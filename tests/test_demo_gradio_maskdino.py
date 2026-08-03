#!/usr/bin/env python3
"""
CPU tests for the Gradio viewer's glue (`demos/demo_gradio.py`), which now serves BOTH
checkpoint families: the active MaskDINO head and the retired D4RT arms.

The module is imported in "import-only" mode (`VGGT_DEMO_SKIP_BACKBONE=1`, `--no_seg`), so no
1B-parameter backbone is downloaded and no checkpoint is auto-discovered; the aggregator is
stubbed. What is under test is the wiring that a unit test of `train/maskdino_viz3d.py` cannot
see: that a checkpoint is routed to the right loader, that the scene dropdown is built from the
run's own scene list restricted to what exists on disk, and that the colouring path runs.

    python tests/test_demo_gradio_maskdino.py
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demos"))       # the demo imports visual_util as a sibling
sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ["VGGT_DEMO_SKIP_BACKBONE"] = "1"
sys.argv = [sys.argv[0], "--no_seg"]          # no checkpoint auto-discovery at import

from maskdino_fixtures import _tiny_head      # noqa: E402
import demo_gradio as dg                     # noqa: E402

torch.manual_seed(0)


def _tiny_checkpoint(path: Path, val_scenes: str, multi_frame: bool = True):
    """A minimal but real MaskDINO checkpoint: the head_config a run would save, plus args."""
    head = _tiny_head(two_stage=False, dn="no", learn_tgt=True, initialize_box_type="no")
    torch.save({
        "head_config": head.head_config,
        "head_state_dict": head.state_dict(),
        "epoch": 12,
        "args": {"val_scenes": val_scenes, "train_scenes": "scene9999_99",
                 "feature_mode": "bundle", "feature_layers": [-1], "multi_frame": multi_frame,
                 "num_frames": 8, "backbone_dtype": "float32"},
    }, path)
    return head


class _StubAggregator:
    """Stands in for VGGT: [1, S, 3, H, W] → (token list, patch_start_idx), 36 patch tokens."""

    def __init__(self, dim=64, patches=36, patch_start_idx=5):
        self.dim, self.patches, self.psi = dim, patches, patch_start_idx

    def aggregator(self, imgs):
        S = imgs.shape[1]
        return [torch.randn(1, S, self.psi + self.patches, self.dim)], self.psi


def test_dispatch_routes_by_checkpoint_family():
    print("=== Testing checkpoint dispatch ===")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        scans = tmp / "scans"
        for name in ("scene0011_00", "scene0015_00"):
            (scans / name / "raw_data").mkdir(parents=True)
        dg.SEG_SCANS_ROOT = str(scans)

        ckpt = tmp / "checkpoint_best_bundle.pth"
        _tiny_checkpoint(ckpt, "scene0011_00,scene0015_00,scene0404_99")
        dg.load_seg_checkpoint(str(ckpt))

        assert dg.SEG["kind"] == "maskdino", dg.SEG["kind"]
        assert dg.SEG["head"] is not None and dg.SEG["train_args"]["feature_mode"] == "bundle"
        # scene0404_99 is in the run's val list but not on this disk → must not be offered
        assert dg.SEG["scene_labels"] == ["scene0011_00 (val)", "scene0015_00 (val)"], \
            dg.SEG["scene_labels"]
        # selecting a scene must not touch the D4RT-only fields
        dg._select_seg_scene(1)
        assert dg.SEG["images"] is None and dg.SEG["coords"] is None

        # a D4RT checkpoint (no head_state_dict) must still go to the legacy loader
        d4rt = tmp / "checkpoint.pth"
        torch.save({"head_config": {}, "decoder_head_state_dict": {}}, d4rt)
        seen = {}
        original = dg._load_d4rt_checkpoint
        dg._load_d4rt_checkpoint = lambda p, c: seen.update(path=p)
        try:
            dg.load_seg_checkpoint(str(d4rt))
        finally:
            dg._load_d4rt_checkpoint = original
        assert seen.get("path") == str(d4rt), "D4RT checkpoint was not routed to its loader"
    print("✓ MaskDINO vs D4RT routing, dropdown limited to scenes present on disk\n")


def test_scene_button_reports_missing_data_instead_of_crashing():
    """A stale/unstaged scans root is the normal failure mode — it must say so, not traceback."""
    print("=== Testing scene loading with missing data ===")
    dg.SEG["kind"] = "maskdino"
    dg.SEG["scenes"] = [{"name": "scene0011_00", "split": "val",
                           "scene_dir": "/nonexistent/scene0011_00/raw_data"}]
    dg.SEG["scene_labels"] = ["scene0011_00 (val)"]
    _, target_dir, paths, msg = dg.load_checkpoint_scene("scene0011_00 (val)")
    assert target_dir == "None" and paths is None, (target_dir, paths)
    assert "scene0011_00" in msg and "Could not read" in msg, msg
    print("✓ missing scene data reported in the log, no crash\n")


def test_colour_path_runs_through_the_demo():
    print("=== Testing the demo's colouring path ===")
    S, hw = 3, 64
    stub = _StubAggregator()
    dg.model = stub
    dg.SEG["kind"] = "maskdino"
    dg.SEG["head"] = _tiny_head(two_stage=False, dn="no", learn_tgt=True,
                                  initialize_box_type="no").eval()
    dg.SEG["train_args"] = {"feature_mode": "bundle", "feature_layers": [-1],
                              "multi_frame": True, "backbone_dtype": "float32"}
    dg.SEG["score_threshold"], dg.SEG["mask_threshold"] = 0.0, 0.5
    dg.SEG["drop_stuff"] = False

    images = torch.rand(1, S, 3, hw, hw)
    colors, legend = dg.compute_seg_colors(images)
    assert colors.shape == (S, hw, hw, 3) and colors.dtype == np.uint8, colors.shape
    assert "colour = query id" in legend, legend
    assert "single-frame checkpoint" not in legend, "multi-frame run must not be warned about"

    # a single-frame checkpoint gets the honest caveat: colours mean nothing across views
    dg.SEG["train_args"]["multi_frame"] = False
    _, legend = dg.compute_seg_colors(images)
    assert "single-frame checkpoint" in legend, legend

    # `single` feature mode must drive the aggregator once per frame, as at training time
    calls = []
    stub_single = _StubAggregator()
    original = stub_single.aggregator
    stub_single.aggregator = lambda imgs: (calls.append(imgs.shape[1]), original(imgs))[1]
    dg.model = stub_single
    dg.SEG["train_args"]["feature_mode"] = "single"
    dg.compute_seg_colors(images)
    assert calls == [1] * S, f"single mode ran {calls} frames per pass"
    print("✓ head driven with the run's own feature mode, legend caveats correct\n")


if __name__ == "__main__":
    test_dispatch_routes_by_checkpoint_family()
    test_scene_button_reports_missing_data_instead_of_crashing()
    test_colour_path_runs_through_the_demo()
    print("✅ All Gradio-glue tests passed")
