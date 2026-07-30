#!/usr/bin/env python3
"""
Full-resolution eval (`--eval_full_res`, docs/MASKDINO.md §6.5). Standalone, CPU-only.

  - the two protocol helpers (logit upsampling, GT from the full-res id map);
  - the ruler difference itself: a prediction PERFECT on the mask grid scores < 1 on the
    full-resolution ruler when the GT has sub-cell detail — exactly the signal the grid
    protocol cannot see, and the reason the flag exists;
  - eval integration, single-frame and multi-frame: `full_*` keys appear with the flag on,
    are absent with it off (backward compatibility with pre-flag args namespaces), and the
    bundle_* keys stay on the mask grid.
"""

import sys
from argparse import Namespace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from maskdino_fixtures import _tiny_head

torch.manual_seed(0)


def test_helpers():
    print("=== Testing upsample_mask_logits / gt_masks_from_id_map ===")
    from train.perframe import gt_masks_from_id_map, upsample_mask_logits

    # same-size passthrough is exact; a zero-row tensor keeps its emptiness at the new size
    m = torch.randn(3, 8, 8)
    assert upsample_mask_logits(m, (8, 8)) is m
    assert upsample_mask_logits(torch.zeros(0, 8, 8), (32, 32)).shape == (0, 32, 32)
    # bilinear on logits: values stay inside the input range, shape is the target
    up = upsample_mask_logits(m, (16, 16))
    assert up.shape == (3, 16, 16)
    assert float(up.max()) <= float(m.max()) + 1e-6
    assert float(up.min()) >= float(m.min()) - 1e-6

    id_map = torch.zeros(8, 8, dtype=torch.int16)
    id_map[0:3, 0:3] = 1
    id_map[5:8, 5:8] = 3
    got = gt_masks_from_id_map(id_map, torch.tensor([3, 1]))
    assert got.shape == (2, 8, 8)
    assert torch.equal(got[0], (id_map == 3).float())   # order follows global_ids
    assert torch.equal(got[1], (id_map == 1).float())
    assert gt_masks_from_id_map(id_map, torch.zeros(0, dtype=torch.long)).shape == (0, 8, 8)
    print("✅ helpers: logit upsampling + id-map GT extraction\n")


def test_fullres_ruler_sees_subcell_detail():
    """A mask that is PERFECT on the grid ruler must lose on the full-res ruler when the GT has
    structure finer than a grid cell — the whole point of §6.5."""
    print("=== Testing that the full-res ruler measures sub-cell detail ===")
    import torch.nn.functional as F

    from train.eval_metrics import compute_instance_segmentation_metrics
    from train.perframe import upsample_mask_logits

    H, g = 16, 4
    gt_full = torch.zeros(1, H, H)
    gt_full[0, ::2, :] = 1.0            # 1-px horizontal stripes: invisible at 4x4
    # the cache's own downsampling rule (train/maskdino_data.py::build_frame_targets)
    occ = F.interpolate(gt_full[None], size=(g, g), mode="area")[0]
    gt_grid = ((occ >= min(0.5, float(occ.max()))) & (occ > 0)).float()
    assert float(gt_grid.sum()) == g * g  # every cell is half-covered → all on

    cl = torch.full((1, 20), -10.0)
    cl[0, 4] = 10.0
    common = dict(class_logits=cl, gt_classes=torch.tensor([4]),
                  background_class=0, score_mode="sigmoid", score_threshold=0.25)

    pred_grid = gt_grid * 20 - 10       # perfect at grid resolution
    on_grid = compute_instance_segmentation_metrics(
        pred_masks=pred_grid, gt_masks=gt_grid, **common)
    assert on_grid["mIoU"] > 0.99, on_grid

    on_full = compute_instance_segmentation_metrics(
        pred_masks=upsample_mask_logits(pred_grid, (H, H)), gt_masks=gt_full, **common)
    assert on_full["mIoU"] < 0.75, on_full          # stripes cover half the square
    assert on_full["mIoU"] > 0.25, on_full
    print(f"✅ grid mIoU {on_grid['mIoU']:.2f} vs full-res mIoU {on_full['mIoU']:.2f} "
          "— the new ruler sees what the old one cannot\n")


def _fullres_scene(s, hh, mem, name="sceneA"):
    """A cached-scene dict the way prepare_scenes builds it, id maps consistent with targets."""
    from train.maskdino_data import build_frame_targets

    H = hh * 4                                       # "full" resolution, 4x the mask grid
    id_maps = torch.zeros(s, H, H, dtype=torch.long)
    id_maps[:, 2:14, 2:14] = 1
    id_maps[:, 20:30, 18:30] = 2
    targets = build_frame_targets(
        {"classes": torch.tensor([3, 7]), "masks": id_maps}, (hh, hh), "cpu")
    return {"name": name, "split": "val", "bundles": [
        {"features": torch.randn(s, 5 + hh * hh, mem), "patch_start_idx": 5,
         "targets": targets, "images": None, "gt_id_maps": id_maps.to(torch.int16)}]}


def test_eval_scenes_full_res():
    print("=== Testing eval_scenes with --eval_full_res (single-frame path) ===")
    from train.maskdino_eval import eval_scenes

    torch.manual_seed(0)
    s, hh, mem = 2, 8, 64
    model = torch.nn.Module()
    model.head = _tiny_head(dec_layers=2, enc_layers=1, dn="no", num_queries=10)
    scenes = [_fullres_scene(s, hh, mem)]

    args = Namespace(multi_frame=False, eval_topk=100, score_threshold=0.25,
                     eval_batch_frames=4, eval_full_res=True)
    m = eval_scenes(model, scenes, args, "cpu")["sceneA"]
    for k in ("full_mIoU", "full_AP50", "full_AP75", "full_mAP", "full_mIoU_all",
              "full_num_gt", "full_num_gt_all"):
        assert k in m, k
    # the full-res ruler scores the SAME GT instances (only the resolution changes) ...
    assert m["full_num_gt_all"] == m["num_gt_all"], m
    for k, v in m.items():
        if "num_" not in k:
            assert 0.0 <= v <= 1.0, (k, v)

    # ... and a pre-flag args namespace (no eval_full_res attribute) stays untouched
    off = Namespace(multi_frame=False, eval_topk=100, score_threshold=0.25,
                    eval_batch_frames=4)
    m_off = eval_scenes(model, scenes, off, "cpu")["sceneA"]
    assert not any(k.startswith("full_") for k in m_off), sorted(m_off)
    assert set(k for k in m if not k.startswith("full_")) == set(m_off)
    print("✅ full_* keys appear with the flag, are absent without it\n")


def test_eval_scenes_multiframe_full_res():
    print("=== Testing eval_scenes with --eval_full_res (multi-frame path) ===")
    from train.maskdino_eval import eval_scenes

    torch.manual_seed(0)
    s, hh, mem = 3, 8, 64
    model = torch.nn.Module()
    model.head = _tiny_head(dec_layers=2, enc_layers=1, dn="no", num_queries=10,
                            cross_frame_attn=True)
    scenes = [_fullres_scene(s, hh, mem)]

    args = Namespace(multi_frame=True, eval_topk=100, score_threshold=0.25,
                     eval_batch_frames=s, eval_full_res=True)
    m = eval_scenes(model, scenes, args, "cpu")["sceneA"]
    for k in ("full_mIoU", "full_AP50", "bundle_mIoU", "bundle_AP50"):
        assert k in m, k
    # bundle_* stays on the mask grid — no full-res bundle keys in either spelling
    assert not any(k.startswith("bundle_full") or k.startswith("full_bundle") for k in m), \
        sorted(m)
    print("✅ multi-frame eval reports full_* per frame, bundle_* stays on the grid\n")


if __name__ == "__main__":
    test_helpers()
    test_fullres_ruler_sees_subcell_detail()
    test_eval_scenes_full_res()
    test_eval_scenes_multiframe_full_res()
    print("All test_maskdino_fullres tests passed! ✅")
