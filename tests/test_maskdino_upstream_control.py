#!/usr/bin/env python3
"""
CPU tests for the upstream-MaskDINO control arm (third_party/maskdino_control/).

Unlike every other test in tests/, this one runs under the REFERENCE env, not the project's
myenv/ — it needs detectron2 0.6:

    /cluster/scratch/niacobone/MaskDINO/myenv/bin/python tests/test_maskdino_upstream_control.py

What it protects. The whole value of the control row is that ONLY the listed axes differ from
upstream. A silently-wrong squash, a schedule that is cosine-ish rather than ours, or a yaml key
that stopped overriding its base would produce a number that looks fine and means nothing. So:

  1. the squash mapper's geometry and targets, including the hflip;
  2. the LR schedule against the ACTUAL lambda in scripts/train_maskdino_coco.py (parsed out of
     the source, so the two cannot drift);
  3. every axis of the config table, asserted against the loaded config;
  4. the axes that must NOT have changed, asserted against upstream's own base config.
"""

import ast
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
MASKDINO_ROOT = os.environ.get("MASKDINO_ROOT", "/cluster/scratch/niacobone/MaskDINO")
sys.path.insert(0, MASKDINO_ROOT)
sys.path.insert(0, str(REPO))

from detectron2.config import get_cfg                                   # noqa: E402
from detectron2.projects.deeplab import add_deeplab_config              # noqa: E402
from maskdino import add_maskdino_config                                # noqa: E402

from third_party.maskdino_control.config import add_control_config      # noqa: E402
from third_party.maskdino_control.lr import matched_lr_lambda           # noqa: E402
from third_party.maskdino_control.squash_mapper import CocoSquashDatasetMapper  # noqa: E402

CONFIG_DIR = REPO / "third_party/maskdino_control/configs"
MATCHED = CONFIG_DIR / "maskdino_upstream_matched.yaml"
UPSTREAM_BASE = Path(MASKDINO_ROOT) / "configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s.yaml"


def load_cfg(path):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    add_control_config(cfg)
    cfg.merge_from_file(str(path))
    return cfg


# --------------------------------------------------------------------------------------------
def test_lr_matches_our_arms():
    """
    Reads `build_step_scheduler` out of scripts/train_maskdino_coco.py and runs it. Importing
    that module would drag in torch+VGGT, so the FunctionDef is extracted with ast — which is
    the point: the comparison is against the source our three arms actually ran, not a copy.
    """
    src = (REPO / "scripts/train_maskdino_coco.py").read_text()
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "build_step_scheduler")
    ns = {"np": np, "LambdaLR": torch.optim.lr_scheduler.LambdaLR}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<train_maskdino_coco>", "exec"), ns)

    total, warmup, ratio = 87948, 1000, 0.01
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=1.0)
    ref = ns["build_step_scheduler"](opt, total, warmup, ratio)
    ref_lambda = ref.lr_lambdas[0]

    for step in [0, 1, 499, 999, 1000, 1001, 5000, 43974, 87947, 87948, 90000]:
        a, b = ref_lambda(step), matched_lr_lambda(step, total, warmup, ratio)
        assert abs(a - b) < 1e-12, f"step {step}: ours {b} vs arms {a}"

    # sanity on the shape itself, not just on agreement
    assert matched_lr_lambda(0, total, warmup, ratio) == 1.0 / warmup
    assert abs(matched_lr_lambda(warmup - 1, total, warmup, ratio) - 1.0) < 1e-12
    assert abs(matched_lr_lambda(total, total, warmup, ratio) - ratio) < 1e-9
    print("✓ LR schedule identical to scripts/train_maskdino_coco.py's, incl. the 0.01 floor")


# --------------------------------------------------------------------------------------------
def _fake_dataset_dict(tmp, w=200, h=100):
    from PIL import Image
    p = Path(tmp) / "img.png"
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, : w // 2] = 255                      # left half white: the flip test needs asymmetry
    Image.fromarray(arr).save(p)
    # one box-shaped polygon in the left half
    poly = [[10.0, 10.0, 40.0, 10.0, 40.0, 50.0, 10.0, 50.0]]
    return {
        "file_name": str(p), "image_id": 7, "height": h, "width": w,
        "annotations": [{"bbox": [10, 10, 40, 50], "bbox_mode": 0,  # BoxMode.XYXY_ABS == 0
                         "category_id": 3, "segmentation": poly, "iscrowd": 0}],
    }


def test_squash_mapper():
    cfg = load_cfg(MATCHED)
    size = cfg.INPUT.SQUASH_SIZE

    with tempfile.TemporaryDirectory() as tmp:
        d = _fake_dataset_dict(tmp)

        cfg_noflip = cfg.clone(); cfg_noflip.CONTROL.HFLIP_PROB = 0.0
        out = CocoSquashDatasetMapper(cfg_noflip, True)(d)

        assert tuple(out["image"].shape) == (3, size, size), out["image"].shape
        assert out["image"].dtype == torch.uint8, out["image"].dtype
        inst = out["instances"]
        assert len(inst) == 1
        assert tuple(inst.gt_masks.shape) == (1, size, size), inst.gt_masks.shape
        assert isinstance(inst.gt_masks, torch.Tensor), "MaskDINO.prepare_targets wants a tensor"
        assert int(inst.gt_classes[0]) == 3
        # the polygon lived in x:[10,40] of a 200-wide image -> [0.05,0.20] of the squashed frame
        x0, _, x1, _ = inst.gt_boxes.tensor[0].tolist()
        assert abs(x0 / size - 0.05) < 0.01 and abs(x1 / size - 0.20) < 0.01, (x0, x1)
        # the squash is an aspect-ratio change, so the mask must NOT be letterboxed
        assert inst.gt_masks[0].any(), "mask vanished"
        col = inst.gt_masks[0].any(0).nonzero().flatten()
        assert col.min() < size * 0.10 and col.max() < size * 0.25

        # height/width stay ORIGINAL: that is what inverts the squash in MaskDINO.forward
        assert (out["height"], out["width"]) == (100, 200)

        cfg_flip = cfg.clone(); cfg_flip.CONTROL.HFLIP_PROB = 1.0
        outf = CocoSquashDatasetMapper(cfg_flip, True)(_fake_dataset_dict(tmp))
        colf = outf["instances"].gt_masks[0].any(0).nonzero().flatten()
        assert colf.min() > size * 0.75, f"hflip did not mirror the mask: {colf.min()}"
        assert torch.equal(outf["image"], torch.flip(out["image"], dims=[2])), "image not flipped"

        # test mode: squash, no flip, no annotations
        outt = CocoSquashDatasetMapper(cfg, False)(_fake_dataset_dict(tmp))
        assert tuple(outt["image"].shape) == (3, size, size)
        assert "instances" not in outt and "annotations" not in outt
        assert torch.equal(outt["image"], out["image"]), "eval-time augmentation leaked"
    print("✓ squash mapper: 518² geometry, bitmask targets, hflip, eval path clean")


# --------------------------------------------------------------------------------------------
def test_config_axes():
    """The table in third_party/maskdino_control/train_control.py's docstring, as assertions."""
    cfg = load_cfg(MATCHED)
    base = load_cfg(UPSTREAM_BASE)

    changed = {
        "SOLVER.MAX_ITER": (87948, 368750),
        "MODEL.BACKBONE.FREEZE_AT": (5, 0),
        "SOLVER.BACKBONE_MULTIPLIER": (0.0, 0.1),
        "SOLVER.LR_SCHEDULER_NAME": ("WarmupCosineMatched", "WarmupMultiStepLR"),
        "SOLVER.WARMUP_ITERS": (1000, 10),
        "SOLVER.CLIP_GRADIENTS.CLIP_VALUE": (0.1, 0.01),
        "INPUT.DATASET_MAPPER_NAME": ("coco_instance_squash", "coco_instance_lsj"),
        "MODEL.MaskDINO.SIZE_DIVISIBILITY": (1, 32),
    }
    unchanged = [
        "SOLVER.IMS_PER_BATCH", "SOLVER.BASE_LR", "SOLVER.WEIGHT_DECAY", "SOLVER.OPTIMIZER",
        "SOLVER.CLIP_GRADIENTS.CLIP_TYPE", "SOLVER.AMP.ENABLED",
        "MODEL.MaskDINO.NUM_OBJECT_QUERIES", "MODEL.MaskDINO.DEC_LAYERS",
        "MODEL.MaskDINO.TRAIN_NUM_POINTS", "MODEL.MaskDINO.DN", "MODEL.MaskDINO.DN_NUM",
        "MODEL.MaskDINO.TWO_STAGE", "MODEL.MaskDINO.INITIALIZE_BOX_TYPE",
        "MODEL.MaskDINO.INITIAL_PRED", "MODEL.SEM_SEG_HEAD.NUM_CLASSES",
        "MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS", "MODEL.SEM_SEG_HEAD.DIM_FEEDFORWARD",
        "TEST.DETECTIONS_PER_IMAGE", "MODEL.WEIGHTS", "MODEL.PIXEL_MEAN", "MODEL.PIXEL_STD",
        "INPUT.FORMAT",
    ]

    def get(c, dotted):
        for k in dotted.split("."):
            c = getattr(c, k)
        return c

    for key, (want, want_base) in changed.items():
        assert get(cfg, key) == want, f"{key}: {get(cfg, key)} != {want}"
        assert get(base, key) == want_base, \
            f"upstream {key} is {get(base, key)}, not {want_base} — the clone changed?"
    for key in unchanged:
        assert get(cfg, key) == get(base, key), \
            f"{key} drifted from upstream: {get(cfg, key)} vs {get(base, key)}"

    # our arms' values, from scripts/train_maskdino_coco.py's argparser
    assert cfg.SOLVER.BASE_LR == 1e-4 and cfg.SOLVER.WEIGHT_DECAY == 0.05
    assert cfg.INPUT.SQUASH_SIZE == 518 and cfg.CONTROL.HFLIP_PROB == 0.5
    assert cfg.CONTROL.COSINE_END_LR_RATIO == 0.01
    assert cfg.TEST.DETECTIONS_PER_IMAGE == 100
    print(f"✓ config: {len(changed)} axes changed as designed, {len(unchanged)} held at upstream")


def test_overfit_config():
    cfg = load_cfg(CONFIG_DIR / "maskdino_upstream_matched_overfit.yaml")
    assert cfg.SOLVER.MAX_ITER == 600, cfg.SOLVER.MAX_ITER
    assert cfg.CONTROL.COCO_ROOT.endswith("coco_overfit64")
    # the gate must inherit every axis of the real run except the budget and the data root
    real = load_cfg(MATCHED)
    for k in ("MODEL.BACKBONE.FREEZE_AT", "INPUT.SQUASH_SIZE", "INPUT.DATASET_MAPPER_NAME",
              "SOLVER.IMS_PER_BATCH", "SOLVER.CLIP_GRADIENTS.CLIP_VALUE",
              "MODEL.MaskDINO.SIZE_DIVISIBILITY", "SOLVER.LR_SCHEDULER_NAME",
              "SOLVER.BASE_LR", "SOLVER.WEIGHT_DECAY", "CONTROL.COSINE_END_LR_RATIO",
              "CONTROL.HFLIP_PROB"):
        a, b = cfg, real
        for p in k.split("."):
            a, b = getattr(a, p), getattr(b, p)
        assert a == b, f"gate drifted from the real run at {k}: {a} vs {b}"

    # The gate DOES depart on the LR schedule, and must: it has to spend its 600 steps at the
    # recipe's LR rather than measure a cosine compressed into them. Assert the departure is the
    # intended one — near-constant 1e-4 — and that the REAL run does not inherit it.
    assert real.CONTROL.LR_HORIZON_ITERS == 0, "the real run must take its horizon from MAX_ITER"
    assert cfg.CONTROL.LR_HORIZON_ITERS == real.SOLVER.MAX_ITER
    mults = [matched_lr_lambda(s, cfg.CONTROL.LR_HORIZON_ITERS, cfg.SOLVER.WARMUP_ITERS,
                               cfg.CONTROL.COSINE_END_LR_RATIO)
             for s in (10, 200, 400, 599)]
    assert all(m > 0.99 for m in mults), f"gate lr should barely decay, got {mults}"
    print("✓ overfit gate config inherits the real run's axes; lr held at ~1.0x through step 600")


def test_ops_build_present():
    """The A100 (sm_80) rebuild. Missing here == a silent 10x slowdown at run time."""
    so = REPO / "third_party/maskdino_control/ops_build"
    hits = list(so.glob("MultiScaleDeformableAttention*.so"))
    assert hits, f"{so} is empty — run third_party/maskdino_control/build_ops.sh"
    print(f"✓ ops_build present: {hits[0].name}")


if __name__ == "__main__":
    test_lr_matches_our_arms()
    test_squash_mapper()
    test_config_axes()
    test_overfit_config()
    test_ops_build_present()
    print("\nAll upstream-control tests passed.")
