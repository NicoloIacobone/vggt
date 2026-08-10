"""
CPU tests for `--class_agnostic` (docs/todo.md 6e).

The whole mechanism is one rule — **a one-class head means class-agnostic** — so the tests pin
both halves of it: that `num_classes == 1` keeps every instance and collapses its label, and that
`num_classes == 19` is bit-for-bit the behaviour every published run was trained under.

Run: `myenv/bin/python tests/test_class_agnostic.py` (no GPU, no backbone weights).
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.maskdino.head import (  # noqa: E402
    NUM_SCANNET_CLASSES,
    MaskDINOVGGTHead,
    to_scannet_class_logits,
)
from train.maskdino_data import build_frame_targets  # noqa: E402

PASSED = []


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    PASSED.append(message)


def make_batch():
    """
    Two frames, three instances. Instance 3's class (25) is outside the 19-class head, which is
    exactly the case the class-aware path drops and the class-agnostic path must keep.
    """
    masks = torch.zeros(2, 16, 16, dtype=torch.int32)
    masks[0, 0:8, 0:8] = 1
    masks[0, 8:16, 8:16] = 2
    masks[1, 0:8, 0:8] = 1
    masks[1, 8:16, 0:8] = 3
    return {
        "masks": masks,
        "classes": torch.tensor([5, 12, 25], dtype=torch.long),
        "scene_name": "unit-test",
    }


def labels_of(targets):
    return [t["labels"].tolist() for t in targets]


def ids_of(targets):
    return [t["global_ids"].tolist() for t in targets]


def test_class_aware_is_unchanged():
    targets = build_frame_targets(make_batch(), (8, 8), "cpu", num_classes=NUM_SCANNET_CLASSES)
    check(ids_of(targets) == [[1, 2], [1]],
          f"class-aware still drops the out-of-range instance 3, got {ids_of(targets)}")
    check(labels_of(targets) == [[4, 11], [4]],
          f"class-aware labels stay 1..C shifted to 0..C-1, got {labels_of(targets)}")


def test_class_agnostic_keeps_every_instance():
    targets = build_frame_targets(make_batch(), (8, 8), "cpu", num_classes=1)
    check(ids_of(targets) == [[1, 2], [1, 3]],
          f"class-agnostic keeps the instance whose class the head cannot name, "
          f"got {ids_of(targets)}")
    check(labels_of(targets) == [[0, 0], [0, 0]],
          f"every label collapses onto the single class, got {labels_of(targets)}")
    check(all(int(t["labels"].max()) < 1 for t in targets),
          "no label can index past a 1-logit class head")


def test_agnostic_drops_only_invalid_ids():
    """A class index below 1 is corrupt GT, not a foreign taxonomy — it stays dropped."""
    batch = make_batch()
    batch["classes"] = torch.tensor([5, 0, 25], dtype=torch.long)
    targets = build_frame_targets(batch, (8, 8), "cpu", num_classes=1)
    check(ids_of(targets) == [[1], [1, 3]],
          f"instance 2 (class 0 = background) is dropped even class-agnostic, got {ids_of(targets)}")


def test_masks_and_boxes_survive_the_collapse():
    targets = build_frame_targets(make_batch(), (8, 8), "cpu", num_classes=1)
    frame0 = targets[0]
    check(frame0["masks"].shape == (2, 8, 8), f"masks keep the grid, got {frame0['masks'].shape}")
    check(frame0["masks"].sum() > 0, "collapsing labels does not empty the masks")
    check(frame0["boxes"].shape == (2, 4) and float(frame0["boxes"].max()) <= 1.0,
          "boxes stay normalised cxcywh")


def test_one_class_head_builds_and_reports_itself():
    head = MaskDINOVGGTHead(memory_dim=64, hidden_dim=32, mask_dim=32, num_classes=1,
                            num_queries=8, num_feature_levels=3, enc_layers=1, dec_layers=1,
                            nheads=2, two_stage=False, learn_tgt=True, dn="no")
    check(head.num_classes == 1, "the head reports its single class")
    check(head.head_config["num_classes"] == 1,
          "head_config carries num_classes, so a checkpoint alone decides how its GT is built")
    logits = torch.randn(8, 1)
    scannet = to_scannet_class_logits(logits)
    check(scannet.shape == (8, 2), f"[Q,1] → [Q,2] with a background column, got {scannet.shape}")
    check(bool(torch.isinf(scannet[:, 0]).all()), "the background column stays -inf")


def main() -> int:
    test_class_aware_is_unchanged()
    test_class_agnostic_keeps_every_instance()
    test_agnostic_drops_only_invalid_ids()
    test_masks_and_boxes_survive_the_collapse()
    test_one_class_head_builds_and_reports_itself()
    for message in PASSED:
        print(f"  ok  {message}")
    print(f"\n{len(PASSED)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
