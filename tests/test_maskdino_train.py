#!/usr/bin/env python3
"""
MaskDINO training path (docs/MASKDINO.md). Standalone, CPU-only, no VGGT weights.

  - the per-frame GT builder (labels/boxes/masks) from a synthetic scene batch, including the
    drop of classes the 19-logit head cannot represent;
  - the per-frame metric slicing in train/perframe.py, shared with scripts/eval_perframe.py;
  - a 60-step overfit of the whole head on one synthetic frame (loss must drop a lot).
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from maskdino_fixtures import _synthetic_targets, _tiny_head
from models.maskdino import HungarianMatcher, SetCriterion, build_weight_dict

torch.manual_seed(0)


def test_frame_targets_builder():
    print("=== Testing per-frame GT builder ===")
    from train.maskdino_data import build_frame_targets

    S, H, W, out_hw = 2, 16, 16, (8, 8)
    masks = torch.zeros(S, H, W, dtype=torch.long)
    masks[0, 2:10, 2:10] = 1     # instance 1 visible in frame 0 only
    masks[:, 12:15, 12:15] = 2   # instance 2 visible in both frames
    batch = {"classes": torch.tensor([3, 7]), "masks": masks,
             "coordinates": torch.zeros(2, 2), "frame_names": ["a", "b"]}

    per_frame = build_frame_targets(batch, out_hw, "cpu")
    assert len(per_frame) == S
    # frame 0 sees both instances, frame 1 only the second
    assert per_frame[0]["labels"].tolist() == [2, 6], per_frame[0]["labels"]  # 1-based → 0-based
    assert per_frame[1]["labels"].tolist() == [6], per_frame[1]["labels"]
    assert per_frame[0]["masks"].shape == (2, 8, 8)
    assert per_frame[0]["masks"].sum() > 0 and per_frame[1]["masks"].sum() > 0
    assert per_frame[0]["boxes"].shape == (2, 4)
    # instance 1 covers pixels 2..9 of 16 → normalized box centre ≈ 0.375, size 0.5
    assert torch.allclose(per_frame[0]["boxes"][0],
                          torch.tensor([0.375, 0.375, 0.5, 0.5]), atol=0.13), \
        per_frame[0]["boxes"][0]
    assert per_frame[0]["global_ids"].tolist() == [1, 2]
    print("✅ build_frame_targets produces per-frame labels/masks/boxes\n")


def test_frame_targets_out_of_range_class():
    """
    `data/scannet_overfit.py::SCANNET_CLASSES` has TWENTY names — index 20 ('otherfurniture')
    has no logit in the 19-class MaskDINO head. Such instances must be dropped (= background,
    what the official-GT builder already does), never crash the matcher / DN label embedding.
    """
    print("=== Testing out-of-range GT class handling ===")
    from train.maskdino_data import build_frame_targets

    S, H, W, out_hw = 2, 16, 16, (8, 8)
    masks = torch.zeros(S, H, W, dtype=torch.long)
    masks[:, 2:10, 2:10] = 1     # instance 1: class 3 (trainable)
    masks[:, 12:15, 12:15] = 2   # instance 2: class 20 = 'otherfurniture' → unrepresentable
    batch = {"classes": torch.tensor([3, 20]), "masks": masks, "scene_name": "sceneXXXX_00"}

    per_frame = build_frame_targets(batch, out_hw, "cpu", num_classes=19)
    for f, t in enumerate(per_frame):
        assert t["labels"].tolist() == [2], (f, t["labels"])          # only the trainable one
        assert t["global_ids"].tolist() == [1], (f, t["global_ids"])
        assert t["masks"].shape == (1, 8, 8) and t["boxes"].shape == (1, 4)
        assert int(t["labels"].max()) < 19

    # ... and the dropped instance must not resurface just because the head is wider
    wide = build_frame_targets(batch, out_hw, "cpu", num_classes=20)
    assert wide[0]["labels"].tolist() == [2, 19], wide[0]["labels"]

    # Everything in range → byte-identical to the default 19-class behaviour (the official-GT
    # path: labels 0..18 must be produced exactly as before this guard existed).
    ok = {"classes": torch.tensor([3, 7]), "masks": masks}
    a = build_frame_targets(ok, out_hw, "cpu")
    b = build_frame_targets(ok, out_hw, "cpu", num_classes=19)
    for ta, tb in zip(a, b):
        assert ta["labels"].tolist() == tb["labels"].tolist() == [2, 6]
        assert torch.equal(ta["masks"], tb["masks"]) and torch.equal(ta["boxes"], tb["boxes"])
    print("✅ out-of-range classes dropped with a warning; in-range GT unchanged\n")


def test_perframe_metrics():
    """The per-frame protocol used to score BOTH this trial and (via scripts/eval_perframe.py)
    the existing D4RT checkpoints — the only apples-to-apples comparison (trial doc §6)."""
    print("=== Testing per-frame metric slicing ===")
    from train.perframe import perframe_metrics

    S, h, w, Ng, N = 3, 6, 6, 2, 4
    gt_masks = torch.zeros(Ng, S, h, w)
    gt_masks[0, 0, 0:3, 0:3] = 1        # instance 0: frame 0 only
    gt_masks[1, 0:2, 3:6, 3:6] = 1      # instance 1: frames 0 and 1
    gt_classes = torch.tensor([4, 9])   # frame 2 has no GT at all

    pred_masks = torch.full((N, S, h, w), -10.0)
    pred_masks[0] = gt_masks[0] * 20 - 10
    pred_masks[1] = gt_masks[1] * 20 - 10
    class_logits = torch.full((N, 20), -10.0)
    class_logits[0, 4] = 10.0
    class_logits[1, 9] = 10.0
    class_logits[2:, 0] = 10.0          # two background queries, dropped by the softmax path

    rows = perframe_metrics(pred_masks, class_logits, gt_masks, gt_classes)
    assert len(rows) == 2, f"frame 2 has no GT and must be skipped, got {len(rows)} rows"
    assert rows[0]["num_gt"] == 2 and rows[1]["num_gt"] == 1, [r["num_gt"] for r in rows]
    # Frame 1: instance 0 is not visible, so query 0's empty mask must be DROPPED, not counted
    # as a false positive (it is correct multi-view behaviour) — otherwise AP50 would be 0.5.
    assert rows[0]["num_pred"] == 2 and rows[1]["num_pred"] == 1, [r["num_pred"] for r in rows]
    assert all(r["mIoU"] > 0.99 and r["AP50"] > 0.99 for r in rows), rows

    # A prediction that is right in frame 0 but empty in frame 1 must only score in frame 0.
    partial = pred_masks.clone()
    partial[1, 1] = -10.0
    rows2 = perframe_metrics(partial, class_logits, gt_masks, gt_classes)
    assert rows2[0]["mIoU"] > 0.99 and rows2[1]["mIoU"] == 0.0, [r["mIoU"] for r in rows2]
    print("✅ perframe_metrics slices frames, skips empty GT, and scores per frame\n")


def test_overfit_single_frame():
    print("=== Testing 60-step overfit on one synthetic frame ===")
    torch.manual_seed(0)
    h, mem = 8, 64
    head = _tiny_head(dec_layers=2, enc_layers=1, dn="seg", num_queries=12)
    tokens = torch.randn(1, 5 + h * h, mem)
    targets = _synthetic_targets(1, 2, (h, h))
    weight_dict = build_weight_dict(dec_layers=2, two_stage=True, dn="seg")
    criterion = SetCriterion(19, HungarianMatcher(num_points=64), weight_dict,
                             losses=["labels", "masks", "boxes"], num_points=0,
                             dn="seg", dn_losses=["labels", "masks", "boxes"])
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)

    head.train()
    first = last = None
    for step in range(60):
        opt.zero_grad()
        out, mask_dict = head(tokens, 5, targets)
        losses = criterion(out, targets, mask_dict)
        total = sum(losses[k] * weight_dict[k] for k in losses if k in weight_dict)
        total.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        if step == 0:
            first = float(total)
        last = float(total)
    print(f"    loss {first:.2f} → {last:.2f}")
    assert last < 0.6 * first, f"overfit did not converge: {first:.2f} → {last:.2f}"
    print("✅ Head overfits a single synthetic frame\n")


if __name__ == "__main__":
    test_frame_targets_builder()
    test_frame_targets_out_of_range_class()
    test_perframe_metrics()
    test_overfit_single_frame()
    print("All test_maskdino_train tests passed! ✅")
