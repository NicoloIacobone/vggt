#!/usr/bin/env python3
"""
MaskDINO matcher + criterion (docs/MASKDINO.md). Standalone, CPU-only.

  - HungarianMatcher on a planted-perfect-prediction case (must recover the identity matching);
  - SetCriterion key set vs. build_weight_dict, and a zero-loss sanity check on perfect preds;
  - the defensive guard that turns an out-of-range GT label into a named AssertionError rather
    than an opaque IndexError deep in the class head.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from maskdino_fixtures import _synthetic_targets, _tiny_head
from models.maskdino import HungarianMatcher, SetCriterion, build_weight_dict

torch.manual_seed(0)


def test_matcher_recovers_planted_assignment():
    print("=== Testing HungarianMatcher on planted predictions ===")
    n, q, hw = 3, 8, (8, 8)
    targets = _synthetic_targets(1, n, hw)
    t = targets[0]
    # Queries 5, 1, 6 carry (almost) exactly GT 0, 1, 2; the rest are empty background queries.
    planted = [5, 1, 6]
    logits = torch.full((1, q, 19), -8.0)
    masks = torch.full((1, q, *hw), -8.0)
    boxes = torch.rand(1, q, 4) * 0.02 + 0.5
    for gi, qi in enumerate(planted):
        logits[0, qi, t["labels"][gi]] = 8.0
        masks[0, qi] = t["masks"][gi] * 16.0 - 8.0
        boxes[0, qi] = t["boxes"][gi]
    outputs = {"pred_logits": logits, "pred_masks": masks, "pred_boxes": boxes}

    matcher = HungarianMatcher(num_points=64)
    (pred_idx, tgt_idx), = matcher(outputs, targets)
    got = {int(g): int(p) for p, g in zip(pred_idx, tgt_idx)}
    assert got == {0: 5, 1: 1, 2: 6}, got

    # a frame with no GT must yield an empty (not crashing) assignment
    (pi, ti), = matcher(outputs, [{"labels": torch.zeros(0, dtype=torch.long),
                                   "masks": torch.zeros(0, *hw),
                                   "boxes": torch.zeros(0, 4)}])
    assert pi.numel() == 0 and ti.numel() == 0
    print("✅ Matcher recovers the planted assignment and survives empty GT\n")


def test_criterion_keys_and_perfect_predictions():
    print("=== Testing SetCriterion (keys, weights, perfect-prediction floor) ===")
    B, h, nq, dec_layers = 2, 8, 12, 2
    targets = _synthetic_targets(B, 3, (h, h))
    head = _tiny_head(dec_layers=dec_layers, dn="seg")
    head.train()
    tokens = torch.randn(B, 5 + h * h, 64)
    out, mask_dict = head(tokens, 5, targets)

    weight_dict = build_weight_dict(dec_layers=dec_layers, two_stage=True, dn="seg")
    criterion = SetCriterion(19, HungarianMatcher(num_points=64), weight_dict,
                             losses=["labels", "masks", "boxes"], num_points=64,
                             dn="seg", dn_losses=["labels", "masks", "boxes"])
    losses = criterion(out, targets, mask_dict)
    missing = [k for k in losses if k not in weight_dict]
    assert not missing, f"loss keys with no weight: {missing}"
    total = sum(losses[k] * weight_dict[k] for k in losses)
    assert torch.isfinite(total), total
    total.backward()
    grads = [p.grad for p in head.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)

    # Perfect predictions on the final layer only: class + mask + box losses must be tiny.
    t = targets[0]
    n = t["labels"].shape[0]
    logits = torch.full((1, nq, 19), -12.0)
    masks = torch.full((1, nq, h, h), -12.0)
    boxes = torch.rand(1, nq, 4) * 0.01 + 0.5
    for i in range(n):
        logits[0, i, t["labels"][i]] = 12.0
        masks[0, i] = t["masks"][i] * 24.0 - 12.0
        boxes[0, i] = t["boxes"][i]
    perfect = {"pred_logits": logits, "pred_masks": masks, "pred_boxes": boxes}
    crit2 = SetCriterion(19, HungarianMatcher(num_points=0 or 64), {},
                         losses=["labels", "masks", "boxes"], num_points=0)
    l = crit2(perfect, [t])
    assert l["loss_ce"] < 1e-2, l["loss_ce"]
    assert l["loss_mask"] < 1e-2, l["loss_mask"]
    assert l["loss_dice"] < 1e-2, l["loss_dice"]
    assert l["loss_bbox"] < 1e-4, l["loss_bbox"]
    assert l["loss_giou"] < 1e-4, l["loss_giou"]
    print("✅ Criterion keys match build_weight_dict; perfect predictions give ~0 loss\n")


def test_out_of_range_label_guard():
    """A caller that bypasses build_frame_targets must get a clear error, not an IndexError."""
    print("=== Testing defensive out-of-range label guard ===")
    Q, C, hw = 6, 19, (8, 8)
    outputs = {"pred_logits": torch.randn(1, Q, C),
               "pred_masks": torch.randn(1, Q, *hw),
               "pred_boxes": torch.rand(1, Q, 4)}
    bad = [{"labels": torch.tensor([2, C]),          # C == num_classes: one past the last logit
            "masks": torch.zeros(2, *hw),
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]])}]

    matcher = HungarianMatcher(num_points=32)
    crit = SetCriterion(C, matcher, {}, losses=["labels"], num_points=0)
    for name, fn in (("HungarianMatcher", lambda: matcher(outputs, bad)),
                     ("SetCriterion", lambda: crit(outputs, bad))):
        try:
            fn()
            raise SystemExit(f"{name} accepted an out-of-range label")
        except AssertionError as e:
            assert "out of range" in str(e) and "19-class head" in str(e), str(e)

    # negative labels are caught too, and valid labels still pass
    neg = [{**bad[0], "labels": torch.tensor([-1, 3])}]
    try:
        matcher(outputs, neg)
        raise SystemExit("matcher accepted a negative label")
    except AssertionError:
        pass
    good = [{**bad[0], "labels": torch.tensor([0, C - 1])}]
    matcher(outputs, good)  # must not raise
    print("✅ out-of-range labels raise a named AssertionError in matcher + criterion\n")


if __name__ == "__main__":
    test_matcher_recovers_planted_assignment()
    test_criterion_keys_and_perfect_predictions()
    test_out_of_range_label_guard()
    print("All test_maskdino_loss tests passed! ✅")
