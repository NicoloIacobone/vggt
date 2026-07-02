"""
CPU test for train/postprocess.py::select_instances — the shared 2D/3D instance selector.

Verifies the rules the visualizers depend on:
  - background-class queries are dropped,
  - low-confidence queries (< score_thr) are dropped,
  - overlaps resolve by winner-takes-all (highest mask prob owns the pixel),
  - kept order is by descending class score,
  - nearest upsampling preserves labels and changes only resolution.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.postprocess import select_instances, upsample_assignment


def _logits_for(class_idx, conf, num_classes=20):
    """Build a [C] logit vector whose softmax puts ~`conf` on `class_idx`."""
    v = torch.zeros(num_classes)
    # softmax(x)=conf on one logit vs zeros on (C-1): x = log(conf*(C-1)/(1-conf))
    import math
    v[class_idx] = math.log(conf * (num_classes - 1) / (1 - conf))
    return v


def test_filters_and_winner_takes_all():
    N, S, h, w = 4, 1, 2, 2
    class_logits = torch.stack([
        _logits_for(5, 0.90),   # 0: chair, high conf      -> keep
        _logits_for(0, 0.99),   # 1: background            -> drop (bg)
        _logits_for(7, 0.30),   # 2: table, low conf       -> drop (< 0.5)
        _logits_for(3, 0.70),   # 3: cabinet, mid conf     -> keep
    ])

    pred_masks = torch.full((N, S, h, w), -10.0)  # ~0 prob everywhere by default
    # Instance 0 (chair) claims the whole 2x2 strongly.
    pred_masks[0] = 5.0
    # Instance 3 (cabinet) claims the top-left pixel even more strongly -> should win it.
    pred_masks[3, 0, 0, 0] = 9.0

    keep, labels, scores, assign = select_instances(class_logits, pred_masks, score_thr=0.5)

    # Kept set = {0, 3}, ordered by descending score => chair(0.90) before cabinet(0.70).
    assert keep == [0, 3], keep
    assert int(labels[0]) == 5 and int(labels[3]) == 3

    # assign indexes into keep: 0 -> chair, 1 -> cabinet.
    # Top-left pixel goes to cabinet (c=1), the rest to chair (c=0).
    assert assign[0, 0, 0].item() == 1, assign
    assert assign[0, 0, 1].item() == 0
    assert assign[0, 1, 0].item() == 0
    assert assign[0, 1, 1].item() == 0
    print("OK: filtering + winner-takes-all + ordering")


def test_all_background_gives_empty():
    N, S, h, w = 3, 1, 2, 2
    class_logits = torch.stack([_logits_for(0, 0.9) for _ in range(N)])
    pred_masks = torch.full((N, S, h, w), 5.0)
    keep, _, _, assign = select_instances(class_logits, pred_masks)
    assert keep == []
    assert (assign == -1).all()
    print("OK: all-background -> no instances, empty assignment")


def test_mask_threshold_leaves_unclaimed_background():
    N, S, h, w = 1, 1, 2, 2
    class_logits = _logits_for(5, 0.9).unsqueeze(0)
    pred_masks = torch.full((N, S, h, w), -10.0)  # prob ~0 < mask_thr -> nothing claimed
    keep, _, _, assign = select_instances(class_logits, pred_masks, mask_thr=0.5)
    assert keep == [0]
    assert (assign == -1).all(), "no pixel exceeds mask_thr so all stay background"
    print("OK: mask_thr keeps low-prob pixels as background")


def test_upsample_preserves_labels():
    assign = torch.tensor([[[0, 1], [-1, 1]]])  # [1, 2, 2]
    up = upsample_assignment(assign, size=(4, 4))
    assert up.shape == (1, 4, 4)
    assert set(up.unique().tolist()) == {-1, 0, 1}, up
    # Each native cell becomes a 2x2 block (nearest), no interpolation between labels.
    assert up[0, 0, 0] == 0 and up[0, 0, 3] == 1
    assert up[0, 3, 0] == -1 and up[0, 3, 3] == 1
    print("OK: nearest upsample preserves integer labels")


if __name__ == "__main__":
    test_filters_and_winner_takes_all()
    test_all_background_gives_empty()
    test_mask_threshold_leaves_unclaimed_background()
    test_upsample_preserves_labels()
    print("\nAll postprocess tests passed.")
