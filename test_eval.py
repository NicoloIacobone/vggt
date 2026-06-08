#!/usr/bin/env python3
"""
Validation for the instance-segmentation evaluation metrics (item 8.4 eval metric).

Checks the metric on three controlled cases: perfect predictions, no detections, and a
partially-correct prediction set.
"""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))

from train.eval_metrics import compute_instance_segmentation_metrics, mask_iou_matrix


def _disjoint_gt(num_inst=3, S=2, h=8, w=8):
    """Build num_inst disjoint binary masks (each owns a distinct row band) + classes."""
    gt_masks = torch.zeros(num_inst, S, h, w)
    rows = torch.linspace(0, h, num_inst + 1).long()
    for i in range(num_inst):
        gt_masks[i, :, rows[i]:rows[i + 1], :] = 1.0
    gt_classes = torch.arange(1, num_inst + 1)  # classes 1..num_inst
    return gt_masks, gt_classes


def _logits_from_binary(binary, pos=10.0, neg=-10.0):
    """Turn a binary mask into large +/- logits so sigmoid>0.5 reproduces it exactly."""
    return binary * (pos - neg) + neg


def test_iou_matrix():
    print("=== Testing IoU matrix ===")
    a = torch.zeros(2, 16)
    a[0, :8] = 1
    a[1, 8:] = 1
    iou = mask_iou_matrix(a, a)
    assert torch.allclose(iou, torch.eye(2), atol=1e-5), f"IoU of identical/disjoint wrong:\n{iou}"
    print("✅ IoU matrix test passed!\n")


def test_perfect():
    print("=== Testing Perfect Predictions ===")
    gt_masks, gt_classes = _disjoint_gt()
    num_classes = 20

    pred_masks = _logits_from_binary(gt_masks)  # exact masks
    class_logits = torch.full((gt_classes.shape[0], num_classes), -10.0)
    for i, c in enumerate(gt_classes):
        class_logits[i, c] = 10.0

    m = compute_instance_segmentation_metrics(pred_masks, class_logits, gt_masks, gt_classes)
    print("  metrics:", {k: round(v, 3) for k, v in m.items()})

    assert m["mIoU"] > 0.99, f"mIoU should be ~1, got {m['mIoU']}"
    assert m["AP50"] > 0.99 and m["AP75"] > 0.99, "AP should be ~1 for perfect preds"
    assert m["mAP"] > 0.99, f"mAP should be ~1, got {m['mAP']}"
    assert m["class_acc"] > 0.99, "class_acc should be 1"
    print("✅ Perfect prediction test passed!\n")


def test_no_detections():
    print("=== Testing No Detections ===")
    gt_masks, gt_classes = _disjoint_gt()
    num_classes = 20

    pred_masks = _logits_from_binary(gt_masks)
    # All predictions are background (class 0) -> dropped.
    class_logits = torch.full((gt_classes.shape[0], num_classes), -10.0)
    class_logits[:, 0] = 10.0

    m = compute_instance_segmentation_metrics(pred_masks, class_logits, gt_masks, gt_classes)
    print("  metrics:", {k: round(v, 3) for k, v in m.items()})

    assert m["num_pred"] == 0, "All preds should be dropped as background"
    assert m["mIoU"] == 0.0 and m["mAP"] == 0.0, "No detections -> zero metrics"
    print("✅ No detection test passed!\n")


def test_partial():
    print("=== Testing Partial Correctness ===")
    gt_masks, gt_classes = _disjoint_gt(num_inst=4)
    num_classes = 20

    # 4 GT instances; predict 2 perfectly, 1 with wrong class, drop 1 (background).
    pred_masks = _logits_from_binary(gt_masks)
    class_logits = torch.full((4, num_classes), -10.0)
    class_logits[0, gt_classes[0]] = 10.0          # correct
    class_logits[1, gt_classes[1]] = 10.0          # correct
    class_logits[2, (int(gt_classes[3]))] = 10.0   # wrong class for GT #2 region
    class_logits[3, 0] = 10.0                       # background -> dropped

    m = compute_instance_segmentation_metrics(pred_masks, class_logits, gt_masks, gt_classes)
    print("  metrics:", {k: round(v, 3) for k, v in m.items()})

    assert m["num_pred"] == 3, f"Expected 3 non-bg detections, got {m['num_pred']}"
    assert 0.0 < m["mIoU"] < 1.0, f"Partial mIoU should be between 0 and 1, got {m['mIoU']}"
    assert 0.0 < m["AP50"] < 1.0, f"Partial AP50 should be strictly between 0 and 1, got {m['AP50']}"
    print("✅ Partial correctness test passed!\n")


if __name__ == "__main__":
    try:
        test_iou_matrix()
        test_perfect()
        test_no_detections()
        test_partial()
        print("=" * 60)
        print("✅ Eval Metrics Validation PASSED!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
