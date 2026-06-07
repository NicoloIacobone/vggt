#!/usr/bin/env python3
"""
Phase 5 Validation: Test the Loss Formulation with Bipartite Matching

This script validates the Hungarian matching and combined loss computation.
"""

import sys
from pathlib import Path
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from train.loss import (
    FocalLoss,
    DiceLoss,
    PointBipartiteMatcher,
    D4RTLoss,
    create_dummy_targets,
)


def test_focal_loss():
    """Test Focal Loss."""
    print("=== Testing Focal Loss ===")
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)

    # Create dummy predictions and targets
    logits = torch.randn(10, 20)  # 10 samples, 20 classes
    targets = torch.randint(0, 20, (10,))

    loss = focal_loss(logits, targets)

    print(f"Logits shape: {logits.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Focal loss: {loss.item():.6f}")

    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert not torch.isinf(loss), "Loss should not be Inf"

    print("✅ Focal Loss test passed!\n")
    return True


def test_dice_loss():
    """Test Dice Loss."""
    print("=== Testing Dice Loss ===")
    dice_loss = DiceLoss(smooth=1e-5)

    # Create dummy predictions and targets
    pred_masks = torch.randn(8, 256, 256)  # 8 masks, 256x256
    target_masks = torch.randint(0, 2, (8, 256, 256)).float()  # Binary masks

    loss = dice_loss(pred_masks, target_masks)

    print(f"Pred masks shape: {pred_masks.shape}")
    print(f"Target masks shape: {target_masks.shape}")
    print(f"Dice loss: {loss.item():.6f}")

    assert 0 <= loss.item() <= 1, "Dice loss should be in [0, 1]"
    assert not torch.isnan(loss), "Loss should not be NaN"

    print("✅ Dice Loss test passed!\n")
    return True


def test_bipartite_matcher():
    """Test PointBipartiteMatcher."""
    print("=== Testing PointBipartiteMatcher ===")
    matcher = PointBipartiteMatcher(
        class_weight=1.0, mask_weight=1.0, coord_weight=1.0
    )

    # Create dummy predictions and ground truth
    B = 2
    N_pred = 12
    N_gt = 8
    num_classes = 20
    mask_dim = 256

    pred_classes = torch.randn(N_pred, num_classes)
    pred_mask_embed = torch.randn(N_pred, mask_dim)
    pred_coords = torch.rand(N_pred, 2)

    gt_classes = torch.randint(0, num_classes, (N_gt,))
    gt_mask_embed = torch.randn(N_gt, mask_dim)
    gt_coords = torch.rand(N_gt, 2)

    # Run matcher
    pred_indices, gt_indices, cost_matrix = matcher(
        pred_classes, pred_mask_embed, pred_coords,
        gt_classes, gt_mask_embed, gt_coords
    )

    print(f"Predicted instances: {N_pred}")
    print(f"Ground truth instances: {N_gt}")
    print(f"Matched pairs: {len(pred_indices)}")
    print(f"Cost matrix shape: {cost_matrix.shape}")
    print(f"Cost matrix stats: min={cost_matrix.min():.4f}, max={cost_matrix.max():.4f}")

    # Verify matching
    assert len(pred_indices) == len(gt_indices), "Indices should have same length"
    assert len(pred_indices) <= min(N_pred, N_gt), "Matches should not exceed min(N_pred, N_gt)"
    assert len(pred_indices) == min(N_pred, N_gt), "Hungarian algorithm should match all min(N,M)"

    # Verify no duplicates
    assert len(set(pred_indices)) == len(pred_indices), "Pred indices should be unique"
    assert len(set(gt_indices)) == len(gt_indices), "GT indices should be unique"

    print("✅ PointBipartiteMatcher test passed!\n")
    return True


def test_d4rt_loss():
    """Test complete D4RT Loss."""
    print("=== Testing D4RT Loss ===")
    loss_fn = D4RTLoss(
        num_classes=20,
        focal_alpha=0.25,
        focal_gamma=2.0,
        class_loss_weight=1.0,
        mask_embed_loss_weight=1.0,
        coord_loss_weight=1.0,
        mask_loss_weight=1.0,
    )

    # Create dummy predictions
    N_pred = 12
    N_gt = 8
    num_classes = 20
    mask_dim = 256
    H, W = 256, 256

    pred_classes = torch.randn(N_pred, num_classes)
    pred_mask_embed = torch.randn(N_pred, mask_dim)
    pred_coords = torch.rand(N_pred, 2)

    # Create ground truth
    gt_classes = torch.randint(0, num_classes, (N_gt,))
    gt_mask_embed = torch.randn(N_gt, mask_dim)
    gt_coords = torch.rand(N_gt, 2)
    gt_masks = torch.randint(0, 2, (N_gt, H, W)).float()
    pred_masks = torch.randn(N_pred, H, W)

    # Compute loss
    total_loss, loss_dict = loss_fn(
        pred_classes, pred_mask_embed, pred_coords,
        gt_classes, gt_mask_embed, gt_coords,
        gt_masks, pred_masks
    )

    print(f"Total loss: {total_loss.item():.6f}")
    print(f"Loss components:")
    for name, value in loss_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"  {name}: {value.item():.6f}")
        else:
            print(f"  {name}: {value}")

    assert total_loss.item() > 0, "Total loss should be positive"
    assert not torch.isnan(total_loss), "Loss should not be NaN"
    assert not torch.isinf(total_loss), "Loss should not be Inf"
    assert loss_dict["num_matches"] == min(N_pred, N_gt)

    print("✅ D4RT Loss test passed!\n")
    return True


def test_gradient_flow():
    """Test gradient flow through loss."""
    print("=== Testing Gradient Flow ===")
    loss_fn = D4RTLoss(num_classes=20)

    N_pred = 10
    N_gt = 8
    mask_dim = 256

    # Create predictions with requires_grad=True
    pred_classes = torch.randn(N_pred, 20, requires_grad=True)
    pred_mask_embed = torch.randn(N_pred, mask_dim, requires_grad=True)
    pred_coords = torch.rand(N_pred, 2, requires_grad=True)

    # Ground truth
    gt_classes = torch.randint(0, 20, (N_gt,))
    gt_mask_embed = torch.randn(N_gt, mask_dim)
    gt_coords = torch.rand(N_gt, 2)

    # Compute loss
    total_loss, _ = loss_fn(
        pred_classes, pred_mask_embed, pred_coords,
        gt_classes, gt_mask_embed, gt_coords
    )

    # Backward
    total_loss.backward()

    print(f"Total loss: {total_loss.item():.6f}")
    print(f"Pred classes grad norm: {pred_classes.grad.norm():.6f}")
    print(f"Pred mask embed grad norm: {pred_mask_embed.grad.norm():.6f}")
    print(f"Pred coords grad norm: {pred_coords.grad.norm():.6f}")

    assert pred_classes.grad is not None, "Classes should have gradients"
    assert pred_mask_embed.grad is not None, "Mask embeddings should have gradients"
    assert pred_coords.grad is not None, "Coordinates should have gradients"

    assert pred_classes.grad.norm() > 0
    assert pred_mask_embed.grad.norm() > 0
    assert pred_coords.grad.norm() > 0

    print("✅ Gradient flow test passed!\n")
    return True


def test_empty_ground_truth():
    """Test with no ground truth instances."""
    print("=== Testing Empty Ground Truth ===")
    loss_fn = D4RTLoss(num_classes=20)

    N_pred = 5
    mask_dim = 256

    pred_classes = torch.randn(N_pred, 20)
    pred_mask_embed = torch.randn(N_pred, mask_dim)
    pred_coords = torch.rand(N_pred, 2)

    # Empty ground truth
    gt_classes = torch.zeros(0, dtype=torch.long)
    gt_mask_embed = torch.randn(0, mask_dim)
    gt_coords = torch.rand(0, 2)

    # Should handle gracefully
    total_loss, loss_dict = loss_fn(
        pred_classes, pred_mask_embed, pred_coords,
        gt_classes, gt_mask_embed, gt_coords
    )

    print(f"Total loss (empty GT): {total_loss.item():.6f}")
    print(f"Num matches: {loss_dict['num_matches']}")

    assert total_loss.item() == 0, "Loss should be 0 for empty ground truth"
    assert loss_dict["num_matches"] == 0

    print("✅ Empty ground truth test passed!\n")
    return True


def test_empty_predictions():
    """Test with no predictions."""
    print("=== Testing Empty Predictions ===")
    loss_fn = D4RTLoss(num_classes=20)

    # Empty predictions
    pred_classes = torch.randn(0, 20)
    pred_mask_embed = torch.randn(0, 256)
    pred_coords = torch.rand(0, 2)

    N_gt = 5
    gt_classes = torch.randint(0, 20, (N_gt,))
    gt_mask_embed = torch.randn(N_gt, 256)
    gt_coords = torch.rand(N_gt, 2)

    # Should handle gracefully
    total_loss, loss_dict = loss_fn(
        pred_classes, pred_mask_embed, pred_coords,
        gt_classes, gt_mask_embed, gt_coords
    )

    print(f"Total loss (empty predictions): {total_loss.item():.6f}")
    print(f"Num matches: {loss_dict['num_matches']}")

    assert total_loss.item() == 0, "Loss should be 0 for empty predictions"
    assert loss_dict["num_matches"] == 0

    print("✅ Empty predictions test passed!\n")
    return True


def test_perfect_match():
    """Test with perfect predictions (should have low loss)."""
    print("=== Testing Perfect Match ===")
    loss_fn = D4RTLoss(num_classes=20)

    N = 5
    mask_dim = 256

    # Create predictions that match ground truth
    gt_classes = torch.tensor([1, 2, 3, 4, 5])
    gt_mask_embed = torch.randn(N, mask_dim)
    gt_coords = torch.rand(N, 2)

    # Predictions are identical to ground truth
    pred_classes = torch.zeros(N, 20)
    for i, cls in enumerate(gt_classes):
        pred_classes[i, cls] = 10.0  # High logit for correct class

    pred_mask_embed = gt_mask_embed.clone()
    pred_coords = gt_coords.clone()

    total_loss, loss_dict = loss_fn(
        pred_classes, pred_mask_embed, pred_coords,
        gt_classes, gt_mask_embed, gt_coords
    )

    print(f"Total loss (perfect match): {total_loss.item():.6f}")
    print(f"Individual losses:")
    for k, v in loss_dict.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.item():.6f}")

    # Perfect match should have very low loss
    assert total_loss.item() < 1.0, "Perfect match should have low loss"

    print("✅ Perfect match test passed!\n")
    return True


def test_loss_components_weighted():
    """Test different loss weightings."""
    print("=== Testing Loss Component Weighting ===")

    N_pred = 8
    N_gt = 6
    mask_dim = 256

    pred_classes = torch.randn(N_pred, 20)
    pred_mask_embed = torch.randn(N_pred, mask_dim)
    pred_coords = torch.rand(N_pred, 2)

    gt_classes = torch.randint(0, 20, (N_gt,))
    gt_mask_embed = torch.randn(N_gt, mask_dim)
    gt_coords = torch.rand(N_gt, 2)

    # Test different weightings
    for class_weight in [0.0, 0.5, 1.0, 2.0]:
        loss_fn = D4RTLoss(
            num_classes=20,
            class_loss_weight=class_weight,
            mask_embed_loss_weight=1.0,
            coord_loss_weight=1.0,
        )

        total_loss, _ = loss_fn(
            pred_classes, pred_mask_embed, pred_coords,
            gt_classes, gt_mask_embed, gt_coords
        )

        print(f"Loss with class_weight={class_weight}: {total_loss.item():.6f}")

    print("✅ Loss weighting test passed!\n")
    return True


if __name__ == "__main__":
    try:
        test_focal_loss()
        test_dice_loss()
        test_bipartite_matcher()
        test_d4rt_loss()
        test_gradient_flow()
        test_empty_ground_truth()
        test_empty_predictions()
        test_perfect_match()
        test_loss_components_weighted()

        print("=" * 60)
        print("✅ Phase 5 Validation PASSED!")
        print("=" * 60)
        print("\nLoss Formulation successfully:")
        print("  1. Focal Loss for class imbalance")
        print("  2. Dice Loss for mask generation")
        print("  3. Hungarian matching via linear_sum_assignment")
        print("  4. Combined loss with configurable weights")
        print("  5. Handles edge cases (empty predictions/GT)")
        print("  6. Supports full gradient flow for training")
        print("  7. Produces stable, valid loss values")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
