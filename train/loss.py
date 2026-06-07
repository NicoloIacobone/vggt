# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import Dict, Optional, Tuple, List


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    From "Focal Loss for Dense Object Detection" (Lin et al., ICCV 2017).

    Args:
        alpha (float): Weighting factor in [0, 1] to balance positive/negative examples
        gamma (float): Exponent of the modulating factor (1 - p_t)^gamma
        reduction (str): 'none' | 'mean' | 'sum'
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): [N, C] model logits
            targets (torch.Tensor): [N] target class indices

        Returns:
            torch.Tensor: Scalar focal loss
        """
        p = F.softmax(logits, dim=-1)
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t) ** self.gamma

        loss = self.alpha * focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class DiceLoss(nn.Module):
    """
    Dice Loss for instance segmentation.

    Computes 1 - (2 * intersection) / (union)

    Args:
        smooth (float): Smoothing constant to avoid division by zero
        reduction (str): 'none' | 'mean' | 'sum'
    """

    def __init__(self, smooth: float = 1e-5, reduction: str = "mean"):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred_masks: torch.Tensor, target_masks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_masks (torch.Tensor): [N, H, W] predicted masks (logits or probabilities)
            target_masks (torch.Tensor): [N, H, W] target masks (binary: 0 or 1)

        Returns:
            torch.Tensor: Dice loss
        """
        # Convert logits to probabilities if needed
        if pred_masks.max() > 1.0:
            pred_masks = torch.sigmoid(pred_masks)

        # Flatten spatial dimensions
        pred_masks = pred_masks.view(pred_masks.size(0), -1)  # [N, H*W]
        target_masks = target_masks.view(target_masks.size(0), -1).float()  # [N, H*W]

        # Compute Dice coefficient
        intersection = (pred_masks * target_masks).sum(dim=1)  # [N]
        union = pred_masks.sum(dim=1) + target_masks.sum(dim=1)  # [N]

        dice = 1 - (2 * intersection + self.smooth) / (union + self.smooth)

        if self.reduction == "mean":
            return dice.mean()
        elif self.reduction == "sum":
            return dice.sum()
        else:
            return dice


class PointBipartiteMatcher(nn.Module):
    """
    Bipartite matching between predicted instances and ground truth instances.

    Uses Hungarian algorithm (linear_sum_assignment) to find the optimal one-to-one
    matching between predictions and targets based on a cost matrix.

    The cost is computed as:
        cost = (1 - class_prob) + mask_dist + coord_dist

    where:
        - class_prob: probability of the predicted class (higher is better)
        - mask_dist: L2 distance between predicted and target mask embeddings
        - coord_dist: L2 distance between predicted and target coordinates

    Args:
        class_weight (float): Weight for class prediction cost (default: 1.0)
        mask_weight (float): Weight for mask embedding distance (default: 1.0)
        coord_weight (float): Weight for coordinate distance (default: 1.0)
    """

    def __init__(
        self,
        class_weight: float = 1.0,
        mask_weight: float = 1.0,
        coord_weight: float = 1.0,
    ):
        super().__init__()
        self.class_weight = class_weight
        self.mask_weight = mask_weight
        self.coord_weight = coord_weight

    def forward(
        self,
        class_logits: torch.Tensor,
        mask_embeddings: torch.Tensor,
        coordinates: torch.Tensor,
        gt_classes: torch.Tensor,
        gt_mask_embeddings: torch.Tensor,
        gt_coordinates: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Match predictions to ground truth instances.

        Args:
            class_logits (torch.Tensor): [N_pred, num_classes] predicted class logits
            mask_embeddings (torch.Tensor): [N_pred, mask_dim] predicted mask embeddings
            coordinates (torch.Tensor): [N_pred, 2] predicted coordinates
            gt_classes (torch.Tensor): [N_gt] ground truth class labels
            gt_mask_embeddings (torch.Tensor): [N_gt, mask_dim] ground truth mask embeddings
            gt_coordinates (torch.Tensor): [N_gt, 2] ground truth coordinates

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - matched_pred_indices: [N_matched] indices of matched predictions
                - matched_gt_indices: [N_matched] indices of matched ground truth
                - cost_matrix: [N_pred, N_gt] the cost matrix used for matching
        """
        N_pred = class_logits.shape[0]
        N_gt = gt_classes.shape[0]

        if N_pred == 0 or N_gt == 0:
            return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long), torch.zeros((N_pred, N_gt))

        device = class_logits.device

        # Compute class prediction cost: 1 - probability of correct class
        probs = F.softmax(class_logits, dim=-1)  # [N_pred, num_classes]
        class_cost = torch.ones((N_pred, N_gt), device=device)

        for j, gt_class_id in enumerate(gt_classes):
            gt_class_id = int(gt_class_id.item()) if isinstance(gt_class_id, torch.Tensor) else int(gt_class_id)
            class_cost[:, j] = 1 - probs[:, gt_class_id]

        # Compute mask embedding distance (L2)
        # [N_pred, 1, mask_dim] - [1, N_gt, mask_dim] → [N_pred, N_gt]
        mask_cost = torch.cdist(mask_embeddings, gt_mask_embeddings, p=2)  # [N_pred, N_gt]

        # Compute coordinate distance (L2)
        coord_cost = torch.cdist(coordinates, gt_coordinates, p=2)  # [N_pred, N_gt]

        # Combine costs
        cost_matrix = (
            self.class_weight * class_cost
            + self.mask_weight * mask_cost
            + self.coord_weight * coord_cost
        )

        # Hungarian algorithm (linear_sum_assignment minimizes cost)
        cost_np = cost_matrix.cpu().detach().numpy()
        pred_indices, gt_indices = linear_sum_assignment(cost_np)

        return (
            torch.tensor(pred_indices, dtype=torch.long, device=device),
            torch.tensor(gt_indices, dtype=torch.long, device=device),
            cost_matrix,
        )


class D4RTLoss(nn.Module):
    """
    Combined loss for D4RT instance segmentation.

    Combines:
    1. Class prediction loss (Focal Loss)
    2. Mask embedding loss (L2 distance)
    3. Coordinate loss (L2 distance)
    4. Mask generation loss (Dice Loss)

    Args:
        num_classes (int): Number of classes (default: 20 for ScanNet)
        focal_alpha (float): Focal loss alpha parameter
        focal_gamma (float): Focal loss gamma parameter
        class_loss_weight (float): Weight for class loss
        mask_embed_loss_weight (float): Weight for mask embedding loss
        coord_loss_weight (float): Weight for coordinate loss
        mask_loss_weight (float): Weight for mask generation loss
        matcher_kwargs (dict): Keyword arguments for PointBipartiteMatcher
    """

    def __init__(
        self,
        num_classes: int = 20,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        class_loss_weight: float = 1.0,
        mask_embed_loss_weight: float = 1.0,
        coord_loss_weight: float = 1.0,
        mask_loss_weight: float = 1.0,
        matcher_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction="mean")
        self.dice_loss = DiceLoss(reduction="mean")
        self.matcher = PointBipartiteMatcher(**(matcher_kwargs or {}))

        self.class_loss_weight = class_loss_weight
        self.mask_embed_loss_weight = mask_embed_loss_weight
        self.coord_loss_weight = coord_loss_weight
        self.mask_loss_weight = mask_loss_weight

    def forward(
        self,
        class_logits: torch.Tensor,
        mask_embeddings: torch.Tensor,
        coordinates: torch.Tensor,
        gt_classes: torch.Tensor,
        gt_mask_embeddings: torch.Tensor,
        gt_coordinates: torch.Tensor,
        gt_masks: Optional[torch.Tensor] = None,
        pred_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute D4RT loss.

        Args:
            class_logits (torch.Tensor): [B, N, num_classes] or [N, num_classes] predicted class logits
            mask_embeddings (torch.Tensor): [B, N, mask_dim] or [N, mask_dim] predicted mask embeddings
            coordinates (torch.Tensor): [B, N, 2] or [N, 2] predicted coordinates (normalized)
            gt_classes (torch.Tensor): [N_gt] ground truth class labels
            gt_mask_embeddings (torch.Tensor): [N_gt, mask_dim] ground truth mask embeddings
            gt_coordinates (torch.Tensor): [N_gt, 2] ground truth coordinates (normalized)
            gt_masks (torch.Tensor, optional): [N_gt, H, W] ground truth instance masks
            pred_masks (torch.Tensor, optional): [N_pred, H, W] predicted instance masks

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
                - total_loss: scalar loss
                - loss_dict: dictionary with individual loss components
        """
        # Flatten batch dimension if present
        if class_logits.dim() == 3:  # [B, N, num_classes]
            B = class_logits.shape[0]
            class_logits = class_logits.view(B * class_logits.shape[1], -1)  # [B*N, num_classes]
            mask_embeddings = mask_embeddings.view(B * mask_embeddings.shape[1], -1)  # [B*N, mask_dim]
            coordinates = coordinates.view(B * coordinates.shape[1], -1)  # [B*N, 2]

        # Match predictions to ground truth
        pred_indices, gt_indices, cost_matrix = self.matcher(
            class_logits, mask_embeddings, coordinates,
            gt_classes, gt_mask_embeddings, gt_coordinates
        )

        # If no matches, return zero loss
        if len(pred_indices) == 0:
            total_loss = torch.tensor(0.0, device=class_logits.device, requires_grad=True)
            return total_loss, {
                "class_loss": torch.tensor(0.0),
                "mask_embed_loss": torch.tensor(0.0),
                "coord_loss": torch.tensor(0.0),
                "mask_loss": torch.tensor(0.0),
                "num_matches": 0,
            }

        # Extract matched predictions and targets
        matched_pred_classes = class_logits[pred_indices]  # [N_matched, num_classes]
        matched_pred_mask_embed = mask_embeddings[pred_indices]  # [N_matched, mask_dim]
        matched_pred_coords = coordinates[pred_indices]  # [N_matched, 2]

        matched_gt_classes = gt_classes[gt_indices]  # [N_matched]
        matched_gt_mask_embed = gt_mask_embeddings[gt_indices]  # [N_matched, mask_dim]
        matched_gt_coords = gt_coordinates[gt_indices]  # [N_matched, 2]

        # Compute individual losses
        losses = {}

        # 1. Class loss (Focal Loss)
        class_loss = self.focal_loss(matched_pred_classes, matched_gt_classes)
        losses["class_loss"] = class_loss

        # 2. Mask embedding loss (L2 distance)
        mask_embed_loss = torch.norm(matched_pred_mask_embed - matched_gt_mask_embed, dim=-1).mean()
        losses["mask_embed_loss"] = mask_embed_loss

        # 3. Coordinate loss (L2 distance)
        coord_loss = torch.norm(matched_pred_coords - matched_gt_coords, dim=-1).mean()
        losses["coord_loss"] = coord_loss

        # 4. Mask generation loss (Dice Loss) - optional
        mask_loss = torch.tensor(0.0, device=class_logits.device)
        if gt_masks is not None and pred_masks is not None:
            matched_gt_masks = gt_masks[gt_indices]  # [N_matched, H, W]
            matched_pred_masks = pred_masks[pred_indices]  # [N_matched, H, W]
            mask_loss = self.dice_loss(matched_pred_masks, matched_gt_masks)

        losses["mask_loss"] = mask_loss
        losses["num_matches"] = len(pred_indices)

        # Combine losses
        total_loss = (
            self.class_loss_weight * class_loss
            + self.mask_embed_loss_weight * mask_embed_loss
            + self.coord_loss_weight * coord_loss
            + self.mask_loss_weight * mask_loss
        )

        return total_loss, losses


def create_dummy_targets(
    batch_size: int = 2,
    num_instances: int = 8,
    num_classes: int = 20,
    mask_dim: int = 256,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Create dummy ground truth targets for testing.

    Args:
        batch_size (int): Batch size
        num_instances (int): Number of instances per batch
        num_classes (int): Number of classes
        mask_dim (int): Dimension of mask embeddings
        device (str): Device to create tensors on

    Returns:
        Dict with gt_classes, gt_mask_embeddings, gt_coordinates
    """
    gt_classes = torch.randint(0, num_classes, (batch_size, num_instances), device=device)
    gt_mask_embeddings = torch.randn(batch_size, num_instances, mask_dim, device=device)
    gt_coordinates = torch.rand(batch_size, num_instances, 2, device=device)

    return {
        "classes": gt_classes,
        "mask_embeddings": gt_mask_embeddings,
        "coordinates": gt_coordinates,
    }
