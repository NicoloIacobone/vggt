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
        apply_sigmoid (bool): If True, treat `pred_masks` as logits and apply sigmoid before
            computing the Dice coefficient. Set False if `pred_masks` are already probabilities
            in [0, 1]. (Replaces the old, unreliable `pred_masks.max() > 1.0` heuristic.)
    """

    def __init__(self, smooth: float = 1e-5, reduction: str = "mean", apply_sigmoid: bool = True):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.apply_sigmoid = apply_sigmoid

    def forward(self, pred_masks: torch.Tensor, target_masks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_masks (torch.Tensor): [N, ...] predicted masks (logits if apply_sigmoid=True,
                else probabilities). All dims after the first are flattened.
            target_masks (torch.Tensor): [N, ...] target masks (binary: 0 or 1)

        Returns:
            torch.Tensor: Dice loss
        """
        if self.apply_sigmoid:
            pred_masks = torch.sigmoid(pred_masks)

        # Flatten all dims after the instance dim (supports [N, H, W], [N, S, h, w], ...).
        pred_masks = pred_masks.reshape(pred_masks.size(0), -1)  # [N, K]
        target_masks = target_masks.reshape(target_masks.size(0), -1).float()  # [N, K]

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


def batch_dice_cost(pred_masks: torch.Tensor, gt_masks: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    """
    Pairwise Dice cost between every predicted and every GT mask (for bipartite matching).

    Args:
        pred_masks (torch.Tensor): [N_pred, K] mask LOGITS (flattened spatial dims)
        gt_masks (torch.Tensor): [N_gt, K] binary masks (flattened)

    Returns:
        torch.Tensor: [N_pred, N_gt] Dice cost in [0, 1] (lower = better overlap)
    """
    pred = torch.sigmoid(pred_masks)                      # [N_pred, K]
    gt = gt_masks.float()                                 # [N_gt, K]
    intersection = pred @ gt.t()                          # [N_pred, N_gt]
    union = pred.sum(dim=1, keepdim=True) + gt.sum(dim=1)[None, :]  # [N_pred, N_gt]
    dice = 1 - (2 * intersection + smooth) / (union + smooth)
    return dice


def batch_bce_cost(pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
    """
    Pairwise binary-cross-entropy cost between every predicted and every GT mask.

    Args:
        pred_masks (torch.Tensor): [N_pred, K] mask LOGITS (flattened)
        gt_masks (torch.Tensor): [N_gt, K] binary masks (flattened)

    Returns:
        torch.Tensor: [N_pred, N_gt] mean BCE cost (lower = better)
    """
    gt = gt_masks.float()
    # BCE-with-logits decomposed so it can be computed for all (pred, gt) pairs at once.
    pos = F.binary_cross_entropy_with_logits(
        pred_masks, torch.ones_like(pred_masks), reduction="none"
    )  # [N_pred, K]
    neg = F.binary_cross_entropy_with_logits(
        pred_masks, torch.zeros_like(pred_masks), reduction="none"
    )  # [N_pred, K]
    cost = pos @ gt.t() + neg @ (1 - gt).t()  # [N_pred, N_gt]
    return cost / pred_masks.shape[1]


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
        pred_masks: Optional[torch.Tensor] = None,
        gt_masks: Optional[torch.Tensor] = None,
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
            pred_masks (torch.Tensor, optional): [N_pred, ...] dense predicted mask LOGITS. If
                given together with `gt_masks`, the mask matching cost is the dense Dice+BCE
                cost (Mask2Former-style) instead of the mask-embedding L2 distance.
            gt_masks (torch.Tensor, optional): [N_gt, ...] dense binary GT masks.

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

        # Mask matching cost: prefer the dense Dice+BCE cost when masks are available,
        # otherwise fall back to the mask-embedding L2 distance.
        if pred_masks is not None and gt_masks is not None:
            pred_flat = pred_masks.reshape(N_pred, -1)  # [N_pred, K] logits
            gt_flat = gt_masks.reshape(N_gt, -1)        # [N_gt, K] binary
            mask_cost = batch_dice_cost(pred_flat, gt_flat) + batch_bce_cost(pred_flat, gt_flat)
        elif gt_mask_embeddings is not None:
            # [N_pred, 1, mask_dim] - [1, N_gt, mask_dim] → [N_pred, N_gt]
            mask_cost = torch.cdist(mask_embeddings, gt_mask_embeddings, p=2)  # [N_pred, N_gt]
        else:
            mask_cost = torch.zeros((N_pred, N_gt), device=device)

        # Compute coordinate distance (L2); zero if no GT coordinates are provided
        if gt_coordinates is not None:
            coord_cost = torch.cdist(coordinates, gt_coordinates, p=2)  # [N_pred, N_gt]
        else:
            coord_cost = torch.zeros((N_pred, N_gt), device=device)

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
        bce_pos_weight_cap: float = 20.0,
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
        # Upper bound on the BCE positive-class weight (neg/pos ratio) to avoid huge weights
        # when an instance's foreground is tiny at the patch resolution.
        self.bce_pos_weight_cap = bce_pos_weight_cap

    def forward(
        self,
        class_logits: torch.Tensor,
        mask_embeddings: torch.Tensor,
        coordinates: torch.Tensor,
        gt_classes,
        gt_mask_embeddings=None,
        gt_coordinates=None,
        gt_masks=None,
        pred_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute D4RT loss (batch-aware, item 8.2).

        The Hungarian matcher runs PER BATCH SAMPLE against that sample's own GT set
        (DETR-style); the loss components are averaged over the samples that have at least
        one GT instance. For B > 1 the GT arguments must therefore be lists of per-sample
        tensors (per-sample instance counts may differ); for B == 1 (or 2D predictions) a
        plain tensor per GT argument is accepted as before.

        Args:
            class_logits (torch.Tensor): [B, N, num_classes] or [N, num_classes] predicted class logits
            mask_embeddings (torch.Tensor): [B, N, mask_dim] or [N, mask_dim] predicted mask embeddings
            coordinates (torch.Tensor): [B, N, 2] or [N, 2] predicted coordinates (normalized)
            gt_classes: [N_gt] tensor, or list of B such tensors
            gt_mask_embeddings: [N_gt, mask_dim] tensor or list of B such tensors (optional)
            gt_coordinates: [N_gt, 2] tensor or list of B such tensors
            gt_masks: [N_gt, ...] dense binary masks, or list of B such tensors (optional)
            pred_masks (torch.Tensor, optional): [B, N, ...] or [N, ...] dense mask logits

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
                - total_loss: scalar loss
                - loss_dict: dictionary with individual loss components (averaged over batch)
        """
        # Normalize predictions to have an explicit batch dimension.
        if class_logits.dim() == 2:
            class_logits = class_logits.unsqueeze(0)
            mask_embeddings = mask_embeddings.unsqueeze(0)
            coordinates = coordinates.unsqueeze(0)
            if pred_masks is not None:
                pred_masks = pred_masks.unsqueeze(0)
        B = class_logits.shape[0]

        def _as_per_sample_list(gt, name):
            """GT may be one tensor (B==1) or a list of B per-sample tensors (B>1)."""
            if gt is None:
                return [None] * B
            if isinstance(gt, (list, tuple)):
                if len(gt) != B:
                    raise ValueError(f"{name}: expected {B} per-sample tensors, got {len(gt)}")
                return list(gt)
            if B == 1:
                return [gt]
            raise ValueError(
                f"{name} must be a list of {B} per-sample tensors when batch size > 1 "
                f"(per-sample instance counts may differ)."
            )

        gt_classes_list = _as_per_sample_list(gt_classes, "gt_classes")
        gt_embed_list = _as_per_sample_list(gt_mask_embeddings, "gt_mask_embeddings")
        gt_coord_list = _as_per_sample_list(gt_coordinates, "gt_coordinates")
        gt_mask_list = _as_per_sample_list(gt_masks, "gt_masks")

        device = class_logits.device
        zero = torch.tensor(0.0, device=device)
        sums = {"class_loss": zero.clone(), "mask_embed_loss": zero.clone(),
                "coord_loss": zero.clone(), "mask_loss": zero.clone()}
        total_matches = 0
        num_valid_samples = 0

        for b in range(B):
            sample_losses, n_matches = self._forward_single(
                class_logits[b], mask_embeddings[b], coordinates[b],
                gt_classes_list[b], gt_embed_list[b], gt_coord_list[b],
                gt_mask_list[b],
                pred_masks[b] if pred_masks is not None else None,
            )
            if n_matches == 0:
                continue
            num_valid_samples += 1
            total_matches += n_matches
            for k in sums:
                sums[k] = sums[k] + sample_losses[k]

        if num_valid_samples == 0:
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return total_loss, {
                "class_loss": torch.tensor(0.0),
                "mask_embed_loss": torch.tensor(0.0),
                "coord_loss": torch.tensor(0.0),
                "mask_loss": torch.tensor(0.0),
                "num_matches": 0,
            }

        losses = {k: v / num_valid_samples for k, v in sums.items()}
        losses["num_matches"] = total_matches

        total_loss = (
            self.class_loss_weight * losses["class_loss"]
            + self.mask_embed_loss_weight * losses["mask_embed_loss"]
            + self.coord_loss_weight * losses["coord_loss"]
            + self.mask_loss_weight * losses["mask_loss"]
        )

        return total_loss, losses

    def _forward_single(
        self,
        class_logits: torch.Tensor,
        mask_embeddings: torch.Tensor,
        coordinates: torch.Tensor,
        gt_classes: torch.Tensor,
        gt_mask_embeddings: Optional[torch.Tensor],
        gt_coordinates: Optional[torch.Tensor],
        gt_masks: Optional[torch.Tensor],
        pred_masks: Optional[torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        """Match and compute the loss terms for ONE sample (un-batched tensors)."""
        # Match predictions to ground truth (mask-aware when dense masks are available)
        pred_indices, gt_indices, cost_matrix = self.matcher(
            class_logits, mask_embeddings, coordinates,
            gt_classes, gt_mask_embeddings, gt_coordinates,
            pred_masks, gt_masks,
        )

        # If no matches, this sample contributes nothing (handled by the caller)
        if len(pred_indices) == 0:
            zero = torch.tensor(0.0, device=class_logits.device)
            return {
                "class_loss": zero,
                "mask_embed_loss": zero,
                "coord_loss": zero,
                "mask_loss": zero,
            }, 0

        # Extract matched predictions and targets
        matched_pred_classes = class_logits[pred_indices]  # [N_matched, num_classes]
        matched_gt_classes = gt_classes[gt_indices]  # [N_matched]

        # Compute individual losses
        losses = {}

        # 1. Class loss (Focal Loss)
        class_loss = self.focal_loss(matched_pred_classes, matched_gt_classes)
        losses["class_loss"] = class_loss

        # 2. Mask embedding loss (L2 distance) - only when descriptor targets are provided.
        # With dense-mask supervision (item 8.4) the mask embeddings are trained purely via the
        # dense mask loss, so this proxy term is typically disabled (gt_mask_embeddings=None).
        if gt_mask_embeddings is not None:
            matched_pred_mask_embed = mask_embeddings[pred_indices]  # [N_matched, mask_dim]
            matched_gt_mask_embed = gt_mask_embeddings[gt_indices]  # [N_matched, mask_dim]
            mask_embed_loss = torch.norm(matched_pred_mask_embed - matched_gt_mask_embed, dim=-1).mean()
        else:
            mask_embed_loss = torch.tensor(0.0, device=class_logits.device)
        losses["mask_embed_loss"] = mask_embed_loss

        # 3. Coordinate term (item 8.5 — resolved as "matching only"): there is no
        # coordinate-regression head, so the "predicted" coordinates are the fixed input query
        # points and a coordinate loss would have no gradient path. Coordinates participate in
        # the matcher cost; this value is a diagnostic of the matched coordinate error and is
        # excluded from the total loss by keeping coord_loss_weight=0.
        if gt_coordinates is not None:
            matched_pred_coords = coordinates[pred_indices]  # [N_matched, 2]
            matched_gt_coords = gt_coordinates[gt_indices]  # [N_matched, 2]
            coord_loss = torch.norm(matched_pred_coords - matched_gt_coords, dim=-1).mean()
        else:
            coord_loss = torch.tensor(0.0, device=class_logits.device)
        losses["coord_loss"] = coord_loss

        # 4. Mask generation loss (Dice + foreground-weighted BCE on dense masks) - optional.
        # Instance masks are sparse (mostly background), so an unweighted BCE collapses to
        # predicting empty masks. We weight the positive (foreground) term by the neg/pos ratio
        # so foreground pixels are not drowned out, and combine it with the scale-invariant Dice.
        mask_loss = torch.tensor(0.0, device=class_logits.device)
        if gt_masks is not None and pred_masks is not None:
            matched_gt_masks = gt_masks[gt_indices].float()  # [N_matched, ...]
            matched_pred_masks = pred_masks[pred_indices]    # [N_matched, ...] (logits)
            num_pos = matched_gt_masks.sum().clamp(min=1.0)
            num_neg = matched_gt_masks.numel() - matched_gt_masks.sum()
            pos_weight = (num_neg / num_pos).clamp(max=self.bce_pos_weight_cap)
            dice = self.dice_loss(matched_pred_masks, matched_gt_masks)
            bce = F.binary_cross_entropy_with_logits(
                matched_pred_masks, matched_gt_masks, pos_weight=pos_weight
            )
            mask_loss = dice + bce

        losses["mask_loss"] = mask_loss

        return losses, len(pred_indices)


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
