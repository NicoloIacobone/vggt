# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Evaluation metrics for multi-view instance segmentation.

Provides interpretable, mask-based metrics computed from the model's dense mask logits and
class logits against the ground-truth instance masks/classes:

  - mIoU       : mean over GT instances of the best IoU achieved by a same-class prediction
                 (recall-oriented; a missed instance contributes 0).
  - AP50/AP75  : average precision at IoU thresholds 0.50 / 0.75 (class-aware, single scene).
  - mAP        : AP averaged over IoU thresholds 0.50:0.05:0.95 (COCO-style).
  - class_acc  : among IoU-matched (pred, GT) pairs, fraction with the correct predicted class.

All masks may have arbitrary trailing spatial dims (e.g. [N, S, h, w]); they are flattened to
[N, K] internally, so the metric naturally treats the multi-view mask of an instance as one set
of pixels across all frames.
"""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Optional


@torch.no_grad()
def mask_iou_matrix(pred_binary: torch.Tensor, gt_binary: torch.Tensor) -> torch.Tensor:
    """
    Pairwise IoU between predicted and GT binary masks.

    Args:
        pred_binary (torch.Tensor): [N_pred, K] binary (0/1) masks
        gt_binary (torch.Tensor): [N_gt, K] binary (0/1) masks

    Returns:
        torch.Tensor: [N_pred, N_gt] IoU values in [0, 1]
    """
    pred = pred_binary.float()
    gt = gt_binary.float()
    inter = pred @ gt.t()                              # [N_pred, N_gt]
    area_pred = pred.sum(dim=1, keepdim=True)          # [N_pred, 1]
    area_gt = gt.sum(dim=1)[None, :]                    # [1, N_gt]
    union = area_pred + area_gt - inter
    return inter / union.clamp(min=1e-6)


def _voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """All-point (VOC2010-style) average precision from a precision/recall curve."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _average_precision(iou: torch.Tensor, scores: torch.Tensor, pred_labels: torch.Tensor,
                       gt_labels: torch.Tensor, iou_threshold: float) -> Optional[float]:
    """
    Class-aware AP at a single IoU threshold for one scene.

    A prediction is a true positive if it has the highest score among still-unmatched GTs of the
    SAME class with IoU >= threshold. Returns None if there are no GT instances.
    """
    n_gt = gt_labels.shape[0]
    if n_gt == 0:
        return None
    n_pred = scores.shape[0]
    if n_pred == 0:
        return 0.0

    order = torch.argsort(scores, descending=True)
    gt_matched = torch.zeros(n_gt, dtype=torch.bool)
    tp = np.zeros(n_pred, dtype=np.float64)
    fp = np.zeros(n_pred, dtype=np.float64)

    for rank, p in enumerate(order.tolist()):
        # Candidate GTs: same class, not yet matched, IoU >= threshold.
        same_class = gt_labels == pred_labels[p]
        ious_p = iou[p].clone()
        ious_p[~same_class] = 0.0
        ious_p[gt_matched] = 0.0
        best_iou, best_gt = torch.max(ious_p, dim=0)
        if best_iou.item() >= iou_threshold:
            tp[rank] = 1.0
            gt_matched[best_gt] = True
        else:
            fp[rank] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    return _voc_ap(recall, precision)


@torch.no_grad()
def compute_instance_segmentation_metrics(
    pred_masks: torch.Tensor,
    class_logits: torch.Tensor,
    gt_masks: torch.Tensor,
    gt_classes: torch.Tensor,
    mask_threshold: float = 0.5,
    score_threshold: float = 0.0,
    background_class: int = 0,
    iou_thresholds: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Compute instance-segmentation metrics for a single scene.

    Args:
        pred_masks (torch.Tensor): [N_pred, ...] dense mask LOGITS (any trailing spatial dims).
        class_logits (torch.Tensor): [N_pred, C] class logits (index `background_class` is bg).
        gt_masks (torch.Tensor): [N_gt, ...] binary GT masks (same trailing dims as pred_masks).
        gt_classes (torch.Tensor): [N_gt] GT class labels.
        mask_threshold (float): probability threshold to binarize predicted masks.
        score_threshold (float): minimum class confidence for a prediction to count as a detection.
        background_class (int): class index treated as background (predictions of this class are
            dropped before evaluation).
        iou_thresholds (list[float], optional): IoU thresholds for mAP (default 0.50:0.05:0.95).

    Returns:
        dict with keys: mIoU, AP50, AP75, mAP, class_acc, num_pred, num_gt.
    """
    if iou_thresholds is None:
        iou_thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]  # 0.50 .. 0.95

    n_gt = gt_masks.shape[0]
    pred_masks = pred_masks.reshape(pred_masks.shape[0], -1)
    gt_flat = (gt_masks.reshape(n_gt, -1) > 0.5)

    # Predicted label = argmax class; score = its softmax probability.
    probs = torch.softmax(class_logits, dim=-1)
    scores_all, labels_all = probs.max(dim=-1)

    # Keep only confident, non-background detections.
    keep = (labels_all != background_class) & (scores_all >= score_threshold)
    pred_bin = (torch.sigmoid(pred_masks[keep]) > mask_threshold)
    pred_labels = labels_all[keep]
    pred_scores = scores_all[keep]

    empty = {"mIoU": 0.0, "AP50": 0.0, "AP75": 0.0, "mAP": 0.0, "class_acc": 0.0,
             "num_pred": int(keep.sum().item()), "num_gt": int(n_gt)}
    if n_gt == 0 or pred_bin.shape[0] == 0:
        return empty

    iou = mask_iou_matrix(pred_bin, gt_flat)  # [N_keep, N_gt]

    # --- mIoU: mean over GT of best same-class prediction IoU ---------------------------------
    same_class = pred_labels[:, None] == gt_classes[None, :]  # [N_keep, N_gt]
    iou_same = iou.clone()
    iou_same[~same_class] = 0.0
    best_iou_per_gt = iou_same.max(dim=0).values if iou_same.shape[0] > 0 else torch.zeros(n_gt)
    mIoU = float(best_iou_per_gt.mean().item())

    # --- AP at each threshold -----------------------------------------------------------------
    aps = [_average_precision(iou, pred_scores, pred_labels, gt_classes, t) for t in iou_thresholds]
    aps = [a for a in aps if a is not None]
    ap_by_t = {t: _average_precision(iou, pred_scores, pred_labels, gt_classes, t)
               for t in (0.5, 0.75)}
    mAP = float(np.mean(aps)) if aps else 0.0

    # --- class accuracy on IoU-Hungarian-matched pairs (class-agnostic matching) --------------
    cost = (-iou).cpu().numpy()
    pi, gi = linear_sum_assignment(cost)
    correct, total = 0, 0
    for p, g in zip(pi, gi):
        if iou[p, g].item() > 0:  # only count overlapping matches
            total += 1
            correct += int(pred_labels[p].item() == gt_classes[g].item())
    class_acc = (correct / total) if total > 0 else 0.0

    return {
        "mIoU": mIoU,
        "AP50": float(ap_by_t[0.5]) if ap_by_t[0.5] is not None else 0.0,
        "AP75": float(ap_by_t[0.75]) if ap_by_t[0.75] is not None else 0.0,
        "mAP": mAP,
        "class_acc": class_acc,
        "num_pred": int(pred_bin.shape[0]),
        "num_gt": int(n_gt),
    }
