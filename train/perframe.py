"""
The per-frame evaluation protocol (docs/MASKDINO.md §6).

Both model families are scored through this module, which is the only reason their numbers
are comparable at all: the MaskDINO trainer scores its own single-frame predictions, and
`scripts/eval_perframe.py` runs the multi-view D4RT arms through the identical rules.
"""

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from train.eval_metrics import compute_instance_segmentation_metrics

# The metric keys every per-frame scorer reports.
METRIC_KEYS = ["mIoU", "AP50", "AP75", "mAP", "class_acc", "num_pred", "num_gt"]


def upsample_mask_logits(pred_masks: torch.Tensor, size) -> torch.Tensor:
    """
    Bilinearly upsample [N, h, w] mask LOGITS to `size` — the full-resolution eval protocol
    (docs/MASKDINO.md §6.5). Logits, not probabilities: sigmoid is monotone, so thresholding the
    upsampled logits at 0 equals thresholding the upsampled-in-logit-space probabilities at 0.5,
    and it is exactly how upstream MaskDINO/COCO evaluation upsamples before binarising.
    """
    size = tuple(size)
    if tuple(pred_masks.shape[-2:]) == size:
        return pred_masks
    if pred_masks.shape[0] == 0:
        return pred_masks.new_zeros((0, *size))
    return F.interpolate(pred_masks[None], size=size, mode="bilinear", align_corners=False)[0]


def gt_masks_from_id_map(id_map: torch.Tensor, global_ids: torch.Tensor) -> torch.Tensor:
    """
    [H, W] global-instance-id map + [n] instance ids → [n, H, W] binary float masks.

    The id map is the dataset's full-resolution GT for one frame (the same tensor
    `build_frame_targets` area-downsamples to the mask grid); pulling the frame's instances out
    of it by the *target's* `global_ids` reuses the class-drop decision made there instead of
    duplicating it.
    """
    if global_ids.numel() == 0:
        return id_map.new_zeros((0, *id_map.shape), dtype=torch.float32)
    return torch.stack([(id_map == int(g)).float() for g in global_ids.tolist()])


def drop_empty_masks(pred_masks: torch.Tensor, class_logits: torch.Tensor,
                     mask_threshold: float = 0.5):
    """
    Keep only the predictions that actually claim pixels in this frame.

    THE rule that makes the per-frame protocol fair to both model families: a multi-view D4RT
    query is *supposed* to be empty in the frames where its object is not visible, so counting
    those as false positives would punish it for behaving correctly. A single-frame MaskDINO
    query with an empty mask is likewise not a detection. Mask2Former/MaskDINO reach the same
    place by multiplying the class score with the mask's mean foreground probability, which
    sinks empty masks to the bottom of the ranking.

    Args:
        pred_masks: [N, h, w] mask logits for one frame; class_logits: [N, C].
    Returns:
        (pred_masks_kept, class_logits_kept) — possibly zero rows.
    """
    nonempty = (torch.sigmoid(pred_masks).flatten(1) > mask_threshold).any(dim=1)
    return pred_masks[nonempty], class_logits[nonempty]


def topk_predictions(pred_masks: torch.Tensor, class_logits: torch.Tensor, k: int):
    """
    Keep the k highest-scoring predictions (COCO's `test_topk_per_image`, MaskDINO uses 100).

    Score = max sigmoid class probability. `k <= 0` disables the cap. Beyond matching the COCO
    protocol this bounds eval cost: AP loops over every kept prediction at 10 IoU thresholds.
    """
    if k <= 0 or class_logits.shape[0] <= k:
        return pred_masks, class_logits
    scores = torch.sigmoid(class_logits).max(dim=-1).values
    keep = torch.topk(scores, k).indices
    return pred_masks[keep], class_logits[keep]


def perframe_metrics(pred_masks: torch.Tensor, class_logits: torch.Tensor,
                     gt_masks: torch.Tensor, gt_classes: torch.Tensor,
                     score_threshold: float = 0.0, score_mode: str = "softmax",
                     background_class: int = 0, mask_threshold: float = 0.5
                     ) -> List[Dict[str, float]]:
    """
    Per-frame instance metrics from MULTI-VIEW predictions (the D4RT side of the comparison).

    Args:
        pred_masks: [N, S, h, w] mask logits; class_logits: [N, C].
        gt_masks:   [Ng, S, h, w] binary; gt_classes: [Ng].
    Returns:
        one metric dict per frame that has at least one visible GT instance (frames with no GT
        are skipped: their mIoU/AP are undefined and would only dilute the mean). Predictions
        that are empty in the frame are dropped first — see `drop_empty_masks`.
    """
    rows = []
    S = pred_masks.shape[1]
    for f in range(S):
        visible = (gt_masks[:, f].flatten(1).sum(dim=1) > 0).nonzero(as_tuple=True)[0]
        if visible.numel() == 0:
            continue
        pm, cl = drop_empty_masks(pred_masks[:, f], class_logits, mask_threshold)
        rows.append(compute_instance_segmentation_metrics(
            pred_masks=pm,
            class_logits=cl,
            gt_masks=gt_masks[visible, f],
            gt_classes=gt_classes[visible],
            score_threshold=score_threshold,
            background_class=background_class,
            score_mode=score_mode,
            mask_threshold=mask_threshold,
        ))
    return rows


def mean_rows(rows: List[Dict[str, float]], keys: List[str] = None) -> Dict[str, float]:
    """Mean of each metric across per-frame rows (zeros if there were no scorable frames)."""
    keys = keys or METRIC_KEYS
    if not rows:
        return {k: 0.0 for k in keys}
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}
