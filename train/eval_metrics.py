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

`multiview_consistency_metrics` (docs/MASKDINO.md §6.6) is the exception: it needs the frame
axis kept separate, because it measures whether ONE query explains an instance in EVERY view.
`tracking_consistency_metrics` answers that same question with the FORMAL metrics of the
tracking literature — HOTA / AssA / DetA / IDF1, views read as timesteps — which is what the
outward-facing consistency claim is quoted on; the pair above stays as the internal diagnostic.
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


# The keys `multiview_consistency_metrics` returns (the eval prefixes them with `bundle_`).
CONSISTENCY_KEYS = ["view_consistency", "id_switch", "num_matched"]


@torch.no_grad()
def multiview_consistency_metrics(
    pred_masks: torch.Tensor,
    gt_masks: torch.Tensor,
    mask_threshold: float = 0.5,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Cross-view consistency of shared queries (docs/MASKDINO.md §6.6, RELATED_WORK.md gap 2).

    Makes "3D consistent" a measured claim: `bundle_AP50` already rewards a query whose mask
    VOLUME matches a GT instance, but a volume can be right on average while a *different* query
    owns the object in each view. These two numbers separate those cases.

    Each query is matched to a GT instance ONCE, at bundle level — class-agnostic Hungarian on
    the IoU of the flattened [S*h*w] volumes, i.e. exactly the assignment `class_acc` uses, one
    dimension larger. Then, for every matched pair (q, g) and every view where g is visible:

      - view_consistency: fraction of those views where IoU(q, g) in that view >= `iou_threshold`
        (0.5). 1.0 = the bundle-matched query segments the instance in every view it appears in.
      - id_switch: fraction of those views where the *best-IoU* query is not q. 0.0 = no view is
        better explained by some other query. Views where NO query overlaps the instance are
        excluded from this fraction (nothing owns the instance there, so nothing switched) —
        that failure mode is what view_consistency counts.

    Both are means over the matched GT instances, so they are recall-flavoured (an instance no
    query overlaps at all is simply not matched and does not enter either mean). The prediction
    set is whatever the caller passes; `train/maskdino_eval.py` passes the threshold-free bundle
    pool, so the numbers do not depend on `--score_threshold`.

    Args:
        pred_masks (torch.Tensor): [N_pred, S, h, w] mask LOGITS (the frame axis must be axis 1).
        gt_masks (torch.Tensor): [N_gt, S, h, w] binary GT volumes, all-zero where not visible.
        mask_threshold (float): probability threshold to binarize predicted masks.
        iou_threshold (float): per-view IoU a query must reach to "explain" the instance there.

    Returns:
        dict with keys `view_consistency`, `id_switch`, `num_matched`. Degenerate cases (no GT,
        no predictions, nothing matched) return all-zeros — read them next to `num_matched`,
        since a zero `id_switch` there means "undefined", not "perfect".
    """
    empty = {"view_consistency": 0.0, "id_switch": 0.0, "num_matched": 0.0}
    if pred_masks.shape[0] == 0 or gt_masks.shape[0] == 0:
        return empty

    s = pred_masks.shape[1]
    pred_bin = (torch.sigmoid(pred_masks) > mask_threshold).flatten(2)   # [N, S, K]
    gt_bin = (gt_masks > 0.5).flatten(2)                                 # [N_gt, S, K]

    # --- one assignment for the whole bundle (the [S*h*w] volume IoU) -------------------------
    vol_iou = mask_iou_matrix(pred_bin.flatten(1), gt_bin.flatten(1))    # [N, N_gt]
    pi, gi = linear_sum_assignment((-vol_iou).cpu().numpy())
    pairs = [(int(p), int(g)) for p, g in zip(pi, gi) if vol_iou[p, g].item() > 0]
    if not pairs:
        return empty

    view_iou = torch.stack([mask_iou_matrix(pred_bin[:, f], gt_bin[:, f]) for f in range(s)])
    visible = gt_bin.any(dim=2)                                          # [N_gt, S]

    consistency, switches = [], []
    for p, g in pairs:
        views = visible[g].nonzero(as_tuple=True)[0]
        if views.numel() == 0:
            continue                       # matched to an instance visible nowhere: undefined
        ious = view_iou[views][:, :, g]                                  # [V, N]
        consistency.append(float((ious[:, p] >= iou_threshold).float().mean().item()))
        best_iou, best_q = ious.max(dim=1)
        covered = best_iou > 0
        switches.append(float((best_q[covered] != p).float().mean().item())
                        if bool(covered.any()) else 0.0)
    if not consistency:
        return empty
    return {"view_consistency": float(np.mean(consistency)),
            "id_switch": float(np.mean(switches)),
            "num_matched": float(len(consistency))}


# The keys `tracking_consistency_metrics` returns (the eval prefixes them with `bundle_`).
TRACKING_KEYS = ["hota", "assa", "deta", "idf1", "num_gt_tracks", "num_pred_tracks"]

# The standard HOTA localisation sweep (Luiten et al., IJCV 2021): alpha = 0.05 : 0.05 : 0.95.
HOTA_ALPHAS = np.arange(0.05, 0.99, 0.05)


def _per_view_presence_and_similarity(pred_masks: torch.Tensor, gt_masks: torch.Tensor,
                                      mask_threshold: float):
    """
    Turn a bundle into the (detections, similarities) form the tracking metrics are defined on.

    A bundle's S views are the S timesteps of a sequence, each GT instance is a ground-truth
    trajectory, and each shared query is a predicted track — the mapping is exact here because a
    query IS one identity across the whole bundle by construction (docs/MASKDINO.md §8), so no
    tracker association step has to be invented to score us.

    Returns:
        present_gt (np.ndarray): [N_gt, S] bool — the instance has pixels in that view
        present_pred (np.ndarray): [N_pred, S] bool
        sims (list[np.ndarray]): per view, the [n_gt_t, n_pred_t] IoU of the present ones only
    """
    pred_bin = (torch.sigmoid(pred_masks) > mask_threshold).flatten(2)    # [N, S, K]
    gt_bin = (gt_masks > 0.5).flatten(2)                                  # [N_gt, S, K]
    present_pred = pred_bin.any(dim=2).cpu().numpy()                      # [N, S]
    present_gt = gt_bin.any(dim=2).cpu().numpy()                          # [N_gt, S]

    sims = []
    for t in range(pred_bin.shape[1]):
        g = np.nonzero(present_gt[:, t])[0]
        p = np.nonzero(present_pred[:, t])[0]
        if g.size == 0 or p.size == 0:
            sims.append(np.zeros((g.size, p.size)))
            continue
        iou = mask_iou_matrix(pred_bin[p, t], gt_bin[g, t])               # [n_p, n_g]
        sims.append(iou.t().cpu().numpy())                                # [n_g, n_p]
    return present_gt, present_pred, sims


@torch.no_grad()
def tracking_consistency_metrics(pred_masks: torch.Tensor,
                                 gt_masks: torch.Tensor,
                                 mask_threshold: float = 0.5,
                                 idf1_threshold: float = 0.5) -> Dict[str, float]:
    """
    Cross-view identity scored with the FORMAL metrics of the tracking literature.

    `multiview_consistency_metrics` above answers the same question with project-defined numbers.
    Those have no published counterpart — SegVGGT, FAST3DIS and IGGT report no cross-view
    consistency metric at all — so the claim "consistency is intrinsic to the query" is stated
    here in the vocabulary a reviewer already has, with the bundle's views read as timesteps:

      - HOTA = sqrt(DetA x AssA), averaged over the standard alpha sweep (Luiten et al., IJCV
        2021). The single number.
      - AssA : association accuracy — the formal counterpart of `id_switch`. It asks, over all
        matched detections, how much of each identity's trajectory the same track explains.
      - DetA : detection accuracy — the formal counterpart of `view_consistency`.
      - IDF1 : identity F1 of the globally optimal one-to-one identity assignment
        (Ristani et al., ECCVW 2016), the MOTChallenge identity metric.

    Both follow the TrackEval reference implementation: HOTA matches per view under the global
    alignment score (so a locally better but identity-breaking match is not rewarded), IDF1
    matches identities once for the whole bundle.

    Args:
        pred_masks (torch.Tensor): [N_pred, S, h, w] mask LOGITS (frame axis must be axis 1).
        gt_masks (torch.Tensor): [N_gt, S, h, w] binary GT volumes, all-zero where not visible.
        mask_threshold (float): probability threshold to binarize predicted masks.
        idf1_threshold (float): the IoU at which IDF1 calls a detection a match (0.5, standard).

    Returns:
        dict with keys `hota`, `assa`, `deta`, `idf1`, `num_gt_tracks`, `num_pred_tracks`.
        Degenerate bundles (no GT, no predictions) return all-zeros — read them next to the two
        track counts, since a zero there means "undefined", not "perfect".
    """
    empty = {k: 0.0 for k in TRACKING_KEYS}
    if pred_masks.shape[0] == 0 or gt_masks.shape[0] == 0:
        return empty

    present_gt, present_pred, sims = _per_view_presence_and_similarity(
        pred_masks, gt_masks, mask_threshold)
    n_gt, n_pred = present_gt.shape[0], present_pred.shape[0]
    num_gt_dets = int(present_gt.sum())
    num_pred_dets = int(present_pred.sum())
    if num_gt_dets == 0 or num_pred_dets == 0:
        return empty

    counts = {"num_gt_tracks": float((present_gt.any(axis=1)).sum()),
              "num_pred_tracks": float((present_pred.any(axis=1)).sum())}

    # --- pass 1: the global alignment score, over the whole bundle -----------------------------
    # How much of the two identities' lifetimes co-occur, in the Jaccard sense. HOTA uses it to
    # break per-view ties in favour of the assignment that keeps identities whole.
    potential = np.zeros((n_gt, n_pred))
    gt_count = np.zeros((n_gt, 1))
    pred_count = np.zeros((1, n_pred))
    for t, sim in enumerate(sims):
        g = np.nonzero(present_gt[:, t])[0]
        p = np.nonzero(present_pred[:, t])[0]
        gt_count[g] += 1
        pred_count[0, p] += 1
        if sim.size:
            potential[np.ix_(g, p)] += sim
    alignment = potential / np.maximum(1.0, gt_count + pred_count - potential)

    # --- pass 2: one matching per alpha ---------------------------------------------------------
    deta, assa, hota = [], [], []
    for alpha in HOTA_ALPHAS:
        matches = np.zeros((n_gt, n_pred))
        tp = 0
        for t, sim in enumerate(sims):
            if sim.size == 0:
                continue
            g = np.nonzero(present_gt[:, t])[0]
            p = np.nonzero(present_pred[:, t])[0]
            score = alignment[np.ix_(g, p)] * sim
            rows, cols = linear_sum_assignment(-score)
            keep = sim[rows, cols] >= alpha - np.finfo(float).eps
            rows, cols = rows[keep], cols[keep]
            matches[g[rows], p[cols]] += 1
            tp += int(rows.size)
        fn, fp = num_gt_dets - tp, num_pred_dets - tp
        d = tp / max(1.0, tp + fn + fp)
        # AssA: each matched detection contributes its identity pair's trajectory-level Jaccard.
        ass_iou = matches / np.maximum(1.0, gt_count + pred_count - matches)
        a = float((matches * ass_iou).sum() / max(1.0, tp))
        deta.append(d)
        assa.append(a)
        hota.append(float(np.sqrt(d * a)))

    # --- IDF1: one identity assignment for the whole bundle -------------------------------------
    hits = np.zeros((n_gt, n_pred))
    for t, sim in enumerate(sims):
        if sim.size == 0:
            continue
        g = np.nonzero(present_gt[:, t])[0]
        p = np.nonzero(present_pred[:, t])[0]
        hits[np.ix_(g, p)] += (sim >= idf1_threshold)
    # Square cost matrix over real + dummy identities, as in the MOTChallenge formulation: an
    # unmatched GT track costs all its detections as FN, an unmatched predicted track as FP.
    size = n_gt + n_pred
    fn_mat = np.zeros((size, size))
    fp_mat = np.zeros((size, size))
    fn_mat[:n_gt, :] = gt_count
    fp_mat[:, :n_pred] = pred_count
    fn_mat[:n_gt, :n_pred] -= hits
    fp_mat[:n_gt, :n_pred] -= hits
    rows, cols = linear_sum_assignment(fn_mat + fp_mat)
    id_fn = float(fn_mat[rows, cols].sum())
    id_fp = float(fp_mat[rows, cols].sum())
    id_tp = num_gt_dets - id_fn
    idf1 = id_tp / max(1e-9, id_tp + 0.5 * id_fn + 0.5 * id_fp)

    return {"hota": float(np.mean(hota)), "assa": float(np.mean(assa)),
            "deta": float(np.mean(deta)), "idf1": float(idf1), **counts}


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
    score_mode: str = "softmax",
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
        score_mode (str): how `class_logits` become per-query (label, score).
            "softmax" (default, every D4RT arm): scores = softmax probs, so "is this an object?"
            is decided by argmax != background_class.
            "sigmoid" (MaskDINO trial, docs/MASKDINO.md §6): scores = per-class sigmoid
            probabilities — there is no background column, so objectness comes purely from
            `score_threshold`. Callers pass a logits tensor whose background column is -inf.

    Returns:
        dict with keys: mIoU, AP50, AP75, mAP, class_acc, num_pred, num_gt.
    """
    if iou_thresholds is None:
        iou_thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]  # 0.50 .. 0.95

    n_gt = gt_masks.shape[0]
    # flatten(1), not reshape(n, -1): the latter raises on a zero-row tensor (which callers can
    # legitimately produce, e.g. after filtering out predictions that claim no pixels).
    pred_masks = pred_masks.flatten(1)
    gt_flat = (gt_masks.flatten(1) > 0.5)

    # Predicted label = argmax class; score = its softmax (or sigmoid) probability.
    if score_mode == "softmax":
        probs = torch.softmax(class_logits, dim=-1)
    elif score_mode == "sigmoid":
        probs = torch.sigmoid(class_logits)
    else:
        raise ValueError(f"score_mode must be 'softmax' or 'sigmoid', got {score_mode!r}")
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
