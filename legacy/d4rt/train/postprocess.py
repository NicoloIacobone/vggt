"""
Shared inference-time post-processing: turn raw D4RT head outputs into a single,
deduplicated set of instances for visualization.

Both the 2D overlay renderer (`legacy/d4rt/scripts/visualize_masks.py`) and the 3D Gradio viewer
(`demos/demo_gradio.py`) MUST select instances the same way, otherwise the two views
disagree about which queries become objects (the exact bug where a chair showed up in 3D
but not in 2D). `select_instances` is the single source of truth for that selection:

  - filter queries whose argmax class is background (index 0) or whose class score < score_thr
    (default 0.5 — the same operating point as the AP50 metric and the 2D overlays),
  - resolve overlaps by per-pixel winner-takes-all over the kept instances' mask probabilities,
    computed at the head's NATIVE mask resolution (the patch grid, or the upsampled grid when
    mask_upsample > 1) so callers can nearest-upsample the integer assignment and get identical,
    resolution-honest masks in both 2D and 3D.

This deliberately does NOT use ground truth or any query ordering assumption, so it is valid
for point / learned / hybrid query modes alike.
"""

from typing import List, Tuple

import torch
import torch.nn.functional as F


def select_instances(
    class_logits: torch.Tensor,
    pred_masks: torch.Tensor,
    score_thr: float = 0.5,
    mask_thr: float = 0.5,
    bg_index: int = 0,
) -> Tuple[List[int], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Select and deduplicate predicted instances (winner-takes-all).

    Args:
        class_logits: [N, C] raw class logits for the N queries (no batch dim).
        pred_masks:   [N, S, h, w] raw mask logits (no batch dim), h/w = native mask resolution.
        score_thr:    min softmax class confidence for a query to count as an instance.
        mask_thr:     min sigmoid mask probability for a pixel to be claimed by any instance.
        bg_index:     class index treated as background (dropped). ScanNet convention: 0.

    Returns:
        keep:   list of kept query indices (into 0..N-1), ordered by descending class score so
                the most confident instance wins ties and gets the first color.
        labels: [N] argmax class per query.
        scores: [N] max softmax prob per query.
        assign: [S, h, w] long tensor at NATIVE resolution; value c indexes into `keep`
                (i.e. the c-th kept instance owns that pixel), -1 = background/unclaimed.
    """
    probs = torch.softmax(class_logits, dim=-1)   # [N, C]
    labels = probs.argmax(dim=-1)                  # [N]
    scores = probs.max(dim=-1).values              # [N]

    candidates = [
        i for i in range(class_logits.shape[0])
        if int(labels[i]) != bg_index and float(scores[i]) >= score_thr
    ]
    # Most confident first: ties in the winner-takes-all go to the higher-scoring instance.
    keep = sorted(candidates, key=lambda i: float(scores[i]), reverse=True)

    S, h, w = pred_masks.shape[1], pred_masks.shape[2], pred_masks.shape[3]
    assign = torch.full((S, h, w), -1, dtype=torch.long, device=pred_masks.device)
    if keep:
        mask_prob = torch.sigmoid(pred_masks)  # [N, S, h, w]
        best = torch.full((S, h, w), float(mask_thr), device=pred_masks.device)
        for c, i in enumerate(keep):
            pv = mask_prob[i]                  # [S, h, w]
            better = pv > best
            best[better] = pv[better]
            assign[better] = c
    return keep, labels, scores, assign


def upsample_assignment(assign: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """
    Nearest-neighbour upsample a per-pixel integer assignment map to (H, W).

    Nearest (not bilinear) keeps instance labels intact and is honest about the true
    patch-grid resolution — the same mode used for the GT masks, so GT and predictions are
    rendered at matching sharpness.

    Args:
        assign: [S, h, w] long tensor from `select_instances`.
        size:   target (H, W).

    Returns:
        [S, H, W] long tensor.
    """
    up = F.interpolate(assign[:, None].float(), size=size, mode="nearest")[:, 0]
    return up.round().long()
