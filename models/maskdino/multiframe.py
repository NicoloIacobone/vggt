"""
Multi-frame extension of the MaskDINO track: ONE query set per bundle of S frames
(docs/MASKDINO.md §8, step 2).

The single-frame protocol treats every frame as an independent image, so query 7 in frame 0 and
query 7 in frame 1 mean nothing to each other. Here the query set is *shared* across the frames
of a bundle:

  - the decoder's initial content queries are selected once per bundle and broadcast to every
    frame (`MaskDINODecoder.forward(..., frames_per_sample=S)`), while each frame keeps its own
    anchor box and refines it independently — a query is one 3D instance seen from S viewpoints;
  - a `CrossFrameAttention` block after every decoder layer lets the S copies of a query talk to
    each other, which is what actually ties the identity together;
  - matching happens ONCE per bundle over the concatenated mask volume
    (`MultiFrameHungarianMatcher`), so the Hungarian assignment itself enforces "query q = the
    same instance in every view". The per-frame losses are then applied through
    `expand_bundle_indices`, unchanged.

Consequence for evaluation: the multi-view (per-bundle) metric of the retired D4RT arms becomes
meaningful again, because a query now owns a mask *volume* — see `train/maskdino_eval.py`.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from .matcher import batch_dice_loss, batch_sigmoid_ce_loss, check_target_labels

# ------------------------------------------------------------------------------------------
# The block that ties a query's identity across views
# ------------------------------------------------------------------------------------------


class CrossFrameAttention(nn.Module):
    """
    Self-attention across the S frame copies of each shared query (one block per decoder layer).

    The attention runs over a sequence of length S — the same query index in every frame of the
    bundle — with batch = (queries x bundles). There is deliberately **no frame positional
    encoding**: the frames of a bundle are an unordered set of views, so the block must be
    permutation-equivariant in S.

    Denoising queries (the `pad_size` slots at the FRONT of the query dimension) are excluded:
    they are per-frame reconstruction targets whose slot i means a different GT instance in each
    frame, so mixing them across frames would both leak and confuse. `num_shared` says how many
    trailing queries take part.
    """

    def __init__(self, d_model: int = 256, nheads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nheads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt: Tensor, frames_per_sample: int, num_shared: Optional[int] = None
                ) -> Tensor:
        """tgt: [nq, bs, d] with bs = B*S and the S frames of a bundle CONTIGUOUS."""
        s = frames_per_sample
        if s <= 1:
            return tgt
        nq, bs, d = tgt.shape
        assert bs % s == 0, f"batch {bs} is not a multiple of frames_per_sample {s}"
        b = bs // s
        n_shared = nq if num_shared is None else min(num_shared, nq)
        head, shared = tgt[:nq - n_shared], tgt[nq - n_shared:]

        x = shared.view(n_shared, b, s, d).permute(2, 0, 1, 3).reshape(s, n_shared * b, d)
        y = self.attn(x, x, x)[0]
        y = y.view(s, n_shared, b, d).permute(1, 2, 0, 3).reshape(n_shared, bs, d)
        shared = self.norm(shared + self.dropout(y))
        return torch.cat([head, shared], dim=0) if head.numel() else shared


# ------------------------------------------------------------------------------------------
# Bundle-level ground truth
# ------------------------------------------------------------------------------------------


def build_bundle_target(frame_targets: Sequence[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    """
    Merge the S per-frame targets of one bundle into one multi-view target.

    The per-frame targets built by `train/maskdino_data.py::build_frame_targets` already carry
    the dataset's GLOBAL instance id, which is exactly the cross-view link the single-frame
    protocol throws away. Re-using it here costs nothing and keeps one GT builder.

    Returns:
        labels     [n]          class 0..C-1
        masks      [n, S, h, w] binary; all-zero in the frames where the instance is not visible
        boxes      [n, S, 4]    normalized cxcywh; zero where not visible
        valid      [n, S]       bool visibility
        frame_row  [n, S]       row of this instance inside frame f's per-frame target (-1 if
                                absent) — what `expand_bundle_indices` needs to reuse the
                                per-frame losses unchanged
        global_ids [n]
    """
    s = len(frame_targets)
    ref = frame_targets[0]
    device = ref["labels"].device
    hw = tuple(ref["masks"].shape[-2:])

    # Instances are ordered by GLOBAL id, not by first appearance, so the bundle target does not
    # depend on the order the frames happen to be drawn in.
    gids = sorted({gid for t in frame_targets for gid in t["global_ids"].tolist()})
    order: Dict[int, int] = {gid: i for i, gid in enumerate(gids)}
    n = len(order)

    labels = torch.zeros(n, dtype=torch.long, device=device)
    masks = torch.zeros(n, s, *hw, device=device, dtype=ref["masks"].dtype)
    boxes = torch.zeros(n, s, 4, device=device, dtype=ref["boxes"].dtype)
    valid = torch.zeros(n, s, dtype=torch.bool, device=device)
    frame_row = torch.full((n, s), -1, dtype=torch.long, device=device)

    for f, t in enumerate(frame_targets):
        for r, gid in enumerate(t["global_ids"].tolist()):
            i = order[gid]
            labels[i] = t["labels"][r]
            masks[i, f] = t["masks"][r]
            boxes[i, f] = t["boxes"][r]
            valid[i, f] = True
            frame_row[i, f] = r

    return {
        "labels": labels, "masks": masks, "boxes": boxes, "valid": valid,
        "frame_row": frame_row,
        "global_ids": torch.as_tensor(gids, dtype=torch.long, device=device),
    }


def bundle_targets_to_device(bundle_targets: List[Dict[str, Tensor]], device: str
                             ) -> List[Dict[str, Tensor]]:
    return [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in bundle_targets]


def expand_bundle_indices(indices: List[Tuple[Tensor, Tensor]],
                          bundle_targets: List[Dict[str, Tensor]],
                          frames_per_sample: int) -> List[Tuple[Tensor, Tensor]]:
    """
    Turn one (query, bundle-instance) assignment per bundle into one (query, row) assignment per
    FRAME, keeping only the frames where the matched instance is actually visible.

    That restriction is the whole point: in a frame where the instance is not visible the query
    stays unmatched, so `loss_labels` supervises it as "no object" and no mask/box loss is
    applied — precisely the behaviour the per-frame evaluation protocol rewards
    (`train/perframe.py::drop_empty_masks`).
    """
    per_frame: List[Tuple[Tensor, Tensor]] = []
    for b, (src, tgt) in enumerate(indices):
        frame_row = bundle_targets[b]["frame_row"]
        device = frame_row.device
        src = src.to(device)
        tgt = tgt.to(device)
        for f in range(frames_per_sample):
            if src.numel() == 0:
                per_frame.append((torch.zeros(0, dtype=torch.int64, device=device),
                                  torch.zeros(0, dtype=torch.int64, device=device)))
                continue
            rows = frame_row[tgt, f]
            keep = rows >= 0
            per_frame.append((src[keep].to(torch.int64), rows[keep].to(torch.int64)))
    return per_frame


# ------------------------------------------------------------------------------------------
# Bundle-level Hungarian matching
# ------------------------------------------------------------------------------------------


class MultiFrameHungarianMatcher(nn.Module):
    """
    One assignment per BUNDLE: query q is matched to a 3D instance, not to a 2D detection.

    Same cost terms and weights as `HungarianMatcher`, aggregated over the S views:
      - class: focal cost on the mean sigmoid score over frames (a query has one class);
      - mask:  BCE + Dice over the CONCATENATED [S*h*w] mask volume, i.e. exactly the multi-view
               mask the retired D4RT arms were matched on;
      - box:   L1 + GIoU per frame, averaged over the frames where the instance is visible.

    `num_points > 0` subsamples the flattened mask volume with one shared random column set for
    every (query, instance) pair — the same trick as the single-frame matcher, applied to the
    volume instead of the frame.
    """

    def __init__(self, cost_class: float = 4.0, cost_mask: float = 5.0, cost_dice: float = 5.0,
                 cost_box: float = 5.0, cost_giou: float = 2.0, num_points: int = 12544):
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.cost_box = cost_box
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"
        self.num_points = num_points

    @torch.no_grad()
    def forward(self, outputs, bundle_targets, frames_per_sample: int):
        """
        Args:
            outputs: pred_logits [B*S, Q, C], pred_masks [B*S, Q, h, w], pred_boxes [B*S, Q, 4]
                     with the S frames of a bundle contiguous in the batch dimension.
            bundle_targets: B dicts from `build_bundle_target`.
        Returns:
            B (query_idx, instance_idx) pairs — one assignment per bundle.
        """
        s = frames_per_sample
        bs, num_queries, num_classes = outputs["pred_logits"].shape
        assert bs % s == 0 and bs // s == len(bundle_targets)
        check_target_labels(bundle_targets, num_classes, where="MultiFrameHungarianMatcher")
        b_size = bs // s

        logits = outputs["pred_logits"].view(b_size, s, num_queries, num_classes)
        masks = outputs.get("pred_masks")
        boxes = outputs.get("pred_boxes")
        if masks is not None:
            masks = masks.view(b_size, s, num_queries, *masks.shape[-2:])
        if boxes is not None:
            boxes = boxes.view(b_size, s, num_queries, 4)

        indices = []
        for b in range(b_size):
            tgt_ids = bundle_targets[b]["labels"]
            if tgt_ids.numel() == 0:
                indices.append((torch.zeros(0, dtype=torch.int64),
                                torch.zeros(0, dtype=torch.int64)))
                continue

            # class: one score per query, averaged over the views
            out_prob = logits[b].sigmoid().mean(0)                       # [Q, C]
            alpha, gamma = 0.25, 2.0
            neg = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos[:, tgt_ids] - neg[:, tgt_ids]

            if masks is not None:
                out_mask = masks[b].permute(1, 0, 2, 3).flatten(1).float()   # [Q, S*h*w]
                tgt_mask = bundle_targets[b]["masks"].flatten(1).to(out_mask)
                if 0 < self.num_points < out_mask.shape[1]:
                    cols = torch.randperm(out_mask.shape[1], device=out_mask.device)
                    cols = cols[:self.num_points]
                    out_mask, tgt_mask = out_mask[:, cols], tgt_mask[:, cols]
                cost_mask = batch_sigmoid_ce_loss(out_mask, tgt_mask)
                cost_dice = batch_dice_loss(out_mask, tgt_mask)
            else:
                cost_mask = cost_dice = torch.zeros_like(cost_class)

            if boxes is not None:
                valid = bundle_targets[b]["valid"].to(cost_class.device)     # [n, S]
                tgt_boxes = bundle_targets[b]["boxes"]                       # [n, S, 4]
                cost_bbox = torch.zeros_like(cost_class)
                cost_giou = torch.zeros_like(cost_class)
                for f in range(s):
                    vis = valid[:, f].to(cost_class.dtype)                   # [n]
                    if float(vis.sum()) == 0:
                        continue
                    l1 = torch.cdist(boxes[b, f], tgt_boxes[:, f], p=1)
                    giou = -generalized_box_iou(box_cxcywh_to_xyxy(boxes[b, f]),
                                                box_cxcywh_to_xyxy(tgt_boxes[:, f]))
                    cost_bbox = cost_bbox + l1 * vis[None]
                    cost_giou = cost_giou + giou * vis[None]
                denom = valid.sum(1).clamp(min=1).to(cost_class.dtype)[None]
                cost_bbox = cost_bbox / denom
                cost_giou = cost_giou / denom
            else:
                cost_bbox = cost_giou = torch.zeros_like(cost_class)

            c = (self.cost_mask * cost_mask + self.cost_class * cost_class
                 + self.cost_dice * cost_dice + self.cost_box * cost_bbox
                 + self.cost_giou * cost_giou)
            c = c.reshape(num_queries, -1).cpu()
            if not torch.isfinite(c).all():
                c = torch.nan_to_num(c, nan=0.0, posinf=1e6, neginf=-1e6)
            i, j = linear_sum_assignment(c)
            indices.append((torch.as_tensor(i, dtype=torch.int64),
                            torch.as_tensor(j, dtype=torch.int64)))
        return indices

    def __repr__(self):
        return (f"MultiFrameHungarianMatcher(cost_class={self.cost_class}, "
                f"cost_mask={self.cost_mask}, cost_dice={self.cost_dice}, "
                f"cost_box={self.cost_box}, cost_giou={self.cost_giou}, "
                f"num_points={self.num_points})")
