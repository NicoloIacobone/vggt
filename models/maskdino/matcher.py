"""
Hungarian matcher for the MaskDINO trial — port of `maskdino/modeling/matcher.py`.

Cost = focal class + point-sampled mask BCE + point-sampled Dice + box L1 + GIoU. The mask cost
is evaluated on `num_points` random points shared by all (query, GT) pairs, which is what makes
matching over 300 queries cheap.
"""

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from .utils import point_sample


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    inputs = inputs.sigmoid().flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    return 1 - (numerator + 1) / (denominator + 1)


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    hw = inputs.shape[1]
    pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction="none")
    neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction="none")
    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum("nc,mc->nm", neg, (1 - targets))
    return loss / hw


class HungarianMatcher(nn.Module):
    """One-to-one assignment between the decoder's queries and the frame's GT instances."""

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
    def forward(self, outputs, targets, cost=("cls", "box", "mask")):
        """
        Args:
            outputs: dict with pred_logits [B, Q, C], pred_masks [B, Q, h, w], pred_boxes [B, Q, 4].
            targets: list of B dicts with labels [n], masks [n, h, w], boxes [n, 4].
        Returns:
            list of B (pred_idx, tgt_idx) int64 tensor pairs.
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        indices = []

        for b in range(bs):
            tgt_ids = targets[b]["labels"]
            if tgt_ids.numel() == 0:
                indices.append((torch.as_tensor([], dtype=torch.int64),
                                torch.as_tensor([], dtype=torch.int64)))
                continue

            out_prob = outputs["pred_logits"][b].sigmoid()  # [Q, C]
            alpha, gamma = 0.25, 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            if "box" in cost and "pred_boxes" in outputs:
                out_bbox = outputs["pred_boxes"][b]
                tgt_bbox = targets[b]["boxes"]
                cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
                cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                                 box_cxcywh_to_xyxy(tgt_bbox))
            else:
                cost_bbox = cost_giou = torch.zeros_like(cost_class)

            if "mask" in cost and outputs.get("pred_masks") is not None:
                out_mask = outputs["pred_masks"][b][:, None]              # [Q, 1, h, w]
                tgt_mask = targets[b]["masks"].to(out_mask)[:, None]      # [n, 1, h, w]
                # one shared set of random points for every pair — the matching-cost trick
                point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
                tgt_mask = point_sample(tgt_mask, point_coords.repeat(tgt_mask.shape[0], 1, 1),
                                        align_corners=False).squeeze(1).float()
                out_mask = point_sample(out_mask, point_coords.repeat(out_mask.shape[0], 1, 1),
                                        align_corners=False).squeeze(1).float()
                cost_mask = batch_sigmoid_ce_loss(out_mask, tgt_mask)
                cost_dice = batch_dice_loss(out_mask, tgt_mask)
            else:
                cost_mask = cost_dice = torch.zeros_like(cost_class)

            C = (self.cost_mask * cost_mask + self.cost_class * cost_class
                 + self.cost_dice * cost_dice + self.cost_box * cost_bbox
                 + self.cost_giou * cost_giou)
            C = C.reshape(num_queries, -1).cpu()
            # A non-finite entry would make scipy raise and kill the run; the D4RT matcher hit
            # this in the hybrid arm, so guard here too (warn-free: NaN → 0 keeps assignment sane).
            if not torch.isfinite(C).all():
                C = torch.nan_to_num(C, nan=0.0, posinf=1e6, neginf=-1e6)
            i, j = linear_sum_assignment(C)
            indices.append((torch.as_tensor(i, dtype=torch.int64),
                            torch.as_tensor(j, dtype=torch.int64)))

        return indices

    def __repr__(self):
        return (f"HungarianMatcher(cost_class={self.cost_class}, cost_mask={self.cost_mask}, "
                f"cost_dice={self.cost_dice}, cost_box={self.cost_box}, "
                f"cost_giou={self.cost_giou}, num_points={self.num_points})")
