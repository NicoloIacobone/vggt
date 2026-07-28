"""
Set criterion for the MaskDINO trial — port of `maskdino/modeling/criterion.py`.

Losses (weights from MaskDINO's COCO instance config): sigmoid-focal classification (4.0),
point-sampled mask BCE (5.0) + Dice (5.0), box L1 (5.0) + GIoU (2.0). Applied to

  - the final decoder layer,
  - every intermediate layer + the initial prediction (deep supervision, `aux_outputs`),
  - the two-stage encoder prediction (`interm_outputs`),
  - the denoising queries (`*_dn`, matched by construction rather than by Hungarian).

Differences from upstream: no distributed all-reduce of `num_masks` (single-GPU runs here), no
detectron2 point-sampling import (local copies in `.utils`), and every mask/box loss is guarded
against an empty match set so a frame with no GT can never produce a NaN.
"""

from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn

from . import box_ops
from .utils import (calculate_uncertainty, cat_matched,
                    get_uncertain_point_coords_with_randomness, point_sample)


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean(1).sum() / num_boxes


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float):
    inputs = inputs.sigmoid().flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    return (1 - (numerator + 1) / (denominator + 1)).sum() / num_masks


def sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float):
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.mean(1).sum() / num_masks


class SetCriterion(nn.Module):
    """
    Args:
        num_classes: foreground classes (no background column — DINO's sigmoid convention).
        matcher: `HungarianMatcher`.
        weight_dict: loss-name → weight, including the `_i` (aux), `_interm` and `_dn` suffixes.
        losses: which of "labels" / "masks" / "boxes" to compute for the matched queries.
        num_points / oversample_ratio / importance_sample_ratio: PointRend mask-loss sampling.
            `num_points <= 0` uses every pixel of the mask grid instead (cheap at 37x37 and
            removes the sampling noise; MaskDINO needs sampling only because COCO masks are big).
        dn: "no" | "seg"; dn_losses: which losses to apply to the denoising queries.
    """

    def __init__(self, num_classes: int, matcher, weight_dict: Dict[str, float],
                 losses: List[str], num_points: int = 12544, oversample_ratio: float = 3.0,
                 importance_sample_ratio: float = 0.75, dn: str = "no",
                 dn_losses: List[str] = (), focal_alpha: float = 0.25):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = list(losses)
        self.dn = dn
        self.dn_losses = list(dn_losses)
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.focal_alpha = focal_alpha

    # ---- individual losses ---------------------------------------------------------------

    def loss_labels(self, outputs, targets, indices, num_boxes):
        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = cat_matched(targets, indices, "labels")
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        if target_classes_o.numel() > 0:
            target_classes[idx] = target_classes_o

        # one-hot with an extra "no object" column that is then dropped: an unmatched query's
        # target is the all-zeros vector (DINO/Deformable-DETR convention).
        onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                             dtype=src_logits.dtype, device=src_logits.device)
        onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        onehot = onehot[:, :, :-1]

        loss_ce = sigmoid_focal_loss(src_logits, onehot, num_boxes,
                                     alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        return {"loss_ce": loss_ce}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = cat_matched(targets, indices, "boxes")
        if src_boxes.numel() == 0:
            zero = outputs["pred_boxes"].sum() * 0.0
            return {"loss_bbox": zero, "loss_giou": zero}

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none").sum() / num_boxes
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes), box_ops.box_cxcywh_to_xyxy(target_boxes)))
        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou.sum() / num_boxes}

    def loss_masks(self, outputs, targets, indices, num_masks):
        src_idx = self._get_src_permutation_idx(indices)
        src_masks = outputs["pred_masks"][src_idx]
        target_masks = cat_matched(targets, indices, "masks").to(src_masks)
        if src_masks.numel() == 0:
            zero = outputs["pred_masks"].sum() * 0.0
            return {"loss_mask": zero, "loss_dice": zero}

        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        if self.num_points > 0:
            with torch.no_grad():
                point_coords = get_uncertain_point_coords_with_randomness(
                    src_masks, calculate_uncertainty, self.num_points,
                    self.oversample_ratio, self.importance_sample_ratio)
                point_labels = point_sample(target_masks, point_coords,
                                            align_corners=False).squeeze(1)
            point_logits = point_sample(src_masks, point_coords, align_corners=False).squeeze(1)
        else:
            # dense: the mask grid is small enough that sampling buys nothing
            point_logits = src_masks.flatten(1)
            point_labels = target_masks.flatten(1)

        return {"loss_mask": sigmoid_ce_loss(point_logits, point_labels, num_masks),
                "loss_dice": dice_loss(point_logits, point_labels, num_masks)}

    # ---- plumbing ------------------------------------------------------------------------

    @staticmethod
    def _get_src_permutation_idx(indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {"labels": self.loss_labels, "masks": self.loss_masks, "boxes": self.loss_boxes}
        assert loss in loss_map, f"unknown loss {loss}"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def prep_for_dn(self, mask_dict):
        output_known_lbs_bboxes = mask_dict["output_known_lbs_bboxes"]
        scalar, pad_size = mask_dict["scalar"], mask_dict["pad_size"]
        assert pad_size % scalar == 0
        single_pad = pad_size // scalar
        num_tgt = mask_dict["known_indice"].numel()
        return output_known_lbs_bboxes, num_tgt, single_pad, scalar

    def forward(self, outputs, targets, mask_dict=None):
        """Returns the (unweighted) loss dict; multiply by `weight_dict` to get the total."""
        device = outputs["pred_logits"].device
        outputs_without_aux = {k: v for k, v in outputs.items()
                               if k not in ("aux_outputs", "interm_outputs")}

        exc_idx = []
        scalar = 1
        if self.dn != "no" and mask_dict is not None:
            output_known_lbs_bboxes, _, single_pad, scalar = self.prep_for_dn(mask_dict)
            for i in range(len(targets)):
                n = len(targets[i]["labels"])
                if n > 0:
                    t = torch.arange(0, n, device=device).long().unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = ((torch.arange(scalar, device=device) * single_pad).long()
                                  .unsqueeze(1) + t).flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([], device=device).long()
                exc_idx.append((output_idx, tgt_idx))

        indices = self.matcher(outputs_without_aux, targets)
        num_masks = max(1.0, float(sum(len(t["labels"]) for t in targets)))

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        if self.dn != "no" and mask_dict is not None:
            l_dict = {}
            for loss in self.dn_losses:
                l_dict.update(self.get_loss(loss, output_known_lbs_bboxes, targets, exc_idx,
                                            num_masks * scalar))
            losses.update({k + "_dn": v for k, v in l_dict.items()})
        elif self.dn != "no":
            losses.update(self._zero_dn_losses(device))

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                aux_indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, aux_indices, num_masks)
                    losses.update({f"{k}_{i}": v for k, v in l_dict.items()})
                start = 0 if "interm_outputs" in outputs else 1
                if i >= start:
                    if self.dn != "no" and mask_dict is not None:
                        out_ = output_known_lbs_bboxes["aux_outputs"][i]
                        l_dict = {}
                        for loss in self.dn_losses:
                            l_dict.update(self.get_loss(loss, out_, targets, exc_idx,
                                                        num_masks * scalar))
                        losses.update({f"{k}_dn_{i}": v for k, v in l_dict.items()})
                    elif self.dn != "no":
                        losses.update(self._zero_dn_losses(device, suffix=f"_{i}"))

        if "interm_outputs" in outputs:
            interm_indices = self.matcher(outputs["interm_outputs"], targets)
            for loss in self.losses:
                l_dict = self.get_loss(loss, outputs["interm_outputs"], targets, interm_indices,
                                       num_masks)
                losses.update({f"{k}_interm": v for k, v in l_dict.items()})

        return losses

    def _zero_dn_losses(self, device, suffix: str = ""):
        """Placeholder zeros so the logged loss keys are stable when a step has no DN group."""
        z = torch.as_tensor(0.0, device=device)
        out = {f"loss_bbox_dn{suffix}": z, f"loss_giou_dn{suffix}": z, f"loss_ce_dn{suffix}": z}
        if self.dn == "seg":
            out[f"loss_mask_dn{suffix}"] = z
            out[f"loss_dice_dn{suffix}"] = z
        return out


def build_weight_dict(class_weight=4.0, mask_weight=5.0, dice_weight=5.0, box_weight=5.0,
                      giou_weight=2.0, dec_layers=9, two_stage=True, dn="seg",
                      deep_supervision=True) -> Dict[str, float]:
    """
    The MaskDINO weight dict: base weights replicated over aux layers (`_i`), the encoder's
    interm output (`_interm`) and the denoising group (`_dn`, `_dn_i`).

    `dec_layers` counts decoder layers; with `initial_pred` there are `dec_layers + 1`
    predictions, i.e. `dec_layers` aux entries plus the final one.
    """
    base = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight,
            "loss_bbox": box_weight, "loss_giou": giou_weight}
    weight_dict = dict(base)

    if dn != "no":
        dn_base = {"loss_ce_dn": class_weight, "loss_bbox_dn": box_weight,
                   "loss_giou_dn": giou_weight}
        if dn == "seg":
            dn_base.update({"loss_mask_dn": mask_weight, "loss_dice_dn": dice_weight})
        weight_dict.update(dn_base)

    if deep_supervision:
        for i in range(dec_layers):
            weight_dict.update({f"{k}_{i}": v for k, v in base.items()})
            if dn != "no":
                # dn_base keys already end in `_dn`, so this yields loss_ce_dn_0, ...
                weight_dict.update({f"{k}_{i}": v for k, v in dn_base.items()})
    if two_stage:
        weight_dict.update({f"{k}_interm": v for k, v in base.items()})
    return weight_dict
