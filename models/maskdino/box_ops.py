"""
Box utilities for the MaskDINO trial (docs/MASKDINO.md).

Ported from MaskDINO's `maskdino/utils/box_ops.py` (itself from DETR), plus a local
`masks_to_boxes` that replaces detectron2's `BitMasks.get_bounding_boxes` (not installed here).
All boxes are float tensors; `cxcywh` boxes are normalized to [0, 1] by the image size.
"""

import torch


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    x_c, y_c, w, h = x.unbind(-1)
    return torch.stack([(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)], dim=-1)


def box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    return torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)], dim=-1)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """Area of xyxy boxes."""
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6), union


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Generalized IoU (https://giou.stanford.edu/) between two sets of xyxy boxes → [N, M].

    Degenerate boxes (x1 < x0 or y1 < y0) would give nonsense; upstream DETR asserts on them.
    Our boxes come from sigmoid outputs / masks, so we clamp instead of asserting: a run must
    never die on a transient bad box.
    """
    boxes1 = torch.cat([boxes1[:, :2], torch.maximum(boxes1[:, 2:], boxes1[:, :2])], dim=-1)
    boxes2 = torch.cat([boxes2[:, :2], torch.maximum(boxes2[:, 2:], boxes2[:, :2])], dim=-1)

    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area.clamp(min=1e-6)


def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """
    Tight xyxy boxes (in PIXEL units of the mask grid) around binary masks [N, H, W].

    Replaces `detectron2.structures.BitMasks.get_bounding_boxes` (MaskDINO's
    `initialize_box_type='bitmask'`) and DETR's `masks_to_boxes` in one function. Empty masks
    get a zero box, matching BitMasks' behaviour.
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device, dtype=torch.float32)

    m = masks.bool()
    h, w = m.shape[-2:]
    y = torch.arange(h, dtype=torch.float32, device=m.device)
    x = torch.arange(w, dtype=torch.float32, device=m.device)

    x_mask = m * x[None, None, :]
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~m, 1e8).flatten(1).min(-1)[0]

    y_mask = m * y[None, :, None]
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~m, 1e8).flatten(1).min(-1)[0]

    boxes = torch.stack([x_min, y_min, x_max + 1, y_max + 1], dim=1)
    empty = ~m.flatten(1).any(-1)
    boxes[empty] = 0.0
    return boxes


def masks_to_boxes_normalized(masks: torch.Tensor) -> torch.Tensor:
    """`masks_to_boxes` in normalized cxcywh — the format the decoder/criterion use."""
    h, w = masks.shape[-2:]
    boxes = masks_to_boxes(masks)
    scale = torch.as_tensor([w, h, w, h], dtype=torch.float32, device=masks.device)
    return box_xyxy_to_cxcywh(boxes / scale).clamp(0.0, 1.0)
