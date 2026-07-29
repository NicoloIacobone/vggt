"""
COCO scoring for the backbone-swap study: MaskDINO instance inference + `COCOeval`.

Deliberately **not** `train/perframe.py`. That module implements this project's ScanNet protocol
(drop empty masks, top-k per frame, IoU against per-frame GT) and its numbers are only comparable
to the D4RT arms. Here the whole point is comparability with *upstream MaskDINO*, so the standard
COCO protocol is used verbatim:

  * `instance_inference` as in `MaskDINO.instance_inference`: sigmoid scores flattened over
    (query × class), top 100 per image, the query's score multiplied by its mask's mean
    foreground probability;
  * masks upsampled from the prediction grid straight to the *original* image size — which
    undoes the squash exactly, and is where the grid resolution actually costs AP;
  * `pycocotools` `COCOeval` for `segm` and `bbox`.

The resolution ceiling this protocol implies is measured, GT-only, by
`scripts/coco_mask_resolution_oracle.py`; read that before reading any AP produced here.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def instance_inference(mask_cls: torch.Tensor, mask_pred: torch.Tensor,
                       box_pred: Optional[torch.Tensor], topk: int = 100):
    """
    One image's decoder output → up to `topk` detections (upstream `MaskDINO.instance_inference`).

    Args:
        mask_cls: [Q, C] class logits (sigmoid convention, no background column).
        mask_pred: [Q, h, w] mask logits on the prediction grid.
        box_pred: [Q, 4] cxcywh in [0,1], or None.
    Returns:
        scores [k], labels [k] (contiguous 0..C-1), masks [k, h, w] logits, boxes [k, 4] or None.
    """
    q, c = mask_cls.shape
    scores = mask_cls.sigmoid()
    labels = torch.arange(c, device=mask_cls.device).unsqueeze(0).repeat(q, 1).flatten(0, 1)
    k = min(topk, q * c)
    scores_per_image, topk_idx = scores.flatten(0, 1).topk(k, sorted=False)
    labels_per_image = labels[topk_idx]
    query_idx = torch.div(topk_idx, c, rounding_mode="floor")

    mask_pred = mask_pred[query_idx]
    boxes = box_pred[query_idx] if box_pred is not None else None

    # mask quality folded into the score, exactly as upstream: a query whose mask is confident
    # over its own foreground outranks one that is barely above threshold.
    binary = (mask_pred > 0).flatten(1).float()
    mask_scores = (mask_pred.sigmoid().flatten(1) * binary).sum(1) / (binary.sum(1) + 1e-6)
    return scores_per_image * mask_scores, labels_per_image, mask_pred, boxes


@torch.no_grad()
def predictions_for_image(mask_cls, mask_pred, box_pred, orig_hw, image_id: int,
                          contig2cat: List[int], topk: int = 100,
                          score_threshold: float = 0.0) -> List[Dict]:
    """One image's detections as COCO result dicts (RLE segmentation + xywh box)."""
    from pycocotools import mask as mask_util

    h, w = orig_hw
    scores, labels, masks, boxes = instance_inference(mask_cls, mask_pred, box_pred, topk)

    # Upsample the mask logits to the ORIGINAL image size: the squash is a per-axis linear map,
    # so this inverts it exactly. Thresholding after upsampling (not before) is what makes the
    # bilinear interpolation able to recover sub-cell boundaries — and what the oracle measures.
    masks = F.interpolate(masks[:, None].float(), size=(int(h), int(w)), mode="bilinear",
                          align_corners=False)[:, 0] > 0

    out = []
    scores_l = scores.tolist()
    labels_l = labels.tolist()
    boxes_np = None
    if boxes is not None:
        b = boxes.clone()
        xyxy = torch.stack([(b[:, 0] - b[:, 2] / 2) * w, (b[:, 1] - b[:, 3] / 2) * h,
                            (b[:, 0] + b[:, 2] / 2) * w, (b[:, 1] + b[:, 3] / 2) * h], dim=1)
        boxes_np = torch.stack([xyxy[:, 0], xyxy[:, 1],
                                xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]], dim=1).tolist()

    masks_np = masks.cpu().numpy().astype(np.uint8)
    for i, (s, lab) in enumerate(zip(scores_l, labels_l)):
        if s < score_threshold:
            continue
        rle = mask_util.encode(np.asfortranarray(masks_np[i]))
        rle["counts"] = rle["counts"].decode("ascii")
        d = {"image_id": int(image_id), "category_id": int(contig2cat[lab]),
             "segmentation": rle, "score": float(s)}
        if boxes_np is not None:
            d["bbox"] = [float(x) for x in boxes_np[i]]
        out.append(d)
    return out


@torch.no_grad()
def evaluate_coco(model, loader, dataset, device: str, topk: int = 100,
                  amp_dtype: torch.dtype = torch.bfloat16, max_images: int = 0,
                  verbose: bool = True) -> Dict[str, float]:
    """
    Run the model over `loader` and score with `COCOeval`.

    Returns a flat dict: `segm_AP`, `segm_AP50`, ..., `bbox_AP`, ... (all ×100, COCO convention).
    """
    import contextlib
    import io

    from pycocotools.cocoeval import COCOeval

    model.eval()
    results: List[Dict] = []
    seen_ids: List[int] = []
    n_seen = 0
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        use_amp = device.startswith("cuda") and amp_dtype != torch.float32
        with (torch.autocast("cuda", dtype=amp_dtype) if use_amp else contextlib.nullcontext()):
            out, _ = model(images, targets=None)
        logits = out["pred_logits"].float()
        masks = out["pred_masks"].float()
        boxes = out.get("pred_boxes")
        boxes = boxes.float() if boxes is not None else None
        for i, img_id in enumerate(batch["image_ids"]):
            results += predictions_for_image(
                logits[i], masks[i], boxes[i] if boxes is not None else None,
                batch["orig_sizes"][i], img_id, dataset.contig2cat, topk=topk)
            seen_ids.append(int(img_id))
        n_seen += len(batch["image_ids"])
        if verbose and n_seen % 1000 < len(batch["image_ids"]):
            print(f"  [eval] {n_seen}/{len(dataset)} images", flush=True)
        if max_images and n_seen >= max_images:
            break

    metrics: Dict[str, float] = {"num_images": float(n_seen), "num_dets": float(len(results))}
    if not results:
        return metrics

    coco_gt = dataset.coco
    # loadRes prints an unconditional banner per call; silence it, it fires on every eval.
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(list(results))
    keys = ["AP", "AP50", "AP75", "APs", "APm", "APl"]
    for iou_type in ("segm", "bbox"):
        if iou_type == "bbox" and "bbox" not in results[0]:
            continue
        ev = COCOeval(coco_gt, coco_dt, iou_type)
        ev.params.imgIds = seen_ids
        ev.evaluate()
        ev.accumulate()
        if verbose:
            print(f"--- {iou_type} ---", flush=True)
            ev.summarize()
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                ev.summarize()
        for k, v in zip(keys, ev.stats[:6]):
            metrics[f"{iou_type}_{k}"] = round(float(v) * 100, 3)
    return metrics
