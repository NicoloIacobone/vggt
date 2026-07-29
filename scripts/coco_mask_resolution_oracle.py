"""
How much COCO mask AP does the VGGT token grid cost, before any model is trained?

Motivation
----------
MaskDINO on a ResNet-50 predicts masks on `mask_features` at **stride 4** of an 800x1333 input
(~200x333 cells). Our VGGT front end (docs/MASKDINO.md §3) predicts them on the **37x37 patch
grid** of a 518px input. That is ~30x fewer mask cells. Before spending GPU-weeks on a COCO
training run it is worth knowing the *ceiling*: what mask AP would a model score that is
**perfect** except that its masks must live on that grid?

Method
------
For every non-crowd GT instance of COCO val2017:

    GT mask [H,W]  --area-downsample-->  [g_h, g_w]  --bilinear-upsample-->  [H,W]  --thr 0.5-->

and submit the result as a detection with score 1.0 and the correct category. Area-downsampling
gives the per-cell foreground fraction, which is the best a soft logit map on that grid can encode;
bilinear-up + threshold is exactly what `train/perframe.py` / MaskDINO inference do. So the
resulting AP is a genuine upper bound for that grid, isolated from every other error source.

Grids compared (`--grids`):
  squash:G     GT squashed to a GxG grid, aspect ratio ignored (what a fixed 518x518 VGGT resize
               does; no pixels wasted, geometry distorted)
  pad:G        GT centre-padded to a square first, then GxG (what `load_and_preprocess_images_square`
               does; geometry preserved, the padding wastes grid cells on empty pixels)
  ar:N         aspect-preserving, ~N cells total, each side a multiple of 1 cell (what a
               variable-shape VGGT forward at a fixed token budget gives)
  stride:S@R   short side resized to R (long side capped at 1333), grid = that / S. `stride:4@800`
               is MaskDINO-on-R50's own mask resolution and is the reference ceiling.

Usage
-----
    myenv/bin/python scripts/coco_mask_resolution_oracle.py --limit 500          # smoke
    myenv/bin/python scripts/coco_mask_resolution_oracle.py                      # full 5000
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

DEFAULT_ANN = "/cluster/scratch/niacobone/coco/annotations/instances_val2017.json"
DEFAULT_GRIDS = ["squash:37", "pad:37", "squash:52", "squash:74", "squash:148",
                 "ar:1369", "stride:4@800", "stride:8@800"]


def grid_shape(spec: str, h: int, w: int):
    """(g_h, g_w, pad) for a grid spec on an image of size h x w. `pad` = square-pad first."""
    kind, _, rest = spec.partition(":")
    if kind == "squash":
        g = int(rest)
        return g, g, False
    if kind == "pad":
        g = int(rest)
        return g, g, True
    if kind == "ar":
        n = int(rest)
        # keep aspect ratio, ~n cells total
        s = (n / (h * w)) ** 0.5
        return max(1, round(h * s)), max(1, round(w * s)), False
    if kind == "stride":
        stride_s, _, short = rest.partition("@")
        stride, short = int(stride_s), int(short)
        scale = short / min(h, w)
        if max(h, w) * scale > 1333:
            scale = 1333 / max(h, w)
        return (max(1, int(round(h * scale / stride))),
                max(1, int(round(w * scale / stride))), False)
    raise ValueError(f"unknown grid spec: {spec}")


def quantize(mask: np.ndarray, g_h: int, g_w: int, pad: bool) -> np.ndarray:
    """GT mask -> best achievable mask on a g_h x g_w grid -> back to full resolution."""
    h, w = mask.shape
    t = torch.from_numpy(mask).float()[None, None]
    if pad:
        m = max(h, w)
        top, left = (m - h) // 2, (m - w) // 2
        t = F.pad(t, (left, m - w - left, top, m - h - top))
    # area interpolation = per-cell foreground fraction = the best soft logit map on this grid
    small = F.interpolate(t, size=(g_h, g_w), mode="area")
    up = F.interpolate(small, size=t.shape[-2:], mode="bilinear", align_corners=False)
    if pad:
        m = max(h, w)
        top, left = (m - h) // 2, (m - w) // 2
        up = up[..., top:top + h, left:left + w]
    return (up[0, 0] > 0.5).numpy().astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ann", default=DEFAULT_ANN)
    ap.add_argument("--limit", type=int, default=0, help="first N images (0 = all)")
    ap.add_argument("--grids", nargs="+", default=DEFAULT_GRIDS)
    ap.add_argument("--out", default=None, help="write the summary JSON here")
    args = ap.parse_args()

    coco = COCO(args.ann)
    img_ids = sorted(coco.getImgIds())
    if args.limit:
        img_ids = img_ids[: args.limit]
    print(f"[oracle] {len(img_ids)} images, grids={args.grids}", flush=True)

    # ---- one pass over the GT, building the quantized detections for every grid at once --------
    dets = {g: [] for g in args.grids}
    patch_areas = []          # GT area measured in 37x37-grid cells, for the size histogram
    n_inst = 0
    for i, img_id in enumerate(img_ids):
        info = coco.loadImgs(img_id)[0]
        h, w = info["height"], info["width"]
        anns = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=img_id)) if not a.get("iscrowd", 0)]
        for a in anns:
            m = coco.annToMask(a)
            n_inst += 1
            patch_areas.append(a["area"] / ((h / 37.0) * (w / 37.0)))
            for spec in args.grids:
                g_h, g_w, pad = grid_shape(spec, h, w)
                q = quantize(m, g_h, g_w, pad)
                if q.sum() == 0:                       # object vanished at this grid
                    continue
                rle = mask_util.encode(np.asfortranarray(q))
                rle["counts"] = rle["counts"].decode("ascii")
                dets[spec].append({"image_id": img_id, "category_id": a["category_id"],
                                   "segmentation": rle, "score": 1.0})
        if (i + 1) % 500 == 0:
            print(f"  ... {i + 1}/{len(img_ids)} images", flush=True)

    # ---- how big is a COCO object in VGGT patches? --------------------------------------------
    pa = np.array(patch_areas)
    hist = {
        "instances": int(n_inst),
        "median_area_in_37grid_cells": float(np.median(pa)),
        "frac_under_1_cell": float((pa < 1).mean()),
        "frac_under_4_cells": float((pa < 4).mean()),
        "frac_under_16_cells": float((pa < 16).mean()),
    }
    print("\n[oracle] GT object size in 37x37-grid cells:", json.dumps(hist, indent=2), flush=True)

    # ---- score every grid ----------------------------------------------------------------------
    results = {}
    for spec in args.grids:
        print(f"\n================ {spec} ({len(dets[spec])} dets) ================", flush=True)
        if not dets[spec]:
            continue
        coco_dt = coco.loadRes(dets[spec])
        ev = COCOeval(coco, coco_dt, "segm")
        ev.params.imgIds = img_ids
        ev.evaluate(); ev.accumulate(); ev.summarize()
        keys = ["AP", "AP50", "AP75", "APs", "APm", "APl"]
        results[spec] = {k: round(float(v) * 100, 3) for k, v in zip(keys, ev.stats[:6])}

    print("\n================ CEILING SUMMARY (mask AP with PERFECT classification) ============")
    print(f"{'grid':<14}{'AP':>8}{'AP50':>8}{'AP75':>8}{'APs':>8}{'APm':>8}{'APl':>8}")
    for spec, r in results.items():
        print(f"{spec:<14}{r['AP']:>8.1f}{r['AP50']:>8.1f}{r['AP75']:>8.1f}"
              f"{r['APs']:>8.1f}{r['APm']:>8.1f}{r['APl']:>8.1f}")

    summary = {"num_images": len(img_ids), "size_histogram": hist, "ceilings": results}
    out = Path(args.out) if args.out else Path("mask_resolution_oracle.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n[oracle] wrote {out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
