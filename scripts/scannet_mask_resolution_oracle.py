#!/usr/bin/env python3
"""
GT-only mask-resolution ceiling on ScanNet — the analogue of
`scripts/coco_mask_resolution_oracle.py` for the ScanNet track, measured under the
full-resolution ruler of docs/MASKDINO.md §6.5.

For every GT instance of every scored frame: area-downsample its binary 518x518 mask onto the
prediction grid (the best *soft* logit map that grid can hold), bilinearly upsample back to
518x518, threshold at 0.5, and submit it as a prediction with score ~1.0 and the correct class.
The resulting metrics are a hard **upper bound** for a model predicting on that grid, isolated
from every other error source — a model perfect at everything else cannot beat them.

The scoring mirrors the training eval exactly: per-frame, frames with no GT skipped,
`drop_empty_masks`, our own metric implementation (`train/eval_metrics.py`), frame rows averaged
per scene and then across scenes. Grids: 37 (the native patch grid), 74 (`--mask_upsample 2`),
148 (`--mask_upsample 4`), 259, 518 (sanity: must be ~1.0).

Needs the dataset (stage the tar or point --scans_root at an unpacked tree); CPU-only:

    myenv/bin/python scripts/scannet_mask_resolution_oracle.py --scans_root $SCANNET_ROOT
    sbatch slurm/scannet_oracle.sh          # stages the 500-scene tar on a CPU node
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.scannet_overfit import ScanNetMultiSceneDataset
from train.common import DEFAULT_SCANS_ROOT, resolve_scene_dirs
from train.eval_metrics import compute_instance_segmentation_metrics
from train.perframe import METRIC_KEYS, drop_empty_masks

NUM_CLASSES = 19  # the head's width; classes outside 1..19 are background (docs/MASKDINO.md §4)


def quantised_prediction(gt_full: torch.Tensor, grid: int) -> torch.Tensor:
    """[n, H, W] binary GT → the best mask LOGITS a `grid`x`grid` prediction can encode."""
    H, W = gt_full.shape[-2:]
    if grid >= min(H, W):
        return gt_full * 20.0 - 10.0
    occ = F.interpolate(gt_full[None], size=(grid, grid), mode="area")[0]
    up = F.interpolate(occ[None], size=(H, W), mode="bilinear", align_corners=False)[0]
    return up * 20.0 - 10.0          # sigmoid(logit) > 0.5  ⇔  soft occupancy > 0.5


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--scans_root", type=str, default=DEFAULT_SCANS_ROOT)
    p.add_argument("--scenes", type=str,
                   default=",".join(f"scene{i:04d}_00" for i in range(80, 90)),
                   help="Comma-separated scene names (default: the val split 0080-0089)")
    p.add_argument("--num_frames", type=int, default=8)
    p.add_argument("--grids", type=str, default="37,74,148,259,518")
    p.add_argument("--score_threshold", type=float, default=0.25)
    p.add_argument("--out", type=str, default="scannet_mask_resolution_oracle.json")
    args = p.parse_args()

    grids = [int(g) for g in args.grids.split(",")]
    scene_dirs = resolve_scene_dirs(args.scenes, args.scans_root)
    dataset = ScanNetMultiSceneDataset(scene_dirs, num_frames=args.num_frames,
                                       frame_sampling="even", img_size=518,
                                       instance_level=True)

    per_grid = {g: {} for g in grids}                       # grid -> scene -> mean row
    for idx in range(len(dataset)):
        item = dataset[idx]
        name = Path(scene_dirs[idx]).parent.name
        id_maps = item["masks"]                             # [S, 518, 518] global-id map
        classes = [int(c) for c in item["classes"].tolist()]
        rows = {g: [] for g in grids}
        for f in range(id_maps.shape[0]):
            ids = [int(i) for i in torch.unique(id_maps[f]).tolist()
                   if i > 0 and 1 <= classes[int(i) - 1] <= NUM_CLASSES]
            if not ids:
                continue                                    # protocol: frames with no GT skipped
            gt_full = torch.stack([(id_maps[f] == g).float() for g in ids])
            gt_cls = torch.tensor([classes[g - 1] for g in ids])
            # score ~1.0 for the correct class, ~0 for the rest (sigmoid scoring, §6.2)
            cl = torch.full((len(ids), NUM_CLASSES + 1), -10.0)
            cl[torch.arange(len(ids)), gt_cls] = 10.0
            for g in grids:
                pm, pcl = drop_empty_masks(quantised_prediction(gt_full, g), cl)
                rows[g].append(compute_instance_segmentation_metrics(
                    pred_masks=pm, class_logits=pcl, gt_masks=gt_full, gt_classes=gt_cls,
                    background_class=0, score_mode="sigmoid",
                    score_threshold=args.score_threshold))
        for g in grids:
            per_grid[g][name] = ({k: float(np.mean([r[k] for r in rows[g]]))
                                  for k in METRIC_KEYS} if rows[g]
                                 else {k: 0.0 for k in METRIC_KEYS})
        print(f"  {name}: " + "  ".join(
            f"{g}px mIoU={per_grid[g][name]['mIoU']:.3f}" for g in grids))

    summary = {g: {k: float(np.mean([m[k] for m in per_grid[g].values()]))
                   for k in METRIC_KEYS} for g in grids}
    print("\nGT-only ceiling (mean over scenes of per-frame means, full-resolution ruler):")
    print(f"{'grid':>6} {'mIoU':>7} {'AP50':>7} {'AP75':>7} {'mAP':>7}")
    for g in grids:
        s = summary[g]
        print(f"{g:>4}px {s['mIoU']:>7.3f} {s['AP50']:>7.3f} {s['AP75']:>7.3f} {s['mAP']:>7.3f}")

    out = {"scenes": args.scenes, "num_frames": args.num_frames,
           "score_threshold": args.score_threshold,
           "summary": {str(g): summary[g] for g in grids},
           "per_scene": {str(g): per_grid[g] for g in grids}}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n✓ Wrote {args.out}")


if __name__ == "__main__":
    main()
