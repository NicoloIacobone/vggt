#!/usr/bin/env python3
"""
Score an existing D4RT checkpoint under the MaskDINO trial's PER-FRAME protocol.

Why this exists: the D4RT arms (A–E) score one multi-view instance against its 8-frame GT mask
(a single IoU over the concatenated frames), while the single-frame MaskDINO trial scores each
frame on its own (docs/MASKDINO_TRIAL.md §6). Those two numbers are not interchangeable, so
"is MaskDINO better than arm C?" can only be answered by running arm C through the *same*
per-frame protocol. That is what this script does.

It reuses the checkpoint's own stored bundles (frames + GT), so no dataset staging is needed
beyond `--scans_root` for `--checkpoint_light` runs. Queries are the honest/unprompted ones:
the learned object queries for `query_mode=learned`, the uniform grid for `point`/`hybrid`.

    python scripts/eval_perframe.py --checkpoint <run_dir>/checkpoint_best.pth
    → <run_dir>/perframe_eval_<ckpt stem>.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.scannet_overfit import decode_checkpoint_images
from train.eval_metrics import compute_instance_segmentation_metrics
from train_multiscene import DEFAULT_SCANS_ROOT


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


def perframe_metrics(pred_masks: torch.Tensor, class_logits: torch.Tensor,
                     gt_masks: torch.Tensor, gt_classes: torch.Tensor,
                     score_threshold: float = 0.0, score_mode: str = "softmax",
                     background_class: int = 0, mask_threshold: float = 0.5
                     ) -> List[Dict[str, float]]:
    """
    Per-frame instance metrics from multi-view predictions.

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


def _mean(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = ["mIoU", "AP50", "AP75", "mAP", "class_acc", "num_pred", "num_gt"]
    if not rows:
        return {k: 0.0 for k in keys}
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def build_model_from_checkpoint(ckpt: Dict, device: str):
    """Rebuild the D4RT model described by a checkpoint's `head_config` and load its weights."""
    from train_overfit import D4RTModel

    cfg = ckpt.get("head_config", {})
    model = D4RTModel(
        freeze_backbone=True,
        num_views=int(cfg.get("num_views", 10)),
        decoder_hidden_dim=int(cfg.get("hidden_dim", 256)),
        mask_embed_dim=int(cfg.get("mask_embed_dim", 256)),
        dropout=float(cfg.get("dropout", 0.0)),
        query_mode=cfg.get("query_mode", "point"),
        num_learned_queries=int(cfg.get("num_learned_queries", 0)),
        mask_upsample=int(cfg.get("mask_upsample", 1)),
        num_anchors=int(cfg.get("num_anchors", 0)),
        anchor_knn=int(cfg.get("anchor_knn", 8)),
        anchor_content=cfg.get("anchor_content", "pooled"),
        anchor_coord_scale=float(cfg.get("anchor_coord_scale", 1.0)),
    ).to(device)
    model.decoder_head.load_state_dict(ckpt["decoder_head_state_dict"])
    model.eval()
    return model, cfg


@torch.no_grad()
def eval_checkpoint(ckpt_path: Path, args) -> Dict:
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model, cfg = build_model_from_checkpoint(ckpt, device)
    query_mode = cfg.get("query_mode", "point")
    if query_mode == "anchor3d":
        raise SystemExit("anchor3d checkpoints need their anchors rebuilt from the point head; "
                         "not supported by this diagnostic (arm C is `learned`).")
    print(f"query_mode={query_mode}, mask_upsample={cfg.get('mask_upsample', 1)}")

    from train_overfit import generate_grid_queries

    scenes = ckpt.get("scenes", [])
    if not scenes:
        raise SystemExit("Checkpoint has no stored scenes.")

    per_scene = {}
    for scene in scenes:
        if args.split and scene.get("split") != args.split:
            continue
        images = decode_checkpoint_images(scene, args.scans_root).to(device)  # [1, S, 3, H, W]
        S = images.shape[1]
        agg_list, patch_start_idx = model.backbone.aggregator(images)
        features = agg_list[-1]

        if query_mode == "learned":
            n = int(cfg.get("num_learned_queries", 0))
            coords = torch.zeros(1, n, 2, device=device)
            views = torch.zeros(1, n, dtype=torch.long, device=device)
        else:
            coords, views = generate_grid_queries(S, args.grid_size, device)
            if query_mode == "hybrid":
                m = int(cfg.get("num_learned_queries", 0))
                coords = torch.cat([torch.zeros(1, m, 2, device=device), coords], dim=1)
                views = torch.cat([torch.zeros(1, m, dtype=torch.long, device=device), views], 1)

        class_logits, _, pred_masks = model.decoder_head(
            coords, views, images, features, int(patch_start_idx), anchors=None)

        gt = scene["gt"]
        rows = perframe_metrics(pred_masks[0], class_logits[0],
                                gt["masks"].to(device), gt["classes"].to(device),
                                score_threshold=args.score_threshold, score_mode="softmax")
        per_scene[scene["name"]] = {"split": scene.get("split"), "frames": len(rows), **_mean(rows)}
        print(f"  [{scene.get('split')}] {scene['name']}: frames={len(rows)} "
              f"mIoU={per_scene[scene['name']]['mIoU']:.3f} "
              f"AP50={per_scene[scene['name']]['AP50']:.3f}")

    summary = {}
    for split in ("train", "val"):
        rows = [m for m in per_scene.values() if m["split"] == split]
        if rows:
            summary[split] = {k: float(np.mean([r[k] for r in rows]))
                              for k in ("mIoU", "AP50", "AP75", "mAP", "class_acc", "num_pred")}
    return {"checkpoint": str(ckpt_path), "protocol": "per-frame (MASKDINO_TRIAL.md §6)",
            "query_mode": query_mode, "score_threshold": args.score_threshold,
            "grid_size": args.grid_size, "per_scene": per_scene, "summary": summary}


def main():
    p = argparse.ArgumentParser(description="Per-frame eval of a D4RT checkpoint")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--scans_root", type=str, default=DEFAULT_SCANS_ROOT)
    p.add_argument("--split", type=str, default=None, choices=[None, "train", "val"],
                   help="Restrict to one split (default: all stored scenes)")
    p.add_argument("--grid_size", type=int, default=6,
                   help="Unprompted query grid for point/hybrid checkpoints")
    p.add_argument("--score_threshold", type=float, default=0.0,
                   help="Class-score threshold; 0.0 reproduces the D4RT arms' own convention "
                        "(argmax != background is the only filter)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    result = eval_checkpoint(ckpt_path, args)
    print("\nSummary (per-frame protocol):")
    for split, m in result["summary"].items():
        print(f"  {split}: mIoU={m['mIoU']:.3f} AP50={m['AP50']:.3f} AP75={m['AP75']:.3f} "
              f"mAP={m['mAP']:.3f}")
    out = Path(args.out) if args.out else ckpt_path.parent / f"perframe_eval_{ckpt_path.stem}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"✓ Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
