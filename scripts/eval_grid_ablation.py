#!/usr/bin/env python3
"""
Grid-density ablation (eval-only) — docs/todo.md "Grid-density vs unprompted recall".

Re-evaluates a trained point/hybrid-mode checkpoint's UNPROMPTED (grid) metrics at several
--grid_size values without retraining: the eval grid is generated at inference time, so the
only cost is one frozen-backbone pass per stored scene plus one head forward per grid size.

Learned-query checkpoints are rejected: their queries ignore coordinates entirely, so a
grid-density sweep is undefined for them.

Usage:
    python scripts/eval_grid_ablation.py \
        --checkpoint <run_dir>/checkpoint_best_ap50.pth \
        --grid_sizes 2,4,6,8,10,12 [--split val] [--output <path>.json]

Writes per-scene + mean metrics (mIoU/AP50/AP75/mAP, plus a kept-predictions duplicate
diagnostic) to <ckpt_dir>/grid_ablation_<ckpt_stem>.json and prints a summary table.
A "prompted" row (the checkpoint's stored GT-centroid queries) is included as a
reproduction check against the run's metrics.jsonl.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train.eval_metrics import compute_instance_segmentation_metrics  # noqa: E402


def _kept_predictions(class_logits: torch.Tensor, score_threshold: float,
                      background_class: int = 0) -> int:
    """#queries whose argmax class is foreground with confidence >= threshold (the same rule
    select_instances uses to keep a query) — the duplicate/FP-pressure diagnostic."""
    probs = class_logits.softmax(-1)
    scores, labels = probs.max(-1)
    return int(((labels != background_class) & (scores >= score_threshold)).sum().item())


@torch.no_grad()
def eval_bundle_at_grid_sizes(
    decoder_head,
    images: torch.Tensor,
    features: torch.Tensor,
    patch_start_idx: int,
    gt: dict,
    grid_sizes,
    device: str,
    score_threshold: float = 0.5,
    prompted_queries=None,
) -> dict:
    """
    Head-only sweep on one cached scene bundle (CPU-testable: takes precomputed features).

    Returns {"grid_<g>": metrics, ...} (+ "prompted" when prompted_queries=(coords, view_ids)
    is given). Metrics are compute_instance_segmentation_metrics() plus num_queries and
    num_kept (foreground predictions at `score_threshold`).
    """
    from train_overfit import generate_grid_queries  # local import keeps the fn CPU-testable

    mode = getattr(decoder_head, "query_mode", "point")
    M = getattr(decoder_head, "num_learned_queries", 0)
    if mode in ("learned", "anchor3d"):
        raise ValueError(f"{mode}-query heads ignore coordinates; grid density is undefined")

    S = images.shape[1]
    runs = {}
    if prompted_queries is not None:
        runs["prompted"] = (prompted_queries[0].to(device), prompted_queries[1].to(device))
    for g in grid_sizes:
        runs[f"grid_{g}"] = generate_grid_queries(S, g, device)

    results = {}
    for label, (coords, view_ids) in runs.items():
        if mode == "hybrid":
            ph_c = torch.zeros(coords.shape[0], M, 2, device=device)
            ph_v = torch.zeros(coords.shape[0], M, dtype=torch.long, device=device)
            coords = torch.cat([ph_c, coords], dim=1)
            view_ids = torch.cat([ph_v, view_ids], dim=1)
        class_logits, _, pred_masks = decoder_head(
            coords, view_ids, images, features, patch_start_idx)
        metrics = compute_instance_segmentation_metrics(
            pred_masks=pred_masks[0],
            class_logits=class_logits[0],
            gt_masks=gt["masks"],
            gt_classes=gt["classes"],
            background_class=0,
        )
        metrics["num_queries"] = int(coords.shape[1])
        metrics["num_kept"] = _kept_predictions(class_logits[0], score_threshold)
        results[label] = metrics
    return results


def main():
    parser = argparse.ArgumentParser(description="Eval-only grid-density ablation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--grid_sizes", type=str, default="2,4,6,8,10,12",
                        help="Comma-separated grid sizes to sweep")
    parser.add_argument("--split", type=str, default="val", choices=["val", "train", "all"],
                        help="Which stored scenes to evaluate (default: val)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--score_threshold", type=float, default=0.5,
                        help="Confidence for the kept-predictions diagnostic (not the metrics)")
    parser.add_argument("--scans_root", type=str, default=None,
                        help="Only needed for --checkpoint_light checkpoints")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON (default <ckpt_dir>/grid_ablation_<stem>.json)")
    args = parser.parse_args()

    from train_overfit import D4RTModel
    from visualize_masks import scenes_from_checkpoint
    from data.scannet_overfit import decode_checkpoint_images

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    grid_sizes = [int(g) for g in args.grid_sizes.split(",") if g.strip()]
    ckpt_path = Path(args.checkpoint)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ck_args = ckpt.get("args", {})
    head_config = ckpt.get("head_config", {}) or {}
    query_mode = head_config.get("query_mode", ck_args.get("query_mode", "point"))
    if query_mode in ("learned", "anchor3d"):
        raise SystemExit(f"This checkpoint uses {query_mode} queries — grid density "
                         "does not apply.")

    scenes = [(label, s) for label, s in scenes_from_checkpoint(ckpt)
              if args.split == "all" or s.get("split", "train") == args.split]
    if not scenes:
        raise SystemExit(f"No stored scenes with split={args.split} in this checkpoint")
    print(f"{len(scenes)} {args.split} scene(s), grid sizes {grid_sizes}, "
          f"query_mode={query_mode}, device={device}")

    num_views = ck_args.get("num_views", 10)
    model = D4RTModel(
        freeze_backbone=True,
        num_views=num_views if isinstance(num_views, int) else 10,
        decoder_hidden_dim=256,
        mask_embed_dim=256,
        dropout=0.0,
        query_mode=query_mode,
        num_learned_queries=head_config.get("num_learned_queries",
                                            ck_args.get("num_learned_queries", 0)),
        mask_upsample=head_config.get("mask_upsample", ck_args.get("mask_upsample", 1)),
    ).to(device)
    model.decoder_head.load_state_dict(ckpt["decoder_head_state_dict"])
    model.eval()

    per_scene = {}
    for label, scene in scenes:
        images = decode_checkpoint_images(scene, scans_root=args.scans_root).to(device)
        gt = {k: v.to(device) for k, v in scene["gt"].items()}
        with torch.no_grad():
            agg_list, patch_start_idx = model.backbone.aggregator(images)
            features = agg_list[-1]
        per_scene[scene["name"]] = eval_bundle_at_grid_sizes(
            model.decoder_head, images, features, patch_start_idx, gt, grid_sizes, device,
            score_threshold=args.score_threshold,
            prompted_queries=(scene["coordinates"], scene["view_ids"]),
        )
        stored = scene.get("metrics") or {}
        rep = per_scene[scene["name"]]["prompted"]
        print(f"  {label}: gt={rep['num_gt']}  prompted mIoU {rep['mIoU']:.3f}"
              + (f" (stored {stored['mIoU']:.3f})" if stored.get("mIoU") is not None else ""))

    # Mean over scenes per query setting.
    settings = ["prompted"] + [f"grid_{g}" for g in grid_sizes]
    keys = ["mIoU", "AP50", "AP75", "mAP", "num_queries", "num_kept", "num_gt"]
    mean = {s: {k: float(torch.tensor([per_scene[n][s][k] for n in per_scene],
                                      dtype=torch.float64).mean().item())
                for k in keys} for s in settings}

    header = f"{'queries':>10} | " + " | ".join(f"{k:>11}" for k in keys)
    print("\n" + header + "\n" + "-" * len(header))
    for s in settings:
        print(f"{s:>10} | " + " | ".join(f"{mean[s][k]:11.3f}" for k in keys))

    out_path = (Path(args.output) if args.output
                else ckpt_path.parent / f"grid_ablation_{ckpt_path.stem}.json")
    out_path.write_text(json.dumps({
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "grid_sizes": grid_sizes,
        "score_threshold": args.score_threshold,
        "mean": mean,
        "per_scene": per_scene,
    }, indent=2))
    print(f"\n✓ Wrote {out_path}")


if __name__ == "__main__":
    main()
