#!/usr/bin/env python3
"""
Cross-GT checkpoint evaluation (eval-only) — docs/OFFICIAL_GT_MIGRATION_PLAN.md Phase 4.

Evaluates a trained checkpoint against GT rebuilt FRESH from a --scans_root tree,
instead of the GT stored inside the checkpoint. This is what quantifies label-noise
fitting: e.g. eval the SAM3-trained arm-C checkpoint against the official-GT val
scenes. Works for all query modes incl. learned (uses the same eval path as
training: prompted GT-centroid queries + unprompted grid; for learned-query heads
the two coincide and are reported under "unprompted").

Usage:
    python scripts/eval_checkpoint.py \
        --checkpoint <run_dir>/checkpoint_best_ap50.pth \
        --scans_root <staged official-GT scans root> \
        [--scenes scene0080_00,...]   # default: the checkpoint's stored val scenes
        [--tag official_gt]           # output name suffix

Writes per-scene + mean prompted/unprompted metrics to
<ckpt_dir>/cross_eval_<ckpt_stem>_<tag>.json and prints a summary.
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def build_eval_args(ck_args: dict, cli) -> SimpleNamespace:
    """Bundle-building settings: the checkpoint's training args, overridable via CLI.

    Eval bundles must match how the run built its own eval bundles (frames per
    scene, query count, grid size, mask resolution); only the GT source (scans
    root / instance_level) is meant to vary in a cross-GT eval.
    """

    def pick(name, default):
        v = getattr(cli, name, None)
        return v if v is not None else ck_args.get(name, default)

    return SimpleNamespace(
        num_frames=int(pick("num_frames", 8)),
        num_queries=int(pick("num_queries", 32)),
        instance_level=bool(ck_args.get("instance_level", False)
                            if cli.instance_level is None else cli.instance_level),
        grid_size=int(pick("grid_size", 6)),
        mask_upsample=int(ck_args.get("mask_upsample", 1)),
        bundles_per_scene=1,
        color_jitter=0.0,
        cache_device=None,
    )


def resolve_eval_scenes(ckpt: dict, scenes_arg: str | None) -> list[str]:
    """Scene names to evaluate: --scenes if given, else the checkpoint's val scenes."""
    if scenes_arg:
        return [s for s in scenes_arg.split(",") if s]
    names = [s["name"] for s in ckpt.get("scenes", []) if s.get("split") == "val"]
    if not names:
        raise ValueError("checkpoint stores no val scenes; pass --scenes explicitly")
    return names


def main():
    parser = argparse.ArgumentParser(description="Cross-GT checkpoint evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--scans_root", type=str, required=True,
                        help="scans tree providing the (fresh) GT, e.g. the staged "
                             "official-GT dataset")
    parser.add_argument("--scenes", type=str, default=None,
                        help="comma-separated scene ids (default: checkpoint's val scenes)")
    parser.add_argument("--instance_level", type=int, default=None, choices=[0, 1],
                        help="override the checkpoint's instance_level (default: inherit)")
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--num_queries", type=int, default=None)
    parser.add_argument("--grid_size", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tag", type=str, default=None,
                        help="output filename suffix (default: scans_root dir name)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    from train_overfit import D4RTModel
    from train_multiscene import prepare_scene_bundles, eval_all, mean_metric

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    ckpt_path = Path(args.checkpoint)
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ck_args = ckpt.get("args", {})
    head_config = ckpt.get("head_config", {}) or {}
    query_mode = head_config.get("query_mode", ck_args.get("query_mode", "point"))

    scene_names = resolve_eval_scenes(ckpt, args.scenes)
    scene_dirs = [str(Path(args.scans_root) / s / "raw_data") for s in scene_names]
    missing = [d for d in scene_dirs if not Path(d).exists()]
    if missing:
        raise SystemExit(f"scene dirs not found under {args.scans_root}: {missing}")

    eval_args = build_eval_args(ck_args, args)
    print(f"{len(scene_names)} scene(s), query_mode={query_mode}, "
          f"instance_level={eval_args.instance_level}, num_frames={eval_args.num_frames}, "
          f"grid_size={eval_args.grid_size}, device={device}")

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

    scenes = prepare_scene_bundles(model, scene_dirs, eval_args, device, split="val")
    for s, name in zip(scenes, scene_names):
        s["name"] = name  # loader may shorten names; keep the canonical ids

    prompted = eval_all(model, scenes, device, unprompted=False)
    unprompted = eval_all(model, scenes, device, unprompted=True)

    keys = ["mIoU", "AP50", "AP75", "mAP", "num_gt"]
    mean = {"prompted": {k: mean_metric(prompted, k) for k in keys},
            "unprompted": {k: mean_metric(unprompted, k) for k in keys}}
    header = f"{'queries':>10} | " + " | ".join(f"{k:>8}" for k in keys)
    print("\n" + header + "\n" + "-" * len(header))
    for label, m in mean.items():
        print(f"{label:>10} | " + " | ".join(f"{m[k]:8.3f}" for k in keys))
    if query_mode == "learned":
        print("(learned queries ignore coordinates: prompted == unprompted)")

    tag = args.tag or Path(args.scans_root).parent.name or "crossgt"
    out_path = (Path(args.output) if args.output
                else ckpt_path.parent / f"cross_eval_{ckpt_path.stem}_{tag}.json")
    out_path.write_text(json.dumps({
        "checkpoint": str(ckpt_path),
        "scans_root": str(args.scans_root),
        "scenes": scene_names,
        "query_mode": query_mode,
        "eval_args": vars(eval_args),
        "mean": mean,
        "per_scene": {"prompted": prompted, "unprompted": unprompted},
    }, indent=2))
    print(f"\n✓ Wrote {out_path}")


if __name__ == "__main__":
    main()
