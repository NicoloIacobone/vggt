#!/usr/bin/env python3
"""
Re-render the RGB | GT | prediction panels of a finished MaskDINO run, without retraining.

`scripts/train_maskdino.py` writes these figures once at the end of a run, so any change to the
drawing code (e.g. the identity-keyed colouring in `train/maskdino_eval.py`) would otherwise
only reach *future* runs. This script rebuilds the head from a checkpoint's `head_config`,
re-caches the frozen-backbone features for the scenes it needs, and calls the same `visualize()`.

Everything about how the scenes are built (frame count, feature mode, mask resolution, …) comes
from the checkpoint's own stored `args`, so the figures match the run they came from.

    python scripts/visualize_maskdino.py --checkpoint <run_dir>/checkpoint_best.pth
    → <run_dir>/visualizations/   (overwrites the existing PNGs; --out_dir to write elsewhere)
"""

import argparse
import sys
from argparse import Namespace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.maskdino.model import MaskDINOVGGTModel
from train.common import DEFAULT_SCANS_ROOT, resolve_scene_dirs
from train.maskdino_data import prepare_scenes
from train.maskdino_eval import visualize


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--scenes", type=str, default=None,
                   help="Comma-separated scene names. Default: the run's own --val_scenes.")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Default: <checkpoint dir>/visualizations")
    p.add_argument("--scans_root", type=str, default=DEFAULT_SCANS_ROOT,
                   help="Override if the data lives somewhere else than during training "
                        "(the tar is staged per job, so the training path is usually stale).")
    p.add_argument("--max_scenes", type=int, default=2)
    p.add_argument("--max_frames", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    return p


def main():
    cli = build_argparser().parse_args()
    ckpt_path = Path(cli.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "head_config" not in ckpt or "head_state_dict" not in ckpt:
        raise SystemExit(f"{ckpt_path} is not a MaskDINO checkpoint "
                         f"(keys: {sorted(ckpt)[:8]}…)")

    # The run's own args drive scene construction; only the paths and the device may differ now.
    args = Namespace(**ckpt["args"])
    args.scans_root = cli.scans_root
    device = cli.device if (cli.device != "cuda" or torch.cuda.is_available()) else "cpu"
    args.device = device
    args.cache_device = device
    print(f"Checkpoint: {ckpt_path.name} (epoch {ckpt.get('epoch', '?')}, "
          f"multi_frame={getattr(args, 'multi_frame', False)}, "
          f"feature_mode={getattr(args, 'feature_mode', 'single')})")

    scenes_arg = cli.scenes if cli.scenes else args.val_scenes
    scene_dirs = resolve_scene_dirs(scenes_arg, args.scans_root)
    if not scene_dirs:
        raise SystemExit(f"No scenes resolved from {scenes_arg!r} under {args.scans_root}")
    # Only the first --max_scenes are ever drawn; caching the rest would waste backbone time.
    scene_dirs = scene_dirs[:cli.max_scenes]
    print(f"Scenes: {', '.join(Path(d).name for d in scene_dirs)}")

    model = MaskDINOVGGTModel(ckpt["head_config"]).to(device)
    model.head.load_state_dict(ckpt["head_state_dict"])
    model.eval()

    print("\n=== Caching frozen-backbone features ===")
    scenes = prepare_scenes(model, scene_dirs, args, device, "val")

    out_dir = Path(cli.out_dir) if cli.out_dir else ckpt_path.parent / "visualizations"
    print(f"\n=== Rendering to {out_dir} ===")
    n = visualize(model, scenes, args, device, out_dir,
                  max_scenes=cli.max_scenes, max_frames=cli.max_frames)
    return 0 if n > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
