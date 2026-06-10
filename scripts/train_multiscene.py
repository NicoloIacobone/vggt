#!/usr/bin/env python3
"""
Multi-scene training for the D4RT instance-segmentation head (Milestone 2).

Milestone-1 trained on one fixed bundle per scene (pure overfit protocol). Milestone 2 turns
this into a regularized training loop that also supports UNPROMPTED inference:

  - No-object loss (DETR-style): unmatched queries are supervised toward the background
    class (down-weighted by --no_object_weight), so at inference time background/empty
    queries can be filtered by their predicted class — no GT-ordered queries needed.
  - Unprompted evaluation: besides the prompted eval (queries at GT centroids), every eval
    also runs a uniform --grid_size x --grid_size query grid per frame and computes the same
    metrics. This measures detection, not just point-prompted segmentation.
  - Regularization: --bundles_per_scene cached bundles per scene (bundle 0 uses evenly-spaced
    frames and is the eval/checkpoint bundle; the rest use random frame sampling + optional
    --color_jitter), Gaussian --query_jitter on instance-centroid queries, and fresh random
    background queries every step (disable with --fixed_bg).
  - Model selection: tracks val prompted mIoU at every eval, saves checkpoint_best.pth when
    it improves, and optionally stops after --early_stop_patience evals without improvement.
  - Scaling: --cache_device cpu keeps the cached backbone features/images in host memory
    (moved to the GPU per step), so scene count is not limited by GPU memory.

The frozen VGGT backbone still runs only ONCE per bundle up front; every epoch trains just
the ~6.5M-param head.

Usage:
    python scripts/train_multiscene.py \
        --train_scenes scene0000_00,scene0001_00,scene0002_00,scene0003_00 \
        --val_scenes scene0004_00 \
        --num_epochs 1000 --num_frames 8 --num_queries 32 --bundles_per_scene 4 \
        --save_checkpoint /cluster/work/igp_psr/niacobone/distillation/output/<run>/checkpoint.pth
"""

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train.loss import D4RTLoss
from train.eval_metrics import compute_instance_segmentation_metrics
from data.scannet_overfit import ScanNetMultiSceneDataset
from train_overfit import (
    D4RTModel, build_gt_targets, generate_query_points, generate_grid_queries,
    _format_metrics,
)

DEFAULT_SCANS_ROOT = "/cluster/work/igp_psr/niacobone/distillation/dataset/scannet/scans"


def resolve_scene_dirs(spec: str, scans_root: str) -> List[str]:
    """Accept comma-separated scene names (resolved under scans_root/<name>/raw_data) or paths."""
    dirs = []
    for token in [t.strip() for t in spec.split(",") if t.strip()]:
        p = Path(token)
        if not p.exists():
            p = Path(scans_root) / token / "raw_data"
        if not p.exists():
            raise ValueError(f"Scene not found: {token} (tried {p})")
        dirs.append(str(p))
    return dirs


def photometric_jitter(images: torch.Tensor, strength: float) -> torch.Tensor:
    """One random brightness/contrast draw applied to a whole bundle (masks are unaffected)."""
    if strength <= 0:
        return images
    contrast = 1.0 + (torch.rand(1, device=images.device).item() * 2 - 1) * strength
    brightness = (torch.rand(1, device=images.device).item() * 2 - 1) * strength
    return ((images - 0.5) * contrast + 0.5 + brightness).clamp(0.0, 1.0)


def bundle_to_device(bundle: Dict, device: str) -> Dict:
    """Shallow copy of a bundle with its tensors on `device` (no-op if already there)."""
    out = {}
    for k, v in bundle.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, dict):
            out[k] = {kk: vv.to(device, non_blocking=True) if isinstance(vv, torch.Tensor) else vv
                      for kk, vv in v.items()}
        else:
            out[k] = v
    return out


@torch.no_grad()
def prepare_scene_bundles(
    model: D4RTModel,
    scene_dirs: List[str],
    args,
    device: str,
    split: str,
) -> List[Dict]:
    """
    Build per-scene cached bundles: images, query points, dense GT targets, and the frozen
    backbone features. Returns one dict per scene: {"name", "split", "bundles": [...]}.

    Bundle 0 always uses evenly-spaced frames (deterministic — it is the eval + checkpoint
    bundle and also carries the unprompted grid queries). Train scenes additionally get
    `bundles_per_scene - 1` randomly-sampled-frame bundles (with optional photometric
    jitter) as regularization.
    """
    if not scene_dirs:
        return []
    num_bundles = args.bundles_per_scene if split == "train" else 1
    common = dict(num_frames=args.num_frames, img_size=518)
    even_loader = DataLoader(ScanNetMultiSceneDataset(scene_dirs, frame_sampling="even", **common),
                             batch_size=1, shuffle=False, num_workers=0)
    rand_dataset = (ScanNetMultiSceneDataset(scene_dirs, frame_sampling="random", **common)
                    if num_bundles > 1 else None)

    def build_bundle(batch, jitter: bool) -> Dict:
        images = batch["images"].to(device)  # [1, S, 3, H, W]
        if jitter:
            images = photometric_jitter(images, args.color_jitter)
        coordinates, view_ids = generate_query_points(batch, args.num_queries, device)
        num_inst_queries = int(batch["num_instances"])

        agg_list, patch_start_idx = model.backbone.aggregator(images)
        features = agg_list[-1].detach()  # [1, S, P, 2048] — cached; backbone never reruns
        num_patch_tokens = features.shape[2] - patch_start_idx
        gt = build_gt_targets(batch, patch_start_idx, num_patch_tokens, device)
        return {
            "images": images,
            "coordinates": coordinates,
            "view_ids": view_ids,
            "num_inst_queries": num_inst_queries,
            "features": features,
            "patch_start_idx": int(patch_start_idx),
            "num_patch_tokens": int(num_patch_tokens),
            "gt": gt,
            "frame_names": batch.get("frame_names", None),
        }

    scenes = []
    for batch in even_loader:
        name = batch["scene_name"][0] if isinstance(batch["scene_name"], (list, tuple)) else str(batch["scene_name"])
        t0 = time.time()
        bundle = build_bundle(batch, jitter=False)
        S = bundle["images"].shape[1]
        bundle["grid_coordinates"], bundle["grid_view_ids"] = generate_grid_queries(
            S, args.grid_size, device)
        scenes.append({"name": name, "split": split,
                       "bundles": [bundle_to_device(bundle, args.cache_device)]})
        print(
            f"  [{split}] {name}: frames={S}, queries={bundle['coordinates'].shape[1]} "
            f"(+{bundle['grid_coordinates'].shape[1]} grid), "
            f"instances={bundle['gt']['classes'].shape[0]}, "
            f"features={tuple(bundle['features'].shape)} ({time.time() - t0:.1f}s backbone)"
        )

    # Extra randomly-sampled bundles (train only): each pass over the dataset resamples frames.
    for k in range(1, num_bundles):
        rand_loader = DataLoader(rand_dataset, batch_size=1, shuffle=False, num_workers=0)
        for idx, batch in enumerate(rand_loader):
            t0 = time.time()
            bundle = build_bundle(batch, jitter=args.color_jitter > 0)
            scenes[idx]["bundles"].append(bundle_to_device(bundle, args.cache_device))
            print(f"  [{split}] {scenes[idx]['name']} bundle {k}: "
                  f"instances={bundle['gt']['classes'].shape[0]} ({time.time() - t0:.1f}s backbone)")
    return scenes


def make_train_queries(b: Dict, args, device: str):
    """
    Per-step query augmentation on a (device-resident) bundle:
      - Gaussian jitter (std --query_jitter) on the instance-centroid queries.
      - Fresh random background query points + view ids (unless --fixed_bg).
    """
    coords = b["coordinates"].clone()
    view_ids = b["view_ids"].clone()
    ni = b["num_inst_queries"]
    S = b["images"].shape[1]
    if args.query_jitter > 0 and ni > 0:
        coords[:, :ni] = (coords[:, :ni] + torch.randn_like(coords[:, :ni]) * args.query_jitter
                          ).clamp(0.0, 1.0)
    nbg = coords.shape[1] - ni
    if nbg > 0 and not args.fixed_bg:
        coords[:, ni:] = torch.rand(coords.shape[0], nbg, 2, device=device)
        view_ids[:, ni:] = torch.randint(0, S, (view_ids.shape[0], nbg), device=device)
    return coords, view_ids


def head_forward(model: D4RTModel, b: Dict, coordinates=None, view_ids=None):
    """Decoder-head-only forward on a device-resident bundle (cached backbone features)."""
    return model.decoder_head(
        coordinates if coordinates is not None else b["coordinates"],
        view_ids if view_ids is not None else b["view_ids"],
        b["images"], b["features"], b["patch_start_idx"],
    )


@torch.no_grad()
def eval_scene(model: D4RTModel, scene: Dict, device: str, unprompted: bool = False) -> Dict[str, float]:
    """Metrics on the deterministic bundle 0; prompted (GT-centroid) or unprompted (grid) queries."""
    was_training = model.training
    model.eval()
    b = bundle_to_device(scene["bundles"][0], device)
    if unprompted:
        class_logits, _, pred_masks = head_forward(model, b, b["grid_coordinates"], b["grid_view_ids"])
    else:
        class_logits, _, pred_masks = head_forward(model, b)
    metrics = compute_instance_segmentation_metrics(
        pred_masks=pred_masks[0],
        class_logits=class_logits[0],
        gt_masks=b["gt"]["masks"],
        gt_classes=b["gt"]["classes"],
        background_class=0,
    )
    if was_training:
        model.train()
    return metrics


@torch.no_grad()
def eval_all(model: D4RTModel, scenes: List[Dict], device: str, unprompted: bool = False) -> Dict[str, Dict[str, float]]:
    return {s["name"]: eval_scene(model, s, device, unprompted) for s in scenes}


def mean_metric(per_scene: Dict[str, Dict[str, float]], key: str) -> float:
    vals = [m[key] for m in per_scene.values()]
    return float(np.mean(vals)) if vals else 0.0


def build_scheduler(optimizer, num_epochs: int, warmup_epochs: int, min_lr_ratio: float = 0.05):
    """Linear warmup followed by cosine decay to min_lr_ratio * base_lr."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return LambdaLR(optimizer, lr_lambda)


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch, args,
                    train_scenes, val_scenes, train_metrics, val_metrics,
                    train_unprompted=None, val_unprompted=None, best_info=None):
    """
    Demo-compatible multi-scene checkpoint. Top-level keys mirror the single-scene format
    (pointing at the FIRST training scene) so older tooling keeps working; the full
    per-scene data lives under "scenes" (bundle 0 of each scene only).
    """
    def scene_entry(s, metrics):
        b = s["bundles"][0]
        return {
            "name": s["name"],
            "split": s["split"],
            "images": b["images"].cpu(),
            "coordinates": b["coordinates"].cpu(),
            "view_ids": b["view_ids"].cpu(),
            "gt": {k: v.cpu() for k, v in b["gt"].items()},
            "patch_start_idx": b["patch_start_idx"],
            "num_patch_tokens": b["num_patch_tokens"],
            "frame_names": b["frame_names"],
            "metrics": metrics.get(s["name"], {}),
        }

    first = train_scenes[0]["bundles"][0]
    head_config = {
        "num_views": int(args.num_views),
        "hidden_dim": 256,
        "num_classes": 20,
        "num_decoder_layers": 4,
        "patch_size": 9,
        "mask_embed_dim": 256,
        "memory_dim": 2048,
        "dropout": float(args.dropout),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "decoder_head_state_dict": model.decoder_head.state_dict(),
            "head_config": head_config,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            # Back-compat single-scene view (first training scene)
            "images": first["images"].cpu(),
            "coordinates": first["coordinates"].cpu(),
            "view_ids": first["view_ids"].cpu(),
            "gt": {k: v.cpu() for k, v in first["gt"].items()},
            "patch_start_idx": first["patch_start_idx"],
            "num_patch_tokens": first["num_patch_tokens"],
            "frame_names": first["frame_names"],
            "final_metrics": train_metrics.get(train_scenes[0]["name"], {}),
            # Multi-scene payload
            "scenes": [scene_entry(s, train_metrics) for s in train_scenes]
                      + [scene_entry(s, val_metrics) for s in val_scenes],
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "train_metrics_unprompted": train_unprompted or {},
            "val_metrics_unprompted": val_unprompted or {},
            "best_info": best_info or {},
        },
        path,
    )
    print(f"✓ Checkpoint saved to {path} ({path.stat().st_size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="D4RT multi-scene training (Milestone 2)")
    parser.add_argument("--train_scenes", type=str,
                        default="scene0000_00,scene0001_00,scene0002_00,scene0003_00",
                        help="Comma-separated scene names (under --scans_root) or paths")
    parser.add_argument("--val_scenes", type=str, default="scene0004_00",
                        help="Held-out scene(s), same format as --train_scenes")
    parser.add_argument("--scans_root", type=str, default=DEFAULT_SCANS_ROOT)
    parser.add_argument("--num_epochs", type=int, default=600)
    parser.add_argument("--warmup_epochs", type=int, default=20)
    parser.add_argument("--num_frames", type=int, default=8, help="Frames per scene per bundle")
    parser.add_argument("--num_queries", type=int, default=32)
    parser.add_argument("--num_views", type=int, default=None,
                        help="View-embedding table size; defaults to max(num_frames, 10)")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cache_device", type=str, default=None,
                        help="Where cached bundles live ('cpu' to scale past GPU memory); "
                             "defaults to --device")
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_checkpoint", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint to resume head/optimizer/scheduler from")
    # --- Milestone 2: no-object loss + unprompted eval ---
    parser.add_argument("--no_object_weight", type=float, default=0.1,
                        help="DETR eos coefficient: weight of the background class loss on "
                             "unmatched queries (0 disables no-object supervision)")
    parser.add_argument("--grid_size", type=int, default=6,
                        help="Unprompted eval uses a grid_size^2 query grid per frame")
    # --- Milestone 2: regularization ---
    parser.add_argument("--bundles_per_scene", type=int, default=1,
                        help="Cached bundles per train scene; bundle 0 is evenly-spaced frames "
                             "(eval/checkpoint), the rest are random frame samples")
    parser.add_argument("--query_jitter", type=float, default=0.0,
                        help="Std of Gaussian jitter on instance-centroid queries (0 disables)")
    parser.add_argument("--fixed_bg", action="store_true",
                        help="Keep the bundle's fixed background queries instead of resampling "
                             "them every step (Milestone-1 behavior)")
    parser.add_argument("--color_jitter", type=float, default=0.0,
                        help="Brightness/contrast jitter strength for random bundles (0 disables)")
    # --- Milestone 2: model selection ---
    parser.add_argument("--early_stop_patience", type=int, default=0,
                        help="Stop after this many evals without val mIoU improvement (0 disables)")
    args = parser.parse_args()

    if args.num_views is None:
        args.num_views = max(args.num_frames, 10)
    if args.num_frames > args.num_views:
        raise ValueError("num_frames must be <= num_views (view-embedding table size)")
    if args.cache_device is None:
        args.cache_device = args.device

    # Seed everything (bundle frame sampling + per-step query augmentation draws).
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    print(f"Using device: {device} (bundle cache on: {args.cache_device})")

    train_dirs = resolve_scene_dirs(args.train_scenes, args.scans_root)
    val_dirs = resolve_scene_dirs(args.val_scenes, args.scans_root) if args.val_scenes else []
    print(f"Train scenes ({len(train_dirs)}): {[Path(d).parent.name for d in train_dirs]}")
    print(f"Val scenes   ({len(val_dirs)}): {[Path(d).parent.name for d in val_dirs]}")

    print("\n=== Initializing Model ===")
    model = D4RTModel(
        freeze_backbone=True,
        num_views=args.num_views,
        decoder_hidden_dim=256,
        mask_embed_dim=256,
        dropout=args.dropout,
    ).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    print("\n=== Preparing Scene Bundles (one frozen-backbone pass per bundle) ===")
    train_scenes = prepare_scene_bundles(model, train_dirs, args, device, "train")
    val_scenes = prepare_scene_bundles(model, val_dirs, args, device, "val")

    loss_fn = D4RTLoss(
        num_classes=20,
        class_loss_weight=1.0,
        mask_embed_loss_weight=0.0,  # mask embeddings train via the dense mask loss
        coord_loss_weight=0.0,       # item 8.5: coordinates are matching-only
        mask_loss_weight=1.0,
        no_object_weight=args.no_object_weight if args.no_object_weight > 0 else None,
    ).to(device)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(optimizer, args.num_epochs, args.warmup_epochs)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.decoder_head.load_state_dict(ckpt["decoder_head_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0))
        print(f"✓ Resumed from {args.resume} at epoch {start_epoch}")

    print("\n=== Initial Metrics (before training) ===")
    for name, m in eval_all(model, train_scenes, device).items():
        print(f"  [train] {name}: {_format_metrics(m)}")
    for name, m in eval_all(model, val_scenes, device).items():
        print(f"  [val]   {name}: {_format_metrics(m)}")

    best_path = None
    if args.save_checkpoint:
        best_path = Path(args.save_checkpoint).parent / "checkpoint_best.pth"

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    history = []
    best = {"val_mIoU": -1.0, "epoch": -1}
    evals_since_best = 0
    t_start = time.time()
    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        epoch_loss, epoch_class, epoch_mask, epoch_matches = 0.0, 0.0, 0.0, 0
        for scene in train_scenes:
            bundle = scene["bundles"][random.randrange(len(scene["bundles"]))]
            b = bundle_to_device(bundle, device)
            coords, view_ids = make_train_queries(b, args, device)

            optimizer.zero_grad()
            class_logits, mask_embeddings, pred_masks = head_forward(model, b, coords, view_ids)
            total_loss, comps = loss_fn(
                class_logits, mask_embeddings, coords,
                b["gt"]["classes"],
                gt_mask_embeddings=None,
                gt_coordinates=b["gt"]["coordinates"],
                gt_masks=b["gt"]["masks"],
                pred_masks=pred_masks,
            )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), max_norm=args.grad_clip)
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_class += comps["class_loss"].item()
            epoch_mask += comps["mask_loss"].item() if isinstance(comps["mask_loss"], torch.Tensor) else float(comps["mask_loss"])
            epoch_matches += comps["num_matches"]
        scheduler.step()

        n = len(train_scenes)
        history.append({"total": epoch_loss / n, "class": epoch_class / n, "mask": epoch_mask / n})

        if epoch == start_epoch or (epoch + 1) % args.log_interval == 0 or epoch == args.num_epochs - 1:
            print(
                f"[Epoch {epoch + 1:4d}/{args.num_epochs}] "
                f"loss/scene: {epoch_loss / n:8.4f} "
                f"(class {epoch_class / n:6.4f}, mask {epoch_mask / n:7.4f}) "
                f"matches {epoch_matches}  lr {scheduler.get_last_lr()[0]:.2e}"
            )
        if (epoch + 1) % args.eval_interval == 0:
            tr = eval_all(model, train_scenes, device)
            va = eval_all(model, val_scenes, device)
            va_un = eval_all(model, val_scenes, device, unprompted=True)
            print(f"    train mIoU={mean_metric(tr, 'mIoU'):.3f} AP50={mean_metric(tr, 'AP50'):.3f} | "
                  f"val mIoU={mean_metric(va, 'mIoU'):.3f} AP50={mean_metric(va, 'AP50'):.3f} | "
                  f"val[grid] mIoU={mean_metric(va_un, 'mIoU'):.3f} AP50={mean_metric(va_un, 'AP50'):.3f}")

            # Model selection on held-out prompted mIoU (falls back to train mIoU w/o val scenes).
            select = mean_metric(va, "mIoU") if val_scenes else mean_metric(tr, "mIoU")
            if select > best["val_mIoU"]:
                best = {"val_mIoU": select, "epoch": epoch + 1}
                evals_since_best = 0
                if best_path is not None:
                    tr_un = eval_all(model, train_scenes, device, unprompted=True)
                    save_checkpoint(best_path, model, optimizer, scheduler, epoch + 1, args,
                                    train_scenes, val_scenes, tr, va, tr_un, va_un, best)
            else:
                evals_since_best += 1
                if args.early_stop_patience > 0 and evals_since_best >= args.early_stop_patience:
                    print(f"⏹ Early stop at epoch {epoch + 1}: no val mIoU improvement in "
                          f"{evals_since_best} evals (best {best['val_mIoU']:.3f} @ epoch {best['epoch']}).")
                    break
        if np.isnan(history[-1]["total"]):
            print("⚠ Loss is NaN — stopping.")
            break

    print(f"\nTraining took {(time.time() - t_start) / 60:.1f} min")
    print("=" * 70)
    print("FINAL METRICS (last epoch — see checkpoint_best.pth for the model-selected head)")
    print("=" * 70)
    train_metrics = eval_all(model, train_scenes, device)
    val_metrics = eval_all(model, val_scenes, device)
    train_unprompted = eval_all(model, train_scenes, device, unprompted=True)
    val_unprompted = eval_all(model, val_scenes, device, unprompted=True)
    print("Train scenes — prompted (queries at GT centroids):")
    for name, m in train_metrics.items():
        print(f"  {name}: {_format_metrics(m)}")
    print("Train scenes — UNPROMPTED (uniform query grid):")
    for name, m in train_unprompted.items():
        print(f"  {name}: {_format_metrics(m)}")
    print("Held-out scene(s) — prompted:")
    for name, m in val_metrics.items():
        print(f"  {name}: {_format_metrics(m)}")
    print("Held-out scene(s) — UNPROMPTED:")
    for name, m in val_unprompted.items():
        print(f"  {name}: {_format_metrics(m)}")
    print(f"\nMean train mIoU={mean_metric(train_metrics, 'mIoU'):.3f} "
          f"(unprompted {mean_metric(train_unprompted, 'mIoU'):.3f}), "
          f"AP50={mean_metric(train_metrics, 'AP50'):.3f} "
          f"(unprompted {mean_metric(train_unprompted, 'AP50'):.3f})")
    if val_metrics:
        print(f"Mean val   mIoU={mean_metric(val_metrics, 'mIoU'):.3f} "
              f"(unprompted {mean_metric(val_unprompted, 'mIoU'):.3f}), "
              f"AP50={mean_metric(val_metrics, 'AP50'):.3f} "
              f"(unprompted {mean_metric(val_unprompted, 'AP50'):.3f})")
    if best["epoch"] > 0:
        print(f"Best val mIoU during training: {best['val_mIoU']:.3f} @ epoch {best['epoch']}"
              + (f" (saved to {best_path})" if best_path is not None else ""))

    if history:
        first, last = history[0]["total"], history[-1]["total"]
        red = (first - last) / first * 100 if first > 0 else 0.0
        print(f"\nLoss/scene: {first:.4f} → {last:.4f}  ({red:.1f}% reduction)")

    if args.save_checkpoint:
        save_checkpoint(Path(args.save_checkpoint), model, optimizer, scheduler,
                        args.num_epochs, args, train_scenes, val_scenes,
                        train_metrics, val_metrics, train_unprompted, val_unprompted, best)

    ok = mean_metric(train_metrics, "mIoU") > 0.5
    print("\n✅ SUCCESS" if ok else "\n⚠ Train mIoU below 0.5 — inspect the run.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
