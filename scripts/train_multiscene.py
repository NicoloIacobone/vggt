#!/usr/bin/env python3
"""
Multi-scene training for the D4RT instance-segmentation head (MILESTONE_1 item 8.7).

Trains the decoder head on several ScanNet scenes simultaneously (overfit protocol: one
fixed, deterministic batch per scene) and evaluates on a held-out scene the model never
trains on. The frozen VGGT backbone is run ONCE per scene up front and its global features
are cached, so every epoch only runs the lightweight decoder head.

Note on evaluation: queries are point prompts (D4RT-style). On the held-out scene the
query points are the GT instance centroids, i.e. we measure "given a point on the object,
can the model segment + classify it in an unseen scene", not unprompted detection.

Usage:
    python scripts/train_multiscene.py \
        --train_scenes scene0000_00,scene0001_00,scene0002_00,scene0003_00 \
        --val_scenes scene0004_00 \
        --num_epochs 600 --num_frames 8 --num_queries 32 \
        --save_checkpoint /cluster/work/igp_psr/niacobone/distillation/output/<run>/checkpoint.pth
"""

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

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
from train_overfit import D4RTModel, build_gt_targets, generate_query_points, _format_metrics

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


@torch.no_grad()
def prepare_scene_bundles(
    model: D4RTModel,
    dataset: ScanNetMultiSceneDataset,
    num_queries: int,
    device: str,
    split: str,
) -> List[Dict]:
    """
    Build one FIXED training/eval bundle per scene: images, query points, dense GT targets,
    and the cached frozen-backbone features. Deterministic (frame_sampling='even' + seeded
    background points), so every epoch sees the identical batch per scene.
    """
    bundles = []
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for batch in loader:
        name = batch["scene_name"][0] if isinstance(batch["scene_name"], (list, tuple)) else str(batch["scene_name"])
        images = batch["images"].to(device)  # [1, S, 3, H, W]
        coordinates, view_ids = generate_query_points(batch, num_queries, device)

        t0 = time.time()
        agg_list, patch_start_idx = model.backbone.aggregator(images)
        features = agg_list[-1].detach()  # [1, S, P, 2048] — cached; backbone never reruns
        num_patch_tokens = features.shape[2] - patch_start_idx
        gt = build_gt_targets(batch, patch_start_idx, num_patch_tokens, device)

        bundles.append({
            "name": name,
            "split": split,
            "images": images,
            "coordinates": coordinates,
            "view_ids": view_ids,
            "features": features,
            "patch_start_idx": int(patch_start_idx),
            "num_patch_tokens": int(num_patch_tokens),
            "gt": gt,
            "frame_names": batch.get("frame_names", None),
        })
        print(
            f"  [{split}] {name}: frames={images.shape[1]}, queries={coordinates.shape[1]}, "
            f"instances={gt['classes'].shape[0]}, features={tuple(features.shape)} "
            f"({time.time() - t0:.1f}s backbone)"
        )
    return bundles


def head_forward(model: D4RTModel, bundle: Dict):
    """Decoder-head-only forward on a scene bundle (uses the cached backbone features)."""
    return model.decoder_head(
        bundle["coordinates"], bundle["view_ids"], bundle["images"],
        bundle["features"], bundle["patch_start_idx"],
    )


@torch.no_grad()
def eval_bundle(model: D4RTModel, bundle: Dict) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    class_logits, _, pred_masks = head_forward(model, bundle)
    metrics = compute_instance_segmentation_metrics(
        pred_masks=pred_masks[0],
        class_logits=class_logits[0],
        gt_masks=bundle["gt"]["masks"],
        gt_classes=bundle["gt"]["classes"],
        background_class=0,
    )
    if was_training:
        model.train()
    return metrics


@torch.no_grad()
def eval_all(model: D4RTModel, bundles: List[Dict]) -> Dict[str, Dict[str, float]]:
    return {b["name"]: eval_bundle(model, b) for b in bundles}


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
                    train_bundles, val_bundles, train_metrics, val_metrics):
    """
    Demo-compatible multi-scene checkpoint. Top-level keys mirror the single-scene format
    (pointing at the FIRST training scene) so older tooling keeps working; the full
    per-scene data lives under "scenes".
    """
    def bundle_entry(b, metrics):
        return {
            "name": b["name"],
            "split": b["split"],
            "images": b["images"].cpu(),
            "coordinates": b["coordinates"].cpu(),
            "view_ids": b["view_ids"].cpu(),
            "gt": {k: v.cpu() for k, v in b["gt"].items()},
            "patch_start_idx": b["patch_start_idx"],
            "num_patch_tokens": b["num_patch_tokens"],
            "frame_names": b["frame_names"],
            "metrics": metrics.get(b["name"], {}),
        }

    first = train_bundles[0]
    head_config = {
        "num_views": int(args.num_views),
        "hidden_dim": 256,
        "num_classes": 20,
        "num_decoder_layers": 4,
        "patch_size": 9,
        "mask_embed_dim": 256,
        "memory_dim": 2048,
        "dropout": 0.0,
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
            "final_metrics": train_metrics.get(first["name"], {}),
            # Multi-scene payload
            "scenes": [bundle_entry(b, train_metrics) for b in train_bundles]
                      + [bundle_entry(b, val_metrics) for b in val_bundles],
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        },
        path,
    )
    print(f"✓ Checkpoint saved to {path} ({path.stat().st_size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="D4RT multi-scene training")
    parser.add_argument("--train_scenes", type=str,
                        default="scene0000_00,scene0001_00,scene0002_00,scene0003_00",
                        help="Comma-separated scene names (under --scans_root) or paths")
    parser.add_argument("--val_scenes", type=str, default="scene0004_00",
                        help="Held-out scene(s), same format as --train_scenes")
    parser.add_argument("--scans_root", type=str, default=DEFAULT_SCANS_ROOT)
    parser.add_argument("--num_epochs", type=int, default=600)
    parser.add_argument("--warmup_epochs", type=int, default=20)
    parser.add_argument("--num_frames", type=int, default=8, help="Frames per scene (evenly spaced)")
    parser.add_argument("--num_queries", type=int, default=32)
    parser.add_argument("--num_views", type=int, default=None,
                        help="View-embedding table size; defaults to max(num_frames, 10)")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_checkpoint", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint to resume head/optimizer/scheduler from")
    args = parser.parse_args()

    if args.num_views is None:
        args.num_views = max(args.num_frames, 10)
    if args.num_frames > args.num_views:
        raise ValueError("num_frames must be <= num_views (view-embedding table size)")

    # Seed everything: with frame_sampling='even' the whole run is deterministic.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    print(f"Using device: {device}")

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

    print("\n=== Preparing Scene Bundles (one frozen-backbone pass per scene) ===")
    common = dict(num_frames=args.num_frames, img_size=518, frame_sampling="even")
    train_bundles = prepare_scene_bundles(
        model, ScanNetMultiSceneDataset(train_dirs, **common), args.num_queries, device, "train")
    val_bundles = prepare_scene_bundles(
        model, ScanNetMultiSceneDataset(val_dirs, **common), args.num_queries, device, "val") if val_dirs else []

    loss_fn = D4RTLoss(
        num_classes=20,
        class_loss_weight=1.0,
        mask_embed_loss_weight=0.0,  # mask embeddings train via the dense mask loss
        coord_loss_weight=0.0,       # item 8.5: coordinates are matching-only
        mask_loss_weight=1.0,
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
    init_train = eval_all(model, train_bundles)
    for name, m in init_train.items():
        print(f"  [train] {name}: {_format_metrics(m)}")
    init_val = eval_all(model, val_bundles)
    for name, m in init_val.items():
        print(f"  [val]   {name}: {_format_metrics(m)}")

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    history = []
    t_start = time.time()
    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        epoch_loss, epoch_class, epoch_mask, epoch_matches = 0.0, 0.0, 0.0, 0
        for bundle in train_bundles:
            optimizer.zero_grad()
            class_logits, mask_embeddings, pred_masks = head_forward(model, bundle)
            total_loss, comps = loss_fn(
                class_logits, mask_embeddings, bundle["coordinates"],
                bundle["gt"]["classes"],
                gt_mask_embeddings=None,
                gt_coordinates=bundle["gt"]["coordinates"],
                gt_masks=bundle["gt"]["masks"],
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

        n = len(train_bundles)
        history.append({"total": epoch_loss / n, "class": epoch_class / n, "mask": epoch_mask / n})

        if epoch == start_epoch or (epoch + 1) % args.log_interval == 0 or epoch == args.num_epochs - 1:
            print(
                f"[Epoch {epoch + 1:4d}/{args.num_epochs}] "
                f"loss/scene: {epoch_loss / n:8.4f} "
                f"(class {epoch_class / n:6.4f}, mask {epoch_mask / n:7.4f}) "
                f"matches {epoch_matches}  lr {scheduler.get_last_lr()[0]:.2e}"
            )
        if (epoch + 1) % args.eval_interval == 0 and epoch != args.num_epochs - 1:
            tr = eval_all(model, train_bundles)
            va = eval_all(model, val_bundles)
            print(f"    train mIoU={mean_metric(tr, 'mIoU'):.3f} AP50={mean_metric(tr, 'AP50'):.3f} | "
                  f"val mIoU={mean_metric(va, 'mIoU'):.3f} AP50={mean_metric(va, 'AP50'):.3f}")
        if np.isnan(history[-1]["total"]):
            print("⚠ Loss is NaN — stopping.")
            break

    print(f"\nTraining took {(time.time() - t_start) / 60:.1f} min")
    print("=" * 70)
    print("FINAL METRICS")
    print("=" * 70)
    train_metrics = eval_all(model, train_bundles)
    val_metrics = eval_all(model, val_bundles)
    print("Train scenes (overfit):")
    for name, m in train_metrics.items():
        print(f"  {name}: {_format_metrics(m)}")
    print("Held-out scene(s) (generalization):")
    for name, m in val_metrics.items():
        print(f"  {name}: {_format_metrics(m)}")
    print(f"\nMean train mIoU={mean_metric(train_metrics, 'mIoU'):.3f}, "
          f"AP50={mean_metric(train_metrics, 'AP50'):.3f}, "
          f"class_acc={mean_metric(train_metrics, 'class_acc'):.3f}")
    if val_metrics:
        print(f"Mean val   mIoU={mean_metric(val_metrics, 'mIoU'):.3f}, "
              f"AP50={mean_metric(val_metrics, 'AP50'):.3f}, "
              f"class_acc={mean_metric(val_metrics, 'class_acc'):.3f}")

    if history:
        first, last = history[0]["total"], history[-1]["total"]
        red = (first - last) / first * 100 if first > 0 else 0.0
        print(f"\nLoss/scene: {first:.4f} → {last:.4f}  ({red:.1f}% reduction)")

    if args.save_checkpoint:
        save_checkpoint(Path(args.save_checkpoint), model, optimizer, scheduler,
                        args.num_epochs, args, train_bundles, val_bundles,
                        train_metrics, val_metrics)

    ok = mean_metric(train_metrics, "mIoU") > 0.5
    print("\n✅ SUCCESS" if ok else "\n⚠ Train mIoU below 0.5 — inspect the run.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
