#!/usr/bin/env python3
"""
Single-frame MaskDINO training on frozen VGGT features (docs/MASKDINO.md).

This file is the entry point only: CLI, model/criterion construction, the epoch loop and
checkpointing. The parts worth reading on their own live next door —

    train/maskdino_data.py   per-frame GT + frozen-backbone feature cache + batching
    train/maskdino_eval.py   per-frame scoring and the RGB|GT|pred figures
    train/perframe.py        the scoring rules shared with scripts/eval_perframe.py
    models/maskdino/         the ported decoder, pixel decoder, matcher and criterion

What is different from the legacy D4RT training loop (legacy/d4rt/scripts/train_multiscene.py):
  - The sample is a FRAME, not a scene bundle. Frames are pooled across scenes and shuffled;
    one step is `--batch_frames` independent images (the supervisor's single-frame constraint).
  - GT is per frame: labels (0..18), binary masks at the mask-grid resolution, and boxes
    derived from those masks (MaskDINO is a box-aware detector).
  - Loss = MaskDINO's SetCriterion (focal + point-sampled BCE/Dice + L1/GIoU) over the final
    layer, every intermediate layer, the encoder's interm output and the denoising queries.
  - Eval is per frame with sigmoid scoring (docs/MASKDINO.md §6).

The frozen VGGT backbone still runs ONCE per frame up front; every epoch trains only the head.

Usage (smoke test):
    python scripts/train_maskdino.py --train_scenes scene0000_00 --val_scenes scene0001_00 \
        --num_epochs 20 --num_frames 4 --num_queries 100 --dec_layers 3 --enc_layers 3
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.maskdino import (NUM_SCANNET_CLASSES, HungarianMatcher, MultiFrameHungarianMatcher,
                             SetCriterion, build_weight_dict)
from models.maskdino.model import MaskDINOVGGTModel
from train.common import (DEFAULT_SCANS_ROOT, append_jsonl, build_scheduler, resolve_scene_dirs)
from train.maskdino_data import (bundle_index, frame_index, gather_batch, gather_bundle_batch,
                                 gather_token_xyz, prepare_scenes)
from train.eval_metrics import CONSISTENCY_KEYS
from train.maskdino_eval import eval_scenes, fmt, mean_metric, visualize

# ------------------------------------------------------------------------------------------
# Checkpointing
# ------------------------------------------------------------------------------------------

def save_checkpoint(path: Path, model, args, epoch, train_metrics, val_metrics, best_info,
                    optimizer=None, scheduler=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "head_state_dict": model.head.state_dict(),
        "head_config": model.head.head_config,
        "epoch": epoch,
        "args": vars(args),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "best_info": best_info or {},
        "trial": "maskdino_single_frame",
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, path)
    print(f"✓ Checkpoint saved to {path} ({path.stat().st_size / 1e6:.1f} MB)")


def update_best(best, metric_key, select, epoch, path, model, args, train_metrics, val_metrics):
    """Track a single best-so-far metric and save its checkpoint when it improves.

    `best` and the returned dict both use `metric_key` for the tracked value and "epoch" for
    the epoch it was reached at, matching the shape `save_checkpoint`'s `best_info` expects.
    """
    if select > best[metric_key]:
        best = {metric_key: select, "epoch": epoch}
        if path:
            save_checkpoint(path, model, args, epoch, train_metrics, val_metrics, best)
    return best


# ------------------------------------------------------------------------------------------

def build_argparser():
    p = argparse.ArgumentParser(description="Single-frame MaskDINO on frozen VGGT features")
    # data
    p.add_argument("--train_scenes", type=str, default="scene0000_00,scene0001_00")
    p.add_argument("--val_scenes", type=str, default="scene0080_00")
    p.add_argument("--scans_root", type=str, default=DEFAULT_SCANS_ROOT)
    p.add_argument("--num_frames", type=int, default=8, help="Cached frames per scene")
    p.add_argument("--eval_num_frames", type=int, default=None,
                   help="Frames per VAL bundle, if different from --num_frames (default: same). "
                        "Training at a wider bundle silently moves the per-bundle ruler — a "
                        "volume over 16 views is a harder object than one over 8 — so pin this "
                        "to the baseline's width to keep bundle_* comparable "
                        "(docs/MASKDINO.md §8.4). Train scenes always use --num_frames.")
    p.add_argument("--class_agnostic", action="store_true",
                   help="Train ONE class ('object') instead of the 19 ScanNet ones "
                        "(docs/todo.md 6e). Required to mix datasets whose taxonomies the head "
                        "cannot name — ScanNet++, Infinigen — and it is also the setting "
                        "FAST3DIS and IGGT report in. The head is built with num_classes=1 and "
                        "`build_frame_targets` collapses every GT label onto it, so no instance "
                        "is dropped for being unnameable. Default off: every published number "
                        "in this project is class-aware.")
    p.add_argument("--class_level", action="store_true",
                   help="Use per-class masks instead of the default per-instance GT "
                        "(masks_instance/); per-instance is what an instance-segmentation "
                        "decoder should be trained on, so it is the default here.")
    p.add_argument("--bundles_per_scene", type=int, default=1,
                   help="Extra frame draws per train scene (bundle 0 = evenly-spaced frames)")
    p.add_argument("--color_jitter", type=float, default=0.0)
    # features
    p.add_argument("--feature_mode", type=str, default="single", choices=["single", "bundle"],
                   help="'single' (default) runs VGGT per frame (S=1) — a true single-frame "
                        "model. 'bundle' runs it once per scene so tokens are multi-view aware.")
    p.add_argument("--feature_layers", type=str, default="-1",
                   help="Comma-separated aggregator layer indices to concatenate (e.g. "
                        "'4,11,17,23'). Default '-1' = last layer only (same cache size as the "
                        "D4RT arms).")
    p.add_argument("--backbone_dtype", type=str, default="float32",
                   choices=["float32", "bfloat16", "float16"])
    p.add_argument("--cache_dtype", type=str, default="float32",
                   choices=["float32", "bfloat16", "float16"],
                   help="Storage dtype of the cached tokens (float16 halves host memory; the "
                        "head always runs in float32)")
    p.add_argument("--cache_device", type=str, default=None, help="'cpu' to scale scene count")
    # head
    p.add_argument("--num_queries", type=int, default=300)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--enc_layers", type=int, default=6)
    p.add_argument("--dec_layers", type=int, default=9)
    p.add_argument("--num_feature_levels", type=int, default=3)
    p.add_argument("--nheads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--two_stage", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dn", type=str, default="seg", choices=["no", "seg"])
    p.add_argument("--dn_num", type=int, default=100)
    p.add_argument("--noise_scale", type=float, default=0.4)
    p.add_argument("--initialize_box_type", type=str, default="bitmask",
                   choices=["no", "bitmask"],
                   help="Seed the decoder's anchor boxes from the initial predicted masks. "
                        "Upstream's third option 'mask2box' is not ported.")
    # multi-frame (docs/MASKDINO.md §8) — off by default, so every single-frame result stands
    p.add_argument("--multi_frame", action="store_true",
                   help="Share ONE query set across the frames of a bundle: the sample becomes a "
                        "bundle of --num_frames frames, queries are selected once per bundle and "
                        "matched once per bundle over the multi-view mask volume. Adds the "
                        "per-bundle (multi-view) metrics the D4RT arms were scored on.")
    p.add_argument("--cross_frame_attn", action=argparse.BooleanOptionalAction, default=None,
                   help="Cross-frame self-attention block after every decoder layer (defaults to "
                        "on with --multi_frame, off otherwise). --no-cross_frame_attn isolates "
                        "how much of the multi-frame gain comes from the block rather than from "
                        "shared query init + bundle matching.")
    p.add_argument("--anchor_3d", action="store_true",
                   help="Replace the decoder's 2D DAB anchor box with a 3D anchor read off "
                        "VGGT's own point head (docs/MASKDINO.md §8.3, docs/todo.md 2d). The "
                        "anchor is one (x,y,z,log r) per query per bundle, projected into each "
                        "view as a soft nearest patch and refined by Delta(xyz, log r) instead "
                        "of Delta(box). Needs --feature_mode bundle; meant to be run with "
                        "--multi_frame. An ABLATION against the 2D-box default, not a new "
                        "mechanism (FAST3DIS owns it) — see docs/RELATED_WORK.md.")
    p.add_argument("--batch_bundles", type=int, default=1,
                   help="Bundles per step in --multi_frame mode (batch = this x --num_frames "
                        "frames). Ignored in single-frame mode, which uses --batch_frames.")
    p.add_argument("--mask_upsample", type=int, default=1, choices=[1, 2, 4],
                   help="1 (default) = masks on the 37x37 patch grid, the same grid every D4RT "
                        "arm was scored on; 2/4 predict (and supervise) at 74/148 px.")
    # losses
    p.add_argument("--class_weight", type=float, default=4.0)
    p.add_argument("--mask_weight", type=float, default=5.0)
    p.add_argument("--dice_weight", type=float, default=5.0)
    p.add_argument("--box_weight", type=float, default=5.0)
    p.add_argument("--giou_weight", type=float, default=2.0)
    p.add_argument("--train_num_points", type=int, default=0,
                   help="PointRend mask-loss points; 0 (default) supervises every pixel, which "
                        "is affordable on a 37x37 grid. MaskDINO's COCO value is 12544.")
    p.add_argument("--matcher_num_points", type=int, default=1369,
                   help="Points used for the matcher's mask cost (37*37 by default)")
    # optim
    p.add_argument("--num_epochs", type=int, default=300)
    p.add_argument("--schedule_epochs", type=int, default=None)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--batch_frames", type=int, default=8)
    p.add_argument("--eval_batch_frames", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=0.1)
    p.add_argument("--score_threshold", type=float, default=0.25,
                   help="Objectness threshold for the metrics (MaskDINO's OBJECT_MASK_THRESHOLD)")
    p.add_argument("--eval_topk", type=int, default=100,
                   help="Max detections kept per frame when scoring (COCO/MaskDINO's "
                        "test_topk_per_image). 0 = keep every query. Also bounds eval cost.")
    p.add_argument("--eval_full_res", action="store_true",
                   help="Additionally score every frame at the dataset's full 518x518 GT "
                        "resolution (keys full_*, docs/MASKDINO.md §6.5): predictions are "
                        "bilinearly upsampled, GT comes from the cached full-res id map. The "
                        "kept prediction set is still decided on the mask grid, so full_* "
                        "isolates boundary quality. Costs ~2 GB cache at 500 scenes.")
    p.add_argument("--eval_train_scenes", type=int, default=10,
                   help="How many train scenes to score for the train-side diagnostic metric "
                        "(evenly spaced; 0 = all). The train number is only a "
                        "memorisation/overfit read-out, and scoring every scene makes eval cost "
                        "grow with the training set — at 190 scenes it dominates the run.")
    # bookkeeping
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--eval_interval", type=int, default=10)
    p.add_argument("--log_interval", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_checkpoint", type=str, default=None)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--no_visualize", action="store_true")
    return p


def main():
    args = build_argparser().parse_args()
    if args.schedule_epochs is None:
        args.schedule_epochs = args.num_epochs
    if args.cache_device is None:
        args.cache_device = args.device
    if args.cross_frame_attn is None:
        args.cross_frame_attn = args.multi_frame
    if args.cross_frame_attn and not args.multi_frame:
        raise SystemExit("--cross_frame_attn needs --multi_frame: with one frame per sample the "
                         "block has nothing to attend across.")
    if args.anchor_3d:
        if args.feature_mode != "bundle":
            raise SystemExit("--anchor_3d needs --feature_mode bundle (docs/MASKDINO.md §8.3): "
                             "in 'single' mode the aggregator sees one frame at a time, so each "
                             "frame's pointmap is in its own coordinate frame and a 3D anchor "
                             "shared across views has no meaning.")
        if not args.multi_frame:
            print("⚠ --anchor_3d without --multi_frame: the anchor is still one 3D point per "
                  "query, but each frame gets its own query set, so it cannot tie identity "
                  "across views. The intended base is --multi_frame --feature_mode bundle.")
    args.feature_layers = [int(x) for x in args.feature_layers.split(",") if x.strip()]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    print(f"Using device: {device} (feature cache on {args.cache_device}, {args.cache_dtype})")

    train_dirs = resolve_scene_dirs(args.train_scenes, args.scans_root)
    val_dirs = resolve_scene_dirs(args.val_scenes, args.scans_root) if args.val_scenes else []
    print(f"Train scenes ({len(train_dirs)}), val scenes ({len(val_dirs)})")

    head_kwargs = dict(
        memory_dim=2048 * len(args.feature_layers), hidden_dim=args.hidden_dim,
        mask_dim=args.hidden_dim,
        num_classes=1 if args.class_agnostic else NUM_SCANNET_CLASSES,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels, enc_layers=args.enc_layers,
        dec_layers=args.dec_layers, nheads=args.nheads, dropout=args.dropout,
        two_stage=args.two_stage, learn_tgt=not args.two_stage, initial_pred=True,
        initialize_box_type=args.initialize_box_type if args.two_stage else "no",
        dn=args.dn, dn_num=args.dn_num, noise_scale=args.noise_scale,
        mask_upsample=args.mask_upsample, cross_frame_attn=args.cross_frame_attn,
        anchor_3d=args.anchor_3d,
    )
    print("\n=== Initializing model ===")
    model = MaskDINOVGGTModel(head_kwargs).to(device)
    trainable = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    print(f"Trainable head parameters: {trainable:,}")

    # Built BEFORE the feature cache on purpose: caching a 3520-scene mixture takes ~3 h,
    # and an error in any of these constructors used to surface only after it (job 9901119
    # died exactly that way, docs/todo.md 2f). Nothing here reads the cache.
    matcher = HungarianMatcher(cost_class=args.class_weight, cost_mask=args.mask_weight,
                               cost_dice=args.dice_weight, cost_box=args.box_weight,
                               cost_giou=args.giou_weight, num_points=args.matcher_num_points)
    weight_dict = build_weight_dict(args.class_weight, args.mask_weight, args.dice_weight,
                                    args.box_weight, args.giou_weight, dec_layers=args.dec_layers,
                                    two_stage=args.two_stage, dn=args.dn)
    # In multi-frame mode the assignment is made once per bundle over the multi-view mask volume
    # and then projected onto the frames; every loss stays the per-frame loss it already was.
    bundle_matcher = MultiFrameHungarianMatcher(
        cost_class=args.class_weight, cost_mask=args.mask_weight, cost_dice=args.dice_weight,
        cost_box=args.box_weight, cost_giou=args.giou_weight,
        num_points=args.matcher_num_points) if args.multi_frame else None
    criterion = SetCriterion(model.head.num_classes, matcher, weight_dict,
                             losses=["labels", "masks", "boxes"],
                             num_points=args.train_num_points, dn=args.dn,
                             dn_losses=["labels", "masks", "boxes"],
                             bundle_matcher=bundle_matcher).to(device)

    optimizer = AdamW([p for p in model.head.parameters() if p.requires_grad],
                      lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args.schedule_epochs, args.warmup_epochs)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.head.load_state_dict(ckpt["head_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0))
        print(f"✓ Resumed from {args.resume} at epoch {start_epoch}")

    print("\n=== Caching frozen-backbone features ===")
    train_scenes = prepare_scenes(model, train_dirs, args, device, "train")
    val_scenes = prepare_scenes(model, val_dirs, args, device, "val")
    if args.multi_frame:
        # One sample = one bundle of --num_frames frames, sharing a query set.
        train_samples = bundle_index(train_scenes)
        step_size = args.batch_bundles
        print(f"Training samples (bundles with >=1 instance): {len(train_samples)} "
              f"x {args.num_frames} frames")
        if args.eval_num_frames and args.eval_num_frames != args.num_frames:
            print(f"  val bundles pinned to {args.eval_num_frames} frames "
                  f"(--eval_num_frames): bundle_* stays on the {args.eval_num_frames}-view "
                  f"ruler while training sees {args.num_frames}")
    else:
        train_samples = frame_index(train_scenes)
        step_size = args.batch_frames
        print(f"Training samples (frames with >=1 instance): {len(train_samples)}")
    # Evenly-spaced subset of train scenes for the diagnostic train metric (see
    # --eval_train_scenes): eval must not scale with the training-set size.
    if 0 < args.eval_train_scenes < len(train_scenes):
        idx = np.linspace(0, len(train_scenes) - 1, args.eval_train_scenes).round().astype(int)
        train_eval_scenes = [train_scenes[i] for i in sorted(set(idx.tolist()))]
    else:
        train_eval_scenes = train_scenes
    print(f"Train scenes scored at each eval: {len(train_eval_scenes)}/{len(train_scenes)}")
    if not train_samples:
        raise SystemExit("No training frames with ground-truth instances — check the GT tree.")

    run_dir = Path(args.save_checkpoint).parent if args.save_checkpoint else None
    metrics_path = run_dir / "metrics.jsonl" if run_dir else None
    best_path = run_dir / "checkpoint_best.pth" if run_dir else None
    best_ap_path = run_dir / "checkpoint_best_ap50.pth" if run_dir else None
    best_bundle_path = run_dir / "checkpoint_best_bundle.pth" if run_dir and args.multi_frame else None
    if run_dir:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))

    print("\n=== Initial metrics (untrained head) ===")
    for name, m in eval_scenes(model, val_scenes, args, device).items():
        print(f"  [val] {name}: {fmt(m)}")

    print("\n" + "=" * 70 + "\nTRAINING\n" + "=" * 70)
    best = {"val_mIoU": -1.0, "epoch": -1}
    best_ap = {"val_AP50": -1.0, "epoch": -1}
    best_bundle = {"val_bundle_AP50": -1.0, "epoch": -1}
    t_start = time.time()
    steps_per_epoch = max(1, (len(train_samples) + step_size - 1) // step_size)

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        random.shuffle(train_samples)
        epoch_loss = epoch_ce = epoch_mask = epoch_box = 0.0
        for step in range(steps_per_epoch):
            chunk = train_samples[step * step_size:(step + 1) * step_size]
            if not chunk:
                continue

            optimizer.zero_grad(set_to_none=True)
            xyz = gather_token_xyz(train_scenes, chunk, device)
            if args.multi_frame:
                feats, targets, bundles, psi, s = gather_bundle_batch(train_scenes, chunk, device)
                out, mask_dict = model.head(feats, psi, targets, frames_per_sample=s,
                                            token_xyz=xyz)
                losses = criterion(out, targets, mask_dict, bundle_targets=bundles,
                                   frames_per_sample=s)
            else:
                feats, targets, psi = gather_batch(train_scenes, chunk, device)
                out, mask_dict = model.head(feats, psi, targets, token_xyz=xyz)
                losses = criterion(out, targets, mask_dict)
            total = sum(losses[k] * weight_dict[k] for k in losses if k in weight_dict)
            total.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.head.parameters(), args.grad_clip)
            optimizer.step()

            epoch_loss += float(total)
            epoch_ce += float(losses["loss_ce"])
            epoch_mask += float(losses["loss_mask"]) + float(losses["loss_dice"])
            epoch_box += float(losses["loss_bbox"]) + float(losses["loss_giou"])
        scheduler.step()
        n = steps_per_epoch
        mean_loss = epoch_loss / n

        if epoch == start_epoch or (epoch + 1) % args.log_interval == 0 \
                or epoch == args.num_epochs - 1:
            print(f"[Epoch {epoch + 1:4d}/{args.num_epochs}] loss {mean_loss:8.3f} "
                  f"(cls {epoch_ce / n:6.3f}, mask+dice {epoch_mask / n:6.3f}, "
                  f"box+giou {epoch_box / n:6.3f})  lr {scheduler.get_last_lr()[0]:.2e}")

        if (epoch + 1) % args.eval_interval == 0 or epoch == args.num_epochs - 1:
            tr = eval_scenes(model, train_eval_scenes, args, device)
            va = eval_scenes(model, val_scenes, args, device)
            print(f"    train mIoU={mean_metric(tr, 'mIoU'):.3f} AP50={mean_metric(tr, 'AP50'):.3f}"
                  f" | val mIoU={mean_metric(va, 'mIoU'):.3f} AP50={mean_metric(va, 'AP50'):.3f}"
                  f" AP75={mean_metric(va, 'AP75'):.3f} mAP={mean_metric(va, 'mAP'):.3f}"
                  f" | val all-query mIoU={mean_metric(va, 'mIoU_all'):.3f} "
                  f"AP50={mean_metric(va, 'AP50_all'):.3f} "
                  f"(kept {mean_metric(va, 'num_pred'):.1f}/frame)")
            if args.multi_frame:
                # the multi-view protocol of the retired arms — meaningful again now that a
                # query owns one instance across all views (docs/MASKDINO.md §8)
                print(f"    val per-bundle mIoU={mean_metric(va, 'bundle_mIoU'):.3f} "
                      f"AP50={mean_metric(va, 'bundle_AP50'):.3f} "
                      f"AP75={mean_metric(va, 'bundle_AP75'):.3f} "
                      f"mAP={mean_metric(va, 'bundle_mAP'):.3f}")
                # cross-view consistency of the shared queries (docs/MASKDINO.md §6.6)
                print(f"    val cross-view consistency="
                      f"{mean_metric(va, 'bundle_view_consistency'):.3f} "
                      f"id_switch={mean_metric(va, 'bundle_id_switch'):.3f} "
                      f"(matched {mean_metric(va, 'bundle_num_matched'):.1f}/bundle)")
            if args.eval_full_res:
                # the full-resolution ruler (docs/MASKDINO.md §6.5) — same detections, 518x518
                print(f"    val full-res  mIoU={mean_metric(va, 'full_mIoU'):.3f} "
                      f"AP50={mean_metric(va, 'full_AP50'):.3f} "
                      f"AP75={mean_metric(va, 'full_AP75'):.3f} "
                      f"mAP={mean_metric(va, 'full_mAP'):.3f}")
            record = {"epoch": epoch + 1, "lr": float(scheduler.get_last_lr()[0]),
                      "loss": mean_loss, "class_loss": epoch_ce / n,
                      "mask_loss": epoch_mask / n, "box_loss": epoch_box / n}
            keys = ["mIoU", "AP50", "AP75", "mAP", "class_acc", "num_pred",
                    "mIoU_all", "AP50_all", "AP75_all", "mAP_all"]
            if args.multi_frame:
                keys += [f"bundle_{k}" for k in keys]
                keys += [f"bundle_{k}" for k in CONSISTENCY_KEYS]   # §6.6
            if args.eval_full_res:
                # after the bundle expansion: bundle_* stays on the mask grid (§6.5)
                keys += [f"full_{k}" for k in ("mIoU", "AP50", "AP75", "mAP", "class_acc",
                                               "num_pred", "mIoU_all", "AP50_all", "AP75_all",
                                               "mAP_all")]
            for split, d in (("train", tr), ("val", va)):
                for key in keys:
                    record[f"{split}_{key}"] = mean_metric(d, key)
            if metrics_path:
                append_jsonl(metrics_path, record)

            select = mean_metric(va, "mIoU") if val_scenes else mean_metric(tr, "mIoU")
            best = update_best(best, "val_mIoU", select, epoch + 1, best_path, model, args, tr, va)
            select_ap = mean_metric(va, "AP50") if val_scenes else mean_metric(tr, "AP50")
            best_ap = update_best(best_ap, "val_AP50", select_ap, epoch + 1, best_ap_path, model,
                                  args, tr, va)
            if args.multi_frame:
                select_bundle = (mean_metric(va, "bundle_AP50") if val_scenes
                                 else mean_metric(tr, "bundle_AP50"))
                best_bundle = update_best(best_bundle, "val_bundle_AP50", select_bundle,
                                          epoch + 1, best_bundle_path, model, args, tr, va)

        if not np.isfinite(mean_loss):
            print("⚠ Loss is not finite — stopping.")
            break

    print(f"\nTraining took {(time.time() - t_start) / 60:.1f} min")
    print("=" * 70 + "\nFINAL METRICS (last epoch)\n" + "=" * 70)
    train_metrics = eval_scenes(model, train_eval_scenes, args, device)
    val_metrics = eval_scenes(model, val_scenes, args, device)
    for name, m in train_metrics.items():
        print(f"  [train] {name}: {fmt(m)}")
    for name, m in val_metrics.items():
        print(f"  [val]   {name}: {fmt(m)}")
    print(f"\nMean train mIoU={mean_metric(train_metrics, 'mIoU'):.3f} "
          f"AP50={mean_metric(train_metrics, 'AP50'):.3f}")
    print(f"Mean val   mIoU={mean_metric(val_metrics, 'mIoU'):.3f} "
          f"AP50={mean_metric(val_metrics, 'AP50'):.3f}")
    if args.multi_frame:
        print(f"Mean val per-bundle mIoU={mean_metric(val_metrics, 'bundle_mIoU'):.3f} "
              f"AP50={mean_metric(val_metrics, 'bundle_AP50'):.3f} "
              f"(multi-view protocol — comparable to the D4RT arms' per-bundle numbers, "
              f"NOT to the per-frame ones; docs/RESULTS.md §1)")
    if best["epoch"] > 0:
        summary = (f"Best val mIoU {best['val_mIoU']:.3f} @ epoch {best['epoch']}; "
                  f"best val AP50 {best_ap['val_AP50']:.3f} @ epoch {best_ap['epoch']}")
        if args.multi_frame and best_bundle["epoch"] > 0:
            summary += (f"; best val bundle AP50 {best_bundle['val_bundle_AP50']:.3f} "
                       f"@ epoch {best_bundle['epoch']}")
        print(summary)

    if args.save_checkpoint:
        save_checkpoint(Path(args.save_checkpoint), model, args, args.num_epochs,
                        train_metrics, val_metrics, best, optimizer, scheduler)
        if not args.no_visualize:
            ckpt = best_path if (best_path and best_path.exists()) else Path(args.save_checkpoint)
            print(f"\n=== Visualizations from {ckpt.name} ===")
            try:
                state = torch.load(ckpt, map_location="cpu", weights_only=False)
                model.head.load_state_dict(state["head_state_dict"])
                visualize(model, val_scenes + train_scenes, args, device,
                          run_dir / "visualizations")
            except Exception as e:  # training succeeded — never fail the run on rendering
                print(f"⚠ Visualization failed ({e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
