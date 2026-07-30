#!/usr/bin/env python3
"""
MaskDINO instance segmentation on COCO with an interchangeable **frozen** backbone.

Why this exists (docs/MASKDINO_COCO.md): every number in `docs/MASKDINO.md` compares our ported
MaskDINO against *our own* ScanNet baselines. `scripts/coco_transplant_eval.py` closed one loop —
our modules reproduce upstream's COCO result when driven by upstream's weights — but it never
touches the training path, and it says nothing about the backbone. This script closes the other
loop: **train** the same decoder on COCO, on frozen features, and see how far the published
MaskDINO numbers survive when the ResNet-50 is replaced by VGGT.

Three arms, identical decoder / schedule / data / augmentation, only `--backbone` differs:

    --backbone resnet50   ImageNet R50, frozen. Levels res3/res4/res5, mask_features from res2
                          (stride 4) — the same pyramid upstream MaskDINO consumes. The control.
    --backbone vggt       frozen VGGT-1B aggregator. One 37x37 token map at 518px; the pyramid is
                          synthesised ViTDet-style and mask_features is upsampled by deconv.
    --backbone dinov2     frozen DINOv2 ViT-L/14. Token geometry IDENTICAL to VGGT, so
                          `vggt vs dinov2` isolates VGGT's 3D pretraining from the 37x37 grid.

Read `scripts/coco_mask_resolution_oracle.py` first. It shows a **perfect** model is capped at
44.7 mask AP on the 37x37 grid and 84.2 at 148x148, which is why `--mask_upsample 4` is the
default for the ViT arms and why the 37x37 grid the ScanNet track uses would make this experiment
unanswerable.

The backbone runs **inline**, not cached: 118 k images x 1369 tokens x 2048 ch would be 618 GB,
and horizontal flipping invalidates a per-image cache anyway.

Smoke test (CPU-sized, needs the COCO tree):
    myenv/bin/python scripts/train_maskdino_coco.py --backbone resnet50 --limit_train 64 \
        --limit_val 32 --max_steps 20 --eval_interval 10 --num_queries 30 --dec_layers 2
"""

import argparse
import contextlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.maskdino import HungarianMatcher, SetCriterion, build_weight_dict  # noqa: E402
from models.maskdino.head_coco import NUM_COCO_CLASSES  # noqa: E402
from models.maskdino.model_coco import MaskDINOCocoModel  # noqa: E402
from train.coco_data import build_loaders, targets_to_device  # noqa: E402
from train.coco_eval import evaluate_coco  # noqa: E402
from train.common import append_jsonl  # noqa: E402

DEFAULT_COCO_ROOT = "/cluster/scratch/niacobone/coco"
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def build_step_scheduler(optimizer, total_steps: int, warmup_steps: int,
                         min_lr_ratio: float = 0.01):
    """Linear warmup then cosine decay, stepped per ITERATION (COCO runs are step-budgeted)."""
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return LambdaLR(optimizer, lr_lambda)


def save_checkpoint(path: Path, model, args, step, metrics, best, optimizer=None, scheduler=None,
                    scaler=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "head_state_dict": model.head.state_dict(),
        "head_config": model.head.head_config,
        "backbone_name": model.backbone_name,
        "step": step,
        "args": vars(args),
        "metrics": metrics,
        "best": best or {},
        "trial": "maskdino_coco",
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)   # atomic: a requeue that lands mid-save must not find a truncated file
    print(f"✓ Checkpoint saved to {path} ({path.stat().st_size / 1e6:.1f} MB)", flush=True)


def build_argparser():
    p = argparse.ArgumentParser(description="MaskDINO on COCO with a frozen, swappable backbone")
    # data
    p.add_argument("--coco_root", type=str, default=DEFAULT_COCO_ROOT)
    p.add_argument("--img_size", type=int, default=518,
                   help="Square (squashed) input. 518 = VGGT's native 37x37 token grid.")
    p.add_argument("--gt_mask_size", type=int, default=296,
                   help="Resolution GT masks are rasterised at. Independent of the prediction "
                        "grid (both are compared through PointRend point sampling) and SHARED by "
                        "every arm, so the supervision signal is identical across backbones.")
    p.add_argument("--hflip", type=float, default=0.5)
    p.add_argument("--limit_train", type=int, default=0, help="first N train images (0 = all)")
    p.add_argument("--limit_val", type=int, default=0, help="first N val images (0 = all 5000)")
    p.add_argument("--num_workers", type=int, default=8)
    # backbone
    p.add_argument("--backbone", type=str, default="vggt",
                   choices=["vggt", "dinov2", "resnet50"])
    p.add_argument("--backbone_dtype", type=str, default="bfloat16",
                   choices=["float32", "bfloat16", "float16"])
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
                   choices=["no", "bitmask"])
    p.add_argument("--mask_upsample", type=int, default=4, choices=[1, 2, 4, 8],
                   help="ViT arms only (a ResNet gets its stride-4 map from res2 instead). "
                        "Default 4 -> 148x148 masks at 518px, ceiling 84.2 AP; 1 would cap the "
                        "whole experiment at 44.7 (scripts/coco_mask_resolution_oracle.py).")
    # losses — MaskDINO's COCO values
    p.add_argument("--class_weight", type=float, default=4.0)
    p.add_argument("--mask_weight", type=float, default=5.0)
    p.add_argument("--dice_weight", type=float, default=5.0)
    p.add_argument("--box_weight", type=float, default=5.0)
    p.add_argument("--giou_weight", type=float, default=2.0)
    p.add_argument("--train_num_points", type=int, default=12544,
                   help="PointRend mask-loss points (MaskDINO's COCO value). Dense supervision "
                        "is affordable on ScanNet's 37x37 grid but not on 148x148.")
    p.add_argument("--matcher_num_points", type=int, default=12544)
    # optim
    p.add_argument("--epochs", type=int, default=12,
                   help="COCO epochs; the '1x' detection schedule. Ignored if --max_steps is set.")
    p.add_argument("--max_steps", type=int, default=0, help="override the step budget (0 = auto)")
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=16,
                   help="EFFECTIVE images per optimiser step (MaskDINO's COCO IMS_PER_BATCH).")
    p.add_argument("--micro_batch", type=int, default=4,
                   help="Images per forward/backward. Upstream reaches batch 16 with 16 GPUs at "
                        "1 image each; on one 24 GB card the mask tensors "
                        "(Q x mask_grid^2 x (dec_layers+2), doubled by denoising) do not fit at "
                        "16, so the step is split and gradients accumulated. Pure memory knob — "
                        "the optimiser sees the same batch.")
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=0.1)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                   help="bf16 autocast for the head (the backbone always runs in "
                        "--backbone_dtype). bf16 needs no GradScaler.")
    # eval / bookkeeping
    p.add_argument("--eval_topk", type=int, default=100, help="COCO's test_topk_per_image")
    p.add_argument("--eval_interval", type=int, default=5000, help="steps between evals")
    p.add_argument("--eval_images", type=int, default=1000,
                   help="val images scored at each periodic eval (0 = all 5000). The FINAL eval "
                        "always uses all of them.")
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--ckpt_interval", type=int, default=2000,
                   help="steps between `checkpoint_last.pth` writes (requeue granularity)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run_dir", type=str, default=None)
    p.add_argument("--resume", type=str, default=None,
                   help="checkpoint to resume from; 'auto' picks <run_dir>/checkpoint_last.pth "
                        "when it exists, which is what the SLURM requeue uses")
    p.add_argument("--time_budget_hours", type=float, default=0.0,
                   help="Stop cleanly after this many hours, save checkpoint_last and exit "
                        "WITHOUT writing summary.json, so the SLURM job's self-resubmit fires. "
                        "0 = unlimited. Must be set BELOW the job's wall clock: at the wall clock "
                        "SLURM kills the whole batch script, including the resubmit line, so a "
                        "run that relies on being killed never continues.")
    return p


def main():
    args = build_argparser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    amp_dtype = DTYPES[args.backbone_dtype]
    print(f"Using device: {device} | backbone {args.backbone} ({args.backbone_dtype})", flush=True)

    run_dir = Path(args.run_dir) if args.run_dir else None
    if run_dir:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))

    # ---- data ----------------------------------------------------------------------------
    print("\n=== Data ===", flush=True)
    train_set, val_set, train_loader, val_loader = build_loaders(args)
    print(f"train images {len(train_set)} | val images {len(val_set)}", flush=True)

    if args.batch_size % args.micro_batch:
        raise SystemExit(f"--batch_size ({args.batch_size}) must be a multiple of --micro_batch "
                         f"({args.micro_batch})")
    accum = args.batch_size // args.micro_batch
    steps_per_epoch = max(1, len(train_loader) // accum)
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    print(f"micro-batch {args.micro_batch} x accum {accum} = effective batch {args.batch_size}",
          flush=True)
    print(f"steps/epoch {steps_per_epoch} | total steps {total_steps} "
          f"({total_steps / steps_per_epoch:.1f} epochs)", flush=True)

    # ---- model ---------------------------------------------------------------------------
    print("\n=== Model ===", flush=True)
    head_kwargs = dict(
        hidden_dim=args.hidden_dim, mask_dim=args.hidden_dim, num_classes=NUM_COCO_CLASSES,
        num_queries=args.num_queries, num_feature_levels=args.num_feature_levels,
        enc_layers=args.enc_layers, dec_layers=args.dec_layers, nheads=args.nheads,
        dropout=args.dropout, two_stage=args.two_stage, learn_tgt=not args.two_stage,
        initial_pred=True,
        initialize_box_type=args.initialize_box_type if args.two_stage else "no",
        dn=args.dn, dn_num=args.dn_num, noise_scale=args.noise_scale,
        mask_upsample=args.mask_upsample,
    )
    model = MaskDINOCocoModel(args.backbone, head_kwargs,
                              backbone_kwargs={"dtype": amp_dtype}
                              if args.backbone in ("vggt", "dinov2") else {}).to(device)
    trainable = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.backbone.parameters())
    grid = model.mask_grid(args.img_size)
    print(f"trainable head params {trainable:,} | frozen backbone params {frozen:,}", flush=True)
    print(f"backbone level strides {model.backbone.strides} | mask_features {grid}x{grid} "
          f"(GT rasterised at {args.gt_mask_size}x{args.gt_mask_size})", flush=True)

    # ---- loss ----------------------------------------------------------------------------
    matcher = HungarianMatcher(cost_class=args.class_weight, cost_mask=args.mask_weight,
                               cost_dice=args.dice_weight, cost_box=args.box_weight,
                               cost_giou=args.giou_weight, num_points=args.matcher_num_points)
    weight_dict = build_weight_dict(args.class_weight, args.mask_weight, args.dice_weight,
                                    args.box_weight, args.giou_weight, dec_layers=args.dec_layers,
                                    two_stage=args.two_stage, dn=args.dn)
    criterion = SetCriterion(NUM_COCO_CLASSES, matcher, weight_dict,
                             losses=["labels", "masks", "boxes"],
                             num_points=args.train_num_points, dn=args.dn,
                             dn_losses=["labels", "masks", "boxes"]).to(device)

    optimizer = AdamW([p for p in model.head.parameters() if p.requires_grad],
                      lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = build_step_scheduler(optimizer, total_steps, args.warmup_steps)

    # ---- resume (the SLURM requeue path) --------------------------------------------------
    start_step, best = 0, {"segm_AP": -1.0, "step": -1}
    resume = args.resume
    if resume == "auto":
        cand = run_dir / "checkpoint_last.pth" if run_dir else None
        resume = str(cand) if cand and cand.exists() else None
        print(f"[resume] auto -> {resume or 'nothing to resume, starting fresh'}", flush=True)
    if resume:
        ckpt = torch.load(resume, map_location="cpu", weights_only=False)
        model.head.load_state_dict(ckpt["head_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_step = int(ckpt.get("step", 0))
        best = ckpt.get("best", best)
        print(f"✓ Resumed from {resume} at step {start_step}", flush=True)

    metrics_path = run_dir / "metrics.jsonl" if run_dir else None
    last_path = run_dir / "checkpoint_last.pth" if run_dir else None
    best_path = run_dir / "checkpoint_best.pth" if run_dir else None

    def run_eval(step: int, max_images: int, tag: str):
        t0 = time.time()
        m = evaluate_coco(model, val_loader, val_set, device, topk=args.eval_topk,
                          amp_dtype=amp_dtype, max_images=max_images, verbose=True)
        m["step"] = step
        m["eval_minutes"] = round((time.time() - t0) / 60, 2)
        m["tag"] = tag
        print(f"[eval@{step}] {tag} segm_AP={m.get('segm_AP', float('nan')):.3f} "
              f"AP50={m.get('segm_AP50', float('nan')):.3f} "
              f"APs={m.get('segm_APs', float('nan')):.3f} | "
              f"bbox_AP={m.get('bbox_AP', float('nan')):.3f} "
              f"({m['eval_minutes']:.1f} min)", flush=True)
        if metrics_path:
            append_jsonl(metrics_path, m)
        model.train()
        return m

    # ---- train ----------------------------------------------------------------------------
    print("\n" + "=" * 70 + f"\nTRAINING  {args.backbone}\n" + "=" * 70, flush=True)
    model.train()
    step = start_step
    t_start = time.time()
    running = {"loss": 0.0, "ce": 0.0, "mask": 0.0, "box": 0.0, "n": 0}
    use_amp = device.startswith("cuda") and amp_dtype != torch.float32

    micro = 0
    oom_skips = 0
    out_of_time = False
    budget_s = args.time_budget_hours * 3600
    optimizer.zero_grad(set_to_none=True)
    while step < total_steps and not out_of_time:
        for batch in train_loader:
            if step >= total_steps or out_of_time:
                break
            images = batch["images"].to(device, non_blocking=True)
            targets = targets_to_device(batch["targets"], device)

            try:
                with (torch.autocast("cuda", dtype=amp_dtype) if (use_amp and args.amp)
                      else contextlib.nullcontext()):
                    out, mask_dict = model(images, targets)
                # Losses in fp32: focal/dice over 12544 points underflow badly in reduced
                # precision, and the Hungarian cost matrix is built on CPU anyway. `mask_dict`
                # carries the DN branch's own logits, so it needs the same treatment.
                losses = criterion(_to_float(out), targets, _to_float(mask_dict))
                total = sum(losses[k] * weight_dict[k] for k in losses if k in weight_dict)
            except torch.cuda.OutOfMemoryError:
                # Peak memory scales with the number of GT instances in the micro-batch, and COCO
                # has images with 90+. Over a 40 h run one unlucky draw must not kill the job and
                # cost every step since the last checkpoint — skip it and move on.
                n_inst = sum(int(t["labels"].numel()) for t in targets)
                print(f"⚠ OOM on a micro-batch of {images.shape[0]} images / {n_inst} instances "
                      f"at step {step}; skipping it", flush=True)
                oom_skips += 1
                optimizer.zero_grad(set_to_none=True)
                micro = 0
                torch.cuda.empty_cache()
                continue

            if not torch.isfinite(total):
                print(f"⚠ non-finite loss in micro-batch of step {step}; skipping it", flush=True)
                del out, mask_dict, losses, total
                continue
            # Mean over the accumulated micro-batches, so the gradient equals the one a single
            # forward at --batch_size would have produced.
            (total / accum).backward()

            running["loss"] += float(total)
            running["ce"] += float(losses["loss_ce"])
            running["mask"] += float(losses["loss_mask"]) + float(losses["loss_dice"])
            running["box"] += float(losses["loss_bbox"]) + float(losses["loss_giou"])
            running["n"] += 1
            del out, mask_dict, losses, total

            micro += 1
            if micro < accum:
                continue
            micro = 0
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.head.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_interval == 0:
                n = max(1, running["n"])
                el = time.time() - t_start
                done = step - start_step
                eta = (total_steps - step) * el / max(1, done) / 3600
                print(f"[step {step:6d}/{total_steps}] loss {running['loss'] / n:7.3f} "
                      f"(cls {running['ce'] / n:6.3f}, mask+dice {running['mask'] / n:6.3f}, "
                      f"box+giou {running['box'] / n:6.3f})  lr "
                      f"{scheduler.get_last_lr()[0]:.2e}  {done / max(el, 1e-9):.2f} it/s  "
                      f"ETA {eta:.1f} h", flush=True)
                running = {"loss": 0.0, "ce": 0.0, "mask": 0.0, "box": 0.0, "n": 0}

            if last_path and step % args.ckpt_interval == 0:
                save_checkpoint(last_path, model, args, step, {}, best, optimizer, scheduler)

            if budget_s and (time.time() - t_start) > budget_s:
                out_of_time = True
                print(f"\n=== time budget ({args.time_budget_hours} h) reached at step {step}"
                      f"/{total_steps} — checkpointing and exiting for the resubmit ===",
                      flush=True)
                if last_path:
                    save_checkpoint(last_path, model, args, step, {}, best, optimizer, scheduler)
                break

            if step % args.eval_interval == 0 and step < total_steps:
                m = run_eval(step, args.eval_images, f"periodic@{args.eval_images or 'all'}")
                if best_path and m.get("segm_AP", -1) > best["segm_AP"]:
                    best = {"segm_AP": m["segm_AP"], "step": step}
                    save_checkpoint(best_path, model, args, step, m, best)

    if out_of_time:
        # Deliberately no final eval and NO summary.json: its absence is the signal the SLURM
        # script tests to decide whether to resubmit (slurm/train_maskdino_coco.sh).
        print(f"\nSegment ran {(time.time() - t_start) / 3600:.2f} h and stopped at step {step}"
              f"/{total_steps} ({oom_skips} micro-batches skipped on OOM). "
              f"Resume with --resume auto.", flush=True)
        return 0

    # ---- final: the full 5000-image val2017 ------------------------------------------------
    print("\n" + "=" * 70 + "\nFINAL EVAL (full val2017)\n" + "=" * 70, flush=True)
    final = run_eval(step, 0, "final@full")
    if last_path:
        save_checkpoint(last_path, model, args, step, final, best, optimizer, scheduler)
    if best_path and final.get("segm_AP", -1) > best["segm_AP"]:
        best = {"segm_AP": final["segm_AP"], "step": step}
        save_checkpoint(best_path, model, args, step, final, best)

    print(f"\nTraining took {(time.time() - t_start) / 3600:.2f} h "
          f"({oom_skips} micro-batches skipped on OOM)", flush=True)
    print(json.dumps({"backbone": args.backbone, "steps": step, "final": final, "best": best},
                     indent=2), flush=True)
    if run_dir:
        (run_dir / "summary.json").write_text(json.dumps(
            {"backbone": args.backbone, "args": vars(args), "final": final, "best": best},
            indent=2, default=str))
    return 0


def _to_float(out):
    """
    Cast the decoder's outputs back to fp32 for the loss, recursing into the aux/interm lists
    and the denoising `mask_dict`. Integer tensors (DN index bookkeeping) are left alone.
    """
    def cast(v):
        if torch.is_tensor(v):
            return v.float() if v.is_floating_point() else v
        if isinstance(v, list):
            return [cast(x) for x in v]
        if isinstance(v, dict):
            return {k: cast(x) for k, x in v.items()}
        return v
    return None if out is None else cast(out)


if __name__ == "__main__":
    sys.exit(main())
