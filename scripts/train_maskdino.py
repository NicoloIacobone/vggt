#!/usr/bin/env python3
"""
Single-frame MaskDINO training on frozen VGGT features (docs/MASKDINO_TRIAL.md).

Parallel to scripts/train_multiscene.py — it deliberately shares nothing but the dataset loader,
the metric function and a few tiny helpers, so the D4RT arms stay untouched.

What is different from the D4RT training loop:
  - The sample is a FRAME, not a scene bundle. Frames are pooled across scenes and shuffled;
    one step is `--batch_frames` independent images (the supervisor's single-frame constraint).
  - GT is per frame: labels (0..18), binary masks at the mask-grid resolution, and boxes
    derived from those masks (MaskDINO is a box-aware detector).
  - Loss = MaskDINO's SetCriterion (focal + point-sampled BCE/Dice + L1/GIoU) over the final
    layer, every intermediate layer, the encoder's interm output and the denoising queries.
  - Eval is per frame with sigmoid scoring (docs/MASKDINO_TRIAL.md §6).

The frozen VGGT backbone still runs ONCE per frame up front; every epoch trains only the head.

Usage (smoke test):
    python scripts/train_maskdino.py --train_scenes scene0000_00 --val_scenes scene0001_00 \
        --num_epochs 20 --num_frames 4 --num_queries 100 --dec_layers 3 --enc_layers 3
"""

import argparse
import contextlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.scannet_overfit import IDX_TO_CLASS, ScanNetMultiSceneDataset
from eval_perframe import drop_empty_masks
from models.maskdino import (HungarianMatcher, MaskDINOVGGTHead, SetCriterion,
                             build_head_from_config, build_weight_dict, to_scannet_class_logits)
from models.maskdino.box_ops import masks_to_boxes_normalized
from train.eval_metrics import compute_instance_segmentation_metrics
from train_multiscene import (DEFAULT_SCANS_ROOT, append_jsonl, build_scheduler,
                              photometric_jitter, resolve_scene_dirs)

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


# ------------------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------------------

class MaskDINOVGGTModel(torch.nn.Module):
    """Frozen VGGT-1B aggregator + the MaskDINO head. Only the head has trainable parameters."""

    def __init__(self, head_kwargs: Dict, load_backbone: bool = True):
        super().__init__()
        self.backbone = None
        if load_backbone:
            from vggt.models.vggt import VGGT
            print("Loading VGGT backbone...")
            try:
                self.backbone = VGGT.from_pretrained("facebook/VGGT-1B")
                print("✓ Loaded pretrained VGGT-1B")
            except Exception as e:  # offline / no HF cache → random init (tests only)
                print(f"⚠ Could not load pretrained VGGT: {e}\n  Falling back to random init.")
                self.backbone = VGGT()
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
        self.head = MaskDINOVGGTHead(**head_kwargs)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone is not None:
            self.backbone.eval()  # frozen: never leave eval (dropout/norm stay deterministic)
        return self


# ------------------------------------------------------------------------------------------
# Per-frame ground truth
# ------------------------------------------------------------------------------------------

def _squeeze_batch(t):
    return t.squeeze(0) if isinstance(t, torch.Tensor) and t.dim() > 1 and t.shape[0] == 1 else t


@torch.no_grad()
def build_frame_targets(batch: Dict, out_hw: Tuple[int, int], device: str) -> List[Dict]:
    """
    Split a scene sample into per-frame MaskDINO targets.

    The dataset paints one GLOBAL instance id per object across all frames; the single-frame
    protocol simply forgets that link and treats every (frame, instance) pair as an independent
    2D target. Masks are area-downsampled to the mask grid with the same peak-preserving rule as
    the D4RT `build_gt_targets`, so a small-but-visible object never disappears.

    Returns one dict per frame with:
        labels     [n]      class index 0..18 (dataset classes are 1..19)
        masks      [n,h,w]  binary float
        boxes      [n,4]    normalized cxcywh, derived from `masks`
        global_ids [n]      the dataset's global instance id (bookkeeping / visualisation)
    """
    classes = _squeeze_batch(batch["classes"]).to(device)   # [Ng], values 1..19
    masks = _squeeze_batch(batch["masks"]).to(device)       # [S, H, W] global-id map
    S = masks.shape[0]

    per_frame = []
    for f in range(S):
        ids = torch.unique(masks[f])
        ids = ids[ids > 0]
        frame_masks, frame_labels, frame_ids = [], [], []
        for gid in ids.tolist():
            binary = (masks[f] == gid).float()
            occ = F.interpolate(binary[None, None], size=out_hw, mode="area")[0, 0]
            thr = min(0.5, float(occ.max()))
            small = ((occ >= thr) & (occ > 0)).float()
            if small.sum() == 0:
                continue
            frame_masks.append(small)
            frame_labels.append(int(classes[gid - 1]) - 1)  # 1..19 → 0..18
            frame_ids.append(gid)

        if frame_masks:
            m = torch.stack(frame_masks)
            per_frame.append({
                "labels": torch.as_tensor(frame_labels, dtype=torch.long, device=device),
                "masks": m,
                "boxes": masks_to_boxes_normalized(m),
                "global_ids": torch.as_tensor(frame_ids, dtype=torch.long, device=device),
            })
        else:
            per_frame.append({
                "labels": torch.zeros(0, dtype=torch.long, device=device),
                "masks": torch.zeros(0, *out_hw, device=device),
                "boxes": torch.zeros(0, 4, device=device),
                "global_ids": torch.zeros(0, dtype=torch.long, device=device),
            })
    return per_frame


def targets_to_device(targets: List[Dict], device: str) -> List[Dict]:
    return [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]


# ------------------------------------------------------------------------------------------
# Feature caching
# ------------------------------------------------------------------------------------------

@torch.no_grad()
def extract_features(model: MaskDINOVGGTModel, images: torch.Tensor, args,
                     device: str) -> Tuple[torch.Tensor, int]:
    """
    Frozen-backbone tokens for one scene's frames.

    `--feature_mode single` (default) runs the aggregator once per frame (S=1) — a genuinely
    single-frame model. `bundle` runs it once over all frames, so VGGT's global attention makes
    the tokens multi-view aware while the decoder still sees one frame at a time (the first
    step of the multi-frame extension, docs/MASKDINO_TRIAL.md §8).

    Returns (features [S, P, C], patch_start_idx), C = 2048 * len(--feature_layers).
    """
    autocast_dtype = DTYPES[args.backbone_dtype]
    use_autocast = device.startswith("cuda") and autocast_dtype != torch.float32

    def run(imgs):
        ctx = (torch.autocast("cuda", dtype=autocast_dtype) if use_autocast
               else contextlib.nullcontext())
        with ctx:
            agg_list, patch_start_idx = model.backbone.aggregator(imgs)
        feats = torch.cat([agg_list[i].float() for i in args.feature_layers], dim=-1)
        return feats, patch_start_idx

    S = images.shape[1]
    if args.feature_mode == "bundle":
        feats, patch_start_idx = run(images)          # [1, S, P, C]
        return feats[0], int(patch_start_idx)
    per_frame = []
    for f in range(S):
        feats, patch_start_idx = run(images[:, f:f + 1])
        per_frame.append(feats[0, 0])
    return torch.stack(per_frame), int(patch_start_idx)


@torch.no_grad()
def prepare_scenes(model: MaskDINOVGGTModel, scene_dirs: List[str], args, device: str,
                   split: str) -> List[Dict]:
    """
    One cached entry per scene: per-frame features + per-frame targets (+ uint8 images for
    visualisation). Train scenes optionally get `--bundles_per_scene` extra frame draws.
    """
    if not scene_dirs:
        return []
    num_bundles = args.bundles_per_scene if split == "train" else 1
    common = dict(num_frames=args.num_frames, img_size=518, instance_level=not args.class_level)
    even = DataLoader(ScanNetMultiSceneDataset(scene_dirs, frame_sampling="even", **common),
                      batch_size=1, shuffle=False, num_workers=0)
    rand_dataset = (ScanNetMultiSceneDataset(scene_dirs, frame_sampling="random", **common)
                    if num_bundles > 1 else None)
    cache_dtype = DTYPES[args.cache_dtype]

    def build(batch, jitter: bool) -> Dict:
        images = batch["images"].to(device)                       # [1, S, 3, H, W]
        if jitter:
            images = photometric_jitter(images, args.color_jitter)
        features, patch_start_idx = extract_features(model, images, args, device)
        num_patch = features.shape[1] - patch_start_idx
        grid = int(round(num_patch ** 0.5))
        out_hw = (grid * args.mask_upsample, grid * args.mask_upsample)
        targets = build_frame_targets(batch, out_hw, device)
        return {
            "features": features.to(args.cache_device, dtype=cache_dtype),
            "patch_start_idx": patch_start_idx,
            "targets": [{k: v.to(args.cache_device) for k, v in t.items()} for t in targets],
            "images": (images[0].clamp(0, 1) * 255).round().to(torch.uint8).cpu(),
            "frame_names": batch.get("frame_names", None),
        }

    scenes = []
    for idx, batch in enumerate(even):
        name = batch["scene_name"][0] if isinstance(batch["scene_name"], (list, tuple)) \
            else str(batch["scene_name"])
        t0 = time.time()
        b = build(batch, jitter=False)
        n_inst = sum(int(t["labels"].numel()) for t in b["targets"])
        scenes.append({"name": name, "split": split, "scene_dir": scene_dirs[idx], "bundles": [b]})
        print(f"  [{split}] {name}: frames={b['features'].shape[0]}, "
              f"tokens={tuple(b['features'].shape[1:])}, per-frame instances={n_inst} "
              f"({time.time() - t0:.1f}s backbone)")

    for k in range(1, num_bundles):
        loader = DataLoader(rand_dataset, batch_size=1, shuffle=False, num_workers=0)
        for idx, batch in enumerate(loader):
            b = build(batch, jitter=args.color_jitter > 0)
            b["images"] = None  # only bundle 0 is ever visualised
            scenes[idx]["bundles"].append(b)
        print(f"  [{split}] extra bundle {k} cached for {len(scenes)} scenes")
    return scenes


def frame_index(scenes: List[Dict], require_gt: bool = True) -> List[Tuple[int, int, int]]:
    """Flat list of (scene, bundle, frame) samples — the unit of a training step."""
    out = []
    for si, s in enumerate(scenes):
        for bi, b in enumerate(s["bundles"]):
            for fi, t in enumerate(b["targets"]):
                if (not require_gt) or int(t["labels"].numel()) > 0:
                    out.append((si, bi, fi))
    return out


def gather_batch(scenes, samples, device):
    """Stack the cached tokens of a list of (scene, bundle, frame) samples into one batch."""
    feats = torch.stack([scenes[si]["bundles"][bi]["features"][fi] for si, bi, fi in samples])
    feats = feats.to(device, dtype=torch.float32, non_blocking=True)
    targets = targets_to_device(
        [scenes[si]["bundles"][bi]["targets"][fi] for si, bi, fi in samples], device)
    patch_start_idx = scenes[samples[0][0]]["bundles"][samples[0][1]]["patch_start_idx"]
    return feats, targets, patch_start_idx


# ------------------------------------------------------------------------------------------
# Eval
# ------------------------------------------------------------------------------------------

@torch.no_grad()
def eval_scenes(model: MaskDINOVGGTModel, scenes: List[Dict], args, device: str
                ) -> Dict[str, Dict[str, float]]:
    """
    Per-frame instance-segmentation metrics, averaged over the frames of each scene.

    Frames with no GT instance are skipped (they have no defined mIoU/AP and would only
    dilute the mean). Every frame is scored TWICE (docs/MASKDINO_TRIAL.md §6):

      - thresholded (`--score_threshold`, MaskDINO's OBJECT_MASK_THRESHOLD): the headline
        numbers, the closest analogue to the D4RT arms' "argmax != background" filter;
      - threshold-free, suffix `_all`: every query kept and ranked by score — the standard
        COCO detection protocol. It is informative from epoch 1, whereas the thresholded
        numbers stay at 0 for a long while (focal-trained sigmoid scores start near zero).
    """
    was_training = model.training
    model.eval()
    keys = ["mIoU", "AP50", "AP75", "mAP", "class_acc", "num_pred", "num_gt"]
    per_scene = {}
    for si, scene in enumerate(scenes):
        samples = [(si, 0, fi) for fi, t in enumerate(scene["bundles"][0]["targets"])
                   if int(t["labels"].numel()) > 0]
        rows = []
        for start in range(0, len(samples), args.eval_batch_frames):
            chunk = samples[start:start + args.eval_batch_frames]
            feats, targets, psi = gather_batch(scenes, chunk, device)
            out, _ = model.head(feats, psi, None)
            for b in range(len(chunk)):
                # A query that claims no pixels in this frame is not a detection — the same
                # rule scripts/eval_perframe.py applies to the D4RT baselines, so the two
                # protocols stay comparable.
                pm, cl = drop_empty_masks(out["pred_masks"][b],
                                          to_scannet_class_logits(out["pred_logits"][b]))
                # COCO keeps at most `test_topk_per_image` (100) detections per image. Enforcing
                # it here is both protocol-correct and a large speedup: the AP computation loops
                # over every kept prediction at 10 IoU thresholds, so an unbounded 300-query set
                # costs ~3x more per frame and dominates eval on large scene counts.
                pm, cl = topk_predictions(pm, cl, args.eval_topk)
                common = dict(
                    pred_masks=pm,
                    class_logits=cl,
                    gt_masks=targets[b]["masks"],
                    gt_classes=targets[b]["labels"] + 1,
                    background_class=0,
                    score_mode="sigmoid",
                )
                row = compute_instance_segmentation_metrics(
                    score_threshold=args.score_threshold, **common)
                row.update({f"{k}_all": v for k, v in
                            compute_instance_segmentation_metrics(score_threshold=0.0,
                                                                  **common).items()})
                rows.append(row)
        all_keys = keys + [f"{k}_all" for k in keys]
        per_scene[scene["name"]] = ({k: float(np.mean([r[k] for r in rows])) for k in all_keys}
                                    if rows else {k: 0.0 for k in all_keys})
    if was_training:
        model.train()
    return per_scene


def topk_predictions(pred_masks: torch.Tensor, class_logits: torch.Tensor, k: int):
    """
    Keep the k highest-scoring predictions (COCO's `test_topk_per_image`, MaskDINO uses 100).

    Score = max sigmoid class probability. `k <= 0` disables the cap. Beyond matching the COCO
    protocol this bounds eval cost: AP loops over every kept prediction at 10 IoU thresholds.
    """
    if k <= 0 or class_logits.shape[0] <= k:
        return pred_masks, class_logits
    scores = torch.sigmoid(class_logits).max(dim=-1).values
    keep = torch.topk(scores, k).indices
    return pred_masks[keep], class_logits[keep]


def mean_metric(per_scene: Dict[str, Dict[str, float]], key: str) -> float:
    vals = [m[key] for m in per_scene.values()]
    return float(np.mean(vals)) if vals else 0.0


def fmt(m: Dict[str, float]) -> str:
    return (f"mIoU={m['mIoU']:.3f}  AP50={m['AP50']:.3f}  AP75={m['AP75']:.3f}  "
            f"mAP={m['mAP']:.3f}  class_acc={m['class_acc']:.3f}  "
            f"pred/gt={m['num_pred']:.1f}/{m['num_gt']:.1f}  "
            f"| all-query: mIoU={m['mIoU_all']:.3f} AP50={m['AP50_all']:.3f}")


# ------------------------------------------------------------------------------------------
# Visualisation (RGB | GT | prediction, per frame)
# ------------------------------------------------------------------------------------------

@torch.no_grad()
def visualize(model: MaskDINOVGGTModel, scenes: List[Dict], args, device: str, out_dir: Path,
              max_scenes: int = 2, max_frames: int = 4) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for si, scene in enumerate(scenes[:max_scenes]):
        bundle = scene["bundles"][0]
        if bundle.get("images") is None:
            continue
        samples = [(si, 0, fi) for fi, t in enumerate(bundle["targets"])
                   if int(t["labels"].numel()) > 0][:max_frames]
        if not samples:
            continue
        feats, targets, psi = gather_batch(scenes, samples, device)
        out, _ = model.head(feats, psi, None)
        for b, (_, _, fi) in enumerate(samples):
            rgb = bundle["images"][fi].permute(1, 2, 0).numpy() / 255.0
            gt = targets[b]["masks"]
            gt_lbl = targets[b]["labels"]
            scores = torch.sigmoid(out["pred_logits"][b])
            best, labels = scores.max(-1)
            keep = (best >= args.score_threshold).nonzero(as_tuple=True)[0]
            keep = keep[torch.argsort(best[keep], descending=True)]
            probs = torch.sigmoid(out["pred_masks"][b])

            h, w = gt.shape[-2:]
            gt_map = torch.zeros(h, w, dtype=torch.long)
            for i in range(gt.shape[0]):
                gt_map[gt[i].cpu() > 0.5] = i + 1
            pred_map = torch.zeros(h, w, dtype=torch.long)
            best_prob = torch.full((h, w), 0.5, device=probs.device)
            for c, qi in enumerate(keep.tolist()):
                better = probs[qi] > best_prob
                best_prob[better] = probs[qi][better]
                pred_map[better.cpu()] = c + 1

            fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
            axes[0].imshow(rgb); axes[0].set_title(f"{scene['name']} frame {fi}")
            axes[1].imshow(gt_map, cmap="tab20", interpolation="nearest")
            axes[1].set_title(f"GT ({gt.shape[0]} inst: "
                              + ",".join(IDX_TO_CLASS[int(l) + 1] for l in gt_lbl[:4]) + ")")
            axes[2].imshow(pred_map, cmap="tab20", interpolation="nearest")
            axes[2].set_title(f"Pred ({len(keep)} inst @ score>={args.score_threshold}: "
                              + ",".join(IDX_TO_CLASS[int(labels[q]) + 1] for q in keep[:4]) + ")")
            for ax in axes:
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(out_dir / f"{scene['name']}_frame{fi:03d}.png", dpi=110)
            plt.close(fig)
            written += 1
    print(f"✓ Wrote {written} figures to {out_dir}")
    return written


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


# ------------------------------------------------------------------------------------------

def build_argparser():
    p = argparse.ArgumentParser(description="Single-frame MaskDINO on frozen VGGT features")
    # data
    p.add_argument("--train_scenes", type=str, default="scene0000_00,scene0001_00")
    p.add_argument("--val_scenes", type=str, default="scene0080_00")
    p.add_argument("--scans_root", type=str, default=DEFAULT_SCANS_ROOT)
    p.add_argument("--num_frames", type=int, default=8, help="Cached frames per scene")
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
                   choices=["no", "bitmask", "mask2box"])
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
        mask_dim=args.hidden_dim, num_classes=19, num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels, enc_layers=args.enc_layers,
        dec_layers=args.dec_layers, nheads=args.nheads, dropout=args.dropout,
        two_stage=args.two_stage, learn_tgt=not args.two_stage, initial_pred=True,
        initialize_box_type=args.initialize_box_type if args.two_stage else "no",
        dn=args.dn, dn_num=args.dn_num, noise_scale=args.noise_scale,
        mask_upsample=args.mask_upsample,
    )
    print("\n=== Initializing model ===")
    model = MaskDINOVGGTModel(head_kwargs).to(device)
    trainable = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    print(f"Trainable head parameters: {trainable:,}")

    print("\n=== Caching frozen-backbone features ===")
    train_scenes = prepare_scenes(model, train_dirs, args, device, "train")
    val_scenes = prepare_scenes(model, val_dirs, args, device, "val")
    train_samples = frame_index(train_scenes)
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

    matcher = HungarianMatcher(cost_class=args.class_weight, cost_mask=args.mask_weight,
                               cost_dice=args.dice_weight, cost_box=args.box_weight,
                               cost_giou=args.giou_weight, num_points=args.matcher_num_points)
    weight_dict = build_weight_dict(args.class_weight, args.mask_weight, args.dice_weight,
                                    args.box_weight, args.giou_weight, dec_layers=args.dec_layers,
                                    two_stage=args.two_stage, dn=args.dn)
    criterion = SetCriterion(19, matcher, weight_dict, losses=["labels", "masks", "boxes"],
                             num_points=args.train_num_points, dn=args.dn,
                             dn_losses=["labels", "masks", "boxes"]).to(device)

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

    run_dir = Path(args.save_checkpoint).parent if args.save_checkpoint else None
    metrics_path = run_dir / "metrics.jsonl" if run_dir else None
    best_path = run_dir / "checkpoint_best.pth" if run_dir else None
    best_ap_path = run_dir / "checkpoint_best_ap50.pth" if run_dir else None
    if run_dir:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))

    print("\n=== Initial metrics (untrained head) ===")
    for name, m in eval_scenes(model, val_scenes, args, device).items():
        print(f"  [val] {name}: {fmt(m)}")

    print("\n" + "=" * 70 + "\nTRAINING\n" + "=" * 70)
    best = {"val_mIoU": -1.0, "epoch": -1}
    best_ap = {"val_AP50": -1.0, "epoch": -1}
    t_start = time.time()
    steps_per_epoch = max(1, (len(train_samples) + args.batch_frames - 1) // args.batch_frames)

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        random.shuffle(train_samples)
        epoch_loss = epoch_ce = epoch_mask = epoch_box = 0.0
        for step in range(steps_per_epoch):
            chunk = train_samples[step * args.batch_frames:(step + 1) * args.batch_frames]
            if not chunk:
                continue
            feats, targets, psi = gather_batch(train_scenes, chunk, device)

            optimizer.zero_grad(set_to_none=True)
            out, mask_dict = model.head(feats, psi, targets)
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
            record = {"epoch": epoch + 1, "lr": float(scheduler.get_last_lr()[0]),
                      "loss": mean_loss, "class_loss": epoch_ce / n,
                      "mask_loss": epoch_mask / n, "box_loss": epoch_box / n}
            for split, d in (("train", tr), ("val", va)):
                for key in ("mIoU", "AP50", "AP75", "mAP", "class_acc", "num_pred",
                            "mIoU_all", "AP50_all", "AP75_all", "mAP_all"):
                    record[f"{split}_{key}"] = mean_metric(d, key)
            if metrics_path:
                append_jsonl(metrics_path, record)

            select = mean_metric(va, "mIoU") if val_scenes else mean_metric(tr, "mIoU")
            if select > best["val_mIoU"]:
                best = {"val_mIoU": select, "epoch": epoch + 1}
                if best_path:
                    save_checkpoint(best_path, model, args, epoch + 1, tr, va, best)
            select_ap = mean_metric(va, "AP50") if val_scenes else mean_metric(tr, "AP50")
            if select_ap > best_ap["val_AP50"]:
                best_ap = {"val_AP50": select_ap, "epoch": epoch + 1}
                if best_ap_path:
                    save_checkpoint(best_ap_path, model, args, epoch + 1, tr, va, best_ap)

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
    if best["epoch"] > 0:
        print(f"Best val mIoU {best['val_mIoU']:.3f} @ epoch {best['epoch']}; "
              f"best val AP50 {best_ap['val_AP50']:.3f} @ epoch {best_ap['epoch']}")

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
