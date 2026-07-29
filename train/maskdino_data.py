"""
Data pipeline for the single-frame MaskDINO track (docs/MASKDINO.md).

Two jobs, both run ONCE per training run before the first epoch:
  - turn each scene sample into per-frame MaskDINO targets (labels / masks / boxes), and
  - cache the frozen backbone's tokens for every frame, so each epoch trains only the head.

Everything downstream (`train/maskdino_eval.py`, `scripts/train_maskdino.py`) consumes the
"scene" dicts built here; `frame_index` + `gather_batch` are what turn them into batches.
"""

import contextlib
import time
from collections import Counter
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.scannet_overfit import IDX_TO_CLASS, ScanNetMultiSceneDataset
from models.maskdino import NUM_SCANNET_CLASSES, build_bundle_target
from models.maskdino.box_ops import masks_to_boxes_normalized
from train.common import photometric_jitter

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


# ------------------------------------------------------------------------------------------
# Per-frame ground truth
# ------------------------------------------------------------------------------------------

def _squeeze_batch(t):
    return t.squeeze(0) if isinstance(t, torch.Tensor) and t.dim() > 1 and t.shape[0] == 1 else t


@torch.no_grad()
def build_frame_targets(batch: Dict, out_hw: Tuple[int, int], device: str,
                        num_classes: int = NUM_SCANNET_CLASSES) -> List[Dict]:
    """
    Split a scene sample into per-frame MaskDINO targets.

    The dataset paints one GLOBAL instance id per object across all frames; the single-frame
    protocol simply forgets that link and treats every (frame, instance) pair as an independent
    2D target. Masks are area-downsampled to the mask grid with the same peak-preserving rule as
    the D4RT `build_gt_targets`, so a small-but-visible object never disappears.

    `num_classes` is the class head's width (default 19). Instances whose dataset class index
    falls outside 1..num_classes are DROPPED — treated as background, exactly as the official-GT
    builder already does upstream (`legacy/dataset_build/scripts/build_official_masks.py` maps
    every NYU40 class outside the 19 trainable ones, `otherfurniture` included, to background).
    Without this a GT tree built against the full 20-name
    `data/scannet_overfit.py::SCANNET_CLASSES` list produces label 19 against a 19-logit head,
    which is an IndexError in the matcher and in the denoising label embedding — a crash, not a
    degradation. The drop is reported once per call, never silent. Neither official-GT tar nor
    the SAM3 tar contains such a class today, so this changes nothing for any completed run.

    Returns one dict per frame with:
        labels     [n]      class index 0..num_classes-1 (dataset classes are 1..num_classes)
        masks      [n,h,w]  binary float
        boxes      [n,4]    normalized cxcywh, derived from `masks`
        global_ids [n]      the dataset's global instance id (bookkeeping / visualisation)
    """
    classes = _squeeze_batch(batch["classes"]).to(device)   # [Ng], values 1..num_classes
    masks = _squeeze_batch(batch["masks"]).to(device)       # [S, H, W] global-id map
    S = masks.shape[0]

    # Decide per global instance id (1-based) whether the head can represent its class.
    class_list = [int(c) for c in classes.tolist()]
    droppable = {gid for gid, c in enumerate(class_list, start=1)
                 if not 1 <= c <= num_classes}
    if droppable:
        counts = Counter(IDX_TO_CLASS.get(class_list[gid - 1], f"class_{class_list[gid - 1]}")
                         for gid in sorted(droppable))
        scene = batch.get("scene_name", "?")
        scene = scene[0] if isinstance(scene, (list, tuple)) else scene
        detail = ", ".join(f"{name} x{n}" for name, n in sorted(counts.items()))
        print(f"⚠ build_frame_targets [{scene}]: dropped {len(droppable)} instance(s) whose class "
              f"is outside the {num_classes}-class head and is therefore treated as background "
              f"({detail}).")

    per_frame = []
    for f in range(S):
        ids = torch.unique(masks[f])
        ids = ids[ids > 0]
        frame_masks, frame_labels, frame_ids = [], [], []
        for gid in ids.tolist():
            if gid in droppable:
                continue
            binary = (masks[f] == gid).float()
            occ = F.interpolate(binary[None, None], size=out_hw, mode="area")[0, 0]
            thr = min(0.5, float(occ.max()))
            small = ((occ >= thr) & (occ > 0)).float()
            if small.sum() == 0:
                continue
            frame_masks.append(small)
            frame_labels.append(class_list[gid - 1] - 1)  # 1..num_classes → 0..num_classes-1
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
def extract_features(model, images: torch.Tensor, args, device: str) -> Tuple[torch.Tensor, int]:
    """
    Frozen-backbone tokens for one scene's frames.

    `--feature_mode single` (default) runs the aggregator once per frame (S=1) — a genuinely
    single-frame model. `bundle` runs it once over all frames, so VGGT's global attention makes
    the tokens multi-view aware while the decoder still sees one frame at a time (the first
    step of the multi-frame extension, docs/MASKDINO.md §8).

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
def prepare_scenes(model, scene_dirs: List[str], args, device: str, split: str) -> List[Dict]:
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
        targets = build_frame_targets(batch, out_hw, device, num_classes=model.head.num_classes)
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


# ------------------------------------------------------------------------------------------
# Batching
# ------------------------------------------------------------------------------------------

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
# Multi-frame batching: the sample is a BUNDLE of S frames (docs/MASKDINO.md §8)
# ------------------------------------------------------------------------------------------

def bundle_index(scenes: List[Dict], require_gt: bool = True) -> List[Tuple[int, int]]:
    """Flat list of (scene, bundle) samples — the unit of a multi-frame training step."""
    return [(si, bi) for si, s in enumerate(scenes) for bi, b in enumerate(s["bundles"])
            if (not require_gt) or any(int(t["labels"].numel()) > 0 for t in b["targets"])]


def gather_bundle_batch(scenes, samples, device):
    """
    Stack whole bundles: the S frames of a bundle stay CONTIGUOUS in the batch dimension, which
    is the layout `frames_per_sample=S` assumes everywhere downstream.

    Unlike `gather_batch` this keeps frames with no GT instance: a shared query has to learn to
    predict "not here" in the views where its object is invisible, and dropping those frames
    would remove exactly that supervision.

    Returns (features [B*S, P, C], per-frame targets [B*S], bundle targets [B], patch_start_idx,
             frames_per_sample).
    """
    bundles = [scenes[si]["bundles"][bi] for si, bi in samples]
    frames_per_sample = bundles[0]["features"].shape[0]
    assert all(b["features"].shape[0] == frames_per_sample for b in bundles), \
        "multi-frame batching needs the same frame count in every bundle"

    feats = torch.cat([b["features"] for b in bundles]).to(device, dtype=torch.float32,
                                                           non_blocking=True)
    frame_targets, bundle_targets = [], []
    for b in bundles:
        per_frame = targets_to_device(b["targets"], device)
        frame_targets.extend(per_frame)
        bundle_targets.append(build_bundle_target(per_frame))
    return feats, frame_targets, bundle_targets, bundles[0]["patch_start_idx"], frames_per_sample
