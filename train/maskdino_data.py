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
from models.maskdino import NUM_SCANNET_CLASSES, build_bundle_target, normalize_token_xyz
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
def patch_token_positions(world_points: torch.Tensor, conf: torch.Tensor,
                          patch_size: int = 14) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    3D position per VGGT patch token: the confidence-weighted mean of the point head's
    predictions over that token's `patch_size` x `patch_size` pixel cell (docs/MASKDINO.md §8.3).

    This is what makes `--anchor_3d` affordable: the full pointmap is ~26 MB per bundle and is
    never stored, while its pooling to the token grid is ~66 kB in fp16. Deliberately
    re-implemented here rather than imported from the frozen `legacy/d4rt/models/anchor_queries.py`
    (arm E's version), so that arm E's published numbers can never move.

    Args:
        world_points: [1, S, H, W, 3] point-head output (H, W divisible by patch_size).
        conf:         [1, S, H, W] point-head confidence (>= 0).
    Returns:
        positions: [S, hp*wp, 3] — row-major per frame, the aggregator's patch-token order.
        weights:   [S, hp*wp]    — mean confidence per token.
    """
    B, S, H, W, _ = world_points.shape
    assert B == 1, f"expected batch size 1, got {B}"
    hp, wp = H // patch_size, W // patch_size
    assert hp * patch_size == H and wp * patch_size == W, (
        f"H, W ({H}, {W}) must be divisible by patch_size ({patch_size})")

    pts = world_points[0].reshape(S, hp, patch_size, wp, patch_size, 3)
    pts = pts.permute(0, 1, 3, 2, 4, 5).reshape(S, hp, wp, patch_size * patch_size, 3)
    w = conf[0].reshape(S, hp, patch_size, wp, patch_size)
    w = w.permute(0, 1, 3, 2, 4).reshape(S, hp, wp, patch_size * patch_size).clamp_min(0)

    denom = w.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    positions = (pts * w.unsqueeze(-1)).sum(dim=-2) / denom          # [S, hp, wp, 3]
    return positions.reshape(S, hp * wp, 3), w.mean(dim=-1).reshape(S, hp * wp)


@torch.no_grad()
def extract_features(model, images: torch.Tensor, args, device: str, need_xyz: bool = False):
    """
    Frozen-backbone tokens for one scene's frames.

    `--feature_mode single` (default) runs the aggregator once per frame (S=1) — a genuinely
    single-frame model. `bundle` runs it once over all frames, so VGGT's global attention makes
    the tokens multi-view aware while the decoder still sees one frame at a time (the first
    step of the multi-frame extension, docs/MASKDINO.md §8).

    `need_xyz` (`--anchor_3d`) additionally runs the frozen POINT head on the aggregator output
    we already have and pools it to one 3D position per patch token, normalised over the whole
    bundle. It is only defined for `--feature_mode bundle`: in `single` mode the aggregator sees
    one frame at a time, so each frame's pointmap lives in its own coordinate frame and a 3D
    anchor shared across views would be meaningless.

    Returns (features [S, P, C], patch_start_idx, token_xyz or None), C = 2048 * len(layers).
    """
    autocast_dtype = DTYPES[args.backbone_dtype]
    use_autocast = device.startswith("cuda") and autocast_dtype != torch.float32

    def ctx():
        return (torch.autocast("cuda", dtype=autocast_dtype) if use_autocast
                else contextlib.nullcontext())

    def run(imgs, with_agg=False):
        with ctx():
            agg_list, patch_start_idx = model.backbone.aggregator(imgs)
        feats = torch.cat([agg_list[i].float() for i in args.feature_layers], dim=-1)
        return (feats, patch_start_idx, agg_list) if with_agg else (feats, patch_start_idx)

    S = images.shape[1]
    if args.feature_mode == "bundle":
        feats, patch_start_idx, agg_list = run(images, with_agg=True)   # [1, S, P, C]
        token_xyz = None
        if need_xyz:
            with torch.autocast("cuda", enabled=False) if device.startswith("cuda") \
                    else contextlib.nullcontext():
                # the fork's aggregator returns None for layers no head indexes
                agg32 = [a.float() if a is not None else None for a in agg_list]
                pts, pconf = model.backbone.point_head(agg32, images=images,
                                                       patch_start_idx=patch_start_idx)
            xyz, w = patch_token_positions(pts, pconf)                  # [S, hw, 3], [S, hw]
            token_xyz = normalize_token_xyz(xyz, w)
            del pts, pconf, agg32
        del agg_list
        return feats[0], int(patch_start_idx), token_xyz
    if need_xyz:
        raise ValueError("--anchor_3d needs --feature_mode bundle (docs/MASKDINO.md §8.3): with "
                         "per-frame features every frame's pointmap is in its own coordinate "
                         "frame, so a 3D anchor shared across views has no meaning.")
    per_frame = []
    for f in range(S):
        feats, patch_start_idx = run(images[:, f:f + 1])
        per_frame.append(feats[0, 0])
    return torch.stack(per_frame), int(patch_start_idx), None


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
        features, patch_start_idx, token_xyz = extract_features(
            model, images, args, device, need_xyz=getattr(args, "anchor_3d", False))
        num_patch = features.shape[1] - patch_start_idx
        grid = int(round(num_patch ** 0.5))
        out_hw = (grid * args.mask_upsample, grid * args.mask_upsample)
        targets = build_frame_targets(batch, out_hw, device, num_classes=model.head.num_classes)
        return {
            "features": features.to(args.cache_device, dtype=cache_dtype),
            # --anchor_3d only: [S, h*w, 3], ~66 kB per bundle in fp16 against ~45 MB of tokens
            # (+0.2%). The full pointmap it comes from is never stored (docs/MASKDINO.md §8.3).
            "token_xyz": (None if token_xyz is None
                          else token_xyz.to(args.cache_device, dtype=cache_dtype)),
            "patch_start_idx": patch_start_idx,
            "targets": [{k: v.to(args.cache_device) for k, v in t.items()} for t in targets],
            "images": (images[0].clamp(0, 1) * 255).round().to(torch.uint8).cpu(),
            "frame_names": batch.get("frame_names", None),
            # Full-resolution GT id map [S, H, W] for --eval_full_res (docs/MASKDINO.md §6.5).
            # int16 keeps 500 scenes ≈ 2 GB; only bundle 0 is ever evaluated, so extra bundles
            # null it below, like `images`.
            "gt_id_maps": (_squeeze_batch(batch["masks"]).to(torch.int16).cpu()
                           if getattr(args, "eval_full_res", False) else None),
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
            b["images"] = None      # only bundle 0 is ever visualised
            b["gt_id_maps"] = None  # ... and only bundle 0 is ever evaluated
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


def gather_token_xyz(scenes, samples, device):
    """
    The `--anchor_3d` companion of `gather_batch` / `gather_bundle_batch`: stack the cached
    per-patch 3D positions of the same samples, in the same batch order.

    Kept a separate call rather than a fourth return value so that every existing call site of
    the two gather functions is untouched — a run without `--anchor_3d` never sees this.

    `samples` are (scene, bundle, frame) triples or (scene, bundle) pairs. Returns None when the
    cache holds no positions (i.e. every run before this flag existed).
    """
    if not samples:
        return None
    if len(samples[0]) == 3:
        rows = [scenes[si]["bundles"][bi].get("token_xyz") for si, bi, _ in samples]
        if any(r is None for r in rows):
            return None
        xyz = torch.stack([r[fi] for r, (_, _, fi) in zip(rows, samples)])
    else:
        rows = [scenes[si]["bundles"][bi].get("token_xyz") for si, bi in samples]
        if any(r is None for r in rows):
            return None
        xyz = torch.cat(rows)
    return xyz.to(device, dtype=torch.float32, non_blocking=True)


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
