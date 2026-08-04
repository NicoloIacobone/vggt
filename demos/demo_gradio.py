# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import contextlib
import cv2
import torch
import numpy as np
import gradio as gr
import sys
import shutil
import argparse
from datetime import datetime
import glob
import gc
import time

import torch.nn.functional as F

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

from visual_util import predictions_to_glb, INSTANCE_PALETTE
from dualview3d import dual_view_html, message_html
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from legacy.d4rt.models.d4rt_decoder import D4RTInstanceSegmentationHead
from legacy.d4rt.models.anchor_queries import build_anchors
from legacy.d4rt.train.postprocess import select_instances, upsample_assignment
from data.scannet_overfit import IDX_TO_CLASS, ScanNetMultiSceneDataset, decode_checkpoint_images
from train.common import DEFAULT_SCANS_ROOT
from train.maskdino_data import DTYPES
from train.maskdino_viz3d import (colorize, head_features, is_maskdino_checkpoint,
                                  load_maskdino_seg_head, maskdino_seg_colors)

# Root for reloading frames from --checkpoint_light checkpoints (no stored pixels) and, for
# MaskDINO checkpoints, for ALL frames — those checkpoints store no pixels at all.
SEG_SCANS_ROOT = DEFAULT_SCANS_ROOT

# A MaskDINO run's --train_scenes can list 1201 scenes; probing every one of them on the work
# filesystem at startup is slow and the dropdown becomes unusable. Val scenes are listed first.
MAX_SCENE_CHOICES = 200

device = "cuda" if torch.cuda.is_available() else "cpu"

# Import-only mode (tests/test_demo_gradio_maskdino.py): the module's glue — checkpoint
# dispatch, instance colouring, scene loading — is worth testing on CPU, and none of it needs
# the 1B-parameter backbone. Everything else behaves identically.
if os.environ.get("VGGT_DEMO_SKIP_BACKBONE"):
    print("VGGT_DEMO_SKIP_BACKBONE set — skipping backbone load (import-only mode).")
    model = None
else:
    print("Initializing and loading VGGT model...")
    # model = VGGT.from_pretrained("facebook/VGGT-1B")  # another way to load the model

    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))

    model.eval()
    model = model.to(device)


# -------------------------------------------------------------------------
# 0) Optional instance-segmentation head (for coloring the 3D cloud by predicted
#    instances). TWO checkpoint families are accepted, told apart by their keys:
#
#    * **MaskDINO** (`scripts/train_maskdino.py`, the active model): stores the head
#      config/weights and the run's own args, but NO pixels — frames are reloaded
#      from `--seg_scans_root`. With a `--multi_frame` checkpoint one query set is
#      shared by the whole bundle, so a query is a scene-level identity and its
#      colour is stable across views; that is the property the 3D view exists to show.
#    * **D4RT** (retired arms A-E, `legacy/d4rt/`): bundles the trained decoder head
#      together with the exact fixed overfit batch (scene frames + query coordinates
#      + view ids). See `legacy/d4rt/scripts/visualize_masks.py` for the 2D version.
#
#    The colour convention is the one used by the runs' own 2D panels
#    (`train/maskdino_eval.paint_identity_map`), so a query has the same colour in
#    the PNG figures and here — see `train/maskdino_viz3d.py`.
# -------------------------------------------------------------------------
SEG = {
    "kind": None,       # "maskdino" | "d4rt"
    "head": None,       # MaskDINOVGGTHead | D4RTInstanceSegmentationHead
    "train_args": {},   # MaskDINO only: the run's own CLI args (feature_mode, multi_frame, …)
    "coords": None,     # D4RT only: [1, N, 2] saved query coordinates (selected scene)
    "view_ids": None,   # D4RT only: [1, N] saved query view ids
    "gt_classes": None, # [Ng] GT classes (for reference)
    "images": None,     # [1, S, 3, H, W] the exact scene frames of the selected scene
    "frame_names": None,
    "scenes": None,        # list of per-scene dicts (multi-scene / MaskDINO checkpoints)
    "scene_labels": [],    # human-readable labels for the scene dropdown
    "score_threshold": 0.25,   # MaskDINO: keep queries at or above this class score
    "mask_threshold": 0.5,     # MaskDINO: a pixel joins its argmax query above this prob
    "drop_stuff": False,       # MaskDINO: drop wall/floor (what the 3D benchmark scores)
    "gt_id_maps": None,        # [S, H, W] GT instance ids of the loaded scene, in gallery order
}


def _select_seg_scene(idx: int):
    """Point the active SEG fields at scene `idx` of the loaded checkpoint."""
    s = SEG["scenes"][idx]
    if SEG["kind"] == "maskdino":
        # A MaskDINO checkpoint carries no per-scene state: the head is scene-agnostic and the
        # frames are read from disk when the scene is loaded. Selecting is just remembering.
        SEG["images"], SEG["frame_names"] = None, None
        return
    SEG["coords"] = s["coordinates"]
    SEG["view_ids"] = s["view_ids"]
    SEG["gt_classes"] = s["gt"]["classes"]
    # Handles float / uint8 / light (reloaded from disk) checkpoint image formats.
    SEG["images"] = decode_checkpoint_images(s, scans_root=SEG_SCANS_ROOT)
    SEG["frame_names"] = s.get("frame_names", None)


def _find_default_seg_checkpoint():
    """Auto-discover the most recent training checkpoint, if any."""
    pattern = "/cluster/work/igp_psr/niacobone/distillation/output/*/checkpoint.pth"
    candidates = sorted(glob.glob(pattern), key=os.path.getmtime)
    return candidates[-1] if candidates else None


def resolve_seg_checkpoint(explicit):
    """
    Which checkpoint to load: the explicit one, else the most recent, else None.

    An explicit path that does not exist is **fatal**. Starting anyway produces a viewer with no
    scene button and no instance colours, which reads as a broken UI rather than as a wrong path
    — and on a GPU node that mistake costs a backbone load to discover. (A `...` left in a
    copied command line is the usual cause.)
    """
    if explicit:
        if not os.path.exists(explicit):
            raise SystemExit(f"--seg_checkpoint does not exist: {explicit}\n"
                             "Pass the full path (a '...' copied from a command line is the "
                             "usual cause), or omit the flag to auto-discover the newest run.")
        return explicit
    found = _find_default_seg_checkpoint()
    if not found:
        print("No segmentation checkpoint found; 3D mask coloring disabled. "
              "Pass --seg_checkpoint /path/to/checkpoint.pth to enable it.")
    return found


def load_seg_checkpoint(ckpt_path: str):
    """Detect the checkpoint family (MaskDINO or legacy D4RT) and load it into SEG."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if is_maskdino_checkpoint(ckpt):
        return _load_maskdino_checkpoint(ckpt_path)
    return _load_d4rt_checkpoint(ckpt_path, ckpt)


def _load_maskdino_checkpoint(ckpt_path: str):
    """
    Load a `scripts/train_maskdino.py` checkpoint: head + the run's own args.

    The scene dropdown is built from the run's OWN scene lists (val first), keeping only the
    scenes that actually exist under `SEG_SCANS_ROOT` — the dataset tar is staged per job, so
    the path stored at training time is almost always stale.
    """
    print(f"Loading MaskDINO segmentation checkpoint: {ckpt_path}")
    head, train_args, ckpt = load_maskdino_seg_head(ckpt_path, device)
    SEG["kind"], SEG["head"], SEG["train_args"] = "maskdino", head, train_args

    scenes = []
    for split in ("val", "train"):
        names = [t.strip() for t in str(train_args.get(f"{split}_scenes") or "").split(",")
                 if t.strip()]
        for name in names:
            if len(scenes) >= MAX_SCENE_CHOICES:
                break
            scene_dir = os.path.join(SEG_SCANS_ROOT, name, "raw_data")
            if os.path.isdir(scene_dir):
                scenes.append({"name": name, "split": split, "scene_dir": scene_dir})
    SEG["scenes"] = scenes or None
    SEG["scene_labels"] = [f"{s['name']} ({s['split']})" for s in scenes]

    multi = bool(train_args.get("multi_frame", False))
    print(f"✓ MaskDINO head ready (epoch {ckpt.get('epoch', '?')}, multi_frame={multi}, "
          f"feature_mode={train_args.get('feature_mode', 'single')}, "
          f"{head.num_queries} queries)")
    if not multi:
        print("  ⚠ single-frame checkpoint: every view gets its own query set, so instance "
              "colours are NOT comparable across views in the 3D cloud. Use a --multi_frame "
              "run's checkpoint_best_bundle.pth for a multi-view-consistent picture.")
    if scenes:
        n_val = sum(s["split"] == "val" for s in scenes)
        print(f"  {len(scenes)} scene(s) available under {SEG_SCANS_ROOT} "
              f"({n_val} val); pick one in the dropdown or just upload images.")
    else:
        print(f"  No scene of this run found under {SEG_SCANS_ROOT} — pass "
              f"--seg_scans_root <staged scans dir> to enable the scene button. Uploading "
              f"your own images still works.")


def _load_d4rt_checkpoint(ckpt_path: str, ckpt):
    """Load a retired-arm checkpoint: decoder head + its fixed query batch (legacy/d4rt/)."""
    print(f"Loading D4RT segmentation checkpoint: {ckpt_path}")
    SEG["kind"] = "d4rt"
    head_config = ckpt.get("head_config") or dict(
        num_views=10, hidden_dim=256, num_classes=20, num_decoder_layers=4,
        patch_size=9, mask_embed_dim=256, memory_dim=2048, dropout=0.0,
    )
    head = D4RTInstanceSegmentationHead(**head_config)
    head.load_state_dict(ckpt["decoder_head_state_dict"])
    head.eval().to(device)
    SEG["head"] = head

    # Multi-scene checkpoints (train_multiscene.py) carry a "scenes" list; single-scene
    # checkpoints (train_overfit.py) are adapted into a one-entry list.
    scenes = ckpt.get("scenes")
    if not scenes:
        scenes = [{
            "name": "checkpoint scene",
            "split": "train",
            "images": ckpt["images"],
            "coordinates": ckpt["coordinates"],
            "view_ids": ckpt["view_ids"],
            "gt": ckpt["gt"],
            "frame_names": ckpt.get("frame_names", None),
            "metrics": ckpt.get("final_metrics", {}),
        }]
    SEG["scenes"] = scenes
    SEG["scene_labels"] = [f"{s['name']} ({s.get('split', 'train')})" for s in scenes]
    _select_seg_scene(0)

    print(f"✓ Segmentation head ready: {len(scenes)} scene(s)")
    for s in scenes:
        m = s.get("metrics", {}) or {}
        n_frames = s["images"].shape[1] if s.get("images") is not None \
            else len(s.get("frame_names") or [])
        print(
            f"    {s['name']} [{s.get('split', 'train')}]: {n_frames} frames, "
            f"{s['coordinates'].shape[1]} queries, mIoU={m.get('mIoU', float('nan')):.3f}, "
            f"class_acc={m.get('class_acc', float('nan')):.3f}"
        )


@torch.no_grad()
def compute_seg_colors(images_dev: torch.Tensor, mask_thr: float = 0.5, score_thr: float = 0.5):
    """Dispatch to the loaded head's colouring path; returns (seg_colors, legend_str)."""
    if SEG["kind"] == "maskdino":
        return _maskdino_seg_colors(images_dev)
    return _d4rt_seg_colors(images_dev, mask_thr, score_thr)


@torch.no_grad()
def _maskdino_seg_colors(images_dev: torch.Tensor):
    """
    Run the MaskDINO head on `images_dev` and colour every pixel by the query that owns it.

    The tokens are rebuilt exactly as the run built them (`--feature_mode`, `--feature_layers`,
    `--backbone_dtype`), and a `--multi_frame` checkpoint sees the whole set of frames as ONE
    bundle — that is what gives a query a single identity across views, and therefore an object
    a single colour in the 3D cloud. Selection and colouring live in `train/maskdino_viz3d.py`
    so the viewer, the 2D figures and the 3D ruler cannot drift apart.
    """
    train_args = SEG["train_args"]
    dtype = DTYPES.get(train_args.get("backbone_dtype", "float32"), torch.float32)

    def aggregator(imgs):
        ctx = (torch.autocast("cuda", dtype=dtype)
               if (device == "cuda" and dtype != torch.float32) else contextlib.nullcontext())
        with ctx:
            return model.aggregator(imgs)

    feats, patch_start_idx = head_features(aggregator, images_dev, train_args)
    S = feats.shape[0]
    multi = bool(train_args.get("multi_frame", False))
    out, _ = SEG["head"](feats, patch_start_idx, None, frames_per_sample=S if multi else 1)
    seg_colors, legend = maskdino_seg_colors(
        out, images_dev, score_threshold=SEG["score_threshold"],
        mask_threshold=SEG["mask_threshold"], drop_stuff=SEG["drop_stuff"])
    if not multi:
        legend += ("  ⚠ single-frame checkpoint: each view has its own query set, so a "
                   "colour means nothing across views.")
    return seg_colors, legend


@torch.no_grad()
def _d4rt_seg_colors(images_dev: torch.Tensor, mask_thr: float = 0.5, score_thr: float = 0.5):
    """
    Run the D4RT decoder head on `images_dev` and build a per-pixel instance-colored image.

    Args:
        images_dev (torch.Tensor): [1, S, 3, H, W] preprocessed scene frames (on device).
        mask_thr: sigmoid threshold for a pixel to belong to an instance's mask.
        score_thr: min class confidence for a query to count as a real instance.

    Returns:
        (seg_colors, legend_str):
          seg_colors: np.uint8 [S, H, W, 3] — instance-colored image (background keeps RGB).
          legend_str: human-readable "color -> class" legend.
    """
    _, S, _, H, W = images_dev.shape

    agg_list, patch_start_idx = model.aggregator(images_dev)
    global_features = agg_list[-1]

    coords = SEG["coords"].to(device)
    view_ids = SEG["view_ids"].to(device).clamp_max(S - 1)  # guard if scene has fewer frames

    # Arm E (anchor3d): rebuild the 3D anchors from the frozen point head (deterministic
    # given the same frames); coordinates become ignored placeholders.
    head = SEG["head"]
    anchors = None
    if getattr(head, "query_mode", "point") == "anchor3d":
        pts3d, pts3d_conf = model.point_head(
            agg_list, images=images_dev, patch_start_idx=patch_start_idx)
        anchors = build_anchors(global_features, patch_start_idx, pts3d, pts3d_conf,
                                num_anchors=head.num_anchors, knn=head.anchor_knn)
        K = anchors["xyz"].shape[1]
        coords = torch.zeros(1, K, 2, device=device)
        view_ids = torch.zeros(1, K, dtype=torch.long, device=device)

    class_logits, _, pred_masks = head(
        coords, view_ids, images_dev, global_features, patch_start_idx, anchors=anchors
    )
    class_logits = class_logits[0]   # [N, C]
    pred_masks = pred_masks[0]       # [N, S, h, w]
    N = class_logits.shape[0]

    # Honest, GT-free instance selection — the SAME rule as the 2D overlay renderer
    # (train/postprocess.select_instances): drop background/low-score queries and resolve
    # overlaps by per-pixel winner-takes-all. This replaces the old "first Ng queries are the
    # real instances" assumption, which only held for the overfit point-query checkpoints and
    # silently mis-selected for learned/hybrid object queries (a chair appearing in 3D but not
    # in the 2D overlays).
    keep, labels, scores, assign = select_instances(
        class_logits, pred_masks, score_thr=score_thr, mask_thr=mask_thr,
    )
    # Nearest-upsample the native-resolution assignment so the 3D coloring matches the 2D
    # "honest" panel exactly (same instances, same patch-grid sharpness).
    best_k = upsample_assignment(assign, (H, W)).cpu().numpy()  # [S, H, W], values index keep; -1 = bg

    base_rgb = images_dev[0].permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy()  # [S, H, W, 3]
    seg = base_rgb.copy()

    legend_lines = []
    for color_i, i in enumerate(keep):
        col = INSTANCE_PALETTE[color_i % len(INSTANCE_PALETTE)].astype(np.float32) / 255.0
        seg[best_k == color_i] = col
        cls_name = IDX_TO_CLASS.get(int(labels[i]), str(int(labels[i])))
        legend_lines.append(f"{cls_name} ({float(scores[i]):.2f})")

    seg_colors = (np.clip(seg, 0, 1) * 255).astype(np.uint8)
    legend_str = "Predicted instances: " + ", ".join(legend_lines) if legend_lines else "No instances detected."
    return seg_colors, legend_str


# Parse the optional segmentation checkpoint and load it before the UI is built
# (the "Load D4RT Checkpoint Scene" button is only shown when a checkpoint is available).
_arg_parser = argparse.ArgumentParser(
    description="VGGT Gradio demo (+ optional predicted instance masks in 3D)")
_arg_parser.add_argument(
    "--seg_checkpoint", type=str, default=None,
    help="Path to a MaskDINO checkpoint (scripts/train_maskdino.py — use a --multi_frame run's "
         "checkpoint_best_bundle.pth) or a legacy D4RT checkpoint.pth, to enable 3D instance "
         "segmentation coloring. If omitted, the most recent checkpoint under the output dir "
         "is auto-discovered.",
)
_arg_parser.add_argument(
    "--seg_scans_root", type=str, default=SEG_SCANS_ROOT,
    help="ScanNet scans root the 'Load Checkpoint Scene' button reads frames from (MaskDINO "
         "checkpoints store no pixels). Defaults to $SCANNET_ROOT or the copy on work.",
)
_arg_parser.add_argument(
    "--seg_score_threshold", type=float, default=SEG["score_threshold"],
    help="MaskDINO: keep queries whose best class score is at least this (0.25 = the figures').",
)
_arg_parser.add_argument(
    "--seg_mask_threshold", type=float, default=SEG["mask_threshold"],
    help="MaskDINO: a pixel joins its argmax query only above this sigmoid probability.",
)
_arg_parser.add_argument(
    "--seg_drop_stuff", action="store_true",
    help="MaskDINO: hide wall/floor instances — what the official 3D benchmark scores.",
)
_arg_parser.add_argument(
    "--no_seg", action="store_true", help="Disable segmentation coloring even if a checkpoint exists.",
)
_cli_args, _ = _arg_parser.parse_known_args()

if not _cli_args.no_seg:
    SEG_SCANS_ROOT = _cli_args.seg_scans_root
    SEG["score_threshold"] = _cli_args.seg_score_threshold
    SEG["mask_threshold"] = _cli_args.seg_mask_threshold
    SEG["drop_stuff"] = _cli_args.seg_drop_stuff
    _seg_ckpt = resolve_seg_checkpoint(_cli_args.seg_checkpoint)
    if _seg_ckpt:
        try:
            load_seg_checkpoint(_seg_ckpt)
        except Exception as e:  # pragma: no cover - demo robustness
            if _cli_args.seg_checkpoint:      # asked for by name → do not start half-working
                raise SystemExit(f"Could not load --seg_checkpoint {_seg_ckpt}: {e!r}")
            print(f"⚠ Could not load auto-discovered checkpoint ({_seg_ckpt}): {e}")


# -------------------------------------------------------------------------
# 1) Core model inference
# -------------------------------------------------------------------------
def run_model(target_dir, model) -> dict:
    """
    Run the VGGT model on images in the 'target_dir/images' folder and return predictions.
    """
    print(f"Processing images from {target_dir}")

    # Device check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Check your environment.")

    # Move model to device
    model = model.to(device)
    model.eval()

    # Load and preprocess images
    image_names = glob.glob(os.path.join(target_dir, "images", "*"))
    image_names = sorted(image_names)
    print(f"Found {len(image_names)} images")
    if len(image_names) == 0:
        raise ValueError("No images found. Check your upload.")

    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {images.shape}")

    # Run inference
    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    # Convert pose encoding to extrinsic and intrinsic matrices
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Convert tensors to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension
    predictions['pose_enc_list'] = None # remove pose_enc_list

    # Generate world points from depth map
    print("Computing world points from depth map...")
    depth_map = predictions["depth"]  # (S, H, W, 1)
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points

    # Predicted instance segmentation colors (item: 3D mask visualization). Computed here so
    # they are cached in predictions.npz and can be toggled in the viewer without recompute.
    if SEG["head"] is not None:
        images_dev = images.unsqueeze(0) if images.dim() == 4 else images
        seg_colors, seg_legend = compute_seg_colors(images_dev)
        predictions["seg_colors"] = seg_colors
        predictions["seg_legend"] = np.array(seg_legend)
        print(seg_legend)

        # GT colours for the side-by-side 3D view: the SAME painting rule and palette as the
        # prediction, keyed to the GT global instance id instead of the query index. Only
        # available for a checkpoint scene — uploaded images have no annotation.
        gt_ids = SEG.get("gt_id_maps")
        if gt_ids is not None and np.shape(gt_ids) == tuple(images_dev.shape[1:2]) + \
                tuple(images_dev.shape[3:]):
            # identity = the GT global instance id itself (0 = background → unpainted), the same
            # identity space and palette slot the run's 2D GT panels use
            ids = np.asarray(gt_ids).astype(np.int64)
            predictions["gt_colors"] = colorize(images_dev[0].cpu(), np.where(ids > 0, ids, -1))
            print(f"GT instances in these frames: {len(np.unique(ids[ids > 0]))}")
        elif gt_ids is not None:
            print(f"⚠ GT id maps {np.shape(gt_ids)} do not match the loaded frames "
                  f"{tuple(images_dev.shape)} — the GT panel stays empty.")

    # Clean up
    torch.cuda.empty_cache()
    return predictions


# -------------------------------------------------------------------------
# 2) Handle uploaded video/images --> produce target_dir + images
# -------------------------------------------------------------------------
def handle_uploads(input_video, input_images):
    """
    Create a new 'target_dir' + 'images' subfolder, and place user-uploaded
    images or extracted frames from video into it. Return (target_dir, image_paths).
    """
    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Create a unique folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = f"input_images_{timestamp}"
    target_dir_images = os.path.join(target_dir, "images")

    # Clean up if somehow that folder already exists
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    os.makedirs(target_dir_images)

    image_paths = []

    # --- Handle images ---
    if input_images is not None:
        for file_data in input_images:
            if isinstance(file_data, dict) and "name" in file_data:
                file_path = file_data["name"]
            else:
                file_path = file_data
            dst_path = os.path.join(target_dir_images, os.path.basename(file_path))
            shutil.copy(file_path, dst_path)
            image_paths.append(dst_path)

    # --- Handle video ---
    if input_video is not None:
        if isinstance(input_video, dict) and "name" in input_video:
            video_path = input_video["name"]
        else:
            video_path = input_video

        vs = cv2.VideoCapture(video_path)
        fps = vs.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * 1)  # 1 frame/sec

        count = 0
        video_frame_num = 0
        while True:
            gotit, frame = vs.read()
            if not gotit:
                break
            count += 1
            if count % frame_interval == 0:
                image_path = os.path.join(target_dir_images, f"{video_frame_num:06}.png")
                cv2.imwrite(image_path, frame)
                image_paths.append(image_path)
                video_frame_num += 1

    # Sort final images for gallery
    image_paths = sorted(image_paths)

    end_time = time.time()
    print(f"Files copied to {target_dir_images}; took {end_time - start_time:.3f} seconds")
    return target_dir, image_paths


# -------------------------------------------------------------------------
# 3) Update gallery on upload
# -------------------------------------------------------------------------
def update_gallery_on_upload(input_video, input_images):
    """
    Whenever user uploads or changes files, immediately handle them
    and show in the gallery. Return (target_dir, image_paths).
    If nothing is uploaded, returns "None" and empty list.
    """
    if not input_video and not input_images:
        return None, None, None, None, message_html("Nothing uploaded yet.")
    # Uploaded frames carry no annotation: the GT panel must not keep showing the last scene's.
    SEG["gt_id_maps"] = None
    target_dir, image_paths = handle_uploads(input_video, input_images)
    return (None, target_dir, image_paths,
            "Upload complete. Click 'Reconstruct' to begin 3D processing.",
            message_html("Click 'Reconstruct' to build the 3D views."))


def _maskdino_scene_frames(scene: dict):
    """
    Read one scene's frames off disk the way the run did: `--num_frames` evenly-spaced views at
    518x518. MaskDINO checkpoints store no pixels, so this is the only way to get the exact
    frames the reported numbers were computed on.
    """
    train_args = SEG["train_args"]
    ds = ScanNetMultiSceneDataset(
        [scene["scene_dir"]], num_frames=train_args.get("num_frames", 8), img_size=518,
        frame_sampling="even", instance_level=not train_args.get("class_level", False))
    sample = ds[0]
    # `masks` is [S, H, W] of GLOBAL instance ids (0 = background), the same identity space the
    # 2D GT panels colour by — that is what makes the GT side of the 3D comparison meaningful.
    return sample["images"], sample.get("frame_names"), sample["masks"]


def load_checkpoint_scene(scene_label=None):
    """
    Populate the gallery with one scene of the loaded checkpoint (written as lossless PNGs so
    VGGT reconstructs them at the same 518x518 resolution the head was trained on). Lets the
    user reconstruct that scene and then color the 3D point cloud by the predicted instances
    ("Color By: Predicted Instances").

    `scene_label` selects the scene: a MaskDINO run's val/train scenes read from
    `--seg_scans_root`, or the scenes a multi-scene D4RT checkpoint stored inside itself (in
    which case it also switches the query points / GT used by `compute_seg_colors`).
    """
    if SEG["scenes"] is None:
        msg = ("No checkpoint scene available. Start the demo with "
               "`--seg_checkpoint /path/to/checkpoint.pth` (and, for a MaskDINO checkpoint, a "
               "`--seg_scans_root` that holds the run's scenes). You can also upload images.")
        return None, "None", None, msg, message_html(msg)

    idx = SEG["scene_labels"].index(scene_label) if scene_label in SEG["scene_labels"] else 0
    _select_seg_scene(idx)

    from PIL import Image

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = f"input_images_{timestamp}"
    target_dir_images = os.path.join(target_dir, "images")
    os.makedirs(target_dir_images, exist_ok=True)

    gt_id_maps = None
    if SEG["kind"] == "maskdino":
        try:
            imgs, names, gt_id_maps = _maskdino_scene_frames(SEG["scenes"][idx])
        except Exception as e:  # pragma: no cover - demo robustness
            msg = f"Could not read {SEG['scenes'][idx]['name']} from {SEG_SCANS_ROOT}: {e}"
            return None, "None", None, msg, message_html(msg)
    else:
        imgs = SEG["images"][0]  # [S, 3, H, W] in [0, 1]
        names = SEG["frame_names"]
    written = []
    for s in range(imgs.shape[0]):
        arr = (imgs[s].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        name = None
        if names is not None:
            name = names[s]
            if isinstance(name, (list, tuple)):
                name = name[0]
        stem = os.path.splitext(str(name))[0] if name else f"frame_{s:05d}"
        dst = os.path.join(target_dir_images, f"{stem}.png")
        Image.fromarray(arr).save(dst)
        written.append((dst, s))

    # `run_model` re-reads the folder in sorted order, so the GT maps must be permuted the same
    # way — otherwise the GT panel would show another frame's annotation on this frame's points.
    written.sort()
    image_paths = [dst for dst, _ in written]
    SEG["gt_id_maps"] = (gt_id_maps[[s for _, s in written]].cpu().numpy()
                         if gt_id_maps is not None else None)
    msg = (
        f"Loaded {SEG['scenes'][idx]['name']} ({len(image_paths)} frames). Click 'Reconstruct', "
        "then set 'Color By' = 'Predicted Instances' to see the masks in 3D."
    )
    return None, target_dir, image_paths, msg, message_html(
        "Click 'Reconstruct' to build the side-by-side view of this scene.")


# -------------------------------------------------------------------------
# 4) Reconstruction: uses the target_dir plus any viz parameters
# -------------------------------------------------------------------------
def gradio_demo(
    target_dir,
    conf_thres=3.0,
    frame_filter="All",
    mask_black_bg=False,
    mask_white_bg=False,
    show_cam=True,
    mask_sky=False,
    prediction_mode="Pointmap Regression",
    color_mode="Image",
):
    """
    Perform reconstruction using the already-created target_dir/images.
    """
    if not os.path.isdir(target_dir) or target_dir == "None":
        return (None, "No valid target directory found. Please upload first.", None,
                message_html("No reconstruction yet."))

    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Prepare frame_filter dropdown
    target_dir_images = os.path.join(target_dir, "images")
    all_files = sorted(os.listdir(target_dir_images)) if os.path.isdir(target_dir_images) else []
    all_files = [f"{i}: {filename}" for i, filename in enumerate(all_files)]
    frame_filter_choices = ["All"] + all_files

    print("Running run_model...")
    with torch.no_grad():
        predictions = run_model(target_dir, model)

    # Save predictions
    prediction_save_path = os.path.join(target_dir, "predictions.npz")
    np.savez(prediction_save_path, **predictions)

    # Handle None frame_filter
    if frame_filter is None:
        frame_filter = "All"

    # Build a GLB file name
    glbfile = os.path.join(
        target_dir,
        f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}_sky{mask_sky}_pred{prediction_mode.replace(' ', '_')}_color{color_mode.replace(' ', '_')}.glb",
    )

    # Convert predictions to GLB
    glbscene = predictions_to_glb(
        predictions,
        conf_thres=conf_thres,
        filter_by_frames=frame_filter,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        show_cam=show_cam,
        mask_sky=mask_sky,
        target_dir=target_dir,
        prediction_mode=prediction_mode,
        color_mode=color_mode,
    )
    glbscene.export(file_obj=glbfile)

    seg_legend = predictions.get("seg_legend")
    seg_legend = str(seg_legend) if seg_legend is not None else None

    dual_html = _dual_view(predictions, conf_thres, frame_filter, mask_black_bg, mask_white_bg,
                           prediction_mode)

    # Cleanup
    del predictions
    gc.collect()
    torch.cuda.empty_cache()

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds (including IO)")
    log_msg = f"Reconstruction Success ({len(all_files)} frames). Waiting for visualization."
    if "Instance" in color_mode and seg_legend:
        log_msg += f"  |  {seg_legend}"

    return (glbfile, log_msg,
            gr.Dropdown(choices=frame_filter_choices, value=frame_filter, interactive=True),
            dual_html)


# -------------------------------------------------------------------------
# 5) Helper functions for UI resets + re-visualization
# -------------------------------------------------------------------------
def _dual_view(predictions, conf_thres, frame_filter, mask_black_bg, mask_white_bg,
               prediction_mode):
    """
    The synchronised GT|prediction panel, built from the SAME controls as the GLB view.

    Never fails the reconstruction: a viewer that cannot be built is a message, not a traceback.
    """
    try:
        return dual_view_html(
            predictions, conf_thres=conf_thres, filter_by_frames=frame_filter or "All",
            mask_black_bg=mask_black_bg, mask_white_bg=mask_white_bg,
            prediction_mode=prediction_mode)
    except Exception as e:  # pragma: no cover - demo robustness
        print(f"⚠ Side-by-side view failed: {e}")
        return message_html(f"Side-by-side view unavailable: {e}")


def clear_fields():
    """
    Clears the 3D viewer, the stored target_dir, and empties the gallery.
    """
    return None


def update_log():
    """
    Display a quick log message while waiting.
    """
    return "Loading and Reconstructing..."


def update_visualization(
    target_dir, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky,
    prediction_mode, color_mode, is_example
):
    """
    Reload saved predictions from npz, create (or reuse) the GLB for new parameters,
    and return it for the 3D viewer. If is_example == "True", skip.
    """

    # If it's an example click, skip as requested
    missing = ("No reconstruction available. Please click the Reconstruct button first.")
    if is_example == "True":
        return None, missing, message_html(missing)

    if not target_dir or target_dir == "None" or not os.path.isdir(target_dir):
        return None, missing, message_html(missing)

    predictions_path = os.path.join(target_dir, "predictions.npz")
    if not os.path.exists(predictions_path):
        msg = f"No reconstruction available at {predictions_path}. Please run 'Reconstruct' first."
        return None, msg, message_html(msg)

    key_list = [
        "pose_enc",
        "depth",
        "depth_conf",
        "world_points",
        "world_points_conf",
        "images",
        "extrinsic",
        "intrinsic",
        "world_points_from_depth",
    ]

    loaded = np.load(predictions_path, allow_pickle=True)
    predictions = {key: np.array(loaded[key]) for key in key_list if key in loaded.files}
    # Optional predicted-instance colors (present only when a seg checkpoint was loaded) and GT
    # colors (only for a checkpoint scene) — both feed the side-by-side view.
    for key in ("seg_colors", "gt_colors"):
        if key in loaded.files:
            predictions[key] = np.array(loaded[key])

    glbfile = os.path.join(
        target_dir,
        f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}_sky{mask_sky}_pred{prediction_mode.replace(' ', '_')}_color{color_mode.replace(' ', '_')}.glb",
    )

    if not os.path.exists(glbfile):
        glbscene = predictions_to_glb(
            predictions,
            conf_thres=conf_thres,
            filter_by_frames=frame_filter,
            mask_black_bg=mask_black_bg,
            mask_white_bg=mask_white_bg,
            show_cam=show_cam,
            mask_sky=mask_sky,
            target_dir=target_dir,
            prediction_mode=prediction_mode,
            color_mode=color_mode,
        )
        glbscene.export(file_obj=glbfile)

    return (glbfile, "Updating Visualization",
            _dual_view(predictions, conf_thres, frame_filter, mask_black_bg, mask_white_bg,
                       prediction_mode))


# -------------------------------------------------------------------------
# Example images
# -------------------------------------------------------------------------

great_wall_video = "examples/videos/great_wall.mp4"
colosseum_video = "examples/videos/Colosseum.mp4"
room_video = "examples/videos/room.mp4"
kitchen_video = "examples/videos/kitchen.mp4"
fern_video = "examples/videos/fern.mp4"
single_cartoon_video = "examples/videos/single_cartoon.mp4"
single_oil_painting_video = "examples/videos/single_oil_painting.mp4"
pyramid_video = "examples/videos/pyramid.mp4"


# -------------------------------------------------------------------------
# 6) Build Gradio UI
# -------------------------------------------------------------------------
theme = gr.themes.Ocean()
theme.set(
    checkbox_label_background_fill_selected="*button_primary_background_fill",
    checkbox_label_text_color_selected="*button_primary_text_color",
)

with gr.Blocks(
    theme=theme,
    css="""
    .custom-log * {
        font-style: italic;
        font-size: 22px !important;
        background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
        -webkit-background-clip: text;
        background-clip: text;
        font-weight: bold !important;
        color: transparent !important;
        text-align: center !important;
    }
    
    .example-log * {
        font-style: italic;
        font-size: 16px !important;
        background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent !important;
    }
    
    #my_radio .wrap {
        display: flex;
        flex-wrap: nowrap;
        justify-content: center;
        align-items: center;
    }

    #my_radio .wrap label {
        display: flex;
        width: 50%;
        justify-content: center;
        align-items: center;
        margin: 0;
        padding: 10px 0;
        box-sizing: border-box;
    }
    """,
) as demo:
    # Instead of gr.State, we use a hidden Textbox:
    is_example = gr.Textbox(label="is_example", visible=False, value="None")
    num_images = gr.Textbox(label="num_images", visible=False, value="None")

    gr.HTML(
        """
    <h1>🏛️ VGGT: Visual Geometry Grounded Transformer</h1>
    <p>
    <a href="https://github.com/facebookresearch/vggt">🐙 GitHub Repository</a> |
    <a href="#">Project Page</a>
    </p>

    <div style="font-size: 16px; line-height: 1.5;">
    <p>Upload a video or a set of images to create a 3D reconstruction of a scene or object. VGGT takes these images and generates a 3D point cloud, along with estimated camera poses.</p>

    <h3>Getting Started:</h3>
    <ol>
        <li><strong>Upload Your Data:</strong> Use the "Upload Video" or "Upload Images" buttons on the left to provide your input. Videos will be automatically split into individual frames (one frame per second).</li>
        <li><strong>Preview:</strong> Your uploaded images will appear in the gallery on the left.</li>
        <li><strong>Reconstruct:</strong> Click the "Reconstruct" button to start the 3D reconstruction process.</li>
        <li><strong>Visualize:</strong> The 3D reconstruction will appear in the viewer on the right. You can rotate, pan, and zoom to explore the model, and download the GLB file. Note the visualization of 3D points may be slow for a large number of input images.</li>
        <li>
        <strong>Adjust Visualization (Optional):</strong>
        After reconstruction, you can fine-tune the visualization using the options below
        <details style="display:inline;">
            <summary style="display:inline;">(<strong>click to expand</strong>):</summary>
            <ul>
            <li><em>Confidence Threshold:</em> Adjust the filtering of points based on confidence.</li>
            <li><em>Show Points from Frame:</em> Select specific frames to display in the point cloud.</li>
            <li><em>Show Camera:</em> Toggle the display of estimated camera positions.</li>
            <li><em>Filter Sky / Filter Black Background:</em> Remove sky or black-background points.</li>
            <li><em>Select a Prediction Mode:</em> Choose between "Depthmap and Camera Branch" or "Pointmap Branch."</li>
            </ul>
        </details>
        </li>
    </ol>
    <p><strong style="color: #0ea5e9;">Please note:</strong> <span style="color: #0ea5e9; font-weight: bold;">VGGT typically reconstructs a scene in less than 1 second. However, visualizing 3D points may take tens of seconds due to third-party rendering, which are independent of VGGT's processing time. </span></p>
    </div>
    """
    )

    target_dir_output = gr.Textbox(label="Target Dir", visible=False, value="None")

    with gr.Row():
        with gr.Column(scale=2):
            input_video = gr.Video(label="Upload Video", interactive=True)
            input_images = gr.File(file_count="multiple", label="Upload Images", interactive=True)

            image_gallery = gr.Gallery(
                label="Preview",
                columns=4,
                height="300px",
                show_download_button=True,
                object_fit="contain",
                preview=True,
            )

        with gr.Column(scale=4):
            with gr.Column():
                gr.Markdown("**3D Reconstruction (Point Cloud and Camera Poses)**")
                log_output = gr.Markdown(
                    "Please upload a video or images, then click Reconstruct.", elem_classes=["custom-log"]
                )
                # Two views of the same reconstruction, driven by the SAME controls below.
                # The side-by-side tab has one camera for both panels (demos/dualview3d.py), so
                # orbiting either one moves the other; the single view keeps the GLB/Babylon
                # viewer, which is the only one that also draws the camera frusta.
                with gr.Tabs():
                    with gr.Tab("Single view"):
                        reconstruction_output = gr.Model3D(height=520, zoom_speed=0.5,
                                                           pan_speed=0.5)
                    with gr.Tab("GT vs Prediction (synced)"):
                        dual_view_output = gr.HTML(
                            message_html("Reconstruct a scene to see GT and prediction "
                                         "side by side."),
                            elem_id="dual_view")

            with gr.Row():
                submit_btn = gr.Button("Reconstruct", scale=1, variant="primary")
                seg_scene_dd = gr.Dropdown(
                    choices=SEG["scene_labels"],
                    value=SEG["scene_labels"][0] if SEG["scene_labels"] else None,
                    label="Checkpoint Scene (train/val)", scale=1,
                    visible=len(SEG["scene_labels"]) > 1,
                )
                # Shown whenever a head is loaded, even with no scenes on disk: clicking it then
                # explains *why* there is nothing to load, which a missing button cannot.
                load_ckpt_btn = gr.Button(
                    "Load Checkpoint Scene", scale=1,
                    variant="secondary", visible=SEG["head"] is not None,
                )
                clear_btn = gr.ClearButton(
                    [input_video, input_images, reconstruction_output, log_output,
                     target_dir_output, image_gallery, dual_view_output],
                    scale=1,
                )

            with gr.Row():
                prediction_mode = gr.Radio(
                    ["Depthmap and Camera Branch", "Pointmap Branch"],
                    label="Select a Prediction Mode",
                    value="Depthmap and Camera Branch",
                    scale=1,
                    elem_id="my_radio",
                )

            with gr.Row():
                color_mode = gr.Radio(
                    ["Image", "Predicted Instances"],
                    label="Color By",
                    value="Image",
                    scale=1,
                    elem_id="my_radio",
                )

            with gr.Row():
                conf_thres = gr.Slider(minimum=0, maximum=100, value=50, step=0.1, label="Confidence Threshold (%)")
                frame_filter = gr.Dropdown(choices=["All"], value="All", label="Show Points from Frame")
                with gr.Column():
                    show_cam = gr.Checkbox(label="Show Camera", value=True)
                    mask_sky = gr.Checkbox(label="Filter Sky", value=False)
                    mask_black_bg = gr.Checkbox(label="Filter Black Background", value=False)
                    mask_white_bg = gr.Checkbox(label="Filter White Background", value=False)

    # ---------------------- Examples section ----------------------
    examples = [
        [colosseum_video, "22", None, 20.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
        [pyramid_video, "30", None, 35.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
        [single_cartoon_video, "1", None, 15.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
        [single_oil_painting_video, "1", None, 20.0, False, False, True, True, "Depthmap and Camera Branch", "True"],
        [room_video, "8", None, 5.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
        [kitchen_video, "25", None, 50.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
        [fern_video, "20", None, 45.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
    ]

    def example_pipeline(
        input_video,
        num_images_str,
        input_images,
        conf_thres,
        mask_black_bg,
        mask_white_bg,
        show_cam,
        mask_sky,
        prediction_mode,
        is_example_str,
    ):
        """
        1) Copy example images to new target_dir
        2) Reconstruct
        3) Return model3D + logs + new_dir + updated dropdown + gallery
        We do NOT return is_example. It's just an input.
        """
        target_dir, image_paths = handle_uploads(input_video, input_images)
        # Always use "All" for frame_filter in examples
        frame_filter = "All"
        glbfile, log_msg, dropdown, dual_html = gradio_demo(
            target_dir, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode
        )
        return glbfile, log_msg, target_dir, dropdown, image_paths, dual_html

    gr.Markdown("Click any row to load an example.", elem_classes=["example-log"])

    gr.Examples(
        examples=examples,
        inputs=[
            input_video,
            num_images,
            input_images,
            conf_thres,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        outputs=[reconstruction_output, log_output, target_dir_output, frame_filter,
                 image_gallery, dual_view_output],
        fn=example_pipeline,
        cache_examples=False,
        examples_per_page=50,
    )

    # -------------------------------------------------------------------------
    # "Reconstruct" button logic:
    #  - Clear fields
    #  - Update log
    #  - gradio_demo(...) with the existing target_dir
    #  - Then set is_example = "False"
    # -------------------------------------------------------------------------
    submit_btn.click(fn=clear_fields, inputs=[], outputs=[reconstruction_output]).then(
        fn=update_log, inputs=[], outputs=[log_output]
    ).then(
        fn=gradio_demo,
        inputs=[
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            color_mode,
        ],
        outputs=[reconstruction_output, log_output, frame_filter, dual_view_output],
    ).then(
        fn=lambda: "False", inputs=[], outputs=[is_example]  # set is_example to "False"
    )

    # Load the D4RT checkpoint scene into the gallery (only when a checkpoint is loaded).
    load_ckpt_btn.click(
        fn=load_checkpoint_scene,
        inputs=[seg_scene_dd],
        outputs=[reconstruction_output, target_dir_output, image_gallery, log_output,
                 dual_view_output],
    )

    # -------------------------------------------------------------------------
    # Real-time Visualization Updates
    # -------------------------------------------------------------------------
    conf_thres.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            color_mode,
            is_example,
        ],
        [reconstruction_output, log_output, dual_view_output],
    )
    frame_filter.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            color_mode,
            is_example,
        ],
        [reconstruction_output, log_output, dual_view_output],
    )
    mask_black_bg.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            color_mode,
            is_example,
        ],
        [reconstruction_output, log_output, dual_view_output],
    )
    mask_white_bg.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            color_mode,
            is_example,
        ],
        [reconstruction_output, log_output, dual_view_output],
    )
    show_cam.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            color_mode,
            is_example,
        ],
        [reconstruction_output, log_output, dual_view_output],
    )
    mask_sky.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            color_mode,
            is_example,
        ],
        [reconstruction_output, log_output, dual_view_output],
    )
    prediction_mode.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            color_mode,
            is_example,
        ],
        [reconstruction_output, log_output, dual_view_output],
    )

    color_mode.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            color_mode,
            is_example,
        ],
        [reconstruction_output, log_output, dual_view_output],
    )

    # -------------------------------------------------------------------------
    # Auto-update gallery whenever user uploads or changes their files
    # -------------------------------------------------------------------------
    input_video.change(
        fn=update_gallery_on_upload,
        inputs=[input_video, input_images],
        outputs=[reconstruction_output, target_dir_output, image_gallery, log_output,
                 dual_view_output],
    )
    input_images.change(
        fn=update_gallery_on_upload,
        inputs=[input_video, input_images],
        outputs=[reconstruction_output, target_dir_output, image_gallery, log_output,
                 dual_view_output],
    )


# Launch only when run as a script: `tests/test_demo_gradio_maskdino.py` imports this module to
# exercise its glue on CPU, and an import that starts serving would hang the test.
if __name__ == "__main__":
    demo.queue(max_size=20).launch(show_error=True, share=False)
