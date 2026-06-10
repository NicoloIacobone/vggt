# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
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
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from models.d4rt_decoder import D4RTInstanceSegmentationHead
from data.scannet_overfit import IDX_TO_CLASS

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Initializing and loading VGGT model...")
# model = VGGT.from_pretrained("facebook/VGGT-1B")  # another way to load the model

model = VGGT()
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))


model.eval()
model = model.to(device)


# -------------------------------------------------------------------------
# 0) Optional D4RT instance-segmentation head (for coloring the 3D cloud by
#    predicted instances). Loaded from a checkpoint produced by
#    `train_overfit.py --save_checkpoint ...`, which bundles the trained
#    decoder head together with the exact fixed overfit batch (scene frames +
#    query coordinates + view ids). See `visualize_masks.py` for the 2D version.
# -------------------------------------------------------------------------
SEG = {
    "head": None,       # D4RTInstanceSegmentationHead
    "coords": None,     # [1, N, 2] saved query coordinates (currently selected scene)
    "view_ids": None,   # [1, N] saved query view ids
    "gt_classes": None, # [Ng] GT classes (for reference)
    "images": None,     # [1, S, 3, H, W] the exact scene frames of the selected scene
    "frame_names": None,
    "scenes": None,        # list of per-scene dicts (multi-scene checkpoints)
    "scene_labels": [],    # human-readable labels for the scene dropdown
}


def _select_seg_scene(idx: int):
    """Point the active SEG fields at scene `idx` of the loaded checkpoint."""
    s = SEG["scenes"][idx]
    SEG["coords"] = s["coordinates"]
    SEG["view_ids"] = s["view_ids"]
    SEG["gt_classes"] = s["gt"]["classes"]
    SEG["images"] = s["images"]
    SEG["frame_names"] = s.get("frame_names", None)


def _find_default_seg_checkpoint():
    """Auto-discover the most recent training checkpoint, if any."""
    pattern = "/cluster/work/igp_psr/niacobone/distillation/output/*/checkpoint.pth"
    candidates = sorted(glob.glob(pattern), key=os.path.getmtime)
    return candidates[-1] if candidates else None


def load_seg_checkpoint(ckpt_path: str):
    """Load the trained decoder head + its fixed query batch into the global SEG dict."""
    print(f"Loading D4RT segmentation checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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
        print(
            f"    {s['name']} [{s.get('split', 'train')}]: {s['images'].shape[1]} frames, "
            f"{s['coordinates'].shape[1]} queries, mIoU={m.get('mIoU', float('nan')):.3f}, "
            f"class_acc={m.get('class_acc', float('nan')):.3f}"
        )


@torch.no_grad()
def compute_seg_colors(images_dev: torch.Tensor, mask_thr: float = 0.5, score_thr: float = 0.5):
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

    class_logits, _, pred_masks = SEG["head"](
        coords, view_ids, images_dev, global_features, patch_start_idx
    )
    class_logits = class_logits[0]   # [N, C]
    pred_masks = pred_masks[0]       # [N, S, h, w]
    N = class_logits.shape[0]

    probs = torch.softmax(class_logits, dim=-1)
    labels = probs.argmax(dim=-1)            # [N]
    scores = probs.max(dim=-1).values        # [N]

    # The fixed overfit queries are ordered [real instances ..., background points ...] (see
    # generate_query_points in train_overfit.py), so the first Ng queries ARE the real
    # instances, in GT order. Coloring only those reproduces the validated 11-instance result
    # and avoids the background query points — which this overfit head does not push to the
    # background class — painting spurious overlapping masks over most of the image.
    n_inst = int(SEG["gt_classes"].shape[0]) if SEG.get("gt_classes") is not None else N
    keep = list(range(min(n_inst, N)))

    # Upsample mask probabilities to full image resolution.
    mask_prob = torch.sigmoid(pred_masks)                                   # [N, S, h, w]
    mask_prob = F.interpolate(
        mask_prob.reshape(N * S, 1, *pred_masks.shape[-2:]),
        size=(H, W), mode="bilinear", align_corners=False,
    ).reshape(N, S, H, W).cpu().numpy()

    base_rgb = images_dev[0].permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy()  # [S, H, W, 3]
    seg = base_rgb.copy()

    # Per-pixel winner-takes-all over kept instances (above mask threshold).
    best_val = np.full((S, H, W), mask_thr, dtype=np.float32)
    best_k = np.full((S, H, W), -1, dtype=np.int64)
    for color_i, i in enumerate(keep):
        pv = mask_prob[i]
        better = pv > best_val
        best_val[better] = pv[better]
        best_k[better] = color_i

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
_arg_parser = argparse.ArgumentParser(description="VGGT Gradio demo (+ optional D4RT masks in 3D)")
_arg_parser.add_argument(
    "--seg_checkpoint", type=str, default=None,
    help="Path to a D4RT checkpoint.pth (from train_overfit.py) to enable 3D instance "
         "segmentation coloring. If omitted, the most recent checkpoint under the output dir "
         "is auto-discovered.",
)
_arg_parser.add_argument(
    "--no_seg", action="store_true", help="Disable segmentation coloring even if a checkpoint exists.",
)
_cli_args, _ = _arg_parser.parse_known_args()

if not _cli_args.no_seg:
    _seg_ckpt = _cli_args.seg_checkpoint or _find_default_seg_checkpoint()
    if _seg_ckpt and os.path.exists(_seg_ckpt):
        try:
            load_seg_checkpoint(_seg_ckpt)
        except Exception as e:  # pragma: no cover - demo robustness
            print(f"⚠ Could not load segmentation checkpoint ({_seg_ckpt}): {e}")
    else:
        print("No D4RT segmentation checkpoint found; 3D mask coloring disabled. "
              "Pass --seg_checkpoint /path/to/checkpoint.pth to enable it.")


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
        return None, None, None, None
    target_dir, image_paths = handle_uploads(input_video, input_images)
    return None, target_dir, image_paths, "Upload complete. Click 'Reconstruct' to begin 3D processing."


def load_checkpoint_scene(scene_label=None):
    """
    Populate the gallery with the exact scene frames stored in the loaded D4RT checkpoint
    (written as lossless PNGs so VGGT reconstructs them at the same 518x518 resolution the
    decoder head was trained on). Lets the user reconstruct that scene and then color the 3D
    point cloud by the predicted instances ("Color By: Predicted Instances").

    `scene_label` selects which checkpoint scene to load (multi-scene checkpoints store the
    training scenes AND the held-out validation scene); it also switches the query points /
    GT used by `compute_seg_colors` to that scene's.
    """
    if SEG["scenes"] is None:
        return None, "None", None, (
            "No segmentation checkpoint loaded. Start the demo with "
            "`--seg_checkpoint /path/to/checkpoint.pth`."
        )

    if scene_label and scene_label in SEG["scene_labels"]:
        _select_seg_scene(SEG["scene_labels"].index(scene_label))

    from PIL import Image

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = f"input_images_{timestamp}"
    target_dir_images = os.path.join(target_dir, "images")
    os.makedirs(target_dir_images, exist_ok=True)

    imgs = SEG["images"][0]  # [S, 3, H, W] in [0, 1]
    names = SEG["frame_names"]
    image_paths = []
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
        image_paths.append(dst)

    image_paths = sorted(image_paths)
    msg = (
        f"Loaded checkpoint scene ({len(image_paths)} frames). Click 'Reconstruct', then set "
        "'Color By' = 'Predicted Instances' to see the masks in 3D."
    )
    return None, target_dir, image_paths, msg


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
        return None, "No valid target directory found. Please upload first.", None, None

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

    # Cleanup
    del predictions
    gc.collect()
    torch.cuda.empty_cache()

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds (including IO)")
    log_msg = f"Reconstruction Success ({len(all_files)} frames). Waiting for visualization."
    if "Instance" in color_mode and seg_legend:
        log_msg += f"  |  {seg_legend}"

    return glbfile, log_msg, gr.Dropdown(choices=frame_filter_choices, value=frame_filter, interactive=True)


# -------------------------------------------------------------------------
# 5) Helper functions for UI resets + re-visualization
# -------------------------------------------------------------------------
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
    if is_example == "True":
        return None, "No reconstruction available. Please click the Reconstruct button first."

    if not target_dir or target_dir == "None" or not os.path.isdir(target_dir):
        return None, "No reconstruction available. Please click the Reconstruct button first."

    predictions_path = os.path.join(target_dir, "predictions.npz")
    if not os.path.exists(predictions_path):
        return None, f"No reconstruction available at {predictions_path}. Please run 'Reconstruct' first."

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
    # Optional predicted-instance colors (present only when a seg checkpoint was loaded).
    if "seg_colors" in loaded.files:
        predictions["seg_colors"] = np.array(loaded["seg_colors"])

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

    return glbfile, "Updating Visualization"


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
                reconstruction_output = gr.Model3D(height=520, zoom_speed=0.5, pan_speed=0.5)

            with gr.Row():
                submit_btn = gr.Button("Reconstruct", scale=1, variant="primary")
                seg_scene_dd = gr.Dropdown(
                    choices=SEG["scene_labels"],
                    value=SEG["scene_labels"][0] if SEG["scene_labels"] else None,
                    label="Checkpoint Scene (train/val)", scale=1,
                    visible=SEG["head"] is not None and len(SEG["scene_labels"]) > 1,
                )
                load_ckpt_btn = gr.Button(
                    "Load D4RT Checkpoint Scene", scale=1,
                    variant="secondary", visible=SEG["head"] is not None,
                )
                clear_btn = gr.ClearButton(
                    [input_video, input_images, reconstruction_output, log_output, target_dir_output, image_gallery],
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
        glbfile, log_msg, dropdown = gradio_demo(
            target_dir, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode
        )
        return glbfile, log_msg, target_dir, dropdown, image_paths

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
        outputs=[reconstruction_output, log_output, target_dir_output, frame_filter, image_gallery],
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
        outputs=[reconstruction_output, log_output, frame_filter],
    ).then(
        fn=lambda: "False", inputs=[], outputs=[is_example]  # set is_example to "False"
    )

    # Load the D4RT checkpoint scene into the gallery (only when a checkpoint is loaded).
    load_ckpt_btn.click(
        fn=load_checkpoint_scene,
        inputs=[seg_scene_dd],
        outputs=[reconstruction_output, target_dir_output, image_gallery, log_output],
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
        [reconstruction_output, log_output],
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
        [reconstruction_output, log_output],
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
        [reconstruction_output, log_output],
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
        [reconstruction_output, log_output],
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
        [reconstruction_output, log_output],
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
        [reconstruction_output, log_output],
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
        [reconstruction_output, log_output],
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
        [reconstruction_output, log_output],
    )

    # -------------------------------------------------------------------------
    # Auto-update gallery whenever user uploads or changes their files
    # -------------------------------------------------------------------------
    input_video.change(
        fn=update_gallery_on_upload,
        inputs=[input_video, input_images],
        outputs=[reconstruction_output, target_dir_output, image_gallery, log_output],
    )
    input_images.change(
        fn=update_gallery_on_upload,
        inputs=[input_video, input_images],
        outputs=[reconstruction_output, target_dir_output, image_gallery, log_output],
    )

    demo.queue(max_size=20).launch(show_error=True, share=False)
