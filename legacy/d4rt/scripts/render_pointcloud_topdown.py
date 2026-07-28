"""
Render top-down point-cloud figures from a trained checkpoint: RGB | predicted instances.

For each scene stored in a checkpoint, this script:
  1. reloads the scene frames (uint8 / float / light checkpoints all supported),
  2. runs the frozen VGGT point head for per-pixel 3D world points + confidence,
  3. runs the trained decoder head and selects instances with the SAME honest, GT-free
     rule as the 2D overlays and the 3D viewer (train/postprocess.select_instances),
  4. renders an orthographic top-down scatter pair: left = RGB-colored points, right =
     instance-colored points over a light-gray background cloud (the "floor-plan" view).

Output: <run_dir>/pointcloud_views/<label>_topdown.png (or --output_dir).

Example:
    python legacy/d4rt/scripts/render_pointcloud_topdown.py \
        --checkpoint <run_dir>/checkpoint_best.pth --scenes scene0080_00

The projection/rendering core (`project_topdown`, `render_topdown_pair`) is pure NumPy /
matplotlib and CPU-testable without the backbone: legacy/d4rt/tests/test_render_topdown.py.
"""

import argparse
import contextlib
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")  # headless cluster nodes

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))         # scripts/

from visualize_masks import (  # noqa: E402
    DEFAULT_SCANS_ROOT, _color, scenes_from_checkpoint,
)
from train_overfit import D4RTModel, generate_grid_queries  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402
from vggt.utils.geometry import unproject_depth_map_to_point_map  # noqa: E402
from legacy.d4rt.train.postprocess import select_instances, upsample_assignment  # noqa: E402
from data.scannet_overfit import (  # noqa: E402
    IDX_TO_CLASS, decode_checkpoint_images, load_frames_by_name,
)
from legacy.d4rt.models.anchor_queries import build_anchors  # noqa: E402

BG_GRAY = np.array([0.82, 0.82, 0.82])


# -------------------------------------------------------------------------
# Pure rendering core (CPU-testable, no backbone)
# -------------------------------------------------------------------------
def estimate_up(points: np.ndarray, hint: np.ndarray = None,
                sample: int = 60_000, seed: int = 0,
                max_tilt_deg: float = 35.0, bin_frac: float = 0.02) -> np.ndarray:
    """
    Estimate the room's up direction from the point cloud (optionally gravity-hinted).

    VGGT's world frame is the (arbitrarily tilted) first-camera frame, so no fixed axis is
    reliably "up".
      - With a `hint` (the mean camera-up from the pose head — scanners are held roughly
        upright but pitch up/down): search directions within `max_tilt_deg` of the hint for
        the one that maximizes point concentration in a thin height slab. Floors, table
        tops, and ceilings are all horizontal, so they vote for the true vertical; this is
        robust to VGGT depth noise, which merely thickens the slab (unlike 3-point plane
        RANSAC, which it derails).
      - Without a hint: PCA — rooms are wide and shallow, so up = the smallest-variance
        axis, sign chosen so the densest height slab (the floor) sits at the bottom.

    Returns a unit [3] vector.
    """
    rng = np.random.default_rng(seed)
    pts = points
    if pts.shape[0] > sample:
        pts = pts[rng.choice(pts.shape[0], sample, replace=False)]

    if hint is None:
        centered = pts - pts.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)  # rows = descending variance
        up = vt[-1]  # smallest-variance direction
        h = centered @ up
        hist, edges = np.histogram(h, bins=50)
        peak = 0.5 * (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1])
        if peak > np.median(h):  # densest slab (floor) must be at low height
            up = -up
        return up

    hint = np.asarray(hint, dtype=np.float64)
    hint = hint / np.linalg.norm(hint)
    # Orthonormal basis ⊥ hint for tilting candidate directions around it.
    ref = np.array([1.0, 0.0, 0.0]) if abs(hint[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(hint, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(hint, e1)

    tilts = np.deg2rad(np.arange(0.0, max_tilt_deg + 1e-9, 2.5))
    azims = np.deg2rad(np.arange(0.0, 360.0, 15.0))
    dirs = [hint]
    for t in tilts[1:]:
        for a in azims:
            dirs.append(np.cos(t) * hint + np.sin(t) * (np.cos(a) * e1 + np.sin(a) * e2))
    dirs = np.stack(dirs)  # [D, 3], all unit norm

    bin_w = bin_frac * np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
    heights = pts @ dirs.T  # [n, D]
    best_i, best_score = 0, -1
    for i in range(dirs.shape[0]):
        h = heights[:, i]
        counts = np.bincount(((h - h.min()) / bin_w).astype(np.int64))
        score = int(counts.max())
        if score > best_score:
            best_i, best_score = i, score
    return dirs[best_i]


def _up_vector(up_axis, points: np.ndarray) -> np.ndarray:
    """Resolve up_axis (unit vector / 'auto' / ±x/±y/±z) into a unit vector."""
    if isinstance(up_axis, np.ndarray):
        return up_axis / np.linalg.norm(up_axis)
    if up_axis == "auto":
        return estimate_up(points)
    name = up_axis.lstrip("+-")
    if name not in ("x", "y", "z"):
        raise ValueError(f"up_axis must be a vector, 'auto', or ±x/±y/±z, got {up_axis!r}")
    up = np.zeros(3)
    up["xyz".index(name)] = -1.0 if up_axis.startswith("-") else 1.0
    return up


def project_topdown(points: np.ndarray, up_axis="auto"):
    """
    Orthographic top-down projection of a point cloud.

    Args:
        points:  [N, 3] world points.
        up_axis: an explicit unit [3] up vector, 'auto' (estimate from the cloud — see
                 estimate_up), or a fixed world axis with sign ("-y" = VGGT first-camera
                 frame, +y is down).

    Returns:
        (xy, order): xy [N, 2] = in-plane coordinates in an orthonormal basis ⊥ up;
        order [N] = indices sorted by height ascending, so drawing in this order paints
        high points last (on top) — what an observer above the scene sees.
    """
    up = _up_vector(up_axis, points)
    # Orthonormal in-plane basis (e1, e2) ⊥ up.
    ref = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(up, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    xy = points @ np.stack([e1, e2], axis=1)
    height = points @ up
    return xy, np.argsort(height, kind="stable")


def _crop_limits(xy: np.ndarray, crop_pct: float):
    """Percentile-based axis limits (with a small margin) so stray points don't blow up the extent."""
    lo = np.percentile(xy, crop_pct, axis=0)
    hi = np.percentile(xy, 100 - crop_pct, axis=0)
    margin = 0.02 * (hi - lo)
    return lo - margin, hi + margin


def render_topdown_pair(
    points: np.ndarray,
    rgb: np.ndarray,
    inst_colors: np.ndarray,
    out_path: Path,
    up_axis: str = "auto",
    point_size: float = 1.5,
    crop_pct: float = 1.0,
    dpi: int = 200,
) -> None:
    """
    Save the two-panel top-down figure: RGB point cloud | instance-colored point cloud.

    Args:
        points:      [N, 3] world points.
        rgb:         [N, 3] float colors in [0, 1] (panel 1).
        inst_colors: [N, 3] float colors in [0, 1] (panel 2; background points pre-grayed).
        out_path:    output PNG path.
        up_axis:     see project_topdown.
        point_size:  matplotlib scatter size.
        crop_pct:    percentile cropped from each in-plane axis extreme (outlier guard).
    """
    xy, order = project_topdown(points, up_axis=up_axis)
    lo, hi = _crop_limits(xy, crop_pct)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, colors in zip(axes, (rgb, inst_colors)):
        ax.scatter(xy[order, 0], xy[order, 1],
                   c=np.clip(colors[order], 0, 1), s=point_size,
                   marker=".", linewidths=0, alpha=1.0, rasterized=True)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal")
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def instance_color_array(assign_flat: np.ndarray, rgb: np.ndarray, gray_background: bool = True,
                         skip: set = None):
    """
    Map a flat instance assignment (-1 = background, else kept-instance index) to colors.

    Background points get flat gray (default) or their RGB; instance k gets palette color k
    — the same palette as the 2D overlay renderer, so figures match across scripts.
    Kept-instance indices in `skip` (e.g. wall/floor stuff instances) stay background-colored.
    """
    colors = np.tile(BG_GRAY, (assign_flat.shape[0], 1)) if gray_background else rgb.copy()
    for k in np.unique(assign_flat):
        if k < 0 or (skip and int(k) in skip):
            continue
        colors[assign_flat == k] = _color(int(k))
    return colors


# -------------------------------------------------------------------------
# Checkpoint-driven pipeline (needs GPU + backbone)
# -------------------------------------------------------------------------
def scene_from_dir(scene_dir: Path, max_frames: int, grid_size: int,
                   query_mode: str, num_views: int) -> dict:
    """
    Build a "virtual" scene dict from a scene's `subset/` frames (full-room coverage,
    instead of the 8-frame bundle stored in the checkpoint). Frames are sampled evenly.
    Point/hybrid checkpoints get unprompted grid queries (no GT anywhere); learned/anchor3d
    coordinates are placeholders replaced in render_scene anyway.
    """
    scene_dir = Path(scene_dir)
    subset = scene_dir / "subset" if (scene_dir / "subset").is_dir() else scene_dir
    paths = sorted(p for p in subset.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not paths:
        raise SystemExit(f"No frames found in {subset}")
    if len(paths) > max_frames:
        keep = np.round(np.linspace(0, len(paths) - 1, max_frames)).astype(int)
        paths = [paths[i] for i in keep]
    # Square 518×518 resize like training/eval — the head requires the square patch grid.
    images = load_frames_by_name(str(subset.parent), [p.stem for p in paths],
                                 image_ext=paths[0].suffix).unsqueeze(0)  # [1,S,3,H,W]
    S = images.shape[1]

    if query_mode in ("point", "hybrid"):
        coordinates, view_ids = generate_grid_queries(S, grid_size=grid_size)
        # The view-embedding table was sized for training bundles; clamp extra frame ids.
        view_ids = view_ids.clamp_max(num_views - 1)
    else:  # learned / anchor3d: coordinates are ignored placeholders
        coordinates = torch.zeros(1, 1, 2)
        view_ids = torch.zeros(1, 1, dtype=torch.long)

    name = scene_dir.parent.name if scene_dir.name == "raw_data" else scene_dir.name
    return {"name": name, "images": images, "coordinates": coordinates, "view_ids": view_ids}


@torch.no_grad()
def render_scene(model, scene: dict, out_path: Path, device: str, args) -> None:
    """Forward one scene (stored bundle or virtual subset scene) and save its figure."""
    images = decode_checkpoint_images(scene, scans_root=args.scans_root).to(device)  # [1,S,3,H,W]
    coordinates = scene["coordinates"].to(device)
    view_ids = scene["view_ids"].to(device)
    S = images.shape[1]
    H, W = images.shape[-2:]

    # bf16 autocast for the frozen backbone on big frame counts (fp32 activations for the
    # full aggregator token list would not fit common GPUs past ~16 frames).
    use_amp = device == "cuda" and not args.no_amp and (args.amp or S > 16)
    amp_ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if use_amp
               else contextlib.nullcontext())

    # Query placeholders per mode — same convention as visualize_masks.visualize_scene.
    mode = getattr(model.decoder_head, "query_mode", "point")
    M = getattr(model.decoder_head, "num_learned_queries", 0)
    if mode in ("learned", "hybrid"):
        ph_c = torch.zeros(coordinates.shape[0], M, 2, device=device)
        ph_v = torch.zeros(coordinates.shape[0], M, dtype=torch.long, device=device)
        if mode == "learned":
            coordinates, view_ids = ph_c, ph_v
        else:
            coordinates = torch.cat([ph_c, coordinates], dim=1)
            view_ids = torch.cat([ph_v, view_ids], dim=1)

    need_pointmap = args.points == "pointmap" or mode == "anchor3d"
    with amp_ctx:
        agg_list, patch_start_idx = model.backbone.aggregator(images)
        pose_enc = model.backbone.camera_head(agg_list)[-1]
        if need_pointmap:
            # Point-head pointmap (feeds arm-E anchors; noisier than depth unprojection).
            pts3d, pts3d_conf = model.backbone.point_head(
                agg_list, images=images, patch_start_idx=patch_start_idx)
        if args.points == "depth":
            depth, depth_conf = model.backbone.depth_head(
                agg_list, images=images, patch_start_idx=patch_start_idx)
    global_features = agg_list[-1].float()
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc.float(), images.shape[-2:])

    # World points: depth-map unprojection by default — the same branch the 3D viewer uses
    # ("Depthmap and Camera Branch"), noticeably cleaner than the raw pointmap head.
    if args.points == "depth":
        world = unproject_depth_map_to_point_map(
            depth[0].float().cpu().numpy(),
            extrinsic[0].float().cpu().numpy(), intrinsic[0].float().cpu().numpy())
        points = np.asarray(world).reshape(-1, 3).astype(np.float32)     # [S*H*W, 3]
        conf = depth_conf[0].float().reshape(-1).cpu().numpy()
    else:
        points = pts3d[0].float().reshape(-1, 3).cpu().numpy()
        conf = pts3d_conf[0].float().reshape(-1).cpu().numpy()

    anchors = None
    if mode == "anchor3d":
        anchors = build_anchors(global_features, patch_start_idx,
                                pts3d.float(), pts3d_conf.float(),
                                num_anchors=model.decoder_head.num_anchors,
                                knn=model.decoder_head.anchor_knn)
        K = anchors["xyz"].shape[1]
        coordinates = torch.zeros(1, K, 2, device=device)
        view_ids = torch.zeros(1, K, dtype=torch.long, device=device)
    extra = {"anchors": anchors} if anchors is not None else {}

    class_logits, _, pred_masks = model.decoder_head(
        coordinates, view_ids, images, global_features, patch_start_idx, **extra
    )

    # Honest, GT-free selection — identical rule to the 2D overlays and the 3D viewer.
    keep, labels, scores, assign = select_instances(
        class_logits[0], pred_masks[0],
        score_thr=args.score_threshold, mask_thr=args.mask_threshold,
    )
    assign_full = upsample_assignment(assign, (H, W))  # [S, H, W]

    kept_classes = [IDX_TO_CLASS.get(int(labels[q]), str(int(labels[q]))) for q in keep]
    gray_names = {c.strip() for c in args.gray_classes.split(",") if c.strip()}
    gray_k = {ki for ki, cls in enumerate(kept_classes) if cls in gray_names}

    # Flatten the remaining per-point arrays.
    rgb = images[0].permute(0, 2, 3, 1).reshape(-1, 3).clamp(0, 1).cpu().numpy()
    assign_flat = assign_full.reshape(-1).cpu().numpy()

    # Confidence filter (percentile, like the 3D viewer's conf_thres slider).
    if args.conf_thres > 0:
        mask = conf >= np.percentile(conf, args.conf_thres)
        points, rgb, assign_flat = points[mask], rgb[mask], assign_flat[mask]

    # Subsample for matplotlib (deterministic).
    if points.shape[0] > args.max_points:
        idx = np.random.default_rng(0).choice(points.shape[0], args.max_points, replace=False)
        points, rgb, assign_flat = points[idx], rgb[idx], assign_flat[idx]

    # Default up = camera-derived gravity + slab refinement. Scanners pitch up/down a lot
    # (so the mean camera-up is a biased gravity estimate) but almost never ROLL, so every
    # camera's right vector is horizontal: gravity = the common normal of the camera right
    # vectors (smallest eigenvector of their scatter). A tight slab search then snaps it to
    # the exact floor normal.
    up_axis = args.up_axis
    if up_axis == "camera":
        R = extrinsic[0, :, :3, :3].float().cpu().numpy()   # [S, 3, 3], world→camera (rows = cam axes)
        rights = R[:, 0, :]                                  # camera x-axes: horizontal if roll≈0
        cam_up = -R[:, 1, :].mean(axis=0)                    # camera +y is image-down
        w, v = np.linalg.eigh(rights.T @ rights)
        if w[1] > 0.05 * w[2]:  # view directions varied enough → null direction is gravity
            gravity = v[:, 0] if v[:, 0] @ cam_up >= 0 else -v[:, 0]
        else:                   # near-degenerate (all cameras look the same way)
            gravity = cam_up / np.linalg.norm(cam_up)
        up_axis = estimate_up(points, hint=gravity, max_tilt_deg=12.0)

    inst_colors = instance_color_array(assign_flat, rgb,
                                       gray_background=not args.rgb_background, skip=gray_k)
    render_topdown_pair(points, rgb, inst_colors, out_path,
                        up_axis=up_axis, point_size=args.point_size,
                        crop_pct=args.crop_pct, dpi=args.dpi)
    legend = ", ".join(f"{cls}({float(scores[q]):.2f})" + (" [gray]" if ki in gray_k else "")
                       for ki, (q, cls) in enumerate(zip(keep, kept_classes)))
    print(f"  {out_path.name}: {points.shape[0]} points, {S} frames, "
          f"{len(keep)} kept instances: {legend}")


def main():
    parser = argparse.ArgumentParser(
        description="Top-down point-cloud renders (RGB | predicted instances) from a checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint.pth")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output dir (default: <run_dir>/pointcloud_views)")
    parser.add_argument("--scenes", type=str, default=None,
                        help="Comma-separated scene names to render (default: all in checkpoint)")
    parser.add_argument("--scene_dir", type=str, default=None,
                        help="Comma-separated scene raw_data dirs: render from ~max_frames "
                             "evenly-spaced subset/ frames (full-room coverage) instead of "
                             "the 8-frame bundles stored in the checkpoint")
    parser.add_argument("--max_frames", type=int, default=32,
                        help="Frame budget per scene in --scene_dir mode")
    parser.add_argument("--grid_size", type=int, default=6,
                        help="Unprompted query grid per frame in --scene_dir mode "
                             "(point/hybrid checkpoints only)")
    parser.add_argument("--amp", action="store_true",
                        help="Force bf16 autocast for the backbone (auto-enabled past 16 frames)")
    parser.add_argument("--no_amp", action="store_true", help="Never use autocast")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--scans_root", type=str, default=DEFAULT_SCANS_ROOT,
                        help="Root for reloading frames from --checkpoint_light checkpoints")
    parser.add_argument("--score_threshold", type=float, default=0.5,
                        help="Min class confidence for a query to count as an instance")
    parser.add_argument("--mask_threshold", type=float, default=0.5,
                        help="Sigmoid threshold for a pixel to belong to an instance")
    parser.add_argument("--conf_thres", type=float, default=50.0,
                        help="Percentile of low-confidence 3D points to drop (0 = keep all)")
    parser.add_argument("--points", type=str, default="depth", choices=("depth", "pointmap"),
                        help="3D point source: 'depth' = depth-head unprojection (cleaner, the "
                             "3D viewer's default branch) or 'pointmap' = raw point head")
    parser.add_argument("--max_points", type=int, default=800_000,
                        help="Random subsample cap for the scatter plot")
    parser.add_argument("--up_axis", type=str, default="camera",
                        help="'camera' (default: PCA axis best aligned with the mean camera-up "
                             "from the pose head), 'auto' (pure PCA, floor-density sign), or a "
                             "fixed axis with sign, e.g. -y (VGGT frame, +y is down)")
    parser.add_argument("--point_size", type=float, default=1.5)
    parser.add_argument("--crop_pct", type=float, default=1.0,
                        help="Percentile cropped from each in-plane extreme (outlier guard)")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--rgb_background", action="store_true",
                        help="Instance panel: keep RGB on background points instead of gray")
    parser.add_argument("--gray_classes", type=str, default="",
                        help="Comma-separated class names rendered as background in the "
                             "instance panel (e.g. wall,floor for a floor-plan look)")
    args = parser.parse_args()

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent / "pointcloud_views"

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ck_args = ckpt.get("args", {})

    scenes = scenes_from_checkpoint(ckpt)
    if args.scenes:
        wanted = {s.strip() for s in args.scenes.split(",") if s.strip()}
        scenes = [(label, sc) for label, sc in scenes if sc.get("name") in wanted]
        if not scenes:
            raise SystemExit(f"None of {sorted(wanted)} found in this checkpoint")

    num_views = ck_args.get("num_views", 10)
    head_config = ckpt.get("head_config", {}) or {}
    model = D4RTModel(
        freeze_backbone=True,
        num_views=num_views if isinstance(num_views, int) else 10,
        decoder_hidden_dim=256,
        mask_embed_dim=256,
        dropout=0.0,
        query_mode=head_config.get("query_mode", ck_args.get("query_mode", "point")),
        num_learned_queries=head_config.get("num_learned_queries",
                                            ck_args.get("num_learned_queries", 0)),
        mask_upsample=head_config.get("mask_upsample", ck_args.get("mask_upsample", 1)),
        num_anchors=head_config.get("num_anchors", ck_args.get("num_anchors", 0)),
        anchor_knn=head_config.get("anchor_knn", ck_args.get("anchor_knn", 8)),
        anchor_content=head_config.get("anchor_content", ck_args.get("anchor_content", "pooled")),
        anchor_coord_scale=head_config.get("anchor_coord_scale",
                                           ck_args.get("anchor_coord_scale", 1.0)),
    ).to(device)
    model.decoder_head.load_state_dict(ckpt["decoder_head_state_dict"])
    model.eval()

    if args.scene_dir:
        mode = getattr(model.decoder_head, "query_mode", "point")
        nv = num_views if isinstance(num_views, int) else 10
        scenes = []
        for d in args.scene_dir.split(","):
            sc = scene_from_dir(d.strip(), args.max_frames, args.grid_size, mode, nv)
            scenes.append((sc["name"], sc))

    for label, scene in scenes:
        name = label if label is not None else (scene.get("name") or "scene")
        print(f"=== {name} ===")
        render_scene(model, scene, out_dir / f"{name}_topdown.png", device, args)

    print(f"\n✓ Wrote {len(scenes)} figure(s) to {out_dir}")


if __name__ == "__main__":
    main()
