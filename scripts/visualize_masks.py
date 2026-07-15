#!/usr/bin/env python3
"""
Visualize the dense instance masks predicted by a trained D4RT head.

Loads a checkpoint produced by `train_overfit.py --save_checkpoint ...` or
`train_multiscene.py --save_checkpoint ...` (both store the trainable decoder head plus the
exact training batch(es)), runs one forward pass through the frozen VGGT backbone + decoder,
Hungarian-matches each prediction to a GT instance (the same matcher used in training),
upsamples the patch-resolution mask logits to full image resolution, and writes per-frame RGB
overlays comparing the predicted masks against the ground truth.

Single-scene (overfit) checkpoints write PNGs directly into the output dir; multi-scene
checkpoints get one subfolder per scene (train and val), e.g. `visualizations/val_scene0004_00/`.

Usage:
    python visualize_masks.py --checkpoint /path/to/run/checkpoint.pth
    # outputs go to <run dir>/visualizations/ by default
    python visualize_masks.py --checkpoint <run>/checkpoint_best.pth --scenes scene0004_00
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")  # headless cluster node
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/ for sibling train_overfit
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

from train_overfit import D4RTModel
from train.loss import PointBipartiteMatcher
from train.postprocess import select_instances, upsample_assignment
from data.scannet_overfit import IDX_TO_CLASS, decode_checkpoint_images
from models.anchor_queries import build_anchors

DEFAULT_SCANS_ROOT = "/cluster/work/igp_psr/niacobone/distillation/dataset/scannet/scans"


# A fixed, perceptually-distinct color palette (RGB in [0,1]) for instances.
_PALETTE = np.array(
    [
        [0.90, 0.10, 0.10], [0.10, 0.60, 0.90], [0.20, 0.80, 0.20],
        [0.95, 0.70, 0.10], [0.70, 0.20, 0.90], [0.10, 0.85, 0.85],
        [0.95, 0.45, 0.75], [0.55, 0.35, 0.15], [0.50, 0.90, 0.30],
        [0.30, 0.30, 0.95], [0.95, 0.55, 0.20], [0.60, 0.60, 0.60],
        [0.80, 0.85, 0.20], [0.40, 0.75, 0.95], [0.85, 0.20, 0.50],
        [0.20, 0.50, 0.40],
    ]
)


def _color(i: int) -> np.ndarray:
    return _PALETTE[i % len(_PALETTE)]


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Alpha-blend a colored mask onto an RGB image and draw a thin contour."""
    out = rgb.copy()
    if mask.any():
        out[mask] = (1 - alpha) * out[mask] + alpha * color
        # 1px contour: pixels in mask adjacent to a non-mask pixel.
        m = mask
        edge = m & ~(
            np.pad(m, ((1, 0), (0, 0)))[:-1] & np.pad(m, ((0, 1), (0, 0)))[1:]
            & np.pad(m, ((0, 0), (1, 0)))[:, :-1] & np.pad(m, ((0, 0), (0, 1)))[:, 1:]
        )
        out[edge] = color
    return np.clip(out, 0, 1)


def scenes_from_checkpoint(ckpt: dict) -> list:
    """
    Normalize a checkpoint into a list of (label, scene_dict) to visualize.

    Multi-scene checkpoints (train_multiscene.py) carry a "scenes" list; each entry becomes
    ("<split>_<name>", scene). Single-scene checkpoints (train_overfit.py) yield one entry
    with label None, built from the top-level keys.
    """
    if ckpt.get("scenes"):
        return [(f"{s.get('split', 'train')}_{s['name']}", s) for s in ckpt["scenes"]]
    return [(None, {
        "name": ckpt.get("args", {}).get("train_scenes"),
        "images": ckpt["images"],
        "scene_dir": ckpt.get("scene_dir"),
        "coordinates": ckpt["coordinates"],
        "view_ids": ckpt["view_ids"],
        "gt": ckpt["gt"],
        "frame_names": ckpt.get("frame_names"),
        "metrics": ckpt.get("final_metrics") or {},
    })]


def visualize_scene(model, scene: dict, out_dir: Path, device: str, args) -> int:
    """Forward + match + per-frame overlay figures for one stored scene batch. Returns #frames."""
    out_dir.mkdir(parents=True, exist_ok=True)

    images = decode_checkpoint_images(
        scene, scans_root=getattr(args, "scans_root", None)).to(device)  # [1, S, 3, H, W]
    coordinates = scene["coordinates"].to(device)  # [1, N, 2]
    view_ids = scene["view_ids"].to(device)        # [1, N]
    gt = {k: v.to(device) for k, v in scene["gt"].items()}
    gt_masks = gt["masks"]      # [Ng, S, h, w]
    gt_classes = gt["classes"]  # [Ng]
    frame_names = scene.get("frame_names", None)

    S = images.shape[1]
    H, W = images.shape[-2:]

    # Phase 3 query modes: learned queries ignore coordinates (use placeholders); hybrid
    # prepends learned-query placeholders to the point queries. Keep the query count aligned
    # with the head's output so the matcher below stays consistent.
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

    print(f"Scene: S={S} frames, {gt_classes.shape[0]} GT instances, image {H}x{W}")
    m = scene.get("metrics") or {}
    if m.get("mIoU") is not None:
        print(f"Checkpoint metrics: mIoU={m.get('mIoU'):.3f} AP50={m.get('AP50'):.3f} "
              f"class_acc={m.get('class_acc'):.3f}")

    # --- Forward pass -------------------------------------------------------------------------
    with torch.no_grad():
        agg_list, patch_start_idx = model.backbone.aggregator(images)
        global_features = agg_list[-1]
        anchors = None
        if mode == "anchor3d":
            # Arm E: rebuild the 3D anchors from the frozen point head — deterministic given
            # the same frames (FPS is deterministic), so nothing anchor-related needs to be
            # stored in the checkpoint. Coordinates become placeholders (ignored by the head).
            pts3d, pts3d_conf = model.backbone.point_head(
                agg_list, images=images, patch_start_idx=patch_start_idx)
            anchors = build_anchors(global_features, patch_start_idx, pts3d, pts3d_conf,
                                    num_anchors=model.decoder_head.num_anchors,
                                    knn=model.decoder_head.anchor_knn)
            K = anchors["xyz"].shape[1]
            coordinates = torch.zeros(1, K, 2, device=device)
            view_ids = torch.zeros(1, K, dtype=torch.long, device=device)
        # anchors passed only when set, so stub/legacy heads without the kwarg keep working.
        extra = {"anchors": anchors} if anchors is not None else {}
        class_logits, mask_embeddings, pred_masks = model.decoder_head(
            coordinates, view_ids, images, global_features, patch_start_idx, **extra
        )
    # Drop batch dim (B=1)
    class_logits = class_logits[0]       # [N, C]
    mask_embeddings = mask_embeddings[0]  # [N, D]
    pred_masks = pred_masks[0]           # [N, S, h, w]

    # --- Match predictions to GT (same matcher as training: weights all 1.0) ------------------
    # Learned/hybrid/anchor3d coordinates are placeholders → drop the coord cost (matches
    # training).
    coord_weight = 0.0 if mode in ("learned", "hybrid", "anchor3d") else 1.0
    matcher = PointBipartiteMatcher(class_weight=1.0, mask_weight=1.0, coord_weight=coord_weight)
    pred_idx, gt_idx, _ = matcher(
        class_logits, mask_embeddings, coordinates[0],
        gt_classes, gt_mask_embeddings=None, gt_coordinates=gt["coordinates"],
        pred_masks=pred_masks, gt_masks=gt_masks,
    )
    pred_idx = pred_idx.cpu().tolist()
    gt_idx = gt_idx.cpu().tolist()

    probs = torch.softmax(class_logits, dim=-1)        # [N, C]
    pred_labels = probs.argmax(dim=-1)                  # [N]
    pred_scores = probs.max(dim=-1).values              # [N]

    # --- Honest, GT-free selection (identical rule to the 3D viewer) --------------------------
    # train/postprocess.select_instances is the single source of truth shared with
    # demos/demo_gradio.py, so the "Prediction (honest)" panel here and the 3D point cloud
    # pick exactly the same instances: drop background/low-score queries, winner-takes-all.
    keep, _, _, assign = select_instances(
        class_logits, pred_masks,
        score_thr=args.score_threshold, mask_thr=args.mask_threshold,
    )
    assign_full = upsample_assignment(assign, (H, W)).cpu().numpy()  # [S, H, W], values index keep

    # All masks are rendered at the head's NATIVE resolution and nearest-upsampled, so GT and
    # both prediction panels share the same (honest) patch-grid sharpness — no panel looks
    # artificially smooth. GT: one binary map per instance.
    gt_full = F.interpolate(
        gt_masks.reshape(-1, 1, *gt_masks.shape[-2:]).float(),
        size=(H, W), mode="nearest",
    ).reshape(gt_masks.shape[0], S, H, W).cpu().numpy() > 0.5
    # Oracle: per-query native mask thresholded then nearest-upsampled (matched to GT below).
    pred_oracle_full = F.interpolate(
        (torch.sigmoid(pred_masks) >= args.mask_threshold).float().reshape(-1, 1, *pred_masks.shape[-2:]),
        size=(H, W), mode="nearest",
    ).reshape(pred_masks.shape[0], S, H, W).cpu().numpy() > 0.5

    imgs_np = images[0].permute(0, 2, 3, 1).cpu().numpy()  # [S, H, W, 3]

    # --- Report matched instances (oracle view) ----------------------------------------------
    print("\nOracle match (color : GT class -> predicted class, score):")
    matches = []  # (color_i, p, g, gt_cls, pred_cls, score, drawn)
    for color_i, (p, g) in enumerate(zip(pred_idx, gt_idx)):
        gt_cls = int(gt_classes[g].item())
        pred_cls = int(pred_labels[p].item())
        score = float(pred_scores[p].item())
        drawn = score >= args.score_threshold and pred_cls != 0
        matches.append((color_i, p, g, gt_cls, pred_cls, score, drawn))
        flag = "" if drawn else "  (below score thr / bg — not drawn)"
        print(f"  [{color_i:2d}] {IDX_TO_CLASS.get(gt_cls, gt_cls):>14s} -> "
              f"{IDX_TO_CLASS.get(pred_cls, pred_cls):<14s} ({score:.2f}){flag}")

    # The honest panel uses its own colors (keep-order); report its instances too.
    print(f"\nHonest selection (score>={args.score_threshold}, non-bg): {len(keep)} instance(s)")
    for c, i in enumerate(keep):
        print(f"  ({c:2d}) {IDX_TO_CLASS.get(int(pred_labels[i]), int(pred_labels[i])):<14s} "
              f"({float(pred_scores[i]):.2f})")

    # --- Per-frame overlays: RGB | GT | Prediction (honest) | Prediction (oracle) -------------
    for s in range(S):
        base = imgs_np[s]
        gt_ov = base.copy()
        honest_ov = base.copy()
        oracle_ov = base.copy()
        # GT + oracle share the matched-pair color scheme.
        for color_i, p, g, gt_cls, pred_cls, score, drawn in matches:
            col = _color(color_i)
            gt_m = gt_full[g, s]
            if gt_m.any():
                gt_ov = overlay_mask(gt_ov, gt_m, col, args.alpha)
            if drawn:
                pred_m = pred_oracle_full[p, s]
                if pred_m.any():
                    oracle_ov = overlay_mask(oracle_ov, pred_m, col, args.alpha)
        # Honest panel: winner-takes-all assignment, colored by keep-order.
        for c in range(len(keep)):
            hm = assign_full[s] == c
            if hm.any():
                honest_ov = overlay_mask(honest_ov, hm, _color(c), args.alpha)

        fname = frame_names[s] if frame_names is not None else f"frame {s}"
        if isinstance(fname, (list, tuple)):
            fname = fname[0]
        fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
        for ax, im, title in zip(
            axes, [base, gt_ov, honest_ov, oracle_ov],
            ["RGB", "Ground truth", "Prediction (honest, no GT)", "Prediction (oracle, GT-matched)"],
        ):
            ax.imshow(im)
            ax.set_title(title, fontsize=11)
            ax.axis("off")
        # GT/oracle legend (shared colors): per-class instance index "{class} #{k}".
        class_counts = {}
        gt_legend = []
        for ci, p, g, gc, pc, sc, dr in matches:
            k = class_counts.get(gc, 0)
            class_counts[gc] = k + 1
            gt_legend.append(Patch(facecolor=_color(ci), label=f"{IDX_TO_CLASS.get(gc, gc)} #{k}"))
        # Honest legend (own colors): predicted class + score.
        honest_legend = [
            Patch(facecolor=_color(c), label=f"{IDX_TO_CLASS.get(int(pred_labels[i]), int(pred_labels[i]))} ({float(pred_scores[i]):.2f})")
            for c, i in enumerate(keep)
        ]
        if gt_legend:
            leg1 = fig.legend(handles=gt_legend, loc="lower left", ncol=min(len(gt_legend), 5),
                              fontsize=7, frameon=False, bbox_to_anchor=(0.02, -0.02),
                              title="GT / oracle", title_fontsize=8)
            fig.add_artist(leg1)
        if honest_legend:
            fig.legend(handles=honest_legend, loc="lower right", ncol=min(len(honest_legend), 5),
                       fontsize=7, frameon=False, bbox_to_anchor=(0.98, -0.02),
                       title="honest", title_fontsize=8)
        fig.suptitle(f"Frame {s} — {fname}", fontsize=13)
        fig.text(0.5, 0.93, "one color = one instance (mask spans all frames jointly); "
                 "honest = same selection as 3D viewer", ha="center", fontsize=9, style="italic")
        fig.tight_layout(rect=[0, 0.06, 1, 0.92])
        out_path = out_dir / f"frame_{s:02d}_overlay.png"
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_path}")

    return S


def main():
    parser = argparse.ArgumentParser(description="Visualize D4RT predicted instance masks")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint.pth")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output dir for PNGs (default: <checkpoint dir>/visualizations)")
    parser.add_argument("--scenes", type=str, default=None,
                        help="Comma-separated scene names to render from a multi-scene "
                             "checkpoint (default: all stored scenes)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--scans_root", type=str, default=DEFAULT_SCANS_ROOT,
                        help="Root for reloading frames from --checkpoint_light checkpoints "
                             "(when no per-scene image pixels are stored)")
    parser.add_argument("--mask_threshold", type=float, default=0.5,
                        help="Sigmoid threshold for a predicted mask pixel")
    parser.add_argument("--score_threshold", type=float, default=0.5,
                        help="Min class confidence for a prediction to be drawn")
    parser.add_argument("--alpha", type=float, default=0.5, help="Mask overlay opacity")
    args = parser.parse_args()

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent / "visualizations"

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ck_args = ckpt.get("args", {})

    scenes = scenes_from_checkpoint(ckpt)
    if args.scenes:
        wanted = {s.strip() for s in args.scenes.split(",") if s.strip()}
        scenes = [(label, sc) for label, sc in scenes if sc.get("name") in wanted]
        if not scenes:
            raise SystemExit(f"None of {sorted(wanted)} found in this checkpoint")

    # --- Rebuild model (frozen backbone from HF + trained decoder head) -----------------------
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
    ).to(device)
    model.decoder_head.load_state_dict(ckpt["decoder_head_state_dict"])
    model.eval()

    total = 0
    for label, scene in scenes:
        scene_dir = out_dir / label if label is not None else out_dir
        if label is not None:
            print(f"\n=== {label} ===")
        total += visualize_scene(model, scene, scene_dir, device, args)

    print(f"\n✓ Wrote {total} overlay figures to {out_dir}")


if __name__ == "__main__":
    main()
