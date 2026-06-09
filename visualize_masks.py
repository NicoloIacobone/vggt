#!/usr/bin/env python3
"""
Visualize the dense instance masks predicted by a trained D4RT head.

Loads a checkpoint produced by `train_overfit.py --save_checkpoint ...` (which stores the
trainable decoder head plus the exact fixed overfit batch), runs one forward pass through the
frozen VGGT backbone + decoder, Hungarian-matches each prediction to a GT instance (the same
matcher used in training), upsamples the patch-resolution mask logits to full image resolution,
and writes per-frame RGB overlays comparing the predicted masks against the ground truth.

Usage:
    python visualize_masks.py --checkpoint /path/to/run/checkpoint.pth
    # outputs go to <run dir>/visualizations/ by default
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

sys.path.insert(0, str(Path(__file__).parent))

from train_overfit import D4RTModel
from train.loss import PointBipartiteMatcher
from data.scannet_overfit import IDX_TO_CLASS


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


def main():
    parser = argparse.ArgumentParser(description="Visualize D4RT predicted instance masks")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint.pth")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output dir for PNGs (default: <checkpoint dir>/visualizations)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mask_threshold", type=float, default=0.5,
                        help="Sigmoid threshold for a predicted mask pixel")
    parser.add_argument("--score_threshold", type=float, default=0.5,
                        help="Min class confidence for a prediction to be drawn")
    parser.add_argument("--alpha", type=float, default=0.5, help="Mask overlay opacity")
    args = parser.parse_args()

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ck_args = ckpt.get("args", {})

    # --- Rebuild model (frozen backbone from HF + trained decoder head) -----------------------
    model = D4RTModel(
        freeze_backbone=True,
        num_views=ck_args.get("num_views", 10) if isinstance(ck_args.get("num_views", 10), int) else 10,
        decoder_hidden_dim=256,
        mask_embed_dim=256,
        dropout=0.0,
    ).to(device)
    model.decoder_head.load_state_dict(ckpt["decoder_head_state_dict"])
    model.eval()

    images = ckpt["images"].to(device)          # [1, S, 3, H, W]
    coordinates = ckpt["coordinates"].to(device)  # [1, N, 2]
    view_ids = ckpt["view_ids"].to(device)        # [1, N]
    gt = {k: v.to(device) for k, v in ckpt["gt"].items()}
    gt_masks = gt["masks"]      # [Ng, S, h, w]
    gt_classes = gt["classes"]  # [Ng]
    frame_names = ckpt.get("frame_names", None)

    S = images.shape[1]
    H, W = images.shape[-2:]

    print(f"Scene: S={S} frames, {gt_classes.shape[0]} GT instances, image {H}x{W}")
    if ckpt.get("final_metrics"):
        m = ckpt["final_metrics"]
        print(f"Checkpoint metrics: mIoU={m.get('mIoU'):.3f} AP50={m.get('AP50'):.3f} "
              f"class_acc={m.get('class_acc'):.3f}")

    # --- Forward pass -------------------------------------------------------------------------
    with torch.no_grad():
        agg_list, patch_start_idx = model.backbone.aggregator(images)
        global_features = agg_list[-1]
        class_logits, mask_embeddings, pred_masks = model.decoder_head(
            coordinates, view_ids, images, global_features, patch_start_idx
        )
    # Drop batch dim (B=1)
    class_logits = class_logits[0]       # [N, C]
    mask_embeddings = mask_embeddings[0]  # [N, D]
    pred_masks = pred_masks[0]           # [N, S, h, w]

    # --- Match predictions to GT (same matcher as training: weights all 1.0) ------------------
    matcher = PointBipartiteMatcher(class_weight=1.0, mask_weight=1.0, coord_weight=1.0)
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

    # Upsample predicted (sigmoid) and GT masks to image resolution.
    pred_prob_full = F.interpolate(
        torch.sigmoid(pred_masks).reshape(-1, 1, *pred_masks.shape[-2:]),
        size=(H, W), mode="bilinear", align_corners=False,
    ).reshape(pred_masks.shape[0], S, H, W).cpu().numpy()
    gt_full = F.interpolate(
        gt_masks.reshape(-1, 1, *gt_masks.shape[-2:]).float(),
        size=(H, W), mode="nearest",
    ).reshape(gt_masks.shape[0], S, H, W).cpu().numpy() > 0.5

    imgs_np = images[0].permute(0, 2, 3, 1).cpu().numpy()  # [S, H, W, 3]

    # --- Report matched instances ------------------------------------------------------------
    print("\nMatched instances (color : GT class -> predicted class, score):")
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

    # --- Per-frame overlays: original | GT | prediction --------------------------------------
    for s in range(S):
        base = imgs_np[s]
        gt_ov = base.copy()
        pred_ov = base.copy()
        for color_i, p, g, gt_cls, pred_cls, score, drawn in matches:
            col = _color(color_i)
            gt_m = gt_full[g, s]
            if gt_m.any():
                gt_ov = overlay_mask(gt_ov, gt_m, col, args.alpha)
            if drawn:
                pred_m = pred_prob_full[p, s] >= args.mask_threshold
                if pred_m.any():
                    pred_ov = overlay_mask(pred_ov, pred_m, col, args.alpha)

        fname = frame_names[s] if frame_names is not None else f"frame {s}"
        if isinstance(fname, (list, tuple)):
            fname = fname[0]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
        for ax, im, title in zip(
            axes, [base, gt_ov, pred_ov], ["RGB", "Ground truth", "Prediction"]
        ):
            ax.imshow(im)
            ax.set_title(title, fontsize=12)
            ax.axis("off")
        legend = [
            Patch(facecolor=_color(ci), label=f"{IDX_TO_CLASS.get(gc, gc)}")
            for ci, p, g, gc, pc, sc, dr in matches
        ]
        fig.legend(handles=legend, loc="lower center", ncol=min(len(legend), 6), fontsize=8,
                   frameon=False, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(f"Frame {s} — {fname}", fontsize=13)
        fig.tight_layout(rect=[0, 0.04, 1, 0.97])
        out_path = out_dir / f"frame_{s:02d}_overlay.png"
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_path}")

    print(f"\n✓ Wrote {S} overlay figures to {out_dir}")


if __name__ == "__main__":
    main()
