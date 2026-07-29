"""
Scoring and visualisation for the single-frame MaskDINO track (docs/MASKDINO.md §6).

The scoring rules themselves — dropping empty masks, the COCO top-k cap — live in
`train/perframe.py`, shared with `scripts/eval_perframe.py` so the two model families are
measured identically. This module only drives them over the cached scenes.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from data.scannet_overfit import IDX_TO_CLASS
from models.maskdino import build_bundle_target, to_scannet_class_logits
from train.eval_metrics import compute_instance_segmentation_metrics
from train.maskdino_data import gather_batch
from train.perframe import METRIC_KEYS, drop_empty_masks, topk_predictions


@torch.no_grad()
def eval_scenes(model, scenes: List[Dict], args, device: str) -> Dict[str, Dict[str, float]]:
    """
    Per-frame instance-segmentation metrics, averaged over the frames of each scene.

    Frames with no GT instance are skipped (they have no defined mIoU/AP and would only
    dilute the mean). Every frame is scored TWICE (docs/MASKDINO.md §6):

      - thresholded (`--score_threshold`, MaskDINO's OBJECT_MASK_THRESHOLD): the headline
        numbers, the closest analogue to the D4RT arms' "argmax != background" filter;
      - threshold-free, suffix `_all`: every query kept and ranked by score — the standard
        COCO detection protocol. It is informative from epoch 1, whereas the thresholded
        numbers stay at 0 for a long while (focal-trained sigmoid scores start near zero).

    With `--multi_frame` the whole bundle has to go through the model at once (the queries are
    shared), so scoring is delegated to `eval_scenes_multiframe`, which reports these same
    per-frame keys plus the `bundle_*` multi-view ones.
    """
    if getattr(args, "multi_frame", False):
        return eval_scenes_multiframe(model, scenes, args, device)
    was_training = model.training
    model.eval()
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
        all_keys = METRIC_KEYS + [f"{k}_all" for k in METRIC_KEYS]
        per_scene[scene["name"]] = ({k: float(np.mean([r[k] for r in rows])) for k in all_keys}
                                    if rows else {k: 0.0 for k in all_keys})
    if was_training:
        model.train()
    return per_scene


@torch.no_grad()
def eval_scenes_multiframe(model, scenes: List[Dict], args, device: str
                           ) -> Dict[str, Dict[str, float]]:
    """
    Scoring for the shared-query multi-frame model (docs/MASKDINO.md §8) — TWO protocols at once.

      - the same per-frame numbers as `eval_scenes` (`mIoU`, `AP50`, …, `*_all`), so a
        multi-frame run is directly comparable to the single-frame bar;
      - `bundle_*`: the multi-view protocol of the retired D4RT arms. A query now owns a mask
        VOLUME [S, h, w], scored against the bundle's GT volume, with one class score per query
        (the max over the views: an instance exists if some view detects it confidently). This
        is the metric that was meaningless while queries were per-frame.

    Both use the shared rules of `train/perframe.py`: a prediction that claims no pixels is
    dropped (per frame for the per-frame numbers, per volume for the bundle numbers), and at
    most `--eval_topk` predictions are kept.
    """
    was_training = model.training
    model.eval()
    per_scene = {}
    for si, scene in enumerate(scenes):
        bundle_targets_all = scene["bundles"][0]["targets"]
        samples = [(si, 0, fi) for fi in range(len(bundle_targets_all))]
        feats, targets, psi = gather_batch(scenes, samples, device)
        s = len(samples)
        out, _ = model.head(feats, psi, None, frames_per_sample=s)

        rows, bundle_rows = [], []
        for b in range(s):
            if int(targets[b]["labels"].numel()) == 0:
                continue                        # no GT in this view → undefined metrics
            pm, cl = drop_empty_masks(out["pred_masks"][b],
                                      to_scannet_class_logits(out["pred_logits"][b]))
            pm, cl = topk_predictions(pm, cl, args.eval_topk)
            rows.append(_score_pair(pm, cl, targets[b]["masks"], targets[b]["labels"] + 1, args))

        bt = build_bundle_target(targets)
        if int(bt["labels"].numel()) > 0:
            # [Q, S, h, w] volumes; one class score per query = max over the views
            vol = out["pred_masks"].permute(1, 0, 2, 3)
            cls = to_scannet_class_logits(out["pred_logits"].max(dim=0).values)
            vol, cls = drop_empty_masks(vol, cls)
            vol, cls = topk_predictions(vol, cls, args.eval_topk)
            bundle_rows.append(_score_pair(vol, cls, bt["masks"], bt["labels"] + 1, args))

        keys = METRIC_KEYS + [f"{k}_all" for k in METRIC_KEYS]
        m = ({k: float(np.mean([r[k] for r in rows])) for k in keys} if rows
             else {k: 0.0 for k in keys})
        m.update({f"bundle_{k}": (float(bundle_rows[0][k]) if bundle_rows else 0.0)
                  for k in keys})
        per_scene[scene["name"]] = m
    if was_training:
        model.train()
    return per_scene


def _score_pair(pred_masks, class_logits, gt_masks, gt_classes, args) -> Dict[str, float]:
    """One scored unit (a frame or a bundle) at both operating points (docs/MASKDINO.md §6.2)."""
    common = dict(pred_masks=pred_masks, class_logits=class_logits, gt_masks=gt_masks,
                  gt_classes=gt_classes, background_class=0, score_mode="sigmoid")
    row = compute_instance_segmentation_metrics(score_threshold=args.score_threshold, **common)
    row.update({f"{k}_all": v for k, v in
                compute_instance_segmentation_metrics(score_threshold=0.0, **common).items()})
    return row


def mean_metric(per_scene: Dict[str, Dict[str, float]], key: str) -> float:
    """Mean of one metric across scenes."""
    vals = [m[key] for m in per_scene.values()]
    return float(np.mean(vals)) if vals else 0.0


def fmt(m: Dict[str, float]) -> str:
    """One-line log rendering of a scene's metric dict."""
    return (f"mIoU={m['mIoU']:.3f}  AP50={m['AP50']:.3f}  AP75={m['AP75']:.3f}  "
            f"mAP={m['mAP']:.3f}  class_acc={m['class_acc']:.3f}  "
            f"pred/gt={m['num_pred']:.1f}/{m['num_gt']:.1f}  "
            f"| all-query: mIoU={m['mIoU_all']:.3f} AP50={m['AP50_all']:.3f}")


@torch.no_grad()
def visualize(model, scenes: List[Dict], args, device: str, out_dir: Path,
              max_scenes: int = 2, max_frames: int = 4) -> int:
    """Write RGB | GT | prediction panels per frame; returns how many figures were written."""
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
        with_gt = [fi for fi, t in enumerate(bundle["targets"]) if int(t["labels"].numel()) > 0]
        if not with_gt:
            continue
        # A shared-query model must see the whole bundle at once; the single-frame model does
        # not care, so it only pays for the frames actually drawn.
        multi = bool(getattr(args, "multi_frame", False))
        frames = list(range(len(bundle["targets"]))) if multi else with_gt[:max_frames]
        samples = [(si, 0, fi) for fi in frames]
        feats, targets, psi = gather_batch(scenes, samples, device)
        out, _ = model.head(feats, psi, None, frames_per_sample=len(frames) if multi else 1)
        drawn = 0
        for b, (_, _, fi) in enumerate(samples):
            if fi not in with_gt or drawn >= max_frames:
                continue
            drawn += 1
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
            # Winner-takes-all over the kept queries, highest score first.
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
