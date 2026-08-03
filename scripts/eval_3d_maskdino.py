#!/usr/bin/env python3
"""
The 3D ruler (docs/MASKDINO.md §9, docs/todo.md 1d): score a `--multi_frame` MaskDINO
checkpoint on the official ScanNet 3D instance benchmark — the protocol SegVGGT and
FAST3DIS report, and the only one that makes our numbers placeable in the literature.

Per scene (official val split, scannet_frames_25k frames — whole-scan coverage):
  1. ONE forward pass over all sampled frames: the frozen VGGT aggregator feeds both the
     MaskDINO head (one query set for the whole scene, `frames_per_sample=S`) and VGGT's
     own depth + camera heads. No GT geometry, depth sensor, or poses enter inference.
  2. Per-view masks are upsampled to 518², each pixel goes to its highest-probability
     query (> --mask_prob_threshold), and is unprojected with the PREDICTED depth +
     cameras into the bundle frame.
  3. Eval-time registration only: a closed-form Sim(3) (Umeyama) from predicted-vs-GT
     camera centers, optionally refined by similarity ICP against the mesh vertices —
     the FAST3DIS convention. VGGT's output scale is arbitrary, so this step is what
     expresses the finished prediction in the benchmark mesh's coordinate frame.
  4. SegVGGT-style lifting: per-vertex votes within --vote_radius, plurality per
     superpoint; one query = one 3D instance, no post-hoc matching anywhere.
  5. The vendored official evaluator (train/benchmark3d.py) scores AP / AP50 / AP25 over
     the benchmark's 18 classes. `otherfurniture` is not predictable by our 19-class head
     (it is background in our 2D GT), so a 17-common-class average is reported as a
     diagnostic next to the official 18-class headline.

    python scripts/eval_3d_maskdino.py --checkpoint <run>/checkpoint_best_bundle.pth \
        --frames_root $TMPDIR/scans25k --gt_root $TMPDIR/scans3d
    → <run_dir>/eval3d_<ckpt stem>.json

NEVER quote these numbers next to the 2D-protocol tables (docs/RESULTS.md §1).
"""

import argparse
import contextlib
import inspect
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.scannet_overfit import load_frames_by_name
from models.maskdino.head import MaskDINOVGGTHead, to_scannet_class_logits
from models.maskdino.model import MaskDINOVGGTModel
from train.benchmark3d import (BENCHMARK_CLASS_NAMES, assign_instances_for_scan,
                               compute_averages, evaluate_matches, format_results,
                               MIN_REGION_SIZE, OVERLAPS)
from train.eval3d_geometry import (accumulate_votes, apply_sim3, assign_pixels_to_queries,
                                   camera_centers_from_extrinsics, icp_refine_sim3,
                                   superpoint_majority, umeyama_sim3,
                                   unproject_masks_to_points)
from train.maskdino_data import DTYPES
from train.scannet3d import SCANNET_IDX_TO_NYU40, load_frames25k_poses, load_scene_3d_gt, \
    sample_frames25k
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

IMG_SIZE = 518
DEFAULT_TSV = ("/cluster/work/igp_psr/niacobone/distillation/dataset/scannet/"
               "scannetv2-labels.combined.tsv")


def build_argparser():
    p = argparse.ArgumentParser(description="Official ScanNet 3D instance eval (the 3D ruler)")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--frames_root", type=str, required=True,
                   help="scans25k tree: <scene>/{color,pose,intrinsics_color.txt}")
    p.add_argument("--gt_root", type=str, required=True,
                   help="scans3d tree: <scene>/{*_vh_clean_2.ply,*.segs.json,*.aggregation.json}")
    p.add_argument("--tsv", type=str, default=DEFAULT_TSV)
    p.add_argument("--scenes", type=str, nargs="*", default=None,
                   help="explicit scene ids (default: every scene present in both roots)")
    p.add_argument("--num_frames", type=int, default=None,
                   help="cap frames per scene (default: all ~16 the 25k export has)")
    p.add_argument("--eval_topk", type=int, default=100,
                   help="max queries kept per scene, ranked by class score (COCO convention)")
    p.add_argument("--min_score", type=float, default=0.0,
                   help="drop queries below this class score before voting")
    p.add_argument("--mask_prob_threshold", type=float, default=0.5,
                   help="a pixel joins its argmax query only above this sigmoid prob")
    p.add_argument("--vote_radius", type=float, default=0.05,
                   help="meters (mesh units): a point votes on the nearest vertex within this")
    p.add_argument("--depth_conf_percentile", type=float, default=0.0,
                   help="drop the lowest-confidence p%% of depth pixels per scene (0 = keep all)")
    p.add_argument("--icp", action=argparse.BooleanOptionalAction, default=True,
                   help="refine the camera-center Sim(3) by similarity ICP against the mesh")
    p.add_argument("--icp_max_dist", type=float, default=0.3)
    p.add_argument("--dump_ply", action="store_true",
                   help="write an instance-colored .ply per scene next to --out (eyeballing)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out", type=str, default=None)
    return p


# Knobs that change the numbers. Two runs of the SAME checkpoint that differ in any of these
# must not write to the same file — that silently cost us job 9503137's JSON on 2026-08-03
# (both knob settings landed on `eval3d_<stem>.json`, the second overwrote the first and only
# the SLURM log preserved the numbers).
RESULT_AFFECTING = ("num_frames", "eval_topk", "min_score", "mask_prob_threshold",
                    "vote_radius", "depth_conf_percentile", "icp", "icp_max_dist", "scenes")


def default_out_path(ckpt_path: Path, args, parser) -> Path:
    """
    `eval3d_<ckpt stem>.json`, plus a compact tag naming every non-default result-affecting knob.

    All-defaults runs keep the documented bare name (docs/MASKDINO.md §9); a tuned run gets e.g.
    `eval3d_checkpoint_best_bundle__vote_radius0.1_depth_conf_percentile25.0.json`.
    """
    parts = []
    for name in RESULT_AFFECTING:
        value = getattr(args, name, None)
        if value == parser.get_default(name):
            continue
        rendered = ("no" + name if value is False else
                    name if value is True else
                    f"{name}{'-'.join(map(str, value)) if isinstance(value, list) else value}")
        parts.append(str(rendered).replace("/", "_"))
    tag = ("__" + "_".join(parts)) if parts else ""
    return ckpt_path.parent / f"eval3d_{ckpt_path.stem}{tag}.json"


# ------------------------------------------------------------------------------------------

def load_model(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "head_config" not in ckpt or "head_state_dict" not in ckpt:
        raise SystemExit(f"{ckpt_path} is not a MaskDINO checkpoint (missing head_config)")
    valid = set(inspect.signature(MaskDINOVGGTHead.__init__).parameters) - {"self"}
    head_kwargs = {k: v for k, v in ckpt["head_config"].items() if k in valid}
    model = MaskDINOVGGTModel(head_kwargs, load_backbone=True).to(device)
    model.head.load_state_dict(ckpt["head_state_dict"])
    model.eval()
    train_args = ckpt.get("args", {})
    if not train_args.get("multi_frame", False):
        print("⚠ checkpoint was NOT trained with --multi_frame: queries are per-frame "
              "hypotheses, so one query is NOT one instance across views — the lifted 3D "
              "instances will be meaningless. Proceeding anyway (ablation use only).")
    return model, train_args


@torch.no_grad()
def run_scene(model, train_args: Dict, scene: str, scene_frames_dir: Path, gt3d: Dict,
              args, device: str) -> Dict:
    """The five pipeline steps for one scene; returns preds + diagnostics."""
    t0 = time.time()
    stems = sample_frames25k(scene_frames_dir, args.num_frames)
    images = load_frames_by_name(str(scene_frames_dir), stems, IMG_SIZE).to(device)  # [S,3,H,W]
    S = images.shape[0]

    # -- 1. one backbone pass: tokens for the head, depth + cameras for the geometry ------
    autocast_dtype = DTYPES[train_args.get("backbone_dtype", "float32")]  # trainer default
    use_autocast = device.startswith("cuda") and autocast_dtype != torch.float32
    ctx = (torch.autocast("cuda", dtype=autocast_dtype) if use_autocast
           else contextlib.nullcontext())
    with ctx:
        agg_list, psi = model.backbone.aggregator(images[None])
    feature_layers = train_args.get("feature_layers", [-1])
    if isinstance(feature_layers, str):     # pre-parse string form, e.g. "-1" or "4,11,23"
        feature_layers = [int(x) for x in feature_layers.split(",") if x.strip()]
    if train_args.get("feature_mode", "single") == "single":
        # faithful to a per-frame-features checkpoint: tokens from S single-frame passes
        # (geometry still needs the bundle pass above — S=1 depth scales don't cohere)
        per_frame = []
        for f in range(S):
            with ctx:
                sf_list, _ = model.backbone.aggregator(images[None, f:f + 1])
            per_frame.append(torch.cat([sf_list[i].float() for i in feature_layers],
                                       dim=-1)[0, 0])
            del sf_list
        feats = torch.stack(per_frame)
    else:
        feats = torch.cat([agg_list[i].float() for i in feature_layers], dim=-1)[0]
    with torch.autocast("cuda", enabled=False) if device.startswith("cuda") \
            else contextlib.nullcontext():
        # the fork's aggregator caches only the layers the heads index (4/11/17/23 + last)
        # and returns None elsewhere — float() only what exists
        agg32 = [a.float() if a is not None else None for a in agg_list]
        depth, depth_conf = model.backbone.depth_head(agg32, images=images[None],
                                                      patch_start_idx=psi)
        pose_enc = model.backbone.camera_head(agg32)[-1]
    del agg_list, agg32
    extri, intri = pose_encoding_to_extri_intri(pose_enc, (IMG_SIZE, IMG_SIZE))
    depth = depth[0].cpu().numpy()                      # [S, H, W, 1]
    depth_conf = depth_conf[0].cpu().numpy()            # [S, H, W]
    extri = extri[0].cpu().numpy()                      # [S, 3, 4] cam-from-world
    intri = intri[0].cpu().numpy()                      # [S, 3, 3]

    out, _ = model.head(feats, int(psi), None, frames_per_sample=S)

    # -- 2. keep queries, assign pixels ----------------------------------------------------
    # one class score per query = max over views (the bundle-protocol convention, §8.2)
    probs = to_scannet_class_logits(out["pred_logits"].max(dim=0).values).sigmoid()  # [Q,20]
    score, cls_idx = probs[:, 1:].max(dim=1)
    cls_idx = cls_idx + 1                                              # 1..19 dataset classes
    nyu40 = torch.as_tensor([SCANNET_IDX_TO_NYU40[int(c)] for c in cls_idx],
                            device=score.device)
    keep = (score >= args.min_score) & (nyu40 > 2)     # drop wall/floor: not benchmark classes
    keep_idx = torch.nonzero(keep).squeeze(1)
    keep_idx = keep_idx[score[keep_idx].argsort(descending=True)][:args.eval_topk]
    if len(keep_idx) == 0:                             # nothing to lift — GT becomes all FNs
        return {"preds": [], "assign": np.full(len(gt3d["vertices"]), -1),
                "stats": {"frames": S, "kept_queries": 0,
                          "seconds": round(time.time() - t0, 1)}}
    q_score = score[keep_idx].cpu().numpy()
    q_label = nyu40[keep_idx].cpu().numpy()

    pixel_query = np.stack([
        assign_pixels_to_queries(out["pred_masks"][f, keep_idx], (IMG_SIZE, IMG_SIZE),
                                 args.mask_prob_threshold)
        for f in range(S)])                                            # [S, H, W]

    # -- 3. unproject with the PREDICTED geometry, register with Sim(3) (eval-only) --------
    world = unproject_depth_map_to_point_map(depth, extri, intri)      # [S, H, W, 3]
    conf_thr = (np.percentile(depth_conf, args.depth_conf_percentile)
                if args.depth_conf_percentile > 0 else -np.inf)
    points, point_query = unproject_masks_to_points(world, pixel_query, depth_conf, conf_thr)

    poses = load_frames25k_poses(scene_frames_dir)
    gt_centers = np.stack([poses[s][:3, 3] for s in stems])
    pred_centers = camera_centers_from_extrinsics(extri)
    s3, R3, t3 = umeyama_sim3(pred_centers, gt_centers)
    center_rms = float(np.sqrt(((apply_sim3(pred_centers, s3, R3, t3) - gt_centers) ** 2)
                               .sum(axis=1).mean()))
    icp_stats = {}
    if args.icp:
        # align the whole predicted cloud (conf-kept pixels), not just the mask pixels
        cloud = world[depth_conf >= conf_thr] if np.isfinite(conf_thr) \
            else world.reshape(-1, 3)
        s3, R3, t3, icp_stats = icp_refine_sim3(cloud, gt3d["vertices"], s3, R3, t3,
                                                max_dist=args.icp_max_dist)
    points = apply_sim3(points, s3, R3, t3)

    # -- 4. votes -> superpoint majority -> instances --------------------------------------
    votes = accumulate_votes(points, point_query, gt3d["vertices"], len(keep_idx),
                             args.vote_radius)
    assign = superpoint_majority(votes, gt3d["superpoints"])
    preds = [{"mask": assign == q, "label_id": int(q_label[q]),
              "confidence": float(q_score[q])} for q in range(len(keep_idx))]

    annotated = gt3d["gt_ids"] > 0
    stats = {
        "frames": S, "kept_queries": int(len(keep_idx)), "points": int(len(points)),
        "sim3_scale": float(s3), "center_rms_m": center_rms, **icp_stats,
        "voted_vertex_frac": float((votes.sum(axis=1) > 0).mean()),
        "annotated_assigned_frac": float((assign[annotated] >= 0).mean())
        if annotated.any() else float("nan"),
        "seconds": round(time.time() - t0, 1),
    }
    return {"preds": preds, "assign": assign, "stats": stats}


def write_instance_ply(path: Path, vertices: np.ndarray, assign: np.ndarray):
    """Ascii ply colored by instance (grey = unassigned) — for eyeballing only."""
    rng = np.random.default_rng(0)
    palette = rng.integers(40, 255, size=(int(assign.max()) + 2, 3))
    colors = np.where((assign >= 0)[:, None], palette[assign], np.array([90, 90, 90]))
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n"
                f"element vertex {len(vertices)}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                "end_header\n")
        for p, c in zip(vertices, colors):
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}\n")


def seventeen_class_mean(avgs: Dict) -> Dict[str, float]:
    """The 17 classes our head shares with the benchmark (no otherfurniture) — DIAGNOSTIC."""
    names = [n for n in BENCHMARK_CLASS_NAMES if n != "otherfurniture"]
    return {key: float(np.nanmean([avgs["classes"][n][k] for n in names]))
            for key, k in (("all_ap", "ap"), ("all_ap_50%", "ap50%"),
                           ("all_ap_25%", "ap25%"))}


def main():
    parser = build_argparser()
    args = parser.parse_args()
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    ckpt_path = Path(args.checkpoint)
    model, train_args = load_model(ckpt_path, device)

    frames_root, gt_root = Path(args.frames_root), Path(args.gt_root)
    if args.scenes:
        scenes = list(args.scenes)
    else:
        scenes = sorted(d.name for d in gt_root.iterdir()
                        if d.is_dir() and (frames_root / d.name).is_dir())
    if not scenes:
        raise SystemExit(f"no scenes found under both {gt_root} and {frames_root}")
    print(f"Scoring {len(scenes)} scene(s), checkpoint {ckpt_path.name} "
          f"(multi_frame={train_args.get('multi_frame', False)}, "
          f"feature_mode={train_args.get('feature_mode', 'single')})")

    out_path = Path(args.out) if args.out else default_out_path(ckpt_path, args, parser)
    matches, per_scene, failed = {}, {}, []
    for i, scene in enumerate(scenes):
        gt3d = load_scene_3d_gt(gt_root, scene, args.tsv)
        preds: List[Dict] = []
        try:
            r = run_scene(model, train_args, scene, frames_root / scene, gt3d, args, device)
            preds, per_scene[scene] = r["preds"], r["stats"]
            print(f"[{i + 1}/{len(scenes)}] {scene}: {r['stats']}", flush=True)
            if args.dump_ply:
                write_instance_ply(out_path.parent / f"eval3d_{scene}.ply",
                                   gt3d["vertices"], r["assign"])
        except Exception:  # noqa: BLE001 — a failed scene still counts (its GT -> FNs)
            failed.append(scene)
            print(f"[{i + 1}/{len(scenes)}] {scene} FAILED:\n"
                  + "".join(traceback.format_exc().splitlines(keepends=True)[-6:]), flush=True)
        gt2pred, pred2gt = assign_instances_for_scan(scene, preds, gt3d["gt_ids"],
                                                     MIN_REGION_SIZE)
        matches[scene] = {"gt": gt2pred, "pred": pred2gt}

    aps = evaluate_matches(matches, OVERLAPS, MIN_REGION_SIZE)
    avgs = compute_averages(aps, OVERLAPS)
    diag17 = seventeen_class_mean(avgs)

    print("\nOfficial 18-class ScanNet 3D instance benchmark (the headline):")
    print(format_results(avgs))
    print(f"\n17-common-class diagnostic (our head cannot predict otherfurniture): "
          f"AP {diag17['all_ap']:.3f}  AP50 {diag17['all_ap_50%']:.3f}  "
          f"AP25 {diag17['all_ap_25%']:.3f}")
    if failed:
        print(f"⚠ {len(failed)} scene(s) failed (their GT counted as misses): "
              + ", ".join(failed))

    result = {
        "checkpoint": str(ckpt_path),
        "protocol": "official ScanNet 3D instance benchmark (docs/MASKDINO.md §9); "
                    "NOT comparable to any 2D-protocol number",
        "args": {k: v for k, v in vars(args).items() if k != "scenes"},
        "num_scenes": len(scenes),
        "failed_scenes": failed,
        "results_18class": {k: (None if isinstance(v, float) and np.isnan(v) else v)
                            for k, v in avgs.items() if k != "classes"},
        "results_17class_diagnostic": diag17,
        "per_class": {n: {k: (None if np.isnan(v) else float(v))
                          for k, v in c.items()} for n, c in avgs["classes"].items()},
        "per_scene": per_scene,
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(f"✓ Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
