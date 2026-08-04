#!/usr/bin/env python3
"""
The oracle check that licenses `--transfer_mode gt_projection` (docs/MASKDINO.md §9.9).

A wrong pixel mapping produces a plausible-looking number, so the transfer is not trusted
until it is shown to be a round trip on REAL val scenes — the same discipline §9.2 applies
to the evaluator (real GT fed back as predictions must score exactly 1.000).

Per scene, with no model anywhere in the loop:

  1. RENDER the 3D GT into every sampled view through the SAME projection the transfer uses
     (`project_vertices_to_view`): each mesh vertex the sensor depth confirms paints its GT
     instance id on the mask pixel it lands on, nearest vertex winning the z-buffer.
     Unannotated vertices win the z-buffer too and paint "no instance", so they occlude
     exactly as real geometry does.
  2. Feed those rendered maps back through the transfer AS IF THEY WERE PREDICTIONS, with
     the ordinary superpoint majority on top.
  3. Score with the vendored official evaluator.

Two numbers come out, and they answer two different questions:

  PURITY  — of the annotated vertices the transfer assigned, the fraction it returned to
            their OWN instance. This is the mapping test: it is ~1.0 iff pixels are read at
            the right place. A shifted, transposed or isotropically-rescaled mapping
            collapses it. `--min_purity` (0.99) fails the run if it does not hold.
  AP/AP50/AP25 — the CEILING of the gt_projection protocol on our frame set. It is < 1.0
            not because the mapping is wrong but because a vertex no frame sees receives no
            vote: with the 25k export's ~17 views per scan, part of every mesh is invisible
            and those vertices are missing from every recovered mask. That is a property of
            the frame budget, and it is the honest upper bound to quote next to any
            gt_projection result.

    myenv/bin/python scripts/eval3d_projection_oracle.py \
        --frames_root $TMPDIR/scans25k --gt_root $TMPDIR/scans3d --num_scenes 20
    → <out>/eval3d_projection_oracle.json

CPU-only; no checkpoint, no GPU, no backbone.
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.benchmark3d import (assign_instances_for_scan, compute_averages, evaluate_matches,
                               format_results, MIN_REGION_SIZE, OVERLAPS)
from train.eval3d_geometry import (mask_grid_intrinsic, project_votes_to_vertices,
                                   project_vertices_to_view, superpoint_majority)
from train.scannet3d import (load_frames25k_color_size, load_frames25k_depth,
                             load_frames25k_intrinsics, load_frames25k_poses,
                             load_scene_3d_gt, sample_frames25k)

IMG_SIZE = 518
DEFAULT_TSV = ("/cluster/work/igp_psr/niacobone/distillation/dataset/scannet/"
               "scannetv2-labels.combined.tsv")


def build_argparser():
    p = argparse.ArgumentParser(description="GT-projection transfer oracle (§9.9)")
    p.add_argument("--frames_root", type=str, required=True)
    p.add_argument("--gt_root", type=str, required=True)
    p.add_argument("--tsv", type=str, default=DEFAULT_TSV)
    p.add_argument("--scenes", type=str, nargs="*", default=None)
    p.add_argument("--num_scenes", type=int, default=None,
                   help="cap the number of scenes (evenly spread over the split)")
    p.add_argument("--num_frames", type=int, default=None,
                   help="cap frames per scene (default: all the 25k export has)")
    p.add_argument("--depth_tolerance", type=float, default=0.1)
    p.add_argument("--mask_size", type=int, default=IMG_SIZE,
                   help="the grid our masks live on (the eval's 518)")
    p.add_argument("--min_purity", type=float, default=0.99,
                   help="fail if the round trip returns fewer than this fraction of "
                        "assigned annotated vertices to their own instance")
    p.add_argument("--out", type=str, default=None)
    return p


def render_gt_instances(vertices: np.ndarray, instance_of: np.ndarray, poses: np.ndarray,
                        K_depth: np.ndarray, K_mask: np.ndarray, depth_maps: np.ndarray,
                        mask_hw) -> np.ndarray:
    """
    The GT 3D instance ids drawn into every view [S, H, W] (-1 = no instance), through the
    same projection + sensor-depth visibility test the transfer uses.

    Nearest vertex wins each pixel (z-buffer), and vertices with `instance_of == -1`
    (unannotated mesh: walls, floors, clutter outside the taxonomy) compete for the pixel
    like everything else and then paint -1 — otherwise a background surface in front of an
    object would fail to hide it and the oracle would be easier than reality.
    """
    mh, mw = mask_hw
    out = np.full((len(poses), mh, mw), -1, dtype=np.int64)
    for f in range(len(poses)):
        vidx, rows, cols, z, _ = project_vertices_to_view(
            vertices, poses[f], K_depth, depth_maps[f], K_mask, mask_hw)
        if len(vidx) == 0:
            continue
        order = np.argsort(-z)                      # farthest first: nearest overwrites
        flat = out[f].reshape(-1)
        flat[rows[order] * mw + cols[order]] = instance_of[vidx[order]]
    return out


def run_scene(scene: str, frames_dir: Path, gt3d: Dict, args) -> Dict:
    t0 = time.time()
    stems = sample_frames25k(frames_dir, args.num_frames, require_depth=True)
    poses = load_frames25k_poses(frames_dir)
    poses = np.stack([poses[s] for s in stems])
    K = load_frames25k_intrinsics(frames_dir)
    K_mask = mask_grid_intrinsic(K["color"], load_frames25k_color_size(frames_dir, stems),
                                 (args.mask_size, args.mask_size))
    depth_maps = load_frames25k_depth(frames_dir, stems)

    # every annotated GT instance becomes one "query"
    gt_ids = gt3d["gt_ids"]
    inst_ids = np.unique(gt_ids[gt_ids > 0])
    instance_of = np.full(len(gt_ids), -1, dtype=np.int64)
    for q, gid in enumerate(inst_ids):
        instance_of[gt_ids == gid] = q

    rendered = render_gt_instances(gt3d["vertices"], instance_of, poses, K["depth"], K_mask,
                                   depth_maps, (args.mask_size, args.mask_size))
    votes, tstats = project_votes_to_vertices(
        gt3d["vertices"], poses, K["depth"], K_mask, depth_maps, rendered, len(inst_ids),
        args.depth_tolerance)
    assign = superpoint_majority(votes, gt3d["superpoints"])

    preds = [{"mask": assign == q, "label_id": int(gid // 1000), "confidence": 1.0}
             for q, gid in enumerate(inst_ids)]

    annotated = gt_ids > 0
    scored = annotated & (assign >= 0)
    stats = {
        "frames": len(stems), "instances": len(inst_ids),
        "purity": float((assign[scored] == instance_of[scored]).mean())
        if scored.any() else float("nan"),
        **tstats,
        "voted_vertex_frac": float((votes.sum(axis=1) > 0).mean()),
        "annotated_assigned_frac": float((assign[annotated] >= 0).mean())
        if annotated.any() else float("nan"),
        "seconds": round(time.time() - t0, 1),
    }
    return {"preds": preds, "stats": stats}


def main():
    args = build_argparser().parse_args()
    frames_root, gt_root = Path(args.frames_root), Path(args.gt_root)
    if args.scenes:
        scenes = list(args.scenes)
    else:
        scenes = sorted(d.name for d in gt_root.iterdir()
                        if d.is_dir() and (frames_root / d.name).is_dir())
        if args.num_scenes and len(scenes) > args.num_scenes:
            idx = np.linspace(0, len(scenes) - 1, args.num_scenes).round().astype(int)
            scenes = [scenes[i] for i in sorted(set(idx.tolist()))]
    if not scenes:
        raise SystemExit(f"no scenes found under both {gt_root} and {frames_root}")
    print(f"GT-projection oracle over {len(scenes)} scene(s) "
          f"(depth_tolerance={args.depth_tolerance}, mask grid {args.mask_size}²)")

    matches, per_scene, failed = {}, {}, []
    for i, scene in enumerate(scenes):
        gt3d = load_scene_3d_gt(gt_root, scene, args.tsv)
        preds: List[Dict] = []
        try:
            r = run_scene(scene, frames_root / scene, gt3d, args)
            preds, per_scene[scene] = r["preds"], r["stats"]
            print(f"[{i + 1}/{len(scenes)}] {scene}: {r['stats']}", flush=True)
        except Exception:  # noqa: BLE001
            failed.append(scene)
            print(f"[{i + 1}/{len(scenes)}] {scene} FAILED:\n"
                  + "".join(traceback.format_exc().splitlines(keepends=True)[-6:]), flush=True)
        gt2pred, pred2gt = assign_instances_for_scan(scene, preds, gt3d["gt_ids"],
                                                     MIN_REGION_SIZE)
        matches[scene] = {"gt": gt2pred, "pred": pred2gt}

    avgs = compute_averages(evaluate_matches(matches, OVERLAPS, MIN_REGION_SIZE), OVERLAPS)
    print("\nOracle — GT rendered through the transfer and scored as a prediction:")
    print(format_results(avgs))

    def agg(key):
        vals = [s[key] for s in per_scene.values() if not np.isnan(s.get(key, np.nan))]
        return float(np.mean(vals)) if vals else float("nan")

    purity, inlier = agg("purity"), agg("depth_inlier_frac")
    worst = min(((s["purity"], n) for n, s in per_scene.items()
                 if not np.isnan(s["purity"])), default=(float("nan"), "-"))
    print(f"\nround-trip purity   {purity:.4f}   (worst scene {worst[1]}: {worst[0]:.4f})")
    print(f"sensor-depth inlier {inlier:.4f}   of projections with a depth reading")
    print(f"visible vertices    {agg('visible_vertex_frac'):.4f}   "
          f"| assigned annotated {agg('annotated_assigned_frac'):.4f}")
    if failed:
        print(f"⚠ {len(failed)} scene(s) failed: " + ", ".join(failed))

    out_path = Path(args.out) if args.out else Path("eval3d_projection_oracle.json")
    out_path.write_text(json.dumps({
        "protocol": "GT-projection transfer oracle (docs/MASKDINO.md §9.9): the 3D GT "
                    "rendered into each view through the transfer's own projection and fed "
                    "back as predictions. Purity tests the pixel mapping; AP is the "
                    "protocol's ceiling on this frame budget, not a model result.",
        "args": {k: v for k, v in vars(args).items() if k != "scenes"},
        "num_scenes": len(scenes), "failed_scenes": failed,
        "purity": purity, "depth_inlier_frac": inlier,
        "visible_vertex_frac": agg("visible_vertex_frac"),
        "annotated_assigned_frac": agg("annotated_assigned_frac"),
        "results_18class": {k: (None if isinstance(v, float) and np.isnan(v) else v)
                            for k, v in avgs.items() if k != "classes"},
        "per_scene": per_scene,
    }, indent=2))
    print(f"✓ Wrote {out_path}")

    if failed:
        raise SystemExit(f"ORACLE FAILED: {len(failed)} scene(s) errored")
    if not (purity >= args.min_purity):
        raise SystemExit(
            f"ORACLE FAILED: round-trip purity {purity:.4f} < {args.min_purity}. The pixel "
            f"mapping or the depth test is wrong — fix it, do not tune around it.")
    print(f"\n✅ ORACLE PASSED: purity {purity:.4f} >= {args.min_purity}. The GT-projection "
          f"transfer reads our mask grid at the right pixel.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
