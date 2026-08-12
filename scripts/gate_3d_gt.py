#!/usr/bin/env python3
"""
The licence gate for a dataset's 3D GT (docs/MASKDINO.md §9.2, docs/todo.md 6d). CPU-only,
no checkpoint, no GPU.

**No dataset ships a number until this passes.** It is the same gate that licensed the
ScanNet evaluator: feed a dataset's own GT back through the pipeline as if it were our
predictions and require the official evaluator to answer exactly

    AP 1.000 / AP50 1.000 / AP25 1.000

Anything less means the GT construction, the label encoding or the instance indexing is
wrong, and no number computed from it is quotable. It is a whole-pipeline check, not a unit
test: it runs the real adapter (`train/datasets3d.py`), the real evaluator
(`train/benchmark3d.py`) and the real class-agnostic collapse over the real tars.

Two optional checks, both about the OTHER way a dataset silently scores ~0 — a wrong pose
convention or a wrong depth scale, which no AP number can distinguish from a bad model:

  `--frames_root`         unprojects each probe frame's sensor depth with the adapter's
                          pose (read as camera-to-world) and intrinsics, and requires the
                          scene's MEDIAN probe distance to the GT mesh to stay under
                          `--max_median_cm` (a single drifted frame is reported, not failed —
                          calibrated on ScanNet, where 3 of 312 val scenes have one probe up
                          to 64.7 cm out while no scene median exceeds 9.6 cm).
                          This is what pinned Replica's depth units to millimetres
                          (0.55 cm against the mesh; the NICE-SLAM 6553.5 constant lands
                          65-91 cm away) and validated its FALLBACK intrinsics.
  `--report_superpoints`  reports how many superpoints a scene has and their purity against
                          the GT instances — the number quoted in `train/replica3d.py` for
                          why Replica's own planar `preseg` is NOT used as the vote's
                          over-segmentation.

    myenv/bin/python scripts/gate_3d_gt.py --dataset replica \
        --gt_root $TMPDIR/scans3d --frames_root $TMPDIR/scans25k

Exit code is non-zero if any scene fails any check.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.benchmark3d import (ID_TO_LABEL, assign_instances_for_scan,
                               collapse_gt_to_class_agnostic,
                               collapse_preds_to_class_agnostic, compute_averages,
                               evaluate_matches, MIN_REGION_SIZE, OVERLAPS)
from train.datasets3d import DATASET_NAMES, DEFAULT_DATASET, get_dataset
from train.scannet3d import DEFAULT_TSV

PERFECT_KEYS = ("all_ap", "all_ap_50%", "all_ap_25%")


def gt_as_predictions(gt_ids: np.ndarray):
    """Every GT instance, verbatim, as a confidence-1.0 prediction of its own class."""
    return [{"mask": gt_ids == inst, "label_id": int(inst // 1000), "confidence": 1.0}
            for inst in np.unique(gt_ids) if inst != 0]


def probe_depth_against_mesh(ds, scene_dir: Path, vertices: np.ndarray, num_probes: int,
                             stride: int = 200):
    """
    Median distance (cm) from unprojected sensor depth to the mesh, over `num_probes` frames.

    The check that catches a wrong pose convention or a wrong depth scale: with the right
    ones the sensor's own surface lands ON the mesh; with either wrong it lands tens of
    centimetres to hundreds of metres away, and every downstream AP is quietly zero.
    """
    from scipy.spatial import cKDTree

    stems = ds.sample_frames(scene_dir, num_probes, require_depth=True)
    K = ds.load_intrinsics(scene_dir)["depth"]
    depth = ds.load_depth(scene_dir, stems)
    poses = ds.load_poses(scene_dir)
    tree = cKDTree(vertices)
    medians = []
    for stem, dmap in zip(stems, depth):
        h, w = dmap.shape
        u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        valid = dmap > 0
        z = dmap[valid].astype(np.float64)
        x = (u[valid] - K[0, 2]) / K[0, 0] * z
        y = (v[valid] - K[1, 2]) / K[1, 1] * z
        cam = np.stack([x, y, z], axis=1)[::stride]
        if len(cam) == 0:
            continue
        P = poses[stem]
        world = cam @ P[:3, :3].T + P[:3, 3]
        medians.append(float(np.median(tree.query(world, k=1)[0])) * 100.0)
    return medians


def superpoint_purity(superpoints: np.ndarray, gt_ids: np.ndarray):
    """(n superpoints, vertex-weighted purity against the GT instances) over labelled verts."""
    labelled = gt_ids > 0
    sp = superpoints[labelled]
    gt = gt_ids[labelled]
    if len(sp) == 0:
        return int(len(np.unique(superpoints))), float("nan")
    pairs, counts = np.unique(np.stack([sp, gt], axis=1), axis=0, return_counts=True)
    starts = np.nonzero(np.r_[True, pairs[1:, 0] != pairs[:-1, 0]])[0]
    best = np.maximum.reduceat(counts, starts)
    total = np.add.reduceat(counts, starts)
    return int(len(np.unique(superpoints))), float((best.sum() / total.sum()))


def main() -> int:
    ap = argparse.ArgumentParser(description="GT-as-predictions gate for a 3D dataset")
    ap.add_argument("--dataset", choices=DATASET_NAMES, default=DEFAULT_DATASET)
    ap.add_argument("--gt_root", required=True)
    ap.add_argument("--frames_root", default=None,
                    help="enables the pose/depth-scale geometry check")
    ap.add_argument("--tsv", default=DEFAULT_TSV)
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--num_scenes", type=int, default=None,
                    help="cap the number of scenes (default: all)")
    ap.add_argument("--num_probes", type=int, default=4,
                    help="frames per scene for the geometry check")
    ap.add_argument("--max_median_cm", type=float, default=15.0)
    ap.add_argument("--report_superpoints", action="store_true")
    ap.add_argument("--out", default=None, help="write the report as json")
    args = ap.parse_args()

    ds = get_dataset(args.dataset)
    gt_root = Path(args.gt_root)
    scenes = args.scenes or sorted(d.name for d in gt_root.iterdir()
                                   if d.is_dir() and not d.name.startswith("_"))
    if args.num_scenes:
        scenes = scenes[:args.num_scenes]
    if not scenes:
        raise SystemExit(f"no scenes under {gt_root}")
    print(f"Gate: {ds.name} ({ds.note}) — {len(scenes)} scene(s) from {gt_root}")

    matches, matches_ca, per_scene, problems, outliers = {}, {}, {}, [], []
    for i, scene in enumerate(scenes):
        t0 = time.time()
        gt3d = ds.load_scene_3d_gt(gt_root, scene, args.tsv)
        gt_ids = gt3d["gt_ids"]
        preds = gt_as_predictions(gt_ids)
        if not preds:
            problems.append(f"{scene}: no GT instances at all")
        if ds.class_aware:
            g, p = assign_instances_for_scan(scene, preds, gt_ids, MIN_REGION_SIZE)
            matches[scene] = {"gt": g, "pred": p}
        g, p = assign_instances_for_scan(scene, collapse_preds_to_class_agnostic(preds),
                                         collapse_gt_to_class_agnostic(gt_ids),
                                         MIN_REGION_SIZE)
        matches_ca[scene] = {"gt": g, "pred": p}

        stats = {"vertices": int(len(gt_ids)), "instances": len(preds),
                 # what the evaluator actually scores: on scannetv2 the GT carries every
                 # nyu40 class and only 18 are benchmark ones, so `instances` overstates it
                 "evaluated_instances": sum(1 for p in preds
                                            if int(p["label_id"]) in ID_TO_LABEL),
                 "labelled_vertex_frac": float((gt_ids > 0).mean()),
                 **{k: v for k, v in gt3d.get("meta", {}).items() if k != "labels"}}
        if args.report_superpoints:
            n_sp, purity = superpoint_purity(gt3d["superpoints"], gt_ids)
            stats["superpoints"] = n_sp
            stats["superpoint_purity"] = round(purity, 4)
            if ds.alt_superpoints is not None:
                n_alt, purity_alt = superpoint_purity(ds.alt_superpoints(gt_root, scene),
                                                      gt_ids)
                stats["alt_superpoints"] = f"{ds.alt_superpoints_name}: {n_alt}"
                stats["alt_superpoint_purity"] = round(purity_alt, 4)
        if args.frames_root:
            scene_dir = Path(args.frames_root) / scene
            if not scene_dir.is_dir():
                problems.append(f"{scene}: no frames directory {scene_dir}")
            else:
                medians = probe_depth_against_mesh(ds, scene_dir, gt3d["vertices"],
                                                   args.num_probes)
                stats["depth_mesh_median_cm"] = [round(m, 2) for m in medians]
                # The scene's value is the MEDIAN over probe frames, not the worst: a wrong
                # pose convention or depth scale puts EVERY frame metres away, while a real
                # scan can carry one bad frame (3 of ScanNet's 312 val scenes do — the worst
                # single probe is 64.7 cm, yet no scene's median exceeds 9.6 cm). Failing on
                # the worst frame would fail the reference dataset, which is how we know the
                # rule would be wrong.
                scene_cm = float(np.median(medians)) if medians else float("inf")
                stats["depth_mesh_scene_cm"] = round(scene_cm, 2)
                if scene_cm > args.max_median_cm:
                    problems.append(f"{scene}: sensor depth lands {scene_cm:.1f} cm from "
                                    f"the mesh (> {args.max_median_cm} cm) — wrong pose "
                                    f"convention, intrinsics or depth scale")
                elif medians and max(medians) > args.max_median_cm:
                    outliers.append(f"{scene} (worst probe {max(medians):.1f} cm)")
        stats["seconds"] = round(time.time() - t0, 1)
        per_scene[scene] = stats
        print(f"[{i + 1}/{len(scenes)}] {scene}: {stats}", flush=True)

    report = {"dataset": ds.name, "num_scenes": len(scenes), "per_scene": per_scene,
              "depth_outlier_scenes": outliers}
    if outliers:
        print(f"\nnote: {len(outliers)}/{len(scenes)} scene(s) carry a single probe frame "
              f"above {args.max_median_cm} cm while their scene median is fine — scan drift, "
              f"not a convention error: {', '.join(outliers[:6])}"
              + (" ..." if len(outliers) > 6 else ""))
    for tag, m in (("class_aware", matches), ("class_agnostic", matches_ca)):
        if not m:
            continue
        avgs = compute_averages(evaluate_matches(m, OVERLAPS, MIN_REGION_SIZE), OVERLAPS)
        got = {k: float(avgs[k]) for k in PERFECT_KEYS}
        report[tag] = got
        ok = all(abs(got[k] - 1.0) < 1e-9 for k in PERFECT_KEYS)
        print(f"{tag}: AP {got['all_ap']:.3f} / AP50 {got['all_ap_50%']:.3f} / "
              f"AP25 {got['all_ap_25%']:.3f}  {'OK' if ok else 'FAIL'}")
        if not ok:
            problems.append(f"{tag} GT-as-predictions is not 1.000/1.000/1.000: {got}")

    report["problems"] = problems
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"✓ Wrote {args.out}")
    if problems:
        print(f"\n✗ GATE FAILED ({len(problems)} problem(s)):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"\n✓ GATE PASSED — {ds.name} is licensed to ship numbers "
          f"(docs/MASKDINO.md §9.2)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
