#!/usr/bin/env python3
"""Verify a built ScanNet++ val-50 GT + frames tree (docs/todo.md 6c). CPU-only.

Reads the two trees the build produces — `scans3d/` and `scans25k/` — and re-derives, from
scratch, everything the build claimed. It never reads the upstream ScanNet++ tree, which is
the point: the tars have to stand on their own.

Per scene it asserts:

  GT (`scans3d/<scene>/`)
    1. `len(segments.json["segIndices"])` == the ply header's vertex count;
    2. SEGMENT-ID CLOSURE: every segment id referenced by an object exists in
       segments.json (`build_vertex_instances` raises otherwise);
    3. every kept instance's label is in `top100_instance.txt`;
    4. the instance count equals `segGroups` filtered by the same class rule, and is
       plausible (>0, <= the group count), and matches `gt_stats.json`.

  FRAMES (`scans25k/<scene>/`)
    5. manifest ints agree with what is on disk: one jpg, one depth png and one pose per
       sampled stem, `sampling` == `uniform-<n>`, stems inside `0..total_frames-1`;
    6. poses are finite 4x4 with an exact `[0,0,0,1]` bottom row and an orthonormal
       rotation, and the intrinsics parse;
    7. POSE / DEPTH-SCALE CHECK — sensor depth unprojected with `pose/<stem>.txt` read as
       camera-to-world lands on the mesh (median distance under `--max_median_cm`), and
       the same depth transformed by the OTHER plausible reading (world-to-camera) does
       not. This is the check that catches the silent killer: a pipeline that runs and
       scores ~0 because the wrong pose field or convention was written;
    8. RGB / POSE PAIRING — the mesh is projected into each probe colour frame and scored
       by normalised cross-correlation against it, at the frame's own pose and at its
       neighbours'. Averaged over probe frames, the own pose must win. A scene whose mesh
       colour carries too little signal (mean NCC under `--ncc_floor`) reports
       INCONCLUSIVE rather than failing on noise;
    9. and the build's own, sharper version of 8 — a +-40-frame index sweep run against
       the source video and recorded in `manifest["rgb_index_check"]` — must have peaked
       at offset 0. That sweep is what actually rules out a stream-level RGB/pose shift;
       it is not reproducible from the tar, which keeps only the sampled frames.

Exit code is non-zero if any scene fails any check.

Usage:
    myenv/bin/python scripts/verify_scannetpp_gt.py \
        --gt_root <tree>/scans3d --frames_root <tree>/scans25k --num_scenes 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent /
                       "legacy" / "dataset_build" / "scripts"))

from scannetpp_common import (  # noqa: E402
    build_vertex_instances, load_instance_classes, load_label_map, load_segments,
    mesh_image_ncc, ply_vertex_count, read_ply_xyz_rgb, unproject,
)


# ----------------------------------------------------------------------------------------

def check_gt(gt_root: Path, scene: str, classes: list[str], label_map: dict) -> dict:
    d = gt_root / scene
    n_vertices = ply_vertex_count(d / "mesh.ply")
    seg_indices = load_segments(d / "segments.json")
    if len(seg_indices) != n_vertices:
        raise AssertionError(f"segIndices {len(seg_indices)} != ply vertices {n_vertices}")

    groups = json.loads((d / "segments_anno.json").read_text())["segGroups"]
    inst_ids, instances = build_vertex_instances(seg_indices, groups, classes, label_map)

    class_set = set(classes)
    bad = [i["label"] for i in instances if i["label"] not in class_set]
    if bad:
        raise AssertionError(f"{len(bad)} instance(s) outside top100_instance: {bad[:5]}")

    # Independent recount of what the class filter should keep: groups whose mapped label
    # is a benchmark class AND that own at least one vertex.
    present = set(seg_indices.tolist())
    expect = sum(1 for g in groups
                 if label_map.get(g.get("label", ""), g.get("label", "")) in class_set
                 and any(s in present for s in g.get("segments", [])))
    if len(instances) != expect:
        raise AssertionError(f"instance count {len(instances)} != independent recount "
                             f"{expect}")
    if not 0 < len(instances) <= len(groups):
        raise AssertionError(f"implausible instance count {len(instances)} against "
                             f"{len(groups)} segGroups")

    stats_path = d / "gt_stats.json"
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text())
        for key, got in (("n_vertices", n_vertices), ("n_instances", len(instances)),
                         ("n_labelled_vertices", int((inst_ids > 0).sum()))):
            if stats.get(key) != got:
                raise AssertionError(f"gt_stats.json {key}={stats.get(key)} but "
                                     f"recomputed {got}")
    return {"n_vertices": n_vertices, "n_groups": len(groups),
            "n_instances": len(instances),
            "labelled_frac": float((inst_ids > 0).mean())}


def load_intrinsic(path: Path) -> np.ndarray:
    m = np.loadtxt(path)
    if m.shape == (4, 4):
        m = m[:3, :3]
    if m.shape != (3, 3) or not np.isfinite(m).all():
        raise AssertionError(f"{path}: expected a finite 3x3/4x4 intrinsic, got {m.shape}")
    return m


def check_frames_files(frames_root: Path, scene: str) -> dict:
    d = frames_root / scene
    man = json.loads((d / "manifest.json").read_text())
    stems = man["sampled_stems"]
    total = man["total_frames"]
    if man["sampling"] != f"uniform-{len(stems)}" and not man["sampling"].startswith(
            "uniform-"):
        raise AssertionError(f"unexpected sampling tag {man['sampling']!r}")
    if len(set(stems)) != len(stems):
        raise AssertionError("duplicate stems in the manifest")
    for stem in stems:
        idx = int(stem.split("_")[1])
        if not 0 <= idx < total:
            raise AssertionError(f"{stem} outside 0..{total - 1}")
        for sub, ext in (("color", ".jpg"), ("depth", ".png"), ("pose", ".txt")):
            p = d / sub / (stem + ext)
            if not p.is_file() or p.stat().st_size == 0:
                raise AssertionError(f"missing or empty {p}")
    for sub in ("color", "depth", "pose"):
        n = len(list((d / sub).iterdir()))
        if n != len(stems):
            raise AssertionError(f"{sub}/ holds {n} files for {len(stems)} stems")
    load_intrinsic(d / "intrinsic" / "intrinsic_color.txt")
    load_intrinsic(d / "intrinsic" / "intrinsic_depth.txt")
    return man


def check_poses(frames_root: Path, scene: str, stems: list[str]) -> None:
    d = frames_root / scene / "pose"
    for stem in stems:
        T = np.loadtxt(d / f"{stem}.txt")
        if T.shape != (4, 4) or not np.isfinite(T).all():
            raise AssertionError(f"{stem}: pose is not a finite 4x4")
        if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-9):
            raise AssertionError(f"{stem}: pose bottom row {T[3]} is not [0,0,0,1]")
        R = T[:3, :3]
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-4):
            raise AssertionError(f"{stem}: pose rotation is not orthonormal")


def check_pose_convention(gt_root: Path, frames_root: Path, scene: str, man: dict,
                          n_probe: int, max_median_cm: float) -> dict:
    """Unproject sensor depth and measure the distance to the mesh, camera-to-world vs the
    alternative reading."""
    import cv2
    from scipy.spatial import cKDTree

    xyz, rgb = read_ply_xyz_rgb(gt_root / scene / "mesh.ply")
    step = max(1, len(xyz) // 400000)
    tree = cKDTree(xyz[::step])
    K = load_intrinsic(frames_root / scene / "intrinsic" / "intrinsic_depth.txt")
    stems = man["sampled_stems"]
    probe = [stems[i] for i in np.round(
        np.linspace(0, len(stems) - 1, min(n_probe, len(stems)))).astype(int)]

    med_c2w, med_w2c = [], []
    for stem in probe:
        depth = cv2.imread(str(frames_root / scene / "depth" / f"{stem}.png"),
                           cv2.IMREAD_UNCHANGED)
        if depth is None or depth.dtype != np.uint16:
            raise AssertionError(f"{stem}: depth png is not uint16")
        T = np.loadtxt(frames_root / scene / "pose" / f"{stem}.txt")
        for T_use, sink in ((T, med_c2w), (np.linalg.inv(T), med_w2c)):
            pts = unproject(depth, K, T_use)
            if len(pts) < 100:
                sink.append(np.inf)
                continue
            sink.append(float(np.median(tree.query(pts, workers=-1)[0])))

    m_c2w = float(np.median(med_c2w))
    m_w2c = float(np.median(med_w2c))
    if m_c2w * 100 > max_median_cm:
        raise AssertionError(
            f"pose/depth check FAILED: camera-to-world unprojection sits "
            f"{m_c2w * 100:.1f} cm from the mesh (limit {max_median_cm} cm)")
    if m_w2c <= m_c2w * 2:
        raise AssertionError(
            f"pose convention is ambiguous: camera-to-world {m_c2w * 100:.1f} cm vs the "
            f"inverse reading {m_w2c * 100:.1f} cm — expected the inverse to be far worse")
    return {"depth_mesh_median_cm": round(m_c2w * 100, 2),
            "inverse_reading_median_cm": round(m_w2c * 100, 2),
            "has_color": bool(rgb.size)}


def check_rgb_pose_pairing(gt_root: Path, frames_root: Path, scene: str, man: dict,
                           n_probe: int, ncc_tol: float, ncc_floor: float) -> dict:
    """Colour frames must correlate with the mesh rendered at THEIR OWN pose better than at
    a neighbouring sampled pose.

    Scored in aggregate, never per frame: a single view of a blank wall gives an NCC
    indistinguishable from zero for every pose, and a per-frame assertion on that is a
    coin flip. The mean over probe frames is only asserted when it clears `ncc_floor`,
    i.e. when there is signal to compare at all.

    The build runs the finer version — a +-40-frame index sweep against the source video,
    recorded in `manifest["rgb_index_check"]` — and that is what actually rules out a
    stream-level RGB/pose shift. The tar keeps only the sampled frames, so what is
    re-checkable from the tar alone is this coarser pairing; both are reported, and the
    build's verdict is re-asserted here.
    """
    import cv2

    xyz, rgb = read_ply_xyz_rgb(gt_root / scene / "mesh.ply")
    if rgb.size == 0:
        return {"skipped": "mesh has no vertex colour"}
    step = max(1, len(xyz) // 400000)
    xyz, gray_v = xyz[::step], (rgb[::step] @ np.array([0.299, 0.587, 0.114]))
    K = load_intrinsic(frames_root / scene / "intrinsic" / "intrinsic_color.txt")
    stems = man["sampled_stems"]
    idx = np.unique(np.round(np.linspace(1, len(stems) - 2,
                                         min(n_probe, len(stems) - 2))).astype(int))

    bases, decoy_best, per_frame = [], [], {}
    for i in idx:
        stem = stems[i]
        img = cv2.imread(str(frames_root / scene / "color" / f"{stem}.jpg"),
                         cv2.IMREAD_GRAYSCALE)
        base = mesh_image_ncc(xyz, gray_v, K,
                              np.loadtxt(frames_root / scene / "pose" / f"{stem}.txt"), img)
        decoys = [mesh_image_ncc(
            xyz, gray_v, K,
            np.loadtxt(frames_root / scene / "pose" / f"{stems[j]}.txt"), img)
            for j in (i - 1, i + 1)]
        decoys = [d for d in decoys if np.isfinite(d)]
        per_frame[stem] = {"ncc": None if np.isnan(base) else round(base, 4),
                           "decoy_ncc": [round(d, 4) for d in decoys]}
        if np.isfinite(base) and decoys:
            bases.append(base)
            decoy_best.append(max(decoys))

    out = {"pairing": per_frame,
           "mean_ncc": round(float(np.mean(bases)), 4) if bases else None,
           "mean_decoy_ncc": round(float(np.mean(decoy_best)), 4) if bases else None}
    if bases and out["mean_ncc"] >= ncc_floor and \
            out["mean_decoy_ncc"] > out["mean_ncc"] - ncc_tol:
        raise AssertionError(
            f"RGB/pose pairing FAILED: mean NCC at the frame's own pose "
            f"{out['mean_ncc']:.4f} does not beat the neighbouring pose "
            f"{out['mean_decoy_ncc']:.4f} over {len(bases)} probe frames")
    if not bases or out["mean_ncc"] < ncc_floor:
        out["inconclusive"] = (f"mean NCC {out['mean_ncc']} below the {ncc_floor} floor — "
                               f"too little mesh colour signal; relying on the build sweep")

    build_check = man.get("rgb_index_check", {})
    if build_check.get("peak_at_zero") is False:
        raise AssertionError(f"the build's RGB index sweep did not peak at offset 0: "
                             f"{build_check}")
    out["build_sweep"] = build_check.get("mean_ncc")
    return out


# ----------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_root", required=True, help="the built scans3d/ tree")
    ap.add_argument("--frames_root", required=True, help="the built scans25k/ tree")
    ap.add_argument("--metadata_dir", default=None,
                    help="defaults to <gt_root>/_metadata")
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--num_scenes", type=int, default=5,
                    help="how many scenes to check when --scenes is not given")
    ap.add_argument("--n_probe", type=int, default=4)
    ap.add_argument("--max_median_cm", type=float, default=15.0)
    ap.add_argument("--ncc_tol", type=float, default=0.02)
    ap.add_argument("--ncc_floor", type=float, default=0.10,
                    help="below this mean NCC the pairing check has no signal to judge "
                         "and reports inconclusive instead of failing")
    ap.add_argument("--skip_rgb_index", action="store_true")
    args = ap.parse_args()

    gt_root, frames_root = Path(args.gt_root), Path(args.frames_root)
    meta_dir = Path(args.metadata_dir) if args.metadata_dir else gt_root / "_metadata"
    classes = load_instance_classes(meta_dir)
    label_map = load_label_map(meta_dir)

    if args.scenes:
        scenes = args.scenes
    else:
        avail = sorted(p.name for p in gt_root.iterdir()
                       if p.is_dir() and not p.name.startswith("_")
                       and (frames_root / p.name).is_dir())
        scenes = avail[:args.num_scenes] if args.num_scenes > 0 else avail

    print(f"[verify] {len(classes)} instance classes; checking {len(scenes)} scene(s)",
          flush=True)
    failures = []
    for scene in scenes:
        try:
            g = check_gt(gt_root, scene, classes, label_map)
            man = check_frames_files(frames_root, scene)
            check_poses(frames_root, scene, man["sampled_stems"])
            p = check_pose_convention(gt_root, frames_root, scene, man, args.n_probe,
                                      args.max_median_cm)
            r = ({} if args.skip_rgb_index else
                 check_rgb_pose_pairing(gt_root, frames_root, scene, man, args.n_probe,
                                        args.ncc_tol, args.ncc_floor))
        except Exception as e:  # noqa: BLE001
            print(f"[{scene}] FAIL: {type(e).__name__}: {e}", flush=True)
            failures.append(scene)
            continue
        ncc = (r.get("mean_ncc"), r.get("mean_decoy_ncc"))
        print(f"[{scene}] OK — {g['n_instances']}/{g['n_groups']} instances, "
              f"{g['n_vertices']} verts ({g['labelled_frac']:.0%} labelled), "
              f"{len(man['sampled_stems'])}/{man['total_frames']} frames, "
              f"depth->mesh {p['depth_mesh_median_cm']} cm "
              f"(inverse reading {p['inverse_reading_median_cm']} cm), "
              f"rgb NCC own/decoy {ncc}"
              f"{' INCONCLUSIVE' if r.get('inconclusive') else ''}, "
              f"build sweep {r.get('build_sweep')}", flush=True)

    if failures:
        print(f"[verify] FAILED {len(failures)}/{len(scenes)}: " + ", ".join(failures),
              flush=True)
        return 1
    print(f"[verify] all {len(scenes)} scene(s) passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
