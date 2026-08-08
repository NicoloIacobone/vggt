"""ScanNet++ v2 iphone frames for a scene list, in the frames25k layout (docs/todo.md 6c).

Per scene, 50 frames sampled uniformly over the WHOLE iphone sequence (the ScanNet++
sequences run 3.4k-26k frames; the 3D eval needs coverage of the whole scan, exactly as
`scannet_frames25k_val312.tar.zst` does — docs/DATASET.md §2):

    scans25k/<scene>/color/<stem>.jpg              1920x1440, decoded from iphone/rgb.mkv
    scans25k/<scene>/depth/<stem>.png              256x192 uint16 MILLIMETRES
    scans25k/<scene>/pose/<stem>.txt               4x4 CAMERA-TO-WORLD, mesh frame
    scans25k/<scene>/intrinsic/intrinsic_color.txt 3x3, colour resolution
    scans25k/<scene>/intrinsic/intrinsic_depth.txt 3x3, depth resolution
    scans25k/<scene>/intrinsics_color.txt          identical, at the ScanNet frames25k path
    scans25k/<scene>/intrinsics_depth.txt          identical, at the ScanNet frames25k path
    scans25k/<scene>/manifest.json

`<stem>` is the upstream `frame_NNNNNN` name, so a stem always names its own index in the
source streams and an off-by-one is visible by inspection.

BOTH intrinsic spellings are written on purpose. `intrinsic/intrinsic_*.txt` is what the
ScanNet++ tar was specified to carry; `intrinsics_*.txt` at the scene root is what the
existing ScanNet loader (`train/scannet3d.py::load_frames25k_intrinsics`) actually reads.
They are byte-identical copies — two 100-byte files buy the evaluator a free path.

THE CONVENTIONS, and why they are these (`scannetpp_common` has the long form):
  - pose = `aligned_pose`, camera-to-world. `pose` is the raw ARKit trajectory and is ~85 m
    away from the mesh.
  - decoded video frame N is `frame_{N:06d}`; asserted per scene by frame count and, via
    the geometry check below, by content.
  - depth.bin = `<4-byte LE size><LZ4 block>` per frame, uint16 millimetres, 192x256.
  - the colour intrinsic VARIES per frame (iPhone autofocus, ~1.5 % of fx). The scene-level
    files hold the MEDIAN over the sampled frames; the exact per-frame intrinsics are in
    `manifest.json["intrinsic_color_per_frame"]` for anyone who needs them.

GEOMETRY SELF-CHECK, per scene, before the scene is marked complete: sampled depth maps are
unprojected with their pose and matched against the mesh vertices. Median point-to-vertex
distance must stay under `--max_median_cm` (default 15 cm; the observed value is 2-4 cm).
This is the guard against the failure mode that would otherwise be invisible — a pipeline
that runs and scores ~0 because pose, depth scale or frame index is wrong. The measured
value is recorded in the manifest.

Resumable: a scene with a `.complete` marker is skipped.

Usage (from the vggt repo):
    myenv/bin/python legacy/dataset_build/scripts/build_scannetpp_frames.py \
        --src_root /cluster/work/igp_psr/nedela/scannetpp_data \
        --out_root $TMPDIR/build/scans25k \
        --scene_list /cluster/work/igp_psr/nedela/scannetpp_data/splits/nvs_sem_val.txt
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scannetpp_common import (  # noqa: E402
    DEPTH_H, DEPTH_W, count_depth_frames, drop_excluded, frame_stems, mesh_image_ncc,
    read_depth_frames, read_ply_xyz_rgb, sample_indices, scale_intrinsic, select_scenes,
    unproject,
)


def write_matrix(path: Path, mat: np.ndarray) -> None:
    np.savetxt(path, np.asarray(mat, dtype=np.float64), fmt="%.18e")


def decode_color(video: Path, indices, out_dir: Path, quality: int) -> dict:
    """Decode and JPEG-write the wanted frames. Returns {index: (h, w)}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    shapes = {}
    try:
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"{video}: decode failed at frame {i}")
            path = out_dir / f"frame_{int(i):06d}.jpg"
            if not cv2.imwrite(str(path), frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), quality]):
                raise RuntimeError(f"cannot write {path}")
            shapes[int(i)] = frame.shape[:2]
    finally:
        cap.release()
    return shapes


def geometry_check(mesh_tree: cKDTree, depths: dict, poses: dict, K_depth: np.ndarray,
                   n_probe: int) -> dict:
    """Unproject a few sampled depth maps and measure the distance to the mesh."""
    idx = sorted(depths)
    probe = [idx[i] for i in np.round(
        np.linspace(0, len(idx) - 1, min(n_probe, len(idx)))).astype(int)]
    medians, cov = [], []
    for i in probe:
        pts = unproject(depths[i], K_depth, poses[i])
        if len(pts) < 100:
            continue
        dist, _ = mesh_tree.query(pts, workers=-1)
        medians.append(float(np.median(dist)))
        cov.append(float(np.mean(dist < 0.05)))
    if not medians:
        raise RuntimeError("geometry check: no probe frame had usable depth")
    return {"probe_frames": probe,
            "depth_mesh_median_cm": round(float(np.median(medians)) * 100, 3),
            "depth_mesh_worst_median_cm": round(float(np.max(medians)) * 100, 3),
            "depth_mesh_frac_within_5cm": round(float(np.mean(cov)), 4)}


def rgb_index_check(video: Path, xyz: np.ndarray, gray_v: np.ndarray, K_color: np.ndarray,
                    pose_json: dict, indices, n_total: int, offsets, n_probe: int) -> dict:
    """Does decoded video frame N really belong to pose `frame_{N:06d}`?

    For a few probe indices, the mesh is rendered with the pose of index i and scored by
    NCC against the video frames at i + offset. The mean NCC over the probes must be
    highest at offset 0 — a whole-sequence shift of the RGB stream against the poses,
    which nothing else here would notice, moves the peak.

    Aggregated over probes on purpose: a single low-texture view is noisy, the mean is not.
    """
    idx = [int(i) for i in indices]
    probe = [idx[i] for i in np.round(
        np.linspace(0, len(idx) - 1, min(n_probe, len(idx)))).astype(int)]
    probe = [i for i in probe if all(0 <= i + o < n_total for o in offsets)]
    if not probe:
        return {"skipped": "sequence too short for the offset sweep"}

    cap = cv2.VideoCapture(str(video))
    try:
        scores: dict[int, list[float]] = {o: [] for o in offsets}
        for i in probe:
            T = np.asarray(pose_json[f"frame_{i:06d}"]["aligned_pose"], dtype=np.float64)
            for o in offsets:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i + o)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                v = mesh_image_ncc(xyz, gray_v, K_color, T, g)
                if np.isfinite(v):
                    scores[o].append(v)
    finally:
        cap.release()

    mean = {o: (float(np.mean(v)) if v else float("nan")) for o, v in scores.items()}
    if not np.isfinite(mean.get(0, np.nan)):
        return {"skipped": "no probe frame had enough geometry in frustum",
                "mean_ncc": mean}
    worse = {o: m for o, m in mean.items()
             if o != 0 and np.isfinite(m) and m >= mean[0]}
    return {"probe_frames": probe,
            "mean_ncc": {str(o): round(m, 4) for o, m in mean.items()},
            "peak_at_zero": not worse,
            "beaten_by": {str(o): round(m, 4) for o, m in worse.items()}}


def build_scene(src_root: Path, out_root: Path, scene: str, args) -> dict:
    src = src_root / "data" / scene
    dst = out_root / scene
    dst.mkdir(parents=True, exist_ok=True)

    pose_json = json.loads((src / "iphone" / "pose_intrinsic_imu.json").read_text())
    stems = frame_stems(pose_json)
    n_total = len(stems)
    expected = [f"frame_{i:06d}" for i in range(n_total)]
    if stems != expected:
        raise ValueError(f"{scene}: pose json frame keys are not a dense 0..N-1 range")

    # Stream lengths must agree, or "decoded frame N is frame_N" is not even well posed.
    cap = cv2.VideoCapture(str(src / "iphone" / "rgb.mkv"))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_wh = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    n_depth = count_depth_frames(src / "iphone" / "depth.bin")
    if not (n_total == n_video == n_depth):
        raise ValueError(f"{scene}: stream length mismatch — pose json {n_total}, "
                         f"video {n_video}, depth.bin {n_depth}")

    indices = sample_indices(n_total, args.num_frames)
    sampled_stems = [f"frame_{int(i):06d}" for i in indices]

    shapes = decode_color(src / "iphone" / "rgb.mkv", indices, dst / "color",
                          args.jpeg_quality)
    bad = {i: s for i, s in shapes.items() if (s[1], s[0]) != vid_wh}
    if bad:
        raise ValueError(f"{scene}: decoded frames disagree with the video header: {bad}")

    depths = read_depth_frames(src / "iphone" / "depth.bin", indices)
    if set(depths) != set(int(i) for i in indices):
        raise ValueError(f"{scene}: depth.bin short — got {len(depths)} of {len(indices)}")
    (dst / "depth").mkdir(parents=True, exist_ok=True)
    for i, d in depths.items():
        if not cv2.imwrite(str(dst / "depth" / f"frame_{i:06d}.png"), d):
            raise RuntimeError(f"cannot write depth for {scene}/frame_{i:06d}")

    (dst / "pose").mkdir(parents=True, exist_ok=True)
    poses, K_per_frame = {}, {}
    for i in indices:
        e = pose_json[f"frame_{int(i):06d}"]
        T = np.asarray(e["aligned_pose"], dtype=np.float64)
        if T.shape != (4, 4) or not np.isfinite(T).all():
            raise ValueError(f"{scene}/frame_{int(i):06d}: bad aligned_pose {T.shape}")
        # Upstream stores T[3,3] as ~0.99999993; normalise so the matrix is exactly affine.
        T = T / T[3, 3]
        T[3, :] = [0.0, 0.0, 0.0, 1.0]
        write_matrix(dst / "pose" / f"frame_{int(i):06d}.txt", T)
        poses[int(i)] = T
        K_per_frame[int(i)] = np.asarray(e["intrinsic"], dtype=np.float64)

    Ks = np.stack([K_per_frame[int(i)] for i in indices])
    K_color = np.median(Ks, axis=0)
    K_depth = scale_intrinsic(K_color, vid_wh, (DEPTH_W, DEPTH_H))
    (dst / "intrinsic").mkdir(parents=True, exist_ok=True)
    write_matrix(dst / "intrinsic" / "intrinsic_color.txt", K_color)
    write_matrix(dst / "intrinsic" / "intrinsic_depth.txt", K_depth)
    shutil.copyfile(dst / "intrinsic" / "intrinsic_color.txt", dst / "intrinsics_color.txt")
    shutil.copyfile(dst / "intrinsic" / "intrinsic_depth.txt", dst / "intrinsics_depth.txt")

    mesh_xyz, mesh_rgb = read_ply_xyz_rgb(src / "scans" / "mesh_aligned_0.05.ply")
    step = (len(mesh_xyz) // args.mesh_subsample + 1) if (
        args.mesh_subsample and len(mesh_xyz) > args.mesh_subsample) else 1
    probe_xyz = mesh_xyz[::step]
    geom = geometry_check(cKDTree(probe_xyz), depths, poses, K_depth, args.n_probe)
    if geom["depth_mesh_median_cm"] > args.max_median_cm:
        raise ValueError(
            f"{scene}: GEOMETRY CHECK FAILED — unprojected depth sits "
            f"{geom['depth_mesh_median_cm']:.1f} cm from the mesh (limit "
            f"{args.max_median_cm} cm). Pose convention, depth scale or frame indexing "
            f"is wrong; refusing to ship this scene.")

    gray_v = mesh_rgb[::step] @ np.array([0.299, 0.587, 0.114])
    rgb_idx = rgb_index_check(src / "iphone" / "rgb.mkv", probe_xyz, gray_v, K_color,
                              pose_json, indices, n_total,
                              tuple(args.index_offsets), args.n_probe)
    if not args.skip_rgb_index_check and rgb_idx.get("peak_at_zero") is False:
        raise ValueError(
            f"{scene}: RGB INDEX CHECK FAILED — the mesh rendered at pose N correlates "
            f"better with video frames at offset(s) {rgb_idx['beaten_by']} than at 0 "
            f"(mean NCC {rgb_idx['mean_ncc']}). The colour stream and the poses are not "
            f"index-aligned; refusing to ship this scene.")

    manifest = {
        "scene": scene,
        "total_frames": n_total,
        "sampled_stems": sampled_stems,
        "sampling": f"uniform-{args.num_frames}",
        # Everything below is extra context; the four keys above are the contract.
        "color_wh": list(vid_wh),
        "depth_wh": [DEPTH_W, DEPTH_H],
        "depth_units": "uint16 millimetres",
        "pose_convention": "aligned_pose, camera-to-world, mesh_aligned_0.05 frame",
        "intrinsic_color_spread_fx": [float(Ks[:, 0, 0].min()), float(Ks[:, 0, 0].max())],
        "intrinsic_color_per_frame": {f"frame_{int(i):06d}": K_per_frame[int(i)].tolist()
                                      for i in indices},
        "geometry_check": geom,
        "rgb_index_check": rgb_idx,
        "source": "ScanNet++ v2 iphone (rgb.mkv / depth.bin / pose_intrinsic_imu.json)",
    }
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (dst / ".complete").touch()
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", required=True)
    ap.add_argument("--out_root", required=True, help="the scans25k/ tree to write")
    ap.add_argument("--scene_list", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=49)
    ap.add_argument("--exclude_scenes", nargs="*", default=[],
                    help="scenes the upstream release ships broken (docs/DATASET.md §2.1). "
                         "Named explicitly, and removed from the tree if already built.")
    ap.add_argument("--num_frames", type=int, default=50)
    ap.add_argument("--jpeg_quality", type=int, default=95)
    ap.add_argument("--n_probe", type=int, default=6,
                    help="sampled frames used by the geometry self-check")
    ap.add_argument("--mesh_subsample", type=int, default=400000,
                    help="cap on mesh vertices in the geometry check's KD-tree (0 = all)")
    ap.add_argument("--max_median_cm", type=float, default=15.0)
    ap.add_argument("--index_offsets", type=int, nargs="*", default=[-40, -10, 0, 10, 40],
                   help="frame-index offsets swept by the RGB/pose alignment check. "
                        "+-1..3 are deliberately absent: at 60 fps adjacent frames are "
                        "nearly identical, so a 1-frame shift is neither detectable nor "
                        "harmful, while a stream-level shift moves the peak far.")
    ap.add_argument("--skip_rgb_index_check", action="store_true")
    args = ap.parse_args()

    src_root, out_root = Path(args.src_root), Path(args.out_root)
    scenes = select_scenes(args.scene_list, args.start, args.end, args.exclude_scenes)
    gone = drop_excluded(out_root, args.exclude_scenes)
    if gone:
        print(f"[frames] removed excluded scene(s) from the tree: {gone}", flush=True)

    ok = skip = fail = 0
    failed: list[str] = []
    frames = 0
    for scene in scenes:
        if (out_root / scene / ".complete").exists():
            skip += 1
            frames += len(json.loads(
                (out_root / scene / "manifest.json").read_text())["sampled_stems"])
            continue
        t0 = time.time()
        try:
            man = build_scene(src_root, out_root, scene, args)
        except Exception as e:  # noqa: BLE001
            print(f"[{scene}] FAIL: {e}", flush=True)
            # Leave the partial dir in place but unmarked; the next run overwrites it.
            fail += 1
            failed.append(scene)
            continue
        frames += len(man["sampled_stems"])
        ok += 1
        g = man["geometry_check"]
        print(f"[{scene}] {len(man['sampled_stems'])}/{man['total_frames']} frames, "
              f"depth->mesh median {g['depth_mesh_median_cm']:.2f} cm "
              f"(worst {g['depth_mesh_worst_median_cm']:.2f}, "
              f"{g['depth_mesh_frac_within_5cm']:.0%} within 5 cm), "
              f"rgb index peak@0 {man['rgb_index_check'].get('peak_at_zero')} "
              f"{man['rgb_index_check'].get('mean_ncc')}, "
              f"{time.time() - t0:.0f}s", flush=True)

    print(f"[frames] Done: ok={ok} skip={skip} fail={fail}, {frames} frames over "
          f"{ok + skip} scenes", flush=True)
    if failed:
        print("[frames] FAILED scenes (re-run to resume): " + ", ".join(failed), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
