"""Extract a scene list's frames from the official scannet_frames_25k.zip (docs/todo.md 1d).

The 25k export samples ~16 frames per scan across the WHOLE video (unlike our stride-5
subsets, which cover only raw frames 0-495) and ships per-frame camera-to-world poses and
intrinsics — the inputs the 3D benchmark eval needs for full-scene coverage and for the
eval-time Sim(3) registration (GT poses are used ONLY there, never at inference).

Keeps, per scene:  color/*.jpg  pose/*.txt  depth/*.png  intrinsics_{color,depth}.txt
(depth is kept for a possible GT-depth oracle ablation; instance/label pngs are dropped —
the 3D eval scores against the 3D GT, not 2D masks). The leading "scannet_frames_25k/"
zip prefix is stripped, so the output tree is  <out_root>/<scene>/color/... .

Verifies per scene: >0 frames, one pose per color frame, intrinsics_color.txt present, and
every pose file parses as a finite 4x4 matrix (guards against the export's known rare
'-inf' poses ending up in the eval unnoticed — those frames are LISTED, not dropped, so the
eval can skip them explicitly).

Usage (from the vggt repo):
    myenv/bin/python legacy/dataset_build/scripts/repack_frames25k.py \
        --zip $TMPDIR/scannet_frames_25k.zip --out_root $TMPDIR/build/scans25k \
        --scene_list data/splits/scannetv2_val.txt --start 0 --end 311
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ZIP_PREFIX = "scannet_frames_25k/"
KEEP_DIRS = ("color/", "pose/", "depth/")
KEEP_FILES = ("intrinsics_color.txt", "intrinsics_depth.txt")


def wanted(rel: str) -> bool:
    """rel is the path inside a scene dir, e.g. 'color/000000.jpg'."""
    return rel.startswith(KEEP_DIRS) or rel in KEEP_FILES


def check_scene(scene_dir: Path, scene: str) -> tuple[str | None, list[str]]:
    """Return (error, bad_pose_frames). bad poses are reported, not fatal."""
    colors = sorted(p.stem for p in (scene_dir / "color").glob("*.jpg"))
    poses = sorted(p.stem for p in (scene_dir / "pose").glob("*.txt"))
    if not colors:
        return "no color frames", []
    if colors != poses:
        return f"color/pose mismatch ({len(colors)} vs {len(poses)})", []
    if not (scene_dir / "intrinsics_color.txt").exists():
        return "missing intrinsics_color.txt", []
    bad = []
    for stem in poses:
        mat = np.loadtxt(scene_dir / "pose" / f"{stem}.txt")
        if mat.shape != (4, 4) or not np.isfinite(mat).all():
            bad.append(f"{scene}/{stem}")
    return None, bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--scene_list", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=311)
    args = ap.parse_args()

    all_scenes = [l.strip() for l in Path(args.scene_list).read_text().splitlines() if l.strip()]
    scenes = set(all_scenes[args.start:args.end + 1])
    out_root = Path(args.out_root)

    zf = zipfile.ZipFile(args.zip)
    by_scene: dict[str, list[str]] = defaultdict(list)
    for name in zf.namelist():
        if not name.startswith(ZIP_PREFIX) or name.endswith("/"):
            continue
        scene, _, rel = name[len(ZIP_PREFIX):].partition("/")
        if scene in scenes and wanted(rel):
            by_scene[scene].append(name)

    missing = sorted(scenes - set(by_scene))
    if missing:
        print(f"MISSING from zip ({len(missing)}): " + ", ".join(missing), flush=True)
        sys.exit(1)

    ok = skip = 0
    frames = 0
    bad_poses: list[str] = []
    for scene in sorted(by_scene):
        scene_dir = out_root / scene
        if (scene_dir / ".complete").exists():
            skip += 1
            continue
        for name in by_scene[scene]:
            rel = name[len(ZIP_PREFIX):]
            dest = out_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src:
                dest.write_bytes(src.read())
        err, bad = check_scene(scene_dir, scene)
        if err:
            print(f"[{scene}] FAIL: {err}", flush=True)
            sys.exit(1)
        bad_poses += bad
        n = len(list((scene_dir / "color").glob("*.jpg")))
        frames += n
        (scene_dir / ".complete").touch()
        ok += 1
        print(f"[{scene}] {n} frames", flush=True)

    print(f"Done: ok={ok} skip={skip} scenes, {frames} frames", flush=True)
    if bad_poses:
        # Non-fatal: the eval must skip these frames (finite-pose check on load).
        print(f"NON-FINITE poses ({len(bad_poses)}): " + ", ".join(bad_poses), flush=True)


if __name__ == "__main__":
    main()
