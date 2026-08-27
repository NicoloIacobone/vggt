"""Stream a ScanNet .sens and write a DENSE whole-scan frame export (docs/todo.md 5b, 6k).

Why this exists. The official `scannet_frames_25k` export samples every 100th frame, i.e.
~17 frames/scene on the val-312 split — and the two published methods our 3D ruler is read
against evaluate on far more views: **FAST3DIS on 50 uniformly sampled views** and
**SegVGGT on every 20th frame (~75-120)**. View count was therefore the last unmatched axis
of the protocol comparison (docs/TRAINING_COMPARABILITY.md §6.3). This script produces the
same tree at an arbitrary stride, so `--num_frames 50` in the eval reproduces FAST3DIS's
budget exactly and the full export approximates SegVGGT's.

Output layout is **byte-compatible with `repack_frames25k.py`** — `<scene>/{color/<idx:06d>.jpg,
depth/<idx:06d>.png, pose/<idx:06d>.txt, intrinsics_color.txt, intrinsics_depth.txt}` — so
`train/scannet3d.py` reads it with no new conventions and `sample_frames25k` subsamples it
uniformly (np.linspace) down to whatever `--num_frames` the eval asks for.

Unlike `extract_sens_subset.py` this CANNOT early-abort: a whole-scan sample needs the last
frame, so the entire .sens is streamed (~1.1 GB/scene, measured ~68 MB/s from ETH, i.e.
~20 s/scene). Nothing but the kept frames touches disk.

Format per the official SensReader/python/SensorData.py, verified field by field:
    header: uint32 version(4) | uint64 strlen + sensor_name
            4 x 4x4 float32 (intrinsic_color, extrinsic_color, intrinsic_depth, extrinsic_depth)
            int32 color_compression | int32 depth_compression
            int32 color_w,color_h,depth_w,depth_h | float32 depth_shift | uint64 num_frames
    frame:  4x4 float32 camera_to_world | uint64 ts_color | uint64 ts_depth
            uint64 color_bytes | uint64 depth_bytes | color payload | depth payload
Color is jpeg (compression 2), depth is zlib'd uint16 millimetres (compression 1) — both
asserted, never assumed. Non-finite poses are written as-is: that is what the official export
does, and `sample_frames25k` filters them out downstream.

Usage:
    myenv/bin/python legacy/dataset_build/scripts/extract_sens_frames25k.py \
        --out_root /cluster/scratch/niacobone/scannet_frames_dense/scans25k \
        --scene_list data/splits/scannetv2_val.txt --start 0 --end 311 --stride 20
"""
from __future__ import annotations

import argparse
import io
import socket
import ssl
import struct
import sys
import time
import urllib.request
import zlib
from pathlib import Path

import numpy as np

ssl._create_default_https_context = ssl._create_unverified_context
BASE = "http://kaldir.vc.cit.tum.de/scannet/v1/scans"

# (width, height) of the color camera. Most scenes are 1296x968; a few were captured at
# 640x480 (docs/DATASET.md §2) — both are legal, anything else fails the scene loudly.
ALLOWED_RES = {(1296, 968), (640, 480)}
COMPRESSION_JPEG = 2          # SensorData.COMPRESSION_TYPE_COLOR['jpeg']
COMPRESSION_ZLIB_USHORT = 1   # SensorData.COMPRESSION_TYPE_DEPTH['zlib_ushort']


def _read_exact(fh, n: int) -> bytes:
    """Read exactly n bytes from a (possibly network) stream, or raise IOError."""
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = fh.read(min(remaining, 1 << 20))
        if not chunk:
            raise IOError(f"stream ended {remaining}/{n} bytes short")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _skip(fh, n: int) -> None:
    """Discard n bytes (HTTP streams can't seek)."""
    remaining = n
    while remaining > 0:
        chunk = fh.read(min(remaining, 1 << 20))
        if not chunk:
            raise IOError(f"stream ended while skipping ({remaining}/{n} left)")
        remaining -= len(chunk)


def _mat4(buf: bytes) -> np.ndarray:
    return np.frombuffer(buf, dtype=np.float32, count=16).reshape(4, 4).astype(np.float64)


def _write_mat(path: Path, mat: np.ndarray) -> None:
    """ScanNet writes 4x4 matrices as space-separated rows; np.loadtxt reads them back."""
    path.write_text("\n".join(" ".join(f"{v:.9g}" for v in row) for row in mat) + "\n")


def extract_frames_from_stream(fh, scene_dir: Path, stride: int,
                               max_frames: int = 0) -> int:
    """Parse a .sens stream, write every `stride`-th frame's color/depth/pose into
    scene_dir. Returns #frames written. Raises ValueError on any format violation."""
    from PIL import Image                      # local: only this path needs Pillow

    version = struct.unpack("I", _read_exact(fh, 4))[0]
    if version != 4:
        raise ValueError(f".sens version {version} != 4")
    strlen = struct.unpack("Q", _read_exact(fh, 8))[0]
    _read_exact(fh, strlen)                                    # sensor_name
    k_color = _mat4(_read_exact(fh, 64))                       # intrinsic_color
    _read_exact(fh, 64)                                        # extrinsic_color
    k_depth = _mat4(_read_exact(fh, 64))                       # intrinsic_depth
    _read_exact(fh, 64)                                        # extrinsic_depth
    color_comp, depth_comp = struct.unpack("ii", _read_exact(fh, 8))
    if color_comp != COMPRESSION_JPEG:
        raise ValueError(f"color compression {color_comp} != jpeg({COMPRESSION_JPEG})")
    if depth_comp != COMPRESSION_ZLIB_USHORT:
        raise ValueError(f"depth compression {depth_comp} != zlib_ushort"
                         f"({COMPRESSION_ZLIB_USHORT})")
    cw, ch, dw, dh = struct.unpack("iiii", _read_exact(fh, 16))
    if (cw, ch) not in ALLOWED_RES:
        raise ValueError(f"color resolution {cw}x{ch} not in {sorted(ALLOWED_RES)}")
    depth_shift = struct.unpack("f", _read_exact(fh, 4))[0]
    if abs(depth_shift - 1000.0) > 1e-3:
        # The whole pipeline reads depth as uint16 MILLIMETRES (train/scannet3d.py
        # load_frames25k_depth divides by 1000). A different shift would silently
        # mis-scale every posed-transfer number.
        raise ValueError(f"depth_shift {depth_shift} != 1000 — depth would be mis-scaled")
    num_frames = struct.unpack("Q", _read_exact(fh, 8))[0]

    for sub in ("color", "depth", "pose"):
        (scene_dir / sub).mkdir(parents=True, exist_ok=True)
    _write_mat(scene_dir / "intrinsics_color.txt", k_color)
    _write_mat(scene_dir / "intrinsics_depth.txt", k_depth)

    wanted = set(range(0, num_frames, stride))
    if max_frames and len(wanted) > max_frames:
        idx = np.linspace(0, num_frames - 1, max_frames).round().astype(int)
        wanted = set(int(i) for i in idx)
    last_wanted = max(wanted) if wanted else -1

    written = 0
    for idx in range(num_frames):
        pose_buf = _read_exact(fh, 64)
        _skip(fh, 16)                                          # 2 timestamps
        color_bytes, depth_bytes = struct.unpack("QQ", _read_exact(fh, 16))
        if idx in wanted:
            data = _read_exact(fh, color_bytes)
            if data[:2] != b"\xff\xd8":
                raise ValueError(f"frame {idx}: color payload is not JPEG")
            (scene_dir / "color" / f"{idx:06d}.jpg").write_bytes(data)

            raw = zlib.decompress(_read_exact(fh, depth_bytes))
            if len(raw) != dw * dh * 2:
                raise ValueError(f"frame {idx}: depth payload {len(raw)} B "
                                 f"!= {dw}x{dh}x2")
            arr = np.frombuffer(raw, dtype=np.uint16).reshape(dh, dw)
            Image.fromarray(arr).save(scene_dir / "depth" / f"{idx:06d}.png")

            _write_mat(scene_dir / "pose" / f"{idx:06d}.txt", _mat4(pose_buf))
            written += 1
        else:
            _skip(fh, color_bytes)
            _skip(fh, depth_bytes)
        if idx == last_wanted:
            break                                              # nothing needed after this
    return written


def check_scene(scene_dir: Path) -> str | None:
    """The same contract repack_frames25k.py enforces: one pose and one depth per color."""
    colors = sorted(p.stem for p in (scene_dir / "color").glob("*.jpg"))
    poses = sorted(p.stem for p in (scene_dir / "pose").glob("*.txt"))
    depths = sorted(p.stem for p in (scene_dir / "depth").glob("*.png"))
    if not colors:
        return "no color frames"
    if colors != poses:
        return f"color/pose mismatch ({len(colors)} vs {len(poses)})"
    if colors != depths:
        return f"color/depth mismatch ({len(colors)} vs {len(depths)})"
    for which in ("color", "depth"):
        if not (scene_dir / f"intrinsics_{which}.txt").exists():
            return f"missing intrinsics_{which}.txt"
    return None


def fetch_scene(scene: str, out_root: Path, stride: int, max_frames: int,
                timeout: int, retries: int) -> str:
    scene_dir = out_root / scene
    marker = scene_dir / ".complete"
    if marker.exists():
        print(f"[{scene}] complete, skip", flush=True)
        return "skip"
    url = f"{BASE}/{scene}/{scene}.sens"
    for attempt in range(1, retries + 1):
        try:
            for sub in ("color", "depth", "pose"):             # partial attempt -> redo
                for p in (scene_dir / sub).glob("*"):
                    p.unlink()
            socket.setdefaulttimeout(timeout)
            t0 = time.time()
            with urllib.request.urlopen(url, timeout=timeout) as r:
                written = extract_frames_from_stream(r, scene_dir, stride, max_frames)
            if written == 0:
                raise IOError("no frames written")
            err = check_scene(scene_dir)
            if err:
                raise IOError(err)
            marker.touch()
            print(f"[{scene}] OK {written} frames in {time.time()-t0:.0f}s", flush=True)
            return "ok"
        except Exception as e:  # noqa: BLE001
            print(f"[{scene}] attempt {attempt}/{retries} failed: {repr(e)[:160]}", flush=True)
            if attempt < retries:
                time.sleep(min(120, 10 * attempt))
    return "fail"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True, help="…/scans25k")
    ap.add_argument("--scene_list", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=311, help="inclusive, 0-based line index")
    ap.add_argument("--stride", type=int, default=20,
                    help="keep every Nth raw frame (SegVGGT's eval convention)")
    ap.add_argument("--max_frames", type=int, default=150,
                    help="hard cap per scene, uniformly resampled (0 = no cap)")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--retries", type=int, default=4)
    args = ap.parse_args()

    all_scenes = [l.strip() for l in Path(args.scene_list).read_text().splitlines() if l.strip()]
    scenes = all_scenes[args.start:args.end + 1]
    out_root = Path(args.out_root)

    ok = skip = fail = 0
    failed = []
    for scene in scenes:
        res = fetch_scene(scene, out_root, args.stride, args.max_frames,
                          args.timeout, args.retries)
        ok += res == "ok"
        skip += res == "skip"
        if res == "fail":
            fail += 1
            failed.append(scene)
    print(f"Done: ok={ok} skip={skip} fail={fail} (lines {args.start}..{args.end})", flush=True)
    if failed:
        print("FAILED scenes (re-run to resume): " + ", ".join(failed), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
