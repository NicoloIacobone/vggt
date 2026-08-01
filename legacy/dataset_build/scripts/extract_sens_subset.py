"""Stream a ScanNet .sens file and extract ONLY the stride-5 subset color frames.

Dataset-extension tooling (500-scene build; follows docs/old/OFFICIAL_GT_MIGRATION_PLAN.md
conventions). Replaces the old SAM3-era pipeline (download whole .sens -> SensReader
exports ALL ~5500 color frames -> copy 100) with a single streaming pass:

- Parses the .sens v4 format incrementally (header, then per frame: 16f pose,
  2x uint64 timestamps, color/depth byte sizes, color bytes, depth bytes —
  verified against the official ScanNet SensReader/python/SensorData.py).
- Color frames are stored as raw JPEG bytes -> written directly to
  <out_root>/<scene>/raw_data/subset/<idx:05d>.jpg for idx in 0,5,...,495
  (fewer if the scene is shorter), no re-encoding.
- **Early abort**: .sens frames are sequential, and frame 495 sits in the first
  ~10% of a typical scene — the HTTP stream is closed as soon as the last
  needed frame is written, so only a fraction of each file is transferred and
  nothing touches disk except the ~100 jpgs.
- Resumable: scenes with a `.subset_complete` marker are skipped; a partial
  subset dir is redone from scratch (cheap).

Sanity guards: version==4, color compression 'jpeg', resolution 1296x968 OR
640x480 — a handful of ScanNet scenes (9 in 0200-0499) were captured with a
640x480 color camera; their 2D GT projections are at the same resolution, so
RGB and masks stay mutually consistent and the loader's resize to 518 handles
the rest. Anything else fails the scene loudly rather than writing wrong data.

Usage (full range):
    myenv/bin/python legacy/dataset_build/scripts/extract_sens_subset.py \
        --out_root /cluster/scratch/niacobone/scannet_official_build/scans \
        --start 200 --end 499

--scene_list FILE switches scene selection from "scene{i:04d}_00 for i in start..end"
to "line[start..end] of FILE" (0-based, inclusive) — for splits that aren't a contiguous
numeric range of _00 scans, e.g. data/splits/scannetv2_train.txt (1201 scenes, includes
_01/_02/... rescans). Default (no --scene_list) is unchanged.
"""
from __future__ import annotations

import argparse
import socket
import ssl
import struct
import sys
import time
import urllib.request
from pathlib import Path

ssl._create_default_https_context = ssl._create_unverified_context
# .sens files live under v1/scans (v2 reuses them; same swap as sam3's download_sens.py).
BASE = "http://kaldir.vc.cit.tum.de/scannet/v1/scans"

SUBSET_STRIDE = 5
SUBSET_COUNT = 100  # frames 0,5,...,495

# (width, height) of the color camera / GT masks. Most scenes are 1296x968;
# a few were captured at 640x480 (GT projections match, loader resizes to 518).
ALLOWED_RES = {(1296, 968), (640, 480)}


class EarlyAbort(Exception):
    """All needed frames written; stop reading the stream."""


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


def extract_subset_from_stream(fh, subset_dir: Path,
                               stride: int = SUBSET_STRIDE,
                               count: int = SUBSET_COUNT) -> int:
    """Parse a .sens stream, write color frames {0, stride, ..., (count-1)*stride}
    as <idx:05d>.jpg into subset_dir. Returns #frames written. Raises EarlyAbort
    (after writing everything needed) or ValueError on format violations.
    """
    version = struct.unpack("I", _read_exact(fh, 4))[0]
    if version != 4:
        raise ValueError(f".sens version {version} != 4")
    strlen = struct.unpack("Q", _read_exact(fh, 8))[0]
    _read_exact(fh, strlen)                    # sensor_name
    _read_exact(fh, 4 * 16 * 4)                # 4 camera matrices (16 floats each)
    color_comp, _depth_comp = struct.unpack("ii", _read_exact(fh, 8))
    if color_comp != 2:                        # 2 = 'jpeg' (SensorData.py)
        raise ValueError(f"color compression {color_comp} != jpeg(2)")
    cw, ch, _dw, _dh = struct.unpack("iiii", _read_exact(fh, 16))
    if (cw, ch) not in ALLOWED_RES:
        raise ValueError(f"color resolution {cw}x{ch} not in {sorted(ALLOWED_RES)}")
    if (cw, ch) != (1296, 968):
        print(f"  note: low-res color camera {cw}x{ch}", flush=True)
    _read_exact(fh, 4)                         # depth_shift (float)
    num_frames = struct.unpack("Q", _read_exact(fh, 8))[0]

    wanted = {i * stride for i in range(count)}
    last_wanted = max(w for w in wanted if w < num_frames) if num_frames else -1
    subset_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for idx in range(num_frames):
        _skip(fh, 16 * 4 + 8 + 8)              # camera_to_world + 2 timestamps
        color_bytes, depth_bytes = struct.unpack("QQ", _read_exact(fh, 16))
        if idx in wanted:
            data = _read_exact(fh, color_bytes)
            if data[:2] != b"\xff\xd8":
                raise ValueError(f"frame {idx}: color payload is not JPEG")
            (subset_dir / f"{idx:05d}.jpg").write_bytes(data)
            written += 1
        else:
            _skip(fh, color_bytes)
        _skip(fh, depth_bytes)
        if idx == last_wanted:
            raise EarlyAbort(written)
    return written


def fetch_scene(scene: str, out_root: Path, timeout: int, retries: int) -> str:
    raw_dir = out_root / scene / "raw_data"
    subset_dir = raw_dir / "subset"
    marker = raw_dir / ".subset_complete"
    if marker.exists():
        print(f"[{scene}] subset complete, skip", flush=True)
        return "skip"
    url = f"{BASE}/{scene}/{scene}.sens"
    for attempt in range(1, retries + 1):
        try:
            # partial subset from a failed attempt -> redo from scratch
            if subset_dir.exists():
                for p in subset_dir.glob("*.jpg"):
                    p.unlink()
            socket.setdefaulttimeout(timeout)
            t0 = time.time()
            written = 0
            with urllib.request.urlopen(url, timeout=timeout) as r:
                try:
                    written = extract_subset_from_stream(r, subset_dir)
                except EarlyAbort as e:
                    written = e.args[0]
            if written == 0:
                raise IOError("no frames written")
            marker.touch()
            print(f"[{scene}] OK {written} frames in {time.time()-t0:.0f}s", flush=True)
            return "ok"
        except Exception as e:  # noqa: BLE001
            print(f"[{scene}] attempt {attempt}/{retries} failed: {repr(e)[:120]}", flush=True)
            if attempt < retries:
                time.sleep(min(120, 10 * attempt))
    return "fail"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True, help="build scans root (…/scans)")
    ap.add_argument("--start", type=int, default=200)
    ap.add_argument("--end", type=int, default=499)
    ap.add_argument("--scene_list", default=None,
                    help="if set, --start/--end index into this file's lines "
                         "(0-based, inclusive) instead of scene{i:04d}_00")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--retries", type=int, default=4)
    args = ap.parse_args()

    if args.scene_list:
        all_scenes = [l.strip() for l in Path(args.scene_list).read_text().splitlines() if l.strip()]
        scenes = all_scenes[args.start:args.end + 1]
    else:
        scenes = [f"scene{i:04d}_00" for i in range(args.start, args.end + 1)]

    out_root = Path(args.out_root)
    ok = skip = fail = 0
    failed = []
    for scene in scenes:
        res = fetch_scene(scene, out_root, args.timeout, args.retries)
        if res == "ok":
            ok += 1
        elif res == "skip":
            skip += 1
        else:
            fail += 1
            failed.append(scene)
    print(f"Done: ok={ok} skip={skip} fail={fail} (range {args.start}..{args.end})", flush=True)
    if failed:
        print("FAILED scenes (re-run to resume): " + ", ".join(failed), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
