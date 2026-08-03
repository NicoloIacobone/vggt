"""Downloader for the official ScanNet v2 3D benchmark GT of a scene list (docs/todo.md 1d).

Per scene, fetches the three files the 3D instance benchmark needs (verified live on
v2/scans, 2026-08-01 — the same server and access path download_2d_gt.py already uses):

    <scene>_vh_clean_2.ply                  the benchmark mesh (~9.5 MB)
    <scene>_vh_clean_2.0.010000.segs.json   its superpoint over-segmentation (~1.5 MB)
    <scene>.aggregation.json                instance -> superpoints + raw label (~10 KB)

`<scene>.aggregation.json` (not `_vh_clean.aggregation.json`) is the one that indexes into
the `_vh_clean_2` segs — the pairing the official benchmark export script uses.

Same timeout/retry/resume structure as download_2d_gt.py; a scene whose three files already
exist and validate is skipped, so re-running resumes. Validation per scene: the ply starts
with the "ply" magic, both jsons parse, and the aggregation's segment ids stay inside the
segs range (cheap guard against truncated downloads that json.load would still accept — the
byte count check in fetch_one catches most, this catches served-but-wrong content).

Honors http(s)_proxy env vars (set by `module load eth_proxy` on Euler).

Usage (from the vggt repo):
    myenv/bin/python legacy/dataset_build/scripts/download_3d_gt.py \
        --out_root $TMPDIR/build/scans3d \
        --scene_list data/splits/scannetv2_val.txt --start 0 --end 311
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import ssl
import sys
import time
import urllib.request
from pathlib import Path

ssl._create_default_https_context = ssl._create_unverified_context
BASE = "http://kaldir.vc.cit.tum.de/scannet/v2/scans"
SUFFIXES = ("_vh_clean_2.ply", "_vh_clean_2.0.010000.segs.json", ".aggregation.json")


def fetch_one(url: str, final: Path, timeout: int, retries: int) -> str:
    if final.is_file() and final.stat().st_size > 0:
        print(f"  present ({final.stat().st_size/1e6:.1f} MB), skip {final.name}", flush=True)
        return "skip"
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(final) + ".part")
    for attempt in range(1, retries + 1):
        try:
            socket.setdefaulttimeout(timeout)
            t0 = time.time()
            with urllib.request.urlopen(url, timeout=timeout) as r:
                total = int(r.headers.get("Content-Length", 0))
                done = 0
                with open(tmp, "wb") as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
            if total and done < total:
                raise IOError(f"short read {done}/{total}")
            os.rename(tmp, final)
            print(f"  OK {final.name} {done/1e6:.1f} MB in {time.time()-t0:.0f}s", flush=True)
            return "ok"
        except Exception as e:  # noqa: BLE001
            print(f"  attempt {attempt}/{retries} failed for {final.name}: "
                  f"{repr(e)[:110]}", flush=True)
            try:
                os.remove(tmp)
            except OSError:
                pass
            if attempt < retries:
                time.sleep(min(120, 10 * attempt))
    return "fail"


def validate_scene(scene_dir: Path, scene: str) -> str | None:
    """Return an error string, or None if the scene's three files look sound."""
    ply = scene_dir / f"{scene}{SUFFIXES[0]}"
    with open(ply, "rb") as f:
        if f.read(3) != b"ply":
            return f"{ply.name}: missing ply magic"
    try:
        segs = json.loads((scene_dir / f"{scene}{SUFFIXES[1]}").read_text())
        agg = json.loads((scene_dir / f"{scene}{SUFFIXES[2]}").read_text())
    except json.JSONDecodeError as e:
        return f"json parse: {e}"
    seg_ids = set(segs["segIndices"])
    for group in agg["segGroups"]:
        if not set(group["segments"]) <= seg_ids:
            return (f"aggregation group {group['id']} references segments outside "
                    f"the segs file")
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True,
                    help="one <scene>/ dir with the three files lands per scene")
    ap.add_argument("--scene_list", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=311)
    ap.add_argument("--timeout", type=int, default=120, help="per-read socket timeout (s)")
    ap.add_argument("--retries", type=int, default=4)
    args = ap.parse_args()

    all_scenes = [l.strip() for l in Path(args.scene_list).read_text().splitlines() if l.strip()]
    scenes = all_scenes[args.start:args.end + 1]
    out_root = Path(args.out_root)

    ok = skip = fail = 0
    failed: list[str] = []
    for scene in scenes:
        scene_dir = out_root / scene
        if (scene_dir / ".complete").exists():
            skip += 1
            continue
        print(f"[{scene}]", flush=True)
        results = [fetch_one(f"{BASE}/{scene}/{scene}{suf}", scene_dir / f"{scene}{suf}",
                             args.timeout, args.retries) for suf in SUFFIXES]
        err = None if "fail" in results else validate_scene(scene_dir, scene)
        if "fail" in results or err:
            if err:
                print(f"[{scene}] VALIDATE FAIL: {err} — wiping for re-download", flush=True)
                for suf in SUFFIXES:
                    (scene_dir / f"{scene}{suf}").unlink(missing_ok=True)
            fail += 1
            failed.append(scene)
            continue
        (scene_dir / ".complete").touch()
        ok += 1
    print(f"Done: ok={ok} skip={skip} fail={fail} (range {args.start}..{args.end})", flush=True)
    if failed:
        print("FAILED scenes (re-run to resume): " + ", ".join(failed), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
