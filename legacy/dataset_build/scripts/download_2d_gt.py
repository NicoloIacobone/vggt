"""Robust, non-interactive downloader for official ScanNet 2D GT zips + convert pipeline.

Adapted from sam3/scripts/download_sens.py (same timeout/retry/resume structure) for
the official-GT migration (docs/old/OFFICIAL_GT_MIGRATION_PLAN.md, Phase 2). Differences:
- Fetches <scene>_2d-instance-filt.zip and <scene>_2d-label-filt.zip from
  v2/scans (NOT the v1/scans swap — that is specific to .sens files).
- Optional --convert_out: after both zips of a scene arrive, runs
  legacy/dataset_build/scripts/build_official_masks.py::convert_scene on them and DELETES the zips on
  success, so peak disk stays ~1 scene of zips (~130 MB).
- Resumable at both levels: existing zips are not re-downloaded; scenes with a
  .complete marker in the build tree are skipped entirely.

Honors http(s)_proxy env vars (set by `module load eth_proxy` on Euler).

Usage (full Phase-2 run, from the vggt repo):
    myenv/bin/python legacy/dataset_build/scripts/download_2d_gt.py \
        --zips_dir /cluster/scratch/niacobone/scannet_official_build/zips \
        --convert_out /cluster/scratch/niacobone/scannet_official_build/scans \
        --subset_root /cluster/scratch/niacobone/scannet_official_build/sam3_subsets/scans \
        --start 0 --end 199

--scene_list FILE switches scene selection from "scene{i:04d}_00 for i in start..end" to
"line[start..end] of FILE" (0-based, inclusive) — see extract_sens_subset.py for the same
convention (used together for splits like data/splits/scannetv2_train.txt).
"""
from __future__ import annotations

import argparse
import os
import socket
import ssl
import sys
import time
import urllib.request
from pathlib import Path

ssl._create_default_https_context = ssl._create_unverified_context
# 2D GT zips live under v2/scans (verified Phase 0; the server 301-redirects
# http->https, urllib follows). The v1/scans swap in download_sens.py is
# specific to .sens files -- do NOT apply it here.
BASE = "http://kaldir.vc.cit.tum.de/scannet/v2/scans"
SUFFIXES = ("_2d-instance-filt.zip", "_2d-label-filt.zip")

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zips_dir", required=True, help="where the zips land")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=199)
    ap.add_argument("--scene_list", default=None,
                    help="if set, --start/--end index into this file's lines "
                         "(0-based, inclusive) instead of scene{i:04d}_00")
    ap.add_argument("--timeout", type=int, default=120, help="per-read socket timeout (s)")
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--convert_out", default=None,
                    help="build-tree scans root; if set, convert each scene after "
                         "download and delete its zips on success")
    ap.add_argument("--subset_root", default=None,
                    help="unpacked SAM3 scans root (subset copy + frame list), "
                         "required with --convert_out")
    ap.add_argument("--tsv", default="/cluster/work/igp_psr/niacobone/distillation/"
                                     "dataset/scannet/scannetv2-labels.combined.tsv")
    args = ap.parse_args()

    if args.convert_out:
        if not args.subset_root:
            ap.error("--convert_out requires --subset_root")
        from legacy.dataset_build.scripts.build_official_masks import convert_scene

    if args.scene_list:
        all_scenes = [l.strip() for l in Path(args.scene_list).read_text().splitlines() if l.strip()]
        scenes = all_scenes[args.start:args.end + 1]
    else:
        scenes = [f"scene{i:04d}_00" for i in range(args.start, args.end + 1)]

    zips_dir = Path(args.zips_dir)
    zips_dir.mkdir(parents=True, exist_ok=True)
    ok = skip = fail = 0
    failed: list[str] = []
    for scene in scenes:
        if args.convert_out:
            marker = Path(args.convert_out) / scene / "raw_data" / ".complete"
            if marker.exists():
                print(f"[{scene}] converted (.complete), skip", flush=True)
                skip += 1
                continue
        print(f"[{scene}]", flush=True)
        zpaths = [zips_dir / f"{scene}{suf}" for suf in SUFFIXES]
        results = [fetch_one(f"{BASE}/{scene}/{scene}{suf}", zp, args.timeout, args.retries)
                   for suf, zp in zip(SUFFIXES, zpaths)]
        if "fail" in results:
            fail += 1
            failed.append(scene)
            continue
        if args.convert_out:
            subset_src = Path(args.subset_root) / scene / "raw_data" / "subset"
            if not subset_src.exists():
                print(f"[{scene}] FAIL: no subset dir at {subset_src}", flush=True)
                fail += 1
                failed.append(scene)
                continue
            try:
                stats = convert_scene(scene, zpaths[0], zpaths[1], args.tsv,
                                      args.convert_out, subset_src=subset_src)
            except Exception as e:  # noqa: BLE001
                print(f"[{scene}] CONVERT FAIL: {repr(e)[:200]}", flush=True)
                fail += 1
                failed.append(scene)
                continue
            marker = Path(args.convert_out) / scene / "raw_data" / ".complete"
            marker.touch()
            for zp in zpaths:
                zp.unlink()
            print(f"[{scene}] converted: {stats['num_instances']} instances, "
                  f"dups={stats['cross_class_duplicates_iou50']}, "
                  f"max cross-IoU={stats['cross_class_max_iou']}", flush=True)
        ok += 1
    print(f"Done: ok={ok} skip={skip} fail={fail} (range {args.start}..{args.end})", flush=True)
    if failed:
        print("FAILED scenes (re-run to resume): " + ", ".join(failed), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
