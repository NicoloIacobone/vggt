#!/usr/bin/env python3
"""
Download a scene range of Aria Synthetic Environments (todo 6n) into a node-local tree.

ASE is the one component of both competitors' training sets we do not have: FAST3DIS trains on
it EXCLUSIVELY and it is the largest part of IGGT's InsScene-15K, whose HuggingFace mirror has
never shipped it (docs/TRAINING_COMPARABILITY.md §5.1-5.3). Without it, arm I is "IGGT's mixture
minus ASE" and the training-matched comparison of docs/RESULTS.md §5.6 reads ~4x behind on a
mixture that is 3819 scenes against their ~100 k.

**This script cannot get the data on its own, and that is not a bug.** ASE is served from a CDN
whose per-chunk URLs live in a json file you receive only after accepting the Project Aria
dataset licence at https://www.projectaria.com/datasets/ase/ . Accepting a licence is the
account holder's act, not an agent's, so the one manual step is: accept, download the json, and
put it where --cdn_file points. Everything after that is this script.

The protocol matches facebookresearch/projectaria_tools' own downloader (Apache-2.0) — chunks of
SCENES_PER_CHUNK scenes named `<set>_chunk_<id:07d>.zip`, sha1 from the metadata — with three
things it does not do, all of which this cluster needs:

  * **resume.** A chunk already unzipped is skipped, so a job that hits the wall clock is
    resubmitted rather than restarted (the fetch_insscene15k pattern).
  * **a time budget.** It stops itself before the wall clock so the caller's packing step
    always runs; a killed job would leave the tree unpacked in $TMPDIR and lose it.
  * **an inode count.** docs/todo.md 6n gates the decision to scale on the measured file count,
    not the byte size — scratch is quota'd on inodes and the InsScene mirror already shipped
    1468 files where ~120 were expected. `--report` writes what the pilot actually cost.

FAST3DIS's own 40 % scene list is unpublished, so no scene range reproduces THEIR training set;
what a range buys is arm I becoming a complete IGGT replication (docs/TRAINING_COMPARABILITY.md
§5.1). Say "ASE scenes N-M", never "FAST3DIS's training data".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path
from zipfile import ZipFile

SCENES_PER_CHUNK = 10           # fixed by the CDN layout, not a knob
CHUNK_RETRIES = 3


def parse_scene_ids(text: str) -> list[int]:
    """`0-999`, `3`, `1,2,5-7` — the official downloader's --scene-ids grammar."""
    ids: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            ids.update(range(int(lo), int(hi) + 1))
        else:
            ids.add(int(part))
    return sorted(ids)


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def chunk_ids_for(scene_ids: list[int]) -> list[int]:
    return sorted({scene // SCENES_PER_CHUNK for scene in scene_ids})


def fetch_chunk(entry: dict, out_dir: Path, tmp_dir: Path, log,
                state_dir: Path | None = None) -> str:
    """
    Download one chunk to $TMPDIR, verify its sha1, unzip into out_dir, delete the zip.

    `state_dir` holds the `.complete` markers. It defaults to out_dir, but the driver points it
    at /cluster/work: out_dir is a node-local block directory that gets WIPED after each block
    is built, so a marker living there would not survive to the next job (docs/todo.md 6n).
    """
    name = entry["filename"]
    state_dir = state_dir or out_dir
    state_dir.mkdir(parents=True, exist_ok=True)
    marker = state_dir / f".{name}.complete"
    if marker.exists():
        return "skip"
    zip_path = tmp_dir / name
    for attempt in range(CHUNK_RETRIES):
        try:
            urllib.request.urlretrieve(entry["cdn"], zip_path)
            got = sha1_file(zip_path)
            if got != entry["sha"]:
                raise ValueError(f"sha1 {got} != {entry['sha']}")
            break
        except Exception as exc:
            log(f"[retry] {name} attempt={attempt}: {type(exc).__name__}: {exc}")
            zip_path.unlink(missing_ok=True)
    else:
        return "fail"
    with ZipFile(zip_path) as archive:
        archive.extractall(out_dir)
    zip_path.unlink(missing_ok=True)
    marker.touch()
    return "ok"


def count_inodes(root: Path) -> int:
    """
    The number docs/todo.md 6n gates on. Counted, never estimated from the byte size.

    Our own `.complete` bookkeeping is excluded: the gate is what the DATA costs, and inflating
    it by one file per chunk would make the pilot's per-scene figure wrong in the direction that
    matters (scratch is quota'd on inodes, so this number decides whether the range scales).
    """
    return sum(1 for p in root.rglob("*") if p.is_file() and not p.name.startswith("."))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cdn_file", type=Path, required=True,
                    help="the licence-gated json from projectaria.com/datasets/ase")
    ap.add_argument("--out_dir", type=Path, required=True, help="node-local scene tree")
    ap.add_argument("--tmp_dir", type=Path, required=True, help="where zips land before unzip")
    ap.add_argument("--state_dir", type=Path, default=None,
                    help="where the .complete markers live (default: --out_dir). Point it at a "
                         "PERSISTENT path when --out_dir is a node-local block dir that gets "
                         "wiped, or a resubmitted job re-downloads everything")
    ap.add_argument("--scene_ids", type=str, default="0-999")
    ap.add_argument("--set", dest="set_type", choices=("train", "test"), default="train")
    ap.add_argument("--time_budget_hours", type=float, default=22.0)
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()

    if not args.cdn_file.exists():
        # The single manual step, named explicitly rather than failing on a KeyError later.
        print(f"missing --cdn_file {args.cdn_file}.\n"
              f"Accept the ASE licence at https://www.projectaria.com/datasets/ase/ , download "
              f"the CDN json it gives you, and put it there. Nothing else in this pipeline "
              f"needs a manual step.", flush=True)
        return 2

    metadata = json.loads(args.cdn_file.read_text())
    by_name = {item["filename"]: item for item in metadata}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    scene_ids = parse_scene_ids(args.scene_ids)
    chunks = chunk_ids_for(scene_ids)
    print(f"[ase] {len(scene_ids)} scenes -> {len(chunks)} chunks of {SCENES_PER_CHUNK} "
          f"({args.set_type} set)", flush=True)

    started = time.time()
    counters = {"ok": 0, "skip": 0, "fail": 0, "missing": 0, "budget": 0}
    failed: list[str] = []
    for i, chunk_id in enumerate(chunks, 1):
        if time.time() - started > args.time_budget_hours * 3600:
            counters["budget"] = len(chunks) - i + 1
            print(f"[ase] time budget reached with {counters['budget']} chunks left — "
                  f"resubmit, the finished chunks are markered and will be skipped", flush=True)
            break
        name = f"{args.set_type}_chunk_{chunk_id:07d}.zip"
        entry = by_name.get(name)
        if entry is None:
            counters["missing"] += 1
            failed.append(name)
            print(f"[ase {i}/{len(chunks)}] {name} not in the CDN metadata", flush=True)
            continue
        outcome = fetch_chunk(entry, args.out_dir, args.tmp_dir,
                              lambda m: print(m, flush=True), args.state_dir)
        counters[outcome] += 1
        if outcome == "fail":
            failed.append(name)
        if i % 10 == 0 or i == 1 or outcome == "fail":
            print(f"[ase {i}/{len(chunks)}] {name}: {outcome} "
                  f"({(time.time() - started) / 60:.0f} min)", flush=True)

    scenes = sorted(p.name for p in args.out_dir.iterdir() if p.is_dir())
    inodes = count_inodes(args.out_dir)
    bytes_on_disk = sum(p.stat().st_size for p in args.out_dir.rglob("*") if p.is_file())
    report = {
        "set": args.set_type, "scene_ids": args.scene_ids, "chunks": len(chunks),
        "counters": counters, "failed": failed, "scenes_on_disk": len(scenes),
        "inodes": inodes, "bytes": bytes_on_disk,
        "inodes_per_scene": round(inodes / len(scenes), 1) if scenes else 0,
        "gb_per_scene": round(bytes_on_disk / 2**30 / len(scenes), 3) if scenes else 0,
        "elapsed_s": round(time.time() - started, 1),
    }
    print(json.dumps(report, indent=1), flush=True)
    if args.report:
        args.report.write_text(json.dumps(report, indent=1))
    # A budget stop is not a failure — the wrapper resubmits. A failed chunk is.
    return 1 if counters["fail"] or counters["missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
