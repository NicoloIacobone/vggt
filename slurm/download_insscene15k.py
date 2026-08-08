#!/usr/bin/env python3
"""Deliverable 2 (docs/todo.md external-dataset task): mirror lifuguan/InsScene-15K shards to
work, one file at a time through $TMPDIR, never unzipped, never accumulated loose on
/cluster/scratch. Self-limits to --time_budget_hours so the SLURM wrapper's tail resubmit
(the train_maskdino_coco.sh pattern) always gets to run instead of being killed at the wall
clock mid-transfer.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = "lifuguan/InsScene-15K"
RESOLVE_ROOT = f"https://huggingface.co/datasets/{REPO}/resolve/main"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_one(rel_path, size, tmp_dir, out_dir, log):
    dst = out_dir / rel_path
    if dst.exists() and dst.stat().st_size == size:
        return "skip"
    tmp_path = tmp_dir / rel_path
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{RESOLVE_ROOT}/{rel_path}"
    for attempt in range(3):
        p = subprocess.run(["wget", "--continue", "-q", "-O", str(tmp_path), url])
        if p.returncode == 0 and tmp_path.exists() and tmp_path.stat().st_size == size:
            break
        log(f"[retry] {rel_path} attempt={attempt} rc={p.returncode} "
            f"got={tmp_path.stat().st_size if tmp_path.exists() else -1} want={size}")
    else:
        return "fail"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.replace(dst)
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tmp_dir", required=True)
    ap.add_argument("--time_budget_hours", type=float, default=22.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--state_file", required=True)
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    out_dir = Path(args.out_dir)
    tmp_dir = Path(args.tmp_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    all_files = []
    for sub, entry in manifest["subsets"].items():
        if entry.get("present"):
            all_files.extend(entry["files"])

    deadline = time.time() + args.time_budget_hours * 3600
    counts = {"ok": 0, "skip": 0, "fail": 0}
    lock_print = lambda msg: print(msg, flush=True)

    todo = [f for f in all_files
            if not ((out_dir / f["path"]).exists()
                    and (out_dir / f["path"]).stat().st_size == f["size"])]
    lock_print(f"[state] {len(all_files) - len(todo)}/{len(all_files)} already mirrored, "
               f"{len(todo)} remaining")

    stopped_early = False
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {}
        it = iter(todo)
        # seed
        for _ in range(args.workers):
            try:
                f = next(it)
            except StopIteration:
                break
            futures[ex.submit(download_one, f["path"], f["size"], tmp_dir, out_dir, lock_print)] = f

        while futures:
            if time.time() > deadline:
                stopped_early = True
                lock_print("[time_budget] deadline reached, letting in-flight downloads finish "
                           "and stopping (no new files started)")
                break
            done, _ = cf.wait(futures, timeout=30, return_when=cf.FIRST_COMPLETED)
            for fut in done:
                f = futures.pop(fut)
                try:
                    res = fut.result()
                except Exception as e:
                    res = "fail"
                    lock_print(f"[error] {f['path']}: {e}")
                counts[res] += 1
                lock_print(f"[{res}] {f['path']} ({counts['ok']} ok, {counts['skip']} skip, "
                           f"{counts['fail']} fail so far)")
                try:
                    nf = next(it)
                    futures[ex.submit(download_one, nf["path"], nf["size"], tmp_dir, out_dir,
                                       lock_print)] = nf
                except StopIteration:
                    pass
        if stopped_early:
            for fut in futures:
                fut.result()

    remaining = [f for f in all_files
                 if not ((out_dir / f["path"]).exists()
                         and (out_dir / f["path"]).stat().st_size == f["size"])]
    state = {
        "total_files": len(all_files), "remaining": len(remaining),
        "counts_this_segment": counts, "complete": len(remaining) == 0,
        "stopped_early_on_time_budget": stopped_early,
    }
    Path(args.state_file).write_text(json.dumps(state, indent=2))
    lock_print(f"[segment done] {json.dumps(state)}")

    if len(remaining) == 0:
        manifest_sha = sha256_file(args.manifest)
        subsets_present = [s for s, e in manifest["subsets"].items() if e.get("present")]
        readme = f"""# InsScene-15K mirror

Source: https://huggingface.co/datasets/{REPO}
Date mirrored: {time.strftime('%Y-%m-%d')}
Licence: Apache-2.0
Shard count: {manifest['total_files']} files
Total bytes: {manifest['total_bytes']} ({manifest['total_bytes']/1e9:.2f} GB)
Manifest sha256: {manifest_sha}
Subsets present: {', '.join(subsets_present)}
Aria/ASE directory present at repo root: {manifest['aria_or_ase_dir_present']}
README on HF still says "still being uploaded": {manifest['readme_says_still_uploading']}

Shards are stored exactly as uploaded (zip parts), NOT unzipped -- one shard is one inode.
This mirror only covers the three subsets present at mirror time; per the upstream README this
dataset is a PARTIAL release (no Aria/ASE data as of this date), so any experiment built on it
must be labelled accordingly.
"""
        (out_dir / "README.md").write_text(readme)
        lock_print("[complete] wrote README.md")


if __name__ == "__main__":
    main()
