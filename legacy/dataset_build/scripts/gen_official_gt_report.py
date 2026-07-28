"""Aggregate per-scene _qa/stats.json of the official-GT build into OFFICIAL_GT_README.md.

Phase-2 QA + deliverable report for docs/old/OFFICIAL_GT_MIGRATION_PLAN.md, in the style
of sam3/scripts/gen_instance_report.py. Also enforces the QA gates:
  1. aggregate cross-class duplicate count == 0 (the migration's acceptance test);
  2. instance-count sanity (same order of magnitude as the SAM3 build's ~4195).
Exits non-zero if a gate fails or scenes are missing.

Usage:
    myenv/bin/python legacy/dataset_build/scripts/gen_official_gt_report.py \
        --build /cluster/scratch/niacobone/scannet_official_build \
        --out /cluster/work/igp_psr/niacobone/distillation/dataset/scannet/OFFICIAL_GT_README.md
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import date
from pathlib import Path

CLASS_ORDER = [
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
    "window", "bookshelf", "picture", "counter", "desk", "curtain",
    "refrigerator", "shower curtain", "toilet", "sink", "bathtub",
]

POLICY = """\
## Provenance

Masks are derived from the **official ScanNet v2 2D annotations**
(`<scene>_2d-instance-filt.zip` + `<scene>_2d-label-filt.zip`, kaldir.vc.cit.tum.de),
which are per-frame projections of the single human-verified 3D annotation —
one class per object and cross-view-consistent instance ids by construction.
They replace the SAM3-generated GT (`scannet_instance_dataset_full.tar.zst`),
whose per-class prompting produced ~15.9% multi-class foreground pixels and
~3.4 cross-class duplicate instances/scene (audit 2026-07-07). The SAM3 tar is
kept unchanged as the baseline for GT-quality comparisons.

Converter: `legacy/dataset_build/scripts/build_official_masks.py` in the vggt repo
(tests: `legacy/dataset_build/tests/test_build_official_masks.py`).

## Layout (identical to the SAM3 dataset — loader-compatible)

```
scans/<scene>/raw_data/subset/<frame>.jpg            # unchanged copy of the SAM3 subset
scans/<scene>/raw_data/masks/<class>/<frame>.png     # per-class union, uint8 {0,255}
scans/<scene>/raw_data/masks_instance/<class>_<k>/<frame>.png
scans/<scene>/raw_data/_qa/stats.json                # per-scene QA metrics
```

- Resolution 1296x968 (native color camera, same as SAM3). Filenames match the
  subset stems (`00375.png` = official frame index 375).
- Masks are written **sparsely**: only frames where the instance is visible
  (the loader skips missing files). This differs from the SAM3 build, which
  wrote all-zero PNGs — both are handled identically by the loader.
- `<k>` is zero-based per class in order of first appearance.
- Class per instance = majority `label-filt` vote mapped raw-id -> NYU40 via
  `scannetv2-labels.combined.tsv`. NYU40 classes outside the 19 trainable ones
  (incl. `otherfurniture`=39) are dropped to background. No speck filter.
- Stuff classes (`wall`/`floor`) keep the official instance ids (NOT merged to
  `_0` as SAM3 did); the loader/matcher are instance-based and unaffected.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="/cluster/scratch/niacobone/scannet_official_build")
    ap.add_argument("--out", default="/cluster/work/igp_psr/niacobone/distillation/"
                                     "dataset/scannet/OFFICIAL_GT_README.md")
    ap.add_argument("--expect_scenes", type=int, default=200)
    args = ap.parse_args()

    scans = Path(args.build) / "scans"
    scenes = sorted(p.name for p in scans.iterdir() if p.is_dir())

    rows, failed = [], []
    total_instances = 0
    total_dups = 0
    max_cross_iou = 0.0
    purities = []
    class_totals: Counter = Counter()
    dropped_nyu: Counter = Counter()
    for scene in scenes:
        sj = scans / scene / "raw_data" / "_qa" / "stats.json"
        if not sj.exists():
            failed.append(scene)
            continue
        d = json.loads(sj.read_text())
        counts = {c: d["instances_per_class"].get(c, 0) for c in CLASS_ORDER}
        total_instances += d["num_instances"]
        total_dups += d["cross_class_duplicates_iou50"]
        max_cross_iou = max(max_cross_iou, d["cross_class_max_iou"])
        if d["min_label_purity"] is not None:
            purities.append(d["min_label_purity"])
        class_totals.update(counts)
        dropped_nyu.update(str(v) for v in d["dropped_out_of_taxonomy"].values())
        rows.append((scene, counts, d["num_instances"],
                     len(d["dropped_out_of_taxonomy"]), d["min_label_purity"]))

    missing = args.expect_scenes - len(rows)
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                cwd=Path(__file__).parent.parent,
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = "unknown"

    lines = ["# ScanNet official-GT instance masks", ""]
    lines.append(f"Built {date.today().isoformat()}, converter commit `{commit}`. "
                 f"{len(rows)} scenes, **{total_instances} instances** "
                 f"(SAM3 build: ~4195 over the same 200 scenes). "
                 f"Min per-instance label purity: "
                 f"{min(purities) if purities else 'n/a'}.")
    lines.append("")
    lines.append(f"**QA gate — cross-class duplicates (IoU>=0.5): {total_dups}** "
                 f"(SAM3 audit: ~3.4/scene); max cross-class instance IoU "
                 f"anywhere: {max_cross_iou}.")
    lines.append("")
    lines.append(POLICY)
    lines.append("## Instances per class (all scenes)")
    lines.append("")
    lines.append("| class | instances |")
    lines.append("|---|---|")
    for c in CLASS_ORDER:
        lines.append(f"| {c} | {class_totals[c]} |")
    lines.append("")
    lines.append("Out-of-taxonomy instances dropped to background, by NYU40 id: "
                 + json.dumps(dict(sorted(dropped_nyu.items(),
                                          key=lambda kv: -kv[1]))) + ".")
    lines.append("")
    lines.append("## Per-scene instance counts")
    lines.append("")
    header = ["scene"] + CLASS_ORDER + ["total", "dropped", "min_purity"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for scene, counts, n_inst, n_drop, purity in rows:
        cells = [scene] + [str(counts[c]) for c in CLASS_ORDER]
        cells += [str(n_inst), str(n_drop), str(purity) if purity is not None else "-"]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Reliability")
    lines.append("")
    lines.append(f"- Scenes missing/failed: {', '.join(failed) if failed else 'none'}"
                 f"{f' (+{missing} not built)' if missing > 0 else ''}.")
    lines.append("")

    Path(args.out).write_text("\n".join(lines))
    print(f"Wrote {args.out}: {len(rows)} scenes, {total_instances} instances, "
          f"{total_dups} cross-class duplicates, max cross-IoU {max_cross_iou}.")

    ok = True
    if failed or missing > 0:
        print(f"GATE FAIL: {len(failed)} failed scenes, {missing} missing.")
        ok = False
    if total_dups != 0:
        print(f"GATE FAIL: {total_dups} cross-class duplicates (must be 0).")
        ok = False
    if not (1000 <= total_instances <= 50000):
        print(f"GATE FAIL: implausible total instance count {total_instances}.")
        ok = False
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
