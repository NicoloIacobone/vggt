#!/usr/bin/env python3
"""
Collect the cross-dataset evaluation matrix (docs/todo.md 6d) from the eval JSONs into the
markdown table that lands in `docs/RESULTS.md` §7. CPU-only, reads nothing but json.

Scans each run directory for `eval3d_*.json`, keeps only the cells run at **defaults** (every
result-affecting knob at its default except `--dataset` and `--transfer_mode`, so a tuned or
subset run can never leak into the matrix), and prints one row per
(checkpoint x dataset x transfer mode).

    myenv/bin/python scripts/collect_eval3d_matrix.py            # markdown
    myenv/bin/python scripts/collect_eval3d_matrix.py --json     # the raw cells
    myenv/bin/python scripts/collect_eval3d_matrix.py \
        --run 'B ScanNet=maskdino_multi_scannet_n1201_a3d_e35_20260821_201002' --only

`--run LABEL=DIR` (repeatable) adds a run the file does not name — the multi-dataset arms of
`docs/MULTIDATASET.md` §10 postdate it — and `--only` drops the three built-in rows so the two
blocks are never printed as one table. DIR is a name under the output root or an absolute path.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT = Path("/cluster/work/igp_psr/niacobone/distillation/output")
RUNS = {                                      # label -> run dir (docs/RESULTS.md §5/§6)
    "mf (1201 control)": "maskdino_sf_list1201_mf_20260802_133826",
    "--anchor_3d": "maskdino_sf_list1201_mf_anchor3d_20260804_171436",
    "--num_frames 16": "maskdino_sf_list1201_mf_s16_20260805_095016",
}
DATASETS = ("scannetv2", "scannet200", "scannetpp", "replica")
MODES = ("unproject", "gt_projection")
# every result-affecting knob except the two the matrix varies
DEFAULTS = {"num_frames": None, "eval_topk": 100, "min_score": 0.0,
            "mask_prob_threshold": 0.5, "depth_tolerance": 0.1, "vote_radius": 0.05,
            "depth_conf_percentile": 0.0, "icp": True, "icp_max_dist": 0.3}


def is_default_cell(args: dict) -> bool:
    return all(args.get(k, v) == v for k, v in DEFAULTS.items())


def triple(block):
    if not block:
        return None
    return (block["all_ap"], block["all_ap_50%"], block["all_ap_25%"])


def fmt(t):
    return "—" if t is None else " / ".join(f"{x:.3f}" for x in t)


def parse_runs(pairs, only: bool):
    """`LABEL=DIR` pairs -> the RUNS mapping to scan. Insertion order is the table's row order."""
    runs = {} if only else dict(RUNS)
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--run wants LABEL=DIR, got {pair!r}")
        label, _, run = pair.partition("=")
        label, run = label.strip(), run.strip()
        if not label or not run:
            raise SystemExit(f"--run wants a non-empty LABEL and DIR, got {pair!r}")
        runs[label] = run
    if not runs:
        raise SystemExit("--only was given without any --run")
    return runs


def collect(runs=None):
    runs = RUNS if runs is None else runs
    cells = {}
    for label, run in runs.items():
        for path in sorted((OUT / run).glob("eval3d_*.json")):
            try:
                d = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            if "results_class_agnostic" not in d or not is_default_cell(d.get("args", {})):
                continue
            key = (label, d.get("dataset", "scannetv2"), d["transfer_mode"])
            cells[key] = {
                "file": path.name,
                "scenes": d["num_scenes"],
                "failed": len(d.get("failed_scenes", [])),
                "class_agnostic": triple(d.get("results_class_agnostic")),
                "class_aware": triple(d.get("results_18class")),
                "frames": round(sum(s.get("frames", 0) for s in d["per_scene"].values())
                                / max(len(d["per_scene"]), 1), 1),
            }
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--run", action="append", metavar="LABEL=DIR",
                    help="add a run the file does not name; repeatable")
    ap.add_argument("--only", action="store_true",
                    help="scan ONLY the --run entries, dropping the three built-in rows")
    args = ap.parse_args()
    runs = parse_runs(args.run, args.only)
    cells = collect(runs)
    if args.json:
        print(json.dumps({" | ".join(k): v for k, v in cells.items()}, indent=2))
        return 0

    print("| checkpoint | dataset | scenes | views/scene | unposed (`unproject`) | "
          "posed (`gt_projection`) |")
    print("|---|---|---|---|---|---|")
    for label in runs:
        for ds in DATASETS:
            up = cells.get((label, ds, "unproject"))
            gp = cells.get((label, ds, "gt_projection"))
            any_cell = up or gp
            if not any_cell:
                continue
            print(f"| {label} | {ds} | {any_cell['scenes']} | {any_cell['frames']} | "
                  f"{fmt(up and up['class_agnostic'])} | {fmt(gp and gp['class_agnostic'])} |")
    print("\nAP / AP50 / AP25, CLASS-AGNOSTIC (the only setting all four datasets share).")
    missing = [f"{lab} x {ds} x {m}" for lab in runs for ds in DATASETS for m in MODES
               if (lab, ds, m) not in cells]
    if missing:
        print(f"\nmissing {len(missing)} cell(s): " + ", ".join(missing))
    print("\nclass-aware ScanNetv2 (the headline ruler, §5), for reference:")
    for label in runs:
        for m in MODES:
            c = cells.get((label, "scannetv2", m))
            if c and c["class_aware"]:
                print(f"  {label:20s} {m:14s} {fmt(c['class_aware'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
