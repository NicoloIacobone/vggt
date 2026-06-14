#!/usr/bin/env python3
"""
Plot the scaling curve (val metrics vs #train scenes N) from the per-run metrics.jsonl files
written by scripts/train_multiscene.py (Phase 0.1). Reads each run's eval history, takes the
BEST and FINAL held-out numbers, and renders:

  (1) held-out val mIoU (prompted, best) and val[grid] AP50 (the honest unprompted detection
      number, best) vs N;
  (2) the train-minus-val mIoU gap vs N (overfitting / data-limitation signal, §3.1).

Usage:
    python scripts/plot_scaling.py \
        --runs 10:/path/scale10 25:/path/scale25 50:/path/scale50 \
        --baseline 4:0.138 \
        --out /path/scaling_curve.png

`--runs` items are `N:run_dir` (run_dir holds metrics.jsonl). `--baseline` items are
`N:val_mIoU` literals for points that predate metrics.jsonl (e.g. N=4 from MILESTONE_2 §6).
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless cluster node
import matplotlib.pyplot as plt
import matplotlib.ticker


def read_run(run_dir: str) -> dict:
    """Summarize one run's metrics.jsonl into best/final held-out + train numbers."""
    path = Path(run_dir) / "metrics.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no metrics.jsonl in {run_dir}")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError(f"empty metrics.jsonl in {run_dir}")

    def best(key):
        vals = [(r.get(key, float("nan")), r["epoch"]) for r in records]
        return max(vals, key=lambda t: (t[0] if t[0] == t[0] else -1))  # NaN-safe max

    last = records[-1]
    bm, bm_ep = best("val_mIoU")
    bap, bap_ep = best("val_grid_AP50")
    return {
        "n_evals": len(records),
        "val_mIoU_best": bm, "val_mIoU_best_epoch": bm_ep,
        "val_grid_AP50_best": bap, "val_grid_AP50_best_epoch": bap_ep,
        "val_mIoU_final": last.get("val_mIoU"),
        "train_mIoU_final": last.get("train_mIoU"),
    }


def main():
    ap = argparse.ArgumentParser(description="Plot the D4RT scaling curve from metrics.jsonl")
    ap.add_argument("--runs", nargs="+", required=True, metavar="N:run_dir",
                    help="One or more N:run_dir (run_dir contains metrics.jsonl)")
    ap.add_argument("--baseline", nargs="*", default=[], metavar="N:val_mIoU",
                    help="Optional literal N:val_mIoU points predating metrics.jsonl")
    ap.add_argument("--out", type=str, default="scaling_curve.png")
    args = ap.parse_args()

    points = []  # (N, summary-dict-or-baseline)
    for item in args.runs:
        n, run_dir = item.split(":", 1)
        s = read_run(run_dir)
        points.append((int(n), s))
        print(f"N={n}: best val mIoU={s['val_mIoU_best']:.3f} @ep{s['val_mIoU_best_epoch']}, "
              f"best val[grid] AP50={s['val_grid_AP50_best']:.3f} @ep{s['val_grid_AP50_best_epoch']}, "
              f"final train mIoU={s['train_mIoU_final']:.3f}, val mIoU={s['val_mIoU_final']:.3f} "
              f"({s['n_evals']} evals)")
    baselines = []
    for item in args.baseline:
        n, v = item.split(":", 1)
        baselines.append((int(n), float(v)))
        print(f"N={n}: baseline val mIoU={float(v):.3f} (literal)")

    points.sort(key=lambda t: t[0])
    Ns = [n for n, _ in points]
    val_miou = [s["val_mIoU_best"] for _, s in points]
    val_ap50 = [s["val_grid_AP50_best"] for _, s in points]
    gap = [(s["train_mIoU_final"] - s["val_mIoU_final"])
           if s["train_mIoU_final"] is not None and s["val_mIoU_final"] is not None else None
           for _, s in points]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(Ns, val_miou, "o-", color="#1f77b4", label="val mIoU (prompted, best)")
    ax1.plot(Ns, val_ap50, "s-", color="#d62728", label="val[grid] AP50 (best, honest)")
    for n, v in baselines:
        ax1.plot([n], [v], "D", color="#1f77b4", markersize=9, fillstyle="none",
                 label=f"val mIoU baseline (N={n})")
    ax1.set_xlabel("number of training scenes N")
    ax1.set_ylabel("held-out metric")
    ax1.set_title("Scaling: held-out val metrics vs N")
    ax1.set_xscale("log")
    ax1.set_xticks(Ns + [n for n, _ in baselines])
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    gap_n = [n for n, g in zip(Ns, gap) if g is not None]
    gap_v = [g for g in gap if g is not None]
    ax2.plot(gap_n, gap_v, "^-", color="#2ca02c")
    ax2.set_xlabel("number of training scenes N")
    ax2.set_ylabel("final train mIoU − val mIoU")
    ax2.set_title("Train–val gap vs N (overfitting signal)")
    ax2.set_xscale("log")
    ax2.set_xticks(gap_n)
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\n✓ Wrote {out}")


if __name__ == "__main__":
    main()
