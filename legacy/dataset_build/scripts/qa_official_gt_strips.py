"""Visual QA for the official-GT build: per-scene overlay strips (Phase-2 gate 3).

For each requested scene, renders one row per sampled frame: RGB | instance
overlay, with a FIXED color per instance dir across all frames — a color that
stays on the same object across the strip demonstrates cross-view identity
consistency. Output: <build>/qa_strips/<scene>_strip.jpg

Usage:
    myenv/bin/python legacy/dataset_build/scripts/qa_official_gt_strips.py \
        --build /cluster/scratch/niacobone/scannet_official_build \
        --scenes scene0000_00,scene0040_00,scene0080_00,scene0120_00,scene0160_00
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def render_scene(raw_dir: Path, out_path: Path, num_frames: int = 6) -> None:
    subset = sorted((raw_dir / "subset").glob("*.jpg"))
    idxs = np.unique(np.linspace(0, len(subset) - 1, num_frames).round().astype(int))
    frames = [subset[i] for i in idxs]

    seg_dirs = sorted(d for d in (raw_dir / "masks_instance").iterdir()
                      if d.is_dir() and not d.name.startswith("_"))
    rng = np.random.default_rng(0)
    colors = {d.name: rng.integers(40, 255, 3) for d in seg_dirs}

    fig, axes = plt.subplots(len(frames), 2, figsize=(14, 5 * len(frames)))
    for r, fp in enumerate(frames):
        rgb = np.array(Image.open(fp).convert("RGB"))
        overlay = rgb.astype(float)
        for d in seg_dirs:
            mp = d / f"{fp.stem}.png"
            if not mp.exists():
                continue
            m = np.array(Image.open(mp)) > 127
            overlay[m] = 0.45 * overlay[m] + 0.55 * colors[d.name]
        axes[r, 0].imshow(rgb)
        axes[r, 0].set_title(f"{fp.stem}.jpg", fontsize=9)
        axes[r, 1].imshow(overlay.astype(np.uint8))
        axes[r, 1].set_title("official-GT instances (fixed color per id)", fontsize=9)
        for ax in axes[r]:
            ax.axis("off")
    # legend: instance dir -> color
    handles = [plt.Line2D([0], [0], marker="s", ls="", markersize=8,
                          markerfacecolor=np.array(c) / 255, label=n)
               for n, c in colors.items()]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8)
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=80)
    plt.close(fig)
    print(f"wrote {out_path} ({len(seg_dirs)} instances)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="/cluster/scratch/niacobone/scannet_official_build")
    ap.add_argument("--scenes", default="scene0000_00,scene0040_00,scene0080_00,"
                                        "scene0120_00,scene0160_00")
    ap.add_argument("--num_frames", type=int, default=6)
    args = ap.parse_args()

    build = Path(args.build)
    for scene in args.scenes.split(","):
        render_scene(build / "scans" / scene / "raw_data",
                     build / "qa_strips" / f"{scene}_strip.jpg", args.num_frames)


if __name__ == "__main__":
    main()
