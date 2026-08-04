#!/usr/bin/env python3
"""
Turn a `.ply` into a single self-contained HTML file you can open in any browser.

Meant for the 3D ruler's `--dump_ply` output (docs/MASKDINO.md §9.7): the benchmark mesh's
vertices coloured by predicted instance, grey where no instance reached them. No MeshLab, no
CloudCompare, no Python on the machine that does the looking — the HTML embeds the points and
its own WebGL viewer (`demos/dualview3d.py`), so it works offline and travels as one file.

    myenv/bin/python scripts/view_ply.py <run_dir>/eval3d_scene0011_00.ply
    → <run_dir>/eval3d_scene0011_00.html      (then: scp it and double-click)

Two files side by side get ONE shared camera — orbit either panel and both move together:

    myenv/bin/python scripts/view_ply.py a.ply b.ply --out compare.html

The viewer subsamples to --max_points (default 200k) so the file stays a few MB; pass a bigger
number for a denser picture and a heavier file.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demos"))

from dualview3d import DEFAULT_MAX_POINTS, standalone_html


def load_ply(path):
    """
    `.ply` → (xyz [N, 3] float64, rgb [N, 3] uint8).

    Colours are optional: an uncoloured cloud comes back light grey rather than failing, so a
    plain geometry dump is still viewable.
    """
    import trimesh

    mesh = trimesh.load(str(path), process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    colors = getattr(getattr(mesh, "visual", None), "vertex_colors", None)
    if colors is None or len(colors) != len(vertices):
        rgb = np.full((len(vertices), 3), 200, dtype=np.uint8)
    else:
        rgb = np.asarray(colors, dtype=np.uint8)[:, :3]
    return vertices, rgb


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ply", nargs="+", help="one or two .ply files (two → synced side by side)")
    p.add_argument("--out", type=str, default=None,
                   help="output .html (default: next to the first .ply, same stem)")
    p.add_argument("--labels", type=str, nargs="*", default=None,
                   help="panel titles (default: the file stems)")
    p.add_argument("--max_points", type=int, default=DEFAULT_MAX_POINTS)
    p.add_argument("--point_size", type=float, default=2.0)
    return p


def main():
    args = build_argparser().parse_args()
    paths = [Path(p) for p in args.ply]
    for path in paths:
        if not path.exists():
            raise SystemExit(f"not found: {path}")
    labels = args.labels or [p.stem for p in paths]
    if len(labels) != len(paths):
        raise SystemExit(f"{len(labels)} label(s) for {len(paths)} file(s)")

    panels = []
    for path, label in zip(paths, labels):
        xyz, rgb = load_ply(path)
        grey = int((rgb.std(axis=1) < 1).sum())     # unassigned vertices are painted grey
        print(f"{path.name}: {len(xyz):,} vertices, {grey:,} ({grey / max(1, len(xyz)):.0%}) "
              f"grey/unassigned")
        panels.append({"label": label, "note": f"{len(xyz):,} vertices",
                       "points": xyz, "colors": rgb})

    out = Path(args.out) if args.out else paths[0].with_suffix(".html")
    out.write_text(standalone_html(panels, title=" | ".join(labels),
                                   max_points=args.max_points, point_size=args.point_size,
                                   canvas_height=560))
    size_mb = out.stat().st_size / 1e6
    print(f"\n✓ {out}  ({size_mb:.1f} MB, self-contained)")
    print("  open it by copying it to your machine:  scp <this file> . && open it in a browser")
    print("  or serve it from here:  python -m http.server 8000 --directory "
          f"{out.parent}   → http://localhost:8000/{out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
