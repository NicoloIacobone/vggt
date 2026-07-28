"""
CPU tests for the top-down point-cloud renderer (legacy/d4rt/scripts/render_pointcloud_topdown.py).

Covers only the pure rendering core — no backbone, no checkpoint, no GPU:
  - project_topdown: axis/sign handling and height-based draw ordering
  - instance_color_array: background gray vs palette colors
  - render_topdown_pair: writes a non-trivial two-panel PNG from a synthetic scene
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))            # repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from render_pointcloud_topdown import (  # noqa: E402
    BG_GRAY, estimate_up, instance_color_array, project_topdown, render_topdown_pair,
)
from visualize_masks import _color  # noqa: E402


def test_project_topdown_axes_and_order():
    pts = np.array([
        [0.0, 2.0, 0.0],   # y=2 → low in "-y" convention (+y is down)
        [1.0, -1.0, 3.0],  # y=-1 → high
        [2.0, 0.5, 1.0],   # middle
    ])

    xy, order = project_topdown(pts, up_axis="-y")
    assert xy.shape == (3, 2)
    # Draw order = ascending height = descending y for "-y": indices 0 (y=2), 2, 1.
    assert order.tolist() == [0, 2, 1]
    # Orthographic projection ⊥ up: points differing only along y project to the same xy,
    # and in-plane (x,z) distances are preserved.
    same_col = project_topdown(np.array([[1.0, 0.0, 2.0], [1.0, 5.0, 2.0]]), up_axis="-y")[0]
    assert np.allclose(same_col[0], same_col[1])
    d3 = np.linalg.norm(pts[0, [0, 2]] - pts[1, [0, 2]])
    assert np.isclose(np.linalg.norm(xy[0] - xy[1]), d3)

    # "+y" flips the ordering.
    _, order_up = project_topdown(pts, up_axis="+y")
    assert order_up.tolist() == [1, 2, 0]

    try:
        project_topdown(pts, up_axis="w")
        raise AssertionError("expected ValueError for bad up_axis")
    except ValueError:
        pass
    print("✓ project_topdown: axes, sign, ordering, rigidity")


def test_estimate_up_recovers_room_axis():
    pts, _, _ = _synthetic_room()
    up = estimate_up(pts)
    # Room extends 4x4 in (x,z) and <1 in y with the floor at +y → up ≈ (0,-1,0):
    # smallest-variance axis is y, sign flipped so the dense floor slab sits at low height.
    assert np.dot(up, [0.0, -1.0, 0.0]) > 0.99, f"estimated up {up}"
    # 'auto' path in project_topdown orders floor (y=1.5) before box tops (y<1.5).
    probe = np.array([[1.0, 1.5, 1.0], [1.0, 0.9, 1.0]])
    _, order = project_topdown(np.concatenate([pts, probe]), up_axis="auto")
    assert list(order).index(len(pts)) < list(order).index(len(pts) + 1)
    print("✓ estimate_up: recovers -y on the synthetic room; auto ordering floor-first")


def test_estimate_up_hinted_ransac_finds_floor():
    # A scan dominated by one wall (wide in x, tall in y, shallow in z) plus a visible
    # floor strip at y=2 — the realistic partial-ScanNet-bundle case where pure PCA picks
    # the wall normal (z), which is WRONG as "up".
    rng = np.random.default_rng(1)
    n = 50000
    wall = np.column_stack([rng.uniform(0, 4, n), rng.uniform(0, 2, n), rng.uniform(0, 0.3, n)])
    floor = np.column_stack([rng.uniform(0, 4, n // 2),
                             np.full(n // 2, 2.0) + rng.normal(0, 0.005, n // 2),
                             rng.uniform(0, 1.5, n // 2)])
    scan = np.concatenate([wall, floor])

    # A tilted camera-up hint (gravity ≈ -y here, hint ~8° off) must snap near the true
    # floor normal via the slab-concentration search (grid resolution ~2.5°).
    up_hinted = estimate_up(scan, hint=np.array([0.05, -1.0, 0.1]))
    assert np.dot(up_hinted, [0.0, -1.0, 0.0]) > 0.998, f"hinted up {up_hinted}"

    # Zero search cone → returns the (normalized) hint itself.
    up_fb = estimate_up(scan, hint=np.array([0.0, -2.0, 0.0]), max_tilt_deg=0.0)
    assert np.allclose(up_fb, [0.0, -1.0, 0.0])

    # Explicit vector up_axis is accepted by project_topdown.
    _, order = project_topdown(scan, up_axis=up_hinted)
    assert order.shape == (scan.shape[0],)
    print("✓ estimate_up: hinted slab search snaps to the floor normal; tilt=0 returns hint")


def test_instance_color_array():
    rng = np.random.default_rng(0)
    rgb = rng.random((100, 3))
    assign = np.full(100, -1, dtype=np.int64)
    assign[10:20] = 0
    assign[50:55] = 3

    colors = instance_color_array(assign, rgb, gray_background=True)
    assert np.allclose(colors[0], BG_GRAY)          # background → gray
    assert np.allclose(colors[15], _color(0))       # instance 0 → palette 0
    assert np.allclose(colors[52], _color(3))       # instance 3 → palette 3

    colors_rgb = instance_color_array(assign, rgb, gray_background=False)
    assert np.allclose(colors_rgb[0], rgb[0])       # background keeps RGB
    assert np.allclose(colors_rgb[15], _color(0))
    print("✓ instance_color_array: gray/rgb background + palette mapping")


def _synthetic_room(n=20000, seed=0):
    """A floor plane with two colored box 'objects' on it (VGGT frame: +y down)."""
    rng = np.random.default_rng(seed)
    floor = np.column_stack([rng.uniform(0, 4, n), np.full(n, 1.5), rng.uniform(0, 4, n)])
    box1 = np.column_stack([rng.uniform(0.5, 1.5, n // 10),
                            rng.uniform(0.8, 1.5, n // 10),
                            rng.uniform(0.5, 1.5, n // 10)])
    box2 = np.column_stack([rng.uniform(2.5, 3.5, n // 10),
                            rng.uniform(1.0, 1.5, n // 10),
                            rng.uniform(2.5, 3.5, n // 10)])
    pts = np.concatenate([floor, box1, box2])
    rgb = np.concatenate([
        np.tile([0.7, 0.6, 0.5], (n, 1)),
        np.tile([0.8, 0.1, 0.1], (n // 10, 1)),
        np.tile([0.1, 0.3, 0.8], (n // 10, 1)),
    ])
    assign = np.concatenate([
        np.full(n, -1, dtype=np.int64),
        np.zeros(n // 10, dtype=np.int64),
        np.ones(n // 10, dtype=np.int64),
    ])
    return pts, rgb, assign


def test_render_topdown_pair_writes_png():
    pts, rgb, assign = _synthetic_room()
    inst_colors = instance_color_array(assign, rgb)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "sub" / "scene_topdown.png"  # exercises parent-dir creation
        render_topdown_pair(pts, rgb, inst_colors, out, point_size=1.0, dpi=100)
        assert out.exists(), "figure not written"
        assert out.stat().st_size > 10_000, f"figure suspiciously small: {out.stat().st_size} B"
    print("✓ render_topdown_pair: two-panel PNG written from synthetic scene")


if __name__ == "__main__":
    test_project_topdown_axes_and_order()
    test_estimate_up_recovers_room_axis()
    test_estimate_up_hinted_ransac_finds_floor()
    test_instance_color_array()
    test_render_topdown_pair_writes_png()
    print("\nAll render_topdown tests passed ✓")
