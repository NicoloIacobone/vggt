#!/usr/bin/env python3
"""
CPU tests for the visualisation colour path (`train/maskdino_eval.py`).

The property under test is the one the figures exist for: **an instance keeps its colour across
the frames of a bundle**. The original code coloured by the instance's rank inside the
per-frame kept list, which is re-filtered and re-sorted every frame, so the same query changed
colour from view to view and the multi-view consistency the model actually has was invisible.

    python tests/test_maskdino_viz.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.maskdino_eval import (NUM_VIZ_COLORS, color_index, identity_cmap,
                                 paint_identity_map)


def test_color_index_stable_and_in_range():
    """Same identity → same slot, always inside 1..NUM_VIZ_COLORS, background never claimed."""
    for ident in range(0, 500):
        c = color_index(ident)
        assert 1 <= c <= NUM_VIZ_COLORS, f"identity {ident} → slot {c} outside 1..20"
        assert c == color_index(ident), "color_index is not deterministic"
    # Distinct identities inside one palette cycle must not collide.
    slots = {color_index(i) for i in range(NUM_VIZ_COLORS)}
    assert len(slots) == NUM_VIZ_COLORS, f"collisions inside one cycle: {sorted(slots)}"
    assert color_index(0) != 0, "slot 0 is reserved for background"
    print("✓ color_index: stable, in range, no collisions inside a cycle")


def test_paint_winner_takes_all():
    """The highest-probability mask claims the pixel; sub-threshold pixels stay background."""
    masks = torch.zeros(2, 4, 4)
    masks[0, :, :2] = 0.9        # left half, strong
    masks[1, :, 1:3] = 0.7       # overlaps column 1, weaker
    out = paint_identity_map(masks, [3, 7])

    assert out[0, 0].item() == color_index(3), "uncontested pixel went to the wrong identity"
    assert out[0, 1].item() == color_index(3), "0.9 should beat 0.7 on the overlapping column"
    assert out[0, 2].item() == color_index(7), "identity 7 should own its uncontested column"
    assert out[0, 3].item() == 0, "a pixel below threshold must stay background"
    print("✓ paint_identity_map: winner-takes-all, threshold respected")


def test_color_survives_reordering_and_filtering():
    """
    THE regression test. Two frames of one bundle: the same three queries, but in frame 2 the
    scores reorder them and one drops below the score threshold. Under the old rank-based
    colouring every surviving query changed colour; keyed to the query index, nothing moves.
    """
    h = w = 6
    # frame 1: queries 5, 12, 40 kept, in that (score) order.
    f1 = torch.zeros(3, h, w)
    f1[0, 0] = 0.9      # query 5  → row 0
    f1[1, 1] = 0.8      # query 12 → row 1
    f1[2, 2] = 0.6      # query 40 → row 2
    map1 = paint_identity_map(f1, [5, 12, 40])

    # frame 2: same objects, scores reordered (40, 5) and query 12 filtered out entirely.
    f2 = torch.zeros(2, h, w)
    f2[0, 2] = 0.95     # query 40 → row 2
    f2[1, 0] = 0.7      # query 5  → row 0
    map2 = paint_identity_map(f2, [40, 5])

    assert map1[0, 0] == map2[0, 0], "query 5 changed colour between frames"
    assert map1[2, 0] == map2[2, 0], "query 40 changed colour between frames"
    assert map2[1, 0] == 0, "the dropped query should leave background, not a recoloured pixel"
    # And the two objects must still be told apart within a frame.
    assert map1[0, 0] != map1[2, 0], "distinct queries collapsed to one colour"
    print("✓ colours survive per-frame reordering and filtering")


def test_empty_and_singleton_inputs():
    """Zero predictions is a legitimate frame (everything filtered out), not a crash."""
    out = paint_identity_map(torch.zeros(0, 5, 5), [])
    assert out.shape == (5, 5) and int(out.sum()) == 0, "empty input must give a blank map"

    binary = torch.zeros(1, 5, 5)
    binary[0, 2, 2] = 1.0
    out = paint_identity_map(binary, [17])
    assert out[2, 2].item() == color_index(17), "binary GT masks must paint too"
    assert int((out > 0).sum()) == 1, "a single-pixel mask painted more than one pixel"
    print("✓ empty and binary/singleton inputs handled")


def test_cmap_covers_every_slot():
    """The colormap must have one entry per slot, background included, for vmin=0/vmax=20."""
    import matplotlib
    matplotlib.use("Agg")

    cmap = identity_cmap()
    assert cmap.N == NUM_VIZ_COLORS + 1, f"cmap has {cmap.N} colours, expected 21"
    bg = cmap(0.0)[:3]
    assert all(c < 0.2 for c in bg), f"slot 0 should be a dark background, got {bg}"
    print("✓ identity_cmap: 21 slots, dark background at 0")


def main():
    print("=" * 70)
    print("MaskDINO visualisation colour tests")
    print("=" * 70)
    test_color_index_stable_and_in_range()
    test_paint_winner_takes_all()
    test_color_survives_reordering_and_filtering()
    test_empty_and_singleton_inputs()
    test_cmap_covers_every_slot()
    print("=" * 70)
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
