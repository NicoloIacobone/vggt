#!/usr/bin/env python3
"""
CPU tests for the 3D-viewer colour path (`train/maskdino_viz3d.py`), the plumbing behind
`demos/demo_gradio.py --seg_checkpoint <maskdino run>/checkpoint_best_bundle.pth`.

What must hold for the viewer to be worth looking at:
  - a MaskDINO checkpoint is told apart from a legacy D4RT one (the demo serves both);
  - `--feature_mode` is honoured, because a single-frame checkpoint fed bundle tokens is
    silently off-distribution;
  - query selection matches the 3D ruler's rule: one score per query = max over views;
  - an instance keeps its colour across views (the same regression `tests/test_maskdino_viz.py`
    guards for the 2D figures — in 3D a colour that flips per view is worse than useless);
  - the whole path runs end to end on a real (tiny) head.

    python tests/test_maskdino_viz3d.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from maskdino_fixtures import _tiny_head
from train.maskdino_eval import NUM_VIZ_COLORS, color_index
from train.maskdino_viz3d import (assign_map, colorize, head_features, identity_palette,
                                  is_maskdino_checkpoint, maskdino_seg_colors,
                                  parse_feature_layers, select_instances)

torch.manual_seed(0)


def test_checkpoint_discrimination():
    print("=== Testing checkpoint discrimination ===")
    assert is_maskdino_checkpoint({"head_config": {}, "head_state_dict": {}, "args": {}})
    # a legacy D4RT checkpoint has head_config too — the state-dict key is what separates them
    assert not is_maskdino_checkpoint({"head_config": {}, "decoder_head_state_dict": {}})
    assert not is_maskdino_checkpoint({"model": {}})
    assert not is_maskdino_checkpoint(None)
    print("✓ MaskDINO vs D4RT checkpoints told apart\n")


def test_parse_feature_layers():
    print("=== Testing --feature_layers parsing ===")
    assert parse_feature_layers({}) == [-1]
    assert parse_feature_layers({"feature_layers": [4, 11, 23]}) == [4, 11, 23]
    assert parse_feature_layers({"feature_layers": "-1"}) == [-1]        # un-parsed string form
    assert parse_feature_layers({"feature_layers": "4,11,23"}) == [4, 11, 23]
    print("✓ list and string forms both parse\n")


def test_head_features_honours_feature_mode():
    """`bundle` = one pass over all frames; `single` = S one-frame passes (what it was trained on)."""
    print("=== Testing feature extraction modes ===")
    S, P, C = 4, 7, 6
    calls = []

    def fake_aggregator(imgs):
        calls.append(tuple(imgs.shape))
        n = imgs.shape[1]
        # a token block whose value encodes the frame index, so we can check the assembly
        base = torch.arange(n, dtype=torch.float32).view(1, n, 1, 1).expand(1, n, P, C).clone()
        return [torch.zeros(1, n, P, C), base], 5

    images = torch.zeros(S, 3, 8, 8)

    calls.clear()
    feats, psi = head_features(fake_aggregator, images, {"feature_mode": "bundle"})
    assert feats.shape == (S, P, C) and psi == 5, feats.shape
    assert calls == [(1, S, 3, 8, 8)], f"bundle must be ONE pass over all frames: {calls}"
    assert torch.equal(feats[:, 0, 0], torch.arange(S, dtype=torch.float32))

    calls.clear()
    feats, _ = head_features(fake_aggregator, images, {"feature_mode": "single"})
    assert feats.shape == (S, P, C)
    assert calls == [(1, 1, 3, 8, 8)] * S, f"single must be S one-frame passes: {calls}"
    assert torch.equal(feats[:, 0, 0], torch.zeros(S)), "each single pass sees frame 0 only"

    # multiple --feature_layers concatenate on the channel axis, like train/maskdino_data
    feats, _ = head_features(fake_aggregator, images,
                             {"feature_mode": "bundle", "feature_layers": [0, 1]})
    assert feats.shape == (S, P, 2 * C), feats.shape
    print("✓ bundle / single / multi-layer feature assembly\n")


def test_select_instances_is_the_3d_rulers_rule():
    print("=== Testing query selection ===")
    S, Q = 3, 5
    logits = torch.full((S, Q, 19), -10.0)
    logits[0, 0, 4] = 3.0        # query 0: chair (idx 5), strong in view 0 only
    logits[2, 1, 0] = 2.0        # query 1: wall (idx 1), strong in view 2 only
    logits[1, 2, 6] = 0.2        # query 2: table, weak (sigmoid 0.55)
    # queries 3, 4 stay background everywhere

    keep, labels, scores = select_instances(logits, score_threshold=0.25, topk=None)
    assert keep.tolist() == [0, 1, 2], f"max-over-views selection failed: {keep.tolist()}"
    assert labels.tolist() == [5, 1, 7], labels.tolist()
    assert scores[0] > scores[1] > scores[2], "keep must be sorted by descending score"

    # a view where an instance is invisible must not veto it: per-view selection would keep
    # nothing in view 1, and the whole point of a shared query set is scene-level identity
    assert 0 in keep.tolist() and 1 in keep.tolist()

    high, _, _ = select_instances(logits, score_threshold=0.9, topk=None)
    assert high.tolist() == [0], f"threshold not applied: {high.tolist()}"
    assert select_instances(logits, 0.25, topk=2)[0].tolist() == [0, 1], "topk not applied"

    nostuff, nolabels, _ = select_instances(logits, 0.25, topk=None, drop_stuff=True)
    assert nostuff.tolist() == [0, 2] and 1 not in nolabels.tolist(), "wall not dropped"

    empty, _, _ = select_instances(torch.full((S, Q, 19), -10.0), 0.25)
    assert empty.numel() == 0
    print("✓ max-over-views scores, threshold, top-k, stuff drop, empty case\n")


def test_colour_is_identity_keyed_across_views():
    """
    THE property. Two views of one bundle where the queries swap rank and one vanishes: the
    colour of a query must not move. Rank-based colouring would repaint the whole scene between
    views — in a 3D cloud that shows up as one object wearing two colours.
    """
    print("=== Testing identity-keyed colouring across views ===")
    S, Q, h = 2, 3, 4
    masks = torch.full((S, Q, h, h), -10.0)
    masks[0, 0, 0] = 5.0      # view 0: query 0 owns row 0
    masks[0, 2, 1] = 5.0      #         query 2 owns row 1
    masks[1, 2, 0] = 5.0      # view 1: query 2 has moved to row 0, query 0 is gone
    masks[1, 1, 2] = 5.0      #         query 1 appears

    assign = assign_map(masks, [2, 0, 1], (h, h))    # keep order = score order, not identity
    assert assign.shape == (S, h, h)
    assert (assign[0, 0] == 0).all() and (assign[0, 1] == 2).all(), assign[0]
    assert (assign[1, 0] == 2).all() and (assign[1, 2] == 1).all(), assign[1]
    assert (assign[0, 2] == -1).all(), "sub-threshold pixels must stay unassigned"

    images = torch.zeros(S, 3, h, h)
    colors = colorize(images, assign)
    assert colors.shape == (S, h, h, 3) and colors.dtype == np.uint8
    q2_view0, q2_view1 = colors[0, 1, 0], colors[1, 0, 0]
    assert np.array_equal(q2_view0, q2_view1), (
        f"query 2 changed colour between views: {q2_view0} vs {q2_view1}")
    assert not np.array_equal(colors[0, 0, 0], q2_view0), "distinct queries share a colour"

    # the palette is the figures' palette: same query → same tab20 slot in PNG and viewer
    palette = identity_palette()
    assert palette.shape == (NUM_VIZ_COLORS + 1, 3)
    assert np.array_equal(colors[0, 0, 0], palette[color_index(0)])
    assert np.array_equal(q2_view0, palette[color_index(2)])

    # background keeps its RGB (the room stays visible in the cloud)
    images = torch.full((S, 3, h, h), 0.5)
    assert np.array_equal(colorize(images, assign)[0, 2, 0], np.array([128, 128, 128])), \
        "unassigned pixels must keep the image colour"

    empty = assign_map(masks, [], (h, h))
    assert (empty == -1).all(), "no kept queries → nothing painted"
    print("✓ colour follows the query index, not its per-view rank\n")


def test_end_to_end_with_a_real_head():
    print("=== Testing end-to-end on a tiny head ===")
    S, P, hw = 3, 5 + 36, 64
    head = _tiny_head(two_stage=False, dn="no", learn_tgt=True, initialize_box_type="no")
    head.eval()
    tokens = torch.randn(S, P, 64)
    with torch.no_grad():
        out, _ = head(tokens, 5, None, frames_per_sample=S)   # one shared query set, S frames
    assert out["pred_logits"].shape[0] == S

    images = torch.rand(S, 3, hw, hw)
    colors, legend = maskdino_seg_colors(out, images, score_threshold=0.0, mask_threshold=0.5)
    assert colors.shape == (S, hw, hw, 3) and colors.dtype == np.uint8, colors.shape
    assert "colour = query id" in legend, legend

    # an impossible threshold must degrade to "the raw images", not crash
    colors, legend = maskdino_seg_colors(out, images, score_threshold=1.1)
    assert colors.shape == (S, hw, hw, 3) and "No instances" in legend, legend
    assert np.array_equal(colors,
                          (images.permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8)), \
        "with nothing kept the frames must come back untouched"

    # a [1, S, 3, H, W] batch (the demo's own tensor layout) is accepted
    colors, _ = maskdino_seg_colors(out, images.unsqueeze(0), score_threshold=0.0)
    assert colors.shape == (S, hw, hw, 3)
    print("✓ real head → colours + legend, degenerate thresholds handled\n")


if __name__ == "__main__":
    test_checkpoint_discrimination()
    test_parse_feature_layers()
    test_head_features_honours_feature_mode()
    test_select_instances_is_the_3d_rulers_rule()
    test_colour_is_identity_keyed_across_views()
    test_end_to_end_with_a_real_head()
    print("✅ All 3D-viewer colour-path tests passed")
