#!/usr/bin/env python3
"""
Cross-view consistency metrics (docs/MASKDINO.md §6.6). Standalone, CPU-only.

  - the planted-perfect case: one query owns the instance in every view → consistency 1.0,
    id_switch 0.0;
  - the planted ID switch: a different query owns the object in one view → consistency drops
    and id_switch rises, by exactly one view's worth;
  - the case bundle_AP50 cannot see: a volume that matches well overall while the ownership
    drifts per view;
  - edge cases: no GT, no predictions, an instance visible in a single view;
  - eval integration: the three `bundle_*` keys appear on the multi-frame path, the per-frame
    path is untouched, and every pre-existing key is still produced.
"""

import sys
from argparse import Namespace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from maskdino_fixtures import _tiny_head

torch.manual_seed(0)

LOGIT = 10.0   # a binary mask as logits: +10 inside, -10 outside


def _volume(s, hw, boxes):
    """[S, h, w] binary volume; `boxes[f]` is None (not visible) or (r0, r1, c0, c1)."""
    v = torch.zeros(s, hw, hw)
    for f, b in enumerate(boxes):
        if b is not None:
            r0, r1, c0, c1 = b
            v[f, r0:r1, c0:c1] = 1.0
    return v


def test_planted_perfect():
    print("=== Testing a perfectly consistent prediction ===")
    from train.eval_metrics import multiview_consistency_metrics

    s, hw = 4, 8
    gt = torch.stack([_volume(s, hw, [(1, 4, 1, 4)] * s),
                      _volume(s, hw, [(5, 8, 5, 8)] * s)])          # [2, S, h, w]
    # Query 0 owns instance 0 in every view, query 1 owns instance 1; query 2 is a distractor
    # that never overlaps either.
    pred = torch.stack([gt[0], gt[1], _volume(s, hw, [(0, 1, 6, 7)] * s)]) * 2 * LOGIT - LOGIT

    m = multiview_consistency_metrics(pred, gt)
    assert m["view_consistency"] == 1.0, m
    assert m["id_switch"] == 0.0, m
    assert m["num_matched"] == 2.0, m
    print(f"✅ consistency={m['view_consistency']:.2f} id_switch={m['id_switch']:.2f} "
          f"over {m['num_matched']:.0f} matched instances\n")


def test_planted_id_switch():
    print("=== Testing a planted ID switch in one view ===")
    from train.eval_metrics import multiview_consistency_metrics

    s, hw = 4, 8
    box = (1, 4, 1, 4)
    gt = _volume(s, hw, [box] * s)[None]                            # [1, S, h, w]
    # Query 0 owns the instance in 3 of the 4 views; in view 3 it is empty and query 1 owns it.
    q0 = _volume(s, hw, [box, box, box, None])
    q1 = _volume(s, hw, [None, None, None, box])
    m = multiview_consistency_metrics(torch.stack([q0, q1]) * 2 * LOGIT - LOGIT, gt)

    assert m["num_matched"] == 1.0, m
    assert abs(m["view_consistency"] - 0.75) < 1e-6, m   # 3 of 4 views explained by the match
    assert abs(m["id_switch"] - 0.25) < 1e-6, m          # exactly one view changed owner
    print(f"✅ one switched view of four → consistency {m['view_consistency']:.2f}, "
          f"id_switch {m['id_switch']:.2f}\n")

    # Both metrics must move MONOTONICALLY with the number of switched views.
    q0b = _volume(s, hw, [box, box, None, None])
    q1b = _volume(s, hw, [None, None, box, box])
    m2 = multiview_consistency_metrics(torch.stack([q0b, q1b]) * 2 * LOGIT - LOGIT, gt)
    assert m2["view_consistency"] < m["view_consistency"], (m, m2)
    assert m2["id_switch"] > m["id_switch"], (m, m2)
    print(f"✅ two switched views → consistency {m2['view_consistency']:.2f}, "
          f"id_switch {m2['id_switch']:.2f} (both moved the right way)\n")


def test_sees_what_volume_iou_cannot():
    """The reason the metric exists: a bundle-level volume IoU can stay high while the
    ownership of the object drifts from view to view (docs/RELATED_WORK.md gap 2)."""
    print("=== Testing that consistency separates two equal-volume-IoU predictions ===")
    from train.eval_metrics import mask_iou_matrix, multiview_consistency_metrics

    s, hw = 4, 8
    box = (1, 5, 1, 5)
    gt = _volume(s, hw, [box] * s)[None]

    # A: one query, whole volume. B: four queries, each owning exactly one view. Their UNION is
    # the same set of pixels, so the best per-query volume IoU is very different — but even the
    # per-view masks of B are individually perfect, which is what makes id_switch the only
    # number that catches it.
    a = _volume(s, hw, [box] * s)[None]
    b = torch.stack([_volume(s, hw, [box if f == g else None for f in range(s)])
                     for g in range(s)])
    ma = multiview_consistency_metrics(a * 2 * LOGIT - LOGIT, gt)
    mb = multiview_consistency_metrics(b * 2 * LOGIT - LOGIT, gt)

    # Every one of B's queries is per-view perfect where it fires ...
    per_view = mask_iou_matrix(b[:, 0].flatten(1), gt[:, 0].flatten(1))
    assert float(per_view.max()) == 1.0
    # ... yet only A explains the instance across the bundle.
    assert ma["view_consistency"] == 1.0 and ma["id_switch"] == 0.0, ma
    assert mb["view_consistency"] == 0.25, mb
    assert mb["id_switch"] == 0.75, mb
    print(f"✅ shared query: {ma['view_consistency']:.2f}/{ma['id_switch']:.2f}  vs  "
          f"per-view queries: {mb['view_consistency']:.2f}/{mb['id_switch']:.2f}\n")


def test_edge_cases():
    print("=== Testing degenerate inputs ===")
    from train.eval_metrics import multiview_consistency_metrics

    s, hw = 3, 8
    box = (2, 5, 2, 5)
    gt = _volume(s, hw, [box] * s)[None]
    pred = gt * 2 * LOGIT - LOGIT
    zeros = {"view_consistency": 0.0, "id_switch": 0.0, "num_matched": 0.0}

    assert multiview_consistency_metrics(pred, torch.zeros(0, s, hw, hw)) == zeros
    assert multiview_consistency_metrics(torch.zeros(0, s, hw, hw), gt) == zeros
    # predictions that overlap nothing are never matched → nothing to average
    away = _volume(s, hw, [(6, 8, 6, 8)] * s)[None] * 2 * LOGIT - LOGIT
    assert multiview_consistency_metrics(away, gt) == zeros

    # an instance visible in ONE view only: the denominator is that view alone
    gt1 = _volume(s, hw, [box, None, None])[None]
    hit = multiview_consistency_metrics(_volume(s, hw, [box, None, None])[None]
                                        * 2 * LOGIT - LOGIT, gt1)
    assert hit["view_consistency"] == 1.0 and hit["id_switch"] == 0.0, hit
    assert hit["num_matched"] == 1.0, hit
    # a query present in the OTHER views cannot help (they are not visible views of the GT) ...
    noisy = multiview_consistency_metrics(
        _volume(s, hw, [box, (0, 2, 0, 2), (0, 2, 0, 2)])[None] * 2 * LOGIT - LOGIT, gt1)
    assert noisy["view_consistency"] == 1.0 and noisy["id_switch"] == 0.0, noisy
    # ... and a half-covering mask in that single view fails the 0.5 IoU bar
    half = multiview_consistency_metrics(
        _volume(s, hw, [(2, 5, 2, 3), None, None])[None] * 2 * LOGIT - LOGIT, gt1)
    assert half["num_matched"] == 1.0 and half["view_consistency"] == 0.0, half
    # ... and it is still the only query overlapping there, so it is a miss, not a switch
    assert half["id_switch"] == 0.0, half
    print("✅ no GT / no predictions / no overlap / single-view instance all handled\n")


def _bundle_scene(s, hh, mem, name="sceneA"):
    """A cached-scene dict the way prepare_scenes builds it (see tests/test_maskdino_fullres)."""
    from train.maskdino_data import build_frame_targets

    id_maps = torch.zeros(s, hh * 4, hh * 4, dtype=torch.long)
    id_maps[:, 2:14, 2:14] = 1
    id_maps[:, 20:30, 18:30] = 2
    targets = build_frame_targets(
        {"classes": torch.tensor([3, 7]), "masks": id_maps}, (hh, hh), "cpu")
    return {"name": name, "split": "val", "bundles": [
        {"features": torch.randn(s, 5 + hh * hh, mem), "patch_start_idx": 5,
         "targets": targets, "images": None}]}


def test_eval_reports_consistency_keys():
    print("=== Testing eval_scenes integration (keys are purely additive) ===")
    from train.eval_metrics import CONSISTENCY_KEYS, TRACKING_KEYS
    from train.maskdino_eval import eval_scenes

    torch.manual_seed(0)
    s, hh, mem = 3, 8, 64
    model = torch.nn.Module()
    model.head = _tiny_head(dec_layers=2, enc_layers=1, dn="no", num_queries=10,
                            cross_frame_attn=True)
    scenes = [_bundle_scene(s, hh, mem)]

    args = Namespace(multi_frame=True, eval_topk=100, score_threshold=0.25, eval_batch_frames=s)
    m = eval_scenes(model, scenes, args, "cpu")["sceneA"]
    for k in CONSISTENCY_KEYS:
        assert f"bundle_{k}" in m, (k, sorted(m))
    assert 0.0 <= m["bundle_view_consistency"] <= 1.0, m
    assert 0.0 <= m["bundle_id_switch"] <= 1.0, m
    # every key the pre-§6.6 eval produced is still there, and only the three are new
    old = {"mIoU", "AP50", "AP75", "mAP", "class_acc", "num_pred", "num_gt"}
    old |= {f"{k}_all" for k in old} | {f"bundle_{k}" for k in old} \
        | {f"bundle_{k}_all" for k in old}
    expected = old | {f"bundle_{k}" for k in CONSISTENCY_KEYS} \
        | {f"bundle_{k}" for k in TRACKING_KEYS} \
        | {f"bundle_{k}_all" for k in TRACKING_KEYS}
    assert set(m) == expected, sorted(set(m) ^ expected)

    # the single-frame path stays exactly as it was: no bundle_* keys at all
    single = Namespace(multi_frame=False, eval_topk=100, score_threshold=0.25,
                       eval_batch_frames=s)
    ms = eval_scenes(model, scenes, single, "cpu")["sceneA"]
    assert not any(k.startswith("bundle_") for k in ms), sorted(ms)
    print("✅ three additive bundle_* keys on the multi-frame path, per-frame path unchanged\n")


if __name__ == "__main__":
    test_planted_perfect()
    test_planted_id_switch()
    test_sees_what_volume_iou_cannot()
    test_edge_cases()
    test_eval_reports_consistency_keys()
    print("All test_maskdino_consistency tests passed! ✅")
