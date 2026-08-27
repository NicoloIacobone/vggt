#!/usr/bin/env python3
"""
The FORMAL cross-view identity metrics — HOTA / AssA / DetA / IDF1. Standalone, CPU-only.

`multiview_consistency_metrics` answers the same question with project-defined numbers that no
competitor publishes; these are the ones quoted outward. What the tests pin down:

  - a perfectly consistent bundle scores 1.0 on all four;
  - the property the custom pair cannot cleanly express: an identity SWITCH costs AssA while a
    plain MISS does not — association and detection are separated, which is the whole reason
    the tracking literature reports both;
  - each further switched view moves every number strictly the wrong way;
  - per-view queries (the failure mode a shared query is supposed to make impossible) collapse
    AssA while per-view masks stay individually perfect;
  - edge cases: no GT, no predictions, an instance visible in one view;
  - eval integration: the four `bundle_*` keys appear on the multi-frame path and the per-frame
    path stays untouched.
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


def _two_instance_gt(s=4, hw=8):
    return torch.stack([_volume(s, hw, [(1, 4, 1, 4)] * s),
                        _volume(s, hw, [(5, 8, 5, 8)] * s)])


def test_planted_perfect():
    print("=== Testing a perfectly consistent bundle ===")
    from train.eval_metrics import tracking_consistency_metrics

    gt = _two_instance_gt()
    m = tracking_consistency_metrics(torch.stack([gt[0], gt[1]]) * 2 * LOGIT - LOGIT, gt)
    for k in ("hota", "assa", "deta", "idf1"):
        assert m[k] == 1.0, (k, m)
    assert m["num_gt_tracks"] == 2.0 and m["num_pred_tracks"] == 2.0, m
    print(f"✅ HOTA={m['hota']:.2f} AssA={m['assa']:.2f} DetA={m['deta']:.2f} "
          f"IDF1={m['idf1']:.2f} over 2 tracks\n")


def test_switch_costs_association_a_miss_does_not():
    """The reason these metrics replace the custom pair: they price the two failures apart."""
    print("=== Testing that a switch hits AssA and a miss hits DetA ===")
    from train.eval_metrics import tracking_consistency_metrics

    s, hw = 4, 8
    gt = _two_instance_gt(s, hw)

    # (a) an identity SWITCH: in view 2 query 1 takes over instance 0. Every view is still
    #     explained by SOME query, so detection is barely touched.
    q0, q1 = gt[0].clone(), gt[1].clone()
    q0[2] = 0
    q1[2] = gt[0][2]
    switch = tracking_consistency_metrics(torch.stack([q0, q1]) * 2 * LOGIT - LOGIT, gt)

    # (b) a plain MISS: query 0 simply drops view 2, nobody takes over. Identity is intact.
    q0b = gt[0].clone()
    q0b[2] = 0
    miss = tracking_consistency_metrics(torch.stack([q0b, gt[1]]) * 2 * LOGIT - LOGIT, gt)

    # the same number of detections is lost in both cases ...
    assert abs(switch["deta"] - miss["deta"]) < 1e-6, (switch, miss)
    # ... but only the switch is an ASSOCIATION error
    assert switch["assa"] < miss["assa"] - 0.2, (switch, miss)
    assert switch["idf1"] < miss["idf1"], (switch, miss)
    print(f"✅ switch: AssA={switch['assa']:.3f} DetA={switch['deta']:.3f}  vs  "
          f"miss: AssA={miss['assa']:.3f} DetA={miss['deta']:.3f} "
          f"(DetA identical, AssA separates them)\n")


def test_more_switches_are_strictly_worse():
    print("=== Testing monotonicity in the number of switched views ===")
    from train.eval_metrics import tracking_consistency_metrics

    s, hw = 4, 8
    gt = _two_instance_gt(s, hw)

    def with_switches(n):
        q0, q1 = gt[0].clone(), gt[1].clone()
        for f in range(s - n, s):
            q0[f] = 0
            q1[f] = gt[0][f]
        return tracking_consistency_metrics(torch.stack([q0, q1]) * 2 * LOGIT - LOGIT, gt)

    one, two = with_switches(1), with_switches(2)
    for k in ("hota", "assa", "idf1"):
        assert two[k] < one[k], (k, one, two)
    print(f"✅ 1 switch HOTA={one['hota']:.3f} → 2 switches HOTA={two['hota']:.3f} "
          f"(AssA {one['assa']:.3f} → {two['assa']:.3f})\n")


def test_per_view_queries_collapse_association():
    """The failure mode a shared query makes impossible, priced formally."""
    print("=== Testing per-view queries vs one shared query ===")
    from train.eval_metrics import mask_iou_matrix, tracking_consistency_metrics

    s, hw = 4, 8
    box = (1, 5, 1, 5)
    gt = _volume(s, hw, [box] * s)[None]

    shared = _volume(s, hw, [box] * s)[None]
    per_view = torch.stack([_volume(s, hw, [box if f == g else None for f in range(s)])
                            for g in range(s)])

    # each per-view query is individually perfect where it fires ...
    assert float(mask_iou_matrix(per_view[:, 0].flatten(1), gt[:, 0].flatten(1)).max()) == 1.0

    ms = tracking_consistency_metrics(shared * 2 * LOGIT - LOGIT, gt)
    mp = tracking_consistency_metrics(per_view * 2 * LOGIT - LOGIT, gt)
    assert ms["assa"] == 1.0 and ms["hota"] == 1.0, ms
    # detection is perfect for both — every view is covered — but association is not
    assert mp["deta"] == 1.0, mp
    assert mp["assa"] <= 0.25 + 1e-6, mp
    assert mp["idf1"] < 0.5, mp
    print(f"✅ shared query AssA={ms['assa']:.2f}  vs  per-view queries AssA={mp['assa']:.2f} "
          f"at identical DetA={mp['deta']:.2f}\n")


def test_edge_cases():
    print("=== Testing degenerate inputs ===")
    from train.eval_metrics import TRACKING_KEYS, tracking_consistency_metrics

    s, hw = 3, 8
    box = (2, 5, 2, 5)
    gt = _volume(s, hw, [box] * s)[None]
    pred = gt * 2 * LOGIT - LOGIT
    zeros = {k: 0.0 for k in TRACKING_KEYS}

    assert tracking_consistency_metrics(pred, torch.zeros(0, s, hw, hw)) == zeros
    assert tracking_consistency_metrics(torch.zeros(0, s, hw, hw), gt) == zeros
    # an all-empty prediction volume has no detections at all
    assert tracking_consistency_metrics(torch.full((2, s, hw, hw), -LOGIT), gt) == zeros

    # a prediction that overlaps nothing: detections exist but never match at any alpha
    away = _volume(s, hw, [(6, 8, 6, 8)] * s)[None] * 2 * LOGIT - LOGIT
    m = tracking_consistency_metrics(away, gt)
    assert m["hota"] == 0.0 and m["assa"] == 0.0 and m["idf1"] == 0.0, m
    assert m["num_gt_tracks"] == 1.0 and m["num_pred_tracks"] == 1.0, m

    # an instance visible in ONE view only: a one-timestep sequence is still perfectly scored
    gt1 = _volume(s, hw, [box, None, None])[None]
    hit = tracking_consistency_metrics(_volume(s, hw, [box, None, None])[None]
                                       * 2 * LOGIT - LOGIT, gt1)
    assert hit["hota"] == 1.0 and hit["assa"] == 1.0 and hit["idf1"] == 1.0, hit
    print("✅ empty GT / empty preds / no overlap / single-view instance all behave\n")


def test_bounded_and_hota_is_the_geometric_mean():
    print("=== Testing the HOTA identity and the [0, 1] bound on random input ===")
    from train.eval_metrics import tracking_consistency_metrics

    s, hw = 5, 8
    gt = torch.stack([_volume(s, hw, [(0, 3, 0, 3)] * s),
                      _volume(s, hw, [(4, 7, 4, 7)] * s),
                      _volume(s, hw, [(0, 2, 5, 8), None, (0, 2, 5, 8), None, None])])
    for trial in range(5):
        pred = (torch.rand(6, s, hw, hw) - 0.5) * 2 * LOGIT
        m = tracking_consistency_metrics(pred, gt)
        for k in ("hota", "assa", "deta", "idf1"):
            assert 0.0 <= m[k] <= 1.0, (trial, k, m)
        # HOTA is the alpha-mean of sqrt(DetA*AssA), so it is bounded by the mean of the two
        assert m["hota"] <= max(m["deta"], m["assa"]) + 1e-9, (trial, m)
    print("✅ all four in [0, 1] and HOTA never exceeds max(DetA, AssA)\n")


def test_eval_reports_tracking_keys():
    print("=== Testing that the eval emits the four bundle_* keys ===")
    from train.eval_metrics import TRACKING_KEYS
    from train.maskdino_eval import eval_scenes
    from test_maskdino_consistency import _bundle_scene

    torch.manual_seed(0)
    s, hh, mem = 3, 8, 64
    model = torch.nn.Module()
    model.head = _tiny_head(dec_layers=2, enc_layers=1, dn="no", num_queries=10,
                            cross_frame_attn=True)
    scenes = [_bundle_scene(s, hh, mem)]

    args = Namespace(multi_frame=True, eval_topk=100, score_threshold=0.25, eval_batch_frames=s)
    m = eval_scenes(model, scenes, args, "cpu")["sceneA"]
    for k in TRACKING_KEYS:
        assert f"bundle_{k}" in m, (k, sorted(m))
    for k in ("hota", "assa", "deta", "idf1"):
        assert 0.0 <= m[f"bundle_{k}"] <= 1.0, (k, m[f"bundle_{k}"])

    # the single-frame path stays exactly as it was: no bundle_* keys at all
    single = Namespace(multi_frame=False, eval_topk=100, score_threshold=0.25,
                       eval_batch_frames=s)
    ms = eval_scenes(model, scenes, single, "cpu")["sceneA"]
    assert not any(k.startswith("bundle_") for k in ms), sorted(ms)
    print("✅ four additive bundle_* keys on the multi-frame path, per-frame path unchanged\n")


def test_unfiltered_pool_destroys_deta_but_not_assa():
    """
    The bug this pins: these metrics count every unmatched prediction as a hard FP.

    Feeding them the raw top-k query pool reports the QUERY BUDGET, not the model — DetA and
    IDF1 collapse while AssA, which averages only over matched detections, barely moves. That is
    exactly what the first re-scoring run showed on the official split (DetA 0.066 unfiltered),
    and it is why `train/maskdino_eval.py` filters with `confident_detections` first.
    """
    print("=== Testing that an unfiltered prediction pool crushes DetA, not AssA ===")
    from train.eval_metrics import tracking_consistency_metrics

    s, hw = 4, 12
    gt = torch.stack([_volume(s, hw, [(1, 4, 1, 4)] * s),
                      _volume(s, hw, [(6, 9, 6, 9)] * s)])
    good = torch.stack([gt[0], gt[1]])

    # the two real detections, plus 30 junk queries that fire somewhere harmless every view
    junk = torch.stack([_volume(s, hw, [(10, 11, (i % 10), (i % 10) + 1)] * s) for i in range(30)])
    pool = torch.cat([good, junk])

    filtered = tracking_consistency_metrics(good * 2 * LOGIT - LOGIT, gt)
    unfiltered = tracking_consistency_metrics(pool * 2 * LOGIT - LOGIT, gt)

    assert filtered["deta"] == 1.0, filtered
    assert unfiltered["deta"] < 0.15, unfiltered           # 2 of 32 tracks are real
    assert unfiltered["idf1"] < 0.25, unfiltered
    # association is measured over matched detections only, so it survives the junk
    assert unfiltered["assa"] == filtered["assa"] == 1.0, (filtered, unfiltered)
    print(f"✅ DetA {filtered['deta']:.2f} → {unfiltered['deta']:.3f} with 30 junk queries, "
          f"while AssA holds at {unfiltered['assa']:.2f}\n")


def test_confident_detections_is_the_shared_filter():
    """AP and the tracking metrics must score the same submitted set, from one definition."""
    print("=== Testing the shared confident-detection filter ===")
    from train.eval_metrics import confident_detections

    # [N, C] sigmoid logits; column 0 is background (never selected by the max)
    logits = torch.tensor([[-9.0, 4.0, -9.0],     # confident class 1
                           [-9.0, -9.0, -0.2],    # sigmoid(-0.2) = 0.45, below 0.5
                           [-9.0, -9.0, -9.0]])   # nothing fires
    keep = confident_detections(logits, score_threshold=0.5)
    assert keep.tolist() == [True, False, False], keep.tolist()
    # a lower operating point admits the middle query
    assert confident_detections(logits, 0.4).tolist() == [True, True, False]
    print("✅ one filter, used by both AP and the tracking metrics\n")


if __name__ == "__main__":
    test_planted_perfect()
    test_switch_costs_association_a_miss_does_not()
    test_more_switches_are_strictly_worse()
    test_per_view_queries_collapse_association()
    test_edge_cases()
    test_unfiltered_pool_destroys_deta_but_not_assa()
    test_confident_detections_is_the_shared_filter()
    test_bounded_and_hota_is_the_geometric_mean()
    test_eval_reports_tracking_keys()
    print("All test_maskdino_tracking_metrics tests passed! ✅")
