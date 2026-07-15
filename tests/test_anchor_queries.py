#!/usr/bin/env python3
"""
Arm E validation: 3D-anchored queries (models/anchor_queries.py + query_mode="anchor3d").

CPU-only, no backbone weights. Covers:
  - FourierPositionalEncoding generalized to 3D (and 2D backward compatibility)
  - patch_token_positions: confidence-weighted per-token 3D positions
  - farthest_point_sample: determinism + spread
  - build_anchors: shapes, normalization invariance to global shift/scale,
    kNN content pooling, tiny-scene padding
  - QueryGenerator/D4RTInstanceSegmentationHead anchor3d forward: shapes, no NaN,
    gradient flow, anchors-required error, head_config round-trip
  - jitter_anchors augmentation
"""

import math
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.anchor_queries import (
    patch_token_positions, farthest_point_sample, build_anchors, jitter_anchors)
from models.d4rt_decoder import (
    FourierPositionalEncoding, QueryGenerator, D4RTInstanceSegmentationHead)


def test_fourier_3d():
    print("=== Fourier encoding: 3D + 2D backward compatibility ===")
    enc = FourierPositionalEncoding(num_freqs=16, max_freq=10.0)

    coords3 = torch.randn(2, 5, 3)
    out3 = enc(coords3)
    assert out3.shape == (2, 5, 96), f"3D: expected [2,5,96], got {out3.shape}"

    # 2D path must be bit-identical to the original layout (sin/cos interleaved per coord).
    coords2 = torch.rand(2, 7, 2)
    out2 = enc(coords2)
    assert out2.shape == (2, 7, 64), f"2D: expected [2,7,64], got {out2.shape}"
    scaled = coords2.unsqueeze(2) * enc.freqs.view(1, 1, -1, 1)
    manual = torch.stack([torch.sin(2 * math.pi * scaled),
                          torch.cos(2 * math.pi * scaled)], dim=-1).reshape(2, 7, 64)
    assert torch.allclose(out2, manual), "2D encoding layout changed — old checkpoints break"
    print("✅ passed\n")


def _synthetic_pointmap(S=2, hp=4, wp=4, ps=14):
    """Pointmap where every pixel of patch cell (s, i, j) sits at xyz = (s, i, j)."""
    wp_map = torch.zeros(1, S, hp * ps, wp * ps, 3)
    for s in range(S):
        for i in range(hp):
            for j in range(wp):
                wp_map[0, s, i * ps:(i + 1) * ps, j * ps:(j + 1) * ps] = torch.tensor(
                    [float(s), float(i), float(j)])
    conf = torch.ones(1, S, hp * ps, wp * ps)
    return wp_map, conf


def test_patch_token_positions():
    print("=== patch_token_positions ===")
    S, hp, wp, ps = 2, 4, 4, 14
    wp_map, conf = _synthetic_pointmap(S, hp, wp, ps)
    pos, w = patch_token_positions(wp_map, conf, patch_size=ps)
    assert pos.shape == (S * hp * wp, 3) and w.shape == (S * hp * wp,)
    # Constant cells -> token position equals the cell value; check token ORDER too
    # (frame-major, then row-major within the frame — the aggregator's patch order).
    expected = torch.tensor([[float(s), float(i), float(j)]
                             for s in range(S) for i in range(hp) for j in range(wp)])
    assert torch.allclose(pos, expected, atol=1e-5), "token positions/order wrong"

    # Confidence weighting: poison half of one cell with garbage at zero confidence.
    wp_map2, conf2 = _synthetic_pointmap(S, hp, wp, ps)
    wp_map2[0, 0, :7, :ps] = 1e6
    conf2[0, 0, :7, :ps] = 0.0
    pos2, _ = patch_token_positions(wp_map2, conf2, patch_size=ps)
    assert torch.allclose(pos2[0], torch.tensor([0.0, 0.0, 0.0]), atol=1e-4), \
        "zero-confidence pixels must not contribute"
    print("✅ passed\n")


def test_fps():
    print("=== farthest_point_sample ===")
    # On a 1D line embedded in 3D, FPS from the centroid must hit both endpoints first.
    t = torch.linspace(0, 1, 101).unsqueeze(1)
    line = torch.cat([t, torch.zeros_like(t), torch.zeros_like(t)], dim=1)
    idx = farthest_point_sample(line, 3)
    assert idx.shape == (3,)
    assert 50 in idx.tolist()[:1], "must start closest to the centroid"
    assert {0, 100}.issubset(set(idx.tolist())), "endpoints are the two farthest points"

    # Deterministic + unique + clamped.
    idx2 = farthest_point_sample(line, 3)
    assert torch.equal(idx, idx2), "FPS must be deterministic"
    assert len(set(idx.tolist())) == 3
    assert farthest_point_sample(line[:2], 5).shape == (2,), "k must clamp to M"
    print("✅ passed\n")


def test_build_anchors():
    print("=== build_anchors ===")
    torch.manual_seed(0)
    S, hp, wp, ps, C, K = 2, 4, 4, 14, 32, 8
    T = S * hp * wp
    wp_map, conf = _synthetic_pointmap(S, hp, wp, ps)
    # De-symmetrize: the perfect integer grid has exact FPS distance ties, whose argmax
    # tie-breaks flip under float-epsilon perturbations (real pointmaps have no ties).
    # Offsets are constant within each patch cell so token positions stay well-defined.
    g = torch.Generator().manual_seed(123)
    cell_offsets = torch.randn(S, hp, wp, 3, generator=g) * 0.05
    wp_map += cell_offsets.repeat_interleave(ps, dim=1).repeat_interleave(ps, dim=2).unsqueeze(0)
    # Features: token t's feature = one-hot-ish signature of t, so pooling is checkable.
    patch_feats = torch.eye(T, C).unsqueeze(0)  # [1, T, C] (T x C slice of identity)
    patch_start_idx = 5
    # [1, S, patch_start_idx + hp*wp, C], special tokens zeroed.
    features = torch.zeros(1, S, patch_start_idx + hp * wp, C)
    features[0, :, patch_start_idx:, :] = patch_feats.reshape(S, hp * wp, C)

    anchors = build_anchors(features, patch_start_idx, wp_map, conf,
                            num_anchors=K, knn=2, patch_size=ps)
    assert anchors["xyz"].shape == (1, K, 3)
    assert anchors["feats"].shape == (1, K, C)
    assert torch.isfinite(anchors["xyz"]).all() and torch.isfinite(anchors["feats"]).all()

    # Normalization: zero mean is not guaranteed on the SAMPLED anchors, but the pool is
    # unit-RMS; anchors drawn from it must live at O(1) scale.
    assert anchors["xyz"].abs().max() < 10, "normalized anchors should be O(1)"

    # Invariance: shifting/scaling the world globally must not change normalized anchors
    # (same tokens selected, same normalized positions) nor the pooled features.
    anchors2 = build_anchors(features, patch_start_idx, wp_map * 7.3 + 42.0, conf,
                             num_anchors=K, knn=2, patch_size=ps)
    assert torch.allclose(anchors["xyz"], anchors2["xyz"], atol=1e-4), \
        "anchor xyz must be invariant to global shift/scale of the pointmap"
    assert torch.allclose(anchors["feats"], anchors2["feats"], atol=1e-5)

    # Content pooling: each anchor's feature = mean of its knn tokens' one-hot signatures
    # -> each row sums to 1 and has exactly knn nonzero entries of value 1/knn.
    row_sums = anchors["feats"].sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(1, K), atol=1e-5), "pooled one-hots must sum to 1"
    nonzero = (anchors["feats"][0] > 1e-6).sum(dim=-1)
    assert (nonzero == 2).all(), f"expected exactly knn=2 pooled tokens, got {nonzero.tolist()}"

    # Tiny scene: fewer valid tokens than K -> padded by cycling, still exactly K anchors.
    tiny_wp, tiny_conf = _synthetic_pointmap(1, 2, 2, ps)
    tiny_feats = torch.zeros(1, 1, patch_start_idx + 4, C)
    tiny_feats[0, :, patch_start_idx:, :] = torch.eye(4, C)
    tiny = build_anchors(tiny_feats, patch_start_idx, tiny_wp, tiny_conf,
                         num_anchors=K, knn=2, patch_size=ps)
    assert tiny["xyz"].shape == (1, K, 3) and tiny["feats"].shape == (1, K, C)
    print("✅ passed\n")


def test_anchor3d_query_generator():
    print("=== QueryGenerator(query_mode='anchor3d') ===")
    torch.manual_seed(0)
    B, K, C, hidden = 1, 8, 64, 256
    qg = QueryGenerator(num_views=4, hidden_dim=hidden, query_mode="anchor3d", memory_dim=C)
    anchors = {"xyz": torch.randn(B, K, 3),
               "feats": torch.randn(B, K, C) * 100.0}  # huge magnitude, like raw VGGT feats
    placeholder_c = torch.zeros(B, K, 2)
    placeholder_v = torch.zeros(B, K, dtype=torch.long)
    images = torch.rand(B, 2, 3, 56, 56)

    q = qg(placeholder_c, placeholder_v, images, anchors=anchors)
    assert q.shape == (B, K, hidden)
    assert torch.isfinite(q).all(), "queries must be finite even with huge feature norms"
    # LayerNorm must tame the magnitude: queries at O(1..10), not O(100).
    assert q.abs().mean() < 20, f"anchor queries not normalized: mean |q| = {q.abs().mean():.1f}"

    # Distinct anchors -> distinct queries (no collapse at init).
    assert (q[0, 0] - q[0, 1]).abs().max() > 1e-4

    # Gradient flows to the anchor projections.
    q.sum().backward()
    assert qg.anchor_pos_proj.weight.grad is not None
    assert qg.anchor_feat_proj.weight.grad is not None

    # anchors required in anchor3d mode.
    try:
        qg(placeholder_c, placeholder_v, images)
        raise AssertionError("anchor3d forward without anchors should raise")
    except ValueError:
        pass

    # Invalid mode still rejected.
    try:
        QueryGenerator(query_mode="anchor2d")
        raise AssertionError("bad query_mode should raise")
    except ValueError:
        pass
    print("✅ passed\n")


def test_anchor3d_head_end_to_end():
    print("=== D4RTInstanceSegmentationHead anchor3d end-to-end (CPU, small dims) ===")
    torch.manual_seed(0)
    B, S, K, C = 1, 2, 8, 64
    hp = wp = 4
    patch_start_idx = 5
    P = patch_start_idx + hp * wp
    head = D4RTInstanceSegmentationHead(
        num_views=4, hidden_dim=32, num_decoder_layers=1, mask_embed_dim=16,
        memory_dim=C, query_mode="anchor3d", num_anchors=K, anchor_knn=2)
    feats = torch.randn(B, S, P, C)
    images = torch.rand(B, S, 3, 56, 56)
    anchors = {"xyz": torch.randn(B, K, 3), "feats": torch.randn(B, K, C)}
    coords = torch.zeros(B, K, 2)
    view_ids = torch.zeros(B, K, dtype=torch.long)

    class_logits, mask_emb, pred_masks = head(
        coords, view_ids, images, feats, patch_start_idx, anchors=anchors)
    assert class_logits.shape == (B, K, 20)
    assert mask_emb.shape == (B, K, 16)
    assert pred_masks.shape == (B, K, S, hp, wp)
    assert torch.isfinite(pred_masks).all()

    # head_config round-trip: rebuilding from the stored config must accept the state dict.
    cfg = dict(num_views=4, hidden_dim=32, num_classes=20, num_decoder_layers=1,
               patch_size=9, mask_embed_dim=16, memory_dim=C, dropout=0.0,
               query_mode="anchor3d", num_learned_queries=0, mask_upsample=1,
               num_anchors=K, anchor_knn=2)
    head2 = D4RTInstanceSegmentationHead(**cfg)
    head2.load_state_dict(head.state_dict())
    assert head2.num_anchors == K and head2.anchor_knn == 2

    # anchor3d without num_anchors must be rejected (checkpoint config safety).
    try:
        D4RTInstanceSegmentationHead(query_mode="anchor3d")
        raise AssertionError("anchor3d head without num_anchors should raise")
    except ValueError:
        pass
    print("✅ passed\n")


def test_jitter():
    print("=== jitter_anchors ===")
    torch.manual_seed(0)
    anchors = {"xyz": torch.randn(1, 8, 3), "feats": torch.randn(1, 8, 16)}
    same = jitter_anchors(anchors, 0.0)
    assert same is anchors, "std=0 must be a no-op"
    j = jitter_anchors(anchors, 0.05)
    assert not torch.equal(j["xyz"], anchors["xyz"])
    assert torch.equal(j["feats"], anchors["feats"]), "content features must not be jittered"
    assert (j["xyz"] - anchors["xyz"]).abs().max() < 1.0
    print("✅ passed\n")


def test_train_multiscene_wiring():
    """anchor3d through the real train/eval helpers (no backbone; synthetic bundle)."""
    print("=== train_multiscene wiring: make_train_queries / head_forward / eval_scene ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from types import SimpleNamespace
    from train_multiscene import make_train_queries, head_forward, eval_scene

    torch.manual_seed(0)
    S, K, C, Ng = 2, 8, 64, 3
    hp = wp = 4
    psi = 5  # patch_start_idx
    head = D4RTInstanceSegmentationHead(
        num_views=4, hidden_dim=32, num_decoder_layers=1, mask_embed_dim=16,
        memory_dim=C, query_mode="anchor3d", num_anchors=K, anchor_knn=2)
    model = SimpleNamespace(decoder_head=head, training=False,
                            eval=lambda: None, train=lambda: None)
    bundle = {
        "images": torch.rand(1, S, 3, 56, 56),
        "coordinates": torch.rand(1, 10, 2),  # unused slot count on purpose (!= K)
        "view_ids": torch.zeros(1, 10, dtype=torch.long),
        "num_inst_queries": Ng,
        "features": torch.randn(1, S, psi + hp * wp, C),
        "patch_start_idx": psi,
        "num_patch_tokens": hp * wp,
        "gt": {"masks": (torch.rand(Ng, S, hp, wp) > 0.5).float(),
               "classes": torch.tensor([1, 2, 3]),
               "coordinates": torch.rand(Ng, 2)},
        "frame_names": None,
        "anchors": {"xyz": torch.randn(1, K, 3), "feats": torch.randn(1, K, C)},
    }
    args = SimpleNamespace(query_mode="anchor3d", num_anchors=K, anchor_jitter=0.05,
                           query_jitter=0.0, fixed_bg=True)

    # Placeholders sized to the anchor count, like learned mode.
    coords, view_ids = make_train_queries(bundle, args, "cpu")
    assert coords.shape == (1, K, 2) and view_ids.shape == (1, K)

    # head_forward falls back to the bundle's cached anchors; K queries come out.
    class_logits, _, pred_masks = head_forward(model, bundle, coords, view_ids)
    assert class_logits.shape == (1, K, 20)
    assert pred_masks.shape == (1, K, S, hp, wp)

    # An explicit (e.g. jittered) anchors dict overrides the bundle's.
    jittered = jitter_anchors(bundle["anchors"], 0.05)
    out_j = head_forward(model, bundle, coords, view_ids, anchors=jittered)
    assert not torch.allclose(out_j[2], pred_masks), "jittered anchors must change the output"

    # eval_scene routes anchor3d through the learned-style branch (GT-free queries).
    scene = {"name": "synthetic", "split": "val", "bundles": [bundle]}
    metrics = eval_scene(model, scene, "cpu")
    for key in ("mIoU", "AP50"):
        assert key in metrics and np.isfinite(metrics[key])
    print("✅ passed\n")


if __name__ == "__main__":
    test_fourier_3d()
    test_patch_token_positions()
    test_fps()
    test_build_anchors()
    test_anchor3d_query_generator()
    test_anchor3d_head_end_to_end()
    test_jitter()
    test_train_multiscene_wiring()
    print("=" * 50)
    print("All Arm-E anchor-query tests passed! ✅")
