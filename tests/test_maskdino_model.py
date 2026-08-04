#!/usr/bin/env python3
"""
MaskDINO model internals (docs/MASKDINO.md). Standalone, CPU-only, no VGGT weights.

  - the pure-PyTorch MSDeformAttn core against a naive explicit-loop reference;
  - VGGTPixelDecoder shapes (levels, mask_features, mask_upsample) + gradient flow;
  - MaskDINODecoder output/aux/interm/DN shapes in every config combination the training
    script can produce (two-stage on/off, dn on/off, initialize_box_type, train/eval);
  - box_ops.masks_to_boxes against hand-computed boxes;
  - head_config round-trip: it must cover every constructor argument, and rebuilding from a
    stored config must reproduce the same state_dict keys;
  - the 3D-anchor geometry of `--anchor_3d` (docs/MASKDINO.md §8.3): the soft-nearest-patch
    projection, its radius behaviour, the pyramid position gather and the normalisation.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from maskdino_fixtures import _synthetic_targets, _tiny_head
from models.maskdino import MaskDINOVGGTHead, build_head_from_config, to_scannet_class_logits
from models.maskdino import box_ops
from models.maskdino.anchor3d import (ANCHOR_LOG_R0, normalize_token_xyz, project_anchors,
                                      pyramid_token_xyz, uv_grid)
from models.maskdino.ms_deform_attn import MSDeformAttn, ms_deform_attn_core_pytorch
from models.maskdino.pixel_decoder import VGGTPixelDecoder

torch.manual_seed(0)


def _naive_ms_deform_attn(value, shapes, sampling_locations, attention_weights):
    """Explicit-loop reference: bilinear-sample each point, weight, sum. Slow but obvious."""
    N, _, M, D = value.shape
    _, Lq, _, L, P, _ = sampling_locations.shape
    splits = value.split([int(h) * int(w) for h, w in shapes], dim=1)
    out = torch.zeros(N, Lq, M, D)
    for n in range(N):
        for lvl in range(L):
            H, W = int(shapes[lvl][0]), int(shapes[lvl][1])
            grid = splits[lvl][n].reshape(H, W, M, D)
            for q in range(Lq):
                for m in range(M):
                    for p in range(P):
                        x, y = sampling_locations[n, q, m, lvl, p]
                        # grid_sample(align_corners=False) pixel centres at (i + 0.5) / size
                        gx, gy = x * W - 0.5, y * H - 0.5
                        x0, y0 = int(torch.floor(gx)), int(torch.floor(gy))
                        wx, wy = gx - x0, gy - y0
                        acc = torch.zeros(D)
                        for dy in (0, 1):
                            for dx in (0, 1):
                                xi, yi = x0 + dx, y0 + dy
                                if 0 <= xi < W and 0 <= yi < H:
                                    w = (wx if dx else 1 - wx) * (wy if dy else 1 - wy)
                                    acc = acc + w * grid[yi, xi, m]
                        out[n, q, m] += attention_weights[n, q, m, lvl, p] * acc
    return out.reshape(N, Lq, M * D)


def test_ms_deform_attn_core():
    print("=== Testing pure-PyTorch MSDeformAttn core vs naive reference ===")
    shapes = torch.as_tensor([[4, 5], [2, 3]], dtype=torch.long)
    N, M, D, Lq, P = 2, 2, 3, 4, 2
    total = int((shapes[:, 0] * shapes[:, 1]).sum())
    value = torch.randn(N, total, M, D)
    loc = torch.rand(N, Lq, M, len(shapes), P, 2)
    attn = torch.rand(N, Lq, M, len(shapes), P)
    attn = attn / attn.flatten(3).sum(-1)[..., None, None]

    fast = ms_deform_attn_core_pytorch(value, shapes, loc, attn)
    slow = _naive_ms_deform_attn(value, shapes, loc, attn)
    assert fast.shape == (N, Lq, M * D), fast.shape
    assert torch.allclose(fast, slow, atol=1e-5), (fast - slow).abs().max()

    # module-level forward + gradients, with both 2-d and 4-d reference points
    mod = MSDeformAttn(d_model=16, n_levels=2, n_heads=4, n_points=2)
    src = torch.randn(N, total, 16, requires_grad=True)
    query = torch.randn(N, Lq, 16)
    lsi = torch.cat((shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1]))
    for ref_dim in (2, 4):
        ref = torch.rand(N, Lq, 2, ref_dim)
        out = mod(query, ref, src, shapes, lsi)
        assert out.shape == (N, Lq, 16), out.shape
        out.sum().backward(retain_graph=True)
        assert src.grad is not None and torch.isfinite(src.grad).all()
    print("✅ MSDeformAttn core matches the naive reference (2-d and 4-d refs)\n")


def test_pixel_decoder():
    print("=== Testing VGGTPixelDecoder ===")
    B, mem, h, patch_start = 2, 64, 8, 5
    tokens = torch.randn(B, patch_start + h * h, mem, requires_grad=True)

    pd = VGGTPixelDecoder(memory_dim=mem, conv_dim=32, mask_dim=32, num_feature_levels=3,
                          enc_layers=2, nheads=4, dim_feedforward=64, enc_n_points=2)
    mask_features, levels = pd(tokens, patch_start)
    assert len(levels) == 3
    assert [tuple(l.shape[-2:]) for l in levels] == [(8, 8), (4, 4), (2, 2)], \
        [l.shape for l in levels]
    assert mask_features.shape == (B, 32, 8, 8), mask_features.shape
    mask_features.sum().backward()
    assert tokens.grad is not None and torch.isfinite(tokens.grad).all()

    # mask_upsample doubles/quadruples only the mask_features resolution
    for up, expect in ((2, 16), (4, 32)):
        pd_up = VGGTPixelDecoder(memory_dim=mem, conv_dim=32, mask_dim=32, num_feature_levels=2,
                                 enc_layers=0, mask_upsample=up)
        mf, lv = pd_up(tokens.detach(), patch_start)
        assert mf.shape == (B, 32, expect, expect), (up, mf.shape)
        assert tuple(lv[0].shape[-2:]) == (8, 8)
    print("✅ VGGTPixelDecoder shapes, mask_upsample and gradients OK\n")


def test_decoder_configs():
    print("=== Testing MaskDINODecoder across configs ===")
    B, mem, h, patch_start, nq = 2, 64, 8, 5, 12
    tokens = torch.randn(B, patch_start + h * h, mem)
    targets = _synthetic_targets(B, 3, (h, h))

    for two_stage in (True, False):
        for dn in ("no", "seg"):
            for box_type in (("no",) if not two_stage else ("no", "bitmask")):
                head = _tiny_head(two_stage=two_stage, dn=dn, initialize_box_type=box_type,
                                  learn_tgt=not two_stage)
                head.train()
                out, mask_dict = head(tokens, patch_start, targets)
                tag = f"two_stage={two_stage} dn={dn} box={box_type}"
                assert out["pred_logits"].shape == (B, nq, 19), (tag, out["pred_logits"].shape)
                assert out["pred_masks"].shape == (B, nq, h, h), (tag, out["pred_masks"].shape)
                assert out["pred_boxes"].shape == (B, nq, 4), (tag, out["pred_boxes"].shape)
                # initial prediction + one entry per decoder layer, minus the final one
                assert len(out["aux_outputs"]) == head.head_config["dec_layers"], tag
                assert ("interm_outputs" in out) == two_stage, tag
                if dn == "seg":
                    assert mask_dict is not None and mask_dict["pad_size"] > 0, tag
                    dn_out = mask_dict["output_known_lbs_bboxes"]
                    assert dn_out["pred_logits"].shape[1] == mask_dict["pad_size"], tag
                else:
                    assert mask_dict is None, tag

                head.eval()
                out_eval, md_eval = head(tokens, patch_start, None)
                assert md_eval is None, tag
                assert out_eval["pred_masks"].shape == (B, nq, h, h), tag
    print("✅ Decoder shapes/aux/interm/DN correct in all configurations\n")


def test_unported_box_init_is_rejected():
    print("=== Testing initialize_box_type guard ===")
    # Upstream MaskDINO offers 'mask2box'; this port implements only 'bitmask'. The two used to
    # share one `!= "no"` branch, so asking for mask2box silently ran bitmask. It must raise.
    for good in ("no", "bitmask"):
        _tiny_head(two_stage=(good != "no"), initialize_box_type=good)
    try:
        _tiny_head(initialize_box_type="mask2box")
    except ValueError as e:
        assert "mask2box" in str(e), f"unhelpful error message: {e}"
    else:
        raise AssertionError("initialize_box_type='mask2box' was silently accepted")
    print("✅ unported initialize_box_type raises instead of aliasing bitmask\n")


def test_masks_to_boxes():
    print("=== Testing masks_to_boxes ===")
    masks = torch.zeros(3, 10, 10)
    masks[0, 2:5, 3:8] = 1        # rows 2..4, cols 3..7
    masks[1] = 1                   # full image
    # masks[2] stays empty
    boxes = box_ops.masks_to_boxes(masks)
    assert torch.equal(boxes[0], torch.tensor([3.0, 2.0, 8.0, 5.0])), boxes[0]
    assert torch.equal(boxes[1], torch.tensor([0.0, 0.0, 10.0, 10.0])), boxes[1]
    assert torch.equal(boxes[2], torch.zeros(4)), boxes[2]

    norm = box_ops.masks_to_boxes_normalized(masks[:1])
    assert torch.allclose(norm[0], torch.tensor([0.55, 0.35, 0.5, 0.3]), atol=1e-6), norm
    xyxy = box_ops.box_cxcywh_to_xyxy(norm)
    assert torch.allclose(xyxy[0], torch.tensor([0.3, 0.2, 0.8, 0.5]), atol=1e-6), xyxy
    assert torch.allclose(torch.diag(box_ops.generalized_box_iou(xyxy, xyxy)), torch.ones(1))
    print("✅ masks_to_boxes / normalization / GIoU self-consistency OK\n")


def test_head_config_round_trip():
    print("=== Testing head_config round-trip ===")
    head = _tiny_head(dn="no", two_stage=False, learn_tgt=True, initialize_box_type="no")

    # head_config is derived from locals(), so it can never silently omit a constructor
    # argument — assert that contract directly rather than trusting the derivation.
    import inspect
    ctor_params = set(inspect.signature(MaskDINOVGGTHead.__init__).parameters) - {"self"}
    assert set(head.head_config) == ctor_params, (
        f"head_config drifted from the constructor: missing "
        f"{ctor_params - set(head.head_config)}, extra {set(head.head_config) - ctor_params}")

    rebuilt = build_head_from_config(head.head_config)
    assert rebuilt.head_config == head.head_config
    assert set(rebuilt.state_dict().keys()) == set(head.state_dict().keys())
    rebuilt.load_state_dict(head.state_dict())
    # unknown keys in a stored config (e.g. added later) must not break the rebuild
    cfg = dict(head.head_config)
    cfg["some_future_option"] = 42
    build_head_from_config(cfg)

    logits = torch.tensor([[0.0, 2.0, -3.0]])
    compat = to_scannet_class_logits(logits)
    assert compat.shape == (1, 4) and compat[0, 0] == float("-inf")
    assert torch.equal(compat[0, 1:], logits[0])
    print("✅ head_config round-trip + ScanNet logit adapter OK\n")


def test_anchor3d_geometry():
    """
    The geometry of --anchor_3d (docs/MASKDINO.md §8.3). This is the part that would silently
    produce a plausible-but-wrong number if the grid convention were off by a transpose, so it
    is asserted against hand-computed values rather than against itself.
    """
    print("=== Testing 3D-anchor projection geometry ===")
    g = 6
    uv = uv_grid(g, torch.device("cpu"), torch.float32)
    assert uv.shape == (g * g, 2)
    for r, c in [(0, 0), (0, g - 1), (g - 1, 2), (3, 4)]:
        # row-major flattening, u horizontal — the same order `tokens_to_map` reshapes with
        assert torch.allclose(uv[r * g + c],
                              torch.tensor([(c + 0.5) / g, (r + 0.5) / g])), (r, c)

    # Patch positions on a plane, one 3D point per patch, deliberately NOT axis-aligned with
    # (u, v) so that a transposed or flipped mapping cannot pass by symmetry.
    pos = torch.stack([uv[:, 0] * 3.0 + uv[:, 1], uv[:, 1] * 5.0, torch.zeros(g * g)], dim=-1)
    token_xyz = pos[None]                                        # [1, g*g, 3]

    # (a) an anchor sitting exactly on a patch, with a small radius, projects to that patch and
    #     collapses to the one-patch size floor.
    for p in (0, 7, g * g - 1):
        anchor = torch.cat([pos[p], torch.tensor([-4.0])])[None, None]   # log r = -4
        ref = project_anchors(anchor, token_xyz, 1)[0, 0]
        assert torch.allclose(ref[:2], uv[p], atol=1e-3), (p, ref[:2], uv[p])
        assert torch.allclose(ref[2:], torch.full((2,), 1.0 / g), atol=1e-3), ref[2:]

    # (b) a huge radius washes the softmax out to uniform: centre → image centre, size → the
    #     spread of the uniform distribution over the g patch centres, widened by the floor.
    anchor = torch.cat([pos[0], torch.tensor([6.0])])[None, None]
    ref = project_anchors(anchor, token_xyz, 1)[0, 0]
    uniform = 2.0 * ((g * g - 1) / (12 * g * g) + (0.5 / g) ** 2) ** 0.5
    assert torch.allclose(ref[:2], torch.full((2,), 0.5), atol=1e-3), ref[:2]
    assert torch.allclose(ref[2:], torch.full((2,), uniform), atol=1e-3), (ref[2:], uniform)

    # (c) ONE anchor, two views whose pointmaps place it on different patches → different
    #     per-view references. This is the whole mechanism: no intrinsics, no extrinsics.
    view1 = pos[torch.arange(g * g).flip(0)]                     # same 3D points, reordered
    two = torch.stack([pos, view1])                              # [2, g*g, 3] = 1 bundle, S=2
    anchor = torch.cat([pos[0], torch.tensor([-4.0])])[None, None]
    ref = project_anchors(anchor, two, 2)
    assert torch.allclose(ref[0, 0, :2], uv[0], atol=1e-3)
    assert torch.allclose(ref[1, 0, :2], uv[g * g - 1], atol=1e-3)

    # (d) differentiable in the anchor — the reason the Delta(xyz, log r) head can learn at all
    a = torch.cat([pos[10], torch.tensor([-1.0])])[None, None].requires_grad_(True)
    project_anchors(a, token_xyz, 1).sum().backward()
    assert a.grad is not None and torch.isfinite(a.grad).all() and float(a.grad.abs().sum()) > 0

    # (e) the pyramid gather: level 0 is the input verbatim, extra levels are nearest resamples
    shapes = torch.as_tensor([[g, g], [3, 3], [2, 2]])
    pyr = pyramid_token_xyz(token_xyz, shapes)
    assert pyr.shape == (1, g * g + 9 + 4, 3)
    assert torch.equal(pyr[:, :g * g], token_xyz)
    lvl1 = pyr[0, g * g:g * g + 9].reshape(3, 3, 3)
    src = token_xyz[0].reshape(g, g, 3)
    for r in range(3):
        for c in range(3):
            assert (lvl1[r, c] == src.reshape(-1, 3)).all(-1).any(), (r, c)  # a real patch, not a blend

    # (f) normalisation is zero-mean / unit-RMS and ignores the low-confidence tail, which is
    #     where VGGT's pointmap puts its wild values
    xyz = torch.randn(2, 16, 3) * 4.0 + 100.0
    conf = torch.ones(2, 16)
    conf[0, :8] = 0.0
    xyz[0, :8] = 1e4                                             # outliers, all low-confidence
    n = normalize_token_xyz(xyz, conf)
    sel = torch.cat([n[0, 8:], n[1]]).reshape(-1, 3)
    assert float(sel.mean(0).abs().max()) < 1e-4
    assert abs(float((sel - sel.mean(0)).pow(2).sum(-1).mean().sqrt()) - 1.0) < 1e-4
    print("✅ soft-nearest-patch projection, radius limits, pyramid gather, normalisation OK\n")


def test_anchor3d_head_wiring():
    """The head must refuse to run an anchor_3d model without positions, and must not grow any
    parameter when the flag is off (every published checkpoint depends on that)."""
    print("=== Testing anchor_3d head wiring ===")
    plain = _tiny_head(dn="no", num_queries=8)
    anch = _tiny_head(dn="no", num_queries=8, anchor_3d=True)
    assert plain.head_config["anchor_3d"] is False
    extra = set(anch.state_dict()) - set(plain.state_dict())
    assert extra and all("anchor_embed" in k for k in extra), extra
    assert set(plain.state_dict()) - set(anch.state_dict()) == set()

    tokens = torch.randn(2, 5 + 64, 64)
    try:
        anch(tokens, 5)
    except ValueError as e:
        assert "token_xyz" in str(e)
    else:
        raise AssertionError("anchor_3d without token_xyz must raise, not silently fall back")

    # ... and the 2D path must ignore token_xyz entirely
    a, _ = plain.eval()(tokens, 5)
    b, _ = plain(tokens, 5, token_xyz=torch.randn(2, 64, 3))
    assert torch.equal(a["pred_boxes"], b["pred_boxes"])

    # non-two-stage falls back to learned anchors, initialised at the documented radius
    lt = _tiny_head(dn="no", two_stage=False, learn_tgt=True, initialize_box_type="no",
                    anchor_3d=True).eval()
    assert torch.allclose(lt.predictor.anchor_query_embed.weight[:, 3],
                          torch.full((lt.num_queries,), ANCHOR_LOG_R0))
    out, _ = lt(tokens, 5, token_xyz=torch.randn(2, 64, 3))
    assert out["pred_masks"].shape[:2] == (2, lt.num_queries)
    print("✅ anchor_3d adds only the Delta(xyz, log r) head and demands its positions\n")


if __name__ == "__main__":
    test_ms_deform_attn_core()
    test_pixel_decoder()
    test_decoder_configs()
    test_unported_box_init_is_rejected()
    test_masks_to_boxes()
    test_head_config_round_trip()
    test_anchor3d_geometry()
    test_anchor3d_head_wiring()
    print("All test_maskdino_model tests passed! ✅")
