#!/usr/bin/env python3
"""
MaskDINO trial validation (docs/MASKDINO_TRIAL.md). Standalone, CPU-only, no VGGT weights.

Checks, component by component:
  - the pure-PyTorch MSDeformAttn core against a naive explicit-loop reference;
  - VGGTPixelDecoder shapes (levels, mask_features, mask_upsample) + gradient flow;
  - MaskDINODecoder output/aux/interm/DN shapes in every config combination that the training
    script can produce (two-stage on/off, dn on/off, initialize_box_type, train/eval);
  - box_ops.masks_to_boxes against hand-computed boxes;
  - HungarianMatcher on a planted-perfect-prediction case (must recover the identity matching);
  - SetCriterion key set vs. build_weight_dict, and a zero-loss sanity check on perfect preds;
  - per-frame GT builder (labels/boxes/masks) from a synthetic scene batch;
  - the per-frame metric slicing shared with scripts/eval_perframe.py;
  - head_config round-trip (rebuild → identical state_dict keys);
  - a 60-step overfit of the whole head on one synthetic frame (loss must drop a lot).
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from models.maskdino import (HungarianMatcher, MaskDINOVGGTHead, SetCriterion,
                             build_head_from_config, build_weight_dict, to_scannet_class_logits)
from models.maskdino import box_ops
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


def _synthetic_targets(B, n, hw, num_classes=19):
    targets = []
    for _ in range(B):
        masks = torch.zeros(n, *hw)
        for i in range(n):
            masks[i, i:i + 2, i:i + 2] = 1.0
        targets.append({
            "labels": torch.randint(0, num_classes, (n,)),
            "masks": masks,
            "boxes": box_ops.masks_to_boxes_normalized(masks),
        })
    return targets


def _tiny_head(**kw):
    cfg = dict(memory_dim=64, hidden_dim=32, mask_dim=32, num_classes=19, num_queries=12,
               num_feature_levels=3, enc_layers=1, dec_layers=2, nheads=4,
               enc_dim_feedforward=64, dec_dim_feedforward=64, enc_n_points=2, dec_n_points=2,
               dn_num=12)
    cfg.update(kw)
    return MaskDINOVGGTHead(**cfg)


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


def test_matcher_recovers_planted_assignment():
    print("=== Testing HungarianMatcher on planted predictions ===")
    n, q, hw = 3, 8, (8, 8)
    targets = _synthetic_targets(1, n, hw)
    t = targets[0]
    # Queries 5, 1, 6 carry (almost) exactly GT 0, 1, 2; the rest are empty background queries.
    planted = [5, 1, 6]
    logits = torch.full((1, q, 19), -8.0)
    masks = torch.full((1, q, *hw), -8.0)
    boxes = torch.rand(1, q, 4) * 0.02 + 0.5
    for gi, qi in enumerate(planted):
        logits[0, qi, t["labels"][gi]] = 8.0
        masks[0, qi] = t["masks"][gi] * 16.0 - 8.0
        boxes[0, qi] = t["boxes"][gi]
    outputs = {"pred_logits": logits, "pred_masks": masks, "pred_boxes": boxes}

    matcher = HungarianMatcher(num_points=64)
    (pred_idx, tgt_idx), = matcher(outputs, targets)
    got = {int(g): int(p) for p, g in zip(pred_idx, tgt_idx)}
    assert got == {0: 5, 1: 1, 2: 6}, got

    # a frame with no GT must yield an empty (not crashing) assignment
    (pi, ti), = matcher(outputs, [{"labels": torch.zeros(0, dtype=torch.long),
                                   "masks": torch.zeros(0, *hw),
                                   "boxes": torch.zeros(0, 4)}])
    assert pi.numel() == 0 and ti.numel() == 0
    print("✅ Matcher recovers the planted assignment and survives empty GT\n")


def test_criterion_keys_and_perfect_predictions():
    print("=== Testing SetCriterion (keys, weights, perfect-prediction floor) ===")
    B, h, nq, dec_layers = 2, 8, 12, 2
    targets = _synthetic_targets(B, 3, (h, h))
    head = _tiny_head(dec_layers=dec_layers, dn="seg")
    head.train()
    tokens = torch.randn(B, 5 + h * h, 64)
    out, mask_dict = head(tokens, 5, targets)

    weight_dict = build_weight_dict(dec_layers=dec_layers, two_stage=True, dn="seg")
    criterion = SetCriterion(19, HungarianMatcher(num_points=64), weight_dict,
                             losses=["labels", "masks", "boxes"], num_points=64,
                             dn="seg", dn_losses=["labels", "masks", "boxes"])
    losses = criterion(out, targets, mask_dict)
    missing = [k for k in losses if k not in weight_dict]
    assert not missing, f"loss keys with no weight: {missing}"
    total = sum(losses[k] * weight_dict[k] for k in losses)
    assert torch.isfinite(total), total
    total.backward()
    grads = [p.grad for p in head.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)

    # Perfect predictions on the final layer only: class + mask + box losses must be tiny.
    t = targets[0]
    n = t["labels"].shape[0]
    logits = torch.full((1, nq, 19), -12.0)
    masks = torch.full((1, nq, h, h), -12.0)
    boxes = torch.rand(1, nq, 4) * 0.01 + 0.5
    for i in range(n):
        logits[0, i, t["labels"][i]] = 12.0
        masks[0, i] = t["masks"][i] * 24.0 - 12.0
        boxes[0, i] = t["boxes"][i]
    perfect = {"pred_logits": logits, "pred_masks": masks, "pred_boxes": boxes}
    crit2 = SetCriterion(19, HungarianMatcher(num_points=0 or 64), {},
                         losses=["labels", "masks", "boxes"], num_points=0)
    l = crit2(perfect, [t])
    assert l["loss_ce"] < 1e-2, l["loss_ce"]
    assert l["loss_mask"] < 1e-2, l["loss_mask"]
    assert l["loss_dice"] < 1e-2, l["loss_dice"]
    assert l["loss_bbox"] < 1e-4, l["loss_bbox"]
    assert l["loss_giou"] < 1e-4, l["loss_giou"]
    print("✅ Criterion keys match build_weight_dict; perfect predictions give ~0 loss\n")


def test_frame_targets_builder():
    print("=== Testing per-frame GT builder ===")
    from train_maskdino import build_frame_targets

    S, H, W, out_hw = 2, 16, 16, (8, 8)
    masks = torch.zeros(S, H, W, dtype=torch.long)
    masks[0, 2:10, 2:10] = 1     # instance 1 visible in frame 0 only
    masks[:, 12:15, 12:15] = 2   # instance 2 visible in both frames
    batch = {"classes": torch.tensor([3, 7]), "masks": masks,
             "coordinates": torch.zeros(2, 2), "frame_names": ["a", "b"]}

    per_frame = build_frame_targets(batch, out_hw, "cpu")
    assert len(per_frame) == S
    # frame 0 sees both instances, frame 1 only the second
    assert per_frame[0]["labels"].tolist() == [2, 6], per_frame[0]["labels"]  # 1-based → 0-based
    assert per_frame[1]["labels"].tolist() == [6], per_frame[1]["labels"]
    assert per_frame[0]["masks"].shape == (2, 8, 8)
    assert per_frame[0]["masks"].sum() > 0 and per_frame[1]["masks"].sum() > 0
    assert per_frame[0]["boxes"].shape == (2, 4)
    # instance 1 covers pixels 2..9 of 16 → normalized box centre ≈ 0.375, size 0.5
    assert torch.allclose(per_frame[0]["boxes"][0],
                          torch.tensor([0.375, 0.375, 0.5, 0.5]), atol=0.13), \
        per_frame[0]["boxes"][0]
    assert per_frame[0]["global_ids"].tolist() == [1, 2]
    print("✅ build_frame_targets produces per-frame labels/masks/boxes\n")


def test_perframe_metrics():
    """The per-frame protocol used to score BOTH this trial and (via scripts/eval_perframe.py)
    the existing D4RT checkpoints — the only apples-to-apples comparison (trial doc §6)."""
    print("=== Testing per-frame metric slicing ===")
    from eval_perframe import perframe_metrics

    S, h, w, Ng, N = 3, 6, 6, 2, 4
    gt_masks = torch.zeros(Ng, S, h, w)
    gt_masks[0, 0, 0:3, 0:3] = 1        # instance 0: frame 0 only
    gt_masks[1, 0:2, 3:6, 3:6] = 1      # instance 1: frames 0 and 1
    gt_classes = torch.tensor([4, 9])   # frame 2 has no GT at all

    pred_masks = torch.full((N, S, h, w), -10.0)
    pred_masks[0] = gt_masks[0] * 20 - 10
    pred_masks[1] = gt_masks[1] * 20 - 10
    class_logits = torch.full((N, 20), -10.0)
    class_logits[0, 4] = 10.0
    class_logits[1, 9] = 10.0
    class_logits[2:, 0] = 10.0          # two background queries, dropped by the softmax path

    rows = perframe_metrics(pred_masks, class_logits, gt_masks, gt_classes)
    assert len(rows) == 2, f"frame 2 has no GT and must be skipped, got {len(rows)} rows"
    assert rows[0]["num_gt"] == 2 and rows[1]["num_gt"] == 1, [r["num_gt"] for r in rows]
    # Frame 1: instance 0 is not visible, so query 0's empty mask must be DROPPED, not counted
    # as a false positive (it is correct multi-view behaviour) — otherwise AP50 would be 0.5.
    assert rows[0]["num_pred"] == 2 and rows[1]["num_pred"] == 1, [r["num_pred"] for r in rows]
    assert all(r["mIoU"] > 0.99 and r["AP50"] > 0.99 for r in rows), rows

    # A prediction that is right in frame 0 but empty in frame 1 must only score in frame 0.
    partial = pred_masks.clone()
    partial[1, 1] = -10.0
    rows2 = perframe_metrics(partial, class_logits, gt_masks, gt_classes)
    assert rows2[0]["mIoU"] > 0.99 and rows2[1]["mIoU"] == 0.0, [r["mIoU"] for r in rows2]
    print("✅ perframe_metrics slices frames, skips empty GT, and scores per frame\n")


def test_head_config_round_trip():
    print("=== Testing head_config round-trip ===")
    head = _tiny_head(dn="no", two_stage=False, learn_tgt=True, initialize_box_type="no")
    rebuilt = build_head_from_config(head.head_config)
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


def test_overfit_single_frame():
    print("=== Testing 60-step overfit on one synthetic frame ===")
    torch.manual_seed(0)
    h, mem = 8, 64
    head = _tiny_head(dec_layers=2, enc_layers=1, dn="seg", num_queries=12)
    tokens = torch.randn(1, 5 + h * h, mem)
    targets = _synthetic_targets(1, 2, (h, h))
    weight_dict = build_weight_dict(dec_layers=2, two_stage=True, dn="seg")
    criterion = SetCriterion(19, HungarianMatcher(num_points=64), weight_dict,
                             losses=["labels", "masks", "boxes"], num_points=0,
                             dn="seg", dn_losses=["labels", "masks", "boxes"])
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)

    head.train()
    first = last = None
    for step in range(60):
        opt.zero_grad()
        out, mask_dict = head(tokens, 5, targets)
        losses = criterion(out, targets, mask_dict)
        total = sum(losses[k] * weight_dict[k] for k in losses if k in weight_dict)
        total.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        if step == 0:
            first = float(total)
        last = float(total)
    print(f"    loss {first:.2f} → {last:.2f}")
    assert last < 0.6 * first, f"overfit did not converge: {first:.2f} → {last:.2f}"
    print("✅ Head overfits a single synthetic frame\n")


if __name__ == "__main__":
    test_ms_deform_attn_core()
    test_pixel_decoder()
    test_decoder_configs()
    test_masks_to_boxes()
    test_matcher_recovers_planted_assignment()
    test_criterion_keys_and_perfect_predictions()
    test_frame_targets_builder()
    test_perframe_metrics()
    test_head_config_round_trip()
    test_overfit_single_frame()
    print("All MaskDINO-trial tests passed! ✅")
