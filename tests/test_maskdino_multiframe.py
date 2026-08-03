#!/usr/bin/env python3
"""
Multi-frame MaskDINO (shared queries across the frames of a bundle, docs/MASKDINO.md §8).
Standalone, CPU-only, no VGGT weights.

  - CrossFrameAttention: mixes only the shared (non-DN) queries, only across frames of the SAME
    bundle, and is permutation-equivariant in the frame order;
  - build_bundle_target / expand_bundle_indices: the per-frame GT is re-linked by global instance
    id and the bundle assignment is projected back onto the frames where the instance is visible;
  - MultiFrameHungarianMatcher recovers a planted multi-view assignment;
  - the head runs with frames_per_sample=S and gives every frame the same query semantics
    (shared query init), and single-frame behaviour is bit-identical to before;
  - a 60-step overfit of the whole multi-frame path (bundle matching + per-frame losses).
  - checkpoint_best_bundle.pth selection (docs/todo.md 2b): update_best in
    scripts/train_maskdino.py picks the right epoch off a synthetic metrics sequence, and the
    bundle checkpoint path only exists for --multi_frame runs.
"""

import inspect
import re
import sys
from argparse import Namespace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from maskdino_fixtures import _tiny_head
from models.maskdino import (CrossFrameAttention, HungarianMatcher, MultiFrameHungarianMatcher,
                             SetCriterion, build_bundle_target, build_weight_dict,
                             expand_bundle_indices)
from models.maskdino.box_ops import masks_to_boxes_normalized

torch.manual_seed(0)


def _bundle_frame_targets(s=3, n=2, hw=(8, 8), num_classes=19):
    """One synthetic bundle: n instances, instance i visible in every frame except frame i."""
    labels = torch.randint(0, num_classes, (n,))
    frames = []
    for f in range(s):
        masks, lbl, gids = [], [], []
        for i in range(n):
            if f == i:                       # deliberately invisible in one view
                continue
            m = torch.zeros(*hw)
            m[i * 2:i * 2 + 2, f:f + 3] = 1.0
            masks.append(m)
            lbl.append(labels[i])
            gids.append(i + 1)
        m = torch.stack(masks) if masks else torch.zeros(0, *hw)
        frames.append({
            "labels": torch.as_tensor(lbl, dtype=torch.long),
            "masks": m,
            "boxes": masks_to_boxes_normalized(m),
            "global_ids": torch.as_tensor(gids, dtype=torch.long),
        })
    return frames, labels


def test_cross_frame_attention():
    print("=== Testing CrossFrameAttention ===")
    torch.manual_seed(0)
    d, nq, s, b, shared = 16, 7, 3, 2, 5
    block = CrossFrameAttention(d, nheads=4).eval()
    tgt = torch.randn(nq, b * s, d)

    out = block(tgt, frames_per_sample=s, num_shared=shared)
    assert out.shape == tgt.shape
    # the DN slots at the front are untouched ...
    assert torch.equal(out[:nq - shared], tgt[:nq - shared])
    assert not torch.allclose(out[nq - shared:], tgt[nq - shared:])
    # ... and S=1 is a no-op, so single-frame runs are unaffected
    assert torch.equal(block(tgt, frames_per_sample=1, num_shared=shared), tgt)

    # bundles must not leak into each other: perturbing bundle 1 leaves bundle 0 unchanged
    other = tgt.clone()
    other[:, s:] = torch.randn(nq, s, d)
    out2 = block(other, frames_per_sample=s, num_shared=shared)
    assert torch.allclose(out[:, :s], out2[:, :s], atol=1e-6)

    # permutation-equivariant in the frame order (views are an unordered set)
    perm = [2, 0, 1]
    permuted = tgt.clone()
    permuted[:, :s] = tgt[:, perm]
    out3 = block(permuted, frames_per_sample=s, num_shared=shared)
    assert torch.allclose(out3[:, :s], out[:, perm], atol=1e-5)
    print("✅ cross-frame block mixes shared queries within a bundle only\n")


def test_bundle_targets_and_index_expansion():
    print("=== Testing bundle GT + index expansion ===")
    s, n = 3, 2
    frames, labels = _bundle_frame_targets(s=s, n=n)
    bt = build_bundle_target(frames)

    assert bt["labels"].tolist() == labels.tolist()
    assert bt["masks"].shape == (n, s, 8, 8) and bt["boxes"].shape == (n, s, 4)
    # instance i is invisible exactly in frame i
    assert bt["valid"].tolist() == [[f != i for f in range(s)] for i in range(n)]
    assert torch.equal(bt["masks"][0, 0].sum(), torch.tensor(0.0))
    for i in range(n):
        for f in range(s):
            row = int(bt["frame_row"][i, f])
            if row < 0:
                assert not bool(bt["valid"][i, f])
            else:
                assert torch.equal(bt["masks"][i, f], frames[f]["masks"][row])
                assert int(frames[f]["global_ids"][row]) == int(bt["global_ids"][i])

    # a bundle assignment (query 3 → instance 0, query 5 → instance 1) becomes per-frame indices
    indices = [(torch.tensor([3, 5]), torch.tensor([0, 1]))]
    per_frame = expand_bundle_indices(indices, [bt], s)
    assert len(per_frame) == s
    for f in range(s):
        src, tgt = per_frame[f]
        # only the instances visible in frame f survive, mapped to their row in that frame
        expected = [(q, int(bt["frame_row"][i, f]))
                    for q, i in zip([3, 5], [0, 1]) if bt["valid"][i, f]]
        assert list(zip(src.tolist(), tgt.tolist())) == expected, (f, src, tgt, expected)
    print("✅ bundle targets re-link frames by global id; indices project back per frame\n")


def test_multiframe_matcher_recovers_planted_assignment():
    print("=== Testing MultiFrameHungarianMatcher ===")
    s, n, q, hw, c = 3, 2, 6, (8, 8), 19
    frames, _ = _bundle_frame_targets(s=s, n=n, hw=hw, num_classes=c)
    bt = build_bundle_target(frames)

    # plant instance 0 on query 4 and instance 1 on query 1, in EVERY frame
    planted = {0: 4, 1: 1}
    pred_masks = torch.full((s, q, *hw), -10.0)
    pred_logits = torch.full((s, q, c), -10.0)
    pred_boxes = torch.rand(s, q, 4) * 0.1
    for i, qi in planted.items():
        for f in range(s):
            pred_masks[f, qi] = bt["masks"][i, f] * 20 - 10
            pred_logits[f, qi, int(bt["labels"][i])] = 10.0
            pred_boxes[f, qi] = bt["boxes"][i, f]

    matcher = MultiFrameHungarianMatcher(num_points=0)
    out = {"pred_logits": pred_logits, "pred_masks": pred_masks, "pred_boxes": pred_boxes}
    (src, tgt), = matcher(out, [bt], s)
    got = {int(t): int(sq) for sq, t in zip(src, tgt)}
    assert got == planted, (got, planted)

    # a bundle with no GT at all matches nothing instead of crashing
    empty = {k: v[:0] for k, v in bt.items()}
    (src0, tgt0), = matcher(out, [empty], s)
    assert src0.numel() == 0 and tgt0.numel() == 0
    print("✅ bundle matcher recovers the planted multi-view assignment\n")


def test_head_shared_queries():
    print("=== Testing shared-query head forward ===")
    torch.manual_seed(0)
    s, b, hh, mem = 3, 2, 8, 64
    head = _tiny_head(dec_layers=2, enc_layers=1, dn="no", num_queries=10,
                      cross_frame_attn=True).eval()
    tokens = torch.randn(b * s, 5 + hh * hh, mem)

    out, _ = head(tokens, 5, None, frames_per_sample=s)
    assert out["pred_logits"].shape == (b * s, 10, 19)
    assert out["pred_masks"].shape[:2] == (b * s, 10)
    assert out["pred_boxes"].shape == (b * s, 10, 4)

    # The initial (pre-decoder) prediction is what shows the queries are shared: the interm
    # output's boxes come from the ONE per-bundle selection broadcast to the frames.
    interm = out["interm_outputs"]["pred_boxes"]
    for f in range(1, s):
        assert torch.allclose(interm[f], interm[0], atol=1e-6), f
        assert torch.allclose(interm[s + f], interm[s], atol=1e-6), f
    assert not torch.allclose(interm[0], interm[s], atol=1e-6)   # different bundles differ

    # frames of a bundle still get their OWN masks/boxes after the per-frame refinement
    assert not torch.allclose(out["pred_masks"][0], out["pred_masks"][1], atol=1e-4)

    # head_config still describes every constructor argument (the checkpoint round-trip contract)
    assert head.head_config["cross_frame_attn"] is True
    print("✅ shared query init broadcast per bundle, per-frame refinement kept\n")


def test_single_frame_path_unchanged():
    """The multi-frame code must be inert at S=1 — the published single-frame numbers depend
    on the default path being byte-identical."""
    print("=== Testing S=1 equivalence ===")
    torch.manual_seed(0)
    head = _tiny_head(dec_layers=2, enc_layers=1, dn="no", num_queries=10).eval()
    tokens = torch.randn(4, 5 + 64, 64)
    a, _ = head(tokens, 5)
    b, _ = head(tokens, 5, None, frames_per_sample=1)
    for k in ("pred_logits", "pred_masks", "pred_boxes"):
        assert torch.equal(a[k], b[k]), k
    assert head.predictor.decoder.cross_frame is None   # no extra parameters by default
    print("✅ frames_per_sample=1 reproduces the single-frame forward exactly\n")


def test_multiframe_overfit():
    print("=== Testing 60-step multi-frame overfit ===")
    torch.manual_seed(0)
    s, hh, mem, q = 3, 8, 64, 12
    head = _tiny_head(dec_layers=2, enc_layers=1, dn="seg", num_queries=q, cross_frame_attn=True)
    tokens = torch.randn(s, 5 + hh * hh, mem)
    frames, _ = _bundle_frame_targets(s=s, n=2, hw=(hh, hh))
    bundle = [build_bundle_target(frames)]

    weight_dict = build_weight_dict(dec_layers=2, two_stage=True, dn="seg")
    criterion = SetCriterion(19, HungarianMatcher(num_points=64), weight_dict,
                             losses=["labels", "masks", "boxes"], num_points=0,
                             dn="seg", dn_losses=["labels", "masks", "boxes"],
                             bundle_matcher=MultiFrameHungarianMatcher(num_points=0))
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)

    head.train()
    first = last = None
    for step in range(60):
        opt.zero_grad()
        out, mask_dict = head(tokens, 5, frames, frames_per_sample=s)
        losses = criterion(out, frames, mask_dict, bundle_targets=bundle, frames_per_sample=s)
        total = sum(losses[k] * weight_dict[k] for k in losses if k in weight_dict)
        total.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        if step == 0:
            first = float(total)
        last = float(total)
    print(f"    loss {first:.2f} → {last:.2f}")
    assert last < 0.6 * first, f"multi-frame overfit did not converge: {first:.2f} → {last:.2f}"

    # every cross-frame block must actually have been trained (i.e. it is in the graph)
    grads = [p.grad for blk in head.predictor.decoder.cross_frame for p in blk.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(float(g.abs().sum()) > 0 for g in grads)
    print("✅ multi-frame path trains end to end (bundle matching + cross-frame blocks)\n")


def test_bundle_batching_and_eval():
    """The training/eval plumbing: bundles keep their frames contiguous, and scoring reports the
    per-frame AND the per-bundle (multi-view) protocol."""
    print("=== Testing bundle batching + multi-frame eval ===")
    from argparse import Namespace

    from train.maskdino_data import bundle_index, gather_bundle_batch

    torch.manual_seed(0)
    s, hh, mem, q = 3, 8, 64, 10
    head = _tiny_head(dec_layers=2, enc_layers=1, dn="no", num_queries=q, cross_frame_attn=True)
    model = torch.nn.Module()
    model.head = head

    scenes = []
    for name in ("sceneA", "sceneB"):
        frames, _ = _bundle_frame_targets(s=s, n=2, hw=(hh, hh))
        scenes.append({"name": name, "split": "val", "bundles": [
            {"features": torch.randn(s, 5 + hh * hh, mem), "patch_start_idx": 5,
             "targets": frames, "images": None}]})

    samples = bundle_index(scenes)
    assert samples == [(0, 0), (1, 0)], samples
    feats, frame_targets, bundles, psi, got_s = gather_bundle_batch(scenes, samples, "cpu")
    assert got_s == s and psi == 5
    assert feats.shape == (len(samples) * s, 5 + hh * hh, mem)
    assert len(frame_targets) == len(samples) * s and len(bundles) == len(samples)
    # frames of a bundle must be contiguous and in order — everything downstream assumes it
    for b, (si, bi) in enumerate(samples):
        for f in range(s):
            assert torch.equal(feats[b * s + f], scenes[si]["bundles"][bi]["features"][f])

    args = Namespace(multi_frame=True, eval_topk=100, score_threshold=0.25, eval_batch_frames=s)
    from train.maskdino_eval import eval_scenes, mean_metric
    per_scene = eval_scenes(model, scenes, args, "cpu")
    assert set(per_scene) == {"sceneA", "sceneB"}
    for m in per_scene.values():
        for k in ("mIoU", "AP50", "bundle_mIoU", "bundle_AP50", "bundle_mIoU_all", "bundle_num_gt"):
            assert k in m, k
        # the bundle has 2 instances; per-frame views see fewer
        assert m["bundle_num_gt_all"] == 2, m["bundle_num_gt_all"]
    assert 0.0 <= mean_metric(per_scene, "bundle_AP50") <= 1.0
    print("✅ bundle batching layout + per-frame/per-bundle scoring\n")


def test_multiframe_visualisation():
    """`visualize` must feed the WHOLE bundle to a shared-query model (it cannot score a subset
    of the frames) while still drawing at most `max_frames` panels."""
    print("=== Testing multi-frame visualisation ===")
    import tempfile
    from argparse import Namespace

    from train.maskdino_eval import visualize

    torch.manual_seed(0)
    s, hh, mem, q = 4, 8, 64, 10
    head = _tiny_head(dec_layers=2, enc_layers=1, dn="no", num_queries=q, cross_frame_attn=True)
    model = torch.nn.Module()
    model.head = head
    frames, _ = _bundle_frame_targets(s=s, n=2, hw=(hh, hh))
    scenes = [{"name": "sceneA", "split": "val", "bundles": [
        {"features": torch.randn(s, 5 + hh * hh, mem), "patch_start_idx": 5, "targets": frames,
         "images": torch.randint(0, 255, (s, 3, 32, 32), dtype=torch.uint8)}]}]

    args = Namespace(multi_frame=True, score_threshold=0.25, eval_topk=100)
    with tempfile.TemporaryDirectory() as tmp:
        written = visualize(model, scenes, args, "cpu", Path(tmp), max_scenes=1, max_frames=2)
        assert written == 2, written
        assert len(list(Path(tmp).glob("*.png"))) == 2
    print("✅ multi-frame visualisation runs the full bundle, draws max_frames panels\n")


def test_update_best_selects_peak_epoch():
    """update_best (scripts/train_maskdino.py, docs/todo.md 2b) must track the epoch a metric
    peaked at and only save the checkpoint on a strict improvement."""
    print("=== Testing update_best selects the right epoch ===")
    import train_maskdino

    saved = []

    def fake_save_checkpoint(path, model, args, epoch, train_metrics, val_metrics, best_info):
        saved.append((path, epoch, dict(best_info)))

    orig_save = train_maskdino.save_checkpoint
    train_maskdino.save_checkpoint = fake_save_checkpoint
    try:
        best = {"val_bundle_AP50": -1.0, "epoch": -1}
        sequence = [0.1, 0.4, 0.3, 0.6, 0.5]   # peak at the 4th synthetic eval
        for i, val in enumerate(sequence):
            best = train_maskdino.update_best(best, "val_bundle_AP50", val, i + 1,
                                              Path("dummy_bundle.pth"), None, None, {}, {})
        assert best == {"val_bundle_AP50": 0.6, "epoch": 4}, best
        # only strict improvements (epochs 1, 2, 4) trigger a save
        assert [e for _, e, _ in saved] == [1, 2, 4], saved
        assert saved[-1][2] == {"val_bundle_AP50": 0.6, "epoch": 4}

        # no path -> selection still tracked, nothing saved
        saved.clear()
        best2 = train_maskdino.update_best({"val_bundle_AP50": -1.0, "epoch": -1},
                                           "val_bundle_AP50", 0.2, 1, None, None, None, {}, {})
        assert best2 == {"val_bundle_AP50": 0.2, "epoch": 1}
        assert saved == []
    finally:
        train_maskdino.save_checkpoint = orig_save
    print("✅ update_best tracks the peak epoch and saves only on improvement\n")


def test_bundle_checkpoint_path_requires_multi_frame():
    """Single-frame args must produce no checkpoint_best_bundle.pth path — the bundle_AP50 key
    does not exist in per-frame metrics, so the path has to be gated on args.multi_frame."""
    print("=== Testing checkpoint_best_bundle.pth is gated on --multi_frame ===")
    import train_maskdino

    src = inspect.getsource(train_maskdino.main)
    m = re.search(r'best_bundle_path = (.+)\n', src)
    assert m, "could not find the best_bundle_path assignment in main()"
    expr = m.group(1)
    assert "args.multi_frame" in expr, expr

    run_dir = Path("/tmp/some_run_dir")
    for multi_frame, expect_path in [(False, False), (True, True)]:
        args = Namespace(multi_frame=multi_frame)
        result = eval(expr, {"Path": Path}, {"run_dir": run_dir, "args": args})
        assert (result is not None) == expect_path, (multi_frame, result)
    print("✅ bundle checkpoint path exists only when --multi_frame is set\n")


if __name__ == "__main__":
    test_cross_frame_attention()
    test_bundle_targets_and_index_expansion()
    test_multiframe_matcher_recovers_planted_assignment()
    test_head_shared_queries()
    test_single_frame_path_unchanged()
    test_multiframe_overfit()
    test_bundle_batching_and_eval()
    test_multiframe_visualisation()
    test_update_best_selects_peak_epoch()
    test_bundle_checkpoint_path_requires_multi_frame()
    print("All test_maskdino_multiframe tests passed! ✅")
