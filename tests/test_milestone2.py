#!/usr/bin/env python3
"""
Milestone 2 validation: no-object loss, unprompted grid queries, and the per-step query
augmentation used by scripts/train_multiscene.py. Standalone (CPU, no backbone weights).
"""

import json
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from train.loss import D4RTLoss
from train_overfit import generate_grid_queries
from train_multiscene import (
    make_train_queries, photometric_jitter, append_jsonl, build_eval_record,
    moving_average, early_stop_should_stop, build_scheduler,
)


def _toy_predictions(num_queries=6, num_classes=20, grid=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    class_logits = torch.randn(num_queries, num_classes, generator=g, requires_grad=True)
    mask_embeddings = torch.randn(num_queries, 256, generator=g)
    coordinates = torch.rand(num_queries, 2, generator=g)
    pred_masks = torch.randn(num_queries, 2, grid, grid, generator=g, requires_grad=True)
    return class_logits, mask_embeddings, coordinates, pred_masks


def _toy_gt(grid=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    gt_classes = torch.tensor([3, 7])
    gt_coordinates = torch.rand(2, 2, generator=g)
    gt_masks = (torch.rand(2, 2, grid, grid, generator=g) > 0.5).float()
    return gt_classes, gt_coordinates, gt_masks


def test_no_object_loss_disabled_matches_old_behavior():
    """no_object_weight=None must reproduce the Milestone-1 matched-only class loss."""
    cl, me, co, pm = _toy_predictions()
    gc, gco, gm = _toy_gt()
    loss_old = D4RTLoss(coord_loss_weight=0.0, mask_embed_loss_weight=0.0)
    loss_new_disabled = D4RTLoss(coord_loss_weight=0.0, mask_embed_loss_weight=0.0,
                                 no_object_weight=None)
    _, comps_a = loss_old(cl, me, co, gc, gt_coordinates=gco, gt_masks=gm, pred_masks=pm)
    _, comps_b = loss_new_disabled(cl, me, co, gc, gt_coordinates=gco, gt_masks=gm, pred_masks=pm)
    assert torch.allclose(comps_a["class_loss"], comps_b["class_loss"])
    print("✓ no_object_weight=None reproduces matched-only class loss")


def test_no_object_loss_supervises_unmatched_queries():
    """With no-object loss on, predicting background on unmatched queries must be cheaper
    than predicting a confident foreground class on them."""
    gc, gco, gm = _toy_gt()
    loss_fn = D4RTLoss(coord_loss_weight=0.0, mask_embed_loss_weight=0.0, no_object_weight=0.1)

    N, C = 6, 20
    # Both predictions match GT perfectly on the first 2 queries (correct classes + masks).
    base_logits = torch.full((N, C), -4.0)
    base_logits[0, 3] = 4.0
    base_logits[1, 7] = 4.0
    pred_masks = torch.full((N, 2, 4, 4), -8.0)
    pred_masks[0] = gm[0] * 16.0 - 8.0
    pred_masks[1] = gm[1] * 16.0 - 8.0
    me = torch.randn(N, 256)
    co = torch.cat([gco, torch.rand(N - 2, 2)], dim=0)

    # Variant A: unmatched queries predict background. Variant B: a foreground class.
    logits_bg = base_logits.clone()
    logits_bg[2:, 0] = 4.0
    logits_fg = base_logits.clone()
    logits_fg[2:, 5] = 4.0

    _, comps_bg = loss_fn(logits_bg, me, co, gc, gt_coordinates=gco, gt_masks=gm, pred_masks=pred_masks)
    _, comps_fg = loss_fn(logits_fg, me, co, gc, gt_coordinates=gco, gt_masks=gm, pred_masks=pred_masks)
    assert comps_bg["class_loss"] < comps_fg["class_loss"], (
        f"background on unmatched should be cheaper: {comps_bg['class_loss']:.4f} vs "
        f"{comps_fg['class_loss']:.4f}")
    print(f"✓ no-object loss prefers background on unmatched queries "
          f"({comps_bg['class_loss']:.4f} < {comps_fg['class_loss']:.4f})")


def test_no_object_loss_gradients():
    cl, me, co, pm = _toy_predictions()
    gc, gco, gm = _toy_gt()
    loss_fn = D4RTLoss(coord_loss_weight=0.0, mask_embed_loss_weight=0.0, no_object_weight=0.1)
    total, comps = loss_fn(cl, me, co, gc, gt_coordinates=gco, gt_masks=gm, pred_masks=pm)
    assert torch.isfinite(total)
    total.backward()
    assert cl.grad is not None and torch.isfinite(cl.grad).all()
    # Unmatched queries must now receive a class gradient too (their rows are non-zero).
    grads_per_query = cl.grad.abs().sum(dim=-1)
    assert (grads_per_query > 0).all(), "every query should receive class supervision"
    print("✓ no-object loss: finite scalar, gradients reach ALL queries")


def test_grid_queries():
    coords, view_ids = generate_grid_queries(num_frames=3, grid_size=5, device="cpu")
    assert coords.shape == (1, 3 * 25, 2) and view_ids.shape == (1, 3 * 25)
    assert coords.min() > 0 and coords.max() < 1
    assert view_ids.min() == 0 and view_ids.max() == 2
    # Each frame gets the identical lattice; per-frame cell count is grid_size^2.
    assert torch.equal(coords[0, :25], coords[0, 25:50])
    assert (view_ids[0, :25] == 0).all() and (view_ids[0, 50:] == 2).all()
    # First cell center of a 5-grid is at 0.1.
    assert torch.allclose(coords[0, 0], torch.tensor([0.1, 0.1]))
    print("✓ grid queries: correct lattice, view ids, and cell centers")


def test_make_train_queries_augmentation():
    torch.manual_seed(0)
    S, Nq, ni = 4, 10, 3
    bundle = {
        "coordinates": torch.rand(1, Nq, 2),
        "view_ids": torch.randint(0, S, (1, Nq)),
        "num_inst_queries": ni,
        "images": torch.zeros(1, S, 3, 8, 8),
    }
    args = Namespace(query_jitter=0.02, fixed_bg=False)
    coords, view_ids = make_train_queries(bundle, args, "cpu")
    assert coords.shape == bundle["coordinates"].shape
    assert coords.min() >= 0 and coords.max() <= 1
    # Instance queries: jittered but close to the originals. Background: resampled.
    assert not torch.equal(coords[:, :ni], bundle["coordinates"][:, :ni])
    assert (coords[:, :ni] - bundle["coordinates"][:, :ni]).abs().max() < 0.2
    assert not torch.equal(coords[:, ni:], bundle["coordinates"][:, ni:])
    # Original bundle must be untouched (clone semantics).
    c2, _ = make_train_queries(bundle, args, "cpu")
    assert not torch.equal(coords, c2)  # fresh draws every call

    args_fixed = Namespace(query_jitter=0.0, fixed_bg=True)
    coords_f, view_ids_f = make_train_queries(bundle, args_fixed, "cpu")
    assert torch.equal(coords_f, bundle["coordinates"])
    assert torch.equal(view_ids_f, bundle["view_ids"])
    print("✓ query augmentation: jitter bounded, bg resampled, M1 behavior recoverable")


def test_train_grid_queries():
    """--train_grid_queries appends a random-offset eval grid to the training queries
    (off by default), so duplicate-suppression is exercised during training (Phase 2)."""
    torch.manual_seed(0)
    S, Nq, ni, g = 3, 10, 3, 4
    bundle = {
        "coordinates": torch.rand(1, Nq, 2),
        "view_ids": torch.randint(0, S, (1, Nq)),
        "num_inst_queries": ni,
        "images": torch.zeros(1, S, 3, 8, 8),
    }
    args = Namespace(query_jitter=0.0, fixed_bg=True, train_grid_queries=True, grid_size=g)
    coords, view_ids = make_train_queries(bundle, args, "cpu")
    extra = S * g * g
    assert coords.shape == (1, Nq + extra, 2), coords.shape
    assert view_ids.shape == (1, Nq + extra)
    assert coords.min() >= 0 and coords.max() <= 1
    # The original (centroid+bg) queries are preserved as the prefix; grid queries appended.
    assert torch.equal(coords[:, :Nq], bundle["coordinates"])
    # Grid view ids cover every frame.
    assert set(view_ids[0, Nq:].tolist()) == set(range(S))

    # Off by default → no extra queries (Milestone-2 behavior).
    args_off = Namespace(query_jitter=0.0, fixed_bg=True, train_grid_queries=False, grid_size=g)
    c_off, _ = make_train_queries(bundle, args_off, "cpu")
    assert c_off.shape == (1, Nq, 2)
    print("✓ train_grid_queries: appends a random-offset grid, off by default")


def test_query_mode_train_queries():
    """Phase 3: make_train_queries builds M placeholder queries for 'learned' and prepends
    M placeholders to the point queries for 'hybrid' (so the head splits them correctly)."""
    torch.manual_seed(0)
    S, Nq, ni, M = 3, 8, 2, 5
    bundle = {
        "coordinates": torch.rand(1, Nq, 2),
        "view_ids": torch.randint(0, S, (1, Nq)),
        "num_inst_queries": ni,
        "images": torch.zeros(1, S, 3, 8, 8),
    }
    common = dict(query_jitter=0.0, fixed_bg=True, train_grid_queries=False, grid_size=4)

    args_l = Namespace(query_mode="learned", num_learned_queries=M, **common)
    c, v = make_train_queries(bundle, args_l, "cpu")
    assert c.shape == (1, M, 2) and v.shape == (1, M)
    assert torch.all(c == 0) and torch.all(v == 0), "learned slots are zero placeholders"

    args_h = Namespace(query_mode="hybrid", num_learned_queries=M, **common)
    c2, v2 = make_train_queries(bundle, args_h, "cpu")
    assert c2.shape == (1, M + Nq, 2)
    assert torch.all(c2[:, :M] == 0), "learned placeholders come first"
    assert torch.equal(c2[:, M:], bundle["coordinates"]), "then the point queries"
    print("✓ query_mode train queries: learned placeholders, hybrid prepend")


def test_photometric_jitter():
    torch.manual_seed(0)
    images = torch.rand(1, 2, 3, 16, 16)
    out = photometric_jitter(images, 0.3)
    assert out.shape == images.shape
    assert out.min() >= 0 and out.max() <= 1
    assert not torch.equal(out, images)
    assert torch.equal(photometric_jitter(images, 0.0), images)
    print("✓ photometric jitter: in-range, identity at strength 0")


def test_metrics_jsonl_writer():
    """build_eval_record + append_jsonl produce one valid JSON line per eval with the
    prompted+grid train/val mIoU & AP50 fields the scaling plots consume."""
    # Two fake evals; per_scene metric dicts keyed by scene name.
    tr = {"s0": {"mIoU": 0.8, "AP50": 0.7}}
    va = {"v0": {"mIoU": 0.4, "AP50": 0.3}}
    tr_un = {"s0": {"mIoU": 0.6, "AP50": 0.5}}
    va_un = {"v0": {"mIoU": 0.2, "AP50": 0.1}}
    comps = {"total": 1.23, "class": 0.4, "mask": 0.83}

    rec = build_eval_record(50, 2e-3, comps, tr, va, tr_un, va_un)
    for k in ("epoch", "lr", "loss", "train_mIoU", "train_AP50", "train_grid_mIoU",
              "train_grid_AP50", "val_mIoU", "val_AP50", "val_grid_mIoU", "val_grid_AP50"):
        assert k in rec, f"missing field {k}"
    assert rec["epoch"] == 50 and abs(rec["val_grid_AP50"] - 0.1) < 1e-9
    assert abs(rec["train_mIoU"] - 0.8) < 1e-9 and abs(rec["loss"] - 1.23) < 1e-9

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "run" / "metrics.jsonl"  # nested dir created on demand
        append_jsonl(path, rec)
        append_jsonl(path, build_eval_record(100, 1e-3, comps, tr, va, tr_un, va_un))
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2, "one JSON line appended per eval"
        parsed = [json.loads(l) for l in lines]
        assert [p["epoch"] for p in parsed] == [50, 100]
    print("✓ metrics.jsonl: build_eval_record fields + append-per-eval semantics")


def test_moving_average():
    assert moving_average([], 3) == 0.0
    assert moving_average([1.0], 3) == 1.0
    assert abs(moving_average([1.0, 2.0, 3.0, 4.0], 2) - 3.5) < 1e-9  # last 2
    assert abs(moving_average([1.0, 2.0], 5) - 1.5) < 1e-9            # window > len
    print("✓ moving_average: window clamps to history length")


def test_early_stop_gate():
    """Noise-robust gate: disabled at patience 0, never fires in the first half of the
    schedule, fires only after `patience` flat evals past the halfway point."""
    N = 1000
    # Disabled when patience <= 0, regardless of stagnation.
    assert not early_stop_should_stop(evals_no_improve=99, patience=0, epoch=900, num_epochs=N)
    # Past patience but still in the first half → must NOT stop (the §2.1 failure mode).
    assert not early_stop_should_stop(evals_no_improve=10, patience=5, epoch=400, num_epochs=N)
    # Second half + enough flat evals → stop.
    assert early_stop_should_stop(evals_no_improve=5, patience=5, epoch=600, num_epochs=N)
    # Second half but not yet enough flat evals → keep going.
    assert not early_stop_should_stop(evals_no_improve=4, patience=5, epoch=600, num_epochs=N)
    # Exactly at the halfway boundary counts as "second half".
    assert early_stop_should_stop(evals_no_improve=5, patience=5, epoch=499, num_epochs=N)
    print("✓ early-stop gate: off at patience 0, half-schedule floor, patience honored")


def test_schedule_epochs_decoupling():
    """The cosine schedule length is set by schedule_epochs, independent of run length:
    a longer schedule decays slower at the same epoch and reaches the floor only at its end."""
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    fn100 = build_scheduler(opt, num_epochs=100, warmup_epochs=10).lr_lambdas[0]
    fn200 = build_scheduler(opt, num_epochs=200, warmup_epochs=10).lr_lambdas[0]
    assert abs(fn100(5) - fn200(5)) < 1e-9, "warmup is identical"
    assert abs(fn100(100) - 0.05) < 1e-6, "short schedule fully decayed to the min ratio"
    assert fn200(100) > 0.4, "long schedule still high at the short schedule's end"
    assert fn200(60) > fn100(60), "longer schedule decays slower at the same epoch"
    print("✓ schedule_epochs decouples cosine length from run length")


if __name__ == "__main__":
    test_no_object_loss_disabled_matches_old_behavior()
    test_no_object_loss_supervises_unmatched_queries()
    test_no_object_loss_gradients()
    test_grid_queries()
    test_make_train_queries_augmentation()
    test_train_grid_queries()
    test_query_mode_train_queries()
    test_photometric_jitter()
    test_metrics_jsonl_writer()
    test_moving_average()
    test_early_stop_gate()
    test_schedule_epochs_decoupling()
    print("\n✅ All Milestone 2 tests passed")
