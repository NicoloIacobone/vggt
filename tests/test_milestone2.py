#!/usr/bin/env python3
"""
Milestone 2 validation: no-object loss, unprompted grid queries, and the per-step query
augmentation used by scripts/train_multiscene.py. Standalone (CPU, no backbone weights).
"""

import sys
from argparse import Namespace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from train.loss import D4RTLoss
from train_overfit import generate_grid_queries
from train_multiscene import make_train_queries, photometric_jitter


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


def test_photometric_jitter():
    torch.manual_seed(0)
    images = torch.rand(1, 2, 3, 16, 16)
    out = photometric_jitter(images, 0.3)
    assert out.shape == images.shape
    assert out.min() >= 0 and out.max() <= 1
    assert not torch.equal(out, images)
    assert torch.equal(photometric_jitter(images, 0.0), images)
    print("✓ photometric jitter: in-range, identity at strength 0")


if __name__ == "__main__":
    test_no_object_loss_disabled_matches_old_behavior()
    test_no_object_loss_supervises_unmatched_queries()
    test_no_object_loss_gradients()
    test_grid_queries()
    test_make_train_queries_augmentation()
    test_photometric_jitter()
    print("\n✅ All Milestone 2 tests passed")
