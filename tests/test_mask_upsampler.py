#!/usr/bin/env python3
"""
Phase 5 validation: the MaskDINO-style pixel decoder (models/mask_upsampler.py) and its
integration into InstanceDecoder + build_gt_targets. Standalone (CPU, no backbone weights).

Checks:
  - MaskUpsampler output shapes for upsample ∈ {1, 2, 4}, gradient flow, and power-of-two guard.
  - InstanceDecoder predicts masks at the upsampled resolution (and is unchanged at 1).
  - build_gt_targets emits GT masks at the matching upsampled resolution.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from models.mask_upsampler import MaskUpsampler
from models.d4rt_decoder import InstanceDecoder


def test_mask_upsampler_shapes():
    print("=== Testing MaskUpsampler shapes ===")
    B, S, h, w, mem, emb = 2, 3, 5, 5, 64, 16
    feats = torch.randn(B, S, h, w, mem)
    for f in (1, 2, 4):
        out = MaskUpsampler(memory_dim=mem, mask_embed_dim=emb, upsample=f)(feats)
        assert out.shape == (B, S, h * f, w * f, emb), (f, out.shape)
    print("✅ MaskUpsampler shapes for upsample ∈ {1,2,4} passed!\n")


def test_mask_upsampler_grad_and_guard():
    print("=== Testing MaskUpsampler gradients + power-of-two guard ===")
    feats = torch.randn(1, 2, 4, 4, 32, requires_grad=True)
    up = MaskUpsampler(memory_dim=32, mask_embed_dim=8, upsample=2)
    out = up(feats)
    out.sum().backward()
    assert feats.grad is not None and torch.isfinite(feats.grad).all()
    for bad in (0, 3, 6):
        try:
            MaskUpsampler(upsample=bad)
            assert False, f"expected ValueError for upsample={bad}"
        except ValueError:
            pass
    print("✅ MaskUpsampler gradient + guard test passed!\n")


def test_decoder_integration():
    print("=== Testing InstanceDecoder with mask_upsample ===")
    B, S, N, h, w, mem, emb = 1, 2, 4, 3, 3, 64, 16
    patch_start = 1
    P = patch_start + h * w
    queries = torch.randn(B, N, 32)
    feats = torch.randn(B, S, P, mem)

    dec1 = InstanceDecoder(hidden_dim=32, num_decoder_layers=1, mask_embed_dim=emb,
                           memory_dim=mem, mask_upsample=1)
    _, _, pm1 = dec1(queries, feats, None, patch_start)
    assert pm1.shape == (B, N, S, h, w), pm1.shape
    assert dec1.mask_upsampler is None

    dec2 = InstanceDecoder(hidden_dim=32, num_decoder_layers=1, mask_embed_dim=emb,
                           memory_dim=mem, mask_upsample=2)
    _, _, pm2 = dec2(queries, feats, None, patch_start)
    assert pm2.shape == (B, N, S, h * 2, w * 2), pm2.shape
    print("✅ InstanceDecoder mask_upsample integration passed!\n")


def test_build_gt_targets_resolution():
    print("=== Testing build_gt_targets at upsampled resolution ===")
    from train_overfit import build_gt_targets

    S, H, W = 2, 12, 12
    masks = torch.zeros(S, H, W, dtype=torch.long)
    masks[:, 2:6, 2:6] = 1  # one instance visible in both frames
    batch = {
        "classes": torch.tensor([5]),
        "coordinates": torch.tensor([[0.4, 0.4]]),
        "masks": masks,
    }
    num_patch_tokens = 4  # h = w = 2
    gt1 = build_gt_targets(batch, 1, num_patch_tokens, "cpu", mask_upsample=1)
    assert gt1["masks"].shape == (1, S, 2, 2), gt1["masks"].shape
    gt2 = build_gt_targets(batch, 1, num_patch_tokens, "cpu", mask_upsample=2)
    assert gt2["masks"].shape == (1, S, 4, 4), gt2["masks"].shape
    # The instance occupies pixels in both → mask must be non-empty at both resolutions.
    assert gt1["masks"].sum() > 0 and gt2["masks"].sum() > 0
    print("✅ build_gt_targets resolution test passed!\n")


if __name__ == "__main__":
    test_mask_upsampler_shapes()
    test_mask_upsampler_grad_and_guard()
    test_decoder_integration()
    test_build_gt_targets_resolution()
    print("All mask_upsampler (Phase 5) tests passed! ✅")
