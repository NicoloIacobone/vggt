#!/usr/bin/env python3
"""
Test for legacy/d4rt/scripts/eval_grid_ablation.py::eval_bundle_at_grid_sizes (CPU, no backbone weights).

Checks: query counts per grid size, all metric keys present and finite, the prompted row,
hybrid-mode placeholder prepending, and that learned-mode heads are rejected.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from legacy.d4rt.models.d4rt_decoder import D4RTInstanceSegmentationHead
from eval_grid_ablation import eval_bundle_at_grid_sizes

METRIC_KEYS = ["mIoU", "AP50", "AP75", "mAP", "num_queries", "num_kept", "num_gt"]


def make_bundle(S=2, grid_hw=4, memory_dim=64, patch_start_idx=3, num_gt=3):
    P = patch_start_idx + grid_hw * grid_hw
    images = torch.rand(1, S, 3, 64, 64)
    features = torch.randn(1, S, P, memory_dim)
    gt = {
        "classes": torch.randint(1, 20, (num_gt,)),
        "coordinates": torch.rand(num_gt, 2),
        "masks": (torch.rand(num_gt, S, grid_hw, grid_hw) > 0.5).float(),
    }
    return images, features, patch_start_idx, gt


def test_point_mode():
    print("=== point mode: grid sweep + prompted row ===")
    torch.manual_seed(0)
    images, features, psi, gt = make_bundle()
    head = D4RTInstanceSegmentationHead(num_views=4, memory_dim=features.shape[-1],
                                        num_decoder_layers=1, dropout=0.0)
    S = images.shape[1]
    N_prompt = 5
    prompted = (torch.rand(1, N_prompt, 2), torch.randint(0, S, (1, N_prompt)))

    results = eval_bundle_at_grid_sizes(head, images, features, psi, gt,
                                        grid_sizes=[2, 3], device="cpu",
                                        prompted_queries=prompted)

    assert set(results) == {"prompted", "grid_2", "grid_3"}, results.keys()
    for label, m in results.items():
        for k in METRIC_KEYS:
            assert k in m, f"{label} missing {k}"
            assert torch.isfinite(torch.tensor(float(m[k]))), f"{label}[{k}] not finite"
        assert m["num_gt"] == gt["classes"].shape[0]
        assert 0 <= m["num_kept"] <= m["num_queries"]
    assert results["prompted"]["num_queries"] == N_prompt
    assert results["grid_2"]["num_queries"] == S * 2 * 2
    assert results["grid_3"]["num_queries"] == S * 3 * 3
    print("✓ point mode OK:", {k: round(results["grid_3"][k], 3) for k in ("mIoU", "AP50")})


def test_hybrid_mode():
    print("=== hybrid mode: learned placeholders prepended ===")
    torch.manual_seed(0)
    images, features, psi, gt = make_bundle()
    M = 4
    head = D4RTInstanceSegmentationHead(num_views=4, memory_dim=features.shape[-1],
                                        num_decoder_layers=1, dropout=0.0,
                                        query_mode="hybrid", num_learned_queries=M)
    S = images.shape[1]
    results = eval_bundle_at_grid_sizes(head, images, features, psi, gt,
                                        grid_sizes=[2], device="cpu")
    assert results["grid_2"]["num_queries"] == M + S * 2 * 2, results["grid_2"]["num_queries"]
    print("✓ hybrid mode OK (query count includes the learned slots)")


def test_learned_mode_rejected():
    print("=== learned mode: must be rejected ===")
    images, features, psi, gt = make_bundle()
    head = D4RTInstanceSegmentationHead(num_views=4, memory_dim=features.shape[-1],
                                        num_decoder_layers=1, dropout=0.0,
                                        query_mode="learned", num_learned_queries=4)
    try:
        eval_bundle_at_grid_sizes(head, images, features, psi, gt, [2], "cpu")
    except ValueError:
        print("✓ learned mode raises ValueError")
    else:
        raise AssertionError("learned-mode head was not rejected")


if __name__ == "__main__":
    test_point_mode()
    test_hybrid_mode()
    test_learned_mode_rejected()
    print("\nAll eval_grid_ablation tests passed!")
