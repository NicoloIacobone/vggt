#!/usr/bin/env python3
"""
Phase 3 Validation: Test the QueryGenerator

This script validates that the QueryGenerator correctly produces query embeddings
from coordinates, view IDs, and RGB images.
"""

import sys
from pathlib import Path
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.d4rt_decoder import QueryGenerator, FourierPositionalEncoding, LocalPatchFeatureExtractor


def test_fourier_encoding():
    """Test Fourier positional encoding."""
    print("=== Testing Fourier Positional Encoding ===")
    encoder = FourierPositionalEncoding(num_freqs=16, max_freq=10.0)

    # Test with batch of coordinates
    B, N = 2, 5
    coords = torch.rand(B, N, 2)  # Random coords in [0, 1]

    encoding = encoder(coords)
    print(f"Input shape: {coords.shape}")
    print(f"Output shape: {encoding.shape}")
    print(f"Expected: [B={B}, N={N}, 2*num_freqs=64]")

    assert encoding.shape == (B, N, 64), f"Expected [2, 5, 64], got {encoding.shape}"
    assert encoding.dtype == torch.float32

    # Check that encoding is bounded (sin/cos output)
    assert encoding.min() >= -1.1 and encoding.max() <= 1.1, "Encoding should be bounded by sin/cos"
    print("✅ Fourier encoding test passed!\n")

    return True


def test_patch_extractor():
    """Test local patch feature extraction."""
    print("=== Testing Local Patch Feature Extractor ===")
    extractor = LocalPatchFeatureExtractor(patch_size=9, hidden_dim=256, in_channels=3)

    # Create dummy images
    B, S, C, H, W = 2, 3, 3, 518, 518
    images = torch.rand(B, S, C, H, W)

    # Random coordinates
    N = 8
    coords = torch.rand(B, N, 2)  # Normalized coords [0, 1]
    view_ids = torch.randint(0, S, (B, N))

    features = extractor(images, coords, view_ids)
    print(f"Images shape: {images.shape}")
    print(f"Coordinates shape: {coords.shape}")
    print(f"View IDs shape: {view_ids.shape}")
    print(f"Output feature shape: {features.shape}")
    print(f"Expected: [B={B}, N={N}, hidden_dim=256]")

    assert features.shape == (B, N, 256), f"Expected [2, 8, 256], got {features.shape}"
    assert features.dtype == torch.float32
    print("✅ Patch extractor test passed!\n")

    return True


def test_query_generator():
    """Test the complete QueryGenerator."""
    print("=== Testing QueryGenerator ===")
    query_gen = QueryGenerator(
        num_views=10,
        hidden_dim=256,
        patch_size=9,
        num_freqs=16,
        max_freq=10.0,
    )

    # Create dummy inputs
    B, S, C, H, W = 2, 4, 3, 518, 518
    images = torch.rand(B, S, C, H, W)

    N = 12  # Number of queries
    coordinates = torch.rand(B, N, 2)  # Normalized coords [0, 1]
    view_ids = torch.randint(0, S, (B, N))

    # Forward pass
    queries = query_gen(coordinates, view_ids, images)

    print(f"Images shape: {images.shape}")
    print(f"Coordinates shape: {coordinates.shape}")
    print(f"View IDs shape: {view_ids.shape}")
    print(f"Queries output shape: {queries.shape}")
    print(f"Expected: [B={B}, N={N}, hidden_dim=256]")

    assert queries.shape == (B, N, 256), f"Expected [2, 12, 256], got {queries.shape}"
    assert queries.dtype == torch.float32
    assert not torch.isnan(queries).any(), "Queries contain NaN values!"
    assert not torch.isinf(queries).any(), "Queries contain Inf values!"

    print(f"Query stats: min={queries.min():.4f}, max={queries.max():.4f}, mean={queries.mean():.4f}, std={queries.std():.4f}")
    print("✅ QueryGenerator test passed!\n")

    return True


def test_gradients():
    """Test that gradients flow correctly through the QueryGenerator."""
    print("=== Testing Gradient Flow ===")
    query_gen = QueryGenerator(num_views=10, hidden_dim=256, patch_size=9)

    # Create dummy inputs
    images = torch.rand(2, 3, 3, 256, 256, requires_grad=True)
    coordinates = torch.rand(2, 8, 2, requires_grad=True)
    view_ids = torch.randint(0, 3, (2, 8))

    # Forward pass
    queries = query_gen(coordinates, view_ids, images)

    # Create a simple loss
    loss = queries.sum()

    # Backward pass
    loss.backward()

    print(f"Images grad exists: {images.grad is not None}")
    print(f"Coordinates grad exists: {coordinates.grad is not None}")
    assert images.grad is not None, "Images should have gradients"
    assert coordinates.grad is not None, "Coordinates should have gradients"
    print(f"Images grad norm: {images.grad.norm():.6f}")
    print(f"Coordinates grad norm: {coordinates.grad.norm():.6f}")
    print("✅ Gradient flow test passed!\n")

    return True


def test_edge_cases():
    """Test edge cases."""
    print("=== Testing Edge Cases ===")
    query_gen = QueryGenerator(num_views=5, hidden_dim=128, patch_size=5)

    # Test with coordinates at boundaries
    B, N = 1, 4
    images = torch.rand(B, 2, 3, 256, 256)

    # Coords at corners and center
    coordinates = torch.tensor(
        [[[0.0, 0.0], [1.0, 1.0], [0.5, 0.5], [0.0, 1.0]]], dtype=torch.float32
    )
    view_ids = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)

    queries = query_gen(coordinates, view_ids, images)
    assert queries.shape == (B, N, 128)
    assert not torch.isnan(queries).any()
    print("✅ Edge case test passed!\n")

    return True


def test_query_modes():
    """Phase 3: point/learned/hybrid query modes produce the right shapes, and learned
    queries are independent of the (placeholder) coordinates."""
    print("=== Testing query modes (point/learned/hybrid) ===")
    B, S, N, M, hidden = 2, 3, 5, 7, 256
    coords = torch.rand(B, N, 2)
    view_ids = torch.randint(0, S, (B, N))
    images = torch.rand(B, S, 3, 28, 28)

    out_p = QueryGenerator(num_views=S, query_mode="point")(coords, view_ids, images)
    assert out_p.shape == (B, N, hidden), out_p.shape

    qg_l = QueryGenerator(num_views=S, query_mode="learned", num_learned_queries=M)
    ph_c = torch.zeros(B, M, 2)
    ph_v = torch.zeros(B, M, dtype=torch.long)
    out_l = qg_l(ph_c, ph_v, images)
    assert out_l.shape == (B, M, hidden), out_l.shape
    # Learned queries ignore coordinate values entirely.
    assert torch.equal(out_l, qg_l(torch.rand(B, M, 2), ph_v, images))

    qg_h = QueryGenerator(num_views=S, query_mode="hybrid", num_learned_queries=M)
    cat_c = torch.cat([torch.zeros(B, M, 2), coords], dim=1)
    cat_v = torch.cat([torch.zeros(B, M, dtype=torch.long), view_ids], dim=1)
    out_h = qg_h(cat_c, cat_v, images)
    assert out_h.shape == (B, M + N, hidden), out_h.shape
    # First M slots are exactly the learned object queries.
    expect = qg_h.learned_queries.weight.unsqueeze(0).expand(B, -1, -1)
    assert torch.allclose(out_h[:, :M], expect)

    for bad in (dict(query_mode="bogus"),
                dict(query_mode="learned", num_learned_queries=0)):
        try:
            QueryGenerator(**bad)
            assert False, f"expected ValueError for {bad}"
        except ValueError:
            pass
    print("✅ query modes test passed!\n")


def test_head_config_roundtrip():
    """Phase 3: a head's state_dict reloads into a head rebuilt from the SAME head_config,
    for every query mode (the checkpoint→demo round-trip the demo relies on)."""
    print("=== Testing head_config round-trip across query modes ===")
    from models.d4rt_decoder import D4RTInstanceSegmentationHead

    for mode, M in [("point", 0), ("learned", 6), ("hybrid", 6)]:
        cfg = dict(num_views=4, hidden_dim=32, num_classes=20, num_decoder_layers=1,
                   patch_size=5, mask_embed_dim=16, memory_dim=64, dropout=0.0,
                   query_mode=mode, num_learned_queries=M)
        head = D4RTInstanceSegmentationHead(**cfg)
        rebuilt = D4RTInstanceSegmentationHead(**cfg)
        missing = rebuilt.load_state_dict(head.state_dict())  # must not raise
        assert not missing.missing_keys and not missing.unexpected_keys, missing
        assert rebuilt.query_mode == mode and rebuilt.num_learned_queries == M
    print("✅ head_config round-trip test passed!\n")


if __name__ == "__main__":
    try:
        test_fourier_encoding()
        test_patch_extractor()
        test_query_generator()
        test_gradients()
        test_edge_cases()
        test_query_modes()
        test_head_config_roundtrip()

        print("=" * 50)
        print("✅ Phase 3 Validation PASSED!")
        print("=" * 50)
        print("\nQuery Generator successfully:")
        print("  1. Encodes (u, v) coordinates with Fourier features")
        print("  2. Embeds view IDs as learned embeddings")
        print("  3. Extracts local RGB patches via grid_sample")
        print("  4. Combines all features into query embeddings")
        print("  5. Supports gradient flow for end-to-end training")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
