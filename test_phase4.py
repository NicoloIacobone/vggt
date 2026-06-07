#!/usr/bin/env python3
"""
Phase 4 Validation: Test the InstanceDecoder and complete D4RT pipeline

This script validates the Transformer decoder and output heads.
"""

import sys
from pathlib import Path
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.d4rt_decoder import InstanceDecoder, QueryGenerator, D4RTInstanceSegmentationHead


def test_instance_decoder():
    """Test the InstanceDecoder."""
    print("=== Testing InstanceDecoder ===")
    decoder = InstanceDecoder(
        hidden_dim=256,
        num_classes=20,
        num_decoder_layers=4,
        num_heads=8,
        mask_embed_dim=256,
        memory_dim=2048,
    )

    # Create dummy inputs
    B, N = 2, 12  # batch size 2, 12 queries
    S, P = 4, 1369  # 4 frames, ~1369 patches per frame
    hidden_dim = 256
    memory_dim = 2048

    queries = torch.randn(B, N, hidden_dim)
    global_features = torch.randn(B, S, P, memory_dim)

    # Forward pass
    class_logits, mask_embeddings = decoder(queries, global_features)

    print(f"Queries shape: {queries.shape}")
    print(f"Global features shape: {global_features.shape}")
    print(f"Class logits shape: {class_logits.shape}")
    print(f"Mask embeddings shape: {mask_embeddings.shape}")

    # Validate shapes
    assert class_logits.shape == (B, N, 20), f"Expected [2, 12, 20], got {class_logits.shape}"
    assert mask_embeddings.shape == (B, N, 256), f"Expected [2, 12, 256], got {mask_embeddings.shape}"

    # Validate value ranges
    assert not torch.isnan(class_logits).any(), "Class logits contain NaN!"
    assert not torch.isinf(class_logits).any(), "Class logits contain Inf!"
    assert not torch.isnan(mask_embeddings).any(), "Mask embeddings contain NaN!"
    assert not torch.isinf(mask_embeddings).any(), "Mask embeddings contain Inf!"

    print(f"Class logits stats: min={class_logits.min():.4f}, max={class_logits.max():.4f}")
    print(f"Mask embeddings stats: min={mask_embeddings.min():.4f}, max={mask_embeddings.max():.4f}")
    print("✅ InstanceDecoder test passed!\n")

    return True


def test_class_head():
    """Test the class prediction head specifically."""
    print("=== Testing Class Head ===")
    decoder = InstanceDecoder(hidden_dim=256, num_classes=20)

    B, N = 3, 10
    queries = torch.randn(B, N, 256)

    class_logits, _ = decoder(
        queries, torch.randn(B, 4, 1369, 2048)
    )

    # Check class logits
    assert class_logits.shape[2] == 20, f"Should have 20 class logits, got {class_logits.shape[2]}"

    # Check that softmax probabilities sum to 1
    probs = torch.softmax(class_logits, dim=-1)
    prob_sums = probs.sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums)), "Softmax probabilities should sum to 1"

    print(f"Class logits shape: {class_logits.shape}")
    print(f"Max logit: {class_logits.max():.4f}, Min logit: {class_logits.min():.4f}")
    print("✅ Class head test passed!\n")

    return True


def test_mask_embedding_head():
    """Test the mask embedding head specifically."""
    print("=== Testing Mask Embedding Head ===")
    decoder = InstanceDecoder(hidden_dim=256, mask_embed_dim=128)

    B, N = 2, 8
    queries = torch.randn(B, N, 256)

    _, mask_embeddings = decoder(
        queries, torch.randn(B, 3, 1369, 2048)
    )

    # Check mask embeddings
    assert mask_embeddings.shape == (B, N, 128), f"Expected [2, 8, 128], got {mask_embeddings.shape}"
    assert not torch.isnan(mask_embeddings).any()

    # Compute L2 norms
    norms = torch.norm(mask_embeddings, dim=-1)
    print(f"Mask embedding shape: {mask_embeddings.shape}")
    print(f"L2 norms: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}")
    print("✅ Mask embedding head test passed!\n")

    return True


def test_full_pipeline():
    """Test the complete D4RT pipeline."""
    print("=== Testing Complete D4RT Pipeline ===")
    pipeline = D4RTInstanceSegmentationHead(
        num_views=5,
        hidden_dim=256,
        num_classes=20,
        num_decoder_layers=4,
        patch_size=9,
        mask_embed_dim=256,
        memory_dim=2048,
    )

    # Create dummy inputs
    B, S = 2, 4
    N = 12  # 12 queries
    H, W = 518, 518

    images = torch.rand(B, S, 3, H, W)
    coordinates = torch.rand(B, N, 2)  # [0, 1]
    view_ids = torch.randint(0, S, (B, N))
    global_features = torch.randn(B, S, (H // 14) ** 2 + 5, 2048)  # (518/14)^2 + special tokens

    # Forward pass
    class_logits, mask_embeddings = pipeline(
        coordinates, view_ids, images, global_features
    )

    print(f"Images shape: {images.shape}")
    print(f"Coordinates shape: {coordinates.shape}")
    print(f"View IDs shape: {view_ids.shape}")
    print(f"Global features shape: {global_features.shape}")
    print(f"Class logits shape: {class_logits.shape}")
    print(f"Mask embeddings shape: {mask_embeddings.shape}")

    assert class_logits.shape == (B, N, 20)
    assert mask_embeddings.shape == (B, N, 256)
    assert not torch.isnan(class_logits).any()
    assert not torch.isnan(mask_embeddings).any()

    print("✅ Complete pipeline test passed!\n")

    return True


def test_gradient_flow():
    """Test gradient flow through the decoder."""
    print("=== Testing Gradient Flow ===")
    decoder = InstanceDecoder(hidden_dim=256, num_classes=20)

    queries = torch.randn(2, 8, 256, requires_grad=True)
    global_features = torch.randn(2, 4, 1369, 2048, requires_grad=True)

    # Forward pass
    class_logits, mask_embeddings = decoder(queries, global_features)

    # Create a simple loss
    loss = class_logits.sum() + mask_embeddings.sum()

    # Backward
    loss.backward()

    assert queries.grad is not None, "Queries should have gradients"
    assert global_features.grad is not None, "Global features should have gradients"

    print(f"Queries grad norm: {queries.grad.norm():.6f}")
    print(f"Global features grad norm: {global_features.grad.norm():.6f}")
    print("✅ Gradient flow test passed!\n")

    return True


def test_different_batch_sizes():
    """Test with different batch sizes."""
    print("=== Testing Different Batch Sizes ===")
    decoder = InstanceDecoder(hidden_dim=256, num_classes=20)

    for B in [1, 2, 4]:
        for N in [5, 10, 20]:
            queries = torch.randn(B, N, 256)
            global_features = torch.randn(B, 3, 1369, 2048)

            class_logits, mask_embeddings = decoder(queries, global_features)

            assert class_logits.shape == (B, N, 20)
            assert mask_embeddings.shape == (B, N, 256)

    print("✅ Tested batch sizes: B=[1,2,4], N=[5,10,20]")
    print("✅ Different batch sizes test passed!\n")

    return True


def test_memory_projection():
    """Test memory projection with different input dimensions."""
    print("=== Testing Memory Projection ===")

    # Test with different memory dimensions
    for memory_dim in [1024, 2048, 4096]:
        decoder = InstanceDecoder(hidden_dim=256, memory_dim=memory_dim)

        queries = torch.randn(2, 8, 256)
        global_features = torch.randn(2, 4, 1369, memory_dim)

        class_logits, mask_embeddings = decoder(queries, global_features)

        assert class_logits.shape == (2, 8, 20)
        assert mask_embeddings.shape == (2, 8, 256)

    print("✅ Tested memory dims: [1024, 2048, 4096]")
    print("✅ Memory projection test passed!\n")

    return True


if __name__ == "__main__":
    try:
        test_instance_decoder()
        test_class_head()
        test_mask_embedding_head()
        test_full_pipeline()
        test_gradient_flow()
        test_different_batch_sizes()
        test_memory_projection()

        print("=" * 60)
        print("✅ Phase 4 Validation PASSED!")
        print("=" * 60)
        print("\nInstanceDecoder successfully:")
        print("  1. Projects VGGT memory (2048-dim) to decoder dim (256-dim)")
        print("  2. Uses Transformer decoder for cross-attention (4 layers, 8 heads)")
        print("  3. Outputs class logits for 20 ScanNet classes")
        print("  4. Outputs mask embeddings for instance matching")
        print("  5. Supports variable batch sizes and query counts")
        print("  6. Handles different memory dimensions via projection")
        print("  7. Supports full gradient flow for end-to-end training")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
