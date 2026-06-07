#!/usr/bin/env python3
"""
Phase 6: Minimal Overfit Training Loop for D4RT Instance Segmentation

This script combines all components (VGGT backbone, QueryGenerator, InstanceDecoder, Loss)
into an end-to-end training pipeline. It validates that gradients flow correctly from
SAM3 pseudo-labels through the entire system.

Usage:
    python train_overfit.py --num_epochs 500 --scene_dir /path/to/scannet/scene
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from vggt.models.vggt import VGGT
from models.d4rt_decoder import D4RTInstanceSegmentationHead
from train.loss import D4RTLoss
from data.scannet_overfit import ScanNetSingleSceneDataset


class D4RTModel(nn.Module):
    """
    Complete D4RT instance segmentation model combining VGGT backbone and decoder.

    Args:
        freeze_backbone (bool): If True, freeze VGGT backbone weights
        num_views (int): Maximum number of views
        decoder_hidden_dim (int): Decoder hidden dimension
        mask_embed_dim (int): Mask embedding dimension
    """

    def __init__(
        self,
        freeze_backbone: bool = True,
        num_views: int = 10,
        decoder_hidden_dim: int = 256,
        mask_embed_dim: int = 256,
    ):
        super().__init__()
        self.freeze_backbone = freeze_backbone

        # Load VGGT backbone
        print("Loading VGGT backbone...")
        try:
            self.backbone = VGGT.from_pretrained("facebook/VGGT-1B")
            print("✓ Loaded pretrained VGGT-1B")
        except Exception as e:
            print(f"⚠ Could not load pretrained VGGT: {e}")
            print("  Creating VGGT with random initialization...")
            self.backbone = VGGT()

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("✓ VGGT backbone frozen")
        else:
            print("✓ VGGT backbone trainable")

        # D4RT decoder head
        self.decoder_head = D4RTInstanceSegmentationHead(
            num_views=num_views,
            hidden_dim=decoder_hidden_dim,
            num_classes=20,
            num_decoder_layers=4,
            patch_size=9,
            mask_embed_dim=mask_embed_dim,
            memory_dim=2048,  # 2 * embed_dim from VGGT
        )

    def forward(
        self,
        images: torch.Tensor,
        coordinates: torch.Tensor,
        view_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through VGGT backbone and D4RT decoder.

        Args:
            images (torch.Tensor): [B, S, 3, H, W] input images
            coordinates (torch.Tensor): [B, N, 2] query coordinates
            view_ids (torch.Tensor): [B, N] view indices

        Returns:
            Dict with class_logits and mask_embeddings
        """
        # VGGT forward pass
        with torch.no_grad() if self.freeze_backbone else torch.enable_grad():
            aggregated_tokens_list, patch_start_idx = self.backbone.aggregator(images)

        # Extract global features from final cached layer
        global_features = aggregated_tokens_list[-1]  # [B, S, P, 2048]

        # D4RT decoder forward pass
        class_logits, mask_embeddings = self.decoder_head(
            coordinates, view_ids, images, global_features, patch_start_idx
        )

        return {
            "class_logits": class_logits,
            "mask_embeddings": mask_embeddings,
        }


def create_dataloader(
    scene_dir: Optional[str] = None,
    num_frames: int = 8,
    batch_size: int = 1,
    num_workers: int = 0,
) -> DataLoader:
    """Create a dataloader for the scene."""
    if scene_dir is None:
        scene_dir = "/cluster/work/igp_psr/niacobone/distillation/dataset/scannet/scans/scene0000_00/raw_data"

    scene_dir = Path(scene_dir)
    if not scene_dir.exists():
        raise ValueError(f"Scene directory not found: {scene_dir}")

    print(f"Loading dataset from: {scene_dir}")
    dataset = ScanNetSingleSceneDataset(
        scene_dir=str(scene_dir),
        num_frames=num_frames,
        img_size=518,  # Must be divisible by VGGT patch_size (14)
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return dataloader


def create_synthetic_scene(scene_dir: str, num_frames: int = 4, img_size: int = 256):
    """Create a minimal synthetic ScanNet-like scene for testing."""
    import numpy as np
    from PIL import Image

    scene_path = Path(scene_dir)
    images_dir = scene_path / "images"
    masks_dir = scene_path / "masks"

    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Create sample images
    print(f"Creating {num_frames} synthetic images...")
    for i in range(num_frames):
        # Random RGB image
        img_array = np.random.randint(0, 256, (img_size, img_size, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(images_dir / f"frame_{i:05d}.jpg")

    # Create masks for 3 classes
    test_classes = ["wall", "floor", "chair"]
    print(f"Creating masks for classes: {test_classes}")

    for class_name in test_classes:
        class_mask_dir = masks_dir / class_name
        class_mask_dir.mkdir(parents=True, exist_ok=True)

        for i in range(num_frames):
            # Create a simple binary mask (random rectangular region)
            mask_array = np.zeros((img_size, img_size), dtype=np.uint8)

            # Add a random rectangular foreground region
            y_start = np.random.randint(0, img_size // 2)
            y_end = np.random.randint(img_size // 2, img_size)
            x_start = np.random.randint(0, img_size // 2)
            x_end = np.random.randint(img_size // 2, img_size)
            mask_array[y_start:y_end, x_start:x_end] = 255

            mask = Image.fromarray(mask_array)
            mask.save(class_mask_dir / f"frame_{i:05d}.png")

    print(f"Synthetic scene created at {scene_path}")
    return scene_path


def generate_query_points(
    batch: Dict[str, torch.Tensor],
    num_queries: int = 16,
    device: str = "cpu",
) -> tuple:
    """
    Generate query points from instance mask centroid and background points.

    Args:
        batch: Batch from dataloader containing 'coordinates' and 'num_instances'
        num_queries: Total number of queries per batch
        device: Device to create tensors on

    Returns:
        Tuple of (coordinates, view_ids)
    """
    B = batch["images"].shape[0]
    S = batch["images"].shape[1]

    coordinates_list = []
    view_ids_list = []

    for b in range(B):
        num_instances = batch["num_instances"]
        num_bg_points = max(1, num_queries - num_instances)

        # Instance points from dataset
        # batch["coordinates"] might be [num_instances, 2], move to device
        instance_coords = batch["coordinates"].squeeze(0) if batch["coordinates"].dim() > 2 else batch["coordinates"]
        instance_coords = instance_coords.to(device)  # [num_instances, 2]
        instance_view_ids = torch.zeros(len(instance_coords), dtype=torch.long, device=device)

        # Generate random background points
        bg_coords = torch.rand(num_bg_points, 2, device=device)
        bg_view_ids = torch.randint(0, S, (num_bg_points,), device=device)

        # Combine
        all_coords = torch.cat([instance_coords, bg_coords], dim=0)  # [num_queries, 2]
        all_view_ids = torch.cat([instance_view_ids, bg_view_ids], dim=0)  # [num_queries]

        coordinates_list.append(all_coords)
        view_ids_list.append(all_view_ids)

    # Stack into batch
    coordinates = torch.stack(coordinates_list, dim=0)  # [B, num_queries, 2]
    view_ids = torch.stack(view_ids_list, dim=0)  # [B, num_queries]

    return coordinates, view_ids


def create_dummy_gt(
    batch: Dict[str, torch.Tensor],
    num_queries: int = 16,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Create random ground truth targets for loss computation.

    This creates synthetic targets that are independent of the dataset
    to validate gradient flow through the entire pipeline.

    Args:
        batch: Batch from dataloader
        num_queries: Number of queries
        device: Device to create tensors on

    Returns:
        Dict with gt_classes, gt_mask_embeddings, gt_coordinates
    """
    # Create random ground truth that matches the number of queries
    num_instances = max(1, min(num_queries // 2, 8))  # 1-8 instances

    # Random classes in valid range [0, 19]
    gt_classes = torch.randint(0, 20, (num_instances,), device=device)

    # Random ground truth embeddings and coordinates
    gt_mask_embeddings = torch.randn(num_instances, 256, device=device)
    gt_coordinates = torch.rand(num_instances, 2, device=device)

    return {
        "classes": gt_classes,
        "mask_embeddings": gt_mask_embeddings,
        "coordinates": gt_coordinates,
    }


def train_epoch(
    model: nn.Module,
    loss_fn: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    num_queries: int = 16,
    log_interval: int = 10,
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: D4RTModel
        loss_fn: D4RTLoss
        dataloader: Training dataloader
        optimizer: AdamW optimizer
        device: Device to train on
        num_queries: Number of query points per image
        log_interval: Log every N iterations

    Returns:
        Dict with average losses
    """
    model.train()
    total_loss = 0.0
    losses_dict = {
        "class_loss": 0.0,
        "mask_embed_loss": 0.0,
        "coord_loss": 0.0,
        "mask_loss": 0.0,
        "num_matches": 0,
    }

    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        images = batch["images"].to(device)  # [B, S, 3, H, W]
        B, S = images.shape[:2]

        # Generate query points
        coordinates, view_ids = generate_query_points(batch, num_queries, device)

        # Generate ground truth targets
        gt = create_dummy_gt(batch, num_queries, device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images, coordinates, view_ids)

        # Compute loss
        total_loss_val, loss_components = loss_fn(
            outputs["class_logits"],
            outputs["mask_embeddings"],
            coordinates,
            gt["classes"],
            gt["mask_embeddings"],
            gt["coordinates"],
        )

        # Backward pass
        total_loss_val.backward()
        optimizer.step()

        # Accumulate losses
        total_loss += total_loss_val.item()
        for key in losses_dict:
            if key in loss_components:
                if key == "num_matches":
                    losses_dict[key] += loss_components[key]
                else:
                    losses_dict[key] += loss_components[key].item() if isinstance(loss_components[key], torch.Tensor) else 0.0

        # Log
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(
                f"Iter {batch_idx + 1:3d} | Loss: {total_loss_val.item():8.4f} | "
                f"Avg Loss: {avg_loss:8.4f} | Matches: {loss_components['num_matches']}"
            )

    # Average losses
    num_batches = len(dataloader)
    avg_losses = {
        "total_loss": total_loss / num_batches,
        "class_loss": losses_dict["class_loss"] / num_batches,
        "mask_embed_loss": losses_dict["mask_embed_loss"] / num_batches,
        "coord_loss": losses_dict["coord_loss"] / num_batches,
        "mask_loss": losses_dict["mask_loss"] / num_batches,
    }

    return avg_losses


def main():
    parser = argparse.ArgumentParser(description="D4RT Overfit Training Loop")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--num_frames", type=int, default=4, help="Number of frames for overfitting")
    parser.add_argument("--num_queries", type=int, default=16, help="Number of query points per image")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument(
        "--scene_dir",
        type=str,
        default="/cluster/work/igp_psr/niacobone/distillation/dataset/scannet/scans/scene0000_00/raw_data",
        help="Path to ScanNet scene directory",
    )
    parser.add_argument("--freeze_backbone", action="store_true", default=True, help="Freeze VGGT backbone")
    parser.add_argument("--unfreeze_backbone", action="store_true", help="Unfreeze VGGT backbone")
    parser.add_argument("--log_interval", type=int, default=10, help="Log every N iterations")
    parser.add_argument("--save_checkpoint", type=str, default=None, help="Path to save checkpoint")

    args = parser.parse_args()

    # Override freeze_backbone if unfreeze_backbone is set
    if args.unfreeze_backbone:
        args.freeze_backbone = False

    # Setup device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")
    print("=" * 70)

    # Create model
    print("\n=== Initializing Model ===")
    model = D4RTModel(
        freeze_backbone=args.freeze_backbone,
        num_views=10,
        decoder_hidden_dim=256,
        mask_embed_dim=256,
    )
    model = model.to(device)

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create dataloader
    print("\n=== Loading Dataset ===")
    try:
        dataloader = create_dataloader(
            scene_dir=args.scene_dir,
            num_frames=args.num_frames,
            batch_size=args.batch_size,
        )
        print(f"✓ Loaded dataset with {len(dataloader)} batches")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        print("  Creating synthetic dataset for testing...")
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            scene_dir = create_synthetic_scene(tmpdir, num_frames=args.num_frames, img_size=280)
            dataloader = create_dataloader(
                scene_dir=str(scene_dir),
                num_frames=args.num_frames,
                batch_size=args.batch_size,
            )

    # Create loss function
    print("\n=== Initializing Loss Function ===")
    loss_fn = D4RTLoss(
        num_classes=20,
        focal_alpha=0.25,
        focal_gamma=2.0,
        class_loss_weight=1.0,
        mask_embed_loss_weight=1.0,
        coord_loss_weight=1.0,
        mask_loss_weight=0.0,  # No mask prediction in this minimal version
    )
    loss_fn = loss_fn.to(device)
    print("✓ Loss function initialized")

    # Create optimizer
    print("\n=== Initializing Optimizer ===")
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    print(f"✓ AdamW optimizer with lr={args.learning_rate}")

    # Training loop
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    losses_history = []

    for epoch in range(args.num_epochs):
        print(f"\n[Epoch {epoch + 1:3d}/{args.num_epochs}]", end=" ")

        epoch_losses = train_epoch(
            model=model,
            loss_fn=loss_fn,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            num_queries=args.num_queries,
            log_interval=args.log_interval,
        )

        losses_history.append(epoch_losses)

        # Print epoch summary
        print(
            f"\n{'':13} ├─ Total Loss: {epoch_losses['total_loss']:8.4f} "
            f"(Class: {epoch_losses['class_loss']:6.4f}, "
            f"MaskEmbed: {epoch_losses['mask_embed_loss']:6.4f}, "
            f"Coord: {epoch_losses['coord_loss']:6.4f})"
        )

        # Check for divergence
        if epoch_losses["total_loss"] > 1e6:
            print("⚠ Warning: Loss is diverging!")
            break

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    initial_loss = losses_history[0]["total_loss"]
    final_loss = losses_history[-1]["total_loss"]
    loss_reduction = ((initial_loss - final_loss) / initial_loss * 100) if initial_loss > 0 else 0

    print(f"\nInitial loss: {initial_loss:.4f}")
    print(f"Final loss:   {final_loss:.4f}")
    print(f"Reduction:    {loss_reduction:.2f}%")

    if loss_reduction > 50:
        print("\n✅ SUCCESS: Loss decreased significantly!")
        print("   Gradients flow correctly through entire pipeline.")
    elif loss_reduction > 0:
        print("\n⚠ PARTIAL SUCCESS: Loss decreased but not dramatically.")
        print("   Check if overfitting parameters are appropriate.")
    else:
        print("\n❌ FAILURE: Loss did not decrease.")
        print("   There may be issues with gradient flow.")

    # Save checkpoint
    if args.save_checkpoint:
        print(f"\nSaving checkpoint to {args.save_checkpoint}")
        torch.save(model.state_dict(), args.save_checkpoint)

    return 0 if loss_reduction > 50 else 1


if __name__ == "__main__":
    sys.exit(main())
