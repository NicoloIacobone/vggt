#!/usr/bin/env python3
"""
Phase 2 Validation: Test the ScanNetSingleSceneDataset

This script creates a minimal synthetic ScanNet scene and validates the dataset loader.
"""

import os
import sys
import tempfile
import torch
import numpy as np
from pathlib import Path
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.scannet_overfit import ScanNetSingleSceneDataset, SCANNET_CLASSES


def create_synthetic_scene(scene_dir: str, num_frames: int = 4, img_size: int = 518):
    """Create a minimal synthetic ScanNet-like scene for testing."""
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
            y_start, y_end = np.random.randint(0, img_size // 2), np.random.randint(img_size // 2, img_size)
            x_start, x_end = np.random.randint(0, img_size // 2), np.random.randint(img_size // 2, img_size)
            mask_array[y_start:y_end, x_start:x_end] = 255

            mask = Image.fromarray(mask_array)
            mask.save(class_mask_dir / f"frame_{i:05d}.png")

    print(f"Synthetic scene created at {scene_path}")
    return scene_path


def test_dataset():
    """Test the ScanNetSingleSceneDataset."""
    # Create synthetic scene
    with tempfile.TemporaryDirectory() as tmpdir:
        scene_dir = create_synthetic_scene(tmpdir, num_frames=4, img_size=256)

        # Initialize dataset
        print("\n=== Initializing ScanNetSingleSceneDataset ===")
        dataset = ScanNetSingleSceneDataset(
            scene_dir=str(scene_dir),
            num_frames=4,
            image_ext=".jpg",
            mask_ext=".png",
            img_size=256,
        )

        print(f"Dataset length: {len(dataset)} (should be 1)")
        assert len(dataset) == 1, "Dataset length should be 1"

        # Get a batch
        print("\n=== Loading batch ===")
        batch = dataset[0]

        # Validate shapes
        print("\nTensor shapes:")
        print(f"  images: {batch['images'].shape}")
        print(f"  masks: {batch['masks'].shape}")
        print(f"  classes: {batch['classes'].shape}")
        print(f"  coordinates: {batch['coordinates'].shape}")
        print(f"  frame_ids: {batch['frame_ids'].shape}")
        print(f"  instance_ids: {batch['instance_ids'].shape}")
        print(f"  num_instances: {batch['num_instances']}")

        # Verify shapes
        assert batch["images"].shape[0] == 4, "Should have 4 frames"
        assert batch["images"].shape[1] == 3, "Should have 3 RGB channels"
        assert batch["images"].shape[2] == 256, "Height should be 256"
        assert batch["images"].shape[3] == 256, "Width should be 256"

        assert batch["masks"].shape[0] == 4, "Masks should have 4 frames"
        assert batch["masks"].shape[1] == 256, "Mask height should be 256"
        assert batch["masks"].shape[2] == 256, "Mask width should be 256"

        assert batch["classes"].shape[0] == batch["num_instances"], "Classes should match num_instances"
        assert batch["coordinates"].shape[0] == batch["num_instances"], "Coordinates should match num_instances"
        assert batch["coordinates"].shape[1] == 2, "Coordinates should be (u, v) pairs"
        assert batch["frame_ids"].shape[0] == batch["num_instances"], "frame_ids should match num_instances"
        assert batch["instance_ids"].shape[0] == batch["num_instances"], "instance_ids should match num_instances"

        # --- Cross-view instance identity (item 8.3) -------------------------------------
        # The synthetic scene has 3 classes (wall/floor/chair) each present in all 4 frames,
        # so it must yield exactly 3 GLOBAL instances, each appearing in multiple frames with
        # ONE consistent ID, not 12 per-(frame, class) instances.
        print("\n=== Cross-view instance identity ===")
        masks = batch["masks"]
        nonzero_ids = torch.unique(masks[masks > 0])
        print(f"  unique instance IDs in masks: {nonzero_ids.tolist()}")
        print(f"  instance_ids returned:        {batch['instance_ids'].tolist()}")

        assert batch["num_instances"] == 3, (
            f"Expected 3 cross-view instances (one per class), got {batch['num_instances']}"
        )
        # The IDs painted into the mask map must equal the returned instance_ids (1..G).
        assert set(nonzero_ids.tolist()) == set(batch["instance_ids"].tolist()), (
            "Mask IDs must match the returned instance_ids"
        )
        # Each instance ID must appear in MORE THAN ONE frame (proves cross-view linking).
        for inst_id in batch["instance_ids"].tolist():
            frames_with_id = [f for f in range(masks.shape[0]) if (masks[f] == inst_id).any()]
            print(f"  instance {inst_id}: present in frames {frames_with_id}")
            assert len(frames_with_id) > 1, (
                f"Instance {inst_id} appears in only one frame — cross-view identity broken"
            )

        # Validate value ranges
        print("\nValue ranges:")
        print(f"  images: [{batch['images'].min():.3f}, {batch['images'].max():.3f}] (expected [0, 1])")
        print(f"  classes: [{batch['classes'].min()}, {batch['classes'].max()}] (expected [0, 19])")
        print(f"  coordinates: [({batch['coordinates'][:, 0].min():.3f}, {batch['coordinates'][:, 1].min():.3f}), "
              f"({batch['coordinates'][:, 0].max():.3f}, {batch['coordinates'][:, 1].max():.3f})] (expected [0, 1])")

        assert batch["images"].min() >= 0.0 and batch["images"].max() <= 1.0, "Images should be in [0, 1]"
        assert batch["classes"].min() >= 1 and batch["classes"].max() <= 19, "Classes should be in [1, 19]"
        assert batch["coordinates"].min() >= 0.0 and batch["coordinates"].max() <= 1.0, "Coordinates should be in [0, 1]"

        # Validate dtypes
        print("\nData types:")
        print(f"  images: {batch['images'].dtype}")
        print(f"  masks: {batch['masks'].dtype}")
        print(f"  classes: {batch['classes'].dtype}")
        print(f"  coordinates: {batch['coordinates'].dtype}")

        assert batch["images"].dtype == torch.float32, "Images should be float32"
        assert batch["masks"].dtype == torch.int32, "Masks should be int32 (instance-ID map)"
        assert batch["classes"].dtype == torch.long, "Classes should be long"
        assert batch["coordinates"].dtype == torch.float32, "Coordinates should be float32"

        print("\n✅ Phase 2 Validation PASSED!")
        print(f"   - Loaded {batch['num_instances']} instances")
        print(f"   - Classes: {batch['classes'].tolist()}")
        print(f"   - Coordinates (u, v): {batch['coordinates'].tolist()}")


if __name__ == "__main__":
    test_dataset()
