# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ScanNet class labels (19 classes + background)
SCANNET_CLASSES = [
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
    "window", "bookshelf", "picture", "counter", "desk", "curtain",
    "refrigerator", "shower curtain", "toilet", "sink", "bathtub", "otherfurniture"
]

CLASS_TO_IDX = {cls_name: idx + 1 for idx, cls_name in enumerate(SCANNET_CLASSES)}
IDX_TO_CLASS = {idx + 1: cls_name for idx, cls_name in enumerate(SCANNET_CLASSES)}
IDX_TO_CLASS[0] = "background"


class ScanNetSingleSceneDataset(Dataset):
    """
    Minimal ScanNet single-scene dataset for overfitting.

    Loads RGB images and corresponding per-class binary masks from a ScanNet scene folder.
    Masks are stored as uint8 PNGs (0 for background, 255 for foreground) in class-specific folders.

    Args:
        scene_dir (str): Path to scene folder containing 'images' and 'masks' subfolders
        num_frames (int): Number of frames to load (randomly sampled from available frames)
        image_ext (str): Image extension (default: '.jpg')
        mask_ext (str): Mask extension (default: '.png')
        img_size (int): Target image size for resizing (default: 518)

    Returns dict with:
        - images: torch.Tensor [num_frames, 3, img_size, img_size] in range [0, 1]
        - masks: torch.Tensor [num_frames, img_size, img_size] instance ID per pixel (0 = background)
        - classes: torch.Tensor [num_instances] class labels (1-19 for ScanNet, 0 for background)
        - coordinates: torch.Tensor [num_instances, 2] (u, v) centroid of each instance
    """

    def __init__(
        self,
        scene_dir: str,
        num_frames: int = 8,
        image_ext: str = ".jpg",
        mask_ext: str = ".png",
        img_size: int = 518,
    ):
        super().__init__()
        self.scene_dir = Path(scene_dir)
        self.num_frames = num_frames
        self.image_ext = image_ext
        self.mask_ext = mask_ext
        self.img_size = img_size

        # Try both 'images' and 'color' folder names (different ScanNet versions)
        self.images_dir = self.scene_dir / "images"
        if not self.images_dir.exists():
            self.images_dir = self.scene_dir / "color"
        if not self.images_dir.exists():
            raise ValueError(f"Images directory not found (tried 'images' and 'color'): {self.scene_dir}")

        self.masks_dir = self.scene_dir / "masks"
        if not self.masks_dir.exists():
            raise ValueError(f"Masks directory not found: {self.masks_dir}")

        # Find all image files
        self.image_files = sorted([
            f for f in self.images_dir.iterdir()
            if f.suffix.lower() == image_ext.lower()
        ])

        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.images_dir}")

        # Find all class folders in masks directory
        self.class_dirs = {
            cls_name: self.masks_dir / cls_name
            for cls_name in SCANNET_CLASSES
            if (self.masks_dir / cls_name).exists()
        }

        if not self.class_dirs:
            raise ValueError(f"No class folders found in {self.masks_dir}")

    def __len__(self):
        return 1  # Single scene dataset - always returns 1 sample

    def __getitem__(self, idx):
        # Sample num_frames random frames
        sampled_indices = random.sample(range(len(self.image_files)),
                                       min(self.num_frames, len(self.image_files)))
        sampled_indices.sort()

        sampled_images = [self.image_files[i] for i in sampled_indices]
        frame_names = [f.stem for f in sampled_images]

        # Load images
        images = []
        for img_path in sampled_images:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
            img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # [3, H, W]
            images.append(img_tensor)

        images = torch.stack(images, dim=0)  # [num_frames, 3, H, W]

        # Load masks and build instance segmentation
        instance_masks = []
        instance_classes = []
        instance_coords = []

        instance_id = 1  # Start from 1 (0 is background)

        for frame_idx, frame_name in enumerate(frame_names):
            frame_mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)

            # Load masks for each class
            for class_idx, (class_name, class_dir) in enumerate(self.class_dirs.items()):
                mask_path = class_dir / f"{frame_name}{self.mask_ext}"

                if mask_path.exists():
                    class_mask = Image.open(mask_path).convert("L")
                    class_mask = class_mask.resize((self.img_size, self.img_size), Image.NEAREST)
                    class_mask_array = np.array(class_mask, dtype=np.uint8)

                    # Find connected components (instances) in this class mask
                    # For simplicity, treat the entire mask as one instance per class
                    if class_mask_array.max() > 0:  # If there's foreground
                        # Find pixels belonging to this class
                        class_pixels = class_mask_array > 127  # Threshold at 127

                        # Assign instance ID
                        frame_mask[class_pixels] = instance_id
                        instance_classes.append(CLASS_TO_IDX[class_name])
                        instance_coords.append(self._get_centroid(class_pixels))
                        instance_id += 1

            instance_masks.append(torch.from_numpy(frame_mask))

        instance_masks = torch.stack(instance_masks, dim=0)  # [num_frames, H, W]

        # Convert to tensors
        classes = torch.tensor(instance_classes, dtype=torch.long) if instance_classes else torch.zeros(0, dtype=torch.long)
        coordinates = torch.tensor(instance_coords, dtype=torch.float32) if instance_coords else torch.zeros((0, 2), dtype=torch.float32)

        return {
            "images": images,
            "masks": instance_masks,
            "classes": classes,
            "coordinates": coordinates,
            "frame_names": frame_names,
            "num_instances": len(instance_classes),
        }

    @staticmethod
    def _get_centroid(mask: np.ndarray) -> Tuple[float, float]:
        """
        Compute (u, v) centroid of a binary mask in normalized coordinates.

        Args:
            mask: Binary numpy array [H, W]

        Returns:
            (u, v) tuple in normalized coordinates [0, 1]
        """
        if not mask.any():
            return (0.5, 0.5)

        coords = np.argwhere(mask)  # [N, 2] in (row, col) format
        centroid_row = coords[:, 0].mean()
        centroid_col = coords[:, 1].mean()

        H, W = mask.shape
        u = centroid_col / (W - 1)  # Normalize to [0, 1]
        v = centroid_row / (H - 1)

        return (float(u), float(v))
