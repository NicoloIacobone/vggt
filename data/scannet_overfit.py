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


def load_frames_by_name(
    scene_dir: str,
    frame_names: List,
    img_size: int = 518,
    image_ext: str = ".jpg",
) -> torch.Tensor:
    """
    Load specific subset frames by their stem name into a float tensor
    [S, 3, img_size, img_size] in [0, 1]. Mirrors ScanNetSingleSceneDataset's image
    loading; used to rehydrate `--checkpoint_light` bundles (which store frame names +
    the scene path instead of the pixels) at visualization/demo time.
    """
    scene_dir = Path(scene_dir)
    images_dir = None
    for cand in ("subset", "images", "color"):
        if (scene_dir / cand).exists():
            images_dir = scene_dir / cand
            break
    if images_dir is None:
        raise ValueError(f"Images directory not found under {scene_dir}")

    imgs = []
    for name in frame_names:
        # Collation may wrap each name in a 1-element list (batch_size=1).
        if isinstance(name, (list, tuple)):
            name = name[0]
        path = images_dir / f"{name}{image_ext}"
        img = Image.open(path).convert("RGB").resize((img_size, img_size), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        imgs.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(imgs, dim=0)  # [S, 3, H, W]


def decode_checkpoint_images(
    scene: Dict,
    scans_root: Optional[str] = None,
    img_size: int = 518,
) -> torch.Tensor:
    """
    Return a scene's frames as a float tensor [1, S, 3, H, W] in [0, 1], handling all three
    checkpoint storage formats:
      - float images (legacy)            → passed through;
      - uint8 images (compact, 4× smaller) → divided by 255;
      - no images (`--checkpoint_light`)   → reloaded from disk via `scene_dir`/`frame_names`
        (falling back to `<scans_root>/<name>/raw_data` when no explicit path was stored).
    """
    imgs = scene.get("images")
    if imgs is not None:
        return imgs.float() / 255.0 if imgs.dtype == torch.uint8 else imgs

    frame_names = scene.get("frame_names")
    if frame_names is None:
        raise ValueError("Light checkpoint scene has no frame_names to reload images from")
    scene_dir = scene.get("scene_dir")
    if scene_dir is None:
        if scans_root is None:
            raise ValueError("Light checkpoint needs --scans_root (no stored scene_dir)")
        scene_dir = str(Path(scans_root) / scene["name"] / "raw_data")
    frames = load_frames_by_name(scene_dir, frame_names, img_size)  # [S, 3, H, W]
    return frames.unsqueeze(0)  # [1, S, 3, H, W]


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
        frame_sampling (str): "random" samples num_frames frames anew on every __getitem__;
            "even" picks num_frames evenly-spaced frames (deterministic — required for a
            stable multi-scene overfit where the same frames must be revisited every epoch)

    Cross-view instance identity (item 8.3): a given ScanNet class present in the scene is
    treated as ONE multi-view instance with a single global ID that is consistent across all
    sampled frames (e.g. the "wall" region keeps the same instance ID in every view it appears
    in), rather than minting a fresh ID for every (frame, class) pair. Because the on-disk masks
    are *binary per-class* PNGs, they carry no information to separate two distinct objects of
    the same class, so class-level linking is the finest cross-view identity the labels support.
    Each returned instance is therefore described once (per-global-instance arrays below) but may
    occupy several frames in the `masks` map.

    Returns dict with:
        - images: torch.Tensor [num_frames, 3, img_size, img_size] in range [0, 1]
        - masks: torch.Tensor [num_frames, img_size, img_size] GLOBAL instance ID per pixel,
                 consistent across frames (0 = background, 1..G = instances)
        - classes: torch.Tensor [num_instances] class label of each global instance (1-19)
        - coordinates: torch.Tensor [num_instances, 2] (u, v) centroid in the instance's
                 representative (largest-area) frame
        - frame_ids: torch.Tensor [num_instances] representative frame index of each instance
        - instance_ids: torch.Tensor [num_instances] the global ID used in `masks` (1..G)
        - frame_names, num_instances: bookkeeping (num_instances == G global instances)
    """

    def __init__(
        self,
        scene_dir: str,
        num_frames: int = 8,
        image_ext: str = ".jpg",
        mask_ext: str = ".png",
        img_size: int = 518,
        images_subdir: Optional[str] = None,
        frame_sampling: str = "random",
    ):
        super().__init__()
        self.scene_dir = Path(scene_dir)
        self.num_frames = num_frames
        self.image_ext = image_ext
        self.mask_ext = mask_ext
        self.img_size = img_size
        if frame_sampling not in ("random", "even"):
            raise ValueError(f"frame_sampling must be 'random' or 'even', got {frame_sampling!r}")
        self.frame_sampling = frame_sampling

        # Locate the image directory.
        # IMPORTANT: masks are only computed for the subsampled set of frames (e.g. a
        # stride-5 subset of a >5000-frame scene). 'color' holds *all* raw frames, most of
        # which have no corresponding mask. We therefore prefer the 'subset' folder (the
        # masked frames) and only fall back to 'images'/'color' if it is absent.
        if images_subdir is not None:
            candidates = [images_subdir]
        else:
            candidates = ["subset", "images", "color"]

        self.images_dir = None
        for cand in candidates:
            if (self.scene_dir / cand).exists():
                self.images_dir = self.scene_dir / cand
                break
        if self.images_dir is None:
            raise ValueError(
                f"Images directory not found (tried {candidates}): {self.scene_dir}"
            )

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

        # Find all class folders in masks directory. On-disk folders may use underscores
        # (e.g. 'shower_curtain') while the canonical class name uses a space
        # ('shower curtain'); accept either so the class is not silently dropped.
        self.class_dirs = {}
        for cls_name in SCANNET_CLASSES:
            for cand in (cls_name, cls_name.replace(" ", "_")):
                cand_dir = self.masks_dir / cand
                if cand_dir.exists():
                    self.class_dirs[cls_name] = cand_dir
                    break

        if not self.class_dirs:
            raise ValueError(f"No class folders found in {self.masks_dir}")

    def __len__(self):
        return 1  # Single scene dataset - always returns 1 sample

    def __getitem__(self, idx):
        k = min(self.num_frames, len(self.image_files))
        if self.frame_sampling == "even":
            # Deterministic, evenly-spaced frames spanning the scene (stable across epochs).
            sampled_indices = np.unique(
                np.linspace(0, len(self.image_files) - 1, k).round().astype(int)
            ).tolist()
        else:
            sampled_indices = random.sample(range(len(self.image_files)), k)
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

        num_frames = len(frame_names)

        # --- Pass 1: load every per-frame, per-class binary mask -----------------------------
        # Collect, for each class that has foreground in ANY sampled frame, the set of frames
        # it appears in and its binary pixel mask there. This lets us assign a SINGLE global
        # instance ID per class (cross-view identity, item 8.3) instead of a fresh ID per
        # (frame, class) pair.
        per_class_frame_pixels: Dict[str, Dict[int, np.ndarray]] = {}

        for frame_idx, frame_name in enumerate(frame_names):
            for class_name, class_dir in self.class_dirs.items():
                mask_path = class_dir / f"{frame_name}{self.mask_ext}"
                if not mask_path.exists():
                    continue

                class_mask = Image.open(mask_path).convert("L")
                class_mask = class_mask.resize((self.img_size, self.img_size), Image.NEAREST)
                class_mask_array = np.array(class_mask, dtype=np.uint8)

                # The on-disk masks are binary semantic masks (one blob per class per frame).
                if class_mask_array.max() == 0:
                    continue
                class_pixels = class_mask_array > 127  # Threshold at 127
                if not class_pixels.any():
                    continue

                per_class_frame_pixels.setdefault(class_name, {})[frame_idx] = class_pixels

        # --- Pass 2: assign global instance IDs and paint the per-frame instance maps --------
        # Deterministic ID order: sort by canonical class index so the same scene always yields
        # the same (instance_id -> class) mapping across runs.
        present_classes = sorted(per_class_frame_pixels.keys(), key=lambda c: CLASS_TO_IDX[c])

        # int32 (not uint8) so the global instance IDs cannot overflow if many classes appear.
        instance_masks = np.zeros((num_frames, self.img_size, self.img_size), dtype=np.int32)

        instance_classes = []
        instance_coords = []   # representative (largest-area frame) centroid per instance
        instance_frames = []   # representative frame index per instance
        instance_ids = []      # the global ID written into `instance_masks` (1..G)

        for global_id, class_name in enumerate(present_classes, start=1):
            frame_pixels = per_class_frame_pixels[class_name]

            best_frame, best_area, best_centroid = -1, -1, (0.5, 0.5)
            for frame_idx, class_pixels in frame_pixels.items():
                # Paint the SAME global ID into every frame this instance appears in.
                instance_masks[frame_idx][class_pixels] = global_id

                # Track the most-visible frame for the representative query point/centroid.
                area = int(class_pixels.sum())
                if area > best_area:
                    best_area = area
                    best_frame = frame_idx
                    best_centroid = self._get_centroid(class_pixels)

            instance_classes.append(CLASS_TO_IDX[class_name])
            instance_coords.append(best_centroid)
            instance_frames.append(best_frame)
            instance_ids.append(global_id)

        instance_masks = torch.from_numpy(instance_masks)  # [num_frames, H, W]

        # Convert to tensors. The i-th instance (0-indexed) has global instance-id (i + 1) in
        # `masks` across ALL frames it appears in; `classes[i]` is its class, `coordinates[i]`
        # and `frame_ids[i]` describe its representative (largest-area) view.
        classes = torch.tensor(instance_classes, dtype=torch.long) if instance_classes else torch.zeros(0, dtype=torch.long)
        coordinates = torch.tensor(instance_coords, dtype=torch.float32) if instance_coords else torch.zeros((0, 2), dtype=torch.float32)
        frame_ids = torch.tensor(instance_frames, dtype=torch.long) if instance_frames else torch.zeros(0, dtype=torch.long)
        instance_ids_t = torch.tensor(instance_ids, dtype=torch.long) if instance_ids else torch.zeros(0, dtype=torch.long)

        return {
            "images": images,
            "masks": instance_masks,
            "classes": classes,
            "coordinates": coordinates,
            "frame_ids": frame_ids,
            "instance_ids": instance_ids_t,
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


class ScanNetMultiSceneDataset(Dataset):
    """
    Multi-scene wrapper (item 8.7): one item per scene, each loaded by its own
    ScanNetSingleSceneDataset. Per-scene instance counts differ, so use batch_size=1
    (or a custom collate_fn) and let the batch-aware D4RTLoss match per sample.

    Args:
        scene_dirs: list of scene directories (each as accepted by ScanNetSingleSceneDataset)
        **kwargs: forwarded to every ScanNetSingleSceneDataset (num_frames, img_size,
            frame_sampling, ...)
    """

    def __init__(self, scene_dirs: List[str], **kwargs):
        super().__init__()
        if not scene_dirs:
            raise ValueError("scene_dirs must contain at least one scene directory")
        self.scenes = [ScanNetSingleSceneDataset(str(d), **kwargs) for d in scene_dirs]
        # Human-readable scene names: the scene folder, not the trailing 'raw_data'.
        self.scene_names = []
        for d in scene_dirs:
            p = Path(d)
            self.scene_names.append(p.parent.name if p.name == "raw_data" else p.name)

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        sample = self.scenes[idx][0]
        sample["scene_name"] = self.scene_names[idx]
        sample["scene_idx"] = idx
        return sample
