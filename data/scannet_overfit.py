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
        instance_level (bool): if False (default), read per-class binary masks from `masks/`
            and assign one global ID per class. If True, read per-instance masks from
            `masks_instance/<class>_<k>/` and assign one global ID per (class, instance) — two
            objects of the same class then become distinct GT instances that share a class
            index (`classes` contains repeated class indices). Stuff classes (wall/floor) are
            single instances on disk, so they behave the same in both modes.

    Cross-view instance identity (item 8.3): each mask SEGMENT present in the scene is treated
    as ONE multi-view instance with a single global ID consistent across all sampled frames
    (e.g. a "wall" region keeps the same ID in every view it appears in), rather than minting a
    fresh ID for every (frame, segment) pair. In the default per-class mode a segment is a
    whole class, so class-level linking is the finest identity the *binary per-class* PNGs
    support; in `instance_level` mode a segment is one tracked instance, so same-class objects
    are separated (SAM3 video tracking provides the cross-frame identity). Each returned
    instance is described once (per-global-instance arrays below) but may occupy several frames
    in the `masks` map.

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
        instance_level: bool = False,
    ):
        super().__init__()
        self.scene_dir = Path(scene_dir)
        self.num_frames = num_frames
        self.image_ext = image_ext
        self.mask_ext = mask_ext
        self.img_size = img_size
        self.instance_level = instance_level
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

        masks_dirname = "masks_instance" if instance_level else "masks"
        self.masks_dir = self.scene_dir / masks_dirname
        if not self.masks_dir.exists():
            raise ValueError(f"Masks directory not found: {self.masks_dir}")

        # Find all image files
        self.image_files = sorted([
            f for f in self.images_dir.iterdir()
            if f.suffix.lower() == image_ext.lower()
        ])

        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.images_dir}")

        # Build the list of mask SEGMENTS — each segment becomes one global instance.
        # A segment is (canonical_class_name, segment_dir); the list is sorted into a
        # deterministic order so the same scene always yields the same (global_id -> class)
        # mapping. On-disk folders may use underscores (e.g. 'shower_curtain') while the
        # canonical class name uses a space ('shower curtain'); accept either.
        #   - per-class mode (default):  one segment per class folder in masks/.
        #   - instance mode:             one segment per masks_instance/<class>_<k>/ folder,
        #                                so two objects of the same class are distinct GT
        #                                instances that share a class index.
        if instance_level:
            # Map both spelling variants of every class name to the canonical form.
            norm_to_canon = {}
            for cls_name in SCANNET_CLASSES:
                norm_to_canon[cls_name] = cls_name
                norm_to_canon[cls_name.replace(" ", "_")] = cls_name
            parsed = []  # (class_idx, k, canonical_class_name, dir)
            for d in sorted(self.masks_dir.iterdir()):
                # Folders are '<class>_<k>'; <class> may itself contain underscores and
                # <k> is a trailing integer. Skip QA/metadata dirs (e.g. '_qa').
                if not d.is_dir() or d.name.startswith("_") or "_" not in d.name:
                    continue
                class_part, k_part = d.name.rsplit("_", 1)
                if not k_part.isdigit():
                    continue
                cls_name = norm_to_canon.get(class_part)
                if cls_name is None:
                    continue
                parsed.append((CLASS_TO_IDX[cls_name], int(k_part), cls_name, d))
            parsed.sort(key=lambda t: (t[0], t[1]))  # class index, then instance index k
            self.segments = [(cls_name, d) for (_, _, cls_name, d) in parsed]
        else:
            class_dirs = {}
            for cls_name in SCANNET_CLASSES:
                for cand in (cls_name, cls_name.replace(" ", "_")):
                    cand_dir = self.masks_dir / cand
                    if cand_dir.exists():
                        class_dirs[cls_name] = cand_dir
                        break
            self.class_dirs = class_dirs  # kept for backward compatibility/inspection
            self.segments = [
                (c, class_dirs[c]) for c in sorted(class_dirs, key=lambda c: CLASS_TO_IDX[c])
            ]

        if not self.segments:
            kind = "instance" if instance_level else "class"
            raise ValueError(f"No {kind} mask folders found in {self.masks_dir}")

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

        # --- Pass 1: load every per-frame, per-segment binary mask ---------------------------
        # Collect, for each segment that has foreground in ANY sampled frame, the set of
        # frames it appears in and its binary pixel mask there. A segment is one class folder
        # (per-class mode) or one instance folder (instance mode); either way it yields a
        # SINGLE global ID consistent across views (cross-view identity, item 8.3) rather than
        # a fresh ID per (frame, segment) pair.
        per_seg_frame_pixels: Dict[int, Dict[int, np.ndarray]] = {}

        for frame_idx, frame_name in enumerate(frame_names):
            for seg_idx, (class_name, seg_dir) in enumerate(self.segments):
                mask_path = seg_dir / f"{frame_name}{self.mask_ext}"
                if not mask_path.exists():
                    continue

                class_mask = Image.open(mask_path).convert("L")
                class_mask = class_mask.resize((self.img_size, self.img_size), Image.NEAREST)
                class_mask_array = np.array(class_mask, dtype=np.uint8)

                # The on-disk masks are binary (one blob per segment per frame).
                if class_mask_array.max() == 0:
                    continue
                class_pixels = class_mask_array > 127  # Threshold at 127
                if not class_pixels.any():
                    continue

                per_seg_frame_pixels.setdefault(seg_idx, {})[frame_idx] = class_pixels

        # --- Pass 2: assign global instance IDs and paint the per-frame instance maps --------
        # Deterministic ID order: segments are already ordered (class index, then instance k),
        # so iterating present segments in sorted seg_idx order gives a stable
        # (instance_id -> class) mapping across runs.
        present_segs = sorted(per_seg_frame_pixels.keys())

        # int32 (not uint8) so the global instance IDs cannot overflow if many segments appear.
        instance_masks = np.zeros((num_frames, self.img_size, self.img_size), dtype=np.int32)

        instance_classes = []
        instance_coords = []   # representative (largest-area frame) centroid per instance
        instance_frames = []   # representative frame index per instance
        instance_ids = []      # the global ID written into `instance_masks` (1..G)

        for global_id, seg_idx in enumerate(present_segs, start=1):
            class_name = self.segments[seg_idx][0]
            frame_pixels = per_seg_frame_pixels[seg_idx]

            best_frame, best_area, best_centroid = -1, -1, (0.5, 0.5)
            for frame_idx, class_pixels in frame_pixels.items():
                # Paint the SAME global ID into every frame this instance appears in.
                # In instance mode same-class instances keep distinct IDs; later painted
                # segments win on cross-class pixel overlaps (matches per-class behavior).
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
