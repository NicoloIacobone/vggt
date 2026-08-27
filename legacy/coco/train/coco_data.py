"""
COCO instance-segmentation data for the backbone-swap study.

Parallel to `train/maskdino_data.py`, which is ScanNet-specific end to end (scene bundles, a
frozen-feature cache, global instance ids). Nothing of that applies here: COCO images are
independent, there are 118 k of them so the features cannot be cached (618 GB at the VGGT token
size), and augmentation would invalidate a cache anyway. So the backbone runs inline in the
training loop and this module only produces images + targets.

Geometry — deliberately identical for every arm, so the comparison is about the backbone:

  image      original H×W  →  **squashed** to `img_size`×`img_size` (aspect ratio discarded)
  GT masks   rasterised at `gt_mask_size`×`gt_mask_size` in that same squashed frame
  GT boxes   cxcywh normalised to [0,1] — invariant under the squash

Squash rather than pad: `scripts/coco_mask_resolution_oracle.py` measures a 4.9 mask-AP ceiling
penalty for centre-padding to a square at a fixed token budget (39.8 vs 44.7 on the 37² grid),
because the padding spends grid cells on nothing. Aspect-preserving variable shapes score the
same as squashing at equal token count (and cost per-shape batching), so squashing wins.

`gt_mask_size` is independent of the prediction grid on purpose: both the matcher and
`SetCriterion.loss_masks` compare masks through PointRend `point_sample` at normalised
coordinates, so GT may live at a finer resolution than the prediction — and it must, or the
supervision would inherit the very quantisation the study is measuring.
"""

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

# `pycocotools` is imported lazily so the CPU tests can exercise the pure-tensor helpers
# (target building, collation) in an environment without it.


def coco_category_mapping(ann_file: str) -> Tuple[Dict[int, int], List[int]]:
    """
    COCO's 80 instance categories carry non-contiguous ids 1..90. MaskDINO's class head has 80
    contiguous sigmoid logits, so map dataset id → 0..79 and keep the inverse for submission.
    """
    with open(ann_file) as f:
        cats = json.load(f)["categories"]
    ids = sorted(c["id"] for c in cats)
    return {cid: i for i, cid in enumerate(ids)}, ids


def masks_to_boxes_normalized(masks: torch.Tensor) -> torch.Tensor:
    """[n,h,w] binary → [n,4] cxcywh in [0,1]. Empty masks give a zero box."""
    if masks.numel() == 0:
        return masks.new_zeros((0, 4))
    n, h, w = masks.shape
    m = masks.bool()
    ys = m.any(2).float()
    xs = m.any(1).float()
    idx_y = torch.arange(h, device=masks.device, dtype=torch.float32)
    idx_x = torch.arange(w, device=masks.device, dtype=torch.float32)
    big = float(max(h, w) + 1)
    y0 = torch.where(ys > 0, idx_y, torch.full_like(idx_y, big)).min(1).values
    y1 = torch.where(ys > 0, idx_y, torch.full_like(idx_y, -1.0)).max(1).values + 1
    x0 = torch.where(xs > 0, idx_x, torch.full_like(idx_x, big)).min(1).values
    x1 = torch.where(xs > 0, idx_x, torch.full_like(idx_x, -1.0)).max(1).values + 1
    empty = ~m.any(2).any(1)
    y0, y1 = y0.clamp(0, h), y1.clamp(0, h)
    x0, x1 = x0.clamp(0, w), x1.clamp(0, w)
    boxes = torch.stack([(x0 + x1) / 2 / w, (y0 + y1) / 2 / h,
                         (x1 - x0) / w, (y1 - y0) / h], dim=1)
    boxes[empty] = 0.0
    return boxes


def xywh_to_cxcywh_normalized(bbox, width: int, height: int) -> List[float]:
    """COCO's absolute xywh box → cxcywh normalised to the image. Squash-invariant."""
    x, y, w, h = bbox
    return [(x + w / 2) / width, (y + h / 2) / height, w / width, h / height]


class CocoInstanceDataset(Dataset):
    """
    COCO instance segmentation, MaskDINO target format.

    Each item is
        images   [3, img_size, img_size]  float32 in [0, 1]
        targets  {"labels": [n] int64 (0..79),
                  "masks":  [n, gt_mask_size, gt_mask_size] uint8,
                  "boxes":  [n, 4] float32 cxcywh in [0,1]}
        image_id, orig_size (H, W)

    Args:
        train: enables the augmentation (horizontal flip) and drops images with no annotation,
            matching detectron2's `filter_empty_annotations` default. Eval keeps every image —
            COCOeval needs the full 5000.
    """

    def __init__(self, img_root: str, ann_file: str, img_size: int = 518,
                 gt_mask_size: int = 296, train: bool = True, hflip: float = 0.5,
                 limit: int = 0):
        from pycocotools.coco import COCO

        self.img_root = img_root.rstrip("/")
        self.img_size = int(img_size)
        self.gt_mask_size = int(gt_mask_size)
        self.train = train
        self.hflip = hflip if train else 0.0

        self.coco = COCO(ann_file)
        self.cat2contig, self.contig2cat = coco_category_mapping(ann_file)

        ids = sorted(self.coco.getImgIds())
        if train:
            # detectron2's `filter_empty_annotations`. Read `imgToAnns` directly rather than
            # calling getAnnIds/loadAnns per image: the latter is ~2 min over 118 k images, and
            # every SLURM resubmit would pay it again.
            img_to_anns = self.coco.imgToAnns
            ids = [i for i in ids if any(self._usable(a) for a in img_to_anns.get(i, ()))]
        if limit:
            ids = ids[:limit]
        self.ids = ids

    @staticmethod
    def _usable(ann) -> bool:
        """Non-crowd, non-degenerate — the one definition used by both the filter and the loader."""
        return (not ann.get("iscrowd", 0) and ann["area"] > 0
                and ann["bbox"][2] > 1 and ann["bbox"][3] > 1)

    def __len__(self) -> int:
        return len(self.ids)

    def _load_image(self, info) -> torch.Tensor:
        img = Image.open(f"{self.img_root}/{info['file_name']}")
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
        arr = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy())
        return arr.permute(2, 0, 1).float() / 255.0

    def _load_targets(self, img_id: int, info) -> Dict[str, torch.Tensor]:
        g = self.gt_mask_size
        anns = [a for a in self.coco.imgToAnns.get(img_id, ()) if self._usable(a)]
        if not anns:
            return {"labels": torch.zeros(0, dtype=torch.long),
                    "masks": torch.zeros(0, g, g, dtype=torch.uint8),
                    "boxes": torch.zeros(0, 4)}

        raw = np.stack([self.coco.annToMask(a) for a in anns]).astype(np.float32)  # [n,H,W]
        # area interpolation = per-cell foreground fraction; >0.5 is the same rule the oracle and
        # the evaluator use, so GT, supervision and scoring quantise identically.
        small = F.interpolate(torch.from_numpy(raw)[None], size=(g, g), mode="area")[0]
        masks = (small > 0.5).to(torch.uint8)

        labels = torch.as_tensor([self.cat2contig[a["category_id"]] for a in anns],
                                 dtype=torch.long)
        boxes = torch.as_tensor(
            [xywh_to_cxcywh_normalized(a["bbox"], info["width"], info["height"]) for a in anns],
            dtype=torch.float32).clamp(0.0, 1.0)
        return {"labels": labels, "masks": masks, "boxes": boxes}

    def __getitem__(self, idx: int) -> Dict:
        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        image = self._load_image(info)
        targets = self._load_targets(img_id, info)

        if self.hflip and torch.rand(()) < self.hflip:
            image = torch.flip(image, dims=[2])
            targets["masks"] = torch.flip(targets["masks"], dims=[2])
            if targets["boxes"].numel():
                targets["boxes"][:, 0] = 1.0 - targets["boxes"][:, 0]

        return {"image": image, "targets": targets, "image_id": img_id,
                "orig_size": (info["height"], info["width"])}


def collate(batch: List[Dict]) -> Dict:
    """Stack the images; keep the targets as a list (MaskDINO's per-image convention)."""
    return {
        "images": torch.stack([b["image"] for b in batch]),
        "targets": [b["targets"] for b in batch],
        "image_ids": [b["image_id"] for b in batch],
        "orig_sizes": [b["orig_size"] for b in batch],
    }


def targets_to_device(targets: List[Dict], device: str) -> List[Dict]:
    return [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]


def build_loaders(args, distributed: bool = False):
    """Train + val `DataLoader`s from the usual CLI arguments."""
    from torch.utils.data import DataLoader

    train_set = CocoInstanceDataset(
        f"{args.coco_root}/train2017", f"{args.coco_root}/annotations/instances_train2017.json",
        img_size=args.img_size, gt_mask_size=args.gt_mask_size, train=True,
        hflip=args.hflip, limit=args.limit_train)
    val_set = CocoInstanceDataset(
        f"{args.coco_root}/val2017", f"{args.coco_root}/annotations/instances_val2017.json",
        img_size=args.img_size, gt_mask_size=args.gt_mask_size, train=False,
        limit=args.limit_val)

    # The loader yields MICRO-batches; the training loop accumulates `batch_size // micro_batch`
    # of them into one optimiser step (see --micro_batch).
    train_loader = DataLoader(train_set, batch_size=args.micro_batch, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate,
                              pin_memory=True, drop_last=True,
                              persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_set, batch_size=args.eval_batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate,
                            pin_memory=True, persistent_workers=args.num_workers > 0)
    return train_set, val_set, train_loader, val_loader
