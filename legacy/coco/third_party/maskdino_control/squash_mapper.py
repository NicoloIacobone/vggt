"""
The dataset mapper that replaces upstream's LSJ pipeline with our arms' input recipe.

Upstream (`COCOInstanceNewBaselineDatasetMapper`, `INPUT.DATASET_MAPPER_NAME=coco_instance_lsj`):
    RandomFlip -> ResizeScale(0.1..2.0, target 1024) -> FixedSizeCrop(1024) .

Ours (`train/coco_data.py`):
    squash to `INPUT.SQUASH_SIZE` x `INPUT.SQUASH_SIZE` (aspect ratio discarded), horizontal flip
    at p=0.5, nothing else.

Squash rather than pad: `scripts/coco_mask_resolution_oracle.py` measures a 4.9 mask-AP ceiling
penalty for centre-padding to a square at a fixed token budget (docs/MASKDINO_COCO.md §1.4).
The squash is inverted exactly at eval time, because upstream's `MaskDINO.forward` resizes each
mask straight to the ORIGINAL `height`/`width` carried in the dataset dict.

Flip and resize commute, so applying the flip first (detectron2's convention) matches
`train/coco_data.py`, which flips the already-squashed tensor.
"""

import copy

import numpy as np
import torch
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

__all__ = ["CocoSquashDatasetMapper", "build_squash_augmentations"]


def build_squash_augmentations(cfg, is_train):
    size = cfg.INPUT.SQUASH_SIZE
    aug = []
    if is_train and cfg.CONTROL.HFLIP_PROB > 0:
        aug.append(T.RandomFlip(prob=cfg.CONTROL.HFLIP_PROB, horizontal=True, vertical=False))
    # T.Resize((h, w)) is an unconditional resize to that exact shape = the squash. It goes
    # through PIL BILINEAR for uint8 input, the same interpolation train/coco_data.py uses.
    aug.append(T.Resize((size, size)))
    return aug


class CocoSquashDatasetMapper:
    """Callable mapping a detectron2 dataset dict to MaskDINO's expected training format."""

    def __init__(self, cfg, is_train: bool):
        self.is_train = is_train
        self.img_format = cfg.INPUT.FORMAT
        self.augmentations = T.AugmentationList(build_squash_augmentations(cfg, is_train))
        self.size = cfg.INPUT.SQUASH_SIZE

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image
        image_shape = image.shape[:2]                      # (H, W) == (size, size)

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            # "height"/"width" stay the ORIGINAL image size; that is what inverts the squash in
            # `MaskDINO.forward`'s postprocess. Annotations are read from the GT json by
            # COCOEvaluator, not from here.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        annos = [
            utils.transform_instance_annotations(obj, transforms, image_shape)
            for obj in dataset_dict.pop("annotations", [])
            if obj.get("iscrowd", 0) == 0
        ]
        # mask_format="bitmask" rasterises the polygons directly at the squashed resolution.
        # `MaskDINO.prepare_targets` reads `gt_masks` as a raw tensor, so BitMasks is unwrapped
        # below -- but only AFTER filter_empty_instances, which needs `.nonempty()`.
        instances = utils.annotations_to_instances(annos, image_shape, mask_format="bitmask")
        if instances.has("gt_masks"):
            instances = utils.filter_empty_instances(instances)
            instances.gt_masks = instances.gt_masks.tensor
        else:
            instances.gt_masks = torch.zeros((len(instances),) + image_shape, dtype=torch.uint8)
        dataset_dict["instances"] = instances
        return dataset_dict
