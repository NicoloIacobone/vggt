"""
MaskDINO decoder on top of frozen VGGT features — a parallel experiment track.

See docs/MASKDINO_TRIAL.md for the plan, the deviations from upstream MaskDINO, and results.
Nothing in this package is imported by the D4RT arms (models/d4rt_decoder.py & co.).
"""

from .criterion import SetCriterion, build_weight_dict
from .head import NUM_SCANNET_CLASSES, MaskDINOVGGTHead, build_head_from_config, to_scannet_class_logits
from .matcher import HungarianMatcher, check_target_labels

__all__ = [
    "MaskDINOVGGTHead",
    "NUM_SCANNET_CLASSES",
    "build_head_from_config",
    "to_scannet_class_logits",
    "HungarianMatcher",
    "check_target_labels",
    "SetCriterion",
    "build_weight_dict",
]
