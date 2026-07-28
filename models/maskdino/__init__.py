"""
MaskDINO decoder on top of frozen VGGT features — a parallel experiment track.

See docs/MASKDINO_TRIAL.md for the plan, the deviations from upstream MaskDINO, and results.
Nothing in this package is imported by the D4RT arms (models/d4rt_decoder.py & co.).
"""

from .criterion import SetCriterion, build_weight_dict
from .head import MaskDINOVGGTHead, build_head_from_config, to_scannet_class_logits
from .matcher import HungarianMatcher

__all__ = [
    "MaskDINOVGGTHead",
    "build_head_from_config",
    "to_scannet_class_logits",
    "HungarianMatcher",
    "SetCriterion",
    "build_weight_dict",
]
