"""
MaskDINO decoder on top of frozen VGGT features — the project's active model track.

See docs/MASKDINO.md for the architecture, the deviations from upstream MaskDINO, and results.
The retired D4RT query-strategy arms live under legacy/d4rt/ and share nothing with this package.

Layout:
    head.py            MaskDINOVGGTHead — pixel decoder + decoder, the trainable unit
    model.py           MaskDINOVGGTModel — frozen VGGT backbone + head
    pixel_decoder.py   VGGT tokens → feature pyramid → MSDeformAttn encoder
    decoder.py         MaskDINODecoder — two-stage selection, denoising, deep supervision
    decoder_layers.py  the generic DAB/DINO decoder stack it drives
    matcher.py         HungarianMatcher (class + box + point-sampled mask cost)
    criterion.py       SetCriterion (focal + BCE/Dice + L1/GIoU, incl. DN and aux losses)
    ms_deform_attn.py  pure-PyTorch multi-scale deformable attention (no CUDA extension)
    box_ops.py         box conversions + masks_to_boxes (replaces detectron2's BitMasks)
    utils.py           MLP, positional encodings, PointRend sampling — ported dependencies

`model.py` is deliberately NOT imported here: it pulls in the VGGT backbone, and the CPU tests
must be able to import the head without it.
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
