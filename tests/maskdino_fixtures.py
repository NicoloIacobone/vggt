#!/usr/bin/env python3
"""Shared fixtures for the MaskDINO test modules (tests/test_maskdino_*.py)."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.maskdino import MaskDINOVGGTHead
from models.maskdino import box_ops


def _synthetic_targets(B, n, hw, num_classes=19):
    targets = []
    for _ in range(B):
        masks = torch.zeros(n, *hw)
        for i in range(n):
            masks[i, i:i + 2, i:i + 2] = 1.0
        targets.append({
            "labels": torch.randint(0, num_classes, (n,)),
            "masks": masks,
            "boxes": box_ops.masks_to_boxes_normalized(masks),
        })
    return targets


def _tiny_head(**kw):
    cfg = dict(memory_dim=64, hidden_dim=32, mask_dim=32, num_classes=19, num_queries=12,
               num_feature_levels=3, enc_layers=1, dec_layers=2, nheads=4,
               enc_dim_feedforward=64, dec_dim_feedforward=64, enc_n_points=2, dec_n_points=2,
               dn_num=12)
    cfg.update(kw)
    return MaskDINOVGGTHead(**cfg)
