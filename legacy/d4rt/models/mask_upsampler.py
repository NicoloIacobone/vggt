# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
"""
MaskDINO / Mask2Former-style pixel decoder (Phase 5).

The dense mask head produces a mask per query as the cosine similarity between the query's
mask embedding and a per-pixel feature map. Today that feature map is the raw 37×37 VGGT
patch grid, which caps mask sharpness. This module upsamples the patch-feature map by a
power-of-two factor (bilinear + 3×3 conv stages) BEFORE the query⊗feature product, so masks
are predicted at 74×74 / 148×148 instead. The cosine-similarity + learnable-temperature mask
logit (the hard-won constraint) stays in the decoder; this module only changes the resolution
and channel projection of the pixel features.
"""

import math
import torch
import torch.nn as nn


class MaskUpsampler(nn.Module):
    """
    Upsample VGGT patch features into a higher-resolution per-pixel feature map.

    Input : patch features `[B, S, h, w, memory_dim]` (the square VGGT patch grid).
    Output: pixel features `[B, S, h*upsample, w*upsample, mask_embed_dim]`.

    `upsample` must be a power of two. The decoder only instantiates this for `upsample > 1`
    (at 1 it keeps its original `Linear` projection, so default behavior is byte-for-byte
    unchanged); `upsample=1` here is still valid and degenerates to a 1×1 projection.

    Args:
        memory_dim (int): channels of the input patch features (2048 for VGGT).
        mask_embed_dim (int): output channels (must match the query mask-embedding dim).
        upsample (int): spatial upsampling factor (power of two).
        norm_groups (int): groups for the per-stage GroupNorm.
    """

    def __init__(self, memory_dim: int = 2048, mask_embed_dim: int = 256,
                 upsample: int = 2, norm_groups: int = 8):
        super().__init__()
        if upsample < 1 or (upsample & (upsample - 1)) != 0:
            raise ValueError(f"upsample must be a power of two, got {upsample}")
        self.upsample = upsample
        self.mask_embed_dim = mask_embed_dim

        self.input_proj = nn.Conv2d(memory_dim, mask_embed_dim, kernel_size=1)
        stages = []
        for _ in range(int(math.log2(upsample))):
            stages += [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(mask_embed_dim, mask_embed_dim, kernel_size=3, padding=1),
                nn.GroupNorm(min(norm_groups, mask_embed_dim), mask_embed_dim),
                nn.ReLU(inplace=True),
            ]
        self.stages = nn.Sequential(*stages)

    def forward(self, patch_feats: torch.Tensor) -> torch.Tensor:
        """`[B, S, h, w, memory_dim]` → `[B, S, h*upsample, w*upsample, mask_embed_dim]`."""
        B, S, h, w, C = patch_feats.shape
        x = patch_feats.permute(0, 1, 4, 2, 3).reshape(B * S, C, h, w)  # [B*S, C, h, w]
        x = self.stages(self.input_proj(x))                            # [B*S, D, h*f, w*f]
        Hf, Wf = x.shape[-2:]
        return x.reshape(B, S, self.mask_embed_dim, Hf, Wf).permute(0, 1, 3, 4, 2)
