"""Frozen backbone + `MaskDINOCocoHead` — the model of the COCO backbone-swap study."""

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from models.maskdino.coco_backbones import build_backbone
from models.maskdino.head_coco import MaskDINOCocoHead


class MaskDINOCocoModel(torch.nn.Module):
    """
    One frozen backbone (`vggt` / `dinov2` / `resnet50`) + the MaskDINO head.

    Only the head trains, in every arm — that is what makes the three comparable. The backbone
    reports the channel widths and strides the head must be built for, so a caller never has to
    hardcode "2048" or "stride 14" per arm.
    """

    def __init__(self, backbone_name: str, head_kwargs: Dict, load_backbone: bool = True,
                 backbone_kwargs: Optional[Dict] = None):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = None
        if load_backbone:
            self.backbone = build_backbone(backbone_name, **(backbone_kwargs or {}))
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
            head_kwargs = dict(head_kwargs)
            head_kwargs.setdefault("in_channels", self.backbone.out_channels)
            head_kwargs.setdefault("highres_channels", self.backbone.highres_channels)
        self.head = MaskDINOCocoHead(**head_kwargs)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone is not None:
            self.backbone.eval()   # frozen: never leave eval (dropout/norm stay deterministic)
        return self

    def mask_grid(self, img_size: int) -> int:
        """
        Side of the `mask_features` grid for a square `img_size` input.

        Computed from the level-0 grid, NOT as `img_size // stride`: a ViT arm's effective stride
        is fractional (14/4 = 3.5 at `mask_upsample=4`), so the integer-stride form reported
        172x172 for what is really a 148x148 map.
        """
        if self.backbone.highres_channels is not None:
            return -(-img_size // self.backbone.highres_stride)      # ResNet: ceil, like its convs
        level0 = img_size // self.backbone.strides[0]
        return level0 * self.head.head_config["mask_upsample"]

    @torch.no_grad()
    def extract(self, images: Tensor) -> Tuple[List[Tensor], Optional[Tensor]]:
        """Frozen features for a batch of images in [0, 1], [B, 3, H, W]."""
        out = self.backbone(images)
        return out["levels"], out["highres"]

    def forward(self, images: Tensor, targets=None):
        levels, highres = self.extract(images)
        return self.head(levels, highres, targets)
