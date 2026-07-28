"""Frozen VGGT-1B backbone + the MaskDINO head — the trainable unit of this track."""

from typing import Dict

import torch

from models.maskdino.head import MaskDINOVGGTHead


class MaskDINOVGGTModel(torch.nn.Module):
    """Frozen VGGT-1B aggregator + the MaskDINO head. Only the head has trainable parameters."""

    def __init__(self, head_kwargs: Dict, load_backbone: bool = True):
        super().__init__()
        self.backbone = None
        if load_backbone:
            from vggt.models.vggt import VGGT
            print("Loading VGGT backbone...")
            try:
                self.backbone = VGGT.from_pretrained("facebook/VGGT-1B")
                print("✓ Loaded pretrained VGGT-1B")
            except Exception as e:  # offline / no HF cache → random init (tests only)
                print(f"⚠ Could not load pretrained VGGT: {e}\n  Falling back to random init.")
                self.backbone = VGGT()
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
        self.head = MaskDINOVGGTHead(**head_kwargs)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone is not None:
            self.backbone.eval()  # frozen: never leave eval (dropout/norm stay deterministic)
        return self
