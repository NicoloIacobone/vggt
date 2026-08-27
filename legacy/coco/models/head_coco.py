"""
`MaskDINOCocoHead` — the trainable unit of the COCO backbone-swap study.

Same contract as `MaskDINOVGGTHead` (pixel decoder + MaskDINO decoder, `head_config` derived from
`locals()` so a new constructor argument can never be silently missing from a checkpoint), but
built on `CocoPixelDecoder`, which accepts either a single ViT map or a ResNet pyramid.

Not re-exported from `models/maskdino/__init__.py`: the CPU tests import it without pulling in any
backbone weights, exactly as the ScanNet head does.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from .decoder import MaskDINODecoder
from .pixel_decoder_coco import CocoPixelDecoder

#: COCO instance segmentation: 80 contiguous foreground classes, no background column (MaskDINO
#: represents "no object" as all sigmoid logits low — CLAUDE.md, docs/MASKDINO.md §4).
NUM_COCO_CLASSES = 80


class MaskDINOCocoHead(nn.Module):
    def __init__(
        self,
        in_channels: Sequence[int] = (2048,),
        highres_channels: Optional[int] = None,
        hidden_dim: int = 256,
        mask_dim: int = 256,
        num_classes: int = NUM_COCO_CLASSES,
        num_queries: int = 300,
        num_feature_levels: int = 3,
        enc_layers: int = 6,
        dec_layers: int = 9,
        nheads: int = 8,
        enc_dim_feedforward: int = 1024,
        dec_dim_feedforward: int = 2048,
        dropout: float = 0.0,
        enc_n_points: int = 4,
        dec_n_points: int = 4,
        two_stage: bool = True,
        learn_tgt: bool = False,
        initial_pred: bool = True,
        initialize_box_type: str = "bitmask",
        dn: str = "seg",
        dn_num: int = 100,
        noise_scale: float = 0.4,
        mask_upsample: int = 1,
    ):
        super().__init__()
        # The checkpoint round-trip contract (CLAUDE.md): `head_config` must describe every
        # constructor argument, so it is captured rather than restated.
        self.head_config = {k: v for k, v in locals().items()
                            if k not in ("self", "__class__")}
        self.head_config["in_channels"] = list(in_channels)

        self.pixel_decoder = CocoPixelDecoder(
            in_channels=in_channels, highres_channels=highres_channels, conv_dim=hidden_dim,
            mask_dim=mask_dim, num_feature_levels=num_feature_levels, enc_layers=enc_layers,
            nheads=nheads, dim_feedforward=enc_dim_feedforward, dropout=dropout,
            enc_n_points=enc_n_points, mask_upsample=mask_upsample,
        )
        self.predictor = MaskDINODecoder(
            in_channels=hidden_dim, num_classes=num_classes, hidden_dim=hidden_dim,
            num_queries=num_queries, nheads=nheads, dim_feedforward=dec_dim_feedforward,
            dec_layers=dec_layers, mask_dim=mask_dim, two_stage=two_stage, dn=dn,
            noise_scale=noise_scale, dn_num=dn_num, initialize_box_type=initialize_box_type,
            initial_pred=initial_pred, learn_tgt=learn_tgt,
            total_num_feature_levels=num_feature_levels, dropout=dropout,
            dec_n_points=dec_n_points,
        )

    @property
    def num_queries(self) -> int:
        return self.predictor.num_queries

    @property
    def num_classes(self) -> int:
        return self.predictor.num_classes

    def forward(self, levels: List[Tensor], highres: Optional[Tensor] = None,
                targets: Optional[List[Dict[str, Tensor]]] = None) -> Tuple[Dict, Optional[Dict]]:
        """
        Args:
            levels: backbone feature maps HIGH→LOW resolution (`FrozenBackbone`'s `levels`).
            highres: the optional stride-4 map for `mask_features` (ResNet only).
            targets: per-image GT dicts (labels/boxes/masks); needed only for denoising.
        Returns:
            (out, mask_dict) exactly as MaskDINO's decoder returns them.
        """
        mask_features, multi_scale = self.pixel_decoder(levels, highres)
        return self.predictor(multi_scale, mask_features, targets)


def build_coco_head_from_config(config: Dict) -> MaskDINOCocoHead:
    """Rebuild a head from a checkpoint's `head_config` (unknown keys are ignored)."""
    import inspect

    valid = set(inspect.signature(MaskDINOCocoHead.__init__).parameters) - {"self"}
    return MaskDINOCocoHead(**{k: v for k, v in config.items() if k in valid})
