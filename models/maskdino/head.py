"""
`MaskDINOVGGTHead` — the trainable head of the MaskDINO trial: pixel decoder + MaskDINO decoder
on top of frozen VGGT aggregator tokens (docs/MASKDINO_TRIAL.md).

Everything the head needs to be rebuilt from a checkpoint lives in `head_config` (mirroring the
D4RT head's round-trip contract in CLAUDE.md), so the visualizer / any eval script can
reconstruct it without knowing the training flags.
"""

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from .decoder import MaskDINODecoder
from .pixel_decoder import VGGTPixelDecoder

# The 19 trainable ScanNet classes are stored 1..19 in the dataset (0 = background). MaskDINO
# has no background column, so labels are shifted to 0..18 inside this trial and shifted back
# for the shared metric/visualisation code.
NUM_SCANNET_CLASSES = 19


class MaskDINOVGGTHead(nn.Module):
    def __init__(
        self,
        memory_dim: int = 2048,
        hidden_dim: int = 256,
        mask_dim: int = 256,
        num_classes: int = NUM_SCANNET_CLASSES,
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
        self.pixel_decoder = VGGTPixelDecoder(
            memory_dim=memory_dim, conv_dim=hidden_dim, mask_dim=mask_dim,
            num_feature_levels=num_feature_levels, enc_layers=enc_layers, nheads=nheads,
            dim_feedforward=enc_dim_feedforward, dropout=dropout, enc_n_points=enc_n_points,
            mask_upsample=mask_upsample,
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
        self.head_config = dict(
            memory_dim=memory_dim, hidden_dim=hidden_dim, mask_dim=mask_dim,
            num_classes=num_classes, num_queries=num_queries,
            num_feature_levels=num_feature_levels, enc_layers=enc_layers, dec_layers=dec_layers,
            nheads=nheads, enc_dim_feedforward=enc_dim_feedforward,
            dec_dim_feedforward=dec_dim_feedforward, dropout=dropout, enc_n_points=enc_n_points,
            dec_n_points=dec_n_points, two_stage=two_stage, learn_tgt=learn_tgt,
            initial_pred=initial_pred, initialize_box_type=initialize_box_type, dn=dn,
            dn_num=dn_num, noise_scale=noise_scale, mask_upsample=mask_upsample,
        )

    @property
    def num_queries(self) -> int:
        return self.predictor.num_queries

    def forward(self, tokens: Tensor, patch_start_idx: int = 5,
                targets: Optional[List[Dict[str, Tensor]]] = None) -> Tuple[Dict, Optional[Dict]]:
        """
        Args:
            tokens: [B, P, memory_dim] VGGT aggregator tokens for B independent FRAMES
                (single-frame protocol: the batch dimension is frames, not scenes).
            patch_start_idx: index where patch tokens start (5 for VGGT-1B).
            targets: per-frame GT dicts (labels/boxes/masks); needed only for denoising.
        Returns:
            (out, mask_dict) exactly as MaskDINO's decoder returns them.
        """
        mask_features, multi_scale = self.pixel_decoder(tokens, patch_start_idx)
        return self.predictor(multi_scale, mask_features, targets)


def build_head_from_config(config: Dict) -> MaskDINOVGGTHead:
    """Rebuild a head from a checkpoint's `head_config` (unknown keys are ignored)."""
    import inspect

    valid = set(inspect.signature(MaskDINOVGGTHead.__init__).parameters) - {"self"}
    return MaskDINOVGGTHead(**{k: v for k, v in config.items() if k in valid})


@torch.no_grad()
def to_scannet_class_logits(pred_logits: Tensor) -> Tensor:
    """
    [Q, 19] MaskDINO sigmoid logits → [Q, 20] in the project's ScanNet layout (index 0 =
    background, 1..19 = classes) so the shared metric / visualisation code can consume them.

    Column 0 is filled with -inf: there is no background *logit* in MaskDINO — "no object" is
    "all class scores low", which the metric handles via `score_mode="sigmoid"` + a score
    threshold, not via an argmax against a background column.
    """
    q = pred_logits.shape[0]
    out = pred_logits.new_full((q, pred_logits.shape[1] + 1), float("-inf"))
    out[:, 1:] = pred_logits
    return out
