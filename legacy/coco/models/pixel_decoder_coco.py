"""
Pixel decoder for the COCO backbone-swap study — one module, two pyramid modes.

`VGGTPixelDecoder` (the ScanNet path) assumes exactly one input map and builds the extra levels
ViTDet-style. That is the right thing for a plain ViT and the wrong thing for a ResNet, which
already has a pyramid *and* a stride-4 map that upstream MaskDINO uses for `mask_features`.
Comparing backbones fairly means letting each one use its own natural structure, so this module
handles both:

  **ViTDet mode** (`levels` has 1 entry — `vggt`, `dinov2`)
      1×1 conv on the token map → level 0; stride-2 convs → levels 1..L-1; `mask_features` from
      level 0 through `mask_upsample` transposed convs. Identical maths to `VGGTPixelDecoder`.

  **FPN mode** (`levels` has L entries + a `highres` map — `resnet50`)
      1×1 conv per level → encoder; then upstream MaskDINO's top-down step onto the stride-4
      lateral → `mask_features`. `mask_upsample` is ignored (the lateral already sets the stride).

Everything between (sine PE, level embed, flatten, the 6-layer MSDeformAttn encoder) is shared
with the ScanNet path — the encoder classes are imported from `pixel_decoder.py`, not copied.

Why `mask_upsample` matters here and did not on ScanNet: `scripts/coco_mask_resolution_oracle.py`
shows a perfect model capped at **44.7 mask AP** on the 37×37 grid and **84.2** at 148×148
(docs/MASKDINO_COCO.md §2). ScanNet objects are large enough that 37×37 was never the binding
constraint; COCO objects are not.
"""

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.init import constant_, normal_, xavier_uniform_

from .ms_deform_attn import MSDeformAttn
from .pixel_decoder import MSDeformAttnEncoder, MSDeformAttnEncoderLayer
from .utils import PositionEmbeddingSine


class CocoPixelDecoder(nn.Module):
    """
    Backbone feature maps → (`mask_features`, multi-scale memory) for `MaskDINODecoder`.

    Args:
        in_channels: channels of each entry of the backbone's `levels`, HIGH→LOW resolution.
            A single entry selects ViTDet mode.
        highres_channels: channels of the backbone's extra stride-4 map, or None. Present ⇒
            FPN mode for `mask_features`.
        num_feature_levels: how many levels the encoder sees (MaskDINO: 3). In ViTDet mode the
            missing ones are synthesised with stride-2 convs.
        mask_upsample: ViTDet mode only — 1/2/4/8 transposed-conv doublings of `mask_features`.
        conv_dim / mask_dim / enc_layers / nheads / dim_feedforward / enc_n_points: as MaskDINO.

    Forward returns:
        mask_features: [B, mask_dim, H_m, W_m]
        multi_scale:   list of [B, conv_dim, H_l, W_l], ordered HIGH→LOW resolution.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        highres_channels: Optional[int] = None,
        conv_dim: int = 256,
        mask_dim: int = 256,
        num_feature_levels: int = 3,
        enc_layers: int = 6,
        nheads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        enc_n_points: int = 4,
        mask_upsample: int = 1,
    ):
        super().__init__()
        in_channels = list(in_channels)
        if not in_channels:
            raise ValueError("in_channels must list at least one backbone level")
        if num_feature_levels < len(in_channels):
            raise ValueError(f"num_feature_levels ({num_feature_levels}) < backbone levels "
                             f"({len(in_channels)}): the encoder cannot drop levels")
        if mask_upsample not in (1, 2, 4, 8):
            raise ValueError(f"mask_upsample must be 1, 2, 4 or 8, got {mask_upsample}")

        self.in_channels = in_channels
        self.num_native_levels = len(in_channels)
        self.num_feature_levels = num_feature_levels
        self.conv_dim = conv_dim
        self.mask_dim = mask_dim
        self.vitdet_mode = self.num_native_levels == 1
        self.mask_upsample = mask_upsample if self.vitdet_mode else 1

        # --- level projections -----------------------------------------------------------------
        proj: List[nn.Module] = [
            nn.Sequential(nn.Conv2d(c, conv_dim, kernel_size=1), nn.GroupNorm(32, conv_dim))
            for c in in_channels
        ]
        # ViTDet: synthesise the missing (coarser) levels from the last projected one.
        for _ in range(num_feature_levels - self.num_native_levels):
            proj.append(nn.Sequential(
                nn.Conv2d(conv_dim, conv_dim, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(32, conv_dim),
            ))
        self.input_proj = nn.ModuleList(proj)
        for p in self.input_proj:
            xavier_uniform_(p[0].weight, gain=1)
            constant_(p[0].bias, 0)

        self.pe_layer = PositionEmbeddingSine(conv_dim // 2, normalize=True)
        self.level_embed = nn.Parameter(torch.empty(num_feature_levels, conv_dim))
        normal_(self.level_embed)

        self.enc_layers = enc_layers
        if enc_layers > 0:
            layer = MSDeformAttnEncoderLayer(conv_dim, dim_feedforward, dropout, "relu",
                                             num_feature_levels, nheads, enc_n_points)
            self.encoder = MSDeformAttnEncoder(layer, enc_layers)
            self._reset_encoder_parameters()
        else:
            self.encoder = None

        # --- mask_features -----------------------------------------------------------------
        self.highres_channels = highres_channels
        if highres_channels is not None:
            # upstream MaskDINO's FPN step: lateral 1x1 on the stride-4 map + upsampled encoder
            # output, then a 3x3 smoothing conv.
            self.lateral_conv = nn.Sequential(nn.Conv2d(highres_channels, conv_dim, kernel_size=1),
                                              nn.GroupNorm(32, conv_dim))
            self.output_conv = nn.Sequential(
                nn.Conv2d(conv_dim, conv_dim, kernel_size=3, padding=1),
                nn.GroupNorm(32, conv_dim), nn.ReLU(inplace=True))
            xavier_uniform_(self.lateral_conv[0].weight, gain=1)
            constant_(self.lateral_conv[0].bias, 0)
            self.mask_up = nn.Identity()
        else:
            up: List[nn.Module] = []
            for _ in range(int(mask_upsample).bit_length() - 1):   # 1→0 steps, 2→1, 4→2, 8→3
                up += [nn.ConvTranspose2d(conv_dim, conv_dim, kernel_size=2, stride=2),
                       nn.GroupNorm(32, conv_dim), nn.GELU()]
            self.mask_up = nn.Sequential(*up) if up else nn.Identity()
            self.lateral_conv = self.output_conv = None

        self.mask_features = nn.Conv2d(conv_dim, mask_dim, kernel_size=1)
        xavier_uniform_(self.mask_features.weight, gain=1)
        constant_(self.mask_features.bias, 0)

    def _reset_encoder_parameters(self):
        for p in self.encoder.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
        for m in self.encoder.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    def forward(self, levels: List[Tensor],
                highres: Optional[Tensor] = None) -> Tuple[Tensor, List[Tensor]]:
        if len(levels) != self.num_native_levels:
            raise ValueError(f"expected {self.num_native_levels} backbone levels, got {len(levels)}")

        srcs, poss = [], []
        for lvl, proj in enumerate(self.input_proj):
            # native levels project their own input; synthesised ones stack on the previous output
            cur = proj(levels[lvl] if lvl < self.num_native_levels else srcs[-1])
            srcs.append(cur)
            poss.append(self.pe_layer(cur))

        if self.encoder is None:
            outs = srcs
        else:
            src_flatten, pos_flatten, shapes = [], [], []
            for lvl, (src, pos) in enumerate(zip(srcs, poss)):
                _, _, h, w = src.shape
                shapes.append((h, w))
                src_flatten.append(src.flatten(2).transpose(1, 2))
                pos_flatten.append(pos.flatten(2).transpose(1, 2)
                                   + self.level_embed[lvl].view(1, 1, -1))
            src_flatten = torch.cat(src_flatten, 1)
            pos_flatten = torch.cat(pos_flatten, 1)
            spatial_shapes = torch.as_tensor(shapes, dtype=torch.long, device=src_flatten.device)
            level_start_index = torch.cat(
                (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
            bs = src_flatten.shape[0]
            # Every image is a fixed square resize with no padding, so all levels are fully valid
            # (upstream reaches the same state via its `enable_mask == 0` shortcut).
            valid_ratios = torch.ones(bs, self.num_feature_levels, 2, device=src_flatten.device)
            memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios,
                                  pos_flatten, None)
            outs = []
            for lvl, (h, w) in enumerate(shapes):
                start = int(level_start_index[lvl])
                outs.append(memory[:, start:start + h * w].transpose(1, 2).view(bs, -1, h, w))

        if self.lateral_conv is not None:
            if highres is None:
                raise ValueError("this decoder was built with highres_channels but got highres=None")
            lat = self.lateral_conv(highres)
            y = lat + F.interpolate(outs[0], size=lat.shape[-2:], mode="bilinear",
                                    align_corners=False)
            mask_feat_in = self.output_conv(y)
        else:
            mask_feat_in = self.mask_up(outs[0])

        return self.mask_features(mask_feat_in), outs
