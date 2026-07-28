"""
Pixel decoder for the MaskDINO trial: VGGT tokens → feature pyramid → MSDeformAttn encoder.

MaskDINO's `MaskDINOEncoder` consumes a CNN backbone's res2..res5 pyramid. VGGT's aggregator is
a plain ViT-style stack: one token resolution (37×37 patches at 518 px input), so the pyramid is
synthesised **ViTDet-style** (a "simple feature pyramid" from the last block — shown to match FPN
for plain-ViT detectors) before the deformable encoder runs. Full rationale + diagram:
docs/MASKDINO_TRIAL.md §3.

Everything downstream (the encoder, the two-stage proposal generation, the decoder) is the
upstream MaskDINO code path.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.init import constant_, normal_, xavier_uniform_

from .ms_deform_attn import MSDeformAttn
from .utils import PositionEmbeddingSine, _get_activation_fn, _get_clones


class MSDeformAttnEncoderLayer(nn.Module):
    """Deformable self-attention + FFN (MaskDINO `MSDeformAttnTransformerEncoderLayer`)."""

    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="relu",
                 n_levels=3, n_heads=8, n_points=4):
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        return self.norm2(src + self.dropout3(src2))

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index,
                padding_mask=None):
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src,
                              spatial_shapes, level_start_index, padding_mask)
        src = self.norm1(src + self.dropout1(src2))
        return self.forward_ffn(src)


class MSDeformAttnEncoder(nn.Module):
    """Stack of `MSDeformAttnEncoderLayer` with per-level reference points."""

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            H_, W_ = int(H_), int(W_)
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            reference_points_list.append(torch.stack((ref_x, ref_y), -1))
        reference_points = torch.cat(reference_points_list, 1)
        return reference_points[:, :, None] * valid_ratios[:, None]

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None,
                padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for layer in self.layers:
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index,
                           padding_mask)
        return output


class VGGTPixelDecoder(nn.Module):
    """
    VGGT aggregator tokens → (mask_features, multi-scale memory) for `MaskDINODecoder`.

    Args:
        memory_dim: channel width of the VGGT tokens (2048 for VGGT-1B's aggregator output;
            `len(feature_layers) * 2048` when several layers are concatenated upstream).
        conv_dim: transformer width inside the pixel decoder (MaskDINO: 256).
        mask_dim: channel width of `mask_features` (MaskDINO: 256).
        num_feature_levels: how many pyramid levels the encoder sees (MaskDINO: 3).
            Level 0 is the native token grid; each extra level is a stride-2 conv on the
            previous one.
        enc_layers: MSDeformAttn encoder layers (MaskDINO: 6). 0 disables the encoder, which
            turns the module into a plain projection — useful as an ablation.
        mask_upsample: 1 (default) keeps `mask_features` on the 37×37 patch grid, so the mask
            metrics are computed on the same grid as the D4RT arms; 2/4 add transposed-conv
            upsampling steps (GT must be built at the matching resolution).

    Forward returns:
        mask_features: [B, mask_dim, h*u, w*u]
        multi_scale:   list of [B, conv_dim, H_l, W_l], ordered HIGH→LOW resolution.
    """

    def __init__(
        self,
        memory_dim: int = 2048,
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
        if num_feature_levels < 1:
            raise ValueError("num_feature_levels must be >= 1")
        if mask_upsample not in (1, 2, 4):
            raise ValueError(f"mask_upsample must be 1, 2 or 4, got {mask_upsample}")

        self.memory_dim = memory_dim
        self.conv_dim = conv_dim
        self.mask_dim = mask_dim
        self.num_feature_levels = num_feature_levels
        self.mask_upsample = mask_upsample

        # Level 0: 1x1 projection of the token grid. Extra levels: stride-2 convs (ViTDet).
        proj = [nn.Sequential(nn.Conv2d(memory_dim, conv_dim, kernel_size=1),
                              nn.GroupNorm(32, conv_dim))]
        for _ in range(num_feature_levels - 1):
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

        # mask_features: (optionally upsampled) level-0 output → 1x1 conv to mask_dim.
        up_layers: List[nn.Module] = []
        c = conv_dim
        for _ in range(int(mask_upsample).bit_length() - 1):  # 1→0 steps, 2→1, 4→2
            up_layers += [nn.ConvTranspose2d(c, conv_dim, kernel_size=2, stride=2),
                          nn.GroupNorm(32, conv_dim), nn.GELU()]
            c = conv_dim
        self.mask_up = nn.Sequential(*up_layers) if up_layers else nn.Identity()
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

    @staticmethod
    def tokens_to_map(tokens: Tensor, patch_start_idx: int) -> Tensor:
        """
        [B, P, C] VGGT tokens (P = special tokens + h*w patch tokens) → [B, C, h, w].

        The camera + register tokens carry no spatial position, so they are dropped here (they
        are re-injected nowhere: MaskDINO's decoder gets all its global context through the
        deformable encoder).
        """
        patches = tokens[:, patch_start_idx:, :]
        b, n, c = patches.shape
        h = int(round(n ** 0.5))
        if h * h != n:
            raise ValueError(f"patch tokens ({n}) do not form a square grid")
        return patches.transpose(1, 2).reshape(b, c, h, h)

    def forward(self, tokens: Tensor, patch_start_idx: int = 5) -> Tuple[Tensor, List[Tensor]]:
        x = self.tokens_to_map(tokens, patch_start_idx) if tokens.dim() == 3 else tokens

        srcs, poss = [], []
        cur = x
        for lvl, proj in enumerate(self.input_proj):
            cur = proj(cur)
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
                pos_flatten.append(pos.flatten(2).transpose(1, 2) + self.level_embed[lvl].view(1, 1, -1))
            src_flatten = torch.cat(src_flatten, 1)
            pos_flatten = torch.cat(pos_flatten, 1)
            spatial_shapes = torch.as_tensor(shapes, dtype=torch.long, device=src_flatten.device)
            level_start_index = torch.cat(
                (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
            bs = src_flatten.shape[0]
            # No padding in this pipeline: every frame is a fixed 518x518 square resize, so all
            # levels are fully valid (MaskDINO reaches the same state via its `enable_mask == 0`
            # shortcut). valid_ratios are therefore all ones.
            valid_ratios = torch.ones(bs, self.num_feature_levels, 2, device=src_flatten.device)

            memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios,
                                  pos_flatten, None)
            outs = []
            for lvl, (h, w) in enumerate(shapes):
                start = int(level_start_index[lvl])
                outs.append(memory[:, start:start + h * w].transpose(1, 2).view(bs, -1, h, w))

        mask_features = self.mask_features(self.mask_up(outs[0]))
        return mask_features, outs
