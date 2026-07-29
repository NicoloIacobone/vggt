"""
The generic DINO decoder stack used by MaskDINO (docs/MASKDINO.md).

Port of MaskDINO's `maskdino/modeling/transformer_decoder/dino_decoder.py`. Nothing here is
MaskDINO-specific — it is DAB-DETR's decoder: sine-embedded anchor boxes as query positional
embeddings, deformable cross-attention into the multi-scale memory, and iterative per-layer
box refinement. `MaskDINODecoder` (decoder.py) supplies the box head and drives it.
"""

import torch
from torch import nn

from .ms_deform_attn import MSDeformAttn
from .multiframe import CrossFrameAttention
from .utils import (MLP, _get_activation_fn, _get_clones, gen_sineembed_for_position,
                    inverse_sigmoid)


class DeformableTransformerDecoderLayer(nn.Module):
    """Self-attention (with query positional embedding) → deformable cross-attention → FFN."""

    def __init__(self, d_model=256, d_ffn=1024, dropout=0.0, activation="relu",
                 n_levels=3, n_heads=8, n_points=4):
        super().__init__()
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        return self.norm3(tgt + self.dropout4(tgt2))

    def forward(self, tgt, tgt_query_pos=None, tgt_reference_points=None,
                memory=None, memory_key_padding_mask=None, memory_level_start_index=None,
                memory_spatial_shapes=None, self_attn_mask=None):
        # self attention among queries (DN queries are isolated by `self_attn_mask`)
        q = k = self.with_pos_embed(tgt, tgt_query_pos)
        tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        # deformable cross attention around each query's anchor box
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
            tgt_reference_points.transpose(0, 1).contiguous(),
            memory.transpose(0, 1), memory_spatial_shapes, memory_level_start_index,
            memory_key_padding_mask,
        ).transpose(0, 1)
        tgt = self.norm1(tgt + self.dropout1(tgt2))

        return self.forward_ffn(tgt)


class TransformerDecoder(nn.Module):
    """DINO decoder: sine-embedded anchor boxes as query positions + per-layer box refinement."""

    def __init__(self, decoder_layer, num_layers, norm=None, d_model=256, query_dim=4,
                 num_feature_levels=3, nheads=8, dropout=0.0, cross_frame_attn=False):
        super().__init__()
        self.layers = (_get_clones(decoder_layer, num_layers) if num_layers > 0
                       else nn.ModuleList())
        self.num_layers = num_layers
        # Multi-frame only (docs/MASKDINO.md §8): one block per layer, tying the S copies of a
        # shared query together. Built only when asked for, so single-frame checkpoints keep
        # exactly the parameter set they had.
        self.cross_frame = (_get_clones(CrossFrameAttention(d_model, nheads, dropout), num_layers)
                            if cross_frame_attn and num_layers > 0 else None)
        self.norm = norm
        self.query_dim = query_dim
        assert query_dim in (2, 4)
        self.num_feature_levels = num_feature_levels
        self.d_model = d_model
        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        self.bbox_embed = None  # set by MaskDINODecoder (shared box head, iterative refinement)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    def forward(self, tgt, memory, memory_key_padding_mask=None, refpoints_unsigmoid=None,
                level_start_index=None, spatial_shapes=None, valid_ratios=None, tgt_mask=None,
                frames_per_sample=1, num_shared_queries=None):
        """
        tgt: [nq, bs, d_model]; memory: [hw, bs, d_model]; refpoints_unsigmoid: [nq, bs, 4].
        `frames_per_sample > 1` means the batch is B bundles of S frames (frames contiguous) that
        share their queries; the per-layer `cross_frame` block then mixes the S copies of each of
        the trailing `num_shared_queries` queries (docs/MASKDINO.md §8).
        Returns (list of per-layer outputs [bs, nq, d], list of reference boxes [bs, nq, 4]).
        """
        output = tgt
        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]

        for layer_id, layer in enumerate(self.layers):
            reference_points_input = (reference_points[:, :, None]
                                      * torch.cat([valid_ratios, valid_ratios], -1)[None, :])
            query_sine_embed = gen_sineembed_for_position(
                reference_points_input[:, :, 0, :], d_model=self.d_model)
            query_pos = self.ref_point_head(query_sine_embed)

            output = layer(
                tgt=output,
                tgt_query_pos=query_pos,
                tgt_reference_points=reference_points_input,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_level_start_index=level_start_index,
                memory_spatial_shapes=spatial_shapes,
                self_attn_mask=tgt_mask,
            )

            # multi-frame: give the S copies of each shared query a look at each other before
            # this layer's box refinement, so the anchors they refine stay one instance
            if self.cross_frame is not None and frames_per_sample > 1:
                output = self.cross_frame[layer_id](output, frames_per_sample, num_shared_queries)

            # iterative box refinement (detached, DINO-style "look forward once" off)
            if self.bbox_embed is not None:
                delta_unsig = self.bbox_embed[layer_id](output)
                new_reference_points = (delta_unsig + inverse_sigmoid(reference_points)).sigmoid()
                reference_points = new_reference_points.detach()
                ref_points.append(new_reference_points)

            intermediate.append(self.norm(output))

        return (
            [itm.transpose(0, 1) for itm in intermediate],
            [ref.transpose(0, 1) for ref in ref_points],
        )
