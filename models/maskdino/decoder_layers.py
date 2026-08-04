"""
The generic DINO decoder stack used by MaskDINO (docs/MASKDINO.md).

Port of MaskDINO's `maskdino/modeling/transformer_decoder/dino_decoder.py`. Nothing here is
MaskDINO-specific — it is DAB-DETR's decoder: sine-embedded anchor boxes as query positional
embeddings, deformable cross-attention into the multi-scale memory, and iterative per-layer
box refinement. `MaskDINODecoder` (decoder.py) supplies the box head and drives it.
"""

import torch
from torch import nn

from .anchor3d import project_anchors
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
        # --anchor_3d only (docs/MASKDINO.md §8.3): the Delta(xyz, log r) head, also set by
        # MaskDINODecoder. None keeps the 2D DAB path exactly as it was.
        self.anchor_embed = None
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
                frames_per_sample=1, num_shared_queries=None, anchor=None, token_xyz=None):
        """
        tgt: [nq, bs, d_model]; memory: [hw, bs, d_model]; refpoints_unsigmoid: [nq, bs, 4].
        `frames_per_sample > 1` means the batch is B bundles of S frames (frames contiguous) that
        share their queries; the per-layer `cross_frame` block then mixes the S copies of each of
        the trailing `num_shared_queries` queries (docs/MASKDINO.md §8).

        `anchor` [B, nq_shared, 4] switches on the 3D-anchor path (docs/MASKDINO.md §8.3): the
        trailing `num_shared_queries` queries take their per-layer reference from the anchor's
        projection into each view (`token_xyz` [bs, h*w, 3]) instead of from an independently
        refined 2D box, and are refined by Delta(xyz, log r). The leading rows — the denoising
        queries, which have no 3D anchor because their slot means a different GT instance in each
        frame — keep the 2D DAB path untouched.

        Returns (list of per-layer outputs [bs, nq, d], list of reference boxes [bs, nq, 4]).
        """
        output = tgt
        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]
        nq = tgt.shape[0]
        n_shared = nq if num_shared_queries is None else min(num_shared_queries, nq)
        n_dn = nq - n_shared          # the denoising rows, always at the FRONT

        for layer_id, layer in enumerate(self.layers):
            if anchor is not None:
                # The shared queries' reference is not a state that gets refined — it is
                # recomputed from the current 3D anchor every layer. `reference_points` is what
                # the layer consumes (DN rows detached, as before); `ref_points[-1]` is the
                # non-detached twin `pred_box` uses as the box base, and only its DN rows differ.
                ref_shared = project_anchors(anchor, token_xyz, frames_per_sample).transpose(0, 1)
                reference_points = torch.cat([reference_points[:n_dn], ref_shared], dim=0)
                ref_points[-1] = torch.cat([ref_points[-1][:n_dn], ref_shared], dim=0)

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

            # iterative 3D-anchor refinement. NOT detached, unlike the box above: the anchor has
            # no loss of its own, so a detached Delta(xyz, log r) head would receive exactly zero
            # gradient — the gradient reaches it only through the soft projection of the NEXT
            # layers' references. One anchor per bundle, so the S views' deltas are averaged
            # (the permutation-equivariant reduction, as in CrossFrameAttention).
            if anchor is not None and self.anchor_embed is not None:
                b, s = anchor.shape[0], frames_per_sample
                delta = self.anchor_embed[layer_id](output[n_dn:])          # [n_shared, bs, 4]
                anchor = anchor + delta.transpose(0, 1).reshape(b, s, n_shared, 4).mean(dim=1)

            intermediate.append(self.norm(output))

        return (
            [itm.transpose(0, 1) for itm in intermediate],
            [ref.transpose(0, 1) for ref in ref_points],
        )
