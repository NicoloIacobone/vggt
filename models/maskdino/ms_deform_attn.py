"""
Multi-scale deformable attention (Deformable DETR / MaskDINO), **pure PyTorch**.

MaskDINO ships this module with a fused CUDA extension (`MultiScaleDeformableAttention`) plus a
`grid_sample`-based reference path used for debugging. This repo has no compiled extension (and
the trial's tests must run on CPU), so the reference path is the only path here — see
docs/MASKDINO_TRIAL.md §2. At our sizes (≈1830 memory tokens, ≤400 queries, 8 heads, 4 points,
3 levels) the fused kernel would buy little: the whole op is three `grid_sample` calls.

Numerically identical to the CUDA op up to floating-point ordering; `tests/test_maskdino.py`
checks it against a naive explicit-loop implementation.
"""

import math
import warnings

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, xavier_uniform_


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    """
    Args:
        value: [N, sum(H*W), n_heads, head_dim]
        value_spatial_shapes: [n_levels, 2] (H, W) per level
        sampling_locations: [N, Lq, n_heads, n_levels, n_points, 2] in [0, 1] (x, y)
        attention_weights: [N, Lq, n_heads, n_levels, n_points], softmaxed over (level, point)
    Returns:
        [N, Lq, n_heads * head_dim]
    """
    N_, _, M_, D_ = value.shape
    _, Lq_, _, L_, P_, _ = sampling_locations.shape
    value_list = value.split([int(H_) * int(W_) for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        H_, W_ = int(H_), int(W_)
        # [N, H*W, M, D] -> [N*M, D, H, W]
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
        # [N, Lq, M, P, 2] -> [N*M, Lq, P, 2]
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # [N*M, D, Lq, P]
        sampling_value_list.append(
            F.grid_sample(value_l_, sampling_grid_l_, mode="bilinear",
                          padding_mode="zeros", align_corners=False)
        )
    # (N*M, 1, Lq, L*P)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1)
    return output.view(N_, M_ * D_, Lq_).transpose(1, 2).contiguous()


def _is_power_of_2(n):
    return isinstance(n, int) and n > 0 and (n & (n - 1)) == 0


class MSDeformAttn(nn.Module):
    """
    Multi-Scale Deformable Attention. Constructor/initialisation identical to Deformable DETR's
    (the sampling-offset bias is seeded with `n_heads` evenly-spaced directions at radii
    1..n_points, which is what makes the module trainable from scratch).
    """

    def __init__(self, d_model: int = 256, n_levels: int = 4, n_heads: int = 8, n_points: int = 4):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, got {d_model} and {n_heads}")
        head_dim = d_model // n_heads
        if not _is_power_of_2(head_dim):
            warnings.warn(
                "MSDeformAttn is most efficient with a power-of-2 head dim; "
                f"got d_model/n_heads = {head_dim}."
            )

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
        grid_init = grid_init.view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes,
                input_level_start_index, input_padding_mask=None):
        """
        Args:
            query: [N, Lq, C]
            reference_points: [N, Lq, n_levels, 2] (normalized xy) or [N, Lq, n_levels, 4]
                (normalized cxcywh — the box then modulates the sampling radius, DAB-style)
            input_flatten: [N, sum(H*W), C] flattened multi-scale memory
            input_spatial_shapes: [n_levels, 2]
            input_level_start_index: [n_levels]
            input_padding_mask: [N, sum(H*W)] True where padded
        Returns:
            [N, Lq, C]
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert int((input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum()) == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)

        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = (reference_points[:, :, None, :, None, :]
                                  + sampling_offsets / offset_normalizer[None, None, None, :, None, :])
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, got {reference_points.shape[-1]}.")

        output = ms_deform_attn_core_pytorch(
            value, input_spatial_shapes, sampling_locations, attention_weights)
        return self.output_proj(output)
