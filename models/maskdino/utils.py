"""
Small building blocks for the MaskDINO trial (docs/MASKDINO_TRIAL.md).

Everything here is a port of code MaskDINO pulls in from other packages, so that the trial has
NO dependency on detectron2 / fvcore (neither is installed in this repo's `myenv`):

  - `MLP`, `inverse_sigmoid`, `gen_sineembed_for_position`, `_get_clones`,
    `_get_activation_fn`, `gen_encoder_output_proposals` → MaskDINO `utils/utils.py`
  - `PositionEmbeddingSine` → MaskDINO `modeling/pixel_decoder/position_encoding.py`
  - `point_sample`, `get_uncertain_point_coords_with_randomness`, `calculate_uncertainty`
    → detectron2 PointRend (`projects/point_rend/point_features.py`), used by both the
    matcher and the mask loss.
"""

import copy
import math
from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MLP(nn.Module):
    """Plain FFN with ReLU between layers (DETR's `MLP`)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def inverse_sigmoid(x: Tensor, eps: float = 1e-5) -> Tensor:
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def _get_activation_fn(activation: str):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def _get_clones(module: nn.Module, n: int, layer_share: bool = False) -> nn.ModuleList:
    if layer_share:
        return nn.ModuleList([module for _ in range(n)])
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def gen_sineembed_for_position(pos_tensor: Tensor, d_model: int = 256) -> Tensor:
    """
    DAB-DETR positional sine embedding of a 2-d point or 4-d box.

    Args:
        pos_tensor: [nq, bs, 2] (x, y) or [nq, bs, 4] (cx, cy, w, h), all in [0, 1].
    Returns:
        [nq, bs, d_model] for 2-d input, [nq, bs, 2*d_model] for 4-d input.
    """
    half = d_model // 2
    scale = 2 * math.pi
    dim_t = torch.arange(half, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / half)

    def embed(v):
        p = v[:, :, None] * scale / dim_t
        return torch.stack((p[:, :, 0::2].sin(), p[:, :, 1::2].cos()), dim=3).flatten(2)

    pos_x = embed(pos_tensor[:, :, 0])
    pos_y = embed(pos_tensor[:, :, 1])
    if pos_tensor.size(-1) == 2:
        return torch.cat((pos_y, pos_x), dim=2)
    if pos_tensor.size(-1) == 4:
        pos_w = embed(pos_tensor[:, :, 2])
        pos_h = embed(pos_tensor[:, :, 3])
        return torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    raise ValueError(f"Unknown pos_tensor shape(-1): {pos_tensor.size(-1)}")


class PositionEmbeddingSine(nn.Module):
    """2-d sine position embedding over a feature map (DETR / MaskDINO)."""

    def __init__(self, num_pos_feats: int = 64, temperature: int = 10000,
                 normalize: bool = False, scale: float = None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and not normalize:
            raise ValueError("normalize should be True if scale is passed")
        self.scale = 2 * math.pi if scale is None else scale

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


def gen_encoder_output_proposals(memory: Tensor, memory_padding_mask: Tensor,
                                 spatial_shapes: Tensor):
    """
    Two-stage query selection input: turn every encoder token into an (unsigmoided) box proposal
    centred on its own cell, with a level-dependent size. Ported verbatim from MaskDINO.

    Returns (output_memory [bs, sum(hw), C], output_proposals [bs, sum(hw), 4] unsigmoided).
    """
    N_, S_, C_ = memory.shape
    proposals = []
    _cur = 0
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        H_, W_ = int(H_), int(W_)
        mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
        valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
        valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device),
            indexing="ij",
        )
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

        scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
        grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
        wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
        proposals.append(torch.cat((grid, wh), -1).view(N_, -1, 4))
        _cur += H_ * W_

    output_proposals = torch.cat(proposals, 1)
    output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
    output_proposals = torch.log(output_proposals / (1 - output_proposals))
    output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float("inf"))
    output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))

    output_memory = memory
    output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
    output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
    return output_memory, output_proposals


# --------------------------------------------------------------------------------------------
# PointRend point sampling (detectron2 `projects/point_rend/point_features.py`), used by the
# mask loss and the matcher's mask cost.
# --------------------------------------------------------------------------------------------

def point_sample(inputs: Tensor, point_coords: Tensor, **kwargs) -> Tensor:
    """
    Bilinearly sample `inputs` [N, C, H, W] at normalized [0,1] `point_coords` [N, P, 2].
    Returns [N, C, P].
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(inputs, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


def calculate_uncertainty(logits: Tensor) -> Tensor:
    """Uncertainty = -|logit| (points near the 0.5 decision boundary are the most uncertain)."""
    assert logits.shape[1] == 1
    return -torch.abs(logits.clone())


def get_uncertain_point_coords_with_randomness(
    coarse_logits: Tensor,
    uncertainty_func,
    num_points: int,
    oversample_ratio: float,
    importance_sample_ratio: float,
) -> Tensor:
    """
    Sample `num_points` points, biased towards uncertain (boundary) locations:
    oversample `num_points * oversample_ratio` uniformly, keep the most uncertain
    `num_points * importance_sample_ratio` of them, fill the rest uniformly at random.
    Returns [N, num_points, 2] in [0, 1].
    """
    assert oversample_ratio >= 1
    assert 0 <= importance_sample_ratio <= 1
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    # Uncertainty must be computed on the SAMPLED logits, not by interpolating an uncertainty
    # map (interpolating then measuring would make points near the boundary look certain).
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(num_boxes, num_uncertain_points, 2)
    if num_random_points > 0:
        point_coords = torch.cat(
            [point_coords, torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device)],
            dim=1,
        )
    return point_coords


def cat_matched(targets: List[dict], indices, key: str) -> Tensor:
    """
    Concatenate the matched GT entries of `key` across the batch, in matcher order.

    MaskDINO pads the per-image GT into a nested tensor and then index-selects with
    (batch_idx, tgt_idx); every image in this trial shares one fixed mask grid, so gathering
    per sample and concatenating is equivalent and avoids the padding machinery entirely.
    """
    parts = [t[key][j] for t, (_, j) in zip(targets, indices) if len(j) > 0]
    if not parts:
        ref = targets[0][key]
        return ref.new_zeros((0,) + tuple(ref.shape[1:]))
    return torch.cat(parts, dim=0)
