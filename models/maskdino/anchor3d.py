"""
3D anchors instead of 2D DAB boxes (docs/MASKDINO.md §8.3, docs/todo.md 2d).

An **ablation**, not a contribution: FAST3DIS (arXiv 2603.25993) already publishes 3D-anchored
queries. What is ours is the controlled comparison — 3D anchors vs 2D DAB boxes inside the same
DINO-family decoder, same frozen backbone, same data, same protocol.

The mechanism, in one paragraph. Every VGGT patch token has a cached 3D position (the
confidence-weighted mean of its 14x14 pixels' point-head predictions, built once per bundle in
`train/maskdino_data.py::patch_token_positions`). A query's positional prior is then a single
**3D anchor** `(x, y, z, log r)` shared by all S views of a bundle instead of a per-view 4-d box.
To use it where the decoder needs a 2D reference — the DAB query positional embedding and the
deformable sampling locations — the anchor is projected into each view as a **soft nearest
patch**:

    w = softmax(-||p_patch - a||^2 / r^2)   over that view's 37x37 grid
    (cx, cy) = sum_p w_p * (u, v)_p
    (w,  h)  = 2 * sqrt(Var_w[(u, v)] + (0.5/37)^2)

so the reference *size* falls out of the same distribution as its centre (this is §8.3's
"anchor = a 3D point (+ scale)", realised as the softmax temperature), floored at one patch so a
collapsed anchor cannot degenerate the deformable sampling to a single point. No intrinsics or
extrinsics are needed — unlike FAST3DIS, which projects with its predicted camera — and the
whole thing is differentiable in the anchor, which is what lets the per-layer Delta(xyz, log r)
refinement learn at all (see `decoder_layers.py`: unlike the DAB box, the anchor has no loss of
its own, so its refinement cannot be detached).
"""

import math
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

# Initial anchor radius in the per-bundle normalised frame (which has unit RMS radius by
# construction, so 0.25 is "a quarter of the scene's spread" — roughly object scale).
ANCHOR_LOG_R0 = math.log(0.25)


def normalize_token_xyz(xyz: Tensor, conf: Optional[Tensor] = None, eps: float = 1e-6) -> Tensor:
    """
    Zero-mean, unit-RMS-radius normalisation of a bundle's patch-token positions.

    VGGT's pointmap magnitudes vary scene to scene, and the softmax temperature `r` is a single
    learned scalar per query — it can only mean something if the coordinate frame is comparable
    across bundles. The centre and scale are estimated on the tokens whose point-head confidence
    is at or above the bundle median, because the unreliable tail of the pointmap is exactly what
    would otherwise drag them (this is the only place confidence is used besides the intra-patch
    pooling itself; it is deliberately NOT a bias inside the softmax).

    Args:
        xyz:  [..., 3] positions of one bundle (all S frames together).
        conf: matching [...] confidences, or None to use every token.
    Returns:
        the same shape, normalised.
    """
    flat = xyz.reshape(-1, 3).float()
    sel = flat
    if conf is not None:
        c = conf.reshape(-1).float()
        keep = c >= c.median()
        if int(keep.sum()) >= 8:
            sel = flat[keep]
    center = sel.mean(dim=0)
    scale = (sel - center).pow(2).sum(dim=-1).mean().sqrt().clamp_min(eps)
    return ((xyz.float() - center) / scale).to(xyz.dtype)


def pyramid_token_xyz(token_xyz: Tensor, spatial_shapes: Tensor) -> Tensor:
    """
    Give every memory token of the pixel decoder's pyramid a 3D position.

    The decoder's two-stage selection picks its top-k proposals out of the *flattened, multi-level*
    encoder memory, so the 3D anchor gathered for a selected proposal has to be indexable with the
    very same index. Levels 1..L-1 are stride-2 convolutions of level 0, so their positions are
    the level-0 grid resampled to each level's resolution — nearest, never averaged, so a cell
    straddling a depth discontinuity keeps a real surface position instead of a point floating
    between two surfaces.

    Args:
        token_xyz:      [bs, h0*w0, 3] positions on the native token grid (level 0).
        spatial_shapes: [L, 2] the pyramid's (h, w) per level, level 0 first.
    Returns:
        [bs, sum_l h_l*w_l, 3] concatenated level-major, matching the decoder's `src_flatten`.
    """
    bs, hw, _ = token_xyz.shape
    h0, w0 = int(spatial_shapes[0][0]), int(spatial_shapes[0][1])
    assert h0 * w0 == hw, f"token_xyz has {hw} tokens, level 0 is {h0}x{w0}"
    grid = token_xyz.transpose(1, 2).reshape(bs, 3, h0, w0)
    out: List[Tensor] = [token_xyz]
    for lvl in range(1, spatial_shapes.shape[0]):
        h, w = int(spatial_shapes[lvl][0]), int(spatial_shapes[lvl][1])
        res = F.interpolate(grid, size=(h, w), mode="nearest")
        out.append(res.flatten(2).transpose(1, 2))
    return torch.cat(out, dim=1)


def uv_grid(grid: int, device, dtype) -> Tensor:
    """[grid*grid, 2] patch-centre (u, v) in [0, 1], row-major — the token flattening order."""
    c = (torch.arange(grid, device=device, dtype=dtype) + 0.5) / grid
    v, u = torch.meshgrid(c, c, indexing="ij")
    return torch.stack((u.reshape(-1), v.reshape(-1)), dim=-1)


def project_anchors(anchor: Tensor, token_xyz: Tensor, frames_per_sample: int,
                    min_radius: float = 1e-2, max_radius: float = 1e2) -> Tensor:
    """
    Soft-nearest-patch projection of one 3D anchor per query into each of the bundle's views.

    Args:
        anchor:    [B, Q, 4] — (x, y, z, log r), ONE anchor per query per bundle.
        token_xyz: [B*S, h*w, 3] per-frame patch positions, frames of a bundle contiguous.
        frames_per_sample: S.
    Returns:
        [B*S, Q, 4] reference boxes (cx, cy, w, h) in (0, 1) — the same layout the DAB machinery
        already consumes, so `gen_sineembed_for_position`, `MSDeformAttn` and `pred_box` are all
        untouched.
    """
    bs, hw, _ = token_xyz.shape
    s = int(frames_per_sample)
    assert bs % s == 0, f"token_xyz batch {bs} is not a multiple of frames_per_sample {s}"
    assert anchor.shape[0] == bs // s, (
        f"anchor batch {anchor.shape[0]} != {bs // s} bundles")
    grid = int(round(hw ** 0.5))
    assert grid * grid == hw, f"patch tokens ({hw}) do not form a square grid"

    a = anchor.repeat_interleave(s, dim=0).float()               # [bs, Q, 4]
    pts = token_xyz.float()
    xyz = a[..., :3]
    r = a[..., 3].exp().clamp(min_radius, max_radius)[..., None]  # [bs, Q, 1]

    # squared distances written out rather than via cdist: cdist's gradient is undefined at
    # distance 0, which is exactly where an anchor sitting on a patch lands.
    d2 = (xyz.pow(2).sum(-1)[..., None]
          + pts.pow(2).sum(-1)[:, None, :]
          - 2.0 * torch.bmm(xyz, pts.transpose(1, 2))).clamp_min(0)
    w = torch.softmax(-d2 / (r * r), dim=-1)                      # [bs, Q, hw]

    uv = uv_grid(grid, token_xyz.device, w.dtype)                 # [hw, 2]
    mean = torch.matmul(w, uv)                                    # [bs, Q, 2]
    var = torch.matmul(w, uv.pow(2)) - mean.pow(2)
    floor = 0.5 / grid                                            # half a patch
    size = 2.0 * (var.clamp_min(0) + floor * floor).sqrt()
    return torch.cat([mean, size], dim=-1).clamp(1e-4, 1.0 - 1e-4)
