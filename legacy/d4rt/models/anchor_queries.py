# Arm E — 3D-anchored queries (docs/todo.md, docs/RELATED_WORK.md gap #1).
#
# Builds decoder queries from VGGT's OWN predicted pointmap geometry instead of
# image-space (u, v) prompts (arm A) or pure learned embeddings (arm C):
#
#   1. Each VGGT patch token gets a 3D position: the confidence-weighted mean of its
#      14x14 pixels' point-head predictions (world_points).
#   2. Anchors = farthest-point sampling over the confidence-filtered, per-scene
#      normalized token positions -> K well-spread 3D scene locations. The 3D spread
#      is the built-in duplicate suppressor (two anchors on one object are
#      geometrically penalized), aimed at the known over-prediction failure.
#   3. Each anchor pools the features of its k nearest patch tokens in 3D; nearby
#      tokens come from *different views* of the same surface, so the pooled feature
#      is multi-view by construction. No view embedding anywhere: one query per 3D
#      location, shared across all views.
#
# Everything here is frozen-backbone preprocessing: it runs once per bundle at
# caching time (train_multiscene.build_bundle runs the frozen point head on the
# aggregator output it already has) and produces small tensors that ride along in
# the cached bundle ({"xyz": [1, K, 3], "feats": [1, K, C]} — a few hundred KB, vs
# ~26 MB for the full pointmap, which is never stored). The trainable encoding
# lives in legacy/d4rt/models/d4rt_decoder.py::QueryGenerator (query_mode="anchor3d").

import torch
from typing import Dict, Tuple


def patch_token_positions(
    world_points: torch.Tensor,
    conf: torch.Tensor,
    patch_size: int = 14,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    3D position per VGGT patch token: confidence-weighted mean of the point-head
    predictions over the token's patch_size x patch_size pixel cell.

    Args:
        world_points: [1, S, H, W, 3] point-head output (H, W divisible by patch_size;
            518 = 37 * 14 for the standard pipeline)
        conf: [1, S, H, W] point-head confidence (>= 0)
        patch_size: VGGT patch size (14)

    Returns:
        positions: [S * hp * wp, 3] token positions, frame-major then row-major —
            the same token order as the aggregator's patch tokens per frame
        weights: [S * hp * wp] mean confidence per token (for validity filtering)
    """
    B, S, H, W, _ = world_points.shape
    assert B == 1, f"expected batch size 1, got {B}"
    hp, wp = H // patch_size, W // patch_size
    assert hp * patch_size == H and wp * patch_size == W, (
        f"H, W ({H}, {W}) must be divisible by patch_size ({patch_size})")

    # [S, hp, ps, wp, ps, 3] -> [S, hp, wp, ps*ps, 3]
    pts = world_points[0].reshape(S, hp, patch_size, wp, patch_size, 3)
    pts = pts.permute(0, 1, 3, 2, 4, 5).reshape(S, hp, wp, patch_size * patch_size, 3)
    w = conf[0].reshape(S, hp, patch_size, wp, patch_size)
    w = w.permute(0, 1, 3, 2, 4).reshape(S, hp, wp, patch_size * patch_size)
    w = w.clamp_min(0)

    denom = w.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    positions = (pts * w.unsqueeze(-1)).sum(dim=-2) / denom  # [S, hp, wp, 3]
    return positions.reshape(-1, 3), w.mean(dim=-1).reshape(-1)


def farthest_point_sample(points: torch.Tensor, k: int) -> torch.Tensor:
    """
    Deterministic farthest-point sampling: start from the point closest to the
    centroid, then greedily add the point farthest from the selected set.

    Args:
        points: [M, 3]
        k: number of samples (clamped to M)

    Returns:
        indices: [min(k, M)] LongTensor into `points`
    """
    M = points.shape[0]
    k = min(k, M)
    if k <= 0:
        return torch.zeros(0, dtype=torch.long, device=points.device)

    centroid = points.mean(dim=0, keepdim=True)
    first = torch.cdist(points, centroid).squeeze(1).argmin()
    selected = torch.empty(k, dtype=torch.long, device=points.device)
    selected[0] = first
    min_dist = (points - points[first]).pow(2).sum(dim=1)  # squared dists are order-preserving
    for i in range(1, k):
        nxt = min_dist.argmax()
        selected[i] = nxt
        min_dist = torch.minimum(min_dist, (points - points[nxt]).pow(2).sum(dim=1))
    return selected


def build_anchors(
    features: torch.Tensor,
    patch_start_idx: int,
    world_points: torch.Tensor,
    conf: torch.Tensor,
    num_anchors: int,
    knn: int = 8,
    conf_quantile: float = 0.3,
    patch_size: int = 14,
) -> Dict[str, torch.Tensor]:
    """
    Build the per-bundle anchor dict consumed by QueryGenerator(query_mode="anchor3d").

    Args:
        features: [1, S, P, C] aggregator output (the cached backbone features)
        patch_start_idx: index where patch tokens start in P
        world_points: [1, S, H, W, 3] frozen point-head prediction
        conf: [1, S, H, W] point-head confidence
        num_anchors: K, the number of anchor queries (padded by cycling if fewer
            valid tokens exist, so the output size is always exactly K)
        knn: patch-token neighbors pooled into each anchor's content feature
        conf_quantile: tokens below this confidence quantile are dropped before
            sampling (low-confidence pointmap regions give unreliable 3D positions);
            relaxed automatically if it would leave fewer than num_anchors tokens
        patch_size: VGGT patch size (14)

    Returns:
        {"xyz": [1, K, 3] per-scene normalized (zero-mean, unit-RMS) anchor positions,
         "feats": [1, K, C] mean feature of each anchor's knn nearest tokens in 3D}
    """
    tok_pos, tok_conf = patch_token_positions(world_points, conf, patch_size)  # [T,3],[T]
    tok_feats = features[0, :, patch_start_idx:, :].reshape(-1, features.shape[-1])  # [T, C]
    assert tok_feats.shape[0] == tok_pos.shape[0], (
        f"patch-token count mismatch: features give {tok_feats.shape[0]}, "
        f"pointmap gives {tok_pos.shape[0]} — check patch_start_idx / resolutions")

    # Confidence filter (with fallback: never filter below num_anchors tokens).
    valid = tok_conf >= torch.quantile(tok_conf, conf_quantile)
    if int(valid.sum()) < max(num_anchors, 1):
        valid = torch.ones_like(valid, dtype=torch.bool)
    pos_v, feats_v = tok_pos[valid], tok_feats[valid]

    # Per-scene normalization (zero-mean, unit RMS radius): VGGT pointmap magnitudes
    # vary scene to scene, and an un-normalized Fourier encoding would alias/saturate —
    # same class of failure as the un-normalized memory in InstanceDecoder.
    center = pos_v.mean(dim=0, keepdim=True)
    scale = (pos_v - center).pow(2).sum(dim=1).mean().sqrt().clamp_min(1e-6)
    pos_n = (pos_v - center) / scale

    idx = farthest_point_sample(pos_n, num_anchors)
    if idx.shape[0] < num_anchors:  # tiny scenes: cycle so output is always exactly K
        reps = -(-num_anchors // idx.shape[0])
        idx = idx.repeat(reps)[:num_anchors]
    anchor_xyz = pos_n[idx]  # [K, 3]

    # kNN content pooling in normalized 3D space (float32 cdist for numerical safety).
    k_eff = min(knn, pos_n.shape[0])
    d = torch.cdist(anchor_xyz.float(), pos_n.float())  # [K, V]
    nn_idx = d.topk(k_eff, largest=False).indices       # [K, k_eff]
    anchor_feats = feats_v[nn_idx].mean(dim=1)          # [K, C]

    return {"xyz": anchor_xyz.unsqueeze(0), "feats": anchor_feats.unsqueeze(0)}


def jitter_anchors(anchors: Dict[str, torch.Tensor], std: float) -> Dict[str, torch.Tensor]:
    """
    Training-time augmentation (analog of --query_jitter): Gaussian jitter on the
    normalized anchor positions. Content features are left untouched — the jitter
    perturbs where the query *says* it is, not what it pooled.
    """
    if std <= 0:
        return anchors
    return {"xyz": anchors["xyz"] + torch.randn_like(anchors["xyz"]) * std,
            "feats": anchors["feats"]}
