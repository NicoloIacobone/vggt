"""
Geometry for the 3D benchmark eval (docs/MASKDINO.md §9): registration of VGGT's
scale-ambiguous predictions to the GT mesh frame, and the SegVGGT-style vote lifting of
per-view 2D masks onto mesh vertices.

Registration is EVAL-ONLY machinery: inference uses VGGT's own predicted depth + cameras
and never sees GT geometry; the Sim(3) here merely expresses the finished prediction in
the benchmark mesh's coordinate frame so it can be scored — the same convention as
FAST3DIS ("Sim(3) + ICP alignment"). It is solved from predicted-vs-GT camera centers
(closed-form Umeyama), optionally refined by a similarity ICP against the mesh vertices.

Everything here is numpy + scipy (cKDTree) and CPU-testable.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from scipy.spatial import cKDTree

from train.perframe import upsample_mask_logits


# ------------------------------------------------------------------------------------------
# Sim(3) registration
# ------------------------------------------------------------------------------------------

def umeyama_sim3(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Closed-form similarity aligning src to dst (Umeyama 1991): returns (s, R, t) with
    dst ≈ s * R @ src + t. Needs >= 3 non-degenerate correspondences [N, 3].
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3 or src.shape[0] < 3:
        raise ValueError(f"need matching [N>=3, 3] point sets, got {src.shape} / {dst.shape}")
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    x, y = src - mu_src, dst - mu_dst
    cov = y.T @ x / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    var_src = (x ** 2).sum() / len(src)
    if var_src < 1e-12:
        raise ValueError("degenerate source points (zero variance)")
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / var_src)
    t = mu_dst - s * R @ mu_src
    return s, R, t


def apply_sim3(points: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """dst = s * R @ p + t for points [..., 3]."""
    return s * np.asarray(points) @ R.T + t


def icp_refine_sim3(src: np.ndarray, dst: np.ndarray, s: float, R: np.ndarray,
                    t: np.ndarray, iters: int = 10, max_dist: float = 0.25,
                    max_src_points: int = 20000, seed: int = 0
                    ) -> Tuple[float, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Similarity ICP: refine an initial (s, R, t) by alternating nearest-neighbour
    correspondences (within max_dist, in dst units) with a full Umeyama re-fit — so the
    scale keeps being re-estimated, unlike rigid ICP. src is subsampled to at most
    `max_src_points`. Returns the refined transform + {"inliers", "rms"} of the last
    iteration; if any iteration finds < 3 correspondences, returns the transform so far.
    """
    rng = np.random.default_rng(seed)
    src = np.asarray(src, dtype=np.float64)
    if len(src) > max_src_points:
        src = src[rng.choice(len(src), max_src_points, replace=False)]
    tree = cKDTree(np.asarray(dst, dtype=np.float64))
    stats = {"inliers": 0.0, "rms": float("nan")}
    for _ in range(iters):
        moved = apply_sim3(src, s, R, t)
        dists, idx = tree.query(moved, k=1, distance_upper_bound=max_dist)
        keep = np.isfinite(dists)
        if keep.sum() < 3:
            break
        s, R, t = umeyama_sim3(src[keep], tree.data[idx[keep]])
        stats = {"inliers": float(keep.mean()),
                 "rms": float(np.sqrt((dists[keep] ** 2).mean()))}
    return s, R, t, stats


def camera_centers_from_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    """Camera centers [S, 3] from world-to-camera extrinsics [S, 3, 4] (VGGT convention)."""
    R, t = extrinsics[:, :, :3], extrinsics[:, :, 3]
    return -np.einsum("sij,si->sj", R, t)


# ------------------------------------------------------------------------------------------
# Vote lifting (SegVGGT recipe: per-vertex votes, then majority per superpoint)
# ------------------------------------------------------------------------------------------

def accumulate_votes(points: np.ndarray, point_query: np.ndarray, vertices: np.ndarray,
                     num_queries: int, radius: float) -> np.ndarray:
    """
    Vote counts [V, Q]: every 3D point (an unprojected mask pixel, already in mesh
    coordinates) votes for its query on the nearest mesh vertex within `radius` (meters);
    points farther than that from the mesh vote nowhere.
    """
    points = np.asarray(points, dtype=np.float64)
    point_query = np.asarray(point_query)
    votes = np.zeros((len(vertices), num_queries), dtype=np.int32)
    if len(points) == 0:
        return votes
    tree = cKDTree(np.asarray(vertices, dtype=np.float64))
    dists, idx = tree.query(points, k=1, distance_upper_bound=radius, workers=-1)
    keep = np.isfinite(dists)
    np.add.at(votes, (idx[keep], point_query[keep]), 1)
    return votes


def superpoint_majority(votes: np.ndarray, superpoints: np.ndarray) -> np.ndarray:
    """
    Per-vertex query assignment [V] (-1 = unassigned) by plurality vote per superpoint:
    each superpoint sums its vertices' votes and goes entirely to the winning query, or to
    no one if it received no votes at all. This is what makes the lifted masks respect the
    GT over-segmentation boundaries (and what dedups stray pixel votes).
    """
    sp_ids, sp_inverse = np.unique(superpoints, return_inverse=True)
    sp_votes = np.zeros((len(sp_ids), votes.shape[1]), dtype=np.int64)
    np.add.at(sp_votes, sp_inverse, votes)
    winner = sp_votes.argmax(axis=1)
    winner[sp_votes.max(axis=1) == 0] = -1
    return winner[sp_inverse]


def assign_pixels_to_queries(mask_logits: torch.Tensor, out_hw: Tuple[int, int],
                             prob_threshold: float = 0.5) -> np.ndarray:
    """
    Per-pixel owning query [H, W] (-1 = no query) from mask logits [Q, h, w]: logits are
    bilinearly upsampled to `out_hw` (the same rule as the full-resolution ruler, §6.5),
    and a pixel goes to the highest-probability query above `prob_threshold`. A partition,
    not per-query masks — the majority vote downstream expects each pixel to argue for at
    most one instance.
    """
    if mask_logits.numel() == 0:
        return np.full(out_hw, -1, dtype=np.int64)
    probs = upsample_mask_logits(mask_logits, out_hw).sigmoid()
    top_prob, top_query = probs.max(dim=0)
    assign = torch.where(top_prob > prob_threshold, top_query,
                         torch.full_like(top_query, -1))
    return assign.cpu().numpy()


def unproject_masks_to_points(world_points: np.ndarray, pixel_query: np.ndarray,
                              conf: Optional[np.ndarray] = None,
                              conf_threshold: float = -np.inf
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flatten per-view unprojected pixels into a vote list.

    world_points [S, H, W, 3]: each pixel's predicted 3D position (already aligned).
    pixel_query  [S, H, W]:    the query owning the pixel, -1 where no query claims it.
    conf         [S, H, W]:    optional per-pixel confidence, kept where >= conf_threshold.

    Returns (points [N, 3], point_query [N]).
    """
    keep = np.asarray(pixel_query) >= 0
    if conf is not None:
        keep &= np.asarray(conf) >= conf_threshold
    return np.asarray(world_points)[keep], np.asarray(pixel_query)[keep]
