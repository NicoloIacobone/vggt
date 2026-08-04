"""
Geometry for the 3D benchmark eval (docs/MASKDINO.md §9): registration of VGGT's
scale-ambiguous predictions to the GT mesh frame, and the SegVGGT-style vote lifting of
per-view 2D masks onto mesh vertices.

Registration is EVAL-ONLY machinery: inference uses VGGT's own predicted depth + cameras
and never sees GT geometry; the Sim(3) here merely expresses the finished prediction in
the benchmark mesh's coordinate frame so it can be scored — the same convention as
FAST3DIS ("Sim(3) + ICP alignment"). It is solved from predicted-vs-GT camera centers
(closed-form Umeyama), optionally refined by a similarity ICP against the mesh vertices.

Two transfers of a 2D mask onto the mesh live here, and the choice between them is
`--transfer_mode` (docs/MASKDINO.md §9.9):

  - `unproject` (default, the headline): push our pixels into 3D with PREDICTED depth +
    cameras, register with Sim(3)+ICP, vote on the nearest vertex within a radius. Measures
    2D mask quality TIMES feed-forward geometry quality.
  - `gt_projection` (the SegVGGT protocol): pull the mesh vertices into each view with the
    GT pose + GT intrinsics, keep the ones the sensor depth confirms, read the mask there.
    The correspondence is exact by construction, so it measures 2D mask quality alone.

Both are eval-time only — GT geometry never reaches the prediction path in either.

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


# ------------------------------------------------------------------------------------------
# GT-projection transfer (the SegVGGT protocol; docs/MASKDINO.md §9.9)
#
# The mirror image of the vote lifting above. Instead of pushing our pixels into 3D with
# PREDICTED depth + cameras, it pulls the benchmark's own vertices into each view with the
# GT pose + GT intrinsics and reads the mask there. The 3D<->2D correspondence is then exact
# by construction: no Sim(3), no ICP, no scale estimate, no vote radius. It measures 2D mask
# quality alone, where the default protocol measures 2D mask quality TIMES feed-forward
# geometry quality — a different experiment, reported in its own column, never a
# re-baselining of the unprojection number.
#
# Still EVAL-TIME TRANSFER ONLY: the GT pose/intrinsics/sensor depth enter after the head has
# produced its masks, exactly as the Sim(3)+ICP does. The model sees only images.
# ------------------------------------------------------------------------------------------

def mask_grid_intrinsic(K_color: np.ndarray, color_wh: Tuple[int, int],
                        mask_hw: Tuple[int, int]) -> np.ndarray:
    """
    The intrinsic of OUR mask grid, derived from the color intrinsic and the exact resize
    `data/scannet_overfit.py::load_frames_by_name` performs.

    That resize is a SQUASH: `Image.open(jpg).resize((518, 518), BILINEAR)` maps the whole
    1296x968 color image onto 518x518 with no crop, no letterbox and no aspect preservation.
    A full-extent-to-full-extent resample means the continuous pixel coordinate scales
    linearly and *anisotropically*:

        u_mask = u_color * 518 / 1296          v_mask = v_color * 518 / 968

    with both coordinates in the corner convention (u in [0, W), pixel index = floor(u)) —
    the convention `np.floor(u).clip(0, W - 1)` implements and the one the resize's extent
    mapping implies. Folding the scale into the intrinsic gives

        K_mask = diag(518/1296, 518/968, 1) @ K_color

    so a 3D point can be sent straight to a mask pixel.

    Two notes on how this relates to SegVGGT's `u * mask_w / depth_w`
    (SegVGGT/eval/eval_instance_seg.py:293-303):

      - Their rescale is per-axis too, and because ScanNet's color and depth intrinsics are
        proportional it lands within ~0.3 px of this derivation (asserted in
        `tests/test_maskdino_eval3d.py::test_mask_grid_intrinsic`). Going through K_color is
        the exact route for OUR masks — it does not assume the two cameras are proportional
        — but the reference implementation is not wrong, and agreeing with it is a check.
      - What WOULD be wrong is an *isotropic* rescale, the natural assumption if you expect
        aspect-preserving preprocessing (SegVGGT resizes to 518x392, we squash to 518x518).
        The two factors here differ, 0.400 vs 0.535; using 0.400 for both misplaces the
        principal point by ~40 rows and every mask read with it.
    """
    w, h = color_wh
    mh, mw = mask_hw
    return np.diag([mw / float(w), mh / float(h), 1.0]) @ np.asarray(K_color, dtype=np.float64)


def project_vertices_to_view(vertices: np.ndarray, pose_c2w: np.ndarray,
                             K_depth: np.ndarray, depth_map: np.ndarray,
                             K_mask: np.ndarray, mask_hw: Tuple[int, int],
                             depth_tolerance: float = 0.1
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                        Dict[str, int]]:
    """
    Which mesh vertices one frame actually sees, and where they land on our mask grid.

    Mirrors SegVGGT's `map_pred_inst_to_gt_pointcloud`
    (SegVGGT/eval/eval_instance_seg.py:266-311) view loop:

      1. world -> camera with `inv(pose_c2w)` (the 25k export's pose is camera-to-world),
      2. project with `K_depth` into the NATIVE 640x480 depth grid — the sensor depth is
         never resampled, so the occlusion test is exact,
      3. keep vertices whose projected depth agrees with the sensor: a reading must exist
         (> 0) and `|z_proj - z_sensor| < depth_tolerance` (SegVGGT: 0.1 m). This is the
         visibility test: a vertex behind the visible surface projects into the frame but
         its z disagrees with what the sensor saw, so it is dropped,
      4. project the survivors with `K_mask` (see `mask_grid_intrinsic`) to get the pixel of
         OUR mask to read.

    Returns `(vertex_idx [N], rows [N], cols [N], z [N], stats)` — the mask-grid pixel
    (row, col) and camera-space depth of each visible vertex. `stats` counts the funnel
    (front / in the depth image / with a reading / depth-consistent / on the mask grid) so
    an implausible inlier fraction is visible per scene rather than hidden in the AP.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    T = np.linalg.inv(np.asarray(pose_c2w, dtype=np.float64))       # world -> camera
    cam = vertices @ T[:3, :3].T + T[:3, 3]
    z = cam[:, 2]

    empty = np.zeros(0, dtype=np.int64)
    stats = {"vertices": len(vertices), "front": 0, "in_depth": 0, "has_reading": 0,
             "depth_inlier": 0, "on_mask": 0}
    idx = np.nonzero(z > 1e-6)[0]
    stats["front"] = len(idx)
    if len(idx) == 0:
        return empty, empty, empty, np.zeros(0), stats

    ray = np.stack([cam[idx, 0] / z[idx], cam[idx, 1] / z[idx], np.ones(len(idx))], axis=1)

    dh, dw = np.shape(depth_map)
    pd = ray @ np.asarray(K_depth, dtype=np.float64).T
    u_d, v_d = pd[:, 0], pd[:, 1]
    in_depth = (u_d >= 0) & (u_d < dw) & (v_d >= 0) & (v_d < dh)
    stats["in_depth"] = int(in_depth.sum())

    sensor = np.zeros(len(idx), dtype=np.float64)
    sensor[in_depth] = np.asarray(depth_map)[
        np.floor(v_d[in_depth]).astype(np.int64), np.floor(u_d[in_depth]).astype(np.int64)]
    has_reading = in_depth & (sensor > 0)
    stats["has_reading"] = int(has_reading.sum())
    inlier = has_reading & (np.abs(z[idx] - sensor) < depth_tolerance)
    stats["depth_inlier"] = int(inlier.sum())

    mh, mw = mask_hw
    pm = ray @ np.asarray(K_mask, dtype=np.float64).T
    u_m, v_m = pm[:, 0], pm[:, 1]
    keep = inlier & (u_m >= 0) & (u_m < mw) & (v_m >= 0) & (v_m < mh)
    stats["on_mask"] = int(keep.sum())

    return (idx[keep],
            np.floor(v_m[keep]).astype(np.int64),
            np.floor(u_m[keep]).astype(np.int64),
            z[idx][keep], stats)


def project_votes_to_vertices(vertices: np.ndarray, poses: np.ndarray, K_depth: np.ndarray,
                              K_mask: np.ndarray, depth_maps: np.ndarray,
                              pixel_query: np.ndarray, num_queries: int,
                              depth_tolerance: float = 0.1
                              ) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Vote counts [V, Q] by the GT-projection transfer — the drop-in replacement for
    `accumulate_votes` in `--transfer_mode gt_projection`.

    Every vertex a frame sees (`project_vertices_to_view`) casts one vote for whichever
    query owns the mask pixel it lands on; pixels no query claims (`pixel_query == -1`) cast
    none. `superpoint_majority` then consumes the result exactly as in the unprojection
    path, so the two modes differ ONLY in how the votes were gathered.

    `poses` [S, 4, 4] camera-to-world, `depth_maps` [S, dh, dw] meters, `pixel_query`
    [S, mh, mw] the per-pixel owning query (-1 = none). Returns (votes, stats) where stats
    carries the sensor-depth inlier fraction — an implausibly low value means the
    projection is wrong, so it is reported per scene.
    """
    pixel_query = np.asarray(pixel_query)
    if pixel_query.ndim != 3:
        raise ValueError(f"pixel_query must be [S, H, W], got {pixel_query.shape}")
    S, mh, mw = pixel_query.shape
    if len(poses) != S or len(depth_maps) != S:
        raise ValueError(f"{S} mask maps but {len(poses)} poses / {len(depth_maps)} depths")

    votes = np.zeros((len(vertices), num_queries), dtype=np.int32)
    seen = np.zeros(len(vertices), dtype=bool)
    total = {"front": 0, "in_depth": 0, "has_reading": 0, "depth_inlier": 0, "on_mask": 0}
    for f in range(S):
        vidx, rows, cols, _, st = project_vertices_to_view(
            vertices, poses[f], K_depth, depth_maps[f], K_mask, (mh, mw), depth_tolerance)
        for k in total:
            total[k] += st[k]
        if len(vidx) == 0:
            continue
        seen[vidx] = True
        q = pixel_query[f][rows, cols]
        claimed = q >= 0
        if claimed.any():
            np.add.at(votes, (vidx[claimed], q[claimed]), 1)

    denom = max(total["has_reading"], 1)
    stats = {
        "proj_frames": S,
        "depth_inlier_frac": total["depth_inlier"] / denom,
        "depth_reading_frac": total["has_reading"] / max(total["in_depth"], 1),
        "visible_vertex_frac": float(seen.mean()) if len(vertices) else float("nan"),
    }
    return votes, stats


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
