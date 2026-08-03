#!/usr/bin/env python3
"""
3D benchmark eval (docs/MASKDINO.md §9, docs/todo.md 1d). Standalone, CPU-only.

  - `train/scannet3d.py`: the minimal PLY reader (binary + ascii + truncation), the
    per-vertex GT builder against toy segs/aggregation/tsv fixtures, the class tables,
    and 25k frame sampling (non-finite poses excluded);
  - `train/eval3d_geometry.py`: Umeyama recovers a planted Sim(3), ICP repairs a
    perturbed one, unprojection round-trips through VGGT's own geometry utils, vote
    accumulation + superpoint majority on planted data, pixel->query assignment;
  - `train/benchmark3d.py` (the vendored official evaluator): perfect predictions score
    exactly 1.0, and the hand-computable cases — an IoU-0.5 prediction passes AP25 but
    not AP50, a genuine false positive halves AP, a duplicate after full recall does not,
    predictions on void vertices are ignored, sub-100-vertex predictions are skipped;
  - end-to-end on a synthetic scene: planted mask pixels -> votes -> majority ->
    evaluator -> AP 1.0.
"""

import json
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

np.random.seed(0)
torch.manual_seed(0)


# ------------------------------------------------------------------------------------------
# fixtures
# ------------------------------------------------------------------------------------------

def _write_binary_ply(path, xyz, with_faces=True):
    """A binary_little_endian PLY with the exact vertex layout ScanNet ships
    (x/y/z float + red/green/blue/alpha uchar), plus a face element after it."""
    n = len(xyz)
    header = [b"ply", b"format binary_little_endian 1.0",
              f"element vertex {n}".encode()]
    header += [b"property float x", b"property float y", b"property float z",
               b"property uchar red", b"property uchar green", b"property uchar blue",
               b"property uchar alpha"]
    if with_faces:
        header += [b"element face 1", b"property list uchar int vertex_indices"]
    header += [b"end_header"]
    with open(path, "wb") as f:
        f.write(b"\n".join(header) + b"\n")
        for p in xyz:
            f.write(struct.pack("<fffBBBB", *p, 10, 20, 30, 255))
        if with_faces:
            f.write(struct.pack("<Biii", 3, 0, 1, 2))


def _random_rotation():
    q, _ = np.linalg.qr(np.random.randn(3, 3))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def _tsv(path):
    path.write_text("id\traw_category\tcategory\tcount\tnyu40id\n"
                    "1\tchair\tchair\t100\t5\n"
                    "2\toffice chair\tchair\t50\t5\n"
                    "3\ttable\ttable\t80\t7\n"
                    "4\tmystery gadget\tmystery\t1\t\n")


# ------------------------------------------------------------------------------------------
# train/scannet3d.py
# ------------------------------------------------------------------------------------------

def test_class_tables():
    print("=== Testing class tables ===")
    from train.scannet3d import (BENCHMARK_CLASS_IDS, BENCHMARK_CLASS_NAMES,
                                 SCANNET_IDX_TO_NYU40)
    assert SCANNET_IDX_TO_NYU40[1] == 1 and SCANNET_IDX_TO_NYU40[2] == 2   # wall, floor
    assert SCANNET_IDX_TO_NYU40[5] == 5                                    # chair
    assert SCANNET_IDX_TO_NYU40[19] == 36                                  # bathtub
    assert SCANNET_IDX_TO_NYU40[20] == 39                                  # otherfurniture
    assert len(BENCHMARK_CLASS_IDS) == len(BENCHMARK_CLASS_NAMES) == 18
    assert 1 not in BENCHMARK_CLASS_IDS and 2 not in BENCHMARK_CLASS_IDS   # no wall/floor
    assert 39 in BENCHMARK_CLASS_IDS                                       # otherfurniture
    # every head class except wall/floor is a benchmark class (17 shared, benchmark adds 39)
    head_nyu40 = {SCANNET_IDX_TO_NYU40[i] for i in range(1, 20)}
    assert head_nyu40 - {1, 2} < set(BENCHMARK_CLASS_IDS)
    print("✅ class tables\n")


def test_ply_reader():
    print("=== Testing PLY reader ===")
    from train.scannet3d import read_ply_vertices
    xyz = np.random.randn(50, 3).astype(np.float32)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "mesh.ply"
        _write_binary_ply(p, xyz)
        got = read_ply_vertices(p)
        assert got.shape == (50, 3) and got.dtype == np.float64
        assert np.allclose(got, xyz, atol=1e-6)

        _write_binary_ply(p, xyz, with_faces=False)
        assert np.allclose(read_ply_vertices(p), xyz, atol=1e-6)

        # ascii variant
        pa = Path(d) / "mesh_ascii.ply"
        lines = ["ply", "format ascii 1.0", f"element vertex {len(xyz)}",
                 "property float x", "property float y", "property float z", "end_header"]
        lines += [" ".join(f"{v:.6f}" for v in p3) for p3 in xyz]
        pa.write_text("\n".join(lines) + "\n")
        assert np.allclose(read_ply_vertices(pa), xyz, atol=1e-5)

        # truncated binary must raise, not return garbage
        data = p.read_bytes()
        p.write_bytes(data[:len(data) - 100])
        try:
            read_ply_vertices(p)
            raise AssertionError("truncated PLY did not raise")
        except ValueError:
            pass
    print("✅ PLY reader: binary LE (with/without faces), ascii, truncation guard\n")


def test_gt_builder():
    print("=== Testing superpoints + per-vertex GT builder ===")
    from train.scannet3d import build_gt_ids, load_raw_to_nyu40, load_superpoints
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _tsv(d / "labels.tsv")
        raw2nyu = load_raw_to_nyu40(d / "labels.tsv")
        assert raw2nyu == {"chair": 5, "office chair": 5, "table": 7}  # blank nyu40id dropped

        # 12 vertices in 4 superpoints (ids arbitrary, not dense)
        superpoints = np.array([7, 7, 7, 42, 42, 42, 9, 9, 9, 13, 13, 13])
        (d / "segs.json").write_text(json.dumps({"segIndices": superpoints.tolist()}))
        assert np.array_equal(load_superpoints(d / "segs.json"), superpoints)

        agg = {"segGroups": [
            {"id": 0, "objectId": 0, "label": "chair", "segments": [7, 42]},
            {"id": 1, "objectId": 1, "label": "table", "segments": [9]},
            {"id": 2, "objectId": 2, "label": "mystery gadget", "segments": [13]},
        ]}
        (d / "agg.json").write_text(json.dumps(agg))
        gt = build_gt_ids(superpoints, d / "agg.json", raw2nyu)
        assert np.array_equal(gt[:6], np.full(6, 5001))    # chair: 5 * 1000 + 0 + 1
        assert np.array_equal(gt[6:9], np.full(3, 7002))   # table: 7 * 1000 + 1 + 1
        assert np.array_equal(gt[9:], np.zeros(3))         # unmapped raw label -> unannotated
    print("✅ GT builder: benchmark encoding, tsv mapping, unmapped labels -> 0\n")


def test_frames25k_sampling():
    print("=== Testing 25k frame sampling ===")
    from train.scannet3d import load_frames25k_poses, sample_frames25k
    with tempfile.TemporaryDirectory() as d:
        scene = Path(d) / "scene0000_00"
        (scene / "pose").mkdir(parents=True)
        (scene / "color").mkdir()
        eye = "\n".join(" ".join(str(float(i == j)) for j in range(4)) for i in range(4))
        for k in range(10):
            (scene / "pose" / f"{k:06d}.txt").write_text(eye)
            (scene / "color" / f"{k:06d}.jpg").write_bytes(b"x")
        # a known export defect: -inf pose -> the frame must vanish from sampling
        (scene / "pose" / "000003.txt").write_text(eye.replace("1.0", "-inf", 1))
        # a pose without its jpg is unusable too
        (scene / "color" / "000007.jpg").unlink()

        poses = load_frames25k_poses(scene)
        assert "000003" not in poses and len(poses) == 9
        stems = sample_frames25k(scene)
        assert stems == sorted(set(poses) - {"000007"})
        capped = sample_frames25k(scene, num_frames=4)
        assert len(capped) == 4 and capped[0] == stems[0] and capped[-1] == stems[-1]
    print("✅ 25k sampling: finite-pose + jpg filter, even cap keeps the endpoints\n")


# ------------------------------------------------------------------------------------------
# train/eval3d_geometry.py
# ------------------------------------------------------------------------------------------

def test_umeyama_and_icp():
    print("=== Testing Umeyama Sim(3) + similarity ICP ===")
    from train.eval3d_geometry import apply_sim3, icp_refine_sim3, umeyama_sim3
    src = np.random.randn(40, 3)
    R_true, s_true, t_true = _random_rotation(), 2.3, np.array([0.5, -1.0, 4.0])
    dst = apply_sim3(src, s_true, R_true, t_true)

    s, R, t = umeyama_sim3(src, dst)
    assert abs(s - s_true) < 1e-9 and np.allclose(R, R_true) and np.allclose(t, t_true)
    assert np.allclose(apply_sim3(src, s, R, t), dst)

    # noise: recovery stays close
    s, R, t = umeyama_sim3(src, dst + 0.01 * np.random.randn(*dst.shape))
    assert abs(s - s_true) < 0.05 and np.allclose(R, R_true, atol=0.05)

    # ICP repairs a perturbed initialisation against a dense target cloud
    cloud = np.random.randn(3000, 3)
    target = apply_sim3(cloud, s_true, R_true, t_true)
    s0, t0 = s_true * 1.08, t_true + 0.15
    s, R, t, stats = icp_refine_sim3(cloud, target, s0, R_true, t0, iters=15, max_dist=1.0)
    assert abs(s - s_true) < 1e-3 and np.allclose(t, t_true, atol=5e-3)
    assert stats["inliers"] > 0.99 and stats["rms"] < 1e-3

    # degenerate input refuses instead of returning nonsense
    try:
        umeyama_sim3(np.zeros((5, 3)), np.ones((5, 3)))
        raise AssertionError("zero-variance source did not raise")
    except ValueError:
        pass
    print("✅ Umeyama exact + noisy, ICP repair, degeneracy guard\n")


def test_unprojection_roundtrip():
    print("=== Testing unprojection round-trip through VGGT's geometry utils ===")
    from train.eval3d_geometry import camera_centers_from_extrinsics
    from vggt.utils.geometry import unproject_depth_map_to_point_map

    H = W = 8
    intr = np.array([[10.0, 0, 4.0], [0, 12.0, 3.0], [0, 0, 1]])
    R, t = _random_rotation(), np.array([0.2, -0.4, 1.5])
    extr = np.concatenate([R, t[:, None]], axis=1)                # cam from world [3,4]
    depth = np.random.uniform(1.0, 3.0, size=(1, H, W, 1))

    world = unproject_depth_map_to_point_map(depth, extr[None], intr[None])[0]  # [H,W,3]
    # project back by hand: x_cam = R @ w + t must reproduce (u, v, depth)
    cam = world @ R.T + t
    u = intr[0, 0] * cam[..., 0] / cam[..., 2] + intr[0, 2]
    v = intr[1, 1] * cam[..., 1] / cam[..., 2] + intr[1, 2]
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    assert np.allclose(u, uu, atol=1e-4) and np.allclose(v, vv, atol=1e-4)
    assert np.allclose(cam[..., 2], depth[0, ..., 0], atol=1e-5)

    # the camera center is where a zero-depth ray starts: -R^T t
    centers = camera_centers_from_extrinsics(extr[None])
    assert np.allclose(centers[0], -R.T @ t)
    print("✅ unprojection round-trip + camera centers (cam-from-world convention)\n")


def test_votes_and_majority():
    print("=== Testing vote accumulation + superpoint majority ===")
    from train.eval3d_geometry import accumulate_votes, superpoint_majority
    # 6 vertices in 3 superpoints of 2
    vertices = np.array([[0, 0, 0], [0, 0, 1], [5, 0, 0], [5, 0, 1], [9, 0, 0], [9, 0, 1]],
                        dtype=np.float64)
    superpoints = np.array([1, 1, 2, 2, 3, 3])
    # query 0 votes twice near vertex 0, once near vertex 2; query 1 votes twice near
    # vertex 3; a far-away point (beyond radius) must vote nowhere
    points = np.array([[0, 0.01, 0], [0, -0.01, 0], [5, 0.01, 0],
                       [5, 0, 1.01], [5, 0.02, 1], [50, 50, 50]])
    point_query = np.array([0, 0, 0, 1, 1, 0])
    votes = accumulate_votes(points, point_query, vertices, num_queries=2, radius=0.1)
    assert votes.shape == (6, 2)
    assert votes[0, 0] == 2 and votes[2, 0] == 1 and votes[3, 1] == 2
    assert votes.sum() == 5                       # the far point voted nowhere

    assign = superpoint_majority(votes, superpoints)
    assert np.array_equal(assign[:2], [0, 0])     # superpoint 1: only query-0 votes
    assert np.array_equal(assign[2:4], [1, 1])    # superpoint 2: 2 votes q1 vs 1 vote q0
    assert np.array_equal(assign[4:], [-1, -1])   # superpoint 3: no votes at all
    print("✅ votes within radius only, plurality per superpoint, unvoted -> -1\n")


def test_pixel_assignment():
    print("=== Testing pixel -> query assignment ===")
    from train.eval3d_geometry import assign_pixels_to_queries, unproject_masks_to_points
    logits = torch.full((2, 4, 4), -10.0)
    logits[0, :2, :] = 5.0        # query 0 owns the top half
    logits[1, :, :2] = 3.0        # query 1 claims the left half, but loses the overlap
    assign = assign_pixels_to_queries(logits, (8, 8))
    assert assign.shape == (8, 8)
    assert (assign[:3, 5:] == 0).all()            # top-right: only query 0
    assert (assign[5:, :3] == 1).all()            # bottom-left: only query 1
    assert (assign[:3, :3] == 0).all()            # overlap: higher-prob query wins
    assert (assign[5:, 5:] == -1).all()           # nobody above threshold
    assert (assign_pixels_to_queries(torch.zeros(0, 4, 4), (8, 8)) == -1).all()

    wp = np.arange(8 * 8 * 3, dtype=np.float64).reshape(1, 8, 8, 3)
    conf = np.ones((1, 8, 8))
    conf[0, 0, :] = 0.0
    pts, pq = unproject_masks_to_points(wp, assign[None], conf, conf_threshold=0.5)
    assert (pq >= 0).all()
    # the conf filter must remove exactly row 0's assigned pixels, nothing else
    assert len(pts) == int((assign >= 0).sum() - (assign[0] >= 0).sum())
    assert (assign[0] >= 0).any()                 # ... and row 0 did have assigned pixels
    print("✅ pixel assignment: argmax over threshold, empty-query guard, conf filter\n")


# ------------------------------------------------------------------------------------------
# train/benchmark3d.py — the vendored official evaluator
# ------------------------------------------------------------------------------------------

def _scene(num_verts=1000):
    """GT with two 200-vertex chairs and a 200-vertex void block (vertices 600:800)."""
    gt = np.zeros(num_verts, dtype=np.int64)
    gt[0:200] = 5 * 1000 + 1
    gt[200:400] = 5 * 1000 + 2
    return gt


def _pred(sl, label_id=5, confidence=0.9, num_verts=1000):
    mask = np.zeros(num_verts, dtype=bool)
    mask[sl] = True
    return {"mask": mask, "label_id": label_id, "confidence": confidence}


def test_evaluator_perfect():
    print("=== Testing evaluator: perfect predictions ===")
    from train.benchmark3d import evaluate, format_results
    gt = _scene()
    preds = [_pred(slice(0, 200)), _pred(slice(200, 400), confidence=0.8)]
    r = evaluate({"s1": preds}, {"s1": gt})
    assert r["all_ap"] == 1.0 and r["all_ap_50%"] == 1.0 and r["all_ap_25%"] == 1.0
    assert r["classes"]["chair"]["ap50%"] == 1.0
    assert np.isnan(r["classes"]["bed"]["ap50%"])          # no bed GT anywhere -> NaN
    assert "average" in format_results(r)
    print("✅ perfect predictions -> AP = AP50 = AP25 = 1.0, absent classes NaN\n")


def test_evaluator_iou_half():
    print("=== Testing evaluator: IoU-0.5 prediction passes AP25, fails AP50 ===")
    from train.benchmark3d import evaluate
    gt = np.zeros(1000, dtype=np.int64)
    gt[0:200] = 5 * 1000 + 1
    # covers half the chair, nothing else: IoU = 100 / 200 = 0.5, NOT > 0.5
    r = evaluate({"s1": [_pred(slice(0, 100))]}, {"s1": gt})
    assert r["classes"]["chair"]["ap25%"] == 1.0
    assert r["classes"]["chair"]["ap50%"] == 0.0
    print("✅ strict '>' at the overlap threshold, exactly like the official code\n")


def test_evaluator_false_positive():
    print("=== Testing evaluator: a genuine FP halves AP50 ===")
    from train.benchmark3d import evaluate
    gt = _scene()
    preds = [_pred(slice(0, 200), confidence=0.9),          # perfect on chair 1
             _pred(slice(280, 400), confidence=0.8)]        # 120/200 of chair 2: IoU 0.6 -> ok
    r = evaluate({"s1": preds}, {"s1": gt})
    assert r["classes"]["chair"]["ap50%"] == 1.0

    preds = [_pred(slice(0, 200), confidence=0.9),
             _pred(slice(340, 460), confidence=0.8)]        # 60/200 of chair 2: IoU 0.23 -> FP
    r = evaluate({"s1": preds}, {"s1": gt})
    # hand-computed: TP@0.9, FP@0.8, one hard FN -> precision [0.5, 1], recall [0.5, 0.5]
    assert abs(r["classes"]["chair"]["ap50%"] - 0.5) < 1e-9
    print("✅ FP + hard FN give the hand-computed AP50 = 0.5\n")


def test_evaluator_duplicate_and_void():
    print("=== Testing evaluator: duplicates after full recall, void ignore, min size ===")
    from train.benchmark3d import evaluate
    gt = np.zeros(1000, dtype=np.int64)
    gt[0:200] = 5 * 1000 + 1
    # duplicate detection: lower-confidence twin becomes an FP *after* recall is complete,
    # so the official integration still yields AP50 = 1.0
    preds = [_pred(slice(0, 200), confidence=0.9), _pred(slice(0, 200), confidence=0.7)]
    r = evaluate({"s1": preds}, {"s1": gt})
    assert r["classes"]["chair"]["ap50%"] == 1.0

    # a prediction living on void vertices (unannotated / non-benchmark classes) is
    # ignored, not an FP — wall/floor GT is void too, which is why dropping our
    # wall/floor predictions is on us, not on the evaluator
    preds = [_pred(slice(0, 200), confidence=0.9), _pred(slice(600, 800), confidence=0.8)]
    r = evaluate({"s1": preds}, {"s1": gt})
    assert r["classes"]["chair"]["ap50%"] == 1.0

    # sub-100-vertex predictions are skipped entirely (official min region size)
    preds = [_pred(slice(0, 200), confidence=0.9), _pred(slice(400, 450), confidence=0.8)]
    r = evaluate({"s1": preds}, {"s1": gt})
    assert r["classes"]["chair"]["ap50%"] == 1.0
    print("✅ duplicate/void/min-region rules match the official semantics\n")


def test_script_helpers():
    print("=== Testing eval_3d_maskdino script helpers ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from eval_3d_maskdino import seventeen_class_mean, write_instance_ply
    from train.benchmark3d import evaluate
    from train.scannet3d import read_ply_vertices

    # otherfurniture has GT but no prediction: 18-class mean pays for it, the
    # 17-class diagnostic does not
    gt = np.zeros(1000, dtype=np.int64)
    gt[0:200] = 5 * 1000 + 1       # chair
    gt[200:400] = 39 * 1000 + 1    # otherfurniture — unpredictable for our head
    r = evaluate({"s1": [_pred(slice(0, 200))]}, {"s1": gt})
    assert abs(r["all_ap_50%"] - 0.5) < 1e-9
    d = seventeen_class_mean(r)
    assert d["all_ap_50%"] == 1.0 and d["all_ap_25%"] == 1.0

    with tempfile.TemporaryDirectory() as td:
        verts = np.random.randn(20, 3)
        assign = np.array([0] * 10 + [1] * 5 + [-1] * 5)
        write_instance_ply(Path(td) / "x.ply", verts, assign)
        got = read_ply_vertices(Path(td) / "x.ply")   # our reader parses our dump (ascii)
        assert np.allclose(got, verts, atol=1e-3)
    print("✅ 17-class diagnostic vs 18-class headline, ply dump round-trips\n")


# ------------------------------------------------------------------------------------------
# end-to-end on a synthetic scene
# ------------------------------------------------------------------------------------------

def test_end_to_end_synthetic():
    print("=== Testing votes -> majority -> evaluator end to end ===")
    from train.benchmark3d import evaluate
    from train.eval3d_geometry import accumulate_votes, superpoint_majority

    # three 200-vertex clusters; two are chairs, the third is unannotated (void)
    rng = np.random.default_rng(1)
    centers = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], dtype=np.float64)
    vertices = np.concatenate([c + 0.2 * rng.standard_normal((200, 3)) for c in centers])
    superpoints = np.repeat(np.arange(6), 100)            # each cluster = 2 superpoints
    gt = np.zeros(600, dtype=np.int64)
    gt[0:200] = 5 * 1000 + 1
    gt[200:400] = 5 * 1000 + 2

    # "unprojected pixels": noisy copies of 150 vertices per chair, one query each,
    # plus a handful of stray query-0 votes inside chair 2 that the majority must undo
    idx0, idx1 = rng.choice(200, 150, False), 200 + rng.choice(200, 150, False)
    points = np.concatenate([vertices[idx0], vertices[idx1], vertices[210:220]])
    points = points + 0.005 * rng.standard_normal(points.shape)
    point_query = np.array([0] * 150 + [1] * 150 + [0] * 10)

    votes = accumulate_votes(points, point_query, vertices, num_queries=2, radius=0.05)
    assign = superpoint_majority(votes, superpoints)
    assert (assign[400:] == -1).all()                     # nobody voted on the void cluster
    preds = [{"mask": assign == q, "label_id": 5, "confidence": 0.9 - 0.1 * q}
             for q in range(2)]
    r = evaluate({"s1": preds}, {"s1": gt})
    assert r["all_ap"] == 1.0 and r["all_ap_50%"] == 1.0 and r["all_ap_25%"] == 1.0
    print("✅ end to end: stray votes overruled by the superpoint majority, AP 1.0\n")


if __name__ == "__main__":
    test_class_tables()
    test_ply_reader()
    test_gt_builder()
    test_frames25k_sampling()
    test_umeyama_and_icp()
    test_unprojection_roundtrip()
    test_votes_and_majority()
    test_pixel_assignment()
    test_evaluator_perfect()
    test_evaluator_duplicate_and_void()
    test_evaluator_iou_half()
    test_evaluator_false_positive()
    test_script_helpers()
    test_end_to_end_synthetic()
    print("All test_maskdino_eval3d tests passed! ✅")
