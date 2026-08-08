#!/usr/bin/env python3
"""
ScanNet++ val-50 dataset build (docs/todo.md 6c, docs/DATASET.md §2.1). Standalone,
CPU-only, no cluster data — everything runs against a synthetic ScanNet++ scene built in a
tmpdir.

  - `scannetpp_common`: uniform frame sampling, intrinsic rescaling, the PLY header vertex
    count, the `"sceneId:"` (sic) segments key, the LZ4 depth-stream reader, unprojection,
    and the mesh/image NCC;
  - the per-vertex instance GT: `instance_map_to` really is applied before the
    `top100_instance` filter, out-of-taxonomy objects become void, ids are dense and
    1-based, and a segment id that does not exist in `segments.json` raises rather than
    silently dropping geometry (segment-id closure, as `download_3d_gt.py` does);
  - `build_scannetpp_3d_gt.py` end to end on a synthetic scene, including that a
    closure-violating scene FAILS instead of shipping;
  - `build_scannetpp_frames.py` end to end: 4x4 camera-to-world poses, uint16 millimetre
    depth, the manifest contract, and — the point of the whole exercise — that the
    geometry self-check REJECTS a scene whose poses are wrong and the RGB index check
    REJECTS a scene whose colour stream is shifted against its poses;
  - `scripts/verify_scannetpp_gt.py` passing on the built tree, and failing when the GT is
    tampered with.

The synthetic scene is exact, not approximate: a textured plane at z=0, cameras above it,
so the sensor depth, the mesh and the images are analytically consistent and the checks
have a right answer to find.
"""

import json
import shutil
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent /
                       "legacy" / "dataset_build" / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import cv2  # noqa: E402
import lz4.block  # noqa: E402

import build_scannetpp_3d_gt as gtbuild  # noqa: E402
import build_scannetpp_frames as frbuild  # noqa: E402
from scannetpp_common import (  # noqa: E402
    DEPTH_H, DEPTH_W, build_vertex_instances, count_depth_frames, load_instance_classes,
    load_label_map, load_segments, mesh_image_ncc, ply_vertex_count, read_depth_frames,
    sample_indices, scale_intrinsic, unproject,
)

np.random.seed(0)

CLASSES = ["table", "chair", "door"]
N_FRAMES = 12
PLANE_STEP = 0.015
PLANE_HALF = 2.6
CAM_HEIGHT = 2.0
K = np.array([[200.0, 0.0, DEPTH_W / 2], [0.0, 200.0, DEPTH_H / 2], [0.0, 0.0, 1.0]])


def ok(msg):
    print(f"  ok  {msg}")


# ==========================================================================================
# synthetic ScanNet++ source scene
# ==========================================================================================

def texture(x, y):
    """A smooth procedural greyscale over the plane, 0..255."""
    return 127.5 * (1.0 + np.sin(7.0 * x) * np.cos(7.0 * y))


def write_binary_ply(path, xyz, rgb):
    n = len(xyz)
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "element face 0\nproperty list uchar int vertex_indices\n"
              "end_header\n")
    with open(path, "wb") as f:
        f.write(header.encode())
        for (x, y, z), (r, g, b) in zip(xyz, rgb.astype(np.uint8)):
            f.write(struct.pack("<fffBBB", x, y, z, r, g, b))


def plane_mesh():
    g = np.arange(-PLANE_HALF, PLANE_HALF + 1e-9, PLANE_STEP)
    xx, yy = np.meshgrid(g, g, indexing="ij")
    xyz = np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], 1)
    t = texture(xyz[:, 0], xyz[:, 1])
    rgb = np.stack([t, t, t], 1)
    return xyz, rgb


def cam2world(k):
    """Camera k: above the plane, drifting AND tilting.

    The tilt matters: a camera looking straight down with R = diag(1,-1,-1) is its own
    inverse rotation, which makes the world-to-camera reading of the pose land on the
    plane too and leaves the convention check nothing to discriminate. Real trajectories
    are not symmetric like that; neither is this one.
    """
    a, b = 0.20 + 0.05 * np.sin(0.9 * k), 0.13 * np.cos(0.6 * k)
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, -1.0, -1.0]) @ Rx @ Ry
    T[:3, 3] = [0.35 * np.sin(0.7 * k), 0.35 * np.cos(0.7 * k), CAM_HEIGHT]
    return T


def render(T):
    """The exact image and depth this camera sees of the plane z = 0."""
    yy, xx = np.mgrid[0:DEPTH_H, 0:DEPTH_W]
    d_cam = np.stack([(xx - K[0, 2]) / K[0, 0], (yy - K[1, 2]) / K[1, 1],
                      np.ones_like(xx, dtype=float)], -1)
    d_world = d_cam @ T[:3, :3].T
    s = -T[2, 3] / d_world[..., 2]              # ray parameter hitting z = 0
    p = T[:3, 3] + s[..., None] * d_world
    img = np.clip(texture(p[..., 0], p[..., 1]), 0, 255).astype(np.uint8)
    return np.stack([img] * 3, -1), (s * 1000).astype(np.uint16)


def write_depth_bin(path, depths):
    with open(path, "wb") as f:
        for d in depths:
            blob = lz4.block.compress(d.tobytes(), store_size=False)
            f.write(len(blob).to_bytes(4, "little"))
            f.write(blob)


def make_source(root: Path, scene="synth0001", shift_rgb=0, break_closure=False):
    """A minimal but complete ScanNet++ source tree for one scene."""
    d = root / "data" / scene
    (d / "scans").mkdir(parents=True, exist_ok=True)
    (d / "iphone").mkdir(parents=True, exist_ok=True)

    xyz, rgb = plane_mesh()
    write_binary_ply(d / "scans" / "mesh.ply.tmp", xyz, rgb)
    shutil.move(d / "scans" / "mesh.ply.tmp", d / "scans" / "mesh_aligned_0.05.ply")

    n = len(xyz)
    # As in the real release, each vertex is its own segment.
    (d / "scans" / "segments.json").write_text(json.dumps(
        {"sceneId:": scene, "segIndices": list(range(n))}))
    quarter = n // 4
    groups = [
        {"id": 0, "objectId": 0, "label": "work table",     # -> "table" via the map
         "segments": list(range(0, quarter))},
        {"id": 1, "objectId": 1, "label": "chair",
         "segments": list(range(quarter, 2 * quarter))},
        {"id": 2, "objectId": 2, "label": "unlisted thing",  # not a benchmark class
         "segments": list(range(2 * quarter, 3 * quarter))},
    ]
    if break_closure:
        groups.append({"id": 3, "objectId": 3, "label": "door",
                       "segments": [n + 5, n + 6]})
    (d / "scans" / "segments_anno.json").write_text(
        json.dumps({"sceneId": scene, "segGroups": groups}))

    imgs, depths, poses = [], [], []
    for k in range(N_FRAMES):
        T = cam2world(k)
        img, dep = render(T)
        imgs.append(img)
        depths.append(dep)
        poses.append(T)
    if shift_rgb:
        imgs = imgs[shift_rgb:] + imgs[:shift_rgb]   # colour stream rotated against poses

    w = cv2.VideoWriter(str(d / "iphone" / "rgb.mkv"),
                        cv2.VideoWriter_fourcc(*"FFV1"), 30, (DEPTH_W, DEPTH_H))
    assert w.isOpened(), "cv2 cannot write an FFV1 mkv — test environment problem"
    for img in imgs:
        w.write(img)
    w.release()

    write_depth_bin(d / "iphone" / "depth.bin", depths)
    (d / "iphone" / "pose_intrinsic_imu.json").write_text(json.dumps({
        f"frame_{k:06d}": {
            "timestamp": float(k) / 30.0,
            "pose": (np.eye(4) + 50.0).tolist(),    # the raw ARKit frame: far away
            "intrinsic": K.tolist(),
            "imu": {},
            "aligned_pose": poses[k].tolist(),
        } for k in range(N_FRAMES)}))

    # metadata + split, exactly the four files the build copies
    meta = root / "metadata" / "semantic_benchmark"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "top100_instance.txt").write_text("\n".join(CLASSES) + "\n")
    (meta / "map_benchmark.csv").write_text(
        "class,semantic_map_to,instance_map_to\n"
        "work table,table,table\n"
        "table,,\nchair,,\ndoor,,\nunlisted thing,,\n")
    (root / "metadata" / "semantic_classes.txt").write_text("\n".join(CLASSES) + "\n")
    (root / "splits").mkdir(parents=True, exist_ok=True)
    (root / "splits" / "nvs_sem_val.txt").write_text(scene + "\n")
    return scene


def run(module, **kw):
    """Call a build script's main() with argv assembled from kwargs."""
    argv = [module.__name__]
    for k, v in kw.items():
        if v is True:
            argv.append(f"--{k}")
        elif isinstance(v, (list, tuple)):
            argv += [f"--{k}"] + [str(x) for x in v]
        else:
            argv += [f"--{k}", str(v)]
    old, sys.argv = sys.argv, argv
    try:
        module.main()
    finally:
        sys.argv = old


# ==========================================================================================
# unit-level
# ==========================================================================================

def test_sample_indices():
    idx = sample_indices(7207, 50)
    assert len(idx) == 50 and idx[0] == 0 and idx[-1] == 7206
    assert (np.diff(idx) > 0).all(), "indices must be strictly increasing"
    assert list(sample_indices(3, 50)) == [0, 1, 2], "short sequence -> every frame once"
    assert len(sample_indices(0, 50)) == 0
    ok("sample_indices spans the whole sequence, endpoints included")


def test_scale_intrinsic():
    Kc = np.array([[1429.5, 0, 954.2], [0, 1429.5, 724.2], [0, 0, 1.0]])
    Kd = scale_intrinsic(Kc, (1920, 1440), (256, 192))
    assert np.isclose(Kd[0, 0], 1429.5 * 256 / 1920)
    assert np.isclose(Kd[1, 2], 724.2 * 192 / 1440)
    assert Kd[2, 2] == 1.0 and np.allclose(scale_intrinsic(Kc, (10, 10), (10, 10)), Kc)
    ok("scale_intrinsic rescales fx/fy/cx/cy and leaves the homogeneous row alone")


def test_depth_stream(tmp):
    frames = [np.random.randint(0, 5000, (DEPTH_H, DEPTH_W)).astype(np.uint16)
              for _ in range(7)]
    p = tmp / "depth.bin"
    write_depth_bin(p, frames)
    assert count_depth_frames(p) == 7
    got = read_depth_frames(p, [0, 3, 6])
    assert set(got) == {0, 3, 6}
    for i in (0, 3, 6):
        assert got[i].dtype == np.uint16 and (got[i] == frames[i]).all()
    assert read_depth_frames(p, [99]) == {}, "an out-of-range index yields nothing"
    ok("depth.bin: <4-byte LE size><LZ4 block> uint16 frames, seek-skipped when unwanted")


def test_segments_and_instances():
    segs_json = {"sceneId:": "x", "segIndices": [0, 0, 1, 1, 2, 3]}
    tmp = Path(tempfile.mkdtemp())
    (tmp / "s.json").write_text(json.dumps(segs_json))
    si = load_segments(tmp / "s.json")
    assert si.tolist() == [0, 0, 1, 1, 2, 3], "the key really is 'sceneId:' with a colon"

    classes = CLASSES
    label_map = {"work table": "table", "table": "table", "chair": "chair"}
    groups = [{"objectId": 7, "label": "work table", "segments": [0]},
              {"objectId": 8, "label": "unlisted", "segments": [1]},
              {"objectId": 9, "label": "chair", "segments": [2, 3]}]
    ids, inst = build_vertex_instances(si, groups, classes, label_map)
    assert [i["label"] for i in inst] == ["table", "chair"], "instance_map_to is applied"
    assert [i["inst_id"] for i in inst] == [1, 2], "ids are dense and 1-based"
    assert ids.tolist() == [1, 1, 0, 0, 2, 2], "unlisted classes become void 0"
    assert inst[0]["object_id"] == 7 and inst[1]["class_id"] == classes.index("chair")

    try:
        build_vertex_instances(si, [{"objectId": 1, "label": "chair", "segments": [99]}],
                               classes, label_map)
    except ValueError as e:
        assert "closure" in str(e)
    else:
        raise AssertionError("a segment id absent from segments.json must raise")
    ok("instance GT: map -> filter -> dense ids, void 0, segment-id closure enforced")


def test_ply_header(tmp):
    xyz, rgb = np.zeros((5, 3)), np.zeros((5, 3))
    write_binary_ply(tmp / "m.ply", xyz, rgb)
    assert ply_vertex_count(tmp / "m.ply") == 5
    (tmp / "bad.ply").write_bytes(b"not a ply at all")
    try:
        ply_vertex_count(tmp / "bad.ply")
    except ValueError:
        pass
    else:
        raise AssertionError("a file without the ply magic must raise")
    ok("ply_vertex_count reads the header only, and rejects non-ply files")


def test_unproject_and_ncc():
    xyz, rgb = plane_mesh()
    gray = rgb[:, 0]
    T = cam2world(3)
    img, depth = render(T)
    pts = unproject(depth, K, T)
    assert len(pts) == DEPTH_H * DEPTH_W
    # uint16 millimetres, so the round trip is exact only to the millimetre
    assert np.abs(pts[:, 2]).max() < 5e-3, "the plane must come back at z=0"
    good = mesh_image_ncc(xyz, gray, K, T, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    bad = mesh_image_ncc(xyz, gray, K, cam2world(7),
                         cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    assert good > 0.9, f"NCC at the true pose should be ~1, got {good}"
    assert good > bad + 0.1, f"a wrong pose must score lower ({good} vs {bad})"
    ok(f"unprojection lands on the plane; NCC {good:.3f} at the true pose vs {bad:.3f}")


# ==========================================================================================
# end to end
# ==========================================================================================

def test_gt_build(tmp):
    src = tmp / "src_gt"
    scene = make_source(src)
    out = tmp / "scans3d"
    run(gtbuild, src_root=src, out_root=out, scene_list=src / "splits/nvs_sem_val.txt",
        start=0, end=0)

    d = out / scene
    for name in ("mesh.ply", "segments.json", "segments_anno.json", ".complete"):
        assert (d / name).exists(), f"missing {name}"
    assert (d / "mesh.ply").read_bytes() == \
        (src / "data" / scene / "scans" / "mesh_aligned_0.05.ply").read_bytes(), \
        "mesh.ply must be a verbatim copy"
    for f in ("top100_instance.txt", "map_benchmark.csv", "nvs_sem_val.txt"):
        assert (out / "_metadata" / f).is_file(), f"metadata {f} not copied into the tar"
    stats = json.loads((d / "gt_stats.json").read_text())
    assert stats["n_instances"] == 2 and stats["n_seg_groups"] == 3
    assert stats["classes"] == ["chair", "table"]

    # a closure violation must fail the scene, not ship it
    src2 = tmp / "src_broken"
    make_source(src2, break_closure=True)
    out2 = tmp / "scans3d_broken"
    try:
        run(gtbuild, src_root=src2, out_root=out2,
            scene_list=src2 / "splits/nvs_sem_val.txt", start=0, end=0)
    except SystemExit as e:
        assert e.code == 1
    else:
        raise AssertionError("a closure-violating scene must exit non-zero")
    assert not (out2 / scene / ".complete").exists(), "a failed scene must stay unmarked"
    ok("3D GT build copies verbatim, counts instances, and refuses a broken scene")


def test_frames_build(tmp):
    src = tmp / "src_fr"
    scene = make_source(src)
    out = tmp / "scans25k"
    run(frbuild, src_root=src, out_root=out, scene_list=src / "splits/nvs_sem_val.txt",
        start=0, end=0, num_frames=6, n_probe=4, mesh_subsample=0,
        index_offsets=[-4, 0, 4])

    man = json.loads((out / scene / "manifest.json").read_text())
    assert man["scene"] == scene and man["total_frames"] == N_FRAMES
    assert man["sampling"] == "uniform-6" and len(man["sampled_stems"]) == 6
    assert man["sampled_stems"][0] == "frame_000000"
    assert man["sampled_stems"][-1] == f"frame_{N_FRAMES - 1:06d}"
    for stem in man["sampled_stems"]:
        assert (out / scene / "color" / f"{stem}.jpg").is_file()
        assert (out / scene / "pose" / f"{stem}.txt").is_file()
        dep = cv2.imread(str(out / scene / "depth" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        assert dep.dtype == np.uint16 and dep.shape == (DEPTH_H, DEPTH_W)
        assert 1000 < int(np.median(dep)) < 4000, \
            "depth must be written in millimetres, at this scene's ~2 m scale"
        T = np.loadtxt(out / scene / "pose" / f"{stem}.txt")
        assert T.shape == (4, 4) and np.allclose(T[3], [0, 0, 0, 1])
        k = int(stem.split("_")[1])
        assert np.allclose(T, cam2world(k), atol=1e-9), \
            "the written pose must be aligned_pose, camera-to-world, for THIS index"
    for p in ("intrinsic/intrinsic_color.txt", "intrinsic/intrinsic_depth.txt",
              "intrinsics_color.txt", "intrinsics_depth.txt"):
        assert (out / scene / p).is_file(), f"missing {p}"
    assert np.allclose(np.loadtxt(out / scene / "intrinsic/intrinsic_color.txt"), K)
    assert man["geometry_check"]["depth_mesh_median_cm"] < 2.0
    assert man["rgb_index_check"]["peak_at_zero"] is True
    ok(f"frames build: poses, mm depth, intrinsics, manifest; depth->mesh "
       f"{man['geometry_check']['depth_mesh_median_cm']:.2f} cm")
    return src, out, scene


def test_frames_build_rejects_bad_pose(tmp):
    src = tmp / "src_badpose"
    scene = make_source(src)
    p = src / "data" / scene / "iphone" / "pose_intrinsic_imu.json"
    j = json.loads(p.read_text())
    for k, e in j.items():
        e["aligned_pose"] = e["pose"]           # the WRONG field: the raw ARKit frame
    p.write_text(json.dumps(j))
    try:
        run(frbuild, src_root=src, out_root=tmp / "bad_pose_out",
            scene_list=src / "splits/nvs_sem_val.txt", start=0, end=0,
            num_frames=6, n_probe=4, mesh_subsample=0, index_offsets=[-4, 0, 4])
    except SystemExit as e:
        assert e.code == 1
    else:
        raise AssertionError("the wrong pose field must fail the geometry check")
    assert not (tmp / "bad_pose_out" / scene / ".complete").exists()
    ok("a scene written from the WRONG pose field is rejected, not shipped")


def test_frames_build_rejects_shifted_rgb(tmp):
    src = tmp / "src_shift"
    scene = make_source(src, shift_rgb=4)
    try:
        run(frbuild, src_root=src, out_root=tmp / "shift_out",
            scene_list=src / "splits/nvs_sem_val.txt", start=0, end=0,
            num_frames=6, n_probe=4, mesh_subsample=0, index_offsets=[-4, 0, 4])
    except SystemExit as e:
        assert e.code == 1
    else:
        raise AssertionError("a shifted colour stream must fail the RGB index check")
    assert not (tmp / "shift_out" / scene / ".complete").exists()
    ok("a colour stream shifted against the poses is rejected, not shipped")


def test_verify(tmp, gt_src, frames_out, scene):
    import verify_scannetpp_gt as ver

    gt_out = tmp / "verify_scans3d"
    run(gtbuild, src_root=gt_src, out_root=gt_out,
        scene_list=gt_src / "splits/nvs_sem_val.txt", start=0, end=0)

    def call():
        old, sys.argv = sys.argv, [
            "verify", "--gt_root", str(gt_out), "--frames_root", str(frames_out),
            "--num_scenes", "1", "--n_probe", "3"]
        try:
            return ver.main()
        finally:
            sys.argv = old

    assert call() == 0, "the freshly built tree must verify"

    # tamper: drop a vertex from segments.json -> the ply/segIndices check must fire
    p = gt_out / scene / "segments.json"
    backup = p.read_text()
    j = json.loads(backup)
    j["segIndices"] = j["segIndices"][:-1]
    p.write_text(json.dumps(j))
    assert call() == 1, "a segIndices/ply mismatch must fail verification"
    p.write_text(backup)

    # tamper: write world-to-camera poses instead -> the convention check must fire
    stems = json.loads((frames_out / scene / "manifest.json").read_text())["sampled_stems"]
    saved = {}
    for stem in stems:
        pp = frames_out / scene / "pose" / f"{stem}.txt"
        saved[stem] = np.loadtxt(pp)
        np.savetxt(pp, np.linalg.inv(saved[stem]))
    assert call() == 1, "world-to-camera poses must fail verification"
    for stem, T in saved.items():
        np.savetxt(frames_out / scene / "pose" / f"{stem}.txt", T)
    assert call() == 0, "and the tree must verify again once restored"
    ok("verify_scannetpp_gt passes the built tree and catches GT / pose tampering")


def main():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        print("scannetpp_common")
        test_sample_indices()
        test_scale_intrinsic()
        test_depth_stream(tmp)
        test_segments_and_instances()
        test_ply_header(tmp)
        test_unproject_and_ncc()
        print("build scripts")
        test_gt_build(tmp)
        src, frames_out, scene = test_frames_build(tmp)
        test_frames_build_rejects_bad_pose(tmp)
        test_frames_build_rejects_shifted_rgb(tmp)
        print("verification")
        test_verify(tmp, src, frames_out, scene)
    print("\nAll ScanNet++ build tests passed.")


if __name__ == "__main__":
    main()
