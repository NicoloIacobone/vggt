#!/usr/bin/env python3
"""
The cross-dataset adapters of the 3D ruler (docs/todo.md 6d): `train/scannetpp3d.py`,
`train/replica3d.py`, the ScanNet200 taxonomy and the `train/datasets3d.py` registry that
`scripts/eval_3d_maskdino.py --dataset` selects through.

Standalone, CPU-only, no cluster data: every scene is synthesised in a tmpdir in the exact
on-disk format of the real tars (docs/DATASET.md §2.1/§2.2), including the Replica
binary PLY with per-FACE `object_id` and the ScanNet++ `segments.json` / `segments_anno.json`
pair.

The load-bearing test is `test_gt_as_predictions_is_perfect`: for EVERY dataset, that
dataset's own GT fed back as predictions must score exactly 1.000 / 1.000 / 1.000 through the
real evaluator. That is the gate that licensed the ScanNet evaluator (docs/MASKDINO.md §9.2),
and no dataset ships a number until it passes — here on synthetic scenes, and by
`scripts/gate_3d_gt.py` on the real tars.

    myenv/bin/python tests/test_datasets3d.py
"""

import json
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from train.benchmark3d import (AGNOSTIC_LABEL_ID, assign_instances_for_scan,  # noqa: E402
                               collapse_gt_to_class_agnostic,
                               collapse_preds_to_class_agnostic, compute_averages,
                               evaluate_matches, MIN_REGION_SIZE, OVERLAPS)
from train.datasets3d import DATASET_NAMES, DEFAULT_DATASET, get_dataset  # noqa: E402
from train.eval3d_geometry import superpoint_majority  # noqa: E402
from train.replica3d import (face_ids_to_vertex_ids, load_preseg_superpoints,  # noqa: E402
                             read_ply_faces_with_object_ids)

from gate_3d_gt import gt_as_predictions, superpoint_purity  # noqa: E402

PASSED = []


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"{name} FAILED {detail}")
    PASSED.append(name)
    print(f"  ok  {name}")


# ------------------------------------------------------------------------------------------
# synthetic scenes, in the exact on-disk formats
# ------------------------------------------------------------------------------------------

def write_replica_mesh(path, vertices, faces, object_ids):
    """A Replica `mesh_semantic.ply`: binary LE, xyz+normals+rgb verts, per-face object_id."""
    k = faces.shape[1]
    header = (
        "ply\n"
        "comment replica-instance-mesh-format v0\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"element face {len(faces)}\n"
        "property list uint8 uint32 vertex_indices\n"
        "property uint16 object_id\n"
        "end_header\n"
    ).encode()
    body = bytearray()
    for v in vertices:
        body += struct.pack("<6f3B", v[0], v[1], v[2], 0.0, 0.0, 1.0, 10, 20, 30)
    for f, oid in zip(faces, object_ids):
        body += struct.pack("<B", k)
        body += b"".join(struct.pack("<I", int(i)) for i in f)
        body += struct.pack("<H", int(oid))
    Path(path).write_bytes(header + bytes(body))


def make_replica_scene(root: Path, scene="room_0"):
    """
    Four objects in row bands of a quad grid: object 1 = `chair` and object 4 = `table`
    (both kept), object 2 = `wall` (structural, dropped), object 3 = unlabelled (class -1,
    dropped). Object ids are NOT dense and not in band order, so the adapter's re-indexing
    is exercised.
    """
    d = root / scene
    d.mkdir(parents=True, exist_ok=True)
    n = 24                                     # 24x24 grid -> 529 quads in 4 bands
    xs, ys = np.meshgrid(np.arange(n), np.arange(n))
    vertices = np.stack([xs.ravel() * 0.1, ys.ravel() * 0.1,
                         np.zeros(n * n)], axis=1).astype(np.float64)
    bands = [(6, 1), (12, 4), (18, 2), (n, 3)]
    faces, obj = [], []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            faces.append([a, a + 1, a + n + 1, a + n])
            obj.append(next(o for hi, o in bands if j < hi))
    faces = np.asarray(faces, dtype=np.int64)
    obj = np.asarray(obj, dtype=np.int64)
    write_replica_mesh(d / "mesh_semantic.ply", vertices, faces, obj)

    # id_to_label is indexed BY object id: [void, chair, wall, unlabelled, table]
    info = {
        "classes": [{"id": 5, "name": "chair"}, {"id": 7, "name": "wall"},
                    {"id": 6, "name": "table"}],
        "id_to_label": [-2, 5, 7, -1, 6],
        "objects": [{"id": 1, "class_name": "chair"}, {"id": 2, "class_name": "wall"},
                    {"id": 3, "class_name": "unknown"}, {"id": 4, "class_name": "table"}],
        "gravity_dir": [0, 0, -1],
    }
    (d / "info_semantic.json").write_text(json.dumps(info))

    # Replica's own planar preseg: a permutation of face ids grouped by segment. Two
    # segments, the first deliberately straddling the chair/table boundary so the purity
    # check has something to find — which is the real release's failure mode.
    order = np.arange(len(faces))
    sizes = [len(faces) // 2, len(faces) - len(faces) // 2]
    (d / "preseg.json").write_text(json.dumps(
        {"dataset": "./mesh.ply", "segmentation": [{"id": 0, "numPrimitives": sizes[0]},
                                                   {"id": 1, "numPrimitives": sizes[1]}]}))
    (d / "preseg.bin").write_bytes(order.astype("<u8").tobytes())
    return {"num_kept": 2, "num_vertices": len(vertices)}


def make_scannetpp_scene(root: Path, scene="7b6477cb95"):
    """
    ScanNet++ GT: `mesh.ply` + `segments.json` (identity segIndices, as the release ships)
    + `segments_anno.json`, and a shared `_metadata/` with the two class tables. One object
    is out of the instance taxonomy and must become void; one raw label must go through
    `instance_map_to` before the filter.
    """
    meta = root / "_metadata"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "top100_instance.txt").write_text("table\nchair\n")
    (meta / "map_benchmark.csv").write_text(
        "class,instance_map_to\ndesk,table\nchair,chair\nwall,wall\n")

    d = root / scene
    d.mkdir(parents=True, exist_ok=True)
    n_verts = 900
    xyz = np.stack([np.arange(n_verts) * 0.01, np.zeros(n_verts), np.zeros(n_verts)], 1)
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n_verts}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "end_header\n").encode()
    (d / "mesh.ply").write_bytes(header + xyz.astype("<f4").tobytes())

    # the release ships one segment per vertex — the degenerate case the adapter reports
    (d / "segments.json").write_text(json.dumps(
        {"sceneId:": scene, "segIndices": list(range(n_verts))}))
    (d / "segments_anno.json").write_text(json.dumps({"segGroups": [
        {"objectId": 0, "label": "desk", "segments": list(range(0, 300))},
        {"objectId": 1, "label": "chair", "segments": list(range(300, 600))},
        {"objectId": 2, "label": "wall", "segments": list(range(600, 900))},
    ]}))
    return {"num_kept": 2, "num_vertices": n_verts}


def make_scannet_scene(root: Path, tsv_path: Path, scene="scene0000_00"):
    """A ScanNet 3D GT scene: mesh + `.segs.json` superpoints + `.aggregation.json`."""
    d = root / scene
    d.mkdir(parents=True, exist_ok=True)
    n_verts, n_seg = 900, 9
    xyz = np.stack([np.arange(n_verts) * 0.01, np.zeros(n_verts), np.zeros(n_verts)], 1)
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n_verts}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "end_header\n").encode()
    (d / f"{scene}_vh_clean_2.ply").write_bytes(header + xyz.astype("<f4").tobytes())
    seg = np.repeat(np.arange(n_seg) * 7, n_verts // n_seg)     # sparse, non-dense ids
    (d / f"{scene}_vh_clean_2.0.010000.segs.json").write_text(
        json.dumps({"segIndices": seg.tolist()}))
    segs = [(seg[i * 100:(i + 1) * 100]).tolist() for i in range(n_seg)]
    (d / f"{scene}.aggregation.json").write_text(json.dumps({"segGroups": [
        {"objectId": 0, "label": "chair", "segments": sorted(set(segs[0] + segs[1]))},
        {"objectId": 1, "label": "table", "segments": sorted(set(segs[2] + segs[3]))},
        # in nyu40 AND in scannet200
        {"objectId": 2, "label": "bookshelf", "segments": sorted(set(segs[4] + segs[5]))},
        # nyu40 wall (not a benchmark class) but IS a scannet200 class
        {"objectId": 3, "label": "wall", "segments": sorted(set(segs[6] + segs[7]))},
    ]}))

    tsv_path.write_text(
        "id\traw_category\tcategory\tcount\tnyu40id\n"
        "5\tchair\tchair\t1\t5\n"
        "4\ttable\ttable\t1\t7\n"
        "8\tbookshelf\tbookshelf\t1\t10\n"
        "1\twall\twall\t1\t1\n")
    return {"num_vertices": n_verts}


# ------------------------------------------------------------------------------------------
# tests
# ------------------------------------------------------------------------------------------

def test_replica_ply_reader(tmp: Path):
    root = tmp / "rep_ply"
    make_replica_scene(root)
    v, f, o = read_ply_faces_with_object_ids(root / "room_0" / "mesh_semantic.ply")
    check("replica ply: vertex count", len(v) == 24 * 24, f"got {len(v)}")
    check("replica ply: quads", f.shape[1] == 4, f"got {f.shape}")
    check("replica ply: object ids", set(np.unique(o).tolist()) == {1, 2, 3, 4},
          f"{np.unique(o)}")
    check("replica ply: vertex coords", np.allclose(v[1], [0.1, 0.0, 0.0]), f"{v[1]}")

    # a mesh whose face block does not divide evenly must raise, not mis-parse
    bad = (root / "room_0" / "mesh_semantic.ply").read_bytes()[:-3]
    (root / "broken.ply").write_bytes(bad)
    try:
        read_ply_faces_with_object_ids(root / "broken.ply")
        raised = False
    except ValueError:
        raised = True
    check("replica ply: truncated face block raises", raised)


def test_face_to_vertex_plurality():
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 4, 5]])
    ids = np.array([7, 7, 9])
    out, ambiguous = face_ids_to_vertex_ids(faces, ids, 7)
    check("plurality: majority wins", out[0] == 7, f"{out}")
    check("plurality: unique owner", out[3] == 7 and out[5] == 9, f"{out}")
    check("plurality: untouched vertex is -1", out[6] == -1, f"{out}")
    check("plurality: ambiguity reported", abs(ambiguous - 1 / 6) < 1e-9, f"{ambiguous}")
    # tie -> lower id
    out2, _ = face_ids_to_vertex_ids(np.array([[0, 1], [0, 2]]), np.array([9, 4]), 3)
    check("plurality: tie goes to the lower id", out2[0] == 4, f"{out2}")


def test_replica_gt(tmp: Path):
    root = tmp / "rep_gt"
    info = make_replica_scene(root)
    ds = get_dataset("replica")
    gt = ds.load_scene_3d_gt(root, "room_0")
    ids = np.unique(gt["gt_ids"][gt["gt_ids"] > 0])
    check("replica gt: only the non-structural labelled objects survive",
          len(ids) == info["num_kept"] and gt["meta"]["num_instances"] == info["num_kept"],
          f"{ids} {gt['meta']}")
    check("replica gt: benchmark encoding, dense 1-based instances",
          sorted(ids.tolist()) == [1000 * AGNOSTIC_LABEL_ID + 1,
                                   1000 * AGNOSTIC_LABEL_ID + 2], f"{ids}")
    check("replica gt: wall dropped as structural",
          gt["meta"]["dropped_objects"] == {"unlabelled": 1, "structural": 1},
          f"{gt['meta']['dropped_objects']}")
    check("replica gt: identity superpoints (preseg is not used)",
          np.array_equal(gt["superpoints"], np.arange(info["num_vertices"])))
    sp = load_preseg_superpoints(root, "room_0")
    n_alt, purity = superpoint_purity(sp, gt["gt_ids"])
    check("replica gt: preseg is a real segmentation but impure", n_alt == 2 and purity < 1.0,
          f"{n_alt} segments, purity {purity}")


def test_scannetpp_gt(tmp: Path):
    root = tmp / "spp_gt"
    info = make_scannetpp_scene(root)
    ds = get_dataset("scannetpp")
    gt = ds.load_scene_3d_gt(root, "7b6477cb95")
    ids = np.unique(gt["gt_ids"][gt["gt_ids"] > 0])
    check("scannet++ gt: instance_map_to applied before the class filter",
          len(ids) == info["num_kept"], f"{ids}")
    check("scannet++ gt: dense 1-based instances under the collapsed label",
          sorted(ids.tolist()) == [1000 * AGNOSTIC_LABEL_ID + 1,
                                   1000 * AGNOSTIC_LABEL_ID + 2], f"{ids}")
    check("scannet++ gt: out-of-taxonomy object is void",
          (gt["gt_ids"][600:] == 0).all())
    check("scannet++ gt: degeneracy of segIndices reported",
          gt["meta"]["superpoints_degenerate"] is True, f"{gt['meta']}")
    # a missing _metadata/ must fail loudly — the GT is unreadable without the class tables
    (root / "_metadata" / "top100_instance.txt").unlink()
    (root / "_metadata" / "map_benchmark.csv").unlink()
    (root / "_metadata").rmdir()
    try:
        ds.load_scene_3d_gt(root, "7b6477cb95")
        raised = False
    except FileNotFoundError:
        raised = True
    check("scannet++ gt: missing _metadata raises", raised)


def test_scannet_taxonomy_switch(tmp: Path):
    root = tmp / "sn_gt"
    root.mkdir(parents=True, exist_ok=True)
    tsv = tmp / "labels.tsv"
    make_scannet_scene(root, tsv)

    v2 = get_dataset("scannetv2").load_scene_3d_gt(root, "scene0000_00", tsv)
    labels_v2 = sorted({int(i) // 1000 for i in np.unique(v2["gt_ids"]) if i})
    check("scannetv2: nyu40 labels, wall included in the encoding (evaluator filters it)",
          labels_v2 == [1, 5, 7, 10], f"{labels_v2}")

    s200 = get_dataset("scannet200").load_scene_3d_gt(root, "scene0000_00", tsv)
    labels_200 = {int(i) // 1000 for i in np.unique(s200["gt_ids"]) if i}
    inst_200 = sorted(int(i) % 1000 for i in np.unique(s200["gt_ids"]) if i)
    check("scannet200: one collapsed label", labels_200 == {AGNOSTIC_LABEL_ID},
          f"{labels_200}")
    check("scannet200: dense 1-based instances", inst_200 == [1, 2, 3, 4], f"{inst_200}")
    check("scannet200: the default taxonomy is untouched",
          not np.array_equal(v2["gt_ids"], s200["gt_ids"]))


def test_gt_as_predictions_is_perfect(tmp: Path):
    """The §9.2 licence gate, on a synthetic scene of every dataset."""
    root = tmp / "gate"
    tsv = tmp / "labels.tsv"
    make_scannet_scene(root / "sn", tsv)
    make_scannetpp_scene(root / "spp")
    make_replica_scene(root / "rep")
    where = {"scannetv2": (root / "sn", "scene0000_00"),
             "scannet200": (root / "sn", "scene0000_00"),
             "scannetpp": (root / "spp", "7b6477cb95"),
             "replica": (root / "rep", "room_0")}
    for name in DATASET_NAMES:
        ds = get_dataset(name)
        gt_root, scene = where[name]
        gt_ids = ds.load_scene_3d_gt(gt_root, scene, tsv)["gt_ids"]
        preds = gt_as_predictions(gt_ids)
        for tag, (p, g) in {
            "class-aware": (preds, gt_ids),
            "class-agnostic": (collapse_preds_to_class_agnostic(preds),
                               collapse_gt_to_class_agnostic(gt_ids)),
        }.items():
            if tag == "class-aware" and not ds.class_aware:
                continue
            gt2pred, pred2gt = assign_instances_for_scan(scene, p, g, MIN_REGION_SIZE)
            avgs = compute_averages(
                evaluate_matches({scene: {"gt": gt2pred, "pred": pred2gt}},
                                 OVERLAPS, MIN_REGION_SIZE), OVERLAPS)
            got = [float(avgs[k]) for k in ("all_ap", "all_ap_50%", "all_ap_25%")]
            check(f"GATE {name} [{tag}]: GT as predictions scores 1.000/1.000/1.000",
                  all(abs(x - 1.0) < 1e-9 for x in got), f"got {got}")


def test_registry():
    check("registry: the default dataset is scannetv2", DEFAULT_DATASET == "scannetv2")
    check("registry: four datasets",
          DATASET_NAMES == ("scannetv2", "scannet200", "scannetpp", "replica"),
          f"{DATASET_NAMES}")
    for name in DATASET_NAMES:
        ds = get_dataset(name)
        for fn in ("load_scene_3d_gt", "sample_frames", "load_poses", "load_intrinsics",
                   "load_depth", "load_color_size"):
            check(f"registry: {name}.{fn} is callable", callable(getattr(ds, fn)))
    check("registry: only scannetv2 is class-aware",
          [get_dataset(n).class_aware for n in DATASET_NAMES] == [True, False, False, False])
    check("registry: scannet200 keeps wall/floor predictions (they are valid classes there)",
          get_dataset("scannet200").drop_wall_floor_predictions is False)
    check("registry: replica frames are png", get_dataset("replica").color_ext == ".png")
    try:
        get_dataset("nyu")
        raised = False
    except ValueError:
        raised = True
    check("registry: an unknown dataset raises", raised)


def test_eval_script_tags_the_dataset():
    """A second dataset must not overwrite the first one's json (the §9.6 lesson)."""
    import argparse

    import eval_3d_maskdino as ev
    parser = ev.build_argparser()
    check("eval script: dataset is result-affecting", "dataset" in ev.RESULT_AFFECTING)
    ckpt = Path("/tmp/run/checkpoint_best_bundle.pth")
    defaults = parser.parse_args(["--checkpoint", str(ckpt), "--frames_root", "f",
                                  "--gt_root", "g"])
    spp = parser.parse_args(["--checkpoint", str(ckpt), "--frames_root", "f",
                             "--gt_root", "g", "--dataset", "scannetpp"])
    check("eval script: the default keeps the documented bare name",
          ev.default_out_path(ckpt, defaults, parser).name ==
          "eval3d_checkpoint_best_bundle.json",
          str(ev.default_out_path(ckpt, defaults, parser)))
    check("eval script: a non-default dataset tags the filename",
          ev.default_out_path(ckpt, spp, parser).name ==
          "eval3d_checkpoint_best_bundle__datasetscannetpp.json",
          str(ev.default_out_path(ckpt, spp, parser)))
    assert isinstance(defaults, argparse.Namespace)


def test_superpoint_majority_identity_fast_path():
    """The identity-superpoint fast path must be bit-identical to the general path."""
    rng = np.random.default_rng(0)
    votes = rng.integers(0, 3, size=(500, 7)).astype(np.int32)
    votes[::13] = 0                                   # some vertices receive nothing
    identity = np.arange(500)
    fast = superpoint_majority(votes, identity)
    # the general path, with the same partition expressed as non-contiguous ids
    shuffled = identity * 3 + 1
    general = superpoint_majority(votes, shuffled)
    reference = np.where(votes.max(axis=1) == 0, -1, votes.argmax(axis=1))
    check("superpoint fast path: identity == general", np.array_equal(fast, general))
    check("superpoint fast path: identity == per-vertex argmax",
          np.array_equal(fast, reference))
    # and the real grouping still groups
    grouped = superpoint_majority(votes, np.repeat(np.arange(100), 5))
    check("superpoint fast path: grouping still applies when superpoints are real",
          len(np.unique(grouped[:5])) == 1)


def test_replica_frames(tmp: Path):
    """The frame side: png colour, 4x4 c2w poses, one shared intrinsic, mm depth."""
    from PIL import Image

    scene_dir = tmp / "rep_frames" / "room_0"
    for sub in ("color", "depth", "pose", "intrinsic"):
        (scene_dir / sub).mkdir(parents=True, exist_ok=True)
    stems = [f"{i:06d}" for i in range(4)]
    for i, stem in enumerate(stems):
        Image.new("RGB", (12, 8), (i, i, i)).save(scene_dir / "color" / f"{stem}.png")
        Image.fromarray((np.full((8, 12), 1500 + i, dtype=np.uint16))).save(
            scene_dir / "depth" / f"{stem}.png")
        pose = np.eye(4)
        pose[0, 3] = i
        np.savetxt(scene_dir / "pose" / f"{stem}.txt", pose)
    (scene_dir / "intrinsic" / "intrinsic_depth.txt").write_text(
        "600.0 0 599.5 0\n0 600.0 339.5 0\n0 0 1 0\n0 0 0 1\n")

    ds = get_dataset("replica")
    got = ds.sample_frames(scene_dir, None, require_depth=True)
    check("replica frames: all stems found", got == stems, f"{got}")
    check("replica frames: subsampling is uniform",
          ds.sample_frames(scene_dir, 2) == [stems[0], stems[-1]],
          f"{ds.sample_frames(scene_dir, 2)}")
    K = ds.load_intrinsics(scene_dir)
    check("replica frames: one intrinsic serves colour and depth",
          np.array_equal(K["color"], K["depth"]) and K["color"][0, 0] == 600.0)
    depth = ds.load_depth(scene_dir, stems)
    check("replica frames: depth is millimetres -> meters",
          abs(float(depth[0, 0, 0]) - 1.5) < 1e-6, f"{depth[0, 0, 0]}")
    check("replica frames: colour size", ds.load_color_size(scene_dir, stems) == (12, 8))
    poses = ds.load_poses(scene_dir)
    check("replica frames: poses are 4x4", poses[stems[2]].shape == (4, 4)
          and poses[stems[2]][0, 3] == 2)


def main():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        print("replica PLY + face->vertex")
        test_replica_ply_reader(tmp)
        test_face_to_vertex_plurality()
        print("GT construction")
        test_replica_gt(tmp)
        test_scannetpp_gt(tmp)
        test_scannet_taxonomy_switch(tmp)
        print("the §9.2 licence gate")
        test_gt_as_predictions_is_perfect(tmp)
        print("registry + script wiring")
        test_registry()
        test_eval_script_tags_the_dataset()
        test_superpoint_majority_identity_fast_path()
        print("frames")
        test_replica_frames(tmp)
    print(f"\n✓ all {len(PASSED)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
