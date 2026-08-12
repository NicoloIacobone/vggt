"""
Replica data for the 3D ruler (docs/todo.md 6d) — the sibling of `train/scannet3d.py`.

Reads the two tars of `docs/DATASET.md` §2.2 (`replica_3d_gt_8.tar.zst` +
`replica_frames_8.tar.zst`, the 8 scenes FAST3DIS reports on: room_0-2, office_0-4) and
exposes the SAME interface every dataset adapter exposes (`train/datasets3d.py`):

    load_scene_3d_gt, sample_frames, load_poses, load_intrinsics, load_depth,
    load_color_size

Four things are specific to Replica. Each was established against the downloaded bytes —
none is assumed:

**1. The GT lives on FACES, not vertices.** `habitat/mesh_semantic.ply` carries
`property uint16 object_id` on the *face* element (verified in the header, and re-verified
here per scene). Vertices are shared across objects at object boundaries — 2.1-2.2 % of them
on room_0 / office_0 — so a vertex takes the object of the plurality of its incident faces,
ties going to the lower object id. The benchmark evaluator scores per vertex, hence the
conversion.

**2. `object_id` is an instance; its class comes from `info_semantic.json`.**
`id_to_label[object_id]` is the semantic class id (`-1` = unlabelled, `-2` = the void entry
at index 0), and `classes` names them.

**3. The room shell is not an instance.** `wall`, `floor` and `ceiling` objects are dropped
from the GT — the same convention the ScanNet benchmark applies (its 18 classes exclude wall
and floor) and that ScanNet++'s `top100_instance.txt` applies (no wall/floor/ceiling). The
prediction side drops the matching classes, so the two sets stay symmetric
(`train/datasets3d.py::DROP_WALL_FLOOR_PREDICTIONS`). Objects whose class id is not positive
(unlabelled) are dropped too. This is OUR GT construction, and every number from it must say
so (the approved plan's §5 item 10).

**4. No superpoints are used.** Replica ships its own over-segmentation (`preseg.json` +
`preseg.bin`: a permutation of face ids grouped into planar segments, 675 segments on
room_0), but it is a *planar* segmentation, not an object over-segmentation: its
face-weighted purity against the GT objects is only **0.796** on room_0 and **0.909** on
office_0 (measured 2026-08-09; reproduce with `scripts/gate_3d_gt.py --report_superpoints`).
Segments that straddle two objects would cap the achievable AP, so the adapter returns
identity superpoints and `superpoint_majority` degenerates to a per-vertex vote — exactly
as it already does on ScanNet++.

Frames come from the vMAP repack (50 uniformly sampled views/scene of the iMAP trajectory):
`color/<stem>.png` 1200x680, `pose/<stem>.txt` 4x4 camera-to-world (`traj_w_c.txt`),
`depth/<stem>.png` uint16 **millimetres**, and ONE intrinsic file,
`intrinsic/intrinsic_depth.txt`. The colour and depth cameras are the same render at the
same resolution, so that one intrinsic serves both.

> **The intrinsics are a documented FALLBACK.** No camera-parameter file exists anywhere in
> the downloaded tree, so the build wrote the standard habitat/NICE-SLAM/vMAP values
> (fx=fy=600, cx=599.5, cy=339.5 at 1200x680) and flagged them `FALLBACK` per scene in
> `REPORT.json`. They are not a guess left unchecked: unprojecting the sensor depth with
> them and `traj_w_c` read as camera-to-world lands **0.55-0.57 cm (median)** from the mesh
> on every probe frame of room_0 and office_0, while a wrong depth scale (÷6553.5, the
> NICE-SLAM constant) lands 65-91 cm away. `scripts/gate_3d_gt.py --dataset replica` re-runs
> that check.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from train.benchmark3d import AGNOSTIC_LABEL_ID
from train.scannet3d import _PLY_DTYPES, load_frames25k_poses

# The room shell, dropped from the GT instance set — see the module docstring, point 3.
STRUCTURAL_CLASSES = ("wall", "floor", "ceiling")

# vMAP's Replica renders store depth as uint16 millimetres. NOT the NICE-SLAM 6553.5
# constant, which is off by 6.55x — the module docstring records the measurement that
# settles it, and `scripts/gate_3d_gt.py` re-runs it.
DEPTH_UNITS_PER_METER = 1000.0

COLOR_EXT = ".png"


# ------------------------------------------------------------------------------------------
# PLY with per-face object ids
# ------------------------------------------------------------------------------------------

def _parse_ply_header(f):
    """(format, [(element name, count, props)]) from an open binary PLY; leaves f at body."""
    if f.readline().strip() != b"ply":
        raise ValueError("not a PLY file")
    fmt, elements = None, []
    while True:
        line = f.readline()
        if not line:
            raise ValueError("header ended without end_header")
        tok = line.decode("ascii", "replace").split()
        if not tok or tok[0] == "comment":
            continue
        if tok[0] == "format":
            fmt = tok[1]
        elif tok[0] == "element":
            elements.append((tok[1], int(tok[2]), []))
        elif tok[0] == "property":
            if not elements:
                raise ValueError("property before any element")
            if tok[1] == "list":
                elements[-1][2].append(("list", _PLY_DTYPES[tok[2]],
                                        _PLY_DTYPES[tok[3]], tok[4]))
            else:
                elements[-1][2].append(("scalar", _PLY_DTYPES[tok[1]], tok[2]))
        elif tok[0] == "end_header":
            break
    return fmt, elements


def read_ply_faces_with_object_ids(path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    (vertices [V,3] float64, faces [F,k] int64, object_ids [F] int64) of a Replica
    `mesh_semantic.ply`.

    Only the binary_little_endian layout Replica ships is supported, and the faces are
    required to be uniform (all k-gons — they are quads in every scene of the release), so
    the whole face block parses as one structured array instead of a per-face Python loop
    over ~1 M faces. A non-uniform mesh raises rather than being silently mis-parsed.
    """
    path = Path(path)
    with open(path, "rb") as f:
        fmt, elements = _parse_ply_header(f)
        if fmt != "binary_little_endian":
            raise ValueError(f"{path}: unsupported PLY format {fmt}")
        by_name = {name: (count, props) for name, count, props in elements}
        if "vertex" not in by_name or "face" not in by_name:
            raise ValueError(f"{path}: expected both a vertex and a face element, got "
                             f"{[e[0] for e in elements]}")
        if [e[0] for e in elements][:2] != ["vertex", "face"]:
            raise ValueError(f"{path}: expected the vertex element to precede the face one")

        n_verts, vprops = by_name["vertex"]
        if any(p[0] == "list" for p in vprops):
            raise ValueError(f"{path}: list property in the vertex element")
        vdtype = np.dtype([(p[2], "<" + p[1]) for p in vprops])
        raw = f.read(n_verts * vdtype.itemsize)
        if len(raw) < n_verts * vdtype.itemsize:
            raise ValueError(f"{path}: truncated vertex block")
        rec = np.frombuffer(raw, dtype=vdtype, count=n_verts)
        vertices = np.stack([rec["x"], rec["y"], rec["z"]], axis=1).astype(np.float64)

        n_faces, fprops = by_name["face"]
        lists = [p for p in fprops if p[0] == "list"]
        if len(lists) != 1 or lists[0][3] != "vertex_indices":
            raise ValueError(f"{path}: expected exactly one list property "
                             f"'vertex_indices' on the face element")
        if not any(p[0] == "scalar" and p[2] == "object_id" for p in fprops):
            raise ValueError(f"{path}: the face element carries no 'object_id' property — "
                             f"this is not a Replica semantic mesh")
        fbytes = f.read()

    _, count_dt, index_dt, _ = lists[0]
    k = int(np.frombuffer(fbytes[:np.dtype(count_dt).itemsize], dtype="<" + count_dt)[0])
    fields = [("_k", "<" + count_dt), ("vertex_indices", "<" + index_dt, (k,))]
    for p in fprops:
        if p[0] == "scalar":
            fields.append((p[2], "<" + p[1]))
    fdtype = np.dtype(fields)
    if fdtype.itemsize * n_faces != len(fbytes):
        raise ValueError(f"{path}: face block is {len(fbytes)} bytes, not "
                         f"{fdtype.itemsize} x {n_faces} — the faces are not all "
                         f"{k}-gons, which this reader requires")
    frec = np.frombuffer(fbytes, dtype=fdtype, count=n_faces)
    if not (frec["_k"] == k).all():
        raise ValueError(f"{path}: mixed face sizes (first is a {k}-gon)")
    faces = frec["vertex_indices"].astype(np.int64)
    if faces.size and (faces.max() >= n_verts or faces.min() < 0):
        raise ValueError(f"{path}: face index out of range [0, {n_verts})")
    return vertices, faces, frec["object_id"].astype(np.int64)


def face_ids_to_vertex_ids(faces: np.ndarray, face_ids: np.ndarray,
                           num_vertices: int) -> Tuple[np.ndarray, float]:
    """
    Per-vertex id [V] by plurality over the incident faces (ties -> lower id), plus the
    fraction of vertices whose incident faces disagreed.

    Vertices touched by no face get -1. Implemented by sorting the (vertex, id) pairs, so it
    stays linear-ish at ~1 M vertices instead of materialising a [V, num_ids] histogram.
    """
    if len(faces) != len(face_ids):
        raise ValueError(f"{len(faces)} faces but {len(face_ids)} ids")
    out = np.full(num_vertices, -1, dtype=np.int64)
    if len(faces) == 0:
        return out, 0.0

    flat_v = np.asarray(faces).reshape(-1)
    flat_i = np.repeat(np.asarray(face_ids), np.shape(faces)[1])
    pairs, counts = np.unique(np.stack([flat_v, flat_i], axis=1), axis=0,
                              return_counts=True)          # sorted by vertex, then id
    v, i = pairs[:, 0], pairs[:, 1]
    starts = np.nonzero(np.r_[True, v[1:] != v[:-1]])[0]
    group_of = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(v)]))
    best = np.repeat(np.maximum.reduceat(counts, starts), np.diff(np.r_[starts, len(v)]))
    is_best = counts == best
    # first best row of each group == plurality winner, ties to the lower id
    _, first = np.unique(group_of[is_best], return_index=True)
    rows = np.nonzero(is_best)[0][first]
    out[v[starts]] = i[rows]
    ambiguous = float((np.diff(np.r_[starts, len(v)]) > 1).mean())
    return out, ambiguous


# ------------------------------------------------------------------------------------------
# GT
# ------------------------------------------------------------------------------------------

def load_semantic_info(scene_dir) -> Dict:
    """
    `info_semantic.json` as `{"class_name_of_object": {object_id: name}}` plus the raw
    tables. `id_to_label[object_id]` is the class id; `classes` names the class ids.
    """
    info = json.loads((Path(scene_dir) / "info_semantic.json").read_text())
    names = {int(c["id"]): str(c["name"]) for c in info["classes"]}
    id_to_label = np.asarray(info["id_to_label"], dtype=np.int64)
    return {"names": names, "id_to_label": id_to_label,
            "num_objects": len(info.get("objects", []))}


def load_scene_3d_gt(gt_root, scene: str, tsv_path=None) -> Dict[str, np.ndarray]:
    """
    vertices [V,3], superpoints [V], gt_ids [V], meta — one Replica scene.

    `gt_ids` uses the benchmark encoding `1000 * label + instance` with the label collapsed
    to `AGNOSTIC_LABEL_ID` (Replica's taxonomy has no correspondence with our 19 ScanNet
    classes) and a dense 1-based instance index over the kept objects, so
    `train/benchmark3d.py` scores it unchanged. `tsv_path` is accepted for interface
    symmetry with the ScanNet adapters and ignored.
    """
    scene_dir = Path(gt_root) / scene
    vertices, faces, face_obj = read_ply_faces_with_object_ids(
        scene_dir / "mesh_semantic.ply")
    vertex_obj, ambiguous = face_ids_to_vertex_ids(faces, face_obj, len(vertices))

    info = load_semantic_info(scene_dir)
    id_to_label, names = info["id_to_label"], info["names"]
    if face_obj.size and face_obj.max() >= len(id_to_label):
        raise ValueError(f"{scene}: object id {int(face_obj.max())} has no entry in "
                         f"info_semantic.json's id_to_label ({len(id_to_label)} entries)")

    kept, dropped = [], {"unlabelled": 0, "structural": 0}
    for obj_id in np.unique(face_obj):
        class_id = int(id_to_label[obj_id])
        if class_id <= 0:
            dropped["unlabelled"] += 1
            continue
        if names.get(class_id, "") in STRUCTURAL_CLASSES:
            dropped["structural"] += 1
            continue
        kept.append((int(obj_id), names.get(class_id, f"class_{class_id}")))
    if not kept:
        raise ValueError(f"{scene}: no instances survived the class filter")
    if len(kept) >= 1000:
        raise ValueError(f"{scene}: {len(kept)} instances do not fit the "
                         f"1000 * label + instance encoding")

    gt_ids = np.zeros(len(vertices), dtype=np.int64)
    n_empty = 0
    for k, (obj_id, _) in enumerate(kept):
        sel = vertex_obj == obj_id
        if not sel.any():
            n_empty += 1
            continue
        gt_ids[sel] = 1000 * AGNOSTIC_LABEL_ID + (k + 1)

    return {
        "vertices": vertices,
        # Replica's own preseg is a PLANAR segmentation and straddles objects (docstring,
        # point 4): the vote stays per vertex.
        "superpoints": np.arange(len(vertices), dtype=np.int64),
        "gt_ids": gt_ids,
        "meta": {
            "num_instances": len(kept) - n_empty,
            "num_objects": int(len(np.unique(face_obj))),
            "dropped_objects": dropped,
            "ambiguous_vertex_frac": ambiguous,
            "labelled_vertex_frac": float((gt_ids > 0).mean()),
            "labels": sorted({name for _, name in kept}),
        },
    }


def load_preseg_superpoints(gt_root, scene: str) -> np.ndarray:
    """
    Per-vertex segment id [V] from Replica's own `preseg.json` + `preseg.bin`.

    `preseg.bin` is a permutation of FACE ids; `preseg.json`'s `segmentation[k].numPrimitives`
    say how many of them belong to segment k, in order. Faces are converted to vertices by
    the same plurality rule the GT uses.

    NOT used by the evaluation — this is the planar over-segmentation whose purity against
    the GT objects is measured (and found insufficient) in the module docstring, point 4.
    `scripts/gate_3d_gt.py --report_superpoints` calls it to reproduce that number.
    """
    scene_dir = Path(gt_root) / scene
    vertices, faces, _ = read_ply_faces_with_object_ids(scene_dir / "mesh_semantic.ply")
    sizes = [int(s["numPrimitives"])
             for s in json.loads((scene_dir / "preseg.json").read_text())["segmentation"]]
    face_order = np.frombuffer((scene_dir / "preseg.bin").read_bytes(), dtype="<u8")
    if len(face_order) != len(faces) or sum(sizes) != len(faces):
        raise ValueError(f"{scene}: preseg covers {len(face_order)} / {sum(sizes)} faces, "
                         f"but the mesh has {len(faces)}")
    face_seg = np.empty(len(faces), dtype=np.int64)
    offset = 0
    for k, size in enumerate(sizes):
        face_seg[face_order[offset:offset + size]] = k
        offset += size
    seg, _ = face_ids_to_vertex_ids(faces, face_seg, len(vertices))
    return seg


# ------------------------------------------------------------------------------------------
# Frames
# ------------------------------------------------------------------------------------------

def sample_frames(scene_dir, num_frames: Optional[int] = None,
                  require_depth: bool = False) -> List[str]:
    """
    Frame stems of a Replica scene, evenly subsampled to at most `num_frames` (None = all
    50). Mirrors `train/scannet3d.py::sample_frames25k` — a stem qualifies only with a
    finite pose and a colour PNG (and a depth PNG when `require_depth`).
    """
    scene_dir = Path(scene_dir)
    poses = load_frames25k_poses(scene_dir)
    stems = [s for s in sorted(poses)
             if (scene_dir / "color" / f"{s}{COLOR_EXT}").exists()
             and (not require_depth or (scene_dir / "depth" / f"{s}.png").exists())]
    if not stems:
        raise ValueError(f"{scene_dir}: no usable frames (finite pose + colour png"
                         + (" + depth png)" if require_depth else ")"))
    if num_frames is not None and len(stems) > num_frames:
        idx = np.linspace(0, len(stems) - 1, num_frames).round().astype(int)
        stems = [stems[i] for i in sorted(set(idx.tolist()))]
    return stems


def load_poses(scene_dir) -> Dict[str, np.ndarray]:
    """stem -> camera-to-world 4x4 (vMAP's `traj_w_c.txt`, one line per frame index)."""
    return load_frames25k_poses(scene_dir)


def load_intrinsics(scene_dir) -> Dict[str, np.ndarray]:
    """
    `{"color": K, "depth": K}` — ONE 3x3, used for both.

    Replica's colour and depth are the same render at the same 1200x680 resolution, and the
    build wrote a single `intrinsic/intrinsic_depth.txt` (the FALLBACK values; see the
    module docstring for the measurement that validates them).
    """
    path = Path(scene_dir) / "intrinsic" / "intrinsic_depth.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing (needed by the GT-projection transfer)")
    mat = np.loadtxt(path, dtype=np.float64)
    if mat.shape not in ((3, 3), (4, 4)):
        raise ValueError(f"{path}: expected a 3x3 or 4x4 intrinsic, got {mat.shape}")
    if not np.isfinite(mat).all():
        raise ValueError(f"{path}: non-finite intrinsic")
    K = mat[:3, :3].copy()
    return {"color": K, "depth": K.copy()}


def load_depth(scene_dir, stems: List[str]) -> np.ndarray:
    """Sensor depth [S, 680, 1200] in METERS (uint16 millimetres on disk, see the module
    docstring for how the scale was established)."""
    from PIL import Image

    scene_dir = Path(scene_dir)
    maps = []
    for stem in stems:
        arr = np.asarray(Image.open(scene_dir / "depth" / f"{stem}.png"))
        if arr.ndim != 2:
            raise ValueError(f"{scene_dir}/depth/{stem}.png: expected a single-channel "
                             f"depth png, got shape {arr.shape}")
        maps.append(arr.astype(np.float32) / DEPTH_UNITS_PER_METER)
    if len({m.shape for m in maps}) != 1:
        raise ValueError(f"{scene_dir}: depth maps of differing sizes "
                         f"{sorted({m.shape for m in maps})}")
    return np.stack(maps)


def load_color_size(scene_dir, stems: List[str]) -> Tuple[int, int]:
    """(width, height) of the colour pngs — 1200x680. Raises if the frames disagree."""
    from PIL import Image

    scene_dir = Path(scene_dir)
    sizes = {Image.open(scene_dir / "color" / f"{s}{COLOR_EXT}").size for s in stems}
    if len(sizes) != 1:
        raise ValueError(f"{scene_dir}: colour frames of differing sizes {sorted(sizes)}")
    return sizes.pop()
