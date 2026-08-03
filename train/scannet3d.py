"""
ScanNet 3D benchmark data for the 3D ruler (docs/MASKDINO.md §9, docs/todo.md 1d).

Loads what the official 3D instance evaluation needs, with no dependency beyond numpy:

  - the benchmark mesh vertices  (`<scene>_vh_clean_2.ply`, minimal PLY reader — plyfile is
    not in myenv and only the vertex block is needed),
  - its superpoint over-segmentation  (`<scene>_vh_clean_2.0.010000.segs.json`),
  - the per-vertex GT instance ids in the benchmark's `1000 * nyu40_label + instance`
    encoding, built from `<scene>.aggregation.json` + `scannetv2-labels.combined.tsv` —
    the same construction as the official `export_train_mesh_for_evaluation.py`,
  - the scannet_frames_25k frames + camera-to-world poses of a scene (the eval's input
    frames; poses are used ONLY for eval-time Sim(3) registration, never at inference).

Class bookkeeping: the benchmark scores 18 classes (nyu40 ids in `BENCHMARK_CLASS_IDS`) —
no wall/floor, but WITH otherfurniture, which our 19-class head cannot predict. The head's
dataset class indices 1..19 map to nyu40 via `SCANNET_IDX_TO_NYU40`; wall/floor (nyu40 1/2)
are outside the benchmark set and must be dropped before voting.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from data.scannet_overfit import SCANNET_CLASSES

# nyu40 id of each name in data/scannet_overfit.py::SCANNET_CLASSES (dataset index = list
# position + 1). The 19-logit head covers positions 0..18 (wall..bathtub); otherfurniture
# (nyu40 39) exists only in the GT taxonomy.
_NYU40_BY_NAME = {
    "wall": 1, "floor": 2, "cabinet": 3, "bed": 4, "chair": 5, "sofa": 6, "table": 7,
    "door": 8, "window": 9, "bookshelf": 10, "picture": 11, "counter": 12, "desk": 14,
    "curtain": 16, "refrigerator": 24, "shower curtain": 28, "toilet": 33, "sink": 34,
    "bathtub": 36, "otherfurniture": 39,
}
SCANNET_IDX_TO_NYU40 = {i + 1: _NYU40_BY_NAME[n] for i, n in enumerate(SCANNET_CLASSES)}

# The official benchmark's 18 instance classes (ScanNet BenchmarkScripts
# evaluate_semantic_instance.py: VALID_CLASS_IDS / CLASS_LABELS), in the official order.
BENCHMARK_CLASS_IDS = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)
BENCHMARK_CLASS_NAMES = (
    "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf", "picture",
    "counter", "desk", "curtain", "refrigerator", "shower curtain", "toilet", "sink",
    "bathtub", "otherfurniture",
)


# ------------------------------------------------------------------------------------------
# PLY
# ------------------------------------------------------------------------------------------

_PLY_DTYPES = {
    "char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
    "short": "i2", "ushort": "u2", "int16": "i2", "uint16": "u2",
    "int": "i4", "uint": "u4", "int32": "i4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def read_ply_vertices(path) -> np.ndarray:
    """
    The x/y/z of a PLY file's vertex element, as float64 [V, 3].

    Supports the two formats ScanNet ships (binary_little_endian and ascii) for the vertex
    element only; list properties before the vertex element are not supported (no ScanNet
    file has them — faces come after the vertices).
    """
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path}: not a PLY file")
        fmt, num_verts, props, in_vertex = None, None, [], False
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: header ended without end_header")
            tok = line.decode("ascii", "replace").split()
            if not tok:
                continue
            if tok[0] == "format":
                fmt = tok[1]
            elif tok[0] == "element":
                in_vertex = tok[1] == "vertex"
                if in_vertex:
                    num_verts = int(tok[2])
                elif num_verts is not None:
                    break                      # vertex element fully described
            elif tok[0] == "property" and in_vertex:
                if tok[1] == "list":
                    raise ValueError(f"{path}: list property in vertex element")
                props.append((tok[2], _PLY_DTYPES[tok[1]]))
            if tok[0] == "end_header":
                break
        # If we broke on the next element, still need to consume up to end_header.
        while tok[0] != "end_header":
            tok = f.readline().decode("ascii", "replace").split() or ["#"]
        if num_verts is None:
            raise ValueError(f"{path}: no vertex element")

        names = [n for n, _ in props]
        if fmt == "ascii":
            rows = [f.readline().split() for _ in range(num_verts)]
            data = np.array(rows, dtype=np.float64)
            xyz = data[:, [names.index("x"), names.index("y"), names.index("z")]]
        elif fmt == "binary_little_endian":
            dtype = np.dtype([(n, "<" + d) for n, d in props])
            raw = f.read(num_verts * dtype.itemsize)
            if len(raw) < num_verts * dtype.itemsize:
                raise ValueError(f"{path}: truncated vertex block")
            rec = np.frombuffer(raw, dtype=dtype, count=num_verts)
            xyz = np.stack([rec["x"], rec["y"], rec["z"]], axis=1).astype(np.float64)
        else:
            raise ValueError(f"{path}: unsupported format {fmt}")
    return xyz


# ------------------------------------------------------------------------------------------
# Superpoints + per-vertex GT ids (the official benchmark encoding)
# ------------------------------------------------------------------------------------------

def load_superpoints(segs_path) -> np.ndarray:
    """Per-vertex superpoint id [V] from a `.segs.json` (values are arbitrary, not dense)."""
    return np.asarray(json.loads(Path(segs_path).read_text())["segIndices"], dtype=np.int64)


def load_raw_to_nyu40(tsv_path) -> Dict[str, int]:
    """raw_category -> nyu40 id from scannetv2-labels.combined.tsv."""
    lines = Path(tsv_path).read_text().splitlines()
    header = lines[0].split("\t")
    i_raw, i_nyu = header.index("raw_category"), header.index("nyu40id")
    out = {}
    for line in lines[1:]:
        cols = line.split("\t")
        if len(cols) > max(i_raw, i_nyu) and cols[i_nyu]:
            out[cols[i_raw]] = int(cols[i_nyu])
    return out


def build_gt_ids(superpoints: np.ndarray, agg_path, raw_to_nyu40: Dict[str, int]
                 ) -> np.ndarray:
    """
    Per-vertex GT id [V] in the benchmark encoding: `1000 * nyu40_label + (objectId + 1)`,
    0 for unannotated vertices — the same construction as the official
    `export_train_mesh_for_evaluation.py` (instances of ALL classes are encoded; the
    evaluator itself selects the 18 benchmark classes).
    """
    agg = json.loads(Path(agg_path).read_text())
    gt = np.zeros(len(superpoints), dtype=np.int64)
    sp_to_verts: Dict[int, np.ndarray] = {}
    for group in agg["segGroups"]:
        label = raw_to_nyu40.get(group["label"], 0)
        if label == 0:
            continue                       # raw category outside the nyu40 taxonomy
        gid = 1000 * label + group["objectId"] + 1
        for seg in group["segments"]:
            verts = sp_to_verts.get(seg)
            if verts is None:
                verts = sp_to_verts[seg] = np.nonzero(superpoints == seg)[0]
            gt[verts] = gid
    return gt


def load_scene_3d_gt(gt_root, scene: str, tsv_path) -> Dict[str, np.ndarray]:
    """vertices [V,3], superpoints [V], gt_ids [V] of one scene from the 3D GT tree."""
    scene_dir = Path(gt_root) / scene
    vertices = read_ply_vertices(scene_dir / f"{scene}_vh_clean_2.ply")
    superpoints = load_superpoints(scene_dir / f"{scene}_vh_clean_2.0.010000.segs.json")
    if len(superpoints) != len(vertices):
        raise ValueError(f"{scene}: {len(vertices)} vertices but {len(superpoints)} seg ids")
    gt_ids = build_gt_ids(superpoints, scene_dir / f"{scene}.aggregation.json",
                          load_raw_to_nyu40(tsv_path))
    return {"vertices": vertices, "superpoints": superpoints, "gt_ids": gt_ids}


# ------------------------------------------------------------------------------------------
# scannet_frames_25k input frames
# ------------------------------------------------------------------------------------------

def load_frames25k_poses(scene_dir) -> Dict[str, np.ndarray]:
    """
    frame stem -> camera-to-world 4x4 for every finite pose in `<scene>/pose/`.
    Frames whose pose contains -inf (a known rare export defect) are silently absent —
    callers sample frames from the returned keys, so those frames are never used.
    """
    out = {}
    for p in sorted(Path(scene_dir).glob("pose/*.txt")):
        mat = np.loadtxt(p)
        if mat.shape == (4, 4) and np.isfinite(mat).all():
            out[p.stem] = mat
    return out


def sample_frames25k(scene_dir, num_frames: Optional[int] = None) -> List[str]:
    """
    Frame stems of a 25k scene, evenly subsampled to at most `num_frames` (None = all).
    Only frames with a finite pose and an existing color jpg qualify.
    """
    scene_dir = Path(scene_dir)
    poses = load_frames25k_poses(scene_dir)
    stems = [s for s in sorted(poses) if (scene_dir / "color" / f"{s}.jpg").exists()]
    if not stems:
        raise ValueError(f"{scene_dir}: no usable frames (finite pose + color jpg)")
    if num_frames is not None and len(stems) > num_frames:
        idx = np.linspace(0, len(stems) - 1, num_frames).round().astype(int)
        stems = [stems[i] for i in sorted(set(idx.tolist()))]
    return stems
