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
    frames), and — for `--transfer_mode gt_projection` only (docs/MASKDINO.md §9.9) — the
    per-scene color/depth intrinsics and the 16-bit sensor depth maps.

GT poses, GT intrinsics and sensor depth are EVAL-TIME TRANSFER machinery: they place a
finished prediction on the benchmark mesh (Sim(3) registration in the default protocol, the
projection in `gt_projection`) and never reach inference. The model sees only images.

Class bookkeeping: the benchmark scores 18 classes (nyu40 ids in `BENCHMARK_CLASS_IDS`) —
no wall/floor, but WITH otherfurniture, which our 19-class head cannot predict. The head's
dataset class indices 1..19 map to nyu40 via `SCANNET_IDX_TO_NYU40`; wall/floor (nyu40 1/2)
are outside the benchmark set and must be dropped before voting.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from data.scannet_overfit import SCANNET_CLASSES

# The ScanNet labels table every taxonomy here is read through (raw_category -> nyu40id /
# id). It lives on group storage next to the tars, not in the repo.
DEFAULT_TSV = ("/cluster/work/igp_psr/niacobone/distillation/dataset/scannet/"
               "scannetv2-labels.combined.tsv")

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
    return _load_raw_to_label(tsv_path, "nyu40id")


def load_raw_to_scannet_id(tsv_path) -> Dict[str, int]:
    """
    raw_category -> raw ScanNet label id (the TSV's `id` column, 1..1191) — the taxonomy
    ScanNet200 is defined over (`data/scannet200_constants.py`). NOT interchangeable with
    the nyu40 ids above.
    """
    return _load_raw_to_label(tsv_path, "id")


def _load_raw_to_label(tsv_path, column: str) -> Dict[str, int]:
    lines = Path(tsv_path).read_text().splitlines()
    header = lines[0].split("\t")
    i_raw, i_label = header.index("raw_category"), header.index(column)
    out = {}
    for line in lines[1:]:
        cols = line.split("\t")
        if len(cols) > max(i_raw, i_label) and cols[i_label]:
            out[cols[i_raw]] = int(cols[i_label])
    return out


def build_gt_ids(superpoints: np.ndarray, agg_path, raw_to_label: Dict[str, int],
                 valid_label_ids=None, collapse_to: Optional[int] = None) -> np.ndarray:
    """
    Per-vertex GT id [V] in the benchmark encoding: `1000 * label + (objectId + 1)`,
    0 for unannotated vertices — the same construction as the official
    `export_train_mesh_for_evaluation.py`.

    Defaults reproduce the v2 benchmark exactly: `raw_to_label` is the nyu40 map, instances
    of ALL classes are encoded, and the evaluator itself selects the 18 benchmark classes.

    The two optional arguments serve the ScanNet200 taxonomy (docs/todo.md 6d) and are
    inert when omitted:
      `valid_label_ids`  keep only objects whose label is in this set (ScanNet200's 200
                         raw ids; the v2 path keeps everything and lets the evaluator filter),
      `collapse_to`      write this label id instead of the object's own — how a taxonomy
                         our 19-class head cannot address is reported CLASS-AGNOSTICALLY
                         (`train/benchmark3d.py::AGNOSTIC_LABEL_ID`). The instance component
                         is re-indexed densely so it cannot collide after the collapse.
    """
    agg = json.loads(Path(agg_path).read_text())
    gt = np.zeros(len(superpoints), dtype=np.int64)
    valid = None if valid_label_ids is None else set(valid_label_ids)
    sp_to_verts: Dict[int, np.ndarray] = {}
    n_kept = 0
    for group in agg["segGroups"]:
        label = raw_to_label.get(group["label"], 0)
        if label == 0:
            continue                       # raw category outside the taxonomy
        if valid is not None and label not in valid:
            continue
        n_kept += 1
        gid = (1000 * label + group["objectId"] + 1 if collapse_to is None
               else 1000 * collapse_to + n_kept)
        for seg in group["segments"]:
            verts = sp_to_verts.get(seg)
            if verts is None:
                verts = sp_to_verts[seg] = np.nonzero(superpoints == seg)[0]
            gt[verts] = gid
    if collapse_to is not None and n_kept >= 1000:
        raise ValueError(f"{agg_path}: {n_kept} instances do not fit the "
                         f"1000 * label + instance encoding")
    return gt


def load_scene_3d_gt(gt_root, scene: str, tsv_path, taxonomy: str = "nyu40",
                     collapse_to: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    vertices [V,3], superpoints [V], gt_ids [V] of one scene from the 3D GT tree.

    `taxonomy` selects the label set the GT instances are drawn from: `nyu40` (the default,
    the official v2 benchmark — every existing number was produced with it) or `scannet200`
    (the 200 raw ScanNet ids of `data/scannet200_constants.py`). See `build_gt_ids` for
    `collapse_to`.
    """
    scene_dir = Path(gt_root) / scene
    vertices = read_ply_vertices(scene_dir / f"{scene}_vh_clean_2.ply")
    superpoints = load_superpoints(scene_dir / f"{scene}_vh_clean_2.0.010000.segs.json")
    if len(superpoints) != len(vertices):
        raise ValueError(f"{scene}: {len(vertices)} vertices but {len(superpoints)} seg ids")
    if taxonomy == "nyu40":
        raw_to_label, valid_label_ids = load_raw_to_nyu40(tsv_path), None
    elif taxonomy == "scannet200":
        from data.scannet200_constants import VALID_CLASS_IDS_200_SET
        raw_to_label, valid_label_ids = (load_raw_to_scannet_id(tsv_path),
                                         VALID_CLASS_IDS_200_SET)
    else:
        raise ValueError(f"unknown taxonomy {taxonomy!r} (nyu40 | scannet200)")
    gt_ids = build_gt_ids(superpoints, scene_dir / f"{scene}.aggregation.json",
                          raw_to_label, valid_label_ids, collapse_to)
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


def sample_frames25k(scene_dir, num_frames: Optional[int] = None,
                     require_depth: bool = False) -> List[str]:
    """
    Frame stems of a 25k scene, evenly subsampled to at most `num_frames` (None = all).
    Only frames with a finite pose and an existing color jpg qualify; `require_depth` also
    demands the sensor depth png (the GT-projection transfer needs it, §9.9 — the default
    unprojection protocol never reads depth from disk).
    """
    scene_dir = Path(scene_dir)
    poses = load_frames25k_poses(scene_dir)
    stems = [s for s in sorted(poses) if (scene_dir / "color" / f"{s}.jpg").exists()
             and (not require_depth or (scene_dir / "depth" / f"{s}.png").exists())]
    if not stems:
        raise ValueError(f"{scene_dir}: no usable frames (finite pose + color jpg"
                         + (" + depth png)" if require_depth else ")"))
    if num_frames is not None and len(stems) > num_frames:
        idx = np.linspace(0, len(stems) - 1, num_frames).round().astype(int)
        stems = [stems[i] for i in sorted(set(idx.tolist()))]
    return stems


# ------------------------------------------------------------------------------------------
# GT camera geometry of the 25k frames (the GT-projection transfer, docs/MASKDINO.md §9.9)
#
# EVAL-TIME TRANSFER ONLY. Poses, intrinsics and sensor depth loaded here never reach the
# prediction path — exactly like the Sim(3)+ICP of the default protocol. The model still
# sees only images.
# ------------------------------------------------------------------------------------------

def load_frames25k_intrinsics(scene_dir) -> Dict[str, np.ndarray]:
    """
    The scene's `{"color": K, "depth": K}` 3x3 intrinsics from `intrinsics_{color,depth}.txt`
    (ScanNet ships them as 4x4; the top-left 3x3 is the pinhole matrix).

    They are the SAME physical camera at two resolutions: on the val-312 tar the normalised
    (fx/W, fy/H, cx/W, cy/H) of the two files agree to <1e-3, so one pose serves both and a
    point's depth-image and color-image projections are consistent.
    """
    scene_dir = Path(scene_dir)
    out = {}
    for which in ("color", "depth"):
        path = scene_dir / f"intrinsics_{which}.txt"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing (needed by the GT-projection transfer)")
        mat = np.loadtxt(path, dtype=np.float64)
        if mat.shape not in ((3, 3), (4, 4)):
            raise ValueError(f"{path}: expected a 3x3 or 4x4 intrinsic, got {mat.shape}")
        if not np.isfinite(mat).all():
            raise ValueError(f"{path}: non-finite intrinsic")
        out[which] = mat[:3, :3].copy()
    return out


def load_frames25k_depth(scene_dir, stems: List[str]) -> np.ndarray:
    """
    Sensor depth maps [S, H, W] in METERS from `depth/<stem>.png` (uint16 millimeters, the
    ScanNet convention; 0 = no reading, kept as 0 and rejected by the caller's depth test).
    All frames of a scene share one resolution (640x480); a mismatch raises.
    """
    from PIL import Image                      # local: keeps the module numpy-only otherwise

    scene_dir = Path(scene_dir)
    maps = []
    for stem in stems:
        arr = np.asarray(Image.open(scene_dir / "depth" / f"{stem}.png"))
        if arr.ndim != 2:
            raise ValueError(f"{scene_dir}/depth/{stem}.png: expected a single-channel "
                             f"depth png, got shape {arr.shape}")
        maps.append(arr.astype(np.float32) / 1000.0)
    if len({m.shape for m in maps}) != 1:
        raise ValueError(f"{scene_dir}: depth maps of differing sizes "
                         f"{sorted({m.shape for m in maps})}")
    return np.stack(maps)


def load_frames25k_color_size(scene_dir, stems: List[str]) -> Tuple[int, int]:
    """
    The (width, height) of a scene's color jpgs — the resolution `intrinsics_color.txt`
    refers to, and the denominator of the mask-grid rescale. Reads headers only (PIL is
    lazy). Raises if the frames disagree, because one intrinsic could not describe both.
    """
    from PIL import Image

    scene_dir = Path(scene_dir)
    sizes = {Image.open(scene_dir / "color" / f"{s}.jpg").size for s in stems}
    if len(sizes) != 1:
        raise ValueError(f"{scene_dir}: color frames of differing sizes {sorted(sizes)}")
    return sizes.pop()
