"""Shared ScanNet++ v2 source-format helpers (docs/todo.md 6c).

Everything here is about READING the upstream ScanNet++ release. It is imported by the two
build scripts and by `scripts/verify_scannetpp_gt.py`, so the format knowledge lives in one
place.

The four formats that are easy to get silently wrong, and how each was pinned down
(evidence: docs/DATASET.md §2.1, reproduced by `scripts/verify_scannetpp_gt.py`):

1. POSE. `iphone/pose_intrinsic_imu.json` carries BOTH `pose` and `aligned_pose`.
   `aligned_pose` is the CAMERA-TO-WORLD pose in the frame of `mesh_aligned_0.05.ply`;
   `pose` is the raw ARKit trajectory in a completely different frame (its unprojected
   depth lands ~85 m from the mesh). Only `aligned_pose`, and only read as camera-to-world
   (world point = R @ p_cam + t), puts any geometry in the frustum at all.

2. RGB. `iphone/rgb.mkv`, decoded by `cv2.VideoCapture`; decoded frame N is
   `frame_{N:06d}` in the pose json. Verified by frame count AND by content (mesh
   reprojection NCC peaks at offset 0 and falls off by +-10 frames).

3. DEPTH. `iphone/depth.bin` is a per-frame stream of `<4-byte little-endian
   compressed size><LZ4 block>`, each block decompressing to 192*256 uint16 MILLIMETRES.
   Not zlib, not float16. One stream entry per RGB frame.

4. LABELS. `segments_anno.json`'s raw labels are the fine-grained ScanNet++ vocabulary
   (2878 classes). The instance benchmark maps them through `map_benchmark.csv`'s
   `instance_map_to` column FIRST and only then filters to `top100_instance.txt`
   (84 classes). Skipping the map loses ~10 % of the instances.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

DEPTH_H, DEPTH_W = 192, 256
COLOR_W, COLOR_H = 1920, 1440

# Files the build copies verbatim out of the upstream tree.
GT_FILES = {
    "mesh.ply": "scans/mesh_aligned_0.05.ply",
    "segments.json": "scans/segments.json",
    "segments_anno.json": "scans/segments_anno.json",
}
# The class tables are copied into the tar too: nothing downstream may read the upstream
# tree (it belongs to another user and can vanish), and the GT is meaningless without them.
METADATA_FILES = (
    "metadata/semantic_benchmark/top100_instance.txt",
    "metadata/semantic_benchmark/map_benchmark.csv",
    "metadata/semantic_classes.txt",
    "splits/nvs_sem_val.txt",
)


# ----------------------------------------------------------------------------------------
# class tables
# ----------------------------------------------------------------------------------------

def select_scenes(scene_list, start: int, end: int, exclude=()) -> list[str]:
    """The `[start, end]` slice of a scene-list file, minus `exclude`.

    Exclusions exist for scenes the upstream release ships broken — see
    `docs/DATASET.md` §2.1 for the one that is excluded and why. They are named
    explicitly, never inferred, so a scene can only disappear on purpose.
    """
    all_scenes = [l.strip() for l in Path(scene_list).read_text().splitlines() if l.strip()]
    drop = set(exclude or ())
    unknown = drop - set(all_scenes)
    if unknown:
        raise ValueError(f"--exclude_scenes names scenes absent from the list: "
                         f"{sorted(unknown)}")
    return [s for s in all_scenes[start:end + 1] if s not in drop]


def drop_excluded(out_root, exclude) -> list[str]:
    """Remove excluded scenes from a partially built tree. Returns what was removed."""
    import shutil
    removed = []
    for scene in exclude or ():
        d = Path(out_root) / scene
        if d.is_dir():
            shutil.rmtree(d)
            removed.append(scene)
    return removed


def load_instance_classes(metadata_dir) -> list[str]:
    """The 84 instance-benchmark class names, in file order (index = class id)."""
    p = Path(metadata_dir) / "top100_instance.txt"
    return [l.strip() for l in p.read_text().splitlines() if l.strip()]


def load_label_map(metadata_dir) -> dict[str, str]:
    """raw ScanNet++ label -> instance-benchmark label (identity where unmapped)."""
    p = Path(metadata_dir) / "map_benchmark.csv"
    with open(p, newline="") as f:
        return {r["class"]: (r["instance_map_to"] or r["class"])
                for r in csv.DictReader(f)}


# ----------------------------------------------------------------------------------------
# 3D GT
# ----------------------------------------------------------------------------------------

def load_segments(path) -> np.ndarray:
    """`segments.json` -> the per-vertex segment id array.

    The scene-id key is spelled `"sceneId:"`, with a literal trailing colon, upstream.
    """
    d = json.loads(Path(path).read_text())
    if "segIndices" not in d:
        raise ValueError(f"{path}: no segIndices")
    return np.asarray(d["segIndices"], dtype=np.int64)


def build_vertex_instances(seg_indices: np.ndarray, seg_groups: list[dict],
                           instance_classes: list[str], label_map: dict[str, str]):
    """Per-vertex instance ids for the benchmark classes.

    Returns `(inst_ids, instances)`:
      `inst_ids`  int32 per vertex, 0 = void (no benchmark instance);
      `instances` list of dicts, one per kept object, in `segGroups` order:
                  {inst_id, object_id, raw_label, label, class_id, n_vertices}.

    `inst_id` is 1-based and dense, so `inst_ids` never collides with the void 0.

    Raises on segment-id closure violations — every segment id an object references must
    exist in `seg_indices` — the same guard `download_3d_gt.py::validate_scene` applies to
    ScanNet's aggregation files.
    """
    present = np.unique(seg_indices)
    present_set = set(present.tolist())
    order = np.argsort(seg_indices, kind="stable")
    sorted_ids = seg_indices[order]
    class_index = {c: i for i, c in enumerate(instance_classes)}

    inst_ids = np.zeros(len(seg_indices), dtype=np.int32)
    instances: list[dict] = []
    for group in seg_groups:
        segs = group.get("segments", [])
        missing = set(segs) - present_set
        if missing:
            raise ValueError(
                f"segment-id closure violated: object {group.get('objectId')} "
                f"('{group.get('label')}') references {len(missing)} segment id(s) "
                f"absent from segments.json, e.g. {sorted(missing)[:5]}")
        raw = group.get("label", "")
        label = label_map.get(raw, raw)
        if label not in class_index:
            continue
        lo = np.searchsorted(sorted_ids, np.asarray(segs, dtype=np.int64), side="left")
        hi = np.searchsorted(sorted_ids, np.asarray(segs, dtype=np.int64), side="right")
        verts = np.concatenate([order[a:b] for a, b in zip(lo, hi)]) if len(segs) else \
            np.empty(0, dtype=np.int64)
        if verts.size == 0:
            continue
        inst_id = len(instances) + 1
        inst_ids[verts] = inst_id
        instances.append({
            "inst_id": inst_id,
            "object_id": int(group.get("objectId", group.get("id", -1))),
            "raw_label": raw,
            "label": label,
            "class_id": class_index[label],
            "n_vertices": int(verts.size),
        })
    return inst_ids, instances


# ----------------------------------------------------------------------------------------
# PLY vertex count / vertices
# ----------------------------------------------------------------------------------------

def ply_vertex_count(path) -> int:
    """Vertex count from the PLY header alone — no body parse, no dependency."""
    n = None
    with open(path, "rb") as f:
        if f.read(3) != b"ply":
            raise ValueError(f"{path}: missing ply magic")
        f.seek(0)
        for _ in range(200):
            line = f.readline().decode("ascii", "replace").strip()
            if line.startswith("element vertex"):
                n = int(line.split()[2])
            if line == "end_header":
                break
    if n is None:
        raise ValueError(f"{path}: no 'element vertex' in header")
    return n


def read_ply_xyz(path) -> np.ndarray:
    """The (N, 3) float64 vertex positions. Uses plyfile, which myenv ships."""
    from plyfile import PlyData
    v = PlyData.read(str(path))["vertex"]
    return np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)


def read_ply_xyz_rgb(path):
    """`(xyz, rgb)` — positions float64 (N, 3), colours float64 (N, 3) in 0..255."""
    from plyfile import PlyData
    v = PlyData.read(str(path))["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    rgb = np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.float64)
    return xyz, rgb


# ----------------------------------------------------------------------------------------
# iphone streams
# ----------------------------------------------------------------------------------------

def frame_stems(pose_json: dict) -> list[str]:
    """The pose json's frame keys in index order (`frame_000000`, ...)."""
    return sorted(pose_json.keys())


def sample_indices(n_total: int, k: int) -> np.ndarray:
    """`k` frame indices spread uniformly over the WHOLE sequence, endpoints included.

    Deduplicated, so a sequence shorter than `k` yields every frame once.
    """
    if n_total <= 0:
        return np.empty(0, dtype=np.int64)
    if n_total <= k:
        return np.arange(n_total, dtype=np.int64)
    return np.unique(np.round(np.linspace(0, n_total - 1, k)).astype(np.int64))


def count_depth_frames(path) -> int:
    """Number of frames in `depth.bin`, by walking the size prefixes (no decompression)."""
    n = 0
    with open(path, "rb") as f:
        while True:
            head = f.read(4)
            if len(head) < 4:
                break
            f.seek(int.from_bytes(head, "little"), 1)
            n += 1
    return n


def read_depth_frames(path, wanted, h: int = DEPTH_H, w: int = DEPTH_W) -> dict:
    """`{index: uint16 (h, w) millimetre depth}` for `wanted`, one pass over the stream.

    Unwanted frames are seeked over, so the cost is the wanted frames only.
    """
    import lz4.block

    want = set(int(i) for i in wanted)
    out: dict[int, np.ndarray] = {}
    with open(path, "rb") as f:
        i = 0
        while want:
            head = f.read(4)
            if len(head) < 4:
                break
            size = int.from_bytes(head, "little")
            if i in want:
                raw = lz4.block.decompress(f.read(size), uncompressed_size=h * w * 2)
                out[i] = np.frombuffer(raw, dtype=np.uint16).reshape(h, w).copy()
                want.discard(i)
            else:
                f.seek(size, 1)
            i += 1
    return out


def mesh_image_ncc(xyz: np.ndarray, gray_v: np.ndarray, K: np.ndarray,
                   cam2world: np.ndarray, img_gray: np.ndarray) -> float:
    """Normalised cross-correlation between a mesh's vertex greyscale, z-buffered into a
    view, and the image it lands on. NaN when too little geometry is in frustum.

    This is the only handle on "does decoded frame N really go with pose N": geometry and
    photometry have to agree, and they only do at the right pairing.
    """
    M = np.linalg.inv(np.asarray(cam2world, dtype=np.float64))
    cam = xyz @ M[:3, :3].T + M[:3, 3]
    z = cam[:, 2]
    f = z > 1e-6
    if f.sum() < 1000:
        return float("nan")
    uv = (cam[f] / z[f, None]) @ np.asarray(K, dtype=np.float64).T
    ih, iw = img_gray.shape
    u, w = uv[:, 0], uv[:, 1]
    inb = (u >= 0) & (u < iw) & (w >= 0) & (w < ih)
    if inb.sum() < 2000:
        return float("nan")
    ui, wi, zz = u[inb].astype(int), w[inb].astype(int), z[f][inb]
    gv = gray_v[f][inb]
    order = np.argsort(zz)                       # nearest vertex wins each pixel
    _, first = np.unique((wi * iw + ui)[order], return_index=True)
    ui, wi, gv = ui[order][first], wi[order][first], gv[order][first]
    a = gv - gv.mean()
    b = img_gray[wi, ui].astype(np.float64)
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a * b).sum() / denom) if denom > 0 else float("nan")


def scale_intrinsic(K: np.ndarray, src_wh, dst_wh) -> np.ndarray:
    """A 3x3 pinhole intrinsic rescaled from `src_wh` to `dst_wh` pixels."""
    K = np.asarray(K, dtype=np.float64).copy()
    K[0] *= dst_wh[0] / src_wh[0]
    K[1] *= dst_wh[1] / src_wh[1]
    return K


def unproject(depth_mm: np.ndarray, K: np.ndarray, cam2world: np.ndarray,
              min_m: float = 0.05) -> np.ndarray:
    """Sensor depth -> world points, with `cam2world` read as camera-to-world.

    `K` must already be at the depth map's resolution. Zero / near-zero depth is dropped.
    """
    d = depth_mm.astype(np.float64) / 1000.0
    h, w = d.shape
    yy, xx = np.mgrid[0:h, 0:w]
    m = d > min_m
    if not m.any():
        return np.empty((0, 3))
    pts = np.stack([(xx[m] - K[0, 2]) / K[0, 0] * d[m],
                    (yy[m] - K[1, 2]) / K[1, 1] * d[m], d[m]], 1)
    T = np.asarray(cam2world, dtype=np.float64)
    return pts @ T[:3, :3].T + T[:3, 3]
