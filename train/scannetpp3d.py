"""
ScanNet++ v2 data for the 3D ruler (docs/todo.md 6d) — the sibling of `train/scannet3d.py`.

Reads the two tars `docs/DATASET.md` §2.1 describes (`scannetpp_3d_gt_val50.tar.zst` +
`scannetpp_frames_val50.tar.zst`, 49 scenes of the official `nvs_sem_val` split) and exposes
the SAME interface every dataset adapter exposes (`train/datasets3d.py`):

    load_scene_3d_gt, sample_frames, load_poses, load_intrinsics, load_depth,
    load_color_size

Two things are different from ScanNet, and both are properties of the release, not choices:

**1. Class-agnostic only.** ScanNet++'s instance benchmark has 84 classes of its own
(`_metadata/top100_instance.txt`) and our head predicts 19 ScanNet classes. There is no
honest label correspondence, so every instance is emitted under the evaluator's single
collapsed label (`train/benchmark3d.py::AGNOSTIC_LABEL_ID`) and only the class-agnostic
column is reported — which is also the setting FAST3DIS and IGGT report in
(`docs/RELATED_WORK.md`), so it is a fair column, not a concession.

**2. The superpoint vote degenerates.** `segments.json::segIndices` is the identity
permutation — one segment per vertex (`docs/DATASET.md` §2.1, verified again here per scene
via `meta["superpoints_degenerate"]`). `train/eval3d_geometry.py::superpoint_majority`
therefore reduces to a per-vertex vote on ScanNet++. Nothing is done about it: inventing an
over-segmentation would be our GT, not the benchmark's.

The frames tar was built to mirror `scannet_frames25k_val312.tar.zst` file-for-file
(`color/<stem>.jpg`, `pose/<stem>.txt` camera-to-world, `depth/<stem>.png` uint16
millimetres, `intrinsics_{color,depth}.txt`), so the frame loaders here are thin wrappers
over the ScanNet ones rather than reimplementations. The colour intrinsic varies ~1.5 % per
frame (iPhone autofocus); the scene-level file holds the median, which is what the
GT-projection transfer uses.

The upstream-format knowledge (`instance_map_to` before the `top100_instance` filter, the
`"sceneId:"` key, segment-id closure) lives in ONE place — the build's
`legacy/dataset_build/scripts/scannetpp_common.py` — and is imported, not restated.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent /
                       "legacy" / "dataset_build" / "scripts"))

from scannetpp_common import (  # noqa: E402
    build_vertex_instances, load_instance_classes, load_label_map, load_segments,
)

from train.benchmark3d import AGNOSTIC_LABEL_ID  # noqa: E402
from train.scannet3d import (  # noqa: E402
    load_frames25k_color_size, load_frames25k_depth, load_frames25k_intrinsics,
    load_frames25k_poses, read_ply_vertices, sample_frames25k,
)

# The tar ships the class tables + the split file next to the scenes, because the GT is
# unreadable without them and nothing may read the upstream tree at run time.
METADATA_DIR = "_metadata"


def load_scene_3d_gt(gt_root, scene: str, tsv_path=None) -> Dict[str, np.ndarray]:
    """
    vertices [V,3], superpoints [V], gt_ids [V], meta — one ScanNet++ scene.

    `gt_ids` uses the benchmark encoding `1000 * label + instance` with the label collapsed
    to `AGNOSTIC_LABEL_ID` (see the module docstring) and a dense 1-based instance index, so
    `train/benchmark3d.py` scores it unchanged. `tsv_path` is accepted for interface
    symmetry with the ScanNet adapters and ignored.
    """
    scene_dir = Path(gt_root) / scene
    meta_dir = Path(gt_root) / METADATA_DIR
    if not meta_dir.is_dir():
        raise FileNotFoundError(f"{meta_dir} missing — the ScanNet++ GT tar ships the class "
                                f"tables in scans3d/{METADATA_DIR}/, and the GT cannot be "
                                f"read without them")

    vertices = read_ply_vertices(scene_dir / "mesh.ply")
    seg_indices = load_segments(scene_dir / "segments.json")
    if len(seg_indices) != len(vertices):
        raise ValueError(f"{scene}: {len(vertices)} vertices but {len(seg_indices)} "
                         f"segment ids")

    groups = json.loads((scene_dir / "segments_anno.json").read_text())["segGroups"]
    inst_ids, instances = build_vertex_instances(
        seg_indices, groups, load_instance_classes(meta_dir), load_label_map(meta_dir))
    if len(instances) >= 1000:
        # the benchmark encoding is `1000 * label + instance`; the val-50 maximum is 250
        raise ValueError(f"{scene}: {len(instances)} instances do not fit the "
                         f"1000 * label + instance encoding")

    gt_ids = np.where(inst_ids > 0, 1000 * AGNOSTIC_LABEL_ID + inst_ids, 0).astype(np.int64)
    n_segments = int(len(np.unique(seg_indices)))
    return {
        "vertices": vertices,
        "superpoints": seg_indices,
        "gt_ids": gt_ids,
        "meta": {
            "num_instances": len(instances),
            "num_segments": n_segments,
            # True on every val-50 scene: segIndices is one segment per vertex, so the
            # superpoint majority is a per-vertex vote (docs/DATASET.md §2.1)
            "superpoints_degenerate": n_segments == len(seg_indices),
            "labels": sorted({i["label"] for i in instances}),
        },
    }


# ------------------------------------------------------------------------------------------
# Frames — the tar mirrors the ScanNet frames25k layout on purpose (docs/DATASET.md §2.1),
# so these are wrappers, not reimplementations. Keeping the wrappers (instead of importing
# the ScanNet names directly at the call site) is what lets `train/datasets3d.py` treat
# every dataset through one interface.
# ------------------------------------------------------------------------------------------

def sample_frames(scene_dir, num_frames: Optional[int] = None,
                  require_depth: bool = False) -> List[str]:
    """The scene's frame stems, evenly subsampled to at most `num_frames` (None = all 50)."""
    return sample_frames25k(scene_dir, num_frames, require_depth=require_depth)


def load_poses(scene_dir) -> Dict[str, np.ndarray]:
    """stem -> camera-to-world 4x4 (`aligned_pose`, in the `mesh_aligned_0.05` frame)."""
    return load_frames25k_poses(scene_dir)


def load_intrinsics(scene_dir) -> Dict[str, np.ndarray]:
    """`{"color": K, "depth": K}` — 1920x1440 and 256x192, the same physical camera."""
    return load_frames25k_intrinsics(scene_dir)


def load_depth(scene_dir, stems: List[str]) -> np.ndarray:
    """Sensor depth [S, 192, 256] in meters (uint16 millimetres on disk)."""
    return load_frames25k_depth(scene_dir, stems)


def load_color_size(scene_dir, stems: List[str]) -> Tuple[int, int]:
    """(width, height) of the colour jpgs — 1920x1440."""
    return load_frames25k_color_size(scene_dir, stems)
