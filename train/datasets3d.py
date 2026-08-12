"""
The dataset registry of the 3D ruler (docs/todo.md 6d) — one entry per `--dataset` value of
`scripts/eval_3d_maskdino.py`.

Everything that differs between benchmarks lives here and nowhere else: which loaders read a
scene, whether the class-aware column means anything, whether the prediction side may keep
wall/floor, and which image extension the frames use. The pipeline itself (the head, the two
transfer modes of `train/eval3d_geometry.py`, and the vendored official evaluator
`train/benchmark3d.py`) is IDENTICAL across datasets — that is the point of the matrix: the
only variable is the benchmark.

    scannetv2   the headline ruler (docs/MASKDINO.md §9). 18 nyu40 classes, CLASS-AWARE
                headline + class-agnostic column. `--dataset` defaults to it, so every
                published number and every existing command is unchanged.
    scannet200  the same scans, the same annotations, the same tars — only the label set
                changes (200 raw ScanNet ids, `data/scannet200_constants.py`). Zero new
                data. Class-agnostic only, and wall/floor ARE valid classes here.
    scannetpp   ScanNet++ v2 `nvs_sem_val`, 49 scenes (docs/DATASET.md §2.1). 84 classes of
                its own, so class-agnostic only.
    replica     the 8 scenes FAST3DIS reports on (docs/DATASET.md §2.2). Its own taxonomy
                and our own GT construction, so class-agnostic only.

**Why three of the four are class-agnostic-only.** Our head has 19 ScanNet logits; ScanNet200,
ScanNet++ and Replica have taxonomies it cannot address. Rather than invent a
correspondence, those datasets emit every instance under one collapsed label
(`train/benchmark3d.py::AGNOSTIC_LABEL_ID`) and are reported in the class-agnostic setting —
which is what FAST3DIS and IGGT report anyway (docs/RELATED_WORK.md), so it is a fair
column. `collapse_gt_to_class_agnostic` is idempotent on such GT, so the script's
class-agnostic path needs no special case.

**Why `drop_wall_floor_predictions` is per dataset.** The prediction filter mirrors each
dataset's GT taxonomy, which is what keeps the comparison single-variable: the ScanNetv2
benchmark excludes wall/floor and ScanNet++/Replica exclude the room shell, so predictions
of those classes are dropped; ScanNet200 includes them as valid classes, so they are kept
and get their chance to match.
"""

from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Optional

from train import replica3d, scannet3d, scannetpp3d
from train.benchmark3d import AGNOSTIC_LABEL_ID

DEFAULT_DATASET = "scannetv2"


@dataclass(frozen=True)
class Dataset3D:
    """One benchmark, as the 3D ruler sees it."""

    name: str
    class_aware: bool
    drop_wall_floor_predictions: bool
    color_ext: str
    # what a scene needs, all with the interface of `train/scannet3d.py`
    load_scene_3d_gt: Callable      # (gt_root, scene, tsv_path) -> dict
    sample_frames: Callable         # (scene_dir, num_frames, require_depth) -> [stem]
    load_poses: Callable            # (scene_dir) -> {stem: 4x4 camera-to-world}
    load_intrinsics: Callable       # (scene_dir) -> {"color": K, "depth": K}
    load_depth: Callable            # (scene_dir, stems) -> [S, H, W] meters
    load_color_size: Callable       # (scene_dir, stems) -> (width, height)
    note: str = ""
    # an over-segmentation the release ships but the evaluation does NOT use, kept so
    # `scripts/gate_3d_gt.py --report_superpoints` can measure it and show why
    alt_superpoints: Optional[Callable] = None      # (gt_root, scene) -> [V]
    alt_superpoints_name: str = ""


DATASETS: Dict[str, Dataset3D] = {
    "scannetv2": Dataset3D(
        name="scannetv2",
        class_aware=True,
        drop_wall_floor_predictions=True,
        color_ext=".jpg",
        load_scene_3d_gt=scannet3d.load_scene_3d_gt,
        sample_frames=scannet3d.sample_frames25k,
        load_poses=scannet3d.load_frames25k_poses,
        load_intrinsics=scannet3d.load_frames25k_intrinsics,
        load_depth=scannet3d.load_frames25k_depth,
        load_color_size=scannet3d.load_frames25k_color_size,
        note="official 18-class ScanNet v2 3D instance benchmark (docs/MASKDINO.md §9)",
    ),
    "scannet200": Dataset3D(
        name="scannet200",
        class_aware=False,
        # ScanNet200 counts wall and floor as valid classes, unlike the v2 benchmark
        drop_wall_floor_predictions=False,
        color_ext=".jpg",
        load_scene_3d_gt=partial(scannet3d.load_scene_3d_gt, taxonomy="scannet200",
                                 collapse_to=AGNOSTIC_LABEL_ID),
        sample_frames=scannet3d.sample_frames25k,
        load_poses=scannet3d.load_frames25k_poses,
        load_intrinsics=scannet3d.load_frames25k_intrinsics,
        load_depth=scannet3d.load_frames25k_depth,
        load_color_size=scannet3d.load_frames25k_color_size,
        note="ScanNet200 taxonomy over the SAME val-312 tars; class-agnostic only",
    ),
    "scannetpp": Dataset3D(
        name="scannetpp",
        class_aware=False,
        drop_wall_floor_predictions=True,
        color_ext=".jpg",
        load_scene_3d_gt=scannetpp3d.load_scene_3d_gt,
        sample_frames=scannetpp3d.sample_frames,
        load_poses=scannetpp3d.load_poses,
        load_intrinsics=scannetpp3d.load_intrinsics,
        load_depth=scannetpp3d.load_depth,
        load_color_size=scannetpp3d.load_color_size,
        note="ScanNet++ v2 nvs_sem_val (49 scenes, 50 views each); class-agnostic only",
    ),
    "replica": Dataset3D(
        name="replica",
        class_aware=False,
        drop_wall_floor_predictions=True,
        color_ext=replica3d.COLOR_EXT,
        load_scene_3d_gt=replica3d.load_scene_3d_gt,
        sample_frames=replica3d.sample_frames,
        load_poses=replica3d.load_poses,
        load_intrinsics=replica3d.load_intrinsics,
        load_depth=replica3d.load_depth,
        load_color_size=replica3d.load_color_size,
        note="Replica 8 scenes, vMAP renders (50 views each); class-agnostic only, and the "
             "GT instance set is OUR construction (train/replica3d.py)",
        alt_superpoints=replica3d.load_preseg_superpoints,
        alt_superpoints_name="preseg (planar, NOT used by the vote)",
    ),
}

DATASET_NAMES = tuple(DATASETS)


def get_dataset(name: str) -> Dataset3D:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r} (expected one of {DATASET_NAMES})")
    return DATASETS[name]
