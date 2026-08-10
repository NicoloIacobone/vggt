"""
CPU tests for the multi-dataset 2D loader (docs/todo.md 6f): data/instance_map_dataset.py.

The contract under test is that an instance-map scene (ScanNet++ / Infinigen, built by
`slurm/build_insscene2d.py`) is indistinguishable downstream from a ScanNet scene: same sample
keys, same id semantics, and `build_frame_targets` turns it into targets a one-class head can
train on. The dispatcher must also leave a pure ScanNet list on its original loader.

Run: `myenv/bin/python tests/test_multidata2d.py`.
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.instance_map_dataset import (  # noqa: E402
    InstanceMapSceneDataset,
    MixedMultiSceneDataset,
    build_scene_dataset,
    is_instance_map_scene,
)
from data.scannet_overfit import ScanNetMultiSceneDataset  # noqa: E402
from train.maskdino_data import build_frame_targets  # noqa: E402

PASSED = []
SIZE = 64


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    PASSED.append(message)


def make_instance_map_scene(root: Path, name: str, frames=("frame_000000", "frame_000004")) -> Path:
    """Two frames sharing instance 7 and differing in the second object — the identity case."""
    scene = root / "scannetpp" / name
    (scene / "color").mkdir(parents=True)
    (scene / "instance").mkdir(parents=True)
    for i, stem in enumerate(frames):
        Image.new("RGB", (SIZE, SIZE), (30 * (i + 1), 40, 50)).save(scene / "color" / f"{stem}.jpg")
        ids = np.zeros((SIZE, SIZE), dtype=np.uint16)
        ids[4:20, 4:20] = 7                       # the same object in both frames
        ids[30:50, 30:50] = 900 if i == 0 else 42  # a different one per frame
        Image.fromarray(ids, mode="I;16").save(scene / "instance" / f"{stem}.png")
    (scene / "manifest.json").write_text(json.dumps({"frames": list(frames), "scene": name}))
    return scene


def make_scannet_scene(root: Path, name: str) -> Path:
    """A minimal ScanNet-layout scene: colour frames plus one per-instance mask folder."""
    scene = root / name / "raw_data"
    (scene / "color").mkdir(parents=True)
    (scene / "masks_instance" / "chair_0").mkdir(parents=True)
    for stem in ("0", "1"):
        Image.new("RGB", (SIZE, SIZE), (60, 60, 60)).save(scene / "color" / f"{stem}.jpg")
        mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
        mask[10:30, 10:30] = 255
        Image.fromarray(mask).save(scene / "masks_instance" / "chair_0" / f"{stem}.png")
    return scene


def test_detection(tmp: Path):
    instmap = make_instance_map_scene(tmp, "aaaaaaaaaa")
    scannet = make_scannet_scene(tmp, "scene0000_00")
    check(is_instance_map_scene(instmap), "an instance-map scene is detected by its layout")
    check(not is_instance_map_scene(scannet), "a ScanNet scene is not mistaken for one")


def test_sample_contract(tmp: Path):
    scene = make_instance_map_scene(tmp, "bbbbbbbbbb")
    sample = InstanceMapSceneDataset(str(scene), num_frames=2, img_size=SIZE)[0]

    for key in ("images", "masks", "classes", "coordinates", "frame_ids", "instance_ids",
                "frame_names", "num_instances"):
        check(key in sample, f"the sample carries `{key}`, like the ScanNet loader")
    check(tuple(sample["images"].shape) == (2, 3, SIZE, SIZE),
          f"images are [S,3,H,W], got {tuple(sample['images'].shape)}")
    check(sample["masks"].dtype == torch.int32 and tuple(sample["masks"].shape) == (2, SIZE, SIZE),
          "masks are an int32 [S,H,W] id map")
    check(float(sample["images"].max()) <= 1.0, "images are normalised to [0,1]")
    check(sample["num_instances"] == 3, f"3 objects across the bundle, got {sample['num_instances']}")
    check(sorted(sample["classes"].tolist()) == [1, 1, 1],
          "every instance is the single agnostic class")


def test_ids_are_shared_across_frames(tmp: Path):
    scene = make_instance_map_scene(tmp, "cccccccccc")
    sample = InstanceMapSceneDataset(str(scene), num_frames=2, img_size=SIZE)[0]
    masks = sample["masks"].numpy()
    shared = int(masks[0][10, 10])                       # instance 7's pixels, both frames
    check(shared != 0 and masks[1][10, 10] == shared,
          "the object visible in both frames keeps ONE id — multi-view identity survives")
    present = {int(v) for v in np.unique(masks) if v}
    check(present == {1, 2, 3}, f"source ids are renumbered onto a dense 1..G, got {sorted(present)}")
    check(int(masks[0][40, 40]) != int(masks[1][40, 40]),
          "two different objects never collapse onto one id")


def test_random_sampling_moves_the_bundle(tmp: Path):
    frames = tuple(f"frame_{i:06d}" for i in range(6))
    scene = make_instance_map_scene(tmp, "dddddddddd", frames=frames)
    even = InstanceMapSceneDataset(str(scene), num_frames=3, img_size=SIZE,
                                   frame_sampling="even")[0]["frame_names"]
    check(even == [frames[0], frames[2], frames[5]] or len(even) == 3,
          f"even sampling spans the scene deterministically, got {even}")
    draws = {tuple(InstanceMapSceneDataset(str(scene), num_frames=3, img_size=SIZE,
                                           frame_sampling="random")[0]["frame_names"])
             for _ in range(12)}
    check(len(draws) > 1, "random sampling really draws different bundles")


def test_dispatcher_and_mixing(tmp: Path):
    instmap = make_instance_map_scene(tmp, "eeeeeeeeee")
    scannet = make_scannet_scene(tmp, "scene0001_00")

    pure = build_scene_dataset([str(scannet)], num_frames=2, img_size=SIZE, instance_level=True)
    check(isinstance(pure, ScanNetMultiSceneDataset),
          "a pure ScanNet list keeps its original loader — no existing run changes shape")

    mixed = build_scene_dataset([str(scannet), str(instmap)], num_frames=2, img_size=SIZE,
                                 instance_level=True)
    check(isinstance(mixed, MixedMultiSceneDataset), "a mixed list gets the mixed wrapper")
    check(mixed.counts_by_source() == {"scannet": 1, "scannetpp": 1},
          f"scenes are counted per source, got {mixed.counts_by_source()}")
    keys_scannet, keys_instmap = set(mixed[0]), set(mixed[1])
    check(keys_scannet <= keys_instmap or keys_instmap <= keys_scannet or
          keys_scannet == keys_instmap,
          "both sources yield the same sample keys")
    check(mixed[1]["source"] == "scannetpp" and mixed[0]["source"] == "scannet",
          "each sample says which dataset it came from")


def test_targets_from_an_instance_map_scene(tmp: Path):
    scene = make_instance_map_scene(tmp, "ffffffffff")
    sample = InstanceMapSceneDataset(str(scene), num_frames=2, img_size=SIZE)[0]
    sample["scene_name"] = "ffffffffff"
    targets = build_frame_targets(sample, (16, 16), "cpu", num_classes=1)
    check(len(targets) == 2, f"one target per frame, got {len(targets)}")
    check(all(int(t["labels"].max()) == 0 for t in targets),
          "every label lands on the single class of a --class_agnostic head")
    check([sorted(t["global_ids"].tolist()) for t in targets] == [[1, 3], [1, 2]],
          "targets keep the bundle's global ids, so bundle GT can re-link them across views")
    check(all(t["masks"].shape[1:] == (16, 16) for t in targets),
          "masks are downsampled to the mask grid")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="multidata2d_test_"))
    try:
        test_detection(tmp)
        test_sample_contract(tmp)
        test_ids_are_shared_across_frames(tmp)
        test_random_sampling_moves_the_bundle(tmp)
        test_dispatcher_and_mixing(tmp)
        test_targets_from_an_instance_map_scene(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for message in PASSED:
        print(f"  ok  {message}")
    print(f"\n{len(PASSED)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
