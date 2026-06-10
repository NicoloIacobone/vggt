#!/usr/bin/env python3
"""
Validation for the checkpoint-format handling in scripts/visualize_masks.py.

Checks that scenes_from_checkpoint normalizes both checkpoint flavors — single-scene
(train_overfit.py, top-level keys) and multi-scene (train_multiscene.py, "scenes" list) —
into the same (label, scene_dict) interface, and that overlay_mask blends/contours correctly.
Runs on CPU without backbone weights (no model is instantiated).
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visualize_masks import scenes_from_checkpoint, overlay_mask


def _fake_batch():
    return {
        "images": torch.zeros(1, 2, 3, 14, 14),
        "coordinates": torch.zeros(1, 4, 2),
        "view_ids": torch.zeros(1, 4, dtype=torch.long),
        "gt": {"masks": torch.zeros(3, 2, 1, 1), "classes": torch.tensor([1, 2, 3]),
               "coordinates": torch.zeros(3, 2)},
        "frame_names": ["0.jpg", "5.jpg"],
    }


def test_single_scene_checkpoint():
    print("=== Testing single-scene (overfit) checkpoint ===")
    ckpt = dict(_fake_batch())
    ckpt["final_metrics"] = {"mIoU": 0.9}

    scenes = scenes_from_checkpoint(ckpt)
    assert len(scenes) == 1
    label, scene = scenes[0]
    assert label is None, "single-scene checkpoints render into the output dir directly"
    for key in ("images", "coordinates", "view_ids", "gt", "frame_names"):
        assert scene[key] is ckpt[key], f"{key} not passed through"
    assert scene["metrics"] == {"mIoU": 0.9}
    print("✅ Single-scene checkpoint test passed!\n")


def test_multi_scene_checkpoint():
    print("=== Testing multi-scene checkpoint ===")
    entries = []
    for name, split in [("scene0000_00", "train"), ("scene0004_00", "val")]:
        e = dict(_fake_batch())
        e.update(name=name, split=split, metrics={"mIoU": 0.5})
        entries.append(e)
    # Top-level back-compat keys must be ignored when "scenes" is present.
    ckpt = dict(_fake_batch())
    ckpt["scenes"] = entries

    scenes = scenes_from_checkpoint(ckpt)
    assert [label for label, _ in scenes] == ["train_scene0000_00", "val_scene0004_00"]
    assert scenes[0][1] is entries[0] and scenes[1][1] is entries[1]
    print("✅ Multi-scene checkpoint test passed!\n")


def test_missing_metrics():
    print("=== Testing checkpoint without metrics ===")
    scenes = scenes_from_checkpoint(dict(_fake_batch()))
    assert scenes[0][1]["metrics"] == {}, "missing final_metrics must yield an empty dict"
    print("✅ Missing-metrics test passed!\n")


def test_overlay_mask():
    print("=== Testing overlay_mask ===")
    rgb = np.zeros((8, 8, 3))
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    color = np.array([1.0, 0.0, 0.0])

    out = overlay_mask(rgb, mask, color, alpha=0.5)
    assert np.allclose(out[~mask], 0.0), "pixels outside the mask must be untouched"
    assert np.allclose(out[3:5, 3:5, 0], 0.5), "mask interior must be alpha-blended"
    assert np.allclose(out[2, 2:6, 0], 1.0), "mask border must be the full contour color"
    assert np.array_equal(overlay_mask(rgb, np.zeros_like(mask), color), rgb), \
        "empty mask must return the image unchanged"
    print("✅ overlay_mask test passed!\n")


if __name__ == "__main__":
    test_single_scene_checkpoint()
    test_multi_scene_checkpoint()
    test_missing_metrics()
    test_overlay_mask()
    print("All visualize_masks tests passed! ✅")
