"""Test legacy/dataset_build/scripts/build_official_masks.py — synthetic official-GT pairs -> SAM3 layout.

Builds tiny instance-filt / label-filt PNG pairs + a minimal tsv, runs the
converter, and asserts dir naming, sparse writing, class mapping (incl.
otherfurniture -> background), union consistency, k-ordering, 16-bit label
decoding, and the zero cross-class-duplicate acceptance stat.

CPU-only, no downloads, no torch. Run: python legacy/dataset_build/tests/test_build_official_masks.py
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from legacy.dataset_build.scripts.build_official_masks import convert_scene, load_label_map  # noqa: E402

H, W = 48, 64

# Minimal label map: raw id -> nyu40. Raw ids chosen != nyu ids to catch mixups.
# 101->5 (chair), 102->28 (shower curtain), 103->39 (otherfurniture, dropped),
# 104->1 (wall), 105->22 (ceiling, out of taxonomy -> dropped).
TSV_ROWS = [(101, 5, "chair"), (102, 28, "shower curtain"),
            (103, 39, "otherfurniture"), (104, 1, "wall"), (105, 22, "ceiling")]


def write_tsv(path):
    with open(path, "w") as f:
        f.write("id\traw_category\tnyu40id\tnyu40class\n")
        for rid, nyu, name in TSV_ROWS:
            f.write(f"{rid}\tx\t{nyu}\t{name}\n")


def make_frame(inst_boxes, label_boxes):
    """boxes: list of (value, r0, r1, c0, c1). Labels saved as 16-bit."""
    ia = np.zeros((H, W), np.uint8)
    la = np.zeros((H, W), np.uint16)
    for v, r0, r1, c0, c1 in inst_boxes:
        ia[r0:r1, c0:c1] = v
    for v, r0, r1, c0, c1 in label_boxes:
        la[r0:r1, c0:c1] = v
    return ia, la


def build_synthetic(root):
    """3 frames (indices 0, 5, 10) in extracted-dir form.

    inst 3 = chair (frames 0,5) ... appears FIRST -> chair_0
    inst 7 = chair (frame 5 only, sparse)         -> chair_1
    inst 2 = shower curtain (frames 0,10)         -> shower_curtain_0
    inst 9 = otherfurniture (frame 0)             -> dropped
    inst 4 = ceiling (frame 10)                   -> dropped
    inst 6 = wall (all frames)                    -> wall_0
    inst 11 = raw label 999, NOT in tsv (frame 0) -> dropped (-1), no crash
      (regression: scene0091_00 KeyError(30) — voteless instances must be dropped)
    """
    inst_dir = root / "inst" / "instance-filt"
    lab_dir = root / "lab" / "label-filt"
    inst_dir.mkdir(parents=True)
    lab_dir.mkdir(parents=True)

    frames = {
        0: ([(3, 0, 10, 0, 10), (2, 20, 30, 20, 30), (9, 40, 48, 0, 10), (6, 0, 48, 50, 64),
             (11, 40, 48, 20, 30)],
            [(101, 0, 10, 0, 10), (102, 20, 30, 20, 30), (103, 40, 48, 0, 10), (104, 0, 48, 50, 64),
             (999, 40, 48, 20, 30)]),
        5: ([(3, 0, 10, 0, 10), (7, 30, 40, 30, 40), (6, 0, 48, 50, 64)],
            [(101, 0, 10, 0, 10), (101, 30, 40, 30, 40), (104, 0, 48, 50, 64)]),
        10: ([(2, 20, 30, 20, 30), (4, 0, 5, 20, 40), (6, 0, 48, 50, 64)],
             [(102, 20, 30, 20, 30), (105, 0, 5, 20, 40), (104, 0, 48, 50, 64)]),
    }
    for f, (ib, lb) in frames.items():
        ia, la = make_frame(ib, lb)
        Image.fromarray(ia).save(inst_dir / f"{f}.png")          # uint8 -> mode L
        Image.fromarray(la).save(lab_dir / f"{f}.png")           # uint16 -> 16-bit PNG
    return root / "inst", root / "lab"


def main():
    tmp = Path(tempfile.mkdtemp(prefix="test_official_masks_"))
    tsv = tmp / "labels.tsv"
    write_tsv(tsv)
    id2nyu = load_label_map(tsv)
    assert id2nyu[101] == 5 and id2nyu[102] == 28, "tsv parsing"
    print("[1/7] tsv parsing OK")

    inst_src, lab_src = build_synthetic(tmp)
    # subset stems drive the frame list; include a frame (15) with no GT file.
    subset = tmp / "subset"
    subset.mkdir()
    for f in (0, 5, 10, 15):
        Image.fromarray(np.zeros((H, W, 3), np.uint8)).save(subset / f"{f:05d}.jpg")

    out_root = tmp / "scans"
    stats = convert_scene("scene_test", inst_src, lab_src, tsv, out_root,
                          subset_src=subset)
    raw = out_root / "scene_test" / "raw_data"

    # Dir naming + k order of first appearance.
    seg_dirs = sorted(p.name for p in (raw / "masks_instance").iterdir())
    assert seg_dirs == ["chair_0", "chair_1", "shower_curtain_0", "wall_0"], seg_dirs
    print("[2/7] dir naming + k ordering OK:", seg_dirs)

    # Sparseness: chair_1 only in frame 5; chair_0 in 0 and 5, not 10 or 15.
    c0 = sorted(p.name for p in (raw / "masks_instance" / "chair_0").iterdir())
    c1 = sorted(p.name for p in (raw / "masks_instance" / "chair_1").iterdir())
    sc = sorted(p.name for p in (raw / "masks_instance" / "shower_curtain_0").iterdir())
    assert c0 == ["00000.png", "00005.png"], c0
    assert c1 == ["00005.png"], c1
    assert sc == ["00000.png", "00010.png"], sc
    print("[3/7] sparse writing + 5-padded stems OK")

    # Mask content: uint8 {0,255}, correct pixels, dropped classes absent.
    m = np.array(Image.open(raw / "masks_instance" / "chair_0" / "00000.png"))
    assert m.dtype == np.uint8 and set(np.unique(m)) == {0, 255}
    assert m[:10, :10].min() == 255 and m[10:, :].max() == 0
    assert not any("otherfurniture" in d or "ceiling" in d for d in seg_dirs)
    assert set(stats["dropped_out_of_taxonomy"].values()) == {39, 22, -1}
    print("[4/7] mask content + background handling (incl. unmappable label) OK")

    # Union consistency: masks/chair = chair_0 | chair_1 per frame.
    u5 = np.array(Image.open(raw / "masks" / "chair" / "00005.png")) > 127
    i5 = (np.array(Image.open(raw / "masks_instance" / "chair_0" / "00005.png")) > 127) \
        | (np.array(Image.open(raw / "masks_instance" / "chair_1" / "00005.png")) > 127)
    assert (u5 == i5).all()
    assert sorted(p.name for p in (raw / "masks").iterdir()) == ["chair", "shower_curtain", "wall"]
    print("[5/7] per-class union consistency OK")

    # QA stats: zero cross-class duplicates, correct counts, subset copied.
    assert stats["num_instances"] == 4
    assert stats["instances_per_class"] == {"chair": 2, "shower curtain": 1, "wall": 1}
    assert stats["cross_class_duplicates_iou50"] == 0
    assert stats["cross_class_max_iou"] == 0.0
    assert stats["min_label_purity"] == 1.0
    assert json.load(open(raw / "_qa" / "stats.json"))["num_instances"] == 4
    assert (raw / "subset" / "00015.jpg").exists()
    print("[6/7] QA stats + subset copy OK")

    # Loader round-trip: the built tree must parse with instance_level=True.
    try:
        from data.scannet_overfit import ScanNetSingleSceneDataset
        ds = ScanNetSingleSceneDataset(str(raw), num_frames=3, instance_level=True)
        names = sorted(c for c, _ in ds.segments)
        assert names == ["chair", "chair", "shower curtain", "wall"], names
        print("[7/7] loader round-trip OK:", names)
    except ImportError as e:  # torch not installed in a bare CPU env
        print(f"[7/7] loader round-trip SKIPPED ({e})")

    print("\nAll build_official_masks tests passed.")


if __name__ == "__main__":
    main()
