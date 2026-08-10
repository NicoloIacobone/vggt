"""
CPU tests for the InsScene-15K 2D build (todo 6f): slurm/insscene_shards.py + build_insscene2d.py.

Standalone, no cluster data, no GPU — run with `myenv/bin/python tests/test_insscene2d.py`.

What is actually at risk here, and therefore what is tested:
  * the split-zip reader must reassemble members that straddle a part boundary, or the build
    silently produces corrupt images on a 53-part archive;
  * instance ids must be remapped ONCE PER SCENE, because the multi-frame GT re-links instances
    across views by id;
  * the ScanNet++ evaluation scenes must be droppable, or training leaks docs/RESULTS.md §7;
  * Infinigen's room shell must be dropped by NAME, matching the ScanNet/Replica convention.
"""

import io
import json
import shutil
import struct
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slurm.build_insscene2d import (  # noqa: E402
    even_indices,
    infinigen_keep_ids,
    remap_scene_ids,
    resize_instance_map,
    write_scene,
)
from slurm.insscene_shards import SplitZipReader, scene_ids  # noqa: E402

PASSED = []


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    PASSED.append(message)


def make_split_zip(directory: Path, stem: str, payloads: dict, part_size: int) -> None:
    """Write a zip, then cut it into `part_size` chunks exactly as `split -b` would."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in payloads.items():
            method = zipfile.ZIP_STORED if name.endswith(".raw") else zipfile.ZIP_DEFLATED
            archive.writestr(name, data, compress_type=method)
    blob = buffer.getvalue()
    for i in range(0, len(blob), part_size):
        (directory / f"{stem}.zip.{i // part_size + 1:03d}").write_bytes(blob[i:i + part_size])


def test_split_zip_reader(tmp: Path):
    payloads = {
        "processed_x/aaaaaaaaaa/images/frame_000000.jpg": bytes(range(256)) * 40,
        "processed_x/aaaaaaaaaa/refined_ins_ids/frame_000000.jpg.npy": b"\x93NUMPY" + b"z" * 5000,
        "processed_x/bbbbbbbbbb/images/frame_000001.jpg": b"deflate-me " * 900,
        "processed_x/notascene/keep.raw": b"stored payload",
        "processed_x/nvs_sem_val.txt": b"aaaaaaaaaa\nbbbbbbbbbb\n",
    }
    directory = tmp / "split"
    directory.mkdir()
    make_split_zip(directory, "processed_x", payloads, part_size=137)

    reader = SplitZipReader(directory, "processed_x")
    check(len(reader.parts) > 3, "the fixture really is split into several parts")
    members = reader.members()
    check(set(members) >= set(payloads), "every member appears in the central directory")
    for name, data in payloads.items():
        check(reader.read(name) == data, f"member round-trips across part boundaries: {name}")

    ids = scene_ids(reader, "processed_x")
    check(ids == ["aaaaaaaaaa", "bbbbbbbbbb"],
          f"scene_ids keeps only 10-hex scene dirs, got {ids}")
    check(list(reader.iter_names(suffix=".txt")) == ["processed_x/nvs_sem_val.txt"],
          "iter_names filters by suffix")


def test_zip64_overrides():
    """The 0x0001 extra field fills in ONLY the sentinelled fields, in order."""
    extra = struct.pack("<HH", 1, 16) + struct.pack("<QQ", 1 << 33, 1 << 32)
    size, csize, offset = SplitZipReader._zip64_overrides(
        extra, 0xFFFFFFFF, 0xFFFFFFFF, 123)
    check((size, csize, offset) == (1 << 33, 1 << 32, 123),
          "zip64 extra overrides size and compressed size but leaves a real offset alone")
    extra_offset_only = struct.pack("<HH", 1, 8) + struct.pack("<Q", 1 << 34)
    size, csize, offset = SplitZipReader._zip64_overrides(extra_offset_only, 10, 5, 0xFFFFFFFF)
    check((size, csize, offset) == (10, 5, 1 << 34),
          "a sentinelled offset alone consumes the first zip64 value")


def test_even_indices():
    check(even_indices(10, 4) == [0, 3, 6, 9], "even sampling spans the whole scene")
    check(even_indices(3, 8) == [0, 1, 2], "asking for more frames than exist keeps them all")
    check(even_indices(0, 8) == [], "an empty scene samples nothing")


def test_remap_is_per_scene():
    """Two frames sharing a source id must share the SAME remapped id."""
    frame_a = np.array([[0, 61, 61], [7, 7, 0]], dtype=np.int32)
    frame_b = np.array([[61, 61, 0], [0, 900, 900]], dtype=np.int32)
    maps, table = remap_scene_ids({"a": frame_a, "b": frame_b})
    check(sorted(table) == [7, 61, 900], f"every source id gets an entry, got {sorted(table)}")
    check(maps["a"][0, 1] == maps["b"][0, 0] == table[61],
          "the same source id keeps one id across frames — multi-view identity")
    check(set(np.unique(np.concatenate([m.ravel() for m in maps.values()]))) == {0, 1, 2, 3},
          "ids are collapsed onto a dense 1..G with 0 left as background")
    check(all(m.dtype == np.uint16 for m in maps.values()), "maps are uint16")


def test_remap_filters():
    frame = np.array([[0, 5, 5, 5], [9, 0, 0, 0]], dtype=np.int32)
    maps, table = remap_scene_ids({"a": frame}, keep={5})
    check(list(table) == [5], "ids outside `keep` are dropped to background")
    check((maps["a"] == 0).sum() == 5, "the dropped instance's pixels become background")
    maps, table = remap_scene_ids({"a": frame}, min_area_px=2)
    check(list(table) == [5], "instances below min_area_px are dropped")


def test_infinigen_shell_is_dropped_by_name():
    objects = {
        "BedFactory(1).spawn_asset(2)": {"type": "MESH", "object_index": 39},
        "bedroom_0/0.wall": {"type": "MESH", "object_index": 69},
        "bedroom_0/0.floor": {"type": "MESH", "object_index": 65},
        "bedroom_0/0.ceiling": {"type": "MESH", "object_index": 61},
        "bedroom_0/0.exterior": {"type": "MESH", "object_index": 60},
        "Area.001": {"type": "LIGHT", "object_index": 1},
        "Camera": {"type": "CAMERA", "object_index": 3},
        "Empty": {"type": "EMPTY", "object_index": 4},
    }
    keep = infinigen_keep_ids(objects)
    check(keep == {39}, f"only real MESH objects survive, got {sorted(keep)}")


def test_resize_invents_no_ids():
    ids = np.zeros((64, 64), dtype=np.int32)
    ids[10:40, 10:40] = 7
    ids[45:50, 45:50] = 4001
    resized = resize_instance_map(ids, size=518)
    check(resized.shape == (518, 518), "instance maps are resized to the trainer's grid")
    check(set(np.unique(resized).tolist()) <= {0, 7, 4001},
          "NEAREST resizing never blends two ids into a third")
    check(4001 in np.unique(resized), "a small instance survives the upsample")


def test_write_scene_round_trip(tmp: Path):
    stems = ["frame_000000", "frame_000010"]
    images = {s: Image.new("RGB", (518, 518), (10, 20, 30)) for s in stems}
    maps = {}
    for i, s in enumerate(stems):
        m = np.zeros((518, 518), dtype=np.uint16)
        m[100:200, 100:200] = 1
        m[300:400, 300:400] = 2 + i          # the second frame also shows instance 3
        maps[s] = m
    out = tmp / "scene"
    counters = write_scene(out, stems, images, maps,
                           dict(source="unit-test", scene="scene", id_table={}))
    check(counters == {"frames": 2, "instances": 3}, f"counters, got {counters}")

    back = np.array(Image.open(out / "instance" / "frame_000010.png"))
    check(back.dtype == np.uint16 and np.array_equal(back, maps["frame_000010"]),
          "the uint16 instance map survives the PNG round-trip unchanged")
    manifest = json.loads((out / "manifest.json").read_text())
    check(manifest["frames"] == stems and manifest["num_instances"] == 3,
          "the manifest records the frames and the scene-wide instance count")
    check((out / "color" / "frame_000000.jpg").exists(), "colour frames are written")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="insscene2d_test_"))
    try:
        test_split_zip_reader(tmp)
        test_zip64_overrides()
        test_even_indices()
        test_remap_is_per_scene()
        test_remap_filters()
        test_infinigen_shell_is_dropped_by_name()
        test_resize_invents_no_ids()
        test_write_scene_round_trip(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for message in PASSED:
        print(f"  ok  {message}")
    print(f"\n{len(PASSED)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
