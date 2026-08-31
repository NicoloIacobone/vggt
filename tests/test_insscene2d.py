"""
CPU tests for the InsScene-15K 2D build (todo 6f): slurm/insscene_shards.py + build_insscene2d.py.

Standalone, no cluster data, no GPU — run with `myenv/bin/python tests/test_insscene2d.py`.

What is actually at risk here, and therefore what is tested:
  * the split-zip reader must reassemble members that straddle a part boundary, or the build
    silently produces corrupt images on a 53-part archive;
  * instance ids must be remapped ONCE PER SCENE, because the multi-frame GT re-links instances
    across views by id;
  * the ScanNet++ evaluation scenes must be droppable, or training leaks docs/RESULTS.md §7;
  * Infinigen's room shell must be dropped by NAME, matching the ScanNet/Replica convention;
  * RE10K's rgb stems must be ordered NUMERICALLY, because `masklet` is indexed by frame position
    and the stems are 8 OR 9 digits long — a lexicographic sort silently misaligns masks and
    images in the 107 scenes that mix the two widths;
  * RE10K's room shell must be dropped by AREA (it has no names), scene-wide rather than
    per frame, so an instance never flickers in and out of the multi-frame GT;
  * ASE's frames must be rotated to upright with rgb and ids going through the SAME rotation --
    a mismatch there silently trains the head on masks that do not cover their objects -- and
    its shell must go by area on the same scene-wide rule, since ASE ships no id->name table.

The COCO-RLE decoder those last two rest on is tested separately, against `pycocotools` itself:
`tests/test_coco_rle.py`.
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
    RE10K_MAX_AREA_FRAC,
    ase_frame_indices,
    ase_keep_ids,
    ase_probe,
    even_indices,
    infinigen_keep_ids,
    re10k_frame_stems,
    re10k_keep_ids,
    remap_scene_ids,
    resize_instance_map,
    rotate_upright,
    write_scene,
)
from slurm.coco_rle import encode_counts  # noqa: E402
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


def _rle(mask: np.ndarray) -> dict:
    """A bool mask → the COCO RLE dict the RE10K masklets carry. Test helper."""
    flat = mask.T.ravel().astype(np.int8)
    edges = np.concatenate([[0], np.flatnonzero(np.diff(flat)) + 1, [flat.size]])
    runs = np.diff(edges).tolist()
    if flat[0] != 0:
        runs = [0] + runs
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": encode_counts(runs)}


def test_re10k_stems_are_ordered_numerically():
    """8- vs 9-digit timestamps: a lexicographic sort misaligns 107 real scenes."""
    members = [
        "processed_re10k/aaaa/rgb/99999999.png",       # 8 digits, LATER than the 9-digit ones
        "processed_re10k/aaaa/rgb/100000000.png",
        "processed_re10k/aaaa/rgb/100033367.png",
        "processed_re10k/aaaa/cam/99999999.npz",       # a sibling directory, not a frame
        "processed_re10k/sam2_results/aaaa/auto_masks.json",
        "processed_re10k/bbbb/rgb/000000010.png",
    ]
    index = re10k_frame_stems(members)
    check(sorted(index) == ["aaaa", "bbbb"],
          f"only scene dirs with rgb frames are indexed, got {sorted(index)}")
    check(index["aaaa"] == ["99999999", "100000000", "100033367"],
          f"stems sort NUMERICALLY, not lexicographically — got {index['aaaa']}")
    check(index["aaaa"] != sorted(index["aaaa"]),
          "the fixture really does distinguish the two orders (8- and 9-digit stems)")
    try:
        re10k_frame_stems(["processed_re10k/cccc/rgb/not_a_number.png"])
    except ValueError as exc:
        check("non-numeric" in str(exc), "a non-numeric stem raises rather than sorting wrongly")
    else:
        raise AssertionError("a non-numeric rgb stem was accepted")


def test_re10k_shell_is_dropped_by_area_scene_wide():
    h, w = 20, 30
    shell = np.zeros((h, w), dtype=bool); shell[:, :] = True          # 100 % of the frame
    half = np.zeros((h, w), dtype=bool); half[:10, :] = True          # 50 %
    obj = np.zeros((h, w), dtype=bool); obj[2:6, 2:8] = True          # 4 %
    # frame 0 shows all three; frame 1 shows only the object, so `half` averages to 25 % < 30 %
    masklet = [[_rle(shell), _rle(half), _rle(obj)],
               [None, _rle(np.zeros((h, w), dtype=bool)), _rle(obj)]]
    keep, missing = re10k_keep_ids(masklet, [0, 1], h * w, RE10K_MAX_AREA_FRAC)
    check(keep == {2, 3}, f"the 100 %-of-frame shell goes, the object stays — got {sorted(keep)}")
    check(missing == 1, f"`None` masklet entries are counted, got {missing}")
    check(2 in keep,
          "the rule is the SCENE-WIDE mean, not a per-frame threshold — a 50 %/0 % instance "
          "averages to 25 % and survives, so its id cannot flicker across the bundle")
    keep_strict, _ = re10k_keep_ids(masklet, [0, 1], h * w, 0.20)
    check(keep_strict == {3}, "a tighter cap also removes it — the threshold is the only knob")
    check(re10k_keep_ids(masklet, [], h * w, RE10K_MAX_AREA_FRAC) == (set(), 0),
          "a scene with no picked frames keeps nothing rather than dividing by zero")


def test_re10k_empty_masklet_is_not_an_instance():
    h, w = 8, 8
    blank = _rle(np.zeros((h, w), dtype=bool))
    keep, _ = re10k_keep_ids([[blank]], [0], h * w, RE10K_MAX_AREA_FRAC)
    check(keep == set(), "a masklet with zero area never becomes an instance")


# --------------------------------------------------------------------------------------------
# ase (todo 6n)
# --------------------------------------------------------------------------------------------

def _make_ase_scene(root: Path, scene: str, indices, size=(6, 8), extra_rgb=(), extra_inst=()):
    """A minimal ASE scene tree: rgb/vignette%07d.jpg + instances/instance%07d.png."""
    (root / scene / "rgb").mkdir(parents=True, exist_ok=True)
    (root / scene / "instances").mkdir(parents=True, exist_ok=True)
    for i in list(indices) + list(extra_rgb):
        Image.fromarray(np.zeros((*size, 3), dtype=np.uint8)).save(
            root / scene / "rgb" / f"vignette{i:07d}.jpg")
    for i in list(indices) + list(extra_inst):
        Image.fromarray(np.zeros(size, dtype=np.uint16), mode="I;16").save(
            root / scene / "instances" / f"instance{i:07d}.png")


def test_ase_frame_indices_need_both_images(tmp: Path):
    root = tmp / "ase"
    _make_ase_scene(root, "7", [0, 1, 2], extra_rgb=[9], extra_inst=[11])
    got = ase_frame_indices(root / "7")
    check(got == [0, 1, 2],
          f"only frames with BOTH an rgb and an instance map are built, got {got}")


def test_ase_frame_indices_are_numeric(tmp: Path):
    root = tmp / "ase_numeric"
    _make_ase_scene(root, "3", [2, 10, 100])
    got = ase_frame_indices(root / "3")
    check(got == [2, 10, 100], f"frame indices come back in numeric order, got {got}")


def test_ase_rotation_is_identical_for_rgb_and_ids():
    """
    The single failure that would be invisible downstream: masks that no longer cover their
    object. rgb and ids must land on the same pixel after the upright rotation.
    """
    ids = np.zeros((4, 6), dtype=np.int32)
    ids[0, 5] = 42                       # one corner pixel, unambiguous under a rotation
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    rgb[0, 5] = (255, 0, 0)

    r_ids, r_rgb = rotate_upright(ids), rotate_upright(rgb)
    check(r_ids.shape == (6, 4) and r_rgb.shape == (6, 4, 3),
          f"a -90 rotation transposes the frame, got {r_ids.shape} / {r_rgb.shape}")
    where_id = tuple(np.argwhere(r_ids == 42)[0])
    where_rgb = tuple(np.argwhere(r_rgb[..., 0] == 255)[0])
    check(where_id == where_rgb,
          f"the id and its pixel land together after rotation: {where_id} vs {where_rgb}")
    check(np.array_equal(rotate_upright(rotate_upright(rotate_upright(r_ids))), ids),
          "four rotations are the identity, i.e. it really is a 90 deg turn")


def test_ase_shell_is_dropped_by_area_scene_wide():
    """
    Same rule as RE10K's: averaged over the scene, not thresholded per frame. Id 1 covers half
    of frame A and none of frame B -- a per-frame rule at 0.30 would drop it in A and keep it
    in B, making it flicker in the multi-frame GT.
    """
    a = np.zeros((10, 10), dtype=np.int32)
    a[:5, :] = 1            # 50 % of frame A
    a[5:7, :2] = 2          # 4 %
    b = np.zeros((10, 10), dtype=np.int32)
    b[0:2, 0:2] = 2         # 4 %
    per_frame = {"a": a, "b": b}

    keep = ase_keep_ids(per_frame, 0.30)
    check(keep == {1, 2},
          f"id 1 averages 25 % over the two frames and SURVIVES a 0.30 cap, got {keep}")
    check(ase_keep_ids(per_frame, 0.20) == {2},
          "at a 0.20 cap the same id is shell — the cap is what moves, not the rule")
    check(ase_keep_ids({"a": a}, 0.30) == {2},
          "on frame A alone id 1 is 50 % and is dropped — the average is over the SCENE")


def test_ase_probe_measures_and_applies_nothing():
    ids = np.zeros((10, 10), dtype=np.int32)
    ids[:6, :] = 1          # 60 %
    ids[6:8, :5] = 2        # 10 %
    ids[8, 0] = 3           # 1 %
    stats = ase_probe({"a": ids})
    check(stats["instances"] == 3, f"every non-zero id is counted, got {stats['instances']}")
    check(abs(stats["area_frac_max"] - 0.60) < 1e-6,
          f"the largest instance is reported at its true area, got {stats['area_frac_max']}")
    check(stats["dropped_at"]["0.3"] == 1 and stats["dropped_at"]["0.5"] == 1,
          f"a 0.30 cap would remove exactly the 60 % instance, got {stats['dropped_at']}")
    check(stats["dropped_at"]["0.1"] == 1,
          "a 0.10 cap removes the 60 % one and keeps the 10 % one (<= is inclusive)")


def test_ase_empty_map_yields_no_instances():
    check(ase_keep_ids({"a": np.zeros((4, 4), dtype=np.int32)}, 0.30) == set(),
          "a frame with only background produces no ASE instance")
    check(ase_probe({"a": np.zeros((4, 4), dtype=np.int32)})["instances"] == 0,
          "and the probe reports zero rather than dividing by it")


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
        test_re10k_stems_are_ordered_numerically()
        test_re10k_shell_is_dropped_by_area_scene_wide()
        test_re10k_empty_masklet_is_not_an_instance()
        test_ase_frame_indices_need_both_images(tmp)
        test_ase_frame_indices_are_numeric(tmp)
        test_ase_rotation_is_identical_for_rgb_and_ids()
        test_ase_shell_is_dropped_by_area_scene_wide()
        test_ase_probe_measures_and_applies_nothing()
        test_ase_empty_map_yields_no_instances()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for message in PASSED:
        print(f"  ok  {message}")
    print(f"\n{len(PASSED)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
