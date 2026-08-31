"""
CPU tests for the ASE pilot fetch (todo 6n): slurm/download_ase.py.

Standalone, no cluster data, no network, no GPU — `myenv/bin/python tests/test_ase_fetch.py`.

The download itself is licence-gated, so what is testable here is everything AROUND it, which
is also where the failures would be silent:

  * the scene-id grammar must match the official downloader's (`0-999`, `1,2,5-7`), because a
    misparse silently fetches the wrong scenes and nothing downstream would notice;
  * scene ids must map onto CHUNKS of 10 — the CDN's unit — with no chunk fetched twice and
    none missed at a range boundary;
  * a corrupt chunk must fail on its sha1 rather than be unzipped, and must not leave a
    `.complete` marker, or a resumed job would skip a chunk it never got;
  * a finished chunk MUST be skipped on the next run: a 230 GB pilot does not fit one wall
    clock, so resume is the difference between a pilot and an infinite loop;
  * a missing CDN file must exit with the licence instructions, not a stack trace — it is the
    one manual step in the pipeline and the message is the whole UX.
"""

import io
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slurm.download_ase import (  # noqa: E402
    SCENES_PER_CHUNK,
    chunk_ids_for,
    fetch_chunk,
    parse_scene_ids,
    sha1_file,
)

REPO = Path(__file__).resolve().parent.parent
PASSED = []


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    PASSED.append(message)


def make_chunk_zip(path: Path, scene_ids) -> None:
    """A chunk zip shaped like ASE's: one directory per scene, rgb/ + instances/."""
    with zipfile.ZipFile(path, "w") as archive:
        for scene in scene_ids:
            archive.writestr(f"{scene}/rgb/vignette0000000.jpg", b"jpeg")
            archive.writestr(f"{scene}/instances/instance0000000.png", b"png")


def test_scene_id_grammar():
    check(parse_scene_ids("0-999") == list(range(1000)), "a range is inclusive on both ends")
    check(parse_scene_ids("3") == [3], "a bare id is one scene")
    check(parse_scene_ids("1,2,5-7") == [1, 2, 5, 6, 7], "commas and ranges mix")
    check(parse_scene_ids("5-7,6") == [5, 6, 7], "an id repeated by a range is not fetched twice")
    check(parse_scene_ids(" 2 , 1 ") == [1, 2], "whitespace is tolerated and output is sorted")


def test_chunking_is_by_ten():
    check(SCENES_PER_CHUNK == 10, "the CDN packs 10 scenes per chunk — not ours to change")
    check(chunk_ids_for([0]) == [0] and chunk_ids_for([9]) == [0],
          "scenes 0..9 live in chunk 0")
    check(chunk_ids_for([10]) == [1], "scene 10 starts chunk 1 — the boundary is exclusive")
    check(chunk_ids_for(list(range(1000))) == list(range(100)),
          "the 1000-scene pilot is exactly 100 chunks")
    check(chunk_ids_for([5, 7, 8]) == [0],
          "three scenes in one chunk cost ONE download, not three")


def test_good_chunk_unzips_and_markers(tmp: Path):
    out, stage = tmp / "out", tmp / "stage"
    out.mkdir(); stage.mkdir()
    src = tmp / "train_chunk_0000000.zip"
    make_chunk_zip(src, [0, 1])
    entry = {"filename": src.name, "cdn": src.resolve().as_uri(), "sha": sha1_file(src)}

    logs = []
    check(fetch_chunk(entry, out, stage, logs.append) == "ok", "a valid chunk downloads")
    check((out / "0" / "rgb" / "vignette0000000.jpg").exists(),
          "the scene tree is unzipped where the builder expects it")
    check((out / f".{src.name}.complete").exists(), "a finished chunk leaves a marker")
    check(not list(stage.glob("*.zip")), "the zip is deleted — 230 GB does not fit twice")

    check(fetch_chunk(entry, out, stage, logs.append) == "skip",
          "a second run SKIPS it — this is what makes the pilot resumable")


def test_bad_sha_fails_without_markering(tmp: Path):
    out, stage = tmp / "out2", tmp / "stage2"
    out.mkdir(); stage.mkdir()
    src = tmp / "train_chunk_0000001.zip"
    make_chunk_zip(src, [10])
    entry = {"filename": src.name, "cdn": src.resolve().as_uri(), "sha": "0" * 40}

    logs = []
    check(fetch_chunk(entry, out, stage, logs.append) == "fail",
          "a chunk whose sha1 does not match the metadata is a failure")
    check(not (out / "10").exists(), "and it is NEVER unzipped")
    check(not (out / f".{src.name}.complete").exists(),
          "no marker, so a resumed job retries it instead of skipping a chunk it never got")
    check(len(logs) == 3, f"all three attempts are logged, got {len(logs)}")


def test_missing_cdn_file_explains_the_licence(tmp: Path):
    result = subprocess.run(
        [sys.executable, str(REPO / "slurm" / "download_ase.py"),
         "--cdn_file", str(tmp / "nope.json"),
         "--out_dir", str(tmp / "o"), "--tmp_dir", str(tmp / "t")],
        capture_output=True, text=True)
    check(result.returncode == 2, f"a missing CDN file exits 2, got {result.returncode}")
    check("projectaria.com/datasets/ase" in result.stdout,
          "the message names the licence page — it is the only manual step in the pipeline")
    check("Traceback" not in result.stderr, "and it is a message, not a stack trace")


def test_report_counts_inodes_not_bytes(tmp: Path):
    """docs/todo.md 6n gates scaling on the FILE COUNT; the report must actually carry it."""
    out, stage = tmp / "out3", tmp / "stage3"
    out.mkdir(); stage.mkdir()
    src = tmp / "train_chunk_0000002.zip"
    make_chunk_zip(src, [20, 21])
    cdn = tmp / "cdn.json"
    cdn.write_text(json.dumps([{"filename": src.name, "cdn": src.resolve().as_uri(),
                                "sha": sha1_file(src)}]))
    report = tmp / "report.json"
    result = subprocess.run(
        [sys.executable, str(REPO / "slurm" / "download_ase.py"),
         "--cdn_file", str(cdn), "--out_dir", str(out), "--tmp_dir", str(stage),
         "--scene_ids", "20-21", "--report", str(report)],
        capture_output=True, text=True)
    check(result.returncode == 0, f"a clean pilot exits 0, got {result.returncode}: {result.stderr}")
    data = json.loads(report.read_text())
    check(data["scenes_on_disk"] == 2, f"both scenes landed, got {data['scenes_on_disk']}")
    check(data["inodes"] == 4, f"4 files across 2 scenes, got {data['inodes']}")
    check(data["inodes_per_scene"] == 2.0,
          f"the per-scene inode cost is what the gate reads, got {data['inodes_per_scene']}")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="ase_fetch_test_"))
    try:
        test_scene_id_grammar()
        test_chunking_is_by_ten()
        test_good_chunk_unzips_and_markers(tmp)
        test_bad_sha_fails_without_markering(tmp)
        test_missing_cdn_file_explains_the_licence(tmp)
        test_report_counts_inodes_not_bytes(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for message in PASSED:
        print(f"  ok  {message}")
    print(f"\n{len(PASSED)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
