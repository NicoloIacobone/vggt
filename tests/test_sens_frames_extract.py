#!/usr/bin/env python3
"""CPU tests for the dense .sens frame extractor (docs/todo.md 6k).

`legacy/dataset_build/scripts/extract_sens_frames25k.py` parses a binary format by hand off
a network stream, so every offset in its header/frame arithmetic is a silent-corruption risk:
get one field width wrong and it still "works" — it writes plausible jpegs with the wrong
poses attached, which no downstream check would catch. So the parser is exercised here
against a SYNTHETIC .sens built to the official SensorData.py layout, with no network.

What the real build additionally verified once, against the official `scannet_frames_25k`
export of scene0011_00 (docs/DATASET.md §2.5): identical frame stems, **depth pixel-identical**
(max |diff| 0), poses equal to 5e-6, intrinsics to 5e-3. The color jpegs are NOT byte-identical
because the official export re-compressed them (102 KB vs our 260 KB original payload) — ours
carry the .sens bytes verbatim.

    python tests/test_sens_frames_extract.py
"""
import io
import struct
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "legacy" / "dataset_build" / "scripts"))

from extract_sens_frames25k import check_scene, extract_frames_from_stream  # noqa: E402

from PIL import Image  # noqa: E402

CW, CH, DW, DH = 1296, 968, 640, 480
FAIL = 0


def check(name, expected, actual):
    global FAIL
    if expected == actual:
        print(f"PASS: {name}")
    else:
        print(f"FAIL: {name}\n  expected: {expected}\n  actual:   {actual}")
        FAIL = 1


def check_raises(name, fn, needle):
    global FAIL
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        if needle in str(e):
            print(f"PASS: {name}")
        else:
            print(f"FAIL: {name}\n  wanted {needle!r} in: {e}")
            FAIL = 1
        return
    print(f"FAIL: {name} — no exception raised")
    FAIL = 1


def a_jpeg(color=(10, 20, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (CW, CH), color).save(buf, format="JPEG", quality=50)
    return buf.getvalue()


def make_sens(num_frames=10, version=4, color_comp=2, depth_comp=1,
              cw=CW, ch=CH, depth_shift=1000.0, jpeg_ok=True, depth_len_ok=True,
              depths=None, poses=None) -> bytes:
    """A synthetic .sens laid out exactly as SensReader/python/SensorData.py writes it."""
    out = io.BytesIO()
    out.write(struct.pack("I", version))
    name = b"synthetic"
    out.write(struct.pack("Q", len(name)))
    out.write(name)
    k_color = np.eye(4, dtype=np.float32)
    k_color[0, 0], k_color[1, 1], k_color[0, 2], k_color[1, 2] = 1170.1, 1170.2, 647.7, 483.8
    k_depth = np.eye(4, dtype=np.float32)
    k_depth[0, 0], k_depth[1, 1], k_depth[0, 2], k_depth[1, 2] = 577.5, 577.6, 319.5, 239.5
    out.write(k_color.tobytes())                       # intrinsic_color
    out.write(np.eye(4, dtype=np.float32).tobytes())   # extrinsic_color
    out.write(k_depth.tobytes())                       # intrinsic_depth
    out.write(np.eye(4, dtype=np.float32).tobytes())   # extrinsic_depth
    out.write(struct.pack("ii", color_comp, depth_comp))
    out.write(struct.pack("iiii", cw, ch, DW, DH))
    out.write(struct.pack("f", depth_shift))
    out.write(struct.pack("Q", num_frames))

    jpg = a_jpeg() if jpeg_ok else b"NOTAJPEG" * 40
    for i in range(num_frames):
        pose = poses[i] if poses is not None else np.full((4, 4), float(i), dtype=np.float32)
        out.write(np.asarray(pose, dtype=np.float32).tobytes())
        out.write(struct.pack("QQ", 1000 + i, 2000 + i))   # 2 timestamps
        d = depths[i] if depths is not None else np.full((DH, DW), 100 * (i + 1), np.uint16)
        raw = d.astype(np.uint16).tobytes()
        if not depth_len_ok:
            raw = raw[:-2]
        dz = zlib.compress(raw)
        out.write(struct.pack("QQ", len(jpg), len(dz)))
        out.write(jpg)
        out.write(dz)
    return out.getvalue()


def run(blob, tmp, stride=2, max_frames=0):
    d = Path(tmp)
    n = extract_frames_from_stream(io.BytesIO(blob), d, stride=stride, max_frames=max_frames)
    return n, d


# --- 1. the happy path ------------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    depths = [np.random.default_rng(i).integers(0, 6000, (DH, DW), dtype=np.uint16)
              for i in range(10)]
    poses = [np.random.default_rng(100 + i).normal(size=(4, 4)).astype(np.float32)
             for i in range(10)]
    n, d = run(make_sens(10, depths=depths, poses=poses), tmp, stride=3)
    check("stride 3 of 10 frames writes 4", 4, n)
    check("stems are the RAW frame index, %06d",
          ["000000", "000003", "000006", "000009"],
          sorted(p.stem for p in (d / "color").glob("*.jpg")))
    check("one depth png per color jpg", 4, len(list((d / "depth").glob("*.png"))))
    check("one pose txt per color jpg", 4, len(list((d / "pose").glob("*.txt"))))
    check("check_scene accepts the result", None, check_scene(d))
    got = np.asarray(Image.open(d / "depth" / "000006.png"))
    check("depth png round-trips the uint16 millimetres bit-for-bit",
          True, bool(np.array_equal(got, depths[6])))
    check("depth png stays 16-bit", "I;16", Image.open(d / "depth" / "000006.png").mode)
    p = np.loadtxt(d / "pose" / "000009.txt")
    check("pose is the frame's OWN camera_to_world, not a neighbour's",
          True, bool(np.allclose(p, poses[9], atol=1e-6)))
    kc = np.loadtxt(d / "intrinsics_color.txt")
    check("intrinsics_color comes from the header's FIRST matrix",
          True, bool(abs(kc[0, 0] - 1170.1) < 1e-3 and abs(kc[1, 2] - 483.8) < 1e-3))
    kd = np.loadtxt(d / "intrinsics_depth.txt")
    check("intrinsics_depth comes from the header's THIRD matrix (not the extrinsic)",
          True, bool(abs(kd[0, 0] - 577.5) < 1e-3 and abs(kd[1, 2] - 239.5) < 1e-3))
    check("color jpg is the payload verbatim", a_jpeg(), (d / "color" / "000000.jpg").read_bytes())

# --- 2. max_frames caps and resamples uniformly ------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    n, d = run(make_sens(100), tmp, stride=1, max_frames=5)
    check("max_frames caps the count", 5, n)
    check("…and spreads them over the WHOLE scan",
          ["000000", "000025", "000050", "000074", "000099"],
          sorted(p.stem for p in (d / "color").glob("*.jpg")))

# --- 3. stride longer than the scene ------------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    n, _ = run(make_sens(3), tmp, stride=100)
    check("stride past the end still writes frame 0", 1, n)

# --- 4. every format guard fires ----------------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    check_raises("version != 4 rejected",
                 lambda: run(make_sens(2, version=3), tmp), "version 3")
    check_raises("non-jpeg color compression rejected",
                 lambda: run(make_sens(2, color_comp=0), tmp), "color compression 0")
    check_raises("non-zlib depth compression rejected — it would decode as garbage",
                 lambda: run(make_sens(2, depth_comp=0), tmp), "depth compression 0")
    check_raises("unknown color resolution rejected",
                 lambda: run(make_sens(2, cw=800, ch=600), tmp), "800x600")
    check_raises("depth_shift != 1000 rejected — the pipeline divides by 1000",
                 lambda: run(make_sens(2, depth_shift=4000.0), tmp), "depth_shift")
    check_raises("a color payload that is not JPEG is rejected",
                 lambda: run(make_sens(2, jpeg_ok=False), tmp), "not JPEG")
    check_raises("a short depth payload is rejected, not reshaped",
                 lambda: run(make_sens(2, depth_len_ok=False), tmp), "depth payload")

# --- 5. check_scene catches an incomplete tree --------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    n, d = run(make_sens(4), tmp, stride=2)
    (d / "pose" / "000002.txt").unlink()
    check("check_scene catches a missing pose", "color/pose mismatch (2 vs 1)", check_scene(d))
    (d / "pose" / "000002.txt").write_text("0 0 0 0\n0 0 0 0\n0 0 0 0\n0 0 0 0\n")
    (d / "depth" / "000000.png").unlink()
    check("check_scene catches a missing depth", "color/depth mismatch (2 vs 1)", check_scene(d))

print()
print("all extract_sens_frames25k checks passed" if not FAIL else "FAILURES ABOVE")
sys.exit(FAIL)
