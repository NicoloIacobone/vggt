"""Test scripts/extract_sens_subset.py — synthetic .sens stream -> stride-5 subset jpgs.

Builds a tiny in-memory .sens v4 file (JPEG color payloads, dummy depth), runs the
streaming extractor, and asserts: frame selection (0,5,10,...), 5-padded naming,
JPEG payload integrity, early abort (nothing read past the last wanted frame),
short-scene handling, and the format guards (version/compression/resolution).

CPU-only, no network. Run: python tests/test_extract_sens_subset.py
"""
import io
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.extract_sens_subset import (  # noqa: E402
    EarlyAbort, extract_subset_from_stream,
)

W, H = 1296, 968


def jpeg_bytes(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    img = Image.fromarray(rng.integers(0, 255, (8, 8, 3), dtype=np.uint8).repeat(H // 8, 0).repeat(W // 8, 1)[:H, :W])
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    return buf.getvalue()


def build_sens(num_frames: int, version=4, color_comp=2, res=(W, H)) -> tuple[bytes, list[bytes]]:
    """Minimal .sens v4 byte blob (layout per official SensorData.py)."""
    out = io.BytesIO()
    out.write(struct.pack("I", version))
    name = b"synthetic"
    out.write(struct.pack("Q", len(name)) + name)
    out.write(struct.pack("f" * 64, *([0.0] * 64)))          # 4 camera matrices
    out.write(struct.pack("ii", color_comp, 1))               # color jpeg, depth zlib
    out.write(struct.pack("iiii", res[0], res[1], 640, 480))
    out.write(struct.pack("f", 1000.0))                       # depth_shift
    out.write(struct.pack("Q", num_frames))
    payloads = []
    for i in range(num_frames):
        color = jpeg_bytes(i)
        depth = bytes(100 + i)                                # arbitrary compressed depth
        payloads.append(color)
        out.write(struct.pack("f" * 16, *([0.0] * 16)))       # camera_to_world
        out.write(struct.pack("QQ", i, i))                    # timestamps
        out.write(struct.pack("QQ", len(color), len(depth)))
        out.write(color)
        out.write(depth)
    return out.getvalue(), payloads


class CountingReader(io.BytesIO):
    """BytesIO that records how many bytes were consumed (early-abort check)."""

    def __init__(self, data):
        super().__init__(data)
        self.consumed = 0

    def read(self, n=-1):
        chunk = super().read(n)
        self.consumed += len(chunk)
        return chunk


def run(data, tmp, **kw):
    fh = CountingReader(data)
    out = Path(tmp) / "subset"
    try:
        written = extract_subset_from_stream(fh, out, **kw)
    except EarlyAbort as e:
        written = e.args[0]
    return written, out, fh


def main():
    tmp = tempfile.mkdtemp(prefix="test_sens_")

    # 12-frame scene, stride 5 -> frames 0,5,10; early abort after frame 10.
    data, payloads = build_sens(12)
    written, out, fh = run(data, Path(tmp) / "a")
    names = sorted(p.name for p in out.iterdir())
    assert written == 3 and names == ["00000.jpg", "00005.jpg", "00010.jpg"], names
    assert (out / "00005.jpg").read_bytes() == payloads[5], "payload must be byte-identical"
    assert fh.consumed < len(data), "early abort must not consume the whole stream"
    print(f"[1/5] stride selection + naming + byte-identical jpgs + early abort "
          f"({fh.consumed}/{len(data)} bytes read) OK")

    # Short scene (3 frames): only frame 0 qualifies; no abort needed past EOF.
    data3, _ = build_sens(3)
    written, out, _ = run(data3, Path(tmp) / "b")
    assert written == 1 and sorted(p.name for p in out.iterdir()) == ["00000.jpg"]
    print("[2/5] short-scene handling OK")

    # More frames wanted than exist (count=100 over 12 frames) is the same path;
    # explicit small count: count=2 -> frames 0,5 only.
    written, out, _ = run(data, Path(tmp) / "c", count=2)
    assert written == 2 and sorted(p.name for p in out.iterdir()) == ["00000.jpg", "00005.jpg"]
    print("[3/5] count limit OK")

    # Format guards. 640x480 is ALLOWED (low-res color-camera scenes, e.g.
    # scene0240_00 — GT projections match that resolution); others rejected.
    ok640, _ = build_sens(2, res=(640, 480))
    written, out, _ = run(ok640, Path(tmp) / "lowres")
    assert written == 1 and (out / "00000.jpg").exists()
    for bad_kw, msg in [(dict(version=3), "version"),
                        (dict(color_comp=1), "compression"),
                        (dict(res=(320, 240)), "resolution")]:
        bad, _ = build_sens(2, **bad_kw)
        try:
            run(bad, Path(tmp) / f"g_{msg}")
            raise AssertionError(f"expected ValueError for bad {msg}")
        except ValueError:
            pass
    print("[4/5] format guards (640x480 allowed; version/compression/other-res rejected) OK")

    # Truncated stream mid-frame -> IOError, not silent partial success.
    try:
        run(data[: len(data) // 4], Path(tmp) / "d")
        raise AssertionError("expected IOError on truncated stream")
    except IOError:
        pass
    print("[5/5] truncated-stream error OK")

    print("\nAll extract_sens_subset tests passed.")


if __name__ == "__main__":
    main()
