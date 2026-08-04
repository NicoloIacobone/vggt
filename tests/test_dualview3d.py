#!/usr/bin/env python3
"""
CPU tests for the synchronised side-by-side 3D viewer (`demos/dualview3d.py`).

The two properties that make the picture trustworthy:
  - **the panels show the same points.** Whatever the colouring, the geometry and the kept-point
    set must be identical, otherwise "only the colour differs" is a lie and the comparison is
    worthless. That is why the background masks are computed from the image, and why the
    subsample is deterministic.
  - **it shows what the GLB tab shows.** The filtering is a re-implementation of
    `visual_util.predictions_to_glb`'s, so it is asserted vertex-for-vertex against it.

The browser side (WebGL, camera, pointer handling) cannot be executed here — no headless
browser in this environment — so the page is checked structurally: one shared camera object,
one canvas per panel, payload decodable, nothing unescaped in the srcdoc attribute.

    python tests/test_dualview3d.py
"""

import base64
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demos"))

from dualview3d import (DEFAULT_MAX_POINTS, build_payload, dual_view_html, filtered_cloud,
                        message_html, panel_colors, quantize_positions, selected_frame,
                        standalone_html, subsample_index, viewer_iframe)
from visual_util import predictions_to_glb

rng = np.random.default_rng(0)


def _predictions(S=3, H=8, W=8):
    """A synthetic reconstruction with the keys both paths read."""
    pts = rng.normal(size=(S, H, W, 3)).astype(np.float32)
    extrinsic = np.tile(np.eye(4)[:3], (S, 1, 1)).astype(np.float32)
    extrinsic[:, :3, 3] = rng.normal(size=(S, 3)) * 0.1
    return {
        "world_points_from_depth": pts,
        "depth_conf": rng.uniform(0.0, 2.0, size=(S, H, W)).astype(np.float32),
        "images": rng.uniform(0, 1, size=(S, H, W, 3)).astype(np.float32),
        "extrinsic": extrinsic,
        "seg_colors": rng.integers(0, 255, size=(S, H, W, 3), dtype=np.uint8),
        "gt_colors": rng.integers(0, 255, size=(S, H, W, 3), dtype=np.uint8),
    }


def _glb_vertices(predictions, **kw):
    """The point cloud the GLB tab would show, in world coordinates."""
    import trimesh

    scene = predictions_to_glb(predictions, show_cam=False, **kw)
    for name, geom in scene.geometry.items():
        if isinstance(geom, trimesh.PointCloud):
            transform = scene.graph.get(scene.graph.geometry_nodes[name][0])[0]
            return np.asarray(geom.vertices) @ transform[:3, :3].T + transform[:3, 3]
    raise AssertionError("no point cloud in the GLB scene")


def test_matches_the_glb_path():
    print("=== Testing agreement with the GLB tab ===")
    predictions = _predictions()
    for kw in ({}, {"conf_thres": 0.0}, {"conf_thres": 90.0},
               {"filter_by_frames": "1: frame_0001.png"}, {"mask_black_bg": True},
               {"mask_white_bg": True}, {"prediction_mode": "Pointmap Branch"}):
        xyz, keep, fi = filtered_cloud(predictions, **kw)
        expected = _glb_vertices(predictions, **kw)
        assert xyz.shape == expected.shape, (kw, xyz.shape, expected.shape)
        assert np.allclose(xyz, expected, atol=1e-5), f"vertices differ from the GLB path: {kw}"
        assert int(keep.sum()) == len(xyz)
    print("✓ same points, same alignment, across every filter combination\n")


def test_panels_share_geometry_and_index():
    print("=== Testing panel alignment ===")
    predictions = _predictions()
    xyz, keep, fi = filtered_cloud(predictions, conf_thres=40.0)
    gt = panel_colors(predictions["gt_colors"], keep, fi)
    seg = panel_colors(predictions["seg_colors"], keep, fi)
    assert len(gt) == len(seg) == len(xyz), (len(gt), len(seg), len(xyz))

    # the same pixel must land on the same point in both panels
    flat_gt = predictions["gt_colors"].reshape(-1, 3)[keep]
    assert np.array_equal(gt, flat_gt)

    # a frame filter slices colours the same way it slices points
    xyz1, keep1, fi1 = filtered_cloud(predictions, filter_by_frames="2: c.png")
    assert fi1 == 2 and len(panel_colors(predictions["seg_colors"], keep1, fi1)) == len(xyz1)
    assert selected_frame("All") is None and selected_frame("junk") is None
    print("✓ colours follow the kept points, frame filter included\n")


def test_payload_round_trip_and_sharing():
    print("=== Testing payload ===")
    n = 500
    pts = rng.normal(size=(n, 3)) * 3.0
    c1 = rng.integers(0, 255, size=(n, 3), dtype=np.uint8)
    c2 = rng.integers(0, 255, size=(n, 3), dtype=np.uint8)
    payload = build_payload([{"label": "GT", "points": pts, "colors": c1},
                             {"label": "Pred", "points": None, "colors": c2}])

    assert payload["panels"][1]["positions"] is None, "geometry must be shared, not duplicated"
    assert payload["panels"][0]["count"] == payload["panels"][1]["count"] == n

    q = np.frombuffer(base64.b64decode(payload["panels"][0]["positions"]), dtype=np.uint16)
    q = q.reshape(-1, 3).astype(np.float64)
    back = q * np.array(payload["panels"][0]["scale"]) + np.array(payload["panels"][0]["offset"])
    # 16 bits over the bounding box: the error must be invisible next to the scene extent
    extent = pts.max(axis=0) - pts.min(axis=0)
    assert np.all(np.abs(back - pts) <= extent / 65535.0 + 1e-9), "quantisation error too large"

    for i, expected in enumerate((c1, c2)):
        got = np.frombuffer(base64.b64decode(payload["panels"][i]["colors"]), dtype=np.uint8)
        assert np.array_equal(got.reshape(-1, 3), expected)

    # a degenerate (single-point / flat) cloud must not divide by zero
    flat = np.zeros((10, 3))
    flat[:, 0] = np.arange(10)
    q, scale, offset = quantize_positions(flat)
    assert scale[1] == 0.0, "a flat axis must get scale 0, not a division by zero"
    assert np.allclose(q[:, 0] * scale[0] + offset[0], flat[:, 0], atol=9.0 / 65535)

    try:
        build_payload([{"label": "x", "points": None, "colors": c1}])
    except ValueError:
        pass
    else:
        raise AssertionError("a leading panel without geometry must be rejected")
    print("✓ base64 round-trip, shared geometry, quantisation error bounded\n")


def test_subsample_keeps_panels_aligned():
    print("=== Testing subsampling ===")
    n, cap = 1000, 100
    idx = subsample_index(n, cap)
    assert len(idx) == cap and np.array_equal(idx, subsample_index(n, cap)), "not deterministic"
    assert idx[0] == 0 and idx[-1] == n - 1 and np.all(np.diff(idx) > 0)
    assert len(subsample_index(50, cap)) == 50, "no subsampling below the cap"

    pts = rng.normal(size=(n, 3))
    c1 = rng.integers(0, 255, size=(n, 3), dtype=np.uint8)
    c2 = rng.integers(0, 255, size=(n, 3), dtype=np.uint8)
    payload = build_payload([{"label": "a", "points": pts, "colors": c1},
                             {"label": "b", "points": None, "colors": c2}], max_points=cap)
    got1 = np.frombuffer(base64.b64decode(payload["panels"][0]["colors"]), np.uint8).reshape(-1, 3)
    got2 = np.frombuffer(base64.b64decode(payload["panels"][1]["colors"]), np.uint8).reshape(-1, 3)
    assert np.array_equal(got1, c1[idx]) and np.array_equal(got2, c2[idx]), \
        "the two panels were subsampled differently — they would show different points"
    print("✓ deterministic stride, identical in every panel\n")


def test_page_structure():
    print("=== Testing the emitted page ===")
    predictions = _predictions()
    doc = standalone_html([{"label": "GT", "points": rng.normal(size=(20, 3)),
                            "colors": rng.integers(0, 255, (20, 3), dtype=np.uint8)},
                           {"label": "Pred", "points": None,
                            "colors": rng.integers(0, 255, (20, 3), dtype=np.uint8)}],
                          title="t")
    assert doc.startswith("<!doctype html>") and doc.rstrip().endswith("</html>")
    assert doc.count("dv3d-data") >= 2 and ">GT<" not in doc  # labels are set from JSON, not HTML
    payload = json.loads(re.search(r"type='application/json'>(.*?)</script>", doc, re.S).group(1))
    assert [p["label"] for p in payload["panels"]] == ["GT", "Pred"]
    # ONE camera object drives every panel — that is what "synchronised" means here
    assert len(re.findall(r"const CAM = \{", doc)) == 1
    assert "--cols: 2" in doc

    # labels come from filenames and scene names: one must not be able to close the JSON block
    hostile = standalone_html([{"label": "a</script><b>", "points": rng.normal(size=(3, 3)),
                                "colors": rng.integers(0, 255, (3, 3), dtype=np.uint8)}])
    block = hostile.split("type='application/json'>", 1)[1].split("</script>", 1)[0]
    assert "</script>" not in block, "a label escaped out of the JSON block"
    assert json.loads(block.replace("<\\/", "</"))["panels"][0]["label"] == "a</script><b>"

    iframe = viewer_iframe([{"label": "only", "points": rng.normal(size=(5, 3)),
                             "colors": rng.integers(0, 255, (5, 3), dtype=np.uint8)}])
    assert iframe.startswith("<iframe srcdoc=") and iframe.rstrip().endswith("</iframe>")
    body = iframe.split('srcdoc="', 1)[1].split('"', 1)[0]
    assert "<script" not in body and "&lt;script" in body, "srcdoc content is not escaped"

    html = dual_view_html(predictions, conf_thres=50.0)
    assert "iframe" in html and "srcdoc" in html
    assert "Ground truth" in html and "Prediction" in html

    # no checkpoint → a single RGB panel; no GT → RGB on the left, still two panels
    no_seg = {k: v for k, v in predictions.items() if k not in ("seg_colors", "gt_colors")}
    assert "Reconstruction" in dual_view_html(no_seg)
    no_gt = {k: v for k, v in predictions.items() if k != "gt_colors"}
    html_no_gt = dual_view_html(no_gt)
    assert "no GT for these frames" in html_no_gt and "Prediction" in html_no_gt

    assert "nothing yet" in message_html("nothing yet")
    print("✓ one camera, one canvas per panel, escaping, graceful fallbacks\n")


def test_ply_round_trips_into_the_page():
    """
    `scripts/view_ply.py` end to end, minus the GPU: an instance-coloured .ply (the shape
    `--dump_ply` writes) must come back out of the page's payload as the same points and the
    same colours.
    """
    print("=== Testing the .ply → HTML path ===")
    sys.path.insert(0, str(ROOT / "scripts"))
    import tempfile

    from view_ply import load_ply

    xyz = rng.normal(size=(50, 3)) * 2.0
    rgb = rng.integers(0, 255, size=(50, 3), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "eval3d_scene0000_00.ply"
        with open(path, "w") as f:
            f.write("ply\nformat ascii 1.0\n"
                    f"element vertex {len(xyz)}\n"
                    "property float x\nproperty float y\nproperty float z\n"
                    "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                    "end_header\n")
            for p, c in zip(xyz, rgb):
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}\n")

        v, c = load_ply(path)
        assert v.shape == (50, 3) and c.shape == (50, 3) and c.dtype == np.uint8
        assert np.allclose(v, xyz, atol=1e-4) and np.array_equal(c, rgb)

        doc = standalone_html([{"label": path.stem, "points": v, "colors": c}])
        payload = json.loads(re.search(r"type='application/json'>(.*?)</script>",
                                       doc, re.S).group(1))
        panel = payload["panels"][0]
        q = np.frombuffer(base64.b64decode(panel["positions"]), np.uint16).reshape(-1, 3)
        back = q.astype(np.float64) * np.array(panel["scale"]) + np.array(panel["offset"])
        assert np.allclose(back, xyz, atol=(xyz.max(0) - xyz.min(0)).max() / 65535 + 1e-3)
        got = np.frombuffer(base64.b64decode(panel["colors"]), np.uint8).reshape(-1, 3)
        assert np.array_equal(got, rgb)
    print("✓ points and colours survive .ply → page unchanged\n")


if __name__ == "__main__":
    test_matches_the_glb_path()
    test_panels_share_geometry_and_index()
    test_payload_round_trip_and_sharing()
    test_subsample_keeps_panels_aligned()
    test_page_structure()
    test_ply_round_trips_into_the_page()
    print("✅ All dual-view 3D tests passed")
