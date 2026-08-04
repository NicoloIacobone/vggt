#!/usr/bin/env python3
"""
Side-by-side, camera-synchronised 3D point-cloud views.

Two uses, one renderer:

  * `demos/demo_gradio.py` — GT | prediction of the SAME reconstructed cloud, so the only
    difference between the panels is the colouring (docs/MASKDINO.md §9.7);
  * `scripts/view_ply.py` — a standalone HTML file for the 3D ruler's `--dump_ply` output, so a
    `.ply` can be looked at without MeshLab and without downloading a viewer.

Why a hand-written WebGL viewer rather than two `gr.Model3D` components: Gradio 5 renders
Model3D through Babylon.js inside a compiled Svelte component, and neither its camera nor its
scene is reachable from outside — two of them cannot be kept in sync. Here **there is only one
camera**; each panel is a viewport onto it, so synchronisation is structural rather than
something that has to be maintained. All controls (orbit, pan, zoom, point size, reset) act on
that single camera and therefore on every panel at once.

The page is emitted as a self-contained document and embedded via an `<iframe srcdoc=...>`:
scripts inserted through `gr.HTML` never execute (the browser does not run `<script>` added via
innerHTML), while an iframe's srcdoc is a real document and does.

Positions are quantised to uint16 inside the cloud's own bounding box before being base64'd
into the page — 6 bytes/point instead of 12, which matters because the whole payload is re-sent
on every control change. 1/65536 of the scene extent is far below the point size on screen.
"""

import base64
import html
import json
from typing import Dict, List, Optional, Sequence

import numpy as np

# Per-panel cap. The payload travels inside the page, so this is a UX budget as much as a
# rendering one: ~200k points ≈ 1.2 MB of positions + 0.6 MB per colour array.
DEFAULT_MAX_POINTS = 200_000
# The Gradio panel is rebuilt and re-sent on every control change, so it pays that cost over and
# over — a smaller cap there keeps the sliders responsive and is still visually dense.
GRADIO_MAX_POINTS = 120_000


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def selected_frame(filter_by_frames) -> Optional[int]:
    """'3: frame_0007.png' → 3; 'All' / unparseable → None (the GLB path's own rule)."""
    if filter_by_frames in (None, "all", "All"):
        return None
    try:
        return int(str(filter_by_frames).split(":")[0])
    except (ValueError, IndexError):
        return None


def filtered_cloud(predictions: Dict, conf_thres: float = 50.0, filter_by_frames="All",
                   mask_black_bg: bool = False, mask_white_bg: bool = False,
                   prediction_mode: str = "Depthmap and Camera Branch"):
    """
    The same points the GLB tab shows: (xyz [N, 3] float64 aligned to the first camera,
    keep [S*H*W] bool, frame index or None).

    Mirrors `visual_util.predictions_to_glb`'s filtering — same branch selection, same
    percentile confidence threshold, same first-camera alignment — and
    `tests/test_dualview3d.py` asserts vertex-for-vertex equality with it so the two cannot
    drift apart. Two deliberate differences, both because the panels must stay comparable:

      * the black/white-background masks are computed from the **image** colours, never from
        whichever instance colouring a panel happens to use, so every panel keeps the same
        points and the only difference between them is colour;
      * `mask_sky` is ignored (it downloads an ONNX segmenter and is meaningless indoors).
    """
    if "Pointmap" in prediction_mode and "world_points" in predictions:
        points = predictions["world_points"]
        conf = predictions.get("world_points_conf", np.ones_like(points[..., 0]))
    else:
        points = predictions["world_points_from_depth"]
        conf = predictions.get("depth_conf", np.ones_like(points[..., 0]))

    images = predictions["images"]
    if images.ndim == 4 and images.shape[1] == 3:                     # NCHW → NHWC
        images = np.transpose(images, (0, 2, 3, 1))
    cameras = predictions["extrinsic"]

    fi = selected_frame(filter_by_frames)
    if fi is not None and fi < len(points):
        points, conf, images, cameras = (points[fi][None], conf[fi][None],
                                         images[fi][None], cameras[fi][None])

    xyz = points.reshape(-1, 3)
    rgb = (images.reshape(-1, 3) * 255).astype(np.uint8)
    flat_conf = conf.reshape(-1)
    threshold = 0.0 if not conf_thres else np.percentile(flat_conf, conf_thres)
    keep = (flat_conf >= threshold) & (flat_conf > 1e-5)
    if mask_black_bg:
        keep &= rgb.sum(axis=1) >= 16
    if mask_white_bg:
        keep &= ~((rgb[:, 0] > 240) & (rgb[:, 1] > 240) & (rgb[:, 2] > 240))

    extrinsic = np.zeros((4, 4))
    extrinsic[:3, :4] = cameras[0]
    extrinsic[3, 3] = 1
    transform = np.linalg.inv(extrinsic) @ _alignment_matrix()
    xyz = xyz[keep] @ transform[:3, :3].T + transform[:3, 3]
    return xyz, keep, fi


def _alignment_matrix() -> np.ndarray:
    """`opengl_conversion @ rot180y`, the constant half of the GLB path's scene alignment."""
    from scipy.spatial.transform import Rotation

    from visual_util import get_opengl_conversion_matrix

    align = np.eye(4)
    align[:3, :3] = Rotation.from_euler("y", 180, degrees=True).as_matrix()
    return get_opengl_conversion_matrix() @ align


def panel_colors(colors: np.ndarray, keep: np.ndarray, frame_idx: Optional[int]) -> np.ndarray:
    """Per-pixel colours [S, H, W, 3] → [N, 3] for the kept points of `filtered_cloud`."""
    colors = np.asarray(colors)
    if colors.ndim == 4 and colors.shape[1] == 3:                     # NCHW → NHWC
        colors = np.transpose(colors, (0, 2, 3, 1))
    if frame_idx is not None and frame_idx < len(colors):
        colors = colors[frame_idx][None]
    return colors.reshape(-1, 3).astype(np.uint8)[keep]


def quantize_positions(xyz: np.ndarray):
    """
    [N, 3] float → (uint16 [N, 3], scale [3], offset [3]) with `q * scale + offset ≈ xyz`.

    A degenerate axis (all points equal) gets scale 0 and keeps its value in the offset.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    q = np.clip(np.round((xyz - lo) / span * 65535.0), 0, 65535).astype(np.uint16)
    scale = np.where(hi > lo, span / 65535.0, 0.0)
    return q, scale.astype(np.float64), lo.astype(np.float64)


def subsample_index(n: int, max_points: int) -> np.ndarray:
    """
    Deterministic even stride. Deterministic matters: every panel must keep the SAME points,
    otherwise "the difference between the panels is only the colour" stops being true.
    """
    if max_points is None or n <= max_points:
        return np.arange(n)
    return np.linspace(0, n - 1, max_points).astype(np.int64)


def build_payload(panels: Sequence[Dict], max_points: Optional[int] = DEFAULT_MAX_POINTS,
                  point_size: float = 2.0) -> Dict:
    """
    Panels → the JSON the page consumes.

    Each panel is `{"label": str, "points": [N, 3] float | None, "colors": [N, 3] uint8,
    "note": str}`. `points=None` means "share the previous panel's geometry", which is the
    normal case for GT-vs-prediction and halves the payload.
    """
    out, shared = [], None
    for panel in panels:
        pts = panel.get("points")
        colors = np.asarray(panel["colors"], dtype=np.uint8).reshape(-1, 3)
        if pts is None:
            if shared is None:
                raise ValueError("the first panel must carry its own points")
            idx, entry = shared, {"positions": None}
        else:
            pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
            if len(pts) != len(colors):
                raise ValueError(f"panel '{panel.get('label')}': {len(pts)} points vs "
                                 f"{len(colors)} colours")
            idx = subsample_index(len(pts), max_points)
            q, scale, offset = quantize_positions(pts[idx])
            entry = {"positions": _b64(q), "scale": scale.tolist(), "offset": offset.tolist()}
            shared = idx
        out.append({**entry, "label": str(panel.get("label", "")),
                    "note": str(panel.get("note", "")), "count": int(len(idx)),
                    "colors": _b64(colors[idx])})
    return {"pointSize": float(point_size), "panels": out}


# The viewer. One camera; every panel is a viewport onto it. Kept dependency-free on purpose:
# no CDN, so it works on a cluster node behind a proxy and inside a saved single-file HTML.
_VIEWER_JS = r"""
const DATA = JSON.parse(document.getElementById("dv3d-data").textContent);

function b64ToBytes(b64) {
  const bin = atob(b64), out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

const VS = `
attribute vec3 aPos;
attribute vec3 aCol;
uniform mat4 uMVP;
uniform vec3 uScale;
uniform vec3 uOffset;
uniform float uSize;
varying vec3 vCol;
void main() {
  vec3 p = aPos * uScale + uOffset;
  gl_Position = uMVP * vec4(p, 1.0);
  gl_PointSize = uSize;
  vCol = aCol;
}`;

const FS = `
precision mediump float;
varying vec3 vCol;
void main() {
  vec2 d = gl_PointCoord - vec2(0.5);
  if (dot(d, d) > 0.25) discard;      // round points, not squares
  gl_FragColor = vec4(vCol, 1.0);
}`;

// --- minimal matrix helpers (column-major, like WebGL wants) ---------------------------
function perspective(fovy, aspect, near, far) {
  const f = 1.0 / Math.tan(fovy / 2), nf = 1 / (near - far);
  return [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0];
}
function lookAt(eye, center, up) {
  let z = [eye[0] - center[0], eye[1] - center[1], eye[2] - center[2]];
  let zl = Math.hypot(z[0], z[1], z[2]) || 1; z = z.map(v => v / zl);
  let x = [up[1] * z[2] - up[2] * z[1], up[2] * z[0] - up[0] * z[2], up[0] * z[1] - up[1] * z[0]];
  let xl = Math.hypot(x[0], x[1], x[2]);
  x = xl > 1e-8 ? x.map(v => v / xl) : [1, 0, 0];
  const y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]];
  return [x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0,
          -(x[0] * eye[0] + x[1] * eye[1] + x[2] * eye[2]),
          -(y[0] * eye[0] + y[1] * eye[1] + y[2] * eye[2]),
          -(z[0] * eye[0] + z[1] * eye[1] + z[2] * eye[2]), 1];
}
function mul(a, b) {                                   // a * b, both column-major 4x4
  const o = new Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let s = 0;
    for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
    o[c * 4 + r] = s;
  }
  return o;
}

// --- the single shared camera ----------------------------------------------------------
const CAM = { theta: 0, phi: 0.25, dist: 3, target: [0, 0, 0], fov: 50 * Math.PI / 180 };
let HOME = null;

function eyePosition() {
  const cp = Math.cos(CAM.phi), sp = Math.sin(CAM.phi);
  return [CAM.target[0] + CAM.dist * cp * Math.sin(CAM.theta),
          CAM.target[1] + CAM.dist * sp,
          CAM.target[2] + CAM.dist * cp * Math.cos(CAM.theta)];
}

const panels = [];
let sharedGeom = null, dirty = true;

function initPanel(canvas, spec) {
  const gl = canvas.getContext("webgl", { antialias: true, alpha: false })
          || canvas.getContext("experimental-webgl");
  if (!gl) { canvas.parentNode.querySelector(".dv3d-err").textContent =
      "WebGL is not available in this browser."; return null; }

  const compile = (type, src) => {
    const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  };
  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl.VERTEX_SHADER, VS));
  gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FS));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
  gl.useProgram(prog);

  let geom = sharedGeom;
  if (spec.positions !== null) {
    geom = { data: b64ToBytes(spec.positions), scale: spec.scale, offset: spec.offset };
    sharedGeom = geom;
  }
  const posBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
  gl.bufferData(gl.ARRAY_BUFFER, geom.data, gl.STATIC_DRAW);
  const colBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, colBuf);
  gl.bufferData(gl.ARRAY_BUFFER, b64ToBytes(spec.colors), gl.STATIC_DRAW);

  gl.enable(gl.DEPTH_TEST);
  gl.clearColor(0.07, 0.075, 0.09, 1.0);
  return { gl, prog, posBuf, colBuf, canvas, count: spec.count, geom,
           loc: { pos: gl.getAttribLocation(prog, "aPos"),
                  col: gl.getAttribLocation(prog, "aCol"),
                  mvp: gl.getUniformLocation(prog, "uMVP"),
                  scale: gl.getUniformLocation(prog, "uScale"),
                  offset: gl.getUniformLocation(prog, "uOffset"),
                  size: gl.getUniformLocation(prog, "uSize") } };
}

function draw(p, pointSize) {
  const { gl, canvas } = p;
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(canvas.clientWidth * dpr));
  const h = Math.max(1, Math.round(canvas.clientHeight * dpr));
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  gl.viewport(0, 0, w, h);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.useProgram(p.prog);

  const eye = eyePosition();
  const near = Math.max(1e-4, CAM.dist * 0.002), far = CAM.dist * 20 + 10;
  const mvp = mul(perspective(CAM.fov, w / h, near, far), lookAt(eye, CAM.target, [0, 1, 0]));
  gl.uniformMatrix4fv(p.loc.mvp, false, new Float32Array(mvp));
  gl.uniform3fv(p.loc.scale, new Float32Array(p.geom.scale));
  gl.uniform3fv(p.loc.offset, new Float32Array(p.geom.offset));
  gl.uniform1f(p.loc.size, pointSize * dpr);

  gl.bindBuffer(gl.ARRAY_BUFFER, p.posBuf);
  gl.enableVertexAttribArray(p.loc.pos);
  gl.vertexAttribPointer(p.loc.pos, 3, gl.UNSIGNED_SHORT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, p.colBuf);
  gl.enableVertexAttribArray(p.loc.col);
  gl.vertexAttribPointer(p.loc.col, 3, gl.UNSIGNED_BYTE, true, 0, 0);
  gl.drawArrays(gl.POINTS, 0, p.count);
}

let pointSize = DATA.pointSize;
function frame() {
  // A panel inside a hidden Gradio tab has clientWidth 0 and never receives a resize event, so
  // "redraw when the camera moved" alone would leave it blank until the first drag. Cheap size
  // poll instead: becoming visible is a size change, and idle costs nothing.
  panels.forEach(p => {
    if (!p) return;
    const w = p.canvas.clientWidth, h = p.canvas.clientHeight;
    if (w !== p.lastW || h !== p.lastH) { p.lastW = w; p.lastH = h; dirty = true; }
  });
  if (dirty) { panels.forEach(p => p && p.canvas.clientWidth && draw(p, pointSize)); dirty = false; }
  requestAnimationFrame(frame);
}

// --- interaction: every handler edits the ONE camera, so all panels move together -------
function attachControls(canvas) {
  let dragging = null, lastX = 0, lastY = 0;
  canvas.addEventListener("contextmenu", e => e.preventDefault());
  canvas.addEventListener("pointerdown", e => {
    dragging = (e.button === 2 || e.shiftKey || e.ctrlKey) ? "pan" : "orbit";
    lastX = e.clientX; lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  });
  const endDrag = e => {
    dragging = null;
    if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
  };
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);
  canvas.addEventListener("pointermove", e => {
    if (!dragging) return;
    const dx = e.clientX - lastX, dy = e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    if (dragging === "orbit") {
      CAM.theta -= dx * 0.008;
      CAM.phi = Math.max(-1.55, Math.min(1.55, CAM.phi + dy * 0.008));
    } else {
      // pan in the camera's own plane, scaled by distance so it feels the same at any zoom
      const eye = eyePosition();
      let f = [CAM.target[0] - eye[0], CAM.target[1] - eye[1], CAM.target[2] - eye[2]];
      const fl = Math.hypot(f[0], f[1], f[2]) || 1; f = f.map(v => v / fl);
      let r = [-f[2], 0, f[0]];                             // f x (0,1,0) = camera right
      const rl = Math.hypot(r[0], r[1], r[2]) || 1; r = r.map(v => v / rl);
      const u = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]];
      const k = CAM.dist * 0.0018;
      for (let i = 0; i < 3; i++) CAM.target[i] += (-dx * r[i] + dy * u[i]) * k;
    }
    dirty = true;
  });
  canvas.addEventListener("wheel", e => {
    e.preventDefault();
    CAM.dist = Math.max(1e-3, CAM.dist * Math.exp(e.deltaY * 0.0012));
    dirty = true;
  }, { passive: false });
}

// --- boot ------------------------------------------------------------------------------
const grid = document.getElementById("dv3d-grid");
DATA.panels.forEach((spec, i) => {
  const wrap = document.createElement("div");
  wrap.className = "dv3d-panel";
  wrap.innerHTML = '<div class="dv3d-head"><span class="dv3d-label"></span>'
                 + '<span class="dv3d-note"></span></div>'
                 + '<canvas></canvas><div class="dv3d-err"></div>';
  wrap.querySelector(".dv3d-label").textContent = spec.label;
  wrap.querySelector(".dv3d-note").textContent = spec.note;
  grid.appendChild(wrap);
  const canvas = wrap.querySelector("canvas");
  let p = null;
  try { p = initPanel(canvas, spec); }
  catch (err) { wrap.querySelector(".dv3d-err").textContent = "WebGL error: " + err.message; }
  panels.push(p);
  if (p) attachControls(canvas);
});

// Frame the cloud: the quantisation box IS the bounding box, so no scan over points needed.
(function fit() {
  const g = sharedGeom;
  if (!g) return;
  const ext = [g.scale[0] * 65535, g.scale[1] * 65535, g.scale[2] * 65535];
  CAM.target = [g.offset[0] + ext[0] / 2, g.offset[1] + ext[1] / 2, g.offset[2] + ext[2] / 2];
  const radius = 0.5 * Math.hypot(ext[0], ext[1], ext[2]) || 1;
  CAM.dist = radius / Math.tan(CAM.fov / 2) * 0.9;
  HOME = JSON.parse(JSON.stringify(CAM));
})();

document.getElementById("dv3d-size").addEventListener("input", e => {
  pointSize = parseFloat(e.target.value); dirty = true;
});
document.getElementById("dv3d-reset").addEventListener("click", () => {
  if (HOME) { CAM.theta = HOME.theta; CAM.phi = HOME.phi; CAM.dist = HOME.dist;
              CAM.target = HOME.target.slice(); dirty = true; }
});
window.addEventListener("resize", () => { dirty = true; });
frame();
"""

_VIEWER_CSS = """
* { box-sizing: border-box; }
body { margin: 0; background: #111318; color: #d8dbe2;
       font: 13px/1.4 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
#dv3d-grid { display: grid; grid-template-columns: repeat(var(--cols), minmax(0, 1fr));
             gap: 8px; padding: 8px; }
@media (max-width: 720px) { #dv3d-grid { grid-template-columns: 1fr; } }
.dv3d-panel { display: flex; flex-direction: column; min-width: 0;
              border: 1px solid #2a2f3a; border-radius: 8px; overflow: hidden; }
.dv3d-head { display: flex; justify-content: space-between; align-items: baseline; gap: 8px;
             padding: 6px 10px; background: #171a21; border-bottom: 1px solid #2a2f3a; }
.dv3d-label { font-weight: 600; }
.dv3d-note { color: #8b93a4; font-size: 11px; text-align: right; }
canvas { width: 100%; height: var(--canvas-h); display: block; background: #111318;
         touch-action: none; cursor: grab; }
canvas:active { cursor: grabbing; }
.dv3d-err { color: #ff8f8f; padding: 0 10px; font-size: 12px; }
.dv3d-bar { display: flex; align-items: center; gap: 14px; flex-wrap: wrap;
            padding: 6px 12px 10px; color: #8b93a4; font-size: 12px; }
.dv3d-bar input[type=range] { width: 120px; vertical-align: middle; }
.dv3d-bar button { background: #232833; color: #d8dbe2; border: 1px solid #39404f;
                   border-radius: 6px; padding: 3px 10px; cursor: pointer; font: inherit; }
.dv3d-bar button:hover { background: #2c323f; }
"""


def standalone_html(panels: Sequence[Dict], title: str = "3D view",
                    max_points: Optional[int] = DEFAULT_MAX_POINTS, point_size: float = 2.0,
                    canvas_height: int = 460) -> str:
    """A complete, dependency-free HTML document showing `panels` under one shared camera."""
    payload = build_payload(panels, max_points=max_points, point_size=point_size)
    total = sum(p["count"] for p in payload["panels"])
    # A panel label containing "</script>" would end the JSON block early; base64 never does,
    # but labels come from filenames and scene names, so close the hole rather than trust them.
    payload_json = json.dumps(payload).replace("</", "<\\/")
    hint = ("drag = orbit · right-drag or shift-drag = pan · wheel = zoom · "
            "one camera drives every panel")
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title><style>{_VIEWER_CSS}</style>"
        f"<style>:root {{ --cols: {max(1, len(payload['panels']))}; "
        f"--canvas-h: {int(canvas_height)}px; }}</style></head><body>"
        "<div id='dv3d-grid'></div>"
        "<div class='dv3d-bar'>"
        f"<span>{html.escape(hint)}</span>"
        "<span>point size <input id='dv3d-size' type='range' min='1' max='8' step='0.5' "
        f"value='{float(point_size)}'></span>"
        "<button id='dv3d-reset'>Reset view</button>"
        f"<span>{total:,} points shown</span>"
        "</div>"
        f"<script id='dv3d-data' type='application/json'>{payload_json}</script>"
        f"<script>{_VIEWER_JS}</script></body></html>"
    )


def viewer_iframe(panels: Sequence[Dict], height: int = 560, **kwargs) -> str:
    """
    The same document wrapped for `gr.HTML`.

    srcdoc rather than innerHTML: a `<script>` inserted as HTML never runs, an iframe document's
    does. The payload is base64 and the document is HTML-escaped, so nothing can break out of
    the attribute.
    """
    doc = standalone_html(panels, **kwargs)
    return (f"<iframe srcdoc=\"{html.escape(doc, quote=True)}\" "
            f"style='width:100%;height:{int(height)}px;border:0;border-radius:8px;' "
            f"sandbox='allow-scripts'></iframe>")


def dual_view_html(predictions: Dict, height: int = 560,
                   max_points: Optional[int] = GRADIO_MAX_POINTS, **filter_kwargs) -> str:
    """
    GT | prediction of the same reconstructed cloud, under one camera.

    Falls back gracefully: with no GT (uploaded images) the left panel is the RGB cloud; with no
    segmentation checkpoint there is a single RGB panel. The two instance panels use **different
    identity spaces** — GT global instance ids vs query indices — so their colours are not meant
    to agree with each other, only with themselves across views (docs/MASKDINO.md §6.4).
    """
    xyz, keep, fi = filtered_cloud(predictions, **filter_kwargs)
    rgb = panel_colors(predictions["images"], keep, fi)
    seg = predictions.get("seg_colors")
    gt = predictions.get("gt_colors")

    if seg is None:
        panels = [{"label": "Reconstruction", "note": "no segmentation checkpoint loaded",
                   "points": xyz, "colors": rgb}]
    else:
        left = ({"label": "Ground truth", "note": "colour = GT instance id",
                 "points": xyz, "colors": panel_colors(gt, keep, fi)} if gt is not None else
                {"label": "RGB", "note": "no GT for these frames", "points": xyz, "colors": rgb})
        panels = [left, {"label": "Prediction", "note": "colour = query id",
                         "points": None, "colors": panel_colors(seg, keep, fi)}]
    return viewer_iframe(panels, height=height, max_points=max_points,
                         canvas_height=max(160, height - 90))


def message_html(text: str, height: int = 120) -> str:
    """Placeholder for 'there is nothing to show yet', styled like the viewer."""
    return (f"<div style='padding:14px;border:1px solid #2a2f3a;border-radius:8px;"
            f"background:#171a21;color:#8b93a4;min-height:{int(height)}px'>"
            f"{html.escape(text)}</div>")
