#!/usr/bin/env python3
"""Deliverable 1 build step: inspect what facebookresearch/Replica-Dataset and kxic/vMAP
actually ship (no assumptions), then repack the 8 target scenes into the two tars.

Run inside slurm/fetch_replica.sh, node-local. Writes a REPORT.md next to the tars documenting
exactly what was found -- this is the "determine what is actually available, report it" step
the task called for, done empirically against the downloaded bytes, not guessed from docs.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

LICENSE_NOTICE = (
    "Replica dataset: CC-BY-NC-4.0 (facebookresearch/Replica-Dataset LICENSE). "
    "vMAP-rendered RGB-D/pose sequences: derived from Replica under the same terms "
    "(kxic/vMAP, github.com/kxhit/vMAP). Non-commercial research use only; do not redistribute."
)


def parse_ply_header(path):
    """Read only the ASCII header (works for binary_little_endian bodies too) and report
    element/property names -- this is how we verify per-vertex/per-face instance ids exist,
    rather than assuming the schema."""
    props = {}
    order = []
    cur = None
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                break
            s = line.decode("ascii", errors="replace").strip()
            if s == "end_header":
                break
            if s.startswith("element"):
                _, name, count = s.split()
                cur = name
                order.append(cur)
                props[cur] = {"count": int(count), "properties": []}
            elif s.startswith("property") and cur is not None:
                props[cur]["properties"].append(s)
    return {"elements": order, "detail": props}


def find_scene_dir(root: Path, scene: str):
    """vmap.zip's internal layout is not documented anywhere we could read without downloading
    it, so search for it instead of assuming a fixed prefix."""
    candidates = [p for p in root.rglob(scene) if p.is_dir()]
    # prefer the shallowest match
    candidates.sort(key=lambda p: len(p.parts))
    return candidates[0] if candidates else None


def find_first(root: Path, patterns):
    for pat in patterns:
        hits = sorted(root.rglob(pat))
        if hits:
            return hits[0]
    return None


def load_intrinsics(scene_dir: Path, report):
    """Look for an explicit camera-params file; vMAP's data_generation config only fixes
    width=1200,height=680 and leaves fx/fy/cx/cy out of that particular yaml, so we search the
    actual downloaded tree for a params/intrinsics file rather than hardcoding a guess."""
    cand = find_first(scene_dir, ["cam_params.json", "*intrinsic*.json", "*intrinsic*.txt",
                                   "camera*.json", "*.yaml", "*.yml"])
    if cand is not None:
        report["intrinsics_source"] = str(cand)
        try:
            if cand.suffix == ".json":
                data = json.loads(cand.read_text())
                report["intrinsics_raw"] = data
                return data, cand
            else:
                report["intrinsics_raw_text"] = cand.read_text()[:2000]
                return None, cand
        except Exception as e:
            report["intrinsics_parse_error"] = str(e)
    return None, None


def intrinsics_matrix_from(data):
    """Best-effort extraction of fx,fy,cx,cy,w,h from whatever JSON schema was actually found."""
    if data is None:
        return None
    flat = {}
    def walk(d):
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, (int, float)):
                    flat[k.lower()] = v
                else:
                    walk(v)
        elif isinstance(d, list):
            for v in d:
                walk(v)
    walk(data)
    keys = {"fx", "fy", "cx", "cy", "w", "h", "width", "height"}
    found = {k: v for k, v in flat.items() if k in keys}
    if {"fx", "fy", "cx", "cy"} <= found.keys():
        return found
    return None


# Habitat/vMAP's well-known fixed Replica render intrinsics (traj 00, the iMAP trajectory),
# used ONLY as a fallback and clearly labelled as such in the report if no params file is found
# in the actual downloaded tree.
FALLBACK_INTRINSICS = {"fx": 600.0, "fy": 600.0, "cx": 599.5, "cy": 339.5, "w": 1200, "h": 680}


def uniform_sample(n_total, n_sample):
    if n_total <= n_sample:
        return list(range(n_total))
    step = (n_total - 1) / (n_sample - 1)
    return sorted(set(round(i * step) for i in range(n_sample)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replica_orig", required=True)
    ap.add_argument("--vmap_extracted", required=True)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--n_frames", type=int, default=50)
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    replica_orig = Path(args.replica_orig)
    vmap_extracted = Path(args.vmap_extracted)
    work = Path(args.work_dir)
    out_dir = Path(args.out_dir)

    scans3d = work / "scans3d"
    scans25k = work / "scans25k"
    scans3d.mkdir(exist_ok=True)
    scans25k.mkdir(exist_ok=True)

    report = {"license": LICENSE_NOTICE, "scenes": {}}

    for scene in args.scenes:
        sr = {}
        report["scenes"][scene] = sr

        # ---- scans3d: GT mesh + instance/semantic annotation ----
        src_scene = replica_orig / scene
        mesh_sem = src_scene / "habitat" / "mesh_semantic.ply"
        info_sem = src_scene / "habitat" / "info_semantic.json"
        preseg_json = src_scene / "preseg.json"
        preseg_bin = src_scene / "preseg.bin"

        dst = scans3d / scene
        dst.mkdir(parents=True, exist_ok=True)

        if mesh_sem.exists():
            hdr = parse_ply_header(mesh_sem)
            sr["mesh_semantic_ply"] = {"present": True, "header": hdr}
            shutil.copy2(mesh_sem, dst / "mesh_semantic.ply")
        else:
            sr["mesh_semantic_ply"] = {"present": False}
            print(f"[WARN] {scene}: habitat/mesh_semantic.ply NOT FOUND", file=sys.stderr)

        if info_sem.exists():
            info = json.loads(info_sem.read_text())
            n_obj = len(info.get("objects", info.get("id_to_label", [])))
            sr["info_semantic_json"] = {"present": True, "n_entries": n_obj,
                                         "top_level_keys": list(info.keys())}
            shutil.copy2(info_sem, dst / "info_semantic.json")
        else:
            sr["info_semantic_json"] = {"present": False}
            print(f"[WARN] {scene}: habitat/info_semantic.json NOT FOUND", file=sys.stderr)

        # plain colored mesh, kept alongside as a non-semantic reference if present
        plain_mesh = src_scene / "mesh.ply"
        if plain_mesh.exists():
            shutil.copy2(plain_mesh, dst / "mesh.ply")
            sr["mesh_ply_plain"] = True

        if preseg_json.exists() and preseg_bin.exists():
            shutil.copy2(preseg_json, dst / "preseg.json")
            shutil.copy2(preseg_bin, dst / "preseg.bin")
            sr["oversegmentation"] = "preseg.json/.bin (Replica's own planar/non-planar presegmentation)"
        else:
            sr["oversegmentation"] = "NOT AVAILABLE in this scene's release tree"

        # ---- scans25k: frames ----
        scene_dir = find_scene_dir(vmap_extracted, scene)
        if scene_dir is None:
            sr["vmap_frames"] = {"present": False}
            print(f"[WARN] {scene}: no matching dir under vmap_extracted", file=sys.stderr)
            continue

        # traj 00 == the iMAP trajectory (per kxic/vMAP's own dataset description)
        traj_dir = None
        for cand in [scene_dir / "imap" / "00", scene_dir / "00", scene_dir]:
            if cand.exists():
                traj_dir = cand
                break
        sr["vmap_scene_dir"] = str(scene_dir)
        sr["vmap_traj_dir"] = str(traj_dir)
        sr["vmap_traj_dir_listing"] = sorted(p.name for p in traj_dir.iterdir())[:50] if traj_dir else []

        # vMAP's own dataset.py indexes frames by number, not by directory globbing+sorting
        # ("rgb_<idx>.png"/"depth_<idx>.png", 1 pose line per idx in traj_w_c.txt) -- globbing
        # and lexicographically sorting would put rgb_10.png before rgb_2.png and scramble
        # trajectory order, so frames are located by constructing the exact expected filename
        # for each index instead.
        rgb_dir = traj_dir / "rgb"
        depth_dir = traj_dir / "depth"
        traj_file = find_first(traj_dir, ["traj_w_c.txt", "traj.txt", "*traj*.txt"])

        sr["vmap_frames"] = {
            "rgb_dir_present": rgb_dir.exists(), "depth_dir_present": depth_dir.exists(),
            "traj_file": str(traj_file) if traj_file else None,
        }
        if not rgb_dir.exists() or traj_file is None:
            print(f"[WARN] {scene}: incomplete vMAP frame data "
                  f"(rgb_dir={rgb_dir.exists()}, depth_dir={depth_dir.exists()}, traj={traj_file})",
                  file=sys.stderr)
            continue

        def rgb_path(i):
            for ext in (".png", ".jpg"):
                p = rgb_dir / f"rgb_{i}{ext}"
                if p.exists():
                    return p
            return None

        def depth_path(i):
            p = depth_dir / f"depth_{i}.png"
            return p if p.exists() else None

        poses = [l.strip() for l in Path(traj_file).read_text().splitlines() if l.strip()]
        # confirm the indexed convention actually matches what's on disk before trusting it
        # for every index, rather than assuming and silently writing an empty scans25k/.
        n_probe = min(len(poses), 20)
        n_rgb_hits = sum(1 for i in range(n_probe) if rgb_path(i) is not None)
        sr["vmap_frames"]["rgb_<idx>_convention_hits"] = f"{n_rgb_hits}/{n_probe}"
        if n_rgb_hits == 0:
            print(f"[WARN] {scene}: rgb_<idx>.{{png,jpg}} convention did not match any of the "
                  f"first {n_probe} indices -- vmap's on-disk layout differs from its own "
                  f"dataset.py, skipping frames for this scene", file=sys.stderr)
            continue

        n_total = len(poses)
        idx = uniform_sample(n_total, args.n_frames)
        idx = [i for i in idx if rgb_path(i) is not None]

        intr_data, intr_src = load_intrinsics(scene_dir, sr)
        intr = intrinsics_matrix_from(intr_data)
        if intr is None:
            intr = dict(FALLBACK_INTRINSICS)
            sr["intrinsics_used"] = "FALLBACK (habitat default 1200x680, fx=fy=600, cx=599.5, cy=339.5) -- no params file found in the downloaded tree"
        else:
            sr["intrinsics_used"] = "parsed from " + str(intr_src)

        dst25 = scans25k / scene
        for sub in ("color", "pose", "depth", "intrinsic"):
            (dst25 / sub).mkdir(parents=True, exist_ok=True)

        fx = intr.get("fx", FALLBACK_INTRINSICS["fx"])
        fy = intr.get("fy", FALLBACK_INTRINSICS["fy"])
        cx = intr.get("cx", FALLBACK_INTRINSICS["cx"])
        cy = intr.get("cy", FALLBACK_INTRINSICS["cy"])
        (dst25 / "intrinsic" / "intrinsic_depth.txt").write_text(
            f"{fx} 0 {cx} 0\n0 {fy} {cy} 0\n0 0 1 0\n0 0 0 1\n"
        )

        manifest = {"scene": scene, "n_total_frames": n_total, "n_sampled": len(idx),
                    "sampled_indices": idx, "traj_source": str(traj_file),
                    "rgb_source_dir": str(rgb_dir)}
        for i in idx:
            stem = f"{i:06d}"
            rp = rgb_path(i)
            shutil.copy2(rp, dst25 / "color" / f"{stem}{rp.suffix}")
            dp = depth_path(i)
            if dp is not None:
                shutil.copy2(dp, dst25 / "depth" / f"{stem}.png")
            vals = poses[i].split()
            if len(vals) == 16:
                rows = [vals[j * 4:(j + 1) * 4] for j in range(4)]
                text = "\n".join(" ".join(r) for r in rows) + "\n"
            else:
                text = poses[i] + "\n"
            (dst25 / "pose" / f"{stem}.txt").write_text(text)

        (dst25 / "manifest.json").write_text(json.dumps(manifest, indent=2))
        sr["sampled"] = len(idx)
        print(f"[OK] {scene}: sampled {len(idx)}/{n_total} frames")

    (work / "REPORT.json").write_text(json.dumps(report, indent=2))

    # ---- pack tars ----
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, src in [("replica_3d_gt_8", scans3d), ("replica_frames_8", scans25k)]:
        tar_path = work / f"{name}.tar.zst"
        subprocess.run(
            f"tar -C {work} -cf - {src.name} | zstd -T4 -19 -o {tar_path}",
            shell=True, check=True,
        )
        shutil.move(str(tar_path), str(out_dir / f"{name}.tar.zst"))
        print(f"[OK] wrote {out_dir / (name + '.tar.zst')}")

    shutil.copy2(work / "REPORT.json", out_dir / "REPORT.json")
    (out_dir / "LICENSE.txt").write_text(LICENSE_NOTICE + "\n")
    print("=== REPORT summary ===")
    print(json.dumps(report, indent=2)[:4000])


if __name__ == "__main__":
    main()
