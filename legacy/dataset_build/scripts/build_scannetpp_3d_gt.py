"""ScanNet++ v2 3D benchmark GT for a scene list (docs/todo.md 6c).

Copies, per scene, the three files the 3D instance benchmark needs out of the upstream
release and into our own tree — the upstream tree belongs to another user and can vanish,
so nothing downstream may ever read it:

    scans3d/<scene>/mesh.ply            <- scans/mesh_aligned_0.05.ply  (the ALIGNED mesh)
    scans3d/<scene>/segments.json       <- verbatim
    scans3d/<scene>/segments_anno.json  <- verbatim

plus one shared `scans3d/_metadata/` with the class tables and the split file, because the
GT is unreadable without them.

The layout mirrors `scannet_3d_gt_val312.tar.zst` (docs/DATASET.md §2) so the evaluator's
loaders need no new file conventions — only the file NAMES differ, because ScanNet++ has no
`_vh_clean_2` / `.aggregation.json` equivalents.

Validation per scene (a scene that fails is left without `.complete`, so a re-run retries):
  - ply magic + `element vertex N` parses, and N == len(segIndices);
  - both jsons parse;
  - SEGMENT-ID CLOSURE: every segment id referenced by an object exists in segments.json —
    the same guard `download_3d_gt.py::validate_scene` applies to ScanNet's aggregation;
  - every kept object's label is in `top100_instance.txt` after the `instance_map_to` map;
  - at least one benchmark instance survives.

Resumable: a scene with a `.complete` marker is skipped.

Usage (from the vggt repo):
    myenv/bin/python legacy/dataset_build/scripts/build_scannetpp_3d_gt.py \
        --src_root /cluster/work/igp_psr/nedela/scannetpp_data \
        --out_root $TMPDIR/build/scans3d \
        --scene_list /cluster/work/igp_psr/nedela/scannetpp_data/splits/nvs_sem_val.txt
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scannetpp_common import (  # noqa: E402
    GT_FILES, METADATA_FILES, build_vertex_instances, drop_excluded,
    load_instance_classes, load_label_map, load_segments, ply_vertex_count, select_scenes,
)


def copy_metadata(src_root: Path, out_root: Path) -> None:
    dest = out_root / "_metadata"
    dest.mkdir(parents=True, exist_ok=True)
    for rel in METADATA_FILES:
        src = src_root / rel
        if not src.is_file():
            raise FileNotFoundError(f"metadata missing upstream: {src}")
        shutil.copyfile(src, dest / Path(rel).name)


def build_scene(src_root: Path, out_root: Path, scene: str,
                instance_classes: list[str], label_map: dict) -> dict:
    src = src_root / "data" / scene
    dst = out_root / scene
    dst.mkdir(parents=True, exist_ok=True)

    for name, rel in GT_FILES.items():
        s = src / rel
        if not s.is_file():
            raise FileNotFoundError(f"{scene}: missing {s}")
        tmp = dst / (name + ".part")
        shutil.copyfile(s, tmp)
        tmp.replace(dst / name)

    n_vertices = ply_vertex_count(dst / "mesh.ply")
    seg_indices = load_segments(dst / "segments.json")
    if len(seg_indices) != n_vertices:
        raise ValueError(f"{scene}: segIndices {len(seg_indices)} != ply vertices "
                         f"{n_vertices}")
    anno = json.loads((dst / "segments_anno.json").read_text())
    groups = anno["segGroups"]
    # Raises on closure violations; drops objects outside the benchmark class set.
    inst_ids, instances = build_vertex_instances(seg_indices, groups,
                                                 instance_classes, label_map)
    if not instances:
        raise ValueError(f"{scene}: no benchmark instances survived class filtering")

    stats = {
        "scene": scene,
        "n_vertices": int(n_vertices),
        "n_segments": int(len(set(seg_indices.tolist()))),
        "n_seg_groups": len(groups),
        "n_instances": len(instances),
        "n_labelled_vertices": int((inst_ids > 0).sum()),
        "classes": sorted({i["label"] for i in instances}),
    }
    (dst / "gt_stats.json").write_text(json.dumps(stats, indent=1))
    (dst / ".complete").touch()
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", required=True, help="the upstream scannetpp_data root")
    ap.add_argument("--out_root", required=True, help="the scans3d/ tree to write")
    ap.add_argument("--scene_list", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=49)
    ap.add_argument("--exclude_scenes", nargs="*", default=[],
                    help="scenes the upstream release ships broken (docs/DATASET.md §2.1). "
                         "Named explicitly, and removed from the tree if already built.")
    args = ap.parse_args()

    src_root, out_root = Path(args.src_root), Path(args.out_root)
    scenes = select_scenes(args.scene_list, args.start, args.end, args.exclude_scenes)
    gone = drop_excluded(out_root, args.exclude_scenes)
    if gone:
        print(f"[gt] removed excluded scene(s) from the tree: {gone}", flush=True)

    meta_dir = src_root / "metadata" / "semantic_benchmark"
    instance_classes = load_instance_classes(meta_dir)
    label_map = load_label_map(meta_dir)
    print(f"[gt] {len(instance_classes)} instance classes, "
          f"{len(label_map)} label-map rows", flush=True)
    copy_metadata(src_root, out_root)

    ok = skip = fail = 0
    failed: list[str] = []
    total_inst = 0
    for scene in scenes:
        if (out_root / scene / ".complete").exists():
            stats = json.loads((out_root / scene / "gt_stats.json").read_text())
            total_inst += stats["n_instances"]
            skip += 1
            continue
        try:
            stats = build_scene(src_root, out_root, scene, instance_classes, label_map)
        except Exception as e:  # noqa: BLE001
            print(f"[{scene}] FAIL: {e}", flush=True)
            fail += 1
            failed.append(scene)
            continue
        total_inst += stats["n_instances"]
        ok += 1
        print(f"[{scene}] {stats['n_vertices']} verts, {stats['n_seg_groups']} groups -> "
              f"{stats['n_instances']} benchmark instances "
              f"({stats['n_labelled_vertices'] / stats['n_vertices']:.1%} of vertices)",
              flush=True)

    print(f"[gt] Done: ok={ok} skip={skip} fail={fail}, {total_inst} instances over "
          f"{ok + skip} scenes", flush=True)
    if failed:
        print("[gt] FAILED scenes (re-run to resume): " + ", ".join(failed), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
