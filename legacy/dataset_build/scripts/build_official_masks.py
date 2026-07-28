"""Convert official ScanNet 2D GT (instance-filt / label-filt) into the SAM3 mask layout.

Offline converter for the official-GT migration (docs/old/OFFICIAL_GT_MIGRATION_PLAN.md).
For each scene it reads the two official zips (or extracted dirs), derives one class
per instance via the label-filt majority vote + the scannetv2-labels tsv (id -> nyu40id),
and emits exactly the SAM3 on-disk conventions the loader is proven against:

    <out_root>/<scene>/raw_data/
        subset/                       (copied unchanged from the SAM3 tree, optional)
        masks/<class>/<frame>.png              per-class union, uint8 {0,255}
        masks_instance/<class>_<k>/<frame>.png per-instance, uint8 {0,255}
        _qa/stats.json                per-scene QA (counts + cross-class duplicate check)

Conventions mirrored from the SAM3 build (do not change):
- dir names use underscores ('shower_curtain_3'); <k> is zero-based per class in order
  of first appearance (first frame, then instance id).
- masks are written SPARSELY: only frames where the instance is visible; the loader
  skips missing files.
- filenames match the subset stems ('00375.png' for official frame index 375).
- NYU40 classes outside the 19 trainable ones (incl. otherfurniture=39) -> background
  (dropped): the class head has 20 logits = background + classes 1..19.
- instance ids of stuff classes (wall/floor) are kept as-is (NOT merged to _0).

Reads PNGs straight out of the zips (no extraction -> no inode churn).
Resumable: a scene with an existing _qa/stats.json + .complete marker is skipped.

Usage (single scene):
    python legacy/dataset_build/scripts/build_official_masks.py \
        --scene scene0000_00 --zips_dir <dir with <scene>_2d-{instance,label}-filt.zip> \
        --out_root <build>/scans --subset_root <unpacked_sam3>/scans
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

# SCANNET_CLASSES[i] has class index i+1; index 0 = background.  Kept in sync with
# data/scannet_overfit.py::SCANNET_CLASSES (first 19 entries; otherfurniture excluded
# on purpose -- unrepresentable in the 20-logit class head, mapped to background).
TRAINABLE_CLASSES = [
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
    "window", "bookshelf", "picture", "counter", "desk", "curtain",
    "refrigerator", "shower curtain", "toilet", "sink", "bathtub",
]
# NYU40 id of each trainable class, same order (see plan section 2).
NYU40_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36]
NYU40_TO_CLASS = {n: c for n, c in zip(NYU40_IDS, TRAINABLE_CLASSES)}


def load_label_map(tsv_path: str | Path) -> dict[int, int]:
    """scannetv2-labels.combined.tsv -> {raw ScanNet label id: nyu40 id}."""
    id2nyu = {}
    with open(tsv_path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                id2nyu[int(row["id"])] = int(row["nyu40id"])
            except (KeyError, ValueError):
                continue
    if not id2nyu:
        raise ValueError(f"No id->nyu40id rows parsed from {tsv_path}")
    return id2nyu


class _FrameSource:
    """Uniform PNG access over a zip file or an extracted directory."""

    def __init__(self, path: str | Path, inner_dir: str):
        self.path = Path(path)
        self.inner = inner_dir  # 'instance-filt' or 'label-filt'
        if self.path.suffix == ".zip":
            self.zf = zipfile.ZipFile(self.path)
            self.members = set(self.zf.namelist())
        else:
            self.zf = None

    def has(self, frame_idx: int) -> bool:
        if self.zf is not None:
            return f"{self.inner}/{frame_idx}.png" in self.members
        return (self.path / self.inner / f"{frame_idx}.png").exists()

    def read(self, frame_idx: int) -> np.ndarray:
        if self.zf is not None:
            data = self.zf.read(f"{self.inner}/{frame_idx}.png")
            return np.array(Image.open(io.BytesIO(data)))
        return np.array(Image.open(self.path / self.inner / f"{frame_idx}.png"))

    def close(self):
        if self.zf is not None:
            self.zf.close()


def cross_class_duplicate_check(inst_frame_masks, inst_class, iou_thresh=0.5):
    """Pairwise cross-frame IoU between instances of DIFFERENT classes.

    Same metric as the 2026-07-07 SAM3 audit: per instance-pair, IoU of the
    pixel sets unioned over all frames.  By construction (single-valued
    instance-id map) this must be ~0 for official GT; computed as the
    migration's acceptance evidence.
    Returns (n_pairs_over_thresh, max_iou, pairs).
    """
    ids = sorted(inst_frame_masks)
    max_iou, pairs = 0.0, []
    for a_i, a in enumerate(ids):
        for b in ids[a_i + 1:]:
            if inst_class[a] == inst_class[b]:
                continue
            frames = set(inst_frame_masks[a]) | set(inst_frame_masks[b])
            inter = union = 0
            for f in frames:
                ma = inst_frame_masks[a].get(f)
                mb = inst_frame_masks[b].get(f)
                if ma is None:
                    union += int(mb.sum())
                elif mb is None:
                    union += int(ma.sum())
                else:
                    inter += int((ma & mb).sum())
                    union += int((ma | mb).sum())
            iou = inter / union if union else 0.0
            max_iou = max(max_iou, iou)
            if iou >= iou_thresh:
                pairs.append((a, b, round(iou, 4)))
    return len(pairs), max_iou, pairs


def convert_scene(
    scene: str,
    instance_src: str | Path,
    label_src: str | Path,
    tsv_path: str | Path,
    out_root: str | Path,
    subset_src: str | Path | None = None,
    frame_indices: list[int] | None = None,
) -> dict:
    """Build <out_root>/<scene>/raw_data/{masks,masks_instance,_qa} from official GT.

    frame_indices: official color-frame indices to convert. Default: stems of the
    subset_src jpgs if given, else 0..495 step 5 intersected with available frames.
    Returns the stats dict (also written to _qa/stats.json).
    """
    id2nyu = load_label_map(tsv_path)
    inst_zf = _FrameSource(instance_src, "instance-filt")
    lab_zf = _FrameSource(label_src, "label-filt")

    raw_dir = Path(out_root) / scene / "raw_data"

    if frame_indices is None:
        if subset_src is not None:
            frame_indices = sorted(int(p.stem) for p in Path(subset_src).glob("*.jpg"))
        else:
            frame_indices = [i for i in range(0, 500, 5)]
    frame_indices = [f for f in frame_indices if inst_zf.has(f)]
    if not frame_indices:
        raise ValueError(f"{scene}: no requested frames present in {instance_src}")

    # Pass 1: per-frame instance maps + global class vote per instance id.
    inst_votes: dict[int, Counter] = defaultdict(Counter)   # inst id -> nyu40 votes (px)
    inst_frame_masks: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    first_seen: dict[int, tuple[int, int]] = {}
    for f in frame_indices:
        ia = inst_zf.read(f)
        la = lab_zf.read(f)
        if ia.shape != la.shape:
            raise ValueError(f"{scene} frame {f}: instance {ia.shape} vs label {la.shape}")
        for inst in np.unique(ia):
            if inst == 0:
                continue
            m = ia == inst
            labs, cnts = np.unique(la[m], return_counts=True)
            for l, c in zip(labs, cnts):
                nyu = id2nyu.get(int(l))
                if nyu is not None:
                    inst_votes[int(inst)][nyu] += int(c)
            inst_frame_masks[int(inst)][f] = m
            first_seen.setdefault(int(inst), (f, int(inst)))
    inst_zf.close()
    lab_zf.close()

    # Class per instance (global majority vote); drop out-of-taxonomy instances.
    inst_class: dict[int, str] = {}
    dropped: dict[int, int] = {}  # inst id -> winning nyu40 id (out of taxonomy; -1 = no
    #                               mappable label pixels at all, e.g. label-filt 0/unknown)
    label_purity: dict[int, float] = {}
    for inst in list(inst_frame_masks):
        votes = inst_votes.get(inst)
        if not votes:
            dropped[inst] = -1
            continue
        nyu, top = votes.most_common(1)[0]
        label_purity[inst] = top / sum(votes.values())
        cls = NYU40_TO_CLASS.get(nyu)
        if cls is None:
            dropped[inst] = nyu
        else:
            inst_class[inst] = cls
    for inst in dropped:
        inst_frame_masks.pop(inst, None)

    # Per-class k in order of first appearance (frame, then instance id).
    inst_k: dict[int, int] = {}
    per_class_counter: dict[str, int] = defaultdict(int)
    for inst in sorted(inst_class, key=lambda i: first_seen[i]):
        cls = inst_class[inst]
        inst_k[inst] = per_class_counter[cls]
        per_class_counter[cls] += 1

    # Pass 2: write sparse per-instance masks + per-class unions.
    class_union: dict[str, dict[int, np.ndarray]] = defaultdict(dict)  # cls -> frame -> mask
    n_inst_pngs = 0
    for inst, frames in inst_frame_masks.items():
        cls = inst_class[inst]
        seg_dir = raw_dir / "masks_instance" / f"{cls.replace(' ', '_')}_{inst_k[inst]}"
        seg_dir.mkdir(parents=True, exist_ok=True)
        for f, m in frames.items():
            Image.fromarray(m.astype(np.uint8) * 255).save(seg_dir / f"{f:05d}.png")
            n_inst_pngs += 1
            u = class_union[cls].get(f)
            class_union[cls][f] = m if u is None else (u | m)
    n_union_pngs = 0
    for cls, frames in class_union.items():
        cls_dir = raw_dir / "masks" / cls.replace(" ", "_")
        cls_dir.mkdir(parents=True, exist_ok=True)
        for f, m in frames.items():
            Image.fromarray(m.astype(np.uint8) * 255).save(cls_dir / f"{f:05d}.png")
            n_union_pngs += 1

    # Copy subset images unchanged (GT changes, pixels don't).
    if subset_src is not None:
        subset_dst = raw_dir / "subset"
        if not subset_dst.exists():
            shutil.copytree(subset_src, subset_dst)

    # QA stats + acceptance check.
    n_dup, max_iou, dup_pairs = cross_class_duplicate_check(inst_frame_masks, inst_class)
    stats = {
        "scene": scene,
        "source": "official ScanNet v2 2d-instance-filt / 2d-label-filt",
        "num_frames": len(frame_indices),
        "num_instances": len(inst_class),
        "instances_per_class": dict(sorted(per_class_counter.items())),
        "dropped_out_of_taxonomy": {
            str(i): int(n) for i, n in sorted(dropped.items())
        },
        "min_label_purity": round(min(label_purity.values()), 4) if label_purity else None,
        "cross_class_duplicates_iou50": n_dup,
        "cross_class_max_iou": round(max_iou, 4),
        "cross_class_dup_pairs": dup_pairs,
        "mask_pngs_instance": n_inst_pngs,
        "mask_pngs_union": n_union_pngs,
    }
    qa_dir = raw_dir / "_qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    with open(qa_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", required=True, help="scene id (e.g. scene0000_00) or comma list")
    ap.add_argument("--zips_dir", required=True,
                    help="dir holding <scene>_2d-instance-filt.zip + <scene>_2d-label-filt.zip")
    ap.add_argument("--out_root", required=True, help="build tree root (…/scans)")
    ap.add_argument("--subset_root", default=None,
                    help="unpacked SAM3 scans root; copies <scene>/raw_data/subset and "
                         "uses its stems as the frame list")
    ap.add_argument("--tsv", default="/cluster/work/igp_psr/niacobone/distillation/"
                                     "dataset/scannet/scannetv2-labels.combined.tsv")
    ap.add_argument("--force", action="store_true", help="rebuild even if .complete exists")
    args = ap.parse_args()

    scenes = [s for s in args.scene.split(",") if s]
    for scene in scenes:
        raw_dir = Path(args.out_root) / scene / "raw_data"
        marker = raw_dir / ".complete"
        if marker.exists() and not args.force:
            print(f"[{scene}] .complete exists, skip", flush=True)
            continue
        zips = Path(args.zips_dir)
        subset_src = None
        if args.subset_root is not None:
            subset_src = Path(args.subset_root) / scene / "raw_data" / "subset"
            if not subset_src.exists():
                raise FileNotFoundError(subset_src)
        stats = convert_scene(
            scene,
            zips / f"{scene}_2d-instance-filt.zip",
            zips / f"{scene}_2d-label-filt.zip",
            args.tsv,
            args.out_root,
            subset_src=subset_src,
        )
        marker.touch()
        print(f"[{scene}] OK: {stats['num_instances']} instances, "
              f"{stats['mask_pngs_instance']} instance pngs, "
              f"dups(iou>=0.5)={stats['cross_class_duplicates_iou50']}, "
              f"max cross-class IoU={stats['cross_class_max_iou']}", flush=True)


if __name__ == "__main__":
    main()
