"""
Build a 2D instance-segmentation training set out of the InsScene-15K mirror (todo 6f).

InsScene-15K already ships per-frame instance annotations, so NOTHING is rendered here: this is
a selection + re-encoding pass. All three of its subsets can supervise our head — but not with
the same kind of label, and that difference must travel with every number:

  processed_scannetpp_v2   903 scenes, `images/` + `refined_ins_ids/` (int16 per-pixel ids,
                           **globally consistent across the frames of a scene** — verified: two
                           adjacent frames of 00777c41d4 share 34 of 34 ids). USED.
  processed_infinigen      1466 sub-scene zips, `Image/` + `ObjectSegmentation/` (int64 ids that
                           index `Objects/*.json`, so every instance has a NAME). USED.
  processed_re10k          5127 scenes with `sam2_results/<scene>/auto_masks.json` — SA-V
                           masklets, COCO-RLE per frame, ids persistent across the whole clip.
                           USED, but the masks are **SAM2 output, not ground truth**: every row
                           trained on this must say "SAM2-supervised" (docs/MULTIDATASET.md §1.3).

Output, one directory per scene, already at the trainer's input resolution:

    <out>/<source>/<scene>/color/<stem>.jpg        518x518 squash, matching data/scannet_overfit
    <out>/<source>/<scene>/instance/<stem>.png     uint16 instance-id map, 0 = background
    <out>/<source>/<scene>/manifest.json           frames, id table, provenance, QA counters

**Ids are remapped per SCENE, never per frame.** The multi-frame head re-links instances across
views by global instance id (CLAUDE.md, "the batch dimension is FRAMES"), so a per-frame
relabelling would destroy exactly the signal the bundle GT is built on.

**Three exclusions worth reading before quoting any number trained on this:**

1. `--exclude_scenes` drops ScanNet++ scenes from the build. The mirror contains ALL 49 scenes of
   our ScanNet++ evaluation column (docs/RESULTS.md §7), so training on it unfiltered would leak
   the entire zero-shot benchmark. The job script passes the official `nvs_sem_val` list.
2. RE10K's SAM2 masks are unnamed, so its room shell can only be dropped by AREA. Measured over
   60 scenes: the median instance is 0.2 % of the frame and p99 is 22 %, so a scene-wide cap at
   **30 %** removes 0.5 % of instances and 0 % of the labelled pixels of the median scene, while
   the 0.20 cap that also looked plausible costs 22 % of them (docs/MULTIDATASET.md §1.4).
3. Infinigen labels the room shell as ordinary instances (`<room>/N.wall|floor|ceiling|exterior`,
   measured at 21 %, 17 % and 32 % of one frame). ScanNet's benchmark excludes wall/floor and our
   Replica GT excludes the room shell (docs/DATASET.md §2.2), so they are dropped here too, BY
   NAME rather than by an area heuristic — the ids index `Objects/*.json`, which names them.

Usage (see slurm/build_insscene2d.sh for the cluster driver):

    python slurm/build_insscene2d.py --source scannetpp --out $TMPDIR/build --frames 32 \
        --exclude_scenes data/splits/scannetpp_nvs_sem_val.txt
    python slurm/build_insscene2d.py --source infinigen --out $TMPDIR/build --frames 32
    python slurm/build_insscene2d.py --source re10k --out $TMPDIR/build --frames 32
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slurm.coco_rle import decode_counts, masklets_to_instance_map, rle_area  # noqa: E402
from slurm.insscene_shards import SplitZipReader, scene_ids  # noqa: E402

DEFAULT_MIRROR = Path("/cluster/work/igp_psr/niacobone/distillation/dataset/insscene15k")
IMG_SIZE = 518
JPEG_QUALITY = 92
MAX_INSTANCE_ID = 65535          # the on-disk map is uint16

# Infinigen's room shell, dropped to match the ScanNet benchmark and our Replica GT.
SHELL_RE = re.compile(r"\.(wall|floor|ceiling|exterior)s?$", re.IGNORECASE)
INFINIGEN_FRAME_RE = re.compile(r"_(\d+)_\d+_\d+_\d+\.(npy|png|jpg|json|npz)$")

# RE10K's SAM2 masks are unnamed, so the room shell can only go by area. 0.30 of the frame,
# averaged over the kept frames of a scene — measured, not assumed: docs/MULTIDATASET.md §1.4.
RE10K_MAX_AREA_FRAC = 0.30
RE10K_MASK_DIR = "sam2_results"  # a SIBLING of the scene dirs, which is why it was missed once


# --------------------------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------------------------

def even_indices(n: int, k: int) -> List[int]:
    """`k` evenly spaced indices spanning 0..n-1 — the sampling `frame_sampling='even'` uses."""
    if n <= 0:
        return []
    if k >= n:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, k).round().astype(int).tolist()))


def resize_instance_map(ids: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """NEAREST, so ids are never blended into ids that do not exist."""
    return np.array(Image.fromarray(ids.astype(np.int32), mode="I").resize(
        (size, size), Image.NEAREST), dtype=np.int32)


def remap_scene_ids(per_frame: Dict[str, np.ndarray],
                    keep: Optional[set] = None,
                    min_area_px: int = 0) -> Tuple[Dict[str, np.ndarray], Dict[int, int]]:
    """
    Collapse the source ids of ONE scene onto a dense 1..G, shared by every frame.

    `keep` (when given) is the set of source ids allowed to survive; everything else — and
    everything below `min_area_px` summed over the sampled frames — becomes background.
    Returns the rewritten maps and the {source id: new id} table for the manifest.
    """
    areas: Dict[int, int] = {}
    for ids in per_frame.values():
        values, counts = np.unique(ids, return_counts=True)
        for value, count in zip(values.tolist(), counts.tolist()):
            if value == 0 or (keep is not None and value not in keep):
                continue
            areas[value] = areas.get(value, 0) + int(count)

    survivors = sorted(v for v, a in areas.items() if a >= min_area_px)
    if len(survivors) > MAX_INSTANCE_ID:
        raise ValueError(f"{len(survivors)} instances exceed the uint16 map")
    table = {src: new for new, src in enumerate(survivors, start=1)}

    out = {}
    for stem, ids in per_frame.items():
        remapped = np.zeros(ids.shape, dtype=np.uint16)
        for src, new in table.items():
            remapped[ids == src] = new
        out[stem] = remapped
    return out, table


def write_scene(out_dir: Path, stems: Sequence[str], images: Dict[str, Image.Image],
                maps: Dict[str, np.ndarray], meta: dict) -> dict:
    """Write color/, instance/ and manifest.json; return the scene's QA counters."""
    (out_dir / "color").mkdir(parents=True, exist_ok=True)
    (out_dir / "instance").mkdir(parents=True, exist_ok=True)
    per_frame_counts = []
    for stem in stems:
        images[stem].save(out_dir / "color" / f"{stem}.jpg", quality=JPEG_QUALITY)
        Image.fromarray(maps[stem], mode="I;16").save(out_dir / "instance" / f"{stem}.png")
        per_frame_counts.append(int(len(np.unique(maps[stem])) - (0 in maps[stem])))
    num_instances = int(max((int(m.max()) for m in maps.values()), default=0))
    manifest = dict(meta, frames=list(stems), num_instances=num_instances,
                    instances_per_frame=per_frame_counts, image_size=IMG_SIZE)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return dict(frames=len(stems), instances=num_instances)


# --------------------------------------------------------------------------------------------
# processed_scannetpp_v2 — one split zip
# --------------------------------------------------------------------------------------------

def build_scannetpp(mirror: Path, out: Path, frames: int, exclude: set,
                    limit: Optional[int], min_area_px: int) -> dict:
    reader = SplitZipReader(mirror / "processed_scannetpp_v2", "processed_scannetpp_v2")
    print(f"[scannetpp] parsing the central directory of {reader.total / 2**30:.0f} GiB ...",
          flush=True)
    members = reader.members()
    print(f"[scannetpp] {len(members)} entries", flush=True)

    scenes = [s for s in scene_ids(reader, "processed_scannetpp_v2") if s not in exclude]
    skipped = len(scene_ids(reader, "processed_scannetpp_v2")) - len(scenes)
    print(f"[scannetpp] {len(scenes)} scenes to build, {skipped} excluded", flush=True)
    if limit:
        scenes = scenes[:limit]

    report = {"source": "scannetpp", "excluded_scenes": skipped, "scenes": {}, "failed": {}}
    for i, scene in enumerate(scenes, 1):
        started = time.time()
        try:
            prefix = f"processed_scannetpp_v2/{scene}/"
            image_names = sorted(n for n in members
                                 if n.startswith(prefix + "images/") and n.endswith(".jpg"))
            picked = [image_names[j] for j in even_indices(len(image_names), frames)]
            per_frame, images, stems = {}, {}, []
            for name in picked:
                stem = Path(name).stem
                ids_name = f"{prefix}refined_ins_ids/{stem}.jpg.npy"
                if ids_name not in members:
                    continue
                ids = np.load(io.BytesIO(reader.read(ids_name)))
                image = Image.open(io.BytesIO(reader.read(name))).convert("RGB")
                if ids.shape[:2] != (image.height, image.width):
                    raise ValueError(f"{stem}: ids {ids.shape} vs image {image.size}")
                per_frame[stem] = resize_instance_map(ids)
                images[stem] = image.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
                stems.append(stem)
            if not stems:
                raise ValueError("no frame carried an instance map")
            maps, table = remap_scene_ids(per_frame, min_area_px=min_area_px)
            counters = write_scene(
                out / "scannetpp" / scene, stems, images, maps,
                dict(source="insscene15k/processed_scannetpp_v2", scene=scene,
                     source_frames=len(image_names), id_table={str(k): v for k, v in table.items()}))
            report["scenes"][scene] = counters
            print(f"[scannetpp {i}/{len(scenes)}] {scene}: {counters['frames']} frames, "
                  f"{counters['instances']} instances, {time.time() - started:.1f}s", flush=True)
        except Exception as exc:                                  # one bad scene must not stop 900
            report["failed"][scene] = f"{type(exc).__name__}: {exc}"
            print(f"[scannetpp {i}/{len(scenes)}] {scene} FAILED: {exc}", flush=True)
    return report


# --------------------------------------------------------------------------------------------
# processed_infinigen — one ordinary zip per sub-scene
# --------------------------------------------------------------------------------------------

def infinigen_keep_ids(objects: dict) -> set:
    """Instance ids that are real objects: MESH, and not the room shell."""
    keep = set()
    for name, value in objects.items():
        if not isinstance(value, dict) or value.get("type") != "MESH":
            continue
        index = value.get("object_index")
        if index is None or SHELL_RE.search(name):
            continue
        keep.add(int(index))
    return keep


def build_infinigen(mirror: Path, out: Path, frames: int, limit: Optional[int],
                    min_area_px: int) -> dict:
    root = mirror / "processed_infinigen"
    zips = sorted(p for scene in sorted(root.iterdir()) if scene.is_dir()
                  for p in sorted(scene.glob("*.zip")))
    print(f"[infinigen] {len(zips)} sub-scene archives", flush=True)
    if limit:
        zips = zips[:limit]

    report = {"source": "infinigen", "scenes": {}, "failed": {}}
    for i, path in enumerate(zips, 1):
        name = f"{path.parent.name}_{path.stem}"
        started = time.time()
        try:
            with zipfile.ZipFile(path) as archive:
                entries = archive.namelist()

                def by_frame(prefix: str, suffix: str) -> Dict[str, str]:
                    out_map = {}
                    for entry in entries:
                        if entry.startswith(prefix) and entry.endswith(suffix):
                            match = INFINIGEN_FRAME_RE.search(entry)
                            if match:
                                out_map[match.group(1)] = entry
                    return out_map

                images_by = by_frame("frames/Image/", ".png")
                segs_by = by_frame("frames/ObjectSegmentation/", ".npy")
                objs_by = by_frame("frames/Objects/", ".json")
                keys = sorted(set(images_by) & set(segs_by), key=int)
                picked = [keys[j] for j in even_indices(len(keys), frames)]
                if not picked:
                    raise ValueError("no frame has both an image and a segmentation")

                keep = set()
                for key in picked[:1] or picked:                   # the object table is per scene
                    if key in objs_by:
                        keep |= infinigen_keep_ids(json.loads(archive.read(objs_by[key])))

                per_frame, images, stems = {}, {}, []
                for key in picked:
                    stem = f"frame_{int(key):06d}"
                    ids = np.load(io.BytesIO(archive.read(segs_by[key])))
                    image = Image.open(io.BytesIO(archive.read(images_by[key]))).convert("RGB")
                    per_frame[stem] = resize_instance_map(ids)
                    images[stem] = image.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
                    stems.append(stem)
                maps, table = remap_scene_ids(per_frame, keep=keep or None,
                                              min_area_px=min_area_px)
            counters = write_scene(
                out / "infinigen" / name, stems, images, maps,
                dict(source="insscene15k/processed_infinigen", scene=name,
                     archive=str(path.relative_to(root)), source_frames=len(keys),
                     shell_dropped=True, id_table={str(k): v for k, v in table.items()}))
            report["scenes"][name] = counters
            if i % 25 == 0 or i == 1:
                print(f"[infinigen {i}/{len(zips)}] {name}: {counters['frames']} frames, "
                      f"{counters['instances']} instances, {time.time() - started:.1f}s",
                      flush=True)
        except Exception as exc:
            report["failed"][name] = f"{type(exc).__name__}: {exc}"
            print(f"[infinigen {i}/{len(zips)}] {name} FAILED: {exc}", flush=True)
    return report


# --------------------------------------------------------------------------------------------
# processed_re10k — one split zip, SAM2 masklets in a SIBLING top-level directory
# --------------------------------------------------------------------------------------------

def re10k_frame_stems(members: Iterable[str]) -> Dict[str, List[str]]:
    """
    Every scene's rgb stems in **numeric** order — never lexicographic. Indexed once, not per
    scene: the central directory holds 1.22 M names and rescanning it 5127 times costs half an
    hour on its own.

    `masklet` is indexed by frame POSITION; `rgb/` is keyed by a timestamp stem. Those stems are
    8 OR 9 digits long (307 821 vs 287 683 across the mirror), so a lexicographic sort puts every
    9-digit stem before every 8-digit one and silently misaligns the masks in the **107 scenes**
    that mix the two widths. Sorting by int is the whole fix, and it is the reason this helper
    exists rather than a `sorted()` at the call site.
    """
    out: Dict[str, List[str]] = {}
    for name in members:
        parts = name.split("/")
        if len(parts) == 4 and parts[0] == "processed_re10k" and parts[2] == "rgb" \
                and parts[3].endswith(".png"):
            out.setdefault(parts[1], []).append(parts[3][:-4])
    for scene, stems in out.items():
        if not all(stem.isdigit() for stem in stems):
            raise ValueError(f"{scene}: non-numeric rgb stem")
        stems.sort(key=int)
    return out


def re10k_keep_ids(masklet: Sequence, picked: Sequence[int], frame_px: int,
                   max_area_frac: float) -> Tuple[set, int]:
    """
    Masklet indices that are not room shell, by AREA — SAM2's masks carry no names.

    An instance is dropped when its area **averaged over the kept frames** exceeds
    `max_area_frac` of the frame. Averaging over the scene rather than thresholding per frame is
    deliberate: a per-frame rule would make an instance flicker in and out of the GT, and the
    multi-frame head re-links instances across views by id, so a flickering id is worse than
    either keeping or dropping it outright.

    Returns the ids to keep (1-based, as `masklets_to_instance_map` writes them) and the number
    of `None` masklet entries seen, which the report carries as a data-quality counter.
    """
    if not picked:
        return set(), 0
    n_obj = len(masklet[picked[0]])
    area = np.zeros(n_obj, dtype=np.int64)
    missing = 0
    for j in picked:
        for index, rle in enumerate(masklet[j]):
            if rle is None:
                missing += 1
                continue
            area[index] += rle_area(decode_counts(rle["counts"]))
    limit = max_area_frac * frame_px * len(picked)
    return {index + 1 for index in range(n_obj) if 0 < area[index] <= limit}, missing


def build_re10k(mirror: Path, out: Path, frames: int, exclude: set, limit: Optional[int],
                min_area_px: int, max_area_frac: float) -> dict:
    reader = SplitZipReader(mirror / "processed_re10k", "processed_re10k")
    print(f"[re10k] parsing the central directory of {reader.total / 2**30:.0f} GiB ...",
          flush=True)
    members = reader.members()
    print(f"[re10k] {len(members)} entries", flush=True)

    # The masks are a SIBLING of the scene dirs, `processed_re10k/sam2_results/<scene>/`, not a
    # child of them — the original survey grouped by the depth-2 component and never saw them
    # (docs/MULTIDATASET.md §1.3). 5127 of the 5138 rgb scenes have one; the rest cannot be used.
    stems_by_scene = re10k_frame_stems(members)
    with_masks = {name.split("/")[2] for name in members
                  if name.startswith(f"processed_re10k/{RE10K_MASK_DIR}/")
                  and name.endswith("/auto_masks.json")}
    all_scenes = sorted(stems_by_scene)
    scenes = [s for s in all_scenes if s in with_masks and s not in exclude]
    print(f"[re10k] {len(all_scenes)} rgb scenes, {len(with_masks)} with masks, "
          f"{len(scenes)} to build ({len(all_scenes) - len(with_masks)} unannotated, "
          f"{len(exclude & set(all_scenes))} excluded)", flush=True)
    if limit:
        scenes = scenes[:limit]

    report = {"source": "re10k", "supervision": "SAM2 auto-masks, NOT ground truth",
              "unannotated_scenes": len(all_scenes) - len(with_masks),
              "excluded_scenes": len(exclude & set(all_scenes)),
              "max_area_frac": max_area_frac, "scenes": {}, "failed": {}}
    for i, scene in enumerate(scenes, 1):
        started = time.time()
        try:
            meta = json.loads(reader.read(
                f"processed_re10k/{RE10K_MASK_DIR}/{scene}/auto_masks.json"))
            height, width = int(meta["video_height"]), int(meta["video_width"])
            masklet = meta["masklet"]
            stems = stems_by_scene[scene]

            # Frame <-> mask alignment is positional, so the two counts MUST agree. A scene where
            # they do not is skipped and counted, never guessed at.
            if not (len(stems) == len(masklet) == int(meta["video_frame_count"])):
                raise ValueError(f"frame counts disagree: {len(stems)} rgb, {len(masklet)} "
                                 f"masklet rows, video_frame_count {meta['video_frame_count']}")

            picked = even_indices(len(stems), frames)
            keep, missing = re10k_keep_ids(masklet, picked, height * width, max_area_frac)
            per_frame, images, kept_stems = {}, {}, []
            for j in picked:
                stem = stems[j]
                image = Image.open(io.BytesIO(
                    reader.read(f"processed_re10k/{scene}/rgb/{stem}.png"))).convert("RGB")
                if (image.height, image.width) != (height, width):
                    raise ValueError(f"{stem}: rgb {image.size} vs masks ({width}, {height})")
                ids = masklets_to_instance_map(
                    [rle if index + 1 in keep else None for index, rle in enumerate(masklet[j])],
                    height, width)
                per_frame[stem] = resize_instance_map(ids)
                images[stem] = image.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
                kept_stems.append(stem)
            if not kept_stems:
                raise ValueError("no frame survived")
            maps, table = remap_scene_ids(per_frame, min_area_px=min_area_px)
            counters = write_scene(
                out / "re10k" / scene, kept_stems, images, maps,
                dict(source="insscene15k/processed_re10k", scene=scene,
                     supervision="sam2", source_frames=len(stems),
                     masklets=len(masklet[picked[0]]), shell_dropped_by_area=max_area_frac,
                     none_masklet_entries=missing,
                     id_table={str(k): v for k, v in table.items()}))
            report["scenes"][scene] = dict(counters, missing=missing,
                                           masklets=len(masklet[picked[0]]))
            if i % 100 == 0 or i == 1:
                print(f"[re10k {i}/{len(scenes)}] {scene}: {counters['frames']} frames, "
                      f"{counters['instances']} instances of {len(masklet[picked[0]])} masklets, "
                      f"{time.time() - started:.1f}s", flush=True)
        except Exception as exc:                              # one bad scene must not stop 5000
            report["failed"][scene] = f"{type(exc).__name__}: {exc}"
            print(f"[re10k {i}/{len(scenes)}] {scene} FAILED: {exc}", flush=True)
    return report


# --------------------------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["scannetpp", "infinigen", "re10k"], required=True)
    parser.add_argument("--mirror", type=Path, default=DEFAULT_MIRROR)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=32,
                        help="frames kept per scene, evenly spaced (default 32)")
    parser.add_argument("--exclude_scenes", type=Path, default=None,
                        help="file of scene ids to skip — ALWAYS pass the eval split here")
    parser.add_argument("--limit", type=int, default=None, help="first N scenes (smoke tests)")
    parser.add_argument("--min_area_px", type=int, default=64,
                        help="drop instances smaller than this, summed over the kept frames")
    parser.add_argument("--max_area_frac", type=float, default=RE10K_MAX_AREA_FRAC,
                        help="re10k only: drop instances covering more than this fraction of the "
                             "frame on average — the room-shell filter (default %(default)s)")
    args = parser.parse_args()

    exclude = set()
    if args.exclude_scenes:
        exclude = {line.strip() for line in args.exclude_scenes.read_text().split() if line.strip()}
        print(f"excluding {len(exclude)} scenes listed in {args.exclude_scenes}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if args.source == "scannetpp":
        report = build_scannetpp(args.mirror, args.out, args.frames, exclude, args.limit,
                                 args.min_area_px)
    elif args.source == "re10k":
        report = build_re10k(args.mirror, args.out, args.frames, exclude, args.limit,
                             args.min_area_px, args.max_area_frac)
    else:
        report = build_infinigen(args.mirror, args.out, args.frames, args.limit, args.min_area_px)

    report["elapsed_s"] = round(time.time() - started, 1)
    report["frames_per_scene"] = args.frames
    report["min_area_px"] = args.min_area_px
    scenes = report["scenes"]
    report["totals"] = {
        "scenes": len(scenes),
        "failed": len(report["failed"]),
        "frames": sum(s["frames"] for s in scenes.values()),
        "instances": sum(s["instances"] for s in scenes.values()),
        "median_instances_per_scene":
            float(np.median([s["instances"] for s in scenes.values()])) if scenes else 0.0,
    }
    (args.out / f"REPORT_{args.source}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report["totals"], indent=1), flush=True)
    return 1 if report["failed"] and not scenes else 0


if __name__ == "__main__":
    raise SystemExit(main())
