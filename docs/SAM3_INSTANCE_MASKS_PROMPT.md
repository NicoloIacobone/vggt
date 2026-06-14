# Prompt for the SAM3-side agent — per-instance ScanNet masks

Copy everything below the line into the agent session that has access to the SAM3
preprocessing pipeline.

---

## Context

You are working on the **data-preprocessing side** of a two-part project. A separate,
downstream project trains a DETR-like 3D multi-view instance-segmentation head on top of a
frozen VGGT backbone, supervised by SAM3 masks computed on ScanNet scenes. That downstream
project is NOT your concern — your only job is to produce its ground truth.

An existing SAM3 pipeline (locate it in this workspace before doing anything — it is the
pipeline that produced the data described below) already processed **97 ScanNet scenes**
into **per-CLASS binary masks**. The supervisor has now decided (2026-06-12) that the
ground truth must be **per-INSTANCE masks with cross-frame-consistent instance identity**.
Your task: extend/modify the pipeline to emit per-instance masks, QA the result on a few
scenes, then run it over all scenes.

## Current on-disk format (verified — treat as the contract)

Root: `/cluster/work/igp_psr/niacobone/distillation/dataset/scannet/scans/`

For each scene (e.g. `scene0000_00`), under `<scene>/raw_data/`:

- `color/` — all >5500 original RGB frames (not used downstream; ignore).
- `subset/` — exactly **100 stride-5 frames** (`00000.jpg`, `00005.jpg`, …, `00495.jpg`),
  1296×968. These are the ONLY frames that matter.
- `masks/<class>/<frame>.png` — current per-class GT: one uint8 PNG per subset frame per
  class, values **{0, 255}**, full resolution 1296×968, filename matching the subset frame
  (`00000.png` ↔ `00000.jpg`). Every class dir contains all 100 frames; the mask is
  all-zero when the class is absent from that frame.
- The 19 class names (fixed taxonomy — do not add, rename, or reorder): `bathtub`, `bed`,
  `bookshelf`, `cabinet`, `chair`, `counter`, `curtain`, `desk`, `door`, `floor`,
  `picture`, `refrigerator`, `shower_curtain`, `sink`, `sofa`, `table`, `toilet`, `wall`,
  `window`.

97 scenes exist: `scene0000_00` … `scene0096_00`, all with `subset/` + `masks/`.

## Target output format

Write to a **new sibling directory** — do NOT modify, move, or delete the existing
`masks/` (the downstream code still consumes it until its loader is updated):

```
<scene>/raw_data/masks_instance/<class>_<k>/<frame>.png
```

- `<k>` is a zero-based instance index within the class, e.g. `chair_0/`, `chair_1/`,
  `wall_0/`. One directory per physical object instance.
- Same PNG conventions as today: uint8, {0, 255}, 1296×968, one PNG for **every** subset
  frame in **every** instance dir (all-zero in frames where that instance is not visible),
  filenames identical to the subset frame names.

## Hard requirements

1. **Cross-frame instance identity is the whole point.** The same physical object must map
   to the same `<class>_<k>` directory in every subset frame where it appears. Use SAM3's
   tracking/video propagation (or whatever association mechanism the pipeline supports)
   across the 100 subset frames — per-frame independent segmentation with no association
   is NOT acceptable. ScanNet subset frames are an ordered video walkthrough (stride-5),
   so temporal propagation applies. If an instance is lost and re-detected, make a
   best-effort re-association (e.g. mask-overlap / IoU with the last seen state); document
   whatever policy you implement.
2. **"Stuff" classes stay single-instance**: `wall` and `floor` always get exactly one
   instance (`wall_0`, `floor_0`) covering all their pixels. All other classes are "things"
   and must be separated into instances.
3. **Consistency with the per-class GT**: for every class and frame, the union of that
   class's instance masks should match the existing `masks/<class>/<frame>.png` (small
   deviations from re-running the model are acceptable; a systematically different
   segmentation is not — prefer decomposing/associating the existing per-class masks if
   the pipeline allows it, otherwise re-run SAM3 with instance output and verify union-IoU
   against the old masks per scene, reporting it).
4. **Instances within one class must be (near-)disjoint** in every frame; tiny boundary
   overlaps are fine, duplicated objects (two `k`s tracking the same chair) are not.
5. Don't renumber instances mid-scene: once `chair_2` exists, it stays `chair_2`.

## Protocol — QA first, bulk second (do not skip)

**Step 1 — pilot + visual QA on exactly 3 scenes**: `scene0000_00`, `scene0005_00`,
`scene0080_00` (the last is a bathroom — small fixtures stress tracking). For each, render
a QA strip: ~10 evenly spaced subset frames with all instance masks overlaid, **one fixed
color per instance directory, consistent across frames**, legend `"{class}_{k}"`. Save to
`<scene>/raw_data/masks_instance/_qa/overview.jpg` (or a grid of jpgs). Then verify, by
actually looking at the rendered images:
- the same object keeps the same color across all frames (identity holds through
  viewpoint changes / temporary occlusion);
- two same-class objects (e.g. two chairs) get different colors;
- instance counts per class are plausible for the scene;
- union of instances ≈ old per-class mask (report mean IoU per class).

**Step 2 — STOP and report.** Present the QA images, per-scene instance counts, union-IoU
numbers, and your association policy to the user for approval **before** launching the
bulk run. Cross-frame identity is the load-bearing assumption of the downstream project —
if tracking is unreliable, the bulk run is wasted compute.

**Step 3 — bulk run after approval**, in this priority order (downstream needs val scenes
first):
1. `scene0080_00`–`scene0089_00` (validation scenes),
2. `scene0000_00`–`scene0049_00` (scaling-experiment train scenes),
3. `scene0050_00`–`scene0096_00` (the rest).

Make the run resumable (skip scenes whose `masks_instance/` is already complete) and log
per-scene: instance counts per class, union-IoU vs old masks, wall-clock time, failures.

**Step 4 (stretch, only if asked)**: download + preprocess additional ScanNet scenes
beyond the 97 (target: 100+ total for the downstream scaling curve), same format (both
`subset/`, `masks/` per-class for continuity, and `masks_instance/`).

## Deliverables

1. `masks_instance/` for all 97 scenes in the format above.
2. The QA overlays (Step 1) kept in place for spot-checking any scene.
3. A short report file at
   `/cluster/work/igp_psr/niacobone/distillation/dataset/scannet/INSTANCE_MASKS_README.md`
   documenting: exact layout, the association/tracking policy, lost-track re-association
   policy, per-scene instance-count table, union-IoU stats, and any scenes that failed or
   look unreliable (the downstream side will exclude those).

Work incrementally and verify each step before scaling it up. If anything in the existing
pipeline contradicts the format described above, stop and report the discrepancy instead
of guessing.
