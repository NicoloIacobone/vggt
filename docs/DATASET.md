# Dataset, ground truth, and storage

The supervision, the on-disk conventions, and how a job gets the data. This applies to both the
active MaskDINO track and the retired retired baseline heads — they read the same trees through the same
loader (`data/scannet_overfit.py`).

## 1. Ground truth: official ScanNet v2 2D instance annotations

**Default since 2026-07-08.** Projections of the single human-verified 3D annotation
(`_2d-instance-filt` / `_2d-label-filt`): one class per object, cross-view-consistent instance
ids by construction.

**Why the switch.** A 2026-07-07 audit of the previous SAM3-generated GT (20 scenes) found
systematic **cross-class duplicates**: SAM3 prompts each class independently, so the same
physical object is often an instance under two classes — 68 pairs with cross-frame IoU ≥ 0.5
between different classes (~3.4/scene, mostly pixel-identical; desk↔table,
curtain↔shower_curtain, chair↔sofa, …), **15.9 % of foreground pixels multi-class**. Training
effect: the matcher demands two predictions for one object (built-in honest-AP50 false
positives) and the class head gets contradictory supervision.

**What it cost to switch** (the baseline head, N=190, learned queries, per-instance GT, val scenes
0080–0089):

| | val mIoU | honest val[grid] AP50 |
|---|---|---|
| trained on SAM3 GT, evaled on SAM3 GT (old headline) | 0.371 | 0.228 |
| trained on SAM3 GT, evaled on OFFICIAL GT (cross-eval) | 0.285 | **0.117** |
| trained on OFFICIAL GT, evaled on OFFICIAL GT (baseline) | **0.367** | **0.199** |

About half the old honest-AP50 headline did not survive clean GT — the SAM3-trained model had
fit SAM3's duplicate/label idiosyncrasies. Retraining on official GT recovered most of it.
Migration record: `docs/old/OFFICIAL_GT_MIGRATION_PLAN.md`.

Differences vs the SAM3 GT: masks written **sparsely** (a missing PNG means "not visible in this
frame"; the loader skips it); stuff classes keep official per-segment ids (`wall_0..wall_k`, not
forced to a single `_0`); out-of-taxonomy objects (including `otherfurniture`) become background.

## 2. The tars on group storage

**Datasets ship ONLY as tars** under
`/cluster/work/igp_psr/niacobone/distillation/dataset/scannet/`. There is deliberately **no
unpacked `scans/` tree on `work`** (deleted 2026-07: reading thousands of small PNGs off `work`
is slow and pressures the inode quota).

All three share the layout `scans/<scene>/raw_data/{subset,masks,masks_instance,_qa}` — each
scene with `subset/` (~100 stride-5 frames), `masks/<class>/` (per-class binary) and
`masks_instance/<class>_<k>/` (per-instance binary).

| Tar | Contents |
|---|---|
| **`scannet_official_gt_500.tar.zst`** — the **default** since 2026-07-09 | Official GT, **500 scenes (scene0000–0499), 7379 instances, 0 cross-class duplicates**. Scenes 0200–0499 built 2026-07-09 (`legacy/dataset_build/slurm/extend_dataset_500.sh`). 9 scenes (0240, 0243, 0269, 0292, 0354, 0366, 0438, 0456, 0483) have a **640×480** color camera; their GT is 640×480 too, so RGB↔GT stay consistent and the loader's resize to 518 handles the rest. |
| `scannet_official_gt_full.tar.zst` (2.3 GB) | The original 200-scene official-GT tar (scene0000–0199, 2950 instances). Kept for exact reproducibility of the runs trained on it. Instance count is below SAM3's ≈4195 because SAM3 double-counted duplicated objects. |
| `scannet_instance_dataset_full.tar.zst` (~2.6 GB) | The SAM3-generated GT (200 scenes, ≈4195 instances, with the duplicate defect above). Kept as the GT-quality baseline and as a project deliverable. Cross-frame identity from SAM3 video tracking; `wall`/`floor` forced single-instance; all-zero PNGs written where absent. |
| `scannet_3d_gt_val312.tar.zst` (1.2 GB) — **BUILT 2026-08-01**, docs/todo.md 1d | **3D benchmark GT** for the official val split (`data/splits/scannetv2_val.txt`, all 312 scenes): per scene `_vh_clean_2.ply` (the benchmark mesh) + `_vh_clean_2.0.010000.segs.json` (superpoints) + `.aggregation.json`. Layout `scans3d/<scene>/`, validated per scene (ply magic, segment-id closure). What `slurm/eval_3d_maskdino.sh` scores against (docs/MASKDINO.md §9). Built by `legacy/dataset_build/slurm/download_3d_gt_val312.sh` (job 9326394); a lost tar re-downloads in ~20 min. |
| `scannet_frames25k_val312.tar.zst` (1.1 GB) — **BUILT 2026-08-01**, docs/todo.md 1d | The val-312 slice of the official **`scannet_frames_25k`** export: per scene `color/*.jpg` + `pose/*.txt` (camera-to-world) + `depth/*.png` + intrinsics, 5 436 frames (~17/scene, sampled across the WHOLE scan — unlike the subset tars above, which cover only raw frames 0–495). Layout `scans25k/<scene>/`; **not** the `raw_data/{subset,masks,…}` layout, and carries no 2D GT. The 3D eval's input frames + its eval-only registration poses. Built by `legacy/dataset_build/slurm/download_frames25k_val312.sh` (job 9326395) from the one 6 GB `v2/tasks` zip. |
| `scannet_official_gt_1201.tar.zst` (29 GB) — **BUILT 2026-07-30**, docs/todo.md 1c | The full official ScanNet v2 train split (`data/splits/scannetv2_train.txt`, 1201 scan ids — includes `_01`/`_02`/... rescans, not a contiguous `_00` range). **1201 scenes, 17 638 instances, 0 cross-class duplicates, min label purity 1.0, no missing/failed scenes** (`OFFICIAL_GT_README_1201.md`; strips in `qa_strips_1201/`). Built by its own scripts (`extend_dataset_1201.sh` / `pack_official_gt_1201.sh`) into a **node-local** tree, so the 500-scene tar above is untouched *and* the scratch inode quota is not — see §5.1, the first attempt died on that quota. Not yet staged by any training script — point `DATA_TAR` at it, and mind `--tmp` (29 GB to stage, so `--tmp` well above the 500-scene tar's 24000) and the feature-cache RAM budget at 1201 scenes. |
| `scannet_official_gt_val312.tar.zst` (7.4 GB) — **BUILT 2026-08-01**, docs/todo.md 1c | The **val ruler for the official split** — same 2D GT and same `raw_data/{subset,masks,masks_instance}` layout as the 500/1201 tars, over the full official val list (`data/splits/scannetv2_val.txt`, 312 scan ids, `_01`/`_02` rescans included, **zero overlap with the 1201 train tar**). **312 scenes, 4630 instances, 0 cross-class duplicates, max cross-class IoU 0.0, min label purity 1.0, no missing/failed scenes** (`OFFICIAL_GT_README_val312.md`; strips in `qa_strips_val312/`). Archive entry count verified against source (347 439 png/jpg; 348 375 files incl. markers + `_qa/stats.json`). 8 scenes (0088_02, 0144_00, 0300_01, 0406_02, 0430_01, 0474_02, 0658_00, 0704_00) have a **640×480** color camera, handled exactly as in the 500-scene tar. Built by `extend_dataset_val312.sh` (job 9325618, 1 h 17, `ok=312 skip=0 fail=0` in both stages) → `pack_official_gt_val312.sh` (job 9328388, 2 min, chained via `CHAIN_PACK=1`), node-local per §5.1. Pair it with the 1201 tar for official-split training; nothing stages it yet. |

Specs on group storage: `OFFICIAL_GT_README.md`, `INSTANCE_MASKS_README.md` (+ `…_split2.md`).

### 2.1 ScanNet++ v2 — the competitor ruler (docs/todo.md 6c)

Under `/cluster/work/igp_psr/niacobone/distillation/dataset/scannetpp/`. **Evaluation only** —
ScanNet++ is the dataset all three direct competitors report on (SegVGGT zero-shot, FAST3DIS
zero-shot, IGGT; `docs/TRAINING_COMPARABILITY.md`), and we had no ScanNet++ numbers at all.
Licence: **ScanNet++ Terms of Use** (research-only, non-redistributable — these tars stay on
group storage).

The split is the official `nvs_sem_val.txt`, **50 scenes** (49 shipped — see below). Layouts mirror
`scannet_3d_gt_val312.tar.zst` / `scannet_frames25k_val312.tar.zst` exactly, so the evaluator's
loaders need no new file conventions.

| Tar | Contents |
|---|---|
| **`scannetpp_3d_gt_val50.tar.zst`** (1.9 GB) — **BUILT 2026-08-08**, jobs 10089394 / 10091616 | **3D benchmark GT.** Layout `scans3d/<scene>/` with `mesh.ply` (= `mesh_aligned_0.05.ply`, verbatim), `segments.json`, `segments_anno.json`, plus a per-scene `gt_stats.json` and one shared `scans3d/_metadata/` (`top100_instance.txt`, `map_benchmark.csv`, `semantic_classes.txt`, `nvs_sem_val.txt`). **49 scenes, 2585 instances** (median 43/scene, range 4–250 — scene `13c3e046d7` really does have only 4 objects inside the 84-class instance benchmark). Validated per scene: ply magic, `segIndices` length == ply vertex count, segment-id closure, every kept label in `top100_instance`, ≥1 instance. Meshes are ~1.1–3.4 M vertices, an order of magnitude denser than ScanNet's `_vh_clean_2`. |
| **`scannetpp_frames_val50.tar.zst`** (980 MB) — **BUILT 2026-08-08**, same jobs | **The input frames + eval-only registration poses.** Layout `scans25k/<scene>/` with `color/<stem>.jpg` (1920×1440), `depth/<stem>.png` (256×192 uint16 mm), `pose/<stem>.txt` (4×4 **camera-to-world**), `intrinsic/intrinsic_{color,depth}.txt`, `intrinsics_{color,depth}.txt` (byte-identical copies at the ScanNet `frames25k` path, so `train/scannet3d.py::load_frames25k_intrinsics` works unchanged) and `manifest.json`. **49 scenes × 50 frames = 2450 frames**, sampled uniformly over the WHOLE iphone sequence (3 378–26 021 frames/scene). No 2D GT. |

**49, not 50.** `d755b3d9d8` is excluded from **both** tars: its iphone trajectory diverges —
`aligned_pose` translations reach 7.2 km against a 5.3 × 4.0 × 3.6 m mesh, and only 143 of its
8 863 frames put the camera within 3 m of the mesh bounding box. The build's geometry self-check
caught it at 3.9 km and refused to ship it; nothing about it is recoverable by resampling. This is
an upstream defect, not a build defect. Both tars carry the same 49-scene list on purpose — a GT
scene with no frames is a landmine for the evaluator. `EXCLUDE=` on the SLURM script rebuilds all
50 and fails on that scene again. Across the other 49, unprojected sensor depth sits **1.1–7.9 cm
(median 1.9 cm)** from the mesh, and the RGB/pose index sweep peaked at offset 0 in **49/49**.

**Verified as shipped, not just as built** (job 10278989, `verify_scannetpp_tars.sh`): the two
tars were unpacked back off `work` and `scripts/verify_scannetpp_gt.py` re-ran every check on
**all 49 scenes — 49/49 pass**. Both tars hold the identical scene list. The build's own
`CHAIN_VERIFY` pass runs on the pre-tar tree, which is not the same thing: compression, the entry
count check and the copy to `work` sit between the two.


**Where it came from.** `/cluster/work/igp_psr/nedela/scannetpp_data`, read **once**, by the
build. That tree belongs to another user and can vanish; nothing downstream may read it, which is
why the class tables and the split file are copied into the GT tar's `_metadata/`.

**The four conventions that fail silently, and the evidence for each** (all reproducible with
`scripts/verify_scannetpp_gt.py`; the build re-checks 1–3 per scene and refuses to ship a scene
that fails):

| | convention | how it was established |
|---|---|---|
| 1 | **Pose = `aligned_pose`, camera-to-world.** `pose_intrinsic_imu.json` carries both `pose` and `aligned_pose`; `pose` is the raw ARKit trajectory. | Unprojected sensor depth lands **1.4–3.3 cm** (median, per scene) from `mesh_aligned_0.05.ply` under `aligned_pose` read as camera-to-world, **~85 m** away under `pose`, and **2.3–3.2 m** away under the world-to-camera reading. No other combination puts any geometry in the frustum. |
| 2 | **RGB: `cv2.VideoCapture` on `iphone/rgb.mkv`; decoded frame N is `frame_{N:06d}`.** No ffmpeg module exists on this cluster. | Frame count matches the pose json exactly, per scene (asserted). By content: the mesh rendered at pose N correlates with the video over a **±40-frame index sweep** and the mean NCC peaks at offset 0 in every scene (recorded in each `manifest.json["rgb_index_check"]`). |
| 3 | **Depth: `iphone/depth.bin` = per frame `<4-byte LE compressed size><LZ4 block>`, decompressing to 192×256 uint16 MILLIMETRES.** Not zlib, not float16. | Bit-identical to an independently prepared reference export of one scene for 5 frames spanning the sequence; the `/1000 → metres` scale is what makes check 1 land at centimetres (×0.1 gives 23–45 cm, ×10 gives 14–29 m). |
| 4 | **Labels: `segments_anno.json`'s raw label → `map_benchmark.csv`'s `instance_map_to` → filter to `top100_instance.txt` (84 classes).** | Skipping the map costs ~10 % of the instances (48 → 54 on the first val scene). Segment-id closure is asserted per object, exactly as `download_3d_gt.py::validate_scene` does for ScanNet. |

Two things to know before using this GT:

- **`segments.json` is not a superpoint over-segmentation here** — `segIndices` is the identity
  permutation, one segment per vertex. Any superpoint majority vote (`docs/MASKDINO.md` §9.1
  step 4) therefore degenerates to a per-vertex vote on ScanNet++; that is a property of the
  release, not of the build.
- **The colour intrinsic varies per frame** (iPhone autofocus, ~1.5 % of fx). The scene-level
  `intrinsic/intrinsic_*.txt` hold the median over the sampled frames; exact per-frame intrinsics
  are in `manifest.json["intrinsic_color_per_frame"]`.
- **The `segGroups` order decides overlaps.** A few upstream objects share segments; the later
  group wins, so a scene can end with slightly fewer distinct GT ids than kept objects (e.g.
  `25f3b7a318`: 54 kept, 51 distinct). The evaluator sees a consistent per-vertex GT either way
  — the gate below proves it — but do not read the 2585 instance total as "distinct GT ids".

### 2.2 Replica — the third competitor ruler (docs/todo.md 6c, built 2026-08-08, job 10100042)

Under `/cluster/work/igp_psr/niacobone/distillation/dataset/replica/`. **Evaluation only.**
FAST3DIS reports Replica zero-shot (`docs/TRAINING_COMPARABILITY.md`); we had no Replica numbers.
Licence: **CC-BY-NC-4.0** (`LICENSE.txt` beside the tars) — research-only, **not redistributable**.

The 8 scenes of the standard vMAP/iMAP set: `room_0-2`, `office_0-4`. Layouts mirror the ScanNet
tars, so the evaluator's loaders needed no new file conventions — only `train/replica3d.py`, which
documents each convention and the measurement that pinned it.

| Tar | Contents |
|---|---|
| **`replica_3d_gt_8.tar.zst`** (372 MB) | **3D benchmark GT**, `scans3d/<scene>/` with `mesh_semantic.ply` (the habitat instance mesh: per-**face** `object_id`, verified in the PLY header), `info_semantic.json` (`id_to_label[object_id]` → class), the plain `mesh.ply`, and Replica's own `preseg.json`/`preseg.bin`. From the official `facebookresearch/Replica-Dataset` release. |
| **`replica_frames_8.tar.zst`** (417 MB) | **Input frames + eval-only registration poses**, `scans25k/<scene>/` with `color/<stem>.png` (1200×680), `depth/<stem>.png` (uint16 **millimetres**), `pose/<stem>.txt` (4×4 **camera-to-world**, from `traj_w_c.txt`), `intrinsic/intrinsic_depth.txt` and `manifest.json`. **8 × 50 = 400 frames**, uniformly sampled over the iMAP trajectory. From `kxic/vMAP`'s `vmap.zip`. |

Four things to know before using this GT — all measured, none assumed:

- **The instance set is OUR construction.** Replica annotates the room shell as objects; the GT
  here drops `wall`, `floor`, `ceiling` and every object whose class id is not positive
  (unlabelled), which is the convention both the ScanNet benchmark (18 classes, no wall/floor)
  and ScanNet++'s `top100_instance.txt` follow. Per scene that leaves 28–73 instances out of
  52–94 objects. Any Replica number must carry this caveat.
- **The GT lives on faces, and 2.1–2.2 % of vertices are shared between objects.** A vertex takes
  the object of the plurality of its incident faces (ties to the lower id).
- **Depth is uint16 millimetres, not the NICE-SLAM 6553.5 constant.** Unprojected with
  `traj_w_c` as camera-to-world it lands **0.5–0.6 cm** from the mesh on every probe frame;
  ÷6553.5 lands 65–91 cm away. The same check validates the intrinsics, which are a documented
  **FALLBACK** (no camera-parameter file exists in the release; the build wrote the standard
  habitat values fx=fy=600, cx=599.5, cy=339.5 @ 1200×680 and flagged them `FALLBACK` per scene
  in `REPORT.json`). A wrong focal would not land at half a centimetre.
- **`preseg` is not used as the vote's over-segmentation.** It is a *planar* segmentation
  (468–802 segments/scene) whose vertex-weighted purity against the GT objects is only
  **0.77–0.95**, so segments straddle objects and would cap the achievable AP. The adapter uses
  identity superpoints, i.e. a per-vertex vote — the same situation as ScanNet++.

### 2.3 The licence gate — no dataset ships a number until it passes

`scripts/gate_3d_gt.py` (driver `slurm/gate_3d_gt.sh`) feeds a dataset's own GT back as
predictions and requires the official evaluator to answer exactly **1.000 / 1.000 / 1.000** —
the same gate that licensed the ScanNet evaluator (`docs/MASKDINO.md` §9.2) — plus the
pose/depth-scale geometry check above. Reports land beside the tars as `gate_<dataset>.json`.

```bash
sbatch --export=ALL,DATASET=replica slurm/gate_3d_gt.sh     # scannetv2|scannet200|scannetpp|replica
```

**Results, 2026-08-09 — all four pass**, on every scene of the real tars:

| dataset | scenes | evaluated GT instances | median vertices | sensor depth → mesh (scene median / worst scene) | GT-as-predictions |
|---|---|---|---|---|---|
| `scannetv2` | 312 | 4 364 | 146 k | 1.26 cm / 9.56 cm | 1.000 / 1.000 / 1.000 (both label settings) |
| `scannet200` | 312 | **10 045** | 146 k | 1.26 cm / 9.56 cm | 1.000 / 1.000 / 1.000 |
| `scannetpp` | 49 | 2 579 | 1 184 k | 1.42 cm / 5.94 cm | 1.000 / 1.000 / 1.000 |
| `replica` | 8 | 368 | 791 k | 0.55 cm / 0.58 cm | 1.000 / 1.000 / 1.000 |

The geometry check fails on a scene's **median** probe frame, not its worst: 3 of ScanNet's own
312 val scenes and 1 of ScanNet++'s 49 carry a single drifted probe (17–76 cm) while their medians
stay under 6 cm. Those are reported as outliers, not failures — a rule that fails the reference
dataset is the wrong rule.

### 2.4 The 2D **training** tars from InsScene-15K (docs/todo.md 6f) — see docs/MULTIDATASET.md

Everything above is ScanNet GT or 3D **evaluation** GT. There is a second family, built 2026-08-10
under `dataset/insscene2d/`, that supplies extra **2D training** supervision outside ScanNet's
taxonomy:

| Tar | Contents |
|---|---|
| `insscene2d_scannetpp.tar.zst` (1.28 GB) | **853** ScanNet++ scenes from IGGT's InsScene-15K mirror, re-encoded to `<scene>/{color/*.jpg, instance/*.png, manifest.json}` at 518. The 50 `nvs_sem_val` scenes are **excluded** — they contain all 49 scenes of the §2.1 eval column. |
| `insscene2d_infinigen.tar.zst` (2.14 GB) | **1466** Infinigen sub-scenes, same layout; the room shell (`wall`/`floor`/`ceiling`/`exterior`) dropped by name, since every evaluator counts those as false positives. |

Different layout, different purpose, **one class only**: these have no ScanNet taxonomy, so they
can only be trained with `--class_agnostic`. Provenance, the two exclusions, the id-globality
verification, the build and the runs are all in **`docs/MULTIDATASET.md`** — this entry exists so
that a reader of the tar list knows they are here and why they are not in the table above.

## 3. Mask conventions (both GTs)

- uint8 `{0, 255}` PNGs at the scene's color-camera resolution (1296×968; 640×480 for the 9
  scenes above).
- Filename = the subset frame's stem + `.png`.
- `<k>` in `masks_instance/<class>_<k>/` is zero-based per class, in order of first appearance.
- Directory names use underscores: `shower_curtain_3`.
- The union of a class's instance masks equals its `masks/<class>/` mask.
- The loader defaults to per-class `masks/`; `--instance_level` (legacy) / the default in the
  MaskDINO trainer (`--class_level` to opt out) reads `masks_instance/`.
- ScanNet class indices are `1..19`, with `0` = background everywhere. In official GT, NYU40
  classes outside the 19 trainable (including `otherfurniture`) are background — see
  `docs/MASKDINO.md` §4 for how the head guards against a GT tree that keeps a 20th class.

## 4. Data access at training time

Each job copies ONE tar to node-local scratch (`$TMPDIR`) and unpacks it there:

```
slurm/stage_dataset.sh   →  exports SCANNET_ROOT=$TMPDIR/scans
```

`scripts/train_maskdino.py` uses `SCANNET_ROOT` as its default `--scans_root` (via
`train/common.py::DEFAULT_SCANS_ROOT`).

Which tar is staged comes from the `DATA_TAR` env var. `stage_dataset.sh`'s own default is still
the SAM3 tar for backward compatibility, but **every training SLURM script overrides it to the
500-scene official tar** and requests `#SBATCH --tmp=24000` (MB — enough for the tar plus the
unpacked tree). `zstd` lives at `/usr/bin/zstd`.

To run against the SAM3 baseline instead:

```bash
sbatch --export=ALL,DATA_TAR=/cluster/work/igp_psr/niacobone/distillation/dataset/scannet/scannet_instance_dataset_full.tar.zst \
       slurm/train_maskdino.sh
```

**Scratch copies are not canonical.** The official-GT build tree also exists unpacked at
`/cluster/scratch/niacobone/scannet_official_build/scans`, but scratch is purgeable and this
tree is already partly hollow (most scenes' `subset/` dirs were emptied by a purge; scene0000_00
and scene0080_00 still have frames, which is enough for smoke tests). The old SAM3 scratch build
trees (`scannet_build*`) are fully hollow. Re-pack from the tar if a tree is ever needed — never
unpack-to-retar on scratch (inode quota).

### 4.1 Scene splits

Training/val scenes are chosen by *scene id*, not by a split file: the tar holds
`scene0000_00 … scene0499_00`, train = the first `N_SCENES` minus the val ones, val = scenes
**0080–0089** (project convention; docs/RESULTS.md §1.1 for why it stays that way).

`data/splits/scannetv2_val.txt` is the **official ScanNet v2 val list** (312 scenes), fetched
2026-07-28 from `ScanNet/ScanNet@master:Tasks/Benchmark/scannetv2_val.txt`. It is used only by
`VAL_SPLIT=official sbatch slurm/train_maskdino.sh`, which intersects it with our range
(`*_00`, id < `N_SCENES` → **77** scenes at N=490) and trains on the remaining 413. That is a
separate run on purpose: 74 of those 77 scenes are inside the normal training range, so an
existing checkpoint cannot be re-scored on them honestly. That intersection is a *workaround* for
the 500-scene tar; with `scannet_official_gt_val312.tar.zst` (§2) the same list is usable as-is,
all 312 scenes, no intersection.

`data/splits/scannetv2_train.txt` is the matching **official train list** (1201 scan ids),
fetched 2026-07-30 the same way. Unlike our own N-scene range it is not `_00`-only or
contiguous by scene number (386 scenes have a `_01` rescan, 215 a `_02`, ...; only 420 of our
existing 500 `scene0000_00..scene0499_00` happen to be official-train). It drives the 1201-scene
dataset build (docs/todo.md 1c) — no trainer reads it yet.

The two lists are **disjoint** (verified: 0 shared scan ids), so the 1201-scene and 312-scene
tars together are the full official train/val protocol — the one every competitor reports on.
Both were built by the same pipeline with the same QA gates, so a number measured across them is
not confounded by GT provenance.

## 5. Rebuilding the GT (only if a tar is lost or conventions change)

The builders are retired to `legacy/dataset_build/` because the tars are canonical and already
built. Run order:

```bash
sbatch legacy/dataset_build/slurm/download_official_gt.sh   # download+convert 200 scenes (resumable)
sbatch legacy/dataset_build/slurm/extend_dataset_500.sh     # scenes 0200–0499, streamed from .sens
sbatch legacy/dataset_build/slurm/pack_official_gt.sh       # QA gates (0 cross-class dups) + strips + tar → work
```

`extend_dataset_500.sh` streams each scene's `.sens` from the TUM server and extracts only the
stride-5 subset jpgs with early abort (frames 0–495 sit in the first ~10 % of the stream), so no
`.sens` is ever stored.

The 1201-scene official-train build (docs/todo.md 1c) reuses the same two Python scripts but
can't use their `[start, end]` scene-number range (the split isn't `_00`-only or contiguous), so
both gained a `--scene_list FILE` option (`--start`/`--end` become 0-based line indices into
FILE instead). It has its own SLURM entry points, build tree, and tar name so the 500-scene
dataset is never touched by it — and it builds **node-local**, for the reason in §5.1:

```bash
sbatch legacy/dataset_build/slurm/extend_dataset_1201.sh <list_start> <list_end>  # one per chunk
sbatch legacy/dataset_build/slurm/pack_official_gt_1201.sh                        # → the tar on work
```

The 312-scene official-**val** build is the same pipeline again, pointed at
`data/splits/scannetv2_val.txt`, with its own chunk dir
(`/cluster/scratch/niacobone/scannet_val312_chunks`), tar, README and strips dir — so neither the
500- nor the 1201-scene tar is touched. It is small enough for one chunk, so the extend job
chains the pack itself and the whole build is one command (~1 h 20 end to end):

```bash
sbatch --export=ALL,CHAIN_PACK=1 legacy/dataset_build/slurm/extend_dataset_val312.sh 0 311
sbatch legacy/dataset_build/slurm/pack_official_gt_val312.sh   # only if packing separately
```

Its `--tmp` is 40000, not the 1201 build's 120000: 312 scenes are ~8 GB of tree plus a ~7.5 GB
tar.

### 5.0 Rebuilding the ScanNet++ val-50 tars (§2.1)

One job, ~25 min, no download — the source is on `work`:

```bash
sbatch legacy/dataset_build/slurm/build_scannetpp_val50.sh
```

Two stages (`build_scannetpp_3d_gt.py`, then `build_scannetpp_frames.py`, both sharing
`scannetpp_common.py`), node-local in `$TMPDIR` per §5.1, **zero loose files on scratch** — and
unlike the ScanNet builds, not even a chunk tar: the deliverables themselves are the checkpoint.
While a stage is incomplete its tar is written as `…partial.tar.zst` and only renamed to the
deliverable name when all 50 scenes are done, so a partial build can never masquerade as the
finished dataset; re-running restores from whichever exists and skips `.complete` scenes.

`CHAIN_VERIFY=1` (default) runs `scripts/verify_scannetpp_gt.py` on the finished tree.
`tests/test_scannetpp_build.py` covers the same code on a synthetic scene, CPU-only.

### 5.1 The scratch inode quota — why the 1201 and val-312 builds never touch scratch

**`/cluster/scratch` is quota'd on FILE COUNT, not just bytes: 1.0 M soft / 1.5 M hard.** The
build tree is ~1046 files per scene (100 subset jpgs + ~950 per-instance/per-class mask pngs +
markers + `_qa/stats.json`), so:

| scenes | files | verdict |
|---|---|---|
| 312 (official val) | ~0.35 M | fits |
| 500 | ~0.52 M | fits |
| **1201** | **~1.26 M** | **over the soft limit; leaves no room for anything else** |

`extend_dataset_val312.sh` / `pack_official_gt_val312.sh` follow the same node-local discipline
even though 312 scenes would fit: the point is that a build should not spend inodes it doesn't
have to, and the 1201-scene tree already proved what happens when one does. Its build cost scratch
**1 inode** (one 7.5 GB chunk tar).

The first 1201-scene attempt (2026-07-30, jobs 9079912/14/15/17) materialised the tree on scratch
and died with `OSError(122, 'Disk quota exceeded')` after 1090 of 1201 scenes — and left the
account at 1 499 966 / 1 500 000 files, i.e. unable to write *any* new file anywhere on scratch.

The fix is not a bigger quota, it is to never keep loose files there. Scratch's **block** quota is
2.4 TB and barely used, so bytes are free while inodes are not, and **one tar is one inode**:

- `extend_dataset_1201.sh` builds the tree in `$TMPDIR` — node-local SSD, requested via
  `#SBATCH --tmp=120000` (nodes have 729 GB–1.8 TB; see `sinfo -o %d`), wiped when the job ends,
  not quota'd. Only one compressed **chunk tar** per range lands on scratch, in
  `/cluster/scratch/niacobone/scannet_1201_chunks/`.
- `pack_official_gt_1201.sh` unpacks those chunk tars back into `$TMPDIR`, runs the QA gates
  there, and copies only the finished tar to work.
- Net loose-file cost on scratch: **zero**. This is the same `$TMPDIR` discipline
  `slurm/stage_dataset.sh` already uses at training time, pushed back into the build.

Resumability survives the round trip, which is what makes this safe:

- Within a tree, `.subset_complete` / `.complete` markers skip finished scenes (unchanged).
- Across jobs, the chunk tar is restored at startup and rewritten at the end — **including after
  a partial failure**, which is the durability checkpoint. The download stage is bounded by
  `DL_BUDGET_H` (default 20 of the 24 h wall clock) so the job always reaches that rewrite; being
  killed at the wall clock would otherwise discard every scene converted in the run.
- An incomplete chunk exits non-zero *and resubmits itself*, up to `MAX_RESUBMITS` (default 5) —
  capped because a scene that is gone upstream can never reach `.complete` and would otherwise
  loop forever on the group's allocation. Because the resubmission is a new job id, a
  `--dependency=afterok` chain onto the original id would never fire, so the completing job
  submits the packing step itself when `CHAIN_PACK=1` (leave it 0 with several parallel chunks and
  submit the pack by hand). If scenes really are gone upstream, pack with `EXPECT_SCENES=<n>` —
  the QA gate counts scenes, so it must be told the real target.
- Before converting, scenes lacking `.complete` have `masks/ masks_instance/ _qa/` wiped:
  `build_official_masks.convert_scene()` only ever `mkdir(exist_ok=True)`-and-overwrites, so a
  conversion interrupted mid-write leaves residue it would not itself clean up. `subset/` is kept
  — that is the expensive `.sens` stream.

**One-off rescue.** `legacy/dataset_build/slurm/snapshot_build_1201.sh` folds an existing
on-scratch build tree into a chunk tar and deletes the tree, reclaiming the inodes. It is how the
failed attempt above was recovered rather than re-downloaded: all 1201 scenes already had their
`.sens` subsets extracted (~90 node-hours of streaming) and 1090 were fully converted, so only 111
scenes' GT zips remained. It verifies that *every regular file* — markers and `stats.json`
included, not just the png/jpg bulk — round-tripped before it removes anything.

## 6. Runs and checkpoints

- Output: `/cluster/work/igp_psr/niacobone/distillation/output/<run_name>/` (timestamped run
  names, e.g. `maskdino_sf_n490_20260727_161200`).
- `checkpoint_best.pth` = best val mIoU — the one to use for eval/demos.
  `checkpoint_best_ap50.pth` = the same run selected on AP50 instead.
- `metrics.jsonl` = one JSON line per eval (epoch, lr, losses, all metrics). Plots read this,
  not the logs.
- Checkpoints are **self-contained**: head weights + `head_config` + the cached scenes +
  optimizer/scheduler state (for `--resume`). The frozen backbone is reloaded from HF
  (`facebook/VGGT-1B`), never stored.
- SLURM job logs go to `slurm/logs/` (gitignored), not the repo root.
