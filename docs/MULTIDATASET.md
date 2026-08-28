# Multi-dataset training — ScanNet v2 + ScanNet++ + Infinigen + RE10K (todo 6e + 6f + 6j)

Opened 2026-08-10. This is the **data-scaling** workstream: every number in `docs/RESULTS.md` was
produced by a head trained on ScanNet v2 and nothing else, and the scaling curve (50 → 190 → 490 →
1201 scenes) was still rising when ScanNet ran out of scenes. This file is the home of what more
data required, what it cost, and what it bought.

> **Nothing here is comparable to a published number in `docs/RESULTS.md` §2/§3/§6.** Those are
> class-aware. Everything trained under this workstream is **class-agnostic** (one class,
> "object"), because the added datasets do not share ScanNet's taxonomy. The honest baseline for
> any run here is a **ScanNet-only run with `--class_agnostic`** — that is §6 below, mirrored as
> `docs/RESULTS.md` §6.2, and it is the *only* row over there a run here may be read against.

## 1. What the added data actually is — and what it is not

`docs/TRAINING_COMPARABILITY.md` §4 assumed ScanNet++ 2D supervision would have to be *rendered*
from the mesh. It does not: **InsScene-15K already ships per-frame instance annotations**, and the
mirror has been on work since 2026-08-08 (522 GB, `dataset/insscene15k/`). Read from the archives
themselves, not from the paper:

| subset | what it holds | usable as 2D supervision? |
|---|---|---|
| `processed_scannetpp_v2` | 903 scenes; `images/*.jpg`, `depth/*.png`, `refined_ins_ids/<stem>.jpg.npy` (int16 per-pixel ids at the image's own 920×690) | **yes** |
| `processed_infinigen` | 1466 sub-scene zips (156 scenes); `Image/`, `Depth/`, `ObjectSegmentation/*.npy` (int64 ids), `Objects/*.json` (names + `object_index`), `camview/*.npz` (K, T) | **yes** |
| `processed_re10k` | 5138 scenes; `<scene>/{rgb,cam}/` **and** a sibling `sam2_results/<scene>/auto_masks.{json,avi}` — COCO-RLE masklets, per frame, ids persistent across the clip (5127 of 5138 scenes) | **yes, but SAM2-generated** — §1.3, built §1.4. Every row trained on it says **SAM2-supervised** |

Two properties were verified before any of it was used, because both are load-bearing:

1. **Ids are global per scene, not per frame.** ScanNet++ scene `00777c41d4`: two adjacent frames
   share **34 of 34** ids. Infinigen: ids are sparse scene-level indices (61, 69, 717, 766 in a
   42-object frame), not a per-frame 1..N relabelling. This is exactly what the multi-frame GT
   needs — it re-links instances across views by global id (CLAUDE.md).
2. **Infinigen's ids index `Objects/*.json`**, so every instance has a name. All 42 ids of a test
   frame resolved (`BedFactory(...)`, `bedroom_0/0.wall`). That is what makes the room-shell drop
   below principled rather than a heuristic.

### 1.3 RE10K IS annotated — corrected 2026-08-24

The row above used to read *"`rgb/` + `cam/` only — no instance annotation of any kind"*. **That was
wrong**, and the error was structural: the original survey grouped member paths by their depth-2
component, which for `processed_re10k/<scene>/rgb/…` is `rgb`/`cam` — but the masks live under a
**sibling top-level directory**, `processed_re10k/sam2_results/<scene>/`, so they never appeared in
that histogram. Re-read from the split zip 2026-08-24 (1 221 783 members in 43 parts, no
unpacking):

| | measured |
|---|---|
| scenes with `rgb/` | 5138 |
| scenes with `sam2_results/auto_masks.json` | **5127** (all of them inside the 5138) |
| `auto_masks.json` schema | SA-V: `masklet[frame][obj]` = COCO RLE `{size, counts}`, plus `masklet_id`, `masklet_type`, `masklet_num`, `video_frame_count`, `video_height`, `video_width` |
| json size | median 1.73 MB uncompressed, p95 5.68, max 20.8; 2.19 GiB compressed in total |
| masklets per scene | median ~60, range 10–667 |
| frame counts | `len(rgb) == video_frame_count == len(masklet)` in every scene sampled, and asserted per scene at build time |

**`masklet_num` is NOT the number of masklets.** In every scene checked it equals
`video_frame_count`, i.e. the OUTER dimension. The outer index is the frame, the inner one the
masklet, and the inner length is constant per scene. The first scene inspected (218 frames,
60 masklets) happened to have `masklet_num == video_frame_count == 218` *and* was read correctly
by luck; scenes where the two differ settle it. Index by position on the outer axis, never on
`masklet_num`.

**Ids are persistent across the clip**, which is the property §1 verifies before using a source —
and it is verified, not assumed from the word "masklet". Constant inner length proves nothing on
its own, so the check is a *tracking* one: over 475 instance pairs in 10 random scenes, the IoU
between masklet *i* on adjacent frames is **median 0.932** against **0.475** for the best match to
any *other* index, and the same index is the best match **93.7 %** of the time. End to end through
the build, 53/60, 61/64, 68/68, 54/55 and 43/46 instance ids survive across four sampled frames of
the first scenes built.

**The one caveat that must travel with every number: the masks are SAM2 output, not ground truth.**
ScanNet++ and Infinigen ship human or engine GT; these are automatic and unnamed. That is a
different *kind* of supervision, not merely more of the same, so this source gets its own labelled
arm and is **never folded into A/A-long's row** — the same rule `docs/TRAINING_COMPARABILITY.md` §2
states for extra data, with one extra caveat nobody else's row carries.

Two further properties, neither of which blocks training but both of which shape how it is read:

1. **RE10K is video of *scenes*, not a 3D instance benchmark** — no mesh, no benchmark cloud, so
   it can never appear on the 3D ruler, only in training. It is also therefore **not one of the
   four benchmarks of `docs/RESULTS.md` §7**, so unlike ScanNet++ there is **nothing it can leak
   and no exclusion list to pass**. Stated explicitly so the next reader does not go looking.
2. It is the only source in the mirror with **clip-length temporal id persistence at 5127 scenes**,
   i.e. cheap `--multi_frame` identity supervision — which is the reason to want it at all.

### 1.4 Building RE10K — four decisions, three of them traps (2026-08-24)

`slurm/build_insscene2d.py --source re10k` follows `build_scannetpp` exactly — same split-zip
reader, same per-scene `remap_ids` onto a dense 1..G, same 518×518 squash, same `manifest.json`.
Four things are specific to it:

**1. COCO RLE without `pycocotools`.** It is not in `myenv`, and adding a C extension to the
critical path of a venv scratch has already destroyed twice (CLAUDE.md) is a bad trade for ~60
lines. `slurm/coco_rle.py` implements the format directly: LEB128-style base-32 run lengths,
column-major, **delta against the run two places back from the third run onwards**. Two of the
three easy bugs there (wrong delta start, row- vs column-major) survive a round trip through one's
own encoder, so it is tested against **`pycocotools` itself** — `tests/data/coco_rle_fixture.json`
was generated once under the reference env and carries the mask bits next to every `counts` string.
`tests/test_coco_rle.py`, 47 checks.

**2. Frame ↔ mask alignment is positional, and the stems are not sortable as strings.** `masklet`
is indexed by frame *position*; `rgb/` is keyed by a timestamp stem. Those stems are **8 or 9
digits** (307 821 vs 287 683 across the mirror), so a lexicographic sort puts every 9-digit stem
before every 8-digit one and **silently misaligns masks and images in 107 scenes**. They are sorted
by `int`. The build then *asserts* `len(rgb) == video_frame_count == len(masklet)` per scene and
skips-and-counts any scene where it does not hold, rather than guessing.

**3. Resolution is per scene, not global.** 360×640 dominates but 540×960, 506×960 and 1080×1920
all occur, and the RLE `size` matches the scene's own `video_height × video_width`. The build
checks the rgb PNG against them before resizing **both** to 518×518.

**4. Overlaps: the SMALLER instance wins.** SAM2 masklets are not a partition — a wall or floor
blob routinely contains the objects in front of it — while the instance-map format is one id per
pixel. Painting largest-first keeps every small object intact and costs the big blob only the
pixels it was occluding. Ties go to the lower masklet index. Both halves are deterministic, which
is what a per-scene id table requires.

**The room-shell filter, measured before it was chosen.** Infinigen needed one (walls/floors 21–32 %
of a frame) and ScanNet++ did not (largest median area 0.18–0.32). RE10K has no names, so the only
available rule is area — and the measurement, over 60 random scenes and 4638 instances:

| | measured |
|---|---|
| per-instance scene-wide area, percentiles | p50 **0.002**, p75 0.010, p90 0.039, p95 0.076, p99 0.223, max 0.758 |
| per-scene *median largest* instance (30 scenes) | median **0.247**, p25 0.185, p75 0.383, max 0.727 — i.e. inside ScanNet++'s unfiltered 0.18–0.32 band, but with a much worse tail |
| border contact by area bucket | <0.10 → 0.000 median, 0.10–0.30 → 0.134, **>0.30 → 0.223** (and 17 % of them touch more than half the frame border) |

So the big instances *are* shell-shaped — they run off the edge of the frame the way a wall does
and a chair does not — and the cut is placed at **0.30 of the frame, averaged over the kept frames
of the scene**:

| cap | instances dropped | labelled pixels dropped (median scene) | scenes emptied |
|---|---|---|---|
| 0.20 | 1.3 % | **21.8 %** | 0 |
| **0.30** | **0.5 %** | **0.0 %** | 0 |
| 0.40 | 0.3 % | 0.0 % | 0 |

The average is taken **scene-wide, not per frame**, deliberately: a per-frame threshold would make
an instance flicker in and out of the GT, and the multi-frame head re-links instances across views
by id, so a flickering id is worse than either keeping or dropping it outright. `--max_area_frac`
is the only knob.

> ⚠ **The cap does NOT make RE10K shell-free, and arm D must be read knowing that.** Looked at
> rather than counted — overlays of built scenes — SAM2 splits a wall, a ceiling or a floor into
> several sub-regions that each sit comfortably under 30 %, so **walls, floors and ceilings are
> still supervised instances in this source**. They are not in the other three (ScanNet's benchmark
> excludes wall/floor, Infinigen's shell is dropped **by name**, our Replica GT excludes the room
> shell — §1.1). Lowering the cap does not fix it, because **there is no knee to cut at**: pooled
> over 25 scenes, border contact — the thing that distinguishes a wall from a chair — rises
> *smoothly* with area (0.000 → 0.012 → 0.048 → 0.090 → 0.160 → 0.193 → 0.296 across the 0–2 %,
> 2–5 %, 5–10 %, 10–15 %, 15–20 %, 20–30 % and >30 % bands), and even above 30 % only **two thirds**
> of instances are shell-shaped, so a third of what any cap removes is a legitimate large object.
> Meanwhile the pixel mass is spread almost evenly across those bands (14 / 12 / 15 / 16 / 13 /
> 12 / 19 %), so a cap at 0.10 would delete **60 % of the labelled pixels** — that is not a shell
> filter, that is gutting the supervision.
>
> The honest statement: **SAM2 auto-masks are a class-agnostic over-segmentation of the whole
> image, not an object-vs-shell partition, and no cheap rule recovers one from them.** The 0.30 cap
> removes the frame-dominating blobs and claims nothing more. This remains a **second confound in
> arm D on top of the SAM2 one** — the arm adds both new scenes *and* shell supervision the other
> three sources do not have.
>
> **What this box predicted has since been tested, and the prediction was WRONG.** It said shell
> supervision would be "the first thing to suspect if arm D loses AP", and arm D then lost AP
> catastrophically — but the cause was **not** this (§11.3): the *same* data at the *same* dose
> trains normally once the learning rate is halved, which a label conflict could not do. Kept as
> written, with the correction attached, because the shell fact is still true and still
> unquantified — it is simply not what broke the first arm. If a *converged* arm D underperforms
> A-long, this becomes the live hypothesis again, and the follow-up is a border-contact rule rather
> than a tighter area one; both halves of the border statistic are computable from the RLE runs
> without decoding the mask.

### 1.1 The two exclusions

- **The 50 `nvs_sem_val` ScanNet++ scenes are dropped from training** — at BUILD time
  (`slurm/build_insscene2d.sh` passes `--exclude_scenes data/splits/scannetpp_nvs_sem_val.txt`;
  there is no such flag on the trainer, and none is needed: the tar simply does not contain them).
  The mirror contains **all 49 scenes of our ScanNet++ evaluation column** (`docs/RESULTS.md` §7);
  training on it unfiltered would leak the whole zero-shot benchmark. 903 − 50 = **853 training
  scenes**.

  **Re-verified against the shipped artefacts, 2026-08-12** — this is the invariant the whole §7
  zero-shot column rests on, so it is checked from the tars, not from the build's intent:
  `REPORT_scannetpp.json` lists 853 built scenes and 50 excluded; built ∩ `nvs_sem_val` = **0**;
  and the eval column's own 49 scenes (from `gate_scannetpp.json`) are **all 49 inside** the
  excluded 50. Re-run those three set operations after any rebuild of either side.
- **Infinigen's room shell is dropped by name** (`<room>/N.wall|floor|ceiling|exterior`, measured
  at 21 %, 17 % and 32 % of one frame). The ScanNet benchmark excludes wall/floor and our Replica
  GT excludes the room shell (`docs/DATASET.md` §2.2), so supervising them here would teach the
  head to emit masks that every evaluator counts as false positives.

ScanNet++'s own annotations needed **no** area filter: no instance dominates a frame (largest
median area 0.18–0.32 over the scenes checked), so there is no wall/floor blob to remove.

### 1.2 The added data is DENSER, not just more (measured 2026-08-12)

From the build reports, instances per scene: **ScanNet++ median 91** (p95 199, max 524),
**Infinigen median 38** (p95 56, max 633), against ScanNet's **~7 per frame / ~14 per bundle**.
Per 8-frame bundle the smoke saw 295 and 250 instance-frames for a ScanNet++ and an Infinigen
scene against ScanNet's 56. So the mixture changes **two** things at once — scene count and
supervision density — and a gain cannot be attributed to "more scenes" alone. Say "more data",
not "more scenes", until the per-source ablation (§8) separates them.

Two consequences checked rather than assumed:

- **10 scenes exceed the 300-query budget scene-wide** (7 ScanNet++, 3 Infinigen), though an
  8-frame bundle sees far fewer. If a bundle ever did, `HungarianMatcher` matches `min(Q, T)`
  pairs, leaves the surplus targets unmatched and neither crashes nor double-assigns a query
  (verified directly). It caps recall on those bundles; it does not corrupt training.
- **The class head is unaffected** — `--class_agnostic` means one class regardless of how many
  instances a frame holds.

## 2. The build

`slurm/build_insscene2d.py` (driver `slurm/build_insscene2d.sh`) is a **selection and
re-encoding** pass — nothing is rendered. Per scene it keeps `--frames` evenly spaced frames,
resizes to the trainer's 518×518 squash, remaps the ids **once per scene** onto a dense 1..G, and
writes:

    <source>/<scene>/color/<stem>.jpg        518×518 RGB
    <source>/<scene>/instance/<stem>.png     uint16 id map, 0 = background
    <source>/<scene>/manifest.json           frames, id table, provenance, counters

`slurm/insscene_shards.py` reads the 211 GiB ScanNet++ and 169 GiB RE10K archives **without
unpacking them**: the parts are a plain `split -b` of one zip, so the central directory sits at the
tail of the last part and any member can be reached by seeking across the concatenation.
Concatenating would materialise 211 GiB to read ~0.3 % of it. zip64 is mandatory at this size and
is handled explicitly.

Storage discipline per `docs/DATASET.md` §5.1: the tree is built in `$TMPDIR` (≈148 k files for
ScanNet++ + Infinigen, a further ≈333 k for RE10K) and only one tar per source lands on work —
**scratch inode cost zero**.

Measured cost per source: ScanNet++ + Infinigen together were **1 h 42** (job 10286143); RE10K is
**~2.6 s/scene × 5127 ≈ 3 h 45** and **~2.0 MB/scene ≈ 10 GB**, which is why
`slurm/build_insscene2d.sh` asks for 24 h rather than 12. `--source re10k` is **not** in the
script's default `SOURCES`: it is opt-in because its supervision is model-generated (§1.3).

## 3. `--class_agnostic` (todo 6e)

One rule, no second flag: **a one-class head means class-agnostic.**

- `scripts/train_maskdino.py --class_agnostic` builds the head with `num_classes=1`.
- `train/maskdino_data.py::build_frame_targets` sees `num_classes == 1` and (a) keeps every
  instance instead of dropping those whose class index the head cannot name, (b) collapses every
  label onto the single class.

Because `head_config` carries `num_classes`, a **checkpoint alone** decides how its GT is built,
and because every scorer (per-frame, per-bundle, full-res) takes its GT classes from those same
targets, the collapse reaches training and evaluation through one code path. Instances with an
invalid class index (< 1) are still dropped — that is corrupt GT, not a foreign taxonomy.

Default off. Every published number in this project stays class-aware.

## 4. The mixed loader

`data/instance_map_dataset.py` reads the instance-map layout and returns **the same sample dict**
as `data/scannet_overfit.py::ScanNetSingleSceneDataset`, so the feature cache, the target builder
and both evaluators consume the two interchangeably. `build_scene_dataset` dispatches **per
directory**: a list of paths may mix all three sources, and a pure-ScanNet list keeps its original
loader untouched, so no existing run changes shape.

`prepare_scenes` **refuses** a mixed scene list against a multi-class head rather than silently
supervising every ScanNet++/Infinigen object as ScanNet class 1.

## 5. Running it

```bash
# 1. the build (CPU, ~1 h 42; writes two tars to dataset/insscene2d/)
sbatch slurm/build_insscene2d.sh
sbatch --export=ALL,SOURCES=re10k slurm/build_insscene2d.sh            # +the SAM2 source, ~3 h 45

# 2. the training — class-agnostic by construction, val stays the official ScanNet 312
sbatch slurm/train_maskdino_multi.sh                                   # all three sources
sbatch --export=ALL,SOURCES='scannet scannetpp' slurm/train_maskdino_multi.sh
sbatch --export=ALL,CAP_SCANNETPP=200,CAP_INFINIGEN=200 slurm/train_maskdino_multi.sh
sbatch --export=ALL,SOURCES='scannet scannetpp infinigen re10k',CAP_RE10K=1500,EPOCHS=17,\
EXTRA_ARGS='--anchor_3d --learning_rate 5e-5' --cpus-per-task=26 \
    slurm/train_maskdino_multi.sh                                      # arm D-long, §11.4
DRY_RUN=1 bash slurm/train_maskdino_multi.sh                           # lists + schedule only

# CPU tests
myenv/bin/python tests/test_insscene2d.py        # the reader, the build's transforms
myenv/bin/python tests/test_coco_rle.py          # the RLE decoder, against pycocotools' output
myenv/bin/python tests/test_class_agnostic.py    # 6e, both directions
myenv/bin/python tests/test_multidata2d.py       # the loader and the dispatcher
bash tests/test_train_maskdino_multi_sh.sh       # the driver's scene lists + the §7.1 regression
```

**Val never moves.** It is the official ScanNet v2 312-scene list in every mixture, scored
class-agnostic — otherwise "more data helped" and "the ruler got easier" are indistinguishable.

**The memory bound is the feature cache**, not the GPU: the trainer caches frozen VGGT features
plus GT for every scene up front. Measured, not projected: **135 GB for 1513 scenes** and
**258 GiB for 3832** (arm A-long, job 11498642) — ≈ 69 MiB per cached bundle. The job's default
16 × 16 GB = 256 GB is sized for the 1201-scene arm; every larger mixture overrides
`--cpus-per-task` at submit time (§8 sizing), and `CAP_*` shrinks it.

**And `--learning_rate` for anything past 3520 scenes.** The default 1e-4 diverges on the
5020-scene four-source mixture; 5e-5 does not, at the same data (§11.3). It is a third knob the
submitter owns, alongside `--cpus-per-task` and `EPOCHS`.

**`EPOCHS` must be set by hand for any large mixture.** The driver's default is `20000/N_TRAIN`
clamped to [6, 40], which at ~5000 scenes returns the floor of 6 — badly under-budgeted. Reading a
step-matched deficit as "this data hurts" is the trap this workstream fell into twice (§9 reading
1, §10.3 reading 2).

## 6. The baseline — ScanNet-only, class-agnostic (job 10287578, 2026-08-10)

**The mixture's reference row, and the only training this workstream has produced so far.** Official
1201/312 split, `--class_agnostic --multi_frame --feature_mode bundle`, S=8, **b1** and
**jitter off** — the driver's defaults, and the two together are one fact: with
`--bundles_per_scene 1` only bundle 0 is ever cached and bundle 0 is *never* jittered
(`prepare_scenes`), so `--color_jitter` is inert at b1. 16 epochs = **19 216 steps**, 6 h 26. Also in `docs/RESULTS.md` §6.2 — that is its home for cross-referencing; the
reading is here.

| run | per-frame mIoU / AP50 | per-bundle mIoU / AP50 | `id_switch` ↓ | `view_consistency` ↑ |
|---|---|---|---|---|
| class-**aware** control, 12 ep (9386666) | 0.623 / 0.650 | 0.529 / 0.525 | 0.498 | 0.717 |
| **class-agnostic, 16 ep (10287578)** | 0.657 / 0.658 | 0.536 / **0.505** | 0.509 | 0.692 |

Best per-frame is epoch 13 (0.657 / 0.658), best bundle is epoch 16 (0.505).
`checkpoint_best_bundle.pth` is the one a mixture run must be compared against.

**Reading 1 — collapsing the taxonomy is nearly free on this ruler, and the −0.020 is an upper
bound, not a measurement.** The gap to the class-aware control is ~2× the 0.009 seed spread with
`id_switch` and `view_consistency` essentially unmoved, so the head was never leaning on the 18-way
class head for its instance separation — the premise the whole workstream rests on. But the row is
confounded **three** ways, not one: one class vs 18, 16 vs 12 epochs, **and b1 vs b2** — the control
ran `--bundles_per_scene 2 --color_jitter 0.2`, i.e. 28 824 steps over 16 views/scene against this
row's 19 216 over 8. Two of the three push this row *down*, so −0.020 is the most the collapse can
cost, and probably more than it does. Read it as "small", never as a measured Δ.

**Reading 2 — it had NOT converged, and that constrains every comparison built on it.**
`bundle_AP50` climbs monotonically to the last epoch: 0.487 → 0.495 → 0.496 → **0.505** at epochs
13–16. A mixture run must get the **same or a longer step budget**, or "more data helped" and
"more steps helped" cannot be separated — the same failure mode the fixed val split exists to
prevent.

## 7. Status

| step | state |
|---|---|
| shard survey + id-consistency verification | done 2026-08-10 |
| `slurm/insscene_shards.py`, `build_insscene2d.py` + 29 CPU checks | done |
| `--class_agnostic` (6e) + 13 CPU checks | done |
| `data/instance_map_dataset.py` + 29 CPU checks | done |
| the build itself (job 10286143) | **done 2026-08-10**, 1 h 42 — the two tars below |
| class-agnostic ScanNet-only baseline (job 10287578) | **done 2026-08-10** — §6 |
| end-to-end smoke run (job 10287385) | FAILED 2026-08-10 — §7.1, a driver bug, not the data |
| 〃 re-run after the §7.1 fix (job **10479399**) | **PASSED 2026-08-12** — see below |
| first uncapped mixture attempt (job 10480614) | FAILED 2026-08-12 — §7.2, the argv cap; lists right, `execve` too long |
| the full mixture run (job **10484000**) | **DONE 2026-08-12** after the §7.2 fix — §9 |
| the three data-scaling arms (§10) | launched 2026-08-21, **done 2026-08-22** — §10.3, §10.4 |
| C-long, the step-matched control for A-long | **FAILED** (job 11632049) — unstable run, §10.5; control still open |
| **C-long′** at lr 5e-5, the re-run | submitted 2026-08-26 (job 11831105) — §10.5; pairs with A-long′ |
| RE10K survey corrected: it IS annotated (6g) | done 2026-08-24 — §1.3 |
| `slurm/coco_rle.py` + `build_re10k` + 47 + 39 CPU checks | done 2026-08-24 — §1.4 |
| the RE10K build itself (job **11641723**) | **done 2026-08-24** — 9.7 GB, 5127 scenes, 0 failed — §11.1 |
| arm **D** at lr 1e-4 (job 11642516) | **DIVERGED 2026-08-25** — §11.2; cause isolated in §11.3 |
| **D-long** + **A-long′** at lr 5e-5 (11830140 / 11830142) | launched 2026-08-26 — §11.4 |

**What the smoke established** (18 scenes, 6 per source, 2 epochs, 10 min): the driver reaches the
end; staging unpacks all three sources; the dispatcher reports
`train scene sources: infinigen=6, scannet=6, scannetpp=6`; every scene caches with sane instance
counts (ScanNet 48–66 per bundle, ScanNet++ 75–295, Infinigen 57–250); the loss is finite and
falls (300.9 → 276.1, with `cls`, `mask+dice` and `box+giou` all sane); checkpoints and figures
are written. Its **metrics are 0.000 and that is expected** — 36 gradient steps from a random head
leaves every sigmoid under the 0.25 score threshold, so `pred/gt = 0.0/7.0`. A smoke proves the
plumbing, not the model.

Build output on work, `dataset/insscene2d/`: `insscene2d_scannetpp.tar.zst` (1.28 GB, 853 scenes)
and `insscene2d_infinigen.tar.zst` (2.14 GB, 1466 sub-scenes), with `REPORT_<source>.json`
alongside each. Scratch loose-file cost 0, as designed.

### 7.1 Why the smoke run died — `set -e` inherited through `stage_dataset.sh`

Job 10287385 (all three sources, `CAP_*=6`) exited after 2 minutes, having printed the ScanNet and
ScanNet++ scene counts and **not** Infinigen's. Nothing in the `.err` but the usual module banner.

`slurm/stage_dataset.sh` line 28 is `set -euo pipefail`, and `train_maskdino_multi.sh` **sources**
it — so the whole driver inherits `errexit` from that point on. The scene-list loop then contains

```bash
[ "$CAP" -gt 0 ] && LIST=$(echo "$LIST" | head -n "$CAP")
```

which is a pipeline whose left side is killed by **SIGPIPE** once `head` has its N lines and the
list exceeds the 64 KB pipe buffer. With `pipefail` the substitution returns 141; being the final
command of the `&&` list, it is *not* exempt from `errexit`, and the job dies silently. Reproduced
in isolation: exit 141, consistent with the `ExitCode 13:0` (SIGPIPE = 13) sacct recorded.

The threshold is the buffer, which is why it looked data-dependent: ScanNet++'s 853 paths (~47 KB)
fit and printed, Infinigen's 1466 (~88 KB) did not. It only fired when `CAP_*` was set — the
uncapped mixture never evaluates that line — so the smoke path was broken while the real run was
merely untested.

**FIXED 2026-08-12.** All three cap sites now use a here-string, `head -n "$CAP" <<< "$LIST"`,
which has no pipe and therefore no writer to kill. Two things came out of fixing it:

- **`SCANNET_ROOT` was expanded unguarded** in the two `sed` calls that build the paths, so the
  driver also died under `set -u` whenever it was unset. Now `${SCANNET_ROOT:-}` — identical in a
  real run, where staging always sets it.
- **`tests/test_train_maskdino_multi_sh.sh`** (16 checks) covers the list logic, and its
  regression check runs the driver under `bash -o errexit -o pipefail -o nounset`. That matters:
  **the first version of the test passed against the broken script**, because `DRY_RUN=1` skips
  the block that sources `stage_dataset.sh` and therefore never inherits `set -e`. A dry-run test
  of this driver cannot see a silent-abort bug unless it forces the options itself. Verified by
  reintroducing the pipe and watching the test go red.

### 7.2 The second scale-only failure — the argv cap (job 10480614)

The first uncapped mixture run got the scene lists *right* (1201 + 853 + 1466 = 3520 train, 312
val, epochs=6) and then died at `execve`:

```
slurm_script: line 149: myenv/bin/python: Argument list too long
```

`--train_scenes` took one comma-joined string of 3520 absolute paths ≈ **211 KB**. Linux caps a
**single** argv entry at `MAX_ARG_STRLEN` = 32 pages = **131 072 bytes**, independently of the much
larger total `ARG_MAX`, so no amount of `ulimit -s` helps. Cost: 5 minutes of compute, but **after
117 GB of staging** — the same shape as §7.1, a bug that only exists at full scale and that every
smaller run (the 1201-scene arm at ~66 KB, the 18-scene smoke at ~1 KB) sails past.

**Fixed:** `train/common.py::resolve_scene_dirs` accepts `@<file>` (one entry per line; commas
still work inside it, blank lines ignored, a missing file names itself), and the driver writes
`$RUN/{train,val}_scenes.txt` and passes those. The run directory now also *records* the exact
scene list, which is provenance a 3520-scene mixture wants anyway. Comma strings are unchanged, so
no existing command or test moves. Covered by
`tests/test_maskdino_train.py::test_scene_list_from_a_file` (incl. a 3600-path fixture asserted to
exceed the cap) and by two checks in `tests/test_train_maskdino_multi_sh.sh` — one of which
asserts the real uncapped list *is* past 131 072 bytes, so the test fails if the fixture ever
stops exercising the condition.

`slurm/train_maskdino.sh` still passes its lists as argv: at 1201 scenes it is ~66 KB, half the
cap. It is not broken, but it has the same ceiling — and unlike §7.1 it fails loudly.

## 8. Open, in order

- [x] Score a class-agnostic **ScanNet-only** run — done, §6.
- [x] Fix the `CAP_*` SIGPIPE — done 2026-08-12, §7.1.
- [x] **Run the mixture at a step-matched budget, then the 3D ruler on it — done 2026-08-12/13,
      §9. It LOST on the ScanNet ruler, and §9 says why that was the wrong budget.**
- [~] **The three data-scaling arms at ~42 k steps each — launched 2026-08-21, §10.** The
      per-source ablation (`SOURCES='scannet scannetpp'`: is the gain ScanNet++ the real
      same-domain data, or Infinigen the synthetic one?) is arm **C** of that set, so it is no
      longer a separate item.
- [ ] The cross-dataset 3D matrix (`docs/RESULTS.md` §7) on the arms — 4 benchmarks × 2 bridges,
      `CKPTS=<run dir> bash slurm/eval_3d_matrix.sh`. **That table, not the ScanNet val ruler, is
      what the extra data is for** (§9 reading 3).

**Sizing, once, for every arm** (measured, not projected): the feature cache is the binding
constraint and it is **not** the GPU. The 1201-scene baseline peaked at **135 GB RSS for 1513
scenes** and the 3520-scene mixture at **258 GiB for 3832** (job 11498642, `sacct MaxRSS`) —
≈ 69 MiB per cached bundle including GT, not the 45 MB of features alone. A linear fit through the
two points is **44 GB + 60 MB × scenes**. So override the script's default (16 × 16 GB = 256 GB,
sized for the 1201-scene arm) at submit time rather than editing it: `--cpus-per-task=26` (416 GB)
for the full mixture *and* for the ~5020-scene four-source arm D (≈360 GB, ~20 % headroom),
20 (320 GB) for ScanNet+ScanNet++, the default for ScanNet alone. All four sources **uncapped** is
8647 train scenes ≈ 640 GB, i.e. 40–44 CPUs, and may not schedule.

**The 3D ruler is ready for a one-class checkpoint** (done 2026-08-12):
`scripts/eval_3d_maskdino.py::label_setting` derives the label setting from the dataset **and**
`head_config`, so a `num_classes == 1` head is scored class-agnostic even on ScanNetv2, the
nyu40-keyed wall/floor prediction filter is skipped for it, and no 18-class table is written.
**Without it the failure would have been silent and total**: the head's single logit is read as
dataset class 1, `SCANNET_IDX_TO_NYU40[1]` is **wall**, and the wall/floor prediction filter drops
wall — so every query of every scene would have been discarded and the run would have reported
AP 0.000 / 0.000 / 0.000 with no error. Covered by
`tests/test_maskdino_eval3d.py::test_label_setting_takes_the_head_into_account`; 19-class runs are
bit-for-bit unchanged.

## 9. The first mixture, measured (job 10484000, 2026-08-12; 3D by job 10596569, 2026-08-13)

3520 train scenes (1201 + 853 + 1466), the §6 recipe unchanged except the schedule: **6 epochs =
21 120 steps** against the baseline's 19 216, i.e. the step-matched budget §8 asked for. Same val
(official ScanNet 312, class-agnostic), same S=8/b1, `--anchor_3d` **off**. 6 h 50 wall, 26 CPUs,
274 GB peak RSS, 0 failures.

| run | steps | per-frame mIoU / AP50 | per-bundle mIoU / AP50 | `id_switch` ↓ | `view_consistency` ↑ |
|---|---|---|---|---|---|
| ScanNet-only baseline, 16 ep (10287578) | 19 216 | 0.641 / **0.656** | 0.536 / **0.505** | **0.509** | **0.692** |
| **mixture 3520, 6 ep (10484000)** | 21 120 | 0.639 / 0.629 | 0.508 / 0.434 | 0.621 | 0.671 |

3D ruler, ScanNetv2 val-312, unposed, defaults, class-agnostic (the only setting a one-class head
has): **AP 0.008 / AP50 0.026 / AP25 0.240**, against the ScanNet-only class-**aware**
`--anchor_3d` checkpoint's 0.042 / 0.138 / 0.504 in the same setting (`docs/RESULTS.md` §7.2).

**Reading 1 — at matched steps the mixture loses on the ScanNet ruler, and that is arithmetic, not
a surprise.** Only 1201 of 3520 scenes are ScanNet, so 6 epochs give each ScanNet scene **6 passes
against the baseline's 16** — 7 206 ScanNet steps against 19 216, a **2.7× cut in exposure to the
domain the val set is drawn from**. Matching *total* steps therefore does not match the thing that
predicts the ScanNet val number. §8's "match STEPS, not epochs" is right about not confounding data
with compute and wrong about the budget being sufficient: both must rise.

**Reading 2 — it had not converged either.** `bundle_AP50` climbs 0.286 → 0.376 → **0.434** over
the last three epochs, steeper than the baseline was at *its* end. The 6-epoch schedule ran the
cosine to 5e-6 on a curve that was still moving.

**Reading 3 — the ScanNet val ruler is the wrong place to look for what this data buys.** The
mixture's whole purpose is the columns where a ScanNet-only head scores **0.000** — ScanNet++ and
Replica unposed (`docs/RESULTS.md` §7.2, finding 3). Nothing in this run addresses that, because
only the ScanNetv2 cell was ever scored. The next arms fix both halves: budget (§10) and evaluation
(§8, the matrix).

## 10. The data-scaling arms (launched 2026-08-21)

Three runs, **one recipe, one ruler, one budget**, differing only in the training mixture — the
shape `docs/TRAINING_COMPARABILITY.md` §2 says the field expects, and the first table in this
project able to say what multi-dataset training is worth.

| arm | train sources | scenes | epochs | steps | job |
|---|---|---|---|---|---|
| **A** | ScanNet + ScanNet++ + Infinigen | 3520 | 12 | 42 240 | 11435332 |
| **B** | ScanNet only | 1201 | 35 | 42 035 | 11435335 |
| **C** | ScanNet + ScanNet++ | 2054 | 20 | 41 080 | 11435338 |
| **A-long** | = A, run to convergence | 3520 | 24 | 84 480 | 11498642 |
| ~~**C-long**~~ | = C, step-matched to A-long | 2054 | 40 | 82 160 | 11632049 — **FAILED, §10.5** |
| **C-long′** | = C-long at **lr 5e-5** — the live control | 2054 | 40 | 82 160 | 11831105 (partner: A-long′ 11830142, §11.4) |

Recipe, identical across the three: `--class_agnostic --multi_frame --feature_mode bundle
--anchor_3d`, S=8, b1, 300 queries, lr 1e-4, warmup 2, val = official ScanNet 312.

**Why ~42 k steps and not the 21 k of §9.** Twice §9's budget, so arm A gets 12 passes over each
ScanNet scene where §9 gave 6, and arms B/C get the same *total* gradient steps as A rather than
the same epochs. Step-matching cannot flatter B: every arm keeps `checkpoint_best_bundle.pth`,
selected on val, so a longer schedule can only raise the number an arm reports.

**Why `--anchor_3d` on all three.** It is the strongest 3D mechanism measured anywhere in this
project (+67 % AP50, `docs/MASKDINO.md` §8.3) and the arms exist to produce 3D benchmark rows. It
is held **constant**, so it confounds nothing inside this block — but it does mean no arm here is a
one-flag comparison against §6 or §9, both of which ran without it.

**A ⇄ B is the data claim; A ⇄ C and C ⇄ B split it by source** — C ⇄ B adds real, same-domain
ScanNet++, A ⇄ C adds synthetic Infinigen. Read them only after the matrix (§8), never off the
ScanNet val ruler alone (§9 reading 3).

### 10.1 What the three arms are gated on, and what they chain into

**Gate: smoke 11434972** (18 scenes, 6 per source, 2 epochs, `--anchor_3d`). `--anchor_3d` had
never been run against the instance-map loader — it runs VGGT's frozen **point head** over the
cached bundle, which is dataset-agnostic by construction but had no evidence behind it. The smoke
cached all three sources with it, the loss fell (257.9 → 236.6 over 4 steps) and the checkpoint
carries `anchor_3d: True`, `num_classes: 1` and the Δ(xyz, log r) head. The three arms were
submitted `--dependency=afterok` on it, so a failure would have cancelled them instead of burning
~38 GPU-hours.

**Chain: `slurm/chain_eval3d_matrix.sh`** submits the 4 datasets × 2 bridges of `docs/RESULTS.md`
§7 for an arm the moment it finishes. It exists because the run directory carries a timestamp
minted when the training job *starts*, so the matrix cannot be named at submit time; the chain
reads it back out of the training log's `scene lists written to …` line. One chain per arm
(11436321 / 11436323 / 11436324), `afterok`, 24 cells in total.

**Those three chain jobs FAILED in 10 s and the matrix was submitted by hand instead** — a bug in
the chain script, not in the arms: it opened with `cd "$(dirname "$0")/.."`, but **SLURM spools the
batch script**, so inside a job `$0` is `…/slurm_script` and that cd lands nowhere near the repo
("no log for job … under slurm/logs"). Fixed to the hardcoded `REPO`, like every other SLURM driver
here. The DRY_RUN test could not see it because it ran *from* the repo, where the wrong cd is
harmless; `tests/test_eval_3d_matrix_sh.sh` now runs a **copy** of the script from a foreign cwd,
which is the shape SLURM gives it — verified red against the old line.

### 10.2 Two fixes this arm forced, both already in

- **The optimizer is now built BEFORE the feature cache** (`scripts/train_maskdino.py`). Caching
  the 3520-scene mixture takes ~3 h and the matcher/criterion/optimizer/scheduler used to be
  constructed after it, so a typo in any of them cost the whole pass — how job 9901119 died
  (`docs/todo.md` 2f, which asked for exactly this). Nothing in that block reads the cache.
- **`slurm/eval_3d_matrix.sh` takes a run directory**, not only its three hard-coded keys, so an
  arm that postdates the file can be scored without editing it
  (`bash tests/test_eval_3d_matrix_sh.sh`, DRY_RUN, 15 checks incl. the chain).

**One submission trap checked rather than assumed.** `--export=ALL,SOURCES='scannet scannetpp
infinigen',…` survives intact: Slurm splits that list on **commas**, not whitespace, and the shell
quoting keeps it one argv word. (The whitespace warning in `slurm/eval_3d_matrix.sh` is about the
*unquoted* form, where the shell — not sbatch — splits the word and the remainder is read as the
script name.) Verified with a 2-minute probe job before the arms were left to run.

### 10.3 The 2D result (all three done 2026-08-22)

Official ScanNet 312 val, class-agnostic, `checkpoint_best_bundle.pth` (selected on val
`bundle_AP50`). Same ruler for all three — only the training mixture moves.

| arm | scenes | steps | best ep | per-frame mIoU / AP50 | per-bundle mIoU / AP50 | `id_switch` ↓ | `view_consistency` ↑ | wall |
|---|---|---|---|---|---|---|---|---|
| **B** ScanNet only | 1201 | 42 035 | 25/35 | 0.654 / 0.675 | 0.557 / 0.548 | **0.441** | 0.707 | 13 h 16 |
| **C** + ScanNet++ | 2054 | 41 080 | 19/20 | **0.659 / 0.677** | **0.568 / 0.554** | 0.472 | **0.714** | 11 h 34 |
| **A** + Infinigen | 3520 | 42 240 | **12/12** | 0.630 / 0.628 | 0.521 / 0.479 | 0.531 | 0.693 | 9 h 50 |
| **A-long** = A converged | 3520 | 84 480 | 20/24 | **0.676 / 0.704** | **0.600 / 0.604** | **0.414** | **0.734** | 18 h 47 |
| (§6 reference: ScanNet only, 16 ep, no `--anchor_3d`) | 1201 | 19 216 | 16/16 | 0.641 / 0.656 | 0.536 / 0.505 | 0.509 | 0.692 | 6 h 26 |

**Reading 1 — adding real, same-domain ScanNet++ is free on this ruler; adding synthetic Infinigen
is not.** C − B = **+0.006** per-bundle AP50, *inside* the 0.009 seed spread (`docs/RESULTS.md`
§6.1), i.e. neutral, with `view_consistency` its best anywhere in this block. A − C = **−0.075**,
eight times the spread.

**Reading 2 — only A failed to converge, so −0.075 was an upper bound on Infinigen's cost. A-long
settled it, and the sign FLIPS.** B was flat over its last ten epochs (peak 0.548 at 25 of 35) and
C nearly so (0.541 / 0.530 / 0.554 / 0.549 over 17–20); **A's best epoch was its last**, still
climbing ~+0.010/epoch. Doubling its budget to 84 480 steps makes the full 3520-scene mixture the
**best run of the block on every 2D axis** — per-bundle AP50 0.479 → **0.604** (+0.056 over C, 6×
the seed spread), per-frame **0.704**, the best `id_switch` (0.414) and `view_consistency` (0.734)
measured anywhere — and A-long is itself converged (0.604 / 0.589 / 0.579 / 0.592 / 0.581 over its
last five). **The larger the mixture, the more steps it needs before it pays**; reading a
step-matched deficit as "this data hurts" is the trap, and this workstream fell into it twice
(§9 reading 1, then here).

**Reading 2b — what A-long still owes, and why it is still owed.** It had 2× the steps of B and C,
so "more data" and "more compute" are not separated at the top end. B is saturated (flat over ten
epochs, so steps cannot rescue it) and C nearly so, but the clean measurement is **C-long** (2054
scenes, 82 160 steps): A-long ⇄ C-long would be step-matched with both converged, isolating exactly
what Infinigen contributes. **Job 11632049 was that run and it failed — an unstable optimisation,
not a slow one (§10.5). The control is still open and nothing here is settled.**

**Reading 3 — `--anchor_3d` plus a real step budget is worth +0.043 to the ScanNet-only
class-agnostic row**, 0.505 (§6) → 0.548, with `id_switch` 0.509 → 0.441. Two variables at once,
so not attributable; recorded because §6's row is what MULTIDATASET rows used to be read against
and B now replaces it as the control.

**Reading 4 — none of this is the deliverable.** Every row above is scored on ScanNet, which is
in-domain for all three arms. What the extra data was bought for is the cross-dataset matrix, where
a ScanNet-only head scores 0.000 unposed on ScanNet++ and Replica — §10.4.

### 10.4 The cross-dataset matrix on the arms — the answer (2026-08-22)

24 cells, 0 failed scenes, jobs 11498511–11498543. **The table lives in `docs/RESULTS.md` §7.5**;
what it means for this workstream is here. A-long's own matrix is chained to job 11498642.

**More data buys exactly what the ScanNet ruler could not see.** §10.3 called C ⇄ B a tie
(+0.006 per-bundle AP50, inside the seed spread). On the benchmarks the mixture exists for, the
same pair is **+59 % AP50 on ScanNet++** (0.043 vs 0.027) and **+70 % on Replica** (0.080 vs
0.047) under the posed bridge, while staying flat in domain. So the workstream's premise holds —
but only under one of the two bridges, and the ScanNet val ruler was actively misleading about it.
**Score a data arm on the matrix, never on the val ruler.**

**And it says where the ceiling is.** Every out-of-domain **unposed** cell is still 0.000 for every
arm. The registration diagnostics are identical across the three arms to three decimals (ICP
inliers 0.963 / 0.924 / 0.660, camera RMS 0.097 / 0.116 / 0.143 m on ScanNetv2 / ScanNet++ /
Replica) because they depend only on **VGGT's frozen cameras**, which head-only training cannot
touch. The 2D masks improved measurably and the unposed number did not leave zero.
**Out of domain the binding constraint is the frozen backbone's geometry, not the decoder and not
the training data** — which is `docs/todo.md` §5's lifting workstream, now with a much sharper
statement of what it must fix and evidence that no amount of supervision substitutes for it.

**And "Infinigen is the odd one out" did not survive A-long.** At 42 k steps arm A trailed B and C
on every ScanNet/ScanNet200 cell and on ScanNet++. At 84 k the same mixture **takes all eight of
its cells**: unposed ScanNetv2 0.057 / 0.166 / 0.516 (+29 % AP50 over C), posed
0.177 / 0.389 / 0.708 (+19 %), and **2.5× B's zero-shot AP50 on both** ScanNet++ (0.068 vs 0.027)
and Replica (0.119 vs 0.047). It also beats the project's published ScanNet-only `--anchor_3d` row
(0.042 / 0.138 / 0.504) by +20 % AP50 on the same ruler, which makes it the best 3D row anywhere in
this project — recorded in `docs/RESULTS.md` §8.2 as a **separately labelled extra-data row**, per
the field norm in `docs/TRAINING_COMPARABILITY.md` §2, not folded into the headline.

**How the two winners pay is not the same mechanism.** B → C buys **coverage** (posed
annotated-assigned 0.657 → 0.766 on ScanNet++, 0.671 → 0.739 on Replica). A-long buys **mask
quality**: its coverage is *lower* than C's out of domain (0.716 / 0.681) and its AP far higher.

Pending a **valid** C-long (11632049 failed — §10.5), quote A-long for the scaling claim, keep the
ScanNet-only row as the headline, and keep the compute/data confound stated rather than dropped.

### 10.5 C-long FAILED — an unstable run, not a slow one (job 11632049, 2026-08-25)

The step-matched control §10.3 reading 2b asks for. It completed (18 h 01, 40/40 epochs, all 8
matrix cells green, 0 failed scenes) and **must not be used**: the optimisation destabilised and
never recovered, so the checkpoint prices nothing.

**It is worse than arm C on identical data with twice the steps — including on TRAIN loss**, which
rules out val noise, overfitting and best-epoch selection:

| | arm C (11435338) | C-long (11632049) |
|---|---|---|
| data / seed / recipe | 2054 scenes, seed 0 | **identical** (51 config keys equal; only `num_epochs` 20 → 40) |
| epochs / steps | 20 / 41 080 | 40 / 82 160 |
| epochs where train loss ROSE | 1 (+0.9) | **16, total +45.2, worst +11.4 @ ep 6** |
| final train loss | **93.2** | 122.6 — worse than arm C at *epoch 5* |
| final train `bundle_AP50` | **0.485** | 0.349 |
| val `bundle_AP50` (best ckpt) | **0.554** | 0.416, **best epoch = last → a lower bound** |

**What is excluded as the cause.** Data: the train lists are identical by scene name (2054), val is
the same 312, 0 leakage into either. Config: identical but for the schedule length. Head:
`head_config` identical to arm C and A-long. **Code: commit `9da8dfe` (08-24 12:17, 70 min before
the run) moved the matcher/criterion/optimizer construction ahead of the feature cache — measured
RNG-inert** (the moved constructors draw zero random numbers, so the training stream is unchanged)
and it touches no data or hyperparameter. **Schedule length alone: arm B ran 35 epochs with ZERO
loss rises.**

**What it is: the learning rate — the same failure §11.3 isolated independently.** Arm D diverged
at 1e-4 with the identical signature (train loss rising shortly after warmup, `train_AP50`
collapsing) and **halving to 5e-5 removed it completely at the same data**. The trigger there was
*dose* — the fraction of dense frames per batch; here it is **exposure**: a 40-epoch cosine holds
the LR near 1e-4 for twice as long as arm C's 20-epoch one (at epoch 7, 9.60e-5 against 8.30e-5),
so C-long takes roughly double arm C's dose at high LR before the schedule pulls it down. Same
operative knob, different trigger. This also explains why arm B's 35 epochs were clean: 1201
ScanNet-only scenes are the sparsest mixture in the block.
The node (`eu-ts-02`, AMD EPYC / A100-80GB, against `eu-a65-0x` Intel / A100-40GB for B/C/A-long)
remains the only *environmental* difference, but arm A ran on `eu-ts-02` without failing, so it is
at most a contributing factor.

**Consequence.** Infinigen's contribution at convergence is **not measured**. A-long ⇄ C-long as it
stands compares a healthy run to a broken one and would attribute the whole +0.188 per-bundle AP50
to data — the §9 reading 1 / §10.3 reading 2 trap in a new costume.

**The re-run: C-long′, job 11831105** (2054 scenes, 40 epochs = 82 160 steps, **lr 5e-5**, seed 0,
`--anchor_3d`, matrix chained as 11831106). Its partner is **A-long′** (job 11830142, §11.4), which
§11.4 launched at 5e-5 for arm D's sake and which serves here unchanged: **A-long′ ⇄ C-long′ is
step-matched (84 480 vs 82 160), same-LR and one-variable.** That is what the driver's rule — "re-run
the CONTROL at the same LR, or the comparison moves two variables at once" — requires, and it costs
one extra job rather than two because A-long′ already exists.

## 11. Arm D — RE10K as a fourth source, SAM2-supervised (todo 6j, 2026-08-24/26)

**A separate, separately-labelled arm, and it must stay that way.** Arms A/B/C differ only in *how
much* human/engine ground truth they see. Arm D changes the **kind** of supervision: RE10K's masks
are SAM2 output, so a gain here is "model-generated pseudo-labels at scale help" — a different
claim, and a weaker one, than "more annotated data helps". Never fold it into A/A-long's row, and
write **SAM2-supervised** on every row it produces (`docs/TRAINING_COMPARABILITY.md` §2, and §1.3
above for why this row carries one caveat more than the field norm requires).

### 11.1 The build (job 11641723, done 2026-08-24)

`insscene2d_re10k.tar.zst`, **9.7 GB**, on work beside the other two. 4 h 42 wall, 8 CPUs, ~11 GB
of `$TMPDIR`, **scratch inode cost zero**.

| | measured over all 5127 scenes |
|---|---|
| scenes built / failed | **5127 / 0** — the `len(rgb) == video_frame_count == len(masklet)` assertion held everywhere |
| `None` masklet entries | **0** in the entire dataset |
| frames | 158 903 (median 32; 426 scenes have fewer than 32 to give) |
| instances | **370 562**, median **61**/scene, p95 157, max 681 |
| dropped by the 0.30 cap + `min_area_px` | 6914 of 377 476 masklets = **1.83 %** |

**RE10K is the densest source in the mixture**, which §1.2's argument did not anticipate: the smoke
cached 223–476 instance-frames per 8-frame bundle for RE10K against ScanNet++'s 75–295,
Infinigen's 57–250 and ScanNet's 48–66. Combined with 0.64–0.98 of every frame being labelled
foreground, an RE10K batch is a very different object from a ScanNet one — which is exactly what
§11.3 turns out to be about.

### 11.2 The first attempt DIVERGED — job 11642516 is a failed run, not a measurement

Recipe identical to A-long (`--class_agnostic --multi_frame --feature_mode bundle --anchor_3d`,
S=8, b1, 300 queries, **lr 1e-4**, warmup 2, val = official ScanNet 312), 5020 scenes × 17 epochs =
**85 340 steps** against A-long's 84 480. It ran to completion: 17 h 15, **352 GiB peak RSS** of the
416 requested at `--cpus-per-task=26`, 0 failures, no NaN, no warning, all 5020 + 312 scenes cached
with the right per-source counts. **The infrastructure was fine and the result is still garbage.**

| | ep1 | ep2 | ep3 | ep4 | ep5 | ep6 | … | last |
|---|---|---|---|---|---|---|---|---|
| **arm D** loss / val bundle AP50 | 146 / 0.120 | 132 / **0.136** | 132 / 0.120 | 148 / 0.107 | 157 / 0.072 | 158 / 0.009 | … | 161 / 0.061 |
| **A-long** loss / val bundle AP50 | 156 / 0.099 | 135 / 0.140 | 130 / 0.208 | 125 / 0.288 | 114 / 0.336 | 107 / 0.364 | … | 68 / 0.581 |

Best `bundle_AP50` **0.136 at epoch 2** (per-frame 0.334 / 0.256, per-bundle 0.291 / 0.136,
`id_switch` 0.663, `view_consistency` 0.485), against A-long's 0.604.

**Three readings, and the third is the one that matters.**

1. **It is a divergence, not the under-convergence §9/§10.3 warned about.** The two runs are
   indistinguishable for two epochs and then arm D turns over at exactly the epoch warmup ends and
   the LR first sits at its 1e-4 peak. Loss *rises* from 132 to 169 while the LR decays, and only
   creeps back as the cosine reaches 5e-6.
2. **`train_AP50` collapses too — 0.211 → 0.006 → 0.058.** The head fails on the data it is being
   fit to, so this cannot be read as "RE10K taught it something that hurts ScanNet val". Anything
   that only moved val would leave the training fit intact.
3. **Every one of the 8 3D cells collapses with it** (job 11642519's matrix, `docs/RESULTS.md`
   §7.5): unposed ScanNetv2 0.001 / 0.007 / 0.172 against A-long's 0.057 / 0.166 / 0.516, and
   *below even the ScanNet-only 1201-scene control*. A run this far below its own control on the
   ruler it trains on is not measuring its training data.

### 11.3 What actually broke it — two diagnostics, one variable each (2026-08-25)

Six epochs each, which is all a divergence that starts at epoch 3 needs. Both are read against
A-long's own first six epochs, not against each other.

| run | scenes | RE10K share | lr | ep6 loss / val bundle AP50 |
|---|---|---|---|---|
| **A-long** (reference trajectory) | 3520 | — | 1e-4 | 107 / 0.364 |
| **arm D** as run | 5020 | 30 % | 1e-4 | 158 / **0.009** |
| **D-lr** (11744294) — same data, half the LR | 5020 | 30 % | **5e-5** | **105 / 0.369** |
| **D-dose** (11744296) — same LR, 5× less RE10K | 3820 | 8 % | 1e-4 | 115 / 0.348 |

**The cause is the learning rate, not RE10K's supervision.** `D-lr` holds the mixture, the dose and
`--anchor_3d` fixed, halves the LR, and the collapse disappears completely: it tracks A-long epoch
for epoch (0.085 / 0.169 / 0.238 / 0.277 / 0.331 / **0.369** against 0.099 / 0.140 / 0.208 / 0.288 /
0.336 / 0.364) and is *marginally ahead* at epoch 6 on 43 % more scenes. **A label conflict cannot
be fixed by halving a learning rate**, so §1.4's shell hypothesis and §1.3's "SAM2 masks are a
different kind of supervision" are both **refuted as the cause of this failure** — they remain open
as questions about a converged arm's *quality*, but they did not break this run.

`D-dose` says the instability is dose-dependent at the original LR: at 8 % RE10K the same 1e-4
trains normally. That is consistent with the trigger being the fraction of batches that are dense,
near-fully-covered RE10K frames — but **the LR is the operative knob**, because it removes the
problem at the full dose.

**So arm D's number is 0.136-at-epoch-2 and it means nothing about RE10K.** It is recorded as a
diverged run so the next reader does not re-derive it, and it is never quoted as what this data is
worth.

### 11.4 The real arm, and why it needs a control it did not need before

Two runs, launched 2026-08-26, one variable between them:

| arm | train sources | scenes | lr | epochs | steps | job |
|---|---|---|---|---|---|---|
| **D-long** | ScanNet + ScanNet++ + Infinigen + RE10K@1500 | 5020 | 5e-5 | 17 | **85 340** | 11830140 |
| **A-long′** | ScanNet + ScanNet++ + Infinigen | 3520 | 5e-5 | 24 | **84 480** | 11830142 |

**A-long′ is not optional.** The published A-long ran at 1e-4 and arm D *cannot* run there at all,
so quoting D-long against it would compare two things that differ in both the data and the LR —
the two-variables-at-once flaw this file has flagged twice already (§6 reading 1, §10.3 reading 3).
A-long′ re-runs A-long's exact mixture at D-long's LR, so **D-long ⇄ A-long′ is one variable: the
RE10K data**, at matched steps and matched schedule. Both chain their own 4 × 2 matrix
(11830144 / 11830145).

Read the pair on the **matrix**, never on the ScanNet val ruler (§10.4). And when they land, A-long′
also prices the LR change itself against the published A-long — a free second read-out nobody
asked for and everyone will want.

### 11.5 What arm D can and cannot settle, once it converges

It is one variable against A-long′ *as a source*, so it measures what 1500 scenes of
SAM2-supervised video add on top of the best annotated mixture. It does **not** separate
"pseudo-labels help" from "1500 more scenes help" — that would need a fourth source of equal size
with real GT, which the mirror does not have. Say the former, never the latter.

And it still carries §1.4's second confound: RE10K adds new scenes **and** supervises the room
shell, which none of the other three sources does. That is now the *live* hypothesis if a
**converged** D-long underperforms A-long′ — it is no longer available as an explanation for the
first attempt, which the LR alone explains.

### 11.6 Status

| step | state |
|---|---|
| RE10K read + schema verified from the split zip | done 2026-08-24 — §1.3 |
| `slurm/coco_rle.py`, verified against `pycocotools` (47 checks) | done 2026-08-24 |
| `build_re10k` + the room-shell measurement (39 checks) | done 2026-08-24 — §1.4 |
| the driver takes a 4th source with **zero code change** (24 checks) | verified 2026-08-24 |
| `insscene2d_re10k.tar.zst` (job 11641723) | **done 2026-08-24** — 9.7 GB, 5127 scenes, 0 failed |
| smoke, 24 scenes / 2 epochs (job 11642515) | **passed 2026-08-24** — loss 252.6 → 224.1, all four sources cached |
| arm D at lr 1e-4 (job 11642516) + its matrix (11642519) | **DIVERGED 2026-08-25** — §11.2 |
| the two divergence diagnostics (11744294 / 11744296) | **done 2026-08-25** — §11.3, the LR is the cause |
| **D-long** (11830140) + **A-long′** (11830142), + matrices (11830144 / 11830145) | **both done 2026-08-27** — §11.7; the 3D matrices are scoring (11996431 ff.) |

### 11.7 The answer in 2D — RE10K COSTS in-domain accuracy at matched compute (2026-08-27)

Both arms finished: D-long 17 epochs, A-long′ 24. **The pair is compute-matched, not
epoch-matched, and that is deliberate** — the epoch counts were chosen to hold the gradient-step
budget constant, and they do to within 1 %:

| arm | train mix | scenes | epochs | **steps** |
|---|---|---|---|---|
| **A-long′** (11830142) | ScanNet 1201 + ScanNet++ 853 + Infinigen 1466 | 3520 | 24 | **84 480** |
| **D-long** (11830140) | 〃 **+ RE10K 1500** (SAM2-supervised) | 5020 | 17 | **85 340** |

Both at lr 5e-5, both `--class_agnostic`, and both validated on the **same** ScanNet val-312 — so
the only difference in the comparison is the 1500 RE10K scenes.

| | A-long′ (no RE10K) | D-long (+RE10K) | Δ |
|---|---|---|---|
| **per-bundle AP50** | **0.5753** | 0.5241 | **−0.051** |
| per-frame AP50 | **0.6821** | 0.6522 | −0.030 |
| `id_switch` (lower better) | **0.4035** | 0.4587 | +0.055 |
| `view_consistency` (higher better) | **0.7301** | 0.7169 | −0.013 |

**1. Every axis moves the wrong way, and the margin is real.** −0.051 per-bundle AP50 is ~5.7× the
measured 0.009 seed spread. This is not noise.

**2. Read it as DISPLACEMENT, not as "RE10K is bad data" — and §12.3 now proves that reading.**
At a fixed step budget, a fourth source buys its steps from the others: each ScanNet scene is seen
17 times instead of 24. The prediction that follows is that RE10K should *help* wherever ScanNet is
absent, and **it does** — see §12.3, where the same 1500 scenes are worth 1.8–2.1× on the same
ruler. So the claim is not "RE10K is worthless"; it is *"RE10K is redundant with ScanNet, and at
matched compute redundancy costs"*.

**3. The out-of-domain matrices came back, and they do not rescue it** (11996431 ff., all 16 cells,
0 failures). RE10K's case was always out-of-domain generalisation, and on this arm it fails there
too — every cell moves the wrong way (class-agnostic AP50):

| benchmark | bridge | A-long′ | D-long | Δ |
|---|---|---|---|---|
| ScanNetv2 | unposed | 0.155 | 0.090 | −0.064 |
| ScanNetv2 | posed | 0.360 | 0.264 | −0.096 |
| ScanNet200 | unposed | 0.115 | 0.087 | −0.028 |
| ScanNet200 | posed | 0.280 | 0.223 | −0.058 |
| ScanNet++ | posed | 0.059 | 0.045 | −0.014 (−24 % relative) |
| Replica | posed | 0.127 | 0.117 | −0.010 (−8 % relative) |

(The two unposed out-of-domain cells are 0.000–0.001 for both arms and carry no signal — zero-shot
dies under the unposed bridge whatever the training mixture, as every arm in this file shows.)
**So, in a mixture that already contains ScanNet, RE10K is negative everywhere.**

**4. The learning-rate diagnosis held.** D-long's best epoch is 15 of 17 with the loss falling
monotonically (83.3 → 82.5 → 82.0), against arm D's best epoch 2 of 17 with training AP50
collapsing to 0.006 (§11.2). Halving the LR removed the collapse exactly as §11.3 predicted, and
the run above prices a converged model rather than a broken one.

## 12. The ZERO-SHOT arms — matching what the competitors train on (todo 6l, 2026-08-26)

Everything in §9–§11 adds data **on top of ScanNet**. This section removes it. The reason is in
`docs/TRAINING_COMPARABILITY.md` §6.2: **FAST3DIS and IGGT never train on ScanNet and every arm
here does**, so every "we lead them" row in `docs/RESULTS.md` §8.2 is favourable to us on the
training axis before a single number is read. Two arms close that, and they also complete a 2 × 2.

| arm | train sources | scenes | epochs | steps | job | matrix |
|---|---|---|---|---|---|---|
| **I** | ScanNet++ + Infinigen + RE10K@1500 | 3819 | 22 | 84 018 | 11839134 | 11839151 |
| **I-gt** | ScanNet++ + Infinigen | 2319 | 36 | 83 484 | 11839135 | 11839152 |

Both `--class_agnostic --anchor_3d --learning_rate 5e-5`, seed 0, S=8, the §10 recipe otherwise
unchanged — so they are step- and schedule-matched to A-long′ (11830142) and D-long (11830140):

| | no RE10K | + RE10K@1500 |
|---|---|---|
| **+ ScanNet** | A-long′ (3520) | D-long (5020) |
| **no ScanNet** | **I-gt (2319)** | **I (3819)** |

Each edge is one variable. The row edges price **ScanNet training data**; the column edges price
**SAM2-supervised RE10K** (§1.3) — each measured twice, in and out of the ScanNet domain.

**Arm I is IGGT's training set minus ASE** (`docs/TRAINING_COMPARABILITY.md` §6.1a), which is the
closest replication the mirror allows and will stay so: ASE is 9.2 TB and its scene list is
unpublished (§5.1–5.2 there). RE10K is capped at 1500 of 5127 scenes because the feature cache is
the binding constraint — uncapped this arm is ~550 GB, which does not schedule. Say **"IGGT's
mixture minus ASE, RE10K subsampled"**, never "IGGT's training data".

**Two differences run in our favour and one against**, all to be stated wherever these rows appear:
we drop the 50 `nvs_sem_val` ScanNet++ scenes from training (§1.1) so our ScanNet++ column is
honest and IGGT's is not; our backbone stays **frozen** where IGGT finetunes VGGT; and IGGT spends
~16 GPU-days against our ~0.8.

### 12.1 The val ruler on an arm that never saw ScanNet

Val stays the official ScanNet v2 312 for every arm — that is the rule the whole file rests on
(§5) — so on these two it is a **zero-shot** read-out rather than an in-domain one. One driver
change was needed: `slurm/train_maskdino_multi.sh` used to set `VAL=""` whenever `scannet` was
absent from `SOURCES`, i.e. these arms would have trained with no val set at all. It now stages
the val-312 tar and builds the val list **independently of `SOURCES`**; 4 new checks in
`tests/test_train_maskdino_multi_sh.sh` pin it, and no existing arm's behaviour changes.

One caveat travels with the numbers: `checkpoint_best_bundle.pth` is *selected* on that zero-shot
ruler. **Measured 2026-08-27 on arm I-gt, this is worse than a caveat — the selection does not
work at all.** Its `val_bundle_AP50` peaks at **epoch 5 of 36** (0.077) while `train_AP50` is
still 0.19 and climbing to 0.24 by epoch 31; the val curve then wanders between 0.02 and 0.07 for
the rest of the run without ever beating epoch 5. A zero-shot ruler at that level is noise, and
selecting on noise is worse than not selecting: `checkpoint_best_bundle.pth` for these two arms is
an early, half-trained checkpoint. **Score the FINAL `checkpoint.pth`** — `CKPT_NAME` in
`slurm/chain_eval3d_matrix.sh` (added 2026-08-27, default unchanged, 3 new checks in
`tests/test_eval_3d_matrix_sh.sh`); jobs 11946406 (I-gt) / 11946413 (I) are chained for exactly
that. Report the final-epoch row as the headline for arms I and I-gt and say why.

**Second thing arm I-gt showed, and it is a warning for the next run.** Its loss curve has the
§10.5 excursion signature at **half** the LR that caused it there: train loss 157.7 → 121.5 by
epoch 5, then **rising to 136.9 by epoch 18** (13 of 30 epoch-to-epoch steps up), then recovering
to 119.3 by epoch 30 with `train_AP50` back to 0.24. It **recovered** — unlike arm D at 1e-4,
whose `train_AP50` went to 0.006 and stayed — so this is a rough run, not a failed one. But it is
the most *exposed* run in the block (36 epochs, the longest cosine, on the smallest mixture), which
is precisely §10.5's "exposure, not dose" trigger. If its matrix lands anomalously below arm I's,
re-run at 2.5e-5 before concluding anything about the data. Arm I (22 epochs, 3819 scenes) carries
far less exposure and is the arm the IGGT comparison actually rests on.

### 12.2 What these arms can and cannot settle

They make our ScanNetv2, ScanNet200 and Replica columns **genuinely zero-shot**, which is
FAST3DIS's setting on all three and IGGT's on ScanNet. They do **not** make ScanNet++ zero-shot —
arm I trains on 853 ScanNet++ scenes and is scored on 49 held-out ones, exactly as IGGT is (minus
the leak). And they cannot separate "no ScanNet" from "less data": I-gt is 2319 scenes against
A-long′'s 3520. That is what the 2 × 2's *other* edge is for — read the square, not one cell.

### 12.3 THE RESULT — the training-matched comparison, and RE10K's sign flip (2026-08-28)

Both arms and their full matrices landed (0 failed scenes anywhere). **All rows below are the
FINAL `checkpoint.pth`**, per §12.1 — `checkpoint_best_bundle` is selected on a zero-shot ruler
that does not work, and must not be used for these two arms.

Both arms are **compute-matched to within 0.6 %**: I-gt 2319 × 36 = 83 484 steps, I 3819 × 22 =
84 018.

#### The competitor-facing cell — ScanNetv2, unposed, class-agnostic

| row | trains on ScanNet? | AP / AP50 / AP25 |
|---|---|---|
| FAST3DIS (published) | no — ASE only | 0.038 / 0.096 / 0.316 |
| IGGT (via FAST3DIS) | no — InsScene-15K | 0.028 / 0.112 / 0.287 |
| **ours, arm I** — IGGT's mixture **minus ASE** | **no** | **0.005 / 0.023 / 0.251** |
| ours, arm I-gt — the same minus RE10K too | no | 0.003 / 0.013 / 0.212 |
| *ours, headline (17 views)* | *yes* | *0.042 / 0.138 / 0.504* |
| *ours, headline (50 views)* | *yes* | *0.053 / 0.170 / 0.542* |

**1. The training-data asymmetry was real, and it was carrying most of the lead.** Removing
ScanNet costs a factor of **6 in AP50** (0.138 → 0.023 at the same 17-view budget). Against the
published rows we go from leading to **~4× behind**. The deck's long-standing wording — *"not
training-matched, and the asymmetry FAVOURS us"* — was correct, and this prices it.

**2. What this does NOT establish is that our recipe is worse than theirs.** Arm I is missing
**ASE entirely** — FAST3DIS's *whole* training set and the largest component of IGGT's — because
its scene list is unpublished and it is 9.2 TB (§5). We are running **3819 scenes against their
~100 k**, with a frozen backbone against adapted ones, at ~0.8 GPU-days against ~16. The
supportable statement is *"we cannot match their training setting, and without ScanNet we are well
behind"*, **not** *"our method loses at equal data"* — that comparison has never been run and
cannot be, on this cluster.

**3. AP25 degrades far less than AP50** (0.504 → 0.251, a factor 2, against AP50's factor 6). Even
without ScanNet the model still finds and coarsely localises objects; what collapses is clearing
the 0.5-IoU bar. Same signature as everywhere else in this project.

#### RE10K's sign FLIPS on whether ScanNet is in the mixture

Two compute-matched pairs, same ruler, class-agnostic AP50 on ScanNetv2:

| pair | ScanNet in mixture? | unposed | posed |
|---|---|---|---|
| A-long′ → **D-long** (§11.7) | **yes** | 0.155 → 0.090 (**−42 %**) | 0.360 → 0.264 (**−27 %**) |
| I-gt → **I** | **no** | 0.013 → 0.023 (**1.8×**) | 0.029 → 0.063 (**2.1×**) |

And without ScanNet it helps on every other cell too: ScanNet200 1.6× unposed / 1.8× posed,
Replica 1.4× posed.

**This is the reading, and it is what the 2 × 2 was built to produce.** RE10K supplies real-world
visual diversity that **ScanNet already supplies, better and in-domain**. Where ScanNet is present
RE10K is redundant, and at a fixed step budget redundancy is not free — it displaces. Where
ScanNet is absent, RE10K is the best real-world proxy in the mixture and is worth roughly a
doubling. **Neither cell alone supports a claim about "what RE10K is worth"; the square does.**

⚠ It remains **SAM2-supervised** — those masks are model output, not ground truth — wherever it
appears, in either direction.

#### The full matrix, for the record (class-agnostic, final `checkpoint.pth`)

**8/8 cells on each arm, 0 failed scenes anywhere.** Two cells (`replica_gt`, `scannetpp_gt`) died
on their first attempt with `CUDA error: CUDA-capable device(s) is/are busy or unavailable` —
**node GPU contention, not code or data** — and were re-run as 12077651 / 12077653; the recovered
values match the rest of the matrix.

| benchmark | bridge | I-gt (no RE10K) | I (+RE10K) | ratio |
|---|---|---|---|---|
| ScanNetv2 | unposed | 0.003 / 0.013 / 0.212 | 0.005 / 0.023 / 0.251 | **1.8×** |
| ScanNetv2 | posed | 0.008 / 0.029 / 0.323 | 0.018 / 0.063 / 0.399 | **2.1×** |
| ScanNet200 | unposed | 0.005 / 0.021 / 0.183 | 0.009 / 0.033 / 0.223 | 1.6× |
| ScanNet200 | posed | 0.015 / 0.049 / 0.287 | 0.030 / 0.086 / 0.364 | 1.8× |
| ScanNet++ | unposed | 0.000 / 0.000 / 0.001 | 0.000 / 0.000 / 0.003 | no signal |
| ScanNet++ | posed | 0.000 / 0.001 / 0.096 | 0.001 / 0.006 / 0.128 | 6× *(on ~0)* |
| Replica | unposed | 0.000 / 0.000 / 0.005 | 0.000 / 0.001 / 0.006 | no signal |
| Replica | posed | 0.003 / 0.016 / 0.194 | 0.005 / 0.023 / 0.278 | 1.4× |

RE10K helps on **every cell that carries signal**. The unposed out-of-domain pair is 0.000 on both
arms, as it is on every arm in this file — that is the bridge, not the data. The ScanNet++ posed
ratio is arithmetically 6× but sits on 0.001 → 0.006 and should not be quoted as a multiple.
