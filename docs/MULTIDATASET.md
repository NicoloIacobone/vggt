# Multi-dataset training — ScanNet v2 + ScanNet++ + Infinigen (todo 6e + 6f)

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
| `processed_re10k` | `rgb/` + `cam/` only | **no** — no instance annotation of any kind, so it is skipped entirely (169 GB never read) |

Two properties were verified before any of it was used, because both are load-bearing:

1. **Ids are global per scene, not per frame.** ScanNet++ scene `00777c41d4`: two adjacent frames
   share **34 of 34** ids. Infinigen: ids are sparse scene-level indices (61, 69, 717, 766 in a
   42-object frame), not a per-frame 1..N relabelling. This is exactly what the multi-frame GT
   needs — it re-links instances across views by global id (CLAUDE.md).
2. **Infinigen's ids index `Objects/*.json`**, so every instance has a name. All 42 ids of a test
   frame resolved (`BedFactory(...)`, `bedroom_0/0.wall`). That is what makes the room-shell drop
   below principled rather than a heuristic.

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

`slurm/insscene_shards.py` reads the 211 GiB ScanNet++ archive **without unpacking it**: the parts
are a plain `split -b` of one zip, so the central directory sits at the tail of the last part and
any member can be reached by seeking across the concatenation. Concatenating would materialise
211 GiB to read ~0.3 % of it. zip64 is mandatory at this size and is handled explicitly.

Storage discipline per `docs/DATASET.md` §5.1: the tree is built in `$TMPDIR` (≈148 k files) and
only one tar per source lands on work — **scratch inode cost zero**.

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
# 1. the build (CPU, ~3 h; writes two tars to dataset/insscene2d/)
sbatch slurm/build_insscene2d.sh

# 2. the training — class-agnostic by construction, val stays the official ScanNet 312
sbatch slurm/train_maskdino_multi.sh                                   # all three sources
sbatch --export=ALL,SOURCES='scannet scannetpp' slurm/train_maskdino_multi.sh
sbatch --export=ALL,CAP_SCANNETPP=200,CAP_INFINIGEN=200 slurm/train_maskdino_multi.sh
DRY_RUN=1 bash slurm/train_maskdino_multi.sh                           # lists + schedule only

# CPU tests
myenv/bin/python tests/test_insscene2d.py        # the reader, the build's transforms
myenv/bin/python tests/test_class_agnostic.py    # 6e, both directions
myenv/bin/python tests/test_multidata2d.py       # the loader and the dispatcher
bash tests/test_train_maskdino_multi_sh.sh       # the driver's scene lists + the §7.1 regression
```

**Val never moves.** It is the official ScanNet v2 312-scene list in every mixture, scored
class-agnostic — otherwise "more data helped" and "the ruler got easier" are indistinguishable.

**The memory bound is the feature cache**, not the GPU: the trainer caches frozen VGGT features
for every scene up front, ~45 MB per 8-frame bundle, so ~54 GB for ScanNet's 1201 alone and
~160 GB for the full 3520-scene mixture. That is what the job's 16×16 GB request buys; `CAP_*`
shrinks it.

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
| the three data-scaling arms (§10) | launched 2026-08-21 |

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
scenes** and the 3520-scene mixture at **274 GB for 3832** — ≈ 71–89 MB per cached bundle including
GT, not the 45 MB of features alone. So override the script's default (16 × 16 GB = 256 GB, sized
for the 1201-scene arm) at submit time rather than editing it: `--cpus-per-task=26` (416 GB) for
the full mixture, 20 (320 GB) for ScanNet+ScanNet++, the default for ScanNet alone.

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
| (§6 reference: ScanNet only, 16 ep, no `--anchor_3d`) | 1201 | 19 216 | 16/16 | 0.641 / 0.656 | 0.536 / 0.505 | 0.509 | 0.692 | 6 h 26 |

**Reading 1 — adding real, same-domain ScanNet++ is free on this ruler; adding synthetic Infinigen
is not.** C − B = **+0.006** per-bundle AP50, *inside* the 0.009 seed spread (`docs/RESULTS.md`
§6.1), i.e. neutral, with `view_consistency` its best anywhere in this block. A − C = **−0.075**,
eight times the spread.

**Reading 2 — but only A failed to converge, so −0.075 is an upper bound on Infinigen's cost, not
a measurement of it.** B is flat over its last six epochs (0.534 … 0.544, peak 0.548 at epoch 25 of
35) and C nearly so (0.541 / 0.530 / 0.554 / 0.549 over 17–20); **A's best epoch is its last**, and
its curve is still climbing ~+0.010/epoch (0.454 → 0.469 → 0.479). At a matched *step* budget the
2.9×-larger mixture gets 2.9× fewer passes over any one scene, which is §9 reading 1 again one
level up. **Arm A-long (24 epochs, 84 480 steps, job 11498642) settles it** — and extending only A
is what convergence requires, not favouritism, precisely because B and C are already flat.

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

**Infinigen is the odd one out.** Arm A is below B and C on all six ScanNet/ScanNet200 cells and on
ScanNet++, but takes Replica's AP50 and AP25 — the one benchmark that is, like Infinigen, synthetic
renders. A had not converged, so its in-domain deficit is an upper bound; A-long settles it. Until
then the defensible multi-dataset setting is **C: ScanNet + ScanNet++, 2054 scenes**.
