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
1201/312 split, `--class_agnostic --multi_frame --feature_mode bundle`, S=8, b2, jitter 0.2, 16
epochs, 6 h 26. Also in `docs/RESULTS.md` §6.2 — that is its home for cross-referencing; the
reading is here.

| run | per-frame mIoU / AP50 | per-bundle mIoU / AP50 | `id_switch` ↓ | `view_consistency` ↑ |
|---|---|---|---|---|
| class-**aware** control, 12 ep (9386666) | 0.623 / 0.650 | 0.529 / 0.525 | 0.498 | 0.717 |
| **class-agnostic, 16 ep (10287578)** | 0.657 / 0.658 | 0.536 / **0.505** | 0.509 | 0.692 |

Best per-frame is epoch 13 (0.657 / 0.658), best bundle is epoch 16 (0.505).
`checkpoint_best_bundle.pth` is the one a mixture run must be compared against.

**Reading 1 — collapsing the taxonomy is nearly free on this ruler.** −0.020 per-bundle AP50
against the class-aware control, ~2× the 0.009 seed spread, with `id_switch` and
`view_consistency` essentially unmoved. The head was never leaning on the 18-way class head for
its instance separation, which is the premise the whole workstream rests on. Note the row is not
like-for-like (12 vs 16 epochs), so read it as "small", not as a measured Δ.

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
| the full mixture run (job **10484000**) | **LAUNCHED 2026-08-12** after the §7.2 fix |

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
- [ ] Run the mixture, then the 3D ruler (`docs/MASKDINO.md` §9) on it in both transfer modes.
      **Budget: match STEPS, not epochs.** The driver's auto schedule gives 6 epochs × 3520 =
      21 120 steps against the baseline's 16 × 1201 = 19 216 — already comparable, which is the
      point (§5, "hold the gradient-step budget near the 1201-scene runs'"). Reading §6.2's "it
      had not converged" as "give the mixture more epochs" would confound data with compute.
      **Memory is the binding constraint, not the GPU:** the baseline peaked at **135 GB RSS for
      1513 scenes** (≈ 89 MB/scene incl. GT, not the 45 MB of features alone), so 3520 + 312
      scenes projects to **≈ 340 GB** — past the script's 16 × 16 GB = 256 GB. Override at submit
      time (`sbatch --cpus-per-task=26 --mem-per-cpu=16384`, 416 GB) on an `eu-a65` node; do not
      change the script's default, which is sized for the 1201-scene arm.
      **The 3D ruler is ready for a one-class checkpoint** (done 2026-08-12, was open here):
      `scripts/eval_3d_maskdino.py::label_setting` derives the label setting from the dataset
      **and** `head_config`, so a `num_classes == 1` head is scored class-agnostic even on
      ScanNetv2, the nyu40-keyed wall/floor prediction filter is skipped for it, and no 18-class
      table is written. **Without it the failure would have been silent and total**: the head's
      single logit is read as dataset class 1, `SCANNET_IDX_TO_NYU40[1]` is **wall**, and the
      wall/floor prediction filter drops wall — so every query of every scene would have been
      discarded and the run would have reported AP 0.000 / 0.000 / 0.000 with no error. Covered by
      `tests/test_maskdino_eval3d.py::test_label_setting_takes_the_head_into_account`; 19-class
      runs are bit-for-bit unchanged.
      Note `scripts/eval_3d_maskdino.py` maps predicted labels to ScanNet ids; a 1-class
      checkpoint must be scored class-agnostic, which the evaluator already supports but does not
      yet *force* from `head_config`.
- [ ] Per-source ablation: is the gain ScanNet++ (real, same domain) or Infinigen (synthetic,
      512×288 upsampled to 518)? The mixture alone cannot say. `SOURCES='scannet scannetpp'` is
      the first arm.
