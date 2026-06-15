# Milestone 3 — Scaling instrumentation, query-mode & pixel-decoder work

This milestone executes `docs/NEXT_STEPS_PLAN.md`. It is organized by the plan's phases.
Phase 0 (instrumentation & small fixes) is **CPU-side and done**; Phases 1–3 are GPU
experiments; Phases 4 and 6 are gated on the SAM3 per-instance data
(`docs/SAM3_INSTANCE_MASKS_PROMPT.md`).

---

## Phase 0 — Instrumentation & small fixes (DONE, 2026-06-14)

All Phase-0 edits are CPU-side, default to previous behavior, and ship with standalone
tests. Full suite (`test_phase2…5`, `test_eval`, `test_milestone2`, `test_visualize_masks`)
passes.

### 0.1 — Persist eval history → `<run_dir>/metrics.jsonl`
`scripts/train_multiscene.py` now appends one JSON line per eval to
`<run_dir>/metrics.jsonl` (sibling of the checkpoint). Each record carries
`epoch, lr, loss, class_loss, mask_loss` and the prompted + grid train/val `mIoU`/`AP50`
(`train_mIoU, train_AP50, train_grid_mIoU, train_grid_AP50, val_*`). Scaling plots now read
this file instead of scraping the log.
- New helpers: `append_jsonl(path, record)`, `build_eval_record(epoch, lr, comps, tr, va, tr_un, va_un)`.
- The eval block now also computes **train** grid metrics every eval (previously only when a
  new best was saved), so the record is complete.
- Test: `tests/test_milestone2.py::test_metrics_jsonl_writer`.

### 0.2 — Smaller checkpoints (uint8 images + `--checkpoint_light`)
Per-scene images dominated checkpoint size (~29 MB/scene). Two levers:
- **Default now stores images as `uint8`** (`(img.clamp(0,1)*255).round().uint8`), 4× smaller
  than float. Decoded back to float on load.
- **`--checkpoint_light`** drops per-scene pixels entirely and stores `frame_names` + the
  scene's `raw_data` path (`scene_dir`); the visualizer/demo reload frames from disk.
- Central decode helper `data/scannet_overfit.py::decode_checkpoint_images(scene, scans_root, img_size)`
  handles all three formats (float passthrough / uint8 → /255 / light → reload), plus
  `load_frames_by_name(...)` for the reload path. Consumers updated:
  `scripts/visualize_masks.py` (new `--scans_root`), `train_multiscene.run_visualizations`
  (threads `scans_root`), and `demos/demo_gradio.py`.
- Checkpoints now also carry `scene_dir` per scene (and top-level) so light checkpoints work
  even with a non-default `--scans_root`.
- Tests: `tests/test_visualize_masks.py::{test_decode_checkpoint_images_formats, test_scene_dir_passthrough}`.

### 0.3 — Noise-robust early stopping (off by default)
Early stopping (`--early_stop_patience`, default 0 = disabled) is now:
- compared against a **moving average** of the selection metric (`--early_stop_window`,
  default 3) with a **min improvement delta** (`--early_stop_min_delta`, default 0.005), and
- **refused before half the schedule** (`epoch+1 ≥ 0.5·num_epochs`) — this is the
  §2.1/§4.3 failure that invalidated the first scale25 run (stopped at peak LR, underfit).
- Pure helpers `moving_average(history, window)` and
  `early_stop_should_stop(evals_no_improve, patience, epoch, num_epochs)`.
- Test: `tests/test_milestone2.py::{test_moving_average, test_early_stop_gate}`.
- **The scaling runs keep `--early_stop_patience 0`** (full schedule); this is robustness for
  later use, not a behavior change for the curve.

### 0.4 — Second best checkpoint on `val[grid] AP50`
Alongside `checkpoint_best.pth` (selected on prompted val mIoU), the run now saves
`checkpoint_best_ap50.pth` selected on the honest unprompted detection number,
val[grid] AP50 (falls back to train[grid] AP50 without val scenes). Tests the
SCALING_RUNS_ANALYSIS §3.2 hypothesis that prompted-mIoU selection picks a poor detection
checkpoint. The final selection metric is to be decided after the Phase-1 re-runs; the
auto-visualization still renders from the mIoU-best checkpoint.

### 0.5 — `--schedule_epochs`
Decouples the cosine schedule length from `--num_epochs`. Default `None` → equals
`--num_epochs` (unchanged behavior). Passing a fixed `--schedule_epochs` means changing the
run length no longer rescales LR decay (removes the §2.1 failure mode permanently).
- Test: `tests/test_milestone2.py::test_schedule_epochs_decoupling`.

### 0.6 — Visualization polish
`scripts/visualize_masks.py`:
- Legend entries are now `"{class} #{k}"` with a per-class instance index, so two same-class
  objects are distinguishable once per-instance GT lands.
- Each frame figure carries the caption *"one color = one predicted instance (mask spans all
  frames jointly)"*.
- `--score_threshold` is exposed (default 0.5) and respected — ready for the §3.3
  under-confidence sweep in Phase 1 (re-render with `--score_threshold 0.3`).

### 0.7 — SLURM scripts (identical protocol)
`slurm/train_scale{10,25,50}.sh`: now all use `--eval_interval 50 --early_stop_patience 0`
(identical protocol across N), and `--time` trimmed to `02:00:00` (jobs run in minutes).

---

## Phase 1 — Fair scaling re-runs (GPU; DONE 2026-06-14, val set is the caveat)

Full-schedule runs (1000 ep, `--early_stop_patience 0`), identical protocol, val =
scene0080–0082. `scripts/plot_scaling.py` reads each run's `metrics.jsonl` into the curve
(`scaling_curve_full.png`).

| N  | best val mIoU (prompted) | best val[grid] AP50 (honest) | final train mIoU | train−val gap | run |
|----|--------------------------|------------------------------|------------------|---------------|-----|
| 4  | 0.138 | — | — | large | MILESTONE_2 §6 |
| 10 | 0.289 @ep650 | 0.171 @ep900 | 0.610 | 0.35 | d4rt_m2_scale10_20260614_194415 |
| 25 | 0.204 @ep600 | 0.147 @ep400 | 0.371 | 0.20 | d4rt_m2_scale25_20260614_143039 |
| 50 | 0.347 @ep750 | 0.159 @ep650 | 0.419 | 0.16 | d4rt_m2_scale50_20260614_194415 |

**Reading it (honestly):**
1. Prompted val mIoU trends up with N (0.14→0.35) but is **non-monotonic** (N=25 dips below
   N=10). With only **3 val scenes** the per-eval signal swings ±0.05–0.08, so the inversion
   is mostly noise — the val set is too thin to trust point-by-point. **Action: widen val to
   scene0080–0089 (no new preprocessing) and re-run** before drawing scaling conclusions.
2. **Unprompted val[grid] AP50 is flat ~0.15 across N=10/25/50** — scaling alone does NOT
   improve the honest detection number. This is the duplicate-suppression / under-confidence
   problem (Phase 2 `--train_grid_queries`, Phase 3 query modes), not a data-volume problem.
3. train−val gap shrinks with N (0.35→0.20→0.16) → more scenes reduce overfitting as
   expected; N=10 overfits hardest (train 0.61), N=25/50 are mildly underfit at the fixed
   1000-ep/LR budget.
4. In every run the mIoU-best and AP50-best checkpoints land at different epochs — consistent
   §3.2 evidence that the selection metric matters (Phase 0.4 was worth adding).

Score-threshold sweep (no retrain, scale25 `visualizations_thr03/` at `--score_threshold
0.3`) confirms §3.3: lowering 0.5→0.3 recovers correct-but-under-confident predictions (chair
0.31, desk 0.54, table 0.36), but predictions where the head outputs the *background class*
(table/desk/sofa → bg) stay undrawn — class confusion, a separate failure the threshold can't
fix.

**Next:** (a) widen val to 0080–0089 and re-run the four points; (b) Phase 2
`--train_grid_queries` A/B on scale25/scale50 — target metric is the flat unprompted AP50.

## Phase 2 — `--train_grid_queries` (CODE DONE, 2026-06-14; experiment PENDING GPU)

`scripts/train_multiscene.py::make_train_queries` now optionally appends the eval grid
(random per-step offset < half a cell, to avoid overfitting cell positions) to the
centroid+background training queries. Hungarian keeps each GT's single best query and
`no_object_weight` pushes the other on-object grid queries to background — DETR-style
duplicate suppression, exercised at train time. New flag `--train_grid_queries` (default
off; stored in checkpoint args). No change to `train/loss.py` (matcher/loss already handle
arbitrary query counts). Test: `tests/test_milestone2.py::test_train_grid_queries`.
**Experiment** (pending GPU, after Phase 1): scale10/scale25 with vs. without the flag;
success metric = unprompted val AP50.

## Phase 3 — `--query_mode {point, learned, hybrid}` (CODE DONE, 2026-06-14; experiment PENDING GPU)

`models/d4rt_decoder.py::QueryGenerator` gained a `query_mode`:
- **point** (default) — current (u,v)/view/patch prompt queries.
- **learned** — `nn.Embedding(num_learned_queries, 256)` true DETR object queries; the
  forward ignores coordinates (the caller passes length-M placeholders so the count stays
  aligned with the matcher/loss). The training loss sets the matcher's `coord_weight=0`.
- **hybrid** — the first M slots are learned object queries, the rest are point queries
  (`coordinates[:, M:]`).

Output length always equals the input query count, so the matcher/loss/eval stay aligned in
every mode. Threaded through `D4RTInstanceSegmentationHead`, `D4RTModel`, and the
`head_config` (`query_mode`, `num_learned_queries`) for the checkpoint→demo round-trip;
`scripts/train_multiscene.py` (flags `--query_mode`, `--num_learned_queries`; learned/hybrid
build placeholder queries and use `matcher_kwargs={"coord_weight":0}`); eval and
`scripts/visualize_masks.py` rebuild mode-aware queries (learned reports under the
unprompted column). Tests: `tests/test_phase3.py::{test_query_modes, test_head_config_roundtrip}`,
`tests/test_milestone2.py::test_query_mode_train_queries`.
**Experiment arms** (pending GPU): A current / B Phase-2 grid / C learned / D hybrid; metric
= held-out unprompted AP50/mIoU. Honest expectation: learned queries are data-hungry and may
underperform at ≤50 scenes; the interesting outcome is the crossover as N grows.

## Phase 5 — MaskDINO-style pixel decoder (CODE DONE, 2026-06-14; training PENDING after 1–3)

New module `models/mask_upsampler.py::MaskUpsampler`: projects the cached patch features and
upsamples the 37×37 map by a power-of-two factor (bilinear + 3×3 conv + GroupNorm + ReLU
stages) before the cosine-similarity mask product. Wired into `InstanceDecoder`
(`mask_upsample`, default 1 = the original Linear path at 37×37, behavior byte-for-byte
unchanged; >1 routes through the upsampler), the head, `D4RTModel`, and `head_config`.
`scripts/train_overfit.py::build_gt_targets` gained `mask_upsample` so GT masks are built at
the matching resolution (74×74 / 148×148). Flag `--mask_upsample` in
`scripts/train_multiscene.py`; `visualize_masks` rebuilds the head at the stored factor. The
cosine-sim + learnable-temperature mask logit (hard-won constraint) is preserved. Test:
`tests/test_mask_upsampler.py`. Train once Phases 1–3 settle. Fallback if the learned pixel
decoder underperforms: reuse VGGT's frozen depth-head DPT features (zero new params).

## Dataset update (2026-06-15): per-instance SAM3 masks landed — Phases 4 & 6 UNBLOCKED

The SAM3-side run is **done**: 97 scenes (scene0000–0096), 2056 instances, per-instance
binary masks at `masks_instance/<class>_<k>/<frame>.png` (zero-based per-class index `k`;
same PNG conventions as `masks/`; cross-frame identity from SAM3 video tracking; `wall`/
`floor` forced to a single instance; union of a class's instances ≈ the old per-class mask,
union-IoU ≈ 1.0). Full spec in
`/cluster/work/igp_psr/niacobone/distillation/dataset/scannet/INSTANCE_MASKS_README.md`.

**Data-access change (now mandatory):** the dataset ships as one zstd tar
`scannet_instance_dataset.tar.zst` (~1.3 GB). Jobs must **not** read the small PNGs off
`work`; instead copy the tar to node-local `$TMPDIR` and unpack it there once. Implemented
in `slurm/stage_dataset.sh` (sourced by `slurm/train_scale{10,25,50}.sh`), which exports
`SCANNET_ROOT=$TMPDIR/scans`; `scripts/train_multiscene.py` uses `SCANNET_ROOT` as the
default `--scans_root`. SLURM headers now request `--tmp=8000` MB of local scratch. The
per-class loader is unchanged and still runs against either the staged tree or `work`, so
all Phase-1/2/3 experiments stay reproducible.

## Phase 4 — Per-instance loader (CODE DONE 2026-06-15; experiment PENDING GPU)

Goal: switch supervision from one-ID-per-class to one-ID-per-`(class, instance)` so the
model is trained/evaluated on true instances (and same-class separation becomes
demonstrable — supervisor §1). Implemented behind a default-off flag, all existing tests
unchanged.

What shipped:

1. **Loader (`data/scannet_overfit.py`).** New ctor flag `instance_level=False`. The two
   `__getitem__` passes were generalized from "per class" to "per **segment**", where a
   segment is one mask folder that becomes one global instance ID:
   - default (per-class): segments = the `masks/<class>/` folders (one per class) — behavior
     **byte-identical** to before (same deterministic class-index ordering, same IDs).
   - `instance_level=True`: segments = the `masks_instance/<class>_<k>/` folders. Folder
     names are parsed with `rsplit("_", 1)` (so multi-word classes like `shower_curtain_0`
     work), `_qa`/metadata dirs are skipped, and segments are sorted by `(class_idx, k)` for
     a stable `global_id -> class` mapping. `classes[i]` carries the instance's class index
     and **repeats** across same-class instances — class head (19+bg) untouched;
     `coordinates`/`frame_ids` per instance are computed exactly as before.
2. **Flags wired through** `scripts/train_overfit.py` (`--instance_level` →
   `create_dataloader`) and `scripts/train_multiscene.py` (`--instance_level` → the dataset
   `common` kwargs). `train_multiscene` stores `vars(args)` in the checkpoint, so the mode is
   persisted; the head architecture is mode-independent and the visualizer reads the baked-in
   `gt` from the cached bundles, so the checkpoint→demo round-trip needs no extra plumbing.
3. **Tests.** `tests/test_phase2.py::test_instance_dataset` synthesizes a scene with two
   same-class instances (`chair_0`, `chair_1`) + a `wall_0` and asserts: 3 instances, the
   chair class index appears twice, the two chairs get distinct cross-view-consistent IDs in
   disjoint pixels. The original `test_dataset` (per-class path) is kept and still passes,
   guarding the default behavior. Smoke-tested on real `scene0000_00` (23 segments → matches
   the README total; `shower_curtain` correctly split into 2 instances).
4. **No changes** in `train/loss.py` / `train/eval_metrics.py` (already Hungarian-over-GT-
   instances); `test_phase5`/`test_eval` pass unchanged.

**Experiment (PENDING GPU):** add `--instance_level` to the scale scripts and re-run the
Phase-1 curve on instance GT (val 0080–0089). **Caveat for slides:** mIoU/AP will likely
**drop** vs per-class (more, smaller, harder instances; chairs especially) — expected, not a
regression. New headline figure: two same-class objects in two colors (visualizer colors by
match; the Phase-0 `"{class} #{k}"` legend labels them).

## Phase 6 — Data-gated ablations (after Phase 4 + the fair scaling re-runs)

Now meaningful with 97 scenes. All on held-out val (widen to scene0080–0089 first — no new
preprocessing), metric = unprompted val[grid] AP50 (the honest number) + prompted mIoU.

1. **No-object weight sweep** (0.05 / 0.1 / 0.4) — tests whether `no_object_weight 0.1`
   drives the §3.3 under-confidence.
2. **Augmentation ablation** — `bundles_per_scene` 1 vs 4, `query_jitter` on/off,
   `color_jitter` on/off.
3. **Grid density vs unprompted recall** — `--grid_size` 4/6/8.
4. **Longer-term:** partial backbone unfreezing once the train−val gap vs N says the
   dataset can support it.

## Suggested execution order from here

```
Phase 4 (instance loader + tests)            ← do first; everything downstream is on instance GT
  └─► re-run the Phase-1 scaling curve on instance GT (N∈{10,25,50}, val 0080–0089)
        └─► Phase 2 --train_grid_queries A/B   (the flat unprompted-AP50 target)
        └─► Phase 3 query-mode arms A/B/C/D
        └─► Phase 5 --mask_upsample train      (mask sharpness; independent of what is learned)
        └─► Phase 6 ablations
```
Note the pending GPU experiments for Phases 1/2/3/5 were all coded against the per-class
loader; after Phase 4 they should be re-run on instance GT so the whole curve is on one GT
definition. The N=100+ scaling point still needs more scenes downloaded+preprocessed (the
only remaining SAM3-side stretch item).
