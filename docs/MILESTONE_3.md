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

## Phase 1 — Fair scaling re-runs (GPU; PENDING user compute)

Pure GPU runs, no new code — the Phase-0 instrumentation (`metrics.jsonl`, the AP50
checkpoint, the identical SLURM protocol) is what they depend on. Submit
`slurm/train_scale{25,10,50}.sh` (now `--eval_interval 50 --early_stop_patience 0`), then
plot val mIoU (prompted+grid) and val AP50 (grid) vs N ∈ {4,10,25,50} from each run's
`metrics.jsonl`. See `docs/NEXT_STEPS_PLAN.md` §Phase 1.

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

## Phases 4 & 6 — Data-gated (BLOCKED on SAM3 per-instance masks)

Gated on the per-instance ScanNet masks produced by the SAM3-side agent
(`docs/SAM3_INSTANCE_MASKS_PROMPT.md`). No downstream work until that data lands and is
QA-approved by the user.
