# Next-Steps Plan — Consolidated Open Issues (written 2026-06-12)

This is an actionable, ordered plan that closes **every open issue** currently recorded in
`docs/todo.md`, `docs/supervisor_feedback_jun_12.md` (§1–§5), `docs/SCALING_RUNS_ANALYSIS.md`
(§4), and `docs/MILESTONE_2.md` (§7). It is written to be used directly as the prompt for
the next working session.

**How to execute (per CLAUDE.md working rules):** proceed phase by phase, in order. Every
code change gets a standalone CPU-runnable test in `tests/` (new or extended) and a doc
update (`docs/todo.md` + the relevant milestone doc) before moving to the next item. New
flags default to off / previous behavior so all existing tests pass unchanged. Keep the
checkpoint `head_config` round-trip intact whenever the head constructor changes.

---

## ⚠ User actions required (SAM3 ground-truth side)

These items depend on the SAM3 preprocessing side, which the user drives separately. A
ready-to-use prompt for the SAM3-side agent is in `docs/SAM3_INSTANCE_MASKS_PROMPT.md`.
Status verified on disk (2026-06-12): **97 scenes (scene0000–0096) are already preprocessed
with per-class masks**, including val candidates scene0083–0089 — so widening the val set
(SCALING_RUNS_ANALYSIS §4.4) needs **no new preprocessing for per-class runs**, only scene
lists, and scale50 + a wider-val protocol are unblocked today.

What still needs the SAM3 side (gates Phases 4 and 6):

1. **Per-INSTANCE SAM3 masks** (decision recorded Jun 12, supervisor_feedback §4):
   - One binary mask per *instance*, layout `masks_instance/<class>_<k>/<frame>.png` next
     to (not replacing) the existing `masks/`; same PNG conventions (uint8 {0,255},
     1296×968, all 100 subset-frame filenames per instance dir, all-zero when absent).
   - Cross-frame instance identity comes from SAM3 tracking/propagation — the load-bearing
     assumption of the GT. The prompt mandates a **visual QA on 3 pilot scenes + user
     approval before the bulk run**, plus a union-IoU check against the old per-class masks.
   - Stuff classes (wall/floor) stay single instances; class taxonomy unchanged (19 classes,
     indices 1..19, 0 = bg downstream).
   - Bulk priority order: val scenes 0080–0089 first, then 0000–0049, then the rest.
2. **More scenes beyond the 97** (stretch, for the N=100+ scaling point): download +
   preprocess additional ScanNet scenes in the same format.

Until per-instance data lands, all training/ablation phases below run on the existing
per-class masks — results stay comparable to the current baselines, and the loader switch
in Phase 4 is isolated.

---

## Phase 0 — Instrumentation & small fixes (CPU-side, no GPU needed)

Small, independent edits from SCALING_RUNS_ANALYSIS §4.3 + supervisor_feedback §1. Do these
first: everything downstream produces data through them.

1. **Persist eval history** — in `scripts/train_multiscene.py`, append one JSON line per
   eval (epoch, lr, mean loss, train/val mIoU & AP50, prompted + grid) to
   `<run_dir>/metrics.jsonl`. Scaling plots then come from files, not log scraping.
   Test: extend `tests/test_milestone2.py` (or a tiny new test) to check the JSONL writer.
2. **Shrink checkpoints** (~29 MB/scene today → bites at N≥50):
   - Store bundle images as uint8 (`(img.clamp(0,1)*255).to(torch.uint8)`); convert back to
     float in `scenes_from_checkpoint` and the demo loader (4× smaller).
   - Add `--checkpoint_light`: drop per-scene images, store `frame_names` + scene path; the
     visualizer reloads frames from `--scans_root`.
   - Keep the head-config round-trip and `tests/test_visualize_masks.py` passing (extend it
     for the uint8/light formats).
3. **Noise-robust early stopping** (off by default; early stopping stays disabled for the
   scaling runs): `--early_stop_min_delta` (e.g. 0.005), compare against a 2–3-eval moving
   average of val mIoU, and refuse to stop before `epoch ≥ 0.5·num_epochs`.
4. **Second "best" checkpoint selected on val[grid] AP50** (`checkpoint_best_ap50.pth`, or
   at minimum log which epoch would have been chosen) — tests the SCALING_RUNS_ANALYSIS
   §3.2 hypothesis that prompted-mIoU selection picks a poor detection checkpoint. Decide
   the selection metric after the fair re-runs.
5. **`--schedule_epochs`** to decouple the cosine schedule length from `--num_epochs`
   (removes the §2.1 failure mode permanently).
6. **Visualization polish** (supervisor_feedback §1):
   - Legend entries `"{class} #{k}"` in `scripts/visualize_masks.py` so duplicate class
     names are distinguishable once per-instance GT exists.
   - Caption overlays "one color = one predicted instance (mask spans all frames jointly)".
   - Make sure `--score_threshold` is exposed/respected for the §3.3 sweep in Phase 1.
7. **Fix the SLURM scripts** (no Python): in `slurm/train_scale10.sh` / `train_scale25.sh`
   / `train_scale50.sh`, set `--eval_interval 50 --early_stop_patience 0` everywhere
   (identical protocol), and cut `--time` allocations (~2 h is generous; jobs run minutes).

Regression gate: full test suite (`tests/test_phase2.py` … `test_milestone2.py`,
`test_visualize_masks.py`) passes before Phase 1.

## Phase 1 — Fair scaling re-runs (GPU; the prerequisite for all experiments)

SCALING_RUNS_ANALYSIS §4.2: the scale10-vs-scale25 comparison is invalid (scale25 was
early-stopped at epoch 200, still at peak LR, underfit). With the Phase-0 protocol fix:

1. Re-run **scale25** (fixed protocol) → the real N=25 point.
2. Re-run **scale10** with `--early_stop_patience 0` → matching full-schedule N=10 point.
3. Launch **scale50** only after 1–2 look sane.
4. **Plot** val mIoU (prompted + grid) and val AP50 (grid) vs N ∈ {4, 10, 25, 50} from
   `metrics.jsonl` (N=4 = 0.138, MILESTONE_2 §6). Also plot the train−val gap vs N (§3.1).
5. Cheap add-ons (each a ~30 min job, mostly caching):
   - **Score-threshold sweep at viz time** (no retraining): re-render the scale10
     checkpoint with `--score_threshold 0.3` — the head is systematically under-confident
     (correct classes at 0.28–0.49 get dropped, §3.3).
   - **LR sanity check at N=25**: one run at `--learning_rate 1e-3` (2e-3 was tuned in the
     4–10-scene regime).
   - **Capacity probe only if** the fair N=25/N=50 runs underfit (train loss stuck above
     the N=10 level with flat val): `num_decoder_layers` 4→6 or `hidden_dim` 256→384.
6. When the wider val set is preprocessed (user action above), redo model selection /
   final curve points with it.

Document results in a new section of SCALING_RUNS_ANALYSIS.md (or a MILESTONE_3 doc) and
tick the corresponding `todo.md` lines.

## Phase 2 — `--train_grid_queries`: trained duplicate suppression (supervisor §3)

The most actionable supervisor comment: training never exercises DETR's duplicate-
suppression mechanism because grid queries exist only at eval, so multiple grid cells fire
on one object at inference → duplicate false positives → depressed unprompted AP50.

1. In `make_train_queries` (`scripts/train_multiscene.py`), optionally append the eval
   grid (or a random-offset grid, to avoid overfitting cell positions) to the
   centroid+background queries. Hungarian assigns each GT its single best query;
   `no_object_weight` pushes every other on-object query to background. **No change needed
   in `train/loss.py`** (matcher/loss already handle arbitrary N).
2. New flag `--train_grid_queries`, default off; store in checkpoint args. Cost: 32 → ~320
   queries/step — irrelevant at 3–4 min/run.
3. CPU test: extend `tests/test_milestone2.py` — grid queries included in matching →
   unmatched on-object queries receive no-object loss.
4. **Experiment**: scale10/scale25 (Phase-1 protocol) with vs. without the flag. Success
   metric: unprompted val AP50 (gap vs prompted should close substantially). Only if
   duplicates persist, discuss NMS as a band-aid.
5. While at it, add the reporting caveat everywhere unprompted numbers appear (slides
   included): unprompted mIoU is optimistic; **AP50 is the honest unprompted number**
   (SCALING_RUNS_ANALYSIS §3.5).

## Phase 3 — Query-mode ablation: point prompts vs learned object queries (supervisor §5)

1. Add `--query_mode {point, learned, hybrid}` to `QueryGenerator` / the head constructor
   (stored in `head_config` for the checkpoint round-trip):
   - `point` = current behavior (default).
   - `learned` = `nn.Embedding(M, 256)`, M ≈ 64–100, true DETR object queries; set the
     matcher's `coord_weight=0` in this mode (coordinates already carry no loss term).
   - `hybrid` (optional) = learned + centroid prompts concatenated.
2. Eval plumbing: for `learned`, the prompted/unprompted distinction collapses — report
   under the unprompted column.
3. CPU test for the new QueryGenerator path + checkpoint round-trip.
4. **Experiment arms** (on the corrected protocol, N=25 or 50; metric = held-out
   unprompted AP50/mIoU):
   | Arm | Train queries | Eval queries |
   |---|---|---|
   | A (current) | GT centroids + bg | centroids (prompted) / grid (unprompted) |
   | B (= Phase 2) | + grid queries | same |
   | C (learned) | M learned embeddings | same M embeddings |
   | D (hybrid, optional) | learned + centroids | learned only or learned+prompt |
   Honest expectation: with ≤50 scenes, learned queries may underperform (DETR queries are
   data-hungry); the interesting outcome is the crossover as N grows. Arm B goes first.

## Phase 4 — Per-instance loader (gated on user's SAM3 per-instance data)

When the per-instance masks land (user action §4 above):

1. `data/scannet_overfit.py`: replace "one global ID per class" with one ID per
   (class, instance); `gt["classes"]` then contains repeated class indices — class head
   (19+bg) untouched; centroids per instance work as-is.
2. Matcher/loss/eval are already instance-based — no changes.
3. Update `tests/test_phase2.py` to assert per-*instance* (not per-class) cross-view
   consistency.
4. Re-run the Phase 1–3 headline experiments on the instance GT. **Expect mIoU/AP to drop
   at first** (more, smaller, harder instances) — flag this in the next slides so it isn't
   read as a regression. New headline figure: two same-class objects in two colors
   (the existing visualizer already colors by match; with the Phase-0 `"{class} #{k}"`
   legend this directly answers supervisor comment §1).

## Phase 5 — MaskDINO-style pixel decoder (supervisor §2; CPU-developable in parallel)

The current mask head is already the Mask2Former/MaskDINO mechanism; the missing piece is
upsampling the 37×37 feature map before the query⊗feature product.

1. New module `models/mask_upsampler.py`: input cached per-frame patch features
   `[S, 37, 37, 2048]`, project to 256, 2–3 bilinear+conv (or ConvTranspose2d) stages →
   `[S, 256, 74, 74]` or `[S, 256, 148, 148]`. Mask logits stay cosine similarity with
   learnable temperature (**hard-won constraint — keep it**).
2. Supervision: downsample the full-res GT masks to the new output resolution instead of
   37×37. If dense Dice+BCE on `[N, S, 148, 148]` doesn't fit in memory, adopt
   Mask2Former's point-sampled mask loss (~3k points/mask).
3. Flag `--mask_upsample 1` = current behavior (default); store
   `head_config["mask_upsample"]` for the checkpoint→demo round-trip.
4. Standalone CPU phase test (`tests/test_mask_upsampler.py` or similar).
5. Fallback if the learned pixel decoder underperforms: reuse VGGT's frozen depth-head DPT
   intermediate features as the high-res map (zero new params, heavier cache).
6. Train it once Phases 1–3 settle (those change *what* is learned; this changes mask
   sharpness). It may also resolve the window/door/picture confusion if that is a
   resolution problem rather than a coverage problem (SCALING_RUNS_ANALYSIS §5).

## Phase 6 — Data-gated ablations (after the bulk per-instance dataset lands)

MILESTONE_2 §7.2–7.4, only meaningful with enough scenes:

1. **No-object weight sweep** (0.05 / 0.1 / 0.4) on unprompted AP50 — also answers whether
   `no_object_weight 0.1` is what pushes scores down (§3.3 under-confidence).
2. **Augmentation ablation**: bundles_per_scene 1 vs 4, query_jitter on/off,
   color_jitter on/off, on held-out scenes.
3. **Grid density vs unprompted recall**: `--grid_size` 4/6/8.
4. **Longer-term** (decide after the scaling curve): partial backbone unfreezing once the
   train−val gap vs N says the dataset can support it.

---

## Dependency summary

```
Phase 0 (instrumentation) ──► Phase 1 (fair scaling runs) ──► Phase 2 (--train_grid_queries)
                                                          └──► Phase 3 (query-mode ablation)
USER: per-instance SAM3 run ─────────────────────────────────► Phase 4 (instance loader)
Phase 5 (pixel decoder) — CPU development anytime; training after 1–3
USER: bulk scenes + wider val set ───────────────────────────► Phase 6 (ablations)
```

After each phase: update `docs/todo.md`, the milestone doc, and CLAUDE.md if commands or
conventions changed.
