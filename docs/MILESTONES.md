# Project Summary — Milestones 1–3 (consolidated)

**Project:** A DETR-like / D4RT-style decoder for **multi-view-consistent 3D instance
segmentation**, trained on top of a **frozen VGGT-1B backbone**. Ground truth = SAM3 masks on
ScanNet scenes. The VGGT backbone is never modified; only the ~6.5M-param head trains.

This file is the single, current summary. The detailed per-milestone docs, the executed plans,
and addressed supervisor feedback are archived in `docs/old/` (read those for the full
debugging narrative). Companion live files: `docs/todo.md` (open work), `docs/HOOK_PLAN.md`
(where the decoder hooks into VGGT), `CLAUDE.md` (commands, storage layout, hard-won
constraints).

---

## Architecture (the pipeline, one component per file)

Hook point: `aggregated_tokens_list[-1]` from VGGT's aggregator → scene features
`F: [B, S, P, 2048]` (S frames, P = patch + 1 camera + 4 register tokens; `patch_start_idx`
splits them). Backbone runs under `no_grad`.

1. **`data/scannet_overfit.py`** — `ScanNetSingleSceneDataset` / `ScanNetMultiSceneDataset`.
   Loads the ~100 stride-5 frames from `subset/` (NOT `color/`) + binary mask PNGs.
   - Default (per-class): one global cross-view-consistent ID per class from `masks/<class>/`.
   - `instance_level=True` (`--instance_level`): one ID per `(class, instance)` from
     `masks_instance/<class>_<k>/`; same-class objects become distinct instances; `classes`
     repeats class indices. Matcher/loss/eval unchanged (already instance-based).
   - Image size 518 (÷14 = VGGT patch size); mask/eval at the 37×37 patch grid.
2. **`models/d4rt_decoder.py`** — `QueryGenerator` + `InstanceDecoder` (4-layer/8-head
   `nn.TransformerDecoder`) → `class_head` (20 logits = 19 classes + bg@0), `mask_embed_head`,
   dense mask head → `pred_masks [B, N, S, h, w]`. `D4RTInstanceSegmentationHead` chains them.
   - `query_mode`: `point` (default; Fourier(u,v)+view-embed+9×9 RGB-patch MLP), `learned`
     (`nn.Embedding` DETR object queries, matcher `coord_weight=0`), `hybrid` (first M learned,
     rest point).
   - `mask_upsample`: 1 (default, 37×37 Linear path) / 2 / 4 routes through
     `models/mask_upsampler.py::MaskUpsampler` for sharper masks (GT built at matching res).
   - Both stored in `head_config` for the checkpoint→demo round-trip.
3. **`train/loss.py`** — `PointBipartiteMatcher` (Hungarian, Dice+BCE cost) + `D4RTLoss`
   (Focal class + Dice + fg-weighted BCE; optional DETR no-object loss via `no_object_weight`,
   default 0.1). Batch-aware (lists of per-sample tensors for B>1).
4. **`train/eval_metrics.py`** — mIoU / AP50 / AP75 / mAP / class_acc. Reports **prompted**
   (queries at GT centroids) and **unprompted** (uniform grid, no GT). **Unprompted val[grid]
   AP50 is the honest detection number.**
5. **`scripts/train_multiscene.py`** — caches frozen-backbone features **once per bundle up
   front**, then every epoch runs only the head (→ minutes, not hours). `--cache_device cpu`
   removes the GPU-memory bound on scene count.

### Hard-won constraints (violating these silently breaks training — keep them)
- **LayerNorm the projected memory** + keep the **query skip connection** in `InstanceDecoder`
  (raw VGGT features have huge magnitudes → otherwise all queries collapse to one vector: loss
  falls, mIoU stays 0).
- Mask logits use **cosine similarity with a learnable temperature**, not raw dot products.
- BCE uses a foreground `pos_weight`; gradient clipping on.
- Coordinates are query *prompts* (enter the matcher cost only), never a loss term.
- An overfit test must hold inputs **and** targets fixed across epochs.

---

## Milestone 1 — Prototype (DONE)
Full phase-1–6 prototype: dataset loader, QueryGenerator, InstanceDecoder + dense mask head,
matcher + losses, eval metrics. Validated single-scene overfit (gradient flow) and 4-scene
multi-scene training (**mean train mIoU 0.967**). Eval on a 5th unseen scene: mIoU 0.027 final
(peaked ~0.13 mid-training) — **no real generalization with only 4 scenes**, motivating M2.
The hard-won constraints above were discovered here. Full detail: `docs/old/MILESTONE_1.md`.

## Milestone 2 — Regularized, unprompted training (DONE)
Converted the overfit pipeline into a real training loop usable **without GT prompts**:
- **No-object loss** (DETR `eos`, `no_object_weight` default 0.1): class loss over all N
  queries, unmatched → background. Train prompted AP50 jumped 0.54→0.77.
- **Unprompted grid inference/eval** (`generate_grid_queries`, default 6×6/frame): "find the
  instances yourself". Dual prompted/unprompted metrics reported everywhere.
- **Regularization:** `--bundles_per_scene K` (bundle 0 = deterministic eval bundle, 1..K-1 =
  random frames), `--query_jitter`, per-step background resampling, `--color_jitter`,
  `--cache_device cpu`.
- **Model selection:** `checkpoint_best.pth` on val mIoU + optional early stopping; auto-renders
  2D overlays after training (`--no_visualize` to skip).
- M1 behavior exactly recovered with `--no_object_weight 0 --bundles_per_scene 1 --query_jitter
  0 --fixed_bg`. Validated on 5 scenes (best val mIoU 0.138). **Scaling was blocked on data.**
Full detail: `docs/old/MILESTONE_2.md`.

## Milestone 3 — Scaling, query modes, pixel decoder, per-instance GT (largely DONE)

> **Status (2026-06-22):** the arm-A *point*-prompt scaling curve is complete through N=200 and
> has **plateaued** (val mIoU ~0.22, honest AP50 ~0.10). But **arm C (learned queries) scaled to
> N=200 breaks that plateau** — val mIoU **0.371**, honest AP50 **0.228** (>2× the point
> baseline), with its old overfitting resolved. The ceiling was the head, not the data. Learned
> queries are now the default for further work; next levers (pixel decoder, ablations) stack on
> top of arm C. See the arm tables below and `docs/todo.md`.

**Phase 0 (instrumentation, CPU, DONE):**
- `metrics.jsonl` per run (one line per eval: epoch, lr, loss, prompted+grid train/val
  mIoU/AP50) — scaling plots read this, not logs.
- Smaller checkpoints: **uint8 images** (4× smaller) + `--checkpoint_light` (drop pixels,
  reload from disk via `decode_checkpoint_images`).
- Noise-robust early stopping (moving avg + min-delta, refuses before half schedule; off by
  default, scaling runs use patience 0).
- Second checkpoint `checkpoint_best_ap50.pth` selected on the honest val[grid] AP50.
- `--schedule_epochs` decouples cosine length from `--num_epochs`.
- Viz polish: `"{class} #{k}"` legend, per-instance caption, `--score_threshold` exposed.

**Phase 4 — per-instance loader (DONE):** `instance_level` flag wired through
train_overfit/train_multiscene; per-`(class,instance)` IDs. No changes to matcher/loss/eval.
Tests added. This is now the real supervision target.

**Phases 1–4 results (instance GT, wide val = scene0080–0089), arm A = point prompts:**

| N   | val mIoU  | val[grid] AP50 (honest) | final train mIoU |
|-----|-----------|-------------------------|------------------|
| 10  | 0.152     | 0.089                   | 0.526            |
| 25  | 0.174     | 0.111                   | 0.353            |
| 50  | 0.212     | **0.125**               | 0.338            |
| 100 | **0.228** | 0.103                   | 0.272            |
| 200 | 0.216     | 0.105                   | 0.265            |

- **The curve has plateaued (updated 2026-06-22).** Through N=50 both columns climbed
  monotonically, but extending to N=100 (`d4rt_m2_scale100_inst`) and N=200 / all 190 non-val
  scenes (`d4rt_full_inst`) flattens val mIoU at ~0.21–0.23 and leaves the honest val[grid] AP50
  at ~0.10 — *below* its N=50 peak of 0.125. **More scenes is no longer the lever**; the ceiling
  is now architecture/resolution, not data quantity.
- Instance-GT numbers are ≈half the per-class equivalents — **the predicted cost of the harder
  task** (more, smaller objects), not a regression.
- train−val gap keeps shrinking with N (0.37→0.18→0.13→0.04→0.05) → the model is no longer
  overfitting; it has hit a capacity/resolution ceiling rather than a data ceiling. This is what
  motivates pivoting to the head/training items (learned queries at large N, the pixel decoder)
  over further scaling.

**Phases 2/3 query-mode arms (all N=50, instance GT, wide val):**

| Arm           | train queries        | val mIoU  | val[grid] AP50 | train mIoU | outcome |
|---------------|----------------------|-----------|----------------|------------|---------|
| A point       | GT centroids + bg    | 0.212     | 0.125          | 0.338      | baseline |
| B grid (P2)   | + random-offset grid | **0.047** | 0.146          | **0.055**  | mask learning collapsed |
| C learned     | M learned embeddings | **0.259** | **0.146**      | 0.749      | **best val**; overfits (gap 0.49) |
| D hybrid      | learned + centroids  | ~0.27*    | —              | 0.54*      | **crashed** (NaN @~ep555) |

\* D's last eval before the crash.

**Arm C scaled to N=200 (instance GT, wide val) — the headline result (2026-06-22):**

| Arm C learned | val mIoU  | honest val[grid] AP50 | train mIoU | gap   |
|---------------|-----------|-----------------------|------------|-------|
| N=50          | 0.259     | 0.146                 | 0.749      | 0.49  |
| N=200 best (@ep600) | **0.371** | 0.228           | 0.457      | 0.086 |
| N=200 final (@ep1000) | 0.326 | **0.228**            | 0.560      | 0.23  |

- **Learned queries break the plateau — the ceiling was the head, not the data.** Against the
  N=200 *point* baseline (val mIoU 0.216, AP50 0.105), arm C gives +0.15 val mIoU and **>2× the
  honest AP50** (0.105 → 0.228). Run: `d4rt_full_inst_learned_20260622_183203`.
- **Learned queries keep scaling** where point prompts saturated: val mIoU 0.259 → 0.371 from
  N=50 → N=200, and their N=50 overfitting **resolved** (train−val gap 0.49 → 0.086 at best epoch)
  — exactly the predicted crossover.
- For learned queries there are no point prompts, so prompted == grid metrics: **0.228 is the
  unconditional honest detection number.** Next levers stack *on top* of arm C (pixel decoder —
  since tested, neutral, see Phase 5; no-object/aug ablations).

- **C (learned object queries) was the surprise winner** already at N=50 — *against* the "DETR
  queries are data-hungry" prior — and the win compounds with scale (table above).
- **B (`--train_grid_queries`) backfired:** train mIoU stuck ~0.05 while class loss fell —
  learned to classify but not to mask. Mechanism: ~320 queries/step routes many GTs to grid
  queries, under-supervising the GT-centroid queries eval uses, and `no_object_weight` over ~10×
  more (mostly bg) queries swamps the few matched-mask gradients. AP50 did tick up (intended
  duplicate suppression). **Fix:** normalize the no-object term by query count (or lower its
  weight with grid on), and/or keep centroid queries always matched.
- **D (hybrid) is numerically unstable:** NaN/inf into `linear_sum_assignment`
  (`train/loss.py:253`) ~ep555 — exploding gradients in the mixed path. Was the best arm before
  dying. **Fix:** guard the matcher cost (`nan_to_num` + finite assert) + tighter grad-clip /
  lower LR on the learned-embedding params.

**Phase 5 — MaskDINO-style pixel decoder (TRAINED 2026-06-30 — neutral result):**
`models/mask_upsampler.py` upsamples the 37×37 map before the cosine-sim mask product
(`--mask_upsample 1/2/4`). Cosine-sim + learnable temperature preserved.

`--mask_upsample 2` (74×74) on the arm-C base (learned, 64 queries, instance GT, N=190; run
`d4rt_full_inst_learned_us2_20260630_161537`, job 5275027 — hit the 4h walltime *after* training
finished, so only the auto-render was cut; overlays re-rendered afterwards, and
`slurm/train_full.sh` now passes `--no_visualize`):

| vs us=1 baseline | honest val[grid] AP50 | val mIoU | gap @best |
|------------------|-----------------------|----------|-----------|
| us=1 (`d4rt_full_inst_learned_20260622_183203`) | 0.228 | **0.371** (@ep600) | 0.086 |
| us=2 best        | **0.236** (@ep500)    | 0.355 (@ep250) | 0.098 |
| us=2 final (@ep1000) | 0.200             | 0.311    | —     |

- **A wash: doubling mask resolution doesn't move the numbers** (+0.008 AP50, −0.016 mIoU —
  within run-to-run noise). Mask resolution is NOT the current bottleneck; `--mask_upsample 4`
  is deprioritized per the decision rule (only on a win).
- Implication: the window/door/picture confusion is semantic rather than resolution-limited —
  the next levers are the score-threshold/under-confidence cluster (no-object sweep) and fixing
  arm D (hybrid), not sharper masks.

Full detail (incl. the SCALING_RUNS_ANALYSIS protocol fixes that made the curve fair):
`docs/old/MILESTONE_3.md`, `docs/old/SCALING_RUNS_ANALYSIS.md`.

---

## Persistent qualitative findings (carry into next experiments)
- **Under-confidence:** many correct predictions land at score 0.28–0.49 and are dropped by the
  0.5 threshold. `--score_threshold 0.3` recovers them; predictions where the head emits the
  *background class* stay undrawn (class confusion, not a threshold issue).
  **Update 2026-07-03 — point-prompt-only phenomenon.** On the arm-C learned-query N=200
  checkpoint, thr 0.3 adds 76 val instances of which 2 are correct (IoU≥0.5) — pure noise —
  and the model already keeps 338 instances vs 144 GT at thr 0.5. For learned queries the
  problem is **over**-prediction (duplicates/FPs), pointing at the no-object-weight sweep,
  not a lower threshold. Keep 0.5.
- **Class-confusion cluster:** `window ↔ door ↔ picture ↔ curtain` (flat, wall-mounted,
  rectangular — hard at 37×37; where RGB evidence + the pixel decoder should help most).
- **Coverage gaps:** bathroom fixtures (`toilet`/`sink` → `chair`) when train scenes lack them —
  fixed by more/varied scenes.
- **Unprompted mIoU is optimistic** (more candidates to match, unmatched FPs unpunished by mIoU);
  **AP50 is the honest unprompted number** — always report it.
- mIoU-best and AP50-best checkpoints land at different epochs → selection metric matters
  (hence `checkpoint_best_ap50.pth`).

## Dataset status (updated 2026-06-17)
- **200 scenes: scene0000–scene0199, ≈4195 instances.** All packed in a single
  `…/scannet/scannet_instance_dataset_full.tar.zst` (~2.6 GB; unpacked ~5.4 GB), containing
  `scans/<scene>/raw_data/...`. Per-scene/per-class counts: `INSTANCE_MASKS_README.md`
  (scene0000–0096, 2056 inst.) + `INSTANCE_MASKS_README_split2.md` (scene0097–0199, 2139 inst.).
  (The two older split tars `scannet_instance_dataset.tar.zst` + `…_split2.tar.zst` are
  superseded by the full tar and can be deleted.)
- Per-instance masks: `masks_instance/<class>_<k>/<frame>.png` (uint8 {0,255}, 1296×968, cross-
  frame identity from SAM3 video tracking; `wall`/`floor` forced single instance; union of a
  class's instances ≈ old per-class mask, union-IoU ≈ 1.0). Per-class `masks/` retained.
- **Data access:** `slurm/stage_dataset.sh` copies the full tar to node-local `$TMPDIR`, unpacks
  once, exports `SCANNET_ROOT=$TMPDIR/scans` (read off local SSD, never the small PNGs off
  `work`); `train_multiscene.py` honors it as default `--scans_root`. SLURM headers request
  `--tmp=16000` MB. Canonical uncompressed source tree:
  `/cluster/scratch/niacobone/scannet_build/scans`.

## Storage layout
- Repo: `/cluster/scratch/niacobone/vggt`. Runs/checkpoints:
  `/cluster/work/igp_psr/niacobone/distillation/output/<run_name>/` (timestamped).
- `checkpoint_best.pth` (best val mIoU) = use for eval/demos; `checkpoint_best_ap50.pth` = same
  run on honest val[grid] AP50. Each run dir has `metrics.jsonl`.
- Checkpoints are self-contained (head weights + `head_config` + scene bundles + optim/sched for
  `--resume`); frozen backbone reloaded from HF `facebook/VGGT-1B`, never stored.
</content>
