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

**Arm B/D fix reruns (fixes 2026-07-03, results 2026-07-07) — both fixes work, neither beats arm C:**

| Fix rerun (N=50, instance GT, wide val) | best val[grid] AP50 | best val[grid] mIoU | train[grid] mIoU @ep1000 | outcome |
|---|---|---|---|---|
| B `_gridq_fix` (`--no_object_norm matched`; job 5647527) | **0.161** (@ep700) | 0.284 | 0.458 | collapse fixed (was 0.055); clears the ≥0.125 success bar |
| D `_hybrid_fix` (`--learned_query_lr_scale 0.1`, grad-clip 0.5, guarded matcher; job 5647528) | 0.146 (@ep200) | 0.247 (@ep250) | 0.750 | NaN fixed — full 1000 epochs, zero non-finite warnings; only *ties* arm C N=50 (0.259/0.146) |

- Judge arm B by the **grid** columns only: its prompted val mIoU stays ~0.05–0.11 because with
  288 trained grid queries the GT-centroid prompts used by the prompted eval get little
  supervision — that's routing, not the old mask collapse (train[grid] mIoU learns 0.13→0.46).
- **B scaled to N=190** (`d4rt_full_inst_gridq_fix_20260703_184456`, job 5658375): val[grid]
  mIoU reaches **0.372** @ep1000 (still rising; matches arm C's 0.371) but the honest AP50 peaks
  at only **0.185** (@ep650) and is unstable across evals (0.071 @ep1000) — well below arm C's
  0.228. Trained grid queries recover mask quality at scale but not stable detection.
- **D no longer NaNs but is not a win** at N=50 → per the decision rule (scale only on a win),
  no N=190 run. The centroid prompts reintroduce the point-path overfitting that pure learned
  queries had resolved (val decays 0.247→0.177 after ep250 while train[grid] climbs to 0.75).
- **Verdict: arm C (pure learned queries) stays the base.** Arms B and D are closed; the next
  levers are the Phase-6 ablations (no-object-weight sweep / duplicate suppression first).

**Grid-density ablation (2026-07-07, eval-only, job 6111639 — negative: 6×6 was not the
bottleneck; the learned-vs-grid gap is architectural, not a density artifact):**
`scripts/eval_grid_ablation.py` sweeps the unprompted `--grid_size` (2/4/6/8/10/12) on a
checkpoint's stored val bundles without retraining (grid-6 reproduces the training-time
`val_grid_AP50` of the selected epoch exactly). Val AP50 by density — arm A
(`d4rt_full_inst`, best_ap50 ckpt): 0.023/0.124/0.134/**0.138**/0.116/0.109, i.e. flat from
6→8 and *falling* beyond; arm B (`gridq_fix`, trained 6×6 grid): 0.018/0.063/**0.185**/
0.100/0.134/0.067 — a sharp peak exactly at its training density. Mechanism confirmed:
kept foreground predictions explode with density (arm A: 58 @g6 → 236 @g12 vs 14.4
GT/scene) — denser grids die by duplicate FPs (no NMS). Unprompted mIoU instead rises
monotonically with density (0.297→0.336): the known "unprompted mIoU is optimistic"
artifact — judge density on AP50 only. The best grid number over all densities and
checkpoints (0.185) stays well below arm C's 0.228.

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

## Dataset status (updated 2026-07-08 — official ScanNet GT is now the default supervision)

**GT migration (2026-07-08, `docs/OFFICIAL_GT_MIGRATION_PLAN.md`).** A 2026-07-07 audit of the
SAM3 GT (20 scenes) found systematic **cross-class duplicates**: SAM3 prompts each class
independently, so the same physical object is often an instance under two classes — 68 pairs
with cross-frame IoU ≥ 0.5 between different classes (~3.4/scene, mostly pixel-identical;
desk↔table, curtain↔shower_curtain, chair↔sofa, …), **15.9% of foreground pixels multi-class**.
Training effect: the matcher demands two predictions for one object (built-in honest-AP50 false
positives) and the class head gets contradictory supervision. → Supervision switched to the
**official ScanNet v2 2D instance GT** (`_2d-instance-filt`/`_2d-label-filt`: projections of the
single human-verified 3D annotation; one class per object, cross-view-consistent ids by
construction). Converted into the exact SAM3 on-disk layout by `scripts/build_official_masks.py`
(zero loader/tooling changes), QA'd (**200 scenes, 2950 instances, 0 cross-class duplicates,
label purity 1.0**; count < SAM3's ≈4195 because SAM3 double-counted duplicated objects), packed
as `…/scannet/scannet_official_gt_full.tar.zst` (2.3 GB). Spec: `…/scannet/OFFICIAL_GT_README.md`.
Differences vs SAM3 GT: masks written sparsely (missing PNG = not visible); stuff classes keep
official per-segment ids (`wall_0..k`, not forced `_0`); out-of-taxonomy objects (incl.
`otherfurniture`) → background. Smoke-tested end-to-end (overfit: loss falls, mIoU rises).

- **Both tars** contain 200 scenes (scene0000–0199), layout
  `scans/<scene>/raw_data/{subset,masks,masks_instance}`:
  - `scannet_official_gt_full.tar.zst` (2.3 GB, **default**) — official GT, 2950 instances.
  - `scannet_instance_dataset_full.tar.zst` (~2.6 GB) — SAM3 GT, ≈4195 instances (with the
    duplicate defect above); kept as the GT-quality baseline and as a project deliverable.
    Per-scene counts: `INSTANCE_MASKS_README.md` + `…_split2.md`.
- Mask conventions (both): `masks_instance/<class>_<k>/<frame>.png`, uint8 {0,255}, 1296×968,
  `<k>` per class by first appearance; union of a class's instances = its `masks/<class>/` mask.
- **Data access:** `slurm/stage_dataset.sh` copies ONE tar (selected by `DATA_TAR`; train SLURM
  scripts default it to the official tar, `sbatch --export=ALL,DATA_TAR=<sam3 tar>` restores the
  baseline) to node-local `$TMPDIR`, unpacks once, exports `SCANNET_ROOT=$TMPDIR/scans`;
  `train_multiscene.py` honors it as default `--scans_root`. SLURM headers request
  `--tmp=16000` MB. No unpacked `scans/` tree exists on work; the official-GT build tree is
  currently also unpacked at `/cluster/scratch/niacobone/scannet_official_build/scans` (scratch,
  purgeable — the tar on work is canonical). Old `scannet_build*` scratch trees are hollow.

## Storage layout
- Repo: `/cluster/scratch/niacobone/vggt`. Runs/checkpoints:
  `/cluster/work/igp_psr/niacobone/distillation/output/<run_name>/` (timestamped).
- `checkpoint_best.pth` (best val mIoU) = use for eval/demos; `checkpoint_best_ap50.pth` = same
  run on honest val[grid] AP50. Each run dir has `metrics.jsonl`.
- Checkpoints are self-contained (head weights + `head_config` + scene bundles + optim/sched for
  `--resume`); frozen backbone reloaded from HF `facebook/VGGT-1B`, never stored.
</content>
