# Milestone 2 — Toward Generalizable, Unprompted Multi-View Instance Segmentation

**Status:** 🟡 Code complete and validated on the 5 available scenes (see §6); the **scaling
experiments are blocked on data** (tens-to-hundreds of preprocessed scenes) — see §7.
**Goal:** Convert the Milestone-1 overfit pipeline into a *regularized training loop* whose
head can be used **without ground-truth query prompts**, and put in place everything needed
to obtain real generalization the moment more training scenes are available.

This milestone implements the four follow-ups identified in `MILESTONE_1.md` §9:

| §9 item | What Milestone 2 does |
|---|---|
| (b) supervise unmatched queries | **No-object loss** (DETR-style), `train/loss.py` |
| — (enabled by b) | **Unprompted inference/eval** on a uniform query grid |
| (c) augmentation / random frames | **Multi-bundle sampling + query/photometric jitter** |
| (d) early stopping on val mIoU | **Best-checkpoint tracking + optional early stop** |
| (a) scale training scenes | **Code-ready** (`--cache_device cpu`); experiments flagged in §6, pending data |

---

## 1. No-Object Loss (DETR-style) — `train/loss.py`

**Problem (M1 §9):** only *matched* queries received a class loss, so the head never learned
to push unmatched/background queries toward the background class. Consequences: AP was
dragged down by confident spurious detections, and inference fundamentally required
GT-ordered queries (you couldn't tell real detections from noise).

**Change:** `D4RTLoss(no_object_weight=...)` (default `None` = exact Milestone-1 behavior).
When set, the class loss is computed over **all** N queries:

- matched queries → their Hungarian-matched GT class (weight 1.0);
- unmatched queries → `background_class` (index 0), each down-weighted by
  `no_object_weight` (DETR's `eos_coef`, default **0.1** in the training script) so the many
  background queries don't drown out the few matched ones.

The per-query focal loss is weighted and normalized by the total weight. Mask/Dice/BCE
supervision is unchanged (still matched-only). Matching itself is also unchanged, so all
Milestone-1 tests pass bit-identically with `no_object_weight=None`.

**Validated by** `tests/test_milestone2.py`: disabled-mode equivalence with the old loss,
"background-on-unmatched is cheaper than foreground-on-unmatched", and gradient flow to
*every* query row (not just matched ones).

## 2. Unprompted (Grid) Queries — `scripts/train_overfit.py::generate_grid_queries`

With no-object supervision in place, the model can be queried **without GT information**:
`generate_grid_queries(S, grid_size)` lays a `grid_size × grid_size` lattice of cell-center
points over **every frame** (default 6×6 → 36 queries/frame, 288 for 8 frames). Each query
is just (cell center, frame id) — no centroids, no GT ordering. Predictions whose argmax
class is background are dropped by the (unchanged) eval metrics; what remains is scored as
detections.

Every evaluation in `train_multiscene.py` now reports **two** metric sets per scene:

- **prompted** — queries at GT centroids (Milestone-1 protocol): "given a point on the
  object, segment + classify it";
- **unprompted** — the uniform grid: "find, segment and classify the instances yourself".

The unprompted numbers are the honest detection metrics; expect them to be lower than the
prompted ones (multiple grid cells can fire on the same instance and count as duplicates
under AP — a deduplication/NMS step or learned object queries are future work, see §7).

## 3. Regularization — `scripts/train_multiscene.py`

Milestone 1 trained on **one fixed batch per scene** (deliberately: an overfit test).
Milestone 2 replaces that with cheap, cache-friendly augmentation while keeping the
"frozen backbone runs only up front" efficiency:

- **`--bundles_per_scene K`** (default 1 = M1 behavior): each train scene gets K cached
  bundles. Bundle 0 uses **evenly-spaced frames** (deterministic — it is the eval and
  checkpoint bundle); bundles 1..K-1 use **random frame sampling**, so across epochs the head
  sees different views of each scene. Each bundle pays one frozen-backbone pass at startup.
- **`--color_jitter s`**: random brightness/contrast (one draw per random bundle) applied
  *before* the backbone pass, so the cached features are consistent with the stored images.
- **`--query_jitter σ`** (Gaussian, clamped to [0,1]): instance-centroid queries are
  perturbed **every step** — the head can no longer key on exact centroid positions.
- **Background queries are resampled every step** (disable with `--fixed_bg`), which
  together with the no-object loss gives dense, varied background supervision.
- **`--cache_device cpu`**: bundles (features ~90 MB/bundle at 8 frames) can live in host
  memory and are moved to the GPU per step — scene count is no longer bounded by GPU memory.
  Defaults to the training device (no overhead) for small runs.

Per-epoch training cost is unchanged (one head forward/backward per scene; the bundle is
*sampled*, not stacked).

## 4. Model Selection — best checkpoint + early stopping

M1 §9 observed val mIoU peaking mid-training (~epoch 600) and then decaying — the signal
existed but the final checkpoint had already overfit past it. Now, at every `--eval_interval`:

- mean **val prompted mIoU** is computed (falls back to train mIoU when no val scenes);
- on improvement, the full checkpoint is saved as **`checkpoint_best.pth`** next to
  `--save_checkpoint` (same demo-compatible format, including both prompted and unprompted
  metrics and `best_info = {val_mIoU, epoch}`);
- with `--early_stop_patience P > 0`, training stops after P consecutive evals without
  improvement.

The final checkpoint is still written at the end; `checkpoint_best.pth` is the one to use
for evaluation/demos.

## 5. Files Changed / Added

| File | Change |
|------|--------|
| `train/loss.py` | `no_object_weight` / `background_class` options in `D4RTLoss` (default off → M1-identical) |
| `scripts/train_overfit.py` | `generate_grid_queries` (unprompted query lattice) |
| `scripts/train_multiscene.py` | multi-bundle scene cache, per-step query augmentation, photometric jitter, unprompted eval, best-checkpoint + early stopping, `--cache_device` |
| `tests/test_milestone2.py` | standalone tests for all of the above (CPU, no backbone weights) |

New CLI flags (`train_multiscene.py`): `--no_object_weight` (default 0.1), `--grid_size`
(default 6), `--bundles_per_scene` (default 1), `--query_jitter`, `--fixed_bg`,
`--color_jitter`, `--early_stop_patience`, `--cache_device`.
Milestone-1 behavior is exactly recovered with
`--no_object_weight 0 --bundles_per_scene 1 --query_jitter 0 --fixed_bg`.

### How to run

```bash
# Milestone 2 unit tests + full regression suite
python tests/test_milestone2.py
python tests/test_phase2.py ... tests/test_phase5.py tests/test_eval.py

# Regularized multi-scene training with no-object loss + unprompted eval
python scripts/train_multiscene.py \
    --train_scenes scene0000_00,scene0001_00,scene0002_00,scene0003_00 \
    --val_scenes scene0004_00 \
    --num_epochs 1000 --warmup_epochs 30 --num_frames 8 --num_queries 32 \
    --learning_rate 2e-3 --bundles_per_scene 3 --query_jitter 0.02 --color_jitter 0.2 \
    --no_object_weight 0.1 --grid_size 6 --eval_interval 50 \
    --save_checkpoint /cluster/work/igp_psr/niacobone/distillation/output/<run>/checkpoint.pth
```

## 6. Validation Run (5 available scenes)

**Run:** `d4rt_m2_5scenes_20260610_133100` — 4 train scenes + held-out `scene0004_00`,
1000 epochs, 8 frames/bundle, 32 queries, lr 2e-3 (30-epoch warmup → cosine),
`--bundles_per_scene 3 --query_jitter 0.02 --color_jitter 0.2 --no_object_weight 0.1
--grid_size 6`. 13 backbone passes up front (~20 s total), training 2.2 min on one
RTX 4090. Checkpoints (final + best):
`/cluster/work/igp_psr/niacobone/distillation/output/d4rt_m2_5scenes_20260610_133100/`.
Loss/scene fell 2.79 → 0.59 (−78.8%).

Final-epoch metrics (prompted = queries at GT centroids; **unprompted** = 6×6 grid/frame,
no GT information):

| Scene | Split | mIoU (prompted) | AP50 (prompted) | mIoU (unprompted) | AP50 (unprompted) | class_acc |
|-------|-------|------|------|------|------|-----------|
| scene0000_00 | train | 0.507 | 0.567 | 0.601 | 0.281 | 0.700 |
| scene0001_00 | train | 0.754 | 0.900 | 0.765 | 0.417 | 0.900 |
| scene0002_00 | train | 0.540 | 0.618 | 0.613 | 0.478 | 0.900 |
| scene0003_00 | train | 0.862 | 1.000 | 0.732 | 0.608 | 1.000 |
| **mean (train)** | | **0.666** | **0.771** | **0.678** | **0.446** | **0.875** |
| scene0004_00 | **val** | 0.109 | 0.125 | 0.172 | 0.009 | 0.333 |

Best-checkpoint tracking fired 5 times; **best val prompted mIoU 0.138 @ epoch 450**,
preserved in `checkpoint_best.pth` (the final-epoch val mIoU had decayed to 0.109 — exactly
the M1 §9 overfitting pattern, now captured instead of lost).

**Reading the numbers (vs Milestone 1 §9):**

- **The no-object loss works.** Train *prompted* AP50 jumped **0.54 → 0.77** even though
  train mIoU *dropped* (0.97 → 0.67, expected — the head can no longer memorize one fixed
  batch thanks to bundle/query augmentation). In M1, AP was crushed by background queries
  predicting foreground; now they predict background and get filtered out.
- **Unprompted inference is real.** With zero GT information, the grid queries reach train
  mIoU **0.678 — on par with prompted (0.666)**. The no-object head keeps only ~90 of 288
  grid queries as detections (≈⅔ correctly pushed to background). Unprompted AP50 (0.45
  train) trails prompted because several grid cells fire on the same instance and the
  duplicates count as false positives — a dedup/NMS pass or learned object queries are the
  obvious next step (§7.4).
- **Generalization is still data-limited, as expected with 4 scenes.** Best val mIoU 0.138
  is slightly above M1's mid-training peak (~0.13) and far above M1's *final* 0.027; the
  gain here is from model selection + regularization, not from real generalization. The
  scaling experiment (§7.1) is the actual test.
- Throughput is unchanged: the multi-bundle cache keeps the per-epoch cost at one head
  forward/backward per scene.

## 7. ⚑ Experiments Pending More Scenes

The following are **flagged for execution once more scenes are downloaded and
preprocessed** (SAM3 masks). The code requires no changes — only the scene lists:

1. **Scaling curve (the §9(a) experiment).** Train with N ∈ {10, 25, 50, 100+} scenes,
   1–3 held-out val scenes, `--bundles_per_scene 3-4`, `--early_stop_patience 5`. The key
   question: does held-out prompted mIoU climb with N (M1 plateaued at ~0.13 with N=4)?
   Use `--cache_device cpu` once bundles exceed GPU memory (~90 MB × scenes × bundles).
2. **No-object weight sweep** (0.05 / 0.1 / 0.4) — only meaningful with enough scenes that
   the head can't memorize; measures the precision/recall trade-off on **unprompted** AP50.
3. **Augmentation ablation** — bundles_per_scene 1 vs 4, query_jitter on/off, color_jitter
   on/off; measured on held-out scenes. With 4 scenes these all just slow the memorization;
   the comparison needs a val signal above noise.
4. **Grid density vs unprompted recall** — `--grid_size` 4/6/8; small objects may fall
   between cells at 6×6. Decide if a dedup/NMS pass before AP is warranted.
5. **Same-class instance separation** (M1 §8.3 limitation) — needs instance-encoded masks
   from the preprocessing side, not just more scenes. Flagging here so the preprocessing
   pipeline can consider emitting per-instance (not per-class) masks.

Items beyond data: learned object queries (drop the point-prompt requirement entirely,
closer to true DETR), mask upsampling above the 37×37 patch grid (FPN-style pixel decoder),
and partial backbone unfreezing once the dataset is large enough to support it.
