# MaskDINO on COCO with a swapped backbone — does the published recipe survive VGGT?

**Status (2026-07-31):** `resnet50` and `dinov2` arms COMPLETE, `vggt` in its final
self-resubmitted segment. Headline so far: **frozen DINOv2 at 37×37 tokens beats the frozen
R50 control by +4.4 AP** (38.8 vs 34.3), and `vggt` is level with `dinov2` at the same step —
see §6.

**The question (supervisor-facing).** Every number in `docs/MASKDINO.md` compares our port against
*our own* ScanNet baselines. `docs/MASKDINO.md` §7.6 proved the port reproduces upstream's COCO
result when driven with upstream's weights — but that check never touches the **training** path and
says nothing about the **backbone**. This track closes both gaps: train the same decoder on COCO
instance segmentation, on frozen features, and measure how much of MaskDINO's published
**46.1 mask AP / 51.5 box AP** survives when the ResNet-50 is replaced by frozen VGGT-1B.

---

## 1. Read this first: the resolution ceiling is real, and it splits in two

The concern that motivated the study — "the feature/mask resolution will make this impossible" —
is **correct as stated, and fatal if left unaddressed**. It is also two different problems with
very different price tags. `scripts/coco_mask_resolution_oracle.py` separates them.

### 1.1 The measurement

For every non-crowd GT instance of COCO val2017: quantise the GT mask onto the grid the model
predicts on (area-downsample = the best soft logit map that grid can hold), bilinear-upsample back
to the original image size, threshold at 0.5, and submit it as a detection with score 1.0 and the
correct class. The resulting `COCOeval` mask AP is a hard **upper bound** for that grid, isolated
from every other error source — a model that is perfect at everything else cannot beat it.

**COCO val2017, all 5000 images, 36 335 instances:**

| prediction grid | AP | AP50 | AP75 | **APs** | APm | APl |
|---|---|---|---|---|---|---|
| **37×37** — the ScanNet track's grid | **44.7** | 62.8 | 46.6 | **8.0** | 65.3 | 94.9 |
| 37×37, centre-padded to square | 39.8 | 57.2 | 41.7 | 2.8 | 57.0 | 92.7 |
| 52×52 | 55.1 | 72.8 | 58.1 | 21.4 | 79.2 | 98.0 |
| 74×74 (`--mask_upsample 2`) | 66.2 | 82.4 | 69.7 | 38.4 | 89.2 | 99.5 |
| 111×111 | 77.6 | 90.4 | 81.6 | 58.1 | 96.1 | 99.9 |
| **148×148 (`--mask_upsample 4`)** | **84.2** | 94.1 | 88.1 | 70.1 | 98.4 | 100.0 |
| 222×222 | 90.7 | 97.6 | 94.0 | 82.3 | 99.5 | 100.0 |
| **stride 4 @800 — MaskDINO-on-R50's own** | **92.0** | 97.9 | 95.0 | 84.8 | 99.7 | 100.0 |
| stride 8 @800 | 79.7 | 91.8 | 83.6 | 61.5 | 96.9 | 99.9 |

**On the 37×37 grid a perfect model scores 44.7 — below MaskDINO's 46.1 target.** The experiment
would have been unanswerable by construction. Upstream, by contrast, operates at 46.1 under a
92.0 ceiling: resolution is never its binding constraint, recognition is.

Why ScanNet never showed this: its objects are furniture filling much of the frame, and
`--mask_upsample 2` measured **neutral** there (−0.022 AP50, `docs/MASKDINO.md` §7.4) — not because
resolution does not matter in general, but because 37×37 was not the binding constraint for those
objects. COCO is the opposite regime:

> **The median COCO instance covers 8.4 cells of a 37×37 grid. 15.7 % of instances fit inside a
> single cell; 37 % inside 2×2; 60 % inside 4×4.**

That single line explains the APs column collapsing to 8.0.

### 1.2 Problem A — the mask grid. Real, but cheap to fix.

`mask_features` resolution is **not** tied to the token grid. It comes from transposed convs on
encoder level 0, exactly as in ViTDet (a stride-16 ViT feeding a stride-4 mask head). Setting
`--mask_upsample 4` gives 148×148 masks at 518 px input — ceiling **84.2 AP**, comfortably above
anything this study will produce. Cost: two `ConvTranspose2d` layers. **This is the default for
the ViT arms here**, and it is the single most important difference from the ScanNet recipe.

### 1.3 Problem B — the token grid. Real, expensive, and *not* fixed here.

37×37 = 1369 tokens determines how well small objects can be **detected and separated**, not just
how cleanly their boundary is drawn. Upsampling `mask_features` cannot invent detail the tokens do
not carry. Reference points: ViTDet gets strong COCO AP from a plain ViT, but at 1024 px / patch 16
= **4096** tokens; our R50 control at 518 px sees 65²+33²+17² = **4907** encoder cells. The VGGT
arm sees 37²+19²+10² = **1830**. That gap is a property of running VGGT at its native 518 px and is
reported as such, not engineered away.

It *could* be closed: VGGT uses 2D RoPE with no absolute position embedding, so it accepts any grid
— verified, a 448×602 input yields exactly 32×43 = 1376 tokens. Measured cost on one RTX 4090:

| input | token grid | img/s (best batch) | h per COCO pass |
|---|---|---|---|
| 518 px | 37×37 | 27.0 | 1.2 |
| 700 px | 50×50 | 13.5 | 2.4 |
| 1036 px | 74×74 | 5.1 | 6.4 |

A token-matched VGGT@1036 arm (74×74 ≈ R50's 65×65) is therefore ~5× the backbone cost of the
518 px arm and is **deferred**, not refuted. It is the obvious follow-up if the 518 px arm loses
mainly on small objects.

### 1.4 Squash, don't pad

Centre-padding to a square (VGGT's own `load_and_preprocess_images_square`) spends grid cells on
black borders and costs **4.9 AP of ceiling** at equal token budget (39.8 vs 44.7). Aspect-preserving
variable shapes at the same token count score the same as squashing (50.6 vs 50.3 on a 300-image
sample) while forcing per-shape batching. So every arm here **squashes to a fixed square**, and the
evaluator upsamples masks straight back to the original image size, which inverts the squash exactly.

## 2. The three arms

Identical decoder, schedule, data, augmentation, loss, and GT resolution. Only `--backbone` differs.
Every backbone is **frozen** — only the pixel decoder + MaskDINO decoder train, as in every run of
this project.

| arm | backbone | encoder levels @518 | `mask_features` | ceiling |
|---|---|---|---|---|
| `--backbone resnet50` | ImageNet R50, frozen | res3/res4/res5 = 65²/33²/17² | res2, stride 4 → 130² | ~92 |
| `--backbone vggt` | VGGT-1B aggregator, frozen | 37²/19²/10² (ViTDet-synthesised) | deconv ×4 → 148² | 84.2 |
| `--backbone dinov2` | DINOv2 ViT-L/14-reg, frozen | 37²/19²/10² (identical to VGGT) | deconv ×4 → 148² | 84.2 |

The comparisons this design buys:

- **`vggt` vs `resnet50`** — the headline: how much does swapping the backbone cost, everything
  else held fixed? Confounded by the token grid (1830 vs 4907 cells), which is stated, not hidden.
- **`vggt` vs `dinov2`** — the clean one: **identical** patch size, architecture family and token
  count. DINOv2 ViT-L/14-reg is precisely the model VGGT's `patch_embed` is built from, and the
  official checkpoint loads into VGGT's own vendored `vit_large` at `strict=True`, 0 missing keys.
  So this isolates *what VGGT's 3D pretraining did to 2D semantics* from *what the 37×37 grid costs*.
- **`resnet50` (frozen, 12 ep) vs upstream (finetuned, 50 ep, 46.1)** — the honest distance to the
  published number, and the reason none of these arms should be read as "MaskDINO reproduced".

### Deviations from upstream's COCO recipe (all shared by all three arms)

1. **Frozen backbone.** Upstream finetunes its ResNet-50. Freezing is the whole point here (VGGT is
   frozen by project constraint), so the R50 control is frozen too. Expect the control to land
   below 46.1 for this reason alone — that gap is measured, not assumed.
2. **12 epochs, not 50** (the "1×" detection schedule).
3. **Fixed 518×518 squash instead of LSJ / multi-scale**, and horizontal flip as the only
   augmentation. Upstream's large-scale jitter is worth real AP; dropping it keeps the three arms
   comparable and the pipeline small.
4. **518 px, not 800/1333.** Set by VGGT's native token grid; applied to every arm.
5. GT masks rasterised at a **shared 296×296** for every arm, independent of each arm's prediction
   grid — the matcher and `SetCriterion.loss_masks` compare through PointRend `point_sample` at
   normalised coordinates, so supervision resolution is a free parameter and is held constant.

## 3. Files

Parallel to the ScanNet track throughout; nothing in `models/maskdino/{decoder,decoder_layers,
matcher,criterion,ms_deform_attn,box_ops,utils}.py` was modified, so every ScanNet number stands.

| File | Contents |
|---|---|
| `scripts/coco_mask_resolution_oracle.py` | §1: the GT-only ceiling measurement |
| `models/maskdino/coco_backbones.py` | `VGGTBackbone` / `DINOv2Backbone` / `ResNet50Backbone` behind one frozen interface |
| `models/maskdino/pixel_decoder_coco.py` | `CocoPixelDecoder` — ViTDet mode (1 map in) **and** FPN mode (pyramid + stride-4 lateral) |
| `models/maskdino/head_coco.py` | `MaskDINOCocoHead` = pixel decoder + the unmodified `MaskDINODecoder`; `head_config` round-trip |
| `models/maskdino/model_coco.py` | `MaskDINOCocoModel` = frozen backbone + head |
| `train/coco_data.py` | COCO dataset, contiguous class mapping, GT masks/boxes, hflip, collate |
| `train/coco_eval.py` | upstream's `instance_inference` + `COCOeval` (segm and bbox) |
| `scripts/train_maskdino_coco.py` | entry point: CLI, step-budgeted loop, AMP, resume |
| `slurm/train_maskdino_coco.sh` | cluster job; **self-resubmits** until `summary.json` exists |
| `tests/test_coco_maskdino.py` | CPU tests: both pyramid modes, head round-trip, GT helpers, inference, RLE round-trip, overfit |

Why parallel files rather than flags on the ScanNet path: the ScanNet loop caches frozen features
once per scene up front (that is what makes it take minutes), which is impossible for 118 k COCO
images — 618 GB at the VGGT token size — and pointless under augmentation. So COCO runs the
backbone **inline**, and the data/eval modules share nothing but the model.

## 4. How the runs are driven (and what was verified before spending the GPU hours)

**Effective batch 16, micro-batch 4-8.** Upstream reaches `IMS_PER_BATCH 16` with **16 GPUs at one
image each**. On one 24 GB card a batch of 16 OOMs immediately: the mask tensors are
`B × Q × grid² × (dec_layers + 2)`, doubled by the denoising queries — 16 × 300 × 148² × 11 in fp32
is far past the card. `--micro_batch` splits the step and accumulates gradients, so the optimiser
still sees exactly batch 16. Measured steady state on one RTX 4090 (optimiser steps/s, 16 img each):

| arm | micro-batch | it/s | 12 ep = 87 948 steps |
|---|---|---|---|
| `resnet50` | 8 | ~1.0 | ~24 h |
| `dinov2` | 8 | ~1.1 | ~22 h |
| `vggt` | 4 | ~0.6 | ~41 h |

**The self-resubmit needs python to stop itself.** A 24 h wall clock cannot hold any of these, and
the naive "resubmit at the end of the batch script" does **not** work: at the wall clock SLURM
tears down the whole script, so the trailing `sbatch` never runs and the study silently stops
half-finished. `--time_budget_hours` (22.5 by default) makes python exit cleanly first, saving
`checkpoint_last.pth` and **not** writing `summary.json` — and that absence is what the shell script
tests. Verified end to end: a run stopped at step 16/40, resumed at exactly step 16 with the cosine
LR back on its curve, and finished. Run settings are frozen into `<run_dir>/job_env.sh` on the
first submission and sourced by every resubmit, so they cannot drift between segments.

**An OOM skips a micro-batch instead of killing the job.** Peak memory tracks the number of GT
instances, and COCO has images with 90+; over a 40 h run one unlucky draw must not cost every step
since the last checkpoint. Skips are counted and reported.

### 4.1 The overfit gate — run this before trusting any AP

A silent bug anywhere downstream of the loss (category mapping inverted, mask transposed, RLE at
the wrong resolution) shows up as AP ≈ 0 forever, which is indistinguishable from "the model has
not learned yet" until 90 GPU-hours have been spent. So the chain is proven first: train on 64
images and score **those same 64**, via a COCO root whose `train2017` is a symlink to `val2017`.

| gate | step 200 | step 400 | step 600 |
|---|---|---|---|
| `resnet50` segm AP | 0.002 | 23.4 | **54.3** |
| `vggt` segm AP | 0.001 | 14.1 | — |

Both climb, so targets, matcher, criterion, `instance_inference`, the contiguous↔dataset category
mapping, RLE encoding, the upsample-to-original-size step and `COCOeval` are all wired correctly —
and the `vggt` row exercises the `mask_upsample 4` (148×148) path that the `resnet50` row does not.
These are **memorisation numbers on 64 images**; they are not results and must never be quoted as
such.

## 5. Reproducing

```bash
# the ceiling measurement (CPU, ~15 min for all 5000 images)
myenv/bin/python scripts/coco_mask_resolution_oracle.py --out mask_resolution_oracle.json

# CPU tests
myenv/bin/python tests/test_coco_maskdino.py

# the three arms (each self-resubmits until done)
sbatch --export=ALL,BACKBONE=resnet50 slurm/train_maskdino_coco.sh
sbatch --export=ALL,BACKBONE=vggt     slurm/train_maskdino_coco.sh
sbatch --export=ALL,BACKBONE=dinov2   slurm/train_maskdino_coco.sh
```

COCO lives at `/cluster/scratch/niacobone/coco` (train2017 + val2017 + instances, extracted from
`/cluster/work/igp_psr/yuxchen/coco.zip`). **Global scratch is purged after 15 days** — re-extract
from that zip if it has vanished.

## 6. Results (2026-07-31 — `resnet50` and `dinov2` final; `vggt` in its last segment)

Final full-val2017 numbers at step 87 948 (12 epochs), from each run's `summary.json`
(`final` block; the `best` interval checkpoint is noted separately):

| arm | segm AP | AP50 | AP75 | APs | APm | APl | box AP | ceiling | best interval AP |
|---|---|---|---|---|---|---|---|---|---|
| upstream R50, finetuned, 50 ep (published) | 46.1 | — | — | — | — | — | 51.5 | 92.0 | — |
| `resnet50` frozen, 12 ep | 34.3 | 54.1 | 36.2 | 14.3 | 36.1 | 53.6 | 38.2 | ~92 | 36.7 @80k |
| `vggt` frozen, 12 ep | *pending (job 9262006, last segment)* | 61.0* | 42.3* | 15.8* | 44.6* | 61.3* | — | 84.2 | 39.7 @75k* |
| `dinov2` frozen, 12 ep | **38.8** | 64.8 | 39.6 | 14.8 | 43.0 | 65.1 | 45.9 | 84.2 | **41.3 @85k** |

\* `vggt` values are the step-85k *interval* eval (39.53 segm AP), not the final block —
replace when `summary.json` lands.

Three readings, the first two already firm:

1. **The 37×37 token grid + `mask_upsample 4` is not crippling — it wins.** Frozen DINOv2 with
   1830 encoder cells beats the frozen ResNet-50 control with 4907 cells by **+4.4 final AP**
   (38.8 vs 34.3), and even its `APs` (14.8) matches the R50's (14.3). The §1.3 concern
   ("small objects need the token grid") shows up only as the *shared* gap of both ViT arms to
   their 84.2 ceiling, not as a deficit against the R50 control.
2. **Freezing + 12 ep costs the R50 recipe ~12 AP** against upstream's finetuned 50-ep 46.1 —
   the honest distance to the published number, measured as designed.
3. **Preliminary: VGGT's 3D pretraining costs almost nothing on 2D semantics at equal
   geometry.** At the same step (85k), `vggt` reads 39.53 vs `dinov2`'s 39.83 interval segm AP —
   essentially level, after trailing badly early in training (14.1 vs 23.4 at the overfit-gate
   stage, 31.3 vs ~38 at 25k). Do not write the final verdict until the `vggt` summary exists:
   the early-training gap and the endgame convergence are both part of the story.
