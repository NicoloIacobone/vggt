# MaskDINO on COCO with a swapped backbone — does the published recipe survive VGGT?

**Status (2026-08-08):** all three arms COMPLETE. Headline: **frozen DINOv2 at 37×37 tokens
beats the frozen R50 control by +4.4 AP** (38.8 vs 34.3), and **`vggt` trails `dinov2` by
−1.2 AP** (37.7 vs 38.8 final), both reaching ceiling-constrained final steps after converging
late — see §6.

A fourth arm **completed 2026-08-12**: **upstream's own MaskDINO under this recipe**, which turns
§6's "distance to 46.1" from an inference against a released checkpoint into a measurement, and is
the first check of the port's **training** path. It landed at **34.55 segm AP against our own
`resnet50` arm's 34.3** — so the recipe costs ~11.6 AP and our matcher/criterion/DN are
corroborated end to end. See `third_party/maskdino_control/README.md` and §6 reading 2 / §6.1.

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
- **`resnet50` (frozen, 12 ep) vs upstream** — the distance to the published number, and the
  reason none of these arms should be read as "MaskDINO reproduced". Note that 46.1 is a released
  *checkpoint* we only ever ran inference on, so this comparison needs a fourth arm to mean
  anything: **upstream's own code under this exact recipe** (`third_party/maskdino_control/`,
  §6 row 2). Without it, "frozen + 12 ep costs ~12 AP" confounds the schedule, the freezing and
  the input resolution.

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
| `third_party/maskdino_control/` | §6's upstream control row: official MaskDINO under this recipe. Own README; imports the pristine clone, never edits it |
| `slurm/train_maskdino_upstream.sh` | that run's cluster job (A100 80 GB; same self-resubmit contract) |
| `tests/test_maskdino_upstream_control.py` | its CPU tests — mapper geometry, LR parity against `train_maskdino_coco.py`'s own lambda, every config axis. Needs the **reference** env |
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
| upstream control (§6, job 10093469) | 0.275 | 22.4 | **52.1** |

Both climb, so targets, matcher, criterion, `instance_inference`, the contiguous↔dataset category
mapping, RLE encoding, the upsample-to-original-size step and `COCOeval` are all wired correctly —
and the `vggt` row exercises the `mask_upsample 4` (148×148) path that the `resnet50` row does not.
These are **memorisation numbers on 64 images**; they are not results and must never be quoted as
such.

**The gate measures the LR schedule as much as the pipeline — set it deliberately.** Reproducing
the row above for the upstream control took three attempts, and the two failures were pure
schedule artefacts that are indistinguishable from a broken loss:

| gate LR schedule | step 200 / 400 / 600 |
|---|---|
| 1000-step warmup (the real run's), so lr only ramps to 6e-5 | 0.000 / 0.000 / 0.838 |
| 10-step warmup but the cosine's horizon left at 600, so the endgame runs at ~1e-6 | 0.000 / 21.4 / 28.0 |
| 10-step warmup **and** the cosine's horizon at the real 87 948, so lr ≈ 1e-4 throughout | 0.275 / 22.4 / **52.1** |

A 600-step gate must therefore hold lr near its peak — `CONTROL.LR_HORIZON_ITERS` exists only for
that, and the real run leaves it at 0 (≡ `MAX_ITER`). A gate that undershoots for this reason
proves nothing either way, so read the LR before concluding anything from a low number.

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

# the upstream control row of §6 — official MaskDINO, this recipe (third_party/maskdino_control/)
bash third_party/maskdino_control/build_ops.sh                     # ONCE: MSDeformAttn for sm_80
python third_party/maskdino_control/make_overfit_root.py --n 64     # ONCE: the §4.1 gate's root
sbatch --time=4:00:00 --export=ALL,GATE=1 slurm/train_maskdino_upstream.sh   # the gate
sbatch slurm/train_maskdino_upstream.sh                                      # 87 948 iters
```

COCO lives at `/cluster/scratch/niacobone/coco` (train2017 + val2017 + instances, extracted from
`/cluster/work/igp_psr/yuxchen/coco.zip`). **Global scratch is purged after 15 days** — re-extract
from that zip if it has vanished.

## 6. Results (2026-08-12 — all three arms AND the upstream control complete)

Final full-val2017 numbers at step 87 948 (12 epochs), from each run's `summary.json`
(`final` block; the `best` interval checkpoint is noted separately).

> ⚠ **The last column is a DIFFERENT POPULATION from every column left of it, and the two are not
> comparable.** `segm AP`…`box AP` are the **full 5000-image** val2017. `best interval AP` comes
> from a periodic eval, and periodic evals score **1000 images** (`--eval_images`, the first 1000
> sorted image ids; `train/coco_eval.py` restricts `COCOeval.params.imgIds` to what it saw). That,
> and nothing else, is why every arm appears to "drop" ~2 AP at its last step — 36.7→34.3,
> 39.7→37.7, 41.3→38.8. It is a protocol change, **not** a late-training regression, and the
> per-arm consistency of the offset is the evidence. Never subtract across the divide, and never
> place a 1000-image number next to a published one. The upstream control row is built to the same
> split on purpose (`CONTROL.VAL_SUBSET_JSON`, `TEST.EVAL_PERIOD 5000`), so its curve is readable
> against our arms' curve at a matched step **and** its final number against theirs.

| arm | segm AP | AP50 | AP75 | APs | APm | APl | box AP | ceiling | best interval AP **(1000 img)** |
|---|---|---|---|---|---|---|---|---|---|
| upstream R50, finetuned, 50 ep — **released checkpoint, our inference** (§7.6) | 46.1 | — | — | — | — | — | 51.5 | 92.0 | — |
| **upstream MaskDINO, THIS recipe** (frozen R50, 12 ep, 518 squash) — jobs 10094393 → 10279969 → 10427048 | **34.5** | 54.6 | 36.5 | 13.1 | 36.5 | 57.2 | 38.3 | ~92 | 36.9 @85k |
| `resnet50` frozen, 12 ep | 34.3 | 54.1 | 36.2 | 14.3 | 36.1 | 53.6 | 38.2 | ~92 | 36.7 @80k |
| `vggt` frozen, 12 ep | 37.659 | 59.384 | 39.512 | 15.253 | 41.555 | 58.524 | 42.065 | 84.2 | 39.7 @75k |
| `dinov2` frozen, 12 ep | **38.8** | 64.8 | 39.6 | 14.8 | 43.0 | 65.1 | 45.9 | 84.2 | **41.3 @85k** |

Three readings, the first two already firm:

1. **The 37×37 token grid + `mask_upsample 4` is not crippling — it wins.** Frozen DINOv2 with
   1830 encoder cells beats the frozen ResNet-50 control with 4907 cells by **+4.4 final AP**
   (38.8 vs 34.3), and even its `APs` (14.8) matches the R50's (14.3). The §1.3 concern
   ("small objects need the token grid") shows up only as the *shared* gap of both ViT arms to
   their 84.2 ceiling, not as a deficit against the R50 control.
2. **The distance to 46.1 is being MEASURED, not inferred — row 1 is a checkpoint, not a run.**
   Read row 1's label: 46.1 / 51.5 is upstream's **released checkpoint**, scored by our own
   inference (`docs/MASKDINO.md` §7.6, job 8967932: 46.129 unmodified / 46.133 ported). No
   MaskDINO has ever been *trained* in this project, so "freezing + 12 ep costs ~12 AP" was an
   inference against a differently-trained model, confounding three things at once: 50 epochs vs
   12, a finetuned R50 vs a frozen one, and LSJ@1024 vs squash@518. Upstream's README also fences
   that row as COCO-only ("clean models that do not use extra detection data or tricks") — only
   its Swin-L 54.5 row uses Objects365 — so extra data is *not* part of the gap.
   **Row 2 removes the confound**: upstream's own code, our recipe, every axis we control matched
   (`third_party/maskdino_control/`). **LANDED 2026-08-12 at 34.55 segm AP** — so the answer is
   **the recipe costs ~11.6 AP** (46.1 → 34.5), and it is now a measurement, not an inference.
   Everything below the 46.1 row is that recipe: frozen backbone, 12 epochs, squash@518.

   Row 2 is also the first **training**-path check of the port, and it passes. §7.6 certifies
   inference only and explicitly excludes `matcher.py`, `criterion.py` and DN generation; the
   stated failure criterion was "if row 2 lands far above our `resnet50` arm's 34.3, our training
   path has a bug". It landed at **34.55 against 34.3 — Δ +0.25 AP**, with the per-metric spread
   equally small (AP50 +0.5, AP75 +0.3, box AP +0.1) and only `APl` moving materially (+3.6, with
   `APs` −1.2 the other way: upstream's mapper favours large objects slightly, ours small ones).
   **Two independently written matchers, criteria and DN generators converge to a quarter of an AP
   over 87 948 steps of real COCO training.** Those three modules are corroborated end to end, and
   the §4.1 gate (52.1 vs 54.3) plus the full matched-step curve in §6.1 say the same thing three
   ways. No bug to report.
3. **VGGT's 3D pretraining costs ~1–1.6 AP on 2D semantics at identical token geometry.** Best
   checkpoint: `vggt` 39.7 @75k vs `dinov2` 41.3 @85k; final step: 37.659 vs 38.8. Both arms
   start wide apart (14.1 vs 23.4 at overfit-gate), converge gradually through mid-training, and
   diverge in the endgame — `vggt` plateaus at 75k while `dinov2` climbs to 85k. The gap
   reflects 3D domain shift, not token scarcity: both arms beat the frozen R50 control's 34.3.

### 6.1 The matched-step curve — the control against our arms, step for step

Row 2's periodic evals use **our arms' protocol exactly** — every 5000 steps, the same first 1000
val2017 images — so every control point can be laid against the columns below at the same step.
This was written to be readable before the run finished; it is now the complete curve.

`segm AP` on the 1000-image periodic split (`num_images == 1000` in each run's `metrics.jsonl`;
`metrics.json` for the control):

| step | `resnet50` frozen | `vggt` frozen | `dinov2` frozen | upstream control | Δ vs `resnet50` |
|---|---|---|---|---|---|
| 5 000 | 17.48 | 13.42 | 25.35 | **17.10** | −0.38 |
| 10 000 | 22.77 | 21.17 | 32.27 | **23.61** | +0.84 |
| 15 000 | 25.85 | 26.47 | 34.01 | **26.12** | +0.27 |
| 20 000 | 27.80 | 28.76 | 35.84 | **28.55** | +0.75 |
| 25 000 | 29.70 | 31.31 | 37.12 | **29.95** | +0.25 |
| 30 000 | 31.25 | 32.76 | 37.20 | **31.59** | +0.34 |
| 35 000 | 32.73 | 34.15 | 37.76 | **32.50** | −0.23 |
| 40 000 | 33.26 | 35.78 | 39.06 | *(eval lost — see below)* | — |
| 45 000 | 33.89 | 36.94 | 38.99 | **33.84** | −0.05 |
| 50 000 | 34.99 | 38.05 | 39.32 | **34.88** | −0.11 |
| 55 000 | 35.00 | 37.81 | 39.83 | **35.03** | +0.03 |
| 60 000 | 35.68 | 38.59 | 40.45 | **35.76** | +0.08 |
| 65 000 | 35.67 | 38.56 | 41.02 | **36.10** | +0.43 |
| 70 000 | 35.99 | 39.28 | 41.07 | **36.65** | +0.66 |
| 75 000 | 36.46 | 39.73 | 41.05 | **36.64** | +0.18 |
| 80 000 | 36.66 | 39.53 | 41.06 | **36.86** | +0.20 |
| 85 000 | 36.50 | 39.53 | 41.33 | **36.92** | +0.42 |
| best | 36.66 @80k | 39.73 @75k | 41.33 @85k | **36.92 @85k** | +0.26 |
| **final, full 5000 img** | **34.3** | **37.66** | **38.8** | **34.55** | **+0.25** |

**Complete, 2026-08-12.** The two implementations agree to **within ±0.84 AP at all 16 matched
steps** (mean Δ **+0.23**, sign changing four times, no drift with training length), and the two
finals agree to **+0.25 AP**. That is independently written matchers, criteria and DN generation
converging over 88 000 steps of real COCO — the corroboration §6 reading 2 asks for, delivered on
the axis it asked for it.

Two secondary confirmations fall out of the same table. **The control reproduces the population
offset** flagged in the §6 warning box: 36.92 (1000 img) → 34.55 (5000 img) = **−2.37**, against
−2.36 / −2.07 / −2.53 for `resnet50` / `vggt` / `dinov2`. A fourth arm, built to the same split on
purpose, lands on the same offset — the "~2 AP drop at the last step" is protocol, definitively,
not late-training regression. And **both R50 curves plateau in the same place** (ours 36.66 @80k
then down to 36.50; the control 36.92 @85k, flat from 80k), so even the endgame shape matches.

Nothing in the periodic columns is quotable as a result: they are 1000-image numbers. The row-2
figure and the last line of this table are the full-5000 finals.

**The 40 000 eval is missing, and the reason is worth knowing before it eats a run of yours.**
The scratch purge took the **reference venv** mid-run (`/cluster/scratch/niacobone/MaskDINO/myenv`
— the same 15-day mechanism CLAUDE.md documents, different victim than 2026-08-07). It presented
as a **data** error, not an environment one:

```
PIL.UnidentifiedImageError: cannot identify image file '.../coco/val2017/000000000139.jpg'
```

That file was fine — valid JPEG header, 161 KB, all 5000 val and 118 287 train images present.
**PIL was down to 0 `.py` files** (venv-wide: 27 `.py` against 1848 `.pyc`), so it had no JPEG
plugin left to register and reported a good image as unreadable. Training had run 6 992 further
iterations without complaint because its dataloader workers imported PIL *before* the deletion;
the eval spawns fresh workers, which re-import. **So the symptom appears at the next eval, on a
data path, long after the actual damage** — diagnose with the `.py` vs `.pyc` count, not the file
it names.

Recovery cost nothing but wall clock: `last_checkpoint` lives on `/cluster/work` (not purged) and
carries optimiser + scheduler + iteration, so the run resumed at exactly 39 999. Only the eval
scheduled for 40 000 is gone — it is not re-run on resume.

Two traps in the rebuild itself: `install.sh` **cannot run on a CPU node** as written (with no GPU
visible and `TORCH_CUDA_ARCH_LIST` unset, torch's `_get_cuda_arch_flags` returns an empty list and
`arch_list[-1] += '+PTX'` raises `IndexError`) — export `TORCH_CUDA_ARCH_LIST="8.0;8.6"` first,
which also fixes the sm_86-only build the clone shipped with. And rebuilding **in place** on
scratch was the right unblock mid-run but not a fix.

**The durable fix landed 2026-08-12 (todo 6i): the whole clone now lives at
`/cluster/home/niacobone/MaskDINO`**, which is not purged. Every path in this file, in
`third_party/maskdino_control/` and in the two slurm drivers points there; `MASKDINO_ROOT` still
overrides. The one thing a move used to break silently — the configs' absolute `_BASE_` into the
clone, and the overfit gate's *relative* base on top of it — is now re-rooted at `MASKDINO_ROOT`
by `config_paths.py::resolve_base`, which raises rather than falling back to detectron2 defaults.
The scratch path above is left as written: it is where the incident happened.

**`resnet50` is the column to read row 2 against** — it is the same frozen ResNet-50 under the same
recipe, so the two differ only by implementation. That test has now run to completion and
**agreement is the verdict** (+0.25 AP final, ±0.84 at every matched step): our `matcher.py`,
`criterion.py` and DN generation are corroborated end to end, and the 46.1 → 34.5 distance is a
recipe cost, not a bug — §6 reading 2.

**Job ledger for row 2** — one run, four SLURM ids, because the driver self-resubmits at the time
limit and each segment resumes from `last_checkpoint` on `/cluster/work`:

| job | wall | reached iter | note |
|---|---|---|---|
| 10094393 | 22 h 31 | 33 004 | 2026-08-09, first segment |
| **10228029** | 4 h 27 | 39 999 | **FAILED** — the scratch purge below, 6 992 iters after the damage |
| 10279969 | 22 h 32 | 73 376 | resumed from 39 999; the 40 000 eval is the only casualty |
| 10427048 | 10 h 22 | **87 948** | finished 2026-08-12 07:47, ran the full-5000 final |

`summary.json` in `output/maskdino_upstream_matched_20260809_015551/` is the row-2 source.
