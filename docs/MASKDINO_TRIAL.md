# MaskDINO-on-VGGT trial (single-frame)

**Status:** implementation complete, first runs launched 2026-07-27.
**Scope:** a *parallel* experiment track. Nothing in the existing D4RT arms (A–E), the VGGT
backbone, or the shared training scripts is modified — this trial lives in its own package
(`models/maskdino/`), its own training script (`scripts/train_maskdino.py`), its own test
(`tests/test_maskdino.py`) and its own SLURM job (`slurm/train_maskdino.sh`).

**Why:** supervisor request (2026-07-27) — replicate the MaskDINO decoder on top of the frozen
VGGT backbone and see whether a state-of-the-art detection-style decoder breaks the ceiling the
hand-rolled D4RT head hit (arm C: val mIoU 0.367 / honest val[grid] AP50 0.199). Constraint from
the supervisor: **single-frame only** for now. If it looks promising, extending to multi-frame is
the follow-up (§8).

Reference implementation read for the port: `/cluster/scratch/niacobone/MaskDINO`
(IDEA-Research MaskDINO, `maskdino/modeling/{transformer_decoder,pixel_decoder,criterion,matcher}`).

---

## 1. What MaskDINO actually is (and what the D4RT head was missing)

MaskDINO = Mask2Former's mask branch grafted onto DINO's detection decoder. The pieces that
matter, and how the existing D4RT head compares:

| MaskDINO component | D4RT head (arms A–E) | Kept in this trial |
|---|---|---|
| **Pixel decoder**: 6-layer MSDeformAttn encoder over 3 feature scales, produces enhanced multi-scale memory + a high-res `mask_features` map | none — raw VGGT tokens are linearly projected and LayerNormed once, single scale | ✅ ported (scales synthesised from VGGT tokens, §3) |
| **Deformable cross-attention** in the decoder (4 sampling points/head/level around a reference box) | dense `nn.TransformerDecoder` cross-attention over all tokens | ✅ ported (pure-PyTorch MSDeformAttn, §2) |
| **Anchor-box queries (DAB)**: each query owns a 4-d box, sine-encoded into its positional embedding, **refined layer by layer** | queries carry a (u,v) point prompt or a free learned embedding; no refinement | ✅ ported |
| **Two-stage query selection**: encoder tokens are classified/box-regressed, top-k become the decoder's initial content + anchors | queries are hand-seeded (grid / centroid / FPS anchors) | ✅ ported |
| **Mask-enhanced box init**: the initial masks are converted to boxes to seed the anchors | n/a | ✅ ported (`--initialize_box_type mask2box`) |
| **Denoising training (DN)**: noised GT labels+boxes (+masks) as extra queries, isolated by an attention mask — the main convergence accelerator in DINO/DN-DETR | none | ✅ ported (`--dn seg`) |
| **Deep supervision**: loss on all 9 decoder layers + the initial prediction + the encoder's interm output | loss on the final layer only | ✅ ported |
| **Losses**: sigmoid-focal class + point-sampled BCE/Dice masks + L1/GIoU boxes | softmax-CE-ish focal class + Dice + fg-weighted BCE, no boxes | ✅ ported |
| Hungarian matcher over class+mask+dice+**box+giou** | matcher over class+mask+dice+coord-prompt | ✅ ported |

The short version of the hypothesis: the D4RT arms plateaued at ~0.2 honest AP50 mainly on
**detection** (finding and separating objects), not on mask quality. Anchor boxes + iterative
refinement + DN + deep supervision are exactly the machinery that fixes DETR-style detection.

## 2. Deviations from upstream MaskDINO (and why)

Everything here is a deliberate, documented deviation — the decoder logic itself is a faithful port.

1. **No detectron2 / fvcore / compiled CUDA op.** The repo's `myenv` has none of them and the
   MaskDINO CUDA extension is built against a different Python. So:
   - `MSDeformAttn` uses the **pure-PyTorch `grid_sample` core**
     (`ms_deform_attn_core_pytorch`, the reference path shipped in MaskDINO itself). Slower than
     the fused kernel, irrelevant at our token counts (1830 memory tokens, 300 queries) and it
     runs on CPU, which is what makes `tests/test_maskdino.py` possible.
   - `point_sample` / `get_uncertain_point_coords_with_randomness` (PointRend) and
     `BitMasks.get_bounding_boxes` are reimplemented locally in `models/maskdino/utils.py`
     and `models/maskdino/box_ops.py`.
   - `Conv2d(norm=…)` + `c2_xavier_fill` → plain `nn.Conv2d` + `nn.init.xavier_uniform_`.
2. **Backbone**: frozen VGGT-1B aggregator instead of ResNet-50/Swin. Only the pixel decoder +
   transformer decoder train — **20.5 M** params at the full recipe, vs ~6.5 M for the D4RT head.
   VGGT is never touched, exactly as in every other arm.
3. **Single scale in, three scales out** (§3) — VGGT is a plain ViT-style aggregator with one
   token resolution, so the res3/res4/res5 pyramid is synthesised ViTDet-style.
4. **19 classes, sigmoid-focal, no background column.** MaskDINO/DINO classify with `num_classes`
   sigmoid logits and represent "no object" as *all logits low*, whereas the D4RT arms used 20
   softmax logits with background at index 0. Ported faithfully → the eval protocol needs the
   `score_mode="sigmoid"` switch (§6).
5. **Mask resolution** is the VGGT patch grid (37×37) by default, so the mask metrics are computed
   on exactly the same grid as arms A–E. `--mask_upsample 2` gives 74×74 (a transposed-conv step
   in the pixel decoder, GT rebuilt to match) — a separate run, not the headline number.
6. **No LSJ / crop / flip augmentation**: the images are VGGT-preprocessed 518×518 square resizes
   of ScanNet frames; the only augmentation is the project's existing photometric jitter.

## 3. Feature pyramid from a single-scale ViT (`models/maskdino/pixel_decoder.py`)

VGGT's aggregator gives `[B, S, P, 2048]` with `P = 5 + 37·37` (1 camera + 4 register + patch
tokens). Dropping the special tokens and reshaping gives one feature map per frame:
`[B, 2048, 37, 37]` (stride 14 at 518 px).

MaskDINO wants 3 encoder levels + a higher-resolution `mask_features`. Following **ViTDet's
"simple feature pyramid"** (a single-scale ViT is enough; FPN's lateral connections are not the
source of the gain), the pixel decoder builds:

```
VGGT tokens [B,2048,37,37]
   ├── 1×1 conv + GN  ─────────────────────────► level 0: 37×37   (stride 14)
   ├── 3×3 s2 conv + GN ───────────────────────► level 1: 19×19   (stride 28)
   └── 3×3 s2 conv + GN (on level 1) ──────────► level 2: 10×10   (stride 56)
                     │
        6-layer MSDeformAttn encoder over the 3 flattened levels (+ sine PE + level embed)
                     │
   ├── enhanced levels 0..2 ──────────────────► decoder memory (multi_scale_features)
   └── level 0 (+ optional ×2 transposed conv) ─► 1×1 conv → mask_features [B,256,37,37] (or 74×74)
```

`--feature_layers` optionally concatenates several aggregator layers (e.g. `4,11,17,23`, the
layers VGGT's own DPT heads read) before the 1×1 projection, at 4× the feature-cache cost. Default
is the last layer only — identical cache footprint to every other arm.

## 4. Single-frame protocol

- **The decoder never sees more than one frame.** Its batch dimension is *frames*, not scenes:
  one training step takes `--batch_frames` frames sampled from one scene's cached bundle and
  treats them as independent images.
- **Features** are also computed per frame by default (`--feature_mode single`: the aggregator
  runs with `S=1`), so the model is genuinely single-frame end to end. `--feature_mode bundle`
  runs the aggregator once over all `S` frames (tokens are then multi-view-informed by VGGT's
  global attention while the decoder stays per-frame) — that is the cheapest first step of the
  multi-frame extension and is available as an ablation.
- **GT** is per frame: for each frame, every ScanNet instance visible in it becomes one target
  with `labels` (0..18), `masks` (binary, at the mask resolution) and `boxes` (cxcywh normalized,
  derived from the mask). Frames with no visible instance are skipped in training.
- **Cross-view instance identity is not used and not required.** That is the whole point of the
  single-frame restriction — and the reason the numbers are not directly comparable to arms A–E
  (§6).

## 5. Files

| File | Contents |
|---|---|
| `models/maskdino/ms_deform_attn.py` | pure-PyTorch `MSDeformAttn` + `ms_deform_attn_core_pytorch` |
| `models/maskdino/utils.py` | `MLP`, `inverse_sigmoid`, `gen_sineembed_for_position`, `PositionEmbeddingSine`, `gen_encoder_output_proposals`, PointRend `point_sample` / uncertainty sampling |
| `models/maskdino/box_ops.py` | cxcywh↔xyxy, GIoU, `masks_to_boxes` |
| `models/maskdino/pixel_decoder.py` | `VGGTPixelDecoder` (§3) + the MSDeformAttn encoder |
| `models/maskdino/decoder.py` | `MaskDINODecoder` — two-stage selection, DAB anchors, iterative box refinement, DN, deep supervision |
| `models/maskdino/matcher.py` | `HungarianMatcher` (class/mask/dice/box/giou, point-sampled mask cost) |
| `models/maskdino/criterion.py` | `SetCriterion` (focal / point-sampled BCE+Dice / L1+GIoU, aux + interm + DN losses) |
| `models/maskdino/head.py` | `MaskDINOVGGTHead` = pixel decoder + decoder, `head_config` round-trip |
| `scripts/train_maskdino.py` | single-frame training + per-frame eval + checkpoints + `metrics.jsonl` + overlays |
| `scripts/eval_perframe.py` | scores an existing **D4RT** checkpoint under this trial's per-frame protocol (the apples-to-apples baseline) |
| `tests/test_maskdino.py` | CPU-only standalone test of every component |
| `slurm/train_maskdino.sh` | cluster job (stages the 500-scene official-GT tar like every other run) |

The only shared file touched is `train/eval_metrics.py`:
- an optional `score_mode="softmax"|"sigmoid"` argument (default `"softmax"` = previous
  behaviour, existing tests unchanged) so the same metric code can score sigmoid-focal
  predictions;
- `reshape(n, -1)` → `flatten(1)`, which fixes a crash on a zero-row prediction tensor (legal
  input once predictions are pre-filtered, see §6.3). Identical for every non-empty input.

## 6. Evaluation protocol — read this before comparing numbers

Metrics come from the *same* function as every other arm
(`train/eval_metrics.py::compute_instance_segmentation_metrics`). Three things differ:

**6.1 Per-frame, not per-bundle.** Arms A–E score one 8-frame multi-view instance against its
8-frame GT mask (one IoU over the concatenated frames). This trial scores each frame separately
and averages over frames, then over scenes. Frames with no GT instance are skipped. Different
task, different denominator → **the headline numbers are not interchangeable with arm C's
0.367 / 0.199.** Use `scripts/eval_perframe.py` to put an arm-C checkpoint on this protocol.

**6.2 Sigmoid scoring, and two operating points.** With no background class, "is this query an
object?" is `max_c sigmoid(logit_c) ≥ threshold`. Every eval therefore reports two variants:

| variant | threshold | meaning |
|---|---|---|
| headline (`mIoU`, `AP50`, …) | `--score_threshold`, default **0.25** (MaskDINO's `OBJECT_MASK_THRESHOLD`) | closest analogue of the D4RT arms' "argmax ≠ background" filter |
| `*_all` | 0.0 — every query kept and ranked by score | the standard COCO detection protocol; also the only signal that moves early in training, because focal-trained sigmoid scores start near zero |

`mIoU_all` (best IoU over *all* queries) is a mask-quality ceiling, not a detection number —
read it next to `AP50_all`, never on its own.

**6.3 A prediction that claims no pixels in a frame is dropped, not counted as a false
positive** (`scripts/eval_perframe.py::drop_empty_masks`, applied by both scripts). Without this
rule the protocol is unfair to the multi-view arms: a D4RT query is *supposed* to be empty in
the frames where its object is not visible. Mask2Former/MaskDINO get the same effect by folding
the mask's mean foreground probability into the score. In this trial's own tests, the rule
turns a spurious AP50 of 0.5 into the correct 1.0 on a planted-perfect example.

Everything is logged per eval into `<run_dir>/metrics.jsonl`.

## 7. Results

### 7.1 Machinery check (2026-07-27)

CPU test suite (`tests/test_maskdino.py`) green: the pure-PyTorch deformable attention matches a
naive explicit-loop reference to 1e-5; the decoder produces the right shapes for every
two-stage × DN × box-init combination; the matcher recovers a planted assignment; perfect
predictions drive every loss term to ~0.

GPU smoke test — 4 train scenes / 2 val scenes, 32 training frames, full recipe (300 queries,
6 encoder + 9 decoder layers, two-stage, DN "seg", mask-enhanced box init), 200 epochs in
**6.1 min** on one RTX 4090 (0.46 s/step, 24 M trainable params, backbone cached in 12 s):

| | mIoU | AP50 | AP75 | class_acc |
|---|---|---|---|---|
| train (memorised) | **0.969** | **0.992** | 0.982 | 1.000 |
| val (2 unseen scenes) | 0.095 | 0.095 | 0.059 | 0.29 |

The train row is a *sanity check, not a result*: 32 frames are trivially memorisable. What it
proves is that gradient flow, Hungarian matching, DN, box refinement and the metric path all
work end to end. The val row is what 4 training scenes buys — the real runs are below.

### 7.2 Real runs

Submitted 2026-07-27 — jobs **8748952** (50 scenes) and **8748972** (200 scenes, 190 after
removing the held-out 0080–0089). Both: official 500-scene GT tar, per-instance masks, val =
scenes 0080–0089, ~20 k gradient steps (epochs auto-scale with scene count so the comparison
across N is about data, not training length).

### 7.2.1 Data scaling (jobs 8748952 / 8754527 / 8774050)

All runs: official 500-scene GT, per-instance masks, val = scenes 0080-0089, identical recipe
(300 queries, 6 encoder + 9 decoder layers, two-stage, DN "seg", mask-enhanced box init),
epochs auto-scaled to hold the ~20-29 k gradient-step budget. All COMPLETED cleanly.

| Run | Scenes | val mIoU | val AP50 | val AP75 | val mAP | peak @ | train mIoU |
|---|---|---|---|---|---|---|---|
| **arm C — the bar** | 190 | 0.451 | 0.294 | 0.141 | 0.154 | converged | — |
| job 8748952 | 50 | 0.451 | 0.440 | 0.314 | 0.290 | ep 150/400 | 1.000 |
| job 8754527 | 190 | 0.594 | 0.624 | 0.440 | 0.418 | ep 38/100 | 0.994 |
| **job 8774050** | **490** | **0.669** | **0.699** | **0.506** | **0.475** | ep 31/60 | 0.947 |

**At 490 scenes the trial beats arm C by +48 % mIoU, +138 % AP50, 3.6x AP75, 3.1x mAP.** The
curve is still rising at the largest scale available (0.440 -> 0.624 -> 0.699 AP50 for
50 -> 190 -> 490 scenes) and the overfitting eases as data grows (train mIoU 1.000 -> 0.994 ->
0.947), so the model remains data-limited even at 490 scenes. Every run still peaks around
half-way through its schedule; `checkpoint_best.pth` captures it.

**This inverts the project's data-scaling conclusion.** Arm C got *worse* with more data
(0.367@190 -> 0.350@490, `docs/ARMS_SUMMARY.md`), which read as "the dataset is not the
bottleneck". On the same data MaskDINO gains +0.26 AP50 going 50 -> 490. The D4RT head was
**architecture-limited, not data-limited**; the old scaling result was a property of that head,
not of the task.

### 7.2.2 Ablations — no single ingredient carries the win

Each removes ONE MaskDINO component at N=190, everything else identical (jobs 8774052 /
8778736 / 8774056 / 8774065, all COMPLETED). Sorted by cost of removal:

| Config | val mIoU | val AP50 | ΔAP50 vs full | train mIoU |
|---|---|---|---|---|
| full recipe | 0.594 | 0.624 | — | 0.994 |
| `--no-two_stage` (no query selection) | 0.592 | 0.578 | −0.046 | 0.995 |
| `--enc_layers 0` (no deformable encoder) | 0.551 | 0.580 | −0.044 | 0.871 |
| `--dn no` (no denoising) | 0.586 | 0.594 | −0.030 | 0.986 |
| `--initialize_box_type no` (no mask-enhanced box init) | 0.610 | 0.608 | −0.016 | 0.993 |

Two honest readings:

1. **No component is decisive.** Each is worth 0.02-0.05 AP50, and box-init is within the
   ±0.04 eval-to-eval noise (its mIoU is actually *higher* than the full recipe's). The full
   recipe is still the best AP50 of the five, so the pieces are additive rather than redundant —
   but nobody should claim "denoising is what made this work".
2. **Every crippled variant still beats arm C by ~2x on AP50** (0.578-0.608 vs 0.294). The credit
   belongs to the architecture *class* — deformable attention over a multi-scale pyramid,
   per-layer anchor-box refinement, deep supervision, 20.5 M params — not to any one trick.
   And **data scale dominates all of it**: +0.26 AP50 from 50->490 scenes, versus ≤0.05 from any
   single component.

`--enc_layers 0` is the only ablation that also drops train mIoU (0.871 vs ~0.99), i.e. it
removes real capacity rather than just a training aid.

### 7.2.3 Cost note — eval must not scale with the training set

The first N=200 submission (job 8748972) reached only epoch 2 in 30 minutes and was cancelled:
it scored **all 190 train scenes** at every eval, and `_average_precision` loops over every kept
prediction at 10 IoU thresholds, so ~1600 frames x ~180 ms = ~5 min per eval, every 2 epochs.
Two fixes in `scripts/train_maskdino.py`: `--eval_topk 100` (COCO's `test_topk_per_image` —
protocol-correct *and* 3x faster per frame) and `--eval_train_scenes 10` (the train metric is
only an overfit read-out). Eval went ~180 s -> ~6 s.

### 7.2.4 What to run next

1. **Multi-frame extension** (§8) — the single-frame question is answered; this is the actual
   research goal.
2. **More data / augmentation**: the curve has not flattened at 490 scenes, which is all the
   official-GT tar holds. `--bundles_per_scene 2 --color_jitter 0.2` adds frame draws without
   new scenes (costs cache memory).
3. `--mask_upsample 2` (74x74 masks) — the masks are still supervised on a 37x37 grid.

### 7.3 The baseline these must beat (measured 2026-07-27)

Arm C (learned object queries, the current best D4RT head) scored under **this trial's per-frame
protocol** via `scripts/eval_perframe.py` on `d4rt_full_inst_learned_officialgt_20260708_124452`
(the run whose multi-view numbers are the quotable 0.367 / 0.199), all 10 val scenes
(0080–0089), unprompted learned queries:

| | mIoU | AP50 | AP75 | mAP |
|---|---|---|---|---|
| **arm C, per-frame — the bar** | **0.451** | **0.294** | 0.141 | 0.154 |
| arm C, per-bundle (reference only, NOT comparable) | 0.367 | 0.199 | — | — |

Per-scene spread: mIoU 0.34–0.61, AP50 0.18–0.43. Full JSON:
`<run_dir>/perframe_eval_checkpoint_best.json`.

Per-frame scores *higher* than per-bundle for the same checkpoint, which is expected and worth
stating explicitly: an instance only has to match in the frames where it is visible, and an
empty prediction in a frame where the object is absent is dropped rather than penalised (§6.3).
**This is exactly why the MaskDINO numbers must be read against 0.451 / 0.294 and never against
0.367 / 0.199.**

## 8. If it works: the multi-frame extension

Ordered by cost, each step reusing everything above:

1. `--feature_mode bundle` (already implemented): VGGT's global attention makes the per-frame
   tokens multi-view aware; the decoder stays single-frame. Free multi-view signal, no
   architectural change, no cross-frame identity.
2. **Shared queries across frames**: run the same query set against every frame's memory in a
   bundle and add a cross-frame self-attention block between decoder layers → one instance id per
   query across all views (this is what the D4RT arms did natively, and where the multi-view
   metric becomes meaningful again).
3. **3D anchors instead of 2D boxes**: replace the DAB 4-d box with a 3D box / anchor from VGGT's
   point head (arm E showed 3D anchors alone don't beat 2D queries, but arm E had no box
   refinement, no DN and no deep supervision — the ingredients that make anchors work in DINO).
