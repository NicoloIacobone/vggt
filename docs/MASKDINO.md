# MaskDINO on frozen VGGT — the active model track

**Status:** single-frame question answered and won (2026-07-27). At 490 scenes this head scores
**val mIoU 0.669 / AP50 0.699** against the best D4RT arm's **0.451 / 0.294** on the identical
per-frame protocol — +48 % mIoU, +138 % AP50; **0.694 / 0.729 with `--bundles_per_scene 2`**
(§7.4). The multi-frame extension (§8) is implemented through step 2 (shared queries across the
frames of a bundle, `--multi_frame`, 2026-07-28): per-frame it is neutral against its control,
and on the arms' own multi-view ruler it scores **0.535 mIoU / 0.494 AP50 vs arm C's
0.367 / 0.199**. 3D anchors (§8.3) are designed but not implemented.

**The port is verified against upstream (§7.6, 2026-07-29).** Driven with MaskDINO's own released
COCO weights, our ported decoder + deformable encoder reproduce upstream's published COCO val2017
result to **+0.004 mask AP / +0.009 box AP** (46.133 vs 46.129 vs paper 46.1). Read §7.6's scope
table before assuming this covers the training path — it does not.

**Origin:** supervisor request (2026-07-27) — replicate the MaskDINO decoder on top of the frozen
VGGT backbone and see whether a state-of-the-art detection-style decoder breaks the ceiling the
hand-rolled D4RT head hit (arm C: val mIoU 0.367 / honest val[grid] AP50 0.199). Constraint:
**single-frame only** for now.

Reference implementation read for the port: `/cluster/scratch/niacobone/MaskDINO`
(IDEA-Research MaskDINO, `maskdino/modeling/{transformer_decoder,pixel_decoder,criterion,matcher}`).

The D4RT arms A–E it replaced are retired to `legacy/d4rt/` but stay runnable — they are the
baseline every number here is measured against. See `legacy/README.md` and `docs/ARMS_SUMMARY.md`.

---

## 1. What MaskDINO is (and what the D4RT head was missing)

MaskDINO = Mask2Former's mask branch grafted onto DINO's detection decoder. The pieces that
matter, and how the retired D4RT head compares:

| MaskDINO component | D4RT head (arms A–E) | Kept here |
|---|---|---|
| **Pixel decoder**: 6-layer MSDeformAttn encoder over 3 feature scales, produces enhanced multi-scale memory + a high-res `mask_features` map | none — raw VGGT tokens are linearly projected and LayerNormed once, single scale | ✅ ported (scales synthesised from VGGT tokens, §3) |
| **Deformable cross-attention** in the decoder (4 sampling points/head/level around a reference box) | dense `nn.TransformerDecoder` cross-attention over all tokens | ✅ ported (pure-PyTorch MSDeformAttn, §2) |
| **Anchor-box queries (DAB)**: each query owns a 4-d box, sine-encoded into its positional embedding, **refined layer by layer** | queries carry a (u,v) point prompt or a free learned embedding; no refinement | ✅ ported |
| **Two-stage query selection**: encoder tokens are classified/box-regressed, top-k become the decoder's initial content + anchors | queries are hand-seeded (grid / centroid / FPS anchors) | ✅ ported |
| **Mask-enhanced box init**: the initial masks are converted to boxes to seed the anchors | n/a | ✅ ported (`--initialize_box_type bitmask`) |
| **Denoising training (DN)**: noised GT labels+boxes (+masks) as extra queries, isolated by an attention mask — the main convergence accelerator in DINO/DN-DETR | none | ✅ ported (`--dn seg`) |
| **Deep supervision**: loss on all 9 decoder layers + the initial prediction + the encoder's interm output | loss on the final layer only | ✅ ported |
| **Losses**: sigmoid-focal class + point-sampled BCE/Dice masks + L1/GIoU boxes | softmax-CE-ish focal class + Dice + fg-weighted BCE, no boxes | ✅ ported |
| Hungarian matcher over class+mask+dice+**box+giou** | matcher over class+mask+dice+coord-prompt | ✅ ported |

The hypothesis that motivated it: the D4RT arms plateaued at ~0.2 honest AP50 mainly on
**detection** (finding and separating objects), not on mask quality. Anchor boxes + iterative
refinement + DN + deep supervision are exactly the machinery that fixes DETR-style detection.
§7.2.2 shows the hypothesis was right about the *class* of architecture but wrong about any
single ingredient being decisive.

## 2. Deviations from upstream MaskDINO (and why)

Everything here is a deliberate, documented deviation — the decoder logic itself is a faithful port.

1. **No detectron2 / fvcore / compiled CUDA op.** The repo's `myenv` has none of them and the
   MaskDINO CUDA extension is built against a different Python. So:
   - `MSDeformAttn` uses the **pure-PyTorch `grid_sample` core**
     (`ms_deform_attn_core_pytorch`, the reference path shipped in MaskDINO itself). Slower than
     the fused kernel, irrelevant at our token counts (1830 memory tokens, 300 queries) and it
     runs on CPU, which is what makes the CPU test suite possible.
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
   `score_mode="sigmoid"` switch (§6). The width comes from `models/maskdino/head.py::
   NUM_SCANNET_CLASSES`; instances of the 20th `SCANNET_CLASSES` name (`otherfurniture`) are
   dropped rather than crashing the matcher — see §4.
5. **Mask resolution** is the VGGT patch grid (37×37) by default, so the mask metrics are computed
   on exactly the same grid as arms A–E. `--mask_upsample 2` gives 74×74 (a transposed-conv step
   in the pixel decoder, GT rebuilt to match) — a separate run, not the headline number.
6. **No LSJ / crop / flip augmentation**: the images are VGGT-preprocessed 518×518 square resizes
   of ScanNet frames; the only augmentation is the project's existing photometric jitter.
7. **`initialize_box_type` implements only `bitmask`.** Upstream also offers `mask2box`; here the
   two would have shared one `!= "no"` branch, so asking for `mask2box` silently ran `bitmask`.
   The constructor now rejects it and the CLI no longer offers it (cleanup 2026-07-28). No
   completed run used it.

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
- **Classes the head cannot represent are dropped** (added 2026-07-28).
  `train/maskdino_data.py::build_frame_targets` takes `num_classes` (wired from
  `model.head.num_classes`, never a hardcoded `19`) and skips any instance whose dataset class
  index is outside `1..num_classes`, reporting one aggregated warning per scene naming the class
  and the count. This matters because `data/scannet_overfit.py::SCANNET_CLASSES` has **twenty**
  names — index 20 is `otherfurniture`, which the 19-logit sigmoid head has no column for. Such an
  instance used to become label 19 and **crash**: `IndexError` in `matcher.py`
  (`pos_cost_class[:, tgt_ids]`) and in the DN `label_enc` `nn.Embedding(19)` in `decoder.py`;
  `criterion.py::loss_labels` did not crash but silently folded it into the no-object column.
  Dropping matches what `legacy/dataset_build/scripts/build_official_masks.py` already does
  upstream (every NYU40 class outside the 19 trainable ones → background).
  **Nothing changes for the completed runs.** Verified 2026-07-28 by listing both GT sources
  without unpacking them: the official-GT build tree has no `otherfurniture` folders, and the
  SAM3 tar (`scannet_instance_dataset_full.tar.zst`, the realistic trigger since SAM3 ran against
  the full 20-name list) contains **exactly the 19 trainable classes** in both `masks/` and
  `masks_instance/` — `zstd -dc … | tar -t` over all 200 scenes returns zero `otherfurniture`
  entries. The bug was latent, reachable only by a future GT build that keeps the 20th class.
  As a backstop, `matcher.py::check_target_labels` (called from `HungarianMatcher.forward` and
  `SetCriterion.forward`) turns an out-of-range or negative label from any *other* caller into a
  named `AssertionError` instead of an opaque `IndexError`.
- **Cross-view instance identity is not used and not required.** That is the whole point of the
  single-frame restriction — and the reason the numbers are not directly comparable to arms A–E
  (§6).

## 5. Files

### The model (`models/maskdino/`)

| File | Contents |
|---|---|
| `head.py` | `MaskDINOVGGTHead` = pixel decoder + decoder; `head_config` round-trip (derived from the constructor signature, so it can never silently omit an argument) |
| `model.py` | `MaskDINOVGGTModel` = frozen VGGT-1B backbone + head. Not re-exported from `__init__` so CPU tests can import the head without the backbone |
| `pixel_decoder.py` | `VGGTPixelDecoder` (§3) + the MSDeformAttn encoder |
| `decoder.py` | `MaskDINODecoder` — two-stage selection, DAB anchors, iterative box refinement, DN, deep supervision |
| `decoder_layers.py` | the generic DAB/DINO decoder stack `MaskDINODecoder` drives |
| `multiframe.py` | the shared-query multi-frame path (§8.2): `CrossFrameAttention`, `build_bundle_target`, `expand_bundle_indices`, `MultiFrameHungarianMatcher` |
| `matcher.py` | `HungarianMatcher` (class/mask/dice/box/giou, point-sampled mask cost) + `check_target_labels` (out-of-range GT-label guard, §4) |
| `criterion.py` | `SetCriterion` (focal / point-sampled BCE+Dice / L1+GIoU, aux + interm + DN losses) |
| `ms_deform_attn.py` | pure-PyTorch `MSDeformAttn` + `ms_deform_attn_core_pytorch` |
| `box_ops.py` | cxcywh↔xyxy, GIoU, `masks_to_boxes` |
| `utils.py` | `MLP`, `inverse_sigmoid`, `gen_sineembed_for_position`, `PositionEmbeddingSine`, `gen_encoder_output_proposals`, PointRend `point_sample` / uncertainty sampling |

### Training, evaluation, jobs

| File | Contents |
|---|---|
| `scripts/train_maskdino.py` | entry point only: CLI, model/criterion construction, epoch loop, checkpoints |
| `train/maskdino_data.py` | per-frame GT (`build_frame_targets`), frozen-backbone feature cache, batching |
| `train/maskdino_eval.py` | per-frame scoring over cached scenes + the RGB\|GT\|pred figures |
| `train/perframe.py` | the protocol itself — `drop_empty_masks`, `topk_predictions`, `perframe_metrics`. Shared with `eval_perframe.py`, which is what makes the two families comparable |
| `train/common.py` | scene-path resolution, photometric jitter, LR schedule, `metrics.jsonl` append |
| `scripts/eval_perframe.py` | scores an existing **D4RT** checkpoint under this protocol (the apples-to-apples baseline) |
| `slurm/train_maskdino.sh` | cluster job (stages the 500-scene official-GT tar; logs → `slurm/logs/`) |
| `tests/test_maskdino_model.py` | MSDeformAttn vs naive reference, pixel decoder, decoder configs, box ops, `head_config` round-trip, `initialize_box_type` guard |
| `tests/test_maskdino_loss.py` | matcher, criterion key set + perfect-prediction zero-loss, out-of-range-label guard |
| `tests/test_maskdino_train.py` | per-frame GT builder (incl. class drop), per-frame metric slicing, 60-step synthetic overfit |
| `tests/test_maskdino_multiframe.py` | cross-frame block, bundle GT + index expansion, bundle matcher, shared-query forward, S=1 equivalence, multi-frame overfit, bundle batching + scoring |
| `tests/maskdino_fixtures.py` | `_tiny_head`, `_synthetic_targets` shared by the three test modules |

The only shared file this track modified is `train/eval_metrics.py`:
- an optional `score_mode="softmax"|"sigmoid"` argument (default `"softmax"` = previous
  behaviour, existing tests unchanged) so the same metric code can score sigmoid-focal
  predictions;
- `reshape(n, -1)` → `flatten(1)`, which fixes a crash on a zero-row prediction tensor (legal
  input once predictions are pre-filtered, see §6.3). Identical for every non-empty input.

## 6. Evaluation protocol — read this before comparing numbers

Metrics come from the *same* function as every D4RT arm
(`train/eval_metrics.py::compute_instance_segmentation_metrics`). Three things differ:

**6.1 Per-frame, not per-bundle.** Arms A–E score one 8-frame multi-view instance against its
8-frame GT mask (one IoU over the concatenated frames). This track scores each frame separately
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
positive** (`train/perframe.py::drop_empty_masks`, applied by both scorers). Without this rule
the protocol is unfair to the multi-view arms: a D4RT query is *supposed* to be empty in the
frames where its object is not visible. Mask2Former/MaskDINO get the same effect by folding the
mask's mean foreground probability into the score. In the tests, the rule turns a spurious AP50
of 0.5 into the correct 1.0 on a planted-perfect example.

Everything is logged per eval into `<run_dir>/metrics.jsonl`.

## 7. Results

### 7.1 Machinery check (2026-07-27)

CPU test suite green: the pure-PyTorch deformable attention matches a naive explicit-loop
reference to 1e-5; the decoder produces the right shapes for every two-stage × DN × box-init
combination; the matcher recovers a planted assignment; perfect predictions drive every loss
term to ~0.

GPU smoke test — 4 train scenes / 2 val scenes, 32 training frames, full recipe (300 queries,
6 encoder + 9 decoder layers, two-stage, DN "seg", mask-enhanced box init), 200 epochs in
**6.1 min** on one RTX 4090 (0.46 s/step, 24 M trainable params, backbone cached in 12 s):

| | mIoU | AP50 | AP75 | class_acc |
|---|---|---|---|---|
| train (memorised) | **0.969** | **0.992** | 0.982 | 1.000 |
| val (2 unseen scenes) | 0.095 | 0.095 | 0.059 | 0.29 |

The train row is a *sanity check, not a result*: 32 frames are trivially memorisable. What it
proves is that gradient flow, Hungarian matching, DN, box refinement and the metric path all
work end to end.

### 7.2 Data scaling (jobs 8748952 / 8754527 / 8774050)

All runs: official 500-scene GT, per-instance masks, val = scenes 0080-0089, identical recipe
(300 queries, 6 encoder + 9 decoder layers, two-stage, DN "seg", mask-enhanced box init),
epochs auto-scaled to hold the ~20-29 k gradient-step budget. All COMPLETED cleanly.

| Run | Scenes | val mIoU | val AP50 | val AP75 | val mAP | peak @ | train mIoU |
|---|---|---|---|---|---|---|---|
| **arm C — the bar** | 190 | 0.451 | 0.294 | 0.141 | 0.154 | converged | — |
| job 8748952 | 50 | 0.451 | 0.440 | 0.314 | 0.290 | ep 150/400 | 1.000 |
| job 8754527 | 190 | 0.594 | 0.624 | 0.440 | 0.418 | ep 38/100 | 0.994 |
| **job 8774050** | **490** | **0.669** | **0.699** | **0.506** | **0.475** | ep 31/60 | 0.947 |

**At 490 scenes this head beats arm C by +48 % mIoU, +138 % AP50, 3.6x AP75, 3.1x mAP.** The
curve is still rising at the largest scale available (0.440 -> 0.624 -> 0.699 AP50 for
50 -> 190 -> 490 scenes) and the overfitting eases as data grows (train mIoU 1.000 -> 0.994 ->
0.947), so the model remains data-limited even at 490 scenes. Every run still peaks around
half-way through its schedule; `checkpoint_best.pth` captures it.

**This inverts the project's data-scaling conclusion.** Arm C got *worse* with more data
(0.367@190 -> 0.350@490, `docs/ARMS_SUMMARY.md`), which read as "the dataset is not the
bottleneck". On the same data MaskDINO gains +0.26 AP50 going 50 -> 490. The D4RT head was
**architecture-limited, not data-limited**; the old scaling result was a property of that head,
not of the task.

### 7.2.1 Ablations — no single ingredient carries the win

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

### 7.2.2 Cost note — eval must not scale with the training set

The first N=200 submission (job 8748972) reached only epoch 2 in 30 minutes and was cancelled:
it scored **all 190 train scenes** at every eval, and `_average_precision` loops over every kept
prediction at 10 IoU thresholds, so ~1600 frames x ~180 ms = ~5 min per eval, every 2 epochs.
Two fixes: `--eval_topk 100` (COCO's `test_topk_per_image` — protocol-correct *and* 3x faster per
frame) and `--eval_train_scenes 10` (the train metric is only an overfit read-out). Eval went
~180 s -> ~6 s.

### 7.3 The baseline these beat (measured 2026-07-27)

Arm C (learned object queries, the best D4RT head) scored under **this per-frame protocol** via
`scripts/eval_perframe.py` on `d4rt_full_inst_learned_officialgt_20260708_124452` (the run whose
multi-view numbers are the quotable 0.367 / 0.199), all 10 val scenes (0080–0089), unprompted
learned queries:

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

### 7.4 Multi-view features, mask resolution, view draws, multi-frame (2026-07-28)

Five runs, all at N=490 against the **0.669 / 0.699** single-frame bar, identical recipe
otherwise, official 500-scene GT, peak (`checkpoint_best*`) numbers:

| Job | Change | val mIoU | val AP50 | ΔAP50 | train mIoU @peak |
|---|---|---|---|---|---|
| 8774050 | — (the bar) | 0.669 | 0.699 | — | 0.947 |
| 8895540 | `--feature_mode bundle` | 0.622 | 0.651 | **−0.048** | 0.872 |
| 8895551 | `--mask_upsample 2` | 0.662 | 0.677 | −0.022 | 0.812 |
| **8895565** | **`--bundles_per_scene 2 --color_jitter 0.2`** | **0.694** | **0.729** | **+0.030** | 0.816 |
| 8900100 | `--multi_frame --feature_mode bundle` | 0.621 | 0.630 | −0.069 | 0.867 |

1. **Multi-view-aware tokens are not free — they cost 0.048 AP50** (§8.1). Running the aggregator
   over the bundle mixes the views inside the frozen features, and per-frame segmentation gets
   *worse*, not better. So §8.1 is a **negative result**, and it is the correct control for the
   multi-frame decoder (below), not the bar.
2. **`--mask_upsample 2` is neutral** (−0.022, inside the ±0.04 eval-to-eval noise) — the same
   verdict the D4RT arms reached on a different head. Masks stay on the 37×37 patch grid.
3. **More view draws per scene is the best lever left**: +0.030 AP50 and a *new best* 0.729, with
   train mIoU dropping 0.947 → 0.816 (less memorisation). The model is still data-limited, and
   since the tar holds no more scenes, views-per-scene is the cheapest remaining data. **Honest
   caveat:** the `EPOCHS=30` this job was submitted with was silently clamped back to 60 by
   `slurm/train_maskdino.sh` (fixed 2026-07-28), so it had a 2× larger step budget available;
   it nevertheless *peaked* at epoch 19 = 18.6 k steps, against the bar's peak at 15.2 k, so the
   gain is not simply "trained longer". `--bundles_per_scene 4` (job 8950610, `EPOCHS=15`) tests
   the next point on that curve.
4. **Multi-frame** (§8.2) costs 0.069 AP50 against the bar — but 0.048 of that is the bundle
   features it is built on. Against its proper control (8895540, 0.651) the shared-query decoder
   is **−0.021 AP50 per frame, i.e. neutral inside the noise band**, and in exchange it produces
   a genuine multi-view result (below). Job 8950613 re-runs it with per-frame features to
   decouple the two; job 8950617 is the `--no-cross_frame_attn` ablation.

**The per-bundle number is back.** Job 8900100 scores, on the multi-view protocol of arms A–E:

| | mIoU | AP50 | AP75 | mAP |
|---|---|---|---|---|
| arm C (best D4RT head), per-bundle | 0.367 | 0.199 | — | — |
| **MaskDINO `--multi_frame`, per-bundle** | **0.535** | **0.494** | 0.279 | 0.272 |

**+46 % mIoU and 2.5× AP50 over the best D4RT arm on the arms' own ruler** — one query is one
instance across all 8 views, no post-hoc matching. Read it *only* against 0.367 / 0.199, never
against the per-frame 0.699 (docs/RESULTS.md §1).

### 7.5 Official-split read-out (job 8900194)

Same recipe, but val = the 77 official ScanNet v2 val scenes inside our tar and train = the other
413 (docs/RESULTS.md §1.1). Peak **val mIoU 0.589 / AP50 0.604** (epoch 27; the job hit its 12 h
wall clock at epoch 39/60, past its peak, so the number stands).

**Our convention split is ~0.10 AP50 "easier" than the official val scenes** — with 67 fewer
training scenes as a partial explanation. Quote 0.699 as *our* split's number and mention 0.604
whenever comparability to the ScanNet literature comes up (it is still a per-view 2D-mask number,
so it is not a leaderboard-comparable figure either — docs/RELATED_WORK.md).

### 7.6 Upstream-equivalence check on COCO (job 8967932, 2026-07-29)

Everything above measures the port against *our own* baselines, which cannot detect a bug that
is faithfully wrong in both. This check closes that loop: it drives **our** ported modules with
**upstream's released COCO weights** and asks whether they reproduce upstream's published COCO
val2017 numbers.

`scripts/coco_transplant_eval.py` loads
`maskdino_r50_50ep_300q_hid1024_3sd1_instance_maskenhanced_mask46.1ap_box51.5ap.pth`
(config `maskdino_R50_bs16_50ep_3s.yaml` — the one our defaults mirror) into upstream's
detectron2 harness, then swaps in our `MaskDINODecoder` and our `MSDeformAttnEncoder`.
`--mode baseline` leaves upstream untouched and is the control.

**The decoder accepts upstream's weights at `strict=True`: 333/333 parameters, names and shapes.**

| COCO val2017, 5000 images | segm AP | segm AP50 | box AP | box AP50 |
|---|---|---|---|---|
| upstream model zoo, row "MaskDINO (hid 1024)" — *this checkpoint* | 46.1 | — | 51.5 | — |
| `--mode baseline` (upstream code, this env) | 46.129 | 69.021 | 51.540 | 70.509 |
| **`--mode ours` (ported modules)** | **46.133** | 69.036 | **51.549** | 70.514 |

**The 46.1 / 51.5 target is the README model-zoo figure for the exact checkpoint used, not a
paper table value** — get this right, the paper's numbers are different. Table 3's 50-epoch /
300-query ResNet-50 rows are **46.0 / 50.5** (plain) and **46.3 / 51.7** (‡, mask-enhanced box
init). The ‡ row corresponds to the *other* released checkpoint, `hid2048` (52 M params,
286 GFLOPs); ours is the narrower `hid1024` variant (47 M, 226 GFLOPs — encoder FFN 1024 instead
of 2048, which is also why `maskdino_R50_bs16_50ep_3s.yaml` is the matching config). Comparing
our run against 46.3 would be comparing against a wider model we never ran.

The comparison that actually carries the verdict is **ours vs `--mode baseline`** — same code
path, same env, same weights, same data — and there the gap is 0.004 AP.

**Δ = +0.004 segm AP / +0.009 box AP.** On CPU the two modes are *bit-identical* to every
printed digit; the ~0.005 AP drift appears only on GPU, because upstream calls the fused CUDA
MSDeformAttn kernel there while our port always uses the `grid_sample` core (§2.1). That is the
one intended difference, and it is worth 0.01 AP.

**Verified as a live path, not a no-op.** `transplant()` asserts the modules are ours
(`models.maskdino.decoder` / `models.maskdino.pixel_decoder`, 6 + 9 ported `MSDeformAttn`
instances), and `--perturb 1.05` scales one weight inside *our* decoder: segm AP moves
55.702 → 55.608 on the 10-image subset. Identical numbers therefore mean equivalence, not a
silent fallback to upstream code.

**One trap this surfaced, worth knowing before touching level plumbing.** Upstream's pixel
decoder returns `multi_scale_features` LOW→HIGH resolution and its decoder walks that list
*backwards* (`idx = num_feature_levels-1-i`); our port takes it HIGH→LOW and walks forwards.
Both flatten the same tensors in the same order only because the adapter reverses the list.
The decoder's own `input_proj` is an empty `nn.Sequential` under this config, so its index
convention carries no weights and the difference is invisible in a state-dict diff.

**Scope — what this does and does not certify.**

| Certified by this check | Not exercised |
|---|---|
| `ms_deform_attn.py` (encoder *and* decoder) | `matcher.py`, `criterion.py` — training only |
| `pixel_decoder.py`: encoder layer/stack, reference points | DN query generation in `decoder.py` |
| `decoder.py`: two-stage selection, DAB anchors, iterative box refinement, mask-enhanced box init, prediction heads | `multiframe.py` — no upstream counterpart |
| `decoder_layers.py`, `utils.py`, `box_ops.masks_to_boxes` | the VGGT ViTDet pyramid (§3) — no COCO counterpart |

The right-hand column is not a known problem, just untested by *this* route; the loss path still
rests on `tests/test_maskdino_loss.py` (perfect-prediction zero loss) and the overfit tests.
Since the multi-frame work edits the matcher and the criterion, that is the column to extend
next if more assurance is wanted.

Reproduce: `sbatch slurm/coco_transplant.sh` (~32 min on one RTX 3090, both modes; results land in
`/cluster/work/igp_psr/niacobone/distillation/output/coco_transplant/`).
COCO val2017 lives at `/cluster/scratch/niacobone/coco` — **global scratch is purged after
15 days**, so re-download (§ the script header) if it has vanished.

## 8. The multi-frame extension

Ordered by cost, each step reusing everything above.

### 8.1 `--feature_mode bundle` — multi-view *features*, single-frame decoder

Runs the frozen aggregator once over all S frames of a bundle instead of once per frame, so
VGGT's global attention makes each frame's tokens multi-view aware while the decoder still sees
one frame at a time. No architectural change, no cross-frame identity, no extra parameters — only
the feature cache is built differently. Implemented in `train/maskdino_data.py::extract_features`;
first run at N=490 submitted 2026-07-28 (§7.4).

### 8.2 Shared queries across frames (`--multi_frame`) — implemented 2026-07-28

**One query set per bundle**: query *q* is the same object hypothesis in all S views, so it owns a
mask *volume* rather than S unrelated 2D masks. Three changes, all off by default:

1. **Shared query initialisation** (`models/maskdino/decoder.py`, `frames_per_sample=S`).
   The batch is B bundles of S frames, frames contiguous. Two-stage selection now takes the top-k
   proposals from the **union** of the S frames' encoder outputs (one `topk` over `S·HW`) and
   broadcasts the selected content embedding to every frame. Each frame then keeps its **own**
   anchor box: with `--initialize_box_type bitmask` the anchor is re-derived per frame from that
   frame's initial mask, and refined per frame by the usual DAB refinement. Content shared,
   geometry per view. Without two-stage the learned queries were already shared, so nothing
   changes there.
2. **`CrossFrameAttention`** (`models/maskdino/multiframe.py`), one block per decoder layer, run
   after the layer and before its box refinement. Attention over a sequence of length S — the S
   copies of one query — with batch = queries × bundles. Deliberately **no frame positional
   encoding**: the views are an unordered set, so the block is permutation-equivariant in S.
   Denoising queries are excluded (slot *i* means a different GT instance in each frame, so mixing
   them would leak and confuse); DN therefore stays exactly the per-frame recipe of §1.
3. **Bundle-level matching** (`MultiFrameHungarianMatcher` + `expand_bundle_indices`). The
   Hungarian assignment is made **once per bundle** — class cost on the mean sigmoid score over
   views, mask BCE+Dice over the concatenated `[S·h·w]` volume (the D4RT arms' multi-view mask
   cost), box L1+GIoU averaged over the views where the instance is visible. The assignment is
   then projected back onto the frames where the matched instance is actually visible, and
   **every loss stays the per-frame loss it already was**. In a view where its instance is not
   visible the query is simply unmatched, i.e. supervised as "no object" — exactly the behaviour
   the evaluation protocol rewards (§6.3).

Bundle GT costs nothing to build: `build_frame_targets` already stores the dataset's global
instance id per frame target, which is the cross-view link the single-frame protocol threw away;
`build_bundle_target` re-links the frames by it (`masks [n,S,h,w]`, `valid [n,S]`, `frame_row`).

**Both metrics are reported** (`train/maskdino_eval.py::eval_scenes_multiframe`): the per-frame
numbers, directly comparable to the 0.669 / 0.699 single-frame bar, **and** `bundle_*` — the
multi-view protocol of arms A–E (one IoU over the concatenated volume, one class score per query
= max over views), which was meaningless while queries were per-frame and is comparable to arm
C's 0.367 / 0.199. Never mix the two (docs/RESULTS.md §1).

Flags: `--multi_frame` (sample = a bundle of `--num_frames` frames), `--batch_bundles`
(default 1 → 8 frames/step, the same GPU footprint and the same steps/epoch as the single-frame
runs), `--no-cross_frame_attn` (ablate the block, keeping shared init + bundle matching).

### 8.3 3D anchors instead of 2D boxes (designed, not implemented)

Replace the DAB 4-d box with a 3D anchor from VGGT's point head. Arm E showed 3D anchors alone
don't beat 2D queries, but arm E had no box refinement, no DN and no deep supervision — the
ingredients that make anchors work in DINO.

**Framing (settled 2026-07-28, docs/RELATED_WORK.md).** FAST3DIS (arXiv 2603.25993) already
publishes exactly this mechanism — a learned 3D anchor generator plus project-and-sample
cross-attention — on a LoRA-adapted Depth-Anything-V3. So this step is an **ablation inside our
own controlled study** ("3D anchors vs 2D DAB boxes, same frozen backbone, same data, same
protocol", which nobody has run and which re-tests the arm-E negative result), **not** a new
mechanism. Budget it accordingly.

**Dependency: do this on top of §8.2, not before it.** A 3D anchor is only meaningful when a
query is one instance across views — with per-frame queries it is just a 2D box plus a depth.

**Design sketch.**

1. *Geometry cache.* At caching time the frozen point head already runs on the aggregator output
   we have; store only the **per-patch-token 3D position** (confidence-weighted mean of the
   token's 14×14 pixels, as in `legacy/d4rt/models/anchor_queries.py::patch_token_positions`) —
   `[S, 37·37, 3]` + confidence, ~65 kB per scene in fp16, versus ~26 MB for the full pointmap,
   which is never stored. Reimplement the 15-line pooling next to the cache rather than importing
   from frozen `legacy/`.
2. *Anchor = a 3D point (+ scale) per query, shared across the bundle's views.* Two-stage
   selection already picks top-k memory tokens; each selected token's cached 3D position is its
   anchor, so 3D two-stage costs a gather.
3. *Per-view reference points without camera math.* Every patch of view *f* has a cached 3D
   position, so the query's 2D reference in that view is a **soft nearest patch**:
   `w = softmax(-‖p_patch − anchor‖² / τ)` over the 37×37 grid, reference `(u,v) = Σ w · (u,v)_patch`.
   Differentiable in the anchor, needs no intrinsics/extrinsics (unlike FAST3DIS, which projects
   with its predicted camera), and degrades gracefully where the pointmap is unreliable.
   Iterative refinement then predicts Δxyz on the anchor instead of Δbox.
4. *Losses unchanged.* The 2D box stays the prediction target of `_bbox_embed` per view, so the
   matcher, the box/GIoU losses and DN all keep working exactly as they do today; only the query
   *positional prior* and the sampling locations change.

Cheap follow-ups that need one flag each: `--mask_upsample 2` (74×74 masks — currently supervised
on the 37×37 patch grid) and `--bundles_per_scene 2 --color_jitter 0.2` (more frame draws without
new scenes; costs cache memory). Both submitted at N=490 on 2026-07-28 (§7.4).

Cheap follow-ups that need one flag each: `--mask_upsample 2` (74×74 masks — currently supervised
on the 37×37 patch grid) and `--bundles_per_scene 2 --color_jitter 0.2` (more frame draws without
new scenes; costs cache memory). Both submitted at N=490 on 2026-07-28 (§7.4).
