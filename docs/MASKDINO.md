# MaskDINO on frozen VGGT — the active model track

**Status:** single-frame question answered and won (2026-07-27). At 490 scenes this head scores
**val mIoU 0.669 / AP50 0.699** against the retired baseline head's **0.451 / 0.294** on the identical
per-frame protocol — +48 % mIoU, +138 % AP50; **0.694 / 0.729 with `--bundles_per_scene 2`**
(§7.4; 4 draws/scene saturates, §7.4.1). The multi-frame extension (§8) is implemented through
step 2 (shared queries across the frames of a bundle, `--multi_frame`, 2026-07-28): per-frame it
is neutral against its control, and on the baseline head's own multi-view ruler the best run scores
**0.539 mIoU / 0.515 AP50 vs the baseline's 0.367 / 0.199** (`--bundles_per_scene 2 --color_jitter
0.2`, job 9071415, §8.2; 0.535 / 0.494 without the data recipe). The 2026-07-29 ablations
(§7.4.1) show that result rests on two ingredients: removing cross-frame attention costs
**−0.18 bundle AP50** and swapping the bundle features for per-frame ones costs **−0.15**, so
multi-view-aware frozen tokens — a *negative* for per-frame accuracy (§8.1) — are *required*
for the multi-view result. **The resolution question is closed (§7.7):** the 37×37 grid's
GT-only ceiling on ScanNet is 0.956 AP50 on the full-resolution ruler, the model sits at ~0.69,
and `--mask_upsample 2` stays neutral even on that ruler — recognition binds, not resolution.
3D anchors (§8.3) are designed but not implemented.

**The reportable 3D number exists (§9.6, 2026-08-03).** Trained on the official 1201-scene
split and scored by the vendored official evaluator on val-312, the multi-frame model reaches
**AP 0.023 / AP50 0.067 / AP25 0.268** (0.029 / 0.083 / 0.305 with tuned lifting knobs) —
**FAST3DIS's ballpark (0.038 / 0.096 / 0.316) on a strictly frozen backbone**, alongside IGGT
(0.028 / 0.112 / 0.287 — FAST3DIS's re-evaluation of it, not IGGT's own paper). **Both of those
columns are class-agnostic and ours is class-aware; scored their way this checkpoint is 0.017 /
0.060 / 0.334, i.e. ahead on AP25 and ~1.6–2.2× behind on AP50/AP** (§9.11 — read it before
quoting "ballpark"). **The `--anchor_3d` checkpoint (§8.3) is the strongest row: class-agnostic
0.042 / 0.138 / 0.504 — lead on AP50/AP25, lead IGGT on AP, TIE FAST3DIS on AP, untuned**
(job 9866391, §9.11; replicated at seed 1, 0.039 / 0.129 / 0.485, job 9979100) —
and re-sweeping its lifting knobs reaches **0.055 / 0.185 / 0.571** with *every* point of the grid
still ahead, so the lead is not a tuning artefact (§9.8.1). **Seed variance is now measured on
both rulers** (§8.3: per-bundle AP50 ±0.009 *and* class-agnostic 3D AP50 ±0.009, effect ~9×
that), which is the yardstick every Δ in this document should be read
against. SegVGGT's 0.504 / 0.717 / 0.870 is **a different protocol** (§9.9): its
evaluator transfers masks with ScanNet's GT poses and sensor depth, so it measures 2D mask
quality alone where ours measures 2D mask quality times predicted geometry. **Measured, that
protocol is worth 2.3× of the gap and no more** (§9.10): run on our own masks it takes us to
0.060 / 0.156 / 0.408, so a factor of ~4.6 to SegVGGT is real and remains. Two things §9.6
settles: the leak-free
checkpoint *beats* the leaked diagnostic 1.6× (data scale outweighs seeing the val scenes —
the 3D ruler reproducing §7.2's data-limited conclusion), and on this ruler the **lifting step,
not the decoder, is the binding constraint**.

**The port is verified against upstream (§7.6, 2026-07-29).** Driven with MaskDINO's own released
COCO weights, our ported decoder + deformable encoder reproduce upstream's published COCO val2017
result to **+0.004 mask AP / +0.009 box AP** (46.133 vs 46.129 vs paper 46.1). Read §7.6's scope
table before assuming this covers the training path — it does not.

**Origin:** supervisor request (2026-07-27) — replicate the MaskDINO decoder on top of the frozen
VGGT backbone and see whether a state-of-the-art detection-style decoder breaks the ceiling the
hand-rolled baseline head hit (val mIoU 0.367 / honest val[grid] AP50 0.199). Constraint:
**single-frame only** for now.

Reference implementation read for the port: `/cluster/home/niacobone/MaskDINO`
(IDEA-Research MaskDINO, `maskdino/modeling/{transformer_decoder,pixel_decoder,criterion,matcher}`).

The head it replaced is retired to `legacy/` but stays runnable — it is the baseline every number
here is measured against. See `legacy/README.md`; its own story is archived in `docs/old/`.

---

## 1. What MaskDINO is (and what the retired baseline head was missing)

MaskDINO = Mask2Former's mask branch grafted onto DINO's detection decoder. The pieces that
matter, and how the retired baseline head compares:

| MaskDINO component | the retired baseline head | Kept here |
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

The hypothesis that motivated it: the retired baseline head plateaued at ~0.2 honest AP50 mainly on
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
   transformer decoder train — **20.5 M** params at the full recipe, vs ~6.5 M for the retired baseline head.
   VGGT is never touched, exactly as in every other run.
3. **Single scale in, three scales out** (§3) — VGGT is a plain ViT-style aggregator with one
   token resolution, so the res3/res4/res5 pyramid is synthesised ViTDet-style.
4. **19 classes, sigmoid-focal, no background column.** MaskDINO/DINO classify with `num_classes`
   sigmoid logits and represent "no object" as *all logits low*, whereas the retired baseline head used 20
   softmax logits with background at index 0. Ported faithfully → the eval protocol needs the
   `score_mode="sigmoid"` switch (§6). The width comes from `models/maskdino/head.py::
   NUM_SCANNET_CLASSES`; instances of the 20th `SCANNET_CLASSES` name (`otherfurniture`) are
   dropped rather than crashing the matcher — see §4.
5. **Mask resolution** is the VGGT patch grid (37×37) by default, so the mask metrics are computed
   on exactly the same grid as the retired baseline head. `--mask_upsample 2` gives 74×74 (a transposed-conv step
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
is the last layer only — identical cache footprint to every other run.

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
  single-frame restriction — and the reason the numbers are not directly comparable to the retired baseline head
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
| `scripts/eval_perframe.py` | scores an existing **legacy** checkpoint under this protocol (the apples-to-apples baseline) |
| `scripts/visualize_maskdino.py` | re-renders a finished run's figures from its checkpoint, without retraining (§6.4) |
| `slurm/train_maskdino.sh` | cluster job (stages the 500-scene official-GT tar; logs → `slurm/logs/`) |
| `slurm/visualize_maskdino.sh` | the same, for the re-rendering script over one or more run dirs |
| `tests/test_maskdino_model.py` | MSDeformAttn vs naive reference, pixel decoder, decoder configs, box ops, `head_config` round-trip, `initialize_box_type` guard |
| `tests/test_maskdino_loss.py` | matcher, criterion key set + perfect-prediction zero-loss, out-of-range-label guard |
| `tests/test_maskdino_train.py` | per-frame GT builder (incl. class drop), per-frame metric slicing, 60-step synthetic overfit |
| `tests/test_maskdino_multiframe.py` | cross-frame block, bundle GT + index expansion, bundle matcher, shared-query forward, S=1 equivalence, multi-frame overfit, bundle batching + scoring |
| `tests/test_maskdino_viz.py` | identity-keyed figure colouring: stable slots, winner-takes-all painting, colour survives per-frame reordering/filtering (§6.4) |
| `tests/test_maskdino_fullres.py` | the `--eval_full_res` ruler (§6.5): helpers, the grid-vs-full ruler difference, full_* keys in both eval paths |
| `tests/test_maskdino_consistency.py` | the cross-view consistency metrics (§6.6): planted-perfect and planted-switch cases, the case volume IoU cannot see, degenerate inputs, additive `bundle_*` keys |
| `tests/maskdino_fixtures.py` | `_tiny_head`, `_synthetic_targets` shared by the three test modules |
| `scripts/eval_3d_maskdino.py` | the 3D ruler (§9): official ScanNet 3D instance benchmark eval of a `--multi_frame` checkpoint |
| `train/scannet3d.py` | 3D benchmark data (§9): minimal PLY reader, superpoints, per-vertex GT ids, 25k frame/pose loading, class tables |
| `train/benchmark3d.py` | the official 3D instance evaluator, vendored + ported to Python 3 (§9.2) |
| `train/eval3d_geometry.py` | eval-time Sim(3) registration (Umeyama + similarity ICP), pixel→query assignment, vertex votes, superpoint majority (§9) |
| `slurm/eval_3d_maskdino.sh` | cluster job: stages the two val-312 tars, runs the 3D eval |
| `train/maskdino_viz3d.py` | the interactive 3D viewer's colour path (§9.7): checkpoint loading, the 3D ruler's query selection, identity-keyed per-pixel colours |
| `demos/demo_gradio.py` | the viewer itself — serves MaskDINO **and** the retired baseline-head checkpoints (§9.7) |
| `demos/dualview3d.py` | the synchronised GT\|prediction panels (§9.7): GLB-equivalent filtering, quantised payload, a dependency-free WebGL page with ONE camera |
| `scripts/view_ply.py` | a `.ply` → one self-contained HTML file, for looking at `--dump_ply` output without MeshLab (§9.7) |
| `tests/test_maskdino_viz3d.py` | feature-mode fidelity, max-over-views selection, colour survives per-view reordering, end-to-end on a tiny head (§9.7) |
| `tests/test_demo_gradio_maskdino.py` | the viewer's glue: checkpoint-family routing, scene dropdown vs what is on disk, GT/frame ordering, colouring path (§9.7) |
| `tests/test_dualview3d.py` | side-by-side view: vertex-for-vertex agreement with the GLB path, panels sharing points, payload round-trip, `.ply` → page (§9.7) |
| `tests/test_maskdino_eval3d.py` | the whole §9 stack CPU-only: PLY/GT fixtures, Umeyama/ICP, unprojection round-trip, votes + majority, evaluator vs hand-computed APs, synthetic end-to-end |

The only shared file this track modified is `train/eval_metrics.py`:
- an optional `score_mode="softmax"|"sigmoid"` argument (default `"softmax"` = previous
  behaviour, existing tests unchanged) so the same metric code can score sigmoid-focal
  predictions;
- `reshape(n, -1)` → `flatten(1)`, which fixes a crash on a zero-row prediction tensor (legal
  input once predictions are pre-filtered, see §6.3). Identical for every non-empty input;
- a new, self-contained `multiview_consistency_metrics` (§6.6) next to it — additive, called
  only from the `--multi_frame` eval, nothing existing routes through it.

## 6. Evaluation protocol — read this before comparing numbers

Metrics come from the *same* function as every retired baseline head
(`train/eval_metrics.py::compute_instance_segmentation_metrics`). Three things differ:

**6.1 Per-frame, not per-bundle.** Arms A–E score one 8-frame multi-view instance against its
8-frame GT mask (one IoU over the concatenated frames). This track scores each frame separately
and averages over frames, then over scenes. Frames with no GT instance are skipped. Different
task, different denominator → **the headline numbers are not interchangeable with the baseline head's
0.367 / 0.199.** Use `scripts/eval_perframe.py` to put a legacy checkpoint on this protocol.

**6.2 Sigmoid scoring, and two operating points.** With no background class, "is this query an
object?" is `max_c sigmoid(logit_c) ≥ threshold`. Every eval therefore reports two variants:

| variant | threshold | meaning |
|---|---|---|
| headline (`mIoU`, `AP50`, …) | `--score_threshold`, default **0.25** (MaskDINO's `OBJECT_MASK_THRESHOLD`) | closest analogue of the retired baseline head' "argmax ≠ background" filter |
| `*_all` | 0.0 — every query kept and ranked by score | the standard COCO detection protocol; also the only signal that moves early in training, because focal-trained sigmoid scores start near zero |

`mIoU_all` (best IoU over *all* queries) is a mask-quality ceiling, not a detection number —
read it next to `AP50_all`, never on its own.

**6.3 A prediction that claims no pixels in a frame is dropped, not counted as a false
positive** (`train/perframe.py::drop_empty_masks`, applied by both scorers). Without this rule
the protocol is unfair to the retired baseline head: a baseline-head query is *supposed* to be empty in the
frames where its object is not visible. Mask2Former/MaskDINO get the same effect by folding the
mask's mean foreground probability into the score. In the tests, the rule turns a spurious AP50
of 0.5 into the correct 1.0 on a planted-perfect example.

Everything is logged per eval into `<run_dir>/metrics.jsonl`.

**6.4 Figures are coloured by identity, not by rank (fixed 2026-07-29).** The RGB|GT|pred panels
in `<run_dir>/visualizations/` used to colour the *n*-th kept prediction with the *n*-th palette
slot. `keep` is re-filtered by `--score_threshold` and re-sorted by score in every frame, so the
same query drew a different colour in every view and the multi-view consistency the model
actually has was invisible — the single most common thing to misread in these figures. Two
independent causes, both fixed:

- colour is now `paint_identity_map(...)` keyed to a **frame-independent identity**: the query
  index for predictions (with `--multi_frame` that *is* the instance identity across views, §8.2)
  and the GT instance's `global_ids` for the GT panel. Score order still decides who wins an
  overlapping pixel — it no longer decides the colour;
- `imshow` is pinned to `vmin=0, vmax=20` with an explicit background slot, so matplotlib stops
  renormalising the colormap to each panel's own instance count.

The GT and prediction panels use **different identity spaces**, so their colours are not meant to
agree with each other — only with themselves across frames. Nothing about the metrics changed:
colours never entered the scoring path. Existing runs can be re-rendered without retraining via
`scripts/visualize_maskdino.py` / `slurm/visualize_maskdino.sh`.

**6.5 The full-resolution ruler (`--eval_full_res`, added 2026-07-30; measurements in §7.7).**
Every number above is computed *on the prediction grid*: GT is area-downsampled to
37×37 (74×74 with `--mask_upsample 2`) and the masks are compared there. On that ruler boundary
detail finer than a grid cell cannot be rewarded **even in principle**, so the "`--mask_upsample`
is neutral" verdict of §7.4 is partly a property of the ruler, not only of the model. Every
published protocol (and our own COCO track) scores at image resolution instead. `--eval_full_res`
adds that ruler *alongside* the grid one: predictions are bilinearly upsampled **in logit space**
to the dataset's 518×518 GT id map (cached per frame, int16, ~2 GB at 500 scenes) and scored as
`full_*` / `full_*_all` keys next to the unchanged grid keys. The kept prediction set is still
decided on the grid (same `drop_empty_masks` + top-k), so `full_*` isolates *mask-boundary
quality* from detection. `bundle_*` stays on the grid (the full-res volume would cost ~200× the
IoU memory and says nothing extra about cross-view consistency). Off by default; every number in
§7.1–7.6 is grid-resolution, §7.7 carries the full-res measurements. Tests:
`tests/test_maskdino_fullres.py`, including
the ruler-difference demonstration (a grid-perfect prediction scores 0.5 mIoU on full-res striped
GT). The GT-only ceiling of each grid on this ruler comes from
`scripts/scannet_mask_resolution_oracle.py` (`sbatch slurm/scannet_oracle.sh`) — the ScanNet
analogue of the COCO oracle (docs/MASKDINO_COCO.md §1); run/quote it before arguing about mask
resolution on ScanNet. When quoting: `full_*` numbers are still *our* metric implementation on
*our* split — they make mask-resolution claims honest, they do not make numbers
leaderboard-comparable (docs/RELATED_WORK.md). Note the 518×518 id map is itself a square resize
of the native 968×1296 annotation; scoring at native resolution (inverting the squash, as the
COCO track does) is a possible refinement, not the current implementation.

**6.6 Cross-view consistency (`bundle_view_consistency` / `bundle_id_switch`, added
2026-08-01).** Everything in §6.1–6.5 measures *how good a mask is*. Nothing measured whether the
model is **multi-view consistent**, which is the claim the whole `--multi_frame` design rests on
(§8.2) and the one thing that separates us from per-frame + fusion baselines
(docs/RELATED_WORK.md gap 2 — "consistency intrinsic to the query, not post-hoc … claim it").
`bundle_AP50` does not settle it: a query's mask *volume* can match a GT instance well on
average while a *different* query is the one that actually explains the object in each
individual view. These two keys make the claim measurable.

Both come from `train/eval_metrics.py::multiview_consistency_metrics`, reported per bundle by
`eval_scenes_multiframe` next to the existing `bundle_*` keys. The recipe:

1. **Match once, at bundle level.** Class-agnostic Hungarian on the IoU of the flattened
   `[S·h·w]` volumes — literally the assignment `class_acc` already uses, one dimension larger.
   Pairs with zero overlap are not matches. The prediction pool is the *threshold-free* bundle
   pool (`drop_empty_masks` + `--eval_topk`, no `--score_threshold`), so consistency is a
   property of the masks, not of the operating point.
2. **`bundle_view_consistency`** — for each matched GT instance, over the views where it is
   *visible*, the fraction with per-view IoU ≥ 0.5 against **its bundle-matched query**. 1.0 =
   the same query segments the instance in every view it appears in.
3. **`bundle_id_switch`** — for the same pairs, the fraction of visible views whose *best-IoU*
   query is not the bundle-matched one. 0.0 = no view is better explained by somebody else.
   Views where **no** query overlaps the instance are excluded from this fraction (nothing owns
   it there, so nothing switched); that failure is what `view_consistency` counts. The two are
   therefore complementary, not redundant: a **miss** lowers consistency and leaves id_switch
   alone, a **hand-off** moves both.

Both are means over matched GT instances, so both are recall-flavoured — an instance no query
overlaps at all never enters either mean. `bundle_num_matched` is reported alongside for exactly
that reason, and because the degenerate cases (no GT, no predictions, nothing matched) return
**0.0 for both keys**: a zero `id_switch` there means *undefined*, not *perfect*. Read the three
together.

Everything stays on the mask grid, for the same reason `bundle_*` does (§6.5). Purely additive:
no existing key, threshold or scoring path changed, and the single-frame eval is untouched
(there is no cross-view identity to measure when queries are per-frame). Tests:
`tests/test_maskdino_consistency.py`, including the case that motivates the metric — a set of
per-view-perfect queries that hands the object off in every frame scores 0.25/0.75 where one
shared query scores 1.00/0.00.

**No measurement yet.** The metric was added after job 9071415 (§8.2, the current multi-view
best) finished, and it is computed inside the eval loop rather than from a saved artefact, so
the first numbers come from the next `--multi_frame` run. Expect the ablations of §7.4.1 to be
the interesting cut: `--no-cross_frame_attn` should cost consistency *specifically* if the
block is what carries cross-view identity.

## 7. Results

**Every number lives in `docs/RESULTS.md`** — one home per fact. This section keeps only the two
measurements whose *method* belongs next to the architecture (§7.6, §7.7) plus one engineering
note. The run-by-run narrative that used to sit here — job ids, dated readings, the machinery
check — is archived verbatim in `docs/old/MASKDINO_RESULTS_HISTORY.md`.

Subsection numbers are kept stable because the rest of the repo cites them.

| was here | what it measured | now read |
|---|---|---|
| §7.1 | machinery / GPU smoke test | archive |
| §7.2, §7.2.1 | data scaling 50→190→490, single-frame ablations | `RESULTS.md` §2 |
| §7.3 | the retired baseline head on this protocol | `RESULTS.md` §1, §2 |
| §7.4, §7.4.1 | bundle features, mask upsample, view draws, multi-frame ablations | `RESULTS.md` §2, §3 |
| §7.5 | 77-scene official-val read-out | `RESULTS.md` §1.1 |
| §7.8, §7.8.1 | official 1201/312 runs; what cross-frame attention buys | `RESULTS.md` §6 |

Four standing conclusions from that body of runs, kept here because the rest of this document
argues against them:

1. **Data scale dominates every component.** +0.26 AP50 from 50→490 scenes, versus ≤0.05 from
   removing any single MaskDINO ingredient. No component is individually decisive on the
   single-frame ruler.
2. **Cross-frame attention is the one exception** — the only individually decisive component this
   track has found, and its job is *identity*, not recognition (§8.2, `RESULTS.md` §3, §6).
3. **Multi-view consistency has a measured price** in per-frame accuracy: bundle features cost
   ~0.05 AP50 per frame and buy the multi-view metric.
4. **Recognition binds, not resolution** (§7.7).

### 7.2.2 Cost note — eval must not scale with the training set

The first N=200 submission (job 8748972) reached only epoch 2 in 30 minutes: it scored **all 190
train scenes** at every eval, and `_average_precision` loops over every kept prediction at 10 IoU
thresholds (~1600 frames × ~180 ms ≈ 5 min per eval). Two fixes: `--eval_topk 100` (COCO's
`test_topk_per_image` — protocol-correct *and* 3× faster per frame) and `--eval_train_scenes 10`
(the train metric is only an overfit read-out). Eval went ~180 s → ~6 s.

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

### 7.7 The resolution verdict (2026-07-30, jobs 9073136 / 9072738 / 9072749 / 9072761)

Three mutually consistent measurements close the "is the 37×37 grid the bottleneck?" question
on ScanNet. All on the full-resolution ruler of §6.5.

**The GT-only ceiling** (`scripts/scannet_mask_resolution_oracle.py`, job 9073136, val scenes
0080–0089, full JSON in the output dir):

| prediction grid | mIoU | AP50 | AP75 | mAP |
|---|---|---|---|---|
| **37×37** (native patch grid) | 0.910 | **0.956** | 0.919 | 0.863 |
| 74×74 (`--mask_upsample 2`) | 0.963 | 0.992 | 0.982 | 0.953 |
| 148×148 (`--mask_upsample 4`) | 0.983 | 0.997 | 0.995 | 0.988 |
| 259×259 | 0.994 | 1.000 | 1.000 | 0.999 |
| 518×518 (sanity) | 1.000 | 1.000 | 1.000 | 1.000 |

Contrast with COCO, where the same grid caps a perfect model at 44.7 AP
(docs/MASKDINO_COCO.md §1): ScanNet's furniture-scale objects lose only ~0.04 AP50 of ceiling
to the 37×37 quantisation. The model sits at ~0.69 AP50 against a 0.956 ceiling —
**recognition binds, not resolution.**

**The model on both rulers** (N=490, full recipe, `--eval_full_res`, peak epoch):

| Run | grid mIoU / AP50 | full-res mIoU / AP50 | full AP75 / mAP |
|---|---|---|---|
| 9072738 — the bar, re-run | 0.670 / 0.690 | 0.654 / 0.673 | 0.526 / 0.465 |
| 9072749 — `--mask_upsample 2` | 0.650 / 0.685 | 0.648 / 0.680 | 0.505 / 0.460 |

Three readings:

1. **The bar reproduces** (0.670/0.690 vs the original 0.669/0.699) — a free seed-level
   reproducibility check, and the grid→full-res drop is only ~0.017 AP50.
2. **`--mask_upsample 2` stays neutral on the honest ruler too** (full AP50 0.680 vs 0.673,
   inside noise). The §7.4 "neutral" verdict was *not* an artefact of the grid-resolution
   scoring; the oracle explains why (the 37×37 ceiling is far above the model).
3. Job 9072761 (`--mask_upsample 4 --train_num_points 12544`) **OOM'd in backward at 19 min**
   (148² mask tensors, batch 8, 24 GB card; the ScanNet trainer has no gradient accumulation,
   unlike the COCO one). Given 1–2, running it is not worth the engineering — the oracle already
   bounds what it could buy (~0.004 AP50 ceiling over ×2).

**Consequence for the plan:** mask resolution work on ScanNet is de-prioritised; the honest
lever for boundary quality would be the token grid (docs/MASKDINO_COCO.md §1.3, VGGT at higher
input resolution), and even that is bounded by the 0.956→0.99 ceiling gap. Quote §7.7 whenever
resolution comes up.

## 8. The multi-frame extension

Ordered by cost, each step reusing everything above.

### 8.1 `--feature_mode bundle` — multi-view *features*, single-frame decoder

Runs the frozen aggregator once over all S frames of a bundle instead of once per frame, so
VGGT's global attention makes each frame's tokens multi-view aware while the decoder still sees
one frame at a time. No architectural change, no cross-frame identity, no extra parameters — only
the feature cache is built differently. Implemented in `train/maskdino_data.py::extract_features`;
first run at N=490 submitted 2026-07-28 (§7.4). **Verdict: −0.048 AP50 per frame as a standalone
change (§7.4), but required by the multi-frame decoder — swapping it out costs −0.147 bundle AP50
(§7.4.1). The multi-view information in the frozen tokens is what the cross-frame attention
consumes.**

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
   views, mask BCE+Dice over the concatenated `[S·h·w]` volume (the retired baseline head' multi-view mask
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
multi-view protocol of the retired baseline head (one IoU over the concatenated volume, one class score per query
= max over views), which was meaningless while queries were per-frame and is comparable to the baseline
C's 0.367 / 0.199. Never mix the two (docs/RESULTS.md §1). Since 2026-08-01 that same eval also
reports **`bundle_view_consistency` / `bundle_id_switch`** (§6.6): whether one query really owns
an instance in *every* view, which is the property this whole section claims and which
`bundle_AP50` alone cannot distinguish from a per-view hand-off. No run has been scored on it
yet — see §6.6.

Flags: `--multi_frame` (sample = a bundle of `--num_frames` frames), `--batch_bundles`
(default 1 → 8 frames/step, the same GPU footprint and the same steps/epoch as the single-frame
runs), `--no-cross_frame_attn` (ablate the block, keeping shared init + bundle matching).

**Results (2026-07-28/30).** Per frame the full multi-frame model is neutral against its
bundle-features control (−0.021 AP50, §7.4); on the baseline head's multi-view ruler it scores
**0.535 mIoU / 0.494 bundle AP50** vs the baseline head's 0.367 / 0.199. The two ablations (§7.4.1) localise
the result: cross-frame attention is worth 0.183 bundle AP50 and bundle features 0.147 — the
former is the only individually-decisive component found anywhere in this track.

**Adding the best data recipe helps here too** (job 9071415, 2026-07-30: `--multi_frame
--feature_mode bundle --bundles_per_scene 2 --color_jitter 0.2`, EPOCHS=30, peak at 19):
per-frame **0.643 / 0.667** (vs 0.621 / 0.630) and a **new multi-view best 0.539 mIoU /
0.515 bundle AP50** (+0.021 over 0.494). Both peaks fall on the same epoch in this run, so
`checkpoint_best_ap50.pth` captures the multi-view headline as well — but the two peaks *can*
diverge (§7.4.1), so `--multi_frame` runs now also save **`checkpoint_best_bundle.pth`**,
selected on val `bundle_AP50` (docs/todo.md 2b, done 2026-08-01). Off for single-frame runs —
the key doesn't exist there. The end-of-run summary line prints its epoch alongside the other
two when present.

### 8.3 3D anchors instead of 2D boxes — `--anchor_3d` (todo 2d, implemented 2026-08-04)

Replace the DAB 4-d box with a 3D anchor read off VGGT's own point head. The archived 3D-anchored query experiment showed 3D anchors
alone don't beat 2D queries, but the archived 3D-anchored query experiment had no box refinement, no DN and no deep supervision — the
ingredients that make anchors work in DINO.

**Framing (settled 2026-07-28, docs/RELATED_WORK.md).** FAST3DIS (arXiv 2603.25993) already
publishes exactly this mechanism — a learned 3D anchor generator plus project-and-sample
cross-attention — on a LoRA-adapted Depth-Anything-V3. So this step is an **ablation inside our
own controlled study** ("3D anchors vs 2D DAB boxes, same frozen backbone, same data, same
protocol", which nobody has run and which re-tests the archived 3D-anchor negative result), **not** a new
mechanism. Budget it accordingly.

**Why it was promoted above the lifting workstream (2026-08-04).** §9.10 measured what our masks
score with a *perfect* 2D↔3D bridge: 0.156 AP50, against 0.65 AP50 for the same checkpoint on the
per-frame ruler. So what binds at 0.156 is not lifting but the 3D-instance criterion — one query
owning a whole object across *all* its views. That is multi-view completeness and identity, i.e.
the decoder, and the anchor is the decoder's positional prior.

**Dependency: on top of §8.2, not before it.** A 3D anchor is only meaningful when a query is one
instance across views — with per-frame queries it is just a 2D box plus a depth. `--anchor_3d`
therefore **requires `--feature_mode bundle`** (a hard error otherwise): in `single` mode the
aggregator sees one frame at a time, so each frame's pointmap is in its own coordinate frame and
an anchor shared across views has no meaning. Without `--multi_frame` it warns.

#### The design as built

1. *Geometry cache* (`train/maskdino_data.py::patch_token_positions`, `extract_features(...,
   need_xyz=True)`). The frozen point head runs on the aggregator output the cache already has,
   and only the **per-patch-token 3D position** is stored: the confidence-weighted mean of the
   token's 14×14 pixels, `[S, 37·37, 3]` in fp16. **Measured: 65.71 kB per bundle against
   45.02 MB of tokens — +0.146 %** (~178 MB over the whole 1201×2 + 312 official-split cache),
   and no measurable caching-time cost. The ~26 MB pointmap it comes from is never stored. The
   pooling is re-implemented here rather than imported from frozen
   `legacy/d4rt/models/anchor_queries.py`, so the archived 3D-anchored query experiment's published numbers cannot move.
   Positions are normalised **per bundle** to zero mean / unit RMS radius
   (`models/maskdino/anchor3d.py::normalize_token_xyz`) — the softmax temperature below is one
   learned scalar per query and only means something in a comparable coordinate frame. The
   centre/scale are estimated on the tokens at or above the bundle's **median** point-head
   confidence, because the unreliable tail of the pointmap is what would otherwise drag them.
2. *Anchor = `(x, y, z, log r)` per query per bundle* — deliberately 4-d, the same width as the
   DAB box it replaces. Two-stage selection already picks top-k memory tokens, so 3D two-stage
   costs a gather: `pyramid_token_xyz` gives every level of the ViTDet pyramid a position (level
   0 verbatim, coarser levels **nearest**-resampled so a cell straddling a depth discontinuity
   keeps a real surface position), which makes the positions indexable with the very same top-k
   index. `r` starts at 0.25 of the bundle's RMS radius for every query. Without two-stage the
   anchors are a learned `nn.Embedding(num_queries, 4)` instead.
3. *Per-view reference points without camera math* (`anchor3d.py::project_anchors`). Every patch
   of view *f* has a position, so the query's 2D reference in that view is a **soft nearest
   patch**:

   ```
   w        = softmax(-‖p_patch − a‖² / r²)        over that view's 37×37 grid
   (cx, cy) = Σ w · (u, v)_patch
   (w,  h)  = 2 · sqrt(Var_w[(u, v)] + (0.5/37)²)
   ```

   The reference *size* falls out of the same distribution as its centre — this is the sketch's
   "3D point **(+ scale)**", realised as the softmax temperature — floored at one patch so a
   collapsed anchor cannot degenerate the deformable sampling to a single point. Differentiable
   in the anchor, needs no intrinsics or extrinsics (unlike FAST3DIS, which projects with its
   predicted camera), and degrades gracefully where the pointmap is unreliable. The result is a
   `(cx, cy, w, h)` in (0,1), so `gen_sineembed_for_position`, `MSDeformAttn` and `pred_box` are
   all untouched. Iterative refinement predicts Δ(xyz, log r) on the anchor instead of Δbox.
4. *Losses unchanged.* The 2D box stays the prediction target of `_bbox_embed` per view — the box
   at layer *l* is `sigmoid(bbox_embed(h_l) + inverse_sigmoid(ref_l))` exactly as before, only
   with `ref_l` now coming from the anchor's projection — so the matcher, the box/GIoU losses,
   deep supervision and DN all keep working as they do today. Only the query *positional prior*
   and the deformable sampling locations change.

#### Three deviations from the §8.3 sketch, all forced (and one confound, stated)

- **The anchor refinement is NOT detached**, unlike the DAB box. `bbox_embed` can be detached
  because it learns from the box loss; the 3D anchor has *no loss of its own*, so a detached
  Δ(xyz, log r) head would receive exactly zero gradient and never train. The gradient reaches it
  through the soft projection of the following layers' references — which is precisely why the
  sketch specified a differentiable projection.
  `tests/test_maskdino_multiframe.py::test_anchor3d_overfit` asserts the head is in the graph.
- **Δxyz is the mean over the bundle's views** of `anchor_embed(output)`. The anchor is one 3D
  point per bundle, and the mean is the permutation-equivariant reduction — the same reasoning
  that keeps `CrossFrameAttention` free of a frame positional encoding (§8.2).
- **Confidence is used for the intra-patch pooling and the robust normalisation, not as a bias
  inside the softmax.** One fewer mechanism, and the normalisation is where bad pointmap values
  actually do damage.
- **The confound, named:** `initialize_box_type=bitmask` still seeds the *initial* (pre-decoder)
  box prediction and the two-stage/interm losses, so those are byte-identical to the control —
  but from decoder layer 0 onward the anchor projection *is* the reference, so the mask-enhanced
  box init no longer reaches the 9 decoder layers. The ablation therefore moves "2D box refined
  by `bbox_embed`" → "3D anchor refined by `anchor_embed`" as one unit. That is inherent: the
  anchor *is* the positional prior, and you cannot have the prior come from the mask and from the
  3D anchor at once (FAST3DIS has no bitmask init either).

**Files:** `models/maskdino/anchor3d.py` (the geometry, no parameters), the `anchor_3d` argument
threaded through `head.py` → `decoder.py` → `decoder_layers.py`, the cache and gather in
`train/maskdino_data.py`, `token_xyz=` at every head call site (`scripts/train_maskdino.py`,
`train/maskdino_eval.py`, `scripts/eval_3d_maskdino.py`, `demos/demo_gradio.py` via
`train/maskdino_viz3d.py::head_token_xyz`). Off by default everywhere; the head raises rather
than silently falling back if it is on and no positions are supplied.

**Note for the 3D ruler.** An `--anchor_3d` checkpoint is the one case where the *model* consumes
predicted geometry. In `--transfer_mode gt_projection` (§9.10) "no predicted geometry" then
describes the 2D→3D **bridge** only, not the whole column. `scripts/eval_3d_maskdino.py` runs the
point head for such checkpoints in both modes and says so here rather than leaving it implicit.

Cheap follow-ups that need one flag each: `--mask_upsample 2` (74×74 masks — currently supervised
on the 37×37 patch grid) and `--bundles_per_scene 2 --color_jitter 0.2` (more frame draws without
new scenes; costs cache memory). Both answered at N=490 (§7.4: upsample neutral, extra draws
+0.030 AP50 and saturating at 2 per §7.4.1).

#### Result (job 9634920, 2026-08-04) — **neutral on AP, real on identity**

Control: job 9386666 (`maskdino_sf_list1201_mf_20260802_133826`, §7.8). A `config.json` diff of
the two runs returns **exactly one key — `anchor_3d`** — and the train/val scene lists are
byte-identical, so the anchor is the only variable. 233.3 min of training against the control's
203.6 (+15 %, all of it the per-layer projection; the feature cache cost +0.146 %).

| | control (2D DAB box) | `--anchor_3d` | Δ |
|---|---|---|---|
| per-frame mIoU / AP50 *(each run's own per-frame peak)* | **0.623 / 0.650** | 0.611 / 0.641 | −0.012 / −0.009 |
| per-bundle mIoU / AP50 *(each run's `checkpoint_best_bundle`)* | **0.529 / 0.525** | 0.524 / **0.527** | −0.005 / **+0.002** |
| `bundle_view_consistency` | 0.717 | **0.723** | +0.006 |
| **`bundle_id_switch`** (lower is better) | 0.498 | **0.409** | **−0.089 (−18 % rel.)** |
| `bundle_num_matched` | 14.08 | 14.05 | −0.03 |

**The 2D verdict: the 3D anchor does not buy 2D AP. It buys identity stability.** Read on before
concluding it is not worth having — on the 3D ruler that identity is worth **+67 % AP50**.

- Per-bundle AP50 +0.002 and consistency +0.006 are flat — well inside the run-to-run wobble the
  control itself shows across its last three epochs (0.514 → 0.516 → 0.525).
- Per-frame is **mildly negative**, −0.009 AP50, and the anchor run trailed the control on the
  per-frame ruler at *every* epoch. Consistent with §8.1's precedent: consistency machinery has a
  per-frame price. Part of it is likely the named confound above — the 9 decoder layers no longer
  get the mask-enhanced box init.
- **`id_switch` is the one thing that moved, and it is not noise: the anchor run is better in
  12/12 epochs**, by a mean of 0.084 (range 0.052–0.111), with the gap present from epoch 1 and
  stable to the end. At epoch 7 it had already reached the control's *final* value.

**Why this is worth reporting even though AP did not move.** §7.8.1 found that removing
cross-frame attention moved `id_switch` 0.498 → 0.682 **and** bundle AP50 0.525 → 0.389 — the two
travelled together, which is what let us claim the block's job is identity. Here they
**dissociate**: identity improves 18 % relative while AP50 sits still. So the two are not the same
axis, and `bundle_AP50` alone cannot see what the 3D anchor does — precisely the situation §6.6
built the consistency metrics for. It also means the residual §9.10 reading 4 identified
("multi-view completeness *and* identity") is not one quantity: this run bought the identity half
and left the completeness half untouched, and the AP50 that the 3D-instance criterion rewards
followed the half that did not move.

**The archived 3D-anchored query experiment, re-tested.** The archived 3D-anchored query experiment's negative result was "3D anchors alone don't beat 2D queries", on a
decoder with no box refinement, no DN and no deep supervision. With all three present the answer
is not a reversal but a sharpening: 3D anchors are **not worse** here (unlike the archived 3D-anchored query experiment, which lost
outright), they are **AP-neutral**, and they are **better on the one property a 3D anchor should
plausibly help** — the same query staying on the same object as the viewpoint changes.

#### The 3D result — the identity gain is worth +67 % AP50 (jobs 9670882 / 9670883, 2026-08-04)

The 2D read-out above says "AP-neutral". **On the 3D ruler that is emphatically not what happens.**
Same script, same 312 val scenes, **17.42 frames/scene in both runs**, every knob at its default,
0 failed scenes — only the checkpoint differs.

| transfer mode | control (2D box) | `--anchor_3d` | Δ AP50 |
|---|---|---|---|
| `unproject` (**the headline protocol**) | 0.023 / 0.067 / 0.268 | **0.038 / 0.112 / 0.360** | **+67 %** |
| `gt_projection` (posed, §9.10) | 0.060 / 0.156 / 0.408 | **0.104 / 0.257 / 0.504** | **+65 %** |
| — 17-class diagnostic, unposed | 0.024 / 0.071 / 0.284 | 0.040 / 0.119 / 0.381 | +67 % |
| — coverage, unposed (voted / annotated-assigned) | 0.153 / 0.635 | 0.177 / 0.666 | +16 % / +5 % |
| — kept queries per scene | 97.6 | **89.0** | −9 % |

**This makes the unposed row competitive with the published unposed cluster** — FAST3DIS
0.038 / 0.096 / 0.316 and IGGT 0.028 / 0.112 / 0.287 — reached on a **strictly frozen** backbone
where both of theirs are LoRA-adapted. See docs/RESULTS.md §5 before quoting it anywhere.

**Why the 2D rulers could not see this, and the 3D one could.** The per-frame ruler scores each
view independently, so identity is worth nothing to it. The per-bundle ruler scores a mask volume
over the **8** training views, so identity is worth something but is diluted. The 3D ruler runs
the head at **S ≈ 17.4** and then *votes per vertex*: if a different query wins the object in each
view, the votes for that vertex split and the superpoint majority becomes noise. `id_switch` is
therefore not a cosmetic property here — it is the exact failure mode the lifting step integrates
over, and it compounds with view count. The supporting signature is in the table: the anchor model
keeps **9 % fewer** queries yet covers **16 % more** vertices. Fewer, cleaner, more view-consistent
instances — precisely what a per-vertex vote rewards.

**A correction to this document's own reasoning.** The first draft of the 2D verdict inferred from
the dissociation that "the 3D-instance criterion follows completeness, and this run bought
identity, so expect little 3D movement". That inference was **wrong, and the measurement
falsified it**: of §9.10 reading 4's two named residuals — multi-view *completeness* and
*identity* — it is **identity** that the 3D ruler is most sensitive to. Keep the dissociation
finding (identity and `bundle_AP50` really are separate axes); discard the prediction that came
with it.

**Consequence for the protocol set.** `bundle_AP50` at S = 8 is a **poor proxy for the 3D ruler**.
A mechanism can be flat on it and worth +67 % AP50 in 3D. Anything that touches cross-view
identity must be scored on the 3D ruler before it is judged — the 2D per-bundle number alone will
under-report it. That also raises the value of todo 2e (bundle width), since the effect is
view-count-dependent by construction.

**Why the controls cannot have drifted — established from the diff, not from a re-run.** Only two
commits separate the control rows from these: `8ad9aab` (the `--transfer_mode` split, whose
no-op-ness on the unposed path §9.10 verified by re-running job 9607208 to a byte-exact
0.084 / 0.236 / 0.375) and `7c4e890` (this work). `7c4e890` **does not touch
`train/eval3d_geometry.py`, `train/benchmark3d.py` or `train/scannet3d.py` at all**, and its only
change to `scripts/eval_3d_maskdino.py` is the anchor branch guarded by
`if model.head.head_config.get("anchor_3d", False)` — provably inert for a 2D-box checkpoint. The
posed control (job 9607206) additionally post-dates `8ad9aab`, so **that comparison is same-code
by construction**. Both comparisons independently show ~+66 %, which is the stronger evidence:
they share no code path between the mesh and the mask.

#### Replicated across seeds (2026-08-07, jobs 9901124 / 9901125)

The 2D verdict above rested on **one run against one control**, the standing weakness of every
row in this track. Both arms were re-trained with `--seed 1`, everything else identical:

| run | per-frame AP50 | per-bundle AP50 | `id_switch` ↓ | `view_consistency` ↑ |
|---|---|---|---|---|
| control, seed 0 (9386666) | 0.6491 | 0.5249 | 0.4982 | 0.7167 |
| control, seed 1 (9901125) | 0.6505 | 0.5342 | 0.4710 | 0.7173 |
| `--anchor_3d`, seed 0 (9634920) | 0.6408 | 0.5271 | 0.4088 | 0.7229 |
| `--anchor_3d`, seed 1 (9901124) | 0.6466 | 0.5362 | 0.4074 | 0.7279 |

**Seed-to-seed spread on per-bundle AP50 is ≈ 0.009 in both arms** — which is the number to
compare every ΔAP50 in this document against, and it retires "could be seed noise" for the larger
effects (cross-frame attention 0.183, bundle features 0.147, bundle width 0.027) while placing the
3D-anchor per-bundle delta (+0.002 / +0.002) firmly *inside* it. That is the point: **AP-neutral
is now a measured statement, not an absence of evidence.**

**The identity effect survives replication.** `id_switch` falls by **0.089** (seed 0) and
**0.064** (seed 1) — same sign, both 2.4–3.3× the control arm's own seed spread (0.027). Ditto
`view_consistency`, up in both. The per-frame cost also replicates (−0.008 / −0.004).

#### …and so does the 3D result (2026-08-07, jobs 9979100 / 9979101)

The 2D replication above left the *3D* claim resting on one run against one control — the
weakness that mattered most, since the 3D ruler is the only protocol placeable next to published
work. Both seed-1 checkpoints were then scored on it, defaults, 312 scenes, 0 failures, **17.42
frames/scene in all four runs**:

| run | 18-class AP / AP50 / AP25 | **class-agnostic** | kept queries | voted vertices |
|---|---|---|---|---|
| control, seed 0 (9861563) | 0.023 / 0.067 / 0.268 | 0.013 / 0.050 / 0.320 | 97.6 | 0.153 |
| control, seed 1 (**9979101**) | 0.025 / 0.075 / 0.313 | 0.016 / 0.059 / 0.348 | 97.8 | 0.147 |
| `--anchor_3d`, seed 0 (9866391) | 0.038 / 0.112 / 0.360 | **0.042 / 0.138 / 0.504** | 89.0 | 0.177 |
| `--anchor_3d`, seed 1 (**9979100**) | 0.037 / 0.112 / 0.342 | **0.039 / 0.129 / 0.485** | 90.4 | 0.168 |

**Seed spread on the 3D ruler is ≈ 0.009 class-agnostic AP50 in both arms** — the same figure the
2D per-bundle metric shows, which is a useful coincidence to know. The anchor effect is
**+0.088 / +0.070** across the two seeds, i.e. **~9× that spread**, so the 3D gain is not a
single-run artefact. The class-aware ΔAP50 replicates too (+67 % / +49 %), as does the collapse's
sign (only the anchor arm gains on collapse: +0.026 / +0.017, both controls lose), and so does
the mechanism's signature: **~8 % fewer kept queries at ~15 % more voted vertices**, in both seeds.

**One claim must be weakened, and it is the AP column.** Quoted against FAST3DIS's
0.038 / 0.096 / 0.316 and IGGT's 0.028 / 0.112 / 0.287, the two seeds read:

| | AP | AP50 | AP25 |
|---|---|---|---|
| ours, seed 0 | 0.042 | 0.138 | 0.504 |
| ours, seed 1 | 0.039 | 0.129 | 0.485 |
| FAST3DIS | 0.038 | 0.096 | 0.316 |
| IGGT | 0.028 | 0.112 | 0.287 |

**AP50 and AP25 are robust leads** (1.34–1.44× FAST3DIS, 1.15–1.23× IGGT on AP50; 1.53–1.59× and
1.69–1.76× on AP25 — every seed, every competitor). **AP is a tie with FAST3DIS**: 0.039–0.042 vs
0.038, inside our own seed spread (0.003 on that column). Say "we match FAST3DIS on AP and lead on
AP50/AP25, and lead IGGT on all three" — **not** "ahead on all three", which was true of seed 0
alone and is what the pre-2026-08-07 wording claimed.

#### Should it be the default?

On the 2D rulers, no (+15 % training time, −0.009 per-frame AP50, and the per-bundle gain is
inside seed noise). On the 3D ruler — the only protocol that is placeable next to published work —
it is the largest single-flag gain in the track, and §8.4 reading 4 confirms it beats the bundle-
width flag there at half the wall clock. **Recommended for any run whose target is the 3D
benchmark; still off by default**, so no completed 2D number moves.

### 8.4 Bundle width — views per bundle (`--num_frames`, todo 2e, opened 2026-08-04)

**Why now.** §9.10 reading 4 put the decoder back in play on the 3D ruler: with a *perfect*
2D↔3D bridge our masks score 0.156 AP50 while the same checkpoint scores 0.650 per frame, so
what is missing is **multi-view completeness and identity** — one query owning an object across
*all* its views. That is exactly what §8.2's shared queries are for, and the one parameter of
§8.2 that has never moved is how many views a bundle has: `--num_frames 8`, in every multi-frame
run ever.

Two independent reasons to widen it:

1. **A train/test mismatch nobody had closed.** `scripts/eval_3d_maskdino.py` runs the head with
   `frames_per_sample=S` where S is the *whole* frame set of the scene — **17.4 frames on
   average** (§9.10). The head is trained at S=8 and scored at S≈17. `CrossFrameAttention` is
   permutation-equivariant and carries no frame positional encoding (§8.2), so it *runs* at any
   S, but 300 shared queries have never been asked to own a 17-view volume during training.
2. **View count is a named, real residual difference** against SegVGGT (§9.9: "~75–100 views to
   our 17"), not a protocol artefact — §9.10 reading 3.

**`--eval_num_frames`** (default unset = unchanged). Widening the training bundle silently moves
the per-bundle ruler: `bundle_*` scores one query against a whole `[S, h, w]` volume, and a
volume over 16 views is a strictly harder object than one over 8, so a bare `--num_frames 16` run
cannot be laid next to the 0.529 / 0.525 baseline. This flag pins the **val** bundle width while
train widens (`train/maskdino_data.py::bundle_frames_for_split`); the eval reads S off each
cached bundle scene by scene, so the two widths coexist inside one run. Covered by
`tests/test_maskdino_multiframe.py::test_eval_num_frames_pins_the_bundle_ruler`. Train scenes
always use `--num_frames` — the diagnostic train metric is therefore on the wide ruler and is not
comparable to the val row, as usual.

Cost: the feature cache is linear in frames, so S=16 at `--bundles_per_scene 2` is ~230 GB
(1201 × 2 × 16 × 5.63 MB + val), i.e. ~26 CPU × 13 GB and a GPU with headroom — this is the
first run in the track that needs an A100 80 GB rather than a 4090.

#### Runs (2026-08-04)

| job | config | vs | reads |
|---|---|---|---|
| **9668639** `_mf_s16` | `--num_frames 16 --eval_num_frames 8`, b2, jitter 0.2, 12 ep | job 9386666, **one flag different** | does bundle width alone move per-frame / per-bundle / consistency? |
| **9668652** `_mf_s16_long` | same but 20 ep, val also at 16 | — | the intensive model; its own (16-view) bundle ruler, and the checkpoint for the 3D ruler |
| **9668726** `_mf_s16_b1` | `--num_frames 16`, b1, 24 ep, `--eval_num_frames 8` | insurance | 168 GB / any 24 GB GPU, so it schedules when the A100s are full. `--bundles_per_scene 1` makes `--color_jitter` inert (only extra bundles are jittered); step-matched at 28.8 k |

#### Result — widening the bundle helps on every axis (2026-08-06)

All rows on the official 1201/312 split, **val pinned to 8-view bundles** except where noted, so
every `bundle_*` figure below is on the same ruler as the 0.525 baseline.

| run | per-frame mIoU / AP50 | per-bundle mIoU / AP50 | `view_consistency` ↑ | `id_switch` ↓ | `num_matched` |
|---|---|---|---|---|---|
| **control** S=8, b2, 12 ep (9386666) | 0.623 / 0.650 | 0.529 / 0.525 | 0.717 | 0.498 | 14.1 |
| **S=16**, b2, 12 ep (**9668639**) | **0.627 / 0.662** | **0.549 / 0.552** | **0.726** | **0.385** | 14.0 |
| S=16, **b1**, 24 ep, no jitter (9668726) | 0.609 / 0.641 | 0.541 / 0.544 | 0.712 | 0.345 | 14.0 |
| S=16, b2, 20 ep, **val at 16** (9668652) | 0.627 / **0.669** | 0.561 / 0.594 † | 0.710 | **0.323** | 14.4 |

† **different ruler** — a 16-view volume, strictly harder than the 8-view one every other row is
scored on. It is *above* the control's 8-view 0.525 anyway, but it is not a like-for-like cell.

**Reading 1 — one flag, +0.027 bundle AP50, and identity is where it comes from.** Job 9668639
differs from the control by `--num_frames 16` alone. Per-frame moves +0.012 (0.650 → 0.662),
per-bundle moves **+0.027** (0.525 → 0.552, +5 % rel.), and `bundle_id_switch` falls
**0.498 → 0.385** (−23 % rel.) while `bundle_num_matched` is flat at 14.0–14.1. Same signature
as §7.8.1's cross-frame-attention cut, read forwards instead of backwards: **recognition
unchanged, identity improved.** Training the shared queries on twice as many simultaneous views
teaches them to *stay* on their object, which is precisely the property §9.10 reading 4 said was
binding.

**Reading 2 — it is the width, not the extra frames.** The obvious objection to 9668639 is that
S=16 × b2 shows the model 32 frames per scene per epoch against the control's 16, so the gain
could be data. Job 9668726 rules that out: `--bundles_per_scene 1` at S=16 shows **16 frames per
scene per epoch — exactly the control's frame budget** — with `--color_jitter` inert and no extra
draws, and it still reaches per-bundle 0.544 (vs 0.525) and `id_switch` 0.345 (vs 0.498). A run
that is *frame-matched and augmentation-poorer* than the control keeps almost all of the
identity gain. The wide bundle is doing the work; the extra frames add the per-frame AP50 on top
(0.641 vs 0.662).

**Reading 3 — the effect keeps going, and the harder ruler agrees.** Job 9668652 (20 epochs, val
left at 16 views) posts the best per-frame AP50 anywhere on the official split (**0.669**) and
drives `id_switch` to **0.323**, monotonically, with no sign of a floor: 0.537 → 0.323 over
epochs 1–19. Its per-bundle 0.594 is measured over 16-view volumes, so it clears the control's
8-view 0.525 on a strictly harder object. Whether width keeps paying past 16 is the obvious next
question and is *not* answered here.

**Cost.** 11 h 26 (9668639) vs 5 h 42 for the control, ~230 GB of feature cache, A100 80 GB.
Roughly 2× the wall clock and 2× the host RAM for +0.027 bundle AP50 and −0.113 id_switch.

#### Reading 4 — on the 3D ruler width pays, but LESS than `--anchor_3d` (2026-08-07)

§8.3's lesson was that `bundle_AP50` at S=8 is a poor proxy for the 3D ruler, so both width
checkpoints were scored there directly (jobs 9901143 / 9901663 / 9901664 / 9901665, val-312, 0
failures, all knobs default). All rows 18-class, and class-agnostic in brackets (§9.11):

| checkpoint | unposed | posed |
|---|---|---|
| control S=8 (9386666) | 0.023 / 0.067 / 0.268 [0.013 / 0.050 / 0.320] | 0.060 / 0.156 / 0.408 |
| **S=16** (9668639) | 0.033 / 0.098 / 0.336 [0.023 / 0.080 / 0.391] | 0.083 / 0.216 / 0.488 [0.064 / 0.190 / 0.572] |
| S=16, 20 ep (9668652) | 0.032 / 0.115 / 0.414 [0.029 / 0.104 / 0.458] | **0.088 / 0.260 / 0.572** [0.081 / 0.252 / 0.644] |
| `--anchor_3d` S=8 (9634920) | **0.038 / 0.112 / 0.360** [**0.042 / 0.138 / 0.504**] | 0.104 / **0.257** / 0.504 |

1. **Width pays on both bridges** — unposed AP50 0.067 → 0.098 (+46 %), posed 0.156 → 0.216
   (+38 %). A second, independent mechanism confirming §9.10 reading 4: what moves this ruler is
   multi-view identity, and both flags that buy identity buy 3D AP.
2. **`--anchor_3d` still wins the unposed column** (0.112 vs 0.098 AP50; 0.138 vs 0.080
   class-agnostic) at **half the wall clock and a 4090 instead of an A100**. The two have never
   been combined — that run is the open question (todo 2f).
3. **The 20-epoch model posts the best posed row in the project** (0.088 / 0.260 / 0.572) and the
   best unposed AP25 outside `--anchor_3d`, but its class-agnostic column *loses* to its own
   class-aware one, like every non-anchored checkpoint (§9.11).
4. The posed re-run of 9668639 reproduced a pre-existing on-disk JSON exactly
   (0.083 / 0.216 / 0.488) — the pipeline's determinism, re-confirmed on a second checkpoint.

### 8.5 The two identity mechanisms together — `--anchor_3d` + `--num_frames 16` (todo 2f, job 9979913, 2026-08-08)

§8.3 and §8.4 each found the *same* signature — bundle AP50 flat or mildly up, `bundle_id_switch`
sharply down, `bundle_num_matched` unmoved — and each paid on the 3D ruler. They had never been
run together. Job 9979913 is **the 8.4 recipe plus one flag**: `--num_frames 16
--eval_num_frames 8`, b2, jitter 0.2, 12 epochs, seed 0, `--anchor_3d`. Its `config.json` differs
from 9668639's in `anchor_3d` alone, so the comparison is one-flag in both directions (against
9668639 for the anchor, against 9634920 for the width).

| run | per-frame mIoU / AP50 | per-bundle mIoU / AP50 | `view_consistency` ↑ | `id_switch` ↓ | `num_matched` |
|---|---|---|---|---|---|
| control S=8, no anchor (9386666) | 0.623 / 0.650 | 0.529 / 0.525 | 0.717 | 0.498 | 14.1 |
| `--anchor_3d` S=8 (9634920) | 0.611 / 0.641 | 0.524 / 0.527 | 0.723 | 0.409 | 14.0 |
| **S=16**, no anchor (9668639) | **0.627 / 0.662** | **0.549 / 0.552** | **0.726** | 0.385 | 14.0 |
| **S=16 + `--anchor_3d`** (**9979913**) | 0.616 / 0.646 | 0.527 / 0.536 | 0.722 | **0.375** | 14.1 |

Best per-frame AP50 of 9979913 is 0.648 @ epoch 11; every figure above is epoch 12, its
best-bundle epoch. Val pinned to 8-view bundles in both S=16 rows, so all four cells are the same
ruler.

**Reading 1 — on the 2D ruler the two mechanisms do NOT compose.** Stacking them lands at
`id_switch` 0.375 against width-alone's 0.385: a −0.010 improvement, *inside* the control arm's
own seed-to-seed spread on that metric (0.498 vs 0.471, §8.3 / RESULTS.md §6.1). Meanwhile
per-bundle AP50 falls 0.552 → 0.536 and per-frame 0.662 → 0.646, both −0.016, i.e. ~1.8× the
0.009 seed spread. So the combination costs measurable AP and buys no measurable identity **over
the wider bundle alone** — the anchor's −0.089 `id_switch` at S=8 does not survive being added on
top of the width's −0.113. The natural explanation is that both act on one axis and that axis is
near its floor at this scale; the run does not distinguish that from a plain optimisation
interaction, and it was not designed to.

**Reading 2 — the AP cost is the anchor's, unchanged by width.** Against its own one-flag control
9668639 the anchor costs −0.016 per-frame AP50 here and cost −0.009 at S=8 (§8.3) — the same sign
and roughly the same size, so the +15 % training-time price documented in §8.3 buys nothing extra
at S=16 either. `view_consistency` and `num_matched` are flat across all four rows, as everywhere
in this section.

Cost: 11 h 36 on an A100 80 GB, ~230 GB feature cache — the §8.4 sizing, unchanged.

#### Reading 3 — the 3D ruler agrees: stacking costs, it never gains (jobs 10477399 / 10477400, 2026-08-12)

§8.3's lesson is that `bundle_AP50` at S=8 is a *poor proxy* for the 3D ruler — `--anchor_3d` was
flat on it and worth +67 % 3D AP50 — so a 2D-negative row could not settle 2f. Both bridges were
therefore run on `checkpoint_best_bundle.pth`, all knobs default, val-312, **17.42 frames/scene and
0 failures, identical to every row below** (the per-scene frame counts match the reference runs
scene for scene, so this is like-for-like by construction).

| checkpoint | unposed 18-class | unposed **agnostic** | posed 18-class | posed **agnostic** | kept q | voted | annot |
|---|---|---|---|---|---|---|---|
| control S=8 (9386666) | 0.023 / 0.067 / 0.268 | 0.013 / 0.050 / 0.320 | 0.060 / 0.156 / 0.408 | 0.039 / 0.122 / 0.483 | 97.6 | 0.153 | 0.635 |
| **`--anchor_3d` S=8** (9634920) | **0.038 / 0.112 / 0.360** | **0.042 / 0.138 / 0.504** | **0.104 / 0.257 / 0.504** | **0.109 / 0.304 / 0.677** | 89.0 | **0.177** | **0.666** |
| S=16, 12 ep (9668639) | 0.033 / 0.098 / 0.336 | 0.023 / 0.080 / 0.391 | 0.083 / 0.216 / 0.488 | 0.064 / 0.190 / 0.572 | 96.4 | 0.149 | 0.629 |
| S=16, 20 ep (9668652) | 0.032 / 0.115 / 0.414 | 0.029 / 0.104 / 0.458 | 0.088 / 0.260 / 0.572 | 0.081 / 0.252 / 0.644 | 95.2 | 0.151 | 0.629 |
| **2f, S=16 + `--anchor_3d`** (9979913) | 0.032 / 0.109 / 0.353 | 0.041 / 0.139 / 0.504 | 0.082 / 0.236 / 0.501 | 0.098 / 0.297 / 0.679 | **82.2** | 0.155 | 0.643 |

`voted` / `annot` are mean `voted_vertex_frac` / `annotated_assigned_frac` in the **unposed** run.

**The verdict, and it is consistent in all four columns.** Against `--anchor_3d` alone, 2f is a
dead heat where the two are closest and *below* everywhere else: unposed class-agnostic
0.041 / 0.139 / 0.504 vs 0.042 / 0.138 / 0.504 — differences of ±0.001, an order of magnitude
inside the 0.009 seed spread — while posed class-agnostic loses 0.109 → 0.098 AP and
0.304 → 0.297 AP50, and **both** class-aware columns lose clearly (unposed AP 0.038 → 0.032, posed
AP 0.104 → 0.082). Stacking the two identity mechanisms buys nothing on the ruler that is most
sensitive to identity, at 2× the wall clock and an A100 instead of a 4090. **`--anchor_3d` alone
remains the checkpoint to quote (RESULTS.md §8.2); 2f changes no headline.**

**The mechanism, from the diagnostics.** 2f pushes the "fewer, cleaner instances" signature
further than any other row — **82.2 kept queries**, the fewest of the five — but its coverage does
not follow: 0.155 voted vertices against `--anchor_3d`'s 0.177, and 0.643 annotated-assigned
against 0.666. It prunes harder *and* covers less. That is the cost showing up in the one place
§9.6 said would bind: the vote. Both flags reduce the query set; run together they over-prune.

**Method note worth keeping.** In §8.3 the 2D reading *mispredicted* the 3D outcome (flat
`bundle_AP50`, +67 % 3D AP50). Here the 2D reading **held** — 2D-negative, 3D-negative. So
"`bundle_AP50` is a poor proxy" stays true as a warning against deciding from it, but it is not a
systematic inversion to be relied on in the other direction either. The only way to know is to run
the 3D ruler, which is why this run was worth its 1.5 h even though it closed a door.

Not re-swept: §9.8.1's lifting knobs are checkpoint-dependent and would likely lift 2f as they
lifted `--anchor_3d` (+0.047 class-agnostic AP50), but a tuned row cannot rescue an untuned tie —
the comparison above is untuned on both sides, which is what makes it single-variable.

## 9. The 3D ruler — official ScanNet 3D instance benchmark (docs/todo.md 1d, 2026-08-01)

**Why.** Nothing in §6–§8 is comparable to any published number (docs/RESULTS.md §1.2): we score
per-view 2D masks; SegVGGT (50.4 / 71.7 / 87.0 AP/AP50/AP25) and FAST3DIS score **3D instance
masks on the official benchmark point clouds**. This section is that protocol, end to end. It is
a **third ruler** — never quote its numbers next to the per-frame or per-bundle tables, and never
convert between them.

**And the published 3D numbers are themselves two protocols, not one — read §9.9 before quoting
any of them.** What this section implements is **unposed transfer**: masks reach the point cloud
through the model's *own predicted* depth and cameras, which is also what FAST3DIS
(0.038 / 0.096 / 0.316) and IGGT (0.028 / 0.112 / 0.287) do. SegVGGT's 0.504 / 0.717 / 0.870 is
**posed transfer**: its evaluator moves masks with ScanNet's GT poses, intrinsics and sensor
depth, so no geometry error enters at all. Same evaluator, different bridge, and the bridge is
what separates the two clusters.

### 9.1 Protocol

`scripts/eval_3d_maskdino.py`, per official-val scene (`slurm/eval_3d_maskdino.sh` for the full
312):

1. **One forward pass per scene** over all sampled `scannet_frames_25k` frames (~16–25, sampled
   across the *whole* scan — our stride-5 subset tars cover only raw frames 0–495 and would cap
   recall): the frozen aggregator feeds the MaskDINO head (**one query set for the whole scene**,
   `frames_per_sample=S`) and VGGT's own depth + camera heads. **No GT geometry, depth sensor, or
   pose enters inference** — the selling point vs every fusion/splat pipeline; keep it intact.
   Inference S (~17 avg) deliberately exceeds the training S=8: `CrossFrameAttention` has no
   frame positional encoding (§8.2), so the block is defined for any S.
2. **Pixels → queries.** One class score per query = max sigmoid over views (the §8.2 bundle
   convention); wall/floor-classified queries are dropped (not benchmark classes, their GT
   vertices are void — see 9.2); top `--eval_topk` (100) kept. Mask logits are bilinearly
   upsampled to 518² (the §6.5 rule) and each pixel joins its argmax query above
   `--mask_prob_threshold` (0.5) — a partition, which is what the majority vote expects.
3. **Unproject + register.** Pixels are unprojected with the *predicted* depth + intrinsics
   (`vggt/utils/geometry.py`), optionally confidence-filtered (`--depth_conf_percentile`). VGGT's
   output lives in an arbitrary-scale bundle frame, so scoring needs an **eval-only Sim(3)**:
   closed-form Umeyama from predicted-vs-GT *camera centers*, refined by a similarity ICP against
   the mesh vertices (`--icp`, on by default; scale is re-estimated every iteration). This is the
   FAST3DIS "Sim(3)+ICP" convention — GT poses are used only to place the finished prediction in
   the mesh's coordinate frame, never at inference.
4. **Lift.** Every kept pixel-point votes for its query on the nearest mesh vertex within
   `--vote_radius` (5 cm); each superpoint (`.segs.json`) goes entirely to its
   plurality query (unvoted → unassigned). The **superpoint majority vote is the SegVGGT
   convention**; the radius is ours and exists only because our points land near, not on, the
   mesh — under posed transfer (§9.9) there is nothing to bridge. One query = one 3D instance
   across the whole scene —
   no post-hoc matching anywhere, which is exactly what §8.2 buys.
5. **Score** with the vendored official evaluator (9.2): AP (0.50:0.05:0.95) / AP50 / AP25.

Output: `eval3d_<ckpt stem>.json` next to the checkpoint (headline + per-class + per-scene
diagnostics: Sim(3) scale, camera-center RMS, ICP inliers, vote coverage). `--dump_ply` writes an
instance-coloured point cloud per scene for eyeballing.

### 9.2 The evaluator is the official one, vendored

`train/benchmark3d.py` is a line-for-line Python-3 port of
`ScanNet/BenchmarkScripts/3d_evaluation/evaluate_semantic_instance.py` (fetched 2026-08-01;
upstream is Python 2), operating on in-memory arrays instead of the txt-file tree. Everything
score-relevant is untouched: the 10 overlap thresholds, the 100-vertex minimum region, greedy
confidence-ordered matching, the duplicate-detection FP rule, **void handling** (predictions on
vertices whose GT class is outside the 18 benchmark classes are ignored, not false positives —
wall/floor GT is void, which is why dropping wall/floor *predictions* is on us), and the exact
PR integration. Verified three ways (`tests/test_maskdino_eval3d.py` + the one-off check):
hand-computed planted APs (an IoU-0.5 pred passes AP25 and fails AP50 on the strict `>`; a
genuine FP + hard FN gives exactly 0.5; duplicates after full recall cost nothing), a synthetic
end-to-end run scoring 1.0, and **real val scenes' GT fed back as predictions scoring exactly
1.000 / 1.000 / 1.000**.

Per-vertex GT is built as the official export does: `1000 * nyu40 + objectId + 1` from
`.aggregation.json` + the segs file + `scannetv2-labels.combined.tsv`
(`train/scannet3d.py::build_gt_ids`).

**Classes.** The benchmark scores 18 classes: our 19 minus wall/floor, **plus `otherfurniture`
(nyu40 39), which our head cannot predict** (it is background in our 2D GT, §4). The official
18-class average is the headline (comparable to SegVGGT); a 17-common-class average is reported
alongside as a diagnostic. On the two-scene smoke both otherfurniture instances exist in the GT,
so the 18-class headline structurally pays ~1/18 of its mass wherever otherfurniture occurs.

### 9.3 Data (one-time, on work; built 2026-08-01, jobs 9326394/9326395)

- `scannet_3d_gt_val312.tar.zst` (1.2 GB): per val scene `_vh_clean_2.ply` + superpoint segs +
  aggregation, downloaded per scene from the same kaldir v2/scans path the 2D GT came from,
  validated (ply magic, segment-id closure) — `legacy/dataset_build/{scripts/download_3d_gt.py,
  slurm/download_3d_gt_val312.sh}`.
- `scannet_frames25k_val312.tar.zst` (1.1 GB): the val-312 slice of the official
  `scannet_frames_25k.zip` (v2/tasks, 6.0 GB, one resumable download) — color + **camera-to-world
  pose** + intrinsics per frame, 5 436 frames, non-finite poses excluded at load time —
  `legacy/dataset_build/{scripts/repack_frames25k.py, slurm/download_frames25k_val312.sh}`.

### 9.4 Honesty: which checkpoint may quote which number

The official val-312 split overlaps our conventional training range (scenes 0000–0489), so **any
existing checkpoint's 3D numbers are DIAGNOSTIC only** — they verify the pipeline, they are not
reportable. The reportable number needs a checkpoint trained on the official 1201-scene split
(tar built, docs/todo.md 1c) with val-312 never seen. A further caveat for the current diagnostic
checkpoint (`maskdino_sf_n490_mf_b2jit_20260730_105117/checkpoint_best.pth`): it is the epoch-17
mIoU-selected checkpoint — the epoch-19 AP50-selected one that carried the 0.515 bundle headline
did not survive the 2026-07-30 output cleanup (bundle AP50 0.461 at epoch 17).

### 9.5 Diagnostic runs on the leaked checkpoint (2026-08-01, jobs 9327269 / 9327271)

Numbers: `docs/RESULTS.md` §5. Full narrative and per-scene readings:
`docs/old/MASKDINO_RESULTS_HISTORY.md`. Kept here because two of its findings are structural and
the rest of §9 argues from them:

1. **Geometry binds, not recognition.** AP25 ≈ 4–5× AP50: objects are found and coarsely
   localised, but the lifted masks miss the >0.5-IoU bar. Median camera-centre RMS after Sim(3) is
   **0.14 m** and ICP point RMS **~0.10 m** — the same order as the vote radius, and VGGT's own
   depth/pose drift over a whole-scan S≈17 bundle, not a 2D mask-quality problem (the same model
   scores 0.65–0.67 per-frame AP50). This is the price of "no GT geometry at inference".
2. **Coverage is the second cap:** ~15 % of mesh vertices receive any vote, ~63 % of annotated
   vertices get assigned. Every unassigned GT instance is a hard FN.
3. **S-generalisation works as designed:** bundles of 3–55 frames (median 15) through a model
   trained at S=8, no failures — `CrossFrameAttention` has no frame positional encoding, and the
   two-stage top-k just unions over more frames.

### 9.6 The REPORTABLE number (2026-08-03, jobs 9503137 / 9503139)

Checkpoint `maskdino_sf_list1201_mf_20260802_133826/checkpoint_best_bundle.pth` — official
1201-scene train split, **val-312 never seen** (§9.4 satisfied), bundle-selected (todo 2b).
312/312 scenes, 0 failures, ~46 min/run. Numbers: `docs/RESULTS.md` §5.

**Quote the defaults row as the headline.** The tuned row's knobs were chosen on §9.5's leaked
runs, so it is mildly tuned on data that includes val scenes; both rows are otherwise honest.

Three readings that the rest of §9 builds on:

1. **We land in the unposed published cluster on a frozen backbone** — FAST3DIS and IGGT, against
   their LoRA-adapted backbones. SegVGGT is far above but in the **other** protocol (§9.9); state
   that plainly, and state just as plainly that it is a legitimate evaluation choice on their
   part, not a trick, since their model is as unposed as ours.
2. **The leak-free checkpoint BEATS the leaked one** (~1.6× AP50 at identical knobs). §9.5
   predicted "roughly nearby"; the truth is that 1201 official train scenes outweigh *having seen
   the val scenes*. The 3D ruler independently reproduces the 2D conclusion: **data-limited, not
   architecture-limited** — and every §9.5 number was a pessimistic proxy.
3. **The lifting step is now the binding constraint, not the decoder.** Geometry diagnostics are
   unchanged from §9.5 and AP25 is still ~4× AP50. Two *lifting* knobs alone bought +0.016 AP50 —
   more than most single-frame decoder ablations are worth. The next lever on this ruler is
   registration quality / coverage / voting, not query design.

**Filename discipline.** Output files name any non-default result-affecting knob
(`eval3d_<stem>__vote_radius0.1_depth_conf_percentile25.0.json`). Before 2026-08-03 both knob
settings wrote to the same path and the second silently overwrote the first, which is how job
9503137's JSON was lost (the defaults run was repeated as job 9532181 and reproduced it exactly —
0.0228 / 0.0672 / 0.2680 — so the pipeline is deterministic and the headline has a JSON behind
it). Guarded by `tests/test_maskdino_eval3d.py::test_out_path_names_the_knobs`.

### 9.7 Looking at the predictions in 3D (qualitative, 2026-08-03)

Two different pictures, and confusing them is easy: **one shows what the benchmark scores, the
other shows what the model predicts.**

**(a) The scored product — `--dump_ply`.** `scripts/eval_3d_maskdino.py --dump_ply` writes
`eval3d_<scene>.ply` next to its JSON: the *benchmark mesh's own vertices*, coloured by the
instance each one was assigned after the full §9.1 pipeline (unproject with predicted depth +
cameras → Sim(3)+ICP into the mesh frame → votes within `--vote_radius` → superpoint majority).
Grey = no instance reached that vertex. Open it in MeshLab/CloudCompare. This is the object the
AP numbers are computed from, so it is the honest figure for a paper — and the grey is the
result too: `voted_vertex_frac` is 0.05–0.21 per scene (§9.6 reading 3, the lifting bottleneck,
made visible).

```bash
sbatch --export=ALL,CHECKPOINT=<mf_run>/checkpoint_best_bundle.pth,\
EXTRA_ARGS='--dump_ply --scenes scene0011_00 scene0015_00 --vote_radius 0.1 --depth_conf_percentile 25' \
    slurm/eval_3d_maskdino.sh
```

Note `--scenes` is result-affecting, so the JSON gets its own name and a scene subset can never
overwrite a full-val result. A handful of scenes is a *picture*, never a number: 4 easy val
scenes scored 0.084 / 0.236 / 0.375, ~3× the 312-scene averages of §9.6.

**(b) What the model predicts — the Gradio viewer.** `demos/demo_gradio.py` now accepts MaskDINO
checkpoints alongside the retired baseline-head ones (it dispatches on the checkpoint's keys) and colours
**VGGT's own predicted point cloud** by the head's per-view instance assignment. No mesh, no
registration, no superpoint vote, no GT of any kind — it is the raw 2D→3D product, seen
interactively.

```bash
python demos/demo_gradio.py \
    --seg_checkpoint <mf_run>/checkpoint_best_bundle.pth \
    --seg_scans_root <a scans tree with the scenes you want>      # optional; uploads work too
# then: 'Load Checkpoint Scene' (or upload images) → 'Reconstruct' → Color By: Predicted Instances
```

`train/maskdino_viz3d.py` keeps it honest by *inheriting* both conventions rather than inventing
its own: query selection is the 3D ruler's (one class score per query = max over views, then
score threshold + top-k, §9.1 step 2), and colour is keyed to the query index with the same
tab20 slots as the run's 2D panels (§6.4) — so query 7 wears one colour in the PNG figures, in
the viewer, and across every view of the bundle. Tokens are rebuilt from the run's own
`--feature_mode` / `--feature_layers` / `--backbone_dtype`, and a `--multi_frame` checkpoint sees
the frames as one bundle. Feeding it a **single-frame** checkpoint is allowed but the legend says
so: with a per-frame query set a colour means nothing across views, which is precisely the
difference §8 exists to close.

What each picture is good for: (a) answers "how much of the room did we actually label, and
correctly?" — coverage and registration failures are obvious, per-view mask quality is not.
(b) answers "are the masks and the cross-view identities any good?" — mask boundaries and colour
stability across views are obvious, while lifting losses are invisible because there is nothing
to lift onto. A prediction that looks right in (b) and empty in (a) is a lifting problem, which
is exactly the diagnosis §9.6 reached numerically.

### 9.8 Lifting-knob sensitivity (todo 5a; jobs 9503137/39, 9508450–55, 9532181–83, 2026-08-03)

The §9.6 tuned row inherited its knobs from the *leaky* diagnostic runs, so they were re-swept
cleanly on the leak-free checkpoint. **Read this as a sensitivity analysis, not as a better
headline**: the sweep runs on val-312, so quoting its argmax would be test-set tuning. The
headline stays the defaults row of §9.6.

All 312 scenes, 0 failures, 18-class metrics:

| `--vote_radius` | `--depth_conf_percentile` | AP | AP50 | AP25 |
|---|---|---|---|---|
| **0.05 (default)** | **0 (default)** | 0.023 | **0.067** | 0.268 |
| 0.05 | 25 | 0.023 | 0.071 | 0.260 |
| 0.10 | 0 | 0.026 | 0.078 | 0.321 |
| 0.10 | 25 | 0.029 | 0.083 | 0.305 |
| 0.10 | 50 | 0.024 | 0.068 | 0.269 |
| 0.15 | 25 | **0.030** | 0.090 | 0.321 |
| 0.20 | 25 | 0.029 | 0.090 | 0.325 |
| 0.30 | 25 | 0.029 | **0.091** | 0.326 |

1. **The vote radius saturates at ~0.15 m — the scale of the registration error.** AP50 climbs
   0.071 → 0.083 → 0.090 from 5 to 15 cm and then goes flat (0.090 at 20 cm, 0.091 at 30 cm);
   strict AP peaks at 0.15 and declines slightly, the expected mask-bloating cost. The plateau
   is the informative part: **doubling the radius past 0.15 m changes nothing**, so beyond that
   the votes already reach every vertex they are ever going to reach. The radius has to be as
   wide as the median camera-center RMS (0.14 m, §9.5) to bridge the misalignment, and once it
   is, what remains is *coverage and assignment*, not "the point landed a few cm off".
2. **The depth-confidence filter has an interior optimum at 25 %** (0.078 → 0.083 → 0.068 at
   radius 0.10): filtering half the depth throws away usable geometry. It also trades the two
   IoU regimes against each other — no filtering gives the best AP25 (0.321) while 25 % gives
   the best AP50 at the same radius, i.e. it buys boundary precision with coverage.
3. **Knobs cap out below FAST3DIS** (whose column is class-agnostic, §9.11). The whole grid spans 0.067 → 0.091 AP50: lifting
   hyper-parameters are worth up to +0.024 (+36 % relative) — more than any decoder ablation in
   §7.2.1 — and still short of FAST3DIS's 0.096. So the remaining gap is **not** a tuning
   artefact, which is the useful negative result here: it has to come from coverage (§todo 5b)
   and registration quality (5c), the two things the plateau in reading 1 points at.

#### 9.8.1 The same sweep on the `--anchor_3d` checkpoint — reading 3 INVERTS (2026-08-07)

Jobs 9901146/48/49/50/51/52, val-312, 0 failures. The §9.6 checkpoint's sweep above capped
*below* FAST3DIS; re-run on the `--anchor_3d` checkpoint (§8.3) **every point of the grid is
above it.** Same sensitivity-analysis caveat as §9.8 — swept on val, so the headline stays the
defaults row — but the *comparison* claim no longer depends on which point you pick.

| `--vote_radius` | `--depth_conf_percentile` | 18-class AP/AP50/AP25 | **class-agnostic** (§9.11) |
|---|---|---|---|
| **0.05 (default) — the headline** | **0 (default)** | 0.038 / 0.112 / 0.360 | **0.042 / 0.138 / 0.504** |
| 0.10 | 0 | 0.047 / 0.142 / 0.395 | 0.052 / 0.171 / 0.545 |
| **0.15** | **0** | 0.048 / **0.151** / **0.419** | **0.055 / 0.185 / 0.571** |
| 0.10 | 25 | 0.047 / 0.136 / 0.381 | 0.051 / 0.168 / 0.518 |
| 0.15 | 25 | **0.050** / 0.151 / 0.396 | 0.055 / 0.180 / 0.544 |
| 0.20 | 25 | 0.050 / 0.151 / 0.403 | 0.055 / 0.178 / 0.548 |
| default + `--eval_topk 600` | 0 | 0.038 / 0.111 / 0.357 | 0.043 / 0.140 / 0.502 |
| FAST3DIS / IGGT (published, class-agnostic) | | — | 0.038 / 0.096 / 0.316 · 0.028 / 0.112 / 0.287 |

1. **The whole grid leads the published unposed cluster.** The *worst* point is the default,
   0.138 class-agnostic AP50 = 1.44× FAST3DIS's 0.096; the best is **0.055 / 0.185 / 0.571** =
   1.45× / 1.93× / 1.81× on FAST3DIS and 1.96× / 1.65× / 1.99× on IGGT. This is the exact mirror
   of §9.8 reading 3 and it retires that sentence for this checkpoint: **the lead is not a tuning
   artefact, because there is no point in the grid that does not have it.**
2. **The radius still saturates at 0.15 m** (0.151 AP50 at 0.15, 0.151 at 0.20) — same plateau,
   same explanation, on a different checkpoint. That is now a property of the *lifting*, not of
   one model.
3. **The confidence filter's sign flipped.** On the §9.6 checkpoint 25 % was the interior optimum;
   here it is neutral-to-negative (0.185 → 0.180 class-agnostic AP50 at radius 0.15, and it costs
   AP25 0.571 → 0.544). Like the class-collapse sign (§9.11), this knob is **checkpoint-dependent**
   — do not carry a tuned value across checkpoints, re-sweep it.
4. **`--eval_topk` is not a lever — negative result.** Going 100 → 600 kept query-class pairs (the
   count SegVGGT and FAST3DIS use, listed in §9.9 as one of the secondary differences favouring
   them) moves nothing: 0.138 → 0.140 class-agnostic AP50. That difference is now measured and can
   be struck from the list of explanations for the SegVGGT gap.

**Looking at a `.ply` without MeshLab.** `scripts/view_ply.py <file>.ply` writes one
self-contained HTML next to it — points, colours and a small WebGL viewer embedded, no CDN, no
install. `scp` it and double-click, or `python -m http.server` on the node and open the
forwarded port. Two files get one shared camera (`scripts/view_ply.py a.ply b.ply`). It also
prints the grey fraction, which is the coverage number of §9.6 reading 3 in one line:
`eval3d_scene0011_00.ply: 237,360 vertices, 95,075 (40%) grey/unassigned`.

**Side by side, one camera (added 2026-08-03).** The viewer's second tab, *GT vs Prediction
(synced)*, shows the SAME reconstructed cloud twice: left coloured by GT instance id, right by
query id. Every control (confidence threshold, frame filter, background masks, prediction
branch) drives both panels, and there is literally **one camera object** in the page — each
panel is a viewport onto it, so orbit/pan/zoom on either side moves both by construction rather
than by keeping two cameras in sync. This is why `demos/dualview3d.py` renders WebGL directly
instead of using two `gr.Model3D` components: Gradio 5 draws Model3D through Babylon.js inside a
compiled Svelte component whose camera is unreachable from outside.

Three things that make the comparison honest, all asserted in `tests/test_dualview3d.py`:

1. **The panels show the same points.** The filtering is a re-implementation of
   `visual_util.predictions_to_glb`'s and is tested vertex-for-vertex against it, the
   black/white-background masks are computed from the *image* (never from a panel's own
   colouring), and the subsample to `--max_points` is a deterministic stride. If the point sets
   could differ, "only the colour differs" would be false and the picture would mislead.
2. **The GT panel is the dataset's annotation, not a re-labelling.** `masks` [S, H, W] carries
   global instance ids; they are painted with the same palette and rule as the 2D GT panel. The
   frames are written as PNGs and re-read in *sorted* order, so the id maps are permuted to
   match — otherwise one frame's GT would land on another frame's points, which looks plausible
   and is wrong (`test_gt_maps_follow_the_gallery_order`).
3. **The identity spaces still differ** (§6.4): GT ids vs query indices. Colours agree with
   themselves across views, never between the two panels.

Uploaded images have no annotation, so the left panel falls back to RGB; with no checkpoint
loaded there is a single RGB panel. **Not executable here:** this environment has no browser, so
the WebGL/camera/pointer code is checked structurally (one camera object, one canvas per panel,
decodable payload, escaped srcdoc) but has never been *run*. Everything upstream of the browser
— filtering, colours, payload, ordering — is tested numerically.

### 9.9 Two 3D protocols — what our number is and is not comparable to (2026-08-04)

The published ScanNet-3D numbers are printed in the literature as one table. They are **two
protocols**, separated by how a finished 2D mask reaches the benchmark point cloud — and that
step, not the evaluator, is what separates the two clusters of results.

| | **posed transfer** | **unposed / predicted-geometry transfer** |
|---|---|---|
| who | **SegVGGT** 0.504 / 0.717 / 0.870 | **FAST3DIS** 0.038 / 0.096 / 0.316, **IGGT** 0.028 / 0.112 / 0.287, **this section** 0.023 / 0.067 / 0.268 |
| mask → point cloud | the GT cloud is **projected into each view** with ScanNet GT poses + intrinsics; occlusion from the ScanNet **sensor depth** map | pixels are **unprojected** with the model's own predicted depth + cameras, then Sim(3)+ICP into the mesh frame for scoring (§9.1 step 3) |
| geometry error in the bridge | **zero** — correspondence exact by construction | the full feed-forward error (ours: median camera-centre RMS 0.14 m, §9.5) |
| what the score measures | 2D mask quality | 2D mask quality **×** feed-forward geometry quality |
| evaluator | official ScanNet, same options as §9.2 | official ScanNet, same options as §9.2 |

**Evidence** (their released code, cloned at `/cluster/scratch/niacobone/SegVGGT`, read
2026-08-04 — every claim below was checked against the file, not the paper):

- `eval/eval_instance_seg.py:243-336` (`map_pred_inst_to_gt_pointcloud`) **does not unproject**.
  It projects the GT benchmark point cloud into each view and reads the predicted 2D mask at the
  landing pixel.
- Extrinsics come from ScanNet GT `pose/{frame}.txt` (`eval_instance_seg.py:198`); intrinsics
  from GT `intrinsic_depth.txt` (`eval/instance_eval_common.py:68`); occlusion is decided against
  the ScanNet **sensor** depth `depth/{frame}.png` within 0.1 m (`eval_instance_seg.py:178-182`,
  `305-307`, `451`).
- Therefore **no Sim(3), no ICP, no scale estimation, no vote radius** — none of §9.1 step 3
  exists on their side, and none of the failure modes §9.6 reading 3 diagnoses can occur.
- **VGGT's geometry heads are never called** in their instance-eval path:
  `eval/instance_eval_common.py:168-189` runs the aggregator and the semantic head only.
- The metric code is mmdet3d's copy of the official ScanNet evaluator with the **same options as
  our vendored one** — overlaps `[0.5:0.05:0.9] + [0.25]`, `min_region_sizes 100`, 18 classes,
  superpoint majority (`eval/instance_seg_eval.py:523-540` vs `train/benchmark3d.py:36-37`).
  **The evaluator is not the difference.**

Secondary differences, all in their favour but none of them the main effect: **~75–100 views per
scene** (every 20th frame of a full `.sens` extraction — `eval_instance_seg.py:169` plus their
`docs/data_preparation.md`) against our ~17 from the official 25k export (§9.3); masks at
**259×196**, half of their 518×392 input (`return_feature_maps_down_ratio: 2` in
`configs/eval/segvggt_scannetv2.yaml`) against our 37×37 grid; **600** kept query-class pairs
(`instance_eval_common.py:107`) against our `--eval_topk 100`; and they train and are scored on
`otherfurniture`, which our 19-class head cannot predict (§9.2).

**Say this fairly.** SegVGGT is not cheating and must never be described as if it were. Their
*model* consumes unposed RGB only, exactly like ours — the GT geometry appears nowhere in
inference, only in the transfer of finished masks onto the benchmark cloud for scoring. That is a
legitimate design: it deliberately isolates segmentation quality from reconstruction quality,
which is a question worth measuring on its own. The problem is only that both protocols appear in
the literature inside one table without the distinction, and SegVGGT and FAST3DIS are
contemporaneous preprints (2603.19926, 2603.25993) so neither could have cited the other.

**What follows for us.** The right comparison for §9.6 is the unposed cluster, where we sit with
FAST3DIS and IGGT on a strictly frozen backbone — with the label-setting caveat of §9.11. Separately, running the posed protocol on our
*own* masks isolates our 2D mask quality from our lifting error and bounds what fixing §9.6
reading 3 could ever buy — the same decomposition SegVGGT's number already enjoys. That is
docs/todo.md 5e; the mechanism exists as `--transfer_mode gt_projection`, and its number is a
**diagnostic decomposition only** — the headline stays the unposed one.

**MEASURED 2026-08-04 — §9.10, and it corrects this section.** Under SegVGGT's own bridge our
masks score 0.060 / 0.156 / 0.408, against their 0.717 AP50. So the protocol is worth a factor
of **2.3** of the AP50 gap and a factor of **~4.6 is real and remains**. An earlier draft of
this section called the difference "a different protocol, not a different league"; that is
**wrong** and has been struck — it is a different protocol *and* a real gap, and the honest
framing is that the protocol difference makes the raw side-by-side meaningless, not that it
makes the two models equivalent.

### 9.10 The posed-transfer column, measured (todo 5e; jobs 9607206 / 9607208 / 9607210, 2026-08-04)

§9.9 established that the two published clusters differ by their 2D→3D bridge. This section
*measures* the difference on our own masks: the same checkpoint, the same frames, the same
queries, the same evaluator — only the bridge swapped. It is a **decomposition, not a new
headline**: the reportable number stays §9.6's 0.023 / 0.067 / 0.268.

**`--transfer_mode {unproject,gt_projection}`** (`scripts/eval_3d_maskdino.py`, geometry in
`train/eval3d_geometry.py`). `unproject` is the default and is byte-for-byte today's pipeline.
`gt_projection` replaces §9.1 steps 3–4's first half with SegVGGT's
`map_pred_inst_to_gt_pointcloud`: per frame, the mesh vertices are transformed by `inv(pose)`,
projected with the GT **depth** intrinsic into the native 640×480 grid, kept when
`|z_projected − z_sensor| < --depth_tolerance` (0.1 m, SegVGGT's value) against the ScanNet
sensor depth, and each survivor votes for whichever query owns the pixel it lands on. The
superpoint majority, the query selection, the class scores and the vendored evaluator are
untouched, so **the transfer is the only variable**. VGGT's depth and camera heads are not even
run in this mode, which makes "no predicted geometry in this column" structural rather than a
convention.

```bash
sbatch --export=ALL,CHECKPOINT=<mf_run>/checkpoint_best_bundle.pth,\
EXTRA_ARGS='--transfer_mode gt_projection' slurm/eval_3d_maskdino.sh
```

**The pixel mapping, derived not assumed.** This is the one place a silent error would produce a
plausible number, so it is written out in `train/eval3d_geometry.py::mask_grid_intrinsic`.
ScanNet's color (1296×968) and depth (640×480) intrinsics are the same camera at two
resolutions — normalised `fx/W, fy/H, cx/W, cy/H` agree to <1e-3 on the val-312 tar — so one
pose serves both. The depth test runs in the **native** depth grid (the sensor map is never
resampled). Our mask, however, lives on a 518² grid that `load_frames_by_name` produced by
**squashing** the whole color image (`Image.resize((518,518))`: no crop, no letterbox, aspect
ratio not preserved), so `K_mask = diag(518/1296, 518/968, 1) @ K_color`. The two factors differ
(0.400 vs 0.535); an *isotropic* rescale — the natural assumption if you expect aspect-preserving
preprocessing, as SegVGGT's own 518×392 pipeline has — misplaces the principal point by ~40 rows
and every mask read with it. Cross-checked against the reference implementation: SegVGGT's
per-axis `u * mask_w / depth_w` from the depth grid lands within 0.5 px of this derivation.

**The oracle that licenses the number** (`scripts/eval3d_projection_oracle.py`,
`slurm/eval3d_projection_oracle.sh`, job 9607210, CPU-only, 312/312 scenes). Mirroring §9.2's
discipline: the 3D GT is *rendered into every view through the transfer's own projection*
(nearest vertex wins the z-buffer; unannotated vertices occlude and then paint "no instance") and
fed straight back as predictions.

| | value | reading |
|---|---|---|
| round-trip **purity** | **0.9999** (worst scene 0.9977) | of the annotated vertices the transfer assigned, the fraction returned to their **own** instance — this is the mapping test, and it passes |
| AP / AP50 / AP25 | 0.828 / **0.948** / 0.974 | the protocol's **ceiling on our ~17-frame budget** |
| sensor-depth inlier | 0.618 | of projections with a depth reading — the rest are surfaces the frame does not actually see |
| visible vertices | 0.666 | fraction of mesh vertices any frame sees at all |

The ceiling is 0.948 rather than 1.000 for one reason and it is not the mapping: ~9 % of
annotated vertices are seen by no frame, so they are missing from every recovered mask and the
strictest IoU bars fail. Purity 0.9999 is what proves the pixels are read in the right place — a
shifted, transposed or isotropically-rescaled mapping collapses it, which
`tests/test_maskdino_eval3d.py` also asserts synthetically.

**Results.** Checkpoint `maskdino_sf_list1201_mf_20260802_133826/checkpoint_best_bundle.pth`
(leak-free, §9.4 satisfied), all 312 val scenes, 0 failures, all knobs at their defaults. Both
rows saw **the same 17.4 frames/scene and the same 97.6 kept queries/scene** — every frame of
the 25k export has its depth png, so not even the frame set moved.

| transfer | AP / AP50 / AP25 (18-class) | 17-class diagnostic | `voted_vertex_frac` | `annotated_assigned_frac` |
|---|---|---|---|---|
| `unproject` (**the headline**, §9.6) | **0.023 / 0.067 / 0.268** | 0.024 / 0.071 / 0.284 | 0.153 | 0.635 |
| `gt_projection` (SegVGGT's protocol) | 0.060 / 0.156 / 0.408 | 0.064 / 0.166 / 0.432 | **0.342** | **0.791** |
| — oracle ceiling of the second row | 0.828 / 0.948 / 0.974 | — | — | 0.906 |

**What each row measures.** `unproject` = 2D mask quality **×** feed-forward geometry quality;
it is the number FAST3DIS (0.038 / 0.096 / 0.316) and IGGT (0.028 / 0.112 / 0.287) are
comparable to, and it is what our claim "no GT geometry at inference" costs. `gt_projection` =
2D mask quality **alone**, with a perfect 2D↔3D bridge; it is the number SegVGGT
(0.504 / 0.717 / 0.870) is comparable to. Neither is "the real" number — they answer different
questions and must always be printed as two columns.

Four readings:

1. **The bridge costs us a factor of 2.3 in AP50** (0.067 → 0.156), 2.6 in strict AP, 1.5 in
   AP25. That is the price of the frozen-backbone, no-GT-geometry claim, quantified for the
   first time, and it is a hard **upper bound on everything todo 5b/5c can buy**: perfect
   registration and perfect coverage would land at 0.156, not at SegVGGT's 0.717.
2. **Coverage is where the bridge is lost, and it is the single most informative number here.**
   `voted_vertex_frac` 0.153 → 0.342 (2.24×) and `annotated_assigned_frac` 0.635 → 0.791 track
   the AP50 ratio almost exactly. §9.6 reading 3 and §9.8 reading 1 diagnosed "the lifting step
   binds" from the inside; this measures it from the outside and agrees. Note the honest ordering
   this implies for todo 5: the vote radius already saturates (§9.8), so what is left in the
   unposed column is *registration*, and it is worth at most +0.089 AP50.
3. **The protocol explains part of the SegVGGT gap, not the gap.** Under *their* bridge we score
   0.156 AP50 against their 0.717 — the protocol accounts for a factor of 2.3 out of the ~10.7×
   raw ratio; a factor of ~4.6 is real and remains. Do not let §9.9's framing harden into "the
   gap is protocol": it is *partly* protocol, and the rest is a LoRA-adapted backbone, ~75–100
   views to our 17, 259×196 masks to our 37×37 grid, and 600 kept queries to our 100 (§9.9,
   secondary differences). §9.9's line "a different protocol, not a different league" is
   **corrected by this measurement** and has been amended in place.
4. **View count is *not* the binding constraint of the posed column** — the oracle's 0.948 AP50
   ceiling on the same 17 frames says the frame budget could at best take 0.156 → 0.164 in
   relative terms. What binds at 0.156 with a perfect bridge is the **3D-instance criterion
   itself**: one mask must cover a whole object across *all* its views at IoU > 0.5, where the
   per-frame ruler scores each view independently and reports 0.65 AP50 for this checkpoint
   (§7.8). The residual is multi-view completeness and identity — exactly what §8.2 exists to
   improve, so on this ruler the decoder *is* back in play.

**Regression control** (job 9607208): the refactor that split the two transfer paths was checked
by re-running the unposed 4-scene subset of §9.7 with identical knobs — it reproduced
0.084 / 0.236 / 0.375 exactly. No existing number in §9.5–§9.8 moved.

**Filename discipline.** `--transfer_mode` and `--depth_tolerance` are result-affecting, so they
tag the output (`eval3d_<stem>__transfer_modegt_projection.json`) and can never overwrite the
unposed headline's file; `tests/test_maskdino_eval3d.py::test_out_path_names_the_knobs` asserts
it. `--vote_radius`, `--depth_conf_percentile` and `--icp` are **inert** in `gt_projection` mode
and the script says so at startup if they are set.

### 9.11 Class-aware vs class-agnostic — the second axis of the comparison (2026-08-06)

§9.9 split the published numbers by their 2D→3D **bridge**. Re-reading the two papers for the
supervisor meeting turned up a *second* axis, which had been silently ignored:

| | SegVGGT | FAST3DIS / IGGT | us |
|---|---|---|---|
| bridge | posed (GT depth + poses) | unposed (predicted geometry) | unposed |
| **labels** | **class-aware**, 18 classes, per-class mean | **class-agnostic** — labels ignored | **class-aware**, 18 classes |
| AP definition | IoU 0.50:0.05:0.95; AP50 / AP25 fixed | identical | identical |
| views / scene | every 20th frame (~75–100) | 50 | ~17 |

**`mAP` (SegVGGT's column header) and `AP` (FAST3DIS's) are the same quantity** — in the ScanNet
3D instance literature AP is already a mean over classes. The header difference means nothing;
the *setting* difference does. FAST3DIS §4.4: *"In the class-agnostic setting, we ignore the
semantic class labels in the annotations and focus purely on object localization and boundary
quality"*, and it publishes no class-aware ScanNet number. IGGT's row in that table inherits the
setting — and is **FAST3DIS's re-evaluation**: IGGT's own paper (arXiv 2510.22706) reports no
ScanNet AP at all, only tracking (T-mIoU 69.41), reconstruction (Abs.Rel 1.90) and
open-vocabulary semantics (3D mIoU 39.68) over 10 scenes × 8–10 images.

**Measured, and it reverses the obvious guess** (jobs 9861563 / 9861564, val-312, 0 failures; the
18-class columns reproduced §9.6 exactly, so the collapse perturbs nothing). Class-agnostic looks
like the *permissive* setting — a wrong label costs nothing — and for our system it is not:

| unposed, val-312 | class-aware (18) | class-agnostic |
|---|---|---|
| defaults (§9.6) | 0.023 / 0.067 / 0.268 | **0.013 / 0.050 / 0.320** |
| `--vote_radius 0.1 --depth_conf_percentile 25` | 0.029 / 0.083 / 0.305 | **0.017 / 0.060 / 0.334** |
| **`--anchor_3d`, defaults (§8.3, job 9866391)** | 0.038 / 0.112 / 0.360 | **0.042 / 0.138 / 0.504** |
| 〃 **`--seed 1` replicate (job 9979100)** | 0.037 / 0.112 / 0.342 | **0.039 / 0.129 / 0.485** |
| control, `--seed 1` replicate (job 9979101) | 0.025 / 0.075 / 0.313 | 0.016 / 0.059 / 0.348 |
| 〃 best sweep point (`--vote_radius 0.15`, §9.8.1) | 0.048 / 0.151 / 0.419 | **0.055 / 0.185 / 0.571** |
| `--num_frames 16` (§8.4, job 9901143) | 0.033 / 0.098 / 0.336 | 0.023 / 0.080 / 0.391 |
| `--num_frames 16`, 20 ep (job 9901664) | 0.032 / 0.115 / 0.414 | 0.029 / 0.104 / 0.458 |
| FAST3DIS (published) | — | 0.038 / 0.096 / 0.316 |
| IGGT (via FAST3DIS) | — | 0.028 / 0.112 / 0.287 |

**The collapse's sign is `--anchor_3d`'s alone (measured 2026-08-07).** Of the four checkpoints
now scored both ways, only the 3D-anchored one gains: 0.112 → 0.138. The §9.6 control loses
(0.067 → 0.050), and so do both bundle-width checkpoints (0.098 → 0.080, 0.115 → 0.104). So this
is not "wider bundles survive pooling" and not a property of the multi-frame recipe — it is
specific to the mechanism that produces **fewer, cleaner, view-consistent instances** (9 % fewer
kept queries at 16 % more voted vertices, §8.3), which is exactly what one instance-pooled
ranking rewards and what duplicate/fragmented detections lose.

On the **headline** checkpoint: AP25 goes up and crosses both published rows (0.334 vs 0.316 /
0.287); AP50 and AP go down and land ~1.6–2.2× behind. That replaces §9.6's "in FAST3DIS's
ballpark", which was comparing across settings.

**On the `--anchor_3d` checkpoint the collapse goes the other way, and the result is the strongest
row in this project (job 9866391, 2026-08-06, 312 scenes, 0 failures, all lifting knobs at
defaults).** 0.038 / 0.112 / 0.360 class-aware → **0.042 / 0.138 / 0.504 class-agnostic** — i.e.
like-for-like it **leads FAST3DIS (0.038 / 0.096 / 0.316) and IGGT (0.028 / 0.112 / 0.287) on
AP50 and AP25, matches FAST3DIS on AP and leads IGGT on it**, on a strictly frozen backbone
against their LoRA-adapted ones, with ~17 views/scene against FAST3DIS's 50, and untuned.
⚠ The AP column is a *tie*, not a lead — §8.3's seed-1 replicate put our AP at 0.039 against
0.038, inside our own 0.003 seed spread; "ahead on all three" was a seed-0-only reading.
This falsified the prediction recorded in todo 1e
("expect the claim to weaken"): the sign of the collapse is **checkpoint-dependent**, not a
property of the setting. Why it flips — the collapse costs a head whose class-aware mean is
carried by one rare class and gains a head whose *instances* are cleaner: `--anchor_3d` keeps 9 %
fewer, more view-consistent queries (§8.3), so the pooled ranking it produces is less polluted by
duplicate/fragmented detections, which is exactly what instance-pooled AP punishes.

Carry two caveats with the claim: `otherfurniture` and the frame-coverage handicaps still apply
(they cost the class-aware column, not this one); and the comparison is still unposed-protocol
only (§9.10). The third — "single run against a single control" — was **retired 2026-08-07**:
both arms are now scored at two seeds on this ruler (§8.3), effect ~9× the seed spread.

*Why the collapse costs us.* It swaps a mean over 18 classes for one instance-pooled ranking. Our
per-class table (tuned row) is carried by rare distinctive classes — toilet **0.508** AP50 at
1/18 of the mean, sink and refrigerator 0.173 — while the numerous ones are weak (chair 0.053,
cabinet 0.040, bookshelf 0.001) and `otherfurniture` is **0.000**, unpredictable for a 19-class
head. Pooling deletes the rare-class leverage and pours every unmatched `otherfurniture` instance
into a single recall curve. The mechanism is structural, not a bug: masks here are disjoint by
construction (`assign == q`, one class per query via argmax), so no cross-class duplicate can be
inflating either column.

*What it means.* At loose IoU our instance discovery is already competitive with the published
unposed cluster; at strict IoU boundary quality and lifting are not. Same conclusion as §9.8 —
lifting binds — now visible in the competitors' own setting.

**The evaluator produces both.** `train/benchmark3d.py::collapse_gt_to_class_agnostic` /
`collapse_preds_to_class_agnostic` relabel every benchmark class onto one id, which makes the
vendored official logic compute exactly FAST3DIS's number: the 17 unused classes get neither GT
nor predictions, score NaN, and `compute_averages`' nanmean drops them. Only the **label** is
collapsed — the prediction set is unchanged, and predictions carrying a non-benchmark label (our
head's wall/floor) are dropped in both settings — so class-aware vs class-agnostic is a
single-variable comparison. Every `scripts/eval_3d_maskdino.py` run now prints it and writes
`results_class_agnostic` into the JSON, beside (never instead of) the 18-class headline.
`tests/test_maskdino_eval3d.py::test_evaluator_class_agnostic` pins the semantics: masks right
and labels rotated scores AP50 0.0 class-aware and 1.0 class-agnostic; with correct labels the
two agree.

**One more row worth carrying to any "why is your AP so low" question.** SegVGGT's Table 1 also
lists the point-cloud / RGB-D family — Mask3D 55.2 / 73.7 / 85.3, Relation3D 62.5 / 80.2 / 87.0,
SegDINO3D 64.0 / 81.5 / 88.9, ODIN 50.0 / 71.0 / 83.6 — and exactly one image-only baseline,
OneFormer3D†, at **5.4 / 10.2 / 17.4**. The high numbers everyone remembers come from a different
input modality; the image-only entry is below us even in the posed protocol.

### 9.12 The same ruler on four benchmarks — the dataset adapters (todo 6d, 2026-08-09)

§9.9 and §9.11 gave the two axes that make a 3D number quotable (bridge, labels). This adds the
third: **which benchmark**. `scripts/eval_3d_maskdino.py --dataset {scannetv2,scannet200,
scannetpp,replica}` swaps the dataset and *nothing else* — same head, same two transfer modes,
same vendored evaluator, same lifting. It defaults to `scannetv2`, so every number above is
unchanged. The matrix it produces lives in `docs/RESULTS.md` §7.

| | GT source | taxonomy | our column |
|---|---|---|---|
| `scannetv2` | `scannet_3d_gt_val312` + `frames25k_val312` | 18 nyu40 classes | class-aware **and** class-agnostic |
| `scannet200` | **the same two tars** + `data/scannet200_constants.py` | 200 raw ScanNet ids | class-agnostic only |
| `scannetpp` | `scannetpp_3d_gt_val50` + `scannetpp_frames_val50` (49 scenes × 50 views) | 84 ScanNet++ instance classes | class-agnostic only |
| `replica` | `replica_3d_gt_8` + `replica_frames_8` (8 scenes × 50 views) | Replica's own | class-agnostic only |

**Why three of the four are class-agnostic-only.** The head has 19 ScanNet logits and those
taxonomies are not ours; inventing a correspondence would be fabricating a comparison. Instead
every instance is emitted under the evaluator's single collapsed label
(`train/benchmark3d.py::AGNOSTIC_LABEL_ID`) on **both** sides, which is exactly §9.11's setting —
and the one FAST3DIS and IGGT report in. Their class-aware fields are written as `null`, never as
a number.

**The prediction filter follows each dataset's GT taxonomy.** ScanNetv2's benchmark excludes
wall/floor, ScanNet++'s `top100_instance.txt` excludes the room shell, and our Replica GT drops
`wall`/`floor`/`ceiling` — so wall/floor predictions are dropped on those three. ScanNet200
*includes* wall and floor as valid classes, so there they are kept. Either way the two sides
match, which is what keeps each column single-variable
(`train/datasets3d.py::drop_wall_floor_predictions`).

**Every dataset is licensed the same way this evaluator was (§9.2)**: its own GT fed back as
predictions must score exactly 1.000 / 1.000 / 1.000. `scripts/gate_3d_gt.py`
(`slurm/gate_3d_gt.sh`) runs it over all scenes of the real tars and additionally re-derives the
pose/depth-scale convention by unprojecting the sensor depth onto the mesh. Results, 2026-08-09:
**all four pass at 1.000 / 1.000 / 1.000**; the geometry check's scene medians are 1.3 cm
(ScanNet, 312 scenes), 1.4 cm (ScanNet++, 49) and 0.5 cm (Replica, 8). The check fails on the
scene **median** rather than the worst probe frame, because 3 of ScanNet's own 312 val scenes and
1 of ScanNet++'s 49 carry a single drifted probe (up to 76 cm) while their medians stay under
6 cm — a rule that fails the reference dataset is the wrong rule.

**Two properties of the releases, not of the build, that the adapters must live with**
(`docs/DATASET.md` §2.1/§2.2): ScanNet++'s `segIndices` is one segment per vertex and Replica's
own `preseg` is a *planar* segmentation whose purity against the GT objects is only 0.77–0.95, so
on both the superpoint majority (§9.1 step 4) degenerates to a **per-vertex vote**. Replica's GT
instance set is additionally **our** construction (the room shell dropped) and every Replica
number must say so.
