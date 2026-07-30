I need a PLAN (not an implementation) for lifting the resolution bottleneck in my 3D instance
segmentation pipeline: a MaskDINO decoder on a frozen VGGT-1B backbone, supervised by ScanNet v2
2D instance annotations. Repo: /cluster/scratch/niacobone/vggt (read docs/MASKDINO.md §3, §6, §7.4
and docs/MASKDINO_COCO.md §1 before proposing anything).

## Facts already verified — do NOT spend a pass re-deriving these

Backbone → tokens
- `models/maskdino/model.py::MaskDINOVGGTModel` loads `VGGT.from_pretrained("facebook/VGGT-1B")`
  whole (aggregator + camera_head + depth_head + point_head + track_head), then freezes and
  `.eval()`s all of it. So VGGT's pretrained DPT weights are already in process, unused.
- `train/maskdino_data.py::extract_features` (~L127-157) is the only call site into the backbone
  for training: `agg_list, patch_start_idx = model.backbone.aggregator(imgs)` (24 entries), then
  `torch.cat([agg_list[i].float() for i in args.feature_layers], dim=-1)`. Note it CONCATENATES on
  the channel axis — a DPT-style reassemble needs the layers kept SEPARATE, so this contract has
  to change, not just the index list.
- `--feature_layers` already exists (`scripts/train_maskdino.py:96`, parsed L192, feeds
  `memory_dim=2048*len(args.feature_layers)` at L208). Default `-1` = last layer only.
- `train/maskdino_data.py::prepare_scenes` caches per scene: `features` [S,P,C] at
  `--cache_device`/`--cache_dtype` (default float32, on the train device), plus raw
  `images` uint8 [S,3,518,518] — so RGB IS available at train time for a guided upsampler,
  and 4 layers means 4x the cache (the binding constraint on how many scenes fit).

Pixel decoder → what the decoder expects today
- `models/maskdino/pixel_decoder.py::VGGTPixelDecoder.forward(tokens, patch_start_idx)` returns
  `(mask_features [B,256,37u,37u], multi_scale: list of [B,256,H_l,W_l] ordered HIGH→LOW)`.
- `tokens_to_map` drops the 5 special tokens and raises unless the patch count is a perfect square.
- Pyramid is ViTDet-synthetic from ONE map: 1x1 conv → 37x37; 3x3 s2 → 19x19; 3x3 s2 → 10x10;
  6-layer MSDeformAttn encoder over all three; `mask_features` = 1x1 conv on
  `mask_up(outs[0])`, where `mask_up` is `log2(mask_upsample)` x (ConvTranspose2d k2 s2 + GN + GELU).
- `--mask_upsample` is `[1,2,4]`, default 1 (`scripts/train_maskdino.py:137`).
- `models/maskdino/head.py::MaskDINOVGGTHead.forward` is just
  `mask_features, multi_scale = self.pixel_decoder(...)` → `self.predictor(multi_scale,
  mask_features, targets, frames_per_sample)`. `MaskDINODecoder` is in
  `models/maskdino/decoder.py`; `total_num_feature_levels` = `num_feature_levels`.

A multi-level interface already exists on the sibling COCO track
- `models/maskdino/pixel_decoder_coco.py::CocoPixelDecoder` takes `in_channels: Sequence[int]`
  (HIGH→LOW) plus an optional `highres_channels` stride-4 map and runs upstream MaskDINO's
  top-down FPN step for `mask_features`; `models/maskdino/coco_backbones.py` wraps
  vggt/dinov2/resnet50. The ScanNet and COCO paths are deliberately PARALLEL, not shared —
  say explicitly which code you would lift vs. re-derive, and do not couple them.

VGGT's own DPT head (approach 1's "may be reusable")
- `vggt/heads/dpt_head.py::DPTHead`: `intermediate_layer_idx=[4,11,17,23]`, per-layer 1x1
  `projects` → [256,512,1024,1024], `resize_layers` = [ConvTranspose2d k4/s4, ConvTranspose2d
  k2/s2, Identity, Conv2d 3x3 s2] → **148 / 74 / 37 / 19**, then `_make_scratch` +
  4x `FeatureFusionBlock` (`_make_fusion_block`) top-down → 256ch @ 148x148, then
  `custom_interpolate` to `518/down_ratio`.
- `feature_only=True` returns that fused map instead of activated depth — this is exactly how
  `vggt/heads/track_head.py` uses it (`down_ratio=2`, `pos_embed=False`). Confirm whether the
  pretrained weights for `backbone.depth_head` / `backbone.point_head` /
  `backbone.track_head.feature_extractor` are usable as-is (frozen or finetuned) rather than
  reimplementing reassemble+RefineNet, and say which of the three is the right donor and why.

Prior evidence that constrains the framing — engage with it, don't ignore it
- docs/MASKDINO.md §7.4, job 8895551: `--mask_upsample 2` scored 0.662/0.677 vs the 0.669/0.699
  bar = **−0.022 AP50, neutral inside the ±0.04 eval noise**. Simply upsampling `mask_features`
  did NOT help on ScanNet. Any plan must explain why the proposed change is different from that
  (richer token CONTENT from multiple layers / RGB guidance vs. a learned resize of one map).
- docs/MASKDINO_COCO.md §1 + `scripts/coco_mask_resolution_oracle.py`: on COCO a PERFECT model is
  capped at 44.7 mask AP at 37x37 and 84.2 at 148x148. **No equivalent oracle has been measured on
  ScanNet.** Treat "the bottleneck is real on ScanNet" as an unproven premise.

## What I want from you

1. **Ceiling first.** Phase 0 of the plan should be the ScanNet analogue of
   `scripts/coco_mask_resolution_oracle.py`: rasterise the official ScanNet GT at 37x37 / 74x74 /
   148x148 / 296x296, re-score it against itself through `train/perframe.py` +
   `train/eval_metrics.py::compute_instance_segmentation_metrics`, and report the mIoU/AP50/AP75
   ceiling per resolution. Cheap, CPU-only, and it decides whether phases 1-3 are worth the GPU
   hours. Tell me what result would kill the rest of the plan.

2. **Pick among the 4 approaches, given what THIS version of VGGT exposes.**
   1. DPT-style reassemble pyramid: pull layers 4/11/17/23 from the aggregator, per-layer resize
      to 148/74/37/19, fuse RefineNet-style. Say explicitly whether to (a) call
      `DPTHead(feature_only=True)` with pretrained weights, (b) subclass/adapt it, or (c) write a
      thin reassemble in `pixel_decoder.py` and feed real levels into the existing MSDeformAttn
      encoder — and whether the encoder should see 148x148 at all (cost: MSDeformAttn over
      148²+74²+37² tokens vs today's 37²+19²+10²).
   2. Learned image-guided upsampler (LoftUp / FeatUp / ViT-Up) after VGGT, guided by the cached
      RGB (PanSt3R v2's move).
   3. Pixel-shuffle head (VGGT-Ω): one MLP → 2u² channels + rearrange. Cheap, still bounded by
      token content.
   4. Sparse point prompts (VGGT-Segmentor): drop dense upsampling, track points via VGGT's
      `track_head`, decode masks SAM-style. Rule this in or out EARLY and explicitly — it would
      replace the MaskDINO mask path, not extend it.
   Recommend one (or a combination) with a reason, not a survey. Combinations are fine if you say
   what each part buys.

3. **Phased plan, smallest-blast-radius first.** For each phase: which files change, what the
   concrete output resolution is (e.g. "mask_features 37x37 → 148x148"), the extra params /
   VRAM / feature-cache cost, the flag name (new options MUST default to off / current behaviour),
   which of `tests/test_maskdino_model.py`, `test_maskdino_loss.py`, `test_maskdino_train.py`,
   `test_maskdino_multiframe.py`, `test_maskdino_viz.py` must be extended, and the go/no-go
   criterion before moving to the next phase. Mark clearly what is a stretch goal.

4. **Evaluation-protocol interactions — call these out loudly.**
   - `prepare_scenes` builds GT at `out_hw = (37*mask_upsample, 37*mask_upsample)` via
     `build_frame_targets`, and nothing in `train/perframe.py` / `train/maskdino_eval.py` ever
     interpolates predictions to a fixed GT resolution. So **changing mask resolution changes the
     resolution the metric is computed at**, and every number in docs/RESULTS.md and
     docs/MASKDINO.md §7 becomes non-comparable unless we fix a scoring resolution.
     Recommend: score at a FIXED resolution regardless of the model's native mask resolution, or
     argue for the alternative — and say what re-scoring the existing bar would cost.
   - Check the same for `scripts/eval_perframe.py` (the D4RT baseline scorer — it rebuilds a head
     from `head_config` and must not silently score at a different resolution than the MaskDINO run
     it is compared to) and for the per-bundle multi-frame protocol in
     `models/maskdino/multiframe.py` (docs/MASKDINO.md §8.2).
   - Note anything that touches `scripts/visualize_maskdino.py`.

5. **Invariants you must respect in the design** (CLAUDE.md): `head_config` is built from
   `locals()` and `tests/test_maskdino_model.py` asserts it equals the constructor's argument set —
   any new head argument must flow through it; class head is 19 sigmoid logits, no background
   column; `drop_empty_masks` stays; `initialize_box_type` stays `no`/`bitmask`; do not modify
   anything under `legacy/`; do not modify VGGT itself (`vggt/` is upstream and frozen) — if
   reusing `DPTHead`, do it by instantiation/subclassing from outside.

6. **Deliverable format:** a written plan in the response (and, if it's long, a file under
   docs/ — propose the filename, don't create it yet). Include a table of
   phase | files touched | output resolution | est. cost | risk | how it's validated.
   **Do not write any implementation code.** I want to approve the resolution/compute tradeoffs
   first. Ask me before starting if any of the above conflicts with what you find in the repo.

Environment: `myenv/bin/python`, GPU cluster, matplotlib headless. Tests are standalone CPU
scripts, not pytest. Training entry point is `sbatch slurm/train_maskdino.sh`.