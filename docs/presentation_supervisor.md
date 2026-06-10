---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section { font-size: 26px; }
  section h1 { font-size: 40px; }
  .small { font-size: 19px; }
  blockquote { font-size: 21px; color: #555; }
---

<!-- _class: lead -->

# Multi-View Consistent 3D Instance Segmentation on a Frozen VGGT Backbone

## A D4RT-style / DETR-like decoder trained on SAM3 pseudo-labels (ScanNet)

**Nico Iacobone — Research update for supervision meeting**
June 2026

> **Figure:** title background — screenshot of `demos/demo_gradio.py` 3D viewer with *Color By: Predicted Instances* on `scene0000_00`.

<!--
Speaker notes:
One-sentence framing: we keep VGGT-1B (CVPR'25 feed-forward 3D reconstruction) completely frozen and attach a lightweight (~6.5M param) DETR-like decoder that produces multi-view consistent instance segmentation masks, supervised by SAM3 pseudo-labels on ScanNet. Two milestones are complete: a validated end-to-end prototype (M1), and a regularized, unprompted-capable training loop (M2). The scaling experiment is the next step and is blocked only on data preprocessing.
-->

---

# 1 · Goal & Motivation

**Key message:** Reuse VGGT's fused multi-view 3D representation as a free feature extractor; only train a small segmentation decoder on top.

- VGGT (CVPR 2025): feed-forward multi-view transformer; its aggregator alternates per-frame and **global cross-frame attention** → features are already multi-view fused
- Hypothesis: those global features suffice for **cross-view-consistent instance segmentation** without touching the backbone
- Approach: D4RT-style **point-prompted queries** + DETR-like decoding + Mask2Former-style dense masks
- Supervision: **SAM3 masks on ScanNet** (pseudo-labels — no manual annotation)
- Backbone: **1.26B params frozen**; trainable head: **~6.5M params**

> **Figure:** high-level block diagram (frozen VGGT → trainable head); adapt the ASCII pipeline from `docs/MILESTONE_1.md` §0.

<!--
Speaker notes:
Emphasize the research bet: if VGGT's global attention already solves correspondence, segmentation identity should transfer across views "for free" — the decoder only has to read it out. This is also why everything stays frozen: it tests the representation, keeps training cheap (minutes), and makes results attributable to the decoder. D4RT inspiration = queries are (u,v,view) point prompts; DETR inspiration = set prediction with Hungarian matching and (since M2) a no-object class.
-->

---

# 2 · What Was Added to the Repo

**Key message:** The fork adds a self-contained head — VGGT code is untouched; every component has a standalone CPU test.

| Component | File | Test |
|---|---|---|
| Hook analysis | `docs/HOOK_PLAN.md` | — |
| ScanNet+SAM3 loader | `data/scannet_overfit.py` | `tests/test_phase2.py` |
| QueryGenerator + InstanceDecoder | `models/d4rt_decoder.py` | `tests/test_phase3/4.py` |
| Matcher + losses (+ no-object) | `train/loss.py` | `tests/test_phase5.py`, `test_milestone2.py` |
| Instance-seg metrics | `train/eval_metrics.py` | `tests/test_eval.py` |
| Overfit / multi-scene training | `scripts/train_overfit.py`, `train_multiscene.py` | — |
| 2D / 3D visualization | `scripts/visualize_masks.py`, `demos/demo_gradio.py` | — |

> **Table:** as shown; optionally a `git diff --stat` slide-footer to show backbone files are unmodified.

<!--
Speaker notes:
Development discipline: incremental phases, each validated by a standalone script before moving on (per docs/prompt.md). The backbone directory vggt/ is byte-identical to upstream except the gradio demo, which gained a segmentation-checkpoint viewer. This matters for reproducibility and for attributing all results to the head.
-->

---

# 3 · Dataset: ScanNet Scenes

**Key message:** 5 preprocessed ScanNet scenes today; ~100 masked frames per scene; instance identity is global across views.

- Per scene: `subset/` = ~100 stride-5 frames **with masks** (out of >5500 raw frames in `color/`)
- Images resized to **518×518** (divisible by VGGT's patch size 14 → 37×37 patch grid)
- 19 ScanNet classes, indices 1–19; **0 = background** everywhere
- Loader builds a per-pixel **global instance-ID map**: one cross-view-consistent ID per class per scene
- Each instance gets a normalized (u,v) **centroid** + representative frame (largest mask area) → used as its query prompt

> **Figure:** one scene strip — 4 RGB frames from `subset/` with the per-class mask PNGs from `masks/<class>/` overlaid (can be produced with a few lines on top of `data/scannet_overfit.py`).

<!--
Speaker notes:
Two practical pitfalls encoded in the loader: (1) sampling from color/ silently yields frames without masks — the loader prefers subset/; (2) cross-view identity: early version created one instance per (frame, class) pair — 32 "instances" in a 4-frame scene — which made multi-view consistency meaningless. Fixed to one global ID per class: scene0000_00 collapses 32 → 11 true cross-view instances, 8 spanning multiple frames. test_phase2.py asserts these invariants.
-->

---

# 4 · Preprocessing: SAM3 Pseudo-Labels

**Key message:** Supervision is fully automatic — SAM3 run per class on ScanNet frames — but currently **per-class binary**, which caps identity granularity.

- SAM3 produces, per frame, one **binary PNG per ScanNet class** (`masks/wall/00005.png`, …; 0=bg, 255=fg, thresholded at 127)
- Computed on a stride-5 subset (~100 frames/scene) — preprocessing cost is the current scaling bottleneck
- Consequence: two chairs in one scene are **indistinguishable** in the labels → class-level linking is the finest identity the data supports ("semantic-as-instance")
- Open preprocessing decision (M2 §7.5): emit **per-instance** (or panoptic) masks instead of per-class

> **Figure:** side-by-side: RGB frame | SAM3 per-class masks (tiled) | resulting global instance map from the loader.

<!--
Speaker notes:
Be explicit with the supervisor: this is a data limitation, not a model one (MILESTONE_1 §8.3). The decoder architecture is already instance-capable — Hungarian matching over queries doesn't care whether targets are semantic or instance level. The decision point: when preprocessing the next batch of scenes (tens to hundreds), should the SAM3 pipeline emit instance-encoded masks? That changes label format, the loader, and the semantic/instance/panoptic positioning of the whole project — worth deciding before the big preprocessing run.
-->

---

# 5 · Hook Point: Global Scene Features from VGGT

**Key message:** One tensor is read out of VGGT — `aggregated_tokens_list[-1]` — and used as cross-attention memory; nothing else changes.

- Aggregator: 24 blocks of alternating **frame attention** (`[B·S, P, C]`) and **global attention** (`[B, S·P, C]`), C = 1024
- Cached layers concatenate frame ‖ global intermediates → width **2C = 2048**
- Hook: last cached layer → **F : [B, S, P, 2048]**
  - P = 37·37 patch tokens + 1 camera + 4 register tokens (`patch_start_idx` separates them)
- Head consumes F under `torch.no_grad()`; backbone never sees gradients

> **Figure:** annotated diagram of the aggregator's alternating attention with the hook arrow at layer 23 (redraw from `docs/HOOK_PLAN.md`).

<!--
Speaker notes:
Why the last layer: it is the most fused multi-view representation, and it's what VGGT's own dense heads (depth/point) consume — so spatial detail demonstrably survives to this depth. The camera/register tokens are kept in the memory for cross-attention but excluded from the dense mask features (only patch tokens map to pixels). A foreshadow for slide 9: the raw magnitude of these features is enormous, which caused the single hardest bug of the project.
-->

---

# 6 · Decoder Architecture I — QueryGenerator

**Key message:** A query is a *point prompt*: where (u,v), which view, and what it locally looks like — summed into one 256-d token.

```
query = query_proj(  pos_proj(Fourier(u,v))        # 16 freqs → 64-d
                   + view_proj(Embedding[view_id])  # learned, per frame index
                   + patch_proj(MLP(9×9 RGB crop))) # 3·9·9=243 → 256, grid_sample
        → [B, N, 256]
```

- **Fourier positional encoding**: 16 log-spaced frequencies, sin+cos on (u,v)
- **View embedding**: `nn.Embedding(num_views, 256)` — table size is a checkpointed config
- **Local appearance**: 9×9 RGB patch around (u,v) from the query's view, 2-layer MLP (vectorized: one batched `grid_sample` for all B·N queries)
- Training queries: GT centroids (+ jitter); inference can use a **uniform grid** — no GT needed

> **Figure:** diagram of the three branches merging into a query token; `models/d4rt_decoder.py` lines for `QueryGenerator.forward`.

<!--
Speaker notes:
Design intent: the query must carry enough information to bind to one object — position alone is ambiguous across views, so the view embedding routes it to the right frame and the RGB patch disambiguates locally. This is the D4RT flavor: queries are points, not learned object slots. Important downstream consequence: because queries are prompts, coordinates are inputs, not predictions — there is no coordinate regression loss (slide 8).
-->

---

# 7 · Decoder Architecture II — InstanceDecoder & Dense Mask Head

**Key message:** A standard 4-layer transformer decoder cross-attends queries into VGGT memory; masks come out Mask2Former-style as query-kernel × pixel-feature **cosine** similarity.

- Memory path: flatten F → `[B, S·P, 2048]` → linear **2048→256** → **LayerNorm** ← essential
- `nn.TransformerDecoder`: **4 layers × 8 heads**, queries = tgt, normed memory = memory
- **Query skip connection**: `decoded = decoded + queries` ← essential (prevents collapse)
- Output heads:
  - `class_head` → `[B, N, 20]` (19 classes + background at index 0)
  - `mask_embed_head` → `[B, N, 256]` per-query mask kernel
  - `mask_feature_proj` on patch tokens → per-pixel features; mask logit = **cosine(kernel, pixel) × learnable temperature + bias** → `pred_masks [B, N, S, 37, 37]`
- One query → **one mask spanning all S frames** → multi-view consistency by construction

> **Figure:** architecture diagram (memory norm + skip highlighted in red); shape-flow table B,N,S,P.

<!--
Speaker notes:
The structural point for a CV audience: cross-view consistency is not enforced by a loss — it falls out of the architecture. The memory contains all frames jointly, and a single query produces a single mask tensor over all frames, so the same query *is* the same instance everywhere. The three highlighted choices (LayerNorm, skip, cosine) are not hyperparameters — each was required to make training work at all; story on slide 9.
-->

---

# 8 · Matching & Losses

**Key message:** DETR-style set prediction: mask-aware Hungarian matching, then Focal + Dice + weighted BCE; since M2, unmatched queries are pushed to *background*.

- **PointBipartiteMatcher** cost = class cost (1−p) + coordinate L2 + **dense Dice+BCE mask cost** (Mask2Former-style) → `linear_sum_assignment`
- **D4RTLoss** on matched pairs:
  - class: **Focal** (α=0.25, γ=2.0)
  - mask: **Dice + BCE with foreground `pos_weight`** (capped at 20) on `[N, S, 37, 37]`
- **No-object loss (M2):** unmatched queries get class loss toward background, down-weighted by `no_object_weight = 0.1` (DETR's `eos_coef`); mask loss stays matched-only
- Coordinates: matching cost only, **no loss term** (queries are prompts, not predictions)
- Batch-aware: per-sample matching, GT as lists of tensors for B>1

> **Figure:** matcher cost-matrix heatmap for one scene (queries × GT) — easy to extract from `train/loss.py`; loss-breakdown curves from `train.log`.

<!--
Speaker notes:
The no-object loss was the single most impactful M2 change: in M1, background queries confidently predicted foreground, crushing AP (0.54 AP50 despite 0.97 mIoU) and making inference impossible without GT-ordered queries. With it, ~2/3 of grid queries correctly self-classify as background and get filtered, enabling unprompted inference. Default off → M1 behavior is bit-identical, so all old tests still pass — the project's "new options default to previous behavior" convention.
-->

---

# 9 · Implementation Detail: the Query-Collapse Bug

**Key message:** Raw VGGT features nearly killed the project — three small changes (memory LayerNorm, query skip, cosine logits) were the difference between mIoU 0 and 0.90.

- Symptom: loss ↓ steadily, **mIoU stuck at 0**; all queries produced the *same* mask
- Diagnosis: queries (std ≈ 2) → identical decoded vectors (std ≈ 0) — **decoder output collapse**
- Cause: un-normalized VGGT memory has huge magnitude → cross-attention output swamps the query residual stream
- Fixes: **LayerNorm projected memory** · **query skip connection** · **cosine mask logits** (sigmoid saturation) · fg `pos_weight` · grad clipping
- An earlier embedding-L2 proxy loss had *hidden* the bug (supplied distinct targets per query)

> **Figure:** two-panel before/after — predicted masks all-identical vs. per-instance (regenerate by ablating the LayerNorm; or show the std-collapse table from `MILESTONE_1.md` §6).

<!--
Speaker notes:
Worth a full slide because it's the transferable lesson: when grafting heads onto foundation-model features, feature statistics are part of the interface. Also a methodology lesson — the proxy loss made everything look fine; only the real dense mask loss exposed the collapse. And the diagnostic that found it (per-layer std instrumentation) is cheap and reusable. These constraints are documented in CLAUDE.md as "violating these silently breaks training."
-->

---

# 10 · Training Setup

**Key message:** Frozen-backbone feature caching makes a full training run cost ~2 minutes on one GPU — iteration speed is effectively free.

- **Cache once, train fast:** backbone runs once per scene-bundle up front (~20 s for 13 bundles); every epoch trains only the 6.5M-param head → 1000 epochs ≈ **2.2 min** (RTX 4090)
- AdamW, lr 2e-3, linear warmup (30) → cosine decay; grad clipping
- **Regularization (M2):** `--bundles_per_scene 3` (random frame re-sampling), query jitter σ=0.02, color jitter 0.2, background queries resampled per step
- **Model selection (M2):** val prompted mIoU every 50 epochs → `checkpoint_best.pth`; optional early stopping
- `--cache_device cpu`: bundles (~90 MB each) in host RAM → scene count not GPU-bound
- Self-contained checkpoints: head weights + config + scene bundles + optimizer (resume & demo round-trip)

> **Figure:** loss + val-mIoU curves vs. epoch from `train.log` (`d4rt_m2_5scenes_20260610_133100`); annotate the val peak at epoch 450.

<!--
Speaker notes:
The caching design is what makes the whole methodology work: augmentation had to be cache-compatible (color jitter applied *before* the backbone pass, one draw per bundle; query jitter is backbone-free so it's per-step). Trade-off acknowledged: photometric augmentation diversity is limited to one draw per bundle. The 2-minute run time is why we could converge on architecture decisions quickly — and why the scaling experiment is logistically trivial once data exists.
-->

---

# 11 · Evaluation Protocol & Metrics

**Key message:** Two regimes — *prompted* (segment what I point at) and *unprompted* (find everything yourself); unprompted is the honest detection number.

- Metrics (`train/eval_metrics.py`): **mIoU, AP50, AP75, mAP (0.50:0.95), class accuracy** — at the 37×37 patch grid
- **Prompted:** queries at GT centroids → measures segmentation + classification given a point
- **Unprompted (M2):** 6×6 grid of cell centers per frame (288 queries @ 8 frames), zero GT; background-argmax predictions dropped, rest scored as detections
- Always reported on train scenes **and** a held-out scene never used in training
- Known artifact: multiple grid cells firing on one instance count as AP false positives (no NMS yet)

> **Table:** small schematic comparing the two regimes (query source / what it measures / GT needed at test time).

<!--
Speaker notes:
The prompted/unprompted split is the right framing for the supervisor: prompted isolates the decoder's readout quality from detection; unprompted is the deployable capability. Metrics replaced "loss went down" as the success criterion early (M1 §8.4) — the collapse bug showed loss alone is misleading. Patch-grid resolution caveat: all numbers are at 37×37; full-res upsampling (FPN-style pixel decoder) is listed future work and will shift absolute IoU values.
-->

---

# 12 · Quantitative Results

**Key message:** The pipeline demonstrably works (overfit + multi-scene fit + unprompted ≈ prompted); generalization is data-limited at 4 training scenes, exactly as expected.

<div class="small">

| Experiment | mIoU | AP50 | class_acc | Note |
|---|---|---|---|---|
| M1 single-scene overfit (400 ep) | 0.004 → **0.900** | 0.96 | 1.00 | gradient-flow sanity ✓ |
| M1 4-scene train (mean) | **0.967** | 0.54 | 0.94 | AP hurt by no bg supervision |
| M1 held-out scene (final) | 0.027 | 0.00 | 0.29 | peaked ~0.13 mid-training |
| M2 4-scene, prompted (mean) | 0.666 | **0.771** | 0.875 | AP50 0.54→0.77 via no-object loss |
| M2 4-scene, **unprompted** (mean) | **0.678** | 0.446 | — | ≈ prompted mIoU, zero GT! |
| M2 held-out, best ckpt | **0.138** @ ep450 | — | — | > M1 final (0.027) via model selection |

</div>

> **Table:** above (from `MILESTONE_1.md` §9 + `MILESTONE_2.md` §6); per-scene breakdowns in backup slides.

<!--
Speaker notes:
Three headline claims, each with its number: (1) the architecture can represent multiple scenes in one set of weights — M1 mean train mIoU 0.967; (2) unprompted inference works — grid queries match prompted mIoU (0.678 vs 0.666) with zero GT, the no-object head correctly suppresses ~200 of 288 grid queries; (3) generalization is bounded by N=4 scenes — best val mIoU 0.138, and the M1→M2 val gain comes from regularization + checkpoint selection, not real generalization. Pre-empt the question: train mIoU "dropping" 0.97→0.67 in M2 is expected — augmentation removed the fixed memorizable batch.
-->

---

# 13 · Qualitative Results

**Key message:** Predicted masks are visually coherent per frame *and consistent across views* — the same query colors the same object in every frame, including in 3D.

- 2D: `scripts/visualize_masks.py --checkpoint <run>/checkpoint_best.pth` → per-frame overlays, prediction vs. SAM3 GT
- 3D: `demos/demo_gradio.py --seg_checkpoint …` → VGGT point cloud colored by **predicted instance** — cross-view consistency visible as single-colored 3D objects
- Failure modes to show honestly: held-out scene (mostly background / wrong classes), duplicate grid detections on large objects (walls/floor)

> **Figures (to generate for the talk):**
> 1. 3×4 grid: frames × {RGB, GT, prediction} for a train scene
> 2. demo_gradio 3D screenshot, *Color By: Predicted Instances*
> 3. one held-out scene example (honest failure case)

<!--
Speaker notes:
Action item before the meeting: run visualize_masks.py on checkpoint_best.pth of d4rt_m2_5scenes_20260610_133100 (the visualizations/ folder for this run hasn't been generated yet) and capture 2–3 demo_gradio screenshots including the val-scene dropdown. The 3D coloring is the most persuasive artifact: a wall that's one color across 8 viewpoints is multi-view consistency made visible — no metric needed.
-->

---

# 14 · Limitations & Open Questions

**Key message:** The dominant limitation is supervision granularity and scale — both are preprocessing decisions, and they should be made before the big SAM3 run.

**Limitations**
- Per-class binary SAM3 masks → cannot separate same-class objects (semantic-as-instance ceiling)
- 37×37 mask resolution (patch grid); no full-res upsampling yet
- Duplicate detections from grid queries (no NMS / dedup) depress unprompted AP
- 5 scenes total → no statistically meaningful generalization claim yet

**Open questions for discussion**
1. **Label format for the next preprocessing run:** semantic vs. instance vs. panoptic SAM3 masks? (M2 §7.5 — blocks the pipeline design)
2. Point-prompted queries vs. **learned object queries** (true DETR) — when to switch?
3. How many scenes before partial backbone unfreezing is justified?
4. Target benchmark/baseline for an eventual paper: ScanNet semantic-3D? Compare against what?

> **Figure:** none — keep this slide for discussion.

<!--
Speaker notes:
This is the slide to slow down on. Question 1 is genuinely blocking: SAM3 preprocessing of hundreds of scenes is expensive, so the per-class vs per-instance decision must precede it. Question 2 has a natural decision point — if the grid-duplicate problem persists after scaling, learned queries solve detection and dedup at once but abandon the D4RT point-prompt framing. Question 4 is about positioning: prompted mode resembles interactive segmentation (SAM-style point prompts but multi-view consistent), unprompted resembles 3D instance segmentation — which story do we tell?
-->

---

# 15 · Next Steps

**Key message:** Code is ready and validated; the scaling curve is the experiment that decides everything — only data stands between us and it.

1. **Preprocess tens→hundreds of ScanNet scenes** with SAM3 (after deciding label format) — the only blocker
2. **Scaling experiment** (M2 §7.1): N ∈ {10, 25, 50, 100+} train scenes, held-out val, early stopping — *does held-out mIoU climb with N?*
3. Ablations once val signal > noise: no-object weight (0.05/0.1/0.4), augmentation on/off, grid density 4/6/8 (+ NMS decision)
4. Then: learned object queries · FPN-style mask upsampling · partial backbone unfreezing

**Timeline anchor:** training infra is ~2 min/run — the bottleneck is exclusively SAM3 preprocessing throughput.

> **Figure:** placeholder axes for the scaling curve (val mIoU vs. #scenes) with the N=4 point (0.138) already plotted — the plot we want to fill in.

<!--
Speaker notes:
End on the empty scaling plot — it makes the ask concrete: support/compute for preprocessing, and a decision on label format. Everything else (code, tests, eval protocol, model selection) is in place and regression-tested; scaling requires zero code changes, just scene lists. Proposed immediate plan: decide label format this week, preprocess a first 25-scene batch, run N={10,25} within a day of data availability.
-->
