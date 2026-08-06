# Related work & positioning (literature survey 2026-07-08, competitors re-read 2026-07-28)

Source: a Claude-run arXiv harvest (822 papers, 113 classified on-topic; CSVs + landscape
figure live with the project owner, not in the repo). Tiering in that harvest was
LLM-assigned — **spot-check any paper before citing its claimed contribution** (one Tier-1
claim was already found misclassified, see below). Verified entries here were checked
against the actual arXiv abstract on 2026-07-08.

## Headline

"Attach a decoder to a frozen VGGT/DUSt3R-family backbone and do a downstream 3D task" is
now the dominant pattern of the last ~12 months (the "VGGT-X" genre: VGGT-Det, VGGT-Occ,
VGGT-Edit, DriveVGGT, …). **The architecture alone is no longer a contribution.** The
contribution must live in the *how*: query design, cross-view-consistency mechanism,
training signal.

**Update 2026-07-28 (read this before the older framing below).** Two things changed:

1. The MaskDINO result (docs/MASKDINO.md §7) says the retired head’s plateau was **architectural, not
   data**: the same data that made the baseline head *worse* (0.367@190 → 0.350@490) makes a DINO-family
   decoder gain +0.26 AP50 from 50 → 490 scenes. The query-initialisation study is
   therefore the **negative half** of the story — "no initialisation strategy rescues an
   under-powered decoder" — and the positive half is "a faithful DINO-family decoder on a
   **strictly frozen** 3D backbone, measured against that study on one protocol".
2. Re-reading the competitors confirms the *mechanisms* we were going to claim are already
   published (see "What is already claimed" below). What is not published is the **controlled
   comparison**: one backbone, one dataset, one protocol, decoder ingredients varied one at a
   time. Frame the contribution as the study, not as any single mechanism.

## Direct competitors (read before writing anything positioning-shaped)

| Paper | arXiv | Verified? | Why it matters |
|---|---|---|---|
| **SegVGGT — full-text read 2026-07-28, eval code read 2026-08-04** | [2603.19926](https://arxiv.org/abs/2603.19926) | **YES (full text + released eval code)** | **Backbone is NOT frozen**: VGGT (24 layers) with **LoRA on the frame- and global-attention modules**, new parameters trained fully. **400 learnable object queries** inserted as a cross-attention module after global attention in *every* one of the 24 aggregator layers (ablation: last-12 30.5 mAP, all-24 31.9 mAP on ScanNet200) — plain learned queries, **no anchor boxes, no denoising, no two-stage selection**. Masks = dot product of a query with a per-view instance-feature map from a semantic DPT head at H/2×W/2, so one query is one instance across all views (same *class* of mechanism as our `--multi_frame`, docs/MASKDINO.md §8.2 — treat shared queries as table stakes, not as our contribution). **Eval is 3D, but under a DIFFERENT PROTOCOL from ours — see "Two 3D protocols" below; this is the single most important thing to know before quoting their number.** ScanNetv2 50.4 / 71.7 / 87.0; ScanNet200 31.9 / 45.7 / 53.7; ScanNet++ zero-shot 13.3 / 33.9 / 56.4. Trains on the full 1201-scene split, 2–24 frames sampled per scene. Still **no query-initialisation / query-count ablation** — that gap holds. |
| SegVGGT — earlier abstract-level notes | | (2026-07-08) — object queries on multi-level VGGT geometric features, evaluated on ScanNetv2/ScanNet200, generalization on ScanNet++; adds "Frame-level Attention Distribution Alignment" (FADA) — per abstract (2026-07-23 check), a training-time-only auxiliary supervision that explicitly guides object queries to attend to instance-relevant frames, targeting the attention-dispersion problem caused by the large number of global image tokens; adds no inference-time cost. Abstract shows **no query-initialization ablation** (grid vs learned vs point) — our gap holds. | Closest published match to our framing. Must-cite, must-position-against. *"Check its eval protocol"* — **done 2026-08-04, and the answer was bigger than expected**: it scores 3D point-cloud masks (so not comparable to our 2D tables), and it reaches the point cloud with **GT poses + sensor depth**, so it is not comparable to our 3D ruler either. See "Two 3D protocols" below. |
| **EPS3D** — End-to-End Feed-Forward 3D Panoptic Segmentation | [2606.08980](https://arxiv.org/abs/2606.08980) | abstract only (2026-07-28) | Open-**vocabulary** 3D panoptic segmentation, trained by **distillation** into 3D-aware semantic + instance features, with an Ins2Sem/Sem2Ins mutual-enhancement module; benchmarks include Replica (+13 % mIoU semantics), ~1 s/scene. Different supervision regime (distilled, open-vocab) from our closed-set ScanNet-supervised setting — related work, not a baseline. No geometry-anchored-query claim in the abstract. |
| **FAST3DIS** — Feed-forward Anchored Scene Transformer for 3D Instance Segmentation | [2603.25993](https://arxiv.org/abs/2603.25993) | **YES — full text 2026-07-28** | **This is the paper that owns "3D-anchored queries".** Backbone = Depth Anything V3, frozen weights + **LoRA** (dual pass: LoRA-off for geometry, LoRA-on for instance features). A learned 3D **anchor generator** (MLP on global scene context) plus **anchor-sampling cross-attention**: anchors are projected into each view with the predicted camera and the feature map is bilinearly sampled, so a query aggregates evidence from the *same 3D location* across views instead of attending densely. Dual-level regularisation (multi-view contrastive + scheduled spatial-overlap penalty) to stop query collisions. Eval: per-view 2D masks **unprojected** to 3D with the recovered depth/cameras, scored as 3D instance segmentation (AP/AP50/AP25) on ScanNetv2, ScanNet++, Replica with Sim(3)+ICP alignment. ScanNetv2 AP25 31.6 / AP50 9.6 / AP 3.8 vs IGGT 28.7 / 11.2 / 2.8, with 115–250× speed-up. Ablations cover the two losses only — **no 2D-vs-3D-anchor ablation**. |
| **PanSt3R** — Multi-view Consistent Panoptic Segmentation (ICCV 2025) | [2506.21348](https://arxiv.org/abs/2506.21348) | abstract (2026-07-28) | Built on **MUSt3R** (multi-view DUSt3R), predicts geometry + multi-view panoptic segmentation in **one forward pass** — so it is *not* the NeRF-style post-hoc fusion paradigm it is often filed under; it explicitly argues against per-frame 2D + fusion, and revisits the mask-merging step with "a more principled approach for multi-view segmentation". Keep it as the earliest single-pass entry in the lane, and stop describing it as post-hoc fusion. |
| VGGT-Segmentor | [2604.13596](https://arxiv.org/abs/2604.13596) | **YES — full read 2026-07-23, MISCLASSIFIED by the harvest** | Ego-Exo4D ego↔exo mask *transfer* (given a mask in one view, segment the same object in the other), **not** ScanNet 3D instance segmentation — cite as related, not as a scoop. Their key finding (VGGT's raw point/track projections drift under large viewpoint change, but internal attention stays object-consistent) motivates a 3-stage Union Segmentation Head — Mask Prompt Fusion, Point-Guided Prediction (K-means points tracked via VGGT's track head), iterative Mask Refinement — plus a Single-Image Self-Supervised recipe (SAM pseudo-masks + augmented pairs) that alone beats the prior *supervised* SOTA (DOMR). SOTA 67.7/68.0 IoU on Ego→Exo/Exo→Ego (+18.0/+12.8 over DOMR). Same premise as us (frozen VGGT + small trainable head, geometry-alone-is-not-enough), different task/eval entirely (2-view prompted IoU vs. our multi-view GT-free AP50/mIoU) — slide write-up in `docs/old/slides_global_overview.md` (Part 7). |
| MoonSeg3R ([2512.15577](https://arxiv.org/abs/2512.15577)), SAB3R ([2506.02112](https://arxiv.org/abs/2506.02112)), Clutt3R-Seg ([2602.11660](https://arxiv.org/abs/2602.11660)), MV3DIS ([2604.08916](https://arxiv.org/abs/2604.08916)), MV-SAM ([2601.17866](https://arxiv.org/abs/2601.17866)) | — | no | Tier 2: same task on related backbones (CUT3R), or non-query alternatives (SAM-mask matching/lifting). Contrast material for related work. |
| VGGT-Det — Mining VGGT Internal Priors | [2603.00912](https://arxiv.org/abs/2603.00912) | no | Adjacent task (detection) but the "which internal VGGT layers carry object identity" angle transfers directly to our layer-ablation idea. |

## Numbers: what is comparable to what (settled 2026-07-28; 3D protocols split 2026-08-04)

**No published number in the table above is comparable to any 2D number in `docs/RESULTS.md`.**
State this explicitly wherever the two appear on the same page; do not build a table that puts
them side by side.

| | this project (2D tables) | SegVGGT / FAST3DIS / IGGT |
|---|---|---|
| what is scored | **per-view 2D masks** on VGGT's 37×37 patch grid (or 74×74 with `--mask_upsample 2`) | **3D masks** on the scene point cloud |
| against what GT | ScanNet 2D instance annotations rendered per frame | the official ScanNet **benchmark point clouds** |
| how much data | 490 train scenes (all our tar holds), 10 val scenes (0080–0089, project convention) | the official 1201/312 ScanNetv2 split |
| metric code | ours (`train/eval_metrics.py`) | the official 3D instance benchmark AP/AP50/AP25 |
| backbone | **strictly frozen**, features cached once, head-only training in minutes | LoRA-adapted (SegVGGT: VGGT frame+global attention; FAST3DIS: DA3) |

Our own 3D ruler (`docs/MASKDINO.md` §9, `docs/RESULTS.md` §5) exists precisely to cross that
line — but it lands in only *one* of the two protocols the literature actually uses.

### Two 3D protocols, printed in the literature as one (established 2026-08-04)

The published 3D numbers are **not one comparable set**. They split by *how a finished 2D mask
reaches the benchmark point cloud*, and that step dominates the score:

| | **posed transfer** | **unposed / predicted-geometry transfer** |
|---|---|---|
| who | **SegVGGT** (0.504 / 0.717 / 0.870) | **FAST3DIS** (0.038 / 0.096 / 0.316), **IGGT** (0.028 / 0.112 / 0.287 — *not* its own paper's number, see below), **this project** (0.023 / 0.067 / 0.268) |
| how masks reach the cloud | the GT benchmark cloud is **projected into each view** with ScanNet's GT poses + intrinsics; occlusion resolved by the ScanNet **sensor depth** map | per-view pixels are **unprojected** with the model's own predicted depth + cameras, then Sim(3)+ICP-registered into the mesh frame for scoring |
| geometry error in the bridge | **none** — the 3D↔2D correspondence is exact by construction | the full feed-forward reconstruction error (ours: median camera-centre RMS 0.14 m) |
| what the number therefore measures | 2D mask quality alone | 2D mask quality **×** feed-forward geometry quality |
| evaluator | official ScanNet, same options | official ScanNet, same options |
| **class-aware?** | **yes** (18 classes, per-class mean) — and **so are we** | **no**: FAST3DIS/IGGT score **class-agnostic**; our own class-agnostic column is 0.017 / 0.060 / 0.334 (§ below) |
| views per scene | every 20th frame (~75–100) | FAST3DIS 50; ours ~17 |

**The evaluator is not the difference — the 2D→3D bridge is.** This is why FAST3DIS (0.038) and
IGGT (0.028) cluster with us (0.023) while SegVGGT sits far above all three. Any table mixing
the two must say which protocol each row is in.

### Two provenance facts to state whenever these numbers are quoted (verified 2026-08-06)

**1. IGGT's 0.028 / 0.112 / 0.287 is not from the IGGT paper.** IGGT (arXiv 2510.22706, clone at
`/cluster/scratch/niacobone/IGGT_official`) reports **no ScanNet AP/AP50/AP25 at all**: its
ScanNet table gives spatial tracking (T-mIoU 69.41 / T-SR 98.66), reconstruction (Abs.Rel 1.90),
and open-vocabulary semantic segmentation (2D mIoU 60.46, **3D mIoU 39.68**), over **10 scenes ×
8–10 images**. The AP triple is **FAST3DIS's re-evaluation of IGGT** (50 sampled views, unposed,
Sim(3)+ICP). Label it *"IGGT, as re-evaluated by FAST3DIS"* — never *"IGGT (published)"*.

**2. FAST3DIS and IGGT are scored CLASS-AGNOSTIC; SegVGGT and we are class-aware.** FAST3DIS
§4.4: *"In the class-agnostic setting, we ignore the semantic class labels in the annotations and
focus purely on object localization and boundary quality"*, and the paper reports no class-aware
ScanNet numbers. The AP definition itself is identical everywhere (IoU 0.5:0.05:0.95 for AP,
fixed 0.5 / 0.25 for AP50 / AP25) — **`mAP` and `AP` in this literature are the same metric**, so
the naming difference between the SegVGGT and FAST3DIS tables means nothing. The *setting* does.

**We measured our own class-agnostic column rather than reasoning about it** (jobs 9861563 /
9861564, 312 scenes, 0 failures; docs/MASKDINO.md §9.11, RESULTS.md §5). The result **reverses
the intuition** that class-agnostic is simply the easier setting:

| our checkpoint (unposed) | class-aware (18) | class-agnostic |
|---|---|---|
| defaults | 0.023 / 0.067 / 0.268 | **0.013 / 0.050 / 0.320** |
| tuned lifting knobs | 0.029 / 0.083 / 0.305 | **0.017 / 0.060 / 0.334** |
| FAST3DIS (published) | — | 0.038 / 0.096 / 0.316 |
| IGGT (via FAST3DIS) | — | 0.028 / 0.112 / 0.287 |

**The like-for-like verdict: we lead the published cluster on AP25 (0.334 vs 0.316 and 0.287) and
trail it ~1.6–2.2× on AP50 and AP.** Say exactly that. The class-aware "in FAST3DIS's ballpark"
line was flattering us on AP/AP50 through a metric difference, and must not be repeated.

*Why the drop.* Class-agnostic replaces a per-class mean with one instance-pooled ranking. Our
class-aware mean is carried by rare, distinctive classes — toilet alone scores 0.508 AP50 for
1/18 of the mean, sink and refrigerator 0.173 — while the numerous classes are weak (chair 0.053,
cabinet 0.040, bookshelf 0.001) and `otherfurniture`, which our 19-class head cannot predict at
all, scores 0.000. Pooling deletes the rare-class leverage and drops every unmatched
`otherfurniture` instance into one recall curve. Nothing about the collapse is unfair — it is
simply the setting our competitors report in, and it is harsher on a class-conditioned head.

**3. The high ScanNet numbers people remember are a different input modality.** SegVGGT's own
Table 1 places point-cloud-input methods at Mask3D 55.2 / 73.7 / 85.3, Relation3D 62.5 / 80.2 /
87.0, SegDINO3D (P+I+D+C) 64.0 / 81.5 / 88.9, ODIN (I+D+C) 50.0 / 71.0 / 83.6. Those consume a
reconstructed cloud or RGB-**D**; they are not in our lane. The only *image-only* baseline in
that table, OneFormer3D†, scores **5.4 / 10.2 / 17.4** — under our AP50 even though it is scored
in the posed protocol. Keep this row handy: it is the honest anchor for "image-only is hard".

**How much of the gap is protocol — MEASURED 2026-08-04, do not guess at this**
(docs/MASKDINO.md §9.10, RESULTS.md §5.1). We implemented SegVGGT's bridge on our own masks
(`--transfer_mode gt_projection`, oracle-verified round-trip purity 0.9999) and re-scored the
same checkpoint on val-312: **0.023 / 0.067 / 0.268 → 0.060 / 0.156 / 0.408**. So the protocol
is worth a factor of **2.3** on AP50, and a factor of **~4.6** to SegVGGT's 0.717 is *real* —
their LoRA-adapted backbone, 4–6× our views, 259×196 masks against our 37×37, 600 kept queries
against our 100. An earlier draft of this section said the outlier "is the protocol, not the
model"; that overstated it and is struck. **Write it as: the two numbers are not comparable
as printed, and when made comparable a substantial genuine gap remains.**

**Evidence.** The paper says so in as many words (§ implementation details, verified 2026-08-06):
*"For the predicted mask associated with the object query, we project each 3D point from the
ground-truth point cloud onto all sampled views. We then compute its visibility and determine
whether the projected pixel falls within the predicted mask. **We utilize the ground-truth depth
maps and camera poses during this mapping stage for fair comparison.**"* Quote this rather than
our code reading — it is their own sentence, so the point cannot be argued.

The released code (clone at `/cluster/scratch/niacobone/SegVGGT`, read 2026-08-04) matches it —
their evaluator does **not unproject** anything:

- `eval/eval_instance_seg.py:243-336` (`map_pred_inst_to_gt_pointcloud`) projects the GT point
  cloud into each view and reads the predicted 2D mask at the landing pixel.
- Extrinsics are ScanNet GT `pose/{frame}.txt` (`eval_instance_seg.py:198`); intrinsics are GT
  `intrinsic_depth.txt` (`eval/instance_eval_common.py:68`); occlusion is the ScanNet **sensor**
  depth `depth/{frame}.png` within 0.1 m (`eval_instance_seg.py:178-182`, `305-307`, `451`).
- Hence **no Sim(3), no ICP, no scale estimation, no vote radius** anywhere in the path.
- **VGGT's geometry heads are never called** in their instance eval
  (`eval/instance_eval_common.py:168-189` runs the aggregator + the semantic head only).
- Their metric code is mmdet3d's copy of the official ScanNet evaluator with **the same options
  as our vendored one** — overlaps `[0.5:0.05:0.9] + [0.25]`, `min_region_sizes 100`, 18 classes,
  superpoint majority (`eval/instance_seg_eval.py:523-540` vs `train/benchmark3d.py:36-37`).

Secondary differences, all favouring them but none of them the main effect: **~75–100 views per
scene** (`--downsample_factor 20` over a full `.sens` extraction — `eval_instance_seg.py:169`,
their `docs/data_preparation.md`) vs our ~17 from the official 25k export; masks at **259×196**
(half the 518×392 input, `return_feature_maps_down_ratio: 2` in
`configs/eval/segvggt_scannetv2.yaml`) vs our 37×37 grid; **600** kept query-class pairs
(`instance_eval_common.py:107`) vs our `--eval_topk 100`; and they train and are scored on
`otherfurniture`, which our 19-class head cannot predict.

**Be fair about this — it is not cheating, and must never be written as if it were.** SegVGGT's
*model* consumes unposed RGB only, exactly like ours; the GT geometry is used solely to move
finished masks onto the benchmark cloud for scoring, which deliberately isolates segmentation
quality from reconstruction quality. That is a legitimate evaluation choice. The problem is only
that the two protocols appear in the literature inside one table without the distinction —
and SegVGGT and FAST3DIS are contemporaneous preprints (2603.19926 and 2603.25993), so neither
could have cited the other.

Two further consequences worth stating in the thesis: (a) the unposed protocol needs no poses at
inference, which is exactly the input assumption we defend — and it measures something strictly
harder than the posed one, so *lower is not worse*; (b) our frozen-backbone constraint is a
**deliberate, differentiating design choice**, not the default of the field: every direct
competitor adapts its backbone with LoRA.

## What is already claimed (so we do not claim it again)

| Mechanism | Owned by | What is left for us |
|---|---|---|
| Object queries shared across views on a VGGT-family backbone | SegVGGT (400 queries in all 24 aggregator layers) | our `--multi_frame` (§8.2) is the same *class* of idea — report it as a controlled comparison against our own single-frame model, not as a new mechanism |
| **3D-anchored queries** (learned 3D anchor generator + project-and-sample cross-attention) | **FAST3DIS** | our §8.3 is therefore an **ablation, not a contribution**: 3D anchors vs 2D DAB boxes *inside the same DINO-family decoder, same frozen backbone, same data, same protocol* — which nobody has run, and which directly re-tests the archived 3D-anchor negative result |
| Attention-dispersion fix for queries over many global tokens | SegVGGT (FADA, training-time only) | we get part of this for free from deformable attention (sampling 4 points/level around an anchor instead of dense attention); worth one sentence contrasting the two remedies |
| Single-pass multi-view panoptic prediction (no test-time optimisation) | PanSt3R | keep as the "why not splat / why not fuse" comparison point |
| Anti-duplicate regularisation for queries | FAST3DIS (contrastive + spatial-overlap penalty) | our over-prediction problem (duplicate FPs on the retired head) is handled instead by DINO's one-to-one Hungarian matching + DN; that contrast is a legitimate discussion point |

## The four open gaps, mapped to this project

> **Superseded by the 2026-07-28 update above for gap 1 and gap 3.** Gap 1 stands as a *gap in
> the literature* (nobody ablates query initialisation), but our answer to it is negative, and
> the thesis's strongest card is now the frozen-backbone decoder study of docs/MASKDINO.md.
> The query-initialisation study it refers to has since been completed and retired; its tables are
> archived in `docs/old/ARMS_SUMMARY.md` and are not part of the current story.

1. **Query strategy is unsettled — nobody has ablated it (our study, now the negative half).**
   No published work resolves how object queries should be initialized / made view-consistent for
   feed-forward 3D. Our own study of that question is closed and its answer was negative: no query
   initialisation on the retired head reached the bar that a DINO-family decoder clears on the same
   frozen backbone and the same data. The live version of the question is `--anchor_3d`
   (docs/MASKDINO.md §8.3) — 3D anchors vs 2D DAB boxes inside the *current* decoder, which is an
   ablation against FAST3DIS's mechanism, not a contribution.
2. **Consistency intrinsic to the query, not post-hoc — we already have it; claim it.**
   Our decoder produces `pred_masks [B, N, S, h, w]`: one query = one instance across all
   views by construction, vs the PanSt3R/MV3DIS paradigm of fusing/matching per-view 2D
   masks. Under-emphasized in our own framing. **The metric now exists (2026-08-01,
   docs/MASKDINO.md §6.6):** `train/eval_metrics.py::multiview_consistency_metrics` reports
   `bundle_view_consistency` (per matched instance, the fraction of its visible views explained
   at IoU ≥ 0.5 by its bundle-matched query) and `bundle_id_switch` (the fraction where another
   query is the better match). No run has been scored on it yet — the numbers to quote come
   from the next `--multi_frame` run, ideally the §7.4.1 ablation triple.
3. **Backbone-agnostic decoding — SKIP.** Real gap (one decoder across VGGT/CUT3R/Pi3) but
   large engineering scope, Lite3R already owns the "model-agnostic" framing, and it does
   not serve the thesis timeline.
4. **Mining frozen-backbone internals — cheap side chapter.** We hook only
   `aggregated_tokens_list[-1]`. The feature-caching setup makes a which-layer ablation
   nearly free; VGGT-Det shows the appetite for this analysis.

## The "why not just splat?" answer (keep ready for reviewers)

Gaussian-Splatting / NeRF instance methods (Ilov3Splat, CAGS, Chorus, …) get multi-view
consistency *by construction* (one 3D representation rendered to all views) — but require
per-scene optimization. Our pitch: feed-forward, per-scene-optimization-free, no GT
geometry/depth sensor needed, seconds not minutes at inference. Make sure the ScanNet eval
isolates that advantage.
