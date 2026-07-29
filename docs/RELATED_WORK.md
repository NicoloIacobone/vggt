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

1. The MaskDINO result (docs/MASKDINO.md §7) says the D4RT plateau was **architectural, not
   data**: the same data that made arm C *worse* (0.367@190 → 0.350@490) makes a DINO-family
   decoder gain +0.26 AP50 from 50 → 490 scenes. The query-initialisation study (arms A–E) is
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
| **SegVGGT — full-text read 2026-07-28** | [2603.19926](https://arxiv.org/abs/2603.19926) | **YES (full text)** | **Backbone is NOT frozen**: VGGT (24 layers) with **LoRA on the frame- and global-attention modules**, new parameters trained fully. **400 learnable object queries** inserted as a cross-attention module after global attention in *every* one of the 24 aggregator layers (ablation: last-12 30.5 mAP, all-24 31.9 mAP on ScanNet200) — plain learned queries, **no anchor boxes, no denoising, no two-stage selection**. Masks = dot product of a query with a per-view instance-feature map from a semantic DPT head at H/2×W/2, so one query is one instance across all views (same *class* of mechanism as our `--multi_frame`, docs/MASKDINO.md §8.2 — treat shared queries as table stakes, not as our contribution). **Eval is 3D**: per-view masks are unprojected into a common 3D space, majority-voted per superpoint, and scored against the official benchmark point clouds with mAP/mAP50/mAP25. ScanNetv2 50.4 / 71.7 / 87.0; ScanNet200 31.9 / 45.7 / 53.7; ScanNet++ zero-shot 13.3 / 33.9 / 56.4. Trains on the full 1201-scene split, 2–24 frames sampled per scene. Still **no query-initialisation / query-count ablation** — that gap holds. |
| SegVGGT — earlier abstract-level notes | | (2026-07-08) — object queries on multi-level VGGT geometric features, evaluated on ScanNetv2/ScanNet200, generalization on ScanNet++; adds "Frame-level Attention Distribution Alignment" (FADA) — per abstract (2026-07-23 check), a training-time-only auxiliary supervision that explicitly guides object queries to attend to instance-relevant frames, targeting the attention-dispersion problem caused by the large number of global image tokens; adds no inference-time cost. Abstract shows **no query-initialization ablation** (grid vs learned vs point) — our gap holds. | Closest published match to our framing. Must-cite, must-position-against. Check its eval protocol: if it scores lifted 3D point-cloud masks while we score per-view 2D masks at patch resolution, numbers are NOT comparable — say so explicitly. |
| **EPS3D** — End-to-End Feed-Forward 3D Panoptic Segmentation | [2606.08980](https://arxiv.org/abs/2606.08980) | abstract only (2026-07-28) | Open-**vocabulary** 3D panoptic segmentation, trained by **distillation** into 3D-aware semantic + instance features, with an Ins2Sem/Sem2Ins mutual-enhancement module; benchmarks include Replica (+13 % mIoU semantics), ~1 s/scene. Different supervision regime (distilled, open-vocab) from our closed-set ScanNet-supervised setting — related work, not a baseline. No geometry-anchored-query claim in the abstract. |
| **FAST3DIS** — Feed-forward Anchored Scene Transformer for 3D Instance Segmentation | [2603.25993](https://arxiv.org/abs/2603.25993) | **YES — full text 2026-07-28** | **This is the paper that owns "3D-anchored queries".** Backbone = Depth Anything V3, frozen weights + **LoRA** (dual pass: LoRA-off for geometry, LoRA-on for instance features). A learned 3D **anchor generator** (MLP on global scene context) plus **anchor-sampling cross-attention**: anchors are projected into each view with the predicted camera and the feature map is bilinearly sampled, so a query aggregates evidence from the *same 3D location* across views instead of attending densely. Dual-level regularisation (multi-view contrastive + scheduled spatial-overlap penalty) to stop query collisions. Eval: per-view 2D masks **unprojected** to 3D with the recovered depth/cameras, scored as 3D instance segmentation (AP/AP50/AP25) on ScanNetv2, ScanNet++, Replica with Sim(3)+ICP alignment. ScanNetv2 AP25 31.6 / AP50 9.6 / AP 3.8 vs IGGT 28.7 / 11.2 / 2.8, with 115–250× speed-up. Ablations cover the two losses only — **no 2D-vs-3D-anchor ablation**. |
| **PanSt3R** — Multi-view Consistent Panoptic Segmentation (ICCV 2025) | [2506.21348](https://arxiv.org/abs/2506.21348) | abstract (2026-07-28) | Built on **MUSt3R** (multi-view DUSt3R), predicts geometry + multi-view panoptic segmentation in **one forward pass** — so it is *not* the NeRF-style post-hoc fusion paradigm it is often filed under; it explicitly argues against per-frame 2D + fusion, and revisits the mask-merging step with "a more principled approach for multi-view segmentation". Keep it as the earliest single-pass entry in the lane, and stop describing it as post-hoc fusion. |
| VGGT-Segmentor | [2604.13596](https://arxiv.org/abs/2604.13596) | **YES — full read 2026-07-23, MISCLASSIFIED by the harvest** | Ego-Exo4D ego↔exo mask *transfer* (given a mask in one view, segment the same object in the other), **not** ScanNet 3D instance segmentation — cite as related, not as a scoop. Their key finding (VGGT's raw point/track projections drift under large viewpoint change, but internal attention stays object-consistent) motivates a 3-stage Union Segmentation Head — Mask Prompt Fusion, Point-Guided Prediction (K-means points tracked via VGGT's track head), iterative Mask Refinement — plus a Single-Image Self-Supervised recipe (SAM pseudo-masks + augmented pairs) that alone beats the prior *supervised* SOTA (DOMR). SOTA 67.7/68.0 IoU on Ego→Exo/Exo→Ego (+18.0/+12.8 over DOMR). Same premise as us (frozen VGGT + small trainable head, geometry-alone-is-not-enough), different task/eval entirely (2-view prompted IoU vs. our multi-view GT-free AP50/mIoU) — slide write-up in `docs/old/slides_global_overview.md` (Part 7). |
| MoonSeg3R ([2512.15577](https://arxiv.org/abs/2512.15577)), SAB3R ([2506.02112](https://arxiv.org/abs/2506.02112)), Clutt3R-Seg ([2602.11660](https://arxiv.org/abs/2602.11660)), MV3DIS ([2604.08916](https://arxiv.org/abs/2604.08916)), MV-SAM ([2601.17866](https://arxiv.org/abs/2601.17866)) | — | no | Tier 2: same task on related backbones (CUT3R), or non-query alternatives (SAM-mask matching/lifting). Contrast material for related work. |
| VGGT-Det — Mining VGGT Internal Priors | [2603.00912](https://arxiv.org/abs/2603.00912) | no | Adjacent task (detection) but the "which internal VGGT layers carry object identity" angle transfers directly to our layer-ablation idea. |

## Numbers: what is comparable to what (settled 2026-07-28)

**No published number in the table above is comparable to any number in `docs/RESULTS.md`.**
State this explicitly wherever the two appear on the same page; do not build a table that puts
them side by side.

| | this project | SegVGGT / FAST3DIS / IGGT |
|---|---|---|
| what is scored | **per-view 2D masks** on VGGT's 37×37 patch grid (or 74×74 with `--mask_upsample 2`) | **3D masks**: per-view predictions unprojected with depth+cameras into the scene point cloud (SegVGGT additionally majority-votes per superpoint; FAST3DIS aligns with Sim(3)+ICP) |
| against what GT | ScanNet 2D instance annotations rendered per frame | the official ScanNet **benchmark point clouds** |
| how much data | 490 train scenes (all our tar holds), 10 val scenes (0080–0089, project convention) | the official 1201/312 ScanNetv2 split |
| metric code | ours (`train/eval_metrics.py`) | the official 3D instance benchmark AP/AP50/AP25 |
| backbone | **strictly frozen**, features cached once, head-only training in minutes | LoRA-adapted (SegVGGT: VGGT frame+global attention; FAST3DIS: DA3) |

Two consequences worth stating in the thesis: (a) an unprojection step needs depth and poses,
which is exactly the input assumption we avoid — but it also means their metric measures
something strictly harder than ours, so *lower is not worse*; (b) our frozen-backbone constraint
is now a **deliberate, differentiating design choice**, not the default of the field: every
direct competitor adapts its backbone with LoRA.

## What is already claimed (so we do not claim it again)

| Mechanism | Owned by | What is left for us |
|---|---|---|
| Object queries shared across views on a VGGT-family backbone | SegVGGT (400 queries in all 24 aggregator layers) | our `--multi_frame` (§8.2) is the same *class* of idea — report it as a controlled comparison against our own single-frame model, not as a new mechanism |
| **3D-anchored queries** (learned 3D anchor generator + project-and-sample cross-attention) | **FAST3DIS** | our §8.3 is therefore an **ablation, not a contribution**: 3D anchors vs 2D DAB boxes *inside the same DINO-family decoder, same frozen backbone, same data, same protocol* — which nobody has run, and which directly re-tests the arm-E negative result |
| Attention-dispersion fix for queries over many global tokens | SegVGGT (FADA, training-time only) | we get part of this for free from deformable attention (sampling 4 points/level around an anchor instead of dense attention); worth one sentence contrasting the two remedies |
| Single-pass multi-view panoptic prediction (no test-time optimisation) | PanSt3R | keep as the "why not splat / why not fuse" comparison point |
| Anti-duplicate regularisation for queries | FAST3DIS (contrastive + spatial-overlap penalty) | our over-prediction problem (arm B/C duplicate FPs) is handled instead by DINO's one-to-one Hungarian matching + DN; that contrast is a legitimate discussion point |

## The four open gaps, mapped to this project

> **Superseded by the 2026-07-28 update above for gap 1 and gap 3.** Gap 1 stands as a *gap in
> the literature* (nobody ablates query initialisation), but our answer to it is negative, and
> the thesis's strongest card is now the frozen-backbone decoder study of docs/MASKDINO.md.
> "Missing arm E" below has since been run (see docs/ARMS_SUMMARY.md) and lost.

1. **Query strategy is unsettled — nobody has ablated it (our study, now the negative half).**
   No published work resolves how
   object queries should be initialized / made view-consistent for feed-forward 3D. We
   already have: point prompts plateau (arm A), trained grid queries peak at their training
   density and die by duplicate FPs (arm B + grid-density ablation → gap is architectural),
   hybrid fails (arm D), learned queries scale (arm C, honest AP50 0.228 vs 0.185 best-grid).
   **Missing arm: E — 3D-anchored queries**, seeded from VGGT's *own predicted pointmap
   geometry* instead of image-space (u,v). `QueryGenerator` is currently purely 2D
   (Fourier(u,v) + view embedding + RGB patch). A 3D anchor also gives a natural
   one-query-per-object dedup mechanism, attacking our known over-prediction failure
   (338 kept vs 144 GT; duplicate FPs are the identified lever).
2. **Consistency intrinsic to the query, not post-hoc — we already have it; claim it.**
   Our decoder produces `pred_masks [B, N, S, h, w]`: one query = one instance across all
   views by construction, vs the PanSt3R/MV3DIS paradigm of fusing/matching per-view 2D
   masks. Under-emphasized in our own framing. To substantiate: add an explicit
   **cross-view consistency metric** to `train/eval_metrics.py` (e.g. per matched instance,
   IoU agreement of its mask identity across views / ID-switch rate).
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
