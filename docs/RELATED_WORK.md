# Related work & positioning (literature survey 2026-07-08)

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
training signal. Our arms A/B/C/D + the grid-density ablation already constitute the
systematic query-strategy study that the field lacks — that is the thesis's strongest card.

## Direct competitors (read before writing anything positioning-shaped)

| Paper | arXiv | Verified? | Why it matters |
|---|---|---|---|
| **SegVGGT** — Joint 3D Reconstruction and Instance Segmentation from Multi-View Images | [2603.19926](https://arxiv.org/abs/2603.19926) | **YES (2026-07-08)** — object queries on multi-level VGGT geometric features, evaluated on ScanNetv2/ScanNet200, generalization on ScanNet++; adds "Frame-level Attention Distribution Alignment" (FADA) — per abstract (2026-07-23 check), a training-time-only auxiliary supervision that explicitly guides object queries to attend to instance-relevant frames, targeting the attention-dispersion problem caused by the large number of global image tokens; adds no inference-time cost. Abstract shows **no query-initialization ablation** (grid vs learned vs point) — our gap holds. | Closest published match to our framing. Must-cite, must-position-against. Check its eval protocol: if it scores lifted 3D point-cloud masks while we score per-view 2D masks at patch resolution, numbers are NOT comparable — say so explicitly. |
| **EPS3D** — End-to-End Feed-Forward 3D Panoptic Segmentation | [2606.08980](https://arxiv.org/abs/2606.08980) | no | Most recent; claims semantic–instance mutual enhancement + multi-view consistency. Verify it doesn't already claim geometry-anchored queries. |
| **FAST3DIS** — Feed-forward Anchored Scene Transformer for 3D Instance Segmentation | [2603.25993](https://arxiv.org/abs/2603.25993) | no | Query decoder on a frozen feed-forward *depth* backbone, end-to-end, no clustering. Directly relevant to the query-strategy question ("anchored" in the title — check what it anchors to). |
| **PanSt3R** — Multi-view Consistent Panoptic Segmentation | [2506.21348](https://arxiv.org/abs/2506.21348) | no | Earliest well-known entry in the lane; fuses per-view 2D predictions into consistent 3D masks — the *post-hoc fusion* paradigm we contrast with. Standard comparison point. |
| VGGT-Segmentor | [2604.13596](https://arxiv.org/abs/2604.13596) | **YES — full read 2026-07-23, MISCLASSIFIED by the harvest** | Ego-Exo4D ego↔exo mask *transfer* (given a mask in one view, segment the same object in the other), **not** ScanNet 3D instance segmentation — cite as related, not as a scoop. Their key finding (VGGT's raw point/track projections drift under large viewpoint change, but internal attention stays object-consistent) motivates a 3-stage Union Segmentation Head — Mask Prompt Fusion, Point-Guided Prediction (K-means points tracked via VGGT's track head), iterative Mask Refinement — plus a Single-Image Self-Supervised recipe (SAM pseudo-masks + augmented pairs) that alone beats the prior *supervised* SOTA (DOMR). SOTA 67.7/68.0 IoU on Ego→Exo/Exo→Ego (+18.0/+12.8 over DOMR). Same premise as us (frozen VGGT + small trainable head, geometry-alone-is-not-enough), different task/eval entirely (2-view prompted IoU vs. our multi-view GT-free AP50/mIoU) — slide write-up in `docs/old/slides_global_overview.md` (Part 7). |
| MoonSeg3R ([2512.15577](https://arxiv.org/abs/2512.15577)), SAB3R ([2506.02112](https://arxiv.org/abs/2506.02112)), Clutt3R-Seg ([2602.11660](https://arxiv.org/abs/2602.11660)), MV3DIS ([2604.08916](https://arxiv.org/abs/2604.08916)), MV-SAM ([2601.17866](https://arxiv.org/abs/2601.17866)) | — | no | Tier 2: same task on related backbones (CUT3R), or non-query alternatives (SAM-mask matching/lifting). Contrast material for related work. |
| VGGT-Det — Mining VGGT Internal Priors | [2603.00912](https://arxiv.org/abs/2603.00912) | no | Adjacent task (detection) but the "which internal VGGT layers carry object identity" angle transfers directly to our layer-ablation idea. |

## The four open gaps, mapped to this project

1. **Query strategy is unsettled — this is our paper.** No published work resolves how
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
