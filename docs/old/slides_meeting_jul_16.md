---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section { font-size: 25px; }
  section h1 { font-size: 38px; }
  .small { font-size: 18px; }
  .cols { display: flex; gap: 24px; align-items: center; }
  .cols > div { flex: 1; }
  section.compact { font-size: 20px; }
  section.compact h1 { font-size: 34px; }
  section.compact .cols { align-items: flex-start; }
  blockquote { font-size: 20px; color: #555; }
  img { background: transparent; }
  table { font-size: 19px; }
---

<!-- _class: lead -->
<!-- _paginate: false -->

# Multi-View Consistent 3D Instance Segmentation on a Frozen VGGT Backbone

## Progress update: official-GT migration, the arm-E study, and the data-scaling answer

**Nico Iacobone — Research update for supervision meeting**
July 16, 2026

<!--
Speaker notes:
Since Jun 15 three big threads closed: (1) the GT was replaced (SAM3 → official ScanNet) after an audit found systematic label duplicates — all baselines re-measured; (2) the query-strategy ablation is COMPLETE (arms A–E, 8 trained variants) with arm C confirmed and arm E closed today with a clean failure decomposition; (3) the "more data?" question is answered: no. Frame the meeting as "the ablation study is done — this is the thesis's experimental core; next is measurement & protocol work, not new arms."
-->

---

<!-- _class: compact -->

# 1 · What's new since June 15

| Thread | Outcome |
|---|---|
| **GT migration** (audit 2026-07-07) | SAM3 GT had ~15.9% multi-class foreground px (same object labeled under 2 classes → built-in false positives). Replaced with **official ScanNet v2 2D instances**: 200 → **500 scenes, 7379 instances, 0 duplicates**. All baselines re-measured. |
| **New quotable baseline** (arm C, N=190, official GT) | val mIoU **0.367** / honest AP50 **0.199**. Cross-eval showed ~half of the old 0.228 AP50 headline was SAM3-GT-specific (SAM3-trained ckpt scores **0.117** on clean GT). |
| **Literature survey** (`RELATED_WORK.md`) | "Decoder on frozen VGGT" is now a crowded genre (SegVGGT = closest). Repositioned: the **query-strategy study is the contribution**, not the architecture. |
| **Arm E — 3D-anchored queries** (the gap no competitor covers) | v0 + 3-variant v1 ablation run; **closed today** with a clean failure decomposition + a calibration finding (slides 3–5). |
| **Data scaling at N=490** (finished today) | **2.6× more scenes does NOT improve arm C** (slide 6). |

All arms, one table: `docs/ARMS_SUMMARY.md` (new).

---

<!-- _class: compact -->

# 2 · Recap: the arms, and why E existed

| Arm | Query init | Status |
|---|---|---|
| A point prompts | (u,v) centroids / grid | plateaued at N≥50 — superseded |
| B trained grid | + trained grid queries | fixed, still loses (AP50 unstable) — closed |
| C **learned (DETR)** | 64 learned embeddings | **base**: 0.367 / 0.199 (N=190, official GT) |
| D hybrid C+A | learned + prompts | fixed (NaN), only ties C, overfits — closed |
| E **anchor3d** | FPS anchors in VGGT's **own pointmap** | v0 + v1 — **closed today** (next slides) |

**Why arm E**: (i) the one query-init cell nobody published (SegVGGT/EPS3D/FAST3DIS all use image-space or learned queries); (ii) arm C's persistent failure is **over-prediction** (keeps ~1.2–1.4× the GT instance count; threshold & grid-density levers both failed) — 3D-spread anchors are a built-in duplicate suppressor: one query per 3D location, shared across views, GT-free by construction.

---

<!-- _class: compact -->

# 3 · Arm E v0 → diagnosis → v1 (what we changed and why)

**v0** (2026-07-15): anchors = farthest-point sampling over the per-scene-normalized token cloud (positions from the frozen point head); content = kNN(8)-pooled frozen features.
**Result**: val mIoU 0.179 / AP50 0.072 — loses to arm C (0.269 / 0.144)… **but the dedup worked** (kept/GT 1.08× vs C's 1.38× at final epoch; single-scene overfit kept *exactly* 10/10).

**Code-review diagnosis** (2026-07-16) — two separable suspects:
1. **Fourier bug**: the positional bands (1–10 cycles/unit) were sized for (u,v) ∈ [0,1]; anchor xyz spans ±2.5 → the base band *wraps*, no unambiguous coarse position. Fix: `--anchor_coord_scale 0.2`.
2. **Frozen content**: v0 also *underfits train* (mIoU 0.535 vs C's 0.731) with class loss 10× higher — pooled surface-point features are scene-specific and un-specializable.

**v1 = a 3-run ablation that separates the two** (all N=50, official GT, 1000 ep):
(i) **hybrid** — DAB-DETR-style: anchor geometry + per-slot *learned* content; (ii) **positional-only** — content removed entirely; (iii) **pooled+fix** — v0 content, only the Fourier fix.

---

<!-- _class: compact -->

# 4 · Arm E v1 results (today) — no win, but a clean decomposition

| Variant | best val mIoU | honest AP50 | kept/GT¹ |
|---|---|---|---|
| E v0 (pooled, unscaled) | 0.179 | 0.072 | 0.83× |
| E v1 pooled + Fourier fix | 0.156 | 0.086 | 0.59× |
| E v1 hybrid (learned content) | 0.207 | **0.121** | 0.65× |
| E v1 positional-only | **0.230** | 0.099 | 0.86× |
| **arm C control (the bar)** | **0.269** | **0.144** | 1.23× |

<span class="small">¹ honest kept predictions vs GT instances, 10 val scenes, best-mIoU checkpoint, same protocol for all rows.</span>

- **Fourier fix alone changes nothing** → the pooled features were **actively harmful**, not just badly encoded.
- Learned content on anchor geometry: best E-family **AP50** (+0.05 over v0). Removing content: best E-family **mIoU** — 0.230, within 0.04 of arm C, from **pure geometry**.
- Positional-only is the **least overfit run of any arm** (train 0.397 @ep1000, val still ~0.22 — arm C: train 0.73, val decayed to 0.17).

---

<!-- _class: compact -->

# 5 · What arm E means for the thesis

**Per the decision rule (scale only on a win): arm E is closed — arm C stays the base.**

The chapter it buys is stronger than a marginal win would have been:

1. **Failure decomposition** — a 3-run ablation cleanly attributes v0's failure: content ≫ encoding. "Frozen backbone features make bad query content" is a transferable negative result (relevant to every pooled-feature query-init in the literature).
2. **Calibration finding** — geometry-spread queries are the *only* lever (of 4 tried: threshold, grid density, no-object weight pending, anchors) that fixed over-prediction: every E variant ≤ 0.86× kept/GT vs C's 1.23×. Duplicate suppression via 3D spread **works**; at N=50 it just costs more detection quality than it returns.
3. **Regularization signal** — pure-geometry queries barely overfit while reaching 85% of arm C's mIoU. If we ever unfreeze/scale, E-style inits may re-enter.

Query-strategy study now complete: **4 families, 8 trained variants, one winner (C), all closures explained.**

---

<!-- _class: compact -->

# 6 · The data-scaling answer (490-scene run, finished today)

**Question**: does arm C keep scaling past N=190 with 2.6× more (official-GT) scenes?
**Answer: no.** Job 7219652 (N=490, `--bundles_per_scene 1` to fit the node — see below): best val mIoU **0.350** @ep150 / AP50 **0.177** @ep100, vs the N=190 baseline **0.367 / 0.199** — and it overfits *faster* (peak @ep100–150 vs @ep450–500).

Caveats: bundles 1 vs 3 is a real recipe deviation; scenes 0200–0499 unvetted. But direction is clear — **data quantity is not the lever** (mirrors the arm-A plateau at N=50): the ceiling is protocol/capacity.

<div class="small">

**Infra finding worth 30 s**: two earlier attempts at this run stalled 15×+ on *different* nodes. Diagnosis: not node luck, not a code bug — our ~250 GB bundle cache on 16-socket/24 GB-per-NUMA-domain nodes goes memory-remote as it grows (same scenes fast early, slow late). Fix: cut the resident cache below ~120 GB (`bundles_per_scene 1`) → ran clean at full speed. Relevant to any future big-cache run.

</div>

---

<!-- _class: compact -->

# 7 · Proposed next steps (measurement & protocol, not new arms)

1. **Cross-view consistency metric** (eval-only, CPU) — our decoder is consistent *by construction* (one query = one instance in all views) vs the fuse-2D paradigm (PanSt3R etc.). Turn the claim into a number (ID-switch rate / cross-view mask agreement) on existing checkpoints. **Cheap, differentiating, thesis-ready.**
2. **Protocol alignment with SegVGGT** (needs a decision): (a) official-split-∩-our-500 scene split vs our contiguous scaling-curve split; (b) per-iteration random frame sampling vs our cached bundles (the caching is why training takes minutes — a deliberate tradeoff to defend or relax).
3. **Which-layer ablation** — we hook aggregator layer [-1]; sweep every 4th layer on arm C @ N=50. Nearly free with feature caching; good analysis chapter either way.
4. Remaining Phase-6 ablations: no-object-weight sweep (0.05/0.1/0.4), augmentation ablation.
5. **Read SegVGGT line-by-line** (eval protocol above all — their 3D point-cloud masks vs our per-view 2D patch masks are NOT comparable numbers) before writing any comparison.

---

<!-- _class: lead -->

# Summary

**The query-strategy ablation — the thesis's experimental core — is complete.**
Arm C (learned queries): **0.367 val mIoU / 0.199 honest AP50** on clean official GT, N=190.
Arm E closed with a clean decomposition: *pooled frozen features hurt; geometry alone regularizes and calibrates (≤0.86× vs 1.23× over-prediction) but caps detection.*
More data doesn't help (N=490 ≤ N=190) → next: consistency metric, protocol alignment, layer sweep.

<span class="small">Docs: `ARMS_SUMMARY.md` (all arms, one table) · `MILESTONES.md` (full narrative) · `todo.md` (open items)</span>
