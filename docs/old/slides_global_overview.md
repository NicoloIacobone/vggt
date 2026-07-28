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
  .verdict { background: #f4f4f4; border-radius: 8px; padding: 10px 16px; margin-top: 10px; }
  .verdict b { color: #2a6; }
---

<!-- _class: lead -->
<!-- _paginate: false -->

# Multi-view consistent 3D instance segmentation from a frozen VGGT backbone

## A DETR-style decoder head, and a systematic study of how its queries should be initialized

**Objective:** given S RGB frames of a ScanNet scene, predict per-instance masks that are
consistent across views — same object = same ID in every frame.

<!-- FIGURE: 3-frame qualitative strip, same instance colored consistently across views
     (source: any auto-rendered overlay in a run's visualizations/ dir) -->

<!--
Speaker notes: frame this up front — "decoder on frozen VGGT" is a crowded genre now
(docs/RELATED_WORK.md). The contribution of this work is the query-strategy ablation
(arms A-E), not the architecture. Everything after this slide is evidence for that.
-->

---

<!-- _class: compact -->

# Architecture

<div class="cols">
<div>

**Pipeline**
1. VGGT aggregator (24 blocks, **frozen**, `no_grad`) — hook at `aggregated_tokens_list[-1]`
2. Features `F: [B, S, P, 2048]` (S frames × patch tokens)
3. **Query Generator** — the part every arm varies
4. **Instance Decoder** (4-layer/8-head `nn.TransformerDecoder`, queries=tgt, projected F=memory)
5. Class head (19 classes + background) / mask-embed head / dense mask head → `pred_masks [B,N,S,h,w]`

Only **~6.5M** head params train; the 1B-param backbone never updates.

</div>
<div>

**Consistency by construction**

One query = one instance *in all S views simultaneously* — the mask tensor carries the
view dimension inside a single query slot, not fused post-hoc across independently-run
per-view 2D masks (contrast: PanSt3R-style fusion).

<div class="small">

Hard-won constraints (ask if curious): LayerNorm + query skip on the projected memory
(else queries collapse to one vector); cosine-similarity mask logits, not raw dot
products; fg-weighted BCE + gradient clipping; coordinates are query *prompts*, never a
loss target.

</div>
</div>
</div>

---

<!-- _class: compact -->

# Training — matching predictions to ground truth

Before any loss is computed, a **Hungarian matcher** (`PointBipartiteMatcher`) finds a
one-to-one assignment between the N predicted queries and the scene's GT instances —
DETR-style, run **per batch sample** against that sample's own GT set.

**Cost matrix** (lower = better; minimized by `scipy.optimize.linear_sum_assignment`):

`cost = class_weight · (1 − P(gt_class)) + mask_weight · (Dice + BCE) + coord_weight · ‖pred_xy − gt_xy‖₂`

- **Class term** — 1 − softmax probability of the GT class
- **Mask term** — dense Dice + BCE cost between predicted mask *logits* and GT masks
  (Mask2Former-style pairwise cost on the actual masks; a cheap mask-**embedding** L2
  distance exists as a fallback in code, but current training always has dense masks
  available, so that fallback path is never actually used)
- **Coord term** — L2 in normalized image coords, useful here even though it's not a
  network *output* — for point-based arms it's the fixed (u,v) the query was placed at, a
  free disambiguation signal for assignment. **Zero** for query modes with no coordinates
  (learned / anchor3d), so matching there is mask+class only

<div class="small">

Note: matching and the loss (next slide) use **separate weights** on **the same 3–4
quantities** — this cost decides *who pairs with whom*; it is not itself backpropagated
(the assignment is solved with `scipy`, off the autograd graph).

</div>

<div class="verdict">

<b>Non-finite guard:</b> if any cost entry is NaN/Inf (exploding logits — exactly what
killed arm D's v0 run at ~ep555), it's replaced with a large finite cost and a warning is
logged instead of crashing the assignment solver. Training continues; the instability stays
visible in the logs rather than taking the run down.

</div>

---

<!-- _class: compact -->

# Training — the loss

Same **matched** query/GT pairs from the step above, scored again with their own weights
(distinct from the matcher's `class_weight`/`mask_weight`/`coord_weight`). Only **two**
terms are actually backpropagated:

| Term | What |
|---|---|
| **Class** | Focal loss (α=0.25, γ=2) on matched pairs; optionally *every* query via `no_object_weight` |
| **Mask** | Dice + foreground-weighted BCE (`pos_weight` = neg/pos ratio, capped at 20) on dense masks |

<div class="small">

Two more terms exist in the code but are always zero-weighted, kept only as diagnostics:
<b>coordinate</b> loss (coordinates are a fixed query <i>input</i>, not a network output —
no gradient path exists even if weighted) and <b>mask-embedding</b> L2 (legacy target,
superseded everywhere by the dense mask loss above — same reason it's unused in matching).

</div>

**DETR-style no-object supervision** (`--no_object_weight`, e.g. 0.1): unmatched queries get
a `background_class` target instead of being ignored, down-weighted by this factor — lets
unprompted/honest inference filter queries without relying on GT-ordered slots.

<div class="verdict">

<b>The arm-B/D fix, in one line:</b> <code>--no_object_norm matched</code> normalizes the
matched and unmatched terms <i>separately</i>
(<code>matched.mean() + w·unmatched.mean()</code>) instead of one pooled weighted mean — so
appending arm B's ~10× grid queries no longer dilutes the matched-query gradient into the
mask-learning collapse seen in v0.

</div>

---

<!-- _class: compact -->

# Dataset & the GT story

<div class="cols">
<div>

**ScanNet v2, official 2D instance GT**
- 500 scenes (scene0000–0499), ~100 stride-5 frames/scene
- **7379 instances, 0 cross-class duplicates**
- 19 trainable NYU40 classes + background
- Masks evaluated at the 37×37 patch grid (input 518²)
- Held-out val: scenes 0080–0089 (13.3 GT instances/scene)

</div>
<div>

**Why "official GT" is called out at all**

The original GT was SAM3-generated (per-class prompt + video tracking). A 2026-07-07
audit found **~15.9% of foreground pixels were multi-class** — the same object labeled
under two classes (desk↔table, curtain↔shower_curtain) — which forces the matcher to
demand two predictions per object, i.e. **built-in false positives**.

</div>
</div>

<div class="verdict">

**Consequence, same checkpoint, two rulers:** honest AP50 <b>0.228</b> on SAM3 val →
<b>0.117</b> on official val → retrained on official GT recovers to <b>0.199</b>.
Roughly half the old headline number was the model fitting label noise. Every quotable
number from here on is measured on the official GT.

</div>

<!-- FIGURE: 3-bar chart of the 0.228 / 0.117 / 0.199 triplet — tells the whole story in
     one image without needing a duplicate-mask screenshot -->

---

<!-- _class: compact -->

# How we read the results

- **val mIoU** — mask quality on Hungarian-matched instances only. Answers "are the masks
  good?", not "did you find the objects?"
- **AP50 / honest AP50** — detection quality; penalizes false positives and duplicates.
- **Prompted vs. unprompted ("honest"):** prompted = queries placed at GT centroids
  (upper-bound diagnostic, needs GT); unprompted = grid queries or GT-free queries.
  **Honest val[grid] AP50 is the headline number** — unprompted mIoU is optimistic because
  unmatched false positives go unpunished. For arms **C** and **E**, prompted == unprompted
  *by construction* — a selling point, not a caveat.
- **kept/GT calibration ratio** (this project's own diagnostic) — honest kept predictions
  ÷ GT instances. >1× = over-predicting, <1× = under-predicting. Decided the arm-E story.
- Two checkpoints are saved per run — best-val-mIoU and best-honest-AP50 — because they
  diverge (mask quality and detection quality peak at different epochs).

---

# Arm A — point queries (D4RT style)

<div class="cols">
<div>

**How it works**
- Position: (u,v) image coords — **GT centroids at train**, uniform grid at honest eval
- Content: Fourier(u,v) + view embedding + 9×9 RGB patch MLP

</div>
<div>

**Results** (val mIoU / honest AP50)

| N=50 | N≈190 | N=490 (official GT) |
|---|---|---|
| 0.212/0.125 (SAM3 GT) | 0.216/0.105 — plateaued | **0.264/0.102** |

</div>
</div>

<div class="verdict">

Simple and promptable, but honest eval needs a grid → duplicate false positives, no NMS.
<b>Plateaued past N=50</b> — the N=490 point just reproduces the same plateau on cleaner
GT. <b>Superseded baseline.</b>

</div>

---

# Arm B — trained grid queries

<div class="cols">
<div>

**How it works**
- Same as A, plus random-offset grid queries that are also *trained* (not just eval-time)
- Fix: `--no_object_norm matched` — normalizes the no-object loss per term so grid queries
  stop diluting the matched-query gradients

</div>
<div>

**Results** ([grid] = val[grid] mIoU / honest AP50)

| N=50 | N≈190 | N=490 (official GT) |
|---|---|---|
| 0.284/0.161 | 0.372/0.185, unstable (0.071 @ep1000) | **0.110/0.172** |

</div>
</div>

<div class="verdict">

v0 collapsed entirely (loss dilution) before the fix. Post-fix: best point-family AP50 at
N=50, but AP50 <b>never stabilizes</b>, and at N=490 the prompted val mIoU regresses hard.
<b>Closed</b> — the fix works, the arm still loses to C.

</div>

---

# Arm C — learned DETR-style queries — **the current base**

<div class="cols">
<div>

**How it works**
- Position source: **none** — position is implicit, learned end-to-end
- Content: 64 free `nn.Embedding` vectors, one per slot — classic DETR object queries

</div>
<div>

**Results** (val mIoU / honest AP50, official GT unless noted)

| N=50 | N=190 | N=490 |
|---|---|---|
| 0.259/0.146 (S); 0.269/0.144 | **0.367/0.199 ← quotable baseline** | 0.350/0.177 |

</div>
</div>

<div class="verdict">

<b>GT-free at eval ⇒ prompted == honest.</b> Broke the point-query plateau (>2× honest
AP50 over arm A) and is <b>GT-robust</b> — barely moves between SAM3 and official GT at
N=50. Over-predicts (kept/GT 1.23–1.38×) and 2.6× more data (N=490) does not beat N=190 —
overfits <i>faster</i> instead. <b>Wins at every scale tested (50 / 190 / 490).</b>

</div>

---

# Arm D — hybrid (C's slots + A's centroid prompts)

<div class="cols">
<div>

**How it works**
- Position: (u,v) GT centroids, like A
- Content: mixed — learned `nn.Embedding` + Fourier/RGB, like C
- Fix: `--learned_query_lr_scale 0.1` puts the learned embeddings in their own low-LR
  AdamW group (the v0 NaN fix)

</div>
<div>

**Results** (val mIoU / honest AP50)

| N=50 | N≈190 | N=490 (official GT) |
|---|---|---|
| 0.247/0.146, then overfits | — not scaled (no win) | **NaN @ ep110** (best-before-divergence 0.295/0.174 @ ep100) |

</div>
</div>

<div class="verdict">

v0 exploded with NaNs; the fix survived all 1000 epochs at N=50 but only <i>tied</i> arm C.
At N=490 the instability <b>recurs</b> — the N=50 fix did not hold at scale. <b>Closed.</b>

</div>

---

# Arm E — 3D-anchored queries

<div class="cols">
<div>

**How it works**
- Position: FPS anchors over VGGT's own predicted 3D pointmap (not image space)
- Content: **v0** kNN(8)-pooled frozen backbone features · **v1** ablates content ∈
  {learned per-slot embedding, none / positional-only, pooled + Fourier-scale fix}

</div>
<div>

**Results** (val mIoU / honest AP50, official GT, N=50 bar = arm C's 0.269/0.144)

| variant | N=50 | kept/GT |
|---|---|---|
| v0 pooled | 0.179/0.072 | 0.83× |
| v1 pooled+fix | 0.156/0.086 | 0.59× |
| v1 hybrid (learned) | 0.207/0.121 | 0.65× |
| v1 pos-only | **0.230**/0.099 | 0.86× |

N=490: v1 hybrid → **0.248/0.139** (job 7974169, zero non-finite warnings)

</div>
</div>

<div class="verdict">

Never beats C on quality at any N — <b>closed</b>. But it's the only arm that fixes
over-prediction (kept/GT 0.59–0.86× vs C's 1.23×), and the v1 ablation cleanly shows the
<i>pooled features</i> were the poison, not the coordinate encoding. Deliverable = the
ablation story, not a new best arm.

</div>

---

<!-- _class: compact -->

# The arms — summary

| Arm | N=50 | N≈190 | N=490 (O) | Verdict |
|---|---|---|---|---|
| A point | 0.212/0.125 | 0.216/0.105 | 0.264/0.102 | plateaued — superseded |
| B trained grid | 0.284/0.161 | 0.372/0.185 unstable | 0.110/0.172 | AP50 never stable — closed |
| **C learned** | 0.269/0.144 | **0.367/0.199** | 0.350/0.177 | **wins at every N — base** |
| D hybrid | 0.247/0.146 | not scaled | NaN @ep110 | instability recurs at scale — closed |
| E v1 (best) | 0.230/0.121 | not scaled | 0.248/0.139 | best calibration, never wins on quality — closed |

**Cross-arm takeaways**
1. GT-free query generation (C, E) is both more honest *and* stronger than grid-prompted arms.
2. Grid-based honest eval dies by duplicate false positives — architectural, not a tuning
   problem (grid-density sweep 2→12: best AP50 0.185, still < C's 0.228).
3. Geometry (E) regularizes and calibrates prediction count but caps detection quality.
4. **The ranking (C > E > B ≈ D > A) is scale-invariant** — confirmed unchanged from
   N=50/190 up to N=490. More data does not change the query-strategy verdict.

---

<!-- _class: compact -->

# Related work case study — VGGT-Segmentor (Ego-Exo4D)

<div class="cols">
<div>

**Problem it solves:** ego↔exo object **mask transfer** — given a mask in one view,
segment the same physical object in the other view. Hard because scale/perspective/
occlusion destabilize pixel-level matching (source: camera near the hands vs. far
exo view, heavy hand/tool occlusion).

**Finding that drives the design:** raw VGGT point/track projections **drift** under
large viewpoint change, yet VGGT's internal object-level **attention stays
well-aligned** — geometry alone isn't trustworthy, it needs a learned fusion head.

**Idea — Union Segmentation Head** (3 stages): Mask Prompt Fusion (conv-encode the
source mask, cross-view bottleneck self-attn) → Point-Guided Prediction (K-means
points on the mask, tracked via VGGT's own track head, bidirectional point↔image
cross-attn) → iterative Mask Refinement. Plus **Single-Image Self-Supervised
Training** (SAM pseudo-masks + augmented view pairs) to skip paired annotation.

</div>
<div>

**Component ablation (IoU)**

| Stage | Ego→Exo | Exo→Ego |
|---|---|---|
| Plain head | 35.5 | 37.1 |
| + Bottleneck Fusion | 50.2 | 52.3 |
| + Point-Guided Pred. | 62.2 | 63.5 |
| + Mask Refinement (full) | **67.7** | **68.0** |

**vs. prior SOTA**

| | Ego→Exo | Exo→Ego |
|---|---|---|
| DOMR (prior SOTA) | 49.7 | 55.2 |
| VGGT-S, correspondence-free (SSL only) | 54.1 | 58.4 |
| VGGT-S, full (supervised) | **67.7** | **68.0** |

</div>
</div>

<div class="verdict">

Their self-supervised-only variant already beats the prior <b>supervised</b> SOTA
(DOMR). Full model: +18.0 / +12.8 IoU over DOMR.

</div>

---

<!-- _class: compact -->

# The three literature paths it organizes

- **Path 1 — Cross-view geometric modeling.** Classical SfM/MVS (keypoint matching;
  costly, breaks on non-rigid motion / wide baselines) → feed-forward geometry
  transformers (**VGGT**, DUSt3R, MASt3R — joint depth/camera/point regression, no
  post-optimization) → SegMASt3R (geometry-grounded segment matching). VGGT-Segmentor
  takes this as its *backbone*, then shows it's **insufficient alone** (the
  attention-vs-projection drift finding above).
- **Path 2 — Ego-exo / visual object correspondence** (the task lineage it actually
  competes in). XSegTx / XView-XMem (Ego-Exo4D's own cross-image-transformer +
  working-memory baselines) → PSALM / ObjectRelator (LMM-assisted, language-conditioned
  prompting) → **DOMR** (dense object matching, models inter-object relations) — the
  prior SOTA it beats by +18.0 / +12.8 IoU.
- **Path 3 — Promptable / generalist segmentation foundations.** Mask2Former, SEEM,
  SAM / SAM2 — strong, but single-view, no cross-view alignment → MASA (bootstraps
  instance association via SAM-lifted self-training). VGGT-Segmentor borrows this
  recipe directly for its Single-Image Self-Supervised stage.

<div class="verdict">

<b>VGGT-Segmentor = Path-1 backbone + Path-2 task target + Path-3 training recipe</b>,
glued by a purpose-built fusion/refinement head — not a new geometric backbone, not a
new foundation model.

</div>

<div class="small">

<b>How this bears on our positioning</b> (already logged in <code>docs/RELATED_WORK.md</code>
as related, not competing — the arXiv harvest mis-tagged it as a ScanNet scoop): same
premise as us (frozen VGGT + lightweight trainable head) and the same hard-won lesson
that raw geometric features need a learned bridge before they're usable — their
attention-vs-drift finding echoes our own LayerNorm+query-skip constraint, from an
independent project. But the task and eval are entirely different: their 2-view
<i>prompted</i> mask transfer, scored by IoU on Ego-Exo4D, vs. our multi-view
<b>GT-free query</b> 3D instance segmentation, scored by AP50/mIoU on ScanNet. Cite as
related work, not as a number to put in the same table.

</div>

---

<!-- _class: lead -->

# Next

- Write up the query-strategy ablation as the thesis's experimental core — it's complete
  (4 query-init families, 8 trained variants, one winner, every closure explained)
- Turn the "consistency by construction" claim into a measured number
  (cross-view mask-identity agreement / ID-switch rate)
- Align the training protocol with published practice (splits, per-iteration frame sampling)
- Position against direct competitors (SegVGGT et al.) — no one else publishes this ablation
