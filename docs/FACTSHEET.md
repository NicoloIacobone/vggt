# FACTSHEET — the only page you need to quote from

> **Purpose.** This is the *single* source for outward-facing material: slides, supervisor
> updates, abstract drafts. It is a read-out of `docs/RESULTS.md` §8, `docs/RELATED_WORK.md`
> and `docs/TRAINING_COMPARABILITY.md` §6 — **nothing here is new**, and nothing here may
> contradict them.
>
> **Two rules for anyone (human or agent) building slides from this file.**
> **(1) Every number on a slide must appear verbatim on this page.** If a number you want is not
> here, it is not cleared for quoting — stop and ask, do not go looking for it in the other docs
> and do not compute it.
> **(2) The main deck is built from Tier 1 only** (§1): the 3D rulers that face the published
> competitors. The 2D rulers and the COCO check are **Tier 2 — backup slides**, kept because they
> answer specific questions (§9), never placed beside a competitor number.
>
> Frozen 2026-08-26. Sources: `docs/RESULTS.md` (numbers), `docs/RELATED_WORK.md` (positioning),
> `docs/TRAINING_COMPARABILITY.md` (what is matched), `docs/MASKDINO.md` (mechanism).

---

## 0. The project in five sentences

We attach a **MaskDINO-family decoder** to a **strictly frozen VGGT-1B** backbone and train it for
**3D multi-view-consistent instance segmentation**, supervised by official ScanNet v2 2D instance
annotations. The backbone is never updated — its features are cached once per scene, so head-only
training takes **~0.8 GPU-days**, against the ~16 GPU-days of the closest competitor. One query is
one instance **across all views by construction** (`pred_masks [B, N, S, h, w]`), so multi-view
consistency is intrinsic, not obtained by post-hoc fusion or mask matching. The 2D masks are lifted
to the scene point cloud and scored with the **official, vendored ScanNet 3D instance evaluator**.
The contribution is the **controlled study** — one backbone, one dataset, one protocol, decoder
ingredients varied one at a time — not any single mechanism (§5). **The result that faces the field
is the 3D one (§2); the 2D numbers that built the project are internal evidence (§9).**

---

## 1. THE RULE THAT BREAKS SLIDES — which rulers face the field, and which do not

This project measures on **seven rulers in two tiers**. Numbers may be compared *inside* a tier and
**never across tiers**. Mixing them is the single most likely error in any presentation.

### Tier 1 — the rulers that face the competitors (§2, §3). **These are the paper.**

| id | ruler | what it scores | comparable to a paper? |
|---|---|---|---|
| **D** | **3D official benchmark, UNPOSED** — our own predicted depth + cameras | 3D masks on the benchmark point clouds, official vendored evaluator | **YES — this is the headline** |
| **E** | **3D official benchmark, POSED** — GT poses + sensor depth | as D, but the 2D↔3D bridge is exact by construction | **YES** |
| **G** | 3D on ScanNet200 / ScanNet++ / Replica | same evaluator, class-agnostic only | same protocol, but **no like-for-like published row is held** |

### Tier 2 — internal rulers: model selection and ablation only (§9). **Never next to a paper.**

| id | ruler | what it is for | comparable to a paper? |
|---|---|---|---|
| **A** | 2D single-frame, project val (scenes 0080–0089) | the scaling curve 50 → 190 → 490; "is the direction right" | **NO** — our own metric code, 37×37 patch grid |
| **B** | 2D per-bundle, project val | the multi-view ablations (cross-frame attention, bundle features) | **NO** |
| **C** | 2D, official 1201/312 split | picks which checkpoint goes to the 3D ruler; carries the **data-scaling arms** | **NO** (official split, but our metric code) |
| **F** | COCO val2017 | **port-correctness check.** Not a result of any kind | **NO — not a project result** |

**Why Tier 2 still exists and must not be thrown away.** A/B/C are not superseded results, they are
*different questions*: C is what selects the checkpoint the headline is measured on and what prices
the extra-data row; B holds the only measurement of what cross-frame attention is worth; F is the
answer to *"how do we know your MaskDINO implementation is not buggy?"* — the one hostile question
that cannot be improvised. Keep them as **backup slides**, not main ones.

### Two axes cut across Tier 1 and must be stated with every number

- **class-aware (18 classes, per-class mean) vs class-agnostic (labels ignored).** FAST3DIS and
  IGGT publish *only* class-agnostic; SegVGGT is class-aware. Both are computed for every run.
- **posed vs unposed bridge.** Unposed = 2D mask quality **×** feed-forward geometry quality.
  Posed = 2D mask quality **alone**. The bridge is worth a consistent **2.3× in AP50**.

AP triples are always written **`AP / AP50 / AP25`**. In this literature **`mAP` ≡ `AP`** — the
naming difference between the SegVGGT and FAST3DIS tables means nothing; the *setting* does.

## 2. THE HEADLINE — one claim, one number

> **On the official ScanNet 3D instance benchmark, unposed and class-agnostic — the like-for-like
> setting of the two published feed-forward competitors — we lead on AP50 and AP25, TIE FAST3DIS on
> AP, and lead IGGT on all three — with a strictly frozen backbone, no adaptation, ~1/3 of their
> views, and every lifting knob at its default.**

> ⚠ **"Ahead on all three" belongs to the EXTRA-DATA row only** (A-long, 1.5–2.0× on all three
> against both). On the ScanNet-only headline row the AP column is a **tie with FAST3DIS** across
> two seeds — 0.039–0.042 against 0.038, inside our own 0.003 seed spread on that column. The
> "ahead on all three" wording was a seed-0-only reading of that row and must not be repeated.

| Method | Backbone | Views/scene | AP | AP50 | AP25 |
|---|---|---|---|---|---|
| IGGT *(as re-evaluated by FAST3DIS)* | adapted | 50 | 0.028 | 0.112 | 0.287 |
| FAST3DIS | LoRA-adapted DA3 | 50 | 0.038 | 0.096 | 0.316 |
| **Ours, `--anchor_3d`, defaults — ScanNet-only training** | **frozen VGGT-1B** | **~17** | **0.042** | **0.138** | **0.504** |
| — seed 1 (replication) | 〃 | 〃 | 0.039 | 0.129 | 0.485 |
| — best lifting knobs *(sensitivity, not headline)* | 〃 | 〃 | 0.055 | 0.185 | 0.571 |
| **Ours + EXTRA TRAINING DATA** (A-long: ScanNet + ScanNet++ + Infinigen, 3520 scenes) | 〃 | 〃 | **0.057** | **0.166** | **0.516** |

Ratios vs FAST3DIS across the two seeds: **AP50 1.34–1.44×**, **AP25 1.53–1.59×**, **AP tie**.
Lead over IGGT on all three. Survives the whole lifting-knob grid — the *worst* point of the sweep
is still 1.44× FAST3DIS's AP50.

**The extra-data row stays separate and the ScanNet-only row stays the headline.** Field norm (and
MaskDINO's own README) fences extra-data rows. The *mechanism* claim rests on the ScanNet-only row;
the *scaling* claim on the extra-data one.

### 2.1 The four caveats that must travel with the headline

1. **Two seeds; one run per cell.** Both arms were replicated at `--seed 1` on 2026-08-07, so the
   old "single run against a single control" caveat is **retired**. What remains: one run per cell
   in the cross-dataset matrix, and **one published row per competitor**.
2. **Not training-matched, and it favours us.** Both published rows are **zero-shot on ScanNet**
   (FAST3DIS trains only on Aria/ASE; IGGT only on InsScene-15K); every row of ours trains on
   ScanNet. Protocol-matched and setting-matched, **not** training-matched. Closing: arms **I** /
   **I-gt** (§6).
3. **The class-collapse sign is checkpoint-dependent.** The same recipe *without* `--anchor_3d`
   scores 0.017 / 0.060 / 0.334 class-agnostic **with its lifting knobs tuned** (0.013 / 0.050 /
   0.320 at defaults) — ahead on AP25, ~2× behind on AP50/AP. **Carry the "tuned" label**: the
   headline row is untuned, so this is not a defaults-vs-defaults comparison.
4. **Two asymmetries run the other way** and are already stated: the frozen backbone, and ~17 views
   to their 50 (closing separately via the dense frame export).

---

## 3. THE COMPETITOR-FACING NUMBERS (Tier 1)

*The internal 2D rulers and the COCO check are in §9, and are backup material.*

### 3.1 Rulers D / E — the two bridges on the same masks (18-class, class-aware)

| Checkpoint | UNPOSED (own geometry) | POSED (GT bridge) | bridge cost |
|---|---|---|---|
| multi-frame, official split — the §5 control, **not** §2's headline row | 0.023 / 0.067 / 0.268 | 0.060 / 0.156 / 0.408 | **2.3× AP50** |
| `--anchor_3d` | 0.038 / 0.112 / 0.360 | 0.104 / 0.257 / 0.504 | 2.3× |
| **S=16, 20 epochs** | 0.032 / 0.115 / 0.414 | **0.088 / 0.260 / 0.572** | 2.3× |
| S=16 + `--anchor_3d` | 0.032 / 0.109 / 0.353 | 0.082 / 0.236 / 0.501 | 2.2× |
| *oracle — GT through the posed bridge* | — | *0.828 / 0.948 / 0.974* | ceiling of a ~17-frame budget |
| **SegVGGT (published, posed)** | — | **0.504 / 0.717 / 0.870** | — |

**Print both bridges or neither.** Of the gap to SegVGGT, a factor **2.3 is the bridge** and a
factor **~4.6 is real**: LoRA-adapted backbone vs frozen, 75–100 views vs 17, 259×196 masks vs
37×37, 600 kept queries vs 100.

⚠ **The ×10.7 → ×2.3 → ×4.6 decomposition is anchored to the CONTROL row** (0.067 unposed →
0.156 posed, against SegVGGT's 0.717) — that is the row the analysis was run on. The bolded rows of
this table are better than the control, so on them the residual is *smaller* than ~4.6×; no figure
for that is cleared. **Quote the decomposition with its row named, and never recompute it against a
different row of this table.**

**The answer to "why is your AP low":** the image-only baseline in SegVGGT's *own* table,
OneFormer3D†, scores **5.4 / 10.2 / 17.4**. Keep this slide-ready.

### 3.2 Ruler G — the other three benchmarks (class-agnostic, posed, `--anchor_3d`)

**ScanNet-only checkpoint** — the fenced row, matching §2's headline: ScanNet200
**0.124 / 0.275 / 0.523** · ScanNet++ **0.009 / 0.038 / 0.178** · Replica **0.006 / 0.028 / 0.190**.

**With EXTRA TRAINING DATA (A-long)**, same ruler, same bridge, fenced exactly like §2's extra-data
row: ScanNet200 **0.132 / 0.287 / 0.539** · ScanNet++ **0.019 / 0.068 / 0.275** · Replica
**0.040 / 0.119 / 0.480**. This is where the extra data pays most — the only measurement in the
project of something the ScanNet ruler cannot see.

**Zero-shot dies under the unposed bridge** — every out-of-domain unposed cell is 0.000 AP /
0.000–0.001 AP50 across all four data arms — and survives weakly under the posed one, which
localises the failure to **geometry, not masks**. ScanNet200 costs zero extra data. No like-for-like
published row is held for any of these.

## 4. WHAT ACTUALLY BUYS THE RESULT — ranked

Read every Δ against the **measured seed spread of 0.009 per-bundle AP50**.

| Lever | Effect | Size | Measured on |
|---|---|---|---|
| **Training data, 50 → 490 scenes** | +0.26 per-frame AP50 | dominates everything | Tier 2 (A) |
| **Cross-frame attention** | +0.183 per-bundle AP50 | 20× seed noise | **Tier 2 only (B)** ⚠ |
| **Bundle features** | +0.147 per-bundle AP50 (−0.048 per-frame) | 16× | **Tier 2 only (B)** ⚠ |
| **`--anchor_3d`** | +66 % 3D AP50 in *both* bridges; `id_switch` −0.089 | AP-neutral in 2D (inside noise) | **Tier 1 (D + E)** |
| **Bundle width 8 → 16** | +0.027 per-bundle AP50; `id_switch` −0.113; +46 % unposed 3D AP50 | 3× | **Tier 1 (D)** |
| Lifting knobs (`--vote_radius`, depth conf.) | +0.016 → +0.047 3D AP50 | larger than most decoder ablations | **Tier 1 (D)** |
| Any single decoder component (two-stage, encoder, DN, box init) | ≤0.046 per-frame AP50 | at N=190 | Tier 2 (A) |
| Mask resolution 37² → 74² | −0.022 (neutral) | **not the bottleneck** | Tier 2 (A) |

> ⚠ **The hole in the ablation table — take this to the supervisors.** The two levers marked
> Tier-2-only are the *most* decisive components measured anywhere in this track (+0.183 and
> +0.147 bundle AP50, 20× and 16× the seed spread) — and **neither has a Tier-1 counterpart**.
> `--anchor_3d`, bundle width and the lifting knobs all have 3D numbers; cross-frame attention and
> bundle features do not. A 3D-ruler ablation of `--no-cross_frame_attn` / `--feature_mode single`
> is therefore the cheapest way to make the paper's ablation table land on the same ruler as its
> headline — but **the two halves do not cost the same**, and the difference is a decision the
> supervisors have to take. Prices in §6.5.

### The four conclusions

1. **Data-limited, not architecture-limited.** The leak-free 1201-scene checkpoint even beats the
   one that had *seen* the val scenes (0.083 vs 0.052 AP50) — data beats leakage.
2. **Recognition and cross-view identity are separate axes.** `bundle_AP50` alone cannot see what
   `--anchor_3d` does; score identity mechanisms on the 3D ruler, never on `bundle_AP50` alone.
3. **The lifting step, not the decoder, now binds** (AP25 ≈ 4× AP50). Masks are not the constraint;
   the 2D→3D bridge is.
4. *(Corollary, and the strongest ablation-level finding)* Mask **resolution is not the ceiling**:
   the 37×37 grid's GT-only ceiling is **0.956 AP50** against the model's ~0.69. **Recognition
   binds.**

---

## 5. POSITIONING — what is ours and what is not

**"Frozen VGGT + a decoder for a downstream 3D task" is the dominant pattern of the last ~12 months
(the "VGGT-X" genre). The architecture alone is not a contribution.** Say this before claiming
anything.

| Mechanism | Already owned by | What is left for us |
|---|---|---|
| Object queries shared across views on a VGGT-family backbone | **SegVGGT** (400 queries in all 24 aggregator layers) | our `--multi_frame` is the same *class* of idea — report it as a controlled comparison against our own single-frame model, **not** as a new mechanism |
| **3D-anchored queries** (learned 3D anchors + project-and-sample cross-attention) | **FAST3DIS** | our `--anchor_3d` is an **ablation, not a contribution**: 3D anchors vs 2D DAB boxes inside the same decoder, same frozen backbone, same data, same protocol — **which nobody has run** |
| Attention-dispersion fix over many global tokens | SegVGGT (FADA, train-time only) | we get part of it free from deformable attention (4 sampled points/level vs dense) — one sentence of contrast |
| Single-pass multi-view panoptic prediction | PanSt3R | the "why not splat / why not fuse" comparison point |
| Anti-duplicate query regularisation | FAST3DIS (contrastive + overlap penalty) | we use DINO's one-to-one Hungarian matching + DN instead — a legitimate discussion point |

### What IS ours — the three defensible claims

1. **The controlled comparison nobody has run**: one backbone, one dataset, one protocol, decoder
   ingredients varied one at a time — including 3D-vs-2D anchors *inside the same decoder*.
2. **Competitive 3D results from a strictly frozen backbone** at ~0.8 GPU-days against ~16, with no
   adaptation of any kind. Everyone else LoRA-adapts.
3. **Consistency intrinsic to the query, not post-hoc** — and now *measured*:
   `bundle_view_consistency` **0.734** / `bundle_id_switch` **0.414** at best (A-long).

### "Why not just splat?" — keep ready for reviewers

Gaussian-Splatting / NeRF instance methods get multi-view consistency by construction but need
**per-scene optimisation**. Ours is feed-forward, optimisation-free, needs no GT geometry or depth
sensor, and runs in **seconds, not minutes**.

---

## 6. WHAT IS STILL OPEN — the ablation menu for the supervisors

### 6.1 In flight right now (jobs launched, results pending)

| axis | what it settles | job |
|---|---|---|
| **Zero-shot arms I / I-gt** | trains on IGGT's mixture **minus ASE**, never on ScanNet → makes the competitor comparison **training-matched**. Completes a {±ScanNet} × {±RE10K} 2×2 | 11839134 / 11839135 |
| **C-long′ ⇄ A-long′** | separates **more data** from **more compute** at the top end (the one unresolved caveat of §9.3); re-run at lr 5e-5 after C-long destabilised | 11831105 / 11830142 |
| **D-long** (RE10K, **SAM2-supervised**) | whether a fourth, model-labelled source helps; needs A-long′ as its same-LR control | 11830140 |
| **Views per scene, 17 → 50 / 100** | the last unmatched *evaluation* axis vs FAST3DIS's 50 views | 11840822 → 11841445 ff. |

**One arm already failed, and that is why D-long is a re-run.** The first RE10K arm (D, lr 1e-4)
**diverged**: best epoch 2 of 17, training loss rising, `train_AP50` collapsing to 0.006, and 3D
cells landing *below* the ScanNet-only control on the domain it trains on. The cause was isolated
one variable at a time to the **learning rate** — halving it to 5e-5 removes the collapse entirely.
**Do not quote arm D's numbers as what RE10K data is worth**: they price a broken run, not a source.

### 6.2 With what setting each comparison was produced — the matched-axes table

**The rule: every competitor-facing row is produced with the competitor's own setting.** Where an
axis is not reproducible, we use the closest one available and **declare it as such**. Three states,
never blurred into one another:

| axis | their setting | ours | state |
|---|---|---|---|
| evaluator | official ScanNet 3D instance benchmark | the same, vendored, same options | **matched** |
| bridge, unposed | FAST3DIS / IGGT: own predicted geometry + Sim(3)+ICP | the same | **matched** |
| bridge, posed ("geometric GT") | SegVGGT: GT poses + intrinsics + sensor depth | `--transfer_mode gt_projection`; its oracle returns **99.99 %** of assigned annotated vertices to their own instance | **matched — certified, not assumed** |
| label setting | FAST3DIS / IGGT class-agnostic; SegVGGT class-aware | both columns computed for every run | **matched** |
| benchmarks | ScanNetv2 / ScanNet200 / ScanNet++ / Replica | all four | **matched** |
| train split (SegVGGT) | official ScanNetv2 1201 | identical | **matched** |
| kept queries | SegVGGT 600 | 100 | **measured neutral** (0.138 → 0.140) — struck as an explanation |
| views, ScanNet++ / Replica | 50 | 50 | **matched** |
| views, ScanNetv2 / ScanNet200 | 50 (FAST3DIS) · 75–100 (SegVGGT) | **~17** | **closest available today — and it runs AGAINST us**; the dense export closes it, in flight (§6.1) |
| training data | FAST3DIS: ASE only → ScanNet zero-shot · IGGT: InsScene-15K | we train on ScanNet | **not matched, and it FAVOURS us** — arms I / I-gt close it, in flight (§6.1) |
| training compute | ~16 GPU-days | ~0.8 GPU-days, frozen backbone | **permanently unmatchable — and a strength, not an excuse** |
| ASE itself | 9.2 TB, unpublished 40 % scene list | — | **permanently out of reach** (§6.4) |

Read it top-down: everything above the "views, ScanNetv2" row is matched; the three rows below it
are the honest gaps, two of them already in flight, one permanent.

### 6.3 Open and costed, not started

**ScanNet200 supervision** — SegVGGT trains a 200-class checkpoint; our 2D GT is 19-class. This is
a *supervision* limit, not a scoring one, and the two are routinely confused: **class-agnostic is
how we score** (labels ignored at evaluation), **19 classes is what the head was taught**. It caps
the class-**aware** column on ScanNet200 only; the class-agnostic headline (§2) is untouched.

### 6.4 Permanently out of reach — state it, do not promise it

- **FAST3DIS's training set is unreproducible at any scale**: 9.2 TB *and* an unpublished 40 %
  scene list. Every FAST3DIS comparison is a cross-training-set comparison. Permanent.
- **ASE has no annotations on this cluster**; a fresh download under Project Aria terms would be
  needed.
- **InsScene-15K is incomplete** — only Infinigen / RE10K / ScanNet++ are published; the Aria
  portion is absent. Any replication is **partial** and must say so.
- **Training compute cannot be matched** (~0.8 vs ~16 GPU-days) — and it is a *strength*, not an
  excuse.
- **FAST3DIS never states which scenes it evaluates.** Do not claim identical evaluation sets.

---

### 6.5 The ablation-table hole — proposed, not yet launched

The headline lives on Tier 1 (3D). Two of the study's strongest levers are measured only on Tier 2
(2D) — see the ⚠ under §4. **A 3D-ruler ablation of `--no-cross_frame_attn` and
`--feature_mode single`** would put the ablation table on the same ruler as the result.

**Checked 2026-08-27: no 3D eval of either exists — and the two halves cost very differently.**

| half | checkpoint that exists | price |
|---|---|---|
| `--no-cross_frame_attn` | job **9503176** — official 1201 split, leak-free | **~free**: one existing 3D eval job on an existing checkpoint |
| `--feature_mode single` | job **8950613** — trained on scenes 0000–0489, which **overlap val-312** | **needs a new training run** on the official split; the existing checkpoint yields only a leaked, diagnostic number |

Put both prices in front of the supervisors, never their average.

---

## 7. HARD RULES — the errors most likely to reach a slide

1. **Never mix tiers.** Every number gets its ruler label (§1). **A Tier-2 number (A/B/C/F) on the
   same slide as a published competitor number is the worst error available here** — our 2D numbers
   are per-view masks on a 37×37 grid scored by our own metric code, and no published number is on
   that ruler. Tier 2 belongs in backup slides only.
2. **Never quote COCO next to ScanNet.** It is a port check, not a result — but keep it to hand:
   it is the answer to *"how do we know the implementation is not buggy?"*
3. **Label IGGT's triple `"IGGT, as re-evaluated by FAST3DIS"`** — never *"IGGT (published)"*. IGGT
   publishes **no ScanNet AP at all**.
4. **Never put a SegVGGT number in the unposed table.** SegVGGT is posed transfer; and its own paper
   reports **two different protocols** (Table 1 full-val; Table 2 ten sampled scenes) — never put a
   Table 1 and a Table 2 number in one row.
5. **Class-aware and class-agnostic are different columns.** FAST3DIS/IGGT are class-agnostic only.
6. **Anything trained on RE10K is SAM2-supervised** — masks are model output, not ground truth. Its
   rows are separately labelled, always.
7. **Extra-data rows stay fenced** from the ScanNet-only headline.
8. **Any pre-2026-07-08 number is on the retired SAM3 ground truth** and does not transfer — the
   switch to official ScanNet GT cost about half the AP50 headline.
9. **`docs/old/` and `legacy/` are archives.** Never cite them as current.
10. **Read every Δ against 0.009 per-bundle AP50** (the measured seed spread). Smaller is noise.
11. **On the ScanNet-only headline row, the AP column is a TIE, not a lead.** Lead AP50, lead AP25,
    tie FAST3DIS on AP, lead IGGT on all three. *"Ahead on all three"* is licensed **only** for the
    extra-data row (A-long), and only with its extra-data label attached. This file itself carried
    the wrong wording until 2026-08-27 — if you meet it again anywhere, it is a bug.
12. **Every competitor-facing number must name the setting it was produced under** (§6.2): matched,
    closest-available-and-declared, or permanently impossible. A comparison with no state named is
    not ready for a slide.

---

## 8. THE ONE-SLIDE SUMMARY

**Tier 1 — the rulers that face the field. This table is the paper.**

| # | Ruler | Our best | vs | Verdict |
|---|---|---|---|---|
| **D** | **3D official, UNPOSED, class-agnostic** | **0.042 / 0.138 / 0.504** | FAST3DIS 0.038 / 0.096 / 0.316 · IGGT\* 0.028 / 0.112 / 0.287 | **lead AP50 + AP25, tie FAST3DIS on AP, lead IGGT on all three** |
| **E** † | 3D official, POSED, class-aware | 0.088 / 0.260 / 0.572 | SegVGGT 0.504 / 0.717 / 0.870 | behind; **2.3× is the bridge**, ~4.6× is real |
| **G** | 3D, ScanNet200 / ScanNet++ / Replica | 0.124 / 0.009 / 0.006 AP (posed) | no like-for-like row held | zero-shot fails unposed, survives posed → **geometry, not masks** |

\* as re-evaluated by FAST3DIS — never *"IGGT (published)"*.
† Row E is a **class-aware** number, because that is what SegVGGT publishes. The data-scaling arms
(A-long and its block) are `--class_agnostic` runs with no class-aware column at all, so they cannot
appear on this row — not because they score worse.

**Row D is the paper. Everything else is evidence for it.**

**Tier 2 — backup only, never on the same slide as the table above.**

| # | Ruler | Our best | vs | What it is for |
|---|---|---|---|---|
| A | 2D single-frame, project val, N=490 | 0.694 / **0.729** | baseline head 0.451 / 0.294 | the scaling curve; direction-of-travel |
| B | 2D per-bundle, project val, N=490 | 0.539 / **0.515** | baseline head 0.367 / 0.199 | the multi-view ablations |
| C | 2D, official 1201/312 | per-frame **0.669** · per-bundle **0.552** ‡ | job 8900194, 0.604 | checkpoint selection; the data-scaling arms |
| F | COCO | 46.133 / 51.549 | upstream 46.1 / 51.5 | port is faithful — **not a project result** |

‡ Best of each column, from **two different checkpoints** — not one model doing both. The
class-agnostic data-scaling arms read higher still (0.704 / 0.604, §9.3), but that is a different
label setting *and* a different training mixture, so it is not "our best" on this row either.

---

## 9. APPENDIX — the internal rulers (Tier 2), backup slides only

**Nothing in this section may appear on a slide next to a published competitor number.** These are
our own metric code on 2D per-view masks (A/B/C) and an upstream-reproduction check (F).

### 9.1 Ruler A — 2D single-frame, project val, N=490

| Model | mIoU | AP50 | AP75 | mAP |
|---|---|---|---|---|
| retired baseline head (N=190) — **the bar** | 0.451 | 0.294 | 0.141 | 0.154 |
| MaskDINO, N=50 | 0.451 | 0.440 | 0.314 | 0.290 |
| MaskDINO, N=190 | 0.594 | 0.624 | 0.440 | 0.418 |
| MaskDINO, N=490 | 0.669 | 0.699 | 0.506 | 0.475 |
| **MaskDINO + 2 view-draws/scene + colour jitter, N=490** | **0.694** | **0.729** | **0.582** | **0.526** |

**+48 % mIoU / +138 % AP50** for the plain N=490 row (0.669 / 0.699); **+54 % / +148 %** for the
bolded best row (0.694 / 0.729). *Quote one pair with its own row — never cross them.*
The curve is **still rising** at 490 scenes; overfitting eases with scale (train mIoU 1.000 → 0.994
→ 0.947): the track is **data-limited**.

### 9.2 Ruler B — 2D per-bundle (the baseline head's own protocol), N=490

| Model | mIoU | AP50 |
|---|---|---|
| retired baseline head (N=190) | 0.367 | 0.199 |
| MaskDINO `--multi_frame` | 0.535 | 0.494 |
| … `--no-cross_frame_attn` | 0.393 | 0.311 |
| … `--feature_mode single` | 0.429 | 0.347 |
| **… + 2 view-draws + colour jitter** | **0.539** | **0.515** |

**+47 % mIoU, 2.6× AP50, with no post-hoc matching or fusion.** Cross-frame attention alone is
worth **+0.183** bundle AP50 — the only individually decisive component found anywhere in this
track. Bundle features are worth **+0.147** on this ruler but **−0.048** per frame: multi-view
consistency has a measured price (0.729 single-frame best vs 0.630 per-frame for the best
multi-view model).

### 9.3 Ruler C — 2D, official 1201/312 split

Per-frame AP50 **0.669** · per-bundle AP50 **0.552**, against job 8900194's 0.604 per-frame.
Scale holds on the honest split. **Seed-to-seed spread is 0.009 per-bundle AP50** — read every Δ in
this file against it.

The data-scaling arms (matched ~42 k steps, class-agnostic, official val-312):

| arm | train data | scenes | steps | per-frame AP50 | per-bundle AP50 | `id_switch` ↓ | `view_consistency` ↑ |
|---|---|---|---|---|---|---|---|
| **B** | ScanNet | 1201 | 42 035 | 0.675 | 0.548 | 0.441 | 0.707 |
| **C** | + ScanNet++ | 2054 | 41 080 | 0.677 | 0.554 | 0.472 | 0.714 |
| **A** | + Infinigen | 3520 | 42 240 | 0.628 | 0.479 | 0.531 | 0.693 |
| **A-long** | = A, converged | 3520 | 84 480 | **0.704** | **0.604** | **0.414** | **0.734** |

Same-domain ScanNet++ is **free** (C − B = +0.006, inside the 0.009 spread). Infinigen looked like
it cost −0.075 **at matched steps and that was a budget artefact**: doubled to 84 k steps, the full
3520-scene mixture is the best run of the block on every 2D axis. **The open caveat: A-long is
compared at convergence, not at matched steps** — "more data" and "more compute" are still not
separated at the top end (closing: C-long′ / A-long′, §6).

### 9.4 Ruler F — COCO, port verification only

**46.133 mask AP / 51.549 box AP** on COCO val2017, reproducing upstream MaskDINO's own released
checkpoint (46.1 / 51.5) to +0.004 AP. Independently, upstream's own code trained under **our**
recipe lands at **34.55 segm AP vs our arm's 34.3**, certifying matcher / criterion / DN on the
*training* path. **This is a correctness proof, not a project result. Never put it next to a
ScanNet number.**

---
