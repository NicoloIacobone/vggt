---
marp: true
paginate: true
---

<style>
section {
  font-size: 24px;
  line-height: 1.3;
  padding: 46px 56px;
}
h1 { font-size: 1.7em; margin: 0 0 .3em; }
h2 { font-size: 1.3em; margin: 0 0 .5em; }
ul, ol { margin: .3em 0; padding-left: 1.1em; }
li { margin: .16em 0; }
p  { margin: .38em 0; }
table { width: 100%; border-collapse: collapse; font-size: .8em; }
th, td { padding: .2em .45em; }
footer { font-size: 13px; }
section.mid   { font-size: 21px; }
section.dense { font-size: 19px; }
</style>

# Frozen VGGT-1B + a MaskDINO decoder
## 3D multi-view-consistent instance segmentation

Two decisions for this meeting:

**(a)** which last ablations to run
**(b)** whether to write a CVPR 2027 paper (November deadline)

---

<!-- _class: mid -->

## 2. The setup

- **Backbone: VGGT-1B, strictly frozen.** Never updated — no finetuning, no LoRA, no adaptation of any kind.
- **Its features are cached, not recomputed.** The frozen backbone is run over a scene's views **once, before training**, and its output tokens are stored; the training loop then only ever reads tokens and never runs the backbone again. That is what makes head-only training cost **~0.8 GPU-days** against the **~16 GPU-days** of the closest competitor.
- **Supervision is 2D masks only** — no 3D label ever enters training:
  - the **headline run: official ScanNet v2 2D instance annotations and nothing else**;
  - the **extra-data runs** add **ScanNet++ + Infinigen** per-frame instance annotations (3520 scenes) and are trained **class-agnostic** — they have no class-aware column at all;
  - the **RE10K runs are SAM2-supervised** — those masks are model output, not ground truth — and are labelled as such wherever they appear.
- **Decoder: MaskDINO-family** — the only trained part of the system.

---

<!-- _class: mid -->

## 3. How to read every number in this deck

Four labels travel with every 3D number. A number without them is not comparable to anything.

- **`AP / AP50 / AP25`** — always one triple, always in that order. In this literature `mAP` ≡ `AP`: the naming difference between papers means nothing, the *setting* does.
- **Unposed vs posed** — how the 2D masks reach the point cloud. *Unposed* uses **our own predicted** depth and cameras, so it scores mask quality **×** geometry quality. *Posed* uses **ScanNet's GT** poses, intrinsics and sensor depth, so it scores **mask quality alone**. The difference is a consistent **2.3× in AP50**.
- **Class-agnostic vs class-aware** — labels ignored, or the 18-class mean. FAST3DIS and IGGT publish **class-agnostic only**; SegVGGT is **class-aware**. We compute both columns for every run.
- **Views per scene** — how many views of one scene enter a single forward pass.
- **`id_switch` / `view_consistency`** — the two cross-view *identity* numbers, and they are **not** what AP measures. For a query matched to one GT instance over a whole bundle: `view_consistency` is the fraction of views where that query still segments the instance (**higher is better**); `id_switch` the fraction where a *different* query explains it better (**lower is better**). AP is scored on the fused 3D instance, so it sees identity only indirectly — a broken identity shows up there as a worse mask, never as its own number.

**One further rule:** the 3D-benchmark numbers are the ones that face a published paper. The project's internal 2D numbers (our own metric code, per-view masks) pick checkpoints and rank ablations — they are marked wherever they appear here and are never placed beside a competitor.

---

<!-- _footer: "Method — one query set per scene, shared by all its views" -->

<!-- _class: mid -->

## 4. The method — one query, one instance, across views

- **One query is one instance across all views, by construction.** Multi-view consistency is **intrinsic to the query** — it is not obtained by post-hoc fusion, tracking or mask matching.
- The whole output is a single tensor, `pred_masks [B, N, S, h, w]`:

| symbol | what it is |
|---|---|
| **B** | bundles in the batch — one bundle = the S views of one scene |
| **N** | object queries, shared by the whole bundle; the top **100** are kept at scoring (SegVGGT keeps 600 — measured neutral for us) |
| **S** | views of that scene in the *same* forward pass — **~17** per ScanNet scene |
| **h × w** | the grid the masks are predicted on: **37 × 37**, VGGT's own patch grid (SegVGGT: 259×196) |

- Query *n* carries the **same object in every one of the S views**, so cross-view identity costs no extra step.
- Those 2D masks then reach the benchmark point cloud through one of two **bridges** (slide 7).

---

<!-- _footer: "The study's main ablation — same decoder, same frozen backbone, same data, same protocol" -->

<!-- _class: mid -->

## 5. Two ways to anchor a query — the study's main ablation

| | where the query's positional prior lives |
|---|---|
| **2D boxes** *(the default)* | one 4-d box per query **per view**, refined layer by layer — the DAB/DINO recipe: the query *carries* its box, every decoder layer moves it |
| **3D anchors** *(`--anchor_3d`)* | one 3D point **+ radius** per query **per scene**, read off VGGT's own frozen point head and projected into each view |

- **Neither variant needs camera intrinsics or extrinsics.** A 3D anchor reaches a view through a *soft nearest-patch* — a softmax over the distances between the anchor and that view's patch positions — not through a perspective projection, which is what keeps it usable when the predicted cameras are imprecise.
- One variable: same decoder, same frozen backbone, same training data, same protocol.
- **The headline row is the 3D-anchor one.** It is worth **+66 % 3D AP50 in *both* bridges** and drops cross-view identity switches (`id_switch` **−0.089**), while being neutral in 2D.
- **Neither mechanism is ours.** 3D-anchored queries are already published by **FAST3DIS**; queries shared across views are already published by **SegVGGT**. What nobody has run is the **two of them against each other inside one decoder** — that is the contribution (slide 17).

---

<!-- _footer: "The ruler, part 1 — one evaluator, four benchmarks" -->

<!-- _class: mid -->

## 6. The ruler — the evaluator

- **Official ScanNet 3D instance benchmark**, and the evaluator is the **official script vendored** into the repo: same overlap thresholds, same confidence-ordered matching, same void handling, same options.
- **Licensed, not assumed.** On each of the four benchmarks, that dataset's own GT fed back as a prediction scores exactly **1.000 / 1.000 / 1.000**.
- **The same evaluator scores all four benchmarks** — ScanNetv2, ScanNet200, ScanNet++, Replica. Only the dataset adapter changes: its point clouds, its instance GT, its taxonomy. Nothing about the head, the bridge or the lifting moves.
- On the other three we report **class-agnostic only**: our head has 19 ScanNet classes and their taxonomies are not ours, so a class-aware column there would be an invented correspondence, not a measurement.
- Consequence: one ruler, four datasets — and on three of them **no published like-for-like row exists**, so they are evidence, not a competitor comparison.

---

<!-- _footer: "The ruler, part 2 — the 2D→3D bridge is the single biggest protocol variable" -->

<!-- _class: mid -->

## 7. The ruler — the two bridges from 2D masks to the point cloud

- **Unposed — our own predicted depth + cameras, then Sim(3)+ICP.** VGGT reconstructs a scene only up to an unknown **rotation, translation and global scale**; a *similarity transform* — that is what **Sim(3)** means — puts the finished prediction into the benchmark's coordinate frame. It is estimated in closed form from predicted-vs-GT **camera centres**, then refined by **ICP** against the mesh. **GT poses place the finished prediction; they never enter inference.** This is FAST3DIS's and IGGT's own setting → matched.
- **Posed — GT poses + intrinsics + sensor depth.** The bridge is exact by construction. This is SegVGGT's *"geometric GT"*, reproduced and **certified rather than assumed**: its oracle returns **99.99 %** of assigned annotated vertices to their own instance.
- **The bridge alone is worth a consistent 2.3× in AP50**: the *same* checkpoint and the *same* masks, scored posed instead of unposed. It is a constant of the protocol, not a property of a model — which is why a 3D number without its bridge named says nothing.
- **Ceiling of the whole setup — a separate measurement, not that 2.3×:** **GT** masks, not ours, pushed through the posed bridge score **0.828 / 0.948 / 0.974**. It says how much of a scene ~17 views can reach at all, and it belongs beside the *posed* column (our best posed row: 0.088 / 0.260 / 0.572, slide 12) — never beside the unposed headline.

---

<!-- _footer: "Official ScanNet 3D benchmark · UNPOSED (own predicted geometry) · class-agnostic — the competitors' own setting" -->

<!-- _class: mid -->

## 8. The headline — like-for-like against the two published competitors

| Method | Backbone | Views/scene | AP | AP50 | AP25 |
|---|---|---|---|---|---|
| IGGT *(as re-evaluated by FAST3DIS)* | adapted | 50 | 0.028 | 0.112 | 0.287 |
| FAST3DIS | LoRA-adapted DA3 | 50 | 0.038 | 0.096 | 0.316 |
| **Ours — 3D anchors, defaults, trained on ScanNet only** | **frozen VGGT-1B** | **~17** | **0.042** | **0.138** | **0.504** |
| — same recipe, second seed | 〃 | 〃 | 0.039 | 0.129 | 0.485 |
| — best lifting parameters *(sensitivity, not the headline — backup B1)* | 〃 | 〃 | 0.055 | 0.185 | 0.571 |
| **Ours + EXTRA TRAINING DATA** (ScanNet + ScanNet++ + Infinigen, 3520 scenes) | 〃 | 〃 | **0.057** | **0.166** | **0.516** |

**Across two seeds, on the ScanNet-only row: lead on AP50 (1.34–1.44×) and on AP25 (1.53–1.59×), TIE with FAST3DIS on AP, lead over IGGT on all three.** *"Ahead on all three"* describes the extra-data row only, and only with that label attached.

The extra-data row is kept separate from the headline, as the field does: the **mechanism** claim rests on the ScanNet-only row, the **scaling** claim on the extra-data one. The tuned row is a sensitivity check on the two lifting parameters — the sweep runs on the split we report on, so its argmax is not quotable as a result; it is here only because the *worst* point of that sweep is still 1.44× FAST3DIS's AP50 (backup B1).

---

<!-- _footer: "The caveats that travel with the headline — stated here, not waited for" -->

<!-- _class: mid -->

## 9. The four caveats that travel with that table

1. **Two seeds, but one run per cell — and one published row per competitor.** Both arms were replicated at a second seed, so "single run against a single control" is retired; what remains is the thin competitor side.
2. **Not training-matched, and the asymmetry FAVOURS us.** Both published rows are **zero-shot on ScanNet** (FAST3DIS trains on Aria/ASE only, IGGT on InsScene-15K); every row of ours trains on ScanNet. Protocol- and setting-matched, **not** training-matched. Closing: the no-ScanNet arms, in flight (slide 19).
3. **The class-collapse sign is checkpoint-dependent.** The same recipe *without* 3D anchors scores **0.017 / 0.060 / 0.334** class-agnostic **with its lifting parameters tuned** (0.013 / 0.050 / 0.320 at defaults) — ahead on AP25, ~2× behind on AP50/AP. So this is **not** a defaults-vs-defaults comparison, and the *tuned* label belongs on it wherever it appears.
4. **Two asymmetries run the other way**: our backbone is frozen where theirs is adapted, and we use **~17 views to their 50**.

---

<!-- _footer: "Matched axes — every competitor-facing row is produced with the competitor's OWN setting" -->

<!-- _class: dense -->

## 10. With what setting — the axes that are matched

| axis | their setting | ours | state |
|---|---|---|---|
| evaluator | official ScanNet 3D instance benchmark | the same, vendored, same options | **matched** |
| bridge, unposed | FAST3DIS / IGGT: own predicted geometry + Sim(3)+ICP | the same | **matched** |
| bridge, posed ("geometric GT") | SegVGGT: GT poses + intrinsics + sensor depth | reproduced; oracle returns **99.99 %** of assigned annotated vertices | **matched — certified, not assumed** |
| label setting | FAST3DIS / IGGT class-agnostic; SegVGGT class-aware | both columns computed for every run | **matched** |
| benchmarks | ScanNetv2 / ScanNet200 / ScanNet++ / Replica | all four | **matched** |
| train split (SegVGGT) | official ScanNetv2 1201 | identical | **matched** |
| kept queries | SegVGGT 600 | 100 | **measured neutral** (0.138 → 0.140) — struck as an explanation |
| views, ScanNet++ / Replica | 50 | 50 | **matched** |

Read it as a single claim: **everything above is produced under the competitor's own setting.** The three axes that are *not* matched are on the next slide, each with its direction stated.

---

<!-- _footer: "The three honest gaps — two in flight, one permanent" -->

<!-- _class: mid -->

## 11. …and the three axes that are not

| axis | their setting | ours | state |
|---|---|---|---|
| views, ScanNetv2 / ScanNet200 | 50 (FAST3DIS) · 75–100 (SegVGGT) | **~17** | **closest available today — and it runs AGAINST us**; dense frame export in flight |
| training data | FAST3DIS: ASE only → ScanNet zero-shot · IGGT: InsScene-15K | we train on ScanNet | **not matched, and it FAVOURS us** — the no-ScanNet arms close it, in flight |
| training compute | ~16 GPU-days | **~0.8 GPU-days**, frozen backbone | **permanently unmatchable — a strength, not an excuse** |
| ASE itself | 9.2 TB, unpublished 40 % scene list | — | **permanently out of reach** |

---

<!-- _footer: "Official ScanNet 3D benchmark · both bridges · class-aware, 18 classes — the gap to SegVGGT and its decomposition" -->

<!-- _class: dense -->

## 12. The posed comparison vs SegVGGT — and where the gap goes

| Checkpoint | UNPOSED (own geometry) | POSED (GT bridge) | bridge cost |
|---|---|---|---|
| **the control run** — the row the decomposition below is anchored to | 0.023 / 0.067 / 0.268 | 0.060 / 0.156 / 0.408 | **2.3× AP50** |
| 3D anchors | 0.038 / 0.112 / 0.360 | 0.104 / 0.257 / 0.504 | 2.3× |
| **wider bundle — 16 views, 20 epochs** | 0.032 / 0.115 / 0.414 | **0.088 / 0.260 / 0.572** | 2.3× |
| 16 views + 3D anchors | 0.032 / 0.109 / 0.353 | 0.082 / 0.236 / 0.501 | 2.2× |
| *oracle — GT masks through the posed bridge* | — | *0.828 / 0.948 / 0.974* | ceiling of a ~17-view budget |
| **SegVGGT (published, posed)** | — | **0.504 / 0.717 / 0.870** | — |

- **The same masks under a different bridge change by 2.3×**, on every row — a constant of the protocol, not of a checkpoint. Both columns therefore always travel together: a posed number on its own reads as a better result than it is.
- Of the **×10.7** raw gap to SegVGGT, a factor **2.3 is the bridge** and a factor **~4.6 is real**. That ~4.6 is bought with four things we chose not to have: **LoRA-adapted backbone** vs strictly frozen · **75–100 views** vs ~17 · **259×196 masks** vs 37×37 · **600 kept queries** vs 100 — and the last one is **measured neutral** for us (0.138 → 0.140), so it is struck off the list.

---

<!-- _footer: "3D ruler on ScanNet200 / ScanNet++ / Replica · POSED · class-agnostic · no published like-for-like row is held" -->

<!-- _class: mid -->

## 13. The other three benchmarks — how far the same checkpoint travels

**Not an ablation: nothing is removed from the model.** One checkpoint, one ruler, four datasets — it measures *transfer*. ScanNet200 is not out of domain either (same scenes, same tars, a 200-class taxonomy); ScanNet++ and Replica are.

**ScanNet-only checkpoint** — the same row as the headline, 3D anchors, posed, class-agnostic:
ScanNet200 **0.124 / 0.275 / 0.523** · ScanNet++ **0.009 / 0.038 / 0.178** · Replica **0.006 / 0.028 / 0.190**

**With the extra training data**, same ruler, same bridge, kept separate exactly like the extra-data headline row:
ScanNet200 **0.132 / 0.287 / 0.539** · ScanNet++ **0.019 / 0.068 / 0.275** · Replica **0.040 / 0.119 / 0.480**
→ this is **where the extra data pays most** — the only measurement in the project of something the ScanNet ruler cannot see.

- **Zero-shot dies under the unposed bridge**: every out-of-domain unposed cell is **0.000 AP / 0.000–0.001 AP50** across all four data arms, and survives weakly under the posed one. That **localises the failure to geometry, not masks**.
- **ScanNet200 costs zero extra data** — same scenes, same tars, a different taxonomy.
- **No like-for-like published row is held for any of these**: evidence, not a comparison.

---

<!-- _footer: "Ablation evidence — the ruler each Δ was measured on is named in the last column. No competitor number on this slide." -->

<!-- _class: dense -->

## 14. What actually buys the result — ranked

| Lever | Effect | Size | Measured on |
|---|---|---|---|
| **Training data, 50 → 490 scenes** *(project val)* | +0.26 per-frame AP50 | dominates everything | internal 2D |
| **Cross-frame attention** | +0.183 per-bundle AP50 | 20× seed noise | **internal 2D only** ⚠ |
| **Bundle features** | +0.147 per-bundle AP50 (−0.048 per-frame) | 16× | **internal 2D only** ⚠ |
| **3D anchors** | +66 % 3D AP50 in *both* bridges; `id_switch` −0.089 | AP-neutral in 2D (inside noise) | **3D, both bridges** |
| **Bundle width 8 → 16 views** | +0.027 per-bundle AP50; `id_switch` −0.113; +46 % unposed 3D AP50 | 3× | **3D, unposed** |
| Lifting parameters (vote radius, depth confidence) | +0.016 → +0.047 3D AP50 | larger than most decoder ablations | **3D, unposed** |
| Any single decoder component (two-stage, encoder, denoising, box init) | ≤0.046 per-frame AP50 | at 190 scenes | internal 2D |
| Mask resolution 37² → 74² | −0.022 (neutral) | **not the bottleneck** | internal 2D |

**Why the data row stops at 490:** that is the range of the *project-val* curve. The 1201 → 3520-scene arms are measured on the official split instead — per-bundle AP50 **0.548** (ScanNet, 1201) → **0.604** (+ ScanNet++ + Infinigen, 3520, converged), `id_switch` 0.441 → 0.414 — and their 3D counterpart is the extra-data row of slide 8 (0.042 / 0.138 / 0.504 → 0.057 / 0.166 / 0.516). One caveat on that pair: it is compared at convergence, not at matched steps, so "more data" and "more compute" are not yet separated — that is what the second row of slide 19 settles.

The two ⚠ rows are the reason for **decision (a)** — slide 16.

---

<!-- _class: mid -->

## 15. What that table says — four conclusions

Read every Δ on the previous slide against the **measured seed spread of 0.009 per-bundle AP50**. Anything smaller is noise.

1. **Data-limited, not architecture-limited.** A historical check, from before the official split existed: an early checkpoint trained on scenes that **overlapped its own validation scenes** — that overlap is the *leak* — scored **0.052** 3D AP50, while the leak-free one trained on the official 1201-scene split scored **0.083**. More training data outweighs even having seen the answers. *(Every number in this deck is on the official, non-overlapping 1201/312 split; the leaked checkpoint survives only as the diagnostic in slide 16.)*
2. **The lifting step, not the decoder, now binds** — AP25 is about **4×** AP50. The masks are not the constraint; the 2D→3D bridge is.
3. **Resolution is not the ceiling either.** On the 37×37 grid, GT-only scores **0.956** AP50 while the model sits at **~0.69** — what binds is **recognition**, not how finely the mask is drawn.
4. **Recognition and cross-view identity are separate axes.** A mechanism can buy identity without buying accuracy, so an identity mechanism is scored on the 3D ruler, never on 2D AP alone.

---

<!-- _footer: "DECISION (a) — the two strongest levers of the study have no 3D counterpart. No competitor number on this slide." -->

<!-- _class: mid -->

## 16. The hole in the ablation table → **decision (a)**

The headline lives on the 3D ruler. **The two most decisive levers measured anywhere in this project are measured only in 2D**: cross-frame attention **+0.183** and bundle features **+0.147** per-bundle AP50 — **20× and 16×** the 0.009 seed spread — and **neither has a 3D counterpart**. 3D anchors, bundle width and the lifting parameters all do; these two do not.

A 3D-ruler ablation of *"no cross-frame attention"* / *"per-frame features"* is the cheapest way to put the ablation table on the same ruler as the headline. **Checked 2026-08-27: no 3D eval of either exists — and the two halves do not cost the same.**

| half | checkpoint that exists | price |
|---|---|---|
| no cross-frame attention | job **9503176** — official 1201 split, leak-free | **~free**: one 3D eval job on an existing checkpoint |
| per-frame features | job **8950613** — trained on scenes 0000–0489, which **overlap the val split** | **needs a new training run**; the existing checkpoint yields only a leaked, diagnostic number |

**The two halves are priced separately, and their average means nothing: the question is whether to buy the free half only, or both.**

---

<!-- _class: mid -->

## 17. Positioning → **decision (b)**

**"Frozen VGGT + a decoder for a downstream 3D task" is the dominant pattern of the last ~12 months. The architecture alone is not a contribution.**

| Mechanism | Already owned by | What is left for us |
|---|---|---|
| Queries shared across views on a VGGT-family backbone | **SegVGGT** — 400 queries, inside all 24 aggregator layers | ours is the same *class* of idea, built outside the frozen backbone: report it as a controlled comparison against our own single-frame model, **not** as a new mechanism |
| **3D-anchored queries** | **FAST3DIS** | ours is an **ablation, not a contribution**: 3D anchors vs 2D boxes inside one decoder, one backbone, one dataset, one protocol — **which nobody has run** |
| Attention-dispersion fix over many global tokens | SegVGGT (FADA, train-time only) | partly free from deformable attention — one sentence of contrast |
| Single-pass multi-view panoptic prediction | PanSt3R | the "why not splat / why not fuse" comparison point |
| Anti-duplicate query regularisation | FAST3DIS | we use one-to-one Hungarian matching + denoising instead — a discussion point |

---

<!-- _class: mid -->

## 18. What IS ours — the three defensible claims

1. **The controlled comparison nobody has run.** One backbone, one dataset, one protocol, decoder ingredients varied one at a time — including **3D vs 2D anchors inside the same decoder**.
2. **Competitive 3D results from a strictly frozen backbone**, at **~0.8 GPU-days against ~16**, with no adaptation of any kind — where everyone else LoRA-adapts.
3. **Consistency intrinsic to the query, not post-hoc** — and now *measured*: view consistency **0.734**, identity switches **0.414** at best.

---

<!-- _class: mid -->

## 19. In flight right now

| axis | what it settles | job |
|---|---|---|
| **The two no-ScanNet arms** | train on IGGT's mixture **minus ASE**, never on ScanNet → makes the competitor comparison **training-matched**; completes a {± ScanNet} × {± RE10K} 2×2 | 11839134 / 11839135 |
| **More data ⇄ more compute** | separates the two at the top end; re-run at a halved learning rate after the first pair destabilised | 11831105 / 11830142 |
| **RE10K arm** (**SAM2-supervised** — masks are model output, not GT) | whether a fourth, model-labelled source helps; needs the same-LR control above | 11830140 |
| **Views per scene, 17 → 50 / 100** | the last unmatched *evaluation* axis against FAST3DIS's 50 views | 11840822 → 11841445 ff. |

**One arm already failed, which is why the RE10K arm is a re-run.** The first attempt **diverged**: best epoch 2 of 17, training loss rising, training AP50 collapsing to **0.006**, 3D cells below the ScanNet-only control on the very domain it trains on. The cause was isolated one variable at a time to the **learning rate**; halving it removes the collapse. That run prices a broken optimisation, not a data source, and is not what RE10K is worth.

---

<!-- _class: mid -->

## 20. Permanently out of reach

- **FAST3DIS's training set is unreproducible at any scale**: 9.2 TB *and* an unpublished 40 % scene list. Every FAST3DIS comparison is a cross-training-set comparison. Permanent.
- **ASE has no annotations on this cluster**; a fresh download under Project Aria terms would be needed.
- **InsScene-15K is incomplete** — only Infinigen / RE10K / ScanNet++ are published, the Aria portion is absent. Any replication is **partial** and must say so.
- **FAST3DIS never states which scenes it evaluates.** We do not claim identical evaluation sets.

---

<!-- _footer: "The three 3D rulers — the only numbers here that face the field" -->

<!-- _class: mid -->

## 21. Where we stand, and the two questions

| Ruler | Our best | vs | Verdict |
|---|---|---|---|
| **3D official, UNPOSED, class-agnostic** | **0.042 / 0.138 / 0.504** | FAST3DIS 0.038 / 0.096 / 0.316 · IGGT\* 0.028 / 0.112 / 0.287 | **lead AP50 + AP25, tie FAST3DIS on AP, lead IGGT on all three** |
| 3D official, POSED, class-aware † | 0.088 / 0.260 / 0.572 | SegVGGT 0.504 / 0.717 / 0.870 | behind; **2.3× is the bridge**, ~4.6× is real |
| 3D on ScanNet200 / ScanNet++ / Replica | 0.124 / 0.009 / 0.006 AP (posed) | no like-for-like row held | zero-shot fails unposed, survives posed → **geometry, not masks** |

\* as re-evaluated by FAST3DIS: IGGT publishes no ScanNet AP of its own. † class-aware because that is what SegVGGT publishes; the class-agnostic scaling runs have no class-aware column at all, so they cannot appear on that row — not because they score worse.

---

<!-- _footer: "BACKUP · sensitivity of the 2D→3D lifting — why the argmax of the sweep is NOT the headline" -->

<!-- _class: mid -->

## B1 — backup. The two lifting parameters, and why they are not the headline

The 2D→3D bridge has exactly **two knobs**, and neither is part of the model:

- the **vote radius** — how far a lifted pixel may reach to claim a mesh vertex;
- the **depth-confidence filter** — how much of the least reliable predicted depth is discarded before lifting.

**The headline row runs both at their defaults.** The tuned row (**0.055 / 0.185 / 0.571**) is reported as a *sensitivity analysis*, never as the result, for one reason: the sweep runs on the val split we report on, so quoting its argmax would be test-set tuning.

**How it is justified:** the sweep is quoted to show the lead is **not** a tuning artefact. On the 3D-anchor checkpoint **every point of the grid is above FAST3DIS** — the *worst* point of the sweep is still **1.44×** its AP50. The physics is the same story: the radius stops helping once it covers the registration error, so beyond that the votes already reach every vertex they will ever reach.
