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

**Status update — 2026-08-27.**

Where the project stands on the 3D benchmark, under what setting each number was produced,
what is running right now, and what is still open.

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
- **Views per scene** — how many views of one scene enter a single forward pass. **This one moved on 2026-08-27**: everything is now produced at the competitors' own 50 views (slide 9).

**One further rule:** the 3D-benchmark numbers are the ones that face a published paper. Everything else in this deck is either a *setting* statement or a cross-view **identity** measurement, and neither is ever placed on the same row as a competitor.

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
| **S** | views of that scene in the *same* forward pass — trained at 8, and it generalises: **up to 50 at evaluation**, which is the competitors' budget |
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
- **The headline row is the 3D-anchor one.** It is worth **+66 % 3D AP50 in *both* bridges** — measured on the benchmark, and that is the claim. It is neutral in 2D accuracy. ⚠ Its effect on cross-view *identity* is **being re-measured**: the project's own metric said the gain was large, the formal association metric (AssA) says it is +0.005. See slide 14.
- **Neither mechanism is ours.** 3D-anchored queries are already published by **FAST3DIS**; queries shared across views are already published by **SegVGGT**. What nobody has run is the **two of them against each other inside one decoder** — that is the contribution (slide 13).

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
- **Ceiling of the whole setup — a separate measurement, not that 2.3×:** **GT** masks, not ours, pushed through the posed bridge score **0.828 / 0.948 / 0.974**. It says how much of a scene a bounded view budget can reach at all, and it belongs beside the *posed* column (slide 11) — never beside the unposed headline.

---

<!-- _footer: "Official ScanNet 3D benchmark · UNPOSED (own predicted geometry) · class-agnostic · 50 views — the competitors' own setting" -->

<!-- _class: mid -->

## 8. The headline — like-for-like against the two published competitors

| Method | Backbone | Views/scene | AP | AP50 | AP25 |
|---|---|---|---|---|---|
| IGGT *(as re-evaluated by FAST3DIS)* | adapted | 50 | 0.028 | 0.112 | 0.287 |
| FAST3DIS | LoRA-adapted DA3 | 50 | 0.038 | 0.096 | 0.316 |
| **Ours — 3D anchors, defaults, trained on ScanNet only** | **frozen VGGT-1B** | **50** | **0.053** | **0.170** | **0.542** |
| — the same checkpoint at a 17-view budget *(two seeds)* | 〃 | 17 | 0.042 | 0.138 | 0.504 |
| **Ours + EXTRA TRAINING DATA** (ScanNet + ScanNet++ + Infinigen, 3520 scenes) | 〃 | **50** | **0.069** | **0.193** | **0.560** |

**At the competitors' own 50-view budget we lead on all three columns** — 1.39× / 1.77× / 1.72× on FAST3DIS, more on IGGT — with a **strictly frozen** backbone and every lifting parameter at its default. The extra-data row leads by 1.8–2.0×.

**Two things this table needs said out loud.** The view budget is now **matched, not conceded**: at 17 views the AP column was a *tie* with FAST3DIS, and the lead on all three only exists because the comparison is finally view-for-view. And more views are **not** an open lever — 50 → 71 views is flat-to-negative, so this saturates exactly where they report.

---

<!-- _footer: "Matched axes — every competitor-facing row is produced with the competitor's OWN setting" -->

<!-- _class: dense -->

## 9. With what setting — the axes that are matched

| axis | their setting | ours | state |
|---|---|---|---|
| evaluator | official ScanNet 3D instance benchmark | the same, vendored, same options | **matched** |
| bridge, unposed | FAST3DIS / IGGT: own predicted geometry + Sim(3)+ICP | the same | **matched** |
| bridge, posed ("geometric GT") | SegVGGT: GT poses + intrinsics + sensor depth | reproduced; oracle returns **99.99 %** of assigned annotated vertices | **matched — certified, not assumed** |
| label setting | FAST3DIS / IGGT class-agnostic; SegVGGT class-aware | both columns computed for every run | **matched** |
| benchmarks | ScanNetv2 / ScanNet200 / ScanNet++ / Replica | all four | **matched** |
| **views per scene, all four benchmarks** | 50 (FAST3DIS, IGGT) · 75–100 (SegVGGT) | **50** — achieved mean 46.7 on ScanNetv2, 50 on ScanNet++ / Replica | **matched — closed 2026-08-27** |
| **train split — SegVGGT only** | official ScanNetv2 1201 | identical | **matched** |
| kept queries | SegVGGT 600 | 100 | **measured neutral** (0.138 → 0.140) — struck as an explanation |

Two notes on the rows in bold. **Views** was the last unmatched *evaluation* axis; the dense frame export closed it, and at their budget our lead widens rather than shrinks. **Train split is matched for SegVGGT and only for SegVGGT** — the other two never train on ScanNet at all, which is the first row of the next slide.

---

<!-- _footer: "The honest gaps — one closing, two permanent" -->

<!-- _class: mid -->

## 10. …and the axes that are not

| axis | their setting | ours | state |
|---|---|---|---|
| training data — **FAST3DIS, IGGT** | FAST3DIS: ASE only → ScanNet zero-shot · IGGT: InsScene-15K, no ScanNet either | we train on ScanNet | **not matched, and it FAVOURS us** — the no-ScanNet arms close it, running now |
| training compute | ~16 GPU-days | **~0.8 GPU-days**, frozen backbone | **permanently unmatchable — a strength, not an excuse** |
| ASE itself | 9.2 TB, and an unpublished 40 % scene list | — | **the scene list is permanently out of reach** (slide 16) |

- **This is the one asymmetry that runs against the comparison rather than against us**, and it is stated rather than waited for: both published rows are **zero-shot on ScanNet** and every row of ours trains on it. Verified from the papers, not assumed.
- **One asymmetry runs the other way and is not in the table:** our backbone is *strictly frozen* where both of theirs are adapted.

---

<!-- _footer: "Official ScanNet 3D benchmark · both bridges · class-aware except where marked — the gap to SegVGGT and its decomposition" -->

<!-- _class: dense -->

## 11. The posed comparison vs SegVGGT — and where the gap goes

| Checkpoint | UNPOSED (own geometry) | POSED (GT bridge) | bridge cost |
|---|---|---|---|
| **the control run** | 0.023 / 0.067 / 0.268 | 0.060 / 0.156 / 0.408 | **2.3× AP50** |
| **3D anchors** — the row the decomposition below is anchored to | 0.038 / 0.112 / 0.360 | 0.104 / 0.257 / 0.504 | 2.3× |
| wider bundle — 16 views, 20 epochs | 0.032 / 0.115 / 0.414 | 0.088 / 0.260 / 0.572 | 2.3× |
| **extra data, at 50 views** — ⚠ **class-AGNOSTIC**, see below | 0.069 / 0.193 / 0.560 | **0.200 / 0.419 / 0.725** | 2.2× |
| *oracle — GT masks through the posed bridge* | — | *0.828 / 0.948 / 0.974* | the protocol's ceiling |
| **SegVGGT (published, posed)** | — | **0.504 / 0.717 / 0.870** | — |

- **The same masks under a different bridge change by 2.3×**, on every row — a constant of the protocol, not of a checkpoint. Both columns therefore always travel together: a posed number on its own reads as a better result than it is.
- **The residual gap is checkpoint-dependent, so it is quoted with its checkpoint.** On the 3D-anchor row the total distance to SegVGGT is **×6.4**, of which **×2.3 is the bridge** and **×2.8 is real**. That ×2.8 is bought with three things we chose not to have: **LoRA-adapted backbone** vs strictly frozen · **75–100 views** vs 50 · **259×196 masks** vs 37×37. A fourth candidate — 600 kept queries vs 100 — is **measured neutral** and struck off.
- **The last row is class-agnostic and the others are class-aware** — the extra-data arms are trained `--class_agnostic` and have no class-aware column *at all*, which is why it cannot be placed on the same footing rather than because it scores worse. On it the raw distance to SegVGGT falls to **1.71×**: a *direction*, not a like-for-like ratio.

---

<!-- _footer: "Putting the two consistency mechanisms on the same ruler as the headline. No competitor number on this slide." -->

<!-- _class: mid -->

## 12. Closing the ablation table on the 3D ruler

The headline lives on the 3D benchmark. But the **two mechanisms that carry multi-view consistency** — cross-frame attention, and letting a query see the whole bundle's features rather than one view's — had only ever been measured on the project's internal 2D metrics. 3D anchors, bundle width and the lifting parameters all had 3D numbers; those two did not. Both were launched today; **the first has landed.**

| Checkpoint | 18-class AP / AP50 / AP25 | class-agnostic |
|---|---|---|
| the control — full model, official split | 0.023 / 0.067 / 0.268 | 0.013 / 0.050 / 0.320 |
| **without cross-frame attention** | **0.010 / 0.029 / 0.167** | **0.005 / 0.021 / 0.214** |
| ratio | 0.46× / **0.43×** / 0.62× | 0.38× / **0.42×** / 0.67× |

- **Removing cross-frame attention costs 57 % of the 3D AP50** — far more than any decoder ingredient measured anywhere in this project, and now on the **same ruler as the headline** rather than on an internal one. The effect survives the label setting (0.43× class-aware, 0.42× class-agnostic), so it is not an artefact of class collapse.
- **AP25 falls least.** Objects are still found without it; what degrades is whether the fused 3D instance clears the 0.5-IoU bar — the same signature as everything else here: the mechanism buys quality, the lifting step converts it into AP50.
- **Single variable, verified not assumed:** a config diff of the two runs returns exactly one differing key.

**The other half is still running** (`--feature_mode single`, job 11986440): no leak-free checkpoint of that arm existed, so it needed a new 12-epoch training run before it could be evaluated at all. That asymmetry in price is why they were two jobs and not one.

---

<!-- _class: mid -->

## 13. Positioning — what the field already owns

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

## 14. What IS ours — the three defensible claims

1. **The controlled comparison nobody has run.** One backbone, one dataset, one protocol, decoder ingredients varied one at a time — including **3D vs 2D anchors inside the same decoder**.
2. **Competitive 3D results from a strictly frozen backbone**, at **~0.8 GPU-days against ~16**, with no adaptation of any kind — where everyone else LoRA-adapts.
3. **Consistency intrinsic to the query, not post-hoc — and now measured on a published ruler.** The evaluation reports **HOTA / AssA / DetA / IDF1**, the tracking literature's own metrics, with a bundle's views read as timesteps and one query read as one track. That mapping is exact rather than invented, which is the point: nothing has to be tracked, matched or fused first. On the headline checkpoint: **HOTA 0.42, AssA 0.58, DetA 0.31, IDF1 0.49**.

**Why this mattered more than expected.** The consistency numbers this project had been quoting were **its own definitions**, with no published counterpart — none of the three competitors reports a cross-view consistency metric at all. Switching to the formal ones immediately changed a claim: where our own `id_switch` said 3D anchors cut identity errors by a large margin (−0.088), **AssA — the published counterpart of exactly that quantity — moves by +0.005**. `id_switch` flips on near-ties between queries segmenting the same object; AssA asks how much of each identity's trajectory is actually explained. The seed spread for these metrics is being measured before either number is quoted as an effect. *(The +66 % 3D AP50 from 3D anchors is untouched by this — it is measured on the benchmark, not on an identity metric.)*

---

<!-- _class: mid -->

## 15. In flight right now

| what | what it settles | job | state |
|---|---|---|---|
| **The two no-ScanNet arms** | train on IGGT's mixture **minus ASE**, never on ScanNet → makes the competitor comparison **training-matched** | 11839134 · 11839135 | one **done**, one running; the done one's 3D evals are scoring now |
| **More data ⇄ more compute** | separates the two at the top end; re-run at a halved learning rate after the first pair destabilised | 11831105 · 11830142 | **both done** |
| **RE10K arm** (**SAM2-supervised** — masks are model output, not GT) | whether a fourth, model-labelled source helps | 11830140 | **done — it COSTS in-domain**, see below; 3D matrices scoring |
| **The ablation table on the 3D ruler** (slide 12) | cross-frame attention and bundle features, priced on the headline's own ruler | 11986399 · 11986440 | **first half DONE** — 57 % of the 3D AP50; second running |
| **Formal identity metrics** (slide 14) | HOTA / AssA / DetA / IDF1 on the headline checkpoint and its control | 11994637 · 11994639 | **DONE — and they revised an identity claim** (slide 14) |
| **Seed spread for those metrics** | whether the +0.005 AssA of 3D anchors is an effect or noise — no spread has ever been measured for them | 11997568 · 11997569 | launched today |
| ~~Views per scene, 17 → 50~~ | the last unmatched *evaluation* axis | 11841445 ff. | **DONE — and it moved the headline (slide 8)** |

**The RE10K arm and its control both landed today, and the answer is negative in-domain.** Against its same-learning-rate control, at a gradient-step budget matched to within 1 % and the same ScanNet val-312, adding 1500 SAM2-supervised RE10K scenes costs **−0.051 per-bundle AP50** (5.7× the seed spread) and worsens cross-view identity. Read it as **displacement, not as "bad data"**: at fixed compute the fourth source buys its steps from the other three. It also does not settle the question RE10K was added for — that one is *out-of-domain*, and those 3D matrices are still scoring.

*(The earlier attempt at this arm diverged — best epoch 2 of 17, training AP50 collapsing to 0.006. The cause was isolated one variable at a time to the learning rate; halving it removed the collapse, and this run is the converged replacement.)*

---

<!-- _class: mid -->

## 16. Open, and permanently out of reach

**Open and costed — the highest-value data item left:**

- **A partial ASE download is affordable, and ASE is *not* unobtainable.** The public Aria Synthetic Environments release ships **2D instance segmentation ground truth** — exactly the supervision we train on — and downloads **by scene range**. At ~230 MB/scene a **1000-scene pilot is ~230 GB**, which fits our quota. It would turn our IGGT replication from "their mixture minus ASE" into the complete one, and it is the only route to training on FAST3DIS's own source.

**Permanently out of reach — state it, do not promise it:**

- **FAST3DIS's exact training set.** Not the data — the **scene list**: 40 % of it is unpublished. Every FAST3DIS comparison is therefore a cross-training-set comparison, at any download size.
- **InsScene-15K is incomplete** — only Infinigen / RE10K / ScanNet++ are published, the Aria portion is absent. Any replication is **partial** and must say so.
- **FAST3DIS never states which scenes it evaluates.** We do not claim identical evaluation sets.
- **Training compute** — ~0.8 vs ~16 GPU-days, and it is a *strength*, not an excuse.

---

<!-- _footer: "The three 3D rulers — the only numbers here that face the field" -->

<!-- _class: mid -->

## 17. Where we stand

| Ruler | Our best | vs | Verdict |
|---|---|---|---|
| **3D official, UNPOSED, class-agnostic, 50 views** | **0.053 / 0.170 / 0.542** | FAST3DIS 0.038 / 0.096 / 0.316 · IGGT\* 0.028 / 0.112 / 0.287 | **lead on all three, at their own view budget** |
| 3D official, POSED, class-aware † | 0.088 / 0.260 / 0.572 | SegVGGT 0.504 / 0.717 / 0.870 | behind; **2.3× is the bridge**, ×2.8 is the real residual |
| 3D on ScanNet200 / ScanNet++ / Replica | 0.124 / 0.009 / 0.006 AP (posed) | no like-for-like row held | zero-shot fails unposed, survives posed → **geometry, not masks** |

\* as re-evaluated by FAST3DIS: IGGT publishes no ScanNet AP of its own. † class-aware because that is what SegVGGT publishes; the class-agnostic scaling runs have no class-aware column at all, so they cannot appear on that row — not because they score worse.

**The one-line read:** on the setting the two unposed competitors publish in, matched axis by axis including their view budget, a **frozen** backbone with a trained decoder leads them — and the remaining distance to the posed state of the art is now mostly priced and partly explained.
