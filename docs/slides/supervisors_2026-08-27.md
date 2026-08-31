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
section.tighter { font-size: 17px; }
</style>

# Frozen VGGT-1B + a MaskDINO decoder
## 3D multi-view-consistent instance segmentation

**Status update.**

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

<!-- _class: dense -->

## 3. How to read every number in this deck

Five labels travel with every number in this deck. A number without them is not comparable to anything.

- **`AP / AP50 / AP25`** — always one triple, always in that order. In this literature `mAP` ≡ `AP`: the naming difference between papers means nothing, the *setting* does.
- **Unposed vs posed** — how the 2D masks reach the point cloud. *Unposed* uses **our own predicted** depth and cameras: mask quality **×** geometry quality. *Posed* uses **ScanNet's GT** poses, intrinsics and sensor depth: **mask quality alone**. A consistent **2.3× in AP50** apart.
- **Class-agnostic vs class-aware** — labels ignored, or the 18-class mean. FAST3DIS and IGGT publish **class-agnostic only**; SegVGGT is **class-aware**. We compute both columns for every run.
- **Views per scene** — how many views of one scene enter a single forward pass. **This one moved**: everything is now produced at the competitors' own 50 views (slide 10).
- **Cross-view identity — `HOTA / AssA / DetA / IDF1`.** The tracking literature's own metrics, mapped exactly: a bundle's S views *are* S timesteps, one query *is* one track, by construction. They are **2D, per bundle**, and get **no competitor column** — none of the three publishes any consistency metric at all. Our older `view_consistency` / `id_switch` are ours alone: internal diagnostic, never quoted outward (slide 14).

**One further rule:** only the **3D-benchmark** numbers face a published paper. A *setting* statement and an *identity* measurement are never put on the same row as a competitor.

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
- **The headline row is the 3D-anchor one.** It is worth **+66 % 3D AP50 in *both* bridges** — measured on the benchmark, and that is the claim. It is neutral in 2D accuracy. ⚠ It does **not** carry a separate cross-view *identity* claim any more: across two seeds no published identity metric distinguishes it from the control (slide 14).
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

<!-- _footer: "The training axis first — every number in this deck is read against this table" -->

<!-- _class: dense -->

## 8. First: the **training-data** axis

Only one of the three competitors trains on what we train on. Every number in this deck is read against that.

| competitor | they train on | the run of ours that matches it | ours vs theirs |
|---|---|---|---|
| **SegVGGT** | ScanNetv2 1201 | **the identical split** → **TRAINING-MATCHED** | posed 0.104 / 0.257 / 0.504 vs **0.504 / 0.717 / 0.870** — **×2.8 behind** once the ×2.3 bridge is out |
| **FAST3DIS** | Aria/ASE only — zero real data | **our no-ScanNet run** — never sees ScanNet, but has **no ASE** either | 0.005 / 0.023 / 0.251 vs 0.038 / 0.096 / 0.316 — **~4× behind** |
| **IGGT** | InsScene-15K = ASE + Infinigen + RE10K + ScanNet++ | **the same run** = their mixture **minus ASE** | 〃 vs 0.028 / 0.112 / 0.287 — **~4× behind** |
| *for reference* | — | *the same recipe **with** ScanNet* | *0.053 / 0.170 / 0.542 — ahead of both (slide 9)* |

**Wherever the training data is matched or approximated, we are behind.** The lead on slide 9 exists in the one configuration where we train on the evaluation domain and they do not: removing ScanNet costs a factor **6 in AP50**.

⚠ **It does not follow that the recipe loses at equal data.** That run has **no ASE at all** — 3819 scenes against ~100 k, frozen against adapted, ~0.8 GPU-days against ~16. *"We cannot match their training setting"* is supportable; *"we lose at equal data"* has never been measured and cannot be here.

---

<!-- _footer: "Official ScanNet 3D benchmark · UNPOSED (own predicted geometry) · class-agnostic · 50 views — the competitors' own setting" -->

<!-- _class: tighter -->

## 9. The headline — like-for-like on protocol, **not** on training data (slide 8)

| Method *(backbone)* | trains on ScanNet? | Views | AP | AP50 | AP25 |
|---|---|---|---|---|---|
| IGGT *(adapted; re-evaluated by FAST3DIS)* | **no** | 50 | 0.028 | 0.112 | 0.287 |
| FAST3DIS *(LoRA-adapted DA3)* | **no** | 50 | 0.038 | 0.096 | 0.316 |
| **Ours — 3D anchors, defaults, ScanNet only** *(**frozen** VGGT-1B)* | **yes** | **50** | **0.053** | **0.170** | **0.542** |
| — the same checkpoint at 17 views *(two seeds)* | yes | 17 | 0.042 | 0.138 | 0.504 |
| **Ours + EXTRA DATA** (+ScanNet++ +Infinigen, 3520 scenes) | yes | **50** | **0.069** | **0.193** | **0.560** |
| **Ours with ScanNet REMOVED** — the no-ScanNet run *(slide 8)* | **no** | 17 | **0.005** | **0.023** | **0.251** |

**At the competitors' own 50-view budget we lead on all three columns** — 1.39× / 1.77× / 1.72× on FAST3DIS, more on IGGT — with a **strictly frozen** backbone and every lifting parameter at its default. Extra data leads by 1.8–2.0×.

**The last row is why the slide is titled this way.** Evaluator, bridge and views — the three a reviewer checks first — are matched; the fourth, training data, is not, and it runs in our favour. **Quote the lead only with that attached.** The view budget is **matched, not conceded** (at 17 views the AP column was a *tie*), and more views are **not** an open lever: 50 → 71 is flat-to-negative.

---

<!-- _footer: "Matched axes — everything except training data and compute" -->

<!-- _class: dense -->

## 10. With what setting — the axes that ARE matched

| axis | their setting | ours | state |
|---|---|---|---|
| evaluator | official ScanNet 3D instance benchmark | the same, vendored, same options | **matched** |
| bridge, unposed | FAST3DIS / IGGT: own predicted geometry + Sim(3)+ICP | the same | **matched** |
| bridge, posed ("geometric GT") | SegVGGT: GT poses + intrinsics + sensor depth | reproduced; oracle returns **99.99 %** of annotated vertices | **matched — certified, not assumed** |
| label setting | FAST3DIS / IGGT class-agnostic; SegVGGT class-aware | both columns computed for every run | **matched** |
| **views per scene, all four benchmarks** | 50 (FAST3DIS, IGGT) · 75–100 (SegVGGT) | **50** — achieved mean 46.7 on ScanNetv2 | **matched — closed** |
| kept queries | SegVGGT 600 | 100 | **measured neutral** (0.138 → 0.140) — struck as an explanation |
| **training data** | see slide 8 | matched for SegVGGT only | **matched once of three** |
| **training compute** | ~16 GPU-days | **~0.8 GPU-days**, frozen backbone | **permanently unmatchable — a strength, not an excuse** |

**Views** was the last unmatched *evaluation* axis; the dense frame export closed it, and at their budget our lead widens. Everything above the two bold rows is matched axis for axis — which is what makes those two rows the whole story.

---

<!-- _footer: "Official ScanNet 3D benchmark · both bridges · class-aware except where marked — the ONE training-matched comparison" -->

<!-- _class: tighter -->

## 11. SegVGGT — **the one training-matched comparison**

**Same training data, same split, same 1201 scenes** (slide 8) — the one row where nothing is conceded on the data axis, and the row we are behind on. *Ahead where they never train on the domain, behind where we train on the same data they do.*

| Checkpoint | UNPOSED (own geometry) | POSED (GT bridge) | bridge cost |
|---|---|---|---|
| **the control run** | 0.023 / 0.067 / 0.268 | 0.060 / 0.156 / 0.408 | **2.3× AP50** |
| **3D anchors** — the decomposition below is anchored here | 0.038 / 0.112 / 0.360 | 0.104 / 0.257 / 0.504 | 2.3× |
| wider bundle — 16 views, 20 epochs | 0.032 / 0.115 / 0.414 | 0.088 / 0.260 / 0.572 | 2.3× |
| **extra data, 50 views** — ⚠ **class-AGNOSTIC** | 0.069 / 0.193 / 0.560 | **0.200 / 0.419 / 0.725** | 2.2× |
| *oracle — GT masks through the posed bridge* | — | *0.828 / 0.948 / 0.974* | the protocol's ceiling |
| **SegVGGT (published, posed)** | — | **0.504 / 0.717 / 0.870** | — |

- **The same masks under a different bridge change by 2.3×**, on every row — a constant of the protocol, not of a checkpoint. Both columns always travel together: a posed number alone reads better than it is.
- **×2.8 is the training-matched verdict.** Total distance on the 3D-anchor row is **×6.4**: **×2.3 bridge**, **×2.8 real** — against a competitor trained on our exact split. It is checkpoint-dependent, so it travels with its checkpoint. That ×2.8 buys three things we chose not to have: **LoRA-adapted backbone** · **75–100 views** · **259×196 masks** vs 37×37. A fourth candidate, 600 kept queries vs 100, is **measured neutral** and struck off.
- **The extra-data row is class-agnostic, the others class-aware** — those runs have no class-aware column *at all*, which is why it cannot sit on the same footing; not because it scores worse. Its raw distance to SegVGGT is **1.71×**: a *direction*, not a ratio.

---

<!-- _footer: "Putting the two consistency mechanisms on the same ruler as the headline. No competitor number on this slide." -->

<!-- _class: dense -->

## 12. The ablation table, now on the 3D ruler

The headline lives on the 3D benchmark, but the **two mechanisms that carry multi-view consistency** — cross-frame attention, and letting a query see the whole bundle's features rather than one view's — had only ever been measured on the project's internal 2D metrics. **Both now have 3D numbers.**

| Checkpoint | 18-class AP / AP50 / AP25 | class-agnostic | AP50 vs control |
|---|---|---|---|
| the control — full model, official split | 0.023 / 0.067 / 0.268 | 0.013 / 0.050 / 0.320 | — |
| **without cross-frame attention** | 0.010 / 0.029 / 0.167 | 0.005 / 0.021 / 0.214 | **0.43× / 0.42×** |
| **per-frame features** | 0.020 / 0.051 / 0.251 | 0.007 / 0.025 / 0.234 | **0.76× / 0.51×** |

- **Both levers are worth roughly half the 3D AP50, and both dwarf every decoder ingredient** (two-stage, encoder, denoising, box init were all ≤0.046 on the 2D ruler). These are the two mechanisms the study leans on, and they are now priced where the headline is.
- **Cross-frame attention survives the label setting; bundle features does not** — 0.43×/0.42× against 0.76×/0.51×. The bundle-features row must always be quoted **with its label setting**.
- **The 2D ordering held, its spacing did not.** In 2D the two looked close (1.24× apart); class-aware on the 3D ruler they are 2.4× apart. That correction is what the exercise bought.
- **Single variable in both cases, verified not assumed:** a config diff against the control returns exactly one differing key. The second row is schedule-matched, not convergence-matched — both runs peak at their last epoch.

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
2. **The first measurement of what a *strictly frozen* backbone reaches here.** Everyone in this space LoRA-adapts; nobody has reported the unadapted case. We have, at **~0.8 GPU-days against ~16** on **1201 scenes against ~100 k**: in-domain it is enough to lead two adapted competitors, on their own training setting it is **not** (slide 8). **Both halves are the finding** — *"competitive 3D results"* alone would drop the data axis.
3. **Consistency intrinsic to the query, not post-hoc — and now measured on a published ruler.** The evaluation reports **HOTA / AssA / DetA / IDF1**, the tracking literature's own metrics, with a bundle's views read as timesteps and one query read as one track. That mapping is exact rather than invented, which is the point: nothing has to be tracked, matched or fused first. On the headline checkpoint: **HOTA 0.42, AssA 0.58, DetA 0.31, IDF1 0.49**.

**It cost us a claim, and that is the exercise working.** The consistency numbers this project quoted were **its own definitions**, with no published counterpart. Re-measured on the formal ones across **two seeds**, the secondary claim that 3D anchors improve cross-view identity **does not hold**: every published metric moves by less than its own seed spread (AssA +0.001 against 0.005), while only our `id_switch` sees an effect. The mechanism's real result — **+66 % 3D AP50 in both bridges** — is measured on the benchmark and untouched.

---

<!-- _class: dense -->

## 15. What landed — and what each result settled

**Everything that was in flight has landed.** Five results, in order of how much they change:

| what | what it settled |
|---|---|
| **Views per scene, 17 → 50** | The last unmatched *evaluation* axis. **It moved the headline** (slide 9): at their own budget we lead on all three columns. |
| **The two no-ScanNet runs** | The last unmatched *training* axis. **It priced the asymmetry, and it now OPENS the deck** (slide 8): without ScanNet we are ~4× behind, so the lead rests on training data they do not use. |
| **The ablation table on the 3D ruler** | Both consistency levers now have 3D numbers (slide 12). The 2D ordering held; its **spacing** did not. |
| **Formal identity metrics + their seed spread** | **Retired a claim** (slide 14): no published identity metric separates 3D anchors from the control. |
| **RE10K** (**SAM2-supervised** — masks are model output, not GT) | Its **sign flips**: −42 % AP50 added to a mixture that has ScanNet, **+1.8×** added to one that does not. Redundant where ScanNet is, valuable where it is not. |

**The one that is worth a sentence of its own: RE10K's sign depends on what else is in the mixture.** Two compute-matched pairs on the same ruler. *With* ScanNet present, adding 1500 SAM2-supervised RE10K scenes costs **42 % of the unposed AP50** and is negative on all 8 matrix cells. *Without* ScanNet, the same 1500 scenes are worth **1.8× unposed and 2.1× posed**, and help on every cell. RE10K supplies real-world diversity that **ScanNet already supplies better** — redundant where ScanNet is present, and at fixed compute redundancy displaces; the best available proxy where it is absent. Neither half alone supports a claim about what RE10K is worth; the 2×2 does.

---

<!-- _class: dense -->

## 16. Open, and permanently out of reach

**Open and costed — the highest-value data item left:**

- **A partial ASE download is affordable, ASE is *not* unobtainable, and the job is WRITTEN.** The public Aria Synthetic Environments release ships **2D instance segmentation ground truth** — exactly the supervision we train on — and downloads **by scene range**. At ~230 MB/scene a **1000-scene pilot is ~230 GB**, which fits our quota. `slurm/fetch_ase.sh` fetches it in blocks, verifies each chunk's sha1, measures the inode cost, probes the shell-cap distribution and packs one tar; the 2D builder has an `ase` source with CPU tests. **The one remaining step is a signature**: the CDN urls arrive only after the Project Aria licence is accepted, which is the account holder's act, not the pipeline's. It would turn our IGGT replication from "their mixture minus ASE" into the complete one — i.e. it is what would let slide 8's second row be read as a *method* comparison instead of a data one.

**Permanently out of reach — state it, do not promise it:**

- **FAST3DIS's exact training set.** Not the data — the **scene list**: 40 % of it is unpublished. Every FAST3DIS comparison is therefore a cross-training-set comparison, at any download size.
- **InsScene-15K is incomplete** — only Infinigen / RE10K / ScanNet++ are published, the Aria portion is absent. Any replication is **partial** and must say so.
- **FAST3DIS never states which scenes it evaluates.** We do not claim identical evaluation sets.
- **Training compute** — ~0.8 vs ~16 GPU-days, and it is a *strength*, not an excuse.

---

<!-- _footer: "The three 3D rulers — the only numbers here that face the field" -->

<!-- _class: dense -->

## 17. Where we stand

| Ruler | training data | Our best | vs | Verdict |
|---|---|---|---|---|
| **3D official, UNPOSED, class-agnostic, 50 views** | **not matched — favours us** | **0.053 / 0.170 / 0.542** | FAST3DIS 0.038 / 0.096 / 0.316 · IGGT\* 0.028 / 0.112 / 0.287 | **lead on all three, at their own view budget** |
| **〃 with ScanNet removed** — the no-ScanNet run, 17 views | **approximated (no ASE)** | **0.005 / 0.023 / 0.251** | 〃 | **~4× behind** — the lead's price, measured |
| 3D official, POSED, class-aware † | **MATCHED** — the same 1201 split | 0.088 / 0.260 / 0.572 *(best posed row)* | SegVGGT 0.504 / 0.717 / 0.870 | behind; **2.3× is the bridge**, and on the `--anchor_3d` row the residual is **×2.8 — the training-matched number** |
| 3D on ScanNet200 / ScanNet++ / Replica | n/a — no like-for-like row held | 0.124 / 0.009 / 0.006 AP (posed) | — | zero-shot fails unposed, survives posed → **geometry, not masks** |

\* as re-evaluated by FAST3DIS: IGGT publishes no ScanNet AP of its own. † class-aware because that is what SegVGGT publishes; the class-agnostic scaling runs have no class-aware column at all, so they cannot appear on that row — not because they score worse.

**The one-line read, in the order the table is meant to be read:** where the training data is matched or approximated we are **behind** — ×2.8 against SegVGGT, ~4× against FAST3DIS/IGGT. The **lead** on row 1 is real and matched on evaluator, bridge, label setting and views, and rests on data those two never use. What is ours is not the leaderboard position: a **strictly frozen backbone at ~0.8 GPU-days**, and a controlled ablation nobody else has run (slides 13–14).
