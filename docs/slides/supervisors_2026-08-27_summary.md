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
## 3D multi-view-consistent instance segmentation — the short version

**Status.** Six slides. The full deck is the long-form version of this one.

- **What it is.** VGGT-1B, **strictly frozen** — no finetuning, no LoRA. Its features are cached once per scene before training, so only the decoder is ever trained: **~0.8 GPU-days against the ~16** of the closest competitor.
- **What it is trained on.** **2D masks only** — no 3D label ever enters training. The headline run sees the official ScanNet v2 2D instance annotations and nothing else.
- **What it produces.** One query = one instance **across all views by construction**, so multi-view consistency is intrinsic to the query rather than obtained by fusion, tracking or mask matching.

---

<!-- _footer: "Official ScanNet 3D benchmark · UNPOSED (own predicted geometry) · class-agnostic · 50 views — the competitors' own setting" -->

<!-- _class: tighter -->

## 2. The headline — and the axis it is not matched on

| Method | trains on ScanNet? | Backbone | Views | AP | AP50 | AP25 |
|---|---|---|---|---|---|---|
| IGGT *(as re-evaluated by FAST3DIS)* | **no** | adapted | 50 | 0.028 | 0.112 | 0.287 |
| FAST3DIS | **no** | LoRA-adapted DA3 | 50 | 0.038 | 0.096 | 0.316 |
| **Ours — 3D anchors, defaults, trained on ScanNet only** | **yes** | **frozen VGGT-1B** | **50** | **0.053** | **0.170** | **0.542** |
| **Ours + EXTRA TRAINING DATA** (ScanNet + ScanNet++ + Infinigen) | yes | 〃 | **50** | **0.069** | **0.193** | **0.560** |
| **Ours with ScanNet REMOVED** — IGGT's mixture COMPLETE | **no** | 〃 | 17 | **0.009** | **0.032** | **0.301** |

**At the competitors' own 50-view budget we lead on all three columns** — 1.39× / 1.77× / 1.72× on FAST3DIS, more on IGGT — from a **strictly frozen** backbone, with every lifting parameter at its default.

**The last row is not a footnote, it is how the first four are read.** Evaluator, bridge, label setting and view budget are matched; **training data is not**, and it runs in our favour. Removing ScanNet costs a factor **4.3 in AP50** and turns the lead into ~3× behind — measured on **IGGT's mixture reproduced in full**, ASE included. On **AP25 the gap closes** (0.301 vs 0.316 / 0.287) at a third of their views. ⚠ It does *not* show the recipe loses at equal data — 1000 ASE scenes against ~100 k, ~0.8 GPU-days against ~16.

**And on the ONE comparison where the training data IS matched — SegVGGT, our exact 1201 split — we are behind by ×2.8** once the ×2.3 evaluation bridge is taken out, measured on the `--anchor_3d` checkpoint (the residual is checkpoint-dependent; slide 6).

---

<!-- _footer: "Every competitor-facing row is produced under the competitor's OWN setting" -->

<!-- _class: dense -->

## 3. Under what setting — and the two gaps

**Matched, axis by axis:** the official ScanNet 3D evaluator, **vendored**; the unposed bridge (own predicted depth + cameras, Sim(3)+ICP) that FAST3DIS and IGGT use; SegVGGT's posed "geometric GT" bridge, **certified at 99.99 %** rather than assumed; both label settings computed for every run; all four benchmarks; **50 views**; and SegVGGT's own official 1201-scene train split.

**Not matched — two axes, and they do not run in the same direction:**

| axis | state |
|---|---|
| **training data** (FAST3DIS, IGGT) | **not matched, it FAVOURS us, and it is now measured on their COMPLETE mixture.** Both are zero-shot on ScanNet; every headline row of ours trains on it. With ScanNet removed we score **0.032 AP50 against their 0.096 / 0.112** — ~3× behind — while **matching them on AP25** (0.301 vs 0.316 / 0.287) at a third of their views. The lead rests on training data they do not use, and that belongs next to the lead. ⚠ It does *not* show the recipe loses at equal data: 1000 ASE scenes against ~100 k. |
| **training data** (SegVGGT) | **MATCHED** — the official ScanNetv2 1201 split, identical. It is the one training-matched comparison in the deck, and we are **×2.8 behind** on it after the bridge is removed. |
| **training compute** | ~0.8 vs ~16 GPU-days — **permanently unmatchable, and a strength, not an excuse** |

**Read the two training rows together, because they are the same sentence twice: where the data is matched or approximated we are behind; the lead lives in the configuration where we train on the evaluation domain and they do not.**

**One asymmetry runs the other way:** our backbone is strictly frozen where both of theirs are adapted.

---

<!-- _class: dense -->

## 4. What is ours, and what the field already owns

**"Frozen VGGT + a decoder for a downstream 3D task" is the dominant pattern of the last ~12 months. The architecture alone is not a contribution** — 3D-anchored queries are FAST3DIS's, queries shared across views are SegVGGT's.

1. **The controlled comparison nobody has run.** One backbone, one dataset, one protocol, decoder ingredients varied one at a time — including **3D anchors vs 2D boxes inside the same decoder**, which no published paper has put head-to-head.
2. **The first measurement of what a *strictly frozen* backbone reaches on this task.** Everyone else LoRA-adapts; nobody reports the unadapted case. At ~0.8 GPU-days against ~16 and 1201 scenes against ~100 k, it leads two adapted competitors in-domain and does **not** on their own training setting (slide 2). Both halves are the finding — *"competitive 3D results"* on its own would drop the data axis.
3. **Consistency intrinsic to the query, not post-hoc — and now measured on a published ruler.** The evaluation reports **HOTA / AssA / DetA / IDF1**, the tracking literature's own metrics, with a bundle's views read as timesteps and one query as one track. That mapping is exact rather than invented, which is the point. On the headline checkpoint: **HOTA 0.42, AssA 0.58, DetA 0.31, IDF1 0.49**.

⚠ **Switching to them already cost us a claim, and that is the exercise working.** Across two seeds, the secondary claim that 3D anchors improve cross-view *identity* **does not hold** — every published metric moves by less than its own seed spread, while only our own `id_switch` sees an effect. The mechanism's real result, **+66 % 3D AP50**, is measured on the benchmark and untouched.

---

<!-- _class: mid -->

## 5. What landed, and what each result settled

**Everything that was in flight has landed.** In order of how much each changed:

| what | what it settled |
|---|---|
| **Views per scene, 17 → 50** | The last unmatched *evaluation* axis. **It moved the headline**: at their own budget we lead on all three columns. |
| **The no-ScanNet runs, now with ASE** | The last unmatched *training* axis, **closed on composition**. **It priced the asymmetry and it sits ON the headline slide**: ~3× behind on AP50, **level on AP25**; the lead rests on data they do not use. |
| **The ablation table on the 3D ruler** | Both consistency levers now have 3D numbers: cross-frame attention **−57 % AP50**, per-frame features −24 % class-aware / −49 % class-agnostic. |
| **Formal identity metrics + seed spread** | **Retired a claim**: no published identity metric separates 3D anchors from the control. |
| **RE10K** (**SAM2-supervised**) | Its **sign flips** — −42 % AP50 added to a mixture with ScanNet, **+1.8×** added to one without. |

**The result worth a sentence of its own.** RE10K supplies real-world diversity that **ScanNet already supplies better**: redundant where ScanNet is present — and at fixed compute redundancy *displaces*, costing 42 % of the unposed AP50 — but the best available proxy where ScanNet is absent, where the same 1500 scenes are worth 1.8–2.1×. Neither half alone supports a claim about what RE10K is worth; the 2×2 does.

---

<!-- _class: dense -->

## 6. What is still open

**CLOSED — the highest-value data item, and it paid.** **ASE was never unobtainable**: the public Aria release ships **2D instance segmentation GT** and downloads **by scene range**. Licence accepted, **1000 scenes fetched with zero failed blocks** (31 897 frames, 78 347 instances), and arm **I-ase** trained on it. Worth **×1.4 AP50** on the competitor cell, and it **closed the AP25 gap**. Our IGGT replication is now their mixture **complete**, not "minus ASE" — so that row reads as a *method* comparison. It does **not** close the scale gap: 1000 scenes to ~100 k.

**Still open, in spending order:** **ASE *with* ScanNet** (every ASE number above is from a no-ScanNet mixture, and RE10K's sign flipped on exactly that) — then **the lifting, not the decoder**: the same masks cost ×2.3 under the unposed bridge.

**Permanently out of reach — stated, not promised:**

- **FAST3DIS's exact training set.** Not the data — the **scene list**: 40 % of it is unpublished, so every FAST3DIS comparison stays a cross-training-set comparison at any download size.
- **InsScene-15K is incomplete** as published; any replication is **partial** and must say so.
- **FAST3DIS never states which scenes it evaluates.** We do not claim identical evaluation sets.

**Where the remaining distance is.** Against SegVGGT's posed numbers we are behind by **×6.4** on the 3D-anchor row: **×2.3 of that is the evaluation bridge** and **×2.8 is real**, bought with a LoRA-adapted backbone, more views and 259×196 masks against our 37×37. A fourth candidate explanation — their 600 kept queries against our 100 — is **measured neutral** and struck off.
