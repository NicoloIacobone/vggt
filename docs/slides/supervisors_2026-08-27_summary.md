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
## 3D multi-view-consistent instance segmentation — the short version

**Status, 2026-08-27.** Six slides. The full deck is `supervisors_2026-08-27.md`.

- **What it is.** VGGT-1B, **strictly frozen** — no finetuning, no LoRA. Its features are cached once per scene before training, so only the decoder is ever trained: **~0.8 GPU-days against the ~16** of the closest competitor.
- **What it is trained on.** **2D masks only** — no 3D label ever enters training. The headline run sees the official ScanNet v2 2D instance annotations and nothing else.
- **What it produces.** One query = one instance **across all views by construction**, so multi-view consistency is intrinsic to the query rather than obtained by fusion, tracking or mask matching.

---

<!-- _footer: "Official ScanNet 3D benchmark · UNPOSED (own predicted geometry) · class-agnostic · 50 views — the competitors' own setting" -->

<!-- _class: mid -->

## 2. The headline

| Method | Backbone | Views/scene | AP | AP50 | AP25 |
|---|---|---|---|---|---|
| IGGT *(as re-evaluated by FAST3DIS)* | adapted | 50 | 0.028 | 0.112 | 0.287 |
| FAST3DIS | LoRA-adapted DA3 | 50 | 0.038 | 0.096 | 0.316 |
| **Ours — 3D anchors, defaults, trained on ScanNet only** | **frozen VGGT-1B** | **50** | **0.053** | **0.170** | **0.542** |
| **Ours + EXTRA TRAINING DATA** (ScanNet + ScanNet++ + Infinigen) | 〃 | **50** | **0.069** | **0.193** | **0.560** |

**At the competitors' own 50-view budget we lead on all three columns** — 1.39× / 1.77× / 1.72× on FAST3DIS, more on IGGT — from a **strictly frozen** backbone, with every lifting parameter at its default.

**Two things this needs said out loud.** The view budget is now **matched, not conceded**: at the 17-view budget everything before today was produced at, the AP column was a *tie* with FAST3DIS. And more views are not an open lever — 50 → 71 is flat-to-negative, so it saturates exactly where they report.

---

<!-- _footer: "Every competitor-facing row is produced under the competitor's OWN setting" -->

<!-- _class: mid -->

## 3. Under what setting — and the two gaps

**Matched, axis by axis:** the official ScanNet 3D evaluator, **vendored**; the unposed bridge (own predicted depth + cameras, Sim(3)+ICP) that FAST3DIS and IGGT use; SegVGGT's posed "geometric GT" bridge, **certified at 99.99 %** rather than assumed; both label settings computed for every run; all four benchmarks; **50 views**, closed today; and SegVGGT's own official 1201-scene train split.

**Not matched — two axes, and they do not run in the same direction:**

| axis | state |
|---|---|
| **training data** (FAST3DIS, IGGT) | **not matched, it FAVOURS us, and it is now measured.** Both are zero-shot on ScanNet; every headline row of ours trains on it. With ScanNet removed we score **0.023 AP50 against their 0.096 / 0.112** — the lead rests on training data they do not use, and that belongs next to the lead. ⚠ It does *not* show the recipe loses at equal data: that arm is missing ASE entirely, 3819 scenes against ~100 k. |
| **training compute** | ~0.8 vs ~16 GPU-days — **permanently unmatchable, and a strength, not an excuse** |

**One asymmetry runs the other way:** our backbone is strictly frozen where both of theirs are adapted.

---

<!-- _class: mid -->

## 4. What is ours, and what the field already owns

**"Frozen VGGT + a decoder for a downstream 3D task" is the dominant pattern of the last ~12 months. The architecture alone is not a contribution** — 3D-anchored queries are FAST3DIS's, queries shared across views are SegVGGT's.

1. **The controlled comparison nobody has run.** One backbone, one dataset, one protocol, decoder ingredients varied one at a time — including **3D anchors vs 2D boxes inside the same decoder**, which no published paper has put head-to-head.
2. **Competitive 3D results from a strictly frozen backbone**, at ~0.8 GPU-days against ~16, with no adaptation of any kind.
3. **Consistency intrinsic to the query, not post-hoc — and now measured on a published ruler.** The evaluation reports **HOTA / AssA / DetA / IDF1**, the tracking literature's own metrics, with a bundle's views read as timesteps and one query as one track. That mapping is exact rather than invented, which is the point. On the headline checkpoint: **HOTA 0.42, AssA 0.58, DetA 0.31, IDF1 0.49**.

⚠ **Switching to them already cost us a claim, and that is the exercise working.** Across two seeds, the secondary claim that 3D anchors improve cross-view *identity* **does not hold** — every published metric moves by less than its own seed spread, while only our own `id_switch` sees an effect. The mechanism's real result, **+66 % 3D AP50**, is measured on the benchmark and untouched.

---

<!-- _class: mid -->

## 5. What landed, and what each result settled

**Everything that was in flight has landed.** In order of how much each changed:

| what | what it settled |
|---|---|
| **Views per scene, 17 → 50** | The last unmatched *evaluation* axis. **It moved the headline**: at their own budget we lead on all three columns. |
| **The two no-ScanNet arms** | The last unmatched *training* axis. **It priced the asymmetry**: without ScanNet we are ~4× behind, so the lead rests on data they do not use. |
| **The ablation table on the 3D ruler** | Both consistency levers now have 3D numbers: cross-frame attention **−57 % AP50**, per-frame features −24 % class-aware / −49 % class-agnostic. |
| **Formal identity metrics + seed spread** | **Retired a claim**: no published identity metric separates 3D anchors from the control. |
| **RE10K** (**SAM2-supervised**) | Its **sign flips** — −42 % AP50 added to a mixture with ScanNet, **+1.8×** added to one without. |

**The result worth a sentence of its own.** RE10K supplies real-world diversity that **ScanNet already supplies better**: redundant where ScanNet is present — and at fixed compute redundancy *displaces*, costing 42 % of the unposed AP50 — but the best available proxy where ScanNet is absent, where the same 1500 scenes are worth 1.8–2.1×. Neither half alone supports a claim about what RE10K is worth; the 2×2 does.

---

<!-- _class: mid -->

## 6. What is still open

**Open and costed — the highest-value data item left.** **ASE is *not* unobtainable**: the public Aria Synthetic Environments release ships **2D instance segmentation ground truth** — exactly the supervision we train on — and downloads **by scene range**. At ~230 MB/scene a **1000-scene pilot is ~230 GB**, which fits our quota. It would turn our IGGT replication from "their mixture minus ASE" into the complete one, and it is the only route to training on FAST3DIS's own source.

**Permanently out of reach — stated, not promised:**

- **FAST3DIS's exact training set.** Not the data — the **scene list**: 40 % of it is unpublished, so every FAST3DIS comparison stays a cross-training-set comparison at any download size.
- **InsScene-15K is incomplete** as published; any replication is **partial** and must say so.
- **FAST3DIS never states which scenes it evaluates.** We do not claim identical evaluation sets.

**Where the remaining distance is.** Against SegVGGT's posed numbers we are behind by **×6.4** on the 3D-anchor row: **×2.3 of that is the evaluation bridge** and **×2.8 is real**, bought with a LoRA-adapted backbone, more views and 259×196 masks against our 37×37. A fourth candidate explanation — their 600 kept queries against our 100 — is **measured neutral** and struck off.
