# Results — one table per protocol

Every number in this file is on the **official ScanNet v2 1201/312 split or larger**, and on one
of two rulers. **The single most common mistake in this project is comparing across them.** Read
§1 before quoting anything.

## 1. The two rulers

| | 3D — the official benchmark (§5, §7) | 2D — internal, per-frame / per-bundle (§6) |
|---|---|---|
| unit scored | one instance in the scene's **point cloud**, official ScanNet evaluator, vendored | one query's mask, on VGGT's 37×37 patch grid, our own metric code |
| what it is for | **the only numbers placeable next to a published paper** | model selection and ablation ranking |
| per-frame vs per-bundle | — | per-frame scores each view alone; per-bundle scores one query's whole [S, h, w] volume against the bundle's GT |

Per-frame scores **higher** than per-bundle for the same checkpoint: an instance only has to match
in the views where it is visible, and a prediction that claims no pixels in a view is dropped
rather than penalised (`train/perframe.py::drop_empty_masks`).

**A 2D number may never be placed next to a competitor's**, and a §7 row (ScanNet200 / ScanNet++ /
Replica, class-agnostic only) may never be read next to a class-aware ScanNetv2 one.

### 1.1 – §4 — moved to the archive (2026-08-27)

The project-val split (scenes 0080–0089), the retired baseline head's own tables, the SAM3
ground-truth era and the COCO port check are **no longer reported**. They are in
`docs/old/RESULTS_HISTORY.md` for provenance and are not quotable. Everything this project
reports is on the official ScanNet v2 1201/312 split or larger, and starts at §5.

Section numbering below is unchanged, so every existing cross-reference to §5–§8 still resolves.


## 5. The 3D ruler — official ScanNet 3D instance benchmark (docs/MASKDINO.md §9)

**A separate protocol on purpose. Nothing here compares to §2 or §3, in either direction** — but
unlike them, this section's numbers ARE placeable next to published ones, because the metric code
is the official evaluator itself, vendored (`train/benchmark3d.py`), on the official val-312 split
and the official benchmark point clouds. Per-view masks are unprojected with **VGGT's own
predicted depth + cameras** (no GT geometry at inference; eval-only Sim(3) registration),
majority-voted per superpoint, and scored as 3D instances.

**Which published numbers, though — the two-protocol rule (established 2026-08-04).** The
literature's 3D numbers are not one comparable set. They split by *how a finished 2D mask reaches
the point cloud*, and that step dominates the score:

- **Posed transfer — SegVGGT (0.504 / 0.717 / 0.870).** Its released evaluator never unprojects:
  it projects the GT benchmark cloud into each view using ScanNet's **GT poses and intrinsics**,
  resolving occlusion with the **ScanNet sensor depth** map. No Sim(3), no ICP, no vote radius —
  the 3D↔2D correspondence is exact by construction. That number measures **2D mask quality with
  a perfect 2D→3D bridge**. Their paper states it outright: *"We utilize the ground-truth depth
  maps and camera poses during this mapping stage for fair comparison."*
- **Unposed transfer — FAST3DIS (0.038 / 0.096 / 0.316), IGGT (0.028 / 0.112 / 0.287), and us.**
  Masks are unprojected with the model's **own predicted** depth and cameras. These numbers
  measure **2D mask quality × feed-forward geometry quality**. Their two rows are additionally
  **class-agnostic** and ours is class-aware — see the amendment under the table below.

The evaluator is *not* the difference — both are the official ScanNet one with identical options.
So FAST3DIS and IGGT cluster with us because they share our protocol; SegVGGT sits far above all
three because it is in the other one. This is **not** a charge of cheating: SegVGGT's *model*
takes unposed images only, exactly like ours, and using GT geometry solely to transfer finished
masks for scoring is a legitimate way to isolate segmentation from reconstruction quality. Full
evidence, with file:line references into their released code, is in `docs/RELATED_WORK.md`
("Two 3D protocols").

**And a third family, which is where the ScanNet numbers people remember come from.** SegVGGT's
own Table 1 lists Mask3D 55.2 / 73.7 / 85.3, Relation3D 62.5 / 80.2 / 87.0, SegDINO3D 64.0 /
81.5 / 88.9, ODIN 50.0 / 71.0 / 83.6 — all consuming a **reconstructed point cloud or RGB-D**,
not images. The one image-only baseline in that table, OneFormer3D†, scores **5.4 / 10.2 /
17.4**. When someone asks why our AP is "so low", that row is the answer: image-only 3D instance
segmentation is a different difficulty class, and we are above it.

Two structural handicaps to remember when reading the table: `otherfurniture` (1 of the 18
benchmark classes) is unpredictable for our 19-class head (background in our 2D GT — the
17-class diagnostic column isolates this), and coverage is bounded by the ~16–25
`scannet_frames_25k` frames per scene (2–24 is SegVGGT's *training* sampling; at **eval** they
take every 20th frame of a full `.sens` extraction, ~75–100 views per scene — so on coverage we
are not comparable to them, we are behind).

| Checkpoint | trained on | AP / AP50 / AP25 (18-class) | 17-class diagnostic | status |
|---|---|---|---|---|
| **multi-frame, official split (job 9386666 `checkpoint_best_bundle`), defaults — job 9503137** | **1201 official train (leak-free)** | **0.023 / 0.067 / 0.268** | 0.024 / 0.071 / 0.284 | **REPORTABLE** |
| **same, `--vote_radius 0.1 --depth_conf_percentile 25` — job 9503139** | 〃 | **0.029 / 0.083 / 0.305** | 0.030 / 0.088 / 0.323 | **REPORTABLE**, knobs tuned on the diagnostic run below |
| **`--anchor_3d` multi-frame (job 9634920 `checkpoint_best_bundle`), defaults — job 9670882** | **1201 official train (leak-free)** | **0.038 / 0.112 / 0.360** | 0.040 / 0.119 / 0.381 | **REPORTABLE — the best row in this table**, untuned |
| best multi-frame (job 9071415, ep-17 ckpt), defaults — job 9327269 | 0000–0489 (**overlaps val-312!**) | 0.013 / 0.041 / 0.223 | 0.014 / 0.044 / 0.236 | DIAGNOSTIC ONLY — leakage, §9.4 |
| same, `--vote_radius 0.1 --depth_conf_percentile 25` — job 9327271 | 〃 | 0.016 / 0.052 / 0.238 | 0.016 / 0.055 / 0.253 | 〃 |
| FAST3DIS (published; LoRA-adapted DA3), 50 views | official split | 0.038 / 0.096 / 0.316 | — | same **bridge** (unposed), but **CLASS-AGNOSTIC** — see below |
| IGGT, **as re-evaluated by FAST3DIS** (50 views) | official split | 0.028 / 0.112 / 0.287 | — | 〃. IGGT's own paper reports **no ScanNet AP** |
| SegVGGT (published; LoRA-adapted VGGT, 1201-scene train) | official split | 0.504 / 0.717 / 0.870 | — | **DIFFERENT protocol** (posed transfer) — not a like-for-like row |

**Among the methods that share our protocol, the reportable rows land in FAST3DIS's ballpark:
AP25 0.305 vs its 0.316, AP50 0.083 vs 0.096, AP 0.029 vs 0.038** — with a *strictly frozen*
backbone against its LoRA-adapted DA3, and comparably to IGGT. ⚠ **This sentence compares across
label settings and is superseded by the amendment below — use the class-agnostic column.**

**Amended 2026-08-06 — the FAST3DIS/IGGT rows are class-agnostic, ours are class-aware, and the
like-for-like column is now MEASURED** (jobs 9861563 / 9861564; docs/MASKDINO.md §9.11).
FAST3DIS §4.4 scores ScanNet with *"the semantic class labels ignored"* and publishes no
class-aware number; IGGT's row comes from that same table. SegVGGT and our columns above are the
18-class, per-class-mean official evaluation. The metric *definition* is identical everywhere
(AP over IoU 0.5:0.05:0.95, AP50/AP25 at fixed thresholds — `mAP` in SegVGGT's table and `AP` in
FAST3DIS's are the same quantity), so the difference is the **setting**, not the metric.

| unposed row | class-aware (18, the headline) | **class-agnostic** (comparable to FAST3DIS/IGGT) |
|---|---|---|
| ours, defaults | 0.023 / 0.067 / 0.268 | **0.013 / 0.050 / 0.320** |
| ours, tuned lifting knobs | 0.029 / 0.083 / 0.305 | **0.017 / 0.060 / 0.334** |
| **ours, `--anchor_3d`, untuned** (job 9866391) | 0.038 / 0.112 / 0.360 | **0.042 / 0.138 / 0.504** |
| **〃 `--seed 1` replicate** (job 9979100) | 0.037 / 0.112 / 0.342 | **0.039 / 0.129 / 0.485** |
| control, `--seed 1` replicate (job 9979101) | 0.025 / 0.075 / 0.313 | 0.016 / 0.059 / 0.348 |
| 〃 best lifting knob (`--vote_radius 0.15`, sensitivity) | 0.048 / 0.151 / 0.419 | **0.055 / 0.185 / 0.571** |
| ours, `--num_frames 16` (job 9901143) | 0.033 / 0.098 / 0.336 | 0.023 / 0.080 / 0.391 |
| ours, `--num_frames 16`, 20 ep (job 9901664) | 0.032 / 0.115 / 0.414 | 0.029 / 0.104 / 0.458 |
| FAST3DIS (published) | — | 0.038 / 0.096 / 0.316 |
| IGGT (via FAST3DIS) | — | 0.028 / 0.112 / 0.287 |

**Read it as, on the headline checkpoint: like-for-like we LEAD on AP25 (0.334 vs 0.316 and
0.287) and TRAIL ~1.6–2.2× on AP50 and AP.** The earlier "in FAST3DIS's ballpark on AP and AP50"
reading was an artefact of comparing across settings and is struck.

**On the `--anchor_3d` checkpoint the collapse goes the other way and the row leads
(2026-08-06, job 9866391, todo 1e CLOSED): 0.042 / 0.138 / 0.504 vs FAST3DIS's 0.038 / 0.096 /
0.316 and IGGT's 0.028 / 0.112 / 0.287 — on a strictly frozen backbone, untuned, at ~17
views/scene against FAST3DIS's 50.** This is the strongest publishable row in the
project. The sign of the class-collapse is **checkpoint-dependent**, not a property of the
setting: it costs a head whose class-aware mean leans on one rare class (toilet 0.508) and pays a
head whose instances are fewer and more view-consistent. Measured on four checkpoints
(2026-08-07), **only `--anchor_3d` gains** — the control and both bundle-width runs lose — so it
tracks that mechanism, not the multi-frame recipe. Carry one caveat: unposed protocol only (§5.1).

**How to word the lead — settled 2026-08-07 by the seed-1 3D replicate (§5.2).** Two seeds put us
at 0.042 / 0.138 / 0.504 and 0.039 / 0.129 / 0.485. So: **lead on AP50 (1.34–1.44× FAST3DIS,
1.15–1.23× IGGT) and on AP25 (1.53–1.59× and 1.69–1.76×), lead IGGT on AP, and TIE FAST3DIS on AP**
(0.039–0.042 vs 0.038, inside our own 0.003 seed spread on that column). The earlier
"ahead on all three" was a seed-0-only reading and must not be repeated.

**And the lead is not a tuning artefact — the whole knob grid has it** (docs/MASKDINO.md §9.8.1,
jobs 9901146–52). Re-sweeping the two lifting knobs on this checkpoint spans 0.138 → **0.185**
class-agnostic AP50; the *worst* point is the untuned default and is still 1.44× FAST3DIS's 0.096,
the best is **0.055 / 0.185 / 0.571** = 1.45× / 1.93× / 1.81× on FAST3DIS and 1.96× / 1.65× /
1.99× on IGGT. This exactly inverts §9.8's earlier finding on the previous checkpoint, where the
whole grid stayed *below* 0.096. The sweep runs on val-312, so it stays a **sensitivity analysis
and the headline remains the defaults row** — but the comparison claim no longer depends on which
point is picked. Also measured there: `--eval_topk` 100 → 600 (the count SegVGGT and FAST3DIS use)
is **neutral**, 0.138 → 0.140, which strikes one item off the list of explanations for the
SegVGGT gap.
Collapsing the labels *lowers* our AP and AP50 rather than raising them, because it replaces the
per-class mean — carried by rare, distinctive classes (toilet 0.508 AP50 for 1/18 of the mean,
sink and refrigerator 0.173) — with one instance-pooled ranking dominated by the numerous weak
classes (chair 0.053, cabinet 0.040) and by `otherfurniture`, which our 19-class head scores
0.000 on. Every 3D run now emits `results_class_agnostic` (`train/benchmark3d.py`: same
predictions, labels collapsed onto one class); **quote that column, never the 18-class one, when
a row sits next to FAST3DIS or IGGT.** Provenance in `docs/RELATED_WORK.md`.

**Updated 2026-08-04 by todo 2d.** The `--anchor_3d` row is no longer "in the ballpark" — at
**0.038 / 0.112 / 0.360 it matches FAST3DIS on AP (0.038) and exceeds it on AP50 (0.112 vs 0.096)
and AP25 (0.360 vs 0.316)**, and exceeds IGGT on AP and AP25 while matching its AP50 — still on a
strictly frozen backbone, and *untuned* (all lifting knobs at their defaults, so the §9.8 sweep's
headroom is unexplored on it). Three honesty notes that must travel with that sentence: it is a
**single run against a single control**; the structural handicaps above (`otherfurniture`,
and the view budget of the run itself) still apply; and (2026-08-06) **their column is
class-agnostic and ours is class-aware**, so "exceeds FAST3DIS" is a cross-setting statement.
**Job 9866391 has since landed the class-agnostic column for this row and the claim SURVIVES:
0.042 / 0.138 / 0.504 — lead on AP50/AP25, lead IGGT on AP, tie FAST3DIS on AP** (table above,
wording note under it, seed replicate in §5.2). The earlier
expectation that it would weaken — reasoning from the headline checkpoint's 0.083 → 0.060 loss —
was wrong. The comparison to SegVGGT below is
unchanged, because its protocol difference is unaffected by which checkpoint we bring. SegVGGT's much higher number is
**not a like-for-like gap**: it is scored under posed transfer, where GT poses, intrinsics and
sensor depth carry the masks onto the point cloud with no geometry error at all, so it measures
2D mask quality alone while ours measures 2D mask quality times feed-forward geometry quality.
Quote it as a different protocol, never as an order-of-magnitude deficit — and quote it fairly:
their model is as unposed as ours, and isolating segmentation from reconstruction is a defensible
choice.

**The leak-free checkpoint beats the leaked one** (0.083 vs 0.052 AP50 at identical knobs) — a
result worth stating plainly: 1201 official train scenes outweigh the advantage of having *seen
the val scenes*, which is the strongest evidence yet that this track is data-limited rather than
architecture-limited (the same conclusion §2's 2D scaling curve reached). It also means the
diagnostic rows were a *pessimistic* proxy, not an optimistic one.

Two caveats to carry when quoting: the knobs of the second row were selected on the leaked
diagnostic run, so the **defaults row (0.023 / 0.067 / 0.268) is the untuned number**; and both
carry the structural handicaps above (`otherfurniture`, frame coverage).

A clean 8-point sweep of the two lifting knobs on the leak-free checkpoint (docs/MASKDINO.md
§9.8 — a *sensitivity analysis*, not a headline: it is swept on val) spans **0.067 → 0.091
AP50**. The vote radius saturates at ~0.15 m = the median registration error, and the whole
grid stays below FAST3DIS's 0.096 — so the remaining gap is not a tuning artefact.

Diagnosis (docs/MASKDINO.md §9.5): AP25 ≈ 4× AP50 — geometry binds, not recognition. Median
Sim(3) camera-center RMS 0.14 m and ICP point RMS ~0.10 m are on the order of the vote radius,
so lifted masks miss the 0.5-IoU bar that the same model clears in 2D (0.650 per-frame AP50);
coverage caps recall (median ~16 % of vertices voted, ~65 % of annotated vertices assigned).
**The lifting step, not the decoder, is now the binding constraint on this ruler** — the +0.016
AP50 that the two lifting knobs alone bought (0.067 → 0.083) is larger than most decoder
ablations in §2.

### 5.1 POSED TRANSFER — a separate protocol block (docs/MASKDINO.md §9.10, 2026-08-04)

**These rows are not the rows above and may not be merged with them.** Everything in §5 so far
is *unposed* transfer. This block runs the *posed* one — SegVGGT's — on our own masks, via
`--transfer_mode gt_projection`: the benchmark mesh is projected into each view with ScanNet's
GT pose + GT intrinsics and gated on the sensor depth, so the 2D↔3D correspondence is exact and
no predicted geometry enters at all (VGGT's depth and camera heads are not even run). Same
checkpoint, same 17.4 frames/scene, same 97.6 queries/scene, same evaluator — **the bridge is
the only variable**.

The model is still image-only in both blocks: GT geometry transfers finished masks for scoring,
exactly as the Sim(3)+ICP does in the unposed block.

| Protocol | Checkpoint | AP / AP50 / AP25 (18-class) | 17-class diagnostic | coverage: voted / annotated-assigned |
|---|---|---|---|---|
| **unposed** (§5 headline, job 9503137/9532181) | 1201-train, leak-free | **0.023 / 0.067 / 0.268** | 0.024 / 0.071 / 0.284 | 0.153 / 0.635 |
| **posed** (job 9607206) | 〃 (identical file) | **0.060 / 0.156 / 0.408** | 0.064 / 0.166 / 0.432 | **0.342 / 0.791** |
| unposed, `--anchor_3d` (job 9670882) | 1201-train + 3D anchors | **0.038 / 0.112 / 0.360** | 0.040 / 0.119 / 0.381 | 0.177 / 0.666 |
| posed, `--anchor_3d` (job 9670883) | 〃 (identical file) | **0.104 / 0.257 / 0.504** | 0.110 / 0.273 / 0.534 | 0.404 / 0.821 |
| unposed, `--num_frames 16` (job 9901143) | 1201-train, S=16 | 0.033 / 0.098 / 0.336 | 0.035 / 0.104 / 0.356 | — |
| posed, `--num_frames 16` (job 9901663) | 〃 (identical file) | 0.083 / 0.216 / 0.488 | 0.088 / 0.229 / 0.517 | — |
| unposed, S=16 20 ep (job 9901664) | 1201-train, S=16, 20 ep | 0.032 / 0.115 / 0.414 | 0.034 / 0.122 / 0.438 | — |
| **posed, S=16 20 ep** (job 9901665) | 〃 (identical file) | **0.088 / 0.260 / 0.572** | 0.093 / 0.276 / 0.605 | — |
| — *oracle ceiling of the posed row* (job 9607210) | GT rendered back through the bridge | 0.828 / 0.948 / 0.974 | — | — / 0.906 |
| SegVGGT (published) | 1201-train, LoRA-adapted VGGT | 0.504 / 0.717 / 0.870 | — | — |

What each row measures: **unposed = 2D mask quality × feed-forward geometry quality** (the
FAST3DIS/IGGT comparison, and the headline). **Posed = 2D mask quality alone, with a perfect
bridge** (the SegVGGT comparison). Neither is "the real" number; print both or neither.

Three things to carry when quoting this block:

1. **The bridge costs 2.3× AP50** (0.067 → 0.156). That is the measured price of the
   no-GT-geometry-at-inference claim, and a hard **ceiling on lifting work** (todo 5b/5c):
   perfect registration and coverage reach 0.156, not SegVGGT's 0.717.
2. **The protocol explains part of the SegVGGT gap, not the gap.** Under their own bridge we
   score 0.156 against their 0.717 — factor 2.3 explained, factor ~4.6 real and remaining
   (LoRA-adapted backbone, ~75–100 views to our 17, 259×196 masks to our 37×37, 600 kept
   queries to our 100). Never round this off to "it's just the protocol".
3. **The posed row is licensed by an oracle, not by plausibility.** Rendering the 3D GT back
   through the same projection returns **99.99 %** of assigned annotated vertices to their own
   instance (`scripts/eval3d_projection_oracle.py`, all 312 scenes) — a wrong pixel mapping
   collapses that number. Its 0.948 AP50 is the ceiling the ~17-frame budget imposes on the
   posed protocol, which also says **view count is not what binds us at 0.156**.
4. **`--anchor_3d` moves BOTH blocks by ~+66 % AP50** (todo 2d, docs/MASKDINO.md §8.3), from an
   ablation that is *flat* on both 2D rulers (per-bundle AP50 0.525 → 0.527). Same 17.42
   frames/scene, same defaults, 0 failures; it keeps 9 % **fewer** queries and covers 16 % **more**
   vertices. Two consequences worth carrying: (a) the unposed row **0.038 / 0.112 / 0.360** now
   sits inside the published unposed cluster (FAST3DIS 0.038 / 0.096 / 0.316, IGGT
   0.028 / 0.112 / 0.287 — *their* columns class-agnostic, ours class-aware, §5) on a **frozen**
   backbone; (b) **`bundle_AP50` at S = 8 is a poor proxy
   for this ruler** — score any cross-view-identity mechanism here before judging it. Because both
   blocks move by the same factor, the 2.3× bridge cost of reading 1 is unchanged, and so is the
   ceiling it puts on todo 5b/5c (now ~0.257 rather than 0.156).
5. **Bundle width moves both blocks too, but less** (2026-08-07, docs/MASKDINO.md §8.4 reading 4):
   unposed 0.067 → 0.098 (+46 %), posed 0.156 → 0.216 (+38 %). A second, independent mechanism
   confirming reading 4 — what this ruler responds to is multi-view identity, and both flags that
   buy identity buy 3D AP. `--anchor_3d` still wins the unposed column (0.112 vs 0.098) at half
   the wall clock and on a 4090 rather than an A100. The 20-epoch width run posts the best posed
   row anywhere (0.088 / 0.260 / 0.572). The two flags have never been combined — todo 2f.

### 5.2 The 3D anchor result, replicated across seeds (2026-08-07, jobs 9979100 / 9979101)

Until this run every row in §5 was **one run against one control** — the standing weakness of the
whole track, and the one that mattered most, because §5 is the only section placeable next to
published work. Both `--seed 1` checkpoints (§6.1, trained 2026-08-07) were scored on the unposed
ruler at defaults: 312 scenes, 0 failures, **17.42 frames/scene in all four runs**.

| arm | seed | 18-class AP / AP50 / AP25 | **class-agnostic** | kept queries | voted vtx |
|---|---|---|---|---|---|
| control | 0 (9861563) | 0.023 / 0.067 / 0.268 | 0.013 / 0.050 / 0.320 | 97.6 | 0.153 |
| control | 1 (**9979101**) | 0.025 / 0.075 / 0.313 | 0.016 / 0.059 / 0.348 | 97.8 | 0.147 |
| `--anchor_3d` | 0 (9866391) | 0.038 / 0.112 / 0.360 | **0.042 / 0.138 / 0.504** | 89.0 | 0.177 |
| `--anchor_3d` | 1 (**9979100**) | 0.037 / 0.112 / 0.342 | **0.039 / 0.129 / 0.485** | 90.4 | 0.168 |

**Seed spread on the 3D ruler is ≈ 0.009 class-agnostic AP50 in both arms** — the same figure §6.1
measured on the 2D per-bundle metric. The anchor effect is **+0.088 / +0.070**, i.e. **~9× that
spread**, so the 3D gain is not a single-run artefact. Everything else replicates too: the
class-aware ΔAP50 (+67 % / +49 %), the *sign* of the class collapse (only the anchor arm gains,
both controls lose), and the mechanism's signature — **~8 % fewer kept queries at ~15 % more voted
vertices**, both seeds.

**What this changed in the wording:** the "single run against a single control" caveat is retired,
and the AP column is downgraded from a lead to a tie with FAST3DIS — see §5's wording note.

### 5.3 Stacking the two identity mechanisms — a measured NEGATIVE (todo 2f, 2026-08-12)

`--anchor_3d` (§5.2) and bundle width 8→16 (§6) each act on `bundle_id_switch` and each pay on the
3D ruler. Run **together** (job 9979913, one flag against 9668639) they do not compose. Both
bridges, all knobs default, 312 scenes, 17.42 frames/scene, 0 failures — mechanism and diagnostics
in `docs/MASKDINO.md` §8.5:

| checkpoint | unposed 18-class | unposed **agnostic** | posed 18-class | posed **agnostic** |
|---|---|---|---|---|
| **`--anchor_3d` S=8** | **0.038 / 0.112 / 0.360** | **0.042 / 0.138 / 0.504** | **0.104 / 0.257 / 0.504** | **0.109 / 0.304 / 0.677** |
| S=16, 20 ep | 0.032 / 0.115 / 0.414 | 0.029 / 0.104 / 0.458 | 0.088 / 0.260 / 0.572 | 0.081 / 0.252 / 0.644 |
| 2f = both | 0.032 / 0.109 / 0.353 | 0.041 / 0.139 / 0.504 | 0.082 / 0.236 / 0.501 | 0.098 / 0.297 / 0.679 |

A tie on the unposed class-agnostic column (±0.001, an order of magnitude inside the 0.009 seed
spread of §5.2) and a loss on the other three. **No headline moves: `--anchor_3d` alone remains the
row §8.2 quotes.** The signature is over-pruning — 2f keeps the fewest queries of any checkpoint
(82.2) and votes *fewer* vertices than `--anchor_3d` (0.155 vs 0.177), so it prunes harder and
covers less.

### 5.4 VIEW COUNT — the competitors' 50 views, measured (todo 6k, 2026-08-27)

Jobs 11841445/49/51/54/57/62/67, 312 scenes, 0 failures, all lifting knobs at defaults. The last
unmatched axis of the comparison: FAST3DIS and IGGT are scored on **50 uniformly sampled views**
and SegVGGT on **every 20th frame**, while every number above this line was produced at **17.42**
(`scannet_frames_25k` is every 100th frame). The dense export (`docs/DATASET.md` §2.5) closes it.

**Read the rows against the 17-view row of the SAME tar, not against §5's published rows.** The
25k export re-compressed its jpegs and the dense one carries the `.sens` bytes, so dense-vs-dense
is the single-variable form. That control is reassuring: `--anchor_3d` at 17 dense views scores
0.044 / 0.137 / 0.488 against 0.042 / 0.138 / 0.504 published — AP and AP50 inside noise, AP25
−0.016.

| checkpoint | views (mean) | unposed, class-agnostic | posed, class-agnostic | voted vtx | annot. assigned |
|---|---|---|---|---|---|
| `--anchor_3d` (ScanNet-only) | 16.98 | 0.044 / 0.137 / 0.488 | — | 0.171 | 0.667 |
| 〃 | **46.65** | **0.053 / 0.170 / 0.542** | **0.121 / 0.336 / 0.704** | 0.268 | 0.754 |
| 〃 | 71.31 | 0.052 / 0.166 / 0.536 | — | 0.308 | 0.787 |
| A-long (+ScanNet++ +Infinigen) | 16.98 | 0.056 / 0.161 / 0.503 | — | 0.241 | 0.715 |
| 〃 | **46.65** | **0.069 / 0.193 / 0.560** | **0.200 / 0.419 / 0.725** | 0.368 | 0.802 |
| *FAST3DIS (published)* | *50* | *0.038 / 0.096 / 0.316* | — | | |
| *IGGT (via FAST3DIS)* | *50* | *0.028 / 0.112 / 0.287* | — | | |
| *SegVGGT (published)* | *~75–100* | — | *0.504 / 0.717 / 0.870* (class-**aware**) | | |

"46.65" and "71.31" are the achieved means of `--num_frames 50` and `100`: 20 % of val scenes are
shorter than 50 dense frames. FAST3DIS's "50 uniformly sampled views" hits the same wall.

**1. The competitors' view budget is worth ~+24 % AP50, and then it saturates.** 17 → 50 views:
`--anchor_3d` 0.137 → 0.170 (+24 %), A-long 0.161 → 0.193 (+20 %). 50 → 71 views: 0.170 → 0.166,
i.e. **flat, slightly negative**. Whatever more views buy is exhausted by ~50, which is exactly
the budget the two unposed competitors report at.

**2. At MATCHED views the lead widens.** Against FAST3DIS's 0.038 / 0.096 / 0.316, the ScanNet-only
`--anchor_3d` row at 50 views is **1.39× / 1.77× / 1.72×**, and A-long **1.82× / 2.01× / 1.77×** —
against 1.10× / 1.44× / 1.59× at 17 views. The "we lead on a third of their views" caveat is
retired: we lead at their views too, and by more.

**3. This closes todo 5b (coverage) with a measured NEGATIVE.** More frames do buy coverage —
voted vertices 0.171 → 0.268 → 0.308, annotated-assigned 0.667 → 0.754 → 0.787, monotone — but AP
stops moving at 50 while coverage keeps rising to 71. So **coverage is no longer the binding
constraint** on the unposed column; §5's "the lifting step binds" now means registration (5c),
not frames. The oracle already said this for the posed column (§5.1 reading iii); it now holds
unposed too.

**4. The posed column moves more than the unposed one.** A-long posed 0.389 → **0.419** AP50 at
50 views is the best posed row in the project, and the distance to SegVGGT's 0.717 falls to
**1.71×** (from 1.84×). Their row is class-aware and ours class-agnostic, so this is a
*direction*, not a like-for-like ratio — §5.1's decomposition is where that gap is priced.

### 5.5 THE ABLATION TABLE ON THE 3D RULER — both levers (todo 6o, CLOSED 2026-08-28)

Job 11986399, 312 scenes, 0 failures, all lifting knobs at defaults, unposed. The first **Tier-1**
number for the mechanism that carries multi-view consistency: until now cross-frame attention was
measured only on the retired 2D project-val ruler, which is the hole §8.4 marks with ⚠.

**Single variable, verified rather than assumed:** a `config.json` diff of the two runs returns
exactly one differing key, `cross_frame_attn`. Same 1201 train scenes, same 312 val scenes, same
schedule, same everything else.

| checkpoint | 18-class AP/AP50/AP25 | class-agnostic |
|---|---|---|
| the control — `maskdino_sf_list1201_mf_20260802_133826` | 0.023 / 0.067 / 0.268 | 0.013 / 0.050 / 0.320 |
| **`--no-cross_frame_attn`** — `…_mf_noxframe_20260803_111855` | **0.010 / 0.029 / 0.167** | **0.005 / 0.021 / 0.214** |
| ratio vs control | 0.46× / **0.43×** / 0.62× | 0.38× / **0.42×** / 0.67× |
| **`--feature_mode single`** — `…singlefeat_20260827_192650` (job 12012326) | **0.020 / 0.051 / 0.251** | **0.007 / 0.025 / 0.234** |
| ratio vs control | 0.89× / **0.76×** / 0.93× | 0.52× / **0.51×** / 0.73× |

**1. Both mechanisms are worth about half the 3D AP50, and both are far larger than any decoder
ingredient.** Removing cross-frame attention costs **57 %** of the class-aware AP50; switching to
per-frame features costs **24 %** class-aware and **49 %** class-agnostic. Nothing in the decoder
(two-stage, encoder, denoising, box init) came close to either. The two levers the study leaned
on are now priced on the **same ruler as the headline** rather than on a retired 2D one.

**2. Cross-frame attention survives the label setting; per-frame features does not.** Cross-frame
attention reads 0.43× class-aware and 0.42× class-agnostic — the same number twice, so it is not
an artefact of the class collapse. `--feature_mode single` reads **0.76× class-aware but 0.51×
class-agnostic**, i.e. the two columns disagree by a factor of ~1.5 on how much it matters. That
is exactly the checkpoint-dependent collapse §9.11 documents, and it means **the bundle-features
row must always be quoted with its label setting**. Never quote "24 %" alone.

**3. The 2D ranking does not survive intact.** On the retired project-val ruler the two levers were
close (+0.183 vs +0.147 per-bundle AP50, a ratio of 1.24). On the 3D ruler they are close only in
the class-agnostic column (58 % vs 49 %); class-aware they are 57 % vs 24 %, a ratio of 2.4. **The
2D ordering was not wrong, but its *spacing* was — which is the reason the exercise was worth its
GPU time.**

**4. AP25 falls least in both rows** (0.62×/0.67× and 0.93×/0.73× against AP50's 0.43×/0.42× and
0.76×/0.51×). Objects are still found without either mechanism; what degrades is whether the fused
3D instance clears the 0.5-IoU bar — the same signature as everything else in this section: the
mechanisms buy mask/identity quality, and the lifting step is what converts it into AP50.

**Caveats that travel with the second row.** `--feature_mode single` needed a **new training run**
(job 11986440) because no leak-free checkpoint of that arm existed; it is single-variable against
the control (config diff: only `feature_mode`) and **schedule-matched, not convergence-matched** —
both runs peak at epoch 12 of 12, so neither had stopped improving. In 2D it reproduces the
retired figure on the new split: per-bundle AP50 0.5249 → 0.3589, i.e. −0.166 against the −0.147
recorded on project-val.

### 5.6 TRAINING-MATCHED — what the lead costs when ScanNet is removed (todo 6l, 2026-08-28)

The last unmatched *training* axis, measured. Arm I trains on **IGGT's mixture minus ASE**
(ScanNet++ 853 + Infinigen 1466 + RE10K 1500 = 3819 scenes) and **never on ScanNet**, which is
FAST3DIS's and IGGT's setting on this benchmark. Full detail, both arms and all 16 matrix cells:
`docs/MULTIDATASET.md` §12.3. Final `checkpoint.pth` (§12.1: best-bundle selection does not work
on a zero-shot ruler). Compute-matched to within 0.6 %.

**ScanNetv2, unposed, class-agnostic — the competitor-facing cell:**

| row | trains on ScanNet? | AP / AP50 / AP25 |
|---|---|---|
| FAST3DIS (published) | no — ASE only | 0.038 / 0.096 / 0.316 |
| IGGT (via FAST3DIS) | no — InsScene-15K | 0.028 / 0.112 / 0.287 |
| **ours, arm I** — IGGT's mixture minus ASE | **no** | **0.005 / 0.023 / 0.251** |
| ours, arm I-gt — minus RE10K too | no | 0.003 / 0.013 / 0.212 |
| *ours, headline, 17 views* | *yes* | *0.042 / 0.138 / 0.504* |
| *ours, headline, 50 views* | *yes* | *0.053 / 0.170 / 0.542* |

**1. The asymmetry was real and it was carrying most of the lead.** Removing ScanNet costs a factor
**6 in AP50** at the same view budget (0.138 → 0.023), turning a lead into being ~4× behind. Every
"we lead FAST3DIS/IGGT" row in this file is therefore a row produced **with ScanNet in training,
against two methods that never use it** — which §8.2 and the FACTSHEET have always declared, and
which is now priced instead of merely declared.

**2. It does NOT show our recipe is worse than theirs at equal data.** Arm I is missing **ASE
entirely** — FAST3DIS's whole training set and IGGT's largest component — because its scene list is
unpublished and it is 9.2 TB. This is **3819 scenes against their ~100 k**, frozen backbone against
adapted, ~0.8 GPU-days against ~16. The supportable claim is *"we cannot match their training
setting, and without ScanNet we are well behind"* — never *"our method loses at equal data"*, a
comparison that has not been run and cannot be here.

**3. AP25 survives far better than AP50** (factor 2 against factor 6). Without ScanNet the model
still finds and coarsely localises objects; what collapses is clearing the 0.5-IoU bar — the same
signature as §5.4 and §5.5.

**4. RE10K's sign flips with ScanNet's presence**, and this is where that shows: adding it to a
mixture *without* ScanNet is worth **1.8× unposed / 2.1× posed**, while adding it to one *with*
ScanNet costs 42 % (§5.5's sibling result, `docs/MULTIDATASET.md` §11.7/§12.3). Redundant where
ScanNet is present, valuable where it is not.

## 6. Official 1201/312 split — first runs (2026-08-02)

**A new ruler, on purpose.** Train = the full official ScanNet v2 train split (1201 scenes,
`scannet_official_gt_1201.tar.zst`), val = the full official val split (312 scenes,
`scannet_official_gt_val312.tar.zst`) — the exact 2D split every competitor trains on. Nothing
here is comparable to the 0080–0089-val tables in §2/§3 (the official val reads ~0.07 AP50
harder per-frame, consistent with the §1.1 read-out), and these are still per-view 2D-mask
numbers on our own metric code, so they are not leaderboard figures either (§1.2). The only
prior point on a comparable axis is job 8900194 (§1.1: 0.589 mIoU / 0.604 AP50) — and even that
comparison carries a caveat: its val was the 77-scene subset, not the full 312.

Both runs: the best recipe (`--bundles_per_scene 2 --color_jitter 0.2`), 12 epochs ≈ 28.8k
steps (~ the N=490 recipe budget of 29.4k), warmup 2, eval every epoch on all 312 val scenes.

| Run (job) | protocol | mIoU | AP50 | AP75 | mAP |
|---|---|---|---|---|---|
| single-frame + `--eval_full_res` (9329716) | per-frame, 37×37 grid | **0.624** | **0.662** | 0.487 | 0.459 |
| 〃 | per-frame, full 518×518 | 0.611 | 0.651 | 0.466 | 0.437 |
| multi-frame `--multi_frame --feature_mode bundle` (9386666) | per-frame | 0.623 | 0.650 | 0.470 | 0.443 |
| 〃 | **per-bundle (multi-view)** | **0.529** | **0.525** | 0.312 | 0.311 |
| … `--no-cross_frame_attn` (9503176) | per-frame | 0.576 | 0.588 | — | — |
| 〃 | per-bundle | 0.471 | 0.389 | 0.220 | — |
| … `--anchor_3d` (9634920, todo 2d) | per-frame | 0.611 | 0.641 | 0.462 | 0.436 |
| 〃 | per-bundle | 0.524 | 0.527 | 0.305 | 0.306 |
| … **`--num_frames 16`** (9668639, todo 2e) | per-frame | 0.627 | **0.662** | 0.475 | 0.450 |
| 〃 | **per-bundle** (val pinned to 8 views) | **0.549** | **0.552** | 0.339 | 0.332 |
| … S=16, b1, 24 ep, no jitter (9668726) | per-frame | 0.609 | 0.641 | 0.459 | 0.435 |
| 〃 | per-bundle (val pinned to 8 views) | 0.541 | 0.544 | 0.331 | 0.324 |
| … S=16, b2, 20 ep (9668652) | per-frame | 0.627 | **0.669** | 0.476 | 0.453 |
| 〃 | per-bundle — **16-view ruler, not comparable** | 0.561 | 0.594 | 0.356 | 0.351 |
| … **S=16 + `--anchor_3d`** (9979913, todo 2f) | per-frame | 0.616 | 0.646 | 0.466 | 0.442 |
| 〃 | per-bundle (val pinned to 8 views) | 0.527 | 0.536 | 0.314 | 0.314 |
| … `--seed 1` replicate of the control (9901125) | per-bundle | 0.542 | 0.534 | — | — |
| … `--seed 1` replicate of `--anchor_3d` (9901124) | per-bundle | 0.531 | 0.536 | — | — |

The 2f row is the **only** one in this table that stacks two winners, and it is *negative*:
against its one-flag control 9668639 it loses 0.016 per-bundle and 0.016 per-frame AP50 (~1.8× the
§6.1 seed spread) while `bundle_id_switch` improves only 0.385 → 0.375, inside the spread.
**The 3D ruler agrees** — 2f ties `--anchor_3d` alone on the unposed class-agnostic column
(0.041 / 0.139 / 0.504 vs 0.042 / 0.138 / 0.504) and loses on the other three (§5.3). The two
identity mechanisms do not compose; `--anchor_3d` alone stays the checkpoint to quote.

### 6.1 Seed variance — the number every Δ in this file must be read against (2026-08-07)

Every row in §2, §3 and §6 was a single run. Both arms of the `--anchor_3d` comparison were
re-trained with `--seed 1`, nothing else changed (docs/MASKDINO.md §8.3):

| run | per-frame AP50 | per-bundle AP50 | `bundle_id_switch` ↓ | `view_consistency` ↑ |
|---|---|---|---|---|
| control, seed 0 (9386666) | 0.6491 | 0.5249 | 0.4982 | 0.7167 |
| control, seed 1 (9901125) | 0.6505 | 0.5342 | 0.4710 | 0.7173 |
| `--anchor_3d`, seed 0 (9634920) | 0.6408 | 0.5271 | 0.4088 | 0.7229 |
| `--anchor_3d`, seed 1 (9901124) | 0.6466 | 0.5362 | 0.4074 | 0.7279 |

Both seed-1 checkpoints were then scored on the **3D ruler** as well — §5.2: the anchor's 3D gain
replicates at ~9× the seed spread, which is what retires the "single run" caveat on §5's headline.

**Seed-to-seed spread on per-bundle AP50 ≈ 0.009 in both arms.** Consequences: the large effects
in §3 (cross-frame attention 0.183, bundle features 0.147) and §6 (bundle width 0.027) are 3–20×
that spread and stand; the 3D-anchor per-bundle delta (+0.002) sits *inside* it, so **"AP-neutral"
is now a measured claim rather than an absence of evidence**. The identity effect replicates with
the same sign in both seeds (`id_switch` −0.089 and −0.064, 2.4–3.3× the control arm's own
spread), and so does the small per-frame cost.

**Bundle width 8 → 16 (todo 2e, docs/MASKDINO.md §8.4).** `--num_frames` is how many views share
one query set, and it had never moved off 8 while the 3D ruler runs the head at S ≈ 17.4. New
flag `--eval_num_frames` pins the **val** width so `bundle_*` keeps measuring an 8-view volume
while training widens — without it the metric's object changes and the row is uncomparable
(the 9668652 row above is exactly that case, and is marked).

| | control S=8 (9386666) | **S=16 (9668639)** | Δ |
|---|---|---|---|
| per-frame AP50 | 0.650 | **0.662** | +0.012 |
| per-bundle AP50 | 0.525 | **0.552** | **+0.027** |
| `bundle_view_consistency` ↑ | 0.717 | **0.726** | +0.009 |
| `bundle_id_switch` ↓ | 0.498 | **0.385** | **−0.113 (−23 % rel.)** |
| `bundle_num_matched` | 14.1 | 14.0 | ±0 |

One flag different. **Recognition flat, identity improved** — the same dissociation as the
`--no-cross_frame_attn` cut read forwards. And it is the *width*, not the extra frames: job
9668726 runs `--bundles_per_scene 1` at S=16, i.e. **the control's exact frame budget** (16
frames/scene/epoch) with `--color_jitter` inert, and still lands 0.544 per-bundle / 0.345
id_switch. Cost: 2× wall clock (11 h 26 vs 5 h 42), ~230 GB feature cache, A100 80 GB.

**The `--anchor_3d` ablation (todo 2d, docs/MASKDINO.md §8.3).** 3D anchors vs 2D DAB boxes,
`config.json` differing in exactly that one key. **AP-neutral, identity-positive**: per-bundle
AP50 0.525 → 0.527 and consistency 0.717 → 0.723 are flat, per-frame is mildly negative
(0.650 → 0.641), and the one systematic move is **`bundle_id_switch` 0.498 → 0.409** (−18 % rel.,
better in **12/12 epochs**). This is a **dissociation** from the `--no-cross_frame_attn` row above,
where identity and bundle AP50 moved together — so the two are not one axis, and `bundle_AP50`
alone cannot see what a 3D anchor does. Off by default; costs +15 % training time.

- **Scale holds up on the honest split**: 0.662 per-frame AP50 vs 8900194's 0.604 (+0.058, with
  ~3× the training scenes; 77-vs-312-scene val caveat above). The train/val gap at epoch 12
  (train AP50 0.878 vs val 0.662) says data is still the lever, matching the §2 scaling story.
- **The multi-view result transfers**: per-bundle 0.529 / 0.525 on the official split vs
  0.539 / 0.515 on the old one (different rulers — quote per split). Per-frame and per-bundle
  peaks fall on different epochs again (10 vs 12), so `checkpoint_best_bundle.pth` carries the
  multi-view number (docs/MASKDINO.md §8.2).
- **First cross-view consistency measurement** (docs/MASKDINO.md §6.6, todo 2c): at epoch 12,
  `bundle_view_consistency` **0.717** / `bundle_id_switch` 0.498 (14.1 matched
  instances/bundle) — a matched instance is explained by its own query in ~72 % of its visible
  views. Both improve monotonically over training (0.679→0.717 / 0.607→0.498 from epoch 6).
- **Cross-frame attention's job is identity, and the metric proves it** (job 9503176,
  docs/MASKDINO.md §7.8.1): removing the block leaves the number of matched instances unchanged
  (14.0 vs 14.1) and `view_consistency` nearly so (0.692 vs 0.717), but **`id_switch` jumps
  0.498 → 0.682** — in 68 % of views some *other* query fits better. Recognition is intact;
  identity is what breaks, and −0.136 bundle AP50 follows.
- Full-res vs grid stays −0.011 AP50, same as on the old split (§7.7's "recognition binds").
- The multi-frame `checkpoint_best_bundle.pth`
  (`output/maskdino_sf_list1201_mf_20260802_133826/`) is the leak-free checkpoint the 3D ruler
  (§5) has been waiting for.

### 6.2 Class-agnostic baseline on the official split (2026-08-10, job 10287578)

The reference row for the multi-dataset workstream — ScanNet-only, **`--class_agnostic`**, so that
"more data helped" can be told apart from "the taxonomy got easier". Same official 1201/312 split,
`--multi_frame --feature_mode bundle`, S=8, 16 epochs. Full context and the mixture it exists to
baseline: **docs/MULTIDATASET.md**.

| run | per-frame mIoU / AP50 | per-bundle mIoU / AP50 | `id_switch` ↓ | `view_consistency` ↑ |
|---|---|---|---|---|
| class-**aware** control, 12 ep (9386666) | 0.623 / 0.650 | 0.529 / 0.525 | 0.498 | 0.717 |
| class-**agnostic**, 16 ep (10287578) | 0.657 / 0.658 | 0.536 / **0.505** | 0.509 | 0.692 |

Not a like-for-like cell and **never quotable against §2/§3/§6 rows** — it is the anchor for
MULTIDATASET.md's rows and nothing else. It differs from the control **three** ways, not one:
one class vs 18, 16 vs 12 epochs, and **b1 vs b2** (`--bundles_per_scene 1 --color_jitter 0`, the
multi driver's defaults, = 19 216 steps over 8 views/scene against the control's 28 824 over 16).
Two of the three push this row down, so the −0.020 is the **most** the taxonomy collapse can cost,
not a measurement of it — read "small", never a Δ (MULTIDATASET.md §6 reading 1). It also says the
run **had not converged**: `bundle_AP50` climbs monotonically to the last epoch
(0.487 → 0.495 → 0.496 → 0.505).

### 6.3 The multi-dataset arm — first mixture (2026-08-12, job 10484000)

Same block, same ruler, same caveat as §6.2. 3520 train scenes (ScanNet 1201 + ScanNet++ 853 +
Infinigen 1466), 6 epochs = 21 120 steps — step-matched to §6.2 by design.

| run | steps | per-frame mIoU / AP50 | per-bundle mIoU / AP50 | `id_switch` ↓ | `view_consistency` ↑ |
|---|---|---|---|---|---|
| ScanNet-only, 16 ep (10287578) | 19 216 | 0.641 / **0.656** | 0.536 / **0.505** | **0.509** | **0.692** |
| mixture 3520, 6 ep (10484000) | 21 120 | 0.639 / 0.629 | 0.508 / 0.434 | 0.621 | 0.671 |

3D ruler on it (ScanNetv2 val-312, unposed, defaults, class-agnostic, job 10596569):
**0.008 / 0.026 / 0.240**.

**At matched total steps the mixture loses on the ScanNet ruler, and the arithmetic explains it**:
only 1201/3520 scenes are ScanNet, so each ScanNet scene got 6 passes against 16 — a 2.7× cut in
exposure to the domain val is drawn from. The reading, and what the mixture is actually for (the
§7 columns where a ScanNet-only head scores 0.000), is **docs/MULTIDATASET.md §9**; the three
step-matched data-scaling arms that replace this run are §10 there.

### 6.4 The data-scaling arms — one recipe, three mixtures (2026-08-22)

Same block and same caveat as §6.2/§6.3. Three runs at a **matched ~42 k gradient-step budget**,
identical recipe (`--class_agnostic --multi_frame --feature_mode bundle --anchor_3d`, S=8, b1),
differing only in the training mixture; val is the official ScanNet 312 in all three.

| arm | train data | scenes | steps | per-frame mIoU / AP50 | per-bundle mIoU / AP50 | `id_switch` ↓ | `view_consistency` ↑ |
|---|---|---|---|---|---|---|---|
| **B** (11435335) | ScanNet | 1201 | 42 035 | 0.654 / 0.675 | 0.557 / 0.548 | **0.441** | 0.707 |
| **C** (11435338) | + ScanNet++ | 2054 | 41 080 | **0.659 / 0.677** | **0.568 / 0.554** | 0.472 | **0.714** |
| **A** (11435332) | + Infinigen | 3520 | 42 240 | 0.630 / 0.628 | 0.521 / 0.479 | 0.531 | 0.693 |
| **A-long** (11498642) | = A, converged | 3520 | 84 480 | **0.676 / 0.704** | **0.600 / 0.604** | **0.414** | **0.734** |

**At a matched step budget more data looks bad, and that reading was a budget artefact.** At 42 k
steps real same-domain ScanNet++ is free (C − B = +0.006, inside §6.1's 0.009 spread) and synthetic
Infinigen appears to cost −0.075 — but A was the only arm that had not converged (best epoch = its
last, still ~+0.010/epoch, while B was flat over its last ten and C nearly so). **Doubling A's
budget to 84 480 steps makes the full 3520-scene mixture the best run of the whole block on every
2D axis**: per-bundle AP50 0.479 → **0.604** (+0.056 over C, 6× the seed spread), per-frame AP50
**0.704**, and the best `id_switch` (0.414) and `view_consistency` (0.734) measured anywhere.
A-long is itself converged (0.604 / 0.589 / 0.579 / 0.592 / 0.581 over its last five epochs).

**The one caveat, and the run that was meant to close it — which FAILED.** A-long is compared at
*convergence*, not at matched steps: it had 2× the gradient steps of B and C. B is saturated (flat
over 10 epochs, so more steps cannot rescue it) and C nearly so, but "more data" and "more compute"
are **still not separated** at the top end. C-long (2054 scenes, 40 epochs = 82 160 steps, job
11632049) was the step-matched control and **it destabilised**: 16 epochs of rising train loss,
final train loss 122.6 against arm C's 93.2 on identical data at half the steps, best epoch = last.
Its checkpoint prices nothing and is not tabled here. **The caveat therefore stands unresolved.**
The cause is the learning rate (the failure §11.3 isolated independently); the re-run is **C-long′**
at lr 5e-5 (job 11831105) against **A-long′** (11830142), step-matched and same-LR.
Full reading: **docs/MULTIDATASET.md §10.5**.

**Do not read this table as the result of the workstream.** Every cell is scored on ScanNet, which
is in-domain for all three arms; what the extra data was bought for is §7's out-of-domain columns.

#### 6.4.1 Arm D — the fourth source, **SAM2-supervised** (2026-08-25) — a DIVERGED run

**Its own block, and not a measurement of anything.** RE10K's masks are **SAM2 output, not ground
truth** (docs/MULTIDATASET.md §1.3), so this row could never be folded into A/A-long's; but it
must not be quoted as what RE10K is worth either, because **the run diverged**.

| arm | train data | scenes | lr | steps | per-frame mIoU / AP50 | per-bundle mIoU / AP50 | `id_switch` ↓ | `view_consistency` ↑ |
|---|---|---|---|---|---|---|---|---|
| **A-long** (11498642) | ScanNet+ScanNet+++Infinigen | 3520 | 1e-4 | 84 480 | 0.676 / 0.704 | 0.600 / 0.604 | 0.414 | 0.734 |
| **D** (11642516) **SAM2-supervised, DIVERGED** | + RE10K@1500 | 5020 | 1e-4 | 85 340 | 0.334 / 0.256 | 0.291 / **0.136** | 0.663 | 0.485 |

**Why it is a divergence and not the familiar under-convergence.** Best epoch is **2 of 17**; the
training loss *rises* from 132 to 169 while the LR decays; and `train_AP50` collapses 0.211 → 0.006,
i.e. the head stops fitting the data it is being trained on. Nothing that merely made ScanNet val
harder could do that. The run itself was clean — 17 h 15, 352 GiB peak RSS of 416, 0 failures, no
NaN.

**The cause is the learning rate, and it was isolated with one variable at a time**
(docs/MULTIDATASET.md §11.3). Holding the mixture, the RE10K dose and `--anchor_3d` fixed and
halving the LR to 5e-5 removes the collapse entirely: that run tracks A-long epoch for epoch and is
marginally ahead at epoch 6 (0.369 vs 0.364) on 43 % more scenes. **A label conflict cannot be
undone by halving a learning rate**, so neither "SAM2 masks are the wrong kind of supervision" nor
the room-shell confound explains this failure.

The real arm is therefore **D-long** (5020 scenes, lr 5e-5, 85 340 steps, job 11830140) against a
**re-run A-long′ at the same LR** (3520 scenes, 84 480 steps, job 11830142), because A-long as
published cannot be compared to a 5e-5 run without moving two variables at once. Neither is in this
table yet.

## 7. Cross-dataset matrix — the same ruler on four benchmarks (todo 6d, 2026-08-09)

**A block of its own. Never merge these rows into §2/§3/§5**, and never read a row here next to a
class-aware ScanNetv2 number: three of the four datasets are class-agnostic-only.

`scripts/eval_3d_maskdino.py --dataset {scannetv2,scannet200,scannetpp,replica}` swaps the
benchmark and *nothing else* — same head, same lifting, same vendored official evaluator, same two
transfer modes (`docs/MASKDINO.md` §9.12). So every column below is a single-variable comparison,
and the ScanNetv2 column doubles as the regression guard.

**No dataset is in this table until its licence gate passed** (`scripts/gate_3d_gt.py`: its own GT
fed back as predictions must score exactly 1.000 / 1.000 / 1.000, `docs/MASKDINO.md` §9.2). All
four passed on 2026-08-09, on every scene of the real tars.

**Nothing here is fine-tuned.** All three checkpoints are trained on ScanNetv2's official 1201
split and nothing else, so ScanNet200 is a *relabelling* of the training domain and ScanNet++ /
Replica are **zero-shot** — the same posture FAST3DIS and SegVGGT report their cross-dataset rows
in (`docs/TRAINING_COMPARABILITY.md`: nobody fine-tunes on the target benchmark).

### 7.1 The four benchmarks, and their denominators

Measured by the gates (`gate_<dataset>.json` beside each tar), not quoted from papers:

| dataset | scenes | views/scene | mesh vertices (median) | evaluated GT instances | per scene (median) | annotated vertices | our column |
|---|---|---|---|---|---|---|---|
| `scannetv2` | 312 | 17.4 | 146 k | 4 364 | 12 | 0.90 | class-aware **and** class-agnostic |
| `scannet200` | 312 | 17.4 | 146 k | **10 045** | 29 | 0.89 | class-agnostic only |
| `scannetpp` | 49 | 50.0 | 1 184 k | 2 579 | 42 | 0.28 | class-agnostic only |
| `replica` | 8 | 50.0 | 791 k | 368 | 44 | 0.40 | class-agnostic only |

`scannet200` reads the **same two tars** as `scannetv2` — only the label set changes — which is why
its rows share the ScanNetv2 forward pass exactly (identical ICP inliers 0.963, camera RMS 0.136 m).
Its GT is **2.3× denser**: 10 045 evaluated instances against 4 364, because 200 categories are
scored instead of 18, and wall/floor are valid classes there (so the prediction side stops dropping
them, `train/datasets3d.py`).

### 7.2 The matrix — AP / AP50 / AP25, CLASS-AGNOSTIC

Every cell at **defaults** (the tuned lifting knobs were tuned on a leaky diagnostic, §5), one run
each, 0 failed scenes anywhere. Regenerate with `myenv/bin/python scripts/collect_eval3d_matrix.py`.

| checkpoint | dataset | unposed (`unproject`) | posed (`gt_projection`) |
|---|---|---|---|
| **mf** (1201 control, job 9386666) | scannetv2 | 0.013 / 0.050 / 0.320 | 0.039 / 0.122 / 0.483 |
| | scannet200 | 0.023 / 0.069 / 0.278 | 0.075 / 0.171 / 0.411 |
| | scannetpp | 0.000 / 0.000 / 0.004 | 0.003 / 0.009 / 0.075 |
| | replica | 0.000 / 0.000 / 0.007 | 0.004 / 0.023 / 0.096 |
| **`--anchor_3d`** (job 9634920) | scannetv2 | **0.042 / 0.138 / 0.504** | **0.109 / 0.304 / 0.677** |
| | scannet200 | 0.036 / 0.111 / 0.366 | **0.124 / 0.275 / 0.523** |
| | scannetpp | 0.000 / 0.000 / 0.015 | **0.009 / 0.038 / 0.178** |
| | replica | 0.000 / 0.000 / 0.009 | 0.006 / 0.028 / **0.190** |
| **`--num_frames 16`** (job 9668639) | scannetv2 | 0.023 / 0.080 / 0.391 | 0.064 / 0.190 / 0.572 |
| | scannet200 | 0.027 / 0.084 / 0.314 | 0.099 / 0.224 / 0.474 |
| | scannetpp | 0.000 / 0.000 / 0.013 | 0.007 / 0.021 / 0.108 |
| | replica | 0.000 / 0.000 / 0.008 | 0.008 / **0.046** / 0.125 |

The class-aware ScanNetv2 rows are §5's and are unchanged — see 7.3 reading 1.

### 7.3 What the matrix says

**1. The adapters disturbed nothing — the regression guard passed exactly.** Re-running ScanNetv2
through the new `--dataset` machinery reproduces every published class-aware triple to the last
digit: mf 0.023 / 0.067 / 0.268 and 0.060 / 0.156 / 0.408, **`--anchor_3d` 0.038 / 0.112 / 0.360**
(the value todo 6d named as the gate) and 0.104 / 0.257 / 0.504, s16 0.033 / 0.098 / 0.336 and
0.083 / 0.216 / 0.488. Two posed cells also gained the class-agnostic column they never had.

**2. ScanNet200 is a real second column, and its sign is checkpoint-dependent.** Against the same
scans' class-agnostic ScanNetv2 row, the control *gains* (0.050 → 0.069 AP50), `--num_frames 16` is
flat (0.080 → 0.084, inside the seed spread) and `--anchor_3d` *loses* (0.138 → 0.111) — while
all three gain in the posed bridge and all three lose AP25 (e.g. 0.504 → 0.366). Two structural differences drive it
and they pull opposite ways: 2.3× more GT instances (harder recall) but also far more GT for a
given prediction to match, plus wall/floor now being scorable on both sides. This mirrors §9.11's
finding that the label collapse is checkpoint-dependent; it is **not** evidence about 200-way
recognition, which we never attempt.

**3. Zero-shot to ScanNet++ and Replica fails under the unposed bridge and survives, weakly, under
the posed one.** Every unposed out-of-domain cell is 0.000 / 0.000 / ~0.01. The posed cells are
small but real (`--anchor_3d`: 0.038 AP50 on ScanNet++, 0.028 on Replica; AP25 0.178 / 0.190).

**4. …and the two bridges localise *why*, which is the point of reporting both.** Median per-scene
coverage, `--anchor_3d`:

| | posed: annotated vertices assigned | unposed: annotated vertices assigned | ICP inliers | camera RMS |
|---|---|---|---|---|
| scannetv2 | 0.834 | **0.679** | 0.963 | 0.136 m |
| scannetpp | 0.685 | **0.223** | 0.924 | 0.200 m |
| replica | 0.685 | **0.255** | 0.660 | 0.275 m |

With a perfect bridge the out-of-domain scenes are *covered* about as well as ScanNet is
(0.685 vs 0.834) — so the AP collapse there is the **2D masks**, an 8× drop in AP50 against the
same checkpoint's ScanNetv2 posed cell (0.038 vs 0.304). Under the unposed bridge coverage itself
collapses to a third — so on ScanNet++/Replica it is the **feed-forward geometry** that fails
first, exactly the constraint §9.6 identified on ScanNet, amplified out of domain. Replica shows
where: VGGT's cameras degrade badly on its synthetic renders (ICP inliers 0.66 vs 0.96, camera RMS
0.275 m vs 0.136 — twice the 0.05 m vote radius).

**5. `--anchor_3d` leads almost everywhere, and its lead is not a ScanNet artefact.** It is the
best of the three on every ScanNetv2 cell, on both ScanNet200 cells, and on ScanNet++ posed
(0.038 AP50 against 0.021 and 0.009) — i.e. on a relabelling of the training domain *and* on an
unseen dataset. The one exception is Replica, where `--num_frames 16` takes AP and AP50
(0.046 vs 0.028) while `--anchor_3d` keeps AP25 (0.190 vs 0.125) — on 8 scenes, so read it as a
single noisy cell, not a reversal.

**6. What may and may not be compared to a paper.** SegVGGT publishes ScanNet++ zero-shot
13.3 / 33.9 / 56.4 — but on **10 randomly sampled val scenes** (their Table 2, a different protocol
from their own Table 1, `docs/SEGVGGT_ANALYSIS.md`), with a LoRA-adapted backbone, and trained on
ScanNet200. Our 49-scene posed row (0.009 / 0.038 / 0.178) is below it and the comparison carries
three confounds at once; quote it only with all three named. FAST3DIS reports ScanNet++ and Replica
zero-shot too, but this project has **no recorded triple** for those two datasets — do not invent
one for the comparison.

### 7.4 What this block does NOT license

- **One run per cell.** The seed spread measured in §6.1 (0.009 per-bundle AP50; 0.003 3D AP)
  applies here too, and no cell was replicated. Deltas smaller than that are noise.
- **Replica's GT instance set is our construction** — the room shell (`wall`, `floor`, `ceiling`)
  and unlabelled objects are dropped (`docs/DATASET.md` §2.2). Every Replica number must say so.
- **ScanNet++ ships 49 of 50 val scenes**, one being an upstream trajectory defect (§2.1), and its
  256×192 depth confirms only 0.43 of projected vertices against ScanNet's 0.599 — the posed bridge
  is thinner there, though its coverage still lands at 0.685.
- **No claim about 200-way or 84-way recognition.** The head has 19 ScanNet logits; those columns
  collapse labels on both sides by construction.

### 7.5 The matrix on the data-scaling arms (2026-08-22/23; jobs 11498511–11498543, 11540891–11540905)

**The deliverable of the multi-dataset workstream** (docs/MULTIDATASET.md §10). Same ruler, same
defaults, same evaluator as §7.2 — 32 cells, 0 failed scenes anywhere. The arms differ **only** in
the training mixture and the step budget (§6.4); every one is `--class_agnostic --anchor_3d`, so
all rows here are class-agnostic and none is fine-tuned on any benchmark. ScanNet++ and Replica are
**zero-shot** for all of them (the 50 `nvs_sem_val` scenes are excluded from the training tars at
build time, docs/MULTIDATASET.md §1.1); ScanNet200 is a relabelling of the ScanNet training domain.

| arm | dataset | unposed (`unproject`) | posed (`gt_projection`) |
|---|---|---|---|
| **B** ScanNet 1201 (42 k steps) | scannetv2 | 0.041 / 0.130 / 0.474 | 0.139 / 0.323 / 0.669 |
| | scannet200 | 0.032 / 0.099 / 0.343 | 0.107 / 0.241 / 0.496 |
| | scannetpp | 0.000 / 0.000 / 0.010 | 0.006 / 0.027 / 0.143 |
| | replica | 0.000 / 0.000 / 0.004 | 0.014 / 0.047 / 0.211 |
| **C** + ScanNet++ 2054 (41 k steps) | scannetv2 | 0.042 / 0.129 / 0.480 | 0.140 / 0.327 / 0.668 |
| | scannet200 | 0.033 / 0.099 / 0.345 | 0.109 / 0.243 / 0.502 |
| | scannetpp | 0.000 / 0.000 / 0.016 | 0.012 / 0.043 / 0.234 |
| | replica | 0.000 / 0.000 / 0.011 | 0.032 / 0.080 / 0.326 |
| **A** + Infinigen 3520 (42 k steps) | scannetv2 | 0.032 / 0.106 / 0.450 | 0.111 / 0.277 / 0.641 |
| | scannet200 | 0.030 / 0.090 / 0.342 | 0.097 / 0.220 / 0.500 |
| | scannetpp | 0.000 / 0.000 / 0.011 | 0.007 / 0.029 / 0.232 |
| | replica | 0.000 / 0.000 / 0.014 | 0.024 / 0.090 / 0.378 |
| **A-long** = A converged (84 k steps) | scannetv2 | **0.057 / 0.166 / 0.516** | **0.177 / 0.389 / 0.708** |
| | scannet200 | **0.041 / 0.118 / 0.372** | **0.132 / 0.287 / 0.539** |
| | scannetpp | **0.000 / 0.001 / 0.026** | **0.019 / 0.068 / 0.275** |
| | replica | **0.000 / 0.000 / 0.036** | **0.040 / 0.119 / 0.480** |

**A-long takes all 8 of its cells.** Bold marks that; the first three arms are shown unbolded so
the 42 k-step block can still be read against itself.

**1. Adding real, same-domain ScanNet++ is a strict improvement: free in domain, large out of it.**
C vs B is flat on every ScanNetv2 and ScanNet200 cell (±0.004, inside §6.1's spread) and then
**+59 % AP50 on ScanNet++** (0.043 vs 0.027) and **+70 % on Replica** (0.080 vs 0.047, AP 2.3×)
under the posed bridge, with AP25 +64 % and +55 %. This is the first measurement in the project
that more training data buys anything the ScanNet ruler cannot see — and §6.4 shows the ScanNet
ruler called it a tie.

**2. Out of domain the unposed bridge stays at zero AP50 whatever we train on.** Across all four
arms every out-of-domain unposed cell is 0.000 / 0.000–0.001 / 0.004–0.036: even A-long, which
gains 2.5× posed AP50 there, moves only AP25 (ScanNet++ 0.010 → 0.026, Replica 0.004 → 0.036) and
leaves AP50 on the floor. The registration diagnostics are **identical across all four arms to
three decimals** — ICP inliers 0.963 / 0.924 / 0.660 and camera RMS 0.097 / 0.116 / 0.143 m on
ScanNetv2 / ScanNet++ / Replica — because they depend only on VGGT's frozen cameras, which no
head-only training touches. Median annotated-vertex coverage out of domain collapses to 0.21–0.31
unposed against 0.66–0.78 posed. §7.3 finding 4 said the feed-forward geometry fails first out of
domain; four arms spanning 1201 → 3520 scenes and 42 k → 84 k steps say it **binds absolutely**.

**3. The two ways data pays are different, and the diagnostics separate them.** B → C pays in
**coverage**: posed annotated-assigned rises exactly where AP does (ScanNet++ 0.657 → 0.766,
Replica 0.671 → 0.739) while in domain it is already 0.88–0.91 and has nowhere to go. A-long pays
in **mask quality instead**: its coverage is *lower* than C's out of domain (ScanNet++ 0.716,
Replica 0.681) and its AP is far higher, so the extra data bought better masks per assigned vertex,
not more of them. Do not read the AP gains here as one mechanism.

**4. "Infinigen hurts" was a budget artefact — at convergence the full mixture wins every cell.**
At 42 k steps A trails B and C on all six ScanNetv2/ScanNet200 cells and on ScanNet++. At 84 480
steps the same mixture takes **every one of the eight cells**, by margins far outside anything in
§7.2: unposed ScanNetv2 **0.057 / 0.166 / 0.516** against C's 0.042 / 0.129 / 0.480 (+29 % AP50)
and posed **0.177 / 0.389 / 0.708** against 0.140 / 0.327 / 0.668 (+19 %). Zero-shot it is
**2.5× B's AP50 on both** ScanNet++ (0.068 vs 0.027) and Replica (0.119 vs 0.047), and Replica's
AP25 more than doubles (0.480 vs 0.211). The larger the mixture, the more steps it needs before it
pays — which is §6.4's lesson stated on the ruler that matters.

#### 7.5.1 Arm D — **SAM2-supervised**, and a DIVERGED run (2026-08-25, job 11642519)

Same ruler, same defaults, 8 cells, 0 failed scenes. **Do not read these as what RE10K training
data is worth** — the checkpoint behind them comes from a run whose training loss rose and whose
`train_AP50` collapsed to 0.006 (§6.4.1, docs/MULTIDATASET.md §11.2). They are recorded so the next
reader does not re-derive them, and because the collapse is uniform across all 8 cells, which is
itself part of the evidence that the run — not the data — is what failed.

| arm | dataset | unposed (`unproject`) | posed (`gt_projection`) |
|---|---|---|---|
| **D** RE10K@1500, 5020 (85 k steps, lr 1e-4) **SAM2-supervised, DIVERGED** | scannetv2 | 0.001 / 0.007 / 0.172 | 0.006 / 0.025 / 0.320 |
| | scannet200 | 0.004 / 0.014 / 0.154 | 0.013 / 0.040 / 0.269 |
| | scannetpp | 0.000 / 0.000 / 0.000 | 0.000 / 0.001 / 0.042 |
| | replica | 0.000 / 0.000 / 0.006 | 0.001 / 0.003 / 0.128 |

For scale: A-long is 0.057 / 0.166 / 0.516 unposed on ScanNetv2 and the **ScanNet-only 1201-scene
control** is 0.013 / 0.050 / 0.320 (§7.2). Arm D lands **below its own single-source control on the
domain it trains on**, which is the shape of a broken run, not of bad training data. The corrected
pair (D-long 11830140 vs A-long′ 11830142, both lr 5e-5, both step-matched) chains its own matrix;
until it lands, this workstream's scaling claim is still A-long's.

**5. This IS a new headline, and it is the best row the project has.** A-long's unposed ScanNetv2
triple **0.057 / 0.166 / 0.516** beats the published `--anchor_3d` ScanNet-only row
(0.042 / 0.138 / 0.504) by +20 % AP50 on the same ruler, and it is the number §8.2's competitor
table should quote once C-long has separated data from compute (§6.4). Against the published
class-agnostic cluster — FAST3DIS 0.038 / 0.096 / 0.316 and IGGT 0.028 / 0.112 / 0.287 — it leads
on all three measures by 1.5–2.0×, still on a **strictly frozen** backbone and at ~17.4 views/scene
against their 50. The §7.4 caveats apply verbatim: one run per cell, Replica's GT is our
construction, no claim about 200-way or 84-way recognition — and **A-long is still not step-matched** (job
11632049 was that control and failed; C-long′ 11831105 ⇄ A-long′ 11830142 replaces it,
docs/MULTIDATASET.md §10.5).

## 8. Summary table — the numbers to quote, and against what

> **Building slides or anything outward-facing? Use `docs/FACTSHEET.md` instead** — it is this
> section plus the protocol labels, the positioning and the open work, in one page an agent can
> read whole. This section stays the source of truth; FACTSHEET is its read-out. **If the two
> disagree, this file wins and FACTSHEET is the bug — fix it there.**

A read-out of §2–§7, nothing new. **Each block is its own ruler**; rows may be compared *inside* a
block and never across blocks (§1). Every "vs" column names what the number is being compared to.

### 8.1 One line per ruler — where we stand

| # | Ruler (protocol) | Our best | Compared against | Verdict |
|---|---|---|---|---|
| A | **2D single-frame**, our val 0080–0089, N=490 (§2) | mIoU **0.694** / AP50 **0.729** | retired baseline head 0.451 / 0.294 | **+54 % mIoU, +148 % AP50**; curve still rising with data |
| B | **2D multi-view (per-bundle)**, our val, N=490 (§3) | mIoU **0.539** / AP50 **0.515** | retired baseline head 0.367 / 0.199 | **+47 % mIoU, 2.6× AP50**, no post-hoc fusion |
| C | **2D, official 1201/312 split** (§6) | per-frame AP50 **0.669** · per-bundle AP50 **0.552** | job 8900194, 0.604 per-frame | scale holds on the honest split; own metric code → not a leaderboard number |
| D | **3D, official benchmark, UNPOSED** — predicted depth+cameras (§5) | class-agnostic **0.042 / 0.138 / 0.504** (`--anchor_3d`; seed 1: 0.039 / 0.129 / 0.485) | FAST3DIS 0.038 / 0.096 / 0.316 · IGGT 0.028 / 0.112 / 0.287 | **lead on AP50 + AP25** (1.34–1.44× / 1.53–1.59× FAST3DIS), **tie FAST3DIS on AP**, lead IGGT on all three; frozen backbone, untuned, **2 seeds** (§5.2) |
| E | **3D, official benchmark, POSED** — GT poses/intrinsics/depth (§5.1) | **0.088 / 0.260 / 0.572** (S=16, 20 ep) | SegVGGT 0.504 / 0.717 / 0.870 | still behind; the protocol explains 2.3×, the rest is real |
| F | **COCO port check** (`docs/old/MASKDINO_COCO.md`, §1.4) | 46.133 mask AP / 51.549 box AP | upstream MaskDINO's own checkpoint, 46.1 / 51.5 | implementation is faithful; **not a project result** |
| G | **3D, the other three benchmarks** — ScanNet200 / ScanNet++ / Replica (§7) | posed `--anchor_3d` **0.124 / 0.275 / 0.523** (ScanNet200) · 0.009 / 0.038 / 0.178 (ScanNet++) · 0.006 / 0.028 / 0.190 (Replica) | no like-for-like published row is held in this project (§7.3 reading 6) | zero-shot **fails** under the unposed bridge (0.000 everywhere) and survives weakly under the posed one; the split localises it to geometry vs masks |

AP triples are always `AP / AP50 / AP25`. A/B/C are per-view 2D masks on our own metric code
(never placeable next to published ScanNet figures). D/E use the official vendored evaluator on
the official val-312 point clouds — the only rows in this project that may sit next to a paper's.
G uses the same evaluator on three further benchmarks, class-agnostic only.

### 8.2 The competitor table — like-for-like, class-agnostic, unposed (§5)

The one comparison that is fair in every dimension: same evaluator, same *bridge* (each method's
own predicted geometry), same label setting (classes collapsed — FAST3DIS and IGGT publish no
class-aware number).

| Method | Backbone | Views/scene | AP | AP50 | AP25 |
|---|---|---|---|---|---|
| IGGT (re-evaluated by FAST3DIS) | adapted | 50 | 0.028 | 0.112 | 0.287 |
| FAST3DIS | LoRA-adapted DA3 | 50 | 0.038 | 0.096 | 0.316 |
| **Ours, `--anchor_3d`, defaults — ScanNet-only training** | **frozen VGGT-1B** | **~17** | **0.042** | **0.138** | **0.504** |
| **〃 at THEIR view budget** (§5.4, dense export, single seed) | 〃 | **50** | **0.053** | **0.170** | **0.542** |
| — same, best lifting knobs (sensitivity, not headline) | 〃 | ~17 | 0.055 | 0.185 | 0.571 |
| *ours, headline checkpoint (no `--anchor_3d`), tuned* | 〃 | ~17 | *0.017* | *0.060* | *0.334* |
| **Ours + EXTRA TRAINING DATA** (A-long: ScanNet + ScanNet++ + Infinigen, 3520 scenes, §7.5) | 〃 | ~17 | **0.057** | **0.166** | **0.516** |
| **〃 at THEIR view budget** (§5.4, single seed) | 〃 | **50** | **0.069** | **0.193** | **0.560** |

**The claim, in the wording §5 settled on 2026-08-07 after the seed-1 replicate: ahead of both
published unposed methods on AP50 and AP25, TIED with FAST3DIS on AP, and ahead of IGGT on all
three — with a strictly frozen backbone, no adaptation, ~1/3 of their views, and all lifting knobs
at defaults.** **At their OWN 50-view budget (§5.4, 2026-08-27) the same checkpoint reads
0.053 / 0.170 / 0.542, i.e. ahead on all three — but that row is a single seed, so the seed-1
wording above still governs the ~17-view headline and the 50-view row is quoted as
"and the lead widens at matched views", never as a replacement replicate.** It survives the whole knob grid — the worst point of the sweep is still 1.44×
FAST3DIS's AP50 (§5). **"Ahead on all three" was a seed-0-only reading of THIS row and must not be
repeated for it**; it is licensed for the **extra-data** row only (§7.5 reading 5). Two caveats
travel with it: **one run per cell** — the "single run against a single control" caveat is retired,
both arms were replicated at seed 1 (§5.2) — and the class-collapse sign is checkpoint-dependent
(the italic row is the same recipe without `--anchor_3d`, **with its lifting knobs tuned**; at
defaults that control reads 0.013 / 0.050 / 0.320).

**The extra-data row is separate on purpose, and the ScanNet-only row stays the headline.** The
field norm both this project and its competitors follow is that extra training data gets its own
labelled row rather than being folded into the clean one (`docs/TRAINING_COMPARABILITY.md` §2, and
MaskDINO's own README fences its rows the same way). Read the two together: the *mechanism* claim
rests on the ScanNet-only row, the *scaling* claim on the extra-data one. It is also the closer
comparison to IGGT, which likewise trains on a curated mixture. Two things it still owes:
**it is not yet step-matched** (84 k steps against the arms it beats — job 11632049 was that control
and failed as an unstable run — the LR failure of §11.3 — so the compute/data split at the top end
remains unmeasured until C-long′ 11831105 ⇄ A-long′ 11830142 lands, docs/MULTIDATASET.md §10.5) and,
like every row here, it is a single run.

**The one asymmetry this table does NOT yet control for — and it favours us.** Both published
rows are **zero-shot on ScanNet** (FAST3DIS trains only on Aria/ASE, IGGT only on InsScene-15K);
every row of ours trains on ScanNet. The comparison is therefore protocol-matched and
setting-matched but **not training-matched**, and that must be said wherever this table is quoted
until the two zero-shot arms land (**I** 11839134 = IGGT's mixture minus ASE, **I-gt** 11839135;
`docs/MULTIDATASET.md` §12, `docs/TRAINING_COMPARABILITY.md` §6.2). Two smaller asymmetries run
the other way and are already stated: the frozen backbone, and — until 2026-08-27 — ~17 views to
their 50, an axis now **closed and measured** (§5.4: at matched views we lead by more, and the
view-count lever saturates at ~50).

*Not in this table on purpose:* **SegVGGT 0.504 / 0.717 / 0.870** — posed transfer, a different
protocol (§5.1); and the point-cloud/RGB-D family (Mask3D 55.2, SegDINO3D 64.0 AP) — different
input modality. The image-only baseline in SegVGGT's own table, OneFormer3D†, scores
**5.4 / 10.2 / 17.4** — that row is the answer to "why is your AP low".

### 8.3 The same masks under both 3D bridges — what the geometry costs

| Checkpoint | unposed (own geometry) | posed (GT bridge) | bridge cost |
|---|---|---|---|
| multi-frame, official split (headline) | 0.023 / 0.067 / 0.268 | 0.060 / 0.156 / 0.408 | **2.3× AP50** |
| `--anchor_3d` | 0.038 / 0.112 / 0.360 | 0.104 / 0.257 / 0.504 | 2.3× |
| S=16, 20 epochs | 0.032 / 0.115 / 0.414 | **0.088 / 0.260 / 0.572** | 2.3× |
| S=16 + `--anchor_3d` (todo 2f, §5.3) | 0.032 / 0.109 / 0.353 | 0.082 / 0.236 / 0.501 | 2.2× |
| *oracle — GT rendered through the posed bridge* | — | *0.828 / 0.948 / 0.974* | ceiling of the ~17-frame budget |
| SegVGGT (published) | — | 0.504 / 0.717 / 0.870 | — |

18-class, class-aware columns. **Unposed = 2D mask quality × feed-forward geometry quality**;
**posed = 2D mask quality alone.** Print both or neither. Of the gap to SegVGGT, a factor 2.3 is
the bridge and a factor ~4.6 is real (LoRA backbone, 75–100 views vs 17, 259×196 masks vs 37×37,
600 kept queries vs 100).

### 8.4 What actually buys the result — ranked by effect size

Read every Δ against the **measured seed-to-seed spread of 0.009 per-bundle AP50** (§6.1).

| Lever | Effect | Size | Where |
|---|---|---|---|
| **Training data 50 → 490 scenes** | +0.26 per-frame AP50 | dominates everything | §2 |
| **Cross-frame attention** | +0.183 per-bundle AP50 | 20× seed noise | §3 |
| **Bundle features** | +0.147 per-bundle AP50 (but −0.048 per-frame) | 16× | §3 |
| **`--anchor_3d`** | +66 % 3D AP50 in *both* bridges; `id_switch` −0.089 | AP-neutral in 2D (+0.002, inside noise) | §5.1, §6.1 |
| **Bundle width 8 → 16** | +0.027 per-bundle AP50; `id_switch` −0.113; +46 % unposed 3D AP50 | 3× | §6 |
| Lifting knobs (`--vote_radius`, depth conf.) | +0.016 → +0.047 3D AP50 | larger than most decoder ablations | §5 |
| Any single decoder component (two-stage, encoder, DN, box init) | ≤0.046 per-frame AP50 | ablations at N=190 | §2 |
| Mask resolution 37² → 74² | −0.022 (neutral); 37²'s GT ceiling is 0.956 | **not the bottleneck** | §2 |

**The three headline conclusions**: (1) the track is **data-limited, not architecture-limited** —
the leak-free 1201-scene checkpoint even beats the one that had *seen* the val scenes
(0.083 vs 0.052 AP50); (2) **recognition and cross-view identity are separate axes** — `bundle_AP50`
alone cannot see what `--anchor_3d` does, so score identity mechanisms on the 3D ruler; (3) on the
3D ruler **the lifting step, not the decoder, now binds** (AP25 ≈ 4× AP50).
