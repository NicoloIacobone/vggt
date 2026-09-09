# Final report — class-agnostic 3D instance segmentation on a frozen VGGT backbone

**Project closed 2026-09-09.** This is the closing account: what was attempted, what was built,
what was measured, why it stopped, and what a successor should do differently. It supersedes the
supervisor deck and is the one document to read if you read only one.

It is written to be *checkable*: every number below is reproduced from `docs/RESULTS.md` and
`docs/MULTIDATASET.md`, which stay in the repository as the underlying record. Where a result is
weaker than it first looks, the report says so in the same sentence.

**Setting.** A research project carried out at **ETH Zurich — Photogrammetry & Remote Sensing**
between June and September 2026, by **Nicolò Iacobone**, following and independent of the author's
master's thesis. Supervised by **Christos Sakaridis** (Lecturer, and Head of the Artificial Visual
Intelligence group, Photogrammetry and Remote Sensing lab, ETH Zurich), **Mattia Segù** (Research
Scientist, Google Zurich) and **Tatiana Tommasi** (Full Professor, Department of Computer and
Control Engineering, Politecnico di Torino; Director of the ELLIS Unit Turin).

---

## 1. The question

Can a **strictly frozen** feed-forward 3D foundation model — VGGT-1B, never updated, not even with
LoRA — support **multi-view-consistent 3D instance segmentation** through a small trained decoder,
well enough to sit next to methods that adapt their backbone?

The interest of the question is the constraint. Every published competitor in this space
(SegVGGT, FAST3DIS, IGGT) adapts a VGGT-family backbone and spends roughly 16 GPU-days doing it.
Nobody had published the unadapted case. If a frozen backbone were enough, the recipe would be
cheap, composable, and reusable for any downstream 3D task; if it were not, the *size* of the
shortfall would be worth knowing.

Supervision came from the official **ScanNet v2 2D instance annotations** on the official
**1201 train / 312 val** split. Predicted 2D masks were lifted to the scene point cloud and scored
with the **official ScanNet 3D instance evaluator, vendored unmodified** into the repository
(`train/benchmark3d.py`).

## 2. The answer, in one paragraph

**A frozen backbone is enough to be *interesting* and not enough to be *competitive*.** In the
in-domain setting — training on ScanNet, evaluating on ScanNet, unposed and class-agnostic, at the
competitors' own 50-view budget — the model reaches **AP/AP50/AP25 = 0.053 / 0.170 / 0.542**,
ahead of both published feed-forward competitors on all three columns, at **~0.8 GPU-days against
their ~16**. But that lead exists only because those competitors are *zero-shot on ScanNet* and we
are not. When ScanNet is removed from training and IGGT's own data mixture is reproduced in full,
the same recipe scores **0.009 / 0.032 / 0.301**: level with them at the coarse-localisation bar
(AP25) and **~3× behind at 0.5 IoU**. The honest summary is that the method finds and roughly
places objects about as well as adapted backbones do, and delineates them considerably worse — and
that the remaining loss sits in the **2D→3D lifting**, not in the decoder.

## 3. What was built

```
frozen VGGT-1B  ──►  aggregated_tokens_list[-1]        (cached once per scene, no_grad)
                     F : [B, S, P, 2048]
                              │
                     ┌────────┴─────────────────────────────────┐
                     │  pixel decoder — ViTDet 3-level pyramid   │  models/maskdino/pixel_decoder.py
                     │  + MSDeformAttn encoder                   │
                     └────────┬─────────────────────────────────┘
                              │
                     ┌────────┴─────────────────────────────────┐
                     │  MaskDINO decoder — two-stage selection,  │  models/maskdino/decoder.py
                     │  DAB anchors, denoising, deep supervision │
                     │  + cross-frame attention (--multi_frame)  │  models/maskdino/multiframe.py
                     │  + optional 3D anchors (--anchor_3d)      │  models/maskdino/anchor3d.py
                     └────────┬─────────────────────────────────┘
                              │
                     pred_masks : [B, N, S, h, w]     one query = one instance in ALL views
```

Three properties are worth naming because they are what the study actually tested:

1. **The backbone is never touched.** Its features are cached once per scene up front, so
   head-only training runs in minutes per epoch rather than hours, and the whole headline run costs
   ~0.8 GPU-days.
2. **Multi-view consistency is structural, not post-hoc.** A query owns one instance across every
   frame of a bundle by construction — there is no mask matching, no fusion, no tracking stage.
   The batch dimension is *frames*, and a bundle's frames stay contiguous in it.
3. **The lifting is honest by default.** Masks are unprojected with VGGT's **own predicted** depth
   and cameras, majority-voted per superpoint, registered to the benchmark cloud with an
   eval-only Sim(3)+ICP. No ground-truth geometry enters inference.

A second, *posed* bridge (GT poses + intrinsics + sensor depth, the protocol SegVGGT publishes on)
is implemented alongside it, so mask quality and geometry quality can be separated. Its oracle —
ground-truth masks pushed through the same bridge — returns 99.99 % of assigned annotated vertices
to their own instance, which is what certifies that the bridge itself is not the thing being
measured.

Scale of the implementation: ~30 k lines of project Python outside the untouched upstream backbone,
22 standalone CPU test scripts plus 3 shell test harnesses, 22 SLURM drivers, ~88 commits between
2026-06-07 and 2026-09-08.

## 4. What was measured

All 3D numbers below come from the official vendored ScanNet evaluator on the official val-312
split. `AP / AP50 / AP25`, always in that order. **2D numbers are on our own metric code and can
never be placed next to a published figure** — they appear here only as §4.4.

### 4.1 The headline — and the row that prices it

Unposed (own predicted geometry), class-agnostic — the like-for-like setting of the two published
feed-forward competitors, at their own 50-view budget:

| Method | Backbone | Views | AP | AP50 | AP25 |
|---|---|---|---|---|---|
| IGGT *(as re-evaluated by FAST3DIS)* | adapted | 50 | 0.028 | 0.112 | 0.287 |
| FAST3DIS | LoRA-adapted DA3 | 50 | 0.038 | 0.096 | 0.316 |
| **Ours — ScanNet-trained, `--anchor_3d`, defaults** | **frozen VGGT-1B** | 46.7 | **0.053** | **0.170** | **0.542** |
| — the same checkpoint at 17 views (two seeds) | 〃 | 17.0 | 0.042 | 0.138 | 0.504 |
| **Ours + extra training data** (ScanNet + ScanNet++ + Infinigen, 3520 scenes) | 〃 | 46.7 | **0.069** | **0.193** | **0.560** |
| **Ours — ScanNet REMOVED, IGGT's mixture reproduced in full** (4819 scenes) | 〃 | 17.4 | **0.009** | **0.032** | **0.301** |

**Read the last row against the third.** They are one result, not two. The third row is what the
recipe reaches when it trains on the evaluation domain and the competitors do not; the last is what
it reaches when that advantage is removed. In AP50 that is **0.170 → 0.032, a factor of 5.3** —
or 4.3 against the view-matched 17-view headline of 0.138, which is the closer comparison since the
no-ScanNet arm was itself scored at 17.4 views.

The last row is the reason the project closed, and it deserves to be stated precisely rather than
softened:

- It is **~3× behind** both published rows at AP50 (0.032 vs 0.096 / 0.112).
- It is **level at AP25** (0.301 vs 0.316 / 0.287 — above IGGT, marginally below FAST3DIS), at
  **17.4 views against their 50**.
- It does **not** demonstrate that the recipe is worse at equal data, and no experiment in this
  project can: the arm holds **1 000 ASE scenes against FAST3DIS's ~100 k**, a frozen backbone
  against adapted ones, ~0.8 GPU-days against ~16. The supportable claim is *"we cannot match
  their training setting"*, never *"we lose at equal data"*.

The distinction matters and it is also the honest limit of the work: the experiment that would
settle it was never affordable here.

### 4.2 The training-matched comparison — the one place nothing is conceded

SegVGGT trains on exactly the split we train on. It publishes on the *posed* bridge, so the
comparison must be run there:

| | AP | AP50 | AP25 |
|---|---|---|---|
| Ours, posed, class-aware (S=16, 20 epochs) | 0.088 | 0.260 | 0.572 |
| SegVGGT (published, posed) | 0.504 | 0.717 | 0.870 |
| *oracle — GT masks through the same bridge* | *0.828* | *0.948* | *0.974* |

We are behind. Decomposed: the unposed→posed bridge is worth a consistent **2.3×**, and after
removing it a **×2.8 residual** remains on the `--anchor_3d` checkpoint. That residual is real and
attributable — LoRA-adapted backbone vs frozen, 75–100 views vs ~17, 259×196 masks vs 37×37. One
candidate explanation was tested and eliminated: kept queries (600 vs our 100) is **measured
neutral**, 0.138 → 0.140.

For calibration in the other direction: the only *image-only* baseline in SegVGGT's own table,
OneFormer3D†, scores **5.4 / 10.2 / 17.4**. Image-only 3D instance segmentation is a different
difficulty class from the RGB-D and point-cloud methods (Mask3D 55.2, SegDINO3D 64.0 AP) that
dominate the ScanNet leaderboard.

### 4.3 What actually moved the number — the ablation ranking

Every Δ is read against a **measured seed-to-seed spread of 0.009 per-bundle AP50**. All rows below
are on the 3D ruler except where marked.

| Lever | Effect | Verdict |
|---|---|---|
| Training data, 1201 → 3520 scenes | +0.023 AP50 at matched views | **dominates every decoder ingredient** |
| Cross-frame attention (removing it) | **−57 % of the 3D AP50** | largest single mechanism in the study |
| Bundle features → per-frame features | −24 % class-aware / −49 % class-agnostic | second largest; label-setting dependent |
| `--anchor_3d` (3D anchors vs 2D DAB boxes) | **+66 % 3D AP50** in both bridges | the strongest decoder-side ingredient |
| View budget, 17 → 50 | +24 % AP50 — then **saturates** | 50 → 71 views is flat-to-negative |
| Bundle width, 8 → 16 frames | +46 % unposed AP50 | 3× seed noise |
| Lifting knobs (vote radius, depth confidence) | +0.016 → +0.047 AP50 | **larger than most decoder ablations** |
| Mask resolution, 37² → 74² | −0.022 (neutral) | **not the bottleneck** |

Four conclusions follow, and they are the substantive findings of the project:

1. **The system is data-limited, not architecture-limited.** More training data moves the headline
   further than removing any single MaskDINO ingredient does (+0.023 against ≤0.005).
2. **Recognition and cross-view identity are separate axes.** `--anchor_3d` is AP-neutral in 2D and
   worth +66 % in 3D. A 2D per-bundle metric cannot see what it does — which is exactly the kind of
   result that only shows up when both rulers are maintained.
3. **The lifting binds, not the decoder.** AP25 runs ≈ 4× AP50 throughout; the unposed bridge costs
   2.3× on identical masks; every out-of-domain unposed cell in the project reads 0.000 AP. Since
   the view budget saturates by 50, what is left in the bridge is **registration**, not coverage.
4. **Resolution is not the ceiling.** The 37×37 grid's GT-only ceiling is 0.956 AP50 against the
   model's ~0.69. Recognition binds first.

### 4.4 The internal 2D ruler, and cross-view identity

On the official 1201/312 split with our own metric code: **per-frame AP50 0.669**, **per-bundle
AP50 0.552** (0.704 / 0.604 for the best class-agnostic data-scaling arm). These selected which
checkpoint went to the 3D ruler; they are **not comparable to any published number** and were never
put on a slide next to one.

Cross-view identity was scored with the tracking literature's own metrics — a bundle's views read
as timesteps, one query as one track by construction: **HOTA 0.422 / AssA 0.584 / DetA 0.314 /
IDF1 0.492**.

This measurement also **retired a claim the project had been making.** `--anchor_3d` was believed
to improve cross-view identity. Across two seeds, no formal metric distinguishes it from its
control — AssA moves +0.0011 against a spread of 0.0046. Only the project's own `id_switch` sees an
effect, because it flips on near-ties between queries segmenting the same object. The correct
statement is: *"3D anchors reduce our own `id_switch`; no published tracking metric distinguishes
them from the control."* The +66 % AP50 is untouched — it is measured on the benchmark, not on an
identity metric.

### 4.5 Transfer to other benchmarks

Same evaluator, same bridges, class-agnostic, `--anchor_3d`:

| | ScanNet200 | ScanNet++ | Replica |
|---|---|---|---|
| ScanNet-only checkpoint (posed) | 0.124 / 0.275 / 0.523 | 0.009 / 0.038 / 0.178 | 0.006 / 0.028 / 0.190 |
| + extra training data (posed) | 0.132 / 0.287 / 0.539 | 0.019 / 0.068 / 0.275 | 0.040 / 0.119 / 0.480 |

**Zero-shot transfer dies under the unposed bridge** — every out-of-domain unposed cell is 0.000 AP
and 0.000–0.001 AP50 across all four data arms — and survives weakly under the posed one. That
split is diagnostic: the failure is in the **geometry**, not in the masks. It is the single
clearest piece of evidence for conclusion 3 above.

## 5. Why the project closed

Two reasons, in order of weight.

**1. The result does not survive the comparison that matters.** The lead in §4.1 is real, measured
under the competitors' own protocol, evaluator, view budget and label setting — but it rests on
training on the evaluation domain while both competitors are zero-shot on it. Once that is
levelled, the method is ~3× behind at AP50. A headline that only holds in the one configuration
where the data advantage is ours is not a result to build a paper on, and the project's own
documentation was explicit about this from the moment it was measured. Closing on that basis is the
correct reading of the evidence, not a concession.

**2. The compute budget was exhausted.** The experiments that could have changed the picture were
all data-scale experiments — ASE at 5 000 scenes rather than 1 000, ASE mixed *with* ScanNet, a
training run long enough to separate "more data" from "more compute" at the top end. Each is one
training run plus an eval matrix, and none was affordable once the cluster allocation ran out. The
architecture-side questions had, by then, largely been answered: §4.3 shows every decoder
ingredient moving the number by less than the data axis does.

There is no third reason. The implementation is verified (the decoder reproduces upstream
MaskDINO's published COCO result to +0.004 mask AP when driven with upstream's own weights), the
evaluator is the official one unmodified, the protocol axes are matched to the competitors one by
one, and the seed spread is measured rather than assumed.

## 6. What the project got right, methodologically

Independent of the final number, three practices are worth carrying forward:

- **Two rulers, never mixed.** A 2D internal metric and the official 3D evaluator were maintained
  in parallel, with an explicit written rule that a 2D number may never appear beside a published
  one. §4.3 conclusion 2 is a finding that only exists because both were kept: the 2D ruler was
  blind to a mechanism worth +66 % in 3D. It also caught a mis-ranking — two levers that looked
  1.24× apart in 2D are 2.4× apart in 3D.
- **Every competitor axis matched explicitly, or declared unmatched.** Evaluator, bridge, label
  setting, benchmark, view budget and training data were each tracked to one of three states —
  matched, closest-available-and-declared, or permanently impossible — rather than blurred. That
  discipline is what made the §4.1 caveat visible instead of accidental.
- **Claims were retired when the measurement said so.** The `--anchor_3d` identity claim (§4.4) was
  the project's own, and it was withdrawn the day formal metrics contradicted it. So was the
  "Infinigen hurts" reading, which turned out to be a step-budget artefact that reversed once the
  schedule was doubled.

## 7. What a successor should do

Ranked by expected value, from the project's own evidence:

1. **Fix the lifting, not the decoder.** This is where §4.3 conclusion 3 and §4.5 both point. The
   unposed bridge costs 2.3× on identical masks and collapses to zero out of domain, while mask
   resolution and decoder ingredients are demonstrably not the constraint. Concretely: better
   registration than Sim(3)+ICP on camera centres, and depth-confidence handling that degrades
   gracefully rather than voting on bad geometry.
2. **Mix ASE with ScanNet.** ASE was only ever measured on a no-ScanNet mixture, where it was worth
   ×1.4 AP50 and — uniquely among sources tried — helped *more* out of domain than in. Whether it
   adds anything on top of the headline recipe is unknown, and one training run answers it. The
   cautionary precedent is RE10K, whose sign **flipped** depending on ScanNet's presence.
3. **Scale ASE.** 1 000 scenes cost ~223 GB and 2 h 41 min to fetch, with the inode cost measured
   at 1 764/scene. 5 000 is affordable on a normal allocation. This buys a quantitative point on
   the data axis, not a new claim.
4. **Reconsider "strictly frozen".** The constraint was the point of the study and it was worth
   measuring, but §4.1 is now the measurement: it costs ~3× at AP50 against adapted backbones on
   matched data. A follow-up that allows minimal adaptation — the cheapest LoRA that closes part of
   the gap — would turn a negative result into a cost curve, which is more useful.

## 8. Reproducibility — what is here and what is not

**In the repository:** the full model (`models/maskdino/`), training and evaluation entry points
(`scripts/`), the data pipeline (`train/`, `data/`), the vendored official 3D evaluator
(`train/benchmark3d.py`), all cluster drivers (`slurm/`), 22 CPU-runnable test scripts (`tests/`),
and the complete measurement record (`docs/`).

**Not in the repository, and not obtainable from it:**

- **Trained checkpoints and cached backbone features.** They lived on cluster storage that has been
  released. Every number in this report is reproducible from the code and the public datasets, but
  not re-scorable from a checkpoint in this tree.
- **The datasets.** ScanNet v2, ScanNet++, Replica and ASE are all licence-gated; the repository
  contains the fetch/build tooling and the exact scene lists, not the data.
- **A working environment.** `myenv/` was never committed. `docs/RESTORE.md` carries the rebuild
  recipe and the exact resolved versions every published number was produced with
  (torch 2.3.1+cu121 / torchvision 0.18.1+cu121).
### 8.1 What was removed at project close

The repository was cleaned up for publication on 2026-09-09. Everything below was deleted from the
working tree and **remains in git history** (`git log --follow`, or `git show <commit>:<old path>`);
documents that still cite these paths are citing that history, not a live file.

| path | what it was | why it went |
|---|---|---|
| `docs/old/` | the pre-official-split archive: milestone write-ups, the COCO backbone-swap study, superseded result tables, meeting notes, old decks and their figures, closed to-do archives | 89 files of superseded narrative; this report is the surviving account |
| `docs/slides/` | the supervisor deck of 2026-08-27 (Marp), its speaker notes and summary deck | written mid-project, framing superseded by this report |
| `legacy/coco/` | the COCO backbone-swap arm and its upstream-MaskDINO control | retired 2026-08-27; the port-fidelity question it answered is settled (46.133 vs upstream 46.129 mask AP) and the project never reported on it |
| `examples/` | upstream VGGT's demo media, 63 MB | upstream content unrelated to this work. `demos/demo_gradio.py` filters its example gallery by file existence, so the demos start without it |
| `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md` | upstream community files | they govern contributions to upstream VGGT, not to this fork |
| two stray `slurm/*.log` | job logs from the 2026-06 scaling runs | logs belong in `slurm/logs/` (gitignored) |

What was deliberately **kept** despite being retired: `legacy/d4rt/` and `legacy/dataset_build/`
(imported by `scripts/eval_perframe.py`, `demos/demo_gradio.py`, `train/scannetpp3d.py` and three
tests), `docs/todo.md` (cited by ~40 comments in live source files), and `vggt/` + `training/`
(untouched upstream).

Note that removing these does **not** shrink a fresh clone: they were tracked from early in the
project's history, so their blobs are in the pack regardless. Only a history rewrite would change
that, and none was performed.

## 9. Rules that still apply to any number taken from this project

Kept because they are the errors most likely to be made by someone reading the tables cold:

1. **2D and 3D numbers are different rulers.** A 2D number may never be placed beside a published
   figure.
2. **Class-aware and class-agnostic are different columns.** FAST3DIS and IGGT publish
   class-agnostic only; SegVGGT is class-aware.
3. **Posed and unposed are different protocols.** Never put a SegVGGT number in an unposed table.
4. **Label IGGT's triple "as re-evaluated by FAST3DIS"** — IGGT publishes no ScanNet AP of its own.
5. **State the view budget with any comparative claim.** At 50 views the lead is on all three
   columns; at 17 the AP column is a tie with FAST3DIS.
6. **Extra-data rows stay fenced** from the ScanNet-only row, per field norm.
7. **Anything trained on RE10K is SAM2-supervised** — those masks are model output, not ground
   truth.
8. **Any pre-2026-07-08 number is on retired SAM3 ground truth** and does not transfer; the switch
   to official ScanNet GT cost about half the AP50 headline.
9. **Read every Δ against 0.009 per-bundle AP50** (measured seed spread), or against 0.01–0.02 for
   the identity metrics.
10. **Quote §4.1's headline and its no-ScanNet row together.** Quoting the lead alone borrows a
    leaderboard position while dropping the axis it rests on.

---

## Where the detail lives

| document | what it holds |
|---|---|
| `docs/RESULTS.md` | every number, one home; the underlying record for §4 |
| `docs/MASKDINO.md` | architecture, deviations from upstream MaskDINO, the protocols in full |
| `docs/MULTIDATASET.md` | the multi-dataset arms, including the no-ScanNet result of §4.1 (§12.4) |
| `docs/TRAINING_COMPARABILITY.md` | what each competitor trains on vs evaluates on, axis by axis |
| `docs/RELATED_WORK.md` | the competitor landscape and the positioning argument |
| `docs/SEGVGGT_ANALYSIS.md` | the closest competitor dissected; where the residual gap of §4.2 goes |
| `docs/COMMANDS.md` | the full command catalogue with the caveat each one needs |
| `docs/DATASET.md` | ground-truth provenance, mask conventions, the tars |
| `docs/FACTSHEET.md` | the frozen outward-facing read-out, as it stood at closure |
| `docs/RESTORE.md` | environment rebuild and the cluster-archive layout |
| `docs/todo.md` | the work ledger as it stood at closure, frozen |
