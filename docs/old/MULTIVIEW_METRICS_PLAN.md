# Plan — multi-view consistency metrics for the MaskDINO track

**Status: PROPOSAL, nothing implemented.** Written 2026-07-29. Answers `docs/RELATED_WORK.md`
gap 2 ("consistency intrinsic to the query, not post-hoc — we already have it; claim it… to
substantiate: add an explicit cross-view consistency metric").

Scope: how to *measure* whether `--multi_frame` (docs/MASKDINO.md §8.2) actually produces
view-consistent instances, as opposed to producing good per-frame masks that happen to be
scored on a volume. No new model work is proposed here.

---

## 1. What we already have, and why it does not answer the question

`train/maskdino_eval.py::eval_scenes_multiframe` reports `bundle_*`: one IoU per query over the
concatenated `[S·h·w]` mask volume, one class score per query (max over views), scored by
`compute_instance_segmentation_metrics`. Result at N=490: **0.535 mIoU / 0.494 AP50**
(docs/RESULTS.md §3).

That metric is the right *headline* — it is the exact analogue of what the closest published
competitor uses. PanSt3R's scene-level PQ (`PQ^sc`) is defined as "PQ computed at the scene level
by pretending that the scene is a concatenation of all images, effectively tying predictions
between all images", and Panoptic Lifting introduced the same idea. So `bundle_AP50` is already
in the family the field accepts.

**But it conflates three things into one number**, and only one of them is what we want to claim:

1. per-frame mask quality (already measured, per-frame protocol: 0.621 / 0.630);
2. per-frame detection/recall (a query missing in one view shrinks the volume IoU);
3. **cross-view identity** — whether the pixels query *q* claims in view *f* and in view *f'*
   belong to the same physical object.

A model can score well on (3) and badly on the volume, or the reverse. Concretely: our
multi-frame run's per-frame numbers are −0.021 against its own control (neutral), and its
bundle numbers are 2.5× arm C. Right now we **cannot say how much of that gap is association
rather than detection**, and that is precisely the sentence the thesis wants to write.

The fix, in one line: **report a metric that separates detection quality from association
quality**, plus the controls that give the association number a floor and a ceiling.

## 2. Literature — what exists, what we adopt

Nothing in the feed-forward-3D lane defines a consistency metric of its own; they all reuse
video/tracking metrics or the concatenate-the-views trick. The relevant menu:

| Metric | Source | What it measures | Verdict for us |
|---|---|---|---|
| **HOTA** = √(DetA·AssA), with **AssA / AssRe / AssPr** | Luiten et al., IJCV 2021 ([2009.07736](https://arxiv.org/abs/2009.07736)) | explicitly decomposes detection accuracy from association accuracy; sub-metrics separate five error types | **ADOPT — the core of this plan.** `AssA` is exactly "cross-view identity, independent of detection quality". `AssRe` = fragmentation (one object split over several queries), `AssPr` = over-merge (one query grabbing several objects). Those are our two named failure modes. |
| **PQ^sc** (scene-level PQ) | Panoptic Lifting (CVPR 2023, [2212.09802](https://arxiv.org/abs/2212.09802)); PanSt3R (ICCV 2025, [2506.21348](https://arxiv.org/abs/2506.21348)) | quality *and* consistency in one number, by concatenating views | **ALREADY HAVE IT** as `bundle_*` (AP/mIoU rather than PQ). Keep as headline; do not re-implement PQ — it needs stuff classes and a different matching rule, and would not be comparable to anything in docs/RESULTS.md. |
| **STQ = √(AQ·SQ)**, AQ = association quality | STEP, Weber et al. ([2102.11859](https://arxiv.org/abs/2102.11859)) | pixel-level association over a whole video, threshold-free | **SKIP.** Same decomposition idea as HOTA but its SQ is *semantic* mIoU (class-level, instance-agnostic), which does not fit an instance-only 19-class head. HOTA gives us the same split in a form that matches our GT. |
| **IDF1** | Ristani et al., ECCVW 2016 | identity-based F1 after one global ID assignment | **OPTIONAL.** Cheap once HOTA's count matrices exist (~10 lines), widely recognised, but known to over-weight association vs detection — the exact complaint HOTA was written to fix. Report only if a reviewer-familiar number is wanted. |
| **VPQ / track mAP (YouTube-VIS)** | Kim et al. 2020 / Yang et al. 2019 | tube-IoU AP over a temporal window | **SKIP** — tube AP over the full window *is* `bundle_AP50`. Nothing new. |
| CVSC / MRC style feature-agreement scores | multi-view diffusion & reconstruction papers | appearance/geometry agreement across views | **SKIP** — measures reconstruction, not instance identity. |

**Framing note (important, matches docs/RELATED_WORK.md).** Adopting HOTA is *using a standard
metric on a new task*, not inventing a metric. The contribution stays "the controlled study";
this plan makes one of its claims measurable. Do not present AssA as novel.

## 3. The mapping: a bundle is a video, views are timesteps

Everything below rests on one substitution, which must be stated explicitly in the docs because
it is what makes tracking metrics legal here:

| tracking concept | our object |
|---|---|
| video / sequence | one **bundle** of S=8 frames from one scene |
| timestep *t* | view *f* (**unordered** — no metric below uses frame order, which is correct: `CrossFrameAttention` is permutation-equivariant in S) |
| GT track id | the dataset **global instance id** (`build_bundle_target` → `global_ids`, `valid [n,S]`, `frame_row`) |
| predicted track id | the **query index** *q* (with `--multi_frame`, query *q* is one object hypothesis in all views by construction) |
| detection at (t, id) | query *q*'s mask in view *f*, kept iff it is non-empty (`drop_empty_masks`) and passes the score rule of docs/MASKDINO.md §6.2 |
| ID switch | query *q* best-covering GT instance *A* in one view and GT instance *B* in another |

Two consequences that must be encoded, not assumed:

- **An empty mask in a view is a track gap, not an ID switch.** `drop_empty_masks` already
  removes it; the association counters must treat it as "no detection here", exactly as HOTA
  treats a missed frame. Getting this wrong would penalise correct occlusion behaviour and
  break the parallel with docs/MASKDINO.md §6.3.
- **A GT instance visible in only one view carries no association information.** It must be
  excluded from the association denominators (it would score a trivial 1.0 and dilute
  everything). It stays in the *detection* denominators.

## 4. Feasibility gate — MEASURE THIS FIRST (half a day, decides the rest)

The val bundles are built with `frame_sampling="even"` (`train/maskdino_data.py::prepare_scenes`)
— 8 frames spread evenly across the *whole* scene. **If the average GT instance is visible in
only ~1–2 of the 8 views, every association metric has an almost empty denominator and the whole
exercise is noise.** I could not check this offline: the scans tree is gone, only the tar
remains (memory: `sam3-gt-cross-class-duplicates`).

**Step 0 (blocking).** From `build_bundle_target(...)["valid"]` alone — no model, no checkpoint —
report over the 10 val scenes:

- histogram of *frames-per-instance* (how many of the 8 views each global instance appears in);
- fraction of instances visible in ≥2 views (**the association denominator**);
- mean number of co-visible instances per view pair;
- the same for a handful of train scenes, to know whether the training signal is any denser.

Cheapest route: a `--gt_stats_only` flag on a short SLURM job that stages the tar, runs
`prepare_scenes` with the backbone skipped, prints the table, exits. ~15 min wall clock.

**Decision gate:**

| outcome | action |
|---|---|
| ≥60 % of instances visible in ≥2 views, median ≥3 | proceed with the full plan as written |
| 30–60 % | proceed, but every association number is reported with its denominator (`n_assoc_instances`) next to it, and per-scene spread instead of a single mean |
| <30 % | **stop and re-decide.** The honest options are (a) report consistency only on the subset and say so loudly, or (b) add a co-visible bundle sampler (`frame_sampling="window"`) as a **separate eval-only view draw**, which is a protocol change and needs its own justification in docs/RESULTS.md §1 — not something to slip in silently |

This gate is the single largest risk in the plan and costs almost nothing to close.

## 5. The metric suite

Three tiers. Tier A is the deliverable; tier B is nearly free once A exists; tier C is a stretch
that produces the best figure but the weakest guarantee.

### Tier A — HOTA decomposition (the core)

Per bundle, class-agnostic (see §6), for each localisation threshold α ∈ {0.05, 0.10, …, 0.95}:

1. **Per-view matching.** For each view *f*, build the IoU matrix between kept predictions and
   the GT instances visible in *f*; Hungarian-match with IoU ≥ α. As in the official HOTA, the
   matching maximises similarity weighted by the global association score so that the assignment
   is not made greedily per frame. TP/FP/FN follow.
2. **Detection.** `DetA_α = |TP| / (|TP| + |FN| + |FP|)`, plus `DetRe`, `DetPr`.
3. **Association.** For every matched pair *c* = (query *q*, GT instance *g*):
   - `TPA(c)` = # views where *q* is matched to *g*
   - `FNA(c)` = # views where *g* is matched to some other query, or missed
   - `FPA(c)` = # views where *q* is matched to some other GT, or is a FP
   - `A(c) = TPA / (TPA + FNA + FPA)`
   - `AssA_α = mean over TP of A(c)`; `AssRe = mean TPA/(TPA+FNA)`; `AssPr = mean TPA/(TPA+FPA)`
4. `HOTA_α = √(DetA_α · AssA_α)`; **HOTA = mean over α**, and likewise DetA/AssA.

**Reported keys:** `mv_HOTA`, `mv_DetA`, `mv_AssA`, `mv_AssRe`, `mv_AssPr`, `mv_HOTA50`,
`mv_AssA50` (α = 0.5, the one people read), `mv_n_assoc` (the denominator from §4).

**`mv_AssA` is the headline consistency number of this project.** `AssRe` low = the model
fragments an object across queries; `AssPr` low = a query merges several objects across views.

Implementation note: reimplement (~150 lines, numpy + `scipy.optimize.linear_sum_assignment`,
which `train/eval_metrics.py` already uses) rather than vendoring TrackEval — it is CPU-only,
dependency-free and matches the repo's style. Validate once against a hand-computed example in
the test file, and (offline, one-off, not in the repo) against TrackEval on a synthetic sequence.

### Tier B — the two interpretable numbers, and the decomposition of `bundle_*`

These cost almost nothing on top of tier A's IoU matrices and are much easier to defend in a
thesis chapter than HOTA's counters.

- **CVIC (cross-view identity consistency).** For each GT instance visible in *k* ≥ 2 views: in
  each of those views take the query with the highest IoU against it; `CVIC(g) = (count of the
  modal query) / k`. Mean over instances. **ID-switch count** `= k − modal count`, summed.
  Range `[1/k, 1]`; a per-frame model with arbitrary query slots sits near the floor, a perfectly
  consistent model at 1.0. This is literally the metric `docs/RELATED_WORK.md` gap 2 asks for
  ("IoU agreement of its mask identity across views / ID-switch rate"). It is close to a
  simplified `AssRe`; report both and say so — CVIC for the narrative, AssA for rigour.
- **Query purity (the dual).** For each query active in ≥2 views, the fraction of its active
  views whose best-matching GT is its modal GT. Catches over-merge in the same units.
- **Cross-view class consistency.** Fraction of active queries whose argmax class is identical in
  every view where they are active. **GT-free.** With `--multi_frame` the content embedding is
  shared, so this should be ≈1.0 and is mostly a *sanity check on the mechanism*; on the
  single-frame model it is a genuine measurement.
- **The association gap on our own ruler.** Re-score `bundle_AP50` twice: once as today, once
  with **oracle association** (every per-frame prediction re-labelled with the id of the GT
  instance it best matches, then volumes rebuilt). `Δ = bundle_AP50(oracle) − bundle_AP50(model)`
  is "what perfect cross-view identity would be worth, holding per-frame quality fixed", in the
  same units as the number already in docs/RESULTS.md §3. **This is the most quotable single
  result the plan can produce** and it reuses existing code end to end.

### Tier C — GT-free geometric consistency (stretch, decide after A/B land)

Use the frozen point head's per-patch 3D positions (already sketched in docs/MASKDINO.md §8.3
step 1: `[S, 37·37, 3]` + confidence, ~65 kB/scene) to lift each query's per-view mask into 3D,
then measure agreement between the point sets a query produces in different views (voxel IoU on
a coarse grid, or symmetric Chamfer). Needs **no ground truth**, so it runs on any scene and
makes a strong figure ("query 42's mask lands on the same chair from all 8 views").

Two honest caveats: it measures agreement with VGGT's own geometry, which may be wrong; and it
requires the geometry cache that §8.3 has not built yet. **Recommendation: defer.** Only build it
if §8.3 (3D anchors) is actually started, since it shares the cache — otherwise it is a
standalone dependency for a diagnostic.

## 6. Protocol decisions (each needs your yes/no)

1. **Class-agnostic association, class-aware detection.** Association is about identity, so
   matching for AssA/CVIC should ignore class; otherwise a class flip is scored as an ID switch
   and the two failure modes are entangled. `DetA` stays class-aware, consistent with everything
   else in `train/eval_metrics.py`. Cross-view class consistency (tier B) covers the class side
   separately. *Recommended: yes, and report class-aware AssA as a secondary column.*
2. **Two operating points, as everywhere else** (docs/MASKDINO.md §6.2): thresholded
   (`--score_threshold 0.25`) and `_all`. Association metrics are much more sensitive to the
   threshold than AP is (more kept queries → more chances to switch), so both are needed.
3. **`--eval_topk 100` stays applied** before association scoring, for the cost reason of
   docs/MASKDINO.md §7.2.2 and to keep the protocol identical to the AP numbers.
4. **Denominator honesty.** Every association number is reported next to `mv_n_assoc` (instances
   visible in ≥2 views) and the per-scene spread. With 10 val scenes × 1 bundle the sample is
   small — the established eval-to-eval noise band is ±0.04 AP50, and association metrics on a
   smaller denominator will be *noisier*, not less. Report per-scene min/max, and a bootstrap CI
   over scenes if the spread turns out to be wide.
5. **Where it runs.** Tier A + B on the val bundles inside `eval_scenes_multiframe`, logged to
   `metrics.jsonl` like everything else. Cost estimate: Q ≤ 100 after topk, S = 8, 37×37 masks,
   10 scenes, 19 α values → well under a second per eval, i.e. invisible next to the ~6 s eval.
   The **controls in §7 run offline** in a separate scorer, so training cost stays at zero.

## 7. The controls — without these the numbers mean nothing

An AssA of, say, 0.62 is uninterpretable on its own. Five reference points, in priority order:

| # | Reference point | How | What it tells us |
|---|---|---|---|
| 1 | **Oracle association (ceiling)** | single-frame model's per-frame predictions, ids assigned from best-matching GT | upper bound achievable at this detection quality; gives the §5-B association gap |
| 2 | **Shuffled ids (floor)** | permute query ids independently per view | proves the metric has dynamic range; AssA should collapse to ≈1/S |
| 3 | **Post-hoc matching baseline** | single-frame model + cross-view Hungarian matching of per-frame masks, using decoder query embeddings (cosine) and/or lifted-3D IoU | **the decisive experiment.** This is the PanSt3R/MV3DIS paradigm we position against. If `--multi_frame` does not beat it, shared queries are not earning their keep and we must say so |
| 4 | **`--no-cross_frame_attn`** (job 8950617, running) | score its checkpoint on the suite | attributes consistency to the cross-frame block vs to shared init + bundle matching alone |
| 5 | **Single-frame model, query index as id** | score the 0.699 bar's checkpoint on the suite | how much cross-view identity DETR query slots give for free — expected low, and that is the point |

Checkpoints for 1, 2, 4, 5 already exist under
`/cluster/work/igp_psr/niacobone/distillation/output/` (`maskdino_sf_n490_*`), so most of the
table is a scoring job, not a training job. Control 3 is the only one needing new code beyond the
metric itself (~a day), and it is the one that carries the argument.

Arm C is **out of scope**: `scripts/eval_perframe.py` could feed a D4RT checkpoint through the
same suite (its queries are multi-view by construction, so it is well-defined and would be a fair
extra column), but it is not needed for the question you asked. Add it later if the thesis wants
the full matrix — do **not** modify anything under `legacy/`.

## 8. Implementation shape (for when you approve)

```
train/multiview_consistency.py     NEW. Pure functions: masks/ids in, metric dict out.
                                   No model, no dataset, no CUDA → fully CPU-testable.
                                   hota(), cvic(), query_purity(), class_consistency(),
                                   oracle_association_ids(), shuffled_ids()
train/maskdino_eval.py             MODIFIED. eval_scenes_multiframe() adds mv_* keys.
                                   Behind --mv_metrics (default OFF, per CLAUDE.md working
                                   rules: new options default to previous behaviour).
scripts/eval_multiview.py          NEW. Offline scorer: loads any maskdino checkpoint
                                   (single- or multi-frame), runs the full suite + all five
                                   controls, writes multiview_eval_<ckpt>.json + a table.
                                   Mirrors scripts/eval_perframe.py in structure.
tests/test_multiview_consistency.py NEW, CPU-only, standalone (repo convention):
                                   - planted perfectly-consistent bundle → AssA = CVIC = 1.0
                                   - one planted ID switch → exact hand-computed value
                                   - shuffled ids → floor
                                   - instances visible in 1 view excluded from assoc denominator
                                   - empty mask in a view = gap, not a switch
                                   - S=1 degenerate → association metrics undefined/NaN, not 0
                                   - hand-worked HOTA example matching the paper's definition
docs/MASKDINO.md §6.4 (new)        the protocol; docs/RESULTS.md §3.x the table;
docs/RELATED_WORK.md gap 2         mark substantiated; docs/todo.md; CLAUDE.md test list
```

Phasing, with a gate after each:

| Phase | Work | Cost | Gate |
|---|---|---|---|
| 0 | GT co-visibility statistics (§4) | ~half a day | **blocking** — decides whether the rest is meaningful |
| 1 | `train/multiview_consistency.py` + tests, tier A + B | ~1 day | tests green, hand-computed values match |
| 2 | Wire into `eval_scenes_multiframe` behind `--mv_metrics` | ~2 h | existing tests still pass unchanged |
| 3 | `scripts/eval_multiview.py` + controls 1, 2, 4, 5 on existing checkpoints | ~half a day + GPU scoring | the first real table |
| 4 | Control 3 (post-hoc matching baseline) | ~1 day | the decisive comparison |
| 5 | Docs + results | ~2 h | — |
| C | Tier C geometric consistency | ~1–2 days | only if §8.3 starts |

## 9. What could make this fail, stated up front

1. **Empty denominator** (§4). Mitigated by the phase-0 gate.
2. **Small-sample noise.** 10 val scenes × 1 bundle. Mitigation: per-scene spread + bootstrap CI,
   and never quote a consistency delta smaller than the measured spread.
3. **The result may be unflattering.** If `--multi_frame`'s AssA barely beats the post-hoc
   matching baseline, the honest conclusion is "shared queries buy little over matching on this
   data" — which, given docs/RELATED_WORK.md already says shared queries are table stakes rather
   than our contribution, is a publishable negative result inside the controlled study, not a
   failure of the plan. Decide now that we report it either way.
4. **Metric-shopping risk.** Fix the suite and the operating points *before* scoring any
   checkpoint, and score every control with the identical code path — the same discipline
   `train/perframe.py` enforces for the two model families.

## 10. Open questions for you

1. Phase 0 first, or implement tier A in parallel and accept the risk? (Recommendation: gate.)
2. Class-agnostic association as the primary (§6.1) — agree?
3. Is the post-hoc matching baseline (control 3) in scope now, or deferred? It is the strongest
   argument and the largest single chunk of work in the plan.
4. Tier C (geometric, GT-free) — defer until §8.3 starts, or is the figure wanted sooner?
5. Should arm C be scored on this suite too (an extra column in the thesis), or is
   single-frame-MaskDINO-as-floor enough?
