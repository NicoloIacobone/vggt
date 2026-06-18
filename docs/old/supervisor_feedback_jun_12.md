# Supervisor Feedback on Slides (Jun 12, 2026) — Analysis & Action Plan

Point-by-point response to the supervisor's comments on the Jun-11 slides
(`docs/slides_meeting_jun_11.md`), each checked against the actual code. Written offline;
all actions are queued for the next cluster session. Companion to
`docs/SCALING_RUNS_ANALYSIS.md` (the scaling-protocol fixes there come first — these
experiments should run on top of the corrected protocol).

---

## 1. "Same color ≠ same instance ID across frames; you color by category"

**The code says otherwise — colors ARE per instance, and cross-frame identity is
architectural — but the figures cannot prove it with the current GT, so the confusion is
justified.**

Facts (verified in `scripts/visualize_masks.py:154-179`):

- Each **matched (prediction, GT) pair** gets one palette color (`color_i` indexes the
  match, not the class). The legend prints class *names*, which is what creates the
  "colored by category" impression.
- A single query emits **one mask spanning all S frames jointly** — `pred_masks` is
  `[N, S, h, w]` (`models/d4rt_decoder.py`), and the visualizer draws `pred_prob_full[p, s]`
  for the same query `p` in every frame `s`. There is **no per-frame detection + ID-linking
  step that could break identity**: same color = same query = same instance, by
  construction. This is precisely the multi-view-consistency selling point of the
  decoder design.
- **However**: the SAM3 GT is per-class *binary* masks, so the dataset assigns exactly one
  global instance per class (`data/scannet_overfit.py`; known limitation, MILESTONE_1
  §8.3). With at most one instance per class in every scene, per-instance and per-category
  coloring are **visually indistinguishable** — no current figure can demonstrate
  instance-level (as opposed to class-level) cross-view consistency.

**Actions:**
- Reply to supervisor with the two halves: identity is architectural (one query → one
  multi-frame mask), but a *demonstration* of same-class instance separation needs the
  per-instance GT from comment 5 — the two comments resolve each other.
- Once per-instance masks exist, the existing visualizer will automatically show two
  same-class objects in different colors (colors already track matches). Small polish
  edit for that day: legend entries `"{class} #{k}"` instead of `"{class}"` so duplicate
  class names are distinguishable (`visualize_masks.py:192`).
- Optional slide fix in the meantime: caption the overlay figures "one color per predicted
  instance (mask spans all frames jointly)" to preempt the category reading.

## 2. "Solve 37×37 masks MaskDINO-style: decoder embeddings convolved with a high-res feature map"

**Agreed — and it's a natural extension of what's already there.** The current mask head is
already the Mask2Former/MaskDINO *mechanism* (query mask-embeddings ⊗ feature map via
cosine similarity), only the feature map is the raw 37×37 patch grid. The missing piece is
a **pixel decoder** that upsamples before the product. This was flagged abstractly at the
end of MILESTONE_2.md §7 ("FPN-style pixel decoder"); MaskDINO makes the recipe concrete.

Implementation sketch (new component, follows the repo conventions — own file + CPU test +
default-off flag):

- **Pixel decoder module** (`models/` — e.g. `mask_upsampler.py`): input the cached
  per-frame patch features `[S, 37, 37, 2048]` (reshaped from the bundle's
  `features`/`num_patch_tokens`), project to mask_embed_dim (256), then 2–3
  `ConvTranspose2d`/bilinear+conv stages → `[S, 256, 74, 74]` or `[S, 256, 148, 148]`.
  Mask logits stay `einsum(query_embed, pixel_features)` with the learnable-temperature
  **cosine similarity — keep it** (hard-won constraint, MILESTONE_1 §6).
- **Supervision**: GT masks exist at full image resolution (`masks/<class>/*.png`) and are
  currently downsampled to 37×37 in the dataset — downsample to the new output resolution
  instead. **Memory caveat**: dense Dice+BCE on `[N, S, 148, 148]` is 16× the current mask
  tensor; if it doesn't fit, adopt Mask2Former's point-sampled mask loss (sample ~3k
  points/mask instead of dense supervision) — that is what makes high-res training cheap
  in the original papers.
- **Keep it off by default** (`--mask_upsample 1` = current behavior) so all existing
  tests pass unchanged; add `head_config["mask_upsample"]` so the checkpoint→demo
  round-trip keeps working (CLAUDE.md convention).
- Note: VGGT's own DPT heads already upsample the same tokens to full resolution — a
  frozen shortcut would be reusing the depth-head's intermediate DPT features as the
  high-res map. Heavier on cache memory but zero new upsampling params; worth keeping in
  mind if the learned pixel decoder underperforms.

**Priority**: after the duplicate-suppression fix (§3) and instance-mask data (§4) — those
change *what* is learned; this changes mask sharpness. Independent enough to develop in
parallel on CPU (phase-test style).

## 3. "If you do Hungarian matching in training you shouldn't need NMS — that's DETR"

**He's right about DETR — but our training never exercises that mechanism, which is exactly
why duplicates appear.** This is the most actionable comment of the set.

Verified mechanics:

- In DETR, the *same* learned query set runs at train and test time; Hungarian assigns one
  query per object, all other queries that fire on it get pushed to no-object → the model
  *learns* duplicate suppression.
- Here, training queries are **GT-centroid prompts (one per instance, ±0.02 jitter) plus
  random *background* points** (`train_overfit.py:200`, `train_multiscene.py:177`). Two
  queries essentially never sit on the same object during training, so Hungarian matching
  is trivial (one candidate per GT) and **the no-object loss never sees an
  "on-object-but-redundant" query**.
- The 288 grid queries exist **only at eval** (`train_multiscene.py:208-213` — the
  `grid_coordinates` are used in `eval_all`, never in the training loop). At inference,
  several grid cells land on one object; each has only ever been taught "a query on an
  object predicts that object", so they all fire → duplicate false positives → depressed
  unprompted AP50.

**Action — make training match the DETR setup instead of adding NMS** (the supervisor's
intuition, applied):

- In `make_train_queries` (`scripts/train_multiscene.py:177`), optionally append the same
  6×6×S grid used at eval (or a random-offset grid, to avoid overfitting cell positions)
  to the centroid+background queries. The Hungarian matcher then assigns each GT to its
  single best query and `no_object_weight` pushes every *other* on-object query to
  background — duplicate suppression is now actually trained.
- Cost: queries go 32 → 320 per step. The decoder is 4 transformer-decoder layers over
  N queries; training is currently 3–4 min/run, so even ~5–10× on the head forward is
  irrelevant.
- New flag, default off per conventions: e.g. `--train_grid_queries` (and store in
  checkpoint args). The loss/matcher already handle arbitrary N and unmatched-query
  no-object loss — **no change needed in `train/loss.py`**.
- **Experiment**: scale10/scale25 (corrected protocol from SCALING_RUNS_ANALYSIS.md §4)
  with vs. without `--train_grid_queries`; success metric = unprompted val AP50 (the gap
  vs prompted should close substantially). If duplicates persist, *then* discuss NMS as a
  band-aid — but expectation is with him: properly trained one-to-one matching should make
  NMS unnecessary.
- Side effect worth watching: with grid queries trained, the model may rely less on the
  prompt coordinate — a soft step toward learned queries (§5).

## 4. "Label format for the next SAM3 run: instance"

**Decision recorded: per-instance masks.** This unblocks the §8.3/§7.5 limitation and
comment 1's demonstration. Downstream changes when the data lands:

- **Preprocessing side**: emit one binary mask per *instance* (e.g.
  `masks/<class>/<instance_k>/` or `masks/<class>_<k>/` — any layout works, but keep
  per-frame filenames aligned with `subset/` frames as today). Cross-frame instance
  identity must come from SAM3 tracking/propagation — that is now the load-bearing
  assumption of the GT, worth a quick visual QA on 2–3 scenes before a bulk run.
- **Loader** (`data/scannet_overfit.py`): replace "one global ID per class" with one ID
  per (class, instance); `gt["classes"]` then contains repeated class indices — the class
  head (19+bg) is untouched. Centroids/coordinates per instance work as-is.
- **Matcher/loss/eval**: already instance-based (Hungarian over GT instances); only the
  cross-view-invariant tests in `tests/test_phase2.py` need updating to assert per-instance
  (not per-class) consistency.
- **Expected metric impact**: mIoU/AP will *drop* at first (more, smaller, harder GT
  instances; chairs especially) — flag this in the next slides so it isn't read as a
  regression. Same-class separation quality becomes a new headline figure (two chairs,
  two colors — directly answers comment 1).
- Semantic vs instance vs panoptic (MILESTONE_2 §7.5) is hereby settled as **instance**;
  stuff classes (wall/floor) simply remain single instances.

## 5. "Point prompts vs learned object queries: ablate"

**Agreed — and the infrastructure makes the ablation cheap.** Planned arms, all on the
corrected scaling protocol, measured on held-out unprompted AP50/mIoU:

| Arm | Queries at train | Queries at eval | Code change |
|---|---|---|---|
| A (current) | GT centroids + bg | GT centroids (prompted) / grid (unprompted) | none |
| B (A + §3 fix) | + grid queries | same | `--train_grid_queries` |
| C (learned) | M learned embeddings (no coordinates) | same M embeddings | `QueryGenerator` variant |
| D (hybrid, optional) | learned + centroid prompts | learned only or learned+prompt | C + concat |

Implementation notes for arm C (`models/d4rt_decoder.py`):

- `QueryGenerator` currently sums Fourier(u,v) + view embedding + 9×9 RGB-patch MLP. Arm C
  replaces the lot with `nn.Embedding(M, 256)` (M ≈ 64–100), i.e. true DETR object
  queries; a `--query_mode {point,learned,hybrid}` flag on the head constructor, stored in
  `head_config` for the checkpoint round-trip.
- The matcher's coordinate cost term is meaningless for learned queries — set
  `coord_weight=0` in that mode (coordinates already carry no loss term, so only the
  matching cost is affected).
- Eval plumbing: prompted/unprompted distinction collapses for arm C (there is one query
  set); report it under the unprompted column.
- Honest expectation to discuss: with ≤50 scenes, learned queries may *underperform* point
  prompts (DETR-style queries are data-hungry and slow to converge — the classic DETR
  pathology); the interesting outcome is the crossover point as N grows. Arm B is the
  cheap middle ground that may capture most of the benefit, which is why it goes first.

---

## Consolidated execution order (next cluster session)

1. **Scaling re-runs with fixed protocol** — SCALING_RUNS_ANALYSIS.md §4 (prereq for
   everything below; ~30 min/job).
2. **`--train_grid_queries` (§3)** — smallest change, directly tests the supervisor's NMS
   claim, likely the biggest unprompted-AP win. Add a CPU test (extend
   `tests/test_milestone2.py`: grid queries included in matching → unmatched ones get
   no-object loss).
3. **Query-mode ablation (§5)** — arms B/C(/D) on N=25 or 50.
4. **Per-instance SAM3 preprocessing (§4)** — start the data run in parallel (it gates on
   GPU/SAM3 throughput, not on the head code); then loader + test updates.
5. **MaskDINO-style pixel decoder (§2)** — develop CPU-side with a phase test meanwhile;
   train once 1–4 settle.

Slide/report follow-ups: caption overlays "one color = one predicted instance (mask spans
all frames)"; add the §3.5 caveat from SCALING_RUNS_ANALYSIS.md (unprompted mIoU is
optimistic, AP50 is the honest unprompted number) wherever unprompted metrics appear.
