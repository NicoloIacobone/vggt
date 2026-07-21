# Plan-backbone: "Global overview" presentation (requested by supervisor, meeting 2026-07-16)

> Purpose of this file: the skeleton for a self-contained deck giving a **global understanding
> of the whole project up to now** — arms, dataset, metrics, competitors, results, strengths &
> limitations. Not yet slide-final: each section lists (a) the content, (b) the exact numbers
> to quote, (c) the figure/screenshot/schema to show and whether it exists or must be built.
> Update this file, then feed it as the prompt to the agent that writes the final deck
> (Marp format like `docs/slides_meeting_jul_16.md` worked well).
>
> Sources of truth: `docs/MILESTONES.md` (narrative + all tables), `docs/ARMS_SUMMARY.md`
> (arm comparison), `docs/RELATED_WORK.md` (competitors), `CLAUDE.md` (storage/commands).

**Deck size target:** ~20–25 slides. Audience: supervisor + anyone new to the project
(assume they know DETR/VGGT at a high level, nothing about our work).

---

## Part 1 — The problem & the idea (2–3 slides)

1. **Title + one-sentence pitch.** "Multi-view consistent 3D instance segmentation from a
   *frozen* VGGT backbone — a DETR-style decoder head, and a systematic study of how its
   queries should be initialized."
2. **Problem slide.** Task: given S RGB frames of a ScanNet scene, predict per-instance masks
   that are *consistent across views* (same object = same ID in every frame). Constraints
   that define the design space: backbone frozen (only ~6.5M head params train), feed-forward
   (no per-scene optimization — the "why not splatting" contrast), no depth sensor / GT
   geometry at inference.
3. **Key framing slide (positioning).** "Decoder on frozen VGGT" is now a crowded genre
   (VGGT-X family). The architecture is NOT the contribution — **the query-strategy ablation
   is** (4 query-init families, 8 trained variants, one winner, every closure explained).
   State this up front so the rest of the deck reads as evidence for it.

**Figures**
- [BUILD — schema] Task illustration: 3–4 frames of one scene with the same instance
  highlighted in the same color across frames. Can be assembled from any auto-rendered
  overlay (GT panel) of a val scene.
- [BUILD — schema] "Design space" mini-diagram: frozen backbone block + trainable head block
  with param counts (1B frozen vs 6.5M trained).

---

## Part 2 — Architecture: how it works (3–4 slides)

1. **Pipeline schema (the central figure of the deck).** VGGT aggregator (24 blocks, frozen,
   `no_grad`) → hook at `aggregated_tokens_list[-1]` → features `F: [B,S,P,2048]` →
   QueryGenerator (the part the arms vary) → InstanceDecoder (4-layer TransformerDecoder,
   queries=tgt, projected F=memory) → class head (19 classes + background), mask-embed head,
   dense mask head → `pred_masks [B,N,S,h,w]`.
2. **The consistency-by-construction point.** One query = one instance *in all S views
   simultaneously* (the mask tensor has the S dimension inside one query slot). Contrast with
   the fuse-per-view-2D-masks paradigm (PanSt3R). This is a structural claim, currently
   unmeasured → honest "future work: turn it into a metric" note.
3. **Training loop.** Hungarian matching (Dice+BCE cost) + Focal class loss + Dice +
   fg-weighted BCE + DETR no-object loss. Frozen-backbone features cached once per scene
   bundle up front → epochs run head-only (minutes, not hours; this is also the deliberate
   protocol tradeoff vs per-iteration frame sampling).
4. **(Optional, backup) Hard-won constraints slide** — the "what silently breaks it" list:
   LayerNorm on projected memory + query skip (else all queries collapse to one vector),
   cosine-sim mask logits with learnable temperature, fg pos_weight, coords are prompts not
   predictions. Good for questions; shows debugging depth.

**Figures**
- [BUILD — schema, the most important one] Pipeline block diagram with tensor shapes at each
  edge and a highlighted "queries enter here" box (this box is what Part 4 swaps out per arm).
  Suggest one master version + per-arm recolored variants for Part 4.
- [EXISTS] `docs/HOOK_PLAN.md` has the hook description if the agent needs details.
- [BUILD — small schema] Matching/loss cartoon: N query slots ↔ GT instances bipartite
  matching, unmatched → "no object".

---

## Part 3 — Dataset & GT story (2–3 slides)

1. **Dataset composition.** ScanNet v2; 500 scenes (scene0000–0499); per scene ~100 stride-5
   frames (`subset/`), official 2D instance GT projected from the human-verified 3D
   annotation; **7379 instances, 0 cross-class duplicates**; 19 trainable NYU40 classes +
   background; masks binary PNG per instance, evaluated at the 37×37 patch grid (image 518²).
   Held-out val = scenes 0080–0089 (13.3 GT inst/scene). Training input: bundles of 8 frames.
2. **The GT migration story (worth a full slide — it's a quality-control credential).**
   Original GT = SAM3-generated (per-class prompting + video tracking). Audit 2026-07-07:
   **~15.9% of foreground pixels multi-class**, ~3.4 duplicate instances/scene
   (desk↔table, curtain↔shower_curtain, …) → matcher demands two predictions per object =
   built-in false positives. Migrated to official ScanNet GT 2026-07-08.
   Consequence for all numbers: same arm-C checkpoint scores honest AP50 **0.228 on SAM3 val
   but 0.117 on official val**; retrained on official GT → **0.199**. ⇒ ~half the old
   headline was fitting label noise. All quotable numbers in this deck are official-GT.
3. **(Optional merge into 1)** Data pipeline in one line: tars on work storage → staged to
   node-local scratch per job; SAM3 tar kept as GT-quality baseline.

**Numbers to quote:** 500 scenes / 7379 instances / 0 duplicates; SAM3: 200 scenes,
~4195 instances, 15.9% multi-class px; the 0.228 → 0.117 → 0.199 triplet.

**Figures**
- [BUILD — screenshot] A SAM3 duplicate example: same object with two class masks side by
  side vs the official single mask. Needs unpacking one scene from the SAM3 tar (or reuse
  the audit material if any images survive). If too costly: a 3-bar figure of the
  0.228/0.117/0.199 triplet tells the story alone.
- [BUILD — table or pie] Dataset composition mini-table (scenes / frames / instances /
  classes / resolution / val split).

---

## Part 4 — The arms: one slide per arm (5–6 slides; the core of the deck)

Uniform per-arm template — same layout for A, B, C, D, E so comparison is visual:
- **Top:** the pipeline schema with the query box swapped to this arm's mechanism (small).
- **Left:** How it works (2–3 bullets) + why it was tried.
- **Right:** Results table (N=50 and, where run, N≈190–200) + qualitative note.
- **Bottom strip:** Strengths / Limitations / Verdict (closed or base).

Per-arm content (all numbers already in `docs/ARMS_SUMMARY.md` — tag (S)/(O) for GT!):

| Arm | How it works (1-liner) | Quote | Strengths | Limitations / verdict |
|---|---|---|---|---|
| **A point** | Fourier(u,v) + view emb + RGB patch; GT centroids at train, uniform grid at honest eval | N=50: 0.212/0.125 (S); plateaus by N=100–200 (0.216/0.105) | simple, promptable | honest eval needs grid → duplicate FPs, no NMS; **plateaued** — superseded |
| **B trained grid** | A + random-offset grid queries trained too (`--no_object_norm matched` fix) | fixed: 0.284[grid]/0.161 (S); N=190: 0.372[grid]/0.185 unstable | best point-family AP50 at N=50 | v0 collapsed masks (loss dilution — good failure-analysis story); AP50 never stable at scale — closed |
| **C learned (DETR)** | 64 learned `nn.Embedding` object queries, no coordinates at all | **BASE. N=190 official GT: val mIoU 0.367 / honest AP50 0.199**; (S) N=200: 0.371/0.228 | GT-free ⇒ prompted==honest; broke the plateau (>2× point AP50); overfitting resolved at scale | over-predicts (kept/GT 1.23–1.38×); slight val decay late in training |
| **D hybrid** | learned slots + centroid prompts (`--learned_query_lr_scale 0.1` NaN fix) | 0.247/0.146 (S), then decays | was best-arm before v0 crash | v0 NaN'd (fixed); centroid path reintroduces overfitting; only ties C — closed |
| **E anchor3d (v0 + 3-variant v1)** | queries seeded from VGGT's *own* predicted pointmap: FPS anchors over token cloud; content ∈ {pooled, learned, none} | best E: pos-only 0.230 mIoU / hybrid 0.121 AP50 (O); bar was C 0.269/0.144 | **calibration**: only lever that fixed over-prediction (kept/GT 0.59–0.86× vs C 1.23×); least-overfit run of all arms (pos-only); clean failure decomposition (pooled frozen features actively harmful, not the Fourier encoding) | never beats C on quality at N=50 — closed; deliverable = the ablation story |

Close Part 4 with **one summary slide**: the `ARMS_SUMMARY.md` results table condensed +
the three cross-arm takeaways: (1) GT-free query generation (C, E) is both more honest and
stronger; (2) grid-based honest eval dies by duplicate FPs (architectural, confirmed by the
grid-density sweep 2–12: best grid 0.185 < C 0.228, kept preds explode with density);
(3) geometry regularizes and calibrates but caps detection.

**Figures**
- [BUILD — the key schema set] 5 small query-init diagrams (same canvas): A = dots at
  centroids on an image; B = A + jittered grid; C = floating embedding slots (no image);
  D = C+A; E = 3D point cloud with FPS anchor dots. These carry the whole deck.
- [BUILD — chart] Grouped bar chart: val mIoU and honest AP50 per arm (official-GT N=50
  where available, tag SAM3 numbers). Data: ARMS_SUMMARY tables.
- [BUILD — chart] Calibration figure: kept/GT ratio per arm (0.59–0.86× E family, 1.23× C),
  horizontal line at 1.0. This is the arm-E finding in one image.
- [BUILD — chart] Overfitting profiles: train vs val mIoU curves per arm from each run's
  `metrics.jsonl` (esp. C's decay vs E-pos-only's flatness). Script to write; ~20 lines of
  matplotlib on top of `scripts/plot_scaling.py` patterns.
- [EXISTS] Qualitative 4-panel overlays (RGB | GT | honest | oracle) in
  `<run_dir>/visualizations/` for the arm-E runs and old arm-C runs.
  **ACTION NEEDED:** the quotable baseline run
  (`d4rt_full_inst_learned_officialgt_20260708_124452`) has NO visualizations dir — render
  with `scripts/visualize_masks.py --checkpoint .../checkpoint_best.pth --scans_root
  /cluster/scratch/niacobone/scannet_official_build/scans` (light checkpoint reloads frames
  from disk). Pick 2 good val scenes + 1 failure case (window/door confusion).

---

## Part 5 — Scaling & side levers: what we ruled out (2 slides)

1. **Data scaling.** Arm-A curve N=10→200 plateaued (mIoU ~0.22, AP50 ~0.10 — table in
   MILESTONES). Arm C at N=490 (2.6× data, official GT): **0.350/0.177 ≤ the N=190
   baseline** → data quantity is not the lever (caveat: bundles=1 recipe deviation, forced
   by the NUMA/RAM infra finding — 30-second aside, it's a good war story).
2. **Side levers, all neutral/negative (one table slide):** mask upsampling ×2 (wash:
   0.236 vs 0.228 AP50 → resolution not the bottleneck; confusion is semantic);
   grid-density sweep (negative, architectural); score threshold 0.5→0.3 (negative: +76
   kept, 2 correct → over-prediction, not under-confidence). Frame as: "we know where the
   ceiling is NOT."

**Figures**
- [EXISTS, needs update] `scaling_curve_full.png` in the output root — regenerate with
  `scripts/plot_scaling.py` to include the arm-C points and the N=490 point (two series:
  point arm vs learned arm; y = val mIoU and honest AP50).
- [BUILD — chart] Grid-density sweep line plot (AP50 vs grid size, arms A and B) from
  `grid_ablation_*.json` — shows the "peaks at training density, dies by duplicates" shape.

---

## Part 6 — Metrics: which and why (1 slide)

- **mIoU** (matched-instance mask quality) — Hungarian-matched, so it answers "are the masks
  good?" but not "did you find the objects?".
- **AP50 / AP75 / mAP** (detection) — penalizes FPs and duplicates.
- **Prompted vs unprompted (the honest column):** prompted = queries at GT centroids
  (upper-bound diagnostic); unprompted = grid or GT-free queries. **Honest val[grid] AP50 is
  the headline number** because unprompted mIoU is optimistic (unmatched FPs unpunished) —
  learned-vs-grid arms differ exactly here. For C and E, prompted == unprompted by
  construction (a selling point).
- **kept/GT calibration ratio** (ours): honest kept predictions ÷ GT instances — the
  over/under-prediction diagnostic that decided arm E's story.
- Model selection: best-mIoU and best-AP50 checkpoints diverge → both saved (mention why).

**Figures**
- [BUILD — small schema] Prompted vs unprompted cartoon (centroid dots vs uniform grid on
  the same frame). Optional; the concept can also live in a footnote.

---

## Part 7 — Competitors (2 slides)

1. **Landscape.** "VGGT-X" genre exploded in the last 12 months (822-paper harvest, 113
   on-topic). Direct lane: SegVGGT (closest — object queries on multi-level VGGT features,
   ScanNetv2/200, **no query-init ablation** → our gap holds), EPS3D (panoptic, mutual
   semantic-instance enhancement), FAST3DIS (anchored queries on frozen depth backbone —
   check what "anchored" means before claiming novelty), PanSt3R (the post-hoc 2D-fusion
   paradigm we contrast with). Caution note to keep: VGGT-Segmentor was misclassified by
   the harvest (it's ego↔exo mask transfer) — spot-check before citing anything.
2. **Their addition vs our arms (the comparison the supervisor asked for).** Table: paper |
   what they bolt onto the backbone | query init | consistency mechanism | eval protocol.
   Punchlines: (a) nobody publishes the query-init ablation — that's us; (b) our masks are
   per-view 2D at patch resolution vs their lifted 3D point-cloud masks → **numbers are NOT
   directly comparable; do not put their AP next to ours in one table** until the protocol
   alignment work is done (SegVGGT line-by-line read is an open todo). Honest slide:
   "quantitative comparison = future work, here is why."

**Figures**
- [BUILD — schema] 2×2 or quadrant positioning map: axis 1 = query init (image-space /
  learned / geometry-anchored), axis 2 = consistency (post-hoc fusion / by construction).
  Place SegVGGT, PanSt3R, FAST3DIS, EPS3D, and our arms C & E on it.
- [MAYBE EXISTS] The harvest landscape figure lives with the project owner (not in repo) —
  reuse if presentable.

---

## Part 8 — Qualitative results & failure modes (1–2 slides)

- 2–3 four-panel overlays (good val scene, mid scene, failure scene). The honest panel = the
  same selection rule as the 3D viewer, no GT involved — say so.
- Known failure modes (from MILESTONES §qualitative): window↔door↔picture↔curtain confusion
  (flat wall-mounted rectangles at 37×37 — semantic, survives upsampling); coverage gaps for
  classes absent in training scenes (toilet→chair); over-prediction/duplicates for arm C.
- Optional: one 3D-viewer screenshot (`demos/demo_gradio.py`, "Color By: Predicted
  Instances") — the money shot for "multi-view consistent" claims.

**Figures**
- [ACTION] Re-render baseline overlays (see Part 4 action), pick panels.
- [BUILD — screenshot] Gradio 3D viewer screenshot on the best arm-C checkpoint.

---

## Part 9 — Wrap-up (2 slides)

1. **What's established.** Query-strategy study complete (4 families, 8 trained variants);
   arm C base: **0.367 / 0.199** official GT N=190; data not the lever; resolution not the
   lever; GT quality audited and fixed; failure modes identified and mostly explained.
2. **Open directions** (mirror `docs/todo.md` / jul-16 slides): cross-view consistency
   metric (turn the structural claim into a number), protocol alignment with SegVGGT,
   which-layer ablation (nearly free), no-object-weight sweep, then the write-up.

**Figure**
- [BUILD — schema, optional] One-line project timeline: M1 prototype → M2 unprompted
  training → M3 scaling + arms A–D → GT migration → arm E + N=490 closure. Dates from
  MILESTONES.

---

## Consolidated build list (what to prepare before the final deck)

**Numbers (all final, from ARMS_SUMMARY/MILESTONES — no new runs needed):**
arm table (S)/(O) tagged; 0.228/0.117/0.199 GT triplet; 500/7379/0 dataset stats; 15.9%
SAM3 audit; arm-A scaling table; N=490 0.350/0.177; kept/GT ratios; side-lever table;
grid-density sweep values.

**Schemas to draw (no code):**
1. Pipeline block diagram + per-arm query-box variants ← highest value
2. 5 query-init mini-diagrams (one canvas style)
3. Positioning quadrant (competitors vs our arms)
4. Prompted-vs-unprompted cartoon (optional), timeline (optional)

**Charts to script (data all on disk):**
5. Arms grouped bars (mIoU + honest AP50)
6. Calibration bars (kept/GT, line at 1.0)
7. Updated scaling curve incl. arm C + N=490 (`scripts/plot_scaling.py`)
8. Train-vs-val overfitting profiles per arm (from `metrics.jsonl`)
9. Grid-density sweep line plot (from `grid_ablation_*.json`)

**Screenshots to produce:**
10. Re-render official-GT baseline overlays (run `visualize_masks.py` on
    `d4rt_full_inst_learned_officialgt_20260708_124452` with `--scans_root` at the scratch
    build tree) → pick 2 good + 1 failure
11. Arm-E vs arm-C honest-panel side-by-side (both viz dirs exist) — shows the calibration
    difference visually
12. Top-down point-cloud pair (RGB | predicted instances) — now scriptable, no Gradio
    needed: `scripts/render_pointcloud_topdown.py --checkpoint <best_ckpt> --scene_dir
    <scans_root>/<scene>/raw_data --gray_classes wall,floor` (drop --gray_classes to also
    color stuff instances). A Gradio 3D viewer screenshot remains an optional extra angle
13. SAM3 duplicate-GT example (if cheap; else the 3-bar GT triplet chart substitutes)

**Open decisions before finalizing (discuss with supervisor):**
- Deck length (20–25 full story vs ~15 condensed)?
- Include the SAM3-era (S) numbers at all, or only official-GT and a one-line "history" note?
- How much infra/debugging war-story (NUMA, NaN fixes) — 0, 1 aside, or a backup section?
- Show competitor numbers verbally only, or attempt a caveated table?
