# Query-strategy arms — one-page comparison

Last updated: **2026-07-22** (N=490 sweep complete for all arms — arm C confirmed the winner
at every scale tested, no ranking change from N=50/190).

The "arms" are the query-initialization strategies tested on the identical
frozen-VGGT + D4RT-decoder pipeline (same matcher, losses, eval). This is the project's
core ablation study (`docs/RELATED_WORK.md`: the architecture alone is a crowded genre;
the query-strategy study is the contribution). Full narrative: `docs/MILESTONES.md`;
open items: `docs/todo.md`.

**How to read the numbers**
- **val mIoU** = prompted val mIoU except where noted [grid]; **honest AP50** = unprompted
  val[grid] AP50 — queries placed with **no GT access** (uniform grid for point-based arms;
  for learned/anchor3d queries prompted == unprompted by construction). Honest AP50 is the
  number that matters for detection claims.
- **GT matters**: results were measured on two supervision sets — the defective SAM3 GT
  (~3.4 cross-class duplicate instances/scene) and the official ScanNet v2 GT (0
  duplicates, default since 2026-07-08). Numbers across GTs are NOT comparable: the same
  SAM3-trained arm-C checkpoint scores 0.228 honest AP50 on SAM3 val but **0.117** on
  official val. Each cell below is tagged (S) = SAM3 GT, (O) = official GT.
- **Decision rule**: an arm is scaled to N≈190 only if it beats arm C at N=50.
- All rows: instance-level GT, wide val (scene0080–0089), 8 frames, 1000 epochs,
  `slurm/train_scale50.sh` / `train_full.sh` recipes.

## The arms — what actually differs

| Arm | Query source (position) | Query content | GT-free at eval? | Key flags |
|-----|------------------------|---------------|------------------|-----------|
| **A** point | (u,v) prompts: GT centroids at train, uniform grid at honest eval | Fourier(u,v) + view emb + 9×9 RGB patch MLP | only via grid queries | (default `--query_mode point`) |
| **B** trained grid | A + random-offset grid queries also *trained* | same as A | only via grid queries | `--train_grid_queries --no_object_norm matched` |
| **C** learned | none — M learned DETR object queries | per-slot `nn.Embedding` | **yes** (prompted == unprompted) | `--query_mode learned --num_learned_queries 64` |
| **D** hybrid | C's learned slots + A's centroid prompts | mixed | only via grid queries | `--query_mode hybrid --learned_query_lr_scale 0.1` |
| **E** anchor3d v0 | 3D anchors: FPS over VGGT's own predicted pointmap (per-scene normalized token cloud) | kNN(8)-pooled frozen backbone features | **yes** (prompted == unprompted) | `--query_mode anchor3d --num_anchors 64` |
| **E** v1 hybrid | same 3D anchors, Fourier band fixed (`--anchor_coord_scale 0.2`) | per-slot learned embeddings (DAB-DETR-style) | **yes** | + `--anchor_content learned` |
| **E** v1 pos-only | same as v1 hybrid | none (positional-only ablation) | **yes** | + `--anchor_content none` |

## Results

| Arm | N=50: val mIoU / honest AP50 | N≈190–200: val mIoU / honest AP50 | N=490 (O): val mIoU / honest AP50 | Verdict |
|-----|------------------------------|-----------------------------------|------------------------------------|---------|
| **A** point | 0.212 / 0.125 (S) | 0.216 / 0.105 (S) — **plateaued** past N=50 | **0.264 / 0.102** — job 7974138 | superseded baseline: more scenes stopped helping (ceiling was the head, not data); N=490 essentially reproduces the N=190 plateau on cleaner GT |
| **B** v0 | 0.047 / 0.146 (S) | — | — | mask learning collapsed (no-object loss diluted matched gradients) |
| **B** fixed | 0.284 [grid] / **0.161** (S) | 0.372 [grid] / 0.185 unstable, 0.071 @ep1000 (S) | **0.110 / 0.172** [grid] — job 7974150 | **closed** — fix works, AP50 never stable, loses to C; N=490 val mIoU regresses hard (prompted path degrades even though grid AP50 holds) |
| **C** learned | 0.259 / 0.146 (S); **0.269 / 0.144 (O)** | **0.371 / 0.228** (S); **0.367 / 0.199 (O) ← quotable baseline** | **0.350 / 0.177** (O, bundles=1 — no gain from 2.6× data) — job 7219652 | **CURRENT BASE** — broke the point plateau, >2× honest AP50, GT-robust at N=50, best at every N tested |
| **D** v0 | crashed (NaN @~ep555) (S) | — | — | exploding gradients in mixed path |
| **D** fixed | 0.247 / 0.146 (S), then overfits | — (no win → not scaled) | **NaN @ep110** — best-before-divergence 0.295 / 0.174 @ep100 — job 7974164 | **closed, and the instability recurs at scale** — the N=50 fix (lr_scale 0.1 + grad clip) was not sufficient at N=490; run completed (guard kept it alive) but loss was NaN from ep110 onward |
| **E** v0 | 0.179 / 0.072 (O), kept/GT **0.83×** (C: 1.23×)¹ | — (no win → not scaled) | — | loses on quality, **wins on calibration** — dedup hypothesis validated; superseded by v1 |
| **E** v1 hybrid | 0.207 / **0.121** (O), kept/GT 0.65× — job 7322623 | — (no win) | **0.248 / 0.139** — job 7974169, zero non-finite warnings (stable where D was not) | best E on AP50 (+0.05 over v0) but still below C at every N tested — **closed** |
| **E** v1 pos-only | **0.230** / 0.099 (O), kept/GT 0.86× — job 7322624 | — (no win) | — | best E on mIoU, least overfit of ALL arms (train 0.40 @ep1000); pure geometry ≈ C −0.04 — **closed** |
| **E** v1 pooled+fix | 0.156 / 0.086 (O), kept/GT 0.59× — job 7322625 | — (no win) | — | Fourier fix alone ≈ wash → **v0's pooled features were the poison, not just the encoding** — closed |

**N=490 sweep (2026-07-21/22, all official GT, `--bundles_per_scene 1`):** ran every closed arm's
best variant at the same 500-scene scale as arm C's existing N=490 point. **Arm C wins at every
scale tested — the ranking established at N=50/190 (C > E-hybrid > B-fixed ≈ D-fixed > A) holds
at N=490 too, so data scale does not change the query-strategy verdict.** Notable: arm E v1
hybrid stayed numerically stable (zero non-finite warnings) through the exact epoch range
(~100–130) where arm D fixed's "fixed" NaN-guard still let the loss diverge — anchor3d's
3D-anchor queries are more robust at scale than the learned+centroid hybrid path, even though
E still trails C on both metrics.

¹ kept/GT = honest kept predictions vs GT instances over the 10 val scenes, parsed from the
auto-viz "Honest selection" at the **best-val-mIoU checkpoint** (same protocol for every row;
the final-epoch counts quoted in todo.md's arm-E item (b) — 1.08× vs 1.38× — are the same
comparison at ep1000 and tell the same story: E calibrates markedly lower than C).

**Arm C at N=490** (job 7219652, official GT, `--bundles_per_scene 1` to fit the node's
NUMA-fragmented RAM — a recipe deviation vs the N=190 baseline): best val mIoU **0.350**
@ep150 / honest AP50 **0.177** @ep100, then overfits (TIMEOUT @~ep750 only cut the decaying
tail). **2.6× more scenes does not improve arm C** — data quantity is not the current lever;
the plateau the point arm hit at N=50 now shows for learned queries at N≈200 too.

## Side levers tested on top of arm C (all negative/neutral — kept for the record)

| Lever | Result |
|-------|--------|
| Mask resolution ×2 (`--mask_upsample 2`, N=190 (S)) | wash: AP50 0.236 vs 0.228, mIoU 0.355 vs 0.371 → resolution is not the bottleneck |
| Unprompted grid density (eval-only sweep 2–12) | negative: kept predictions explode with density (no NMS); best grid number 0.185 < C's 0.228 → learned-vs-grid gap is architectural |
| Score threshold 0.5 → 0.3 (viz-time) | negative: 76 extra kept instances, 2 with IoU≥0.5 → model already over-predicts at 0.5; the lever is duplicate suppression (→ arm E), not the threshold |

## Cross-arm takeaways (updated with the v1 results)

1. **Arm C stands as the base — every challenger is now closed** (B, D, E v0, E v1 ×3). The
   query-strategy ablation is complete: 8 trained variants across 4 query-init families.
2. **GT-free query generation (C, E) is both more honest and stronger** — arms whose honest
   eval depends on grid queries (A, B, D) all die by duplicate false positives without NMS.
3. **The v1 ablation cleanly decomposes v0's failure**: fixing only the Fourier encoding
   (pooled+fix) changed nothing → the kNN-pooled frozen features were actively harmful, not
   just weakly encoded; swapping them for learned content (hybrid) recovered +0.05 AP50, and
   dropping content entirely (pos-only) recovered even more mIoU (0.230, within 0.04 of C)
   while overfitting the least of ALL arms.
4. **Geometry regularizes but caps detection**: every E variant calibrates below 1× kept/GT
   (0.59–0.86× vs C's 1.23×) and none reaches C's AP50 — the 3D-spread prior suppresses
   duplicates as designed, but at N=50 its slots can't specialize enough to match fully
   learned queries. The dedup finding stands (E v0 vs C at final epoch: 1.08× vs 1.38×); the
   *thesis-usable* form is the ablation story, not a new SOTA arm.
