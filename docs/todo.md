# TODO

See `docs/MILESTONES.md` for the consolidated summary; `docs/old/` for full per-milestone
detail. This list tracks only what is still open.

## Open — next experiments (GPU)

- [ ] **Arm-C rerun on the full 500-scene official GT — TIMED OUT, inconclusive, needs
      relaunch (job 6442237, submitted 2026-07-09, killed at the 12h limit 2026-07-10).**
      `slurm/train_full.sh` updated to pull its "whole dataset" train pool from
      `scannet_official_gt_500.tar.zst` — 0000–0079 + 0090–0499 (490 scenes, was 190), same
      held-out val 0080–0089; submitted with the arm-C recipe (`INSTANCE_LEVEL=1,
      EXTRA_ARGS="--query_mode learned --num_learned_queries 64"`). **Result: only reached
      epoch 150/1000 before TIMEOUT** (`checkpoint_best.pth` epoch field confirms 150; val
      mIoU 0.313 / AP50 0.170 at that point — not comparable to the N=190 baseline's
      converged 0.367/0.199, since that run trained 450–1000 epochs). The 12h budget was
      sized off a linear 2.6x-more-bundles estimate (570→1480 cached bundles) from the
      N=190 run's ~4h/850-epoch pace; actual throughput was **~15–17x slower per epoch**,
      not 2.6x — `sacct` shows MaxRSS ≈ 250 GB, comfortably under the 350 GB request, so
      it's not host-memory swapping. Root cause NOT YET DIAGNOSED: stdout is block-buffered
      when redirected to the SLURM log and the process was SIGTERM'd by the time-limit
      kill, so all of `train_multiscene.py`'s per-epoch prints were lost — only
      `metrics.jsonl`'s eval-interval writes (flushed to disk) survived, giving epoch
      counts but no per-epoch timing to localize the slowdown (caching pass vs. train loop
      vs. eval). **Fix applied**: `slurm/train_full.sh` now `export PYTHONUNBUFFERED=1` and
      moved `${EXTRA_ARGS:-}` to the END of the python invocation (argparse keeps the last
      occurrence of a repeated flag, so it can now override any hardcoded default, e.g.
      `--num_epochs`/`--eval_interval` — previously EXTRA_ARGS sat before the hardcoded
      flags and could only add new ones).

      **Diagnostic run (job 6944946, `d4rt_full_diag`, `--time 2:00:00`,
      `--num_epochs 20 --eval_interval 5`) — bottleneck localized, ALSO timed out, but with a
      real answer this time: the entire 2h was consumed by the up-front feature-caching
      pass — it never reached the `TRAINING` banner, let alone epoch 1.** Per-bundle
      "backbone" timing (the `build_bundle` call: image transfer + jitter + query-point gen
      + one aggregator forward + `build_gt_targets`) is normally ~1.4–1.6s, matching the
      constant `[1, 8, 1374, 2048]` feature shape (compute per bundle should be independent
      of scene content). But a growing fraction of scenes spike to 10–80s+ with **no
      correlation to instance count** (scene0466 had 57 instances @1.8s; scene0472 had 13
      instances @82.4s) — ruling out GT-mask-processing cost as the driver. Spikes were rare
      early in bundle 0, common by bundle 1, and dominant by bundle 2 (490 scenes × 3
      bundles compounding the exposure window). This pattern — constant-shape compute with
      wildly variable wall time, worsening over the run — points to **contention on the
      shared 8-GPU node** (`eu-g6-057`; the first, 12h run was on a different node,
      `eu-g6-014`, which explains why it fared relatively better): other tenants' jobs on
      the same physical node competing for CPU scheduling / host memory bandwidth / local
      SSD I/O, stalling our host-side code between GPU kernel launches. Not a bug in
      `train_multiscene.py` — nothing here scales with our scene count in a way code
      changes would fix.
      **Sanity check attempt #1 (job 6962015) was INVALID — infra bug, not a real test.**
      Tried to override `--train_scenes`/`--val_scenes` back to the historical 190/10 split
      via `EXTRA_ARGS` passed through `sbatch --export=...,EXTRA_ARGS="--train_scenes
      scene0000_00,scene0001_00,..."`. `sbatch --export` splits its whole argument on every
      comma regardless of quoting, so the 190-scene comma-separated list got truncated to
      just the first entry — the run silently trained on **1 scene** (log: `Train scenes
      (1): ['scene0000_00']`), finished in 9 min, and was meaningless as a sanity check.
      (Job 6944946's finding above is unaffected — its `EXTRA_ARGS` had no commas.)
      **Fixed properly**: added a `SANITY200=1` flag to `slurm/train_full.sh` — when set, it
      switches `TRAIN` to the original 190-scene pool via bash arithmetic (`seq`), no
      comma-bearing value ever crosses `--export`. **Relaunched (job 6962655,
      `d4rt_full_sanity200`, node `eu-g6-046`) — COMPLETED cleanly, all 1000 epochs, in
      2h39m total (well under the 4h budget; 132.8 min was training alone).** Best val mIoU
      **0.350** @ep500, best val[grid] AP50 **0.196** @ep350 — matches the original
      0.367/0.199 baseline within normal run-to-run noise (different random bundle/jitter
      sampling). **Conclusion: the original recipe is healthy and fast on an uncontended
      node — the earlier stalls were specific to the 500-scene job's nodes
      (`eu-g6-014`/`eu-g6-057`), not a general cluster-load regression or a code bug.**
      Back-of-envelope for the real 500-scene point under similarly uncontended conditions:
      this run's 570 cached bundles → 132.8 min training; 490-scene run has 1480 bundles
      (2.6x) → ~5.75h training + ~20min staging (bigger tar) + ~35min caching (2.6x this
      run's ~14min) ≈ **~6.5–7h if the node behaves**, vs. the 12h that wasn't enough on a
      bad node. Next: relaunch the real 490-scene arm-C job (`SANITY200` unset) with a
      generous margin over that estimate — plan is `--time` in the 20–24h range on
      `gpuhe.24h` (max 48h, no need for `gpuhe.bulk`/`--exclusive` given this wasn't a
      systemic issue) to absorb node-luck variance, and just retry if unlucky again.

      **Relaunched for real (job 6981912, `d4rt_full`, `--time 24:00:00`, `SANITY200` unset
      → the 490-scene pool, `EXP_TAG=_learned_officialgt_500`, arm-C recipe, node
      `eu-g6-069`) — TIMED OUT AGAIN, worse than attempt #1.** Only reached **epoch 80/1000**
      in the full 24h (vs. epoch 150/1000 in the first 12h attempt) — `metrics.jsonl`'s last
      write (epoch 50, val mIoU 0.246/AP50 0.158) has an mtime ~14h into the 24h job, so
      staging+caching+50 epochs alone ate more than half the budget. `sacct` MaxRSS ≈ 241 GiB
      — again under the 350 GB request, so still not our own cgroup limit.
      **This changes the diagnosis from "bad node luck" to "this job's footprint is
      structurally contention-prone": two different large-footprint attempts (job 6442237 on
      `eu-g6-014`, job 6981912 on `eu-g6-069`) have now both stalled hard on different nodes,
      while the one small-footprint sanity run (190 scenes, ~240 GB less RAM pressure) sailed
      through cleanly on a third node (`eu-g6-046`) in 2h39m.** Two data points isn't proof,
      but it's no longer a single unlucky draw either — the common factor across both bad
      runs is the ~250 GB actual / 350 GB requested memory footprint, which plausibly makes
      this job much more sensitive to *any* neighboring tenant's memory/bandwidth/disk
      pressure than the 190-scene job ever was. **Not relaunching blind a third time.**

      **Root cause refined from "node luck" to a footprint/NUMA effect (2026-07-15) — user
      asked the right question ("is it actually the dataset?"), which prompted re-checking
      the per-bundle timing log by scene-ID range instead of assuming.** Split job 6944946's
      per-bundle timings into old (0000–0199) vs new (0200–0499) scenes by bundle pass:
      bundle 0 both ranges fast (mean ~2s); bundle 1 old scenes STILL fast (mean 1.8s) while
      new scenes (processed later in that same pass) already show p90 21s/max 82s; bundle 2
      old scenes (now badly late in the job) degrade to p90 30s/max 61s. **Same scene IDs
      are fast early / slow late — the slowdown tracks how much has already accumulated in
      the host RAM cache at that point in the job, not scene content.** `scontrol show node`
      on both bad-run nodes reports `Sockets=16, CoresPerSocket=8` for 384100 MB RAM — an
      unusually fragmented ~24 GB-per-NUMA-domain topology. A resident cache that grows past
      a few domains' worth (our ~250–310 GB run does; the working 190-scene run's ~580
      bundles/~122 GB likely doesn't) increasingly touches memory physically remote from the
      compute cores — a deterministic effect of OUR footprint vs. this hardware, independent
      of node identity or other tenants (explains recurring on two different nodes with the
      same accumulation-shaped pattern). Practical corollary: splitting the *tar file* into
      two ~250-scene halves would NOT by itself fix a genuine single N≈490 run —
      `train_multiscene.py` caches every train scene's bundles simultaneously regardless of
      how many tar files they shipped in; only two separate ~245-scene runs would reduce
      footprint, at the cost of not answering the original N≈490-in-one-model question.

      **Relaunched (job 7206201, `d4rt_full`, `--time 8:00:00`,
      `EXP_TAG=_learned_officialgt_500_b1`, `EXTRA_ARGS="--query_mode learned
      --num_learned_queries 64 --bundles_per_scene 1"`)** — keeps the single unified
      490-scene run (still answers the original scaling question) but drops
      `--bundles_per_scene` 3→1, cutting the resident cache to ~500 bundles (~105 GB),
      safely inside the size that ran clean in the 190-scene sanity check (~580
      bundles/~122 GB). Trade-off: loses the 2 extra randomly-sampled augmentation bundles
      per scene that the arm-C recipe used at N=190 — a real recipe deviation, flag it if
      this run's numbers are compared directly to the 0.367/0.199 baseline. Job 7206201 was
      accidentally cancelled by the user before it could run; resubmitted identically as
      **job 7219652**. Pending as of 2026-07-15.
- [X] **Dataset extension to 500 scenes (DONE 2026-07-09; pack job 6423316 shipping the tar).**
      Scenes 0200–0499 had no SAM3-era subset frames, so new tooling streams each scene's
      `.sens` and extracts only the stride-5 subset jpgs with early abort (~10% of each file
      transferred, no .sens on disk): `scripts/extract_sens_subset.py`
      (+ `tests/test_extract_sens_subset.py`), then the usual `download_2d_gt.py` zips+convert
      (`slurm/extend_dataset_500.sh`, jobs 6291743/6291746). 291/300 converted first pass;
      the 9 failures (scene0240/0243/0269/0292/0354/0366/0438/0456/0483) all had a **640×480
      color camera** — their GT projections are 640×480 too (RGB↔GT consistent, loader resizes
      to 518 anyway), so the extractor now allows that resolution and the scenes were healed.
      **QA gates @500: PASS — 500 scenes, 7379 instances, 0 cross-class duplicates**; low-res
      strip (scene0240_00) eyeballed clean. New tar `scannet_official_gt_500.tar.zst` (the
      200-scene `scannet_official_gt_full.tar.zst` is kept for reproducibility of the current
      baseline). Train SLURM scripts now default `DATA_TAR` to the 500-scene tar and request
      `--tmp=24000` (bigger unpacked tree). NOTE: scene lists in the train scripts still
      enumerate 0000–0199; extending train sets to the new scenes is a separate decision.

- [X] **Migrate GT: SAM3 → official ScanNet 2D instance annotations** (DONE 2026-07-08,
      Phases 0–3 + tooling; `docs/old/OFFICIAL_GT_MIGRATION_PLAN.md`). Motivation: 2026-07-07 audit —
      ~3.4 cross-class duplicate instances/scene, 15.9% multi-class foreground px in the SAM3 GT.
      Built `scannet_official_gt_full.tar.zst` (200 scenes, 2950 instances, **0 cross-class
      duplicates**, QA gates + visual strips pass), same layout → zero loader changes; train
      SLURM scripts now stage it by default (`DATA_TAR` env var to switch back). Overfit smoke
      test on converted GT passes. New: `scripts/build_official_masks.py`,
      `scripts/download_2d_gt.py`, `scripts/gen_official_gt_report.py`,
      `scripts/qa_official_gt_strips.py`, `scripts/eval_checkpoint.py` (+ tests).
- [X] **Official-GT training validation (migration Phase 4 — DONE 2026-07-08).**
      (a) Arm-C rerun on official GT (job 6234787,
      `d4rt_full_inst_learned_officialgt_20260708_124452`): best val mIoU **0.367** @ep500,
      honest val[grid] AP50 **0.199** @ep450 → **the new quotable baseline**. Run hit the 4 h
      walltime at ep850/1000 but both metrics had peaked ~ep450–500 and best checkpoints were
      saved — bests valid; official-GT epochs are ~20% slower, budget >4 h next time.
      (b) Cross-eval of the old SAM3-trained checkpoints against official-GT val (job 6234828,
      `scripts/eval_checkpoint.py`): best_ap50 honest AP50 0.228 → **0.117**, best mIoU
      0.371 → **0.285** (official val: 13.3 GT inst/scene). ~Half the honest-AP50 headline was
      SAM3-GT-specific; retraining on clean GT recovers +70% AP50 on the same ruler. Full
      table in MILESTONES §Dataset status. Migration complete — plan archived to `docs/old/`.

- [X] **N=100+ scaling point** (DONE 2026-06-22). Arm-A instance-GT curve now runs N ∈ {10, 25,
      50, 100, 200}, wide val (scene0080–0089). **Result: plateau** — val mIoU flattens at
      ~0.21–0.23 past N=50, honest val[grid] AP50 sits ~0.10 (below its N=50 peak of 0.125). More
      scenes is no longer the lever; train−val gap shrank to ~0.05 (no longer overfitting →
      capacity/resolution ceiling). Runs: `d4rt_m2_scale100_inst`, `d4rt_full_inst`. See MILESTONES.
- [X] **Confirm the learned-query win (arm C) at large N** (DONE 2026-06-22 — decisive win).
      At N=200: val mIoU **0.371** (best @ep600, gap 0.086) / 0.326 final, honest AP50 **0.228** —
      vs the plateaued point baseline (0.216 / 0.105): +0.15 mIoU, >2× AP50. Learned queries keep
      scaling (0.259→0.371 from N=50→200) and their N=50 overfitting resolved (gap 0.49→0.086).
      **The ceiling was the head, not the data.** Run: `d4rt_full_inst_learned_20260622_183203`.
      → Arm C learned is now the default base for all further experiments (below).
- [X] **Fix Phase-2 `--train_grid_queries` loss balance** (DONE 2026-07-07 — fix works, arm B
      closed as a loss to arm C). `--no_object_norm matched` fixed the collapse: N=50 rerun
      (job 5647527, `_gridq_fix`) reaches train[grid] mIoU 0.458 (was 0.055) and best val[grid]
      AP50 **0.161** ≥ the 0.125 bar. Scaled to N=190 (job 5658375,
      `d4rt_full_inst_gridq_fix_20260703_184456`): val[grid] mIoU 0.372 @ep1000 (matches arm C)
      but AP50 peaks at only 0.185 and is unstable (0.071 @ep1000) vs arm C's 0.228 →
      **arm C stays the base**. See MILESTONES.
- [X] **Fix Phase-3 hybrid (arm D) NaN** (DONE 2026-07-07 — NaN fixed, no win, arm D closed).
      N=50 rerun (job 5647528, `_hybrid_fix`, lr_scale 0.1, grad_clip 0.5) survived all 1000
      epochs with zero non-finite matcher warnings, but best val mIoU 0.247 / AP50 0.146 only
      ties arm C N=50 (0.259/0.146) and then overfits (val→0.177 while train[grid]→0.75).
      Per the decision rule (scale only on a win): **no N=190 run**. See MILESTONES.
- [X] **Train the MaskDINO pixel decoder** (`--mask_upsample 2`) — DONE 2026-06-30 (SLURM job
      5275027, run `d4rt_full_inst_learned_us2_20260630_161537`; arm-C base: learned, 64 queries,
      instance-level, N=190). **Result: a wash.** Best honest val[grid] AP50 **0.236** (@ep500,
      marginally above the us=1 baseline's 0.228), best val mIoU **0.355** (@ep250, below the
      baseline's 0.371); final @ep1000: 0.200 / 0.311; gap at best 0.098 (vs 0.086). Doubling the
      mask resolution to 74×74 does not move the numbers → **resolution is NOT the current
      bottleneck**; `--mask_upsample 4` is deprioritized (per the decision rule: only on a win).
      The window/door/picture confusion is therefore likely semantic, not resolution-limited —
      points at the score-threshold / no-object levers and arm-D instead.

## Open — positioning & new research directions (from the 2026-07-08 literature survey)

Context: an arXiv harvest (see `docs/RELATED_WORK.md`) shows "decoder on frozen VGGT" is now
a crowded genre — **SegVGGT** (verified: object queries on VGGT, ScanNetv2/ScanNet200) is the
closest published competitor. The architecture alone is no longer the contribution; the
query-strategy study (arms A–D) is. Direction: don't pivot, reposition.

- [ ] **Read the direct competitors** (`docs/RELATED_WORK.md` table): SegVGGT line-by-line
      first (esp. its eval protocol — 3D point-cloud masks vs our per-view 2D patch-grid masks
      are NOT comparable numbers; note the difference explicitly), then EPS3D, FAST3DIS,
      PanSt3R. Check whether any already claims (a) a query-init ablation or (b) 3D-anchored
      queries — both would change the plan below. Record findings in RELATED_WORK.md.
- [ ] **Arm E — 3D-anchored queries (the main new experiment). CODE DONE + TESTED
      2026-07-15; GPU runs pending.** Seed queries from VGGT's own predicted pointmap
      geometry instead of image-space (u,v). Implemented as `--query_mode anchor3d`
      (+ `--num_anchors 64 --anchor_knn 8 --anchor_jitter`):
      `models/anchor_queries.py` — each patch token gets a 3D position (confidence-weighted
      mean of its 14×14 pixels' point-head output), anchors = deterministic farthest-point
      sampling over the confidence-filtered, per-scene-normalized (zero-mean/unit-RMS) token
      positions, content = mean feature of each anchor's kNN tokens in 3D (inherently
      multi-view); query = proj(Fourier(xyz)) + proj(LayerNorm(pooled feats)), NO view
      embedding — one query per 3D location shared across views. All anchor building is
      frozen preprocessing in `build_bundle` (the point head runs once per bundle on the
      agg_list already in hand; only the small anchor dict is cached, not the pointmap).
      Matcher treats it like learned mode (coord_weight 0, mask-only cost); eval is
      GT-free by construction (prompted == unprompted, like arm C). Tooling wired:
      visualize_masks/demo_gradio rebuild anchors from the point head (deterministic),
      eval_checkpoint.py inherits support via prepare_scene_bundles, eval_grid_ablation
      rejects anchor3d (no coordinates). Tests: `tests/test_anchor_queries.py` (Fourier-3D
      backward compat, token positions, FPS, normalization invariance, kNN pooling, head
      end-to-end + config round-trip, train-loop wiring) — full suite green.
      Rationale: (i) fills the one query-strategy cell no competitor has published
      (RELATED_WORK gap #1 — verify vs SegVGGT/EPS3D/FAST3DIS before claiming);
      (ii) 3D-spread anchors are a natural one-query-per-object dedup mechanism → directly
      attacks the over-prediction failure (338 kept vs 144 GT).
      (a) Single-scene overfit smoke — PASSED 2026-07-15 (scene0000_00, 300 ep, 4 frames,
      64 anchors, official GT, local 4090): loss 2.86→0.28, prompted == unprompted
      mIoU 0.925 / AP50 1.000 / class_acc 1.000, and the honest GT-free selection kept
      EXACTLY 10 predictions for 10 GT instances out of 64 anchors — the intended
      one-query-per-object dedup behavior, visible already in overfit.
      (b) N=50 runs SUBMITTED 2026-07-15: job 7212666 (arm E,
      `EXTRA_ARGS="--query_mode anchor3d --num_anchors 64 --anchor_knn 8
      --anchor_jitter 0.02"`, EXP_TAG=_anchor3d) and job 7212769 (arm-C N=50
      official-GT CONTROL, learned/64 — needed because the existing N=50 arm-C numbers
      0.259/0.146 are SAM3-GT), both `slurm/train_scale50.sh` with INSTANCE_LEVEL=1,
      scenes 0000–0049, val 0080–0089. Compare best val mIoU + honest val AP50.
      (c) Scale to N=190 only on a win vs 0.367/0.199 (arm protocol). Pre-registered
      ablations: K ∈ {32, 64, 128}; positional-only queries (drop pooled feats); pair
      with the no-object-weight sweep (same duplicate-FP target).
- [ ] **Cross-view consistency metric** in `train/eval_metrics.py` (+ test). Our decoder is
      intrinsically consistent by construction (`pred_masks [B,N,S,h,w]`, one query = one
      instance in all views) vs the fuse-2D-masks paradigm (PanSt3R/MV3DIS) — quantify it
      (e.g. per matched instance, cross-view mask-identity agreement / ID-switch rate) so the
      claim is a number, not an assertion. CPU-runnable; no retraining needed (eval-only on
      existing checkpoints).
- [ ] **Which-layer ablation (mining backbone internals; cheap).** We hook only
      `aggregated_tokens_list[-1]`; sweep the hook layer (e.g. every 4th of the 24 blocks) on
      the arm-C recipe at N=50. Nearly free given feature caching; VGGT-Det ("mining VGGT
      internal priors") shows the appetite. Good thesis-analysis chapter even if [-1] wins.
- Deprioritized: backbone-agnostic decoder (VGGT/CUT3R/Pi3) — real gap but out of thesis
  scope; Lite3R already owns the framing.
- [ ] **Align training protocol with best practices (splits + frame sampling).** Audited
      2026-07-13 against SegVGGT's stated protocol: (a) our `--train_scenes`/`--val_scenes`
      are contiguous scene-ID ranges (first N train, fixed 0080–0089 val) chosen for a
      data-scaling curve, not the official ScanNetv2 1201/312/100 split (SegVGGT spans all
      1613 scenes; we only have 500 downloaded, so an exact match isn't possible without
      more data, but an official-split-intersected-with-our-500 subset would be closer than
      the current first-N convention). (b) SegVGGT randomly samples 2–24 frames per scene
      *every training iteration*; we cache `--bundles_per_scene` (3) fixed random frame-sets
      per scene once up front and reuse them for up to 1000 epochs, with `num_frames` fixed
      per run (8) rather than varied. This is a deliberate tradeoff for backbone-feature
      caching (head-only training in minutes, not hours) but risks the head memorizing the
      cached view combinations rather than learning frame-count/view-set robustness. Needs a
      decision before implementing: intersect official split with our 500 scenes vs. keep the
      scaling-curve convention; and whether to raise `--bundles_per_scene` / randomize
      `num_frames` per bundle vs. accept the caching tradeoff as-is.

## Open — data-gated ablations (Phase 6; meaningful now with 200 scenes)

- [ ] No-object-weight sweep (0.05 / 0.1 / 0.4) — tests whether 0.1 drives the under-confidence.
- [ ] Augmentation ablation: `bundles_per_scene` 1 vs 4, `query_jitter` on/off, `color_jitter` on/off.
- [X] Grid-density vs unprompted recall (DONE 2026-07-07, job 6111639 — **negative: density
      is not the lever, learned-vs-grid gap confirmed architectural**). Eval-only sweep
      (`scripts/eval_grid_ablation.py` + `slurm/eval_grid_ablation.sh`, no retraining) of
      `--grid_size` 2/4/6/8/10/12 on the stored val bundles; grid-6 rows reproduce the
      training-time `val_grid_AP50` of the selected epochs exactly (0.134 / 0.185).
      Val AP50 by density —
      arm A `d4rt_full_inst` best_ap50: 0.023 / 0.124 / 0.134 / **0.138** / 0.116 / 0.109
      (flat 6→8, drops beyond); arm B `gridq_fix` best_ap50: 0.018 / 0.063 / **0.185** /
      0.100 / 0.134 / 0.067 (sharp peak exactly at its 6×6 *training* density). Kept
      foreground predictions explode with density (arm A: 58 @6 → 236 @12 vs 14.4 GT/scene)
      → denser grids die by duplicate FPs (no NMS), as predicted. Unprompted mIoU rises
      monotonically with density (0.297→0.336) — the known "unprompted mIoU is optimistic"
      artifact; judge on AP50. Best grid number over all densities/checkpoints (0.185) stays
      well below arm C's 0.228. Results: `<run_dir>/grid_ablation_<ckpt>.json`.
- [X] Score-threshold sweep at viz time (DONE 2026-07-03 — **negative, keep 0.5**). On the
      arm-C best checkpoint (`d4rt_full_inst_learned_20260622_183203`), thr 0.3 surfaces 76
      extra instances over the 10 val scenes of which only 2 have IoU≥0.5 with any GT (1 with
      the right class) — the under-confidence finding was a *point-prompt* phenomenon and does
      not transfer to learned queries. The model already over-predicts at 0.5 (338 kept vs 144
      GT) → the lever is duplicate suppression / no-object weight, not the threshold.
      Renders: `<run>/visualizations_thr03/` (val scenes, vs `visualizations/` at 0.5).
- [ ] Longer-term: partial backbone unfreezing once the train−val gap vs N says data supports it.

## Done (high level — detail in docs/MILESTONES.md)

- [X] M1: prototype, single-scene overfit, 4-scene training (train mIoU 0.967), unseen-scene eval.
- [X] M2: no-object loss, unprompted grid eval, regularization, best-checkpoint + early stop.
- [X] M3 Phase 0: metrics.jsonl, uint8/light checkpoints, noise-robust early stop,
      checkpoint_best_ap50, --schedule_epochs, viz polish, fixed SLURM protocol.
- [X] M3 Phase 4: per-instance loader (`instance_level`) + tests; instance-GT scaling curve
      (monotonic, wide val) N∈{10,25,50}.
- [X] M3 Phases 2/3 CODE + first runs: `--train_grid_queries`, `--query_mode {point,learned,hybrid}`.
      Arms A/B/C/D run at N=50 (C learned = best; B/D need fixes above).
- [X] M3 Phase 5: `models/mask_upsampler.py` + `--mask_upsample`; us=2 trained at N=190 →
      neutral vs us=1 baseline (see MILESTONES) — resolution is not the bottleneck.
- [X] Standardized 2D/3D visualization (2026-07-02): `train/postprocess.py::select_instances`
      is the single instance-selection rule shared by `visualize_masks.py` and the Gradio 3D
      viewer (honest, no-GT, winner-takes-all; + oracle GT-matched panel as diagnostic).
- [X] Per-instance SAM3 dataset: 200 scenes (scene0000–0199) in one `scannet_instance_dataset_full.tar.zst`;
      `stage_dataset.sh` stages it to node-local SSD (`--tmp=16000`).
</content>
