# TODO

Open work only. Everything closed up to 2026-07-28 is in
`docs/old/todo_archive_20260728.md`; the reasoning behind each closed item is in
`docs/old/MILESTONES.md` (the retired head) and `docs/RESULTS.md` (MaskDINO).

**Goal restated (2026-07-30).** A 3D-consistent multi-view instance segmentation model on a
**strictly frozen** VGGT backbone, written up as a controlled decoder study for a top-tier venue
(target: **3DV** — check the CFP dates now; the dataset extension in 1c has the longest
wall-clock tail). The framing is settled in `docs/RELATED_WORK.md`: shared queries and 3D
anchors are published mechanisms (SegVGGT, FAST3DIS) — the paper is the *controlled study* (one
frozen backbone, one dataset, one protocol, ingredients varied one at a time), and the frozen
backbone is the differentiator (every direct competitor LoRA-adapts). The §7.4.1 ablation triple
(cross-frame attention 0.183 / bundle features 0.147 bundle AP50, consistency-vs-per-frame price
quantified) is the core table of that study.

## 1. Paper-blocking: make the numbers placeable (protocol work)

Nothing we report is currently comparable to any published number (docs/RESULTS.md §1.2).
In effort order — each step also de-risks the next:

- [x] **1a + 1b. Full-resolution eval + ScanNet oracle — DONE, question CLOSED 2026-07-30**
      (docs/MASKDINO.md §6.5 implementation, **§7.7 the verdict**). The 37×37 grid's GT-only
      ceiling on ScanNet is **0.956 AP50** on the full-res ruler (COCO: 44.7 — different regime
      entirely); the model sits at ~0.69; the bar re-run reproduces (0.690 grid / 0.673 full)
      and `--mask_upsample 2` stays neutral on the honest ruler (0.680 vs 0.673 full AP50).
      **Recognition binds, not resolution — quote §7.7, stop spending here.** The
      `--mask_upsample 4` run OOM'd (no grad accumulation in the ScanNet trainer); per §7.7 not
      worth fixing (~0.004 AP50 of ceiling over ×2).
- [x] **1c. Extend the dataset to the full official ScanNet v2 train split (1201 scenes) —
      DONE 2026-08-02: tars built AND first runs trained (docs/RESULTS.md §6,
      docs/MASKDINO.md §7.8).** Simultaneously the protocol fix
      (train/eval on the official 1201/312 split like every competitor) and the **biggest
      performance lever left**: +0.26 AP50 came from 50→490 scenes, the curve is still rising,
      and views-per-scene saturated at 2 (§7.4.1). `data/splits/scannetv2_train.txt` (1201 scan
      ids — includes `_01`/`_02`/... rescans, NOT a contiguous `scene{i:04d}_00` range, fetched
      the same way as the existing `scannetv2_val.txt`) drives the build.
      `extract_sens_subset.py` and `download_2d_gt.py` gained a `--scene_list FILE` option
      (start/end become 0-based line indices into FILE; default unchanged) since the old
      range-based selection can't express this split.

      **Attempt 1 failed on the scratch INODE quota** (1.0 M soft / 1.5 M hard *files*): the
      build tree is ~1046 files/scene, so 1201 scenes is ~1.26 M files. Jobs
      9079912/14/15/17 died with `OSError(122, 'Disk quota exceeded')` at 1090/1201 scenes and
      left the account at 1 499 966/1 500 000 files — unable to write anything on scratch.
      The build is now **node-local** (`$TMPDIR`, `--tmp=120000`); only one compressed chunk
      tar per range touches scratch, which costs 1 inode. Full rationale + the resumability
      contract in **docs/DATASET.md §5.1** — read it before changing any build script.

      Attempt 2 (2026-07-30) **SUCCEEDED**, and *preserved* attempt 1's work rather than
      re-downloading — all 1201 scenes already had their `.sens` subsets (~90 node-hours of
      streaming) and 1090 were fully converted, so only 111 scenes' GT zips remained:
      `snapshot_build_1201.sh` (job 9127341: tree → chunk tar, verify every regular file, then
      delete the tree — reclaimed 1.26 M inodes, scratch 1 499 970 → 243 059 files) →
      `extend_dataset_1201.sh 0 1200` (job 9127345: restored the chunk tar, `ok=111 skip=1090
      fail=0`, 42 min) → `pack_official_gt_1201.sh` (job 9161678, submitted by the extend job
      via `CHAIN_PACK=1`). Result: **`scannet_official_gt_1201.tar.zst`, 29 GB on work — 1201
      scenes, 17 638 instances, 0 cross-class duplicates, min label purity 1.0, no
      missing/failed scenes**; archive entry count verified against source (1 328 343).
      `OFFICIAL_GT_README_1201.md` + `qa_strips_1201/` alongside it.
      **The val ruler now exists too — BUILT 2026-08-01.** The 1201 tar is train-split only, and
      the convention val scenes 0080–0089 split 6 train / 4 official-val, so official-split
      training had nothing honest to be scored on. Same pipeline, same QA gates, pointed at
      `data/splits/scannetv2_val.txt` via new sibling scripts (`extend_dataset_val312.sh` job
      9325618, 1 h 17, `ok=312 skip=0 fail=0` in both stages → `pack_official_gt_val312.sh` job
      9328388, 2 min, chained by `CHAIN_PACK=1`). Result: **`scannet_official_gt_val312.tar.zst`,
      7.4 GB on work — 312 scenes, 4630 instances, 0 cross-class duplicates, max cross-class IoU
      0.0, min label purity 1.0, no missing/failed scenes**; archive entry count verified against
      source (347 439 png/jpg). `OFFICIAL_GT_README_val312.md` + `qa_strips_val312/` alongside it.
      The two lists are disjoint (0 shared scan ids), so 1201 + 312 is the full official protocol.
      **First runs DONE 2026-08-01/02** (docs/RESULTS.md §6, docs/MASKDINO.md §7.8; plumbing:
      multi-tar `DATA_TAR` + `TRAIN_LIST`/`VAL_LIST` in the slurm scripts, tested by
      `tests/test_train_maskdino_sh_lists.sh`): single-frame job 9329716 **0.624 mIoU / 0.662
      AP50** (full-res 0.611 / 0.651); multi-frame job 9386666 per-bundle **0.529 / 0.525**,
      per-frame 0.623 / 0.650. 12 CPU × 14 GB + `--tmp=90000` + 12 epochs was the right sizing
      (8h16 / 5h42). Leftover cleanup: the redundant chunk tars in
      `/cluster/scratch/niacobone/scannet_1201_chunks/` (29 GB) and
      `/cluster/scratch/niacobone/scannet_val312_chunks/` (7.5 GB) can be deleted
      (blocks only, 1 inode each).
- [~] **1d. 3D benchmark eval — PIPELINE BUILT + VERIFIED 2026-08-01** (docs/MASKDINO.md §9,
      the full protocol; RESULTS.md §5 is where its numbers go). Per-view masks unprojected
      with VGGT's *own* predicted depth + cameras (no GT geometry at inference), eval-only
      Sim(3) registration (Umeyama on camera centers + similarity ICP — the FAST3DIS
      convention), per-vertex votes + majority per superpoint (the superpoint majority is the
      SegVGGT convention; the radius is ours), scored by the
      **vendored official evaluator** (`train/benchmark3d.py`; real val scenes' GT fed back as
      predictions scores exactly 1.000). Data on work: `scannet_3d_gt_val312.tar.zst` +
      `scannet_frames25k_val312.tar.zst` (whole-scan frames + poses; the stride-5 subsets
      cover only raw frames 0–495 and would cap recall). Run:
      `sbatch --export=ALL,CHECKPOINT=... slurm/eval_3d_maskdino.sh`.
      Diagnostic val-312 runs DONE 2026-08-01 (jobs 9327269/9327271, §9.5 + RESULTS.md §5):
      **AP 0.016 / AP50 0.052 / AP25 0.238** at best knobs — FAST3DIS's order of magnitude
      (0.038/0.096/0.316). Verdict: geometry binds (median Sim(3) RMS
      0.14 m ≈ vote radius; AP25 ≈ 5×AP50), not recognition; coverage caps recall.
      **DONE 2026-08-03 — the reportable number exists** (jobs 9503137 / 9503139, docs/MASKDINO.md
      §9.6, RESULTS.md §5): leak-free 1201-trained multi-frame checkpoint scores **AP 0.023 /
      AP50 0.067 / AP25 0.268** (0.029 / 0.083 / 0.305 with tuned lifting knobs — tuned on the
      leaky diagnostic, so the plain row is the headline). **FAST3DIS's ballpark
      (0.038 / 0.096 / 0.316) on a strictly frozen backbone**, alongside IGGT
      (0.028 / 0.112 / 0.287) — *both of those class-agnostic, ours class-aware, see 1e*.
      SegVGGT (0.504 / 0.717 / 0.870) is far above but **in a different
      protocol** — posed transfer, GT poses + sensor depth, no geometry error (established
      2026-08-04, docs/MASKDINO.md §9.9) — so it is not a like-for-like gap; see 5e.
      Two findings: the leak-free checkpoint **beats** the leaked
      diagnostic 1.6× (data scale > leakage — the 3D ruler reproducing the 2D data-limited
      conclusion), and **the lifting step is the binding constraint, not the decoder** → the new
      workstream 5 below. Fixed while reporting: eval3d output files now name their non-default
      knobs (two knob settings used to overwrite one file; job 9503137's JSON was lost that way,
      numbers recovered from its log). Guarded by
      `tests/test_maskdino_eval3d.py::test_out_path_names_the_knobs`.
- [x] **1e. Class-agnostic column — CLOSED 2026-08-06, and the `--anchor_3d` row LEADS**
      (docs/MASKDINO.md §9.11, RESULTS.md §5).
      Re-reading FAST3DIS turned up that its and IGGT's ScanNet rows are **class-agnostic** while
      ours and SegVGGT's are class-aware. The evaluator now computes both
      (`train/benchmark3d.py::collapse_gt_to_class_agnostic`, `results_class_agnostic` in the
      eval JSON, `tests/…::test_evaluator_class_agnostic`). Jobs 9861563/9861564 measured the
      §9.6 rows: **0.013 / 0.050 / 0.320** (defaults) and **0.017 / 0.060 / 0.334** (tuned) —
      collapsing labels *lowers* AP/AP50 and *raises* AP25, so like-for-like we **lead the
      published cluster on AP25 and trail ~1.6–2.2× on AP50/AP**, and "in FAST3DIS's ballpark on
      AP50" is struck everywhere. **Job 9866391 (the `--anchor_3d` checkpoint) LANDED 2026-08-06
      and the claim SURVIVED: 0.038 / 0.112 / 0.360 class-aware → 0.042 / 0.138 / 0.504
      class-agnostic — ahead of FAST3DIS (0.038 / 0.096 / 0.316) and IGGT (0.028 / 0.112 / 0.287)
      on all three, frozen backbone, untuned, ~17 views/scene vs their 50.** This is the strongest
      publishable row in the project. The prediction written here ("expect that claim to weaken")
      was **wrong**: the sign of the class-collapse is checkpoint-dependent — it punishes a head
      whose class-aware mean leans on rare classes and rewards one whose instances are fewer and
      view-consistent. Outward-facing claim is now licensed, with the single-run caveat.

## 2. Complete the multi-frame study (the contribution)

- [x] **2a. Best data recipe × multi-frame — DONE 2026-07-30** (job 9071415): **new multi-view
      best 0.539 mIoU / 0.515 bundle AP50** (+0.021 over 0.494), per-frame 0.643 / 0.667 (up
      from 0.621 / 0.630). Peak per-frame and per-bundle coincide (epoch 19), so
      `checkpoint_best_ap50.pth` carries the headline. This is the recipe 2d builds on.
- [x] **2b. Bundle-selected checkpoint — DONE 2026-08-01.** `checkpoint_best*` selected on the
      *per-frame* metrics only; the per-bundle peak can fall on a different epoch (§7.4.1). Added
      `checkpoint_best_bundle.pth`, selected on val `bundle_AP50`, saved only for `--multi_frame`
      runs (docs/MASKDINO.md §8.2). No behaviour change for single-frame runs; nothing retrained.
- [x] **2c. Cross-view consistency metric — IMPLEMENTED 2026-08-01** (docs/MASKDINO.md §6.6;
      RELATED_WORK.md gap 2). `train/eval_metrics.py::multiview_consistency_metrics`, reported
      by the `--multi_frame` eval as `bundle_view_consistency` (per matched instance, fraction
      of its visible views explained at IoU ≥ 0.5 by its bundle-matched query) and
      `bundle_id_switch` (fraction of visible views where some *other* query is the best
      match), plus `bundle_num_matched`. Purely additive — no existing key changed.
      **First numbers measured 2026-08-02** (job 9386666, official 1201/312 split,
      docs/MASKDINO.md §7.8): consistency 0.679→0.717 and id_switch 0.607→0.498 over epochs
      6→12, ~14.1 matched/bundle. **CLOSED 2026-08-03 by job 9503176** (docs/MASKDINO.md
      §7.8.1): removing cross-frame attention leaves matched instances unchanged (14.0 vs 14.1)
      and `view_consistency` nearly so (0.692 vs 0.717), but **`id_switch` jumps 0.498 → 0.682**
      and bundle AP50 falls 0.525 → 0.389. The block's job is **identity preservation**, not
      recognition — the mechanism claim the metric was built to support, and a core table for
      the paper.
- [x] **2d. 3D anchors vs 2D DAB boxes — DONE 2026-08-04. Flat in 2D, +67 % AP50 in 3D.**
      (docs/MASKDINO.md §8.3 has the design as built, the three forced deviations from the
      sketch, and the named confound). `--anchor_3d`, default off: the decoder's 2D DAB anchor
      box becomes a 3D `(x, y, z, log r)` per query per **bundle**, gathered from the two-stage
      top-k out of a per-patch-token position cache read off VGGT's frozen point head
      (**+0.146 % cache**, 65.7 kB/bundle measured, no caching-time cost), soft-projected into
      each view over the 37×37 grid — no intrinsics/extrinsics — and refined by Δ(xyz, log r).
      Every loss unchanged; denoising queries stay on the 2D DAB path.
      Framed and budgeted as an **ablation** (FAST3DIS owns the mechanism as a contribution),
      which also closes the archived 3D-anchor loop. Promoted above 5b/5c by §9.10 reading 4.
      **2D RESULT IN (job 9634920 vs control 9386666; `config.json` differs in exactly the one
      key): AP-neutral, identity-positive.** Per-bundle 0.529 / 0.525 → 0.524 / **0.527** and
      consistency 0.717 → 0.723 are flat; per-frame 0.623 / 0.650 → 0.611 / 0.641 is mildly
      negative; and the single systematic move is **`bundle_id_switch` 0.498 → 0.409**
      (−18 % rel., better in **12/12 epochs**, mean −0.084). That **dissociates** identity from
      bundle AP50, which §7.8.1 had found moving together — so they are not one axis, and the
      §9.10 "multi-view completeness *and* identity" residual is not one quantity either: this
      bought the identity half only. Not a default (+15 % training time, −0.009 per-frame AP50).
      **3D RESULT IN (jobs 9670882 / 9670883) — and it overturns the "not a default" reading.**
      Same 312 scenes, same **17.42 frames/scene**, all knobs default, 0 failures:
      unposed **0.023 / 0.067 / 0.268 → 0.038 / 0.112 / 0.360 (+67 % AP50)** and posed
      **0.060 / 0.156 / 0.408 → 0.104 / 0.257 / 0.504 (+65 %)**. The unposed row now **matches
      FAST3DIS on AP (0.038) and beats it on AP50 (0.112 vs 0.096) and AP25 (0.360 vs 0.316)**,
      on a strictly frozen backbone and *untuned*. Signature: 9 % **fewer** kept queries, 16 %
      **more** voted vertices — fewer, cleaner, view-consistent instances is exactly what a
      per-vertex vote rewards, and `id_switch` is the failure mode the vote integrates over.
      **Two lessons.** (i) `bundle_AP50` at S=8 is a **poor proxy for the 3D ruler** — a mechanism
      can be flat on it and worth +67 % in 3D, so score identity mechanisms on the 3D ruler before
      judging them. (ii) This document's own prediction ("bought identity, not completeness, so
      expect little 3D movement") was **wrong and was falsified**: of §9.10 reading 4's two
      residuals it is *identity* the 3D ruler is most sensitive to. Keep the dissociation, discard
      the prediction. **Drift ruled out from the diff, not a re-run**: the only commit between the
      control rows and these is `7c4e890`, which touches no file in `train/eval3d_*` /
      `benchmark3d` / `scannet3d` and adds one `anchor_3d`-guarded branch to
      `scripts/eval_3d_maskdino.py` that is inert for a 2D-box checkpoint; the posed control also
      post-dates the `--transfer_mode` refactor, so it is same-code by construction. Both blocks
      move ~+66 % independently.
      **Recommended for any 3D-benchmark run; still off by default.** Converges with 2e: both
      winners moved `id_switch` and little else.

- [x] **2e. Bundle width: 8 → 16 views per bundle — DONE 2026-08-06, POSITIVE**
      (docs/MASKDINO.md §8.4, RESULTS.md §6). One flag (`--num_frames 16`, job 9668639 vs control
      9386666): per-frame AP50 0.650 → **0.662**, per-bundle AP50 0.525 → **0.552** on the same
      pinned 8-view ruler, `bundle_id_switch` 0.498 → **0.385** (−23 % rel.) with
      `bundle_num_matched` flat at 14.0. **Recognition unchanged, identity improved** — the
      §7.8.1 signature read forwards. Job 9668726 rules out the data confound: `--bundles_per_scene
      1` at S=16 is *frame-matched to the control* with jitter inert and still gets 0.544 /
      0.345, so it is the **width**, not the extra frames. Job 9668652 (20 ep, val at 16) posts
      the best per-frame AP50 on the official split anywhere (**0.669**) and id_switch **0.323**,
      still falling. Costs 2× wall clock and ~230 GB of cache (A100 80 GB).
      **3D ruler run 2026-08-07** (jobs 9901143/63/64/65, docs/MASKDINO.md §8.4 reading 4): width
      pays on both bridges — unposed AP50 0.067 → **0.098** (+46 %), posed 0.156 → **0.216**
      (+38 %) — and the 20-epoch run posts the best posed row anywhere (**0.088 / 0.260 / 0.572**).
      But `--anchor_3d` still wins the unposed column (0.112 vs 0.098) at half the wall clock on a
      4090. Second independent confirmation that this ruler responds to multi-view **identity**.
      Open: does width keep paying past 16? Not answered.
      Why it was tried: §9.10 reading 4 (multi-view completeness/identity binds the posed
      column) plus a standing train/test mismatch — the 3D ruler runs the head at S ≈ 17.4 and
      it was trained at S = 8. New flag **`--eval_num_frames`** (default unset = unchanged) pins
      the VAL bundle width so `bundle_*` stays on the 8-view ruler the 0.525 baseline was
      measured on; without it, widening training silently changes what the metric measures.

- [x] **2f. Combine `--anchor_3d` with `--num_frames 16` — CLOSED 2026-08-12, NEGATIVE on every
      ruler.** (Trained job 9979913; scored in 3D by jobs 10477399 / 10477400.) (docs/MASKDINO.md §8.5, RESULTS.md §6.) The two
      winners of 2d and 2e both act on `bundle_id_switch` and both pay on the 3D ruler, and they
      had never been run together. Config = the 2e recipe plus `--anchor_3d`, one flag against
      9668639; A100 80 GB, 11 h 36.
      **They do not compose on the 2D ruler**: per-bundle AP50 0.552 → **0.536** and per-frame
      0.662 → **0.646** (both −0.016, ~1.8× the 0.009 seed spread) while `id_switch` improves only
      0.385 → **0.375**, *inside* the control arm's own seed spread on that metric (0.027). So the
      anchor's −0.089 `id_switch` at S=8 does not survive being stacked on the width's −0.113 —
      one axis, near its floor — and the run costs measurable AP for it.
      **The 3D ruler was the arbiter and it agrees** (docs/MASKDINO.md §8.5 reading 3, RESULTS.md
      §5.3; both bridges, default knobs, 312 scenes, 17.42 frames, 0 failures). Against
      `--anchor_3d` alone: a dead heat on the unposed class-agnostic column
      (0.041 / 0.139 / 0.504 vs 0.042 / 0.138 / 0.504, ±0.001 — an order of magnitude inside the
      0.009 seed spread) and a **loss** on the other three (posed agnostic 0.109 → 0.098 AP;
      unposed class-aware 0.038 → 0.032; posed class-aware 0.104 → 0.082). Signature:
      **over-pruning** — 2f keeps the fewest queries of any checkpoint (82.2) yet votes *fewer*
      vertices than `--anchor_3d` (0.155 vs 0.177). Both flags shrink the query set; together they
      shrink it past the point the vote can pay for.
      **Consequences.** `--anchor_3d` alone stays the checkpoint to quote (RESULTS.md §8.2) — no
      headline moves. And the §8.3 caution reads more precisely now: `bundle_AP50` mispredicted the
      3D outcome there and predicted it correctly here, so it is unreliable in *both* directions,
      not systematically inverted. Knobs were deliberately NOT re-swept: a tuned row cannot rescue
      an untuned tie, and untuned-on-both-sides is what makes the comparison single-variable.
      **Two earlier attempts were lost to a broken venv, not to the code** (jobs 9901119, 9973805):
      scratch's 15-day purge had eaten torch's `.py` sources while leaving the `.pyc` — see the
      warning in `CLAUDE.md`. Env rebuilt 2026-08-07, all 13 CPU tests green, resubmitted.
      **Still worth fixing:** the optimizer is constructed *after* the ~4.5 h feature-caching pass
      (`scripts/train_maskdino.py:313`), so any error there costs the whole cache — 9901119 died
      exactly that way. Building it first would turn this class of failure into a one-minute one.

- [x] **2g. Seed variance — DONE 2026-08-07** (jobs 9901124 / 9901125, RESULTS.md §6.1,
      docs/MASKDINO.md §8.3). Both arms of the 2d comparison re-trained with `--seed 1`.
      **Per-bundle AP50 spread ≈ 0.009 in both arms**, which is the yardstick every Δ in the docs
      must now be read against: the big ablations (0.183 / 0.147 / 0.027) stand at 3–20× it, and
      the 3D-anchor per-bundle delta (+0.002) is *inside* it — so "AP-neutral, identity-positive"
      became a measured claim. `id_switch` improves in both seeds (−0.089, −0.064). This retires
      the "single run vs single control" objection for the headline comparison.

## 3. Resolution stream — CLOSED, archived

Closed 2026-07-30; the verdict (recognition binds, not mask resolution) is
`docs/MASKDINO.md` §7.7. Item history: `docs/old/todo_archive_20260827.md`.

## 4. Watching

**C-long (11632049) LANDED 2026-08-25 AND FAILED** — an unstable optimisation, not a slow one; the
cause is the learning rate, the same failure §11.3 isolated on arm D (docs/MULTIDATASET.md §10.5).
**Re-run submitted 2026-08-26: C-long′ (11831105) at lr 5e-5, whose control is A-long′ (11830142),
already queued for arm D. A-long′ ⇄ C-long′ is step-matched, same-LR and one-variable.**

**Also in flight since 2026-08-26 — the competitor-matched programme (6k + 6l):**

| job | what | lands as |
|---|---|---|
| 11839134 → 11839151 | **arm I** — ScanNet++ + Infinigen + RE10K@1500, **no ScanNet** (IGGT's mixture minus ASE) | MULTIDATASET §12 |
| 11839135 → 11839152 | **arm I-gt** — ScanNet++ + Infinigen, no ScanNet, no SAM2 supervision | 〃 |
| 11839821 (array) → 11840376 | the **dense** ScanNet val-312 frame export (stride 20) | DATASET §2.5 |

Read I ⇄ D-long and I-gt ⇄ A-long′ as the {±ScanNet} edges of one 2 × 2; do not read either
against a published number until its matrix lands. Everything else in the data-scaling set has landed —
docs/MULTIDATASET.md §10.3/§10.4, docs/RESULTS.md §6.4/§7.5/§8.2. Everything else once listed here
has landed too:

- (todo 2d fully landed — jobs 9634920 / 9670882 / 9670883. A regression re-run of the control
  under current code was submitted as 9848637 and **cancelled**; it is not needed, because the
  drift question is closed from the commit diff instead — see 2d.)
- (todo 2f fully closed 2026-08-12 — jobs 9979913 / 10477399 / 10477400 → 2f. Negative.)
- (the 24-cell cross-dataset matrix landed 2026-08-22, jobs 11498511–11498543, 0 failures →
  docs/RESULTS.md §7.5. It is the deliverable of 6f and it reverses §10.3's ScanNet reading.)
- **Job 11632049 — C-long — FAILED 2026-08-25.** ScanNet + ScanNet++ at 40 epochs / 82 160 steps,
  meant as the **step-matched** partner for A-long. It completed and all 8 matrix cells are green,
  but the run destabilised: 16 epochs of rising train loss (total +45.2, worst +11.4 @ ep 6), final
  train loss 122.6 against arm C's 93.2 on **identical data at half the steps**, and best epoch =
  last. Excluded as causes: data, config, `head_config`, and commit `9da8dfe` (measured RNG-inert);
  schedule length alone is excluded by arm B's clean 35 epochs. **Cause: the learning rate** — the
  same 1e-4 divergence §11.3 isolated on arm D, triggered here by *exposure* (a 40-epoch cosine
  holds LR near peak twice as long as arm C's 20-epoch one) rather than by dose.
  **Re-run: C-long′ 11831105 at lr 5e-5, matrix chained 11831106; control A-long′ 11830142**
  → docs/MULTIDATASET.md §10.5.
- (arm A-long landed 2026-08-23 — job 11498642, 18 h 47, matrix 11540891–11540905 chained and
  green. It **flipped the sign of §10.3**: at 84 k steps the full 3520-scene mixture is the best
  run of the block on every 2D axis AND takes all eight of its 3D cells → docs/RESULTS.md §7.5
  finding 4, §8.2's extra-data row.)
- (the three data-scaling arms landed 2026-08-22 — jobs 11435332 / 11435335 / 11435338, gated on
  smoke 11434972 → docs/MULTIDATASET.md §10.3. Their first chain jobs 11436321/23/24 failed on a
  `$0`-in-SLURM bug in `slurm/chain_eval3d_matrix.sh`, now fixed and regression-tested; the matrix
  was submitted by hand → §10.1.)
- (job 10484000, the first mixture, landed 2026-08-12 and was scored in 3D on 2026-08-13 by
  10596569 → docs/MULTIDATASET.md §9, docs/RESULTS.md §6.3. Two scale-only driver bugs preceded
  it, both fixed and both regression-tested: a SIGPIPE under the inherited `set -e` (§7.1) and the
  128 KB argv cap (§7.2).)
- (todo 6g's control landed 2026-08-12, jobs 10094393 → 10279969 → 10427048 → 6g.)
- (the multi-dataset baseline landed 2026-08-10, job 10287578 → docs/MULTIDATASET.md §6; the
  smoke run 10287385 **failed on a driver bug** → MULTIDATASET.md §7.1.)
- (Everything else previously listed here landed on 2026-08-07: the 3D ruler on 2e's winners
  → 2e; the lifting-knob sweep on the `--anchor_3d` checkpoint → §9.8.1, worth **+0.047**
  class-agnostic AP50, far more than the +0.016 it was worth on the old checkpoint; seed
  variance → 2g.)

## 5. Lifting quality — the new binding constraint (opened 2026-08-03 by §9.6)

**SHARPENED 2026-08-22 by the data-scaling matrix (docs/RESULTS.md §7.5).** Out of domain the
unposed bridge scores **0.000 for every arm** — 1201, 2054 and 3520 training scenes alike — while
the same checkpoints gain +59 % / +70 % AP50 on ScanNet++ / Replica under the *posed* bridge. The
registration diagnostics are identical across arms to three decimals (ICP inliers 0.963 / 0.924 /
0.660, camera RMS 0.097 / 0.116 / 0.143 m) because they depend only on **VGGT's frozen cameras**.
So out of domain this workstream is not one lever among several: **it is the only one**, and no
amount of 2D supervision substitutes for it. That makes 5c (registration) the item with the
clearest evidence behind it, not 5b.

On the 3D ruler the decoder is no longer what limits us: AP25 ≈ 4× AP50, median camera-centre
RMS after Sim(3) is 0.14 m, and only ~16 % of mesh vertices get a vote. 5a bounded what knobs can
do *on the §9.6 checkpoint* (0.067 → 0.091 AP50, under FAST3DIS's 0.096), so the remaining gap was
read as **coverage and registration, in that order**. **AMENDED 2026-08-27 — coverage is out**:
5b closed as a negative (docs/RESULTS.md §5.4), so **registration (5c) is the only item left in
this workstream**.

**AMENDED 2026-08-07 (docs/MASKDINO.md §9.8.1).** Re-swept on the `--anchor_3d` checkpoint the
same two knobs are worth far more — class-agnostic AP50 **0.138 → 0.185**, +0.047 against the
+0.024 they were worth before — and *every* point of the grid now leads FAST3DIS and IGGT rather
than trailing them. Two consequences: the knobs are **checkpoint-dependent** (the confidence
filter even flips sign), so re-sweep them per checkpoint instead of carrying a tuned value; and
the "gap is not a tuning artefact" reading of 5a is now a statement about the *old* checkpoint
only. Ordered by expected value per hour:

**REPRIORITISED 2026-08-04 by 5e.** The posed-transfer measurement puts a **hard ceiling of
0.156 AP50 on 5b + 5c combined** — that is what our *current* masks score with a perfect
2D↔3D bridge. Lifting work is therefore worth at most +0.089 AP50, and its oracle also shows
view count is not the binder. Above that ceiling only the masks themselves move the number, so
**2d (and multi-view completeness/identity generally) now outranks 5b/5c**. Do 5b/5c only for
the parts that are cheap.

- [x] **5a. Knob sweep — DONE 2026-08-03** (8 points, docs/MASKDINO.md §9.8). Reported as a
      **sensitivity analysis, not a headline** (swept on val-312). Findings: the vote radius
      **saturates at ~0.15 m = the median registration error** (0.090 AP50 at 0.15, unchanged at
      0.20/0.30 — doubling it does nothing), the confidence filter has an interior optimum at
      25 % and trades AP25 for AP50, and the **whole grid spans 0.067 → 0.091 AP50, still below
      FAST3DIS's 0.096**. The useful negative: the remaining gap is *not* a tuning artefact, so
      it must come from 5b/5c below. Also confirmed the pipeline is deterministic (the repeated
      defaults run reproduced §9.6 exactly).
      **Re-run on the `--anchor_3d` checkpoint 2026-08-07 (§9.8.1) — the last sentence inverts
      there:** the grid spans 0.138 → 0.185 class-agnostic AP50 and its *worst* point already
      leads FAST3DIS's 0.096 by 1.44×. Radius still saturates at 0.15 m; the confidence filter
      flips to neutral-negative. Also measured: **`--eval_topk` 100 → 600 is neutral** (0.138 →
      0.140), so query count is struck from the list of explanations for the SegVGGT gap.
- [x] **5b. Coverage — CLOSED 2026-08-27, NEGATIVE** (docs/RESULTS.md §5.4 reading 3). More
      frames buy coverage monotonically (voted vertices 0.171 → 0.268 → 0.308 at 17 / 50 / 71
      views, annotated-assigned 0.667 → 0.754 → 0.787) and AP **stops moving at 50** while
      coverage keeps rising to 71. So coverage is no longer what binds the unposed column —
      **5c (registration) is now the only lifting item left**. The original text, for the record:
      **Coverage.** ~16 % of vertices voted / ~65 % of annotated vertices assigned caps
      recall outright. Options: more frames per scene (the 25k export has ~16–30; SegVGGT's
      *eval* uses ~75–100, every 20th frame of a full `.sens` extraction — 2–24 is their
      training sampling, so we are behind here, not level), overlapping bundles, or per-frame
      confidence-weighted voting instead of hard argmax. Measure `annotated_assigned_frac` as
      the intermediate target, not AP.
- [ ] **5c. Registration.** Sim(3)+ICP on camera centres is the FAST3DIS convention, but our RMS
      is at the vote radius. Try ICP on the *voted points* rather than camera centres, or
      per-bundle registration with a consistency check. NOTE: registration is eval-only — it must
      never leak GT geometry into the prediction path (that is the project's selling point).
- [ ] **5d. Only after 5a–5c:** revisit whether the decoder ever becomes the constraint again on
      this ruler. If it does, 2d (3D anchors) is the natural next decoder change — and it acts
      exactly on the geometry that binds here.
- [x] **5e. POSED-transfer protocol on our own masks — DONE 2026-08-04** (docs/MASKDINO.md
      §9.10, RESULTS.md §5.1; jobs 9607206 / 9607208 / 9607210). Implemented as
      `--transfer_mode {unproject,gt_projection}` (default = unchanged behaviour), geometry in
      `train/eval3d_geometry.py`, licensed by `scripts/eval3d_projection_oracle.py`
      (round-trip purity **0.9999** over all 312 scenes; a wrong pixel mapping collapses it).
      Same checkpoint, same 17.4 frames, same 97.6 queries, same evaluator — only the bridge
      moves: **0.023 / 0.067 / 0.268 → 0.060 / 0.156 / 0.408**, coverage 0.153 → 0.342 voted and
      0.635 → 0.791 annotated-assigned. Three results that redirect the workstream:
      **(i)** the bridge costs **2.3× AP50**, so that is a hard **ceiling on 5b + 5c combined** —
      perfect lifting reaches 0.156, not SegVGGT's 0.717;
      **(ii)** the protocol explains **2.3× of the ~10.7× SegVGGT gap and no more** — ~4.6× is
      real (LoRA backbone, 4–6× the views, 259×196 vs 37×37 masks, topk 600 vs 100), so the §9.9
      line "a different protocol, not a different league" was **wrong** and is struck;
      **(iii)** the oracle ceiling on our own frame budget is 0.948 AP50, so **view count is not
      what binds the posed column** — what binds at 0.156 is the 3D-instance criterion itself
      (one mask covering a whole object across *all* its views at IoU > 0.5, where the per-frame
      ruler reports 0.65). That is multi-view completeness and identity, i.e. **the decoder is
      back in play on this ruler** → re-prioritise 2d above 5b/5c.
      The number is a diagnostic decomposition and is never the headline; the headline stays the
      unposed 0.023 / 0.067 / 0.268.

## 6. Comparability programme (supervisor, opened 2026-08-07) — the current top priority

Everything in §1 made our *evaluation* placeable. This makes the *training setting* placeable, and
completes the evaluation to every benchmark the competitors report on. Full plan and budget lines:
the approved plan file; the audit itself: `docs/TRAINING_COMPARABILITY.md`; the multi-dataset
training arm (6e + 6f) has its own home in **`docs/MULTIDATASET.md`**.

- [x] **6a. The training × evaluation audit — DONE 2026-08-07** (`docs/TRAINING_COMPARABILITY.md`).
      Three findings that reshape the rest: **SegVGGT trains on the same official 1201 split we do**
      (so that row needs no retraining, only more evaluation); **FAST3DIS trains on Aria Synthetic
      Environments ONLY and scores all three benchmarks zero-shot**; **IGGT trains on InsScene-15K,
      which contains ScanNet++**, one of its own eval sets. Field practice, read from the papers
      rather than assumed: **nobody finetunes on the target benchmark**, and SegVGGT §4.3 explicitly
      treats baselines that do as weaker for it. Also settled: the ~3.6 TB of `.hdf5` packs under
      `csakarid/data/3D_datasets` are **RGB+depth only** — no instance/semantic/pose annotations
      anywhere, including `ASE.hdf5` — so they cannot supply any competitor's supervision.
- [x] **6b. SegVGGT dissection — DONE 2026-08-07** (`docs/SEGVGGT_ANALYSIS.md`). **Their release has
      no training code at all** (no loss/matcher/optimizer/dataset; FADA ships as the hook without
      its loss), so their training setting is paper-only and unreproducible. Gap decomposition
      updated per checkpoint: on `--anchor_3d` the total is **×6.4**, the bridge **×2.3**, the
      **residual ×2.8** — down from the ×4.6 written in §9.10, bought entirely by 2d + 2e. Quote
      ×2.8 with its checkpoint, not ×4.6.
- [x] **6c. Build the missing eval datasets — DONE 2026-08-09.** ~~ScanNet200 val GT~~,
      ~~ScanNet++ val-50 GT + 50-view frames~~, ~~Replica 8 scenes~~. Node-local per
      `docs/DATASET.md` §5.1; scratch inode cost **zero**.
  - [x] **ScanNet200 DONE 2026-08-09 — zero download, as predicted.** The 200 valid raw ScanNet
        ids (`data/scannet200_constants.py`, cross-checked against the TSV by
        `tests/test_scannet200_taxonomy.py`) plus a `--taxonomy {nyu40,scannet200}` switch in
        `train/scannet3d.py::build_gt_ids` turn the **existing** val-312 tars into the
        ScanNet200 ruler. Default `nyu40`, so no existing number moves.
  - [x] **Replica 8 scenes DONE 2026-08-08** (job 10100042) —
        `dataset/replica/{replica_3d_gt_8,replica_frames_8}.tar.zst` (372 MB + 417 MB, CC-BY-NC-4.0).
        GT mesh (`habitat/mesh_semantic.ply`, per-face `object_id`, verified in the PLY header) +
        `info_semantic.json` from the official `facebookresearch/Replica-Dataset` release; frames
        (50/scene, uniformly sampled, manifest.json with the full list) from `kxic/vMAP`'s
        `vmap.zip`, confirmed to hold exactly these 8 scenes' traj-00 (iMAP) renders. No explicit
        intrinsics file was found in the downloaded tree, so the fallback is the standard
        habitat/NICE-SLAM/vMAP values (fx=fy=600, cx=599.5, cy=339.5 @ 1200×680) — flagged as
        `FALLBACK` per scene in the build's `REPORT.json`, not silently assumed. Adapter (6d) still
        needed to actually score against it.
  - [x] **ScanNet++ val-50 DONE 2026-08-08** (jobs 10089394 / 10091616) —
        `scannetpp_3d_gt_val50.tar.zst` (1.9 GB, **49 scenes / 2585 instances**) +
        `scannetpp_frames_val50.tar.zst` (980 MB, **49 × 50 = 2450 frames**). 49 not 50:
        `d755b3d9d8`'s upstream trajectory diverges to 7.2 km and the build's geometry
        self-check refused to ship it. See `docs/DATASET.md` §2.1 (contents, the four
        silently-failing conventions and the evidence for each) and §5.0 (rebuild).
        `legacy/dataset_build/{scripts/{scannetpp_common,build_scannetpp_3d_gt,
        build_scannetpp_frames}.py, slurm/build_scannetpp_val50.sh}`, verified by
        `scripts/verify_scannetpp_gt.py` and `tests/test_scannetpp_build.py`. Scratch loose-file
        cost 0. **Feeds 6d**: the frames tar is 50 sampled views/scene with GT poses + sensor
        depth, i.e. it supports both transfer modes; note for the adapter that ScanNet++
        `segments.json` is one segment per vertex, so the superpoint vote degenerates.
- [x] **6d. Dataset adapters + the cross-dataset eval matrix — DONE 2026-08-09**
      (`docs/RESULTS.md` **§7**, mechanism in `docs/MASKDINO.md` §9.12). `train/scannetpp3d.py` /
      `train/replica3d.py` as siblings of `train/scannet3d.py`, the `train/datasets3d.py` registry
      behind `--dataset {scannetv2,scannet200,scannetpp,replica}` (default `scannetv2`), 80 new
      CPU checks (`tests/test_datasets3d.py`, `tests/test_scannet200_taxonomy.py`).
      **All four licence gates passed at exactly 1.000/1.000/1.000** on every scene of the real
      tars (`scripts/gate_3d_gt.py`, `slurm/gate_3d_gt.sh`), and the regression guard reproduced
      the `--anchor_3d` ScanNetv2 row **0.038 / 0.112 / 0.360** to the last digit — as did all six
      ScanNetv2 class-aware cells. 24 cells scored (`slurm/eval_3d_matrix.sh`,
      `scripts/collect_eval3d_matrix.py`), 0 failed scenes.
      **Three findings.** (i) **ScanNet200 costs no data at all** and is a genuine second column
      (2.3× the GT instances: 10 045 vs 4 364), whose sign against ScanNetv2 is
      *checkpoint-dependent*, like §9.11's label collapse. (ii) **Zero-shot to ScanNet++ and
      Replica is 0.000 under the unposed bridge** on all three checkpoints, and small but real
      under the posed one (`--anchor_3d`: 0.038 AP50 ScanNet++, 0.028 Replica). (iii) Reporting
      both bridges **localises the failure**: posed coverage out of domain is nearly ScanNet's
      (0.685 vs 0.834 annotated vertices assigned) so the AP loss there is the **2D masks**, while
      unposed coverage collapses to a third (0.223 / 0.255) so out of domain the **feed-forward
      geometry fails first** — §9.6's "lifting binds", amplified. On Replica, VGGT's cameras are
      the specific culprit (ICP inliers 0.66 vs 0.96, camera RMS 0.275 m vs 0.136).
      Two things the builds had flagged were handled and re-verified rather than rediscovered:
      ScanNet++'s `segIndices` really is one segment per vertex on all 49 scenes (the superpoint
      vote degenerates to a per-vertex vote, reported per scene), and Replica's FALLBACK
      intrinsics + `traj_w_c` as camera-to-world + **millimetre** depth put the sensor surface
      0.55 cm from the mesh (÷6553.5, the NICE-SLAM constant, lands 65–91 cm out).
      One correction worth keeping: the gate's geometry check originally failed a scene on its
      *worst* probe frame, which **failed ScanNet itself** — 3/312 val scenes carry one drifted
      probe up to 64.7 cm while no scene median exceeds 9.6 cm. It now fails on the scene median
      and reports single-frame outliers. A rule that fails the reference dataset is the wrong rule.
- [x] **6e. `--class_agnostic` training mode — DONE 2026-08-10** (`docs/MULTIDATASET.md` §3, 13 CPU
      checks in `tests/test_class_agnostic.py`). Needed to train on any dataset without the ScanNet
      taxonomy; also the setting FAST3DIS and IGGT report in. **One rule, no second flag: a
      one-class head means class-agnostic** — `build_frame_targets` reads `num_classes == 1` off
      `head_config`, keeps every instance instead of dropping the ones it cannot name, and
      collapses the label, so training and all three scorers share one code path and a *checkpoint
      alone* decides how its GT is built. Defaults off; every published number stays class-aware.
      **Measured 2026-08-10** (job 10287578, MULTIDATASET.md §6, RESULTS.md §6.2): collapsing the
      taxonomy costs −0.020 per-bundle AP50 with `id_switch`/`view_consistency` unmoved — the head
      was not leaning on the 18-way class head for instance separation, which is the premise the
      whole 6f arm rests on.
- [~] **6f. InsScene-15K arm** (IGGT's own training data, Apache-2.0). **Download DONE 2026-08-08**
      (job 10106802): the full mirror is on work at `dataset/insscene15k/`, 522.07 GB / **1565**
      files (not ~120 — `processed_infinigen` is 1468 small per-scene zips, `processed_re10k` 44,
      `processed_scannetpp_v2` 53), shards untouched (never unzipped), `README.md` alongside with
      licence/date/manifest sha256. Re-checked at mirror time: still **no Aria/ASE directory**
      upstream, so this remains a **partial replication** by construction, not by our choice — an
      ASE arm is out of scope and permanently so: 9.2 TB *and* the sampled scene list is
      unpublished.
      **The 2D supervision turned out to be already there — no rendering needed** (this reverses
      `docs/TRAINING_COMPARABILITY.md` §4's assumption): `processed_scannetpp_v2` ships
      `refined_ins_ids`, `processed_infinigen` ships `ObjectSegmentation`, ids are global per scene
      in both, and `processed_re10k` is skipped — **not** for lack of annotation (it ships SAM2 masklets; corrected 2026-08-24, `docs/MULTIDATASET.md` §1.3) but because they are automatic, not GT. **BUILD DONE 2026-08-10** (job
      10286143, 1 h 42): `dataset/insscene2d/insscene2d_scannetpp.tar.zst` (1.28 GB, **853**
      scenes — the 50 `nvs_sem_val` dropped, they contain all 49 scenes of our ScanNet++ eval
      column) + `insscene2d_infinigen.tar.zst` (2.14 GB, **1466** sub-scenes, room shell dropped by
      name). Loader `data/instance_map_dataset.py`, driver `slurm/train_maskdino_multi.sh`, 71 new
      CPU checks, all documented in **`docs/MULTIDATASET.md`** — that file is this item's home.
      **THE MIXTURE IS TRAINED — job 10484000, done 2026-08-12, scored in 3D 2026-08-13**
      (docs/MULTIDATASET.md §9, docs/RESULTS.md §6.3). Two driver bugs that only exist at full
      scale preceded it and are both fixed and regression-tested: the `CAP_*` SIGPIPE under the
      `set -e` inherited from the sourced `slurm/stage_dataset.sh` (§7.1) and the 128 KB
      `MAX_ARG_STRLEN` argv cap (§7.2).
      **Result: at a step-matched budget the mixture LOSES on the ScanNet val ruler** (per-bundle
      AP50 0.434 vs the ScanNet-only baseline's 0.505) — arithmetic, not a surprise: 1201 of 3520
      scenes are ScanNet, so 6 epochs give each ScanNet scene 6 passes against 16, a 2.7× cut in
      exposure to the domain val is drawn from. Matching *total* steps does not match what predicts
      the ScanNet number, and the run had not converged either.
      **The three data-scaling arms LANDED 2026-08-22** (docs/MULTIDATASET.md §10.3,
      docs/RESULTS.md §6.4): same recipe + `--anchor_3d`, matched ~42 k steps, only the mixture
      moves. On the ScanNet val ruler **adding real ScanNet++ is free** (per-bundle AP50 0.548 →
      0.554, inside the 0.009 seed spread) and **adding synthetic Infinigen costs 0.075** — but A
      is the only arm that had not converged, so read that as an upper bound and wait for A-long
      (11498642).
      **THE MATRIX LANDED 2026-08-22** (24 cells, 0 failures; docs/RESULTS.md §7.5,
      docs/MULTIDATASET.md §10.4) **and it reverses the ScanNet reading**: C ⇄ B is flat in domain
      and **+59 % AP50 on ScanNet++ / +70 % on Replica** under the posed bridge. More data buys
      exactly what the ScanNet val ruler could not see — score a data arm on the matrix, never on
      the val ruler. **Every out-of-domain UNPOSED cell is still 0.000 for every arm**, with ICP
      inliers and camera RMS identical across arms to three decimals, because they depend only on
      VGGT's frozen cameras: out of domain the binding constraint is the backbone's geometry, and
      no amount of 2D supervision substitutes for it (→ §5 below, sharpened).
      **A-LONG LANDED 2026-08-23 AND FLIPPED THE INFINIGEN SIGN.** At 84 480 steps the full
      3520-scene mixture is the best run of the block on every 2D axis (per-bundle AP50 0.479 →
      **0.604**, per-frame **0.704**, best `id_switch` 0.414) and takes **all eight** of its 3D
      cells: unposed ScanNetv2 **0.057 / 0.166 / 0.516** (+20 % AP50 over the published
      ScanNet-only `--anchor_3d` row and 1.5–2.0× the published class-agnostic cluster), and 2.5×
      arm B's zero-shot AP50 on ScanNet++ and Replica. The larger the mixture, the more steps it
      needs before it pays — a step-matched deficit is not evidence that data hurts.
      **STILL OPEN: C-long.** Job 11632049 was that step-matched partner and **failed** as an
      unstable run — the LR failure of §11.3 (docs/MULTIDATASET.md §10.5) — so "more data" vs "more
      compute" at the top end is still unmeasured and A-long's win is still confounded with its 2×
      step budget. Re-run in flight: C-long′ 11831105 ⇄ A-long′ 11830142, both at lr 5e-5.

- [~] **6j. RE10K as a FOURTH training source — SAM2-supervised, its own arm** (opened 2026-08-24,
      `docs/MULTIDATASET.md` §1.3, §1.4, §11). **6f's "processed_re10k has no annotation" was
      wrong**, and structurally so: the survey grouped member paths by their depth-2 component and
      the masks live under a **sibling** top-level directory, `processed_re10k/sam2_results/`.
      Re-read from the 43-part split zip (1 221 783 members, no unpacking): **5127 of 5138 scenes**
      ship `auto_masks.json`, SA-V masklets, COCO-RLE per frame, **ids persistent across the whole
      clip** — the property §1 requires, and the only source in the mirror that has it at this
      scale, i.e. cheap `--multi_frame` identity supervision.
      **The caveat that travels with every number: these are SAM2 output, not ground truth.** Say
      **SAM2-supervised** in every row and never fold it into A/A-long's.
      DONE: `slurm/coco_rle.py` (COCO compressed RLE in pure numpy — `pycocotools` is not in
      `myenv` — verified against `pycocotools`' own output, 47 checks); `build_re10k` +
      `--source re10k` + the room-shell measurement (39 checks); the training driver takes a 4th
      source with **zero code change**, proved by 8 new checks in
      `tests/test_train_maskdino_multi_sh.sh`. Three traps found and handled, all measured rather
      than assumed: rgb stems are **8 OR 9 digits** so a lexicographic sort misaligns masks in 107
      scenes; resolution is **per scene** (360×640, 540×960, 1080×1920); overlaps resolve
      **smaller-instance-wins**; and the room-shell cap sits at **0.30 of the frame** (drops 0.5 %
      of instances and 0 % of the median scene's labelled pixels, against 0.20's 21.8 %).
      **No exclusion list is needed** — RE10K is not one of the four 3D benchmarks, so there is
      nothing it can leak.
      **KNOWN CONFOUND, measured and documented rather than papered over**: the area cap does NOT
      make RE10K shell-free. SAM2 splits a wall or ceiling into sub-regions each under 30 %, and
      there is no knee to cut at — border contact rises smoothly with area and even above 30 %
      only two thirds of instances are shell-shaped, while a 0.10 cap would delete 60 % of the
      labelled pixels. So arm D adds new scenes **and** shell supervision the other three sources
      do not have. First hypothesis if it loses 3D AP with healthy 2D masks.
      **BUILD DONE 2026-08-24** (job 11641723, 4 h 42): `insscene2d_re10k.tar.zst` **9.7 GB**,
      **5127 scenes, 0 failed, 0 `None` masklet entries**, 158 903 frames, 370 562 instances
      (median 61/scene), 1.83 % of masklets dropped by the 0.30 cap + `min_area_px`. Smoke
      (11642515) passed: all four sources staged and cached, loss 252.6 → 224.1. **RE10K is the
      DENSEST source in the mixture** — 223–476 instance-frames per 8-frame bundle against
      ScanNet++'s 75–295 and ScanNet's 48–66, with 0.64–0.98 of every frame labelled foreground.
      **ARM D AT lr 1e-4 DIVERGED — job 11642516 is a failed run, not a measurement**
      (docs/MULTIDATASET.md §11.2, docs/RESULTS.md §6.4.1 + §7.5.1). It completed cleanly (17 h 15,
      352 GiB peak RSS of 416, 0 failures, no NaN) and is still garbage: best `bundle_AP50`
      **0.136 at epoch 2 of 17** vs A-long's 0.604, training loss RISING 132 → 169 while the LR
      decayed, and **`train_AP50` collapsing 0.211 → 0.006** — the head stopped fitting its own
      training data. All 8 3D cells collapsed with it, to *below* the ScanNet-only 1201 control.
      **CAUSE ISOLATED 2026-08-25, one variable at a time (§11.3): the LEARNING RATE, not RE10K.**
      `D-lr` (11744294) holds mixture, dose and `--anchor_3d` fixed and halves the LR to 5e-5 — the
      collapse vanishes and it tracks A-long epoch for epoch, ending ep6 *ahead* (0.369 vs 0.364)
      on 43 % more scenes. `D-dose` (11744296) shows the instability is dose-dependent at 1e-4
      (8 % RE10K trains fine), but the LR is the operative knob because it fixes the full dose.
      **A label conflict cannot be undone by halving a learning rate**, so the SAM2-supervision and
      room-shell hypotheses are refuted *as the cause of this failure* — they stay open only as
      questions about a converged arm's quality.
      IN FLIGHT (launched 2026-08-26): **D-long** (11830140, 5020 scenes, lr 5e-5, 17 ep =
      85 340 steps) and **A-long′** (11830142, 3520 scenes, lr 5e-5, 24 ep = 84 480 steps), each
      chaining its own 4×2 matrix (11830144 / 11830145). **A-long′ is not optional**: published
      A-long ran at 1e-4 and arm D cannot run there at all, so without it the comparison moves two
      variables — the flaw this workstream has already hit twice. D-long ⇄ A-long′ is one variable,
      the RE10K data, at matched steps and matched schedule. Read it on the MATRIX, not the val
      ruler (§10.4).

- [x] **6g. Upstream MaskDINO trained under OUR recipe — DONE 2026-08-12, and it AGREES**
      (docs/old/MASKDINO_COCO.md §6 row 2 + §6.1; jobs 10094393 → 10228029 → 10279969 → 10427048,
      87 948 iters). Built as `third_party/maskdino_control/` (own README); driver
      `legacy/coco/slurm/train_maskdino_upstream.sh`; tests `legacy/coco/tests/test_maskdino_upstream_control.py` (5/5, needs
      the **reference** env). 8 axes moved to our values, 22 asserted still at upstream's.
      **§4.1 gate PASSED: segm AP 52.1 @600 vs our `resnet50` arm's 54.3** — the two implementations
      track each other point-for-point, which was the first evidence for `matcher.py`,
      `criterion.py` and DN generation.
      **RESULT: 34.55 segm AP on full val2017 against our `resnet50` arm's 34.3 — Δ +0.25**, with
      ±0.84 AP at all 16 matched periodic steps (mean +0.23, sign changing) and the same
      1000→5000-image population offset (−2.37 vs our −2.36). The stated failure criterion ("far
      above 34.3 = a bug in our training path") did **not** fire: two independently written
      matchers, criteria and DN generators converge to a quarter of an AP over 88 k steps of real
      COCO. **Two things this licenses:** the port is now certified on the *training* path, not
      just inference (§7.6 excluded exactly those three modules), and **the 46.1 → 34.5 distance is
      a measured recipe cost (~11.6 AP), not an inference against a differently-trained model.**
      ~42 h at 1.70 s/iter, batch 16 peaks at 23.9 GB, so it self-resubmits. Three things this cost,
      all recorded where they bite:
      the clone's MSDeformAttn `.so` is **sm_86-only** and upstream's bare `except:` turns a wrong
      arch into a silent ~10× slowdown (`build_ops.sh`, and an unwrapped kernel call at startup);
      torch 1.10 **rejects** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`; and a 600-step gate
      measures the LR schedule unless the cosine's horizon is pinned to the real budget
      (`CONTROL.LR_HORIZON_ITERS`, docs/old/MASKDINO_COCO.md §4.1 — two failed attempts read as a broken loss).
      Plus a fourth, mid-run: **the scratch purge ate the reference venv** and surfaced as
      `UnidentifiedImageError` on a perfectly good COCO jpg (docs/old/MASKDINO_COCO.md §6.1). Resumed from
      `/cluster/work` at zero cost; the 40 000 eval is the only casualty.
- [x] **6i. Move the MaskDINO reference venv off scratch — DONE 2026-08-12.** `$MASKDINO_ROOT/myenv`
      had been purged once mid-run (6g) and the project's own `myenv/` once before (2026-08-07,
      CLAUDE.md); rebuilding in place restarts the 15-day clock but guarantees a repeat.
      **The WHOLE clone moved, not just the venv** — `/cluster/scratch/niacobone/MaskDINO` →
      **`/cluster/home/niacobone/MaskDINO`** (4.9 GB, 21 102 files; $HOME is at 34.6/45 GB and
      112 k/450 k files after, both comfortable). The clone's own `.py` sources and the sm_80+86
      `MultiScaleDeformableAttention.so` sit under the same 15-day rule as the venv, so splitting
      them would have left half the problem. Verified before deleting the original: identical file
      counts, and a checksum `rsync -nc` differing on exactly the 43 `myenv/bin/` scripts whose
      shebang/`VIRTUAL_ENV` were rewritten (129 bytes = 43 × 3 chars, the path shortening).
      No `.so` carried an RPATH into scratch. `legacy/coco/tests/test_maskdino_upstream_control.py` passes
      6/6 under the moved venv, which is the §7.6 re-verification this item asked for.
      **What made it more than a rename**, and is now fixed so the next move IS one: the control
      configs inherit upstream's own COCO yaml by **absolute path**, and the overfit gate inherits
      *that* by a relative path — so a moved clone breaks the §4.1 gate one level below where you
      look. `third_party/maskdino_control/config_paths.py::resolve_base` re-roots a dead absolute
      `_BASE_` at `$MASKDINO_ROOT`, following the relative chain, never editing the yaml on disk,
      and raising rather than silently falling back to detectron2 defaults. Wired into
      `train_control.py::setup` and the reference test; 23 CPU checks in
      `legacy/coco/tests/test_maskdino_control_paths.py`, which runs under the **project** venv (the module is
      stdlib-only on purpose).
- [ ] **6h. Optional second control: upstream's OWN recipe** (finetuned backbone, LSJ@1024, same
      87 948 iters, ~3 days). 6g alone cannot separate "our recipe" from "our implementation" — it
      pins the two together. This run pins the **recipe** cost, so the pair brackets the 46.1 gap
      from both sides. **Not launched** — out of the scope 6g was requested under; ask before
      spending it. Mechanically it is one more yaml against the same driver (revert FREEZE_AT,
      BACKBONE_MULTIPLIER, the mapper, the schedule and the clip; keep MAX_ITER 87948).

- [~] **6k. Dense ScanNet val-312 frames — the competitors' VIEW COUNT** (opened 2026-08-26,
      `docs/DATASET.md` §2.5, `docs/TRAINING_COMPARABILITY.md` §6.3). The last unmatched axis of
      the protocol comparison, and it runs *against* us: FAST3DIS evaluates on **50** uniformly
      sampled views and SegVGGT on **every 20th frame** (~75–120), while `scannet_frames_25k` gives
      us **17.42**. ScanNet++ and Replica are **already at exactly 50** (measured off the eval
      JSONs, not assumed), so the gap is confined to the two ScanNet columns.
      `extract_sens_frames25k.py` streams the whole `.sens` (~1.15 GB/scene at ~68 MB/s measured;
      no early abort is possible for a whole-scan sample) and writes the same tree `repack_frames25k`
      does. **Verified against the official export on `scene0011_00`: depth pixel-identical, poses
      to 5e-6, intrinsics to 5e-3 — but the color jpegs differ because the 25k export re-compressed
      them**, so the view-count comparison needs its own 17-frame control ON the dense tar.
      23 CPU checks in `tests/test_sens_frames_extract.py` (synthetic `.sens`, every format guard).
      **DONE 2026-08-27, and it closes two things** (docs/RESULTS.md §5.4; build 11840822 →
      pack 11840823 → cells 11841445/49/51/54/57/62/67, 312 scenes, 0 failures).
      **(i) At the competitors' 50 views the lead WIDENS**: `--anchor_3d` 0.137 → **0.170**
      class-agnostic AP50 (+24 %) and A-long 0.161 → **0.193** (+20 %), i.e. 1.77× / 2.01×
      FAST3DIS's 0.096 against 1.44× / 1.73× at 17. The 17-view control on the SAME tar
      reproduces the published row (0.044 / 0.137 / 0.488 vs 0.042 / 0.138 / 0.504), so the
      jpeg-recompression difference is noise-level.
      **(ii) The lever SATURATES at ~50**: 71 views gives 0.166, flat-to-negative against 50's
      0.170 — which is why 5b below closes as a negative. Posed at 50 views is the best posed row
      anywhere (A-long **0.200 / 0.419 / 0.725**, SegVGGT distance 1.84× → 1.71×).
      Cost note for the next run: the 100-view cell needed 7 h 28 and a 40 GB GPU; 50 views fits a
      4090 in ~2 h 45.
- [~] **6l. The ZERO-SHOT arms — matching what the competitors TRAIN on** (opened 2026-08-26,
      `docs/MULTIDATASET.md` §12, `docs/TRAINING_COMPARABILITY.md` §6.2). The largest remaining
      mismatch is not the protocol: **FAST3DIS and IGGT never train on ScanNet and every arm we
      have ever run does**, so §8.2's lead is favourable to us on the training axis before a number
      is read. Two arms, both `--class_agnostic --anchor_3d`, lr 5e-5, step-matched at ~84 k:
      **I** (ScanNet++ + Infinigen + RE10K@1500, 3819 scenes, job 11839134) = **IGGT's mixture minus
      ASE**, and **I-gt** (ScanNet++ + Infinigen, 2319, job 11839135) = the same without the
      SAM2-supervised source. With A-long′ and D-long they complete a **2 × 2 in {±ScanNet} ×
      {±RE10K}**, so each edge is one variable. Matrices chained (11839151 / 11839152).
      Driver change: the val-312 tar is now staged independently of `SOURCES`, so an arm that never
      trains on ScanNet is still scored on the same ruler (there, a zero-shot one) — 4 new checks in
      `tests/test_train_maskdino_multi_sh.sh`. Both verifications the user asked for came back TRUE:
      only ASE is missing for IGGT, and the "geometric GT" setting is SegVGGT's posed bridge, which
      we already implement and report (§6.1 there).

- [ ] **6m. SegVGGT's ScanNet200 checkpoint — the one competitor setting we CANNOT currently match,
      and what it would cost.** SegVGGT trains a *second* checkpoint on **ScanNet200 train** and
      reports it **class-aware** (31.9 / 45.7 / 53.7, posed). We evaluate ScanNet200 as a
      relabelling of the same val meshes (todo 6d) but **class-agnostic only**, because the
      supervision does not exist on our side: the 2D GT tars store instances under the 19 NYU40
      classes and drop everything outside that taxonomy to background (`docs/DATASET.md` §1), so a
      200-way head has nothing to learn from. Closing it means **rebuilding the 1201-scene 2D GT
      from `_2d-label-filt` at the raw label ids** (~1–2 days of streaming + packing, the §5.1
      node-local rules) plus a 200-class training arm. Not launched — state the gap instead:
      *"our ScanNet200 column is class-agnostic; SegVGGT's is class-aware; the two are not
      comparable"* (`docs/RESULTS.md` §7.4).

- [x] **6l. The zero-shot arms — CLOSED 2026-08-28** (`docs/MULTIDATASET.md` §12.3,
      `docs/RESULTS.md` §5.6). Arms I / I-gt done, all 16 matrix cells, 0 failed scenes. **The
      training-data asymmetry is priced**: without ScanNet, 0.005 / 0.023 / 0.251 against
      FAST3DIS's 0.038 / 0.096 / 0.316 — a factor 6 below our own headline AP50. The lead rests on
      training on ScanNet, and that now has a number. **Not** evidence the recipe loses at equal
      data: ASE is absent entirely (3819 scenes vs ~100 k). **Second finding: RE10K's sign flips**
      — −42 % AP50 in a mixture with ScanNet, +1.8× in one without; redundant vs ScanNet, valuable
      without it. Two failed matrix cells (12046077 / 12046106) were **GPU contention, not code**
      (`CUDA-capable device(s) is/are busy`) and were re-run as 12077651 / 12077653.
- [~] **6n. A partial ASE download — costed 2026-08-27, SCRIPTED 2026-08-31, waiting on ONE
      signature.** ASE **is** publicly available (projectaria.com/datasets/ase) and its per-scene
      GT **includes 2D instance segmentation**; the downloader takes scene ranges. Budget:
      ~23 TB / 100 k scenes ≈ **230 MB/scene**, so a **1 000-scene pilot ≈ 230 GB** against
      ~2.34 TB free scratch. What it buys: arm I becomes the **complete** IGGT replication instead
      of "minus ASE", which is what would let `docs/TRAINING_COMPARABILITY.md` §6.6's second and
      third rows be read as a *method* comparison instead of a data one. What it does not buy: a
      FAST3DIS-matched training set — their 40 % scene list is unpublished, permanently (§5).
  - [x] **The pipeline — DONE 2026-08-31** (`docs/TRAINING_COMPARABILITY.md` §6.7).
        `slurm/download_ase.py` (the official chunk protocol + resume markers on work + a time
        budget + the inode report), `slurm/fetch_ase.sh` (fetch → gate → probe → build → pack, in
        **blocks of 100 scenes** so it never holds 230 GB at once: `--tmp` is 60 GB, not 400), and
        `--source ase` in `slurm/build_insscene2d.py`. 26 + 53 CPU checks in
        `tests/test_ase_fetch.py` / `tests/test_insscene2d.py`.
  - [x] **The two non-obvious decisions inside it**, both tested: frames are **rotated upright**
        (ASE stores them 90° off, every other source is upright; rgb and ids go through the same
        numpy call so masks cannot drift off their objects), and the **room-shell cap is NOT
        inherited from RE10K** — `--probe` measures ASE's own area distribution and what each
        candidate cap would remove, `PROBE_ONLY=1` stops there.
  - [x] **The gate is the inode count**, wired into the driver: it prints files-per-scene per
        block and the projection for a 5× range. Scratch is quota'd on files and the InsScene
        mirror shipped 1 468 small zips where ~120 were expected.
  - [ ] **BLOCKED on the licence, and only on that.** The per-chunk CDN urls arrive after
        accepting the Project Aria dataset agreement — the account holder's act, not an agent's.
        Accept it, drop the json at `<work>/dataset/ase/ASE_cdn_urls.json`, then
        `sbatch slurm/fetch_ase.sh`. The job prints those instructions and exits 2 if it is absent.
  - [ ] **After the pilot lands:** read `PROBE_ase_*.json`, pick `MAX_AREA_FRAC` off its
        `dropped_frac_at` table, rebuild, then retrain arm I with ASE as a fourth source and
        re-run its 8-cell matrix — that is the row that closes the IGGT training axis.
- [x] **6o. Land the ablation-table hole on the 3D ruler — CLOSED 2026-08-28.**
  - [x] **`--no-cross_frame_attn` — job 11986399, DONE** (`docs/RESULTS.md` §5.5). 312 scenes, 0
        failures, defaults. Removing it costs **57 % of the 3D AP50** (0.067 → 0.029 class-aware,
        0.050 → 0.021 class-agnostic) — the largest single-mechanism effect in the project, and
        the ratio holds across both label settings. Single variable verified by a `config.json`
        diff: exactly one differing key.
  - [x] **`--feature_mode single` — jobs 11986440 → 12012326, DONE 2026-08-28.** A new 12-epoch
        run on the official 1201/312 split (no leak-free checkpoint of that arm existed), then its
        3D eval. **0.76× the control's AP50 class-aware, 0.51× class-agnostic** — the two columns
        disagree by ~1.5×, so this row must never be quoted without its label setting. In 2D it
        reproduces the retired figure on the new split (−0.166 per-bundle AP50 against −0.147).
        Single-variable (config diff: only `feature_mode`), schedule-matched not
        convergence-matched (both peak at epoch 12/12). **Reading that came out of the move to
        Tier 1: the 2D *ordering* of the two levers held, its *spacing* did not** — 1.24× apart in
        2D, 2.4× apart class-aware in 3D.
- [x] **6q. RE10K at matched compute — ANSWERED NEGATIVE in-domain 2026-08-27.** D-long (11830140,
      +1500 SAM2-supervised RE10K scenes) against A-long′ (11830142) at lr 5e-5, gradient-step
      budgets matched to within 1 % (85 340 vs 84 480 — the 17-vs-24 epoch difference is what
      holds them equal), same ScanNet val-312: **per-bundle AP50 0.5753 → 0.5241 (−0.051, 5.7× the
      seed spread)**, per-frame 0.6821 → 0.6522, `id_switch` 0.4035 → 0.4587. Read as
      **displacement at fixed compute**, not "bad data" — each ScanNet scene is seen 17 times
      instead of 24. **Does not settle the out-of-domain question RE10K was added for**; the 3D
      matrices (11996431 ff.) do, and are scoring. `docs/MULTIDATASET.md` §11.7. The LR diagnosis
      of §11.3 held: best epoch 15 of 17, loss monotone.
- [ ] **6p. Formal cross-view identity metrics — IMPLEMENTED, re-scoring in flight 2026-08-27.**
      `view_consistency` / `id_switch` are project-defined and have **no published counterpart**
      (verified: SegVGGT, FAST3DIS and IGGT report no cross-view consistency metric at all).
      `train/eval_metrics.py::tracking_consistency_metrics` adds **HOTA / AssA / DetA / IDF1**
      (`docs/MASKDINO.md` §6.6.1) with the bundle's views read as timesteps; jobs **11986564 /
      11986565** re-score the headline `--anchor_3d` checkpoint and its control via the new
      `--eval_only` path. **First run exposed a real bug and was re-launched (11994637/11994639):**
      the metrics were being scored on the raw `--eval_topk` pool, so DetA/IDF1 measured the query
      budget rather than the model (DetA 0.066); they now score the *submitted* detections, with
      the unfiltered variant kept under `_all` (`docs/MASKDINO.md` §6.6.1). The run did validate
      the path — `id_switch` reproduced at −0.088 against the recorded −0.089. When the re-run
      lands: put HOTA/AssA on the slides and demote the custom pair to an internal diagnostic.
      **Re-run DONE 2026-08-27** (`docs/MASKDINO.md` §6.6.2): headline checkpoint HOTA 0.422 /
      AssA 0.584 / DetA 0.314 / IDF1 0.492. **It revised a claim**: `id_switch` says `--anchor_3d`
      cuts identity errors by −0.088, AssA says +0.005 — an order-of-magnitude disagreement on the
      same two checkpoints, because `id_switch` flips on near-ties. Jobs **11997568 / 11997569**
      re-scored the seed-1 replicates. **SETTLED 2026-08-27** (§6.6.3): the first seed spread for
      these metrics is HOTA 0.011 / AssA 0.005 / DetA 0.017 / IDF1 0.021, and **every formal Δ for
      `--anchor_3d` is inside it** (AssA +0.0011). Only `id_switch` sees the effect (−0.076 vs a
      0.027 spread). **The identity half of the `--anchor_3d` claim is retired**; its +66 % 3D
      AP50 is untouched. Still to re-score: **A-long** (the 0.734 / 0.414 row), which needs the
      multi-dataset val staged.

## Longer-term / low priority

- [ ] `color_jitter` on/off alone has never been isolated from the extra bundles.
- [ ] Token-grid arm at 700/1036 px input (RoPE accepts any grid) — parked by §7.7: bounded by
      the 0.956→0.99 ceiling gap on ScanNet; only worth revisiting for a different dataset or
      if a reviewer demands it.
- [ ] Which-layer ablation (`--feature_layers 4,11,17,23`) — nearly free with the feature
      cache; VGGT-Det shows the appetite for "which VGGT layers carry object identity".
- [ ] Partial backbone unfreezing, once the train−val gap vs N says data supports it. Note it
      would surrender the frozen-backbone differentiator (docs/RELATED_WORK.md) — a deliberate
      decision, not a default next step.
