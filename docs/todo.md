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

## 3. Resolution stream — CLOSED 2026-07-30 (docs/MASKDINO.md §7.7)

The mask grid is decoupled from the token grid ("VGGT is not an FPN" is answered,
docs/MASKDINO_COCO.md §1.2), the 37×37 GT-only ceiling on ScanNet is 0.956 AP50 vs the model's
~0.69, and `--mask_upsample 2` is neutral on the full-resolution ruler too. **Recognition
binds, not resolution — nothing left to do here on ScanNet.** The only surviving idea (a
700/1036 px token-grid arm) is bounded by the 0.956→0.99 ceiling gap and stays parked in
"Longer-term".

## 4. Watching

**One run open as of 2026-08-12: job 10484000, the multi-dataset mixture.** Everything else once
listed here has landed:

- (todo 2d fully landed — jobs 9634920 / 9670882 / 9670883. A regression re-run of the control
  under current code was submitted as 9848637 and **cancelled**; it is not needed, because the
  drift question is closed from the commit diff instead — see 2d.)
- (todo 2f fully closed 2026-08-12 — jobs 9979913 / 10477399 / 10477400 → 2f. Negative.)
- **Job 10484000** — the multi-dataset mixture, 3520 scenes (docs/MULTIDATASET.md §7/§8). The one
  open run. Two scale-only driver bugs preceded it, both fixed and both regression-tested:
  a SIGPIPE under the inherited `set -e` (§7.1) and the 128 KB argv cap (§7.2).
- (todo 6g's control landed 2026-08-12, jobs 10094393 → 10279969 → 10427048 → 6g.)
- (the multi-dataset baseline landed 2026-08-10, job 10287578 → docs/MULTIDATASET.md §6; the
  smoke run 10287385 **failed on a driver bug** → MULTIDATASET.md §7.1.)
- (Everything else previously listed here landed on 2026-08-07: the 3D ruler on 2e's winners
  → 2e; the lifting-knob sweep on the `--anchor_3d` checkpoint → §9.8.1, worth **+0.047**
  class-agnostic AP50, far more than the +0.016 it was worth on the old checkpoint; seed
  variance → 2g.)

## 5. Lifting quality — the new binding constraint (opened 2026-08-03 by §9.6)

On the 3D ruler the decoder is no longer what limits us: AP25 ≈ 4× AP50, median camera-centre
RMS after Sim(3) is 0.14 m, and only ~16 % of mesh vertices get a vote. 5a bounded what knobs can
do *on the §9.6 checkpoint* (0.067 → 0.091 AP50, under FAST3DIS's 0.096), so **the remaining gap
is coverage and registration, in that order**.

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
- [ ] **5b. Coverage.** ~16 % of vertices voted / ~65 % of annotated vertices assigned caps
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
      in both, and `processed_re10k` has none and is skipped. **BUILD DONE 2026-08-10** (job
      10286143, 1 h 42): `dataset/insscene2d/insscene2d_scannetpp.tar.zst` (1.28 GB, **853**
      scenes — the 50 `nvs_sem_val` dropped, they contain all 49 scenes of our ScanNet++ eval
      column) + `insscene2d_infinigen.tar.zst` (2.14 GB, **1466** sub-scenes, room shell dropped by
      name). Loader `data/instance_map_dataset.py`, driver `slurm/train_maskdino_multi.sh`, 71 new
      CPU checks, all documented in **`docs/MULTIDATASET.md`** — that file is this item's home.
      **STILL OPEN: the mixture has never been trained.** The end-to-end smoke (job 10287385) died
      in 2 minutes on a **driver bug, not on the data** — `slurm/stage_dataset.sh` is sourced and
      carries `set -euo pipefail`, so the `CAP_*` line `[ "$CAP" -gt 0 ] && LIST=$(echo "$LIST" |
      head -n "$CAP")` aborts the job by SIGPIPE once the list exceeds the 64 KB pipe buffer
      (ScanNet++'s 853 paths fit, Infinigen's 1466 do not). Reproduced; it only fires when `CAP_*`
      is set. Fix, re-smoke, then run the mixture with **at least** the baseline's 16-epoch budget
      (it had not converged) — MULTIDATASET.md §7.1 and §8.
- [x] **6g. Upstream MaskDINO trained under OUR recipe — DONE 2026-08-12, and it AGREES**
      (docs/MASKDINO_COCO.md §6 row 2 + §6.1; jobs 10094393 → 10228029 → 10279969 → 10427048,
      87 948 iters). Built as `third_party/maskdino_control/` (own README); driver
      `slurm/train_maskdino_upstream.sh`; tests `tests/test_maskdino_upstream_control.py` (5/5, needs
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
      (`CONTROL.LR_HORIZON_ITERS`, MASKDINO_COCO.md §4.1 — two failed attempts read as a broken loss).
      Plus a fourth, mid-run: **the scratch purge ate the reference venv** and surfaced as
      `UnidentifiedImageError` on a perfectly good COCO jpg (MASKDINO_COCO.md §6.1). Resumed from
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
      No `.so` carried an RPATH into scratch. `tests/test_maskdino_upstream_control.py` passes
      6/6 under the moved venv, which is the §7.6 re-verification this item asked for.
      **What made it more than a rename**, and is now fixed so the next move IS one: the control
      configs inherit upstream's own COCO yaml by **absolute path**, and the overfit gate inherits
      *that* by a relative path — so a moved clone breaks the §4.1 gate one level below where you
      look. `third_party/maskdino_control/config_paths.py::resolve_base` re-roots a dead absolute
      `_BASE_` at `$MASKDINO_ROOT`, following the relative chain, never editing the yaml on disk,
      and raising rather than silently falling back to detectron2 defaults. Wired into
      `train_control.py::setup` and the reference test; 23 CPU checks in
      `tests/test_maskdino_control_paths.py`, which runs under the **project** venv (the module is
      stdlib-only on purpose).
- [ ] **6h. Optional second control: upstream's OWN recipe** (finetuned backbone, LSJ@1024, same
      87 948 iters, ~3 days). 6g alone cannot separate "our recipe" from "our implementation" — it
      pins the two together. This run pins the **recipe** cost, so the pair brackets the 46.1 gap
      from both sides. **Not launched** — out of the scope 6g was requested under; ask before
      spending it. Mechanically it is one more yaml against the same driver (revert FREEZE_AT,
      BACKBONE_MULTIPLIER, the mapper, the schedule and the clip; keep MAX_ITER 87948).

## Recently closed (2026-07-29/30, 2026-08-01) — details in docs/MASKDINO.md §7.4.1, §7.7, §8.2

- [x] `--bundles_per_scene 4` (job 8950610) — **saturates** (0.699 / 0.722 vs b2's
      0.694 / 0.729, inside noise). Views-per-scene lever exhausted at 2; do NOT fold 4 into
      the default recipe.
- [x] `--no-cross_frame_attn` at N=490 (job 8950617) — bundle AP50 0.494 → 0.311.
      **Cross-frame attention is the main carrier of the multi-view result** — the only
      individually-decisive component found anywhere in this track.
- [x] `--multi_frame` on per-frame features (job 8950613) — bundle AP50 0.494 → 0.347.
      **Bundle features are required for multi-view consistency**, despite costing −0.048
      per-frame as a standalone change (§8.1). Consistency has a measured price.
- [x] 2026-07-30: resolution question closed (§7.7, oracle + two full-res runs); multi-view
      best moved to **0.539 / 0.515** (job 9071415, §8.2); COCO r50/dinov2 arms final
      (MASKDINO_COCO.md §6).
- [x] 2026-08-01: COCO `vggt` arm COMPLETE (job 9262006). **vggt final 37.7 AP vs dinov2 38.8**
      (−1.1 AP at identical geometry); best checkpoint vggt 39.7 @75k vs dinov2 41.3 @85k
      (−1.6 AP). Both trail early (14.1 vs 23.4 at overfit-gate), converge mid-training, diverge
      late. Verdict: 3D pretraining costs ~1–1.6 AP on 2D semantics (docs/MASKDINO_COCO.md §6,
      reading 3).

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
