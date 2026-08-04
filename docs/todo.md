# TODO

Open work only. Everything closed up to 2026-07-28 is in
`docs/old/todo_archive_20260728.md`; the reasoning behind each closed item is in
`docs/old/MILESTONES.md` (D4RT arms) and `docs/MASKDINO.md` §7 (MaskDINO).

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
      (0.028 / 0.112 / 0.287). SegVGGT (0.504 / 0.717 / 0.870) is far above but **in a different
      protocol** — posed transfer, GT poses + sensor depth, no geometry error (established
      2026-08-04, docs/MASKDINO.md §9.9) — so it is not a like-for-like gap; see 5e.
      Two findings: the leak-free checkpoint **beats** the leaked
      diagnostic 1.6× (data scale > leakage — the 3D ruler reproducing the 2D data-limited
      conclusion), and **the lifting step is the binding constraint, not the decoder** → the new
      workstream 5 below. Fixed while reporting: eval3d output files now name their non-default
      knobs (two knob settings used to overwrite one file; job 9503137's JSON was lost that way,
      numbers recovered from its log). Guarded by
      `tests/test_maskdino_eval3d.py::test_out_path_names_the_knobs`.

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
- [ ] **2d. 3D anchors vs 2D DAB boxes** (docs/MASKDINO.md §8.3 — full design sketch there).
      Build on `--multi_frame --feature_mode bundle` — settled by §7.4.1, bundle features are
      the right base. Framed and budgeted as an **ablation** (FAST3DIS owns the mechanism as a
      contribution), which also closes the arm-E loop.

## 3. Resolution stream — CLOSED 2026-07-30 (docs/MASKDINO.md §7.7)

The mask grid is decoupled from the token grid ("VGGT is not an FPN" is answered,
docs/MASKDINO_COCO.md §1.2), the 37×37 GT-only ceiling on ScanNet is 0.956 AP50 vs the model's
~0.69, and `--mask_upsample 2` is neutral on the full-resolution ruler too. **Recognition
binds, not resolution — nothing left to do here on ScanNet.** The only surviving idea (a
700/1036 px token-grid arm) is bounded by the 0.956→0.99 ceiling gap and stays parked in
"Longer-term".

## 4. Watching

(Nothing active — everything submitted on 2026-08-03 has landed.)

## 5. Lifting quality — the new binding constraint (opened 2026-08-03 by §9.6)

On the 3D ruler the decoder is no longer what limits us: AP25 ≈ 4× AP50, median camera-centre
RMS after Sim(3) is 0.14 m, and only ~16 % of mesh vertices get a vote. 5a has now bounded what
knobs can do (0.067 → 0.091 AP50, still under FAST3DIS's 0.096), so **the remaining gap is
coverage and registration, in that order**. Ordered by expected value per hour:

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
