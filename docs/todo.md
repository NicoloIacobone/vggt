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

- [~] **1a. Full-resolution 2D eval — IMPLEMENTED 2026-07-30** (`--eval_full_res`,
      docs/MASKDINO.md §6.5, `tests/test_maskdino_fullres.py`): predictions bilinearly upsampled
      in logit space to the 518×518 GT id map, scored as `full_*` keys next to the unchanged
      grid keys; kept-prediction set still decided on the grid, so `full_*` isolates boundary
      quality. Default off, no existing number changes. **Remaining: re-run the headline recipe
      and `--mask_upsample 2/4` with the flag on** — expect upsample to stop being neutral on
      this ruler.
- [~] **1b. ScanNet mask-resolution oracle — IMPLEMENTED + SUBMITTED 2026-07-30**
      (`scripts/scannet_mask_resolution_oracle.py`, `slurm/scannet_oracle.sh`, job 9073136):
      the GT-only ceiling of the 37/74/148/259/518 grids under the full-resolution ruler, val
      scenes, our protocol. Quote it whenever mask resolution comes up. **Remaining: record the
      numbers in docs/MASKDINO.md §6.5 when the job lands.**
      Follow-up runs also submitted: jobs 9072738 (`--eval_full_res`), 9072749 (`… 
      --mask_upsample 2`), 9072761 (`… --mask_upsample 4 --train_num_points 12544` — upsample 4
      has never been trained on ScanNet; point-sampled mask loss because 148² full-pixel
      supervision is the COCO-recipe regime).
- [~] **1c. Extend the dataset to the full official ScanNet v2 train split (1201 scenes) —
      TAR BUILT 2026-07-30; nothing trains on it yet.** Simultaneously the protocol fix
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
      **Remaining:** point a training run at it (`DATA_TAR=...`) — raise `--tmp` (29 GB to
      stage vs the 500-scene tar's 10 GB) and check the feature-cache RAM budget at 1201
      scenes. The redundant 29 GB chunk tar in `/cluster/scratch/niacobone/scannet_1201_chunks/`
      can be deleted (blocks only, 1 inode).
- [ ] **1d. 3D benchmark eval.** Unproject per-view masks with VGGT's *own* predicted depth +
      cameras into the ScanNet benchmark point cloud and score the official 3D instance AP /
      AP50 / AP25 (SegVGGT recipe: majority-vote per superpoint). `--multi_frame` makes this
      natural — one query already is one instance across views, no post-hoc matching. This is
      what lets our table sit next to SegVGGT (50.4 / 71.7) and FAST3DIS; without it a 3DV
      submission has no anchor to the literature. Keep "no GT geometry at inference" as the
      selling point (docs/RELATED_WORK.md).

## 2. Complete the multi-frame study (the contribution)

- [~] **2a. Best data recipe × multi-frame** — `--multi_frame --feature_mode bundle
      --bundles_per_scene 2 --color_jitter 0.2`, N=490, EPOCHS=30: the current multi-view
      headline (0.535 / 0.494) does not yet benefit from the best-known lever (+0.030 per-frame
      AP50). **Submitted 2026-07-30, job 9071415.**
- [ ] **2b. Bundle-selected checkpoint.** `checkpoint_best*` selects on the *per-frame* metrics
      only; the per-bundle peak falls on a different epoch (§7.4.1). Add
      `checkpoint_best_bundle.pth` selected on bundle AP50 for `--multi_frame` runs, so the
      multi-view headline is not read off a per-frame-selected checkpoint.
- [ ] **2c. Cross-view consistency metric** (RELATED_WORK.md gap 2): per matched instance,
      cross-view IoU-agreement / ID-switch rate. Makes "3D consistent" a *measured* claim
      rather than an architectural one — and it is the metric that separates us from
      per-frame + fusion baselines.
- [ ] **2d. 3D anchors vs 2D DAB boxes** (docs/MASKDINO.md §8.3 — full design sketch there).
      Build on `--multi_frame --feature_mode bundle` — settled by §7.4.1, bundle features are
      the right base. Framed and budgeted as an **ablation** (FAST3DIS owns the mechanism as a
      contribution), which also closes the arm-E loop.

## 3. Resolution stream (gated on 1a/1b — do not start before)

The mask grid is already decoupled from the token grid (`--mask_upsample`, ViTDet-style
transposed convs; docs/MASKDINO_COCO.md §1.2 — "VGGT is not an FPN" is answered). The open
question is the **token grid** (detection/separation of small objects, §1.3). Spend here only
if the oracle (1b) or the COCO APs column says it binds on ScanNet:

- [ ] `--mask_upsample 2` and `4` re-measured under full-resolution eval (rides on 1a).
- [ ] Optional: one arm at 700 px (50×50 tokens) or 1036 px (74×74). VGGT is 2D-RoPE-only and
      accepts any grid (verified, docs/MASKDINO_COCO.md §1.3); on ScanNet the ~5× backbone cost
      is amortised by once-per-scene caching — the real costs are ~4× feature cache and ~4×
      encoder tokens in training.

## 4. Watching

- [~] **COCO backbone-swap arms** (jobs 9010539 / 9010540 / 9010546, ~half-way on 2026-07-30).
      Intermediate reads (NOT results): dinov2 39.8 segm AP @55k steps already **above** the
      frozen-R50 control's 35.0 @50k despite 2.7× fewer encoder cells → a 37×37-token ViT with
      `--mask_upsample 4` is not resolution-crippled on COCO; vggt 31.3 @25k (the slow arm,
      far behind on steps — no conclusion yet). Fill docs/MASKDINO_COCO.md §6 when
      `summary.json` exists, then read vggt-vs-dinov2 as "what 3D pretraining did to 2D
      semantics" at identical token geometry.
- [~] Job 9071415 (2a above).

## Recently closed (2026-07-29) — details in docs/MASKDINO.md §7.4.1

- [x] `--bundles_per_scene 4` (job 8950610) — **saturates** (0.699 / 0.722 vs b2's
      0.694 / 0.729, inside noise). Views-per-scene lever exhausted at 2; do NOT fold 4 into
      the default recipe.
- [x] `--no-cross_frame_attn` at N=490 (job 8950617) — bundle AP50 0.494 → 0.311.
      **Cross-frame attention is the main carrier of the multi-view result** — the only
      individually-decisive component found anywhere in this track.
- [x] `--multi_frame` on per-frame features (job 8950613) — bundle AP50 0.494 → 0.347.
      **Bundle features are required for multi-view consistency**, despite costing −0.048
      per-frame as a standalone change (§8.1). Consistency has a measured price.

## Longer-term / low priority

- [ ] `color_jitter` on/off alone has never been isolated from the extra bundles.
- [ ] Which-layer ablation (`--feature_layers 4,11,17,23`) — nearly free with the feature
      cache; VGGT-Det shows the appetite for "which VGGT layers carry object identity".
- [ ] Partial backbone unfreezing, once the train−val gap vs N says data supports it. Note it
      would surrender the frozen-backbone differentiator (docs/RELATED_WORK.md) — a deliberate
      decision, not a default next step.
