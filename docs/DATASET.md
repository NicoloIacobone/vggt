# Dataset, ground truth, and storage

The supervision, the on-disk conventions, and how a job gets the data. This applies to both the
active MaskDINO track and the retired retired baseline heads — they read the same trees through the same
loader (`data/scannet_overfit.py`).

## 1. Ground truth: official ScanNet v2 2D instance annotations

**Default since 2026-07-08.** Projections of the single human-verified 3D annotation
(`_2d-instance-filt` / `_2d-label-filt`): one class per object, cross-view-consistent instance
ids by construction.

**Why the switch.** A 2026-07-07 audit of the previous SAM3-generated GT (20 scenes) found
systematic **cross-class duplicates**: SAM3 prompts each class independently, so the same
physical object is often an instance under two classes — 68 pairs with cross-frame IoU ≥ 0.5
between different classes (~3.4/scene, mostly pixel-identical; desk↔table,
curtain↔shower_curtain, chair↔sofa, …), **15.9 % of foreground pixels multi-class**. Training
effect: the matcher demands two predictions for one object (built-in honest-AP50 false
positives) and the class head gets contradictory supervision.

**What it cost to switch** (the baseline head, N=190, learned queries, per-instance GT, val scenes
0080–0089):

| | val mIoU | honest val[grid] AP50 |
|---|---|---|
| trained on SAM3 GT, evaled on SAM3 GT (old headline) | 0.371 | 0.228 |
| trained on SAM3 GT, evaled on OFFICIAL GT (cross-eval) | 0.285 | **0.117** |
| trained on OFFICIAL GT, evaled on OFFICIAL GT (baseline) | **0.367** | **0.199** |

About half the old honest-AP50 headline did not survive clean GT — the SAM3-trained model had
fit SAM3's duplicate/label idiosyncrasies. Retraining on official GT recovered most of it.
Migration record: `docs/old/OFFICIAL_GT_MIGRATION_PLAN.md`.

Differences vs the SAM3 GT: masks written **sparsely** (a missing PNG means "not visible in this
frame"; the loader skips it); stuff classes keep official per-segment ids (`wall_0..wall_k`, not
forced to a single `_0`); out-of-taxonomy objects (including `otherfurniture`) become background.

## 2. The tars on group storage

**Datasets ship ONLY as tars** under
`/cluster/work/igp_psr/niacobone/distillation/dataset/scannet/`. There is deliberately **no
unpacked `scans/` tree on `work`** (deleted 2026-07: reading thousands of small PNGs off `work`
is slow and pressures the inode quota).

All three share the layout `scans/<scene>/raw_data/{subset,masks,masks_instance,_qa}` — each
scene with `subset/` (~100 stride-5 frames), `masks/<class>/` (per-class binary) and
`masks_instance/<class>_<k>/` (per-instance binary).

| Tar | Contents |
|---|---|
| **`scannet_official_gt_500.tar.zst`** — the **default** since 2026-07-09 | Official GT, **500 scenes (scene0000–0499), 7379 instances, 0 cross-class duplicates**. Scenes 0200–0499 built 2026-07-09 (`legacy/dataset_build/slurm/extend_dataset_500.sh`). 9 scenes (0240, 0243, 0269, 0292, 0354, 0366, 0438, 0456, 0483) have a **640×480** color camera; their GT is 640×480 too, so RGB↔GT stay consistent and the loader's resize to 518 handles the rest. |
| `scannet_official_gt_full.tar.zst` (2.3 GB) | The original 200-scene official-GT tar (scene0000–0199, 2950 instances). Kept for exact reproducibility of the runs trained on it. Instance count is below SAM3's ≈4195 because SAM3 double-counted duplicated objects. |
| `scannet_instance_dataset_full.tar.zst` (~2.6 GB) | The SAM3-generated GT (200 scenes, ≈4195 instances, with the duplicate defect above). Kept as the GT-quality baseline and as a project deliverable. Cross-frame identity from SAM3 video tracking; `wall`/`floor` forced single-instance; all-zero PNGs written where absent. |
| `scannet_3d_gt_val312.tar.zst` (1.2 GB) — **BUILT 2026-08-01**, docs/todo.md 1d | **3D benchmark GT** for the official val split (`data/splits/scannetv2_val.txt`, all 312 scenes): per scene `_vh_clean_2.ply` (the benchmark mesh) + `_vh_clean_2.0.010000.segs.json` (superpoints) + `.aggregation.json`. Layout `scans3d/<scene>/`, validated per scene (ply magic, segment-id closure). What `slurm/eval_3d_maskdino.sh` scores against (docs/MASKDINO.md §9). Built by `legacy/dataset_build/slurm/download_3d_gt_val312.sh` (job 9326394); a lost tar re-downloads in ~20 min. |
| `scannet_frames25k_val312.tar.zst` (1.1 GB) — **BUILT 2026-08-01**, docs/todo.md 1d | The val-312 slice of the official **`scannet_frames_25k`** export: per scene `color/*.jpg` + `pose/*.txt` (camera-to-world) + `depth/*.png` + intrinsics, 5 436 frames (~17/scene, sampled across the WHOLE scan — unlike the subset tars above, which cover only raw frames 0–495). Layout `scans25k/<scene>/`; **not** the `raw_data/{subset,masks,…}` layout, and carries no 2D GT. The 3D eval's input frames + its eval-only registration poses. Built by `legacy/dataset_build/slurm/download_frames25k_val312.sh` (job 9326395) from the one 6 GB `v2/tasks` zip. |
| `scannet_official_gt_1201.tar.zst` (29 GB) — **BUILT 2026-07-30**, docs/todo.md 1c | The full official ScanNet v2 train split (`data/splits/scannetv2_train.txt`, 1201 scan ids — includes `_01`/`_02`/... rescans, not a contiguous `_00` range). **1201 scenes, 17 638 instances, 0 cross-class duplicates, min label purity 1.0, no missing/failed scenes** (`OFFICIAL_GT_README_1201.md`; strips in `qa_strips_1201/`). Built by its own scripts (`extend_dataset_1201.sh` / `pack_official_gt_1201.sh`) into a **node-local** tree, so the 500-scene tar above is untouched *and* the scratch inode quota is not — see §5.1, the first attempt died on that quota. Not yet staged by any training script — point `DATA_TAR` at it, and mind `--tmp` (29 GB to stage, so `--tmp` well above the 500-scene tar's 24000) and the feature-cache RAM budget at 1201 scenes. |
| `scannet_official_gt_val312.tar.zst` (7.4 GB) — **BUILT 2026-08-01**, docs/todo.md 1c | The **val ruler for the official split** — same 2D GT and same `raw_data/{subset,masks,masks_instance}` layout as the 500/1201 tars, over the full official val list (`data/splits/scannetv2_val.txt`, 312 scan ids, `_01`/`_02` rescans included, **zero overlap with the 1201 train tar**). **312 scenes, 4630 instances, 0 cross-class duplicates, max cross-class IoU 0.0, min label purity 1.0, no missing/failed scenes** (`OFFICIAL_GT_README_val312.md`; strips in `qa_strips_val312/`). Archive entry count verified against source (347 439 png/jpg; 348 375 files incl. markers + `_qa/stats.json`). 8 scenes (0088_02, 0144_00, 0300_01, 0406_02, 0430_01, 0474_02, 0658_00, 0704_00) have a **640×480** color camera, handled exactly as in the 500-scene tar. Built by `extend_dataset_val312.sh` (job 9325618, 1 h 17, `ok=312 skip=0 fail=0` in both stages) → `pack_official_gt_val312.sh` (job 9328388, 2 min, chained via `CHAIN_PACK=1`), node-local per §5.1. Pair it with the 1201 tar for official-split training; nothing stages it yet. |

Specs on group storage: `OFFICIAL_GT_README.md`, `INSTANCE_MASKS_README.md` (+ `…_split2.md`).

## 3. Mask conventions (both GTs)

- uint8 `{0, 255}` PNGs at the scene's color-camera resolution (1296×968; 640×480 for the 9
  scenes above).
- Filename = the subset frame's stem + `.png`.
- `<k>` in `masks_instance/<class>_<k>/` is zero-based per class, in order of first appearance.
- Directory names use underscores: `shower_curtain_3`.
- The union of a class's instance masks equals its `masks/<class>/` mask.
- The loader defaults to per-class `masks/`; `--instance_level` (legacy) / the default in the
  MaskDINO trainer (`--class_level` to opt out) reads `masks_instance/`.
- ScanNet class indices are `1..19`, with `0` = background everywhere. In official GT, NYU40
  classes outside the 19 trainable (including `otherfurniture`) are background — see
  `docs/MASKDINO.md` §4 for how the head guards against a GT tree that keeps a 20th class.

## 4. Data access at training time

Each job copies ONE tar to node-local scratch (`$TMPDIR`) and unpacks it there:

```
slurm/stage_dataset.sh   →  exports SCANNET_ROOT=$TMPDIR/scans
```

`scripts/train_maskdino.py` uses `SCANNET_ROOT` as its default `--scans_root` (via
`train/common.py::DEFAULT_SCANS_ROOT`).

Which tar is staged comes from the `DATA_TAR` env var. `stage_dataset.sh`'s own default is still
the SAM3 tar for backward compatibility, but **every training SLURM script overrides it to the
500-scene official tar** and requests `#SBATCH --tmp=24000` (MB — enough for the tar plus the
unpacked tree). `zstd` lives at `/usr/bin/zstd`.

To run against the SAM3 baseline instead:

```bash
sbatch --export=ALL,DATA_TAR=/cluster/work/igp_psr/niacobone/distillation/dataset/scannet/scannet_instance_dataset_full.tar.zst \
       slurm/train_maskdino.sh
```

**Scratch copies are not canonical.** The official-GT build tree also exists unpacked at
`/cluster/scratch/niacobone/scannet_official_build/scans`, but scratch is purgeable and this
tree is already partly hollow (most scenes' `subset/` dirs were emptied by a purge; scene0000_00
and scene0080_00 still have frames, which is enough for smoke tests). The old SAM3 scratch build
trees (`scannet_build*`) are fully hollow. Re-pack from the tar if a tree is ever needed — never
unpack-to-retar on scratch (inode quota).

### 4.1 Scene splits

Training/val scenes are chosen by *scene id*, not by a split file: the tar holds
`scene0000_00 … scene0499_00`, train = the first `N_SCENES` minus the val ones, val = scenes
**0080–0089** (project convention; docs/RESULTS.md §1.1 for why it stays that way).

`data/splits/scannetv2_val.txt` is the **official ScanNet v2 val list** (312 scenes), fetched
2026-07-28 from `ScanNet/ScanNet@master:Tasks/Benchmark/scannetv2_val.txt`. It is used only by
`VAL_SPLIT=official sbatch slurm/train_maskdino.sh`, which intersects it with our range
(`*_00`, id < `N_SCENES` → **77** scenes at N=490) and trains on the remaining 413. That is a
separate run on purpose: 74 of those 77 scenes are inside the normal training range, so an
existing checkpoint cannot be re-scored on them honestly. That intersection is a *workaround* for
the 500-scene tar; with `scannet_official_gt_val312.tar.zst` (§2) the same list is usable as-is,
all 312 scenes, no intersection.

`data/splits/scannetv2_train.txt` is the matching **official train list** (1201 scan ids),
fetched 2026-07-30 the same way. Unlike our own N-scene range it is not `_00`-only or
contiguous by scene number (386 scenes have a `_01` rescan, 215 a `_02`, ...; only 420 of our
existing 500 `scene0000_00..scene0499_00` happen to be official-train). It drives the 1201-scene
dataset build (docs/todo.md 1c) — no trainer reads it yet.

The two lists are **disjoint** (verified: 0 shared scan ids), so the 1201-scene and 312-scene
tars together are the full official train/val protocol — the one every competitor reports on.
Both were built by the same pipeline with the same QA gates, so a number measured across them is
not confounded by GT provenance.

## 5. Rebuilding the GT (only if a tar is lost or conventions change)

The builders are retired to `legacy/dataset_build/` because the tars are canonical and already
built. Run order:

```bash
sbatch legacy/dataset_build/slurm/download_official_gt.sh   # download+convert 200 scenes (resumable)
sbatch legacy/dataset_build/slurm/extend_dataset_500.sh     # scenes 0200–0499, streamed from .sens
sbatch legacy/dataset_build/slurm/pack_official_gt.sh       # QA gates (0 cross-class dups) + strips + tar → work
```

`extend_dataset_500.sh` streams each scene's `.sens` from the TUM server and extracts only the
stride-5 subset jpgs with early abort (frames 0–495 sit in the first ~10 % of the stream), so no
`.sens` is ever stored.

The 1201-scene official-train build (docs/todo.md 1c) reuses the same two Python scripts but
can't use their `[start, end]` scene-number range (the split isn't `_00`-only or contiguous), so
both gained a `--scene_list FILE` option (`--start`/`--end` become 0-based line indices into
FILE instead). It has its own SLURM entry points, build tree, and tar name so the 500-scene
dataset is never touched by it — and it builds **node-local**, for the reason in §5.1:

```bash
sbatch legacy/dataset_build/slurm/extend_dataset_1201.sh <list_start> <list_end>  # one per chunk
sbatch legacy/dataset_build/slurm/pack_official_gt_1201.sh                        # → the tar on work
```

The 312-scene official-**val** build is the same pipeline again, pointed at
`data/splits/scannetv2_val.txt`, with its own chunk dir
(`/cluster/scratch/niacobone/scannet_val312_chunks`), tar, README and strips dir — so neither the
500- nor the 1201-scene tar is touched. It is small enough for one chunk, so the extend job
chains the pack itself and the whole build is one command (~1 h 20 end to end):

```bash
sbatch --export=ALL,CHAIN_PACK=1 legacy/dataset_build/slurm/extend_dataset_val312.sh 0 311
sbatch legacy/dataset_build/slurm/pack_official_gt_val312.sh   # only if packing separately
```

Its `--tmp` is 40000, not the 1201 build's 120000: 312 scenes are ~8 GB of tree plus a ~7.5 GB
tar.

### 5.1 The scratch inode quota — why the 1201 and val-312 builds never touch scratch

**`/cluster/scratch` is quota'd on FILE COUNT, not just bytes: 1.0 M soft / 1.5 M hard.** The
build tree is ~1046 files per scene (100 subset jpgs + ~950 per-instance/per-class mask pngs +
markers + `_qa/stats.json`), so:

| scenes | files | verdict |
|---|---|---|
| 312 (official val) | ~0.35 M | fits |
| 500 | ~0.52 M | fits |
| **1201** | **~1.26 M** | **over the soft limit; leaves no room for anything else** |

`extend_dataset_val312.sh` / `pack_official_gt_val312.sh` follow the same node-local discipline
even though 312 scenes would fit: the point is that a build should not spend inodes it doesn't
have to, and the 1201-scene tree already proved what happens when one does. Its build cost scratch
**1 inode** (one 7.5 GB chunk tar).

The first 1201-scene attempt (2026-07-30, jobs 9079912/14/15/17) materialised the tree on scratch
and died with `OSError(122, 'Disk quota exceeded')` after 1090 of 1201 scenes — and left the
account at 1 499 966 / 1 500 000 files, i.e. unable to write *any* new file anywhere on scratch.

The fix is not a bigger quota, it is to never keep loose files there. Scratch's **block** quota is
2.4 TB and barely used, so bytes are free while inodes are not, and **one tar is one inode**:

- `extend_dataset_1201.sh` builds the tree in `$TMPDIR` — node-local SSD, requested via
  `#SBATCH --tmp=120000` (nodes have 729 GB–1.8 TB; see `sinfo -o %d`), wiped when the job ends,
  not quota'd. Only one compressed **chunk tar** per range lands on scratch, in
  `/cluster/scratch/niacobone/scannet_1201_chunks/`.
- `pack_official_gt_1201.sh` unpacks those chunk tars back into `$TMPDIR`, runs the QA gates
  there, and copies only the finished tar to work.
- Net loose-file cost on scratch: **zero**. This is the same `$TMPDIR` discipline
  `slurm/stage_dataset.sh` already uses at training time, pushed back into the build.

Resumability survives the round trip, which is what makes this safe:

- Within a tree, `.subset_complete` / `.complete` markers skip finished scenes (unchanged).
- Across jobs, the chunk tar is restored at startup and rewritten at the end — **including after
  a partial failure**, which is the durability checkpoint. The download stage is bounded by
  `DL_BUDGET_H` (default 20 of the 24 h wall clock) so the job always reaches that rewrite; being
  killed at the wall clock would otherwise discard every scene converted in the run.
- An incomplete chunk exits non-zero *and resubmits itself*, up to `MAX_RESUBMITS` (default 5) —
  capped because a scene that is gone upstream can never reach `.complete` and would otherwise
  loop forever on the group's allocation. Because the resubmission is a new job id, a
  `--dependency=afterok` chain onto the original id would never fire, so the completing job
  submits the packing step itself when `CHAIN_PACK=1` (leave it 0 with several parallel chunks and
  submit the pack by hand). If scenes really are gone upstream, pack with `EXPECT_SCENES=<n>` —
  the QA gate counts scenes, so it must be told the real target.
- Before converting, scenes lacking `.complete` have `masks/ masks_instance/ _qa/` wiped:
  `build_official_masks.convert_scene()` only ever `mkdir(exist_ok=True)`-and-overwrites, so a
  conversion interrupted mid-write leaves residue it would not itself clean up. `subset/` is kept
  — that is the expensive `.sens` stream.

**One-off rescue.** `legacy/dataset_build/slurm/snapshot_build_1201.sh` folds an existing
on-scratch build tree into a chunk tar and deletes the tree, reclaiming the inodes. It is how the
failed attempt above was recovered rather than re-downloaded: all 1201 scenes already had their
`.sens` subsets extracted (~90 node-hours of streaming) and 1090 were fully converted, so only 111
scenes' GT zips remained. It verifies that *every regular file* — markers and `stats.json`
included, not just the png/jpg bulk — round-tripped before it removes anything.

## 6. Runs and checkpoints

- Output: `/cluster/work/igp_psr/niacobone/distillation/output/<run_name>/` (timestamped run
  names, e.g. `maskdino_sf_n490_20260727_161200`).
- `checkpoint_best.pth` = best val mIoU — the one to use for eval/demos.
  `checkpoint_best_ap50.pth` = the same run selected on AP50 instead.
- `metrics.jsonl` = one JSON line per eval (epoch, lr, losses, all metrics). Plots read this,
  not the logs.
- Checkpoints are **self-contained**: head weights + `head_config` + the cached scenes +
  optimizer/scheduler state (for `--resume`). The frozen backbone is reloaded from HF
  (`facebook/VGGT-1B`), never stored.
- SLURM job logs go to `slurm/logs/` (gitignored), not the repo root.
