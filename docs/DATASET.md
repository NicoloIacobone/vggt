# Dataset, ground truth, and storage

The supervision, the on-disk conventions, and how a job gets the data. This applies to both the
active MaskDINO track and the retired D4RT arms — they read the same trees through the same
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

**What it cost to switch** (arm C, N=190, learned queries, per-instance GT, val scenes
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

Specs on group storage: `OFFICIAL_GT_README.md`, `INSTANCE_MASKS_README.md` (+ `…_split2.md`).

## 3. Mask conventions (both GTs)

- uint8 `{0, 255}` PNGs at the scene's color-camera resolution (1296×968; 640×480 for the 9
  scenes above).
- Filename = the subset frame's stem + `.png`.
- `<k>` in `masks_instance/<class>_<k>/` is zero-based per class, in order of first appearance.
- Directory names use underscores: `shower_curtain_3`.
- The union of a class's instance masks equals its `masks/<class>/` mask.
- The loader defaults to per-class `masks/`; `--instance_level` (D4RT) / the default in the
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
existing checkpoint cannot be re-scored on them honestly.

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
