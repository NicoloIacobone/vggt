# Restoring this project on a new cluster

This repo was archived on **2026-08-13** as `cluster_backup_20260813.zip` (95.37 GB, 14350
entries, sha256 `9d457425debfdba9f013128ec15ce9cc49c861dc66c5e5bb61c465c52981aee4`) before the
original ETH Euler account expired. If you are an agent reading this after an unzip: this file
tells you what you have, what you do not, and the two things that will otherwise break.

Read `CLAUDE.md` next — it is the project's real entry point. This file only covers the move.

## 1. What the unzip gives you

Every entry carries its **original absolute path minus the leading slash**, so extracting into
any directory `$R` reproduces the old layout underneath it:

```
$R/cluster/scratch/niacobone/vggt/                      this repo (with .git, without myenv/)
$R/cluster/work/igp_psr/niacobone/distillation/dataset/ the dataset tars
$R/cluster/work/igp_psr/niacobone/distillation/output/  runs + checkpoints
$R/cluster/scratch/niacobone/.cache/{huggingface,torch}/ model weights incl. frozen VGGT-1B
$R/cluster/home/niacobone/MaskDINO/                     upstream-MaskDINO reference (no venv)
$R/cluster/home/niacobone/.claude/                      project memory + session transcripts
```

Extracting with `cd / && unzip …` restores the exact old absolute paths and nothing below needs
re-rooting — but that needs the same `/cluster/...` tree and root rights, so on a new machine you
will normally pick your own `$R`.

## 2. Rebuild the environment (the venv was NOT archived — it is not portable)

`myenv/` was deliberately excluded from both this repo and the MaskDINO reference. Rebuild it:

```bash
cd $R/cluster/scratch/niacobone/vggt
module purge
deactivate                       # only if a venv is active; harmless to skip
rm -rf myenv
module load stack/2024-06 python/3.12.8 cuda/12.8.0 eth_proxy   # Euler-specific; adapt
python -m venv myenv             # the original was Python 3.12.8
source myenv/bin/activate
pip install --upgrade pip wheel setuptools
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -r requirements_demo.txt
```

Two things this recipe does that the obvious short version does not, both load-bearing:
**`requirements_demo.txt` is required, not optional** — matplotlib, scipy, opencv, trimesh and the
rest of the active code's imports live there, not in `requirements.txt`; and the last two lines
**override** the cu128 line, so what you end up with is **torch 2.3.1+cu121 / torchvision
0.18.1+cu121 / numpy 2.5.2**, which is what every published number was produced with. Verified
2026-08-24 on Euler: 41 491 files, all 20 CPU tests green.

Put it **off** any purging scratch filesystem if you can. On the old cluster a 15-day scratch purge
destroyed the venv twice, and the two failures looked nothing alike — once the `.py` sources went
and the `.pyc` stayed, once the reverse plus the binaries (`Unable to find torch_shm_manager`). See
`CLAUDE.md` for both signatures and the diagnosis. Nothing about that bug is specific to Euler.

Point the caches at the restored weights so nothing re-downloads the frozen VGGT-1B backbone:

```bash
export HF_HOME=$R/cluster/scratch/niacobone/.cache/huggingface
export TORCH_HOME=$R/cluster/scratch/niacobone/.cache/torch
```

## 3. Re-root the hardcoded paths — this is what will actually break

The code carries absolute paths from the old cluster. Two prefixes cover essentially all of it:

| Old prefix | Occurrences in active code | What it points at |
|---|---:|---|
| `/cluster/work/igp_psr/niacobone/distillation` | ~2690 | datasets + run output |
| `/cluster/scratch/niacobone/vggt` | ~110 | this repo (SLURM `--output`, `cd`, …) |

Three levers, in order of preference:

1. **Environment variables**, where the code already offers them — no edits needed:
   - `SCANNET_ROOT` overrides `DEFAULT_SCANS_ROOT` in [train/common.py](train/common.py#L21)
     (default `…/dataset/scannet/scans`). This is the one that matters for training/eval.
   - `MASKDINO_ROOT` and `COCO_ROOT` in [scripts/coco_transplant_eval.py](scripts/coco_transplant_eval.py#L73).
2. **CLI flags** — `--scans_root`, `--save_checkpoint`, `--output_dir`, and `DATA_TAR` /
   `GT_TAR` / `FRAMES_TAR` on the SLURM drivers. See `docs/COMMANDS.md`.
3. **A sweep of the SLURM drivers**, which hardcode paths in `#SBATCH` lines that no variable
   can reach. Review the diff before committing it:
   ```bash
   grep -rl '/cluster/work/igp_psr/niacobone/distillation' slurm/ scripts/ train/ demos/
   # then, once $NEW_WORK and $NEW_REPO are set:
   grep -rlZ '/cluster/work/igp_psr/niacobone/distillation' slurm/ scripts/ train/ demos/ \
     | xargs -0 sed -i "s|/cluster/work/igp_psr/niacobone/distillation|$NEW_WORK|g"
   ```
   `legacy/` and `slurm/logs/` also match; leave both alone — `legacy/` is frozen on purpose and
   the logs are historical records, not code.

Also: the SLURM headers request Euler-specific resources (`--gpus=rtx_4090:1`, `--tmp=…`) and a
`niacobone@student.ethz.ch` mail address. Adjust per your scheduler.

## 4. Datasets ship as tars only — stage them, do not read them in place

There is deliberately **no unpacked `scans/` tree**: reading thousands of small PNGs off group
storage is slow and burns inodes. `slurm/stage_dataset.sh` unpacks a `.tar.zst` onto node-local
scratch and exports `SCANNET_ROOT=$TMPDIR/scans`. Sizes to budget for: `--tmp` well above 24000
for the 500-scene tar, ~35 GB for the 1201-scene one. Provenance and layout: `docs/DATASET.md`.

## 5. What is in the archive

| Block | Size |
|---|---:|
| `dataset/scannet` — official-GT tars 1201 / 500 / val312 / full, 3D GT + frames25k val312, SAM3-GT tar, READMEs, QA strips | 56.02 GB |
| `output/maskdino_*` — every active run in full (checkpoints, `metrics.jsonl`, eval json, figures) | 13.29 GB |
| HF + torch caches, incl. the frozen **VGGT-1B** backbone | 11.37 GB |
| `output/d4rt_*` — the published baseline bar `d4rt_full_inst_learned_officialgt_20260708_124452` in full; the other retired runs record-only, no `.pth` | 4.88 GB |
| `dataset/insscene2d` (2D training tars), `scannetpp` + `replica` (eval tars) | 7.22 GB |
| this repo without `myenv/`, with `.git` | 0.79 GB |
| MaskDINO reference without its venv, `~/.claude`, `demo_scans`, IGGT_official / SegVGGT / sam3 | 1.36 GB |

## 6. What is NOT in the archive

| Missing | Size | How to get it back |
|---|---:|---|
| `dataset/insscene15k` | 522 GB | Public mirror of <https://huggingface.co/datasets/lifuguan/InsScene-15K> (Apache-2.0). Only needed to **rebuild** the `insscene2d` tars, which are included. |
| `dataset/blendedmvs` + its backup tar | 243 GB | Belongs to the earlier distillation work, not the MaskDINO track. |
| `scratch/coco` | 20 GB | Public COCO 2017 download; needed only for the backbone-swap study (`docs/old/MASKDINO_COCO.md`). |
| `myenv/` in both repos, pip cache | ~11 GB | Rebuild — see §2. |
| `~/.claude/.credentials.json`, and `~/.claude/{backups,sessions,session-env,shell-snapshots,ide}` | small | Live OAuth tokens and volatile session state. Excluded on purpose; nothing to restore. |
| `~/.claude/backups/.claude.json.backup.1786628316095` | tiny | The one file the build job could not read (it was rotated away mid-zip). No project data. This is why the archive has 14350 of 14351 manifest entries. |

## 7. Confirm the restore before trusting it

All tests are standalone CPU scripts — no GPU, no backbone weights, no dataset needed:

```bash
cd $R/cluster/scratch/niacobone/vggt
for t in tests/test_*.py; do myenv/bin/python "$t"; done   # skip test_maskdino_upstream_control.py:
                                                           # it needs the MaskDINO reference venv
bash tests/test_train_maskdino_sh_lists.sh                 # DRY_RUN, no cluster needed
```

Then a real smoke test against one scene (`docs/COMMANDS.md` has the full catalogue):

```bash
myenv/bin/python scripts/train_maskdino.py --train_scenes scene0000_00 \
    --val_scenes scene0080_00 --num_epochs 50 --num_queries 300 --scans_root <staged_scans_root>
```

Checkpoints load standalone: each bundles the head plus its `head_config`, and the frozen
VGGT-1B backbone is fetched from HuggingFace (or the restored cache) rather than stored. For the
3D ruler use a `--multi_frame` checkpoint, `checkpoint_best_bundle.pth` — per-frame checkpoints
produce meaningless 3D instances.

## 8. Licences — before you copy this anywhere

The archive contains third-party data licensed to the original researcher for research use and
**not** for redistribution: **ScanNet v2** (per-person Terms of Use), **ScanNet++ v2**
(research-only, explicitly non-redistributable), **Replica** (see its `LICENSE.txt`). Re-hosting
the archive, or the tars inside it, is a licence question — resolve it before sharing.
