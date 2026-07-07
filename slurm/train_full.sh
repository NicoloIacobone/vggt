#!/bin/bash
#
# Scaling experiment: train on the WHOLE dataset (all 190 non-val scenes; val 0080–0089
# held out, as in every scaling run). Submit: sbatch --export=ALL,INSTANCE_LEVEL=1 slurm/train_full.sh
#
#SBATCH --job-name=d4rt_full
#SBATCH --output=train_full_%j.log
#SBATCH --error=train_full_%j.err
#SBATCH --open-mode=append
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=8000
#SBATCH --tmp=16000
#SBATCH --gpus=rtx_4090:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=niacobone@student.ethz.ch

module purge
module load stack/2024-06 python/3.12.8 cuda/12.8.0 eth_proxy
cd /cluster/scratch/niacobone/vggt
source myenv/bin/activate
PYTHON=myenv/bin/python

# Stage the full dataset onto node-local scratch and export SCANNET_ROOT.
# GT tar: official ScanNet GT by default (docs/OFFICIAL_GT_MIGRATION_PLAN.md). For the
# SAM3-GT baseline: sbatch --export=ALL,DATA_TAR=/cluster/work/igp_psr/niacobone/distillation/dataset/scannet/scannet_instance_dataset_full.tar.zst ...
export DATA_TAR="${DATA_TAR:-/cluster/work/igp_psr/niacobone/distillation/dataset/scannet/scannet_official_gt_full.tar.zst}"
source slurm/stage_dataset.sh

# Held-out val set (scene0080–0089), excluded from train so every scaling point is comparable.
VAL=$(seq -f "scene%04g_00" 80 89 | paste -sd, -)
OUT=/cluster/work/igp_psr/niacobone/distillation/output

# Optional per-instance GT: submit with `sbatch --export=ALL,INSTANCE_LEVEL=1 ...`.
INSTANCE_FLAG=""; RUN_TAG=""
if [ "${INSTANCE_LEVEL:-0}" = "1" ]; then INSTANCE_FLAG="--instance_level"; RUN_TAG="_inst"; fi
RUN_TAG="${RUN_TAG}${EXP_TAG:-}"

# Train pool = all scenes EXCEPT the held-out val 0080–0089 → 0000–0079 + 0090–0199 (190 scenes).
TRAIN=$( (seq -f "scene%04g_00" 0 79; seq -f "scene%04g_00" 90 199) | paste -sd, - )

# 190×3 cached bundles → ~120 GB host RAM cache (--cache_device cpu); 20×8000=160 GB requested
# (scale100 used 62.7 GB for 300 bundles, so 570 bundles ≈ 120 GB; nodes have ~500 GB). Self-contained
# checkpoints (default uint8 images, NOT --checkpoint_light) so visualization needs no scans tree.
$PYTHON scripts/train_multiscene.py \
    --scans_root $SCANNET_ROOT $INSTANCE_FLAG ${EXTRA_ARGS:-} \
    --train_scenes $TRAIN \
    --val_scenes $VAL \
    --num_epochs 1000 --warmup_epochs 30 --num_frames 8 --num_queries 32 \
    --learning_rate 2e-3 --bundles_per_scene 3 --query_jitter 0.02 --color_jitter 0.2 \
    --no_object_weight 0.1 --grid_size 6 --eval_interval 50 --early_stop_patience 0 \
    --cache_device cpu --no_visualize \
    --save_checkpoint $OUT/d4rt_full${RUN_TAG}_$(date +%Y%m%d_%H%M%S)/checkpoint.pth

# NOTE: --no_visualize keeps the job inside the 4h walltime. On the full 190-scene set,
# training + staging already takes ~3h45 (more at --mask_upsample 2), so the end-of-run
# auto-render (200 scenes) previously pushed the job past the limit (job 5275027 TIMEOUT at
# 20:15 — training had finished at 19:58, only the viz was truncated). Render overlays
# separately afterward on any GPU node:
#   myenv/bin/python scripts/visualize_masks.py --checkpoint $OUT/<run>/checkpoint_best_ap50.pth
