#!/bin/bash
#
# Scaling experiment: 100 train scenes (curve midpoint between scale50 and the full run).
# Submit: sbatch --export=ALL,INSTANCE_LEVEL=1 slurm/train_scale100.sh
#
#SBATCH --job-name=d4rt_scale100
#SBATCH --output=train_scale100_%j.log
#SBATCH --error=train_scale100_%j.err
#SBATCH --open-mode=append
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=5120
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
source slurm/stage_dataset.sh

# Held-out val set (scene0080–0089), excluded from train so every scaling point is comparable.
VAL=$(seq -f "scene%04g_00" 80 89 | paste -sd, -)
OUT=/cluster/work/igp_psr/niacobone/distillation/output

# Optional per-instance GT: submit with `sbatch --export=ALL,INSTANCE_LEVEL=1 ...`.
INSTANCE_FLAG=""; RUN_TAG=""
if [ "${INSTANCE_LEVEL:-0}" = "1" ]; then INSTANCE_FLAG="--instance_level"; RUN_TAG="_inst"; fi
RUN_TAG="${RUN_TAG}${EXP_TAG:-}"

# First 100 scenes from the non-val pool (0000–0079 + 0090–0109); val 0080–0089 stays held out.
TRAIN=$( (seq -f "scene%04g_00" 0 79; seq -f "scene%04g_00" 90 199) | head -100 | paste -sd, - )

$PYTHON scripts/train_multiscene.py \
    --scans_root $SCANNET_ROOT $INSTANCE_FLAG ${EXTRA_ARGS:-} \
    --train_scenes $TRAIN \
    --val_scenes $VAL \
    --num_epochs 1000 --warmup_epochs 30 --num_frames 8 --num_queries 32 \
    --learning_rate 2e-3 --bundles_per_scene 3 --query_jitter 0.02 --color_jitter 0.2 \
    --no_object_weight 0.1 --grid_size 6 --eval_interval 50 --early_stop_patience 0 \
    --cache_device cpu \
    --save_checkpoint $OUT/d4rt_m2_scale100${RUN_TAG}_$(date +%Y%m%d_%H%M%S)/checkpoint.pth
