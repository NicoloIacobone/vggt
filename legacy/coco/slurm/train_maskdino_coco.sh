#!/bin/bash
#
# COCO backbone-swap study (docs/MASKDINO_COCO.md): train the ported MaskDINO decoder on COCO
# instance segmentation with a frozen, swappable backbone.
#
#   sbatch --export=ALL,BACKBONE=resnet50 slurm/train_maskdino_coco.sh   # the control
#   sbatch --export=ALL,BACKBONE=vggt     slurm/train_maskdino_coco.sh   # the question
#   sbatch --export=ALL,BACKBONE=dinov2   slurm/train_maskdino_coco.sh   # VGGT's token twin
#   sbatch --export=ALL,BACKBONE=vggt,EPOCHS=6,EXP_TAG=_short slurm/train_maskdino_coco.sh
#
# Knobs (via --export=ALL,VAR=...):
#   BACKBONE     vggt | resnet50 | dinov2                   (default vggt)
#   EPOCHS       COCO epochs, the "1x" detection schedule    (default 12)
#   BATCH        EFFECTIVE images per optimiser step         (default 16, MaskDINO's IMS_PER_BATCH)
#   MICRO_BATCH  images per forward/backward                 (default 4; memory only)
#   EXTRA_ARGS   appended verbatim to the python call
#   EXP_TAG      appended to the run directory name
#   RUN_DIR      resume/extend a specific run instead of creating one
#
# A 12-epoch run is ~88 k optimiser steps at ~0.6-1.1 it/s, i.e. 22-42 h depending on the
# backbone, so it does not fit one wall clock. The job therefore **self-resubmits**: the first
# submission freezes its settings into <run_dir>/job_env.sh, `--resume auto` picks up
# checkpoint_last.pth, and the tail re-queues until the python process writes summary.json.
# Nothing is lost beyond the last --ckpt_interval steps.
#
# `--time_budget_hours` (default 22.5, below the 24 h wall clock) is what makes that work.
# **The resubmit CANNOT rely on being killed**: at the wall clock SLURM tears down the whole
# batch script, the `sbatch` at the bottom never executes, and the run silently stops halfway.
# So python stops itself early, checkpoints, and exits 0 without writing summary.json.
#
#SBATCH --job-name=maskdino_coco
#SBATCH --output=/cluster/scratch/niacobone/vggt/slurm/logs/maskdino_coco_%j.log
#SBATCH --error=/cluster/scratch/niacobone/vggt/slurm/logs/maskdino_coco_%j.err
#SBATCH --open-mode=append
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=6144
#SBATCH --gpus=rtx_4090:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=niacobone@student.ethz.ch

set -euo pipefail

module purge
module load stack/2024-06 python/3.12.8 cuda/12.8.0 eth_proxy
cd /cluster/scratch/niacobone/vggt
PYTHON=myenv/bin/python
export PYTHONUNBUFFERED=1
export HF_HOME=/cluster/scratch/niacobone/.cache/huggingface
# COCO images carry up to 90+ instances, so the per-micro-batch peak varies a lot; expandable
# segments keeps that fragmentation from turning into an avoidable OOM.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# A resumed job MUST reuse the run dir AND its settings, or it would silently restart from
# scratch with a different config. The first submission writes them down; every resubmit sources
# that file, so the settings can never drift between segments of one run.
if [ -n "${RUN_DIR:-}" ] && [ -f "$RUN_DIR/job_env.sh" ]; then
    # shellcheck disable=SC1091
    source "$RUN_DIR/job_env.sh"
    echo "[cfg] resumed settings from $RUN_DIR/job_env.sh"
else
    BACKBONE="${BACKBONE:-vggt}"
    EPOCHS="${EPOCHS:-12}"
    BATCH="${BATCH:-16}"
    MICRO_BATCH="${MICRO_BATCH:-4}"
    COCO_ROOT="${COCO_ROOT:-/cluster/scratch/niacobone/coco}"
    OUT=/cluster/work/igp_psr/niacobone/distillation/output
    RUN_DIR="${RUN_DIR:-$OUT/maskdino_coco_${BACKBONE}_${EPOCHS}ep${EXP_TAG:-}_$(date +%Y%m%d_%H%M%S)}"
    mkdir -p "$RUN_DIR"
    {
        echo "BACKBONE=$BACKBONE"; echo "EPOCHS=$EPOCHS"; echo "BATCH=$BATCH"
        echo "MICRO_BATCH=$MICRO_BATCH"; echo "COCO_ROOT=$COCO_ROOT"
        echo "RUN_DIR=$RUN_DIR"
        printf 'EXTRA_ARGS=%q\n' "${EXTRA_ARGS:-}"
    } > "$RUN_DIR/job_env.sh"
fi

echo "=== GPU ==="; nvidia-smi --query-gpu=name,memory.total --format=csv
echo "=== CFG === backbone=$BACKBONE epochs=$EPOCHS batch=$BATCH micro=$MICRO_BATCH"
echo "=== RUN === $RUN_DIR"

# The ViT arms need mask_upsample 4 (the script's default) or the whole experiment sits under a
# 44.7 AP ceiling — scripts/coco_mask_resolution_oracle.py. A ResNet already has a stride-4 map,
# so the flag is inert there and one command line serves all three arms.
$PYTHON scripts/train_maskdino_coco.py \
    --backbone "$BACKBONE" \
    --coco_root "$COCO_ROOT" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH" \
    --micro_batch "$MICRO_BATCH" \
    --run_dir "$RUN_DIR" \
    --resume auto \
    --num_workers 10 \
    --time_budget_hours "${TIME_BUDGET_HOURS:-22.5}" \
    ${EXTRA_ARGS:-}

if [ ! -f "$RUN_DIR/summary.json" ]; then
    echo "=== incomplete: resubmitting with RUN_DIR=$RUN_DIR ==="
    sbatch --export=ALL,RUN_DIR="$RUN_DIR" slurm/train_maskdino_coco.sh
else
    echo "=== COMPLETE ==="
    cat "$RUN_DIR/summary.json"
fi
