#!/bin/bash
#
# The upstream-MaskDINO control arm for docs/MASKDINO_COCO.md §6: official MaskDINO on COCO,
# configured like our `resnet50` arm on every axis we control (12 ep / frozen R50 / squash@518 /
# our cosine schedule / grad clip 0.1). See third_party/maskdino_control/train_control.py.
#
#   bash third_party/maskdino_control/build_ops.sh          # ONCE: MSDeformAttn for sm_80
#   python third_party/maskdino_control/make_overfit_root.py --n 64
#   sbatch --export=ALL,GATE=1 slurm/train_maskdino_upstream.sh    # the §4.1 overfit gate
#   sbatch slurm/train_maskdino_upstream.sh                        # the real 87 948-iter run
#
# Knobs (via --export=ALL,VAR=...):
#   GATE         1 = run the 64-image overfit gate config instead     (default 0)
#   RUN_DIR      resume/extend a specific run instead of creating one
#   EXTRA_OPTS   appended verbatim as detectron2 KEY VALUE pairs
#
# WHY THIS SELF-RESUBMITS THE WAY IT DOES -- read before editing.
# 88 k optimiser steps do not fit one wall clock, and the naive "sbatch at the end of the batch
# script" does NOT work: at the wall clock SLURM tears down the WHOLE script, so the trailing
# sbatch never executes and the study silently stops half-finished. The tool must therefore stop
# ITSELF first. `CONTROL.TIME_BUDGET_HOURS` (22.5, under the 24 h wall) makes python checkpoint,
# exit 0, and NOT write summary.json -- and that absence is exactly what the test below reads.
# detectron2's `last_checkpoint` file plus `--resume` carry iteration, optimiser and LR state,
# so a segment boundary is invisible to the schedule.
#
# A100 80 GB, IMS_PER_BATCH 16, NO gradient accumulation. detectron2 has no accumulation, and
# halving the batch while doubling the iterations would change the optimisation and destroy the
# comparison -- so the batch is bought with memory, not with steps.
#
#SBATCH --job-name=maskdino_upstream
#SBATCH --output=/cluster/scratch/niacobone/vggt/slurm/logs/maskdino_upstream_%j.log
#SBATCH --error=/cluster/scratch/niacobone/vggt/slurm/logs/maskdino_upstream_%j.err
#SBATCH --open-mode=append
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=6144
#SBATCH --gpus=1
#SBATCH --gres=gpumem:80g
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=niacobone@student.ethz.ch

set -euo pipefail

MASKDINO_ROOT=/cluster/home/niacobone/MaskDINO
REPO=/cluster/scratch/niacobone/vggt
cd "$REPO"

# The REFERENCE env (py3.9 / torch 1.10+cu113 / detectron2 0.6), NOT the project's myenv/.
module purge
module load stack/2024-06 gcc/12.2.0 cuda/11.3.1 python/3.9 eth_proxy
PYTHON="$MASKDINO_ROOT/myenv/bin/python"
export PYTHONUNBUFFERED=1
export MASKDINO_ROOT
# the compiled MSDeformAttn op links against this torch build's libc10.so
export LD_LIBRARY_PATH="$MASKDINO_ROOT/myenv/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
# COCO images carry up to 90+ instances, so the per-step peak varies a lot. slurm/train_maskdino_coco.sh
# fights that fragmentation with `expandable_segments:True`, but this env is torch 1.10 — that option
# arrived in 2.1 and torch 1.10 REJECTS the whole PYTORCH_CUDA_ALLOC_CONF string at cuda init
# ("Unrecognized CachingAllocator option", job 10020066). max_split_size_mb is the 1.10-era knob.
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

GATE="${GATE:-0}"
if [ "$GATE" = "1" ]; then
    CONFIG=third_party/maskdino_control/configs/maskdino_upstream_matched_overfit.yaml
    DEFAULT_NAME=maskdino_upstream_gate64
    TIME_BUDGET=0.0
else
    CONFIG=third_party/maskdino_control/configs/maskdino_upstream_matched.yaml
    DEFAULT_NAME=maskdino_upstream_matched
    TIME_BUDGET="${TIME_BUDGET_HOURS:-22.5}"
fi

# A resumed job MUST reuse the run dir AND its settings, or it would silently restart from
# scratch with a different config. The first submission writes them down; every resubmit sources
# that file, so settings can never drift between segments of one run.
if [ -n "${RUN_DIR:-}" ] && [ -f "$RUN_DIR/job_env.sh" ]; then
    # shellcheck disable=SC1091
    source "$RUN_DIR/job_env.sh"
    echo "[cfg] resumed settings from $RUN_DIR/job_env.sh"
    RESUME_FLAG="--resume"
else
    OUT=/cluster/work/igp_psr/niacobone/distillation/output
    RUN_DIR="${RUN_DIR:-$OUT/${DEFAULT_NAME}_$(date +%Y%m%d_%H%M%S)}"
    mkdir -p "$RUN_DIR"
    {
        echo "CONFIG=$CONFIG"; echo "RUN_DIR=$RUN_DIR"; echo "GATE=$GATE"
        echo "TIME_BUDGET=$TIME_BUDGET"
        printf 'EXTRA_OPTS=%q\n' "${EXTRA_OPTS:-}"
    } > "$RUN_DIR/job_env.sh"
    RESUME_FLAG=""
fi

echo "=== GPU ==="; nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
echo "=== CFG === $CONFIG  budget=${TIME_BUDGET}h"
echo "=== RUN === $RUN_DIR"

$PYTHON third_party/maskdino_control/train_control.py \
    --config-file "$CONFIG" \
    --num-gpus 1 \
    $RESUME_FLAG \
    OUTPUT_DIR "$RUN_DIR" \
    CONTROL.TIME_BUDGET_HOURS "$TIME_BUDGET" \
    ${EXTRA_OPTS:-}

if [ ! -f "$RUN_DIR/summary.json" ]; then
    echo "=== incomplete: resubmitting with RUN_DIR=$RUN_DIR ==="
    sbatch --export=ALL,RUN_DIR="$RUN_DIR" slurm/train_maskdino_upstream.sh
else
    echo "=== COMPLETE ==="
    cat "$RUN_DIR/summary.json"
fi
