#!/bin/bash
#
# COCO val2017 weight-transplant validation of the ported MaskDINO stack.
# See scripts/coco_transplant_eval.py and docs/MASKDINO.md §7.5.
#
#   sbatch slurm/coco_transplant.sh              # full 5000-image val2017, both modes
#   sbatch --export=ALL,LIMIT=500 slurm/coco_transplant.sh
#
# Runs under the REFERENCE env (/cluster/scratch/niacobone/MaskDINO/myenv: py3.9 + torch 1.10 +
# detectron2 0.6 + pycocotools + the compiled MSDeformAttn op), NOT the project's myenv/.
# That torch build supports sm_37..sm_86 only, so this must land on a 3090 / A100 —
# an RTX 4090 (sm_89) or RTX PRO 6000 (sm_120) will fail with "no kernel image is available".
#
#SBATCH --job-name=coco_transplant
#SBATCH --output=/cluster/scratch/niacobone/vggt/slurm/logs/coco_transplant_%j.log
#SBATCH --error=/cluster/scratch/niacobone/vggt/slurm/logs/coco_transplant_%j.err
#SBATCH --open-mode=append
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6144
#SBATCH --gpus=rtx_3090:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=niacobone@student.ethz.ch

set -euo pipefail

MASKDINO_ROOT=/cluster/scratch/niacobone/MaskDINO
cd /cluster/scratch/niacobone/vggt

export PYTHONUNBUFFERED=1
export MASKDINO_ROOT
export COCO_ROOT=/cluster/scratch/niacobone
export DETECTRON2_DATASETS=/cluster/scratch/niacobone
# the compiled MSDeformAttn op links against this torch build's libc10.so
export LD_LIBRARY_PATH="$MASKDINO_ROOT/myenv/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

PYTHON="$MASKDINO_ROOT/myenv/bin/python"
LIMIT="${LIMIT:-0}"
OUT=/cluster/work/igp_psr/niacobone/distillation/output/coco_transplant

echo "=== GPU ==="; nvidia-smi --query-gpu=name,compute_cap --format=csv

for MODE in baseline ours; do
  echo "================ mode=$MODE limit=$LIMIT ================"
  $PYTHON scripts/coco_transplant_eval.py \
      --mode "$MODE" --limit "$LIMIT" --output "$OUT/$MODE"
done

echo "=== SUMMARIES ==="
cat "$OUT"/*/summary_*.json
