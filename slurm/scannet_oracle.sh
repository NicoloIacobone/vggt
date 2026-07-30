#!/bin/bash
#
# GT-only mask-resolution ceiling on ScanNet (docs/MASKDINO.md §6.5) — CPU-only, ~minutes.
# Stages the 500-scene official-GT tar and runs scripts/scannet_mask_resolution_oracle.py
# over the val scenes. Result JSON lands in the shared output dir.
#
#   sbatch slurm/scannet_oracle.sh
#
#SBATCH --job-name=scannet_oracle
#SBATCH --output=/cluster/scratch/niacobone/vggt/slurm/logs/oracle_%j.log
#SBATCH --error=/cluster/scratch/niacobone/vggt/slurm/logs/oracle_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8192
#SBATCH --tmp=24000

module purge
module load stack/2024-06 python/3.12.8 eth_proxy
cd /cluster/scratch/niacobone/vggt
source myenv/bin/activate
export PYTHONUNBUFFERED=1

export DATA_TAR="${DATA_TAR:-/cluster/work/igp_psr/niacobone/distillation/dataset/scannet/scannet_official_gt_500.tar.zst}"
source slurm/stage_dataset.sh

OUT=/cluster/work/igp_psr/niacobone/distillation/output/scannet_mask_resolution_oracle.json
myenv/bin/python scripts/scannet_mask_resolution_oracle.py \
    --scans_root "$SCANNET_ROOT" --out "$OUT" ${EXTRA_ARGS:-}
echo "[oracle] wrote $OUT"
