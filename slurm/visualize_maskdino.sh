#!/bin/bash
#
# Re-render the figures of finished MaskDINO runs with the current drawing code, without
# retraining (scripts/visualize_maskdino.py). Needs the backbone (GPU) and the dataset tar,
# hence a job rather than a login-node command.
#
#   sbatch --export=ALL,RUNS='<run_dir_1> <run_dir_2>' slurm/visualize_maskdino.sh
#   sbatch --export=ALL,RUNS=<run_dir>,CKPT=checkpoint_best_ap50.pth slurm/visualize_maskdino.sh
#
# Knobs:
#   RUNS      space-separated run directories (default: the three multi-frame runs)
#   CKPT      checkpoint file inside each run dir (default checkpoint_best.pth)
#   DATA_TAR  dataset tar to stage (default: 500-scene official GT)
#
#SBATCH --job-name=maskdino_viz
#SBATCH --output=/cluster/scratch/niacobone/vggt/slurm/logs/maskdino_viz_%j.log
#SBATCH --error=/cluster/scratch/niacobone/vggt/slurm/logs/maskdino_viz_%j.err
#SBATCH --open-mode=append
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6144
#SBATCH --tmp=24000
#SBATCH --gpus=rtx_4090:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=niacobone@student.ethz.ch

set -euo pipefail

module purge
module load stack/2024-06 python/3.12.8 cuda/12.8.0 eth_proxy
cd /cluster/scratch/niacobone/vggt
source myenv/bin/activate
PYTHON=myenv/bin/python
export PYTHONUNBUFFERED=1

export DATA_TAR="${DATA_TAR:-/cluster/work/igp_psr/niacobone/distillation/dataset/scannet/scannet_official_gt_500.tar.zst}"
source slurm/stage_dataset.sh

OUT=/cluster/work/igp_psr/niacobone/distillation/output
RUNS="${RUNS:-$OUT/maskdino_sf_n490_mf_20260728_185730 \
$OUT/maskdino_sf_n490_mf_singlefeat_20260729_103343 \
$OUT/maskdino_sf_n490_mf_noxframe_20260729_103342}"
CKPT="${CKPT:-checkpoint_best.pth}"

for RUN in $RUNS; do
    echo "================ $RUN ================"
    $PYTHON scripts/visualize_maskdino.py \
        --checkpoint "$RUN/$CKPT" \
        --scans_root "$SCANNET_ROOT"
done
