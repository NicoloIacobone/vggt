#!/bin/bash
# Dataset extension to 500 scenes: for scenes [start..end] (default 200..499),
# stream the .sens to extract the stride-5 subset frames (legacy/dataset_build/scripts/extract_sens_subset.py,
# early-abort streaming — no .sens ever touches disk), then download the official
# 2D GT zips and convert them into the build tree (legacy/dataset_build/scripts/download_2d_gt.py,
# zips deleted per scene). Both stages are resumable — re-run to heal failures
# (markers: raw_data/.subset_complete and raw_data/.complete).
#
# Submit two parallel ranges: sbatch legacy/dataset_build/slurm/extend_dataset_500.sh 200 349
#                             sbatch legacy/dataset_build/slurm/extend_dataset_500.sh 350 499
# After both finish: QA + pack via legacy/dataset_build/slurm/pack_official_gt.sh (EXPECT_SCENES=500).
#
#SBATCH --job-name=extend_gt_500
#SBATCH --output=/cluster/scratch/niacobone/vggt/slurm/logs/extend_gt_500_%j.log
#SBATCH --error=/cluster/scratch/niacobone/vggt/slurm/logs/extend_gt_500_%j.err
#SBATCH --open-mode=append
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4096
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=niacobone@student.ethz.ch

# eth_proxy gives compute nodes outbound network access for the download.
module load eth_proxy 2>/dev/null || module load stack/2024-06 eth_proxy 2>/dev/null || true
set -u

START=${1:-200}
END=${2:-499}
REPO=/cluster/scratch/niacobone/vggt
BUILD=/cluster/scratch/niacobone/scannet_official_build

cd "$REPO"

# Stage 1: subset frames from the .sens streams.
myenv/bin/python legacy/dataset_build/scripts/extract_sens_subset.py \
    --out_root "$BUILD/scans" --start "$START" --end "$END"
SUBSET_RC=$?

# Stage 2: official GT zips + conversion (frame list = the extracted subsets).
# Runs even if stage 1 had failures — it only converts scenes whose subset exists;
# a heal re-run picks up the rest.
myenv/bin/python legacy/dataset_build/scripts/download_2d_gt.py \
    --zips_dir "$BUILD/zips" \
    --convert_out "$BUILD/scans" \
    --subset_root "$BUILD/scans" \
    --start "$START" --end "$END"
CONVERT_RC=$?

exit $(( SUBSET_RC || CONVERT_RC ))
