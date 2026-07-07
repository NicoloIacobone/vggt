#!/bin/bash
# Phase 2 of docs/OFFICIAL_GT_MIGRATION_PLAN.md: download the official ScanNet
# 2D GT zips (instance-filt + label-filt) for scene0000_00..scene0199_00,
# convert each scene into the SAM3 mask layout, and delete the zips per scene
# (peak zip disk ~1 scene, ~130 MB). Fully resumable: re-run to heal failures
# (existing zips are kept, scenes with .complete markers are skipped).
#
# Prereq (done once): the SAM3 subset dirs extracted to
#   /cluster/scratch/niacobone/scannet_official_build/sam3_subsets/scans
# via: zstd -dc <sam3 tar> | tar -x --wildcards 'scans/*/raw_data/subset/*'
#
# Dual-use: `sbatch slurm/download_official_gt.sh [start] [end]` or
# `bash slurm/download_official_gt.sh [start] [end]` on a login node.
#
#SBATCH --job-name=official_gt_dl
#SBATCH --output=official_gt_dl_%j.log
#SBATCH --error=official_gt_dl_%j.err
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

START=${1:-0}
END=${2:-199}
REPO=/cluster/scratch/niacobone/vggt
BUILD=/cluster/scratch/niacobone/scannet_official_build

cd "$REPO"
myenv/bin/python scripts/download_2d_gt.py \
    --zips_dir "$BUILD/zips" \
    --convert_out "$BUILD/scans" \
    --subset_root "$BUILD/sam3_subsets/scans" \
    --start "$START" --end "$END"
