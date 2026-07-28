#!/bin/bash
# Phase 2 (packing) of docs/old/OFFICIAL_GT_MIGRATION_PLAN.md: run the QA gates +
# report, tar the official-GT build tree, verify counts, atomic-move to work.
# Modeled on sam3/scripts/pack_split2.sh. Never unpacks a tar to re-tar it.
#
# Submit after the download/convert job: sbatch legacy/dataset_build/slurm/pack_official_gt.sh
# (or --dependency=afterok:<download job id>). CPU-only.
# EXPECT_SCENES / OUT_TAR are env-overridable (defaults: the 500-scene build).
# The original 200-scene tar (scannet_official_gt_full.tar.zst) is kept as-is
# for reproducibility of the runs trained on it.
#
#SBATCH --job-name=pack_official_gt
#SBATCH --output=/cluster/scratch/niacobone/vggt/slurm/logs/pack_official_gt_%j.log
#SBATCH --error=/cluster/scratch/niacobone/vggt/slurm/logs/pack_official_gt_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4096
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=niacobone@student.ethz.ch

set -e
REPO=/cluster/scratch/niacobone/vggt
BUILD=/cluster/scratch/niacobone/scannet_official_build
WORK=/cluster/work/igp_psr/niacobone/distillation/dataset/scannet
EXPECT_SCENES=${EXPECT_SCENES:-500}
OUT_TAR=${OUT_TAR:-scannet_official_gt_500.tar.zst}
STAGE_TAR=/cluster/scratch/niacobone/$OUT_TAR
WORK_TAR="$WORK/$OUT_TAR"

cd "$REPO"

# QA gates (exits non-zero on failure -> job fails before packing) + README.
myenv/bin/python legacy/dataset_build/scripts/gen_official_gt_report.py \
    --build "$BUILD" --out "$WORK/OFFICIAL_GT_README.md" --expect_scenes "$EXPECT_SCENES"

# Visual spot-check strips (eyeballed separately; not a hard gate).
myenv/bin/python legacy/dataset_build/scripts/qa_official_gt_strips.py --build "$BUILD" \
    --scenes scene0000_00,scene0080_00,scene0160_00,scene0250_00,scene0340_00,scene0430_00,scene0499_00

echo "[pack] building archive ..."
tar --use-compress-program="zstd -1 -T0" -C "$BUILD" -cf "$STAGE_TAR" scans

echo "[pack] archive size: $(du -h "$STAGE_TAR" | cut -f1)"
n_tar=$(tar --use-compress-program="zstd -d" -tf "$STAGE_TAR" | grep -c '\.png$\|\.jpg$' || true)
n_src=$(find "$BUILD/scans" \( -name '*.png' -o -name '*.jpg' \) | wc -l)
echo "[pack] entries: archive=$n_tar source=$n_src"
if [ "$n_tar" != "$n_src" ]; then
    echo "[pack] COUNT MISMATCH — aborting before copy to work"
    exit 1
fi

cp "$STAGE_TAR" "$WORK_TAR.tmp" && mv "$WORK_TAR.tmp" "$WORK_TAR"
echo "[pack] done -> $WORK_TAR ($(du -h "$WORK_TAR" | cut -f1))"
