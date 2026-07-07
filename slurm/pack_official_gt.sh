#!/bin/bash
# Phase 2 (packing) of docs/OFFICIAL_GT_MIGRATION_PLAN.md: run the QA gates +
# report, tar the official-GT build tree, verify counts, atomic-move to work.
# Modeled on sam3/scripts/pack_split2.sh. Never unpacks a tar to re-tar it.
#
# Submit after the download/convert job: sbatch slurm/pack_official_gt.sh
# (or --dependency=afterok:<download job id>). CPU-only.
#
#SBATCH --job-name=pack_official_gt
#SBATCH --output=pack_official_gt_%j.log
#SBATCH --error=pack_official_gt_%j.err
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
STAGE_TAR=/cluster/scratch/niacobone/scannet_official_gt_full.tar.zst
WORK_TAR="$WORK/scannet_official_gt_full.tar.zst"

cd "$REPO"

# QA gates (exits non-zero on failure -> job fails before packing) + README.
myenv/bin/python scripts/gen_official_gt_report.py \
    --build "$BUILD" --out "$WORK/OFFICIAL_GT_README.md"

# Visual spot-check strips (eyeballed separately; not a hard gate).
myenv/bin/python scripts/qa_official_gt_strips.py --build "$BUILD"

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
