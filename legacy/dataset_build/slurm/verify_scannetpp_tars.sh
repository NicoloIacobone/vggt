#!/bin/bash
# Acceptance test for the SHIPPED ScanNet++ tars (docs/DATASET.md §2.1).
#
# build_scannetpp_val50.sh verifies the tree it just built, in $TMPDIR, before packing it.
# That is not the same thing as verifying the deliverable: the tar is what everything
# downstream actually reads, and between the two sit compression, a count check and a copy
# to `work`. This job closes that gap — it unpacks the tars from `work` and runs
# scripts/verify_scannetpp_gt.py against the unpacked result, touching nothing else.
#
# Also the right thing to run after a lost/restored tar, or before a paper number depends
# on this data.
#
# NODE-LOCAL: the ~7900-file tree lives in $TMPDIR. Scratch loose-file cost is ZERO —
# unpacking these tars anywhere on scratch would spend ~7900 inodes for nothing
# (docs/DATASET.md §5.1).
#
# Usage:
#   sbatch legacy/dataset_build/slurm/verify_scannetpp_tars.sh
#   sbatch --export=ALL,VERIFY_SCENES=0 legacy/dataset_build/slurm/verify_scannetpp_tars.sh  # all 49
#
#SBATCH --job-name=verify_scannetpp_tars
#SBATCH --output=/cluster/scratch/niacobone/vggt/slurm/logs/verify_scannetpp_tars_%j.log
#SBATCH --error=/cluster/scratch/niacobone/vggt/slurm/logs/verify_scannetpp_tars_%j.err
#SBATCH --open-mode=append
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4096
#SBATCH --tmp=20000

set -u

REPO=/cluster/scratch/niacobone/vggt
OUT=${OUT_DIR:-/cluster/work/igp_psr/niacobone/distillation/dataset/scannetpp}
WORK=${TMPDIR:?TMPDIR unset — this job needs node-local scratch via #SBATCH --tmp}/verify
mkdir -p "$WORK"
cd "$REPO"

for tar_name in scannetpp_3d_gt_val50 scannetpp_frames_val50; do
    src=$OUT/$tar_name.tar.zst
    [ -f "$src" ] || { echo "[verify] MISSING $src"; exit 1; }
    echo "[verify] unpacking $src ($(du -h "$src" | cut -f1)) ..."
    tar --use-compress-program="zstd -d" -C "$WORK" -xf "$src" || exit 1
done

# The two tars must describe the same scenes: a GT scene with no frames (or the reverse)
# is a silent hole in any evaluation that iterates one of them.
gt_list=$(find "$WORK/scans3d" -mindepth 1 -maxdepth 1 -type d ! -name '_*' -printf '%f\n' | sort)
fr_list=$(find "$WORK/scans25k" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
if [ "$gt_list" != "$fr_list" ]; then
    echo "[verify] SCENE LIST MISMATCH between the two tars:"
    diff <(echo "$gt_list") <(echo "$fr_list")
    exit 1
fi
echo "[verify] both tars hold the same $(echo "$gt_list" | grep -c .) scenes"

myenv/bin/python scripts/verify_scannetpp_gt.py \
    --gt_root "$WORK/scans3d" --frames_root "$WORK/scans25k" \
    --num_scenes "${VERIFY_SCENES:-5}"
RC=$?

echo "[verify] scratch quota (must be unchanged by this job):"
lfs quota -h -u "$USER" /cluster/scratch/niacobone 2>/dev/null | tail -2
echo "[verify] exit $RC"
exit $RC
