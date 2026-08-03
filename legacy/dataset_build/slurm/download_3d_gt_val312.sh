#!/bin/bash
# 3D benchmark GT for the official ScanNet v2 VAL split (docs/todo.md 1d): per scene the
# benchmark mesh (_vh_clean_2.ply), its superpoint segs json and the aggregation json —
# what the 3D instance evaluation (scripts/eval_3d_maskdino.py) scores against.
#
# NODE-LOCAL (docs/DATASET.md §5.1): the ~3.5 GB tree (936 files) is built in $TMPDIR and
# only scannet_3d_gt_val312.tar.zst lands on work. Unlike the 2D GT builds there is no
# cross-job chunk-tar resume: a full re-download is ~20 min, cheaper than the machinery.
# Within a run the downloader skips scenes whose .complete marker exists.
#
#   sbatch legacy/dataset_build/slurm/download_3d_gt_val312.sh
#
#SBATCH --job-name=download_3d_gt_val312
#SBATCH --output=/cluster/scratch/niacobone/vggt/slurm/logs/download_3d_gt_val312_%j.log
#SBATCH --error=/cluster/scratch/niacobone/vggt/slurm/logs/download_3d_gt_val312_%j.err
#SBATCH --open-mode=append
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4096
#SBATCH --tmp=12000
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=niacobone@student.ethz.ch

# eth_proxy gives compute nodes outbound network access for the download.
module load eth_proxy 2>/dev/null || module load stack/2024-06 eth_proxy 2>/dev/null || true
set -euo pipefail

REPO=/cluster/scratch/niacobone/vggt
WORK=/cluster/work/igp_psr/niacobone/distillation/dataset/scannet
OUT_TAR=${OUT_TAR:-scannet_3d_gt_val312.tar.zst}
WORK_TAR="$WORK/$OUT_TAR"
EXPECT_SCENES=312

BUILD=${TMPDIR:?TMPDIR unset — this job needs node-local scratch via #SBATCH --tmp}/build
mkdir -p "$BUILD/scans3d"
cd "$REPO"

myenv/bin/python legacy/dataset_build/scripts/download_3d_gt.py \
    --out_root "$BUILD/scans3d" \
    --scene_list "$REPO/data/splits/scannetv2_val.txt" --start 0 --end 311

n_scenes=$(find "$BUILD/scans3d" -mindepth 2 -name '.complete' | wc -l)
echo "[3dgt] complete scenes: $n_scenes / $EXPECT_SCENES"
if [ "$n_scenes" != "$EXPECT_SCENES" ]; then
    echo "[3dgt] INCOMPLETE — not packing (re-submit to retry the failed scenes)"
    exit 1
fi

echo "[3dgt] building archive ..."
STAGE_TAR=$TMPDIR/$OUT_TAR
tar --use-compress-program="zstd -3 -T0" -C "$BUILD" -cf "$STAGE_TAR" scans3d

n_tar=$(tar --use-compress-program="zstd -d" -tf "$STAGE_TAR" | grep -c '\.ply$\|\.json$' || true)
n_src=$(find "$BUILD/scans3d" \( -name '*.ply' -o -name '*.json' \) | wc -l)
echo "[3dgt] entries: archive=$n_tar source=$n_src, size $(du -h "$STAGE_TAR" | cut -f1)"
if [ "$n_tar" != "$n_src" ]; then
    echo "[3dgt] COUNT MISMATCH — aborting before copy to work"
    exit 1
fi

# work is a shared group filesystem and runs close to full — fail loudly rather than
# leaving a truncated tar where an eval job would pick it up.
avail_kb=$(df -Pk "$WORK" | awk 'NR==2 {print $4}')
need_kb=$(( $(stat -c %s "$STAGE_TAR") / 1024 + 2097152 ))
if [ "$avail_kb" -lt "$need_kb" ]; then
    echo "[3dgt] NOT ENOUGH SPACE on work: need ~$((need_kb/1024)) MB, have $((avail_kb/1024)) MB"
    exit 1
fi

cp "$STAGE_TAR" "$WORK_TAR.tmp" && mv "$WORK_TAR.tmp" "$WORK_TAR"
echo "[3dgt] done -> $WORK_TAR ($(du -h "$WORK_TAR" | cut -f1))"
