#!/bin/bash
# scannet_frames_25k for the official ScanNet v2 VAL split (docs/todo.md 1d): the official
# whole-scan frame export (~16 frames/scene, per-frame pose + intrinsics), the input frames
# of the 3D benchmark eval (scripts/eval_3d_maskdino.py). Our stride-5 subset tars cover
# only raw frames 0-495 and carry no poses — this export fixes both.
#
# The 6.0 GB zip (verified live on v2/tasks, 2026-08-01; v1/tasks is a 404) is downloaded
# ONCE onto scratch with curl -C - (1 inode; a killed job resumes the byte range instead of
# restarting). The val-312 scenes are then repacked NODE-LOCAL (docs/DATASET.md §5.1) and
# only scannet_frames25k_val312.tar.zst lands on work. Delete the zip once the tar exists.
#
#   sbatch legacy/dataset_build/slurm/download_frames25k_val312.sh
#
#SBATCH --job-name=frames25k_val312
#SBATCH --output=/cluster/scratch/niacobone/vggt/slurm/logs/frames25k_val312_%j.log
#SBATCH --error=/cluster/scratch/niacobone/vggt/slurm/logs/frames25k_val312_%j.err
#SBATCH --open-mode=append
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4096
#SBATCH --tmp=16000
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=niacobone@student.ethz.ch

# eth_proxy gives compute nodes outbound network access for the download.
module load eth_proxy 2>/dev/null || module load stack/2024-06 eth_proxy 2>/dev/null || true
set -euo pipefail

REPO=/cluster/scratch/niacobone/vggt
WORK=/cluster/work/igp_psr/niacobone/distillation/dataset/scannet
URL="http://kaldir.vc.cit.tum.de/scannet/v2/tasks/scannet_frames_25k.zip"
ZIP=/cluster/scratch/niacobone/scannet_frames_25k.zip
OUT_TAR=${OUT_TAR:-scannet_frames25k_val312.tar.zst}
WORK_TAR="$WORK/$OUT_TAR"

BUILD=${TMPDIR:?TMPDIR unset — this job needs node-local scratch via #SBATCH --tmp}/build
mkdir -p "$BUILD/scans25k"
cd "$REPO"

echo "[25k] downloading (resumes if partial) ..."
curl -sSL -C - --retry 8 --retry-delay 15 -o "$ZIP" "$URL"
myenv/bin/python - "$ZIP" <<'EOF'
import sys, zipfile
zipfile.ZipFile(sys.argv[1]).testzip() and sys.exit("corrupt zip")
print("[25k] zip integrity OK")
EOF

myenv/bin/python legacy/dataset_build/scripts/repack_frames25k.py \
    --zip "$ZIP" --out_root "$BUILD/scans25k" \
    --scene_list "$REPO/data/splits/scannetv2_val.txt" --start 0 --end 311

echo "[25k] building archive ..."
STAGE_TAR=$TMPDIR/$OUT_TAR
tar --use-compress-program="zstd -3 -T0" -C "$BUILD" -cf "$STAGE_TAR" scans25k

n_tar=$(tar --use-compress-program="zstd -d" -tf "$STAGE_TAR" \
        | grep -c '\.jpg$\|\.png$\|\.txt$' || true)
n_src=$(find "$BUILD/scans25k" \( -name '*.jpg' -o -name '*.png' -o -name '*.txt' \) | wc -l)
echo "[25k] entries: archive=$n_tar source=$n_src, size $(du -h "$STAGE_TAR" | cut -f1)"
if [ "$n_tar" != "$n_src" ]; then
    echo "[25k] COUNT MISMATCH — aborting before copy to work"
    exit 1
fi

# work is a shared group filesystem and runs close to full — fail loudly rather than
# leaving a truncated tar where an eval job would pick it up.
avail_kb=$(df -Pk "$WORK" | awk 'NR==2 {print $4}')
need_kb=$(( $(stat -c %s "$STAGE_TAR") / 1024 + 2097152 ))
if [ "$avail_kb" -lt "$need_kb" ]; then
    echo "[25k] NOT ENOUGH SPACE on work: need ~$((need_kb/1024)) MB, have $((avail_kb/1024)) MB"
    exit 1
fi

cp "$STAGE_TAR" "$WORK_TAR.tmp" && mv "$WORK_TAR.tmp" "$WORK_TAR"
echo "[25k] done -> $WORK_TAR ($(du -h "$WORK_TAR" | cut -f1))"
echo "[25k] $ZIP is now redundant — delete it to reclaim scratch blocks."
