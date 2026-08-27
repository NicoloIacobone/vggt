#!/bin/bash
#
# Pack the dense val-312 frame export (built by build_frames_dense_val312.sh) into the one
# tar `slurm/eval_3d_maskdino.sh` stages via FRAMES_TAR (docs/todo.md 6k, docs/DATASET.md §2.5).
#
#   sbatch legacy/dataset_build/slurm/pack_frames_dense_val312.sh
#   sbatch --dependency=afterok:<array job> legacy/dataset_build/slurm/pack_frames_dense_val312.sh
#
# Refuses to pack an incomplete build: every scene of data/splits/scannetv2_val.txt must carry
# its `.complete` marker. Resume by re-running the array — it skips finished scenes.
#
#SBATCH --job-name=pack_frames_dense
#SBATCH --output=/cluster/scratch/niacobone/vggt/slurm/logs/pack_frames_dense_%j.log
#SBATCH --error=/cluster/scratch/niacobone/vggt/slurm/logs/pack_frames_dense_%j.err
#SBATCH --open-mode=append
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4096
#SBATCH --tmp=40000
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=niacobone@student.ethz.ch

set -euo pipefail
REPO=/cluster/scratch/niacobone/vggt
WORK=/cluster/work/igp_psr/niacobone/distillation/dataset/scannet
SRC=${SRC:-/cluster/scratch/niacobone/scannet_frames_dense}
OUT_TAR=${OUT_TAR:-scannet_frames_dense_val312.tar.zst}
LIST=$REPO/data/splits/scannetv2_val.txt
cd "$REPO"

# --- completeness gate ---------------------------------------------------------------------
missing=0
while read -r s; do
    [ -z "$s" ] && continue
    [ -f "$SRC/scans25k/$s/.complete" ] || { echo "[pack] INCOMPLETE: $s"; missing=$((missing+1)); }
done < "$LIST"
[ "$missing" -eq 0 ] || { echo "[pack] $missing scenes incomplete — re-run the array"; exit 1; }

n_scenes=$(find "$SRC/scans25k" -mindepth 1 -maxdepth 1 -type d | wc -l)
n_jpg=$(find "$SRC/scans25k" -name '*.jpg' | wc -l)
n_png=$(find "$SRC/scans25k" -name '*.png' | wc -l)
n_txt=$(find "$SRC/scans25k" -name '*.txt' | wc -l)
echo "[pack] $n_scenes scenes, $n_jpg color, $n_png depth, $n_txt txt"
[ "$n_jpg" -eq "$n_png" ] || { echo "[pack] color/depth count mismatch"; exit 1; }

STAGE_TAR=${TMPDIR:?}/$OUT_TAR
echo "[pack] building $STAGE_TAR ..."
tar --use-compress-program="zstd -3 -T0" -C "$SRC" --exclude='.complete' -cf "$STAGE_TAR" scans25k

n_tar=$(tar --use-compress-program="zstd -d" -tf "$STAGE_TAR" | grep -c '\.jpg$\|\.png$\|\.txt$')
n_src=$((n_jpg + n_png + n_txt))
echo "[pack] entries: archive=$n_tar source=$n_src, size $(du -h "$STAGE_TAR" | cut -f1)"
[ "$n_tar" -eq "$n_src" ] || { echo "[pack] entry count mismatch — NOT shipping"; exit 1; }

mkdir -p "$WORK"
cp "$STAGE_TAR" "$WORK/$OUT_TAR"
echo "[pack] done -> $WORK/$OUT_TAR"
echo "[pack] the scratch tree ($SRC) can now be deleted; it is ~$n_src files"
