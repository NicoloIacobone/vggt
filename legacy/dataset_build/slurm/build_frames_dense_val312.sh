#!/bin/bash
#
# DENSE whole-scan frames for the official ScanNet v2 val-312 split (docs/todo.md 6k):
# every 20th raw frame straight out of the .sens files, i.e. **SegVGGT's own eval sampling**,
# from which `--num_frames 50` reproduces **FAST3DIS's** budget exactly. The official
# `scannet_frames_25k` export we score on today is every 100th frame (~17/scene), which is
# the last unmatched axis of the protocol comparison (docs/TRAINING_COMPARABILITY.md §6.3).
#
#   sbatch legacy/dataset_build/slurm/build_frames_dense_val312.sh          # the 16-task array
#   sbatch --export=ALL,PACK=1 legacy/dataset_build/slurm/pack_frames_dense.sh   # then this
#
# ~1.15 GB of .sens is streamed per scene and NOTHING but the kept frames touches disk
# (measured 68 MB/s from ETH, ~18 s/scene). The whole file must be read — a whole-scan sample
# needs the last frame, so `extract_sens_subset.py`'s early abort does not apply here.
#
# WHY THIS ONE WRITES TO SCRATCH and not to $TMPDIR (docs/DATASET.md §5.1 says node-local).
# That rule exists because the 1201-scene 2D-GT build materialises ~1.26 M files and blew the
# inode quota. This build is ~94 k files (312 scenes x ~100 frames x {jpg,png,txt}) against
# 211 k used of a 1.0 M soft quota, and an ARRAY cannot share a $TMPDIR across tasks anyway.
# It is resumable per scene (a `.complete` marker per scene dir), so a killed task re-runs free.
#
#SBATCH --job-name=frames_dense_val312
#SBATCH --output=/cluster/scratch/niacobone/vggt/slurm/logs/frames_dense_%A_%a.log
#SBATCH --error=/cluster/scratch/niacobone/vggt/slurm/logs/frames_dense_%A_%a.err
#SBATCH --open-mode=append
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4096
#SBATCH --array=0-15
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=niacobone@student.ethz.ch

# eth_proxy gives compute nodes outbound network access for the stream.
module load eth_proxy 2>/dev/null || module load stack/2024-06 eth_proxy 2>/dev/null || true
set -euo pipefail

REPO=/cluster/scratch/niacobone/vggt
OUT_ROOT=${OUT_ROOT:-/cluster/scratch/niacobone/scannet_frames_dense/scans25k}
LIST=$REPO/data/splits/scannetv2_val.txt
STRIDE=${STRIDE:-20}
MAX_FRAMES=${MAX_FRAMES:-150}

cd "$REPO"
N=$(grep -cvE '^\s*$' "$LIST")
NTASK=${SLURM_ARRAY_TASK_COUNT:-16}
ID=${SLURM_ARRAY_TASK_ID:-0}
# Contiguous blocks, ceil-divided, so the last task simply gets fewer scenes.
PER=$(( (N + NTASK - 1) / NTASK ))
START=$(( ID * PER ))
END=$(( START + PER - 1 ))
[ "$END" -ge "$N" ] && END=$(( N - 1 ))
if [ "$START" -ge "$N" ]; then echo "[dense] task $ID: nothing to do"; exit 0; fi
echo "[dense] task $ID/$NTASK -> scenes $START..$END of $N, stride $STRIDE"

mkdir -p "$OUT_ROOT"
myenv/bin/python legacy/dataset_build/scripts/extract_sens_frames25k.py \
    --out_root "$OUT_ROOT" --scene_list "$LIST" \
    --start "$START" --end "$END" --stride "$STRIDE" --max_frames "$MAX_FRAMES"
