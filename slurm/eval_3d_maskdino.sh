#!/bin/bash
#
# The 3D ruler (docs/MASKDINO.md §9): official ScanNet 3D instance benchmark eval of a
# --multi_frame MaskDINO checkpoint on the official val-312 split.
#
#   sbatch --export=ALL,CHECKPOINT=<run_dir>/checkpoint_best_bundle.pth slurm/eval_3d_maskdino.sh
#
# Knobs (all via --export=ALL,VAR=...):
#   CHECKPOINT   REQUIRED — the checkpoint to score (a multi-frame one; per-frame checkpoints
#                produce meaningless 3D instances, the script warns)
#   EXTRA_ARGS   appended verbatim (e.g. '--num_frames 8', '--no-icp',
#                '--depth_conf_percentile 25', '--dump_ply')
#
# Stages TWO tars node-local: the val-312 3D GT (mesh + superpoints + aggregation) and the
# val-312 scannet_frames_25k repack (whole-scan frames + poses). Results land next to the
# checkpoint as eval3d_<ckpt stem>.json.
#
# NEVER quote the resulting numbers next to the 2D-protocol tables (docs/RESULTS.md §1) —
# and remember checkpoints trained on scenes 0000-0489 overlap this val split: their
# numbers are DIAGNOSTIC only (docs/MASKDINO.md §9.4).
#
#SBATCH --job-name=eval3d_maskdino
#SBATCH --output=/cluster/scratch/niacobone/vggt/slurm/logs/eval3d_%j.log
#SBATCH --error=/cluster/scratch/niacobone/vggt/slurm/logs/eval3d_%j.err
#SBATCH --open-mode=append
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6144
#SBATCH --tmp=16000
#SBATCH --gpus=rtx_4090:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=niacobone@student.ethz.ch

module purge
module load stack/2024-06 python/3.12.8 cuda/12.8.0 eth_proxy
cd /cluster/scratch/niacobone/vggt
source myenv/bin/activate
PYTHON=myenv/bin/python
export PYTHONUNBUFFERED=1

: "${CHECKPOINT:?export CHECKPOINT=<run_dir>/checkpoint_best_bundle.pth}"

WORK=/cluster/work/igp_psr/niacobone/distillation/dataset/scannet
GT_TAR=${GT_TAR:-$WORK/scannet_3d_gt_val312.tar.zst}
FRAMES_TAR=${FRAMES_TAR:-$WORK/scannet_frames25k_val312.tar.zst}

STAGE=${TMPDIR:?TMPDIR unset — this job needs node-local scratch via #SBATCH --tmp}
for t in "$GT_TAR" "$FRAMES_TAR"; do
    [ -f "$t" ] || { echo "missing tar: $t (run the legacy/dataset_build download jobs)"; exit 1; }
    echo "[stage] $(basename "$t") -> $STAGE"
    tar --use-compress-program="zstd -d -T0" -C "$STAGE" -xf "$t"
done

$PYTHON scripts/eval_3d_maskdino.py \
    --checkpoint "$CHECKPOINT" \
    --gt_root "$STAGE/scans3d" \
    --frames_root "$STAGE/scans25k" \
    ${EXTRA_ARGS:-}
