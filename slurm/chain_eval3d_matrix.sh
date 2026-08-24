#!/bin/bash
#
# Chain the cross-dataset 3D matrix (docs/RESULTS.md §7) onto a training job that has not
# finished yet — the run directory carries a timestamp minted when that job STARTS, so it
# cannot be named at submit time.
#
#   sbatch --dependency=afterok:<train job> --export=ALL,TRAIN_JOB=<train job> \
#       slurm/chain_eval3d_matrix.sh
#
# Knobs: DATASETS / MODES pass through to slurm/eval_3d_matrix.sh (defaults: all 4 x both).
#        LOG_DIR   where to look for the training log     (default: slurm/logs)
#        DRY_RUN=1 resolve the run dir and print, submit nothing (this is what the test uses)
#
# Submitting from inside a job is the same pattern `CHAIN_PACK=1` uses in the dataset builds.
#
#SBATCH --job-name=chain_eval3d
#SBATCH --output=/cluster/scratch/niacobone/vggt/slurm/logs/chain_eval3d_%j.log
#SBATCH --error=/cluster/scratch/niacobone/vggt/slurm/logs/chain_eval3d_%j.err
#SBATCH --open-mode=append
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2048
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=niacobone@student.ethz.ch

set -uo pipefail
# NOT `cd "$(dirname "$0")/.."`. SLURM copies the batch script to its spool directory before
# running it, so inside a job $0 is `.../slurm_script` and that cd lands nowhere near the repo:
# jobs 11436321/23/24 died in 10 s with "no log for job ... under slurm/logs" while the DRY_RUN
# test — which runs from the repo — passed. Every other SLURM driver here hardcodes the path.
REPO=${REPO:-/cluster/scratch/niacobone/vggt}
cd "$REPO" || { echo "cannot cd to REPO=$REPO"; exit 1; }

: "${TRAIN_JOB:?export TRAIN_JOB=<the training job id>}"
LOG_DIR=${LOG_DIR:-slurm/logs}

# The trainer prints exactly one such line, before it does any work:
#   [cfg] scene lists written to <RUN>/{train,val}_scenes.txt
LOG=$(ls "$LOG_DIR"/*_"$TRAIN_JOB".log 2>/dev/null | head -1)
[ -n "$LOG" ] || { echo "no log for job $TRAIN_JOB under $LOG_DIR"; exit 1; }
RUN=$(sed -n 's|.*scene lists written to \(.*\)/{train,val}_scenes\.txt|\1|p' "$LOG" | head -1)
[ -n "$RUN" ] || { echo "could not read the run dir out of $LOG"; exit 1; }
echo "[chain] job $TRAIN_JOB -> $RUN"

CKPT=$RUN/checkpoint_best_bundle.pth
if [ ! -f "$CKPT" ] && [ "${DRY_RUN:-0}" != 1 ]; then
    # A --multi_frame run always writes it; its absence means the arm is not one, or died late.
    echo "missing $CKPT — nothing to score"; exit 1
fi

CKPTS="$RUN" DATASETS="${DATASETS:-scannetv2 scannet200 scannetpp replica}" \
    MODES="${MODES:-unproject gt_projection}" bash slurm/eval_3d_matrix.sh
