#!/bin/bash
#
# Score a FINISHED run on the 2D validation protocol — no training (docs/MASKDINO.md §6.6.1).
#
#   sbatch --export=ALL,CHECKPOINT=<run_dir>/checkpoint_best_bundle.pth slurm/eval_only_maskdino.sh
#
# Why this exists: metrics added after a run cannot be read off its `metrics.jsonl`, because the
# 2D numbers are computed inside the eval loop and never stored as an artefact. This stages the
# VAL tar only, caches the val features, runs one validation pass with `--eval_only`, and appends
# a single `eval_only` row to the run's own metrics.jsonl. The run's config.json is not touched.
#
# It is the way HOTA / AssA / DetA / IDF1 (§6.6.1) were put on checkpoints that predate them.
#
# Knobs (all via --export=ALL,VAR=...):
#   CHECKPOINT   REQUIRED — the checkpoint to score
#   VAL_LIST     val scene list (default: the official 312-scene val split)
#   DATA_TAR     tar(s) to stage (default: the val-312 official-GT tar)
#   EXTRA_ARGS   appended verbatim — MUST reproduce the run's protocol flags, e.g.
#                '--multi_frame --feature_mode bundle --num_frames 8'. The eval scores whatever
#                bundle geometry it is given, so a mismatch here silently changes the ruler.
#   DRY_RUN=1    echo the command and exit (run with: bash slurm/eval_only_maskdino.sh)
#
#SBATCH --job-name=maskdino_evalonly
#SBATCH --output=/cluster/scratch/niacobone/vggt/slurm/logs/evalonly_%j.log
#SBATCH --error=/cluster/scratch/niacobone/vggt/slurm/logs/evalonly_%j.err
#SBATCH --open-mode=append
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=10240
#SBATCH --tmp=40000
#SBATCH --gpus=rtx_4090:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=niacobone@student.ethz.ch

cd /cluster/scratch/niacobone/vggt
: "${CHECKPOINT:?export CHECKPOINT=<run_dir>/checkpoint_best_bundle.pth}"
VAL_LIST=${VAL_LIST:-data/splits/scannetv2_val.txt}
RUN_DIR=$(dirname "$CHECKPOINT")

if [ -z "${DRY_RUN:-}" ]; then
    module purge
    module load stack/2024-06 python/3.12.8 cuda/12.8.0 eth_proxy
    source myenv/bin/activate
    export PYTHONUNBUFFERED=1
    DS=/cluster/work/igp_psr/niacobone/distillation/dataset/scannet
    export DATA_TAR="${DATA_TAR:-$DS/scannet_official_gt_val312.tar.zst}"
    source slurm/stage_dataset.sh
fi
PYTHON=myenv/bin/python

# Only ids actually present in the staged tree (the tar's QA report says all 312 are).
if [ -d "${SCANNET_ROOT:-/nonexistent}" ]; then
    VAL=$(comm -12 <(grep -vE '^\s*$' "$VAL_LIST" | sort -u) <(ls "$SCANNET_ROOT" | sort) \
          | paste -sd, -)
else
    VAL=$(grep -vE '^\s*$' "$VAL_LIST" | sort -u | paste -sd, -)
fi
echo "[cfg] eval-only: $(tr ',' '\n' <<< "$VAL" | wc -l) val scenes, checkpoint $CHECKPOINT"

# --train_scenes is inert here: --eval_only never resolves or caches the train split.
CMD="$PYTHON scripts/train_maskdino.py --eval_only \
    --resume $CHECKPOINT --save_checkpoint $RUN_DIR/checkpoint.pth \
    --scans_root ${SCANNET_ROOT:-unset} --val_scenes <312 ids> \
    --num_frames 8 --batch_frames 8 --eval_batch_frames 8 \
    --num_queries 300 --enc_layers 6 --dec_layers 9 \
    --two_stage --dn seg --dn_num 100 --initialize_box_type bitmask \
    --cache_device cpu --cache_dtype float16 ${EXTRA_ARGS:-}"
if [ -n "${DRY_RUN:-}" ]; then
    echo "[dry-run] would exec: $CMD"
    exit 0
fi

$PYTHON scripts/train_maskdino.py --eval_only \
    --resume "$CHECKPOINT" --save_checkpoint "$RUN_DIR/checkpoint.pth" \
    --scans_root "$SCANNET_ROOT" \
    --val_scenes "$VAL" \
    --num_frames 8 --batch_frames 8 --eval_batch_frames 8 \
    --num_queries 300 --enc_layers 6 --dec_layers 9 \
    --two_stage --dn seg --dn_num 100 --initialize_box_type bitmask \
    --cache_device cpu --cache_dtype float16 \
    ${EXTRA_ARGS:-}   # last, so EXTRA_ARGS can override any flag above
