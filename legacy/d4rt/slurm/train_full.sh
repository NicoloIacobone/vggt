#!/bin/bash
#
# Scaling experiment: train on the WHOLE dataset (all 490 non-val scenes out of the
# 500-scene official-GT tar; val 0080–0089 held out, as in every scaling run).
# Submit: sbatch --export=ALL,INSTANCE_LEVEL=1 legacy/d4rt/slurm/train_full.sh
#
#SBATCH --job-name=d4rt_full
#SBATCH --output=/cluster/scratch/niacobone/vggt/slurm/logs/train_full_%j.log
#SBATCH --error=/cluster/scratch/niacobone/vggt/slurm/logs/train_full_%j.err
#SBATCH --open-mode=append
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=17500
#SBATCH --tmp=24000
#SBATCH --gpus=rtx_4090:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=niacobone@student.ethz.ch

module purge
module load stack/2024-06 python/3.12.8 cuda/12.8.0 eth_proxy
cd /cluster/scratch/niacobone/vggt
source myenv/bin/activate
PYTHON=myenv/bin/python
# Unbuffered stdout: without this, prints are lost if the job is SIGTERM'd by the time
# limit (job 6442237 hit exactly this — the log had nothing but the [stage] bash echoes,
# so a 12h TIMEOUT gave zero per-epoch timing to diagnose why). Cheap, always keep it on.
export PYTHONUNBUFFERED=1

# Stage the full dataset onto node-local scratch and export SCANNET_ROOT.
# GT tar: official ScanNet GT by default (docs/old/OFFICIAL_GT_MIGRATION_PLAN.md). For the
# SAM3-GT baseline: sbatch --export=ALL,DATA_TAR=/cluster/work/igp_psr/niacobone/distillation/dataset/scannet/scannet_instance_dataset_full.tar.zst ...
export DATA_TAR="${DATA_TAR:-/cluster/work/igp_psr/niacobone/distillation/dataset/scannet/scannet_official_gt_500.tar.zst}"
source slurm/stage_dataset.sh

# Held-out val set (scene0080–0089), excluded from train so every scaling point is comparable.
VAL=$(seq -f "scene%04g_00" 80 89 | paste -sd, -)
OUT=/cluster/work/igp_psr/niacobone/distillation/output

# Optional per-instance GT: submit with `sbatch --export=ALL,INSTANCE_LEVEL=1 ...`.
INSTANCE_FLAG=""; RUN_TAG=""
if [ "${INSTANCE_LEVEL:-0}" = "1" ]; then INSTANCE_FLAG="--instance_level"; RUN_TAG="_inst"; fi
RUN_TAG="${RUN_TAG}${EXP_TAG:-}"

# Train pool = all scenes EXCEPT the held-out val 0080–0089 → 0000–0079 + 0090–0499 (490 scenes).
# SANITY200=1 reverts to the ORIGINAL 190-scene/200-scene-tar pool (the exact recipe that
# produced the 0.367/0.199 baseline) for an A/B sanity check against the 500-scene point.
# Deliberately a plain flag, NOT a comma-bearing scene list passed through EXTRA_ARGS/--export
# — sbatch's --export splits on every comma regardless of quoting, so a --train_scenes
# override with 190 comma-separated scenes silently truncates to just the first one
# (job 6962015: "Train scenes (1): ['scene0000_00']", not a real sanity check).
if [ "${SANITY200:-0}" = "1" ]; then
    TRAIN=$( (seq -f "scene%04g_00" 0 79; seq -f "scene%04g_00" 90 199) | paste -sd, - )
else
    TRAIN=$( (seq -f "scene%04g_00" 0 79; seq -f "scene%04g_00" 90 499) | paste -sd, - )
fi

# 490×3 cached bundles + 10×1 val ≈ 1480 bundles → ~310 GB host RAM cache (--cache_device cpu),
# scaled from the 190-scene measurement (scale100: 62.7 GB / 300 bundles ≈ 0.21 GB/bundle).
# 20×17500=350 GB requested, headroom under the rtx_4090 nodes' ~375 GB. Self-contained
# checkpoints (default uint8 images, NOT --checkpoint_light) so visualization needs no scans tree.
# --time bumped 4h→12h: ~2.6x more train bundles/epoch than the 190-scene run that already
# used ~4h (bests peaked ~ep450-500 there; watch metrics.jsonl if this run needs the same bump).
$PYTHON legacy/d4rt/scripts/train_multiscene.py \
    --scans_root $SCANNET_ROOT $INSTANCE_FLAG \
    --train_scenes $TRAIN \
    --val_scenes $VAL \
    --num_epochs 1000 --warmup_epochs 30 --num_frames 8 --num_queries 32 \
    --learning_rate 2e-3 --bundles_per_scene 3 --query_jitter 0.02 --color_jitter 0.2 \
    --no_object_weight 0.1 --grid_size 6 --eval_interval 50 --early_stop_patience 0 \
    --cache_device cpu --no_visualize \
    --save_checkpoint $OUT/d4rt_full${RUN_TAG}_$(date +%Y%m%d_%H%M%S)/checkpoint.pth \
    ${EXTRA_ARGS:-}
# EXTRA_ARGS placed LAST so it can override any of the hardcoded defaults above (argparse
# keeps the last occurrence of a repeated flag) — needed e.g. for a short diagnostic run
# via EXTRA_ARGS="--num_epochs 20 --eval_interval 5".

# NOTE: --no_visualize keeps the job inside the walltime — on the 190-scene set, training +
# staging already took ~3h45 (more at --mask_upsample 2) and the end-of-run auto-render
# (200 scenes) previously pushed the job past its 4h limit (job 5275027 TIMEOUT at
# 20:15 — training had finished at 19:58, only the viz was truncated); on the 490-scene set
# the render would be ~2.5x bigger still. Render overlays separately afterward on any GPU node:
#   myenv/bin/python legacy/d4rt/scripts/visualize_masks.py --checkpoint $OUT/<run>/checkpoint_best_ap50.pth
