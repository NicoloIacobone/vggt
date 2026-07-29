#!/bin/bash
#
# MaskDINO-on-VGGT trial, single frame (docs/MASKDINO.md).
# Submit from anywhere: sbatch slurm/train_maskdino.sh
#
# Knobs (all via --export=ALL,VAR=...):
#   N_SCENES     number of train scenes, scene0000..scene<N-1>  (default 50)
#   EXTRA_ARGS   appended verbatim to the python call (e.g. '--mask_upsample 2')
#   EXP_TAG      appended to the run directory name
#   DATA_TAR     which dataset tar to stage (default: 500-scene official GT)
#   VAL_SPLIT    'convention' (default, val = scenes 0080-0089) or 'official' (val = the official
#                ScanNet v2 val scenes inside our range; needs its own run, see below)
#
#SBATCH --job-name=maskdino_sf
#SBATCH --output=/cluster/scratch/niacobone/vggt/slurm/logs/maskdino_%j.log
#SBATCH --error=/cluster/scratch/niacobone/vggt/slurm/logs/maskdino_%j.err
#SBATCH --open-mode=append
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6144
#SBATCH --tmp=24000
#SBATCH --gpus=rtx_4090:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=niacobone@student.ethz.ch

module purge
module load stack/2024-06 python/3.12.8 cuda/12.8.0 eth_proxy
cd /cluster/scratch/niacobone/vggt
source myenv/bin/activate
PYTHON=myenv/bin/python
# Unbuffered: without this, stdout is block-buffered into the .log and the run looks frozen for
# the first ~10 minutes (feature caching) even though it is progressing normally.
export PYTHONUNBUFFERED=1

# Same dataset staging as every other run: one tar → node-local scratch → SCANNET_ROOT.
export DATA_TAR="${DATA_TAR:-/cluster/work/igp_psr/niacobone/distillation/dataset/scannet/scannet_official_gt_500.tar.zst}"
source slurm/stage_dataset.sh

N_SCENES="${N_SCENES:-50}"
if [ "${VAL_SPLIT:-convention}" = "official" ]; then
    # Comparability read-out (docs/RESULTS.md §1): val = the official ScanNet v2 val scenes that
    # exist in our 500-scene tar (*_00 only, id < N_SCENES), train = everything else in range.
    # This needs its OWN run — 77 of those 80 scenes sit inside the usual 0000–0489 train range,
    # so scoring an existing checkpoint on them would be scoring on training data.
    VAL=$(awk -F'_' -v n="$N_SCENES" '$2=="00" && substr($1,6)+0 < n' \
          slurm/../data/splits/scannetv2_val.txt | sort | paste -sd, -)
    TRAIN=$(seq -f "scene%04g_00" 0 $((N_SCENES - 1)) \
            | grep -v -F -x -f <(tr ',' '\n' <<< "$VAL") | paste -sd, -)
    echo "[cfg] OFFICIAL split: $(tr ',' '\n' <<< "$VAL" | wc -l) val scenes, \
$(tr ',' '\n' <<< "$TRAIN" | wc -l) train scenes"
else
    # Scene 0080–0089 are the held-out val scenes of every D4RT scaling run; skip them if the
    # train range would otherwise swallow them.
    TRAIN=$(seq -f "scene%04g_00" 0 $((N_SCENES - 1)) | grep -v -E "scene008[0-9]_00" | paste -sd, -)
    VAL=$(seq -f "scene%04g_00" 80 89 | paste -sd, -)
fi
OUT=/cluster/work/igp_psr/niacobone/distillation/output
RUN=$OUT/maskdino_sf_n${N_SCENES}${EXP_TAG:-}_$(date +%Y%m%d_%H%M%S)

# One step = one batch of 8 frames, one epoch = every training frame once → steps/epoch ≈
# N_SCENES. Hold the TOTAL gradient-step budget roughly constant (~20k) across scene counts,
# so the comparison across N is about data, not about training length.
# An explicitly exported EPOCHS is honoured as-is — the clamps below exist only to keep the
# AUTO-derived schedule sane, and used to silently override e.g. EPOCHS=30 (needed to hold the
# step budget when --bundles_per_scene multiplies the steps per epoch) back up to 60.
if [ -z "${EPOCHS:-}" ]; then
    EPOCHS=$(( 20000 / N_SCENES ))
    [ "$EPOCHS" -lt 60 ] && EPOCHS=60
    [ "$EPOCHS" -gt 400 ] && EPOCHS=400
fi
WARMUP=$(( EPOCHS / 20 )); [ "$WARMUP" -lt 5 ] && WARMUP=5
EVAL_EVERY=$(( EPOCHS / 40 )); [ "$EVAL_EVERY" -lt 1 ] && EVAL_EVERY=1
echo "[cfg] scenes=$N_SCENES epochs=$EPOCHS warmup=$WARMUP eval_every=$EVAL_EVERY"

# MaskDINO's COCO instance recipe (300 queries, 9 decoder / 6 encoder layers, two-stage, DN
# "seg", mask-enhanced box init), single-frame, on the 37x37 patch grid so the mask metrics sit
# on the same grid as the D4RT arms. float16 feature cache keeps 500 scenes inside host RAM.
$PYTHON scripts/train_maskdino.py \
    --scans_root $SCANNET_ROOT \
    --train_scenes $TRAIN --val_scenes $VAL \
    --num_frames 8 --batch_frames 8 --eval_batch_frames 8 \
    --num_queries 300 --enc_layers 6 --dec_layers 9 \
    --two_stage --dn seg --dn_num 100 --initialize_box_type bitmask \
    --num_epochs $EPOCHS --warmup_epochs $WARMUP --learning_rate 1e-4 --weight_decay 0.05 \
    --eval_interval $EVAL_EVERY --log_interval $EVAL_EVERY \
    --cache_device cpu --cache_dtype float16 \
    --save_checkpoint $RUN/checkpoint.pth \
    ${EXTRA_ARGS:-}   # last, so EXTRA_ARGS can override any flag above
