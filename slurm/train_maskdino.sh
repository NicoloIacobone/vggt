#!/bin/bash
#
# MaskDINO-on-VGGT trial, single frame (docs/MASKDINO.md).
# Submit from anywhere: sbatch slurm/train_maskdino.sh
#
# Knobs (all via --export=ALL,VAR=...):
#   N_SCENES     number of train scenes, scene0000..scene<N-1>  (default 50)
#   EXTRA_ARGS   appended verbatim to the python call (e.g. '--mask_upsample 2')
#   EXP_TAG      appended to the run directory name
#   DATA_TAR     dataset tar(s) to stage — a single path (default: 500-scene official GT) or a
#                whitespace-separated list, all unpacked into one scans/ tree
#   VAL_SPLIT    'convention' (default, val = scenes 0080-0089) or 'official' (val = the official
#                ScanNet v2 val scenes inside our range; needs its own run, see below)
#   TRAIN_LIST / VAL_LIST
#                paths to explicit scene-list files (one scan id per line, e.g.
#                data/splits/scannetv2_train.txt). Set BOTH to use them; they override
#                N_SCENES/VAL_SPLIT scene selection entirely. Ids missing from the staged tree
#                are dropped with a loud warning. The full official split needs bigger
#                resources than this header requests — pass them on the sbatch command line:
#                sbatch --time=24:00:00 --cpus-per-task=12 --mem-per-cpu=14336 --tmp=90000 ...
#   EPOCHS / WARMUP
#                override the auto-derived schedule (see the budget block below)
#   DRY_RUN=1    derive and echo the scene lists, schedule and python command, then exit
#                without staging data or training (run with: bash slurm/train_maskdino.sh)
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

cd /cluster/scratch/niacobone/vggt
if [ -z "${DRY_RUN:-}" ]; then
    module purge
    module load stack/2024-06 python/3.12.8 cuda/12.8.0 eth_proxy
    source myenv/bin/activate
    # Unbuffered: without this, stdout is block-buffered into the .log and the run looks frozen
    # for the first ~10 minutes (feature caching) even though it is progressing normally.
    export PYTHONUNBUFFERED=1
    # Same dataset staging as every other run: tar(s) → node-local scratch → SCANNET_ROOT.
    export DATA_TAR="${DATA_TAR:-/cluster/work/igp_psr/niacobone/distillation/dataset/scannet/scannet_official_gt_500.tar.zst}"
    source slurm/stage_dataset.sh
fi
PYTHON=myenv/bin/python

# Turn a split file (one scan id per line) into the comma-joined list the trainer takes,
# dropping ids not present in the staged tree (each tar's QA report says none should be).
scene_list_from_file() {
    local ids; ids=$(grep -vE '^\s*$' "$1" | sort -u)
    if [ -d "${SCANNET_ROOT:-/nonexistent}" ]; then
        local on_disk missing
        on_disk=$(ls "$SCANNET_ROOT" | sort)
        missing=$(comm -23 <(echo "$ids") <(echo "$on_disk"))
        if [ -n "$missing" ]; then
            echo "[cfg] WARNING: $(echo "$missing" | wc -l) scenes from $1 not in" \
                 "$SCANNET_ROOT, dropped: $(echo "$missing" | head -3 | paste -sd' ' -) ..." >&2
        fi
        ids=$(comm -12 <(echo "$ids") <(echo "$on_disk"))
    fi
    echo "$ids" | paste -sd, -
}

N_SCENES="${N_SCENES:-50}"
if [ -n "${TRAIN_LIST:-}${VAL_LIST:-}" ]; then
    # Explicit split files (the official 1201/312 protocol). Both must be set.
    : "${TRAIN_LIST:?set TRAIN_LIST and VAL_LIST together}"
    : "${VAL_LIST:?set TRAIN_LIST and VAL_LIST together}"
    TRAIN=$(scene_list_from_file "$TRAIN_LIST")
    VAL=$(scene_list_from_file "$VAL_LIST")
    OVERLAP=$(comm -12 <(tr ',' '\n' <<< "$TRAIN") <(tr ',' '\n' <<< "$VAL"))
    [ -n "$OVERLAP" ] && { echo "[cfg] ERROR: train/val lists share scenes: $OVERLAP" >&2; exit 1; }
    N_TRAIN=$(tr ',' '\n' <<< "$TRAIN" | wc -l)
    echo "[cfg] LIST split: $N_TRAIN train ($TRAIN_LIST), \
$(tr ',' '\n' <<< "$VAL" | wc -l) val ($VAL_LIST)"
elif [ "${VAL_SPLIT:-convention}" = "official" ]; then
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
if [ -n "${TRAIN_LIST:-}" ]; then
    RUN=$OUT/maskdino_sf_list${N_TRAIN}${EXP_TAG:-}_$(date +%Y%m%d_%H%M%S)
else
    RUN=$OUT/maskdino_sf_n${N_SCENES}${EXP_TAG:-}_$(date +%Y%m%d_%H%M%S)
fi

# One step = one batch of 8 frames, one epoch = every training frame once → steps/epoch ≈
# N_SCENES. Hold the TOTAL gradient-step budget roughly constant (~20k) across scene counts,
# so the comparison across N is about data, not about training length.
# An explicitly exported EPOCHS is honoured as-is — the clamps below exist only to keep the
# AUTO-derived schedule sane, and used to silently override e.g. EPOCHS=30 (needed to hold the
# step budget when --bundles_per_scene multiplies the steps per epoch) back up to 60.
# In TRAIN_LIST mode the min-60 clamp is dropped: at 1201 scenes it would multiply the step
# budget by ~4, not protect it.
N_BUDGET=${N_TRAIN:-$N_SCENES}
if [ -z "${EPOCHS:-}" ]; then
    EPOCHS=$(( 20000 / N_BUDGET ))
    EPOCHS_MIN=60; [ -n "${TRAIN_LIST:-}" ] && EPOCHS_MIN=1
    [ "$EPOCHS" -lt "$EPOCHS_MIN" ] && EPOCHS=$EPOCHS_MIN
    [ "$EPOCHS" -gt 400 ] && EPOCHS=400
fi
if [ -z "${WARMUP:-}" ]; then
    WARMUP=$(( EPOCHS / 20 )); [ "$WARMUP" -lt 5 ] && WARMUP=5
fi
EVAL_EVERY=$(( EPOCHS / 40 )); [ "$EVAL_EVERY" -lt 1 ] && EVAL_EVERY=1
echo "[cfg] scenes=$N_BUDGET epochs=$EPOCHS warmup=$WARMUP eval_every=$EVAL_EVERY \
(steps/epoch ~= scenes x bundles_per_scene; budget ~= that x epochs)"

if [ -n "${DRY_RUN:-}" ]; then
    echo "[dry-run] TRAIN ($(tr ',' '\n' <<< "$TRAIN" | wc -l)): $TRAIN"
    echo "[dry-run] VAL ($(tr ',' '\n' <<< "$VAL" | wc -l)): $VAL"
    echo "[dry-run] RUN=$RUN"
    echo "[dry-run] EXTRA_ARGS=${EXTRA_ARGS:-}"
    echo "[dry-run] would exec: $PYTHON scripts/train_maskdino.py --num_epochs $EPOCHS" \
         "--warmup_epochs $WARMUP --eval_interval $EVAL_EVERY ... ${EXTRA_ARGS:-}"
    exit 0
fi

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
