#!/bin/bash
#
# Multi-dataset training: ScanNet v2 + ScanNet++ + Infinigen, CLASS-AGNOSTIC (docs/todo.md 6e/6f).
#
#   sbatch slurm/train_maskdino_multi.sh                              # the full mixture
#   sbatch --export=ALL,SOURCES='scannet scannetpp' slurm/train_maskdino_multi.sh
#   sbatch --export=ALL,CAP_SCANNETPP=200,CAP_INFINIGEN=200 slurm/train_maskdino_multi.sh
#   DRY_RUN=1 bash slurm/train_maskdino_multi.sh                      # print the lists and exit
#
# Knobs (via --export=ALL,VAR=...):
#   SOURCES         subset of "scannet scannetpp infinigen"   (default: all three)
#   CAP_<SOURCE>    keep at most N scenes of that source, deterministic (default: all)
#   CAP_VAL         keep at most N of the 312 val scenes — smoke runs only, it moves the ruler
#   EPOCHS/WARMUP   override the derived schedule
#   EXTRA_ARGS      appended verbatim to the python call
#   EXP_TAG         appended to the run directory name
#
# WHY CLASS-AGNOSTIC IS NOT OPTIONAL HERE. ScanNet++ (~84 classes) and Infinigen (procedural
# factories) do not share ScanNet's 19-class taxonomy, so there is no honest class-aware label
# for their instances. `--class_agnostic` builds a one-class head and collapses every GT label
# onto it (docs/todo.md 6e); `prepare_scenes` refuses a mixed scene list without it rather than
# silently supervising every object as ScanNet class 1.
#
# WHY VAL STAYS SCANNET-ONLY. The val ruler must not move when the training mixture changes, or
# "more data helped" and "the ruler got easier" become indistinguishable. Val is the official
# ScanNet v2 312-scene list, scored class-agnostic — so the honest baseline for this run is a
# ScanNet-ONLY run with --class_agnostic, not any published class-aware number
# (docs/RESULTS.md §1).
#
# MEMORY. The trainer caches frozen VGGT features for every scene up front (~45 MB per 8-frame
# bundle), so the cache is ~45 MB x scenes: ~54 GB for ScanNet's 1201 alone, ~160 GB for the full
# 3520-scene mixture. That is what the CPU/mem request below buys; use CAP_* to shrink it.
#
#SBATCH --job-name=maskdino_multi
#SBATCH --output=/cluster/scratch/niacobone/vggt/slurm/logs/maskdino_multi_%j.log
#SBATCH --error=/cluster/scratch/niacobone/vggt/slurm/logs/maskdino_multi_%j.err
#SBATCH --open-mode=append
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16384
#SBATCH --tmp=140000
#SBATCH --gpus=1
#SBATCH --gres=gpumem:40g
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=niacobone@student.ethz.ch

cd /cluster/scratch/niacobone/vggt
set -o pipefail

DATASET_ROOT=/cluster/work/igp_psr/niacobone/distillation/dataset
SOURCES="${SOURCES:-scannet scannetpp infinigen}"
STAGE="${TMPDIR:-/tmp}"

if [ -z "${DRY_RUN:-}" ]; then
    module purge
    module load stack/2024-06 python/3.12.8 cuda/12.8.0 eth_proxy
    source myenv/bin/activate
    export PYTHONUNBUFFERED=1

    # --- ScanNet: the usual staging path (train 1201 + val 312) ---------------------------
    if [[ " $SOURCES " == *" scannet "* ]]; then
        export DATA_TAR="$DATASET_ROOT/scannet/scannet_official_gt_1201.tar.zst \
$DATASET_ROOT/scannet/scannet_official_gt_val312.tar.zst"
        source slurm/stage_dataset.sh          # sets SCANNET_ROOT
    fi

    # --- the instance-map datasets: one tar each, unpacked node-local --------------------
    for SRC in $SOURCES; do
        [ "$SRC" = "scannet" ] && continue
        TAR="$DATASET_ROOT/insscene2d/insscene2d_${SRC}.tar.zst"
        [ -f "$TAR" ] || { echo "[cfg] ERROR: missing $TAR — run slurm/build_insscene2d.sh" >&2;
                           exit 1; }
        echo "[stage] unpacking $TAR"
        mkdir -p "$STAGE/insscene2d"
        zstd -dc "$TAR" | tar -C "$STAGE/insscene2d" -xf -
        echo "[stage] $SRC: $(ls "$STAGE/insscene2d/$SRC" | wc -l) scenes"
    done
fi
PYTHON=myenv/bin/python

# ---- scene lists ------------------------------------------------------------------------
# Every entry is an absolute path: `resolve_scene_dirs` takes paths as-is, and
# `build_scene_dataset` picks the loader per directory, so the three sources are one flat list.
cap_of() { eval "echo \${CAP_$(echo "$1" | tr '[:lower:]' '[:upper:]'):-0}"; }

TRAIN_PARTS=()
for SRC in $SOURCES; do
    CAP=$(cap_of "$SRC")
    if [ "$SRC" = "scannet" ]; then
        IDS=$(grep -vE '^\s*$' data/splits/scannetv2_train.txt | sort -u)
        [ -d "${SCANNET_ROOT:-/nonexistent}" ] && \
            IDS=$(comm -12 <(echo "$IDS") <(ls "$SCANNET_ROOT" | sort))
        [ "$CAP" -gt 0 ] && IDS=$(echo "$IDS" | head -n "$CAP")
        LIST=$(echo "$IDS" | sed "s|^|${SCANNET_ROOT}/|; s|$|/raw_data|")
    else
        LIST=$(find "$STAGE/insscene2d/$SRC" -mindepth 1 -maxdepth 1 -type d | sort)
        [ "$CAP" -gt 0 ] && LIST=$(echo "$LIST" | head -n "$CAP")
    fi
    N=$(echo "$LIST" | grep -c . || true)
    echo "[cfg] $SRC: $N train scenes"
    TRAIN_PARTS+=("$LIST")
done
TRAIN=$(printf '%s\n' "${TRAIN_PARTS[@]}" | grep -c . >/dev/null; \
        printf '%s\n' "${TRAIN_PARTS[@]}" | grep . | paste -sd, -)
N_TRAIN=$(tr ',' '\n' <<< "$TRAIN" | grep -c .)

# Val: the official ScanNet v2 312, unchanged across every mixture (see the header).
if [[ " $SOURCES " == *" scannet "* ]]; then
    VAL_IDS=$(grep -vE '^\s*$' data/splits/scannetv2_val.txt | sort -u)
    [ -d "${SCANNET_ROOT:-/nonexistent}" ] && \
        VAL_IDS=$(comm -12 <(echo "$VAL_IDS") <(ls "$SCANNET_ROOT" | sort))
    [ "${CAP_VAL:-0}" -gt 0 ] && VAL_IDS=$(echo "$VAL_IDS" | head -n "${CAP_VAL}")
    VAL=$(echo "$VAL_IDS" | sed "s|^|${SCANNET_ROOT}/|; s|$|/raw_data|" | paste -sd, -)
else
    VAL=""
fi
N_VAL=$(tr ',' '\n' <<< "$VAL" | grep -c . || true)
echo "[cfg] $N_TRAIN train scenes, $N_VAL val scenes (ScanNet official val, class-agnostic)"

# One step = one batch of 8 frames; hold the gradient-step budget near the 1201-scene runs'.
if [ -z "${EPOCHS:-}" ]; then
    EPOCHS=$(( 20000 / (N_TRAIN > 0 ? N_TRAIN : 1) ))
    [ "$EPOCHS" -lt 6 ] && EPOCHS=6
    [ "$EPOCHS" -gt 40 ] && EPOCHS=40
fi
[ -z "${WARMUP:-}" ] && { WARMUP=$(( EPOCHS / 20 )); [ "$WARMUP" -lt 2 ] && WARMUP=2; }
EVAL_EVERY=1
echo "[cfg] epochs=$EPOCHS warmup=$WARMUP eval_every=$EVAL_EVERY"

OUT=/cluster/work/igp_psr/niacobone/distillation/output
RUN=$OUT/maskdino_multi_$(echo "$SOURCES" | tr -d ' ')_n${N_TRAIN}${EXP_TAG:-}_$(date +%Y%m%d_%H%M%S)

if [ -n "${DRY_RUN:-}" ]; then
    echo "[dry-run] RUN=$RUN"
    echo "[dry-run] first train entries:"; tr ',' '\n' <<< "$TRAIN" | head -3
    echo "[dry-run] last train entries:";  tr ',' '\n' <<< "$TRAIN" | tail -3
    echo "[dry-run] first val entry:";     tr ',' '\n' <<< "$VAL" | head -1
    exit 0
fi

mkdir -p "$RUN"
$PYTHON scripts/train_maskdino.py \
    --scans_root "${SCANNET_ROOT:-$STAGE}" \
    --train_scenes "$TRAIN" --val_scenes "$VAL" \
    --class_agnostic \
    --multi_frame --feature_mode bundle \
    --num_frames 8 --batch_frames 8 --eval_batch_frames 8 \
    --num_queries 300 --enc_layers 6 --dec_layers 9 \
    --two_stage --dn seg --dn_num 100 --initialize_box_type bitmask \
    --num_epochs $EPOCHS --warmup_epochs $WARMUP --learning_rate 1e-4 --weight_decay 0.05 \
    --eval_interval $EVAL_EVERY --log_interval $EVAL_EVERY \
    --cache_device cpu --cache_dtype float16 \
    --save_checkpoint "$RUN/checkpoint.pth" \
    ${EXTRA_ARGS:-}   # last, so EXTRA_ARGS can override any flag above
