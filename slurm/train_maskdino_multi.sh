#!/bin/bash
#
# Multi-dataset training: ScanNet v2 + ScanNet++ + Infinigen (+ RE10K), CLASS-AGNOSTIC
# (docs/todo.md 6e/6f/6j).
#
#   sbatch slurm/train_maskdino_multi.sh                              # the full mixture
#   sbatch --export=ALL,SOURCES='scannet scannetpp' slurm/train_maskdino_multi.sh
#   sbatch --export=ALL,CAP_SCANNETPP=200,CAP_INFINIGEN=200 slurm/train_maskdino_multi.sh
#   sbatch --export=ALL,SOURCES='scannet scannetpp infinigen re10k',CAP_RE10K=1500,EPOCHS=17,\
#       EXTRA_ARGS='--anchor_3d --learning_rate 5e-5' --cpus-per-task=26 \
#       slurm/train_maskdino_multi.sh          # the SAM2-supervised arm -- 5e-5 IS load-bearing
#   DRY_RUN=1 bash slurm/train_maskdino_multi.sh                      # print the lists and exit
#
# Knobs (via --export=ALL,VAR=...):
#   SOURCES         subset of "scannet scannetpp infinigen re10k"   (default: the first three —
#                   re10k's masks are SAM2 output, not GT, so it is opt-in and its rows carry
#                   that caveat: docs/MULTIDATASET.md §1.3)
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
# MEMORY -- the binding constraint, and it is NOT the GPU. The trainer caches frozen VGGT
# features plus GT for every scene up front. MEASURED, not projected: 135 GB peak RSS for 1513
# scenes (the ScanNet-only arm) and 258 GiB for 3832 (arm A-long, job 11498642), i.e. ~69 MiB per
# cached bundle. The default 16 x 16 GB = 256 GB is sized for the 1201-scene arm; OVERRIDE IT AT
# SUBMIT TIME rather than editing this file:
#     --cpus-per-task=20 (320 GB)  ScanNet + ScanNet++      ~2054 + 312 scenes
#     --cpus-per-task=26 (416 GB)  the three-source mixture ~3520 + 312, and the CAP_RE10K=1500
#                                  four-source arm at ~5020 + 312 (~360 GB, ~20 % headroom)
# All four sources UNCAPPED is 8647 train scenes ~= 640 GB: ~40-44 CPUs, and it may not schedule.
#
# SCHEDULE. `EPOCHS` defaults to 20000/N_TRAIN clamped to [6, 40], which at a 5000-scene mixture
# is the FLOOR of 6 and badly under-budgets it. Set EPOCHS explicitly for any large mixture: this
# workstream read "more data hurts" off an under-budgeted run twice (docs/MULTIDATASET.md §9
# reading 1, §10.3 reading 2). One step = one 8-frame bundle = one scene at b1, so
# steps = N_TRAIN x EPOCHS; A-long's budget is 84 480.
#
# LEARNING RATE -- the third thing a big mixture can need, learned the expensive way. The default
# 1e-4 is stable up to the 3520-scene three-source mixture and DIVERGES on the 5020-scene
# four-source one: job 11642516 ran 17 h clean and produced garbage, its training loss RISING from
# epoch 3 (the first epoch after warmup) and `train_AP50` collapsing 0.211 -> 0.006. Halving to
# 5e-5 removes it entirely at the same data and the same dose (docs/MULTIDATASET.md §11.3).
# So for any mixture larger or denser than A-long's, pass `--learning_rate 5e-5` in EXTRA_ARGS --
# and re-run the CONTROL at the same LR, or the comparison moves two variables at once.
# The tell, if it happens again: the epoch-1/2 curve looks normal and the run turns over exactly
# when warmup ends. Watch `loss` and `train_AP50`, not just val.
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
# `build_scene_dataset` picks the loader per directory, so all the sources are one flat list.
cap_of() { eval "echo \${CAP_$(echo "$1" | tr '[:lower:]' '[:upper:]'):-0}"; }

# NEVER `echo "$LIST" | head -n N` here. `stage_dataset.sh` is SOURCED above and carries
# `set -euo pipefail`, so this whole driver runs under errexit: once `head` has its N lines it
# exits, `echo` takes SIGPIPE, pipefail propagates 141, and — being the final command of an
# `&&` list — it is not exempt from errexit. The job then dies silently, mid scene-list, with
# nothing in the .err. That is exactly how job 10287385 died (docs/MULTIDATASET.md §7.1), and it
# only bites once the list outgrows the 64 KB pipe buffer, so it looks data-dependent.
# `head -n N <<< "$LIST"` has no pipe and no writer to kill.
TRAIN_PARTS=()
for SRC in $SOURCES; do
    CAP=$(cap_of "$SRC")
    if [ "$SRC" = "scannet" ]; then
        IDS=$(grep -vE '^\s*$' data/splits/scannetv2_train.txt | sort -u)
        [ -d "${SCANNET_ROOT:-/nonexistent}" ] && \
            IDS=$(comm -12 <(echo "$IDS") <(ls "$SCANNET_ROOT" | sort))
        [ "$CAP" -gt 0 ] && IDS=$(head -n "$CAP" <<< "$IDS")
        LIST=$(sed "s|^|${SCANNET_ROOT:-}/|; s|$|/raw_data|" <<< "$IDS")
    else
        LIST=$(find "$STAGE/insscene2d/$SRC" -mindepth 1 -maxdepth 1 -type d | sort)
        [ "$CAP" -gt 0 ] && LIST=$(head -n "$CAP" <<< "$LIST")
    fi
    N=$(grep -c . <<< "$LIST" || true)
    echo "[cfg] $SRC: $N train scenes"
    TRAIN_PARTS+=("$LIST")
done
TRAIN=$(printf '%s\n' "${TRAIN_PARTS[@]}" | { grep . || true; } | paste -sd, -)
N_TRAIN=$(tr ',' '\n' <<< "$TRAIN" | grep -c .)

# Val: the official ScanNet v2 312, unchanged across every mixture (see the header).
if [[ " $SOURCES " == *" scannet "* ]]; then
    VAL_IDS=$(grep -vE '^\s*$' data/splits/scannetv2_val.txt | sort -u)
    [ -d "${SCANNET_ROOT:-/nonexistent}" ] && \
        VAL_IDS=$(comm -12 <(echo "$VAL_IDS") <(ls "$SCANNET_ROOT" | sort))
    [ "${CAP_VAL:-0}" -gt 0 ] && VAL_IDS=$(head -n "${CAP_VAL}" <<< "$VAL_IDS")
    VAL=$(sed "s|^|${SCANNET_ROOT:-}/|; s|$|/raw_data|" <<< "$VAL_IDS" | paste -sd, -)
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
    # the contract the 3520-scene mixture needs: lists reach python as FILES, never as argv
    echo "[dry-run] scene lists: @$RUN/train_scenes.txt @$RUN/val_scenes.txt"
    echo "[dry-run] train list bytes: $(printf '%s' "$TRAIN" | wc -c) (argv cap is 131072)"
    exit 0
fi

mkdir -p "$RUN"
# The lists go in as FILES, not as one giant argv entry. Linux caps a single argument at
# MAX_ARG_STRLEN = 128 KB whatever ARG_MAX says, and the full mixture's 3520 absolute paths are
# ~211 KB: job 10480614 died at execve with "Argument list too long" AFTER staging 117 GB
# (docs/MULTIDATASET.md §7.2). It also leaves the exact scene list in the run dir as provenance.
tr ',' '\n' <<< "$TRAIN" | grep . > "$RUN/train_scenes.txt"
tr ',' '\n' <<< "$VAL"   | grep . > "$RUN/val_scenes.txt" || true
echo "[cfg] scene lists written to $RUN/{train,val}_scenes.txt"

$PYTHON scripts/train_maskdino.py \
    --scans_root "${SCANNET_ROOT:-$STAGE}" \
    --train_scenes "@$RUN/train_scenes.txt" --val_scenes "@$RUN/val_scenes.txt" \
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
