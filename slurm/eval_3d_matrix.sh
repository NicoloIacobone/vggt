#!/bin/bash
#
# The cross-dataset evaluation matrix (docs/todo.md 6d -> docs/RESULTS.md §7): submit
# `slurm/eval_3d_maskdino.sh` for every (checkpoint x dataset x transfer mode) cell.
#
#   bash slurm/eval_3d_matrix.sh                 # the full 3 x 4 x 2 grid
#   DRY_RUN=1 bash slurm/eval_3d_matrix.sh       # print the sbatch lines, submit nothing
#   CKPTS=anchor3d DATASETS=replica bash slurm/eval_3d_matrix.sh    # one row
#
# NOT a SLURM script itself — run it on the login node; it only calls sbatch.
#
# The three headline checkpoints (docs/RESULTS.md §5/§6), all official-split multi-frame:
#   mf        the plain 1201-split control          (job 9386666)
#   anchor3d  --anchor_3d, the strongest 3D row     (job 9634920)
#   s16       --num_frames 16                       (job 9668639)
#
# Every cell runs at DEFAULTS. That is deliberate: the tuned lifting knobs were tuned on a
# leaky diagnostic (docs/MASKDINO.md §9.6), so the plain row is the reportable one, and the
# ScanNetv2 x anchor3d x unproject cell doubles as the REGRESSION GATE — it must reproduce
# 0.038 / 0.112 / 0.360 or the adapters broke something.
#
# Output: one json per cell next to its checkpoint, named by `default_out_path` (the
# dataset tags the filename, so no cell overwrites another).

set -euo pipefail
cd "$(dirname "$0")/.."

OUT=${OUT:-/cluster/work/igp_psr/niacobone/distillation/output}
declare -A RUN=(
    [mf]=maskdino_sf_list1201_mf_20260802_133826
    [anchor3d]=maskdino_sf_list1201_mf_anchor3d_20260804_171436
    [s16]=maskdino_sf_list1201_mf_s16_20260805_095016
)

CKPTS=${CKPTS:-"mf anchor3d s16"}
DATASETS=${DATASETS:-"scannetv2 scannet200 scannetpp replica"}
MODES=${MODES:-"unproject gt_projection"}

# A CKPTS entry is either one of the keys above, or — for arms that postdate this file, e.g.
# the multi-dataset runs of docs/MULTIDATASET.md — a run directory (absolute, or a name under
# $OUT) or an explicit .pth. The keys stay so the three published rows keep their short names.
for c in $CKPTS; do
    run=${RUN[$c]:-}
    if [ -n "$run" ]; then      ckpt=$OUT/$run/checkpoint_best_bundle.pth; tag=$c
    elif [ -f "$c" ];  then     ckpt=$c;                             tag=$(basename "$(dirname "$c")")
    elif [ -d "$c" ];  then     ckpt=$c/checkpoint_best_bundle.pth;  tag=$(basename "$c")
    elif [ -d "$OUT/$c" ]; then ckpt=$OUT/$c/checkpoint_best_bundle.pth; tag=$c
    else echo "unknown checkpoint '$c' — not a key (${!RUN[*]}), a run dir or a .pth"; exit 1; fi
    # SLURM job names are the only place the key is read back, so keep the tag short and safe.
    # A run dir's identity lives in its TAIL (`..._n3520_a3d_e12_<timestamp>`), so drop the
    # timestamp and the shared `maskdino_` prefix and keep the last 28 chars, not the first.
    tag=${tag#maskdino_}
    tag=$(printf '%s' "$tag" | sed -E 's/_[0-9]{8}_[0-9]{6}$//' | tr -c 'A-Za-z0-9_' '_')
    [ ${#tag} -gt 28 ] && tag=${tag: -28}
    if [ ! -f "$ckpt" ]; then
        # A dry run is for reading the sbatch lines, so a missing checkpoint is a warning there
        # and a hard error in a real submission.
        [ "${DRY_RUN:-0}" = 1 ] && echo "WARNING: missing checkpoint $ckpt" \
                               || { echo "missing checkpoint $ckpt"; exit 1; }
    fi
    for d in $DATASETS; do
        for m in $MODES; do
            extra=""
            [ "$m" = gt_projection ] && extra="--transfer_mode gt_projection"
            # EXTRA_ARGS goes through the ENVIRONMENT, not through --export's list: sbatch
            # splits that list on whitespace, so `EXTRA_ARGS=--transfer_mode gt_projection`
            # inside it makes sbatch read "gt_projection" as the script name and die.
            # `--export=ALL` propagates the submitting environment, which carries it intact.
            cmd=(sbatch --job-name="eval3d_${tag}_${d}_${m:0:2}"
                 --export=ALL,DATASET=$d,CHECKPOINT=$ckpt
                 slurm/eval_3d_maskdino.sh)
            if [ "${DRY_RUN:-0}" = 1 ]; then
                echo "EXTRA_ARGS='$extra' ${cmd[*]}"
            else
                EXTRA_ARGS="$extra" "${cmd[@]}"
            fi
        done
    done
done
