#!/bin/bash
#
# DRY_RUN checks for slurm/eval_3d_matrix.sh (docs/RESULTS.md §7): the grid it prints, and the
# three ways a checkpoint may be named. No cluster, no sbatch, no real checkpoint.
#
#   bash tests/test_eval_3d_matrix_sh.sh
#
set -uo pipefail
cd "$(dirname "$0")/.."
FAIL=0
ok()   { echo "PASS: $1"; }
bad()  { echo "FAIL: $1"; FAIL=1; }
check(){ if [ "$2" = "$3" ]; then ok "$1"; else bad "$1 (got '$2', want '$3')"; fi; }

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/runA" "$TMP/runB"
: > "$TMP/runA/checkpoint_best_bundle.pth"
: > "$TMP/runA/checkpoint.pth"                # the FINAL epoch — what CKPT_NAME selects
: > "$TMP/runB/checkpoint_best_bundle.pth"

run() { DRY_RUN=1 OUT="$TMP" bash slurm/eval_3d_matrix.sh 2>&1; }

# --- the grid ---------------------------------------------------------------------------
OUTPUT=$(CKPTS=runA DATASETS="scannetv2 scannet200 scannetpp replica" \
         MODES="unproject gt_projection" run)
check "4 datasets x 2 modes = 8 cells" "$(grep -c '^EXTRA_ARGS=' <<< "$OUTPUT")" 8
check "gt_projection cells pass the flag" \
      "$(grep -c "EXTRA_ARGS='--transfer_mode gt_projection'" <<< "$OUTPUT")" 4
check "unproject cells pass nothing"      "$(grep -c "EXTRA_ARGS='' " <<< "$OUTPUT")" 4

# --- the three ways of naming a checkpoint ----------------------------------------------
OUTPUT=$(CKPTS="runA" DATASETS=replica MODES=unproject run)
grep -q "CHECKPOINT=$TMP/runA/checkpoint_best_bundle.pth" <<< "$OUTPUT" \
    && ok "a run-dir name under \$OUT resolves" || bad "a run-dir name under \$OUT resolves"
OUTPUT=$(CKPTS="$TMP/runB" DATASETS=replica MODES=unproject run)
grep -q "CHECKPOINT=$TMP/runB/checkpoint_best_bundle.pth" <<< "$OUTPUT" \
    && ok "an absolute run dir resolves" || bad "an absolute run dir resolves"
OUTPUT=$(CKPTS="$TMP/runB/checkpoint_best_bundle.pth" DATASETS=replica MODES=unproject run)
grep -q "job-name=eval3d_runB_replica_un" <<< "$OUTPUT" \
    && ok "an explicit .pth takes its tag from the run dir" \
    || bad "an explicit .pth takes its tag from the run dir"

# --- the published keys still work, and unknown names still abort ------------------------
OUTPUT=$(CKPTS="mf anchor3d s16" DATASETS=scannetv2 MODES=unproject run)
check "the three published keys still resolve" "$(grep -c '^EXTRA_ARGS=' <<< "$OUTPUT")" 3
grep -q "job-name=eval3d_anchor3d_scannetv2_un" <<< "$OUTPUT" \
    && ok "a key keeps its short job name" || bad "a key keeps its short job name"
grep -q "WARNING: missing checkpoint" <<< "$OUTPUT" \
    && ok "a dry run warns about a missing checkpoint instead of dying" \
    || bad "a dry run warns about a missing checkpoint instead of dying"

OUTPUT=$(CKPTS="not_a_run" DATASETS=replica MODES=unproject run); RC=$?
check "an unknown name aborts" "$RC" 1
grep -q "unknown checkpoint 'not_a_run'" <<< "$OUTPUT" \
    && ok "…and says so" || bad "…and says so"

# --- slurm/chain_eval3d_matrix.sh: resolving a run dir from a training log ---------------
LOGS=$TMP/logs; mkdir -p "$LOGS"
printf '%s\n' \
  "[cfg] 3520 train scenes, 312 val scenes" \
  "[cfg] scene lists written to $TMP/runA/{train,val}_scenes.txt" \
  "Using device: cuda" > "$LOGS/maskdino_multi_999001.log"

OUTPUT=$(TRAIN_JOB=999001 LOG_DIR="$LOGS" OUT="$TMP" DRY_RUN=1 DATASETS=replica \
         MODES=unproject bash slurm/chain_eval3d_matrix.sh 2>&1)
grep -q "\[chain\] job 999001 -> $TMP/runA" <<< "$OUTPUT" \
    && ok "the chain reads the run dir out of the training log" \
    || bad "the chain reads the run dir out of the training log ($OUTPUT)"
grep -q "CHECKPOINT=$TMP/runA/checkpoint_best_bundle.pth" <<< "$OUTPUT" \
    && ok "…and hands it to the matrix" || bad "…and hands it to the matrix"

# CKPT_NAME picks a different checkpoint in the same run dir — the ZERO-SHOT arms must be
# scored on the FINAL epoch, because the val ruler their best-bundle checkpoint is selected on
# is itself zero-shot and cannot separate epochs (docs/MULTIDATASET.md §12.1).
OUTPUT=$(TRAIN_JOB=999001 LOG_DIR="$LOGS" OUT="$TMP" DRY_RUN=1 DATASETS=replica \
         MODES=unproject CKPT_NAME=checkpoint.pth bash slurm/chain_eval3d_matrix.sh 2>&1)
grep -q "CHECKPOINT=$TMP/runA/checkpoint.pth" <<< "$OUTPUT" \
    && ok "CKPT_NAME selects the final checkpoint" || bad "CKPT_NAME selects the final checkpoint ($OUTPUT)"
grep -q "checkpoint_best_bundle" <<< "$OUTPUT" \
    && bad "…and does NOT also submit the default one" || ok "…and does NOT also submit the default one"
grep -q "eval3d_runA" <<< "$OUTPUT" \
    && ok "…and the job name still identifies the run, not the checkpoint file" \
    || bad "…and the job name still identifies the run ($OUTPUT)"

OUTPUT=$(TRAIN_JOB=999002 LOG_DIR="$LOGS" DRY_RUN=1 bash slurm/chain_eval3d_matrix.sh 2>&1); RC=$?
check "a missing training log aborts" "$RC" 1
: > "$LOGS/maskdino_multi_999003.log"
OUTPUT=$(TRAIN_JOB=999003 LOG_DIR="$LOGS" DRY_RUN=1 bash slurm/chain_eval3d_matrix.sh 2>&1); RC=$?
check "a log without the run-dir line aborts" "$RC" 1

# --- the SLURM shape: a COPY of the script, run with the cwd somewhere else ---------------
# This is what actually killed jobs 11436321/23/24. SLURM spools the batch script, so `$0` is
# not the repo path and a `cd $(dirname $0)/..` lands nowhere. The DRY_RUN checks above cannot
# see it: they run from the repo, where the wrong cd happens to be harmless.
REPO_DEFAULT=$(sed -n 's/^REPO=${REPO:-\(.*\)}$/\1/p' slurm/chain_eval3d_matrix.sh)
if [ "$(cd "$REPO_DEFAULT" 2>/dev/null && pwd -P)" = "$(pwd -P)" ]; then
    FAKE=slurm/logs/maskdino_multi_999009.log
    printf '%s\n' "[cfg] scene lists written to $TMP/runA/{train,val}_scenes.txt" > "$FAKE"
    cp slurm/chain_eval3d_matrix.sh "$TMP/slurm_script"     # SLURM's spooled copy
    OUTPUT=$(cd "$TMP" && TRAIN_JOB=999009 OUT="$TMP" DRY_RUN=1 DATASETS=replica \
             MODES=unproject bash "$TMP/slurm_script" 2>&1)
    rm -f "$FAKE"
    grep -q "\[chain\] job 999009 -> $TMP/runA" <<< "$OUTPUT" \
        && ok "the chain finds the repo when run as SLURM runs it (spooled copy, foreign cwd)" \
        || bad "the chain finds the repo when run as SLURM runs it ($OUTPUT)"
else
    echo "SKIP: chain cwd check (REPO default '$REPO_DEFAULT' is not this checkout)"
fi

echo
[ "$FAIL" = 0 ] && echo "all eval_3d_matrix.sh dry-run checks passed" \
                || { echo "SOME CHECKS FAILED"; exit 1; }
