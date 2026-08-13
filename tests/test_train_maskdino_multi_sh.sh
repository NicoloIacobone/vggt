#!/bin/bash
# CPU-only dry-run test of slurm/train_maskdino_multi.sh scene-list derivation.
#
#   bash tests/test_train_maskdino_multi_sh.sh
#
# The regression it exists for (docs/MULTIDATASET.md §7.1). `stage_dataset.sh` is SOURCED by the
# driver and carries `set -euo pipefail`, so the driver runs under errexit. A `echo "$LIST" |
# head -n "$CAP"` then kills the job by SIGPIPE as soon as the list outgrows the 64 KB pipe
# buffer — silently, mid scene-list, with an empty .err. Job 10287385 died that way and the
# mixture run was blocked on it for two days. The buffer threshold is why it looked
# data-dependent: 853 ScanNet++ paths fit, 1466 Infinigen paths did not.
#
# So check 3 below builds a source with MORE scenes than the buffer holds and caps it. Under the
# old code the whole script dies there and produces no output at all.
#
# ⚠ IT MUST RUN THE SCRIPT UNDER errexit TO SEE THAT. `DRY_RUN=1` skips the block that sources
# `stage_dataset.sh`, so a plain dry run does NOT inherit `set -euo pipefail` and the bug is
# invisible — the first version of this test passed against the broken script. `run_strict`
# therefore invokes bash with the same three options the sourced file would have set. Any future
# check for a silent-abort bug belongs there, not in `run`.
#
# Checks:
#   1. uncapped: every source is listed, val is the official 312, the schedule is derived;
#   2. SOURCES selects a subset and val stays ScanNet-only;
#   3. CAP_* under errexit survives a list far larger than the pipe buffer (the regression);
#   4. CAP_VAL caps the val list, also under errexit;
#   5. every emitted train path is absolute and comes from the right source.
set -uo pipefail
cd "$(dirname "$0")/.."
FAIL=0
check() {  # check <name> <expected> <actual>
    if [ "$2" = "$3" ]; then echo "PASS: $1"; else
        echo "FAIL: $1"; echo "  expected: $2"; echo "  actual:   $3"; FAIL=1; fi
}
grab() { sed -n "s/^\[dry-run\] $1: //p" <<< "$2"; }

# A fake staged tree: $STAGE/insscene2d/<source>/<scene>/ with deliberately long names, so the
# capped list is ~90 KB — comfortably past the 64 KB pipe buffer the regression needs.
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/insscene2d/scannetpp" "$STAGE/insscene2d/infinigen"
for i in $(seq 1 40); do mkdir -p "$STAGE/insscene2d/scannetpp/scene_pp_$(printf '%04d' "$i")"; done
for i in $(seq 1 1466); do
    mkdir -p "$STAGE/insscene2d/infinigen/scene_infinigen_sub_scene_$(printf '%06d' "$i")"
done
export TMPDIR="$STAGE"

N_PP=40
N_INF=1466
N_SCANNET=$(grep -cvE '^\s*$' data/splits/scannetv2_train.txt)
N_VAL_ALL=$(grep -cvE '^\s*$' data/splits/scannetv2_val.txt)

run() { DRY_RUN=1 env "$@" bash slurm/train_maskdino_multi.sh 2>&1; }

# The real job inherits `set -euo pipefail` from the sourced stage_dataset.sh; DRY_RUN skips that
# source, so errexit has to be forced here or silent-abort bugs stay invisible (see the header).
run_strict() {
    DRY_RUN=1 env "$@" bash -o errexit -o pipefail -o nounset slurm/train_maskdino_multi.sh 2>&1
}

# --- 1. the full mixture, uncapped -----------------------------------------------------------
OUT=$(run)
check "uncapped scannet count"   "$N_SCANNET" "$(sed -n 's/^\[cfg\] scannet: \([0-9]*\) .*/\1/p'   <<< "$OUT")"
check "uncapped scannetpp count" "$N_PP"      "$(sed -n 's/^\[cfg\] scannetpp: \([0-9]*\) .*/\1/p' <<< "$OUT")"
check "uncapped infinigen count" "$N_INF"     "$(sed -n 's/^\[cfg\] infinigen: \([0-9]*\) .*/\1/p' <<< "$OUT")"
TOTAL=$((N_SCANNET + N_PP + N_INF))
check "uncapped total + val" "$TOTAL train scenes, $N_VAL_ALL val scenes" \
      "$(sed -n 's/^\[cfg\] \(.*\) (ScanNet official val, class-agnostic)/\1/p' <<< "$OUT")"
check "schedule is derived" "1" "$(grep -c '^\[cfg\] epochs=[0-9]* warmup=[0-9]* eval_every=1$' <<< "$OUT")"

# --- 2. SOURCES selects a subset -------------------------------------------------------------
OUT=$(run SOURCES='scannet scannetpp')
check "subset omits infinigen" "0" "$(grep -c '^\[cfg\] infinigen:' <<< "$OUT")"
check "subset val is still the full 312" "$N_VAL_ALL" \
      "$(sed -n 's/^\[cfg\] [0-9]* train scenes, \([0-9]*\) val.*/\1/p' <<< "$OUT")"

# --- 3. THE REGRESSION: caps on a list bigger than the pipe buffer ----------------------------
BYTES=$(find "$STAGE/insscene2d/infinigen" -mindepth 1 -maxdepth 1 -type d | wc -c)
check "the infinigen list really is past the 64 KB pipe buffer" "1" \
      "$([ "$BYTES" -gt 65536 ] && echo 1 || echo 0)"
OUT=$(run_strict CAP_SCANNET=6 CAP_SCANNETPP=6 CAP_INFINIGEN=6)
check "capped run under errexit reaches the end at all (SIGPIPE regression)" "1" \
      "$(grep -c '^\[dry-run\] RUN=' <<< "$OUT")"
check "capped scannet"   "6" "$(sed -n 's/^\[cfg\] scannet: \([0-9]*\) .*/\1/p'   <<< "$OUT")"
check "capped scannetpp" "6" "$(sed -n 's/^\[cfg\] scannetpp: \([0-9]*\) .*/\1/p' <<< "$OUT")"
check "capped infinigen" "6" "$(sed -n 's/^\[cfg\] infinigen: \([0-9]*\) .*/\1/p' <<< "$OUT")"
check "capped total" "18 train scenes, $N_VAL_ALL val scenes" \
      "$(sed -n 's/^\[cfg\] \(.*\) (ScanNet official val, class-agnostic)/\1/p' <<< "$OUT")"

# --- 4. CAP_VAL -------------------------------------------------------------------------------
OUT=$(run_strict CAP_SCANNET=6 CAP_SCANNETPP=6 CAP_INFINIGEN=6 CAP_VAL=4)
check "CAP_VAL caps the val list under errexit" "18 train scenes, 4 val scenes" \
      "$(sed -n 's/^\[cfg\] \(.*\) (ScanNet official val, class-agnostic)/\1/p' <<< "$OUT")"

# --- 5. the emitted paths ---------------------------------------------------------------------
OUT=$(run CAP_SCANNET=3 CAP_SCANNETPP=3 CAP_INFINIGEN=3)
FIRST=$(grab "first train entries" "$OUT"; sed -n '/first train entries:/,/last train entries:/p' <<< "$OUT" \
        | grep '^/' | head -3)
check "train entries are absolute paths" "3" "$(grep -c '^/' <<< "$FIRST")"
LAST=$(sed -n '/last train entries:/,$p' <<< "$OUT" | grep '^/' | head -3)
check "the last entries come from the last source" "3" "$(grep -c 'infinigen' <<< "$LAST")"

# --- 6. the lists must reach python as FILES (job 10480614: argv cap, MULTIDATASET.md §7.2) ----
OUT=$(run)
check "scene lists are passed as @files, not argv" "1" \
      "$(grep -c '^\[dry-run\] scene lists: @.*/train_scenes.txt @.*/val_scenes.txt$' <<< "$OUT")"
BYTES=$(sed -n 's/^\[dry-run\] train list bytes: \([0-9]*\) .*/\1/p' <<< "$OUT")
check "the uncapped train list really does exceed the 128 KB argv cap" "1" \
      "$([ "${BYTES:-0}" -gt 131072 ] && echo 1 || echo 0)"

[ "$FAIL" -eq 0 ] && echo && echo "all train_maskdino_multi.sh dry-run checks passed"
exit "$FAIL"
