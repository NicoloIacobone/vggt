#!/bin/bash
# CPU-only dry-run test of slurm/train_maskdino.sh scene-list derivation (no staging, no GPU).
#
#   bash tests/test_train_maskdino_sh_lists.sh
#
# Checks, via DRY_RUN=1:
#   1. backward compat — the default and VAL_SPLIT=official invocations produce the exact
#      lists and schedule the pre-TRAIN_LIST script produced;
#   2. TRAIN_LIST/VAL_LIST mode reads the official split files (1201/312, disjoint) and the
#      auto epoch budget is not min-clamped to 60;
#   3. scenes missing from SCANNET_ROOT are dropped with a warning;
#   4. overlapping train/val lists abort.
set -uo pipefail
cd "$(dirname "$0")/.."
FAIL=0
check() {  # check <name> <expected> <actual>
    if [ "$2" = "$3" ]; then echo "PASS: $1"; else
        echo "FAIL: $1"; echo "  expected: $2"; echo "  actual:   $3"; FAIL=1; fi
}
grab() { sed -n "s/^\[dry-run\] $1 ([0-9]*): //p" <<< "$2"; }

# --- 1a. default invocation (N_SCENES=50) ----------------------------------------------------
OUT=$(DRY_RUN=1 bash slurm/train_maskdino.sh 2>&1)
EXP_TRAIN=$(seq -f "scene%04g_00" 0 49 | grep -v -E "scene008[0-9]_00" | paste -sd, -)
EXP_VAL=$(seq -f "scene%04g_00" 80 89 | paste -sd, -)
check "default TRAIN" "$EXP_TRAIN" "$(grab TRAIN "$OUT")"
check "default VAL" "$EXP_VAL" "$(grab VAL "$OUT")"
check "default schedule" "1" "$(grep -c "scenes=50 epochs=400 warmup=20 eval_every=10" <<< "$OUT")"

# --- 1b. N_SCENES=490 keeps the old auto schedule (min-60 clamp intact) ----------------------
OUT=$(DRY_RUN=1 N_SCENES=490 bash slurm/train_maskdino.sh 2>&1)
check "n490 TRAIN count" "480" "$(grab TRAIN "$OUT" | tr ',' '\n' | wc -l)"
check "n490 schedule" "1" "$(grep -c "scenes=490 epochs=60 warmup=5 eval_every=1" <<< "$OUT")"

# --- 1c. VAL_SPLIT=official unchanged --------------------------------------------------------
OUT=$(DRY_RUN=1 N_SCENES=490 VAL_SPLIT=official bash slurm/train_maskdino.sh 2>&1)
EXP_VAL=$(awk -F'_' '$2=="00" && substr($1,6)+0 < 490' data/splits/scannetv2_val.txt \
          | sort | paste -sd, -)
EXP_TRAIN=$(seq -f "scene%04g_00" 0 489 | grep -v -F -x -f <(tr ',' '\n' <<< "$EXP_VAL") \
            | paste -sd, -)
check "official VAL" "$EXP_VAL" "$(grab VAL "$OUT")"
check "official TRAIN" "$EXP_TRAIN" "$(grab TRAIN "$OUT")"

# --- 2. TRAIN_LIST/VAL_LIST with the real split files (no SCANNET_ROOT -> no filtering) ------
OUT=$(DRY_RUN=1 TRAIN_LIST=data/splits/scannetv2_train.txt \
      VAL_LIST=data/splits/scannetv2_val.txt bash slurm/train_maskdino.sh 2>&1)
check "list TRAIN count" "1201" "$(grab TRAIN "$OUT" | tr ',' '\n' | wc -l)"
check "list VAL count" "312" "$(grab VAL "$OUT" | tr ',' '\n' | wc -l)"
check "list TRAIN content" \
    "$(sort -u data/splits/scannetv2_train.txt | paste -sd, -)" "$(grab TRAIN "$OUT")"
check "list auto epochs (20000/1201=16, no min-60 clamp)" "1" \
    "$(grep -c "scenes=1201 epochs=16 " <<< "$OUT")"
check "list run dir" "1" "$(grep -c "RUN=.*maskdino_sf_list1201_" <<< "$OUT")"
OUT=$(DRY_RUN=1 EPOCHS=12 WARMUP=2 TRAIN_LIST=data/splits/scannetv2_train.txt \
      VAL_LIST=data/splits/scannetv2_val.txt bash slurm/train_maskdino.sh 2>&1)
check "explicit EPOCHS/WARMUP honoured" "1" \
    "$(grep -c "scenes=1201 epochs=12 warmup=2 eval_every=1" <<< "$OUT")"

# --- 3. missing-scene filtering against a fake SCANNET_ROOT ----------------------------------
T=$(mktemp -d); mkdir -p "$T/scans"/scene9998_00 "$T/scans"/scene9999_00
printf 'scene9999_00\nscene7777_00\n' > "$T/train.txt"   # scene7777 not on disk
printf 'scene9999_00\n' > "$T/val.txt"   # overlaps train -> tests the abort below
printf 'scene9998_00\n' > "$T/val_ok.txt"
OUT=$(DRY_RUN=1 SCANNET_ROOT="$T/scans" TRAIN_LIST="$T/train.txt" VAL_LIST="$T/val_ok.txt" \
      bash slurm/train_maskdino.sh 2>&1)
check "missing scene dropped" "scene9999_00" "$(grab TRAIN "$OUT")"
check "missing scene warned" "1" "$(grep -c "WARNING: 1 scenes from" <<< "$OUT")"

# --- 4. overlapping lists abort --------------------------------------------------------------
OUT=$(DRY_RUN=1 SCANNET_ROOT="$T/scans" TRAIN_LIST="$T/train.txt" VAL_LIST="$T/val.txt" \
      bash slurm/train_maskdino.sh 2>&1); RC=$?
check "overlap aborts (rc)" "1" "$RC"
check "overlap aborts (msg)" "1" "$(grep -c "ERROR: train/val lists share scenes" <<< "$OUT")"
rm -rf "$T"

[ "$FAIL" -eq 0 ] && echo "ALL PASS" || echo "FAILURES"; exit $FAIL
