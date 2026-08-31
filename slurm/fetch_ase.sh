#!/bin/bash
#
# todo 6n: the ASE pilot — download a scene range of Aria Synthetic Environments, MEASURE what
# it costs in inodes, build the 2D instance training set out of it, and land one tar on work.
#
#   sbatch slurm/fetch_ase.sh                                    # scenes 0-999, the pilot
#   sbatch --export=ALL,SCENE_IDS=0-4999 slurm/fetch_ase.sh      # after the pilot's gate
#   sbatch --export=ALL,PROBE_ONLY=1 slurm/fetch_ase.sh          # download + measure, no build
#
# ONE manual step, and it is a licence, not a path: ASE's per-chunk CDN urls come in a json you
# receive after accepting https://www.projectaria.com/datasets/ase/ . Put it at $CDN_FILE below.
# Everything else — download, sha1, unzip, inode gate, probe, 2D build, pack — is this script.
#
# Why it is worth a job at all (docs/TRAINING_COMPARABILITY.md §5.1-5.3, docs/RESULTS.md §5.6):
# arm I is "IGGT's mixture MINUS ASE" and reads 0.005/0.023/0.251 against FAST3DIS's
# 0.038/0.096/0.316. ASE is the missing component. It does NOT reproduce FAST3DIS's training set
# — their 40 % scene list is unpublished, permanently — so what this buys is a COMPLETE IGGT
# replication, and that is the only claim it licenses.
#
# **It downloads in BLOCKS and never holds the whole range on disk.** 1000 scenes is ~230 GB raw
# but only ~3 GB once built (32 frames/scene at 518x518), so a block is fetched, built, and its
# raw tree deleted before the next block starts. Peak node-local use is one block, which is why
# --tmp is 60 GB and not 400: a 400 GB request schedules badly and a whole-range tree would be
# lost outright on a wall-clock kill. The `.complete` markers live on WORK, so a resubmitted job
# skips every chunk already fetched.
#
# Storage discipline (docs/DATASET.md §5.1): raw scenes and the build both live in $TMPDIR, which
# is node-local and not quota'd. Only `ase2d_<range>.tar.zst` and the json reports land on
# /cluster/work. Zero loose files touch /cluster/scratch, which is quota'd on FILE COUNT.
#
#SBATCH --job-name=fetch_ase
#SBATCH --output=/cluster/scratch/niacobone/vggt/slurm/logs/fetch_ase_%j.log
#SBATCH --error=/cluster/scratch/niacobone/vggt/slurm/logs/fetch_ase_%j.err
#SBATCH --open-mode=append
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4096
#SBATCH --tmp=60000
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=niacobone@student.ethz.ch

set -euo pipefail
module purge
module load stack/2024-06 python/3.12.8 eth_proxy
cd /cluster/scratch/niacobone/vggt
PYTHON=/cluster/scratch/niacobone/vggt/myenv/bin/python
export PYTHONUNBUFFERED=1

OUT_DIR=${OUT_DIR:-/cluster/work/igp_psr/niacobone/distillation/dataset/ase}
CDN_FILE=${CDN_FILE:-$OUT_DIR/ASE_cdn_urls.json}
SCENE_IDS=${SCENE_IDS:-0-999}
SET_TYPE=${SET_TYPE:-train}
FRAMES=${FRAMES:-32}
BLOCK=${BLOCK:-100}                 # scenes per download+build block (~23 GB raw)
# The shell cap is RE10K's measured 0.30 until ASE's own probe replaces it. Step 3 is what
# replaces it; do not silently promote this default (docs/todo.md 6n, docs/MULTIDATASET.md §1.4).
MAX_AREA_FRAC=${MAX_AREA_FRAC:-0.30}
mkdir -p "$OUT_DIR"

TAG=$(printf '%s' "$SCENE_IDS" | tr -c 'A-Za-z0-9' '_')
STATE="$OUT_DIR/state_${TAG}"       # .complete markers — on WORK, so a resubmit resumes
RAW="$TMPDIR/ase_raw"
BUILD="$TMPDIR/ase_build"
mkdir -p "$STATE" "$RAW" "$BUILD"

echo "=== $(date) : node $(hostname), TMPDIR=$TMPDIR ==="
df -h "$TMPDIR"

if [ ! -f "$CDN_FILE" ]; then
    cat <<EOF
MISSING CDN FILE: $CDN_FILE

ASE is licence-gated. The one manual step:
  1. accept the licence at https://www.projectaria.com/datasets/ase/
  2. download the CDN json it hands you
  3. put it at $CDN_FILE

Then resubmit this job unchanged. Nothing else here needs a human.
EOF
    exit 2
fi

# The range as two integers, so it can be walked in blocks. Only a single `LO-HI` range is
# supported here; download_ase.py itself takes the full comma grammar if a job ever needs it.
case "$SCENE_IDS" in
    *-*) LO=${SCENE_IDS%%-*}; HI=${SCENE_IDS##*-} ;;
    *)   LO=$SCENE_IDS;       HI=$SCENE_IDS ;;
esac
echo "=== ASE $SET_TYPE scenes $LO..$HI in blocks of $BLOCK ==="

TOTAL_INODES=0
TOTAL_SCENES=0
for ((lo = LO; lo <= HI; lo += BLOCK)); do
    hi=$((lo + BLOCK - 1)); [ "$hi" -gt "$HI" ] && hi=$HI
    echo "=== block $lo-$hi ==="
    rm -rf "$RAW"; mkdir -p "$RAW"

    $PYTHON slurm/download_ase.py \
        --cdn_file "$CDN_FILE" \
        --out_dir "$RAW" \
        --tmp_dir "$TMPDIR/ase_zips" \
        --state_dir "$STATE" \
        --scene_ids "$lo-$hi" \
        --set "$SET_TYPE" \
        --time_budget_hours "${TIME_BUDGET_HOURS:-20}" \
        --report "$OUT_DIR/FETCH_ase_${TAG}_${lo}_${hi}.json"

    # A block whose chunks were all already fetched in an earlier job has an empty $RAW: the
    # markers are on work but the data was node-local. Its scenes are already inside the tar
    # that job wrote, so skipping is correct — re-fetching would need the markers cleared.
    n=$(find "$RAW" -mindepth 1 -maxdepth 1 -type d | wc -l)
    if [ "$n" -eq 0 ]; then
        echo "    block $lo-$hi already fetched by an earlier job (markers in $STATE) — skipping"
        continue
    fi

    ####################################
    # THE GATE (docs/todo.md 6n): inodes, not gigabytes
    ####################################
    inodes=$(find "$RAW" -type f -not -name '.*' | wc -l)
    TOTAL_INODES=$((TOTAL_INODES + inodes)); TOTAL_SCENES=$((TOTAL_SCENES + n))
    echo "    gate: $n scenes, $inodes files, $(du -sh "$RAW" | cut -f1)"

    ####################################
    # measure the shell-cap distribution BEFORE applying one — once, on the first block
    ####################################
    if [ ! -f "$OUT_DIR/PROBE_ase_${TAG}.json" ]; then
        echo "    probing the instance-area distribution (the cap is measured, not inherited)"
        $PYTHON slurm/build_insscene2d.py --source ase --ase_root "$RAW" \
            --out "$TMPDIR/ase_probe" --frames "$FRAMES" --limit "${PROBE_SCENES:-60}" --probe
        cp "$TMPDIR/ase_probe/REPORT_ase.json" "$OUT_DIR/PROBE_ase_${TAG}.json"
        if [ "${PROBE_ONLY:-0}" = 1 ]; then
            echo "=== PROBE_ONLY: stopping before the build."
            echo "    Read $OUT_DIR/PROBE_ase_${TAG}.json, pick MAX_AREA_FRAC off its"
            echo "    dropped_frac_at table, then resubmit without PROBE_ONLY. ==="
            exit 0
        fi
    fi

    ####################################
    # build this block into the shared output tree
    ####################################
    $PYTHON slurm/build_insscene2d.py --source ase --ase_root "$RAW" \
        --out "$BUILD" --frames "$FRAMES" --max_area_frac "$MAX_AREA_FRAC"
    mv "$BUILD/REPORT_ase.json" "$BUILD/REPORT_ase_${lo}_${hi}.json"
done
rm -rf "$RAW"

########################################
# pack — the ONLY thing that leaves the node
########################################
BUILT=$(find "$BUILD/ase" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
echo "=== $BUILT scenes built; $TOTAL_SCENES fetched this job, $TOTAL_INODES raw files ==="
[ "$BUILT" -gt 0 ] || { echo "nothing was built — check the reports in $OUT_DIR"; exit 1; }
if [ "$TOTAL_SCENES" -gt 0 ]; then
    echo "    per-scene raw cost: $((TOTAL_INODES / TOTAL_SCENES)) files — the number that"
    echo "    decides whether this range scales, NOT the byte size (docs/todo.md 6n)"
fi

echo "=== packing ==="
tar --use-compress-program="zstd -19 -T0" -C "$BUILD" -cf "$OUT_DIR/ase2d_${TAG}.tar.zst" .
ls -la "$OUT_DIR/ase2d_${TAG}.tar.zst"
echo "=== $(date) : done ==="
