#!/bin/bash
#
# todo 6n: the ASE pilot — download a scene range of Aria Synthetic Environments, MEASURE what
# it costs in inodes, build the 2D instance training set out of it, and land the source tar the
# multi-dataset trainer already knows how to stage.
#
#   sbatch --export=ALL,PROBE_ONLY=1 slurm/fetch_ase.sh   # FIRST: one block, measure, stop
#   sbatch slurm/fetch_ase.sh                             # scenes 0-999, the pilot
#   sbatch --export=ALL,SCENE_IDS=0-4999 slurm/fetch_ase.sh          # after the pilot's gate
#
# ONE manual step, and it is a licence, not a path: ASE's per-chunk CDN urls come in a json you
# receive after accepting https://www.projectaria.com/datasets/ase/ . Put it at $CDN_FILE below.
# Everything else — download, sha1, unzip, inode gate, probe, 2D build, pack — is this script.
#
# Why it is worth a job at all (docs/TRAINING_COMPARABILITY.md §6.6-6.7, docs/RESULTS.md §5.6):
# arm I is "IGGT's mixture MINUS ASE" and reads 0.005/0.023/0.251 against FAST3DIS's
# 0.038/0.096/0.316. ASE is the missing component. It does NOT reproduce FAST3DIS's training set
# — their 40 % scene list is unpublished, permanently — so what this buys is a COMPLETE IGGT
# replication, and that is the only claim it licenses.
#
# **It downloads in BLOCKS and never holds the whole range on disk.** Measured 2026-08-31 against
# the live CDN: **2211 MiB/chunk = 221 MiB/scene at 47 MiB/s**, so the 1000-scene pilot is 216 GiB
# in ~1.3 h — one job, comfortably. A block is fetched, built, packed to its own small tar on
# work, and its raw tree deleted before the next block starts, so peak node-local use is one
# block. That is why --tmp is 60 GB and not 400: a 400 GB request schedules badly.
#
# **Resume granularity is the BLOCK, and it is the block TAR that records it** — not the chunk
# download markers, which live in $TMPDIR and die with the job. That distinction is the whole
# correctness argument: a marker says "downloaded", and a block that was downloaded but not built
# (PROBE_ONLY, or a wall-clock kill) must be fetched again or its scenes vanish from the output
# silently. Only a block tar on work proves a block is done.
#
# Storage discipline (docs/DATASET.md §5.1): raw scenes and the build live in $TMPDIR, which is
# node-local and not quota'd. Only the block tars and the final `insscene2d_ase.tar.zst` land on
# /cluster/work, and the block tars are deleted once merged. Zero loose files touch
# /cluster/scratch, which is quota'd on FILE COUNT.
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

DATASET_ROOT=${DATASET_ROOT:-/cluster/work/igp_psr/niacobone/distillation/dataset}
OUT_DIR=${OUT_DIR:-$DATASET_ROOT/ase}
DEST=${DEST:-$DATASET_ROOT/insscene2d}     # where the trainer looks for insscene2d_ase.tar.zst
CDN_FILE=${CDN_FILE:-$OUT_DIR/ASE_cdn_urls.json}
SCENE_IDS=${SCENE_IDS:-0-999}
SET_TYPE=${SET_TYPE:-train}
FRAMES=${FRAMES:-32}
BLOCK=${BLOCK:-100}                 # scenes per download+build block (~22 GiB raw)
# The shell cap is RE10K's measured 0.30 until ASE's own probe replaces it. The probe step is
# what replaces it; do not silently promote this default (docs/todo.md 6n, MULTIDATASET.md §1.4).
MAX_AREA_FRAC=${MAX_AREA_FRAC:-0.30}
mkdir -p "$OUT_DIR" "$DEST"

TAG=$(printf '%s' "$SCENE_IDS" | tr -c 'A-Za-z0-9' '_')
BLOCKS_DIR="$OUT_DIR/blocks_${TAG}"        # per-block tars — the resume record, on WORK
RAW="$TMPDIR/ase_raw"
BUILD="$TMPDIR/ase_build"
mkdir -p "$BLOCKS_DIR" "$RAW" "$BUILD"

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

# The CDN packs 10 scenes per chunk. If a block boundary falls inside a chunk, two blocks want
# the same chunk — and since the download markers are per-JOB, the second block would see it as
# already fetched and silently lose those scenes. Refuse rather than produce a short dataset.
if [ $((LO % 10)) -ne 0 ] || [ $((BLOCK % 10)) -ne 0 ]; then
    echo "LO ($LO) and BLOCK ($BLOCK) must both be multiples of 10 — the CDN chunk size."
    echo "Otherwise two blocks share a chunk and the second one loses its scenes."
    exit 1
fi
echo "=== ASE $SET_TYPE scenes $LO..$HI in blocks of $BLOCK ==="

TOTAL_INODES=0
TOTAL_SCENES=0
for ((lo = LO; lo <= HI; lo += BLOCK)); do
    hi=$((lo + BLOCK - 1)); [ "$hi" -gt "$HI" ] && hi=$HI
    BLOCK_TAR="$BLOCKS_DIR/ase_${lo}_${hi}.tar.zst"

    # THE resume test: a block is done when its tar exists, never when its chunks are markered.
    if [ -f "$BLOCK_TAR" ]; then
        echo "=== block $lo-$hi already built ($(basename "$BLOCK_TAR")) — skipping ==="
        continue
    fi
    echo "=== block $lo-$hi ==="
    rm -rf "$RAW" "$TMPDIR/ase_state"; mkdir -p "$RAW"

    $PYTHON slurm/download_ase.py \
        --cdn_file "$CDN_FILE" \
        --out_dir "$RAW" \
        --tmp_dir "$TMPDIR/ase_zips" \
        --state_dir "$TMPDIR/ase_state" \
        --scene_ids "$lo-$hi" \
        --set "$SET_TYPE" \
        --time_budget_hours "${TIME_BUDGET_HOURS:-20}" \
        --report "$OUT_DIR/FETCH_ase_${TAG}_${lo}_${hi}.json"

    n=$(find "$RAW" -mindepth 1 -maxdepth 1 -type d | wc -l)
    [ "$n" -gt 0 ] || { echo "block $lo-$hi downloaded no scene — see the report"; exit 1; }

    ####################################
    # THE GATE (docs/todo.md 6n): inodes, not gigabytes
    ####################################
    inodes=$(find "$RAW" -type f -not -name '.*' | wc -l)
    TOTAL_INODES=$((TOTAL_INODES + inodes)); TOTAL_SCENES=$((TOTAL_SCENES + n))
    echo "    gate: $n scenes, $inodes files, $(du -sh "$RAW" | cut -f1)"

    ####################################
    # measure the shell-cap distribution BEFORE applying one — once, on the first block built
    ####################################
    if [ ! -f "$OUT_DIR/PROBE_ase_${TAG}.json" ]; then
        echo "    probing the instance-area distribution (the cap is measured, not inherited)"
        rm -rf "$TMPDIR/ase_probe"
        $PYTHON slurm/build_insscene2d.py --source ase --ase_root "$RAW" \
            --out "$TMPDIR/ase_probe" --frames "$FRAMES" --limit "${PROBE_SCENES:-60}" --probe
        cp "$TMPDIR/ase_probe/REPORT_ase.json" "$OUT_DIR/PROBE_ase_${TAG}.json"
        if [ "${PROBE_ONLY:-0}" = 1 ]; then
            echo "=== PROBE_ONLY: stopping before the build."
            echo "    Read $OUT_DIR/PROBE_ase_${TAG}.json, pick MAX_AREA_FRAC off its"
            echo "    dropped_frac_at table, then resubmit without PROBE_ONLY."
            echo "    No block tar was written, so this block WILL be re-fetched — that is"
            echo "    deliberate: a downloaded-but-unbuilt block must never count as done. ==="
            exit 0
        fi
    fi

    ####################################
    # build this block, pack it, and only THEN record it as done
    ####################################
    rm -rf "$BUILD"; mkdir -p "$BUILD"
    $PYTHON slurm/build_insscene2d.py --source ase --ase_root "$RAW" \
        --out "$BUILD" --frames "$FRAMES" --max_area_frac "$MAX_AREA_FRAC"
    mv "$BUILD/REPORT_ase.json" "$BUILD/REPORT_ase_${lo}_${hi}.json"
    tar -C "$BUILD" -cf - . | zstd -T8 -3 -o "$BLOCK_TAR.part" -f
    mv "$BLOCK_TAR.part" "$BLOCK_TAR"          # atomic: a half-written tar is never a "done"
    echo "    block $lo-$hi packed: $(du -h "$BLOCK_TAR" | cut -f1)"
done
rm -rf "$RAW" "$BUILD"

########################################
# merge the block tars into the ONE tar the trainer stages, then drop them
########################################
echo "=== merging $(ls "$BLOCKS_DIR"/ase_*.tar.zst 2>/dev/null | wc -l) block tars ==="
MERGE="$TMPDIR/ase_merge"; rm -rf "$MERGE"; mkdir -p "$MERGE"
for t in "$BLOCKS_DIR"/ase_*.tar.zst; do zstd -dc "$t" | tar -C "$MERGE" -xf -; done

BUILT=$(find "$MERGE/ase" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
echo "=== $BUILT scenes built; $TOTAL_SCENES fetched this job, $TOTAL_INODES raw files ==="
[ "$BUILT" -gt 0 ] || { echo "nothing was built — check the reports in $OUT_DIR"; exit 1; }
if [ "$TOTAL_SCENES" -gt 0 ]; then
    echo "    per-scene raw cost: $((TOTAL_INODES / TOTAL_SCENES)) files — the number that"
    echo "    decides whether this range scales, NOT the byte size (docs/todo.md 6n)"
fi

# The trainer stages exactly one tar per source, `insscene2d_<src>.tar.zst`, and unpacks it into
# $STAGE/insscene2d/ — so the archive must carry `ase/<scene>/...` at its top level.
echo "=== packing $DEST/insscene2d_ase.tar.zst ==="
tar -C "$MERGE" -cf - . | zstd -T8 -3 -o "$DEST/insscene2d_ase.tar.zst" -f
cp "$MERGE"/REPORT_ase_*.json "$OUT_DIR/" 2>/dev/null || true
ls -lh "$DEST/insscene2d_ase.tar.zst"
rm -rf "$BLOCKS_DIR" "$MERGE"
echo "=== $(date) : done. Train with SOURCES='... ase' (docs/COMMANDS.md) ==="
