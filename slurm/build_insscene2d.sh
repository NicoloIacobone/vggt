#!/bin/bash
#
# Build the 2D instance-segmentation training set out of the InsScene-15K mirror (todo 6f).
#
#   sbatch slurm/build_insscene2d.sh                       # scannetpp + infinigen, 32 frames
#   sbatch --export=ALL,SOURCES=infinigen slurm/build_insscene2d.sh
#   sbatch --export=ALL,SOURCES=re10k slurm/build_insscene2d.sh          # the SAM2-supervised arm
#   sbatch --export=ALL,FRAMES=16,LIMIT=20 slurm/build_insscene2d.sh    # smoke run
#
# Knobs (via --export=ALL,VAR=...):
#   SOURCES   space-separated subset of "scannetpp infinigen re10k" (default: the first two —
#             re10k is NOT in the default because its masks are SAM2 output rather than ground
#             truth and every row trained on it carries that caveat, docs/MULTIDATASET.md §1.3)
#   FRAMES    frames kept per scene, evenly spaced                 (default: 32)
#   LIMIT     first N scenes per source, for smoke runs            (default: all)
#
# WHY IT BUILDS NODE-LOCAL AND SHIPS TARS -- read docs/DATASET.md §5.1 before changing this.
# Scratch is quota'd on FILE COUNT (1.0 M soft / 1.5 M hard), and this build writes
# (853 + 1466) scenes x 32 frames x 2 files ~= 148 k files -- and re10k alone another 333 k
# (5127 x 65). It therefore materialises the tree in $TMPDIR and lands ONE tar per source on
# work, costing 1 inode each. Measured: re10k is ~1.96 MB/scene, so ~10 GB of $TMPDIR and a
# ~10 GB tar.
#
# CPU-only: nothing here touches a GPU. The cost is I/O — seeking inside a 211 GiB split zip —
# plus JPEG/PNG re-encoding, so it asks for cores rather than memory bandwidth. The 32 GB is for
# the ScanNet++ central directory: 3.39 M entries parsed into python objects.
#
# TIME. The scannetpp + infinigen build was 1 h 42 (job 10286143). re10k is single-process like
# the others and measured at ~2.5-3 s/scene x 5127 scenes ~= 4 h, so the wall clock below is 24 h
# rather than 12: it is a ceiling, not a reservation, and a build that dies at hour 12 costs the
# whole pass.
#
#SBATCH --job-name=build_insscene2d
#SBATCH --output=/cluster/scratch/niacobone/vggt/slurm/logs/build_insscene2d_%j.log
#SBATCH --error=/cluster/scratch/niacobone/vggt/slurm/logs/build_insscene2d_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4096
#SBATCH --tmp=120000
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=niacobone@student.ethz.ch

set -euo pipefail

REPO=/cluster/scratch/niacobone/vggt
DEST=/cluster/work/igp_psr/niacobone/distillation/dataset/insscene2d
SOURCES="${SOURCES:-scannetpp infinigen}"
FRAMES="${FRAMES:-32}"
LIMIT_ARG=""
[ -n "${LIMIT:-}" ] && LIMIT_ARG="--limit ${LIMIT}"

cd "$REPO"
mkdir -p "$DEST"
BUILD="${TMPDIR:?TMPDIR unset — this build must not run on scratch}/insscene2d"
mkdir -p "$BUILD"

echo "=== node $(hostname) | build $BUILD | dest $DEST | frames $FRAMES ==="
df -h "$TMPDIR" | tail -1

for SRC in $SOURCES; do
    echo "=== $SRC ==="
    # No exclusion list for infinigen or re10k, and that is a fact about the benchmarks rather
    # than an omission: neither is one of the four datasets docs/RESULTS.md §7 evaluates on, so
    # there is nothing either could leak. Only ScanNet++ is both trained on and evaluated on.
    EXCLUDE=""
    if [ "$SRC" = "scannetpp" ]; then
        # NON-NEGOTIABLE: the mirror contains every scene of our ScanNet++ eval column
        # (docs/RESULTS.md §7). Training on them would leak the whole zero-shot benchmark.
        EXCLUDE="--exclude_scenes data/splits/scannetpp_nvs_sem_val.txt"
    fi
    myenv/bin/python slurm/build_insscene2d.py \
        --source "$SRC" --out "$BUILD" --frames "$FRAMES" $EXCLUDE $LIMIT_ARG

    echo "--- packing $SRC ---"
    tar -C "$BUILD" -cf - "$SRC" "REPORT_${SRC}.json" \
        | zstd -T8 -3 -o "$DEST/insscene2d_${SRC}.tar.zst" -f
    ls -lh "$DEST/insscene2d_${SRC}.tar.zst"
    cp "$BUILD/REPORT_${SRC}.json" "$DEST/"
done

echo "=== inventory ==="
find "$BUILD" -mindepth 2 -maxdepth 2 -type d | wc -l
du -sh "$BUILD"
ls -lh "$DEST"
echo "=== DONE ==="
