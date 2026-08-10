#!/bin/bash
#
# Build the 2D instance-segmentation training set out of the InsScene-15K mirror (todo 6f).
#
#   sbatch slurm/build_insscene2d.sh                       # both sources, 32 frames/scene
#   sbatch --export=ALL,SOURCES=infinigen slurm/build_insscene2d.sh
#   sbatch --export=ALL,FRAMES=16,LIMIT=20 slurm/build_insscene2d.sh    # smoke run
#
# Knobs (via --export=ALL,VAR=...):
#   SOURCES   space-separated subset of "scannetpp infinigen"      (default: both)
#   FRAMES    frames kept per scene, evenly spaced                 (default: 32)
#   LIMIT     first N scenes per source, for smoke runs            (default: all)
#
# WHY IT BUILDS NODE-LOCAL AND SHIPS TARS -- read docs/DATASET.md §5.1 before changing this.
# Scratch is quota'd on FILE COUNT (1.0 M soft / 1.5 M hard), and this build writes
# (853 + 1466) scenes x 32 frames x 2 files ~= 148 k files. It therefore materialises the tree in
# $TMPDIR and lands ONE tar per source on work, costing 1 inode each.
#
# CPU-only: nothing here touches a GPU. The cost is I/O — seeking inside a 211 GiB split zip —
# plus JPEG/PNG re-encoding, so it asks for cores rather than memory bandwidth. The 32 GB is for
# the ScanNet++ central directory: 3.39 M entries parsed into python objects.
#
#SBATCH --job-name=build_insscene2d
#SBATCH --output=/cluster/scratch/niacobone/vggt/slurm/logs/build_insscene2d_%j.log
#SBATCH --error=/cluster/scratch/niacobone/vggt/slurm/logs/build_insscene2d_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4096
#SBATCH --tmp=60000
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
