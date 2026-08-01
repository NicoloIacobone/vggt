#!/bin/bash
# ONE-OFF RESCUE (docs/DATASET.md §6): fold the on-scratch 1201-scene build tree into a
# single chunk tar and delete the tree, reclaiming ~1.26M inodes.
#
# Why this exists: the first 1201-scene attempt (2026-07-30) materialised the whole build
# tree on scratch. At ~1046 files/scene that is ~1.26M inodes against a 1.0M soft / 1.5M
# hard file quota, so the run died with OSError(122, 'Disk quota exceeded') after 1090 of
# 1201 scenes. The rebuild pattern is now node-local (extend_dataset_1201.sh); this script
# converts what already exists into that pattern's currency — a chunk tar — instead of
# throwing away ~90 node-hours of .sens streaming.
#
# Scratch's block quota is 2.4 TB and barely used; only the FILE count is scarce. One tar
# is one inode, so the snapshot costs ~24 GB of blocks and 1 inode.
#
# The tree is deleted ONLY after the tar is verified to contain every regular file. Set
# KEEP_TREE=1 to skip the delete (then the inodes are NOT reclaimed).
#
#   sbatch legacy/dataset_build/slurm/snapshot_build_1201.sh
#
#SBATCH --job-name=snapshot_1201
#SBATCH --output=/cluster/scratch/niacobone/vggt/slurm/logs/snapshot_1201_%j.log
#SBATCH --error=/cluster/scratch/niacobone/vggt/slurm/logs/snapshot_1201_%j.err
#SBATCH --open-mode=append
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4096
#SBATCH --tmp=8000
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=niacobone@student.ethz.ch

set -euo pipefail

BUILD=/cluster/scratch/niacobone/scannet_official_build_1201
CHUNKS=/cluster/scratch/niacobone/scannet_1201_chunks
CHUNK_TAR=$CHUNKS/chunk_0000_1200.tar.zst

if [ ! -d "$BUILD/scans" ]; then
    echo "[snapshot] $BUILD/scans does not exist — nothing to do (already snapshotted?)"
    exit 0
fi
mkdir -p "$CHUNKS"

# Intermediate zips are per-scene temporaries the converter deletes on success; any left
# behind are from interrupted scenes and are not part of the dataset.
rm -rf "$BUILD/zips"

echo "[snapshot] counting source files (1.2M inodes on Lustre, this takes a few minutes) ..."
n_src=$(find "$BUILD/scans" -type f | wc -l)
n_scenes=$(find "$BUILD/scans" -mindepth 1 -maxdepth 1 -type d | wc -l)
n_complete=$(find "$BUILD/scans" -mindepth 3 -maxdepth 3 -name '.complete' | wc -l)
echo "[snapshot] source: $n_scenes scenes, $n_complete converted (.complete), $n_src files"

echo "[snapshot] building $CHUNK_TAR ..."
tar --use-compress-program="zstd -1 -T0" -C "$BUILD" -cf "$CHUNK_TAR.tmp" scans
echo "[snapshot] size: $(du -h "$CHUNK_TAR.tmp" | cut -f1)"

# Verify EVERY regular file survived, not just the png/jpg bulk: the markers
# (.complete/.subset_complete) and _qa/stats.json are what make the tree resumable and
# what the QA gate reads, and losing them silently would be unrecoverable after the rm.
echo "[snapshot] verifying archive ..."
n_tar=$(tar --use-compress-program="zstd -d" -tf "$CHUNK_TAR.tmp" | grep -cv '/$')
echo "[snapshot] entries: archive=$n_tar source=$n_src"
if [ "$n_tar" != "$n_src" ]; then
    echo "[snapshot] COUNT MISMATCH — keeping the tree, removing the partial archive"
    rm -f "$CHUNK_TAR.tmp"
    exit 1
fi

mv "$CHUNK_TAR.tmp" "$CHUNK_TAR"
echo "[snapshot] verified -> $CHUNK_TAR"

if [ "${KEEP_TREE:-0}" = "1" ]; then
    echo "[snapshot] KEEP_TREE=1 — tree left in place, inodes NOT reclaimed"
    exit 0
fi

echo "[snapshot] deleting $BUILD (reclaiming ~$n_src inodes) ..."
rm -rf "$BUILD"
echo "[snapshot] done. Quota now:"
lfs quota -u "$USER" /cluster/scratch/"$USER" 2>/dev/null || true
