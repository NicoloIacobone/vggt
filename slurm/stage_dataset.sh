# slurm/stage_dataset.sh — stage the ScanNet instance dataset onto node-local scratch.
#
# Per INSTANCE_MASKS_README.md, the dataset ships as a single ~1.3 GB zstd-compressed
# tar (one big file lives well on the work filesystem; reading the thousands of small
# PNGs directly off `work` is slow and pressures the inode quota). Each job copies that
# one archive to the compute node's local SSD ($TMPDIR) and unpacks it there once, then
# reads everything off fast local disk.
#
# Usage: `source slurm/stage_dataset.sh` after activating the venv. Exports SCANNET_ROOT,
# which scripts/train_multiscene.py picks up as the default --scans_root.
#
# Requires `zstd` (present at /usr/bin/zstd on the GPU nodes; no module needed). Request
# enough node-local scratch in the SBATCH header: the tar (1.3 GB) + unpacked tree
# (~2.7 GB) need ~4 GB, so `#SBATCH --tmp=8000` (MB) is comfortable.

set -euo pipefail

DATA_TAR="${DATA_TAR:-/cluster/work/igp_psr/niacobone/distillation/dataset/scannet/scannet_instance_dataset.tar.zst}"
STAGE_DIR="${TMPDIR:-/tmp}"
TAR_BASENAME="$(basename "$DATA_TAR")"

echo "[stage] copying $TAR_BASENAME -> $STAGE_DIR ..."
cp "$DATA_TAR" "$STAGE_DIR/"

echo "[stage] unpacking with zstd ..."
tar --use-compress-program="zstd -d" -C "$STAGE_DIR" -xf "$STAGE_DIR/$TAR_BASENAME"
rm -f "$STAGE_DIR/$TAR_BASENAME"   # reclaim the 1.3 GB; the unpacked tree is what we read

export SCANNET_ROOT="$STAGE_DIR/scans"
echo "[stage] SCANNET_ROOT=$SCANNET_ROOT ($(ls "$SCANNET_ROOT" | wc -l) scenes)"
