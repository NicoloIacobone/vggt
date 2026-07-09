# slurm/stage_dataset.sh — stage the ScanNet instance dataset onto node-local scratch.
#
# The dataset ships as a single zstd-compressed tar. Which tar is staged is
# controlled by the DATA_TAR env var:
#   - default here: the SAM3-GT tar `scannet_instance_dataset_full.tar.zst`
#     (200 scenes, ~2.6 GB; unpacked ~5.4 GB) — backward compatible.
#   - the train SLURM scripts override it to the official-ScanNet-GT tar
#     `scannet_official_gt_500.tar.zst` (500 scenes, the current default
#     supervision; see docs/old/OFFICIAL_GT_MIGRATION_PLAN.md — they request
#     --tmp=24000 for it). All tars unpack to `scans/<scene>/raw_data/...`.
# One big file lives well on the work filesystem; reading the thousands of small PNGs
# directly off `work` is slow and pressures the inode quota. Each job copies that one
# archive to the compute node's local SSD ($TMPDIR) and unpacks it there once, then reads
# everything off fast local disk. The tar contains `scans/<scene>/raw_data/...`.
#
# Usage: `source slurm/stage_dataset.sh` after activating the venv. Exports SCANNET_ROOT,
# which scripts/train_multiscene.py picks up as the default --scans_root.
#
# Requires `zstd` (present at /usr/bin/zstd on the GPU nodes; no module needed). Request
# enough node-local scratch in the SBATCH header: peak = tar + unpacked tree before the
# tar is deleted — ≈ 8 GB for the 200-scene tars (`--tmp=16000` MB comfortable), ≈ 19 GB
# for the 500-scene official tar (`--tmp=24000` MB, what the train scripts request).

set -euo pipefail

DATA_TAR="${DATA_TAR:-/cluster/work/igp_psr/niacobone/distillation/dataset/scannet/scannet_instance_dataset_full.tar.zst}"
STAGE_DIR="${TMPDIR:-/tmp}"
TAR_BASENAME="$(basename "$DATA_TAR")"

echo "[stage] copying $TAR_BASENAME -> $STAGE_DIR ..."
cp "$DATA_TAR" "$STAGE_DIR/"

echo "[stage] unpacking with zstd ..."
tar --use-compress-program="zstd -d" -C "$STAGE_DIR" -xf "$STAGE_DIR/$TAR_BASENAME"
rm -f "$STAGE_DIR/$TAR_BASENAME"   # reclaim the 1.3 GB; the unpacked tree is what we read

export SCANNET_ROOT="$STAGE_DIR/scans"
echo "[stage] SCANNET_ROOT=$SCANNET_ROOT ($(ls "$SCANNET_ROOT" | wc -l) scenes)"
