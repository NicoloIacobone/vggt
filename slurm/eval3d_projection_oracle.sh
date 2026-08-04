#!/bin/bash
#
# The oracle that licenses `--transfer_mode gt_projection` (docs/MASKDINO.md §9.9).
#
#   sbatch slurm/eval3d_projection_oracle.sh
#   sbatch --export=ALL,EXTRA_ARGS='--num_scenes 20' slurm/eval3d_projection_oracle.sh
#
# Renders the 3D GT into every view through the transfer's OWN projection and feeds it back
# as predictions: round-trip purity must be ~1.000 or the pixel mapping is wrong. No model,
# no GPU, no checkpoint — this measures the eval machinery, not a network.
#
# Knobs (via --export=ALL,VAR=...):
#   OUT          where the json goes (default: next to the val-312 tars' README on work)
#   EXTRA_ARGS   appended verbatim (e.g. '--num_scenes 20', '--depth_tolerance 0.05')
#
#SBATCH --job-name=eval3d_oracle
#SBATCH --output=/cluster/scratch/niacobone/vggt/slurm/logs/eval3d_oracle_%j.log
#SBATCH --error=/cluster/scratch/niacobone/vggt/slurm/logs/eval3d_oracle_%j.err
#SBATCH --open-mode=append
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8192
#SBATCH --tmp=16000
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=niacobone@student.ethz.ch

module purge
module load stack/2024-06 python/3.12.8 eth_proxy
cd /cluster/scratch/niacobone/vggt
source myenv/bin/activate
PYTHON=myenv/bin/python
export PYTHONUNBUFFERED=1

WORK=/cluster/work/igp_psr/niacobone/distillation/dataset/scannet
GT_TAR=${GT_TAR:-$WORK/scannet_3d_gt_val312.tar.zst}
FRAMES_TAR=${FRAMES_TAR:-$WORK/scannet_frames25k_val312.tar.zst}
OUT=${OUT:-/cluster/work/igp_psr/niacobone/distillation/output/eval3d_projection_oracle.json}

STAGE=${TMPDIR:?TMPDIR unset — this job needs node-local scratch via #SBATCH --tmp}
for t in "$GT_TAR" "$FRAMES_TAR"; do
    [ -f "$t" ] || { echo "missing tar: $t (run the legacy/dataset_build download jobs)"; exit 1; }
    echo "[stage] $(basename "$t") -> $STAGE"
    tar --use-compress-program="zstd -d -T0" -C "$STAGE" -xf "$t"
done

$PYTHON scripts/eval3d_projection_oracle.py \
    --gt_root "$STAGE/scans3d" \
    --frames_root "$STAGE/scans25k" \
    --out "$OUT" \
    ${EXTRA_ARGS:-}
