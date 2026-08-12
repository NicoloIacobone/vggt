#!/bin/bash
#
# The §9.2 licence gate for one dataset's 3D GT, over ALL its scenes (docs/todo.md 6d).
# CPU-only, no GPU, no checkpoint.
#
#   sbatch --export=ALL,DATASET=scannetpp slurm/gate_3d_gt.sh
#
# Stages the dataset's two tars node-local and feeds its own GT back as predictions: the
# official evaluator must answer exactly 1.000 / 1.000 / 1.000, and the sensor depth must
# land on the mesh. **No dataset ships a number in docs/RESULTS.md §7 until this passes.**
# The report lands next to the tars as gate_<dataset>.json.
#
# Knobs: DATASET (scannetv2|scannet200|scannetpp|replica), EXTRA_ARGS.
#
#SBATCH --job-name=gate3d
#SBATCH --output=/cluster/scratch/niacobone/vggt/slurm/logs/gate3d_%j.log
#SBATCH --error=/cluster/scratch/niacobone/vggt/slurm/logs/gate3d_%j.err
#SBATCH --open-mode=append
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8192
#SBATCH --tmp=16000
#SBATCH --mail-type=FAIL

module purge
module load stack/2024-06 python/3.12.8 eth_proxy
cd /cluster/scratch/niacobone/vggt
PYTHON=myenv/bin/python
export PYTHONUNBUFFERED=1

DATASET=${DATASET:-scannetv2}
WORK=/cluster/work/igp_psr/niacobone/distillation/dataset
case "$DATASET" in
    scannetv2|scannet200)
        GT_TAR=$WORK/scannet/scannet_3d_gt_val312.tar.zst
        FRAMES_TAR=$WORK/scannet/scannet_frames25k_val312.tar.zst
        OUT_DIR=$WORK/scannet ;;
    scannetpp)
        GT_TAR=$WORK/scannetpp/scannetpp_3d_gt_val50.tar.zst
        FRAMES_TAR=$WORK/scannetpp/scannetpp_frames_val50.tar.zst
        OUT_DIR=$WORK/scannetpp ;;
    replica)
        GT_TAR=$WORK/replica/replica_3d_gt_8.tar.zst
        FRAMES_TAR=$WORK/replica/replica_frames_8.tar.zst
        OUT_DIR=$WORK/replica ;;
    *) echo "unknown DATASET=$DATASET"; exit 1 ;;
esac

STAGE=${TMPDIR:?TMPDIR unset — this job needs node-local scratch via #SBATCH --tmp}
for t in "$GT_TAR" "$FRAMES_TAR"; do
    [ -f "$t" ] || { echo "missing tar: $t"; exit 1; }
    echo "[stage] $(basename "$t") -> $STAGE"
    tar --use-compress-program="zstd -d -T0" -C "$STAGE" -xf "$t"
done

$PYTHON scripts/gate_3d_gt.py \
    --dataset "$DATASET" \
    --gt_root "$STAGE/scans3d" \
    --frames_root "$STAGE/scans25k" \
    --report_superpoints \
    --out "$OUT_DIR/gate_${DATASET}.json" \
    ${EXTRA_ARGS:-}
