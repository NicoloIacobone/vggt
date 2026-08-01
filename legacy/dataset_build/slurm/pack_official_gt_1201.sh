#!/bin/bash
# Packing step for the 1201-scene extension (docs/todo.md 1c). Same QA gates + tar +
# verify + atomic-move pattern as pack_official_gt.sh, but NODE-LOCAL (docs/DATASET.md
# §6): the chunk tars written by extend_dataset_1201.sh are unpacked into $TMPDIR, the
# gates run there, and only the finished tar is copied to work. Nothing is ever
# materialised on scratch as loose files — the 1201-scene tree is ~1.26M inodes against a
# 1.0M soft / 1.5M hard file quota.
#
# Produces a NEW tar name (scannet_official_gt_1201.tar.zst) — does not touch
# scannet_official_gt_500.tar.zst, which other runs train against.
#
# Submit after all extend_dataset_1201.sh chunk jobs finish:
#   sbatch --dependency=afterok:<job1>:<job2>:<job3>:<job4> \
#       legacy/dataset_build/slurm/pack_official_gt_1201.sh
#
#SBATCH --job-name=pack_official_gt_1201
#SBATCH --output=/cluster/scratch/niacobone/vggt/slurm/logs/pack_official_gt_1201_%j.log
#SBATCH --error=/cluster/scratch/niacobone/vggt/slurm/logs/pack_official_gt_1201_%j.err
#SBATCH --open-mode=append
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4096
#SBATCH --tmp=120000
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=niacobone@student.ethz.ch

set -euo pipefail
REPO=/cluster/scratch/niacobone/vggt
CHUNKS=/cluster/scratch/niacobone/scannet_1201_chunks
WORK=/cluster/work/igp_psr/niacobone/distillation/dataset/scannet
EXPECT_SCENES=${EXPECT_SCENES:-1201}
OUT_TAR=${OUT_TAR:-scannet_official_gt_1201.tar.zst}
WORK_TAR="$WORK/$OUT_TAR"

# --tmp=120000 covers: unpacked tree (~31 GB at 1201 scenes) + the output tar (~24 GB).
# The chunk tars are streamed straight off scratch, never copied to the node.
BUILD=${TMPDIR:?TMPDIR unset — this job needs node-local scratch via #SBATCH --tmp}/build
mkdir -p "$BUILD"

cd "$REPO"

shopt -s nullglob
chunk_tars=("$CHUNKS"/chunk_*.tar.zst)
if [ ${#chunk_tars[@]} -eq 0 ]; then
    echo "[pack] no chunk tars in $CHUNKS — run extend_dataset_1201.sh first"
    exit 1
fi
for t in "${chunk_tars[@]}"; do
    echo "[pack] unpacking $(basename "$t") ..."
    tar --use-compress-program="zstd -d" -C "$BUILD" -xf "$t"
done
echo "[pack] build tree: $(find "$BUILD/scans" -mindepth 1 -maxdepth 1 -type d | wc -l) scenes, \
$(find "$BUILD/scans" -type f | wc -l) files"

# QA gates (exits non-zero on failure -> job fails before packing) + README.
myenv/bin/python legacy/dataset_build/scripts/gen_official_gt_report.py \
    --build "$BUILD" --out "$WORK/OFFICIAL_GT_README_1201.md" --expect_scenes "$EXPECT_SCENES"

# Visual spot-check strips (eyeballed separately; not a hard gate). Scenes spread evenly
# across data/splits/scannetv2_train.txt (indices 0, n/6, .., n-1). They are rendered into
# the node-local tree, so copy them out to work before it evaporates.
myenv/bin/python legacy/dataset_build/scripts/qa_official_gt_strips.py --build "$BUILD" \
    --scenes scene0191_00,scene0628_00,scene0614_00,scene0499_00,scene0171_01,scene0489_00,scene0567_01
mkdir -p "$WORK/qa_strips_1201"
cp "$BUILD"/qa_strips/*.jpg "$WORK/qa_strips_1201/"
echo "[pack] QA strips -> $WORK/qa_strips_1201"

echo "[pack] building archive ..."
STAGE_TAR=$TMPDIR/$OUT_TAR
tar --use-compress-program="zstd -1 -T0" -C "$BUILD" -cf "$STAGE_TAR" scans

echo "[pack] archive size: $(du -h "$STAGE_TAR" | cut -f1)"
n_tar=$(tar --use-compress-program="zstd -d" -tf "$STAGE_TAR" | grep -c '\.png$\|\.jpg$' || true)
n_src=$(find "$BUILD/scans" \( -name '*.png' -o -name '*.jpg' \) | wc -l)
echo "[pack] entries: archive=$n_tar source=$n_src"
if [ "$n_tar" != "$n_src" ]; then
    echo "[pack] COUNT MISMATCH — aborting before copy to work"
    exit 1
fi

# work is a shared group filesystem and runs close to full — fail loudly rather than
# leaving a truncated tar where a training job would pick it up.
avail_kb=$(df -Pk "$WORK" | awk 'NR==2 {print $4}')
need_kb=$(( $(stat -c %s "$STAGE_TAR") / 1024 + 2097152 ))
if [ "$avail_kb" -lt "$need_kb" ]; then
    echo "[pack] NOT ENOUGH SPACE on work: need ~$((need_kb/1024)) MB, have $((avail_kb/1024)) MB"
    exit 1
fi

cp "$STAGE_TAR" "$WORK_TAR.tmp" && mv "$WORK_TAR.tmp" "$WORK_TAR"
echo "[pack] done -> $WORK_TAR ($(du -h "$WORK_TAR" | cut -f1))"
echo "[pack] chunk tars in $CHUNKS are now redundant — delete them to reclaim scratch blocks."
