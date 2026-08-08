#!/bin/bash
# ScanNet++ v2 validation set (50 scenes) -> the two tars the 3D eval consumes
# (docs/todo.md 6c, docs/DATASET.md §2.1). Modelled on extend_dataset_val312.sh.
#
# WHY: ScanNet++ is the evaluation dataset of all three direct competitors — SegVGGT
# (zero-shot), FAST3DIS (zero-shot) and IGGT — and we have no ScanNet++ numbers at all
# (docs/TRAINING_COMPARABILITY.md). This is the ruler, not a training set.
#
# Two stages, two deliverables, both under
# /cluster/work/igp_psr/niacobone/distillation/dataset/scannetpp/ :
#
#   scannetpp_3d_gt_val50.tar.zst     scans3d/<scene>/{mesh.ply,segments.json,
#                                     segments_anno.json} + scans3d/_metadata/
#   scannetpp_frames_val50.tar.zst    scans25k/<scene>/{color,depth,pose,intrinsic,
#                                     manifest.json}, 50 uniformly sampled frames
#
# The layout mirrors scannet_3d_gt_val312.tar.zst / scannet_frames25k_val312.tar.zst so
# the evaluator needs no new file conventions.
#
# SOURCE: /cluster/work/igp_psr/nedela/scannetpp_data — ANOTHER USER'S TREE. It is read
# exactly once, here. Everything downstream reads our tars; the class tables and the split
# file are copied into the GT tar for the same reason.
#
# NODE-LOCAL BUILD (docs/DATASET.md §5.1): the tree lives in $TMPDIR and SCRATCH COSTS ZERO
# LOOSE FILES. Unlike the ScanNet builds there is no chunk tar on scratch either — the
# source is on `work`, not the network, so the deliverable tars themselves are the
# checkpoint (see "resume" below), and scratch is touched not at all.
#
# --tmp=40000 covers: scans3d (~4.4 GB of ply+json) + scans25k (~1.0 GB) + a restored tar
# read + the two output tars (~4 GB), with headroom.
#
# RESUMABLE, in two directions:
#   - within the tree: per-scene .complete markers skip finished scenes;
#   - across jobs: an existing tar on work is restored into $TMPDIR at startup. While a
#     stage is incomplete its tar is written as <name>.partial.tar.zst and only renamed to
#     the deliverable name when all scenes are done, so a partial build can never
#     masquerade as the finished dataset. The tar is rewritten even after a partial
#     failure — that is the durability checkpoint — and the job then exits non-zero.
#
# Both stages fail a scene rather than shipping it when its geometry self-check fails
# (build_scannetpp_frames.py: unprojected depth vs the mesh, and the RGB/pose index
# sweep). A missing scene is recoverable; a silently misaligned one is not.
#
# The tars keep the `val50` name (the official split IS 50 scenes) but ship 49: see EXCLUDE
# below. Both tars hold the SAME scene list.
#
# Usage (one job does the whole split, ~50 min):
#   sbatch legacy/dataset_build/slurm/build_scannetpp_val50.sh
#   sbatch --export=ALL,CHAIN_VERIFY=0 legacy/dataset_build/slurm/build_scannetpp_val50.sh
#   sbatch --export=ALL,EXCLUDE= legacy/dataset_build/slurm/build_scannetpp_val50.sh  # all 50
#
#SBATCH --job-name=build_scannetpp_val50
#SBATCH --output=/cluster/scratch/niacobone/vggt/slurm/logs/build_scannetpp_val50_%j.log
#SBATCH --error=/cluster/scratch/niacobone/vggt/slurm/logs/build_scannetpp_val50_%j.err
#SBATCH --open-mode=append
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4096
#SBATCH --tmp=40000
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=niacobone@student.ethz.ch

set -u

REPO=/cluster/scratch/niacobone/vggt
SRC=${SRC_ROOT:-/cluster/work/igp_psr/nedela/scannetpp_data}
OUT=${OUT_DIR:-/cluster/work/igp_psr/niacobone/distillation/dataset/scannetpp}
SCENE_LIST=$SRC/splits/nvs_sem_val.txt
LIST_START=${1:-0}
LIST_END=${2:-49}
NUM_FRAMES=${NUM_FRAMES:-50}

# Scenes the UPSTREAM release ships broken. Excluded from both tars so the two always hold
# the same scene list — a GT scene with no frames is a landmine for the evaluator.
#   d755b3d9d8: its iphone trajectory diverges. `aligned_pose` translations reach 7.2 km
#   against a 5.3 x 4.0 x 3.6 m mesh, and only 143 of 8863 frames put the camera within 3 m
#   of the mesh bbox at all. The geometry self-check caught it at 3.9 km (job 10089394);
#   nothing about it is recoverable by resampling. Not a build defect.
EXCLUDE=${EXCLUDE-d755b3d9d8}

GT_TAR=$OUT/scannetpp_3d_gt_val50.tar.zst
FR_TAR=$OUT/scannetpp_frames_val50.tar.zst

BUILD=${TMPDIR:?TMPDIR unset — this job needs node-local scratch via #SBATCH --tmp}/build
mkdir -p "$BUILD" "$OUT" "$REPO/slurm/logs"
cd "$REPO"

n_want=$(sed -n "$((LIST_START + 1)),$((LIST_END + 1))p" "$SCENE_LIST" | grep -c .)
for e in $EXCLUDE; do n_want=$((n_want - 1)); done
[ -n "$EXCLUDE" ] && echo "[scannetpp] excluding broken upstream scene(s): $EXCLUDE"
echo "[scannetpp] building scenes $LIST_START..$LIST_END ($n_want) from $SRC"
echo "[scannetpp] scratch quota BEFORE:"
lfs quota -h -u "$USER" /cluster/scratch/niacobone 2>/dev/null | tail -2

# ---------------------------------------------------------------------------------------
# restore(): put a previous run's tree back into $TMPDIR so .complete markers apply.
# Prefers the finished tar; falls back to a .partial from an interrupted run.
# ---------------------------------------------------------------------------------------
restore() {
    local final=$1 sub=$2
    for cand in "$final" "${final%.tar.zst}.partial.tar.zst"; do
        if [ -f "$cand" ]; then
            echo "[scannetpp] restoring $cand -> $BUILD ..."
            tar --use-compress-program="zstd -d" -C "$BUILD" -xf "$cand" || return 1
            echo "[scannetpp] restored $(find "$BUILD/$sub" -mindepth 2 -maxdepth 2 \
-name '.complete' 2>/dev/null | wc -l) completed scene(s) into $sub"
            return 0
        fi
    done
    echo "[scannetpp] no existing tar for $sub — starting from scratch"
}

# ---------------------------------------------------------------------------------------
# pack(): tar the subtree, verify the entry count against the source tree, then place it
# at the FINAL name only when every requested scene is complete. Written to a .tmp and
# moved, so an interrupted copy can never truncate a good tar.
# ---------------------------------------------------------------------------------------
pack() {
    local sub=$1 final=$2 n_done=$3
    local partial="${final%.tar.zst}.partial.tar.zst"
    local dest="$final"
    [ "$n_done" -lt "$n_want" ] && dest="$partial"

    echo "[scannetpp] packing $sub -> $dest ..."
    local n_src n_tar
    n_src=$(find "$BUILD/$sub" -type f | wc -l)
    tar --use-compress-program="zstd -3 -T0" -C "$BUILD" -cf "$TMPDIR/$sub.tar.zst" "$sub"
    n_tar=$(tar --use-compress-program="zstd -d" -tf "$TMPDIR/$sub.tar.zst" | grep -cv '/$')
    echo "[scannetpp] $sub entries: archive=$n_tar source=$n_src ($(du -h "$TMPDIR/$sub.tar.zst" | cut -f1))"
    if [ "$n_tar" != "$n_src" ]; then
        echo "[scannetpp] COUNT MISMATCH — refusing to write $dest"
        return 1
    fi
    cp "$TMPDIR/$sub.tar.zst" "$dest.tmp" && mv "$dest.tmp" "$dest"
    rm -f "$TMPDIR/$sub.tar.zst"
    # A completed build supersedes any leftover .partial.
    [ "$dest" = "$final" ] && rm -f "$partial"
    echo "[scannetpp] wrote $dest"
}

count_done() { find "$BUILD/$1" -mindepth 2 -maxdepth 2 -name '.complete' 2>/dev/null | wc -l; }

# ---------------------------------------------------------------------------------------
# Stage 1 — 3D GT (fast: a file copy plus validation, ~2 s/scene)
# ---------------------------------------------------------------------------------------
restore "$GT_TAR" scans3d
myenv/bin/python legacy/dataset_build/scripts/build_scannetpp_3d_gt.py \
    --src_root "$SRC" --out_root "$BUILD/scans3d" --scene_list "$SCENE_LIST" \
    --start "$LIST_START" --end "$LIST_END" --exclude_scenes $EXCLUDE
GT_RC=$?
GT_DONE=$(count_done scans3d)
pack scans3d "$GT_TAR" "$GT_DONE" || GT_RC=1
echo "[scannetpp] stage 1: $GT_DONE / $n_want scenes"

# ---------------------------------------------------------------------------------------
# Stage 2 — iphone frames (~20 s/scene: video seeks, LZ4 depth, two geometry checks)
# ---------------------------------------------------------------------------------------
restore "$FR_TAR" scans25k
myenv/bin/python legacy/dataset_build/scripts/build_scannetpp_frames.py \
    --src_root "$SRC" --out_root "$BUILD/scans25k" --scene_list "$SCENE_LIST" \
    --start "$LIST_START" --end "$LIST_END" --num_frames "$NUM_FRAMES" \
    --exclude_scenes $EXCLUDE
FR_RC=$?
FR_DONE=$(count_done scans25k)
pack scans25k "$FR_TAR" "$FR_DONE" || FR_RC=1
echo "[scannetpp] stage 2: $FR_DONE / $n_want scenes"

echo "[scannetpp] scratch quota AFTER:"
lfs quota -h -u "$USER" /cluster/scratch/niacobone 2>/dev/null | tail -2

if [ "$GT_DONE" -lt "$n_want" ] || [ "$FR_DONE" -lt "$n_want" ]; then
    echo "[scannetpp] INCOMPLETE — re-run this script to resume from the .partial tars."
    echo "[scannetpp] Check the logs above for whether the missing scenes failed a"
    echo "[scannetpp] geometry check (do NOT ship them) or just ran out of wall clock."
    exit 1
fi

echo "[scannetpp] BUILD COMPLETE — $n_want scenes in both tars"

if [ "${CHAIN_VERIFY:-1}" = "1" ]; then
    echo "[scannetpp] CHAIN_VERIFY=1 — verifying the built tree in place"
    myenv/bin/python scripts/verify_scannetpp_gt.py \
        --gt_root "$BUILD/scans3d" --frames_root "$BUILD/scans25k" \
        --num_scenes "${VERIFY_SCENES:-5}"
    VERIFY_RC=$?
    echo "[scannetpp] verify exit $VERIFY_RC"
    exit $(( GT_RC || FR_RC || VERIFY_RC ))
fi

exit $(( GT_RC || FR_RC ))
