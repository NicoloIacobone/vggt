#!/bin/bash
# Dataset extension to the full official ScanNet v2 VAL split (docs/todo.md 1c, the
# counterpart of extend_dataset_1201.sh): for scenes [list_start..list_end] (0-based,
# inclusive indices into data/splits/scannetv2_val.txt, 312 scenes total), stream the
# .sens to extract the stride-5 subset frames (legacy/dataset_build/scripts/
# extract_sens_subset.py, early-abort streaming — no .sens ever touches disk), then
# download the official 2D GT zips and convert them into the build tree
# (legacy/dataset_build/scripts/download_2d_gt.py, zips deleted per scene).
#
# WHY: scannet_official_gt_1201.tar.zst holds only TRAIN-split scenes, so official-split
# training has no val ruler. The convention val scenes 0080-0089 that every earlier run
# used split 6 train / 4 official-val, i.e. they are not a valid official-split val set
# either. This tar is that ruler.
#
# NODE-LOCAL BUILD (docs/DATASET.md §5.1), identical discipline to the 1201 build even
# though 312 scenes (~0.33M files) would fit under the scratch 1.0M soft / 1.5M hard FILE
# quota on their own: the tree lives in $TMPDIR and only ONE compressed tar per chunk ever
# lands on scratch, costing 1 inode. Scratch has 2.4 TB of block quota — bytes are free,
# inodes are not.
#
# Resumable in two directions:
#   - within the tree: .subset_complete / .complete markers skip finished scenes;
#   - across jobs: the chunk tar is restored into $TMPDIR at startup and rewritten at the
#     end, so re-running the SAME chunk picks up exactly where it stopped. The tar is
#     rewritten even when a stage reports failures — that is the durability checkpoint —
#     and the job then exits non-zero.
#
# The official val split is NOT a contiguous scene0000_00.. range (includes _01/_02/...
# rescans, non-contiguous scene ids), so selection is by list index, not scene number —
# unlike extend_dataset_500.sh's [start..end] scene-number range.
#
# Nothing here touches the 500-scene or 1201-scene trees/tars on work.
#
# The whole split fits one chunk (stage 1 runs ~5 s/scene, stage 2 dominates at ~40 s/scene
# — cf. the 1201 build's chunk logs), so submit it as one job and let it chain the pack:
#   sbatch --export=ALL,CHAIN_PACK=1 legacy/dataset_build/slurm/extend_dataset_val312.sh 0 311
#
#SBATCH --job-name=extend_gt_val312
#SBATCH --output=/cluster/scratch/niacobone/vggt/slurm/logs/extend_gt_val312_%j.log
#SBATCH --error=/cluster/scratch/niacobone/vggt/slurm/logs/extend_gt_val312_%j.err
#SBATCH --open-mode=append
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4096
#SBATCH --tmp=40000
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=niacobone@student.ethz.ch

# eth_proxy gives compute nodes outbound network access for the download.
module load eth_proxy 2>/dev/null || module load stack/2024-06 eth_proxy 2>/dev/null || true
set -u

LIST_START=${1:-0}
LIST_END=${2:-311}
REPO=/cluster/scratch/niacobone/vggt
SCENE_LIST=$REPO/data/splits/scannetv2_val.txt
CHUNKS=/cluster/scratch/niacobone/scannet_val312_chunks
CHUNK_TAR=$CHUNKS/chunk_$(printf %04d "$LIST_START")_$(printf %04d "$LIST_END").tar.zst

# Node-local build tree. --tmp=40000 covers: restored tar read (streamed, not copied) +
# unpacked tree (~8 GB at 312 scenes) + transient zips + the output tar (~7 GB).
BUILD=${TMPDIR:?TMPDIR unset — this job needs node-local scratch via #SBATCH --tmp}/build
mkdir -p "$BUILD" "$CHUNKS"

cd "$REPO"

# Restore this chunk's previous state, if any, so finished scenes are skipped by marker.
if [ -f "$CHUNK_TAR" ]; then
    echo "[extend] restoring $CHUNK_TAR -> $BUILD ..."
    tar --use-compress-program="zstd -d" -C "$BUILD" -xf "$CHUNK_TAR"
    echo "[extend] restored $(find "$BUILD/scans" -mindepth 1 -maxdepth 1 -type d | wc -l) scenes, \
$(find "$BUILD/scans" -mindepth 3 -maxdepth 3 -name '.complete' | wc -l) already converted"
else
    echo "[extend] no existing $CHUNK_TAR — starting from scratch"
fi

# Stage 1: subset frames from the .sens streams.
myenv/bin/python legacy/dataset_build/scripts/extract_sens_subset.py \
    --out_root "$BUILD/scans" --scene_list "$SCENE_LIST" --start "$LIST_START" --end "$LIST_END"
SUBSET_RC=$?

# Clear convert residue from scenes that never reached .complete. build_official_masks
# .convert_scene() only ever mkdir(exist_ok=True)+overwrites — it never clears its output
# dir — so a conversion interrupted mid-write (quota error, wall clock, SIGTERM) leaves
# masks/ masks_instance/ _qa/ behind. Re-converting the same scene from the same zips is
# deterministic and would overwrite them all, but only as long as the inputs really are
# identical; wiping first makes that independent of the assumption. subset/ and
# .subset_complete are kept — that is the expensive .sens stream, and stage 1 owns it.
echo "[extend] clearing convert residue from unconverted scenes ..."
n_wiped=0
for rd in "$BUILD"/scans/*/raw_data; do
    [ -d "$rd" ] || continue
    [ -f "$rd/.complete" ] && continue
    if [ -d "$rd/masks" ] || [ -d "$rd/masks_instance" ] || [ -d "$rd/_qa" ]; then
        rm -rf "$rd/masks" "$rd/masks_instance" "$rd/_qa"
        n_wiped=$((n_wiped + 1))
    fi
done
echo "[extend] wiped partial conversions: $n_wiped scene(s)"

# Stage 2: official GT zips + conversion (frame list = the extracted subsets).
# Runs even if stage 1 had failures — it only converts scenes whose subset exists;
# a heal re-run picks up the rest. Zips are node-local and per-scene transient.
#
# Bounded by DL_BUDGET_H so the job always reaches the checkpoint tar below. Without this
# the 24 h wall clock would tear down the batch script mid-download and every scene
# converted in this run would be lost (same failure mode the COCO scripts document).
DL_BUDGET_S=$(( ${DL_BUDGET_H:-20} * 3600 ))   # integer hours; no python module loaded here
timeout --signal=INT "${DL_BUDGET_S}s" \
    myenv/bin/python legacy/dataset_build/scripts/download_2d_gt.py \
    --zips_dir "$TMPDIR/zips" \
    --convert_out "$BUILD/scans" \
    --subset_root "$BUILD/scans" \
    --scene_list "$SCENE_LIST" --start "$LIST_START" --end "$LIST_END"
CONVERT_RC=$?
if [ "$CONVERT_RC" = "124" ]; then
    echo "[extend] download stage hit the ${DL_BUDGET_H:-20}h budget — checkpointing and resubmitting"
fi

# Checkpoint: always persist whatever was built, even on partial failure, so the next
# re-run resumes instead of re-downloading. Written to a .tmp and moved into place so an
# interrupted copy can never truncate a good chunk tar.
echo "[extend] packing chunk -> $CHUNK_TAR ..."
rm -rf "$TMPDIR/zips"
n_src=$(find "$BUILD/scans" -type f | wc -l)
tar --use-compress-program="zstd -1 -T0" -C "$BUILD" -cf "$TMPDIR/chunk.tar.zst" scans
n_tar=$(tar --use-compress-program="zstd -d" -tf "$TMPDIR/chunk.tar.zst" | grep -cv '/$')
echo "[extend] entries: archive=$n_tar source=$n_src ($(du -h "$TMPDIR/chunk.tar.zst" | cut -f1))"
if [ "$n_tar" != "$n_src" ]; then
    echo "[extend] COUNT MISMATCH — refusing to overwrite $CHUNK_TAR"
    exit 1
fi
cp "$TMPDIR/chunk.tar.zst" "$CHUNK_TAR.tmp" && mv "$CHUNK_TAR.tmp" "$CHUNK_TAR"
echo "[extend] chunk saved: $CHUNK_TAR"

# How many of THIS chunk's scenes are converted? (Not the whole tree — a chunk tar only
# ever holds its own range, but be explicit so the resubmit condition is unambiguous.)
n_want=$(sed -n "$((LIST_START + 1)),$((LIST_END + 1))p" "$SCENE_LIST" | grep -c .)
n_done=0
while read -r s; do
    [ -f "$BUILD/scans/$s/raw_data/.complete" ] && n_done=$((n_done + 1))
done < <(sed -n "$((LIST_START + 1)),$((LIST_END + 1))p" "$SCENE_LIST" | grep .)
echo "[extend] converted $n_done / $n_want scenes in range $LIST_START..$LIST_END"

if [ "$n_done" -lt "$n_want" ]; then
    # Resubmit to heal the remainder. The next run restores this chunk tar, so only the
    # missing scenes are re-fetched. Capped, because a scene that is permanently
    # unavailable upstream can never reach .complete and would otherwise loop forever on
    # the group's allocation. RESUBMIT=0 disables it outright.
    n_try=$(( ${RESUBMIT_N:-0} + 1 ))
    if [ "${RESUBMIT:-1}" != "1" ]; then
        echo "[extend] incomplete — RESUBMIT=0, not resubmitting"
    elif [ "$n_try" -gt "${MAX_RESUBMITS:-5}" ]; then
        echo "[extend] incomplete after ${MAX_RESUBMITS:-5} resubmits — STOPPING."
        echo "[extend] $((n_want - n_done)) scene(s) never converted; check the logs above for"
        echo "[extend] whether they are download failures (retry) or gone upstream (accept and"
        echo "[extend] pack with EXPECT_SCENES=$n_done)."
    else
        echo "[extend] incomplete — resubmitting chunk $LIST_START..$LIST_END (attempt $n_try)"
        sbatch --export=ALL,RESUBMIT_N=$n_try \
            legacy/dataset_build/slurm/extend_dataset_val312.sh "$LIST_START" "$LIST_END"
    fi
    exit 1
fi

echo "[extend] chunk $LIST_START..$LIST_END COMPLETE"

# Because an incomplete chunk exits 1 and resubmits itself as a NEW job id, a
# --dependency=afterok chain onto the ORIGINAL id would never fire. So the completing job
# launches the packing step itself. Only meaningful when this chunk covers the whole
# split; with several parallel chunks leave CHAIN_PACK=0 and submit the pack by hand.
if [ "${CHAIN_PACK:-0}" = "1" ]; then
    echo "[extend] CHAIN_PACK=1 — submitting pack_official_gt_val312.sh"
    sbatch legacy/dataset_build/slurm/pack_official_gt_val312.sh
fi

exit $(( SUBSET_RC || CONVERT_RC ))
