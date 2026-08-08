#!/bin/bash
#
# Deliverable 1 (external dataset acquisition, docs/todo.md): Replica, 8 scenes
# (room0, room1, room2, office0..office4) as the third eval dataset for FAST3DIS parity.
#
# Two sources, for two different reasons (see build_replica.py for what was actually
# verified about each, not assumed):
#   - facebookresearch/Replica-Dataset (official release, CC-BY-NC-4.0): ships
#     habitat/mesh_semantic.ply + habitat/info_semantic.json per scene -- the GT mesh with
#     per-face instance+semantic annotation. This is scans3d/.
#   - HuggingFace kxic/vMAP vmap.zip (44.8 GB, the same 8 scenes, traj 00 == the iMAP
#     trajectory): pre-rendered RGB/depth/pose. This is scans25k/.
#
# Everything materialises in $TMPDIR (node-local, not quota'd); only the two output tars
# land on /cluster/work. Zero loose files touch /cluster/scratch.
#
#SBATCH --job-name=fetch_replica
#SBATCH --output=/cluster/scratch/niacobone/vggt/slurm/logs/fetch_replica_%j.log
#SBATCH --error=/cluster/scratch/niacobone/vggt/slurm/logs/fetch_replica_%j.err
#SBATCH --open-mode=append
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4096
#SBATCH --tmp=140000
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=niacobone@student.ethz.ch

set -euo pipefail
module purge
module load stack/2024-06 python/3.12.8 eth_proxy
cd /cluster/scratch/niacobone/vggt
PYTHON=/cluster/scratch/niacobone/vggt/myenv/bin/python
export PYTHONUNBUFFERED=1

OUT_DIR=/cluster/work/igp_psr/niacobone/distillation/dataset/replica
mkdir -p "$OUT_DIR"

WORK="$TMPDIR/replica_build"
mkdir -p "$WORK"
cd "$WORK"

echo "=== $(date) : node $(hostname), TMPDIR=$TMPDIR ==="
df -h "$TMPDIR"

SCENES="room_0 room_1 room_2 office_0 office_1 office_2 office_3 office_4"

########################################
# 1. Official Replica-Dataset: mesh + habitat semantic annotation only, for our 8 scenes.
########################################
mkdir -p replica_parts
echo "=== downloading official Replica-Dataset release (17 parts, ~34 GB) ==="
for suf in aa ab ac ad ae af ag ah ai aj ak al am an ao ap aq; do
    f="replica_v1_0.tar.gz.part$suf"
    if [ ! -f "replica_parts/$f.ok" ]; then
        wget --continue -q -O "replica_parts/$f" \
            "https://github.com/facebookresearch/Replica-Dataset/releases/download/v1.0/$f"
        touch "replica_parts/$f.ok"
    fi
    echo "  got $f"
done

echo "=== extracting only $SCENES (habitat mesh + preseg) from the concatenated stream ==="
mkdir -p replica_orig
PATTERNS=()
for s in $SCENES; do
    PATTERNS+=("$s/habitat/mesh_semantic.ply" "$s/habitat/info_semantic.json" \
               "$s/mesh.ply" "$s/textures" "$s/preseg.json" "$s/preseg.bin")
done
cat replica_parts/replica_v1_0.tar.gz.part* | tar -xz -C replica_orig \
    --wildcards --ignore-command-error --wildcards-match-slash \
    "${PATTERNS[@]}" 2>&1 | grep -v "^tar:.*Not found in archive" || true

echo "=== replica_orig tree ==="
find replica_orig -maxdepth 3 | sort

# free the 34 GB of parts now that extraction is done
rm -rf replica_parts

########################################
# 2. vMAP pre-rendered sequences (RGB/depth/pose), same 8 scenes.
########################################
echo "=== downloading kxic/vMAP vmap.zip (44.8 GB) ==="
wget --continue -q -O vmap.zip \
    "https://huggingface.co/datasets/kxic/vMAP/resolve/main/vmap.zip"
echo "=== vmap.zip top-level listing ==="
unzip -l vmap.zip > vmap_listing.txt
head -60 vmap_listing.txt

mkdir -p vmap_extracted
echo "=== extracting vmap.zip ==="
unzip -q vmap.zip -d vmap_extracted
rm -f vmap.zip
echo "=== vmap_extracted tree (depth 3) ==="
find vmap_extracted -maxdepth 3 | sort > vmap_extracted_tree.txt
head -100 vmap_extracted_tree.txt

########################################
# 3. Inspect + repack (all real parsing/decisions happen in build_replica.py, not here)
########################################
$PYTHON /cluster/scratch/niacobone/vggt/slurm/build_replica.py \
    --replica_orig "$WORK/replica_orig" \
    --vmap_extracted "$WORK/vmap_extracted" \
    --scenes $SCENES \
    --n_frames 50 \
    --work_dir "$WORK" \
    --out_dir "$OUT_DIR"

echo "=== done, contents of $OUT_DIR ==="
ls -la "$OUT_DIR"
du -sh "$OUT_DIR"/*.tar.zst 2>/dev/null || true

echo "=== $(date) : fetch_replica.sh finished ==="
