#!/bin/bash
#
# Deliverable 2 (docs/todo.md external-dataset task): mirror lifuguan/InsScene-15K
# (IGGT's own training set, Apache-2.0) to work. ~522 GB across 1565 files/shards as of
# 2026-08-07 -- does not fit one 24h wall clock, so this self-resubmits exactly like
# slurm/train_maskdino_coco.sh: the python step stops itself at --time_budget_hours (below
# the wall clock) and exits 0, so the resubmit at the script tail always gets to run instead
# of being torn down mid-transfer by SLURM.
#
# Shards are moved to work AS-IS (never unzipped -- one shard is one inode) through $TMPDIR,
# so no loose files ever accumulate on /cluster/scratch.
#
#SBATCH --job-name=fetch_insscene15k
#SBATCH --output=/cluster/scratch/niacobone/vggt/slurm/logs/fetch_insscene15k_%j.log
#SBATCH --error=/cluster/scratch/niacobone/vggt/slurm/logs/fetch_insscene15k_%j.err
#SBATCH --open-mode=append
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4096
#SBATCH --tmp=30000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=niacobone@student.ethz.ch

set -euo pipefail
module purge
module load stack/2024-06 python/3.12.8 eth_proxy
cd /cluster/scratch/niacobone/vggt
PYTHON=/cluster/scratch/niacobone/vggt/myenv/bin/python
export PYTHONUNBUFFERED=1

OUT_DIR=/cluster/work/igp_psr/niacobone/distillation/dataset/insscene15k
mkdir -p "$OUT_DIR"

# The manifest is rebuilt fresh each segment (cheap, a few API calls) so a repo change
# (e.g. an Aria/ASE dir appearing mid-transfer) is picked up rather than a stale list.
MANIFEST="$OUT_DIR/manifest.json"
$PYTHON slurm/insscene15k_manifest.py "$MANIFEST"

RESUBMIT_COUNT="${RESUBMIT_COUNT:-0}"
MAX_RESUBMITS=80   # ~80 days of segments; a circuit breaker, not an expected ceiling

echo "=== $(date) : segment start, resubmit #$RESUBMIT_COUNT, node $(hostname) ==="
df -h "$TMPDIR"

STATE_FILE="$TMPDIR/state.json"
$PYTHON slurm/download_insscene15k.py \
    --manifest "$MANIFEST" \
    --out_dir "$OUT_DIR" \
    --tmp_dir "$TMPDIR/dl" \
    --time_budget_hours 23 \
    --workers 4 \
    --state_file "$STATE_FILE"

cat "$STATE_FILE"
COMPLETE=$(python3 -c "import json; print(json.load(open('$STATE_FILE'))['complete'])")

if [ "$COMPLETE" = "True" ]; then
    echo "=== COMPLETE : all shards mirrored ==="
    ls -la "$OUT_DIR"
else
    if [ "$RESUBMIT_COUNT" -ge "$MAX_RESUBMITS" ]; then
        echo "=== ABORT : hit MAX_RESUBMITS=$MAX_RESUBMITS without completing, needs investigation ==="
        exit 1
    fi
    echo "=== incomplete, resubmitting (resubmit #$((RESUBMIT_COUNT + 1))) ==="
    sbatch --export=ALL,RESUBMIT_COUNT=$((RESUBMIT_COUNT + 1)) slurm/fetch_insscene15k.sh
fi
