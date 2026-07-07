#!/bin/bash
#
# Grid-density ablation (docs/todo.md): eval-only sweep of the unprompted grid size
# (2/4/6/8/10/12 vs the 6x6 default) on the trained point-mode checkpoints. No retraining,
# no dataset staging — the checkpoints store the val bundles (uint8 images); only the frozen
# backbone reruns once per scene. Results: <run_dir>/grid_ablation_<ckpt>.json.
#
#SBATCH --job-name=d4rt_grid_ablate
#SBATCH --output=eval_grid_ablation_%j.log
#SBATCH --error=eval_grid_ablation_%j.err
#SBATCH --open-mode=append
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000
#SBATCH --gpus=rtx_4090:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=niacobone@student.ethz.ch

module purge
module load stack/2024-06 python/3.12.8 cuda/12.8.0 eth_proxy
cd /cluster/scratch/niacobone/vggt
source myenv/bin/activate
PYTHON=myenv/bin/python

OUT=/cluster/work/igp_psr/niacobone/distillation/output
GRIDS=2,4,6,8,10,12

# Arm A (point prompts, N=200): the run whose honest val[grid] AP50 plateaued at ~0.10.
# Both selection checkpoints — AP50-selected is the honest-detection one.
for CKPT in checkpoint_best_ap50.pth checkpoint_best.pth; do
    $PYTHON scripts/eval_grid_ablation.py \
        --checkpoint $OUT/d4rt_full_inst_20260618_160014/$CKPT --grid_sizes $GRIDS
done

# Arm B (trained grid queries, N=190): density sweep is most meaningful here — the grid
# path was actually trained (random-offset 6x6), so density is not confounded with the
# centroid->grid train/eval mismatch that arm A has.
$PYTHON scripts/eval_grid_ablation.py \
    --checkpoint $OUT/d4rt_full_inst_gridq_fix_20260703_184456/checkpoint_best_ap50.pth \
    --grid_sizes $GRIDS
