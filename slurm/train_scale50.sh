#!/bin/bash
#
# Scaling experiment (MILESTONE_2.md §7.1): 50 train scenes.
# Submit from anywhere: sbatch slurm/train_scale50.sh
#
#SBATCH --job-name=d4rt_scale50
#SBATCH --output=train_scale50_%j.log
#SBATCH --error=train_scale50_%j.err
#SBATCH --open-mode=append
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6144
#SBATCH --gpus=rtx_4090:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=niacobone@student.ethz.ch

module purge
module load stack/2024-06 python/3.12.8 cuda/12.8.0 eth_proxy
cd /cluster/scratch/niacobone/vggt
source myenv/bin/activate
PYTHON=myenv/bin/python

VAL=scene0080_00,scene0081_00,scene0082_00
OUT=/cluster/work/igp_psr/niacobone/distillation/output

# 50 scenes (0000–0049); ~14 GB of cached bundles → host RAM cache + extra memory headroom
$PYTHON scripts/train_multiscene.py \
    --train_scenes $(seq -f "scene%04g_00" 0 49 | paste -sd, -) \
    --val_scenes $VAL \
    --num_epochs 1000 --warmup_epochs 30 --num_frames 8 --num_queries 32 \
    --learning_rate 2e-3 --bundles_per_scene 3 --query_jitter 0.02 --color_jitter 0.2 \
    --no_object_weight 0.1 --grid_size 6 --eval_interval 50 --early_stop_patience 0 \
    --cache_device cpu \
    --save_checkpoint $OUT/d4rt_m2_scale50_$(date +%Y%m%d_%H%M%S)/checkpoint.pth
