# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

This is a fork of **VGGT** (Visual Geometry Grounded Transformer, CVPR 2025) — a feed-forward 3D reconstruction model. The project goal is **not** to modify VGGT itself, but to attach and train a **D4RT-style / DETR-like decoder for 3D multi-view consistent segmentation** on top of the frozen VGGT-1B backbone. Ground-truth supervision comes from segmentation masks produced by running **SAM3 on ScanNet scenes** (currently per-class binary masks; semantic vs. instance vs. panoptic is still open — see `docs/MILESTONE_2.md` §7.5).

Project history, design decisions, and results live in `docs/`:
- `docs/MILESTONE_1.md` — full prototype (phases 1–6), validated overfit + 4-scene multi-scene training. Read this first; it documents every component and the debugging that shaped the architecture.
- `docs/MILESTONE_2.md` — no-object loss, unprompted (grid-query) inference, regularization, best-checkpoint/early stopping. Scaling experiments are **blocked on data** (need tens-to-hundreds of preprocessed scenes).
- `docs/HOOK_PLAN.md` — where/how the decoder hooks into VGGT.
- `docs/todo.md` — current task list.
- `docs/prompt.md` — the original phase-by-phase project brief (workflow: incremental, each phase validated by a standalone test before moving on; simplicity over optimization).

## Environment & Commands

A virtualenv lives in-repo at `myenv/` — use `myenv/bin/python` (or `source myenv/bin/activate`). Runs on a GPU cluster node; matplotlib must stay headless (`Agg`).

```bash
# Tests (standalone scripts, not pytest; phase tests run on CPU without backbone weights)
python tests/test_phase2.py      # dataset loader + cross-view instance invariants
python tests/test_phase3.py      # QueryGenerator
python tests/test_phase4.py      # InstanceDecoder + dense mask head
python tests/test_phase5.py      # matcher + losses
python tests/test_eval.py        # instance-segmentation metrics
python tests/test_milestone2.py  # no-object loss, grid queries, augmentation, metrics.jsonl, early-stop, train-grid/query-mode queries
python tests/test_visualize_masks.py  # visualize_masks checkpoint-format handling (float/uint8/light) + overlays
python tests/test_mask_upsampler.py   # Phase-5 MaskUpsampler pixel decoder + GT-resolution match

# Single-scene overfit (sanity check for gradient flow / new components)
python scripts/train_overfit.py --num_epochs 400 --num_frames 4 --num_queries 16 \
    --learning_rate 2e-3 \
    --scene_dir /cluster/work/igp_psr/niacobone/distillation/dataset/scannet/scans/scene0000_00/raw_data

# Multi-scene training (the real training entry point)
python scripts/train_multiscene.py \
    --train_scenes scene0000_00,scene0001_00,scene0002_00,scene0003_00 \
    --val_scenes scene0004_00 \
    --num_epochs 1000 --warmup_epochs 30 --num_frames 8 --num_queries 32 \
    --learning_rate 2e-3 --bundles_per_scene 3 --query_jitter 0.02 --color_jitter 0.2 \
    --no_object_weight 0.1 --grid_size 6 --eval_interval 50 \
    --save_checkpoint /cluster/work/igp_psr/niacobone/distillation/output/<run_name>/checkpoint.pth
# After training it auto-renders the 2D overlays into <run_dir>/visualizations/ from
# checkpoint_best.pth (final checkpoint if no best was saved); opt out with --no_visualize.

# Scaling experiments (MILESTONE_2.md §7.1) as SLURM jobs — submit from anywhere, they cd
# to the repo and use myenv/. Val scenes 0080-0082 are held out of every train set.
sbatch slurm/train_scale10.sh   # scenes 0000-0009
sbatch slurm/train_scale25.sh   # scenes 0000-0024 (--cache_device cpu)
sbatch slurm/train_scale50.sh   # scenes 0000-0049 (--cache_device cpu)

# Visualize predictions manually (re-render or filter scenes)
python scripts/visualize_masks.py --checkpoint <run_dir>/checkpoint.pth   # 2D overlays → <run_dir>/visualizations/ (multi-scene ckpt: one subfolder per train/val scene; --scenes to filter)
python demos/demo_gradio.py --seg_checkpoint <path>   # 3D viewer; auto-discovers latest checkpoint, scene dropdown, "Color By: Predicted Instances"
```

Milestone-1 behavior is exactly recovered with `--no_object_weight 0 --bundles_per_scene 1 --query_jitter 0 --fixed_bg`.

### Storage layout (repo vs. group storage)

- Repo: `/cluster/scratch/niacobone/vggt`
- ScanNet scenes: `/cluster/work/igp_psr/niacobone/distillation/dataset/scannet/scans/<scene>/raw_data` (default `--scans_root`)
- Training runs/checkpoints: `/cluster/work/igp_psr/niacobone/distillation/output/<run_name>/checkpoint.pth` (timestamped run names, e.g. `d4rt_m2_5scenes_20260610_133100`). `checkpoint_best.pth` (best val mIoU) is the one to use for eval/demos; `checkpoint_best_ap50.pth` is the same run selected on the honest unprompted val[grid] AP50 instead.
- Each run dir also gets `metrics.jsonl` — one JSON line per eval (epoch, lr, loss, prompted+grid train/val mIoU & AP50). Scaling plots read this, not the logs.
- Checkpoints are self-contained: head weights + head config + the scene batches + optimizer/scheduler (for `--resume`). The frozen backbone is reloaded from HF (`facebook/VGGT-1B`), never stored. Scene images are stored as **uint8** (4× smaller than float; decoded back via `data/scannet_overfit.py::decode_checkpoint_images`); `--checkpoint_light` drops the pixels entirely and stores `frame_names` + `scene_dir`, so the visualizer/demo reload frames from `--scans_root`.

## Architecture

### Upstream VGGT (do not modify; kept frozen)

`vggt/models/vggt.py::VGGT` wraps `vggt/models/aggregator.py::Aggregator` (24 blocks of alternating per-frame and global cross-frame attention) plus the original heads in `vggt/heads/` (camera, depth, point, track). The `training/` directory is upstream's Co3D finetuning framework — unrelated to this project (our training code is in `scripts/` + `train/`).

### The segmentation head (this project's code)

The hook point is `aggregated_tokens_list[-1]` from the aggregator: global scene features `F: [B, S, P, 2048]` (S frames, P = patch tokens + 1 camera + 4 register tokens; `patch_start_idx` separates them). The head is a separate module — the backbone is untouched and runs under `no_grad`; only ~6.5M head params train.

Pipeline (one component per file, each with its phase test):

1. `data/scannet_overfit.py` — `ScanNetSingleSceneDataset` / `ScanNetMultiSceneDataset`. Loads frames from the scene's `subset/` dir (the ~100 stride-5 frames that actually have masks — **not** `color/`, which has >5500 unmasked frames) and per-class binary mask PNGs from `masks/<class>/`. Assigns one **global, cross-view-consistent instance ID per class** (the binary per-class masks can't separate same-class objects — data limitation, not code). Image size 518 (must be divisible by VGGT's patch size 14); mask/eval resolution is the 37×37 patch grid.
2. `models/d4rt_decoder.py` — `QueryGenerator` (Fourier-encoded (u,v) + learned view embedding + 9×9 RGB patch MLP, summed → `[B, N, 256]`) and `InstanceDecoder` (4-layer/8-head `nn.TransformerDecoder`, queries as tgt, projected F as memory) with `class_head` (20 logits = 19 ScanNet classes + background at index 0), `mask_embed_head`, and a dense Mask2Former-style mask head → `pred_masks [B, N, S, h, w]`. `D4RTInstanceSegmentationHead` chains them. `query_mode` (`point` default / `learned` DETR object queries / `hybrid`) and `mask_upsample` (1 default = 37×37 patch grid; 2/4 route through `models/mask_upsampler.py::MaskUpsampler` for sharper masks, with GT built at the matching resolution) are constructor + `head_config` options — keep the round-trip intact.
3. `train/loss.py` — `PointBipartiteMatcher` (Hungarian, mask-aware Dice+BCE cost) + `D4RTLoss` (Focal class loss + Dice + fg-weighted BCE; optional DETR-style no-object loss on unmatched queries via `no_object_weight`). Batch-aware: for `B > 1`, GT args are lists of per-sample tensors.
4. `train/eval_metrics.py` — mIoU / AP50 / AP75 / mAP / class_acc. Evaluation reports **prompted** (queries at GT centroids) and **unprompted** (uniform grid, no GT) metrics; unprompted is the honest detection number.
5. `scripts/train_multiscene.py` — caches frozen-backbone features **once per scene bundle up front**, then every epoch runs only the head (this is why training is minutes, not hours). `--cache_device cpu` lifts the GPU-memory bound on scene count.

### Hard-won constraints (violating these silently breaks training)

These came out of real debugging (`MILESTONE_1.md` §6) — keep them when touching the decoder:
- **LayerNorm the projected memory** and keep the **query skip connection** in `InstanceDecoder` — raw VGGT features have huge magnitudes and otherwise every query collapses to the same decoded vector (loss falls, mIoU stays 0).
- Mask logits use **cosine similarity** (learnable temperature), not raw dot products, to keep sigmoids from saturating.
- BCE uses a foreground `pos_weight`; gradient clipping is on.
- Coordinates are query *prompts*, not predictions: they enter the matcher cost but carry no loss term.
- An overfit test must hold inputs **and** targets fixed across epochs to be meaningful.

## Working rules

- **Always proceed step by step**: implement incrementally and test every component you add or edit before moving on (run the relevant `tests/test_*.py`, or add one if none covers it).
- **After every change, check whether documentation needs updating or adding** — both the files in `docs/` (milestone docs, `todo.md`) and this CLAUDE.md itself.

## Conventions

- New components follow the established pattern: implement in the matching dir (`data/`, `models/`, `train/`, `scripts/`), add a standalone CPU-runnable test in `tests/`, and document the result in the current milestone doc. New loss/training options must default to off / previous behavior so existing tests pass unchanged.
- ScanNet class indices: `1..19`, with `0` = background everywhere (dataset, class head, no-object target).
- `--num_views` sizes the view-embedding table; checkpoints store the head config so the demo can rebuild the head — keep that round-trip intact when changing the head's constructor.
