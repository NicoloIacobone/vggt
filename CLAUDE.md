# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

This is a fork of **VGGT** (Visual Geometry Grounded Transformer, CVPR 2025) — a feed-forward 3D reconstruction model. The project goal is **not** to modify VGGT itself, but to attach and train a **D4RT-style / DETR-like decoder for 3D multi-view consistent segmentation** on top of the frozen VGGT-1B backbone. Ground-truth supervision is the **official ScanNet v2 2D instance annotations** (default since 2026-07-08; see Storage layout). The previous GT — masks from running **SAM3 on ScanNet scenes** — was found to have systematic cross-class duplicates (audit 2026-07-07: ~15.9% multi-class foreground pixels, the same object labeled under two classes → built-in honest-AP50 false positives) and is kept only as the baseline for GT-quality comparisons; migration record: `docs/OFFICIAL_GT_MIGRATION_PLAN.md`. The loader defaults to per-class, `--instance_level` switches to per-instance.

Project history, design decisions, and results live in `docs/`:
- `docs/MILESTONES.md` — **the single consolidated summary** of Milestones 1–3 (architecture, hard-won constraints, all results, qualitative findings, dataset & storage status). Read this first.
- `docs/todo.md` — current open task list.
- `docs/RIEPILOGO_PROGETTO_IT.md` — full project narrative in Italian (architecture, motivations, experiments, interpretations) for the project owner.
- `docs/HOOK_PLAN.md` — where/how the decoder hooks into VGGT.
- `docs/slides_meeting_jun_15.md` — most recent supervision-meeting slides.
- `docs/old/` — archived per-milestone detail (`MILESTONE_1/2/3.md`), executed plans (`NEXT_STEPS_PLAN.md`), the scaling-protocol analysis (`SCALING_RUNS_ANALYSIS.md`), addressed supervisor feedback, the original project brief (`prompt.md`), and the SAM3 preprocessing prompt. Consult these for the full debugging narrative behind a result.

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
python tests/test_grid_ablation.py    # eval-only grid-density sweep (eval_grid_ablation.py)
python tests/test_build_official_masks.py  # official-GT converter (synthetic zips/tsv → SAM3 layout + loader round-trip)
python tests/test_eval_checkpoint.py       # cross-GT eval plumbing (arg inheritance/overrides, scene resolution)

# Single-scene overfit (sanity check for gradient flow / new components). No unpacked tree
# lives on work anymore — point --scene_dir at the official-GT build tree on scratch (below),
# or at a staged/unpacked tar.
python scripts/train_overfit.py --num_epochs 400 --num_frames 4 --num_queries 16 \
    --learning_rate 2e-3 --instance_level \
    --scene_dir /cluster/scratch/niacobone/scannet_official_build/scans/scene0000_00/raw_data

# Multi-scene training (the real training entry point)
python scripts/train_multiscene.py \
    --train_scenes scene0000_00,scene0001_00,scene0002_00,scene0003_00 \
    --val_scenes scene0004_00 \
    --num_epochs 1000 --warmup_epochs 30 --num_frames 8 --num_queries 32 \
    --query_mode learned --num_learned_queries 64 --instance_level \
    --learning_rate 2e-3 --bundles_per_scene 3 --query_jitter 0.02 --color_jitter 0.2 \
    --no_object_weight 0.1 --grid_size 6 --eval_interval 50 \
    --save_checkpoint /cluster/work/igp_psr/niacobone/distillation/output/<run_name>/checkpoint.pth
# CURRENT BASE: learned object queries (arm C) on per-instance GT (--instance_level) is the
# default starting point for all further experiments — it broke the point-prompt plateau
# (val mIoU 0.371, honest val[grid] AP50 0.228 at N=200; see docs/MILESTONES.md). For the
# superseded point-prompt baseline, drop --query_mode/--num_learned_queries (defaults to point).
# After training it auto-renders the 2D overlays into <run_dir>/visualizations/ from
# checkpoint_best.pth (final checkpoint if no best was saved); opt out with --no_visualize.
# Arm-B/D fixes (2026-07-03): --no_object_norm matched normalizes the no-object class loss
# per term (matched.mean() + w*unmatched.mean()), so appended grid queries no longer dilute
# the matched gradients — use it with --train_grid_queries. --learned_query_lr_scale 0.1
# puts the learned-query embeddings in their own lower-LR AdamW param group (hybrid NaN
# fix; 1.0 = single group, --resume-compatible with old checkpoints). The matcher also
# nan_to_num-guards the Hungarian cost (warns instead of crashing on non-finite entries).
# Rerun outcome (2026-07-07): both fixes verified (B's collapse and D's NaN are gone) but
# neither arm beats arm C — arms B/D closed, arm C stays the base (docs/MILESTONES.md).

# Scaling experiments (docs/MILESTONES.md) as SLURM jobs — submit from anywhere, they cd
# to the repo and use myenv/. Val scenes 0080-0082 are held out of every train set. Each job
# stages the dataset tar onto node-local scratch first (slurm/stage_dataset.sh → SCANNET_ROOT).
# Which tar is staged = DATA_TAR env var: the train scripts default to the OFFICIAL ScanNet GT
# (scannet_official_gt_full.tar.zst); for the SAM3-GT baseline submit with
#   sbatch --export=ALL,DATA_TAR=/cluster/work/igp_psr/niacobone/distillation/dataset/scannet/scannet_instance_dataset_full.tar.zst <script>
sbatch slurm/train_scale10.sh   # scenes 0000-0009
sbatch slurm/train_scale25.sh   # scenes 0000-0024 (--cache_device cpu)
sbatch slurm/train_scale50.sh   # scenes 0000-0049 (--cache_device cpu)
# Add --instance_level (to the script's python call) to run the curve on per-instance GT.

# Official-GT dataset rebuild (only if the tar is lost/needs changes; docs/OFFICIAL_GT_MIGRATION_PLAN.md):
sbatch slurm/download_official_gt.sh   # download+convert 200 scenes (resumable; scripts/download_2d_gt.py + scripts/build_official_masks.py)
sbatch slurm/pack_official_gt.sh       # QA gates (scripts/gen_official_gt_report.py; 0 cross-class dups) + strips + tar → work

# Cross-GT eval: score any checkpoint against GT rebuilt fresh from a scans tree (works for
# learned-query heads, unlike eval_grid_ablation.py) → <run_dir>/cross_eval_<ckpt>_<tag>.json
python scripts/eval_checkpoint.py --checkpoint <run_dir>/checkpoint_best_ap50.pth --scans_root <scans_root> --tag official_gt

# Eval-only grid-density ablation (docs/todo.md): sweeps the unprompted --grid_size on the
# val bundles stored in a point/hybrid checkpoint — no retraining, no dataset staging;
# rejects learned-query checkpoints (their queries ignore coordinates). Results →
# <run_dir>/grid_ablation_<ckpt>.json. Batch job for the trained runs: sbatch slurm/eval_grid_ablation.sh
python scripts/eval_grid_ablation.py --checkpoint <run_dir>/checkpoint_best_ap50.pth --grid_sizes 2,4,6,8,10,12

# Visualize predictions manually (re-render or filter scenes)
python scripts/visualize_masks.py --checkpoint <run_dir>/checkpoint.pth   # 2D overlays → <run_dir>/visualizations/ (multi-scene ckpt: one subfolder per train/val scene; --scenes to filter)
python demos/demo_gradio.py --seg_checkpoint <path>   # 3D viewer; auto-discovers latest checkpoint, scene dropdown, "Color By: Predicted Instances"
```

The 2D overlay and the 3D viewer share ONE instance-selection rule — `train/postprocess.py::select_instances`
(drop background/score<0.5 queries, per-pixel winner-takes-all, no GT, no query-order assumption). The 2D
figure shows 4 panels: RGB | GT | **Prediction (honest, no GT)** | Prediction (oracle, GT-matched). The
"honest" panel and the 3D "Predicted Instances" coloring are identical by construction; the oracle panel
(Hungarian-matched to GT) is the upper-bound diagnostic for "is this a detection miss or a mask-quality
issue?". All masks render at the head's native patch-grid resolution, nearest-upsampled, so GT and
predictions share the same (honest) sharpness — predictions are NOT bilinear-smoothed to look better than
their 37×37 supervision. For genuinely sharper masks, retrain with `--mask_upsample 2/4` (changes the
supervision resolution; compare AP50/mIoU, don't judge by the picture).

Milestone-1 behavior is exactly recovered with `--no_object_weight 0 --bundles_per_scene 1 --query_jitter 0 --fixed_bg`.

### Storage layout (repo vs. group storage)

- Repo: `/cluster/scratch/niacobone/vggt`
- **Datasets ship ONLY as tars on work** (`/cluster/work/igp_psr/niacobone/distillation/dataset/scannet/`); there is NO unpacked `scans/` tree on work (deliberately deleted 2026-07: reading thousands of small PNGs off `work` is slow and pressures the inode quota). Two tars, identical layout `scans/<scene>/raw_data/{subset,masks,masks_instance,_qa}`, 200 scenes (scene0000–0199) each with `subset/` (~100 stride-5 frames), `masks/<class>/` (per-class binary), `masks_instance/<class>_<k>/` (per-instance binary):
  - **`scannet_official_gt_full.tar.zst` (2.3 GB, the DEFAULT)** — official ScanNet v2 GT (projections of the human-verified 3D annotation): one class per object, cross-view-consistent ids, **2950 instances, 0 cross-class duplicates**. Masks written sparsely (missing file = instance not visible; the loader skips). Stuff classes keep official ids (`wall_0..wall_k`). Spec: `…/scannet/OFFICIAL_GT_README.md`. Built by `scripts/build_official_masks.py` from the official `_2d-instance-filt`/`_2d-label-filt` zips (rebuild: `slurm/download_official_gt.sh` + `slurm/pack_official_gt.sh`).
  - `scannet_instance_dataset_full.tar.zst` (~2.6 GB) — the SAM3-generated GT (≈4195 instances, but ~3.4 cross-class duplicate instances/scene; audit 2026-07-07). Kept as baseline + project deliverable. Cross-frame identity from SAM3 video tracking; `wall`/`floor` forced single-instance; all-zero PNGs written where absent. Spec: `…/scannet/INSTANCE_MASKS_README.md` (+`…_split2.md`).
- **Dataset access at training time:** each job copies ONE tar to node-local scratch (`$TMPDIR`) and unpacks it there (`slurm/stage_dataset.sh`, exports `SCANNET_ROOT=$TMPDIR/scans`; `scripts/train_multiscene.py` uses `SCANNET_ROOT` as default `--scans_root`). The tar is chosen by `DATA_TAR`: stage_dataset.sh's own default is still the SAM3 tar (backward compat), but **every train SLURM script overrides it to the official tar**. `#SBATCH --tmp=16000` (MB) covers either tar + unpacked tree. `zstd` at `/usr/bin/zstd`. The official-GT build tree currently also exists unpacked at `/cluster/scratch/niacobone/scannet_official_build/scans` (scratch — purgeable; the tar on work is canonical). The old SAM3 scratch build trees (`scannet_build*`) are hollow (purge ate the PNGs) — re-pack from the tar if ever needed, never unpack-to-retar on scratch (inode quota).
- **Mask conventions (both GTs):** uint8 {0,255}, 1296×968, filename = subset stem + `.png`; `<k>` zero-based per class in order of first appearance; dir names use underscores (`shower_curtain_3`). Union of a class's instance masks = its `masks/<class>/` mask. The loader defaults to per-class `masks/`; `--instance_level` switches to per-instance (`data/scannet_overfit.py::instance_level`). In official GT, NYU40 classes outside the 19 trainable (incl. `otherfurniture`) are background — the class head's 20 logits can't represent index 20.
- Training runs/checkpoints: `/cluster/work/igp_psr/niacobone/distillation/output/<run_name>/checkpoint.pth` (timestamped run names, e.g. `d4rt_m2_5scenes_20260610_133100`). `checkpoint_best.pth` (best val mIoU) is the one to use for eval/demos; `checkpoint_best_ap50.pth` is the same run selected on the honest unprompted val[grid] AP50 instead.
- Each run dir also gets `metrics.jsonl` — one JSON line per eval (epoch, lr, loss, prompted+grid train/val mIoU & AP50). Scaling plots read this, not the logs.
- Checkpoints are self-contained: head weights + head config + the scene batches + optimizer/scheduler (for `--resume`). The frozen backbone is reloaded from HF (`facebook/VGGT-1B`), never stored. Scene images are stored as **uint8** (4× smaller than float; decoded back via `data/scannet_overfit.py::decode_checkpoint_images`); `--checkpoint_light` drops the pixels entirely and stores `frame_names` + `scene_dir`, so the visualizer/demo reload frames from `--scans_root`.

## Architecture

### Upstream VGGT (do not modify; kept frozen)

`vggt/models/vggt.py::VGGT` wraps `vggt/models/aggregator.py::Aggregator` (24 blocks of alternating per-frame and global cross-frame attention) plus the original heads in `vggt/heads/` (camera, depth, point, track). The `training/` directory is upstream's Co3D finetuning framework — unrelated to this project (our training code is in `scripts/` + `train/`).

### The segmentation head (this project's code)

The hook point is `aggregated_tokens_list[-1]` from the aggregator: global scene features `F: [B, S, P, 2048]` (S frames, P = patch tokens + 1 camera + 4 register tokens; `patch_start_idx` separates them). The head is a separate module — the backbone is untouched and runs under `no_grad`; only ~6.5M head params train.

Pipeline (one component per file, each with its phase test):

1. `data/scannet_overfit.py` — `ScanNetSingleSceneDataset` / `ScanNetMultiSceneDataset`. Loads frames from the scene's `subset/` dir (the ~100 stride-5 frames that actually have masks — **not** `color/`, which has >5500 unmasked frames) and per-class binary mask PNGs from `masks/<class>/`. Assigns one **global, cross-view-consistent instance ID per class** by default (the binary per-class masks can't separate same-class objects — data limitation, not code). Pass `instance_level=True` (CLI `--instance_level`) to instead read `masks_instance/<class>_<k>/` and assign one ID per `(class, instance)`, so same-class objects are separated and `classes` repeats class indices (matcher/loss/eval unchanged — already instance-based). Image size 518 (must be divisible by VGGT's patch size 14); mask/eval resolution is the 37×37 patch grid.
2. `models/d4rt_decoder.py` — `QueryGenerator` (Fourier-encoded (u,v) + learned view embedding + 9×9 RGB patch MLP, summed → `[B, N, 256]`) and `InstanceDecoder` (4-layer/8-head `nn.TransformerDecoder`, queries as tgt, projected F as memory) with `class_head` (20 logits = 19 ScanNet classes + background at index 0), `mask_embed_head`, and a dense Mask2Former-style mask head → `pred_masks [B, N, S, h, w]`. `D4RTInstanceSegmentationHead` chains them. `query_mode` (`point` default / `learned` DETR object queries / `hybrid`) and `mask_upsample` (1 default = 37×37 patch grid; 2/4 route through `models/mask_upsampler.py::MaskUpsampler` for sharper masks, with GT built at the matching resolution) are constructor + `head_config` options — keep the round-trip intact.
3. `train/loss.py` — `PointBipartiteMatcher` (Hungarian, mask-aware Dice+BCE cost) + `D4RTLoss` (Focal class loss + Dice + fg-weighted BCE; optional DETR-style no-object loss on unmatched queries via `no_object_weight`). Batch-aware: for `B > 1`, GT args are lists of per-sample tensors.
4. `train/eval_metrics.py` — mIoU / AP50 / AP75 / mAP / class_acc. Evaluation reports **prompted** (queries at GT centroids) and **unprompted** (uniform grid, no GT) metrics; unprompted is the honest detection number.
5. `scripts/train_multiscene.py` — caches frozen-backbone features **once per scene bundle up front**, then every epoch runs only the head (this is why training is minutes, not hours). `--cache_device cpu` lifts the GPU-memory bound on scene count.

### Hard-won constraints (violating these silently breaks training)

These came out of real debugging (`docs/old/MILESTONE_1.md` §6) — keep them when touching the decoder:
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
