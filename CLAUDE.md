# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A fork of **VGGT** (Visual Geometry Grounded Transformer, CVPR 2025) — a feed-forward 3D
reconstruction model. The goal is **not** to modify VGGT, but to attach and train a decoder for
**3D multi-view consistent instance segmentation** on top of the frozen VGGT-1B backbone.
Supervision is the **official ScanNet v2 2D instance annotations**.

**The active model is a MaskDINO decoder** (`models/maskdino/`), currently single-frame. At 490
scenes it scores val mIoU **0.669** / AP50 **0.699**, against the best previous head's
0.451 / 0.294 on the same per-frame protocol. Multi-frame is the open work.

An earlier family of hand-rolled DETR-style heads ("D4RT arms A–E", a query-initialisation study)
is **retired to `legacy/d4rt/`** but still runnable — it is the baseline every MaskDINO number is
measured against, and `scripts/eval_perframe.py` imports it to produce that baseline. See
`legacy/README.md`.

### Docs — read in this order

- `docs/MASKDINO.md` — **the primary document.** Architecture, deviations from upstream MaskDINO,
  the single-frame protocol, the evaluation protocol, all results, and the multi-frame plan.
- `docs/MASKDINO_COCO.md` — the COCO backbone-swap study: does MaskDINO's published recipe survive
  a frozen VGGT backbone? Contains the **mask-resolution ceiling measurement** (§1) — read it
  before proposing anything that depends on mask or token resolution, on COCO *or* on ScanNet.
- `docs/RESULTS.md` — every number in one place, split by protocol. Read §1 before quoting
  anything: per-frame and per-bundle numbers are **not** interchangeable.
- `docs/SUPERVISOR_COMPARISON.md` — the send-outward summary: MaskDINO vs arm C, the COCO
  port-equivalence check, and how the metrics are computed. Derived from the two files above —
  if a number changes there, change it here too.
- `docs/DATASET.md` — GT provenance, the tars, mask conventions, how a job gets the data.
- `docs/todo.md` — open work only.
- `docs/ARMS_SUMMARY.md` — the retired arms A–E in one page (what differed, results, verdicts).
- `docs/RELATED_WORK.md` — competitor landscape & positioning. Read before framing any result as
  a contribution.
- `docs/RIEPILOGO_PROGETTO_IT.md` — full project narrative in Italian for the project owner.
- `docs/old/` — archived detail: `MILESTONES.md` (the full D4RT story), per-milestone docs,
  executed plans, the todo archive, past meeting slides, the original project brief.

## Environment & Commands

A virtualenv lives in-repo at `myenv/` — use `myenv/bin/python`. Runs on a GPU cluster node;
matplotlib must stay headless (`Agg`).

```bash
# --- Tests (standalone scripts, not pytest; all CPU-only, no backbone weights needed) --------
python tests/test_maskdino_model.py   # MSDeformAttn vs naive ref, pixel decoder, decoder configs,
                                      # box ops, head_config round-trip, initialize_box_type guard
python tests/test_maskdino_loss.py    # matcher, criterion keys + perfect-prediction zero loss,
                                      # out-of-range GT-label guard
python tests/test_maskdino_train.py   # per-frame GT builder (incl. class drop), per-frame metric
                                      # slicing, 60-step synthetic overfit
python tests/test_maskdino_multiframe.py  # shared-query multi-frame path: cross-frame block,
                                      # bundle GT + index expansion, bundle matcher, S=1
                                      # equivalence, multi-frame overfit, bundle batching/scoring
python tests/test_maskdino_viz.py     # figure colouring keyed to identity, not per-frame rank
python tests/test_maskdino_consistency.py  # cross-view consistency metrics (MASKDINO.md §6.6):
                                      # planted perfect/switched cases, degenerate inputs,
                                      # additive bundle_* keys
python tests/test_maskdino_fullres.py # --eval_full_res ruler (MASKDINO.md §6.5): helpers,
                                      # grid-vs-full ruler difference, both eval paths
python tests/test_maskdino_eval3d.py  # 3D ruler (MASKDINO.md §9): PLY/GT builders, Umeyama/ICP,
                                      # unprojection round-trip, votes+majority, the vendored
                                      # official evaluator vs hand-computed APs, synthetic E2E
python tests/test_coco_maskdino.py    # COCO track: both pixel-decoder pyramid modes, head
                                      # round-trip, GT helpers, instance inference + RLE, overfit
bash tests/test_train_maskdino_sh_lists.sh  # slurm scene-list logic via DRY_RUN: numeric-range
                                      # back-compat, TRAIN_LIST/VAL_LIST split files, filtering

# --- Training (the entry point) ---------------------------------------------------------------
sbatch slurm/train_maskdino.sh                                 # 50 scenes, ~20k steps
sbatch --export=ALL,N_SCENES=490 slurm/train_maskdino.sh       # epochs auto-scale to hold the budget
sbatch --export=ALL,EXTRA_ARGS='--mask_upsample 2' slurm/train_maskdino.sh
# multi-frame: one query set per bundle of 8 frames (docs/MASKDINO.md §8.2); reports the
# per-bundle (multi-view) metrics as well as the per-frame ones
sbatch --export=ALL,N_SCENES=490,EXTRA_ARGS='--multi_frame --feature_mode bundle' slurm/train_maskdino.sh
# Official 1201/312 split: TRAIN_LIST/VAL_LIST override the numeric-range scene selection, and
# DATA_TAR takes a space-separated list of tars staged into one tree. Needs bigger resources
# than the script header (fp16 cache ~110 GB at 1201 scenes; ~58 GB staged) — pass them on the
# command line. EPOCHS=12 ≈ 28.8k steps at 1201 scenes x 2 bundles (~ the N=490 recipe budget).
# The scene-list logic is covered by tests/test_train_maskdino_sh_lists.sh (dry-run, CPU-only).
DS=/cluster/work/igp_psr/niacobone/distillation/dataset/scannet
sbatch --time=24:00:00 --cpus-per-task=12 --mem-per-cpu=14336 --tmp=90000 \
    --export=ALL,DATA_TAR="$DS/scannet_official_gt_1201.tar.zst $DS/scannet_official_gt_val312.tar.zst",\
TRAIN_LIST=data/splits/scannetv2_train.txt,VAL_LIST=data/splits/scannetv2_val.txt,EPOCHS=12,WARMUP=2,\
EXTRA_ARGS='--bundles_per_scene 2 --color_jitter 0.2' slurm/train_maskdino.sh
DRY_RUN=1 TRAIN_LIST=... VAL_LIST=... bash slurm/train_maskdino.sh   # echo lists/schedule, no data
python scripts/train_maskdino.py --train_scenes scene0000_00 --val_scenes scene0080_00 \
    --num_epochs 50 --num_queries 300 --scans_root <scans_root>       # local smoke test

# --- Full-resolution ruler (docs/MASKDINO.md §6.5) --------------------------------------------
# Adds full_* metrics scored at the 518x518 GT resolution next to the unchanged grid metrics.
sbatch --export=ALL,N_SCENES=490,EXTRA_ARGS='--eval_full_res' slurm/train_maskdino.sh
sbatch slurm/scannet_oracle.sh   # GT-only ceiling of the 37/74/148 grids on ScanNet (CPU-only)

# --- 3D ruler: official ScanNet 3D instance benchmark (docs/MASKDINO.md §9) -------------------
# THE THIRD PROTOCOL — the only one placeable next to SegVGGT/FAST3DIS; never quote next to the
# 2D tables. Unprojects a --multi_frame checkpoint's masks with VGGT's OWN predicted
# depth+cameras (no GT geometry at inference), Sim(3)-registers for scoring only, majority-votes
# per superpoint, scores with the vendored official evaluator. Checkpoints trained on scenes
# 0000-0489 overlap val-312: their numbers are DIAGNOSTIC only (§9.4).
sbatch --export=ALL,CHECKPOINT=<run_dir>/checkpoint_best_bundle.pth slurm/eval_3d_maskdino.sh
# needs the two val-312 tars on work (built 2026-08-01; rebuild in ~20 min if lost):
sbatch legacy/dataset_build/slurm/download_3d_gt_val312.sh       # mesh+superpoints+aggregation
sbatch legacy/dataset_build/slurm/download_frames25k_val312.sh   # whole-scan frames + poses

# --- Re-render a finished run's figures (docs/MASKDINO.md §6.4) -------------------------------
# Colours are keyed to instance identity, so an object keeps its colour across the frames of a
# bundle. Changing the drawing code does NOT touch existing runs — re-render them explicitly:
sbatch --export=ALL,RUNS='<run_dir_1> <run_dir_2>' slurm/visualize_maskdino.sh
python scripts/visualize_maskdino.py --checkpoint <run_dir>/checkpoint_best.pth   # needs GPU+data

# --- Upstream-equivalence check (docs/MASKDINO.md §7.6) ---------------------------------------
# Drives OUR ported decoder/encoder with upstream MaskDINO's released COCO weights and checks we
# reproduce their published val2017 numbers. Needs the REFERENCE env, not myenv/, and a GPU of
# sm<=86 (3090/A100) because that torch build predates Ada.
sbatch slurm/coco_transplant.sh                                # both modes, 5000 imgs, ~32 min
LD_LIBRARY_PATH=/cluster/scratch/niacobone/MaskDINO/myenv/lib/python3.9/site-packages/torch/lib \
  /cluster/scratch/niacobone/MaskDINO/myenv/bin/python scripts/coco_transplant_eval.py \
  --mode ours --limit 10          # CPU-runnable smoke test; --mode baseline is the control

# --- COCO backbone-swap study (docs/MASKDINO_COCO.md) -----------------------------------------
# Trains the SAME decoder on COCO with a frozen, swappable backbone. Parallel scripts throughout
# (COCO cannot use the ScanNet feature cache: 618 GB), nothing in the ScanNet path is shared.
# Each job self-resubmits until <run_dir>/summary.json exists.
sbatch --export=ALL,BACKBONE=resnet50 slurm/train_maskdino_coco.sh   # the control
sbatch --export=ALL,BACKBONE=vggt     slurm/train_maskdino_coco.sh   # the question
sbatch --export=ALL,BACKBONE=dinov2   slurm/train_maskdino_coco.sh   # same token geometry as VGGT
# The GT-only resolution ceiling — run/quote this BEFORE arguing about mask resolution:
myenv/bin/python scripts/coco_mask_resolution_oracle.py   # 37x37 caps a PERFECT model at 44.7 AP

# --- The apples-to-apples baseline ------------------------------------------------------------
# Per-frame metrics are NOT comparable to the retired arms' multi-view numbers. To score a D4RT
# checkpoint under the identical per-frame protocol:
python scripts/eval_perframe.py --checkpoint <d4rt_run>/checkpoint_best.pth   # → perframe_eval_*.json

# --- Retired D4RT arms (still runnable; see legacy/README.md) ---------------------------------
python legacy/d4rt/scripts/train_multiscene.py --train_scenes ... --val_scenes ...
python legacy/d4rt/scripts/visualize_masks.py --checkpoint <run_dir>/checkpoint.pth
python demos/demo_gradio.py --seg_checkpoint <path>   # 3D viewer for D4RT checkpoints
sbatch legacy/d4rt/slurm/train_full.sh
for t in legacy/d4rt/tests/test_*.py; do python "$t"; done

# --- Dataset rebuild (only if a tar is lost; docs/DATASET.md §5) -------------------------------
for t in legacy/dataset_build/tests/test_*.py; do myenv/bin/python "$t"; done   # CPU-only
sbatch legacy/dataset_build/slurm/download_official_gt.sh
sbatch legacy/dataset_build/slurm/extend_dataset_500.sh
sbatch legacy/dataset_build/slurm/pack_official_gt.sh

# --- 1201-scene official-train extension (docs/todo.md 1c; separate tar, does not touch the
# 500-scene one) --------------------------------------------------------------------------------
# BUILDS NODE-LOCAL. /cluster/scratch is quota'd on FILE COUNT (1.0M soft / 1.5M hard) and the
# 1201-scene tree is ~1.26M files — building it there fails, and did (docs/DATASET.md §5.1).
# The tree lives in $TMPDIR; only one compressed chunk tar per range lands on scratch.
sbatch legacy/dataset_build/slurm/extend_dataset_1201.sh <list_start> <list_end>  # one per chunk
sbatch legacy/dataset_build/slurm/pack_official_gt_1201.sh   # after all chunks report COMPLETE
# An incomplete chunk resubmits ITSELF (new job id), so --dependency=afterok on the original id
# never fires — don't chain the pack that way. With a single chunk covering the whole split,
# CHAIN_PACK=1 makes the completing job submit the pack itself:
sbatch --export=ALL,CHAIN_PACK=1 legacy/dataset_build/slurm/extend_dataset_1201.sh 0 1200
# One-off rescue: fold an existing on-scratch build tree into a chunk tar and free the inodes.
sbatch legacy/dataset_build/slurm/snapshot_build_1201.sh

# --- 312-scene official-VAL build — the val ruler for the split above (docs/todo.md 1c) --------
# Same pipeline and QA gates, --scene_list data/splits/scannetv2_val.txt, own chunk dir/tar/README
# so neither the 500- nor the 1201-scene tar is touched. One chunk is enough, so the extend job
# chains the pack itself: the whole build is this one command (~1h20 end to end).
sbatch --export=ALL,CHAIN_PACK=1 legacy/dataset_build/slurm/extend_dataset_val312.sh 0 311
sbatch legacy/dataset_build/slurm/pack_official_gt_val312.sh   # only if packing separately
```

SLURM job logs go to `slurm/logs/` (gitignored). Never let them accumulate in the repo root.

## Architecture

### Upstream VGGT (do not modify; kept frozen)

`vggt/models/vggt.py::VGGT` wraps `vggt/models/aggregator.py::Aggregator` (24 blocks of
alternating per-frame and global cross-frame attention) plus the original heads in `vggt/heads/`.
The `training/` directory is upstream's Co3D finetuning framework — unrelated to this project.

The hook point is `aggregated_tokens_list[-1]`: global scene features `F: [B, S, P, 2048]`
(S frames, P = patch tokens + 1 camera + 4 register tokens; `patch_start_idx` separates them).
The backbone runs under `no_grad` and its features are cached **once per scene up front**, which
is why training takes minutes, not hours.

### The active path (MaskDINO, single-frame)

```
models/maskdino/          the model — see docs/MASKDINO.md §5 for the per-file table
  head.py                 MaskDINOVGGTHead = pixel decoder + decoder (the trainable unit)
  model.py                MaskDINOVGGTModel = frozen VGGT + head
  pixel_decoder.py        VGGT tokens → 3-level ViTDet pyramid → MSDeformAttn encoder
  decoder.py              MaskDINODecoder: two-stage selection, DAB anchors, DN, deep supervision
  decoder_layers.py       the generic DAB/DINO decoder stack it drives
  multiframe.py           --multi_frame: cross-frame attention, bundle GT, bundle matcher
  matcher.py criterion.py ms_deform_attn.py box_ops.py utils.py

scripts/train_maskdino.py entry point: CLI, construction, epoch loop, checkpointing
scripts/eval_perframe.py  scores a D4RT checkpoint on the same protocol (the baseline)
train/maskdino_data.py    per-frame GT + frozen-backbone feature cache + batching
train/maskdino_eval.py    per-frame scoring over cached scenes + figures
train/perframe.py         the protocol itself, shared by both scorers
train/common.py           scene paths, photometric jitter, LR schedule, metrics.jsonl
train/eval_metrics.py     mIoU / AP50 / AP75 / mAP / class_acc (shared with legacy)
data/scannet_overfit.py   the dataset loader (shared with legacy)
```

The batch dimension is **FRAMES**, not scenes. GT is per frame (labels + masks + boxes). With
`--multi_frame` the batch is B bundles of S frames that **stay contiguous** in that dimension
(everything downstream assumes it) and share one query set; the GT is still per frame, re-linked
across views by global instance id at batch time.

### Invariants that silently break things if violated

- **`head_config` must describe every constructor argument** of `MaskDINOVGGTHead`. It is derived
  from `locals()` precisely so a new argument cannot be silently absent from saved checkpoints;
  `tests/test_maskdino_model.py` asserts the two sets are equal. Don't hand-write it back.
- **The class head has 19 sigmoid logits and no background column.** "No object" is *all logits
  low*, so metrics need `score_mode="sigmoid"` plus a score threshold — never an argmax against a
  background column. `build_frame_targets` DROPS instances whose class index falls outside
  `1..num_classes` (with a warning) rather than crashing the matcher; see `docs/MASKDINO.md` §4.
- **A prediction claiming no pixels in a frame is dropped, not counted as a false positive**
  (`train/perframe.py::drop_empty_masks`). Both scorers apply it; removing it makes the protocol
  unfair to the multi-view arms and invalidates the comparison.
- **`initialize_box_type` accepts only `no` and `bitmask`.** Upstream's `mask2box` is not ported
  and the constructor rejects it — it used to share a branch with `bitmask` and alias silently.
- ScanNet class indices are `1..19`, `0` = background, everywhere in the dataset and the loader.
  The MaskDINO head shifts to `0..18` internally and shifts back via `to_scannet_class_logits`.

## Working rules

- **Always proceed step by step**: implement incrementally and test every component you add or
  edit before moving on (run the relevant `tests/test_*.py`, or add one if none covers it).
- **After every change, check whether documentation needs updating** — the files in `docs/` and
  this CLAUDE.md itself.
- New components follow the established pattern: implement in the matching dir (`data/`,
  `models/`, `train/`, `scripts/`), add a standalone CPU-runnable test in `tests/`, document the
  result in `docs/MASKDINO.md`.
- New training options must default to off / previous behaviour so existing tests pass unchanged.
- **Do not "fix" `legacy/`.** It is frozen on purpose: its numbers are published in the docs, and
  changing its behaviour would invalidate the baseline. Bug fixes there need an explicit reason.
