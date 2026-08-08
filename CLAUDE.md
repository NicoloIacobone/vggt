# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

## Project

A fork of **VGGT** (Visual Geometry Grounded Transformer, CVPR 2025) — a feed-forward 3D
reconstruction model. The goal is **not** to modify VGGT, but to attach and train a decoder for
**3D multi-view consistent instance segmentation** on top of the frozen VGGT-1B backbone.
Supervision is the **official ScanNet v2 2D instance annotations**.

The active model is a **MaskDINO decoder** (`models/maskdino/`). Single-frame at 490 scenes:
val mIoU **0.669** / AP50 **0.699**, against the retired baseline head's 0.451 / 0.294 on the same
per-frame protocol. `--multi_frame` (shared queries across a bundle) is implemented; widening the
bundle and improving the 3D lifting are the open work. All numbers: `docs/RESULTS.md`.

`legacy/` is frozen on purpose — the previous hand-rolled head, kept because
`scripts/eval_perframe.py` and `demos/demo_gradio.py` still import it. Its story is archived in
`docs/old/`; do not surface it in new work.

### Docs — read in this order

- `docs/MASKDINO.md` — **the primary document.** Architecture, deviations from upstream MaskDINO,
  the protocols, the evaluation rules, the multi-frame mechanisms.
- `docs/RESULTS.md` — **every number, one home.** Read §1 before quoting anything: per-frame,
  per-bundle and 3D numbers are **not** interchangeable.
- `docs/COMMANDS.md` — the full command catalogue (tests, training, 3D ruler, COCO, dataset
  rebuilds) with the caveats each one needs.
- `docs/MASKDINO_COCO.md` — the COCO backbone-swap study. Contains the **mask-resolution ceiling
  measurement** (§1) — read it before proposing anything that depends on mask or token resolution.
- `docs/DATASET.md` — GT provenance, the tars, mask conventions, how a job gets the data.
- `docs/todo.md` — open work only.
- `docs/RELATED_WORK.md` — competitor landscape & positioning. Read before framing any result as a
  contribution.
- `docs/TRAINING_COMPARABILITY.md` — what each competitor **trains** on vs evaluates on, what is on
  the cluster, what is missing. Read alongside RELATED_WORK: that file covers the evaluation side,
  this one the training side.
- `docs/SEGVGGT_ANALYSIS.md` — the closest competitor, dissected: no training code, where the
  ×10.7→×2.8 AP50 gap goes, and the conceptual difference.
- `docs/SUPERVISOR_COMPARISON.md`, `docs/RIEPILOGO_PROGETTO_IT.md` — send-outward summaries derived
  from the two files above. If a number changes there, change it here too.
- `docs/old/` — archive. Nothing in it is current; don't cite it as a source of truth.

## Environment & core commands

A virtualenv lives in-repo at `myenv/` — use `myenv/bin/python`.

> ⚠ **`myenv/` is on scratch, and scratch purges files 15 days after last access — it destroyed
> the venv on 2026-08-07.** Python imports read `__pycache__/*.pyc` and never touch the `.py`
> source, so the sources' atime goes stale, the purge deletes them, and the *still-present* `.pyc`
> then fails to validate. Symptom: `ModuleNotFoundError: No module named 'torch._vendor...'` or
> `module 'torch._dynamo' has no attribute 'disable'` — a torch that imports partially, in a tree
> with far fewer `.py` than `.pyc`. Diagnose with
> `find myenv -name '*.py' | wc -l` vs `-name '*.pyc' | wc -l`. Rebuilding it on scratch just
> restarts the 15-day clock; `$HOME` is not purged and has room. `requirements.txt` pins
> torch 2.3.1 / torchvision 0.18.1.

Runs on a GPU cluster node;
matplotlib must stay headless (`Agg`). SLURM logs go to `slurm/logs/` (gitignored) — never let them
accumulate in the repo root.

```bash
# Tests — standalone scripts, not pytest; all CPU-only, no backbone weights needed.
for t in tests/test_*.py; do python "$t"; done
bash tests/test_train_maskdino_sh_lists.sh          # slurm scene-list logic, DRY_RUN

# Training (the entry point)
sbatch slurm/train_maskdino.sh                      # 50 scenes, ~20k steps
sbatch --export=ALL,N_SCENES=490 slurm/train_maskdino.sh
sbatch --export=ALL,N_SCENES=490,EXTRA_ARGS='--multi_frame --feature_mode bundle' \
    slurm/train_maskdino.sh
python scripts/train_maskdino.py --train_scenes scene0000_00 --val_scenes scene0080_00 \
    --num_epochs 50 --num_queries 300 --scans_root <scans_root>       # local smoke test

# 3D ruler — the only protocol placeable next to published numbers (docs/MASKDINO.md §9)
sbatch --export=ALL,CHECKPOINT=<run_dir>/checkpoint_best_bundle.pth slurm/eval_3d_maskdino.sh

# Figures / qualitative
sbatch --export=ALL,RUNS='<run_dir>' slurm/visualize_maskdino.sh
python demos/demo_gradio.py --seg_checkpoint <run_dir>/checkpoint_best_bundle.pth \
    --seg_scans_root /cluster/scratch/niacobone/demo_scans/scans
```

Everything else — official-split recipes, `--anchor_3d`, `--eval_num_frames`, `--eval_full_res`,
the two 3D transfer modes and their oracle, COCO, dataset rebuilds — is in `docs/COMMANDS.md`.

## Architecture

### Upstream VGGT (do not modify; kept frozen)

`vggt/models/vggt.py::VGGT` wraps `vggt/models/aggregator.py::Aggregator` (24 blocks of alternating
per-frame and global cross-frame attention) plus the original heads in `vggt/heads/`. The
`training/` directory is upstream's Co3D finetuning framework — unrelated to this project.

The hook point is `aggregated_tokens_list[-1]`: global scene features `F: [B, S, P, 2048]`
(S frames, P = patch tokens + 1 camera + 4 register tokens; `patch_start_idx` separates them). The
backbone runs under `no_grad` and its features are cached **once per scene up front**, which is why
training takes minutes, not hours.

### The active path

```
models/maskdino/          the model — see docs/MASKDINO.md §5 for the per-file table
  head.py                 MaskDINOVGGTHead = pixel decoder + decoder (the trainable unit)
  model.py                MaskDINOVGGTModel = frozen VGGT + head
  pixel_decoder.py        VGGT tokens → 3-level ViTDet pyramid → MSDeformAttn encoder
  decoder.py              MaskDINODecoder: two-stage selection, DAB anchors, DN, deep supervision
  decoder_layers.py       the generic DAB/DINO decoder stack it drives
  multiframe.py           --multi_frame: cross-frame attention, bundle GT, bundle matcher
  anchor3d.py             --anchor_3d: 3D anchors instead of 2D DAB boxes (the §8.3 ablation)
  matcher.py criterion.py ms_deform_attn.py box_ops.py utils.py

scripts/train_maskdino.py entry point: CLI, construction, epoch loop, checkpointing
scripts/eval_perframe.py  scores a legacy checkpoint on the same protocol (the baseline)
train/maskdino_data.py    per-frame GT + frozen-backbone feature cache + batching
train/maskdino_eval.py    per-frame scoring over cached scenes + figures
train/perframe.py         the protocol itself, shared by both scorers
train/common.py           scene paths, photometric jitter, LR schedule, metrics.jsonl
train/eval_metrics.py     mIoU / AP50 / AP75 / mAP / class_acc
data/scannet_overfit.py   the dataset loader
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
  (`train/perframe.py::drop_empty_masks`). Both scorers apply it; removing it changes the protocol
  and invalidates every comparison already published.
- **`initialize_box_type` accepts only `no` and `bitmask`.** Upstream's `mask2box` is not ported and
  the constructor rejects it — it used to share a branch with `bitmask` and alias silently.
- ScanNet class indices are `1..19`, `0` = background, everywhere in the dataset and the loader. The
  MaskDINO head shifts to `0..18` internally and shifts back via `to_scannet_class_logits`.

## Working rules

- **Always proceed step by step**: implement incrementally and test every component you add or edit
  before moving on (run the relevant `tests/test_*.py`, or add one if none covers it).
- **After every change, check whether documentation needs updating** — `docs/` and this file.
- New components follow the established pattern: implement in the matching dir (`data/`, `models/`,
  `train/`, `scripts/`), add a standalone CPU-runnable test in `tests/`, document the result in
  `docs/MASKDINO.md` and the number in `docs/RESULTS.md`.
- New training options must default to off / previous behaviour so existing tests pass unchanged.
- **Do not "fix" `legacy/`.** It is frozen on purpose: its numbers are published, and changing its
  behaviour would invalidate the baseline. Bug fixes there need an explicit reason.
- Keep docs short. One fact, one home — cross-reference instead of restating.
