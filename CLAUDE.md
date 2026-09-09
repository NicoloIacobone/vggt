# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

> **This project is CLOSED (2026-09-09).** No work is in flight, no jobs are pending, and the
> cluster allocation the results were produced on has been released. Read
> `docs/FINAL_REPORT.md` before anything else — it is the closing account and it supersedes every
> "next step" written anywhere else in `docs/`.
>
> The repository is kept as a **reference artefact**: the code, the protocols and the complete
> measurement record. Treat every document here as frozen. If you are asked to change something,
> the default is a *documentation* or *presentation* change, not a research change.

## Project

A research project carried out at **ETH Zurich — Photogrammetry & Remote Sensing**,
June–September 2026 (supervisors: Christos Sakaridis, Mattia Segù, Tatiana Tommasi). It is **not**
a master's thesis — it followed one.

The repository is a fork of **VGGT** (Visual Geometry Grounded Transformer, CVPR 2025) — a
feed-forward 3D reconstruction model. The goal was **not** to modify VGGT, but to attach and train
a decoder for **3D multi-view consistent instance segmentation** on top of the frozen VGGT-1B
backbone, supervised by the **official ScanNet v2 2D instance annotations**.

The model is a **MaskDINO decoder** (`models/maskdino/`) trained on the official ScanNet v2
1201/312 split or larger. On the official 3D instance benchmark, unposed and class-agnostic, at
the competitors' own **50 views**, it scores **0.053 / 0.170 / 0.542** (AP/AP50/AP25) from a
strictly frozen backbone — ahead of FAST3DIS's 0.038 / 0.096 / 0.316 and IGGT's
0.028 / 0.112 / 0.287 on all three.

**That lead is not training-matched, and the price is measured — quote the two together.** Both
competitors are zero-shot on ScanNet and every headline row of ours trains on it. With ScanNet
removed and **IGGT's mixture reproduced in full** (arm I-ase, +ASE, 2026-09-03) the same recipe
reads **0.009 / 0.032 / 0.301**: ~3× behind at AP50, **level at AP25**, at a third of their views
(`docs/MULTIDATASET.md` §12.4). **That gap, together with an exhausted compute budget, is why the
project closed** — the reasoning is in `docs/FINAL_REPORT.md` §5.

`legacy/` is frozen on purpose. It holds the previous hand-rolled head — kept because
`scripts/eval_perframe.py` and `demos/demo_gradio.py` still import it — and the dataset builders,
which `train/scannetpp3d.py`, `scripts/verify_scannetpp_gt.py` and three tests import. Do not
surface any of it in new work.

Material retired at project close — `docs/old/`, `docs/slides/`, `legacy/coco/`, upstream's
`examples/` media — was **deleted** from the tree. It stays in git history and is listed in
`docs/FINAL_REPORT.md` §8.1. Do not restore any of it without being asked, and do not cite it as
current.

### Docs — read in this order

- `docs/FINAL_REPORT.md` — **the closing account. Read this first, always.** What was built, what
  it scored, why it stopped, what a successor should do, and the rules that still apply to any
  number taken from the project. Everything below is the record underneath it.
- `docs/MASKDINO.md` — **the primary technical document.** Architecture, deviations from upstream
  MaskDINO, the protocols, the evaluation rules, the multi-frame mechanisms.
- `docs/RESULTS.md` — **every number, one home.** Everything in it is on the official 1201/312
  split or larger; read §1 before quoting anything, because 2D and 3D numbers are **not**
  interchangeable and only the 3D ones face a published paper.
- `docs/FACTSHEET.md` — the frozen outward-facing read-out as it stood at closure: the numbers
  cleared for quoting, the rulers **in two tiers** (Tier 1 = the 3D rulers that face the published
  competitors; Tier 2 = the internal 2D rulers, backup only), the positioning. It never
  contradicts RESULTS.md — if it does, RESULTS.md wins and FACTSHEET.md is the bug.
- `docs/COMMANDS.md` — the full command catalogue (tests, training, 3D ruler, dataset rebuilds)
  with the caveats each one needs.
- `docs/DATASET.md` — GT provenance, the tars, mask conventions, how a job got the data.
- `docs/MULTIDATASET.md` — the multi-dataset training arm (ScanNet + ScanNet++ + Infinigen, plus
  RE10K and **ASE** in their own arms, `--class_agnostic`). Its rows are **class-agnostic** and
  never comparable to RESULTS' §6. Anything trained on RE10K is additionally **SAM2-supervised** —
  the masks are model output, not ground truth — and carries a separate labelled row (§1.3, §11).
  **§12.4 is the no-ScanNet result that prices the headline**; §12.3 is its pre-ASE version and
  must not be quoted against a competitor.
- `docs/RELATED_WORK.md` — competitor landscape & positioning. Read before framing any result as a
  contribution.
- `docs/TRAINING_COMPARABILITY.md` — what each competitor **trains** on vs evaluates on. Read
  alongside RELATED_WORK: that file covers the evaluation side, this one the training side.
- `docs/SEGVGGT_ANALYSIS.md` — the closest competitor, dissected: no training code, where the
  residual AP50 gap goes (**×2.8 on the `--anchor_3d` checkpoint**, once the ×2.3 bridge is taken
  out — always quote the residual with the checkpoint it was measured on), and the conceptual
  difference.
- `docs/todo.md` — the work ledger, **frozen at closure**. Kept because ~40 comments in live source
  files cite its item numbers. `docs/FINAL_REPORT.md` §7 supersedes every unchecked box in it.
- `docs/RESTORE.md` — the cluster archive layout, the venv rebuild recipe, and the hardcoded
  `/cluster/...` paths that must be re-rooted before anything runs. Read it first if this repo was
  restored from the 2026-08-13 backup zip.

The pre-official-split archive (`docs/old/`) and the supervisor slide decks (`docs/slides/`) were
removed at project close; they are recoverable from git history. Documents that cite them by path
are citing that history — do not treat the links as live.

## Environment & core commands

A virtualenv is expected in-repo at `myenv/` — use `myenv/bin/python`. It is **not** committed.
The rebuild recipe, the exact resolved versions every published number was produced with
(**torch 2.3.1+cu121 / torchvision 0.18.1+cu121**), and the scratch-purge failure mode that
destroyed it twice are all in `docs/RESTORE.md` §2.

Everything ran on a SLURM GPU cluster; matplotlib must stay headless (`Agg`). SLURM logs go to
`slurm/logs/` (gitignored) — never let them accumulate in the repo root.

```bash
# Tests — standalone scripts, not pytest; all CPU-only, no backbone weights needed.
for t in tests/test_*.py; do python "$t"; done
bash tests/test_train_maskdino_sh_lists.sh          # slurm scene-list logic, DRY_RUN
bash tests/test_train_maskdino_multi_sh.sh          # …the multi-dataset driver, incl. errexit
bash tests/test_eval_3d_matrix_sh.sh                # …the cross-dataset eval grid, DRY_RUN

# Training (the entry point)
sbatch slurm/train_maskdino.sh                      # 50 scenes, ~20k steps
sbatch --export=ALL,N_SCENES=490 slurm/train_maskdino.sh
sbatch --export=ALL,N_SCENES=490,EXTRA_ARGS='--multi_frame --feature_mode bundle' \
    slurm/train_maskdino.sh
python scripts/train_maskdino.py --train_scenes scene0000_00 --val_scenes scene0080_00 \
    --num_epochs 50 --num_queries 300 --scans_root <scans_root>       # local smoke test

# 3D ruler — the only protocol placeable next to published numbers (docs/MASKDINO.md §9)
sbatch --export=ALL,CHECKPOINT=<run_dir>/checkpoint_best_bundle.pth slurm/eval_3d_maskdino.sh
# …the same ruler on the other benchmarks (§9.12); DATASET defaults to scannetv2
sbatch --export=ALL,DATASET=scannetpp,CHECKPOINT=<ckpt> slurm/eval_3d_maskdino.sh

# Figures / qualitative
sbatch --export=ALL,RUNS='<run_dir>' slurm/visualize_maskdino.sh
python demos/demo_gradio.py --seg_checkpoint <run_dir>/checkpoint_best_bundle.pth \
    --seg_scans_root <demo_scans_root>
```

Everything else — official-split recipes, `--anchor_3d`, `--eval_num_frames`, `--eval_full_res`,
the two 3D transfer modes and their oracle, dataset rebuilds — is in `docs/COMMANDS.md`.

## Architecture

### Upstream VGGT (do not modify; kept frozen)

`vggt/models/vggt.py::VGGT` wraps `vggt/models/aggregator.py::Aggregator` (24 blocks of alternating
per-frame and global cross-frame attention) plus the original heads in `vggt/heads/`. The
`training/` directory is upstream's Co3D finetuning framework — unrelated to this project and
unused by it.

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

scripts/train_maskdino.py entry point: CLI, construction, epoch loop, checkpointing;
                          `--eval_only` scores a finished run without training (slurm/eval_only_maskdino.sh)
scripts/eval_perframe.py  scores a legacy checkpoint on the same protocol (the baseline)
train/maskdino_data.py    per-frame GT + frozen-backbone feature cache + batching
train/maskdino_eval.py    per-frame scoring over cached scenes + figures
train/perframe.py         the protocol itself, shared by both scorers
train/common.py           scene paths, photometric jitter, LR schedule, metrics.jsonl
train/eval_metrics.py     mIoU / AP50 / AP75 / mAP / class_acc; cross-view identity —
                          HOTA / AssA / DetA / IDF1 (the formal, published ones) next to the
                          project's own view_consistency / id_switch (docs/MASKDINO.md §6.6.1)
data/scannet_overfit.py   the dataset loader

the 3D ruler (docs/MASKDINO.md §9) and its four benchmarks (§9.12)
  train/benchmark3d.py    the VENDORED official ScanNet evaluator — do not touch
  train/eval3d_geometry.py  Sim(3)+ICP, the two 2D→3D transfers, the vote lifting
  train/datasets3d.py     the `--dataset` registry: scannetv2 | scannet200 | scannetpp | replica
  train/scannet3d.py      + train/scannetpp3d.py, train/replica3d.py — one adapter per dataset,
                          same interface; `data/scannet200_constants.py` is the 200-class map
  scripts/eval_3d_maskdino.py  the ruler; scripts/gate_3d_gt.py  the per-dataset licence gate
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

Now that the project is closed, these apply to maintenance work:

- **Do not restart the research.** New experiments, new arms, new training runs are out of scope
  unless explicitly asked for. `docs/FINAL_REPORT.md` §7 is the ranked list of what *would* be
  worth doing, and it is a hand-off, not a plan.
- **Do not edit the numbers.** `docs/RESULTS.md` and `docs/MULTIDATASET.md` are the frozen record.
  If a number looks wrong, say so — do not silently change it.
- **Always proceed step by step**: implement incrementally and test every component you touch
  (run the relevant `tests/test_*.py`, or add one if none covers it).
- **After every change, check whether documentation needs updating** — `docs/` and this file.
- **Do not "fix" `legacy/`.** It is frozen on purpose: its numbers are published, and changing its
  behaviour would invalidate the baseline.
- Keep docs short. One fact, one home — cross-reference instead of restating.
