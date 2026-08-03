# legacy/ — retired code, kept runnable

Nothing here is on the active path. It is kept in-tree (rather than deleted or parked on a
branch) because it is still **executable** and still **cited**: the D4RT numbers are the
baseline every MaskDINO result is measured against, and `scripts/eval_perframe.py` imports
`legacy/d4rt/scripts/train_overfit.py` to produce that baseline.

Every script here `cd`s to the repo root and is run from there, exactly as before:

```bash
python legacy/d4rt/scripts/train_multiscene.py --train_scenes ... --val_scenes ...
sbatch legacy/d4rt/slurm/train_full.sh
python legacy/d4rt/tests/test_phase5.py
```

## legacy/d4rt/ — the D4RT query-strategy arms (Milestones 1–3, closed 2026-07-22)

A DETR-style decoder on frozen VGGT with **multi-view** (8-frame bundle) supervision, studied
across five query-initialisation arms A–E. Arm C (learned object queries) won at every scene
count and peaked at **val mIoU 0.350 / honest AP50 0.177** at N=490 on official GT. Under the
per-frame protocol that makes the two families comparable, arm C scores **0.451 / 0.294** —
which the MaskDINO track beats with 0.669 / 0.699.

Full narrative: `docs/old/MILESTONES.md`; the arm-by-arm verdict table: `docs/ARMS_SUMMARY.md`.

    models/    d4rt_decoder.py (QueryGenerator + InstanceDecoder + mask head),
               anchor_queries.py (arm E 3D anchors), mask_upsampler.py
    train/     loss.py (Hungarian matcher + D4RT loss), postprocess.py (instance selection)
    scripts/   train_multiscene.py, train_overfit.py, visualize_masks.py,
               eval_checkpoint.py, eval_grid_ablation.py, render_pointcloud_topdown.py,
               plot_scaling.py
    tests/     test_phase2–5, test_milestone2, test_anchor_queries, test_postprocess, …
    slurm/     train_full.sh, train_scale{10,25,50,100}.sh, eval_grid_ablation.sh

## legacy/dataset_build/ — the official-GT dataset builders (finished 2026-07-09)

One-shot tooling that produced the ScanNet tars on group storage. The tars are canonical and
already built, so this runs only if a tar is lost or the GT conventions change. Spec and
rebuild order: `docs/DATASET.md`.

    scripts/   build_official_masks.py, download_2d_gt.py, extract_sens_subset.py,
               gen_official_gt_report.py, qa_official_gt_strips.py
    tests/     test_build_official_masks.py, test_extract_sens_subset.py
    slurm/     download_official_gt.sh, extend_dataset_500.sh, pack_official_gt.sh

## What did NOT move

`data/scannet_overfit.py`, `train/eval_metrics.py`, `train/common.py` and `train/perframe.py`
are shared by both families and stay at the repo root. `vggt/` is the untouched upstream
backbone. `demos/demo_gradio.py` stays with the other upstream demos; it serves **both**
families — it dispatches on the checkpoint's keys, so D4RT checkpoints keep working exactly as
before (it still imports from `legacy.d4rt.*`) while MaskDINO checkpoints take the new path in
`train/maskdino_viz3d.py` (docs/MASKDINO.md §9.7).
