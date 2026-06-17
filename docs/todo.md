[X] Download and preprocess (compute SAM3 masks) other 4 scenes (total 5 scenes)
[X] Increase training scenes number (train on 4) — scripts/train_multiscene.py, mean train mIoU 0.967 (MILESTONE_1 §9)
[X] Eval on a scene the model has never seen during training (eval on 5th scene) — scene0004_00: mIoU 0.027 final (peaked ~0.13 mid-training); no real generalization with only 4 scenes, see MILESTONE_1 §9

# Milestone 2 (see docs/MILESTONE_2.md)
[X] No-object loss on unmatched queries (DETR eos) — train/loss.py `no_object_weight`
[X] Unprompted inference/eval on a uniform query grid — `generate_grid_queries` + dual prompted/unprompted metrics
[X] Regularization: multi-bundle random frame sampling, query jitter, bg resampling, color jitter
[X] Best-checkpoint on val mIoU + optional early stopping (`checkpoint_best.pth`)
[X] Download + preprocess more scenes (tens-to-hundreds) — SAM3 masks; per-INSTANCE format. DONE (Jun 15): 97 scenes (scene0000–0096), 2056 instances, `masks_instance/<class>_<k>/`, shipped as `scannet_instance_dataset.tar.zst` (see INSTANCE_MASKS_README.md). Per-class `masks/` retained.
[ ] Scaling experiment: train on N ∈ {10, 25, 50, 100+} scenes, val on held-out scenes (MILESTONE_2 §7.1) — first scale10/scale25 runs done but scale25 invalidated by premature early stop; fix SLURM scripts + re-run per docs/SCALING_RUNS_ANALYSIS.md §4 before launching scale50
[ ] No-object weight + augmentation ablations on the larger dataset (MILESTONE_2 §7.2–7.4) — blocked on data

# Milestone 3 / Phase 0 — instrumentation & small fixes (see docs/MILESTONE_3.md, docs/NEXT_STEPS_PLAN.md)
[X] Persist eval history → <run_dir>/metrics.jsonl (epoch, lr, loss, prompted+grid train/val mIoU & AP50)
[X] Shrink checkpoints: uint8 images (default, 4× smaller) + --checkpoint_light (drop pixels, reload from disk)
[X] Noise-robust early stopping: --early_stop_min_delta + --early_stop_window moving average, refuse before half schedule (off by default)
[X] Second best checkpoint on val[grid] AP50 → checkpoint_best_ap50.pth
[X] --schedule_epochs to decouple cosine schedule length from --num_epochs
[X] Viz polish: legend "{class} #{k}", caption "one color = one predicted instance", --score_threshold exposed
[X] Fix SLURM scripts: identical protocol (--eval_interval 50 --early_stop_patience 0), --time trimmed to 2 h
[ ] Phase 1: fair scaling re-runs (GPU) — scale25/scale10 full-schedule, then scale50; plot mIoU/AP50 vs N from metrics.jsonl
[X] Phase 2 CODE: --train_grid_queries (random-offset grid in make_train_queries, off by default) — [ ] GPU experiment (scale10/25 with vs without) after Phase 1
[X] Phase 3 CODE: --query_mode {point,learned,hybrid} in QueryGenerator + head_config round-trip + matcher coord_weight=0 for learned — [ ] GPU experiment arms A/B/C/D after Phase 1
[X] Phase 5 CODE: MaskDINO pixel decoder (models/mask_upsampler.py + --mask_upsample, default 1 = unchanged) — [ ] train after Phases 1–3 settle

# Supervisor feedback Jun 12 (see docs/supervisor_feedback_jun_12.md)
[X] `--train_grid_queries` CODE: include the eval grid in training so Hungarian + no-object loss learn duplicate suppression (DETR-style, no NMS) — [ ] run the unprompted-AP50 experiment (§3)
[X] `--query_mode` CODE: point prompts vs learned object queries vs hybrid, coord_weight=0 for learned mode — [ ] run the ablation (§5)
[X] Per-instance loader + tests CODE (Jun 15) — data/scannet_overfit.py `instance_level` flag (per-(class,instance) IDs from masks_instance/<class>_<k>/, default off = per-class unchanged); --instance_level in train_overfit/train_multiscene; tests/test_phase2.py::test_instance_dataset.
[X] GPU: instance-GT scaling curve N∈{10,25,50} (Jun 15) — first pass (3-scene val) val mIoU 0.142/0.136/0.185. Fixed exit-code footgun (low train mIoU no longer = FAILED).
[X] GPU: wide-val (0080–0089) instance curve + Phases 2/3 arms at N=50 (Jun 15) — curve now MONOTONIC: val mIoU 0.152/0.174/0.212, val[grid] AP50 0.089/0.111/0.125. Arms: A point 0.212, C learned 0.259 (BEST), B grid_queries 0.047 (mask learning collapsed), D hybrid CRASHED (NaN in matcher). See MILESTONE_3 "Phases 2/3/4 instance-GT, wide-val results".
[ ] Fix Phase-2 grid-query loss balance (normalize no-object by query count / keep centroid queries matched) + rerun; fix hybrid NaN (guard matcher cost, grad-clip learned params) + rerun
[X] MaskDINO-style pixel decoder CODE: models/mask_upsampler.py upsamples patch features before the cosine-sim mask product (--mask_upsample) — [ ] train + (if dense OOM) point-sampled mask loss (§2)
[X] Viz polish: legend "{class} #{k}" for same-class instances; caption "one color = one predicted instance (mask spans all frames jointly)" (§1)

# Data management
[X] Keep file count low: the per-instance dataset is shipped as ONE zstd tar `scannet_instance_dataset.tar.zst` (~1.3 GB on `work`). Each job copies it to node-local `$TMPDIR` and unzips there (`slurm/stage_dataset.sh` → exports `SCANNET_ROOT`; train_multiscene.py honors it as default `--scans_root`). Build was done from the fast `/cluster/scratch/niacobone/scannet_build/scans` tree. See INSTANCE_MASKS_README.md "Consuming the dataset at training time".