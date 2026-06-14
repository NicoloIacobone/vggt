[X] Download and preprocess (compute SAM3 masks) other 4 scenes (total 5 scenes)
[X] Increase training scenes number (train on 4) — scripts/train_multiscene.py, mean train mIoU 0.967 (MILESTONE_1 §9)
[X] Eval on a scene the model has never seen during training (eval on 5th scene) — scene0004_00: mIoU 0.027 final (peaked ~0.13 mid-training); no real generalization with only 4 scenes, see MILESTONE_1 §9

# Milestone 2 (see docs/MILESTONE_2.md)
[X] No-object loss on unmatched queries (DETR eos) — train/loss.py `no_object_weight`
[X] Unprompted inference/eval on a uniform query grid — `generate_grid_queries` + dual prompted/unprompted metrics
[X] Regularization: multi-bundle random frame sampling, query jitter, bg resampling, color jitter
[X] Best-checkpoint on val mIoU + optional early stopping (`checkpoint_best.pth`)
[ ] Download + preprocess more scenes (tens-to-hundreds) — SAM3 masks; label format DECIDED with supervisor (Jun 12): per-INSTANCE masks (docs/supervisor_feedback_jun_12.md §4)
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
[ ] Per-instance loader + tests once instance-mask data lands — data/scannet_overfit.py ID-per-(class,instance), update tests/test_phase2.py (§4) — BLOCKED on SAM3
[X] MaskDINO-style pixel decoder CODE: models/mask_upsampler.py upsamples patch features before the cosine-sim mask product (--mask_upsample) — [ ] train + (if dense OOM) point-sampled mask loss (§2)
[X] Viz polish: legend "{class} #{k}" for same-class instances; caption "one color = one predicted instance (mask spans all frames jointly)" (§1)
