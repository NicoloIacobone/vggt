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

# Supervisor feedback Jun 12 (see docs/supervisor_feedback_jun_12.md)
[ ] `--train_grid_queries`: include the eval grid in training so Hungarian + no-object loss learn duplicate suppression (DETR-style, no NMS) — expected unprompted-AP50 fix (§3)
[ ] Ablation: point prompts vs learned object queries (vs hybrid) — `--query_mode` in QueryGenerator, coord_weight=0 for learned mode (§5)
[ ] Per-instance loader + tests once instance-mask data lands — data/scannet_overfit.py ID-per-(class,instance), update tests/test_phase2.py (§4)
[ ] MaskDINO-style pixel decoder above the 37×37 grid — upsample patch features before the cosine-sim mask product; point-sampled mask loss if dense doesn't fit (§2)
[ ] Viz polish: legend "{class} #{k}" for same-class instances; caption overlays "one color = one predicted instance (mask spans all frames)" (§1)
