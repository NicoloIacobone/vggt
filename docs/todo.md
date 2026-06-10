[X] Download and preprocess (compute SAM3 masks) other 4 scenes (total 5 scenes)
[X] Increase training scenes number (train on 4) — scripts/train_multiscene.py, mean train mIoU 0.967 (MILESTONE_1 §9)
[X] Eval on a scene the model has never seen during training (eval on 5th scene) — scene0004_00: mIoU 0.027 final (peaked ~0.13 mid-training); no real generalization with only 4 scenes, see MILESTONE_1 §9

# Milestone 2 (see docs/MILESTONE_2.md)
[X] No-object loss on unmatched queries (DETR eos) — train/loss.py `no_object_weight`
[X] Unprompted inference/eval on a uniform query grid — `generate_grid_queries` + dual prompted/unprompted metrics
[X] Regularization: multi-bundle random frame sampling, query jitter, bg resampling, color jitter
[X] Best-checkpoint on val mIoU + optional early stopping (`checkpoint_best.pth`)
[ ] Download + preprocess more scenes (tens-to-hundreds) — SAM3 masks; consider per-INSTANCE (not per-class) masks (MILESTONE_2 §7.5)
[ ] Scaling experiment: train on N ∈ {10, 25, 50, 100+} scenes, val on held-out scenes (MILESTONE_2 §7.1) — blocked on data
[ ] No-object weight + augmentation ablations on the larger dataset (MILESTONE_2 §7.2–7.4) — blocked on data
