[X] Download and preprocess (compute SAM3 masks) other 4 scenes (total 5 scenes)
[X] Increase training scenes number (train on 4) — scripts/train_multiscene.py, mean train mIoU 0.967 (MILESTONE_1 §9)
[X] Eval on a scene the model has never seen during training (eval on 5th scene) — scene0004_00: mIoU 0.027 final (peaked ~0.13 mid-training); no real generalization with only 4 scenes, see MILESTONE_1 §9
