"""
The extra config keys the control arm needs on top of upstream's `add_maskdino_config`.

Kept in a `CONTROL` namespace so nothing collides with detectron2's or MaskDINO's own keys —
the same yaml can therefore be diffed against upstream's `maskdino_R50_bs16_50ep_3s.yaml`
and every line that differs is an axis we deliberately matched to our arms.
"""

from detectron2.config import CfgNode as CN


def add_control_config(cfg):
    cfg.INPUT.SQUASH_SIZE = 518          # our arms' fixed square input (VGGT's native 37x37 grid)

    cfg.CONTROL = CN()
    # data ---------------------------------------------------------------------------------
    cfg.CONTROL.COCO_ROOT = "/cluster/scratch/niacobone/coco"
    cfg.CONTROL.TRAIN_JSON = "annotations/instances_train2017.json"
    cfg.CONTROL.TRAIN_IMAGES = "train2017"
    cfg.CONTROL.VAL_JSON = "annotations/instances_val2017.json"
    cfg.CONTROL.VAL_IMAGES = "val2017"
    # The population the PERIODIC evals score, so the control's curve is comparable to our arms'
    # curve step for step. Our arms evaluate on `--eval_images 1000` while training and on all
    # 5000 only at the end; COCOEvaluator scores whatever is in the registered GT json, so the
    # subset has to be a registered dataset of its own. Built once by make_val_subset.py.
    # "" = score the full val at every periodic eval (what the run did before 2026-08-08).
    cfg.CONTROL.VAL_SUBSET_JSON = "annotations/instances_val2017_first1000.json"
    cfg.CONTROL.HFLIP_PROB = 0.5         # our arms' only augmentation
    # schedule -----------------------------------------------------------------------------
    # `scripts/train_maskdino_coco.py::build_step_scheduler`: linear warmup then cosine down to
    # `min_lr_ratio` * BASE_LR. detectron2's own WarmupCosineLR decays to exactly 0, which is a
    # different endgame, so the schedule is reimplemented rather than approximated.
    cfg.CONTROL.COSINE_END_LR_RATIO = 0.01
    # Horizon the cosine is computed over, decoupled from where training stops. 0 = MAX_ITER,
    # which is what the real run and our three arms use. Only the overfit gate sets it: a cosine
    # squeezed into 600 steps runs the gate's endgame at ~1e-6 and caps it at 28 AP, which is a
    # schedule artefact, not a statement about the pipeline the gate exists to check.
    cfg.CONTROL.LR_HORIZON_ITERS = 0
    # driver -------------------------------------------------------------------------------
    # Wall-clock self-stop. python must exit BEFORE SLURM's wall clock or the resubmit at the
    # tail of the batch script is torn down with the rest of the job and the run stops silently
    # half-finished (slurm/train_maskdino_coco.sh header; same failure mode here).
    cfg.CONTROL.TIME_BUDGET_HOURS = 0.0  # 0 = no budget
    # A100s are sm_80. The MSDeformAttn .so shipped in the clone's venv holds sm_86 code only,
    # and upstream's `MSDeformAttn.forward` wraps the CUDA call in a bare `except:` that falls
    # back to the pure-pytorch core — so a wrong-arch build costs ~10x throughput SILENTLY
    # instead of crashing. Assert the kernel really runs before spending days on it.
    cfg.CONTROL.REQUIRE_CUDA_MSDA = True
