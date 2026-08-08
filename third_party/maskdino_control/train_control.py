#!/usr/bin/env python3
"""
Train the OFFICIAL MaskDINO on COCO under OUR recipe -- the missing control row of
docs/MASKDINO_COCO.md §6.

WHY. §6 puts our three frozen-backbone arms (resnet50 34.3, vggt 37.7, dinov2 38.8 mask AP)
next to "46.1". That 46.1 is upstream's RELEASED CHECKPOINT, verified by our own inference
(docs/MASKDINO.md §7.6, job 8967932) -- no MaskDINO has ever been TRAINED here. So the 12 AP gap
confounds three things at once: 50 epochs vs 12, a finetuned R50 vs a frozen one, and LSJ@1024
vs squash@518. This run holds all three at OUR values and changes nothing else, which turns
"upstream is 12 AP ahead" into "our recipe costs the R50 X AP".

It is also the first TRAINING-path check of our port. §7.6 certifies the inference path and
explicitly excludes `matcher.py`, `criterion.py` and DN generation. Upstream trained under our
recipe should land near our `resnet50` arm's 34.3; far above means our training path has a bug.

WHAT IS CHANGED vs upstream's `maskdino_R50_bs16_50ep_3s.yaml` (and nothing else):

    axis            upstream                        this run
    step budget     MAX_ITER 368750 (50 ep)         MAX_ITER 87948 (12 ep)
    backbone        finetuned, BACKBONE_MULT 0.1    FREEZE_AT 5 + BACKBONE_MULTIPLIER 0.0
    input           LSJ 1024, scale 0.1-2.0         518x518 squash + hflip 0.5 (squash_mapper.py)
    LR schedule     multistep @0.889/0.963          linear warmup 1000 -> cosine to 0.01 (lr.py)
    grad clip       full_model 0.01                 full_model 0.1
    padding         SIZE_DIVISIBILITY 32            1  (518 is not a multiple of 32; padding to
                                                    544 would show the backbone a black border
                                                    our arms never see)
Queries / decoder depth / DN / two-stage / bitmask box init / TRAIN_NUM_POINTS 12544 / batch 16 /
AdamW 1e-4 wd 0.05 / COCOeval segm+bbox on all 5000 val2017 at topk 100 were ALREADY identical.

Upstream's clone at $MASKDINO_ROOT is never edited -- it is imported. That is what keeps §7.6
reproducible.

    # the mandatory overfit gate (docs/MASKDINO_COCO.md §4.1): 64 images, train == val
    python third_party/maskdino_control/train_control.py \
        --config-file third_party/maskdino_control/configs/maskdino_upstream_matched_overfit.yaml

    # the real run (driven by slurm/train_maskdino_upstream.sh, which self-resubmits)
    python third_party/maskdino_control/train_control.py \
        --config-file third_party/maskdino_control/configs/maskdino_upstream_matched.yaml \
        --resume OUTPUT_DIR <run_dir> CONTROL.TIME_BUDGET_HOURS 22.5
"""

import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
MASKDINO_ROOT = os.environ.get("MASKDINO_ROOT", "/cluster/scratch/niacobone/MaskDINO")
# ops_build/ holds MultiScaleDeformableAttention rebuilt for sm_80+sm_86 (see build_ops.sh).
# It must precede site-packages, whose copy is sm_86-only and would make upstream's bare
# `except:` in MSDeformAttn.forward fall back to the slow pytorch core on an A100.
sys.path.insert(0, str(_HERE / "ops_build"))
sys.path.insert(0, MASKDINO_ROOT)
sys.path.insert(0, str(_HERE.parent.parent))

import torch  # noqa: E402

import detectron2.utils.comm as comm  # noqa: E402
from detectron2.checkpoint import DetectionCheckpointer  # noqa: E402
from detectron2.config import get_cfg  # noqa: E402
from detectron2.data import MetadataCatalog, build_detection_test_loader, \
    build_detection_train_loader  # noqa: E402
from detectron2.data.datasets import register_coco_instances  # noqa: E402
from detectron2.engine import HookBase, default_argument_parser, default_setup, hooks, \
    launch  # noqa: E402
from detectron2.evaluation import COCOEvaluator  # noqa: E402
from detectron2.projects.deeplab import add_deeplab_config  # noqa: E402
from detectron2.utils.logger import setup_logger  # noqa: E402

from maskdino import add_maskdino_config  # noqa: E402  (upstream clone)
from train_net import Trainer as UpstreamTrainer  # noqa: E402  (upstream clone)

from third_party.maskdino_control.config import add_control_config  # noqa: E402
from third_party.maskdino_control.lr import build_matched_lr_scheduler  # noqa: E402
from third_party.maskdino_control.squash_mapper import CocoSquashDatasetMapper  # noqa: E402

TRAIN_DATASET = "control_coco_2017_train"
VAL_DATASET = "control_coco_2017_val"
VAL_SUBSET_DATASET = "control_coco_2017_val_subset"


class TimeBudgetReached(BaseException):
    """Not an Exception on purpose: detectron2's train loop logs a traceback for those."""


class TimeBudgetHook(HookBase):
    """
    Stops training before SLURM's wall clock so the batch script's resubmit can actually run.

    The naive "resubmit at the end of the batch script" does not work: at the wall clock SLURM
    tears the whole script down, the trailing `sbatch` never executes, and the study silently
    stops half-finished (slurm/train_maskdino_coco.sh header). So python stops itself first.
    """

    def __init__(self, hours: float):
        self.budget_s = hours * 3600.0

    def before_train(self):
        self._t0 = time.time()

    def after_step(self):
        if self.budget_s <= 0:
            return
        if time.time() - self._t0 >= self.budget_s:
            raise TimeBudgetReached


def assert_cuda_msda():
    """
    Upstream wraps the fused MSDeformAttn CUDA call in a bare `except:` and falls back to the
    pure-pytorch core. A wrong-arch build therefore costs ~10x throughput SILENTLY instead of
    crashing -- which over 88k steps is the difference between 3 days and 3 weeks.
    """
    import MultiScaleDeformableAttention as MSDA
    from maskdino.modeling.pixel_decoder.ops.functions.ms_deform_attn_func import (
        MSDeformAttnFunction,
    )
    dev = torch.device("cuda")
    n, h, c, lq, p = 2, 4, 8, 6, 4
    shapes = torch.as_tensor([[8, 8], [4, 4]], dtype=torch.long, device=dev)
    lvl_start = torch.cat((shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1]))
    value = torch.rand(n, int(shapes.prod(1).sum()), h, c, device=dev)
    loc = torch.rand(n, lq, h, len(shapes), p, 2, device=dev)
    attn = torch.rand(n, lq, h, len(shapes), p, device=dev)
    out = MSDeformAttnFunction.apply(value, shapes, lvl_start, loc, attn, 64)   # NOT wrapped
    assert out.shape == (n, lq, h * c), out.shape
    print(f"[msda] fused CUDA MSDeformAttn OK on {torch.cuda.get_device_name(0)} "
          f"(sm_{'.'.join(map(str, torch.cuda.get_device_capability(0)))}) via {MSDA.__file__}",
          flush=True)


def register_control_datasets(cfg):
    """
    Register COCO explicitly from CONTROL.COCO_ROOT instead of relying on d2's builtin
    `coco_2017_*` + $DETECTRON2_DATASETS. That makes the overfit gate a config change
    (point COCO_ROOT at a 64-image root) rather than an environment trick, and it keeps the
    evaluator's GT json in lockstep with the images being scored.
    """
    root = Path(cfg.CONTROL.COCO_ROOT)
    specs = [(TRAIN_DATASET, cfg.CONTROL.TRAIN_JSON, cfg.CONTROL.TRAIN_IMAGES),
             (VAL_DATASET, cfg.CONTROL.VAL_JSON, cfg.CONTROL.VAL_IMAGES)]
    if cfg.CONTROL.VAL_SUBSET_JSON:
        specs.append((VAL_SUBSET_DATASET, cfg.CONTROL.VAL_SUBSET_JSON, cfg.CONTROL.VAL_IMAGES))
    for name, js, imgs in specs:
        if name in MetadataCatalog.list():
            continue
        path = root / js
        assert path.is_file(), (
            f"{name}: {path} missing. The periodic-eval subset is built once by "
            f"`python third_party/maskdino_control/make_val_subset.py --n 1000`.")
        register_coco_instances(name, {}, str(path), str(root / imgs))


class ControlTrainer(UpstreamTrainer):
    """Upstream's Trainer with our data pipeline, our LR schedule and a wall-clock self-stop."""

    @classmethod
    def build_train_loader(cls, cfg):
        assert cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_squash", \
            f"the control arm only runs the squash mapper, got {cfg.INPUT.DATASET_MAPPER_NAME}"
        return build_detection_train_loader(cfg, mapper=CocoSquashDatasetMapper(cfg, True))

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        # d2's default test mapper is ResizeShortestEdge(MIN_SIZE_TEST); the squash must be
        # applied at eval time too or the model sees a distribution it never trained on.
        return build_detection_test_loader(cfg, dataset_name,
                                           mapper=CocoSquashDatasetMapper(cfg, False))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            # Per dataset: the periodic subset and the final full val are DIFFERENT populations
            # and their dumped predictions must not overwrite each other.
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        # Same evaluator upstream uses for `coco`: segm AND bbox, topk from TEST.DETECTIONS_PER_IMAGE.
        return COCOEvaluator(dataset_name, output_dir=output_folder)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        if cfg.SOLVER.LR_SCHEDULER_NAME == "WarmupCosineMatched":
            return build_matched_lr_scheduler(cfg, optimizer)
        return super().build_lr_scheduler(cfg, optimizer)

    def build_writers(self):
        # d2's default writer set includes TensorboardXWriter, whose import chain hits
        # `distutils.version` inside torch 1.10's tensorboard shim -- an AttributeError under this
        # env's setuptools (job 10089104). Nothing here reads tensorboard: metrics.json is the log.
        from detectron2.utils.events import CommonMetricPrinter, JSONWriter
        return [CommonMetricPrinter(self.max_iter),
                JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json"))]

    def build_hooks(self):
        ret = super().build_hooks()
        # 88k iters / CHECKPOINT_PERIOD checkpoints at ~560 MB each fills the output dir fast,
        # and only the newest is ever used for resume.
        for i, h in enumerate(ret):
            if isinstance(h, hooks.PeriodicCheckpointer):
                ret[i] = hooks.PeriodicCheckpointer(
                    self.checkpointer, self.cfg.SOLVER.CHECKPOINT_PERIOD, max_to_keep=2)
        if self.cfg.CONTROL.TIME_BUDGET_HOURS > 0:
            ret.append(TimeBudgetHook(self.cfg.CONTROL.TIME_BUDGET_HOURS))
        return ret


def setup(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    add_control_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.DATASETS.TRAIN = (TRAIN_DATASET,)
    # DATASETS.TEST drives the PERIODIC eval hook only; the final eval overrides it with the full
    # val below. Matching our arms on both axes -- every 5000 steps, first 1000 images -- is what
    # makes the two curves comparable at a step mark. See docs/MASKDINO_COCO.md 6.
    cfg.DATASETS.TEST = ((VAL_SUBSET_DATASET,) if cfg.CONTROL.VAL_SUBSET_JSON
                         else (VAL_DATASET,))
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="maskdino")
    return cfg


def write_summary(cfg, results, iteration):
    """
    The completion marker the batch script tests for. Its ABSENCE is what triggers a resubmit,
    so it must be written only when the full step budget is done.
    """
    segm = results.get("segm", {})
    box = results.get("bbox", {})
    payload = {
        "run": "maskdino_upstream_matched",
        "max_iter": cfg.SOLVER.MAX_ITER,
        "final_iter": iteration,
        "dataset": VAL_DATASET,                       # `final` below is the FULL 5000-image val
        "periodic_eval_dataset": (VAL_SUBSET_DATASET if cfg.CONTROL.VAL_SUBSET_JSON
                                  else VAL_DATASET),  # metrics.json's segm/AP is THIS population
        "periodic_eval_period": cfg.TEST.EVAL_PERIOD,
        "coco_root": cfg.CONTROL.COCO_ROOT,
        "final": {
            "segm_AP": segm.get("AP"), "segm_AP50": segm.get("AP50"),
            "segm_AP75": segm.get("AP75"), "segm_APs": segm.get("APs"),
            "segm_APm": segm.get("APm"), "segm_APl": segm.get("APl"),
            "box_AP": box.get("AP"), "box_AP50": box.get("AP50"),
        },
        "raw": {k: dict(v) for k, v in results.items()},
    }
    out = Path(cfg.OUTPUT_DIR) / "summary.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"=== summary.json written to {out} ===\n{json.dumps(payload['final'], indent=2)}",
          flush=True)


def full_val_cfg(cfg):
    """
    `cfg` with DATASETS.TEST pointing at the FULL 5000-image val2017.

    The periodic evals deliberately score a 1000-image subset (our arms' `--eval_images`), but a
    reported number must be the full val or it is not comparable to anything published -- our own
    arms' finals included. Mixing the two populations in one table is exactly the trap
    docs/MASKDINO_COCO.md 6 documents: our arms appear to "drop" ~2 AP at their last step purely
    because the population changed under them.
    """
    out = cfg.clone()
    out.defrost()
    out.DATASETS.TEST = (VAL_DATASET,)
    out.freeze()
    return out


def main(args):
    cfg = setup(args)
    register_control_datasets(cfg)
    if cfg.CONTROL.REQUIRE_CUDA_MSDA and torch.cuda.is_available():
        assert_cuda_msda()

    if args.eval_only:
        model = ControlTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume)
        results = ControlTrainer.test(full_val_cfg(cfg), model)
        write_summary(cfg, results, cfg.SOLVER.MAX_ITER)
        return results

    trainer = ControlTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    try:
        trainer.train()
    except TimeBudgetReached:
        # Save under the periodic name so `last_checkpoint` points at it and the next segment
        # resumes from exactly here (optimizer + LR schedule + iteration all travel with it).
        trainer.checkpointer.save(f"model_{trainer.iter:07d}")
        print(f"=== TIME BUDGET reached at iter {trainer.iter}/{cfg.SOLVER.MAX_ITER}; "
              f"checkpointed, exiting WITHOUT summary.json so the job resubmits ===", flush=True)
        return None

    print("\n" + "=" * 70 + "\nFINAL EVAL (full val2017)\n" + "=" * 70, flush=True)
    results = ControlTrainer.test(full_val_cfg(cfg), trainer.model)
    if comm.is_main_process():
        write_summary(cfg, results, trainer.iter)
    return results


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()
    print("Command Line Args:", args)
    launch(main, args.num_gpus, num_machines=args.num_machines,
           machine_rank=args.machine_rank, dist_url=args.dist_url, args=(args,))
