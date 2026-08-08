"""
Our arms' LR schedule, reimplemented for detectron2.

`scripts/train_maskdino_coco.py::build_step_scheduler` is:

    step < warmup            ->  (step + 1) / warmup                       (linear warmup)
    otherwise                ->  r + (1 - r) * 0.5 * (1 + cos(pi * p))     (cosine to r)
    with p = (step - warmup) / (total - warmup),  r = min_lr_ratio = 0.01

Upstream MaskDINO instead multisteps at 0.889 / 0.963 of the budget, and detectron2's built-in
`WarmupCosineLR` decays to exactly 0 with a different warmup parameterisation
(`WARMUP_FACTOR + (1 - WARMUP_FACTOR) * alpha`). Neither reproduces the curve our arms ran, so
the lambda is copied verbatim instead of approximated.
"""

import math

from torch.optim.lr_scheduler import LambdaLR

__all__ = ["build_matched_lr_scheduler", "matched_lr_lambda"]


def matched_lr_lambda(step: int, total_steps: int, warmup_steps: int,
                      min_lr_ratio: float = 0.01) -> float:
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def build_matched_lr_scheduler(cfg, optimizer):
    # LR_HORIZON_ITERS decouples the cosine's horizon from the stop point. The real run leaves it
    # at 0 (== MAX_ITER), which is what our arms ran. The overfit gate sets it to the real 87948
    # while stopping at 600, so the gate spends its steps near peak LR instead of riding a cosine
    # compressed into 600 steps -- measured: compressed cosine caps the gate at 28.0 AP and a
    # 1000-step warmup caps it at 0.8, neither of which says anything about the pipeline.
    total = cfg.CONTROL.LR_HORIZON_ITERS or cfg.SOLVER.MAX_ITER
    warmup = cfg.SOLVER.WARMUP_ITERS
    ratio = cfg.CONTROL.COSINE_END_LR_RATIO
    return LambdaLR(optimizer, lambda s: matched_lr_lambda(s, total, warmup, ratio))
