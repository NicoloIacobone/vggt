"""
Training helpers shared by every training entry point (MaskDINO and the legacy D4RT arms).

These five used to live in `legacy/d4rt/scripts/train_multiscene.py`, which made the MaskDINO script
import from a D4RT script for purely generic utilities. Nothing here knows about either
model family — it is scene-path resolution, augmentation, the LR schedule, and metrics I/O.
"""

import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

# Default ScanNet root. Jobs that unpack the zstd dataset tar onto node-local scratch
# (slurm/stage_dataset.sh) export SCANNET_ROOT=$TMPDIR/scans so the loader reads off the fast
# local SSD; otherwise it falls back to the work filesystem.
DEFAULT_SCANS_ROOT = os.environ.get(
    "SCANNET_ROOT",
    "/cluster/work/igp_psr/niacobone/distillation/dataset/scannet/scans",
)


def resolve_scene_dirs(spec: str, scans_root: str) -> List[str]:
    """
    Accept comma-separated scene names (resolved under scans_root/<name>/raw_data) or paths.

    `@<file>` reads the list from a file instead, one entry per line (commas still work inside
    it). **This is not a convenience — past ~2000 absolute paths there is no other way.** Linux
    caps a SINGLE argv entry at `MAX_ARG_STRLEN` = 128 KB regardless of the total `ARG_MAX`, and
    the multi-dataset mixture's 3520 absolute paths are ~211 KB: job 10480614 died at
    `execve` with "Argument list too long" *after* staging 117 GB (docs/MULTIDATASET.md §7.2).
    A file also leaves the exact scene list in the run directory, which is the provenance a
    3520-scene mixture needs anyway.
    """
    if spec.startswith("@"):
        path = Path(spec[1:])
        if not path.is_file():
            raise ValueError(f"scene list file not found: {path}")
        spec = path.read_text()
    dirs = []
    for token in [t.strip() for t in spec.replace("\n", ",").split(",") if t.strip()]:
        p = Path(token)
        if not p.exists():
            p = Path(scans_root) / token / "raw_data"
        if not p.exists():
            raise ValueError(f"Scene not found: {token} (tried {p})")
        dirs.append(str(p))
    return dirs


def photometric_jitter(images: torch.Tensor, strength: float) -> torch.Tensor:
    """One random brightness/contrast draw applied to a whole bundle (masks are unaffected)."""
    if strength <= 0:
        return images
    contrast = 1.0 + (torch.rand(1, device=images.device).item() * 2 - 1) * strength
    brightness = (torch.rand(1, device=images.device).item() * 2 - 1) * strength
    return ((images - 0.5) * contrast + 0.5 + brightness).clamp(0.0, 1.0)


def build_scheduler(optimizer, num_epochs: int, warmup_epochs: int, min_lr_ratio: float = 0.05):
    """Linear warmup followed by cosine decay to min_lr_ratio * base_lr."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return LambdaLR(optimizer, lr_lambda)


def append_jsonl(path: Path, record: Dict) -> None:
    """Append one JSON object as a line to `path` (parent dirs created on demand).

    Used to persist one record per eval to <run_dir>/metrics.jsonl, so scaling plots come
    from a machine-readable file rather than scraping the training log.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
