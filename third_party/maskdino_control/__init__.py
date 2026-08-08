"""
Upstream-MaskDINO control arm for the COCO backbone-swap study (docs/MASKDINO_COCO.md §6).

Everything here lives in THIS repo and is pointed at the pristine upstream clone
(`/cluster/scratch/niacobone/MaskDINO`) from outside. The clone is never edited, so
`docs/MASKDINO.md` §7.6 (the weight-transplant equivalence check) stays reproducible.
"""

from .config import add_control_config          # noqa: F401
from .squash_mapper import CocoSquashDatasetMapper  # noqa: F401
from .lr import build_matched_lr_scheduler      # noqa: F401
