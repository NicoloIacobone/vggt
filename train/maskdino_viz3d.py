#!/usr/bin/env python3
"""
Colour a 3D point cloud by a MaskDINO checkpoint's predicted instances — the plumbing behind
`demos/demo_gradio.py --seg_checkpoint <run_dir>/checkpoint_best_bundle.pth`.

The viewer needs exactly one thing from the head: a per-pixel instance colouring of the frames
VGGT just reconstructed, which it then pastes onto the unprojected point cloud. Both conventions
used to produce it are inherited rather than invented here, so the picture matches the numbers:

  * **Query selection is the 3D ruler's** (`scripts/eval_3d_maskdino.py`, docs/MASKDINO.md §9.1
    step 2): one class score per query = max over the views of the bundle, sigmoid, best
    non-background class; keep `score >= threshold`, then top-k by score. What you see is what
    the 3D benchmark lifts and scores — minus the unprojection, registration and superpoint vote.
  * **Colour is keyed to the query index**, exactly like the 2D figures
    (`train/maskdino_eval.paint_identity_map`): an instance keeps its colour across views, which
    is the property the multi-frame model exists to have. Same tab20 slots as those figures, so
    query 7 is the same colour in the PNG panels and in the 3D viewer.

Read-only presentation layer: nothing here trains, scores, or mutates a checkpoint.
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from data.scannet_overfit import IDX_TO_CLASS
from models.maskdino.head import build_head_from_config, to_scannet_class_logits
from train.eval3d_geometry import assign_pixels_to_queries
from train.maskdino_eval import NUM_VIZ_COLORS, color_index

# NYU40 ids 1 (wall) and 2 (floor) are not ScanNet-benchmark instance classes; the 3D ruler drops
# them (`nyu40 > 2`). Kept optional here: in a viewer a fully coloured room is usually what you
# want to look at, and the two classes are named rather than hard-coded ids on purpose.
STUFF_CLASS_NAMES = ("wall", "floor")


def is_maskdino_checkpoint(ckpt: Dict) -> bool:
    """True for a `scripts/train_maskdino.py` checkpoint, False for a legacy D4RT one."""
    return isinstance(ckpt, dict) and "head_config" in ckpt and "head_state_dict" in ckpt


def load_maskdino_seg_head(ckpt_path, device: str = "cpu"):
    """
    Rebuild the trained head from a checkpoint. Returns (head, train_args, ckpt).

    `train_args` is the run's own CLI namespace as a dict — `feature_mode`, `feature_layers`,
    `multi_frame`, `num_frames`, `val_scenes` all come from there, so the demo reproduces the
    exact token geometry the head was trained on instead of guessing.
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if not is_maskdino_checkpoint(ckpt):
        raise ValueError(f"{ckpt_path} is not a MaskDINO checkpoint (keys: {sorted(ckpt)[:8]}…)")
    head = build_head_from_config(ckpt["head_config"])
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval().to(device)
    return head, dict(ckpt.get("args", {})), ckpt


def parse_feature_layers(train_args: Dict) -> List[int]:
    """`--feature_layers` as stored: already a list, or the un-parsed '-1' / '4,11,23' string."""
    layers = train_args.get("feature_layers", [-1])
    if isinstance(layers, str):
        layers = [int(x) for x in layers.split(",") if x.strip()]
    return list(layers)


@torch.no_grad()
def head_features(aggregator: Callable, images: torch.Tensor, train_args: Dict
                  ) -> Tuple[torch.Tensor, int]:
    """
    Frozen-backbone tokens for the head, honouring the run's `--feature_mode`.

    `aggregator` is any callable with VGGT's signature: [1, S, 3, H, W] → (list of token
    tensors, patch_start_idx). `images` is [S, 3, H, W] or [1, S, 3, H, W].

    `bundle` runs one pass over all frames (tokens are multi-view aware); `single` runs S
    one-frame passes, which is what a single-frame checkpoint was trained on — feeding it bundle
    tokens silently changes its input distribution, so this is not a detail to skip.
    Returns (features [S, P, C], patch_start_idx).
    """
    if images.dim() == 4:
        images = images.unsqueeze(0)
    layers = parse_feature_layers(train_args)
    S = images.shape[1]
    if train_args.get("feature_mode", "single") == "bundle":
        agg_list, psi = aggregator(images)
        return torch.cat([agg_list[i].float() for i in layers], dim=-1)[0], int(psi)
    per_frame, psi = [], 5
    for f in range(S):
        agg_list, psi = aggregator(images[:, f:f + 1])
        per_frame.append(torch.cat([agg_list[i].float() for i in layers], dim=-1)[0, 0])
    return torch.stack(per_frame), int(psi)


@torch.no_grad()
def head_token_xyz(point_head: Callable, agg_list, images: torch.Tensor,
                   patch_start_idx: int) -> torch.Tensor:
    """
    `--anchor_3d` checkpoints only (docs/MASKDINO.md §8.3): rebuild the per-patch 3D positions
    the training cache fed the decoder, from the aggregator output the head already consumed.

    `agg_list` / `patch_start_idx` come from the SAME bundle pass as the tokens — an anchor_3d
    checkpoint is only defined for `--feature_mode bundle`, so there is one coordinate frame for
    all views. Returns [S, h*w, 3], normalised per bundle.
    """
    from train.maskdino_data import patch_token_positions  # local: avoids a viz→data import cycle
    from models.maskdino.anchor3d import normalize_token_xyz

    if images.dim() == 4:
        images = images.unsqueeze(0)
    agg32 = [a.float() if a is not None else None for a in agg_list]
    pts, conf = point_head(agg32, images=images, patch_start_idx=patch_start_idx)
    xyz, w = patch_token_positions(pts, conf)
    return normalize_token_xyz(xyz, w)


def select_instances(pred_logits: torch.Tensor, score_threshold: float = 0.25,
                     topk: Optional[int] = 100, drop_stuff: bool = False
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Bundle-protocol query selection, the 3D ruler's rule (docs/MASKDINO.md §9.1).

    pred_logits [S, Q, 19] (or [Q, 19] for a single view) → (keep_idx [K], labels [K] as ScanNet
    class indices 1..19, scores [K]), sorted by descending score. One score per query, taken as
    the max over views: a query is one instance of the *scene*, so a view where it is invisible
    must not veto it.
    """
    if pred_logits.dim() == 3:
        pred_logits = pred_logits.max(dim=0).values
    probs = to_scannet_class_logits(pred_logits).sigmoid()      # [Q, 20], col 0 = -inf → 0
    scores, cls_idx = probs[:, 1:].max(dim=1)
    cls_idx = cls_idx + 1                                       # back to 1..19
    keep = scores >= score_threshold
    if drop_stuff:
        stuff = torch.as_tensor([IDX_TO_CLASS.get(int(c), "") in STUFF_CLASS_NAMES
                                 for c in cls_idx], device=keep.device)
        keep &= ~stuff
    keep_idx = torch.nonzero(keep).squeeze(1)
    keep_idx = keep_idx[scores[keep_idx].argsort(descending=True)]
    if topk is not None:
        keep_idx = keep_idx[:topk]
    return keep_idx, cls_idx[keep_idx], scores[keep_idx]


def identity_palette() -> np.ndarray:
    """
    [NUM_VIZ_COLORS + 1, 3] uint8 tab20 palette; slot 0 is the figures' background colour.

    Indexed by `train.maskdino_eval.color_index(query_index)`, which is what makes a query's
    colour identical here and in the run's PNG panels.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ((0.08, 0.08, 0.08),) + tuple(plt.get_cmap("tab20").colors)
    return (np.asarray(colors[:NUM_VIZ_COLORS + 1], dtype=np.float64) * 255).round().astype(
        np.uint8)


def assign_map(pred_masks: torch.Tensor, keep_idx: Sequence[int], out_hw: Tuple[int, int],
               mask_threshold: float = 0.5) -> np.ndarray:
    """
    [S, Q, h, w] mask logits → [S, H, W] map of *query indices* (-1 = no instance).

    Delegates the per-view rule to the 3D ruler's `assign_pixels_to_queries` (bilinear upsample
    of the logits, then argmax over the kept queries above the probability threshold), then maps
    the subset positions back to real query indices so the colouring is identity-keyed.
    """
    keep_idx = torch.as_tensor(list(keep_idx), dtype=torch.long)
    S = pred_masks.shape[0]
    if keep_idx.numel() == 0:
        return np.full((S,) + tuple(out_hw), -1, dtype=np.int64)
    lookup = keep_idx.cpu().numpy()
    out = []
    for f in range(S):
        sub = assign_pixels_to_queries(pred_masks[f, keep_idx], out_hw, mask_threshold)
        out.append(np.where(sub >= 0, lookup[np.clip(sub, 0, None)], -1))
    return np.stack(out)


def colorize(images: torch.Tensor, assign: np.ndarray,
             palette: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Paint an assignment map over the RGB frames. Returns uint8 [S, H, W, 3].

    `images` [S, 3, H, W] in [0, 1]; unassigned pixels keep their RGB, which is what makes the
    3D cloud readable (the room stays visible, the instances light up).
    """
    palette = identity_palette() if palette is None else palette
    rgb = (images.detach().cpu().permute(0, 2, 3, 1).clamp(0, 1).numpy() * 255).round().astype(
        np.uint8)   # round, not truncate — same convention as train/maskdino_data.prepare_scenes
    out = rgb.copy()
    for q in np.unique(assign):
        if q < 0:
            continue
        out[assign == q] = palette[color_index(int(q))]
    return out


def maskdino_seg_colors(head_out: Dict, images: torch.Tensor, score_threshold: float = 0.25,
                        mask_threshold: float = 0.5, topk: Optional[int] = 100,
                        drop_stuff: bool = False, palette: Optional[np.ndarray] = None
                        ) -> Tuple[np.ndarray, str]:
    """
    Head output + frames → (uint8 [S, H, W, 3] instance colours, human-readable legend).

    `head_out` is what `MaskDINOVGGTHead.forward` returns: `pred_logits` [S, Q, 19] and
    `pred_masks` [S, Q, h, w] (S = frames in the bundle).
    """
    if images.dim() == 5:
        images = images[0]
    keep_idx, labels, scores = select_instances(head_out["pred_logits"], score_threshold,
                                                topk, drop_stuff)
    assign = assign_map(head_out["pred_masks"], keep_idx, images.shape[-2:], mask_threshold)
    colors = colorize(images, assign, palette)
    if keep_idx.numel() == 0:
        return colors, f"No instances above score {score_threshold}."
    visible = {int(q) for q in np.unique(assign) if q >= 0}
    parts = [f"q{int(q)}:{IDX_TO_CLASS.get(int(c), int(c))} ({float(s):.2f})"
             + ("" if int(q) in visible else " [no pixels]")
             for q, c, s in zip(keep_idx.tolist(), labels.tolist(), scores.tolist())]
    return colors, (f"{len(visible)} instance(s) drawn (of {keep_idx.numel()} kept @ score≥"
                    f"{score_threshold}), colour = query id: " + ", ".join(parts[:12])
                    + (" …" if len(parts) > 12 else ""))
