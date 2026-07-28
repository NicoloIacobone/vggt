"""Test legacy/d4rt/scripts/eval_checkpoint.py plumbing — CPU, no backbone, no downloads.

Covers the two pure helpers (the model/backbone path reuses components already
covered by test_phase4/test_eval/test_grid_ablation):
- build_eval_args: checkpoint args inherited, CLI overrides win, GT-source
  fields (instance_level) overridable independently of bundle-shape fields.
- resolve_eval_scenes: defaults to the checkpoint's stored val scenes; --scenes
  wins; errors when neither is available.

Run: python legacy/d4rt/tests/test_eval_checkpoint.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from eval_checkpoint import build_eval_args, resolve_eval_scenes  # noqa: E402

CK_ARGS = {"num_frames": 8, "num_queries": 32, "instance_level": True,
           "grid_size": 6, "mask_upsample": 2}


def cli(**over):
    base = dict(num_frames=None, num_queries=None, grid_size=None, instance_level=None)
    base.update(over)
    return SimpleNamespace(**base)


def main():
    # Inheritance from checkpoint args.
    ea = build_eval_args(CK_ARGS, cli())
    assert (ea.num_frames, ea.num_queries, ea.grid_size) == (8, 32, 6)
    assert ea.instance_level is True and ea.mask_upsample == 2
    assert ea.bundles_per_scene == 1 and ea.color_jitter == 0.0
    print("[1/4] checkpoint-arg inheritance OK")

    # CLI overrides win; instance_level=0 must beat a True checkpoint value.
    ea = build_eval_args(CK_ARGS, cli(num_frames=4, grid_size=10, instance_level=0))
    assert (ea.num_frames, ea.grid_size, ea.instance_level) == (4, 10, False)
    ea = build_eval_args({}, cli())  # legacy checkpoint without these keys
    assert (ea.num_frames, ea.num_queries, ea.grid_size, ea.mask_upsample) == (8, 32, 6, 1)
    assert ea.instance_level is False
    print("[2/4] CLI overrides + legacy defaults OK")

    # Scene resolution: stored val scenes by default, --scenes wins.
    ckpt = {"scenes": [{"name": "scene0000_00", "split": "train"},
                       {"name": "scene0080_00", "split": "val"},
                       {"name": "scene0081_00", "split": "val"}]}
    assert resolve_eval_scenes(ckpt, None) == ["scene0080_00", "scene0081_00"]
    assert resolve_eval_scenes(ckpt, "scene0004_00") == ["scene0004_00"]
    print("[3/4] scene resolution OK")

    try:
        resolve_eval_scenes({"scenes": []}, None)
        raise AssertionError("expected ValueError for no val scenes")
    except ValueError:
        pass
    print("[4/4] no-val-scenes error OK")

    print("\nAll eval_checkpoint tests passed.")


if __name__ == "__main__":
    main()
