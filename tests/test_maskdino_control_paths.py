#!/usr/bin/env python3
"""
CPU tests for third_party/maskdino_control/config_paths.py (docs/todo.md 6i).

Runs under the PROJECT venv, unlike tests/test_maskdino_upstream_control.py — the module under
test is stdlib-only on purpose, so the thing that keeps the control arm runnable after the clone
moves does not itself need the reference env to be checked.

What it protects: the control config inherits from upstream's own yaml by absolute path. If that
path silently fails to resolve, detectron2 would either crash (fine) or, worse, a future edit
could make it fall back to defaults — a control row that looks fine and means nothing.
"""

import importlib.util
import os
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Loaded by path, NOT as third_party.maskdino_control.config_paths: the package __init__ imports
# detectron2, which only the reference env has. That the module is reachable this way is part of
# what is being protected — it must stay free of package-level imports.
_spec = importlib.util.spec_from_file_location(
    "maskdino_config_paths", REPO / "third_party/maskdino_control/config_paths.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
DEFAULT_MASKDINO_ROOT, maskdino_root, resolve_base = (
    _mod.DEFAULT_MASKDINO_ROOT, _mod.maskdino_root, _mod.resolve_base)

PASSED = 0


def check(cond, what):
    global PASSED
    assert cond, f"FAILED: {what}"
    PASSED += 1
    print(f"  ok  {what}")


def _clone(tmp, tail="coco/instance-segmentation/maskdino_R50_bs16_50ep_3s.yaml"):
    """A fake upstream clone: <tmp>/clone/configs/<tail>."""
    base = Path(tmp) / "clone" / "configs" / tail
    base.parent.mkdir(parents=True, exist_ok=True)
    base.write_text("MODEL:\n  META_ARCHITECTURE: MaskDINO\n")
    return base


def _cfg(tmp, base_path, extra="SOLVER:\n  MAX_ITER: 87948\n"):
    cfg = Path(tmp) / "matched.yaml"
    cfg.write_text(f"# a comment\n_BASE_: {base_path}\n\n{extra}")
    return cfg


def test_existing_base_is_left_alone():
    with tempfile.TemporaryDirectory() as tmp:
        base = _clone(tmp)
        cfg = _cfg(tmp, base)
        check(resolve_base(cfg, Path(tmp) / "clone") == str(cfg),
              "a _BASE_ that exists returns the original path, untouched")
        check(f"_BASE_: {base}" in cfg.read_text(),
              "the original yaml is never rewritten in place")


def test_moved_clone_is_rerooted():
    with tempfile.TemporaryDirectory() as tmp:
        base = _clone(tmp)
        cfg = _cfg(tmp, "/gone/MaskDINO/configs/coco/instance-segmentation/"
                        "maskdino_R50_bs16_50ep_3s.yaml")
        out = resolve_base(cfg, Path(tmp) / "clone")
        check(out != str(cfg), "a missing _BASE_ yields a patched copy, not the original")
        text = Path(out).read_text()
        check(f"_BASE_: {base}" in text, "the patched copy points at the clone's real location")
        check("MAX_ITER: 87948" in text and "# a comment" in text,
              "everything else in the config survives the patch verbatim")
        check(f"_BASE_: {base}" not in cfg.read_text(),
              "the on-disk config is still the record of what the run inherited")


def test_root_comes_from_the_environment():
    with tempfile.TemporaryDirectory() as tmp:
        base = _clone(tmp)
        cfg = _cfg(tmp, "/gone/MaskDINO/configs/coco/instance-segmentation/"
                        "maskdino_R50_bs16_50ep_3s.yaml")
        old = os.environ.get("MASKDINO_ROOT")
        os.environ["MASKDINO_ROOT"] = str(Path(tmp) / "clone")
        try:
            check(maskdino_root() == str(Path(tmp) / "clone"),
                  "MASKDINO_ROOT overrides the default root")
            check(f"_BASE_: {base}" in Path(resolve_base(cfg)).read_text(),
                  "resolve_base with no explicit root follows MASKDINO_ROOT")
        finally:
            os.environ.pop("MASKDINO_ROOT")
            if old is not None:
                os.environ["MASKDINO_ROOT"] = old
        check(maskdino_root() == DEFAULT_MASKDINO_ROOT,
              "with MASKDINO_ROOT unset the default is used")


def test_unresolvable_base_raises_rather_than_falling_back():
    with tempfile.TemporaryDirectory() as tmp:
        _clone(tmp)
        cfg = _cfg(tmp, "/gone/MaskDINO/configs/coco/nope.yaml")
        try:
            resolve_base(cfg, Path(tmp) / "clone")
        except FileNotFoundError as exc:
            check("nope.yaml" in str(exc) and "MASKDINO_ROOT" in str(exc),
                  "a base missing from the clone too raises, naming the file and the knob")
        else:
            raise AssertionError("FAILED: unresolvable _BASE_ did not raise")

        odd = Path(tmp) / "odd.yaml"
        odd.write_text("_BASE_: /gone/elsewhere/base.yaml\n")
        try:
            resolve_base(odd, Path(tmp) / "clone")
        except FileNotFoundError as exc:
            check("configs/" in str(exc),
                  "a base outside any configs/ dir raises instead of guessing")
        else:
            raise AssertionError("FAILED: un-rerootable _BASE_ did not raise")


def test_configs_without_a_base_pass_through():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "plain.yaml"
        cfg.write_text("SOLVER:\n  MAX_ITER: 10\n")
        check(resolve_base(cfg, Path(tmp)) == str(cfg),
              "a config with no _BASE_ is returned unchanged")


def _base_of(path):
    for line in Path(path).read_text().splitlines():
        if line.startswith("_BASE_:"):
            base = line.split(":", 1)[1].strip()
            return base if os.path.isabs(base) else str(Path(path).parent / base)
    return None


def test_a_relative_base_chain_is_followed():
    """The overfit gate inherits the matched config, which inherits upstream's — two levels."""
    with tempfile.TemporaryDirectory() as tmp:
        base = _clone(tmp)
        parent = _cfg(tmp, "/gone/MaskDINO/configs/coco/instance-segmentation/"
                           "maskdino_R50_bs16_50ep_3s.yaml")
        child = Path(tmp) / "overfit.yaml"
        child.write_text(f"_BASE_: {parent.name}\nSOLVER:\n  MAX_ITER: 600\n")

        out = Path(resolve_base(child, Path(tmp) / "clone"))
        check(out != child, "a rewrite two levels down forces the parent to be rewritten too")
        check("MAX_ITER: 600" in out.read_text(), "the overfit config's own keys survive")
        check(os.path.exists(_base_of(out)), "its _BASE_ points at a file that exists")
        check(_base_of(_base_of(out)) == str(base),
              "and that file's own _BASE_ is the re-rooted upstream config")
        check(Path(_base_of(out)).parent == out.parent,
              "parent and child land in the same temp dir, so relative siblings still resolve")


def test_an_intact_relative_chain_is_left_alone():
    with tempfile.TemporaryDirectory() as tmp:
        base = _clone(tmp)
        parent = _cfg(tmp, base)
        child = Path(tmp) / "overfit.yaml"
        child.write_text(f"_BASE_: {parent.name}\n")
        check(resolve_base(child, Path(tmp) / "clone") == str(child),
              "an intact relative chain returns the original path")


def test_the_real_control_configs_resolve():
    """Both shipped configs must resolve against the clone, if the clone is present."""
    root = maskdino_root()
    if not os.path.isdir(root):
        print(f"  ..  skipped: no clone at {root}")
        return
    for name in ("maskdino_upstream_matched.yaml", "maskdino_upstream_matched_overfit.yaml"):
        cfg = REPO / "third_party/maskdino_control/configs" / name
        if not cfg.exists():
            continue
        resolved = _base_of(resolve_base(cfg))
        while resolved is not None:
            check(os.path.exists(resolved), f"{name}: base {Path(resolved).name} exists")
            resolved = _base_of(resolved)


if __name__ == "__main__":
    for fn in [test_existing_base_is_left_alone, test_moved_clone_is_rerooted,
               test_root_comes_from_the_environment,
               test_unresolvable_base_raises_rather_than_falling_back,
               test_configs_without_a_base_pass_through,
               test_a_relative_base_chain_is_followed,
               test_an_intact_relative_chain_is_left_alone,
               test_the_real_control_configs_resolve]:
        fn()
    print(f"\n{PASSED} checks passed")
