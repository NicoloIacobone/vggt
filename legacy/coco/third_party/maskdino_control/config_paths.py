"""
Keep the control configs' `_BASE_` pointing at the clone, wherever the clone currently lives.

`configs/maskdino_upstream_matched*.yaml` inherit from upstream's OWN COCO config inside the
pristine clone (`$MASKDINO_ROOT/configs/coco/instance-segmentation/...`). That is deliberate —
§6's whole claim is that only the listed axes differ from upstream — but it hard-codes an
absolute path into a yaml, and the clone moves: it lived on scratch until 2026-08-12 and moved to
$HOME because scratch's 15-day purge had eaten the reference venv mid-run twice (docs/todo.md 6i).

Every *python* entry point here already reads `MASKDINO_ROOT` from the environment. This module
makes the yaml follow it too, so the next move is one environment variable and no edits.

Stdlib only, and deliberately importable without the package `__init__` (which pulls detectron2),
so `tests/test_maskdino_control_paths.py` runs under the project venv like every other test.
"""

import os
import re
import tempfile
from pathlib import Path

DEFAULT_MASKDINO_ROOT = "/cluster/home/niacobone/MaskDINO"

_BASE_RE = re.compile(r"^_BASE_:[ \t]*(\S+)[ \t]*$", re.MULTILINE)


def maskdino_root() -> str:
    """The clone's location: `$MASKDINO_ROOT`, else the current default."""
    return os.environ.get("MASKDINO_ROOT", DEFAULT_MASKDINO_ROOT)


_MAX_BASE_DEPTH = 8


def resolve_base(config_file, root=None, _depth=0) -> str:
    """
    Return a config path safe to hand to `cfg.merge_from_file`.

    Unchanged — the same path back — unless the config's `_BASE_` chain reaches an absolute path
    that does not exist. In that case the part after `/configs/` is re-rooted at `root` and a
    patched copy is written to a temp file whose path is returned. The original yaml is never
    edited: it stays the record of what the run inherited from.

    **The chain is followed.** `maskdino_upstream_matched_overfit.yaml` inherits the matched
    config by a *relative* base, which in turn inherits upstream's by an absolute one — so the
    §4.1 overfit gate breaks on a moved clone one level down from where it looks. A rewritten
    parent points at the rewritten child, in the same temp directory so relative siblings still
    resolve.

    Raises FileNotFoundError if a base is missing and cannot be re-rooted — silently training
    against upstream's defaults instead of upstream's COCO config would produce a control row
    that looks fine and means nothing.
    """
    config_file = str(config_file)
    if _depth > _MAX_BASE_DEPTH:
        raise FileNotFoundError(f"{config_file}: _BASE_ chain deeper than {_MAX_BASE_DEPTH} — cycle?")

    text = Path(config_file).read_text()
    match = _BASE_RE.search(text)
    if match is None:
        return config_file
    base = match.group(1)
    root = maskdino_root() if root is None else str(root)

    if not os.path.isabs(base):
        # Relative bases are resolved by detectron2 against the config's own directory, and are
        # in-repo, so they never move. Recurse: only a rewrite further down forces one here.
        child = Path(config_file).parent / base
        if not child.exists():
            raise FileNotFoundError(f"{config_file}: relative _BASE_ {base} does not exist")
        resolved = resolve_base(child, root, _depth + 1)
        if os.path.samefile(resolved, child):
            return config_file
        return _write_patched(config_file, text, match, resolved, Path(resolved).parent)

    if os.path.exists(base):
        return config_file

    _, sep, tail = base.partition("/configs/")
    if not sep:
        raise FileNotFoundError(
            f"{config_file}: _BASE_ {base} does not exist and is not under a 'configs/' "
            f"directory, so it cannot be re-rooted at MASKDINO_ROOT={root}")

    rerooted = os.path.join(root, "configs", tail)
    if not os.path.exists(rerooted):
        raise FileNotFoundError(
            f"{config_file}: _BASE_ {base} does not exist, and neither does {rerooted}. "
            f"Point MASKDINO_ROOT at the upstream clone (docs/todo.md 6i).")
    return _write_patched(config_file, text, match, rerooted)


def _write_patched(config_file, text, match, new_base, out_dir=None) -> str:
    out_dir = Path(tempfile.mkdtemp(prefix="maskdino_cfg_")) if out_dir is None else Path(out_dir)
    patched = out_dir / Path(config_file).name
    patched.write_text(text[:match.start(1)] + str(new_base) + text[match.end(1):])
    return str(patched)
