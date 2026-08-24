#!/usr/bin/env python3
"""
CPU checks for `scripts/collect_eval3d_matrix.py` (docs/RESULTS.md §7): the default-cell filter
that keeps tuned runs out of the matrix, and the `--run`/`--only` entry points that let a run the
file does not name — the multi-dataset arms of docs/MULTIDATASET.md §10 — be collected.

    myenv/bin/python tests/test_collect_eval3d_matrix.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scripts.collect_eval3d_matrix as C  # noqa: E402

CHECKS = 0


def ok(msg):
    global CHECKS
    CHECKS += 1
    print(f"  ok  {msg}")


def cell(dataset, mode, triple, **knobs):
    args = {"num_frames": None, "eval_topk": 100, "min_score": 0.0, "mask_prob_threshold": 0.5,
            "depth_tolerance": 0.1, "vote_radius": 0.05, "depth_conf_percentile": 0.0,
            "icp": True, "icp_max_dist": 0.3}
    args.update(knobs)
    return {"dataset": dataset, "transfer_mode": mode, "num_scenes": 3, "failed_scenes": [],
            "args": args, "per_scene": {"a": {"frames": 8}},
            "results_class_agnostic": dict(zip(("all_ap", "all_ap_50%", "all_ap_25%"), triple)),
            "results_18class": None}


def test_default_cell_filter():
    assert C.is_default_cell(cell("scannetv2", "unproject", (0, 0, 0))["args"])
    assert not C.is_default_cell(cell("scannetv2", "unproject", (0, 0, 0),
                                      vote_radius=0.15)["args"])
    assert not C.is_default_cell(cell("scannetv2", "unproject", (0, 0, 0), icp=False)["args"])
    ok("a tuned knob keeps a cell out of the matrix; defaults let it in")


def test_parse_runs():
    base = dict(C.RUNS)
    got = C.parse_runs(["A=dir_a"], only=False)
    assert list(got)[: len(base)] == list(base) and got["A"] == "dir_a"
    assert C.parse_runs(["A=dir_a", "B=dir_b"], only=True) == {"A": "dir_a", "B": "dir_b"}
    assert list(C.parse_runs(["B=dir_b", "A=dir_a"], only=True)) == ["B", "A"], "row order kept"
    for bad in ("noequals", "=dir", "label="):
        try:
            C.parse_runs([bad], only=True)
        except SystemExit:
            continue
        raise AssertionError(f"{bad!r} should have been refused")
    try:
        C.parse_runs([], only=True)
    except SystemExit:
        ok("--run parsing: appends, --only replaces, order kept, bad input refused")
    else:
        raise AssertionError("--only with no --run should abort")


def test_collect_reads_only_the_named_runs():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "run_a").mkdir()
        (root / "run_b").mkdir()
        (root / "run_a" / "eval3d_x.json").write_text(
            json.dumps(cell("scannetpp", "unproject", (0.1, 0.2, 0.3))))
        (root / "run_a" / "eval3d_x_tuned.json").write_text(
            json.dumps(cell("scannetpp", "gt_projection", (0.9, 0.9, 0.9), vote_radius=0.15)))
        (root / "run_b" / "eval3d_y.json").write_text(
            json.dumps(cell("replica", "unproject", (0.4, 0.5, 0.6))))
        old = C.OUT
        C.OUT = root
        try:
            cells = C.collect({"A": "run_a"})
            assert set(cells) == {("A", "scannetpp", "unproject")}, cells
            assert cells[("A", "scannetpp", "unproject")]["class_agnostic"] == (0.1, 0.2, 0.3)
            both = C.collect({"A": "run_a", "B": "run_b"})
            assert ("B", "replica", "unproject") in both
        finally:
            C.OUT = old
    ok("collect() scans exactly the runs it is given and drops the tuned cell")


if __name__ == "__main__":
    print("=== collect_eval3d_matrix ===")
    test_default_cell_filter()
    test_parse_runs()
    test_collect_reads_only_the_named_runs()
    print(f"\n{CHECKS} checks passed")
