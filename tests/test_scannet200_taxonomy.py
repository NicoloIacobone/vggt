#!/usr/bin/env python3
"""
The ScanNet200 taxonomy (`data/scannet200_constants.py`, docs/todo.md 6d). CPU-only.

ScanNet200 needs no new data — only a different label map over the tars we already have — so
the whole risk sits in that map being right. This checks the two things that would silently
produce a plausible-looking but wrong GT:

  1. the id list is the official one: 200 unique ids, in the official order, drawn from the
     TSV's `id` column (1..1191) and NOT from the nyu40 column;
  2. `raw_category -> id` really is a FUNCTION over the labels TSV (one id per raw
     category), so the 200-class GT is a partition of the annotated objects and no object
     can land in two classes.

Check 2's TSV half runs only when the real labels table is present (it lives on group
storage, not in the repo); the synthetic half always runs.

    myenv/bin/python tests/test_scannet200_taxonomy.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.scannet200_constants import VALID_CLASS_IDS_200, VALID_CLASS_IDS_200_SET
from train.scannet3d import DEFAULT_TSV, load_raw_to_nyu40, load_raw_to_scannet_id

PASSED = []


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"{name} FAILED {detail}")
    PASSED.append(name)
    print(f"  ok  {name}")


def test_id_list():
    check("200 class ids", len(VALID_CLASS_IDS_200) == 200, f"{len(VALID_CLASS_IDS_200)}")
    check("ids are unique", len(VALID_CLASS_IDS_200_SET) == 200)
    check("ids are sorted (the official order)",
          list(VALID_CLASS_IDS_200) == sorted(VALID_CLASS_IDS_200))
    check("ids are raw ScanNet ids, not nyu40 (max 1191, not 40)",
          max(VALID_CLASS_IDS_200) == 1191 and min(VALID_CLASS_IDS_200) == 1,
          f"{min(VALID_CLASS_IDS_200)}..{max(VALID_CLASS_IDS_200)}")


def test_map_is_a_function():
    with tempfile.TemporaryDirectory() as td:
        tsv = Path(td) / "labels.tsv"
        tsv.write_text("id\traw_category\tcategory\tnyu40id\n"
                       "5\tchair\tchair\t5\n"
                       "1163\tobject\tobject\t40\n"
                       "3\tfloor\tfloor\t2\n")
        raw_to_id = load_raw_to_scannet_id(tsv)
        raw_to_nyu = load_raw_to_nyu40(tsv)
        check("the two columns are read independently",
              raw_to_id["object"] == 1163 and raw_to_nyu["object"] == 40,
              f"{raw_to_id} {raw_to_nyu}")
        check("each raw category has exactly one id",
              len(raw_to_id) == 3 and len(set(raw_to_id)) == 3)


def test_against_the_real_tsv():
    tsv = Path(DEFAULT_TSV)
    if not tsv.exists():
        print(f"  --  skipped: {tsv} not mounted")
        return
    raw_to_id = load_raw_to_scannet_id(tsv)
    ids = set(raw_to_id.values())
    missing = sorted(VALID_CLASS_IDS_200_SET - ids)
    check("every ScanNet200 id is reachable from some raw category",
          not missing, f"unreachable: {missing[:10]}")
    lines = tsv.read_text().splitlines()
    header = lines[0].split("\t")
    i_raw = header.index("raw_category")
    raws = [line.split("\t")[i_raw] for line in lines[1:] if line.strip()]
    check("raw categories are unique in the TSV (the map is a partition)",
          len(raws) == len(set(raws)), f"{len(raws)} rows, {len(set(raws))} unique")
    covered = sum(1 for r in raws if raw_to_id[r] in VALID_CLASS_IDS_200_SET)
    check("the 200 classes cover a plausible share of the raw categories",
          0.2 < covered / len(raws) < 0.95, f"{covered}/{len(raws)}")


def main():
    print("id list")
    test_id_list()
    print("raw_category -> id")
    test_map_is_a_function()
    print("against the real labels TSV")
    test_against_the_real_tsv()
    print(f"\n✓ all {len(PASSED)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
