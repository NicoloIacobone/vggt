"""
Write the val2017 subset our three COCO arms use for their PERIODIC evals.

Why this file exists. `scripts/train_maskdino_coco.py` evaluates on `--eval_images 1000` while
training and on all 5000 only at the end (`train/coco_eval.py::evaluate_coco`, which restricts
`COCOeval.params.imgIds` to the images it actually saw). The control arm evaluates through
detectron2's `COCOEvaluator`, which scores every image in the registered GT json -- so the only
way to give it the same population is to hand it a GT json holding exactly those images. Without
this, the control's curve sits ~2 AP below our arms' curve for a reason that is pure protocol:
our arms' own final numbers drop ~2 AP against their best interval numbers for exactly the same
reason (docs/MASKDINO_COCO.md 6).

The subset rule is copied from `train/coco_data.py`, not re-invented:

  * ids are `sorted(coco.getImgIds())` -- and detectron2's `load_coco_json` sorts the same way,
    so "first N" means the same set on both sides;
  * the val split is NOT filtered for empty annotations (that branch is `if train:`), so the
    handful of val images with no usable annotation stay in and are counted;
  * the loader is `shuffle=False`, so "first N seen" is "first N sorted".

`categories` is copied VERBATIM. COCOEvaluator builds its contiguous-id mapping from the
registered metadata, so dropping the categories no image in the subset happens to use would
renumber the classes and silently mislabel every prediction.

    python third_party/maskdino_control/make_val_subset.py --n 1000
"""

import argparse
import json
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coco_root", default="/cluster/scratch/niacobone/coco")
    p.add_argument("--n", type=int, default=1000, help="images to keep (our arms' --eval_images)")
    p.add_argument("--out", default=None, help="default: annotations/instances_val2017_first<N>.json")
    args = p.parse_args()

    root = Path(args.coco_root)
    src = root / "annotations" / "instances_val2017.json"
    out = Path(args.out) if args.out else root / "annotations" / f"instances_val2017_first{args.n}.json"

    with open(src) as f:
        full = json.load(f)

    keep = sorted(img["id"] for img in full["images"])[: args.n]
    keep_set = set(keep)
    images = [im for im in full["images"] if im["id"] in keep_set]
    anns = [a for a in full["annotations"] if a["image_id"] in keep_set]

    subset = {
        "info": full.get("info", {}),
        "licenses": full.get("licenses", []),
        "images": images,
        "annotations": anns,
        "categories": full["categories"],          # verbatim -- see the module docstring
    }
    with open(out, "w") as f:
        json.dump(subset, f)

    n_empty = len(keep_set) - len({a["image_id"] for a in anns})
    print(f"{out}\n  images     {len(images)} (ids {keep[0]}..{keep[-1]}, {n_empty} with no annotation)"
          f"\n  annotations {len(anns)}\n  categories  {len(subset['categories'])}")
    assert len(images) == min(args.n, len(full["images"])), "subset lost images"
    assert len(subset["categories"]) == len(full["categories"]), "categories must stay verbatim"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
