#!/usr/bin/env python3
"""
Build the 64-image COCO root the overfit gate trains and scores on (docs/MASKDINO_COCO.md §4.1).

    <out>/train2017 -> <coco>/val2017          (symlink: train and val are the SAME images)
    <out>/val2017   -> <coco>/val2017
    <out>/annotations/instances_{train,val}2017.json   the first N val2017 images, subset

Subsetting the GT json rather than just limiting the loader matters: COCOeval scores every image
in the GT file, so a 5000-image GT with 64 images' worth of predictions would report ~0 AP and
the gate would fail for the wrong reason.

Costs 4 inodes on scratch, which is quota'd on file count -- do not extract a second copy of COCO.

    python third_party/maskdino_control/make_overfit_root.py --n 64
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coco_root", default="/cluster/scratch/niacobone/coco")
    p.add_argument("--out", default="/cluster/scratch/niacobone/coco_overfit64")
    p.add_argument("--n", type=int, default=64)
    args = p.parse_args()

    src, out = Path(args.coco_root), Path(args.out)
    (out / "annotations").mkdir(parents=True, exist_ok=True)
    for link in ("train2017", "val2017"):
        tgt = out / link
        if tgt.is_symlink() or tgt.exists():
            tgt.unlink()
        tgt.symlink_to(src / "val2017")

    data = json.loads((src / "annotations/instances_val2017.json").read_text())
    keep_imgs = sorted(data["images"], key=lambda im: im["id"])[: args.n]
    keep_ids = {im["id"] for im in keep_imgs}
    subset = {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "categories": data["categories"],          # all 80: NUM_CLASSES must not change
        "images": keep_imgs,
        "annotations": [a for a in data["annotations"] if a["image_id"] in keep_ids],
    }
    blob = json.dumps(subset)
    for name in ("instances_train2017.json", "instances_val2017.json"):
        (out / "annotations" / name).write_text(blob)

    n_inst = sum(1 for a in subset["annotations"] if not a.get("iscrowd", 0))
    print(f"✓ {out}: {len(keep_imgs)} images, {n_inst} non-crowd instances "
          f"({len(subset['annotations'])} total anns), images symlinked to {src / 'val2017'}")


if __name__ == "__main__":
    main()
