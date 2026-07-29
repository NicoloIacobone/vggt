#!/usr/bin/env python3
"""
CPU tests for the COCO backbone-swap track (docs/MASKDINO_COCO.md).

Nothing here needs a GPU, backbone weights or the COCO images; the two tests that need COCO
annotations build a tiny synthetic `instances.json` on the fly. Run:

    myenv/bin/python tests/test_coco_maskdino.py
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.maskdino import HungarianMatcher, SetCriterion, build_weight_dict
from models.maskdino.head_coco import MaskDINOCocoHead, build_coco_head_from_config
from models.maskdino.pixel_decoder_coco import CocoPixelDecoder
from train.coco_data import (coco_category_mapping, masks_to_boxes_normalized,
                             xywh_to_cxcywh_normalized)
from train.coco_eval import instance_inference, predictions_for_image

PASS, FAIL = "✓", "✗"
_failures = []


def check(name, cond, extra=""):
    print(f"  {PASS if cond else FAIL} {name}{(' — ' + extra) if extra else ''}")
    if not cond:
        _failures.append(name)


def _tiny_head(**kw):
    cfg = dict(in_channels=(64,), highres_channels=None, hidden_dim=32, mask_dim=32,
               num_classes=6, num_queries=12, num_feature_levels=3, enc_layers=1, dec_layers=2,
               nheads=4, enc_dim_feedforward=64, dec_dim_feedforward=64, enc_n_points=2,
               dec_n_points=2, dn_num=12)
    cfg.update(kw)
    return MaskDINOCocoHead(**cfg)


def _targets(B, n, hw, num_classes=6):
    out = []
    for _ in range(B):
        masks = torch.zeros(n, *hw)
        for i in range(n):
            masks[i, 2 * i:2 * i + 4, 2 * i:2 * i + 4] = 1.0
        out.append({"labels": torch.randint(0, num_classes, (n,)), "masks": masks,
                    "boxes": masks_to_boxes_normalized(masks)})
    return out


# ---------------------------------------------------------------------------------------------
# 1. Pixel decoder — both pyramid modes
# ---------------------------------------------------------------------------------------------

def test_pixel_decoder():
    print("\n[1] CocoPixelDecoder")

    # ViTDet mode: one token map in, 3 levels out, mask_features upsampled by deconv
    pd = CocoPixelDecoder(in_channels=(64,), conv_dim=32, mask_dim=32, num_feature_levels=3,
                          enc_layers=1, nheads=4, dim_feedforward=64, enc_n_points=2,
                          mask_upsample=4)
    x = torch.randn(2, 64, 12, 12)
    mf, levels = pd(([x]))
    check("vitdet: 3 encoder levels", len(levels) == 3)
    check("vitdet: level shapes 12/6/3",
          [tuple(l.shape[-2:]) for l in levels] == [(12, 12), (6, 6), (3, 3)],
          str([tuple(l.shape[-2:]) for l in levels]))
    check("vitdet: mask_features 48x48 (12 x mask_upsample 4)", tuple(mf.shape) == (2, 32, 48, 48),
          str(tuple(mf.shape)))
    check("vitdet: levels ordered HIGH->LOW",
          levels[0].shape[-1] > levels[1].shape[-1] > levels[2].shape[-1])

    # FPN mode: a ResNet pyramid + the stride-4 lateral
    pd2 = CocoPixelDecoder(in_channels=(16, 32, 64), highres_channels=8, conv_dim=32, mask_dim=32,
                           num_feature_levels=3, enc_layers=1, nheads=4, dim_feedforward=64,
                           enc_n_points=2, mask_upsample=4)
    lv = [torch.randn(2, 16, 16, 16), torch.randn(2, 32, 8, 8), torch.randn(2, 64, 4, 4)]
    hr = torch.randn(2, 8, 32, 32)
    mf2, levels2 = pd2(lv, hr)
    check("fpn: 3 encoder levels at the backbone's own strides",
          [tuple(l.shape[-2:]) for l in levels2] == [(16, 16), (8, 8), (4, 4)])
    check("fpn: mask_features at the highres stride (32x32)",
          tuple(mf2.shape) == (2, 32, 32, 32), str(tuple(mf2.shape)))
    check("fpn: mask_upsample is ignored when a highres lateral exists", pd2.mask_upsample == 1)

    # guards
    try:
        CocoPixelDecoder(in_channels=(16, 32, 64), num_feature_levels=2)
        check("guard: num_feature_levels < backbone levels rejected", False)
    except ValueError:
        check("guard: num_feature_levels < backbone levels rejected", True)
    try:
        pd2(lv, None)
        check("guard: FPN mode without a highres map rejected", False)
    except ValueError:
        check("guard: FPN mode without a highres map rejected", True)
    try:
        pd([x, x])
        check("guard: wrong number of backbone levels rejected", False)
    except ValueError:
        check("guard: wrong number of backbone levels rejected", True)


# ---------------------------------------------------------------------------------------------
# 2. Head — forward shapes for both modes, head_config round-trip
# ---------------------------------------------------------------------------------------------

def test_head():
    print("\n[2] MaskDINOCocoHead")
    torch.manual_seed(0)

    head = _tiny_head(mask_upsample=2)
    x = torch.randn(2, 64, 12, 12)
    tg = _targets(2, 3, (24, 24))
    out, mask_dict = head([x], None, tg)
    check("vitdet: pred_logits [B,Q,C]", tuple(out["pred_logits"].shape) == (2, 12, 6),
          str(tuple(out["pred_logits"].shape)))
    check("vitdet: pred_masks at 24x24", tuple(out["pred_masks"].shape[-2:]) == (24, 24),
          str(tuple(out["pred_masks"].shape)))
    check("vitdet: pred_boxes [B,Q,4]", tuple(out["pred_boxes"].shape) == (2, 12, 4))
    check("vitdet: deep supervision present", len(out["aux_outputs"]) >= 2)
    check("vitdet: denoising produced a mask_dict", mask_dict is not None)

    head_r = _tiny_head(in_channels=(16, 32, 64), highres_channels=8)
    lv = [torch.randn(2, 16, 16, 16), torch.randn(2, 32, 8, 8), torch.randn(2, 64, 4, 4)]
    out_r, _ = head_r(lv, torch.randn(2, 8, 32, 32), _targets(2, 3, (32, 32)))
    check("fpn: pred_masks at the highres stride (32x32)",
          tuple(out_r["pred_masks"].shape[-2:]) == (32, 32), str(tuple(out_r["pred_masks"].shape)))

    # the round-trip contract from CLAUDE.md
    import inspect
    sig = set(inspect.signature(MaskDINOCocoHead.__init__).parameters) - {"self"}
    check("head_config covers every constructor argument", set(head.head_config) == sig,
          f"missing {sorted(sig - set(head.head_config))}, "
          f"extra {sorted(set(head.head_config) - sig)}")
    rebuilt = build_coco_head_from_config(head.head_config)
    rebuilt.load_state_dict(head.state_dict())
    check("head rebuilt from head_config accepts the state dict", True)
    check("rebuilt head reports the same class count", rebuilt.num_classes == head.num_classes)


# ---------------------------------------------------------------------------------------------
# 3. GT helpers
# ---------------------------------------------------------------------------------------------

def test_gt_helpers():
    print("\n[3] GT helpers")

    m = torch.zeros(2, 10, 20)
    m[0, 2:6, 4:8] = 1          # y 2..6, x 4..8
    boxes = masks_to_boxes_normalized(m)
    check("masks_to_boxes: cx", abs(float(boxes[0, 0]) - 6 / 20) < 1e-6, f"{float(boxes[0,0]):.4f}")
    check("masks_to_boxes: cy", abs(float(boxes[0, 1]) - 4 / 10) < 1e-6, f"{float(boxes[0,1]):.4f}")
    check("masks_to_boxes: w", abs(float(boxes[0, 2]) - 4 / 20) < 1e-6)
    check("masks_to_boxes: h", abs(float(boxes[0, 3]) - 4 / 10) < 1e-6)
    check("masks_to_boxes: empty mask -> zero box", float(boxes[1].abs().sum()) == 0.0)

    b = xywh_to_cxcywh_normalized([10, 20, 40, 60], width=100, height=200)
    check("xywh->cxcywh normalised", np.allclose(b, [0.3, 0.25, 0.4, 0.3]), str(b))

    # the squash is a per-axis linear map, so a normalised box is invariant to it
    b2 = xywh_to_cxcywh_normalized([20, 20, 80, 60], width=200, height=200)
    b1 = xywh_to_cxcywh_normalized([10, 20, 40, 60], width=100, height=200)
    check("normalised boxes are squash-invariant", np.allclose(b1, b2), f"{b1} vs {b2}")

    # horizontal flip: mask and box must move together
    masks = torch.zeros(1, 8, 8)
    masks[0, 2:4, 1:3] = 1
    box = masks_to_boxes_normalized(masks)
    fm = torch.flip(masks, dims=[2])
    fb = box.clone()
    fb[:, 0] = 1.0 - fb[:, 0]
    check("hflip: flipped mask and flipped box agree",
          torch.allclose(masks_to_boxes_normalized(fm), fb, atol=1e-6),
          f"{masks_to_boxes_normalized(fm).tolist()} vs {fb.tolist()}")


# ---------------------------------------------------------------------------------------------
# 4. Category mapping over a synthetic annotation file
# ---------------------------------------------------------------------------------------------

def test_category_mapping():
    print("\n[4] COCO category mapping")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "instances.json"
        # non-contiguous ids with gaps, exactly like real COCO (1..90 for 80 classes)
        p.write_text(json.dumps({"images": [], "annotations": [],
                                 "categories": [{"id": i} for i in (1, 2, 3, 5, 90)]}))
        c2i, i2c = coco_category_mapping(str(p))
        check("contiguous 0..N-1", sorted(c2i.values()) == [0, 1, 2, 3, 4])
        check("mapping is sorted by dataset id", c2i[1] == 0 and c2i[90] == 4)
        check("inverse recovers the dataset id",
              all(i2c[c2i[k]] == k for k in c2i))


# ---------------------------------------------------------------------------------------------
# 5. Inference — upstream's instance_inference, and a planted-perfect COCO round trip
# ---------------------------------------------------------------------------------------------

def test_inference():
    print("\n[5] instance_inference / predictions_for_image")
    Q, C, h, w = 5, 4, 8, 8
    logits = torch.full((Q, C), -10.0)
    logits[2, 3] = 10.0                       # query 2 is confidently class 3
    logits[0, 1] = 5.0
    masks = torch.full((Q, h, w), -10.0)
    masks[2, 1:5, 1:5] = 10.0
    masks[0, 6:8, 6:8] = 10.0
    boxes = torch.rand(Q, 4) * 0.1 + 0.4

    scores, labels, mask_out, box_out = instance_inference(logits, masks, boxes, topk=3)
    order = torch.argsort(scores, descending=True)
    check("top detection is (query 2, class 3)", int(labels[order[0]]) == 3)
    check("its mask is query 2's mask",
          torch.equal(mask_out[order[0]], masks[2]))
    check("its box is query 2's box", torch.allclose(box_out[order[0]], boxes[2]))
    check("score folds in mask quality (<= class score)",
          float(scores[order[0]]) <= float(logits[2, 3].sigmoid()) + 1e-6)
    check("exactly topk detections", scores.numel() == 3)

    try:
        from pycocotools import mask as mask_util  # noqa: F401
    except ImportError:
        print("  … pycocotools missing, skipping the RLE round trip")
        return
    dets = predictions_for_image(logits, masks, boxes, (16, 32), image_id=7,
                                 contig2cat=[10, 20, 30, 40], topk=3)
    check("prediction dicts carry the DATASET category id",
          any(d["category_id"] == 40 for d in dets),
          str(sorted({d["category_id"] for d in dets})))
    from pycocotools import mask as mu
    top = max(dets, key=lambda d: d["score"])
    dec = mu.decode(top["segmentation"])
    check("mask decoded at the ORIGINAL image size", dec.shape == (16, 32), str(dec.shape))
    # query 2 covers cols 1..5 of 8 -> 4..20 of 32; rows 1..5 of 8 -> 2..10 of 16
    ys, xs = np.nonzero(dec)
    check("upsampled mask lands where the prediction grid says",
          2 <= ys.min() <= 4 and 8 <= ys.max() <= 11 and 3 <= xs.min() <= 6
          and 17 <= xs.max() <= 21,
          f"y {ys.min()}..{ys.max()} x {xs.min()}..{xs.max()}")
    check("bbox present and in xywh", len(top["bbox"]) == 4 and top["bbox"][2] > 0)


# ---------------------------------------------------------------------------------------------
# 6. GT quantisation matches the oracle's rule
# ---------------------------------------------------------------------------------------------

def test_gt_quantisation():
    print("\n[6] GT rasterisation rule")
    raw = np.zeros((1, 40, 40), dtype=np.float32)
    raw[0, 10:30, 10:30] = 1.0
    small = F.interpolate(torch.from_numpy(raw)[None], size=(20, 20), mode="area")[0]
    q = (small > 0.5).to(torch.uint8)
    check("area-downsample + 0.5 threshold preserves a clean box",
          int(q.sum()) == 100, f"{int(q.sum())} cells")
    # a sub-cell object survives if it covers more than half a cell, and vanishes otherwise
    tiny = np.zeros((1, 40, 40), dtype=np.float32)
    tiny[0, 0, 0] = 1.0                          # 1/4 of a 2x2 cell
    tq = (F.interpolate(torch.from_numpy(tiny)[None], size=(20, 20), mode="area")[0] > 0.5)
    check("an object covering <half a cell vanishes at that grid (the measured ceiling)",
          int(tq.sum()) == 0)


# ---------------------------------------------------------------------------------------------
# 7. Overfit — the whole COCO-shaped path drives the loss down
# ---------------------------------------------------------------------------------------------

def test_overfit():
    print("\n[7] synthetic overfit (COCO-shaped: 80-class-style sigmoid head, point-sampled loss)")
    torch.manual_seed(0)
    head = _tiny_head(mask_upsample=2)
    tg = _targets(2, 3, (24, 24))
    x = torch.randn(2, 64, 12, 12)

    matcher = HungarianMatcher(cost_class=4.0, cost_mask=5.0, cost_dice=5.0, cost_box=5.0,
                               cost_giou=2.0, num_points=256)
    wd = build_weight_dict(dec_layers=2, two_stage=True, dn="seg")
    crit = SetCriterion(6, matcher, wd, losses=["labels", "masks", "boxes"], num_points=256,
                        dn="seg", dn_losses=["labels", "masks", "boxes"])
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)

    first = last = None
    for i in range(60):
        opt.zero_grad()
        out, md = head([x], None, tg)
        losses = crit(out, tg, md)
        total = sum(losses[k] * wd[k] for k in losses if k in wd)
        total.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 0.1)
        opt.step()
        if i == 0:
            first = float(total)
        last = float(total)
    check("loss decreased", last < first * 0.75, f"{first:.2f} -> {last:.2f}")
    check("loss is finite", np.isfinite(last))


def main():
    print("=" * 78)
    print("COCO backbone-swap track — CPU tests")
    print("=" * 78)
    test_pixel_decoder()
    test_head()
    test_gt_helpers()
    test_category_mapping()
    test_inference()
    test_gt_quantisation()
    test_overfit()
    print("\n" + "=" * 78)
    if _failures:
        print(f"{FAIL} {len(_failures)} FAILED: {_failures}")
        return 1
    print(f"{PASS} all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
