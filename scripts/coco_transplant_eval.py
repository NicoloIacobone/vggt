"""
COCO val2017 weight-transplant validation of the ported MaskDINO stack.

Purpose
-------
`docs/MASKDINO.md` §2 claims the decoder logic in `models/maskdino/` is a *faithful port* of
IDEA-Research MaskDINO, with only documented deviations. This script tests that claim the only
way that is not self-referential: it drives **our** ported modules with **upstream's own trained
COCO weights** and checks whether they reproduce upstream's published COCO val2017 numbers for
that checkpoint: **46.1 mask AP / 51.5 box AP**, which is the MaskDINO README model-zoo row
"MaskDINO (hid 1024)" for `maskdino_r50_50ep_300q_hid1024_3sd1_instance_maskenhanced`.

That target is the *model zoo* figure, not a paper table value. Paper Table 3's 50-epoch /
300-query ResNet-50 rows read 46.0 / 50.5 (plain) and 46.3 / 51.7 (‡, mask-enhanced box init);
the ‡ row belongs to the wider `hid2048` release (52 M params), not to this checkpoint.
Either way the verdict rests on `--mode ours` vs `--mode baseline`, which holds everything
except the modules under test fixed.

If the port is correct, `--mode ours` and `--mode baseline` must agree to within numerical noise.
Any real gap is a bug in the port, and the per-component structure below localises it.

What this validates (the ported code paths actually exercised here)
-------------------------------------------------------------------
  models/maskdino/ms_deform_attn.py   MSDeformAttn (pure-PyTorch core) — used by encoder AND decoder
  models/maskdino/pixel_decoder.py    MSDeformAttnEncoderLayer / MSDeformAttnEncoder / reference points
  models/maskdino/decoder.py          MaskDINODecoder: two-stage selection, DAB anchors, iterative
                                      box refinement, mask-enhanced box init, prediction heads
  models/maskdino/decoder_layers.py   the DAB/DINO decoder stack
  models/maskdino/utils.py            MLP, inverse_sigmoid, gen_sineembed_for_position,
                                      PositionEmbeddingSine, gen_encoder_output_proposals
  models/maskdino/box_ops.py          masks_to_boxes (initialize_box_type="bitmask")

What it does NOT validate (out of scope by construction)
--------------------------------------------------------
  - The VGGT-specific ViTDet pyramid in `VGGTPixelDecoder` (input_proj + the mask_features path).
    COCO has no VGGT tokens, so upstream's `input_proj`, FPN lateral (`adapter_1`/`layer_1`) and
    `mask_features` conv are used verbatim here. That front end is a *deliberate* deviation
    (docs/MASKDINO.md §3) and has no upstream counterpart to be checked against.
  - `matcher.py` / `criterion.py` / DN query generation — training-only, not reached at eval.
  - `multiframe.py` — no upstream counterpart at all.

Level-ordering note (the one real trap)
---------------------------------------
Upstream's pixel decoder returns `multi_scale_features` LOW→HIGH resolution ([res5, res4, res3])
and its decoder then walks that list *backwards* (`idx = num_feature_levels-1-i`). Our port takes
the list already ordered HIGH→LOW and walks it forwards. Both therefore flatten the same tensors
in the same sequence order — but only if the adapter reverses the list on the way in, which is
what `PortedDecoderAdapter` does. (The decoder's own `input_proj` is an empty `nn.Sequential`
under this config, so its index convention carries no weights.)

Usage
-----
    # control: unmodified upstream, same env/data/checkpoint -> should give ~46.1 / ~51.5
    python scripts/coco_transplant_eval.py --mode baseline --limit 100

    # the actual test: our ported modules, upstream weights
    python scripts/coco_transplant_eval.py --mode ours --limit 100

    # full 5000-image val2017
    sbatch slurm/coco_transplant.sh

Must run under the reference env (detectron2 + pycocotools + the compiled MSDeformAttn op):
    /cluster/scratch/niacobone/MaskDINO/myenv/bin/python
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

MASKDINO_ROOT = os.environ.get("MASKDINO_ROOT", "/cluster/scratch/niacobone/MaskDINO")
COCO_ROOT = os.environ.get("COCO_ROOT", "/cluster/scratch/niacobone")
REPO_ROOT = Path(__file__).resolve().parents[1]

# detectron2 registers coco_2017_val at import time from this env var; must be set first.
os.environ.setdefault("DETECTRON2_DATASETS", COCO_ROOT)
sys.path.insert(0, MASKDINO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from detectron2.checkpoint import DetectionCheckpointer  # noqa: E402
from detectron2.config import get_cfg  # noqa: E402
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader  # noqa: E402
from detectron2.data import DatasetMapper  # noqa: E402
from detectron2.evaluation import COCOEvaluator  # noqa: E402
from detectron2.modeling import build_model  # noqa: E402
from detectron2.projects.deeplab import add_deeplab_config  # noqa: E402
from detectron2.utils.logger import setup_logger  # noqa: E402

from maskdino import add_maskdino_config  # noqa: E402

# ---- the modules under test -------------------------------------------------------------------
from models.maskdino.decoder import MaskDINODecoder  # noqa: E402
from models.maskdino.pixel_decoder import MSDeformAttnEncoder, MSDeformAttnEncoderLayer  # noqa: E402
from models.maskdino.utils import PositionEmbeddingSine  # noqa: E402

# The config the released checkpoint was trained with, and the one our port's defaults mirror
# (docs/MASKDINO.md §2): 3 feature levels, encoder FFN 1024, decoder FFN 2048, 300 queries,
# two-stage, DN "seg", initialize_box_type "bitmask". The `_dowsample1_2048` variants use
# TOTAL_NUM_FEATURE_LEVELS 4 / encoder FFN 2048 and do NOT match these weights.
DEFAULT_CONFIG = f"{MASKDINO_ROOT}/configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s.yaml"
DEFAULT_OUTPUT_ROOT = Path(
    "/cluster/work/igp_psr/niacobone/distillation/output/coco_transplant")
DEFAULT_WEIGHTS = (
    f"{MASKDINO_ROOT}/weights/"
    "maskdino_r50_50ep_300q_hid1024_3sd1_instance_maskenhanced_mask46.1ap_box51.5ap.pth"
)


class PortedEncoderAdapter(nn.Module):
    """
    Upstream `MaskDINOEncoder` with its MSDeformAttn encoder swapped for **our** port.

    Everything VGGT-specific in `VGGTPixelDecoder` (the ViTDet pyramid) has no COCO counterpart,
    so the pieces that turn backbone features into encoder inputs — `input_proj`, the res2 FPN
    lateral and the final `mask_features` conv — are taken from the loaded upstream module by
    reference. The deformable encoder itself, the position embedding, the level embedding and the
    flatten/reference-point plumbing are ours.
    """

    def __init__(self, upstream: nn.Module, conv_dim: int = 256, dim_feedforward: int = 1024,
                 nheads: int = 8, enc_n_points: int = 4, enc_layers: int = 6):
        super().__init__()
        # --- not under test: reused verbatim from the loaded upstream module -------------------
        self.input_proj = upstream.input_proj
        self.lateral_convs = nn.ModuleList(upstream.lateral_convs)
        self.output_convs = nn.ModuleList(upstream.output_convs)
        self.mask_features = upstream.mask_features
        self.in_features = list(upstream.in_features)
        self.transformer_in_features = list(upstream.transformer_in_features)
        self.num_fpn_levels = upstream.num_fpn_levels
        self.high_resolution_index = upstream.high_resolution_index
        self.total_num_feature_levels = upstream.total_num_feature_levels

        # --- under test: our ported encoder ----------------------------------------------------
        self.pe_layer = PositionEmbeddingSine(conv_dim // 2, normalize=True)
        layer = MSDeformAttnEncoderLayer(conv_dim, dim_feedforward, 0.0, "relu",
                                         self.total_num_feature_levels, nheads, enc_n_points)
        self.encoder = MSDeformAttnEncoder(layer, enc_layers)

        # transplant upstream's trained encoder weights (parameter names match 1:1)
        src = upstream.transformer.encoder.state_dict()
        missing, unexpected = self.encoder.load_state_dict(src, strict=True), None
        self.level_embed = nn.Parameter(upstream.transformer.level_embed.data.clone())

    @torch.no_grad()
    def forward_features(self, features, masks):
        # Upstream order: transformer_in_features[::-1] == [res5, res4, res3] (LOW->HIGH res).
        # input_proj[idx] is tied to that position, and so is level_embed[idx].
        srcs, poss, shapes = [], [], []
        for idx, f in enumerate(self.transformer_in_features[::-1]):
            x = features[f].float()
            srcs.append(self.input_proj[idx](x))
            poss.append(self.pe_layer(x))

        src_flatten, pos_flatten = [], []
        for lvl, (src, pos) in enumerate(zip(srcs, poss)):
            _, _, h, w = src.shape
            shapes.append((h, w))
            src_flatten.append(src.flatten(2).transpose(1, 2))
            pos_flatten.append(pos.flatten(2).transpose(1, 2) + self.level_embed[lvl].view(1, 1, -1))
        src_flatten = torch.cat(src_flatten, 1)
        pos_flatten = torch.cat(pos_flatten, 1)
        spatial_shapes = torch.as_tensor(shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        bs = src_flatten.shape[0]
        # Upstream disables the padding mask at both encoder and decoder ("it does not affect
        # performance": enable_mask == 0), so valid_ratios are all ones there too. Our port
        # hardcodes that same state.
        valid_ratios = torch.ones(bs, self.total_num_feature_levels, 2, device=src_flatten.device)

        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios,
                              pos_flatten, None)

        out = []
        for lvl, (h, w) in enumerate(shapes):
            start = int(level_start_index[lvl])
            out.append(memory[:, start:start + h * w].transpose(1, 2).view(bs, -1, h, w))

        multi_scale_features = out[:self.total_num_feature_levels]

        # Upstream FPN top-down step onto res2 -> the 1/4-resolution mask_features (not under test).
        for idx, f in enumerate(self.in_features[:self.num_fpn_levels][::-1]):
            cur_fpn = self.lateral_convs[idx](features[f].float())
            y = cur_fpn + F.interpolate(out[self.high_resolution_index], size=cur_fpn.shape[-2:],
                                        mode="bilinear", align_corners=False)
            out.append(self.output_convs[idx](y))

        return self.mask_features(out[-1]), out[0], multi_scale_features


class PortedDecoderAdapter(nn.Module):
    """Wraps our `MaskDINODecoder` in upstream's predictor call signature."""

    def __init__(self, ours: MaskDINODecoder):
        super().__init__()
        self.ours = ours

    def forward(self, x, mask_features, masks, targets=None):
        # See the level-ordering note in the module docstring: upstream hands over LOW->HIGH,
        # our port wants HIGH->LOW.
        return self.ours(list(reversed(x)), mask_features, targets=targets)


def transplant(model, cfg, verbose=True):
    """Replace the upstream pixel-decoder encoder and predictor with our ported ones, in place."""
    head = model.sem_seg_head
    up_pd = head.pixel_decoder
    up_pred = head.predictor

    ours_dec = MaskDINODecoder(
        in_channels=cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM,
        num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
        hidden_dim=cfg.MODEL.MaskDINO.HIDDEN_DIM,
        num_queries=cfg.MODEL.MaskDINO.NUM_OBJECT_QUERIES,
        nheads=cfg.MODEL.MaskDINO.NHEADS,
        dim_feedforward=cfg.MODEL.MaskDINO.DIM_FEEDFORWARD,
        dec_layers=cfg.MODEL.MaskDINO.DEC_LAYERS,
        mask_dim=cfg.MODEL.SEM_SEG_HEAD.MASK_DIM,
        enforce_input_project=cfg.MODEL.MaskDINO.ENFORCE_INPUT_PROJ,
        two_stage=cfg.MODEL.MaskDINO.TWO_STAGE,
        dn=cfg.MODEL.MaskDINO.DN,
        noise_scale=cfg.MODEL.MaskDINO.DN_NOISE_SCALE,
        dn_num=cfg.MODEL.MaskDINO.DN_NUM,
        initialize_box_type=cfg.MODEL.MaskDINO.INITIALIZE_BOX_TYPE,
        initial_pred=cfg.MODEL.MaskDINO.INITIAL_PRED,
        learn_tgt=cfg.MODEL.MaskDINO.LEARN_TGT,
        total_num_feature_levels=cfg.MODEL.SEM_SEG_HEAD.TOTAL_NUM_FEATURE_LEVELS,
        dropout=cfg.MODEL.MaskDINO.DROPOUT,
        dec_n_points=4,
    )
    result = ours_dec.load_state_dict(up_pred.state_dict(), strict=True)
    if verbose:
        print(f"[transplant] decoder weights -> our MaskDINODecoder: {result}", flush=True)

    ours_pd = PortedEncoderAdapter(
        up_pd,
        conv_dim=cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM,
        dim_feedforward=cfg.MODEL.SEM_SEG_HEAD.DIM_FEEDFORWARD,
        nheads=cfg.MODEL.MaskDINO.NHEADS,
        enc_layers=cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS,
    )
    if verbose:
        print("[transplant] encoder weights -> our MSDeformAttnEncoder: <All keys matched>",
              flush=True)

    head.pixel_decoder = ours_pd.to(model.device)
    head.predictor = PortedDecoderAdapter(ours_dec).to(model.device)

    # Provenance guard. A transplant test is worthless if it silently keeps running upstream's
    # code, and identical numbers would be exactly what that failure mode looks like. Assert the
    # live modules are the ones in this repo.
    assert type(head.predictor.ours).__module__ == "models.maskdino.decoder", \
        f"decoder is not ours: {type(head.predictor.ours).__module__}"
    assert type(head.pixel_decoder.encoder).__module__ == "models.maskdino.pixel_decoder", \
        f"encoder is not ours: {type(head.pixel_decoder.encoder).__module__}"
    from models.maskdino.ms_deform_attn import MSDeformAttn as OurMSDA
    n_msda = sum(isinstance(m, OurMSDA) for m in head.modules())
    assert n_msda == cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS + cfg.MODEL.MaskDINO.DEC_LAYERS, \
        f"expected 6 encoder + 9 decoder ported MSDeformAttn modules, found {n_msda}"
    if verbose:
        print(f"[transplant] live modules: decoder={type(head.predictor.ours).__module__}, "
              f"encoder={type(head.pixel_decoder.encoder).__module__}, "
              f"ported MSDeformAttn instances={n_msda}", flush=True)
    return model


def perturb(model, scale=1.05):
    """
    Negative control: nudge one weight inside OUR decoder. If the ported code is really the live
    path, the metrics must move. If they do not, the transplant is not being exercised and an
    identical-to-baseline result means nothing.
    """
    w = model.sem_seg_head.predictor.ours.decoder.layers[0].linear1.weight
    w.data.mul_(scale)
    print(f"[perturb] scaled our decoder.layers.0.linear1.weight by {scale}", flush=True)
    return model


def setup_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(args.config)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.DATASETS.TEST = ("coco_2017_val",)
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.freeze()
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["baseline", "ours"], required=True,
                    help="baseline = unmodified upstream (the control); ours = ported modules")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N images (0 = all 5000)")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--output", default=None, help="output dir for the evaluator (default: auto)")
    ap.add_argument("--perturb", type=float, default=0.0,
                    help="negative control: scale one weight in OUR decoder by this factor "
                         "(implies --mode ours). Metrics MUST change, proving the port is live.")
    args = ap.parse_args()

    setup_logger(name="detectron2")
    cfg = setup_cfg(args)

    # Group storage, not the repo: the evaluator dumps a ~320 MB predictions file per mode.
    out_dir = Path(args.output or (DEFAULT_OUTPUT_ROOT / args.mode))
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    if args.mode == "ours":
        transplant(model, cfg)
        if args.perturb:
            perturb(model, args.perturb)
    model.eval()

    dataset_dicts = DatasetCatalog.get("coco_2017_val")
    if args.limit:
        dataset_dicts = dataset_dicts[: args.limit]
    img_ids = [d["image_id"] for d in dataset_dicts]
    print(f"[eval] mode={args.mode}  images={len(dataset_dicts)}", flush=True)

    loader = build_detection_test_loader(dataset_dicts, mapper=DatasetMapper(cfg, False),
                                         num_workers=4)
    evaluator = COCOEvaluator("coco_2017_val", output_dir=str(out_dir))
    evaluator.reset()

    with torch.no_grad():
        for i, inputs in enumerate(loader):
            outputs = model(inputs)
            evaluator.process(inputs, outputs)
            if (i + 1) % 250 == 0:
                print(f"  ... {i + 1}/{len(dataset_dicts)}", flush=True)

    results = evaluator.evaluate(img_ids=img_ids if args.limit else None)
    print(json.dumps(results, indent=2, default=str), flush=True)

    summary = {
        "mode": args.mode,
        "num_images": len(dataset_dicts),
        "weights": args.weights,
        "segm_AP": results.get("segm", {}).get("AP"),
        "segm_AP50": results.get("segm", {}).get("AP50"),
        "bbox_AP": results.get("bbox", {}).get("AP"),
        "bbox_AP50": results.get("bbox", {}).get("AP50"),
        # Target = upstream's README model-zoo row for THIS checkpoint ("MaskDINO (hid 1024)",
        # 47M params). NOT a paper table value: Table 3's 50ep/300q R50 rows are 46.0/50.5
        # (plain) and 46.3/51.7 (mask-enhanced), the latter being the wider hid2048 release.
        "model_zoo_reference": {"segm_AP": 46.1, "bbox_AP": 51.5,
                                "source": "IDEA-Research/MaskDINO README, row 'MaskDINO (hid 1024)'"},
    }
    (out_dir / f"summary_{args.mode}_{len(dataset_dicts)}.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
