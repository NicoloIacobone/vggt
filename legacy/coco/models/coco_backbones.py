"""
Frozen backbones for the COCO backbone-swap study (docs/MASKDINO_COCO.md).

The ScanNet track has exactly one backbone (VGGT-1B) hard-wired into
`models/maskdino/model.py`. The COCO study needs three, compared under an identical decoder and
an identical schedule, so they are behind one interface here:

    backbone(images) -> {"levels": [high-res, ..., low-res], "highres": Tensor | None}

`levels` are the maps the deformable encoder runs on (`num_feature_levels` of them, ordered
HIGH→LOW resolution, matching `VGGTPixelDecoder`'s convention). `highres` is an optional extra map
at a *finer* stride, used only to build `mask_features` — a ResNet has one for free (res2, stride
4) whereas a plain ViT does not, which is exactly the asymmetry the study is about.

All three are frozen and in eval mode: only the pixel decoder + MaskDINO decoder train, as in
every run of this project.

| name       | input      | levels (at 518px)       | highres      | channels |
|------------|------------|-------------------------|--------------|----------|
| `vggt`     | 518² squash| 37² (synthesised 19²,10²) | none       | 2048     |
| `dinov2`   | 518² squash| 37² (synthesised 19²,10²) | none       | 1024     |
| `resnet50` | 518² squash| 65², 33², 17² (res3/4/5)| 130² (res2)  | 512/1024/2048 |

`vggt` and `dinov2` share the patch size (14) and therefore the token geometry exactly, which is
what makes "did VGGT's 3D pretraining cost 2D semantics?" a controlled question.
"""

import contextlib
from typing import Dict, List, Optional

import torch
from torch import Tensor, nn

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class FrozenBackbone(nn.Module):
    """Common contract: frozen, eval-only, reports its output channels and strides."""

    #: channel width of each entry of `levels`
    out_channels: List[int]
    #: stride of each entry of `levels` relative to the input image
    strides: List[int]
    #: channels / stride of the extra high-resolution map, or None
    highres_channels: Optional[int] = None
    highres_stride: Optional[int] = None
    #: how many pyramid levels the module produces itself (the pixel decoder synthesises the rest)
    native_levels: int = 1

    #: autocast dtype for the backbone's own forward (the head has its own)
    autocast_dtype: torch.dtype = torch.float32

    def train(self, mode: bool = True):
        super().train(False)          # frozen backbones never leave eval
        return self

    def _amp(self, images: Tensor):
        """Autocast context for this backbone, or a no-op on CPU / in fp32."""
        if images.is_cuda and self.autocast_dtype != torch.float32:
            return torch.autocast("cuda", dtype=self.autocast_dtype)
        return contextlib.nullcontext()

    def forward(self, images: Tensor) -> Dict[str, object]:
        raise NotImplementedError


class VGGTBackbone(FrozenBackbone):
    """
    Frozen VGGT-1B aggregator, run with S=1 (each COCO image is its own scene).

    Returns the last aggregator layer's patch tokens as a single [B, 2048, h, w] map, exactly the
    tensor `VGGTPixelDecoder` consumes on ScanNet. The aggregator normalises internally with the
    ImageNet statistics, so it wants images in [0, 1].
    """

    def __init__(self, feature_layers=(-1,), dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        from vggt.models.vggt import VGGT

        try:
            vggt = VGGT.from_pretrained("facebook/VGGT-1B")
        except Exception as e:  # offline / no HF cache → random init (tests only)
            print(f"⚠ Could not load pretrained VGGT: {e}\n  Falling back to random init.")
            vggt = VGGT()
        self.aggregator = vggt.aggregator
        for p in self.aggregator.parameters():
            p.requires_grad = False
        self.aggregator.eval()

        self.feature_layers = tuple(feature_layers)
        self.autocast_dtype = dtype
        self.out_channels = [2048 * len(self.feature_layers)]
        self.strides = [14]
        self.native_levels = 1
        self.wants_normalized = False        # the aggregator normalises internally

    @torch.no_grad()
    def forward(self, images: Tensor) -> Dict[str, object]:
        b, _, h, w = images.shape
        with self._amp(images):
            agg_list, patch_start_idx = self.aggregator(images.unsqueeze(1))   # [B,1,3,H,W]
        feats = torch.cat([agg_list[i].float() for i in self.feature_layers], dim=-1)
        tokens = feats[:, 0, int(patch_start_idx):, :]                          # [B, h*w, C]
        gh, gw = h // 14, w // 14
        level = tokens.transpose(1, 2).reshape(b, -1, gh, gw)
        return {"levels": [level], "highres": None}


#: Official DINOv2 ViT-L/14 *with registers* — the exact checkpoint VGGT's patch embed descends
#: from, so it loads into VGGT's vendored ViT with zero missing/unexpected keys.
DINOV2_VITL14_REG_URL = ("https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/"
                         "dinov2_vitl14_reg4_pretrain.pth")


class DINOv2Backbone(FrozenBackbone):
    """
    Frozen DINOv2 ViT-L/14 (with registers) — VGGT's own patch-embed ancestor.

    The control that separates "the 37×37 grid is too coarse for COCO" from "VGGT's 3D pretraining
    is a poor 2D-semantics prior": same patch size, same architecture, same token count. Only the
    24 alternating-attention aggregator blocks VGGT stacks on top are missing.

    Built from **VGGT's own vendored ViT** (`vggt/layers/vision_transformer.py::vit_large`) rather
    than `torch.hub.load("facebookresearch/dinov2", ...)`: it is the same class VGGT instantiates
    for `patch_embed="dinov2_vitl14_reg"`, the official weights load into it at `strict=True`, and
    it needs no network access at construction time beyond the cached checkpoint.
    """

    def __init__(self, dtype: torch.dtype = torch.bfloat16, img_size: int = 518):
        super().__init__()
        from vggt.layers.vision_transformer import vit_large

        self.vit = vit_large(img_size=img_size, patch_size=14, num_register_tokens=4,
                             interpolate_antialias=True, interpolate_offset=0.0,
                             block_chunks=0, init_values=1.0)
        try:
            sd = torch.hub.load_state_dict_from_url(DINOV2_VITL14_REG_URL, map_location="cpu")
            self.vit.load_state_dict(sd, strict=True)
            print("✓ Loaded pretrained DINOv2 ViT-L/14-reg")
        except Exception as e:  # offline / no cache → random init (tests only)
            print(f"⚠ Could not load pretrained DINOv2: {e}\n  Falling back to random init.")
        for p in self.vit.parameters():
            p.requires_grad = False
        self.vit.eval()

        self.autocast_dtype = dtype
        self.out_channels = [int(self.vit.embed_dim)]
        self.strides = [14]
        self.native_levels = 1
        self.wants_normalized = True         # plain ViT: caller must ImageNet-normalise
        self.register_buffer("_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, images: Tensor) -> Dict[str, object]:
        b, _, h, w = images.shape
        x = (images - self._mean) / self._std
        with self._amp(images):
            out = self.vit.forward_features(x)
        tokens = out["x_norm_patchtokens"].float()                              # [B, h*w, C]
        gh, gw = h // 14, w // 14
        level = tokens.transpose(1, 2).reshape(b, -1, gh, gw)
        return {"levels": [level], "highres": None}


class ResNet50Backbone(FrozenBackbone):
    """
    Frozen ImageNet ResNet-50 — the "original backbone" reference.

    Gives the encoder res3/res4/res5 (strides 8/16/32) and res2 (stride 4) for `mask_features`,
    which is exactly the pyramid upstream MaskDINO's `MaskDINOEncoder` consumes. Frozen, unlike
    upstream (which finetunes it): the frozen-vs-finetuned gap is a known confound and is stated
    as such in the results, because every arm here freezes its backbone.
    """

    def __init__(self, weights: str = "IMAGENET1K_V1", **_):
        super().__init__()
        from torchvision.models import resnet50

        net = resnet50(weights=weights)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2 = net.layer1, net.layer2      # res2 (s4), res3 (s8)
        self.layer3, self.layer4 = net.layer3, net.layer4      # res4 (s16), res5 (s32)
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

        self.out_channels = [512, 1024, 2048]                  # res3, res4, res5 (HIGH→LOW)
        self.strides = [8, 16, 32]
        self.highres_channels = 256                            # res2
        self.highres_stride = 4
        self.native_levels = 3
        self.wants_normalized = True
        self.register_buffer("_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, images: Tensor) -> Dict[str, object]:
        x = (images - self._mean) / self._std
        x = self.stem(x)
        res2 = self.layer1(x)
        res3 = self.layer2(res2)
        res4 = self.layer3(res3)
        res5 = self.layer4(res4)
        return {"levels": [res3.float(), res4.float(), res5.float()], "highres": res2.float()}


BACKBONES = {"vggt": VGGTBackbone, "dinov2": DINOv2Backbone, "resnet50": ResNet50Backbone}


def build_backbone(name: str, **kwargs) -> FrozenBackbone:
    if name not in BACKBONES:
        raise ValueError(f"unknown backbone {name!r}; choose from {sorted(BACKBONES)}")
    return BACKBONES[name](**kwargs)
