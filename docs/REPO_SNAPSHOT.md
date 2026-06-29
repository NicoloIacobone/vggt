# Repo Snapshot

A single-file orientation document for the VGGT + D4RT segmentation-head project.
Generated 2026-06-29. For the full narrative (architecture, results, constraints) see
[`docs/MILESTONES.md`](MILESTONES.md); this file is the at-a-glance map plus the verbatim
source of the decoder head and the data pipeline.

> Scope: the dataset artifacts (`data/` on group storage / node-local scratch) and
> checkpoint outputs (`output/<run>/checkpoint*.pth`) are excluded. The `data/` *source
> module* (the data pipeline) is the loader code below — it is included in full as requested.

## Directory tree (code, docs, infra; vendored examples/logs/visualizations elided)

```
vggt/
├── CLAUDE.md                      Project instructions for Claude Code (read first)
├── README.md  LICENSE.txt  pyproject.toml  requirements*.txt
│
├── data/                         ── THIS PROJECT'S DATA PIPELINE ──
│   ├── __init__.py
│   └── scannet_overfit.py        ScanNet loader: frames + per-class/per-instance masks →
│                                 cross-view-consistent global instance IDs (full source below)
│
├── models/                       ── SEGMENTATION HEAD (trainable, ~6.5M params) ──
│   ├── d4rt_decoder.py           QueryGenerator + InstanceDecoder + head wrapper (full source below)
│   └── mask_upsampler.py         Optional MaskDINO-style pixel decoder for sharper masks (mask_upsample>1)
│
├── train/                        ── MATCHER / LOSS / METRICS ──
│   ├── loss.py                   PointBipartiteMatcher (Hungarian) + D4RTLoss (focal+dice+BCE, no-object)
│   └── eval_metrics.py           Instance-seg metrics: mIoU / AP50 / AP75 / mAP / class_acc
│
├── scripts/                      ── ENTRY POINTS ──
│   ├── train_multiscene.py       Real training: caches frozen backbone feats once/scene, trains head
│   ├── train_overfit.py          Single-scene overfit sanity check (gradient-flow smoke test)
│   ├── visualize_masks.py        Render 2D mask overlays from a checkpoint → <run>/visualizations/
│   └── plot_scaling.py           Plot val metrics vs #train scenes from per-run metrics.jsonl
│
├── tests/                        ── STANDALONE CPU TESTS (not pytest) ──
│   ├── test_phase2.py            Dataset loader + cross-view instance invariants
│   ├── test_phase3.py            QueryGenerator
│   ├── test_phase4.py            InstanceDecoder + dense mask head
│   ├── test_phase5.py            Matcher + losses
│   ├── test_mask_upsampler.py    MaskUpsampler pixel decoder + GT-resolution match
│   ├── test_eval.py              Instance-segmentation metrics
│   ├── test_milestone2.py        No-object loss, grid queries, augmentation, early-stop, query modes
│   └── test_visualize_masks.py   visualize_masks checkpoint-format handling + overlays
│
├── slurm/                        ── CLUSTER JOBS ──
│   ├── stage_dataset.sh          Copy+unpack dataset tar to $TMPDIR, export SCANNET_ROOT
│   ├── train_scale10/25/50/100.sh  Scaling-curve jobs (scenes 0000-00NN, val 0080-0082 held out)
│   └── train_full.sh             Full-dataset training job
│
├── demos/                        ── UPSTREAM VGGT DEMOS (+ our seg hooks) ──
│   ├── demo_gradio.py            3D viewer; --seg_checkpoint adds "Color By: Predicted Instances"
│   ├── demo_viser.py  demo_colmap.py  visual_util.py
│
├── docs/                         ── PROJECT DOCS ──
│   ├── MILESTONES.md             Consolidated summary of Milestones 1-3 (read first)
│   ├── todo.md                   Current open task list
│   ├── HOOK_PLAN.md              Where/how the decoder hooks into VGGT
│   ├── slides_meeting_jun_15.md  Most recent supervision slides
│   ├── REPO_SNAPSHOT.md          (this file)
│   └── old/                      Archived per-milestone detail, plans, feedback, prompts
│
├── vggt/                         ── UPSTREAM BACKBONE (frozen, do NOT modify) ──
│   ├── models/vggt.py            VGGT wrapper
│   ├── models/aggregator.py      24-block alternating per-frame / global cross-frame attention
│   ├── heads/                    Original camera / depth / point / track heads
│   ├── layers/                   ViT building blocks (attention, block, rope, mlp, patch_embed, …)
│   ├── dependency/               VGGSfM tracker + COLMAP/projection utilities
│   └── utils/                    geometry, pose_enc, rotation, load_fn, visual_track helpers
│
└── training/                     ── UPSTREAM Co3D FINETUNING FRAMEWORK (unrelated to this project) ──
    ├── trainer.py  launch.py  loss.py  config/
    ├── data/                     Upstream datasets (co3d, vkitti), augmentation, dataloaders
    └── train_utils/             checkpoint, distributed, optimizer, freeze, logging helpers
```

### Module purpose, one line each

| Module | Purpose |
|--------|---------|
| `data/scannet_overfit.py` | Load ScanNet `subset/` frames + binary masks; emit one global, cross-view-consistent instance ID per class (or per instance with `--instance_level`). |
| `models/d4rt_decoder.py` | The DETR-like head: point/learned/hybrid query generation, cross-attention decoder over VGGT features, class logits + dense cosine-similarity masks. |
| `models/mask_upsampler.py` | Optional pixel decoder that upsamples the 37×37 patch-feature map for sharper masks when `mask_upsample>1`. |
| `train/loss.py` | Hungarian bipartite matcher (mask-aware Dice+BCE cost) + combined focal/dice/BCE loss with optional no-object term; batch-aware. |
| `train/eval_metrics.py` | Prompted (GT-centroid) and unprompted (grid) instance-segmentation metrics: mIoU, AP50/75, mAP, class accuracy. |
| `scripts/train_multiscene.py` | Main training loop; caches frozen-backbone features once per scene bundle, then trains only the head each epoch. |
| `scripts/train_overfit.py` | Single-scene overfit harness for verifying gradient flow of new components. |
| `scripts/visualize_masks.py` | Re-render 2D prediction overlays from any checkpoint. |
| `scripts/plot_scaling.py` | Build the scaling curve (val metric vs #train scenes) from `metrics.jsonl`. |
| `vggt/` | Upstream frozen 3D-reconstruction backbone; hook point is `aggregated_tokens_list[-1]`. |
| `training/` | Upstream Co3D finetuning framework — not used by this project. |

---

## Full source: `models/d4rt_decoder.py` (the decoder)

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from models.mask_upsampler import MaskUpsampler


class FourierPositionalEncoding(nn.Module):
    """
    Fourier positional encoding for 2D coordinates.

    Encodes (u, v) coordinates using sine and cosine at different frequencies.
    Output dimension is 4 * num_freqs (sin & cos for each of u and v).

    Args:
        num_freqs (int): Number of frequency bands
        max_freq (float): Maximum frequency (controls the frequency range)
    """

    def __init__(self, num_freqs: int = 16, max_freq: float = 10.0):
        super().__init__()
        self.num_freqs = num_freqs
        self.max_freq = max_freq

        # Precompute frequency bands
        freqs = torch.logspace(0, math.log10(max_freq), num_freqs)
        self.register_buffer("freqs", freqs)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Encode coordinates using Fourier features.

        Args:
            coords (torch.Tensor): Coordinates with shape [B, N, 2] where last dim is (u, v)

        Returns:
            torch.Tensor: Fourier encoded features with shape [B, N, 4 * num_freqs]
        """
        B, N, _ = coords.shape

        # Expand coordinates for each frequency: [B, N, num_freqs, 2]
        coords_expanded = coords.unsqueeze(2)  # [B, N, 1, 2]
        freqs = self.freqs.view(1, 1, -1, 1)  # [1, 1, num_freqs, 1]
        scaled_coords = coords_expanded * freqs  # [B, N, num_freqs, 2]

        # Apply sin and cos
        sin_encoding = torch.sin(2 * math.pi * scaled_coords)  # [B, N, num_freqs, 2]
        cos_encoding = torch.cos(2 * math.pi * scaled_coords)  # [B, N, num_freqs, 2]

        # Interleave sin and cos: [B, N, num_freqs, 4]
        encoding = torch.stack([sin_encoding, cos_encoding], dim=-1)  # [B, N, num_freqs, 2, 2]
        encoding = encoding.view(B, N, self.num_freqs * 4)  # [B, N, 4 * num_freqs]

        return encoding


class LocalPatchFeatureExtractor(nn.Module):
    """
    Extract local RGB patch features using grid_sample.

    Extracts a patch around each (u, v) coordinate from the input images,
    then encodes it using a small MLP.

    Args:
        patch_size (int): Size of the patch (e.g., 9 for 9x9 patches)
        hidden_dim (int): Output dimension of the patch features
        in_channels (int): Number of input channels (default: 3 for RGB)
    """

    def __init__(self, patch_size: int = 9, hidden_dim: int = 256, in_channels: int = 3):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        # MLP to encode the flattened patch
        patch_feat_dim = in_channels * (patch_size ** 2)
        self.patch_encoder = nn.Sequential(
            nn.Linear(patch_feat_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(
        self,
        images: torch.Tensor,
        coords: torch.Tensor,
        view_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract and encode local patches around coordinates.

        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W]
            coords (torch.Tensor): Normalized coordinates with shape [B, N, 2] in [0, 1]
            view_ids (torch.Tensor, optional): View indices with shape [B, N] (used for per-view extraction)

        Returns:
            torch.Tensor: Patch features with shape [B, N, hidden_dim]
        """
        B, S, C, H, W = images.shape
        N = coords.shape[1]

        if view_ids is None:
            # Use first view for all queries
            view_ids = torch.zeros((B, N), dtype=torch.long, device=images.device)

        if torch.any(view_ids >= S) or torch.any(view_ids < 0):
            raise ValueError(
                f"view_ids out of range: values must be in [0, {S}) but got "
                f"[{int(view_ids.min())}, {int(view_ids.max())}]"
            )

        # Flatten batch and sequence dimensions (reshape, not view: backbone-adjacent tensors
        # are not guaranteed contiguous)
        images_flat = images.reshape(B * S, C, H, W)  # [B*S, 3, H, W]

        # Convert normalized coords [0, 1] to grid_sample format [-1, 1]
        grid_coords = coords * 2 - 1  # [B, N, 2]

        # Create offset grid for the patch around center
        # Use normalized pixel offsets
        half_size = (self.patch_size - 1) / 2
        patch_offsets = torch.linspace(
            -half_size, half_size, self.patch_size, device=coords.device
        )  # [patch_size]

        # Normalize offsets to [-1, 1] range based on image dimensions
        offset_u = patch_offsets * 2 / W  # Offsets in normalized coords
        offset_v = patch_offsets * 2 / H

        # Create 2D grid for the patch
        grid_u, grid_v = torch.meshgrid(offset_u, offset_v, indexing="ij")
        patch_grid = torch.stack([grid_u, grid_v], dim=-1)  # [patch_size, patch_size, 2]

        # Vectorized extraction (item 8.6): gather each query's source image and run ONE
        # grid_sample over all B*N patches instead of a Python loop of B*N calls.
        batch_offsets = torch.arange(B, device=images.device).unsqueeze(1) * S  # [B, 1]
        img_indices = (batch_offsets + view_ids).reshape(-1)                    # [B*N]
        imgs_q = images_flat[img_indices]                                       # [B*N, 3, H, W]

        # Per-query sampling grid centered at the query point.
        centers = grid_coords.reshape(B * N, 1, 1, 2)                # [B*N, 1, 1, 2]
        sample_grid = patch_grid.unsqueeze(0) + centers              # [B*N, ps, ps, 2]

        patches = F.grid_sample(
            imgs_q,
            sample_grid,
            align_corners=False,
            padding_mode="border",
            mode="bilinear",
        )  # [B*N, 3, patch_size, patch_size]

        patches = patches.reshape(B * N, -1)  # [B*N, 3*patch_size^2]
        patch_features = self.patch_encoder(patches)  # [B*N, hidden_dim]
        patch_features = patch_features.view(B, N, self.hidden_dim)  # [B, N, hidden_dim]

        return patch_features


class QueryGenerator(nn.Module):
    """
    D4RT Query Generator for instance segmentation.

    Generates attention queries by combining:
    1. Fourier positional encoding of (u, v) coordinates
    2. View embeddings (which view the point is from)
    3. Local RGB patch features (9x9 patch around the point)

    Args:
        num_views (int): Maximum number of views in a batch
        hidden_dim (int): Dimension of query embeddings
        patch_size (int): Size of local RGB patch (default: 9)
        num_freqs (int): Number of Fourier frequency bands
        max_freq (float): Maximum frequency for Fourier encoding
    """

    def __init__(
        self,
        num_views: int = 10,
        hidden_dim: int = 256,
        patch_size: int = 9,
        num_freqs: int = 16,
        max_freq: float = 10.0,
        query_mode: str = "point",
        num_learned_queries: int = 0,
    ):
        super().__init__()
        if query_mode not in ("point", "learned", "hybrid"):
            raise ValueError(f"query_mode must be point/learned/hybrid, got {query_mode!r}")
        if query_mode in ("learned", "hybrid") and num_learned_queries <= 0:
            raise ValueError(f"query_mode={query_mode} needs num_learned_queries > 0")
        self.num_views = num_views
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.query_mode = query_mode
        self.num_learned_queries = num_learned_queries

        # Point-prompt branch (Fourier pos + view embedding + local RGB patch). Always built
        # for point/hybrid; harmless (and kept) for "learned" so the module is uniform.
        self.pos_encoder = FourierPositionalEncoding(num_freqs=num_freqs, max_freq=max_freq)
        pos_encoding_dim = 4 * num_freqs  # sin and cos for u and v
        self.view_embedding = nn.Embedding(num_views, hidden_dim)
        self.patch_extractor = LocalPatchFeatureExtractor(
            patch_size=patch_size, hidden_dim=hidden_dim, in_channels=3
        )
        self.pos_proj = nn.Linear(pos_encoding_dim, hidden_dim)
        self.view_proj = nn.Linear(hidden_dim, hidden_dim)
        self.patch_proj = nn.Linear(hidden_dim, hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)

        # Learned object-query table (true DETR queries) for learned/hybrid modes.
        self.learned_queries = (
            nn.Embedding(num_learned_queries, hidden_dim)
            if query_mode in ("learned", "hybrid") else None
        )

    def forward(
        self,
        coordinates: torch.Tensor,
        view_ids: torch.Tensor,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate decoder queries. Output length always equals `coordinates.shape[1]`, so the
        downstream matcher/loss stay aligned regardless of mode:
          - "point": every slot is a point-prompt query built from its (u, v)/view/patch.
          - "learned": all slots are the learned object queries (coordinates ignored; the
            caller passes a length-`num_learned_queries` placeholder).
          - "hybrid": the first `num_learned_queries` slots are learned object queries, the
            remaining slots are point-prompt queries from `coordinates[:, M:]`.
        """
        B, N = coordinates.shape[0], coordinates.shape[1]
        if self.query_mode == "learned":
            return self.learned_queries.weight.unsqueeze(0).expand(B, -1, -1)  # [B, M, hidden]
        if self.query_mode == "hybrid":
            M = self.num_learned_queries
            learned = self.learned_queries.weight.unsqueeze(0).expand(B, -1, -1)  # [B, M, hidden]
            point = self._point_queries(coordinates[:, M:], view_ids[:, M:], images)
            return torch.cat([learned, point], dim=1)  # [B, M + (N-M), hidden]
        return self._point_queries(coordinates, view_ids, images)

    def _point_queries(
        self,
        coordinates: torch.Tensor,
        view_ids: torch.Tensor,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """Point-prompt queries: Fourier(u,v) + view embedding + local RGB patch, summed."""
        B = coordinates.shape[0]
        N = coordinates.shape[1]

        # Guard the view-embedding table bound (item 8.6): a frame index beyond num_views
        # would silently raise an opaque CUDA indexing error inside nn.Embedding. Size
        # `num_views` to the maximum sequence length you intend to train/evaluate with.
        if torch.any(view_ids >= self.num_views) or torch.any(view_ids < 0):
            raise ValueError(
                f"view_ids must be in [0, num_views={self.num_views}) but got "
                f"[{int(view_ids.min())}, {int(view_ids.max())}]; construct the "
                f"QueryGenerator with num_views >= the max number of frames."
            )

        # 1. Fourier positional encoding
        pos_encoding = self.pos_encoder(coordinates)  # [B, N, 4*num_freqs]
        pos_features = self.pos_proj(pos_encoding)  # [B, N, hidden_dim]

        # 2. View embeddings
        view_features = self.view_embedding(view_ids)  # [B, N, hidden_dim]
        view_features = self.view_proj(view_features)  # [B, N, hidden_dim]

        # 3. Local RGB patch features
        patch_features = self.patch_extractor(images, coordinates, view_ids)  # [B, N, hidden_dim]
        patch_features = self.patch_proj(patch_features)  # [B, N, hidden_dim]

        # Combine all features by summing
        queries = pos_features + view_features + patch_features  # [B, N, hidden_dim]
        queries = self.query_proj(queries)  # [B, N, hidden_dim]

        return queries


class InstanceDecoder(nn.Module):
    """
    DETR-like cross-attention decoder for multi-view instance segmentation.

    Uses a Transformer decoder to process queries using global scene features (from VGGT)
    as memory. Outputs class logits and mask embeddings for each query.

    Args:
        hidden_dim (int): Dimension of query/memory embeddings (default: 256)
        num_classes (int): Number of output classes (19 ScanNet + 1 background = 20)
        num_decoder_layers (int): Number of Transformer decoder layers (default: 4)
        num_heads (int): Number of attention heads (default: 8)
        dim_feedforward (int): Dimension of FFN intermediate layer
        dropout (float): Dropout rate
        mask_embed_dim (int): Dimension of mask embeddings
        memory_dim (int): Dimension of memory from VGGT (default: 2048 for 2*embed_dim)
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_classes: int = 20,
        num_decoder_layers: int = 4,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        mask_embed_dim: int = 256,
        memory_dim: int = 2048,
        mask_upsample: int = 1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.mask_embed_dim = mask_embed_dim
        self.mask_upsample = mask_upsample

        # Project memory from VGGT (2048-dim) to decoder hidden dim (256-dim).
        # The LayerNorm is essential: raw VGGT features have a very large magnitude, so without
        # it the cross-attention output dwarfs the query residual in the decoder and every query
        # collapses to the same memory average (identical outputs for all instances).
        self.memory_proj = nn.Linear(memory_dim, hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)

        # Pixel decoder for dense mask prediction (Mask2Former-style): projects the VGGT
        # patch tokens to a per-pixel feature map of dimension `mask_embed_dim`. A dense mask
        # for each query is the COSINE similarity between its mask embedding and this feature
        # map, scaled by a learnable temperature and shifted by a learnable bias. Cosine
        # (rather than a raw dot-product) keeps the mask logits well-scaled regardless of the
        # (large, un-normalized) VGGT feature norms, which otherwise saturate the sigmoid and
        # stall the gradients.
        self.mask_feature_proj = nn.Linear(memory_dim, mask_embed_dim)
        self.mask_logit_scale = nn.Parameter(torch.tensor(10.0))
        self.mask_logit_bias = nn.Parameter(torch.tensor(0.0))

        # Phase 5: optional MaskDINO-style pixel decoder. At mask_upsample=1 (default) the
        # original Linear projection at the 37×37 patch grid is used (behavior unchanged);
        # for >1 the patch-feature map is upsampled before the cosine-similarity mask product.
        self.mask_upsampler = (
            MaskUpsampler(memory_dim=memory_dim, mask_embed_dim=mask_embed_dim,
                          upsample=mask_upsample)
            if mask_upsample > 1 else None
        )

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        # Output heads
        self.class_head = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes),
        )

        self.mask_embed_head = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, mask_embed_dim),
        )

    def forward(
        self,
        queries: torch.Tensor,
        global_features: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        patch_start_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decode instance predictions using cross-attention.

        Args:
            queries (torch.Tensor): Query embeddings from QueryGenerator [B, N, hidden_dim]
            global_features (torch.Tensor): Global scene features from VGGT [B, S, P, 2*embed_dim]
                where B=batch, S=num_frames, P=num_patches, 2*embed_dim=2048
            images (torch.Tensor, optional): Original images [B, S, 3, H, W] (for reference)
            patch_start_idx (int, optional): Index where the patch tokens start. The first
                `patch_start_idx` tokens are special (camera/register) tokens and are skipped
                when building the dense per-pixel feature map. Defaults to 0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - class_logits:    [B, N, num_classes] class predictions per query
                - mask_embeddings: [B, N, mask_embed_dim] per-query mask kernels
                - pred_masks:      [B, N, S, h, w] dense mask LOGITS per query per frame, at
                  the VGGT patch-grid resolution (h = w = sqrt(num_patch_tokens))
        """
        B, N, _ = queries.shape
        B_feat, S, P, _ = global_features.shape

        assert B == B_feat, f"Batch size mismatch: queries {B} vs features {B_feat}"

        # Project memory features to decoder dimension
        # Reshape global_features from [B, S, P, 2048] to [B, S*P, 256]
        global_features_flat = global_features.reshape(B, S * P, -1)  # [B, S*P, 2048]
        memory = self.memory_norm(self.memory_proj(global_features_flat))  # [B, S*P, hidden_dim]

        # Cross-attention decoder
        # tgt: queries [B, N, hidden_dim]
        # memory: global features [B, S*P, hidden_dim]
        decoded = self.transformer_decoder(
            tgt=queries,
            memory=memory,
        )  # [B, N, hidden_dim]

        # Skip connection from the (distinct) input queries. The cross-attention tends to
        # collapse all queries toward the same memory-attended average; adding the queries back
        # preserves each instance's identity so the per-query class/mask outputs stay distinct.
        decoded = decoded + queries

        # Output heads
        class_logits = self.class_head(decoded)  # [B, N, num_classes]
        mask_embeddings = self.mask_embed_head(decoded)  # [B, N, mask_embed_dim]

        # Dense mask prediction (Mask2Former-style): build a per-pixel feature map from the
        # patch tokens and take its dot-product with each query's mask embedding.
        start = patch_start_idx if patch_start_idx is not None else 0
        num_patch = P - start
        h = w = int(round(num_patch ** 0.5))
        assert h * w == num_patch, (
            f"Patch tokens ({num_patch}) do not form a square grid; "
            f"check patch_start_idx ({start}) and P ({P})."
        )

        patch_tokens = global_features[:, :, start:start + h * w, :]  # [B, S, h*w, memory_dim]
        if self.mask_upsampler is not None:
            # Upsample the patch-feature map → [B, S, h*f, w*f, mask_embed_dim] (Phase 5).
            pixel_feats = self.mask_upsampler(patch_tokens.reshape(B, S, h, w, -1))
        else:
            pixel_feats = self.mask_feature_proj(patch_tokens)        # [B, S, h*w, mask_embed_dim]
            pixel_feats = pixel_feats.reshape(B, S, h, w, self.mask_embed_dim)

        # pred_masks[b, n, s, i, j] = scale * cos(mask_embeddings[b, n], pixel_feats[b, s, i, j]) + bias
        emb_n = F.normalize(mask_embeddings, dim=-1)
        pix_n = F.normalize(pixel_feats, dim=-1)
        pred_masks = torch.einsum("bnc,bshwc->bnshw", emb_n, pix_n)
        pred_masks = self.mask_logit_scale * pred_masks + self.mask_logit_bias

        return class_logits, mask_embeddings, pred_masks


class D4RTInstanceSegmentationHead(nn.Module):
    """
    Complete D4RT instance segmentation head combining QueryGenerator and InstanceDecoder.

    This is a convenience wrapper that combines the query generation and decoding steps.

    Args:
        num_views (int): Number of views in a batch
        hidden_dim (int): Hidden dimension (default: 256)
        num_classes (int): Number of classes (default: 20)
        num_decoder_layers (int): Number of decoder layers (default: 4)
        patch_size (int): Size of patches for local features (default: 9)
        mask_embed_dim (int): Dimension of mask embeddings (default: 256)
        memory_dim (int): Dimension of memory from VGGT (default: 2048)
    """

    def __init__(
        self,
        num_views: int = 10,
        hidden_dim: int = 256,
        num_classes: int = 20,
        num_decoder_layers: int = 4,
        patch_size: int = 9,
        mask_embed_dim: int = 256,
        memory_dim: int = 2048,
        dropout: float = 0.1,
        query_mode: str = "point",
        num_learned_queries: int = 0,
        mask_upsample: int = 1,
    ):
        super().__init__()
        self.query_mode = query_mode
        self.num_learned_queries = num_learned_queries
        self.mask_upsample = mask_upsample
        self.query_generator = QueryGenerator(
            num_views=num_views,
            hidden_dim=hidden_dim,
            patch_size=patch_size,
            query_mode=query_mode,
            num_learned_queries=num_learned_queries,
        )
        self.instance_decoder = InstanceDecoder(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_decoder_layers=num_decoder_layers,
            mask_embed_dim=mask_embed_dim,
            memory_dim=memory_dim,
            dropout=dropout,
            mask_upsample=mask_upsample,
        )

    def forward(
        self,
        coordinates: torch.Tensor,
        view_ids: torch.Tensor,
        images: torch.Tensor,
        global_features: torch.Tensor,
        patch_start_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate queries and decode instance predictions.

        Args:
            coordinates (torch.Tensor): [B, N, 2] normalized query coordinates
            view_ids (torch.Tensor): [B, N] view indices
            images (torch.Tensor): [B, S, 3, H, W] input images
            global_features (torch.Tensor): [B, S, P, 2*embed_dim] from VGGT aggregator
            patch_start_idx (int, optional): Index where patch tokens start

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - class_logits:    [B, N, num_classes]
                - mask_embeddings: [B, N, mask_embed_dim]
                - pred_masks:      [B, N, S, h, w] dense mask logits at patch resolution
        """
        # Generate queries
        queries = self.query_generator(coordinates, view_ids, images)

        # Decode predictions
        class_logits, mask_embeddings, pred_masks = self.instance_decoder(
            queries, global_features, images, patch_start_idx
        )

        return class_logits, mask_embeddings, pred_masks
```

---

## Full source: `data/scannet_overfit.py` (the data pipeline)

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ScanNet class labels (19 classes + background)
SCANNET_CLASSES = [
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
    "window", "bookshelf", "picture", "counter", "desk", "curtain",
    "refrigerator", "shower curtain", "toilet", "sink", "bathtub", "otherfurniture"
]

CLASS_TO_IDX = {cls_name: idx + 1 for idx, cls_name in enumerate(SCANNET_CLASSES)}
IDX_TO_CLASS = {idx + 1: cls_name for idx, cls_name in enumerate(SCANNET_CLASSES)}
IDX_TO_CLASS[0] = "background"


def load_frames_by_name(
    scene_dir: str,
    frame_names: List,
    img_size: int = 518,
    image_ext: str = ".jpg",
) -> torch.Tensor:
    """
    Load specific subset frames by their stem name into a float tensor
    [S, 3, img_size, img_size] in [0, 1]. Mirrors ScanNetSingleSceneDataset's image
    loading; used to rehydrate `--checkpoint_light` bundles (which store frame names +
    the scene path instead of the pixels) at visualization/demo time.
    """
    scene_dir = Path(scene_dir)
    images_dir = None
    for cand in ("subset", "images", "color"):
        if (scene_dir / cand).exists():
            images_dir = scene_dir / cand
            break
    if images_dir is None:
        raise ValueError(f"Images directory not found under {scene_dir}")

    imgs = []
    for name in frame_names:
        # Collation may wrap each name in a 1-element list (batch_size=1).
        if isinstance(name, (list, tuple)):
            name = name[0]
        path = images_dir / f"{name}{image_ext}"
        img = Image.open(path).convert("RGB").resize((img_size, img_size), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        imgs.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(imgs, dim=0)  # [S, 3, H, W]


def decode_checkpoint_images(
    scene: Dict,
    scans_root: Optional[str] = None,
    img_size: int = 518,
) -> torch.Tensor:
    """
    Return a scene's frames as a float tensor [1, S, 3, H, W] in [0, 1], handling all three
    checkpoint storage formats:
      - float images (legacy)            → passed through;
      - uint8 images (compact, 4× smaller) → divided by 255;
      - no images (`--checkpoint_light`)   → reloaded from disk via `scene_dir`/`frame_names`
        (falling back to `<scans_root>/<name>/raw_data` when no explicit path was stored).
    """
    imgs = scene.get("images")
    if imgs is not None:
        return imgs.float() / 255.0 if imgs.dtype == torch.uint8 else imgs

    frame_names = scene.get("frame_names")
    if frame_names is None:
        raise ValueError("Light checkpoint scene has no frame_names to reload images from")
    scene_dir = scene.get("scene_dir")
    if scene_dir is None:
        if scans_root is None:
            raise ValueError("Light checkpoint needs --scans_root (no stored scene_dir)")
        scene_dir = str(Path(scans_root) / scene["name"] / "raw_data")
    frames = load_frames_by_name(scene_dir, frame_names, img_size)  # [S, 3, H, W]
    return frames.unsqueeze(0)  # [1, S, 3, H, W]


class ScanNetSingleSceneDataset(Dataset):
    """
    Minimal ScanNet single-scene dataset for overfitting.

    Loads RGB images and corresponding per-class binary masks from a ScanNet scene folder.
    Masks are stored as uint8 PNGs (0 for background, 255 for foreground) in class-specific folders.

    Args:
        scene_dir (str): Path to scene folder containing 'images' and 'masks' subfolders
        num_frames (int): Number of frames to load (randomly sampled from available frames)
        image_ext (str): Image extension (default: '.jpg')
        mask_ext (str): Mask extension (default: '.png')
        img_size (int): Target image size for resizing (default: 518)
        frame_sampling (str): "random" samples num_frames frames anew on every __getitem__;
            "even" picks num_frames evenly-spaced frames (deterministic — required for a
            stable multi-scene overfit where the same frames must be revisited every epoch)
        instance_level (bool): if False (default), read per-class binary masks from `masks/`
            and assign one global ID per class. If True, read per-instance masks from
            `masks_instance/<class>_<k>/` and assign one global ID per (class, instance) — two
            objects of the same class then become distinct GT instances that share a class
            index (`classes` contains repeated class indices). Stuff classes (wall/floor) are
            single instances on disk, so they behave the same in both modes.

    Cross-view instance identity (item 8.3): each mask SEGMENT present in the scene is treated
    as ONE multi-view instance with a single global ID consistent across all sampled frames
    (e.g. a "wall" region keeps the same ID in every view it appears in), rather than minting a
    fresh ID for every (frame, segment) pair. In the default per-class mode a segment is a
    whole class, so class-level linking is the finest identity the *binary per-class* PNGs
    support; in `instance_level` mode a segment is one tracked instance, so same-class objects
    are separated (SAM3 video tracking provides the cross-frame identity). Each returned
    instance is described once (per-global-instance arrays below) but may occupy several frames
    in the `masks` map.

    Returns dict with:
        - images: torch.Tensor [num_frames, 3, img_size, img_size] in range [0, 1]
        - masks: torch.Tensor [num_frames, img_size, img_size] GLOBAL instance ID per pixel,
                 consistent across frames (0 = background, 1..G = instances)
        - classes: torch.Tensor [num_instances] class label of each global instance (1-19)
        - coordinates: torch.Tensor [num_instances, 2] (u, v) centroid in the instance's
                 representative (largest-area) frame
        - frame_ids: torch.Tensor [num_instances] representative frame index of each instance
        - instance_ids: torch.Tensor [num_instances] the global ID used in `masks` (1..G)
        - frame_names, num_instances: bookkeeping (num_instances == G global instances)
    """

    def __init__(
        self,
        scene_dir: str,
        num_frames: int = 8,
        image_ext: str = ".jpg",
        mask_ext: str = ".png",
        img_size: int = 518,
        images_subdir: Optional[str] = None,
        frame_sampling: str = "random",
        instance_level: bool = False,
    ):
        super().__init__()
        self.scene_dir = Path(scene_dir)
        self.num_frames = num_frames
        self.image_ext = image_ext
        self.mask_ext = mask_ext
        self.img_size = img_size
        self.instance_level = instance_level
        if frame_sampling not in ("random", "even"):
            raise ValueError(f"frame_sampling must be 'random' or 'even', got {frame_sampling!r}")
        self.frame_sampling = frame_sampling

        # Locate the image directory.
        # IMPORTANT: masks are only computed for the subsampled set of frames (e.g. a
        # stride-5 subset of a >5000-frame scene). 'color' holds *all* raw frames, most of
        # which have no corresponding mask. We therefore prefer the 'subset' folder (the
        # masked frames) and only fall back to 'images'/'color' if it is absent.
        if images_subdir is not None:
            candidates = [images_subdir]
        else:
            candidates = ["subset", "images", "color"]

        self.images_dir = None
        for cand in candidates:
            if (self.scene_dir / cand).exists():
                self.images_dir = self.scene_dir / cand
                break
        if self.images_dir is None:
            raise ValueError(
                f"Images directory not found (tried {candidates}): {self.scene_dir}"
            )

        masks_dirname = "masks_instance" if instance_level else "masks"
        self.masks_dir = self.scene_dir / masks_dirname
        if not self.masks_dir.exists():
            raise ValueError(f"Masks directory not found: {self.masks_dir}")

        # Find all image files
        self.image_files = sorted([
            f for f in self.images_dir.iterdir()
            if f.suffix.lower() == image_ext.lower()
        ])

        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.images_dir}")

        # Build the list of mask SEGMENTS — each segment becomes one global instance.
        # A segment is (canonical_class_name, segment_dir); the list is sorted into a
        # deterministic order so the same scene always yields the same (global_id -> class)
        # mapping. On-disk folders may use underscores (e.g. 'shower_curtain') while the
        # canonical class name uses a space ('shower curtain'); accept either.
        #   - per-class mode (default):  one segment per class folder in masks/.
        #   - instance mode:             one segment per masks_instance/<class>_<k>/ folder,
        #                                so two objects of the same class are distinct GT
        #                                instances that share a class index.
        if instance_level:
            # Map both spelling variants of every class name to the canonical form.
            norm_to_canon = {}
            for cls_name in SCANNET_CLASSES:
                norm_to_canon[cls_name] = cls_name
                norm_to_canon[cls_name.replace(" ", "_")] = cls_name
            parsed = []  # (class_idx, k, canonical_class_name, dir)
            for d in sorted(self.masks_dir.iterdir()):
                # Folders are '<class>_<k>'; <class> may itself contain underscores and
                # <k> is a trailing integer. Skip QA/metadata dirs (e.g. '_qa').
                if not d.is_dir() or d.name.startswith("_") or "_" not in d.name:
                    continue
                class_part, k_part = d.name.rsplit("_", 1)
                if not k_part.isdigit():
                    continue
                cls_name = norm_to_canon.get(class_part)
                if cls_name is None:
                    continue
                parsed.append((CLASS_TO_IDX[cls_name], int(k_part), cls_name, d))
            parsed.sort(key=lambda t: (t[0], t[1]))  # class index, then instance index k
            self.segments = [(cls_name, d) for (_, _, cls_name, d) in parsed]
        else:
            class_dirs = {}
            for cls_name in SCANNET_CLASSES:
                for cand in (cls_name, cls_name.replace(" ", "_")):
                    cand_dir = self.masks_dir / cand
                    if cand_dir.exists():
                        class_dirs[cls_name] = cand_dir
                        break
            self.class_dirs = class_dirs  # kept for backward compatibility/inspection
            self.segments = [
                (c, class_dirs[c]) for c in sorted(class_dirs, key=lambda c: CLASS_TO_IDX[c])
            ]

        if not self.segments:
            kind = "instance" if instance_level else "class"
            raise ValueError(f"No {kind} mask folders found in {self.masks_dir}")

    def __len__(self):
        return 1  # Single scene dataset - always returns 1 sample

    def __getitem__(self, idx):
        k = min(self.num_frames, len(self.image_files))
        if self.frame_sampling == "even":
            # Deterministic, evenly-spaced frames spanning the scene (stable across epochs).
            sampled_indices = np.unique(
                np.linspace(0, len(self.image_files) - 1, k).round().astype(int)
            ).tolist()
        else:
            sampled_indices = random.sample(range(len(self.image_files)), k)
            sampled_indices.sort()

        sampled_images = [self.image_files[i] for i in sampled_indices]
        frame_names = [f.stem for f in sampled_images]

        # Load images
        images = []
        for img_path in sampled_images:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
            img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # [3, H, W]
            images.append(img_tensor)

        images = torch.stack(images, dim=0)  # [num_frames, 3, H, W]

        num_frames = len(frame_names)

        # --- Pass 1: load every per-frame, per-segment binary mask ---------------------------
        # Collect, for each segment that has foreground in ANY sampled frame, the set of
        # frames it appears in and its binary pixel mask there. A segment is one class folder
        # (per-class mode) or one instance folder (instance mode); either way it yields a
        # SINGLE global ID consistent across views (cross-view identity, item 8.3) rather than
        # a fresh ID per (frame, segment) pair.
        per_seg_frame_pixels: Dict[int, Dict[int, np.ndarray]] = {}

        for frame_idx, frame_name in enumerate(frame_names):
            for seg_idx, (class_name, seg_dir) in enumerate(self.segments):
                mask_path = seg_dir / f"{frame_name}{self.mask_ext}"
                if not mask_path.exists():
                    continue

                class_mask = Image.open(mask_path).convert("L")
                class_mask = class_mask.resize((self.img_size, self.img_size), Image.NEAREST)
                class_mask_array = np.array(class_mask, dtype=np.uint8)

                # The on-disk masks are binary (one blob per segment per frame).
                if class_mask_array.max() == 0:
                    continue
                class_pixels = class_mask_array > 127  # Threshold at 127
                if not class_pixels.any():
                    continue

                per_seg_frame_pixels.setdefault(seg_idx, {})[frame_idx] = class_pixels

        # --- Pass 2: assign global instance IDs and paint the per-frame instance maps --------
        # Deterministic ID order: segments are already ordered (class index, then instance k),
        # so iterating present segments in sorted seg_idx order gives a stable
        # (instance_id -> class) mapping across runs.
        present_segs = sorted(per_seg_frame_pixels.keys())

        # int32 (not uint8) so the global instance IDs cannot overflow if many segments appear.
        instance_masks = np.zeros((num_frames, self.img_size, self.img_size), dtype=np.int32)

        instance_classes = []
        instance_coords = []   # representative (largest-area frame) centroid per instance
        instance_frames = []   # representative frame index per instance
        instance_ids = []      # the global ID written into `instance_masks` (1..G)

        for global_id, seg_idx in enumerate(present_segs, start=1):
            class_name = self.segments[seg_idx][0]
            frame_pixels = per_seg_frame_pixels[seg_idx]

            best_frame, best_area, best_centroid = -1, -1, (0.5, 0.5)
            for frame_idx, class_pixels in frame_pixels.items():
                # Paint the SAME global ID into every frame this instance appears in.
                # In instance mode same-class instances keep distinct IDs; later painted
                # segments win on cross-class pixel overlaps (matches per-class behavior).
                instance_masks[frame_idx][class_pixels] = global_id

                # Track the most-visible frame for the representative query point/centroid.
                area = int(class_pixels.sum())
                if area > best_area:
                    best_area = area
                    best_frame = frame_idx
                    best_centroid = self._get_centroid(class_pixels)

            instance_classes.append(CLASS_TO_IDX[class_name])
            instance_coords.append(best_centroid)
            instance_frames.append(best_frame)
            instance_ids.append(global_id)

        instance_masks = torch.from_numpy(instance_masks)  # [num_frames, H, W]

        # Convert to tensors. The i-th instance (0-indexed) has global instance-id (i + 1) in
        # `masks` across ALL frames it appears in; `classes[i]` is its class, `coordinates[i]`
        # and `frame_ids[i]` describe its representative (largest-area) view.
        classes = torch.tensor(instance_classes, dtype=torch.long) if instance_classes else torch.zeros(0, dtype=torch.long)
        coordinates = torch.tensor(instance_coords, dtype=torch.float32) if instance_coords else torch.zeros((0, 2), dtype=torch.float32)
        frame_ids = torch.tensor(instance_frames, dtype=torch.long) if instance_frames else torch.zeros(0, dtype=torch.long)
        instance_ids_t = torch.tensor(instance_ids, dtype=torch.long) if instance_ids else torch.zeros(0, dtype=torch.long)

        return {
            "images": images,
            "masks": instance_masks,
            "classes": classes,
            "coordinates": coordinates,
            "frame_ids": frame_ids,
            "instance_ids": instance_ids_t,
            "frame_names": frame_names,
            "num_instances": len(instance_classes),
        }

    @staticmethod
    def _get_centroid(mask: np.ndarray) -> Tuple[float, float]:
        """
        Compute (u, v) centroid of a binary mask in normalized coordinates.

        Args:
            mask: Binary numpy array [H, W]

        Returns:
            (u, v) tuple in normalized coordinates [0, 1]
        """
        if not mask.any():
            return (0.5, 0.5)

        coords = np.argwhere(mask)  # [N, 2] in (row, col) format
        centroid_row = coords[:, 0].mean()
        centroid_col = coords[:, 1].mean()

        H, W = mask.shape
        u = centroid_col / (W - 1)  # Normalize to [0, 1]
        v = centroid_row / (H - 1)

        return (float(u), float(v))


class ScanNetMultiSceneDataset(Dataset):
    """
    Multi-scene wrapper (item 8.7): one item per scene, each loaded by its own
    ScanNetSingleSceneDataset. Per-scene instance counts differ, so use batch_size=1
    (or a custom collate_fn) and let the batch-aware D4RTLoss match per sample.

    Args:
        scene_dirs: list of scene directories (each as accepted by ScanNetSingleSceneDataset)
        **kwargs: forwarded to every ScanNetSingleSceneDataset (num_frames, img_size,
            frame_sampling, ...)
    """

    def __init__(self, scene_dirs: List[str], **kwargs):
        super().__init__()
        if not scene_dirs:
            raise ValueError("scene_dirs must contain at least one scene directory")
        self.scenes = [ScanNetSingleSceneDataset(str(d), **kwargs) for d in scene_dirs]
        # Human-readable scene names: the scene folder, not the trailing 'raw_data'.
        self.scene_names = []
        for d in scene_dirs:
            p = Path(d)
            self.scene_names.append(p.parent.name if p.name == "raw_data" else p.name)

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        sample = self.scenes[idx][0]
        sample["scene_name"] = self.scene_names[idx]
        sample["scene_idx"] = idx
        return sample
```
