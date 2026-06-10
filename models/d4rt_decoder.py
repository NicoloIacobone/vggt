# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


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
    ):
        super().__init__()
        self.num_views = num_views
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size

        # Fourier positional encoding
        self.pos_encoder = FourierPositionalEncoding(num_freqs=num_freqs, max_freq=max_freq)
        pos_encoding_dim = 4 * num_freqs  # sin and cos for u and v

        # View embeddings
        self.view_embedding = nn.Embedding(num_views, hidden_dim)

        # Local RGB patch feature extraction
        self.patch_extractor = LocalPatchFeatureExtractor(
            patch_size=patch_size, hidden_dim=hidden_dim, in_channels=3
        )

        # Projection layers to combine features
        self.pos_proj = nn.Linear(pos_encoding_dim, hidden_dim)
        self.view_proj = nn.Linear(hidden_dim, hidden_dim)
        self.patch_proj = nn.Linear(hidden_dim, hidden_dim)

        # Final query projection
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        coordinates: torch.Tensor,
        view_ids: torch.Tensor,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate queries for instance decoder.

        Args:
            coordinates (torch.Tensor): Normalized (u, v) coordinates with shape [B, N, 2] in [0, 1]
            view_ids (torch.Tensor): View indices with shape [B, N] in [0, num_views)
            images (torch.Tensor): Input images with shape [B, S, 3, H, W] in [0, 1]

        Returns:
            torch.Tensor: Query embeddings with shape [B, N, hidden_dim]
        """
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
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.mask_embed_dim = mask_embed_dim

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
        pixel_feats = self.mask_feature_proj(patch_tokens)            # [B, S, h*w, mask_embed_dim]
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
    ):
        super().__init__()
        self.query_generator = QueryGenerator(
            num_views=num_views,
            hidden_dim=hidden_dim,
            patch_size=patch_size,
        )
        self.instance_decoder = InstanceDecoder(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_decoder_layers=num_decoder_layers,
            mask_embed_dim=mask_embed_dim,
            memory_dim=memory_dim,
            dropout=dropout,
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
