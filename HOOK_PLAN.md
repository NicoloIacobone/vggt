# HOOK PLAN: D4RT Decoder Integration with VGGT

## Phase 1 Analysis Summary

### Repository Structure
- **Main Model**: `vggt/models/vggt.py` (VGGT class)
- **Aggregator (Backbone)**: `vggt/models/aggregator.py` (Aggregator class)
- **Heads**: `vggt/heads/` (camera_head.py, dpt_head.py, track_head.py)
- **Existing Training Code**: `training/` directory with data loaders and utilities

### VGGT Forward Pass Flow
```
VGGT.forward(images: [B, S, 3, H, W])
  ↓
Aggregator.forward(images)
  - Patch embed: images → [B*S, P, C]
  - Frame attention: intra-frame processing (B*S, P, C)
  - Global attention: inter-frame fusion, reshape to (B, S*P, C)
  - Output: List of cached tensors at layer indices [4, 11, 17, 23]
  ↓
aggregated_tokens_list[-1]  [shape: B, S, P, 2C]
  ↓
Passed to heads (camera_head, depth_head, point_head, track_head)
```

## Global Scene Features (F) Identification

### **Source of Global Features**
- **Class/File**: `Aggregator` in `vggt/models/aggregator.py`
- **Extraction Point**: `aggregated_tokens_list[-1]` (final cached layer output, index=23)
- **Location in Code**: 
  - Generated at line 258-261 in `aggregator.py`:
    ```python
    concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
    output_list.append(concat_inter)
    ```
  - Accessible in `vggt.py` line 61 after `aggregator(images)` call

### **Shape & Semantics of F**
| Property | Value | Notes |
|----------|-------|-------|
| **Shape** | `[B, S, P, 2C]` | B: batch size, S: num frames, P: patch tokens + special tokens, 2C: concatenated features |
| **B (Batch)** | Variable | Typically 1 for single scene |
| **S (Frames)** | Variable | Number of input views |
| **P (Patches)** | 1370+ | `(H/patch_size)² + 1 (camera) + num_register` = (518/14)² + 1 + 4 ≈ 1370 |
| **2C (Dims)** | 2048 | `2 × embed_dim = 2 × 1024 = 2048` (frame features + global features concatenated) |

### **Semantic Meaning**
- Each element `F[b, s, p, :]` is a 2048-dim token representing:
  - **Frame-local features**: From within-frame self-attention (1024-dim)
  - **Global/cross-view features**: From global cross-frame attention (1024-dim)
- These tokens encode rich 3D geometric information about the scene visible from all frames
- Special tokens:
  - Index 0: Camera token (learned)
  - Indices 1-4: Register tokens (learned)
  - Indices 5+: Patch tokens (from DINOv2 ViT)
- Patch tokens map to spatial image locations: can recover (u, v) pixel coordinates via `patch_start_idx` and patch geometry

## Integration Hook Point

### **Where to Insert D4RT Decoder**

**File**: `vggt/models/vggt.py` (VGGT class)

**Method**: `forward()` (line 29-96)

**Hook Location**: After line 61:
```python
aggregated_tokens_list, patch_start_idx = self.aggregator(images)

# [HOOK POINT] Insert D4RT Decoder here
# - Input: aggregated_tokens_list[-1] as memory for cross-attention
# - Output: instance segmentation predictions
```

**Integration Strategy**:
1. Initialize D4RT decoder as module in `__init__`
   ```python
   self.instance_decoder = InstanceDecoder(...)
   ```

2. In `forward()`, after aggregator call:
   ```python
   # Extract global scene features F
   global_features = aggregated_tokens_list[-1]  # [B, S, P, 2C]
   
   # Generate queries for instance decoder
   # (from pseudo-labels or sampled points)
   queries = self.query_generator(...)  # [B, Num_Queries, Hidden_Dim]
   
   # Cross-attention decoder
   if self.instance_decoder is not None:
       instance_logits, mask_embeddings = self.instance_decoder(
           queries, 
           global_features,  # as memory
           images,
           patch_start_idx
       )
       predictions["instance_logits"] = instance_logits
       predictions["mask_embeddings"] = mask_embeddings
   ```

### **Why This Hook Point is Optimal**
- Global features F are already computed and cached
- No need to modify Aggregator (the computationally expensive backbone)
- Queries can be generated from SAM3 pseudo-labels independently
- Maintains modularity: decoder can be added/removed without affecting VGGT backbone
- Can optionally freeze VGGT backbone and only train decoder + query generator

## Memory Format Considerations

### **For Cross-Attention Memory**
The global features `F: [B, S, P, 2C]` need reshaping depending on decoder type:

**Option A: Per-Frame Memory** (recommended for multi-view consistency)
```python
F_reshaped = global_features  # Keep as [B, S, P, 2C]
# Cross-attention decoder processes each view's memory separately
# Allows view-aware attention
```

**Option B: Flattened Global Memory**
```python
F_flattened = global_features.view(B, S*P, 2*embed_dim)  # [B, S*P, 2048]
# Treats all patches from all views as one global token sequence
# Maximum information fusion but loses frame structure
```

**Option C: Projected Memory** (if dimension mismatch)
```python
# Project 2C → C if decoder expects different dimensionality
# e.g., C = 256 for smaller decoder
memory_projection = nn.Linear(2*embed_dim, hidden_dim)
F_projected = memory_projection(global_features)
```

## Implementation Roadmap

### Phase 2: ScanNet Dataset Loader
- Location: `data/scannet_overfit.py`
- Needs: SAM3 pseudo-label masks (19 ScanNet classes + background)
- Return: images, masks, class labels, point coordinates

### Phase 3: Query Generator
- Location: `models/d4rt_decoder.py` (new file)
- Inputs: continuous coords (u,v), view IDs, RGB images
- Process: Fourier positional encoding → view embedding → RGB patch via grid_sample → concat → queries

### Phase 4: InstanceDecoder
- Location: `models/d4rt_decoder.py` (same file as QueryGenerator)
- Architecture: TransformerDecoder + output heads
- Inputs: queries, global_features (F), images
- Outputs: class logits [B, Num_Queries, 20], mask embeddings [B, Num_Queries, Hidden_Dim]

### Phase 5: Loss & Matching
- Location: `train/loss.py` (new file)
- Hungarian matching between sparse predictions and SAM3 instances
- Combined loss: CE (classes) + Focal/Dice (masks)

### Phase 6: Overfit Training
- Location: `train_overfit.py` (new file)
- Training loop with VGGT backbone (frozen or partially frozen) + decoder (trainable)
- Validate on 4-8 ScanNet frames until loss decreases significantly

---

## Summary for Phase 1 Validation

✅ **Repository explored**: VGGT backbone + heads architecture understood
✅ **Global features identified**: `aggregated_tokens_list[-1]` with shape `[B, S, P, 2048]`
✅ **Hook point determined**: VGGT.forward() after aggregator, before heads
✅ **Memory semantics clarified**: Multi-view fused 3D geometric features
✅ **Integration strategy defined**: Modular decoder attachment without backbone modification

**Ready to proceed to Phase 2: ScanNet Dataset Loader**
