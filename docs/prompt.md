# SYSTEM OBJECTIVE
You are an expert PyTorch developer and 3D Computer Vision researcher. You are operating inside a local clone of the VGGT (Visual Geometry Grounded Transformer) repository. 
Our goal is to build a novel D4RT-style / DETR-like cross-attention decoder for Multi-View Instance Segmentation, attaching it to the global feature output of the VGGT backbone.

# CONSTRAINTS & WORKFLOW
1. **No assumptions:** Do not assume you know the exact file structure or variable names of this specific VGGT repo. You must explore the codebase to find where to hook the new decoder.
2. **Incremental & Testable:** You must execute this plan strictly phase-by-phase. Do not proceed to Phase N+1 until Phase N is fully implemented and validated with a standalone test script.
3. **Simplicity over Optimization:** For now, focus on clear, readable, and functional code. Do not implement complex multi-GPU distributed training or heavy optimizations. We need a working prototype first.

---

## PHASE 1: Repository Exploration & Hook Identification
**Goal:** Understand the VGGT forward pass and find the exact tensor that represents the Global Scene Features (which we will call $F$) to use as the `memory` for our cross-attention decoder.

**Actions:**
1. Read the `README.md` (if present) to understand the basic entry points.
2. Search the codebase for the main VGGT model definition (e.g., look for files like `models/vggt.py`, `network.py`, or similar).
3. Trace the forward pass of the model. Identify the Multi-View Transformer or the global self-attention module that fuses information across views.
4. **Output:** Create a brief text file `HOOK_PLAN.md` detailing exactly which class/file you will modify to extract $F$, and the expected shape of $F$ (e.g., `[Batch, Tokens, Dim]`).

**Validation:** Show me the contents of `HOOK_PLAN.md` before proceeding.

---

## PHASE 2: Minimal ScanNet Dataset Loader
**Goal:** Build a simple, overfit-ready PyTorch `Dataset` for a single ScanNet scene.

**Context:** The dataset consists of a single folder containing 5000+ `.jpg` RGB images and their corresponding SAM3 pseudo-label masks (PNGs or arrays) mapped to 19 valid ScanNet classes + background.

**Actions:**
1. Create a new file `data/scannet_overfit.py`.
2. Implement `ScanNetSingleSceneDataset(Dataset)`. It must:
   - Load $N$ randomly sampled RGB frames from the scene.
   - Load the corresponding SAM3 instance masks and class labels.
   - Return a dictionary containing: `images` (tensor), `masks` (tensor), `classes` (tensor), and valid image coordinates $(u, v)$ for mask boundaries/points.
3. **Validation:** Write a small script `test_phase2.py` that instantiates the dataset, loads `batch_size=2`, and prints the tensor shapes. Run it and ensure it works without crashing.

---

## PHASE 3: The D4RT Query Generator
**Goal:** Build the module that generates the queries for our decoder using continuous coordinates and local RGB patches.

**Actions:**
1. Create a new file `models/d4rt_decoder.py`.
2. Implement `class QueryGenerator(nn.Module)`. It must take as input:
   - Continuous normalized coordinates $(u, v)$ in $[0, 1]$.
   - View IDs (integer indices).
   - The original high-resolution RGB images.
3. The forward pass must compute:
   - **Fourier Positional Encoding:** Convert $(u, v)$ into high-frequency embeddings.
   - **View Embedding:** `nn.Embedding(num_views, hidden_dim)`.
   - **Local RGB Patch:** Use `torch.nn.functional.grid_sample` to extract a $9\times9$ RGB patch around $(u, v)$ from the original images, flatten it, and pass it through a small MLP.
   - Sum these three components to output a tensor of queries `[Batch, Num_Queries, Hidden_Dim]`.
4. **Validation:** Write `test_phase3.py` that passes dummy coordinates and dummy RGB images to `QueryGenerator` and asserts the output shape matches the expected query dimensions.

---

## PHASE 4: The DETR-like Cross-Attention Decoder
**Goal:** Build the Transformer Decoder and the Output Heads.

**Actions:**
1. In `models/d4rt_decoder.py`, add `class InstanceDecoder(nn.Module)`.
2. Instantiate a standard `torch.nn.TransformerDecoder` (e.g., 4 layers, 8 heads).
3. The forward pass takes the queries (from Phase 3) as `tgt` and the global VGGT features $F$ (identified in Phase 1) as `memory`.
4. Add two parallel MLP output heads on top of the decoded features:
   - `class_head`: Outputs logits for 19 classes + 1 background.
   - `mask_embed_head`: Outputs a latent vector to generate/match the instance mask.
5. **Validation:** Write `test_phase4.py` passing dummy VGGT features and dummy queries into the `InstanceDecoder`. Print the output shapes of the class logits and mask embeddings to verify they are correct.

---

## PHASE 5: Loss Formulation (Bipartite Matching)
**Goal:** Implement the Hungarian matching between sparse point predictions and SAM3 pseudo-labels.

**Actions:**
1. Create `train/loss.py`.
2. Look into popular DETR implementations (or standard `scipy.optimize.linear_sum_assignment`) to implement a `PointBipartiteMatcher`.
3. Implement the combined loss:
   - Match predicted queries to Ground Truth SAM3 instances based on class probability and point location/mask embedding.
   - Compute Cross-Entropy for classes.
   - Compute Focal/Dice Loss for the matched masks.
4. **Validation:** Write `test_phase5.py`. Create a fake prediction tensor and a fake SAM3 ground truth target. Run the matching and the loss computation. Assert that the loss is a valid, non-NaN scalar and gradients can flow backward.

---

## PHASE 6: Minimal Overfit Training Loop
**Goal:** Tie everything together in a simple script to overfit a few frames.

**Actions:**
1. Create `train_overfit.py`.
2. Initialize the VGGT backbone (frozen except for the final layers if necessary).
3. Initialize the `QueryGenerator` and `InstanceDecoder`.
4. Load the `ScanNetSingleSceneDataset` (restrict it to just 4 or 8 frames to force overfitting).
5. Write a simple PyTorch training loop (AdamW optimizer, zero_grad, forward, loss.backward, step).
6. Log the loss to the console every 10 iterations.
7. **Validation:** Run `python train_overfit.py` for 500 epochs. The loss must decrease significantly, proving that the gradients flow correctly from the SAM3 pseudo-labels, through the matching, through the D4RT queries, and back to the decoder.

Stop and wait for my confirmation after you complete Phase 1.