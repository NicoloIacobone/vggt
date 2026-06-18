# Milestone 1 — D4RT Multi-View Instance Segmentation Prototype on VGGT

**Status:** ✅ Complete — end-to-end pipeline implemented and validated, **including multi-scene
training**: the model predicts **dense multi-view instance masks**, trains on 4 ScanNet scenes
simultaneously (mean mIoU **0.967**, class_acc 0.94), and is evaluated on a 5th scene it never
saw (see §9). All open issues of §8 are resolved.
**Goal:** Attach a novel D4RT-style / DETR-like cross-attention decoder for multi-view
instance segmentation to the global feature output of the frozen VGGT backbone, and prove
that gradients flow correctly from (pseudo-)labels back through the decoder.

This document is the first project milestone. It describes what was built in each of the
six phases, the design decisions taken, the validation performed, and the open issues that
should be addressed first in the next iteration.

---

## 0. Overview

The system reuses the pretrained **VGGT-1B** backbone (frozen) as a multi-view feature
extractor and adds a lightweight, trainable instance-segmentation head on top:

```
Images [B, S, 3, H, W]
        │
        ▼
VGGT Aggregator (frozen, alternating frame/global attention)
        │  aggregated_tokens_list[-1]
        ▼
Global Scene Features  F : [B, S, P, 2048]     ← cross-attention "memory"
        │
        │   coordinates (u,v), view_ids, images
        ▼
QueryGenerator ───────────────► Queries [B, N, 256]   ← cross-attention "tgt"
        │
        ▼
InstanceDecoder (TransformerDecoder, 4 layers × 8 heads; normed memory + query skip)
        ├──► class_head      → class_logits   [B, N, 20]
        └──► mask_embed_head → mask_embeddings [B, N, 256]  (per-query mask kernels)
                  │  cosine · pixel features (from VGGT patch tokens)
                  ▼
              pred_masks [B, N, S, h, w]   (dense per-frame mask logits, Mask2Former-style)
        │
        ▼
PointBipartiteMatcher (Hungarian, mask-aware Dice+BCE cost) → matched (pred, gt) pairs
        │
        ▼
D4RTLoss = w_cls·Focal + w_mask·(Dice + fg-weighted BCE)   [+ optional emb/coord terms]
        │
        ▼
EvalMetrics: mIoU / AP50 / AP75 / mAP / class_acc
        │
        ▼
backward() → updates QueryGenerator + InstanceDecoder (backbone frozen)
```

**Trainable parameters:** ~6.5M (decoder head, incl. the dense mask projection) on top of
~1.26B frozen backbone params.

### Files produced

| File | Purpose |
|------|---------|
| `docs/HOOK_PLAN.md` | Phase 1 analysis: where/how to hook into VGGT |
| `data/scannet_overfit.py` | Phase 2: single-scene ScanNet loader + `ScanNetMultiSceneDataset` (item 8.7) |
| `models/d4rt_decoder.py` | Phases 3 & 4: QueryGenerator + InstanceDecoder (+ dense mask head) + wrapper |
| `train/loss.py` | Phase 5: Focal/Dice/BCE losses, mask-aware Hungarian matcher, batch-aware `D4RTLoss` |
| `train/eval_metrics.py` | Item 8.4: instance-segmentation metrics (mIoU / AP50 / AP75 / mAP / class_acc) |
| `scripts/train_overfit.py` | Phase 6: end-to-end single-scene overfit loop + evaluation |
| `scripts/train_multiscene.py` | Item 8.7 / §9: multi-scene training + held-out-scene evaluation, LR schedule, checkpoint/resume |
| `tests/test_phase2.py … test_phase5.py`, `tests/test_eval.py` | Standalone validation scripts per phase + metrics |

---

## 1. Phase 1 — Repository Exploration & Hook Identification

**Objective:** Find the exact tensor in VGGT's forward pass that represents the fused
multi-view *Global Scene Features* (`F`) to use as cross-attention memory.

**Findings (see `HOOK_PLAN.md` for full detail):**

- The backbone is `vggt/models/aggregator.py::Aggregator`, wrapped by
  `vggt/models/vggt.py::VGGT`.
- The aggregator runs **alternating attention**: per-frame self-attention
  (`tokens` shaped `[B*S, P, C]`) interleaved with global cross-frame attention
  (`[B, S*P, C]`), for `depth=24` blocks.
- At cached layers it concatenates the frame and global intermediates along the channel
  dim (`torch.cat([frame, global], dim=-1)`), producing tokens of width `2C`.
- **Chosen hook:** `aggregated_tokens_list[-1]` (the last cached layer, index 23), with shape

  ```
  F : [B, S, P, 2C] = [B, S, P, 2048]
  ```

  where `C = embed_dim = 1024`, and `P` = patch tokens + special tokens
  (`(H/14)·(W/14) + 1 camera + 4 register`).

- **Integration point:** in `VGGT.forward`, immediately after
  `aggregated_tokens_list, patch_start_idx = self.aggregator(images)`. The new head is a
  separate module so the backbone is untouched and can stay frozen.

The first `patch_start_idx` tokens are the camera/register special tokens; patch tokens
(which map to spatial image locations) follow.

---

## 2. Phase 2 — Minimal ScanNet Dataset Loader

**File:** `data/scannet_overfit.py` → `ScanNetSingleSceneDataset(Dataset)`

**Expected on-disk layout** (matches the real data at
`/cluster/work/igp_psr/.../scene0000_00/raw_data`):

```
scene_dir/
├── color/                  # ALL raw frames (>5500) — NOT used for training
├── subset/                 # the ~100 stride-5 frames that actually have masks
│   ├── 00000.jpg
│   ├── 00005.jpg
│   └── ...
└── masks/
    ├── wall/      00000.png 00005.png ...   # uint8, 0 = bg, 255 = fg
    ├── floor/     ...
    ├── shower_curtain/  ...                  # folder names may use underscores
    └── <one folder per ScanNet class present>
```

**Important — frame subset.** The scene has >5500 frames in `color/`, but masks were only
computed for a stride-5 **subset** of ~100 frames (`00000, 00005, 00010, …`), stored in
`subset/`. Sampling from `color/` would pick frames with no corresponding mask (loaded as
empty/background). The loader therefore **prefers `subset/`** (search order
`subset → images → color`, overridable via the `images_subdir` argument).

**Behavior:**
- Discovers frames in `subset/` (then `images/`, then `color/` as fallbacks).
- Auto-detects which of the 19 ScanNet class folders exist under `masks/`, tolerating
  space-vs-underscore naming (`shower curtain` ↔ `shower_curtain`).
- Loads each class's binary PNG (thresholded at 127) in every sampled frame and assigns a
  **global, cross-view-consistent instance ID** per class (see §8.3): the same class keeps one
  ID across all frames, painted into a per-pixel integer **instance map** (`0` = background,
  `1..G` = global instances).
- Computes each instance's normalized **(u, v) centroid** in `[0, 1]` from its
  representative (largest-area) frame.

**Returns** (`__getitem__`):

| Key | Shape / type | Meaning |
|-----|--------------|---------|
| `images` | `[num_frames, 3, img_size, img_size]` float32 in `[0,1]` | RGB frames |
| `masks` | `[num_frames, img_size, img_size]` int32 | **global** instance-ID map per pixel, consistent across frames |
| `classes` | `[num_instances]` long, values `1..19` | class of each global instance |
| `coordinates` | `[num_instances, 2]` float32 | (u, v) centroid in the instance's representative frame |
| `frame_ids` | `[num_instances]` long | representative (largest-area) frame of each instance |
| `instance_ids` | `[num_instances]` long, values `1..G` | the global ID used in `masks` |
| `frame_names`, `num_instances` | — | bookkeeping (`num_instances == G`) |

The 19 ScanNet classes are mapped to indices `1..19`; index `0` is reserved for background.
`img_size` is set to **518** in training because VGGT requires the input to be divisible by
its patch size (14).

**Validation (`test_phase2.py`):** synthetic scene with 4 frames × 3 classes → correct shapes,
value ranges (`images∈[0,1]`, `classes∈[1,19]`, `coords∈[0,1]`), dtypes, **and cross-view
identity**: exactly 3 global instances (one per class, not 12 per-`(frame,class)`), the mask IDs
equal the returned `instance_ids`, and every instance ID appears in more than one frame.

---

## 3. Phase 3 — D4RT Query Generator

**File:** `models/d4rt_decoder.py`

Three sub-modules combine into one query per point:

1. **`FourierPositionalEncoding`** — encodes `(u, v)` with `num_freqs=16` log-spaced
   frequencies using sin & cos for each coordinate → `4 · num_freqs = 64` dims.
2. **`LocalPatchFeatureExtractor`** — uses `F.grid_sample` to crop a **9×9 RGB patch**
   around each `(u, v)` from the correct view's image, flattens it (`3·9·9 = 243`), and
   passes it through a 2-layer MLP → `hidden_dim`.
3. **`nn.Embedding(num_views, hidden_dim)`** — a learned **view embedding** per frame index.

**`QueryGenerator.forward(coordinates, view_ids, images)`:**
each component is linearly projected to `hidden_dim=256`, **summed**, and passed through a
final projection:

```
queries = query_proj( pos_proj(fourier) + view_proj(view_emb) + patch_proj(rgb_patch) )
        → [B, N, 256]
```

**Validation (`test_phase3.py`):** correct output shape `[B, N, 256]`, no NaN/Inf, and
gradients flow back to both `images` and `coordinates`.

---

## 4. Phase 4 — DETR-like Cross-Attention Decoder

**File:** `models/d4rt_decoder.py`

**`InstanceDecoder`:**
- Projects VGGT memory `F` from `2048 → 256` (`memory_proj`) after flattening
  `[B, S, P, 2048] → [B, S·P, 2048]`, then **LayerNorm**s it (`memory_norm`).
- A standard `nn.TransformerDecoder` (**4 layers, 8 heads**, `batch_first=True`) with the
  queries as `tgt` and the normed memory as `memory`, followed by a **skip connection**
  `decoded = decoded + queries`.
- Heads on the decoded tokens:
  - **`class_head`** → `class_logits [B, N, 20]` (19 classes + background).
  - **`mask_embed_head`** → `mask_embeddings [B, N, 256]` (per-query **mask kernels**).
  - **Dense mask head (item 8.4):** a `mask_feature_proj` turns the VGGT patch tokens into a
    per-pixel feature map; each query's dense mask is the **cosine** similarity of its mask
    embedding with that map, scaled by a learnable temperature + bias →
    `pred_masks [B, N, S, h, w]` mask logits at the patch grid (`h=w=37`).

> Two design choices were essential to make the dense head trainable (found by diagnosis, §6):
> (1) **normalizing the memory** — raw VGGT features have a huge magnitude, so without the
> LayerNorm the cross-attention output dwarfs the query residual and every query collapses to
> the same memory average; (2) the **query skip connection** — the cross-attention still tends
> to collapse all queries to an identical output, and adding the queries back preserves each
> instance's identity. **Cosine** (not raw dot-product) mask logits keep the sigmoid from
> saturating on the large feature norms.

**`D4RTInstanceSegmentationHead`** is a convenience wrapper chaining
`QueryGenerator → InstanceDecoder`; `dropout` is plumbed through (0 for a clean overfit).

**Validation (`test_phase4.py`):** correct shapes (incl. `pred_masks [B,N,S,h,w]`), softmax
sums to 1, gradient flow to queries, memory, and the dense mask head, and robustness across
batch sizes `{1,2,4}`, query counts `{5,10,20}`, and memory dims `{1024,2048,4096}`.

---

## 5. Phase 5 — Loss Formulation (Bipartite Matching)

**File:** `train/loss.py`

- **`FocalLoss`** (α=0.25, γ=2.0) for class prediction (handles class imbalance).
- **`DiceLoss`** for dense mask supervision, now with an explicit `apply_sigmoid` flag
  (replacing the old `pred.max() > 1` heuristic) and arbitrary trailing dims so it works on the
  multi-view `[N, S, h, w]` masks.
- **`PointBipartiteMatcher`** — builds a cost matrix combining *class cost* `(1 − p_correct)` +
  *coordinate L2* + a **mask cost**, then runs `scipy.optimize.linear_sum_assignment`
  (Hungarian). The mask cost is the dense **Dice+BCE** cost (`batch_dice_cost`/`batch_bce_cost`,
  Mask2Former-style) when dense masks are supplied, otherwise it falls back to mask-embedding L2.
- **`D4RTLoss`** — runs the (mask-aware) matcher, gathers matched pairs, and returns the
  weighted sum of: Focal (class) + **Dice + foreground-weighted BCE** (dense mask), plus optional
  mask-embedding/coordinate terms, with a per-term breakdown. The BCE uses a `pos_weight`
  (capped neg/pos ratio) so the sparse instance masks don't collapse to empty.

**Validation (`test_phase5.py`):** individual losses valid; matcher returns optimal, unique
assignments; combined loss is a non-NaN scalar with working gradients; the **dense-mask path**
(mask-aware matching + Dice+BCE on `[N,S,h,w]`, gradients to mask logits, perfect masks → ~0
loss) is covered; edge cases (empty predictions / empty GT / perfect match → loss 0) handled.

---

## 6. Phase 6 — Minimal Overfit Training Loop

**File:** `train_overfit.py`

**`D4RTModel`** wraps the frozen VGGT backbone + `D4RTInstanceSegmentationHead`. The backbone
runs under `torch.no_grad()` when frozen; only the head's ~6.5M params are optimized with
**AdamW**.

**True overfit protocol.** A genuine overfit test must hold the inputs **and** targets
constant so the model has something stable to memorize. The loop therefore:
1. Seeds all RNGs.
2. Loads **one fixed batch**, generates query coordinates + view IDs, and builds the
   **dense GT targets once** (`build_gt_targets`: real classes/centroids + per-instance binary
   masks at the patch grid; see §8.4).
3. Repeats forward → loss → backward (grad-clipped) → step on that same batch for all epochs.
4. Reports instance-segmentation **metrics** before and after training (item 8.4 eval metric).

**Result (real scene `scene0000_00`, 4 frames, 16 queries / 11 cross-view instances, 400 epochs, lr 2e-3, dropout 0):**

```
Initial loss: 2.91
Final loss:   0.33
Reduction:    88.6%   → ✅ SUCCESS
matches:      11 / 11       (all cross-view instances matched every epoch)
class_loss:   0.69 → 0.00   (perfect classification of matched instances)
mask  (Dice+BCE): 2.22 → 0.33  (dense masks fit)

Instance-segmentation metrics (before → after):
  mIoU      0.004 → 0.900
  AP50      0.000 → 0.962
  AP75      0.000 → 0.876
  mAP       0.000 → 0.785
  class_acc 0.000 → 1.000
```

With dense mask supervision (§8.4) the model now predicts actual masks that overlap the GT
(mIoU 0.90, AP50 0.96) and classifies every matched instance correctly — a genuine, measurable
overfit, not just a falling loss. Gradients flow end-to-end from the dense mask + class loss,
through the decoder and query generator, while the VGGT backbone stays frozen.

> **Debugging note (why the dense head first failed).** With the naive dense head the loss
> plateaued and mIoU stayed 0. Step-by-step instrumentation showed the **decoder output
> collapsing**: distinct queries (std ≈ 2) produced *identical* decoded vectors (std ≈ 0), so
> all instances shared one mask. Cause: the un-normalized VGGT memory dominates the
> cross-attention residual stream. Fixes (all in `InstanceDecoder`): LayerNorm the projected
> memory, add a **query skip connection**, use **cosine** mask logits, plus a foreground
> `pos_weight` on the BCE and gradient clipping. Earlier the (now-removed) embedding-L2 proxy
> had masked this by supplying distinct per-query targets; the real mask loss exposed it.

---

## 7. How to Run

```bash
# Per-phase unit validation
python tests/test_phase2.py
python tests/test_phase3.py
python tests/test_phase4.py
python tests/test_phase5.py
python tests/test_eval.py     # instance-segmentation metrics

# Single-scene end-to-end overfit (uses the real scene by default)
python scripts/train_overfit.py \
    --num_epochs 400 \
    --num_frames 4 \
    --num_queries 16 \
    --learning_rate 2e-3 \
    --scene_dir /cluster/work/igp_psr/niacobone/distillation/dataset/scannet/scans/scene0000_00/raw_data

# Multi-scene training (train on 4 scenes, evaluate on a held-out 5th — see §9)
python scripts/train_multiscene.py \
    --train_scenes scene0000_00,scene0001_00,scene0002_00,scene0003_00 \
    --val_scenes scene0004_00 \
    --num_epochs 2000 --num_frames 8 --num_queries 32 --learning_rate 2e-3 \
    --save_checkpoint /cluster/work/igp_psr/niacobone/distillation/output/<run_name>/checkpoint.pth
```

Key flags (`train_overfit.py`): `--dropout` (0 by default for a clean overfit),
`--unfreeze_backbone`, `--save_checkpoint PATH`, `--device {cuda,cpu}`. If the scene directory
is unavailable the script falls back to an auto-generated synthetic scene.
Key flags (`train_multiscene.py`): `--warmup_epochs`, `--eval_interval`, `--resume CKPT`
(restores head + optimizer + scheduler), `--num_views` (view-embedding table size,
defaults to `max(num_frames, 10)`). Scene names are resolved under `--scans_root`.

To visualize a trained checkpoint in 3D: `python demos/demo_gradio.py --seg_checkpoint
<path>` (auto-discovers the latest checkpoint if omitted). With a multi-scene checkpoint a
dropdown lets you load any bundled scene — including the held-out validation scene — then
"Reconstruct" and set *Color By: Predicted Instances*.

---

## 8. Open Issues — all resolved

These were identified during review. All items have since been **completed**; the details of
each resolution are kept below for reference.

### 8.1 Replace synthetic targets with real SAM3 supervision — ✅ DONE
`create_dummy_gt` (random class/embedding/coordinate targets) was replaced by a target builder
in `train_overfit.py` that derives **all** targets from the data + frozen backbone:
- **classes** ← real ScanNet labels (`batch["classes"]`).
- **coordinates** ← real instance centroids (`batch["coordinates"]`).
- **masks** ← real per-instance GT masks (see §8.4; this superseded the original pooled-feature
  "prototype" descriptor once dense mask supervision was added).

Supporting changes: the dataset returns a per-instance `frame_ids`, instance-ID maps use
`int32` (no overflow), and each instance query takes the **view of its representative frame** (so
its view embedding and local RGB patch come from a frame where the instance is actually visible).
With this (and the cross-view fix of §8.3), all real instances are matched every epoch and the
overfit reaches ~89% loss reduction (§6).

### 8.2 Make `D4RTLoss` batch-aware — ✅ DONE
`D4RTLoss.forward` no longer flattens `[B, N, …] → [B·N, …]`: the Hungarian matcher now runs
**per batch sample** against that sample's own GT set (DETR-style) via an internal
`_forward_single`, and the loss components are averaged over the samples that have at least one
GT instance (`num_matches` is summed). For `B > 1` the GT arguments are **lists of per-sample
tensors** (instance counts may differ per sample); for `B == 1` or 2D predictions the old
single-tensor call signature still works, so all phase tests pass unchanged. Passing a single
GT tensor with `B > 1` now raises an explicit error instead of silently conflating samples.
Verified: the `B=2` batched loss equals the mean of the two single-sample losses.

### 8.3 Fix cross-view instance identity in the dataset — ✅ DONE
Previously every `(frame, class)` with foreground became a *separate* instance ID, so the same
physical object had different IDs across views — effectively per-frame **semantic** masks. The
dataset now links instances across frames: each ScanNet class present in the scene is assigned a
**single global instance ID, consistent across all sampled frames**, painted into the per-pixel
`masks` map, and described once via per-global-instance `classes` / `coordinates` / `frame_ids` /
`instance_ids`. The representative `(u, v)` centroid and query view come from the instance's
largest-area frame. The GT-target builder (`build_gt_targets`, §8.4) consumes this directly: an
instance's dense GT mask spans **every frame it appears in**, so the Dice/BCE supervision is
genuinely multi-view.

On `scene0000_00` (4 frames) this collapses 32 per-`(frame, class)` instances to **11 cross-view
instances**, 8 of which span more than one frame; the overfit matches all 11 every epoch and
reaches ~89% loss reduction with mIoU 0.90 (§6). `test_phase2.py` asserts the cross-view
invariants (one instance per class, mask IDs == `instance_ids`, each ID present in >1 frame).

*Limitation (by data, not code):* the on-disk masks are **binary per-class** PNGs and carry no
information to separate two distinct objects of the same class, so class-level linking is the
finest cross-view identity the labels support. Separating same-class objects (true instance IDs)
would require instance-encoded masks or geometric/temporal correspondence across views.

### 8.4 Wire up real mask prediction + correct Dice handling — ✅ DONE
The decoder now predicts **dense masks** (Mask2Former-style): a `mask_feature_proj` turns the
VGGT patch tokens into a per-pixel feature map, and each query's mask is the cosine similarity of
its mask embedding with that map (scaled by a learnable temperature) → `pred_masks [B,N,S,h,w]`
mask logits at the patch grid (§4). Supervision is **Dice + foreground-weighted BCE** on the
matched masks, the matcher is **mask-aware** (dense Dice+BCE cost), and `DiceLoss` now takes an
explicit `apply_sigmoid` flag (the `pred.max() > 1` heuristic is gone). `build_gt_targets`
produces the per-instance binary GT masks at the patch grid (multi-view, §8.3).

Making this train required fixing a decoder **output-collapse** (un-normalized VGGT memory
swamping the query residual) via memory LayerNorm + a query skip connection + cosine logits — see
the debugging note in §6. Result on `scene0000_00`: **mIoU 0.90, AP50 0.96, class_acc 1.0**.
Covered by `test_phase4.py` (dense shapes/grads), `test_phase5.py` (dense mask loss + matcher),
and `test_eval.py` (metrics). *Note:* masks/metrics are at the 37×37 patch resolution; upsampling
to full image resolution (e.g. an FPN-style pixel decoder) is future work.

**Eval metric (delivered alongside 8.4).** `train/eval_metrics.py` computes interpretable
instance-segmentation metrics — **mIoU, AP50, AP75, mAP (0.50:0.95), and class accuracy** — from
the dense mask + class logits vs. the GT, and `train_overfit.py` reports them before/after
training. This replaces "raw loss reduction" as the success signal (partially addresses §8.7).

### 8.5 Coordinate loss — ✅ DONE (resolved as "matching only")
Decision: **no coordinate-refinement head**; coordinates are query *prompts*, not predictions.
They participate in the Hungarian matcher cost (where they help disambiguate same-class
instances) but carry no loss term (`coord_loss_weight=0`); the reported `coord_loss` is a
diagnostic of the matched coordinate error only. `D4RTLoss`/the matcher now also tolerate
`gt_coordinates=None`. A learnable refinement head remains possible future work if predicted
point tracks are ever needed (true D4RT-style trajectory readout).

### 8.6 Robustness & performance — ✅ DONE
- **View embedding bound:** `QueryGenerator` now raises a clear `ValueError` when
  `view_ids >= num_views`, and `train_multiscene.py` sizes the table via `--num_views`
  (default `max(num_frames, 10)`). The head config is stored in the checkpoint and the demo
  rebuilds the head from it, so changed sizes stay loadable.
- **Vectorized patch extraction:** `LocalPatchFeatureExtractor` now gathers each query's
  source frame and runs **one** batched `grid_sample` over all `B·N` patches (verified
  numerically identical to the old loop).
- **`reshape` over `view`** on image/backbone tensors (no contiguity assumptions).

### 8.7 Training realism — ✅ DONE
Delivered by `scripts/train_multiscene.py` + `ScanNetMultiSceneDataset` (see §9 for results):
- **Multi-scene loading:** one fixed, deterministic bundle per scene
  (`frame_sampling="even"` picks evenly-spaced frames; seeded background queries).
- **Train/val split:** `--train_scenes` / `--val_scenes`; the val scene is never trained on.
- **LR schedule:** linear warmup → cosine decay (`--warmup_epochs`, floor at 5% of base LR).
- **Checkpointing/resume:** checkpoints carry head weights + head config + optimizer +
  scheduler + per-scene data/metrics; `--resume` restores all of them.
- **Efficiency:** the frozen backbone runs **once per scene up front** and its global features
  are cached, so every epoch only runs the ~6.5M-param head (2000 epochs × 4 scenes ≈ 4 min
  on one RTX 4090).
- *Not needed for this milestone:* partial backbone unfreezing (the head overfits the training
  scenes without it); revisit when scaling the dataset.

---

## 9. Multi-Scene Training Result (4 train scenes + 1 held-out scene)

**Run:** `scripts/train_multiscene.py`, 2000 epochs, 8 frames/scene (evenly spaced from the
100 masked `subset/` frames), 32 queries, lr 2e-3 (50-epoch warmup → cosine), dropout 0,
frozen backbone. Checkpoint:
`/cluster/work/igp_psr/niacobone/distillation/output/d4rt_multiscene_4train_20260610_123647/checkpoint.pth`
(bundles the trained head + all 5 scene batches; loadable by `demos/demo_gradio.py`).

Loss/scene fell **2.82 → 0.067 (-97.6%)** with all **37** cross-view training instances matched
every epoch.

| Scene | Split | mIoU | AP50 | AP75 | mAP | class_acc |
|-------|-------|------|------|------|-----|-----------|
| scene0000_00 | train | 0.949 | 0.463 | 0.463 | 0.413 | 1.000 |
| scene0001_00 | train | 0.971 | 0.570 | 0.570 | 0.535 | 1.000 |
| scene0002_00 | train | 0.967 | 0.539 | 0.539 | 0.487 | 0.909 |
| scene0003_00 | train | 0.982 | 0.597 | 0.597 | 0.592 | 0.833 |
| **mean (train)** | | **0.967** | **0.542** | | | **0.936** |
| scene0004_00 | **val (never seen)** | 0.027 | 0.000 | 0.000 | 0.000 | 0.286 |

**Reading the numbers.**
- The head fits **four scenes at once** almost perfectly (mIoU ≥ 0.95 per scene, near-perfect
  classification of matched instances), confirming that a single set of decoder weights can
  represent multiple scenes' instances simultaneously — the multi-scene pipeline works.
- Train AP50 (~0.54) is much lower than mIoU because AP also penalizes the extra background
  query points, which the head does not push to the background class (no "no-object"
  supervision is applied to unmatched queries yet — see below).
- **Generalization is not there yet, as expected with only 4 training scenes.** On the held-out
  scene the final checkpoint reaches mIoU 0.03 / class_acc 0.29. Interestingly, val mIoU
  *peaked* mid-training (~0.13 around epoch 600) and then decayed as the head kept memorizing
  the training scenes — classic overfitting, visible thanks to the held-out-scene evaluation.

**What this implies for Milestone 2:** to obtain real generalization the next iteration should
(a) scale the number of training scenes (tens-to-hundreds, not 4), (b) supervise unmatched
queries toward the background class (DETR's "no-object" loss) so AP becomes meaningful and
inference doesn't need GT-ordered queries, (c) add data augmentation / random frame sampling
as regularization instead of the fixed overfit batch, and (d) consider early stopping on val
mIoU (the mid-training peak shows the signal exists).
