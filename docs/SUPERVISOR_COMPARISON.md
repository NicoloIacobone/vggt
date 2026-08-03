# MaskDINO on frozen VGGT — results summary

**Task.** 3D multi-view-consistent instance segmentation on top of the **frozen VGGT-1B**
backbone. Only the decoder head is trained; supervision is the official ScanNet v2 2D instance
annotations (19 classes). Nothing in VGGT is modified or finetuned.

**Two things are reported below**, and they answer two different questions:

1. **MaskDINO vs arm C** — does the new head beat the best of the previous (hand-rolled DETR-style)
   heads on our ScanNet task? *This is the project result.*
2. **Our MaskDINO vs official MaskDINO on COCO** — is our re-implementation of MaskDINO faithful to
   upstream? *This is a correctness proof for the port, not a project result.*

---

## 1. Headline

| | previous best (arm C) | MaskDINO (best) | gain |
|---|---|---|---|
| single-frame mIoU | 0.451 | **0.694** | +54 % |
| single-frame AP50 | 0.294 | **0.729** | +148 % |
| multi-view mIoU | 0.367 | **0.539** | +47 % |
| multi-view AP50 | 0.199 | **0.515** | 2.6× |

Our MaskDINO port reproduces upstream MaskDINO on COCO val2017 to **0.004 AP**.

Since 2026-08-02 the same recipes are also trained on the **full official ScanNet v2 1201/312
split** (a separate, harder ruler — do not mix with the table above): single-frame **0.624 mIoU
/ 0.662 AP50** per-frame, multi-frame **0.529 / 0.525** per-bundle, plus the first cross-view
consistency read-out (0.717). Details: `docs/RESULTS.md` §6, `docs/MASKDINO.md` §7.8.

**And since 2026-08-03 there is a number that IS comparable to the literature** — see §5.

---

## 2. MaskDINO vs arm C

Arm C = learned DETR object queries, the best of the five previous query-initialisation variants
(A–E), now retired. Same frozen backbone, same data, same metric code.

### 2.1 Single-frame protocol (the primary comparison)

Each frame is scored separately; per-scene averages are then averaged over scenes.
All runs: official ScanNet GT, val = scenes 0080–0089, held out of every training set.

| Model | Train scenes | mIoU | AP50 | AP75 | mAP |
|---|---|---|---|---|---|
| **arm C — the bar** | 190 | 0.451 | 0.294 | 0.141 | 0.154 |
| MaskDINO | 50 | 0.451 | 0.440 | 0.314 | 0.290 |
| MaskDINO | 190 | 0.594 | 0.624 | 0.440 | 0.418 |
| MaskDINO | 490 | 0.669 | 0.699 | 0.506 | 0.475 |
| **MaskDINO + 2 view-draws/scene + colour jitter** | **490** | **0.694** | **0.729** | **0.582** | **0.526** |

Notes worth stating:

- Arm C's per-frame numbers were obtained by re-scoring its released checkpoint through the
  *identical* protocol (`scripts/eval_perframe.py`), not by quoting its old numbers on a
  different ruler.
- **Data scale dominates every architectural ingredient.** Going 50 → 490 scenes is worth
  +0.26 AP50; removing any single MaskDINO component (two-stage query selection, deformable
  encoder, denoising, mask-enhanced box init) costs only 0.02–0.05 AP50 — and every crippled
  variant still beats arm C by ≈2×. The win belongs to the architecture *class*, not to one trick.
- The curve is **still rising at 490 scenes** (all the data we currently have packed) and
  overfitting eases with scale (train mIoU 1.000 → 0.994 → 0.947), i.e. the model is still
  data-limited.
- This **inverts an earlier conclusion**: arm C got *worse* with more data (0.367@190 →
  0.350@490), which read as "the dataset is not the bottleneck". On the same data MaskDINO gains
  +0.26 AP50. The old head was architecture-limited, not data-limited.

### 2.2 Multi-view protocol (arm C's own ruler)

One instance is scored once against its 8-frame ground-truth mask volume — a single IoU over the
concatenated frames, so a prediction is only correct if it is consistent across views. With
`--multi_frame`, one MaskDINO query is one instance across all 8 views by construction, so it can
be scored on exactly this ruler.

| Model | Train scenes | mIoU | AP50 | AP75 | mAP |
|---|---|---|---|---|---|
| arm C — best previous head | 190 | 0.367 | 0.199 | — | — |
| MaskDINO `--multi_frame` | 490 | 0.535 | 0.494 | 0.279 | 0.272 |
| **… + 2 view-draws/scene + colour jitter** | 490 | **0.539** | **0.515** | — | — |

No post-hoc matching, tracking or mask fusion is involved: cross-view consistency comes from the
shared query set.

Two ablations (2026-07-29) localise where the consistency comes from: removing the decoder's
**cross-frame attention** costs **−0.18** multi-view AP50 (0.494 → 0.311), and computing the
frozen features per frame instead of once per bundle costs **−0.15** (0.494 → 0.347) — VGGT's
global attention writes cross-view correspondence into the frozen tokens, and the decoder's
cross-frame attention consumes it. The same bundle-level features cost −0.05 *per-frame* AP50 as
a standalone change, so multi-view consistency has a measured price in per-frame accuracy
(0.729 single-frame best vs 0.630 per-frame for the best multi-view model).

### 2.3 The one caveat when reading §2.1 against §2.2

The two tables are **two rulers, not two results** — never mix them. The same arm-C checkpoint
reads 0.451 / 0.294 per-frame and 0.367 / 0.199 per-bundle. Per-frame always scores higher,
because an instance only has to match in the frames where it is actually visible, and a
prediction that claims no pixels in a frame is dropped rather than penalised (§4.3).

---

## 3. Our MaskDINO vs official MaskDINO (COCO val2017)

**Purpose.** Every number in §2 is measured against our own baselines, which cannot detect a bug
that is faithfully wrong on both sides. This check closes that loop.

**Method.** We load upstream MaskDINO's **released COCO checkpoint**
(`maskdino_r50_50ep_300q_hid1024_3sd1_instance_maskenhanced`) into upstream's own detectron2
harness, then **swap in our ported decoder and our deformable-attention encoder** and re-run the
full COCO val2017 evaluation. `--mode baseline` leaves upstream untouched and is the control.
Our decoder accepts upstream's weights at `strict=True`: **333/333 parameters**, names and shapes.

| COCO val2017, 5000 images | segm AP | segm AP50 | box AP | box AP50 |
|---|---|---|---|---|
| upstream model zoo, this exact checkpoint | 46.1 | — | 51.5 | — |
| `--mode baseline` (upstream code, our env) | 46.129 | 69.021 | 51.540 | 70.509 |
| **`--mode ours` (our ported modules)** | **46.133** | 69.036 | **51.549** | 70.514 |

**Δ = +0.004 segm AP / +0.009 box AP** against the control — same code path, same environment,
same weights, same data. On CPU the two modes are *bit-identical* to every printed digit; the
~0.005 drift appears only on GPU, because upstream calls the fused CUDA deformable-attention
kernel while our port always uses the portable `grid_sample` core. That is the one intended
difference between the two implementations.

**Verified as a live path, not a no-op.** The transplant asserts at runtime that the executing
modules are ours (6 + 9 ported deformable-attention instances), and perturbing a single weight
*inside our decoder* by 1.05× moves the score (55.702 → 55.608 AP on a 10-image subset). Identical
numbers therefore mean equivalence, not a silent fallback to upstream code.

**Scope.** Certified: deformable attention (encoder and decoder), the deformable encoder stack and
its reference points, two-stage query selection, DAB anchors, iterative box refinement,
mask-enhanced box init, the prediction heads. Not exercised by this route (training-only or
project-specific): the matcher, the criterion, denoising query generation, the multi-frame module,
and the VGGT feature pyramid. Those rest on the unit tests (perfect-prediction zero loss,
synthetic overfit).

**Do not put this next to a ScanNet number.** The backbone, dataset and task are all upstream's
here. It says our implementation of MaskDINO is faithful, and nothing more.

---

## 4. How the metrics are computed

All numbers in §2 come from a single function
(`train/eval_metrics.py::compute_instance_segmentation_metrics`), shared by both model families —
that is the only reason they are comparable. Masks are evaluated on VGGT's native **37×37 patch
grid** (518×518 input / 14-px patches), not at image resolution.

### 4.1 The four metrics

Given predicted instance masks (binarised at sigmoid > 0.5), their predicted class and score, and
the GT instance masks and classes:

- **mIoU** — for each *ground-truth* instance, the best IoU achieved by any prediction **of the
  same class**; averaged over GT instances. Recall-oriented: a missed instance contributes 0.
  It answers "how well is each real object covered?" and does not punish extra predictions.
- **AP50 / AP75** — class-aware average precision at IoU ≥ 0.50 / 0.75. Predictions are ranked by
  score; each is a true positive if it reaches the IoU threshold against a still-unmatched GT
  instance of the same class, otherwise a false positive. AP is the all-point (VOC2010-style) area
  under the resulting precision–recall curve. This *does* punish extra predictions and rewards
  well-calibrated scores.
- **mAP** — AP averaged over IoU thresholds 0.50 : 0.05 : 0.95 (COCO-style); the strictest of the
  four and the most sensitive to mask boundary quality.
- **class_acc** — among IoU-matched (prediction, GT) pairs from a class-agnostic Hungarian
  matching, the fraction with the correct predicted class. It separates "found the object but
  labelled it wrong" from "did not find the object".

### 4.2 Scoring, and what counts as a detection

The MaskDINO class head has **19 sigmoid logits and no background column**: "no object" is
*all logits low*. Objectness is therefore a threshold, not an argmax:
`max_c sigmoid(logit_c) ≥ 0.25` (upstream MaskDINO's `OBJECT_MASK_THRESHOLD`). This is the closest
analogue of the previous heads' "argmax ≠ background" filter, so both families are filtered
comparably. As in COCO, at most the **top-100** predictions per frame are kept.

Every evaluation additionally logs an `*_all` variant at threshold 0.0 (every query kept and
ranked by score) — the standard COCO detection protocol, and the only signal that moves early in
training, since focal-trained sigmoid scores start near zero.

### 4.3 The one protocol rule worth knowing

**A prediction that claims no pixels in a given frame is dropped, not counted as a false
positive.** Without it the per-frame protocol would be unfair to the multi-view heads: a
multi-view query is *supposed* to be empty in the frames where its object is not visible.
Mask2Former/MaskDINO reach the same place by folding the mask's mean foreground probability into
the score. Both scorers apply the rule identically; frames containing no GT instance at all are
skipped rather than counted as zeros.

### 4.4 What these numbers are not comparable to

- **Published ScanNet instance-segmentation numbers** (SegVGGT, FAST3DIS, IGGT, …). Those unproject
  per-view masks into the scene point cloud and score the official **3D** instance benchmark.
  We score **per-view 2D masks on a 37×37 grid with our own metric implementation**: different
  task, different GT, different metric code. **§5 is the exception** — it runs exactly their
  protocol and is the only section that may sit next to their numbers.
- **The official ScanNet v2 val split.** Our val = scenes 0080–0089 is a project convention, kept
  because it is the one ruler every arm and every scale point was measured on. A separate
  comparability read-out on the official val list (77 scenes present in our data, 413 train / 77
  val) is run separately rather than by re-scoring existing checkpoints.
- **Anything dated before 2026-07-08**, which used SAM3-generated pseudo-GT instead of official
  ScanNet annotations. That switch alone cost roughly half the headline AP50.

---

## 5. The 3D benchmark — the number that is comparable to the literature (2026-08-03)

Everything above is 2D and self-measured. This section is not: per-view masks are unprojected
into the scene point cloud using **VGGT's own predicted depth and cameras** (no ground-truth
geometry at inference), majority-voted per superpoint, and scored by the **official ScanNet 3D
instance evaluator**, vendored unmodified, on the **official val-312 split** with a model
trained on the official 1201-scene train split (no overlap).

| Method | backbone | AP | AP50 | AP25 |
|---|---|---|---|---|
| SegVGGT (published) | VGGT, **LoRA-adapted** | 0.504 | 0.717 | 0.870 |
| FAST3DIS (published) | Depth-Anything-V3, **LoRA-adapted** | 0.038 | 0.096 | 0.316 |
| **ours** | **VGGT, strictly frozen** | **0.023** | **0.067** | **0.268** |
| ours, tuned lifting knobs | 〃 | 0.029 | 0.083 | 0.305 |

**We are in FAST3DIS's ballpark while never touching the backbone; SegVGGT is an order of
magnitude ahead.** Both statements belong in any honest summary. The tuned row's two knobs were
selected on an earlier (leaky) diagnostic run, so the plain row is the headline.

Two findings worth carrying:

1. **Data scale beats leakage.** An earlier checkpoint that had *seen* the val scenes during
   training scored 0.052 AP50; this leak-free one, trained on 1201 official scenes, scores
   0.083 — 1.6× better despite the disadvantage. The 3D ruler independently reproduces the 2D
   conclusion that this track is **data-limited, not architecture-limited**.
2. **The bottleneck is the 2D→3D lifting, not the decoder.** AP25 is ~4× AP50: objects are
   found and coarsely placed, but the lifted masks miss the strict-IoU bar. The registration
   diagnostics say why — median camera-centre error after alignment is 0.14 m, the same order as
   the voting radius — and only ~16 % of mesh vertices receive any vote. Two *lifting* knobs
   alone were worth more (+0.016 AP50) than most decoder ablations in §2. That is where the next
   effort belongs, and it is the price of the "no GT geometry at inference" design.

Details, per-class tables and reproduction: `docs/MASKDINO.md` §9, `docs/RESULTS.md` §5.

---

*Sources in the repo: `docs/RESULTS.md` (all numbers, split by protocol), `docs/MASKDINO.md`
(architecture, §6 protocol, §7 results, §7.6 COCO equivalence check, §9 the 3D ruler),
`docs/ARMS_SUMMARY.md` (the retired arms A–E).*
