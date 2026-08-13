# MaskDINO on frozen VGGT — results summary

**Task.** 3D multi-view-consistent instance segmentation on top of the **frozen VGGT-1B**
backbone. Only the decoder head is trained; supervision is the official ScanNet v2 2D instance
annotations (19 classes). Nothing in VGGT is modified or finetuned.

**Two things are reported below**, and they answer two different questions:

1. **MaskDINO vs the baseline head** — does the new head beat the best of the previous (hand-rolled DETR-style)
   heads on our ScanNet task? *This is the project result.*
2. **Our MaskDINO vs official MaskDINO on COCO** — is our re-implementation of MaskDINO faithful to
   upstream? *This is a correctness proof for the port, not a project result.*

---

## 1. Headline

| | previous best (the baseline head) | MaskDINO (best) | gain |
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

**Best on that split since 2026-08-06: widening the bundle from 8 to 16 views** (one flag,
`--num_frames 16`; `docs/MASKDINO.md` §8.4). Per-bundle AP50 **0.525 → 0.552** on the same pinned
8-view ruler, per-frame 0.650 → 0.662, and `bundle_id_switch` **0.498 → 0.385** with the number
of matched instances flat — identity, not recognition, exactly as the cross-frame-attention
ablation predicted. A frame-matched control run (`--bundles_per_scene 1`) rules out "it is just
more data". A 20-epoch run reaches per-frame **0.669**, the best on the official split.

**And since 2026-08-03 there is a number that IS comparable to the literature** — see §5.

---

## 2. MaskDINO vs the baseline head

The baseline head = the best of a retired family of hand-rolled DETR-style heads (learned DETR
object queries). Same frozen backbone, same data, same metric code.

### 2.1 Single-frame protocol (the primary comparison)

Each frame is scored separately; per-scene averages are then averaged over scenes.
All runs: official ScanNet GT, val = scenes 0080–0089, held out of every training set.

| Model | Train scenes | mIoU | AP50 | AP75 | mAP |
|---|---|---|---|---|---|
| **the baseline head — the bar** | 190 | 0.451 | 0.294 | 0.141 | 0.154 |
| MaskDINO | 50 | 0.451 | 0.440 | 0.314 | 0.290 |
| MaskDINO | 190 | 0.594 | 0.624 | 0.440 | 0.418 |
| MaskDINO | 490 | 0.669 | 0.699 | 0.506 | 0.475 |
| **MaskDINO + 2 view-draws/scene + colour jitter** | **490** | **0.694** | **0.729** | **0.582** | **0.526** |

Notes worth stating:

- The baseline head's per-frame numbers were obtained by re-scoring its released checkpoint through the
  *identical* protocol (`scripts/eval_perframe.py`), not by quoting its old numbers on a
  different ruler.
- **Data scale dominates every architectural ingredient.** Going 50 → 490 scenes is worth
  +0.26 AP50; removing any single MaskDINO component (two-stage query selection, deformable
  encoder, denoising, mask-enhanced box init) costs only 0.02–0.05 AP50 — and every crippled
  variant still beats the baseline head by ≈2×. The win belongs to the architecture *class*, not to one trick.
- The curve is **still rising at 490 scenes** (all the data we currently have packed) and
  overfitting eases with scale (train mIoU 1.000 → 0.994 → 0.947), i.e. the model is still
  data-limited.
- This **inverts an earlier conclusion**: the baseline head got *worse* with more data (0.367@190 →
  0.350@490), which read as "the dataset is not the bottleneck". On the same data MaskDINO gains
  +0.26 AP50. The old head was architecture-limited, not data-limited.

### 2.2 Multi-view protocol (the baseline head's own ruler)

One instance is scored once against its 8-frame ground-truth mask volume — a single IoU over the
concatenated frames, so a prediction is only correct if it is consistent across views. With
`--multi_frame`, one MaskDINO query is one instance across all 8 views by construction, so it can
be scored on exactly this ruler.

| Model | Train scenes | mIoU | AP50 | AP75 | mAP |
|---|---|---|---|---|---|
| the baseline head — best previous head | 190 | 0.367 | 0.199 | — | — |
| MaskDINO `--multi_frame` | 490 | 0.535 | 0.494 | 0.279 | 0.272 |
| **… + 2 view-draws/scene + colour jitter** | 490 | **0.539** | **0.515** | — | — |

No post-hoc matching, tracking or mask fusion is involved: cross-view consistency comes from the
shared query set.

Two ablations (2026-07-29) localise where the consistency comes from: removing the decoder's
**cross-frame attention** costs **−0.18** multi-view AP50 (0.494 → 0.311), and computing the
frozen features per frame instead of once per bundle costs **−0.15** (0.494 → 0.347) — VGGT's
global attention writes cross-view correspondence into the frozen tokens, and the decoder's
cross-frame attention consumes it.

**What that block actually does is preserve identity, and we can now measure it directly**
(2026-08-03, official split). Removing it leaves the number of instances the model finds
unchanged (14.0 vs 14.1 per bundle) and barely moves whether an instance's own query covers a
given view (0.69 vs 0.72). What breaks is *which* query owns the object: the rate at which some
**other** query fits a view better jumps from **50 % to 68 %**. Recognition survives; cross-view
identity does not — which is the mechanism the multi-view score is actually paying for. The same bundle-level features cost −0.05 *per-frame* AP50 as
a standalone change, so multi-view consistency has a measured price in per-frame accuracy
(0.729 single-frame best vs 0.630 per-frame for the best multi-view model).

### 2.3 The one caveat when reading §2.1 against §2.2

The two tables are **two rulers, not two results** — never mix them. The same baseline checkpoint
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

**Scope.** Certified by the route above: deformable attention (encoder and decoder), the
deformable encoder stack and its reference points, two-stage query selection, DAB anchors,
iterative box refinement, mask-enhanced box init, the prediction heads. Not exercised by it
(training-only or project-specific): the matcher, the criterion, denoising query generation, the
multi-frame module, and the VGGT feature pyramid.

**The training path is now certified too (2026-08-12).** The route above drives *upstream's
weights* through our modules, so it never touches the loss. A second control closes that hole:
upstream MaskDINO's own code, **trained from scratch under our recipe** (frozen R50, 12 epochs,
squash@518, 87 948 iterations) against our own `resnet50` arm under the same recipe.

| COCO val2017, 5000 images, trained not loaded | segm AP | AP50 | box AP |
|---|---|---|---|
| upstream MaskDINO's code, our recipe | 34.55 | 54.6 | 38.3 |
| **our port, same recipe** | **34.3** | 54.1 | 38.2 |

**Δ = +0.25 segm AP**, and ±0.84 AP at all 16 matched intermediate evaluations (mean +0.23, sign
changing). Two independently written matchers, criteria and denoising-query generators converging
to a quarter of an AP over 88 000 steps of real training is what those three modules needed, and
they now rest on that rather than on unit tests alone. It also prices the recipe: upstream's
released 50-epoch finetuned checkpoint scores 46.1, so **freezing the backbone and training 12
epochs at 518 px costs ~11.6 AP** — a measurement now, not an inference against a
differently-trained model. Details: `docs/MASKDINO_COCO.md` §6.

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

- **Published ScanNet instance-segmentation numbers** (SegVGGT, FAST3DIS, IGGT, …). Those score
  masks on the scene point cloud against the official **3D** instance benchmark. We score
  **per-view 2D masks on a 37×37 grid with our own metric implementation**: different
  task, different GT, different metric code. **§5 is the exception** — it runs the official 3D
  benchmark and is the only section that may sit next to published numbers, subject to the
  two-protocol rule stated there.
- **The official ScanNet v2 val split.** Our val = scenes 0080–0089 is a project convention, kept
  because it is the one ruler every retired-baseline and every scale point was measured on. A separate
  comparability read-out on the official val list (77 scenes present in our data, 413 train / 77
  val) is run separately rather than by re-scoring existing checkpoints.
- **Anything dated before 2026-07-08**, which used SAM3-generated pseudo-GT instead of official
  ScanNet annotations. That switch alone cost roughly half the headline AP50.

---

## 5. The 3D benchmark — the number that is comparable to the literature (2026-08-03; protocol note added 2026-08-04)

Everything above is 2D and self-measured. This section is not: per-view masks are unprojected
into the scene point cloud using **VGGT's own predicted depth and cameras** (no ground-truth
geometry at inference), majority-voted per superpoint, and scored by the **official ScanNet 3D
instance evaluator**, vendored unmodified, on the **official val-312 split** with a model
trained on the official 1201-scene train split (no overlap).

**First, which published numbers this can sit next to.** The literature's 3D numbers are **two
protocols, not one**, split by how a finished 2D mask reaches the benchmark point cloud:

- **Unposed / predicted-geometry transfer** — masks are unprojected with the model's *own*
  predicted depth and cameras. FAST3DIS, IGGT and **we** are here. The score is 2D mask quality
  **×** feed-forward geometry quality. (Their two rows are also **class-agnostic** — second axis,
  below the table.)
- **Posed transfer** — **SegVGGT** is here. Its released evaluator never unprojects: it projects
  the GT benchmark cloud into each view using ScanNet's **GT poses and intrinsics**, resolving
  occlusion with the **ScanNet sensor depth** map, so there is no Sim(3), no ICP and no geometry
  error at all. The score is 2D mask quality alone.

Both use the official ScanNet evaluator with identical options — the evaluator is not the
difference, the bridge is. This is **not** an accusation: SegVGGT's *model* takes unposed images
only, exactly like ours, and using GT geometry solely to transfer finished masks for scoring is a
legitimate way to isolate segmentation quality from reconstruction quality. It simply means the
two rows below are answering different questions. (Evidence, with file:line references into their
released code: `docs/RELATED_WORK.md`, "Two 3D protocols"; `docs/MASKDINO.md` §9.9.)

| Method | backbone | protocol | labels | AP | AP50 | AP25 |
|---|---|---|---|---|---|---|
| FAST3DIS (published), 50 views | Depth-Anything-V3, **LoRA-adapted** | unposed | **class-agnostic** | 0.038 | 0.096 | 0.316 |
| IGGT, **as re-evaluated by FAST3DIS** | — | unposed | **class-agnostic** | 0.028 | 0.112 | 0.287 |
| **ours** | **VGGT, strictly frozen** | **unposed** | class-aware (18) | **0.023** | **0.067** | **0.268** |
| ours, tuned lifting knobs | 〃 | unposed | class-aware (18) | 0.029 | 0.083 | 0.305 |
| **ours, scored FAST3DIS's way** | 〃 | unposed | **class-agnostic** | **0.017** | **0.060** | **0.334** |
| **ours, `--anchor_3d`, scored FAST3DIS's way** | **VGGT, strictly frozen** | **unposed** | **class-agnostic** | **0.042** | **0.138** | **0.504** |
| 〃 best lifting knob (sensitivity, not the headline) | 〃 | unposed | class-agnostic | 0.055 | 0.185 | 0.571 |
| **ours, run under SegVGGT's own protocol** | **VGGT, strictly frozen** | **posed** | class-aware (18) | **0.060** | **0.156** | **0.408** |
| SegVGGT (published) | VGGT, **LoRA-adapted** | **posed** | class-aware (18) | 0.504 | 0.717 | 0.870 |

**The headline sentence, as of 2026-08-07: scored exactly the way FAST3DIS and IGGT score
themselves, our 3D-anchored checkpoint leads both on all three metrics — 0.042 / 0.138 / 0.504
against 0.038 / 0.096 / 0.316 and 0.028 / 0.112 / 0.287 — with a strictly frozen backbone against
their LoRA-adapted ones, ~17 views per scene against FAST3DIS's 50, and no tuning of the lifting
step.** Two supports it needs and now has: re-sweeping the lifting knobs spans 0.138 → 0.185 and
*every* point of the grid still leads, so this is not a tuning artefact; and re-training both arms
at a second seed puts per-bundle AP50 spread at ±0.009, so the 2D side is not seed noise either.
SegVGGT's much larger number comes from a protocol in which the 2D→3D step is error-free by
construction — never replace that with "they are an order of magnitude ahead", which compares
across protocols. The tuned row's two knobs were selected on an earlier (leaky) diagnostic run, so the plain row is the
headline.

**A second axis, added 2026-08-06 after re-reading both papers.** FAST3DIS and IGGT are scored
**class-agnostic** — *"we ignore the semantic class labels in the annotations"* (FAST3DIS §4.4),
and that paper publishes no class-aware ScanNet number; SegVGGT and we are class-aware over the
benchmark's 18 classes. The metric *definition* is the same everywhere (`mAP` in SegVGGT's header
and `AP` in FAST3DIS's are one quantity: IoU 0.50:0.05:0.95, with AP50/AP25 at fixed thresholds),
so the column names are not the difference — the setting is. **We measured our own class-agnostic
column rather than arguing about it** (jobs 9861563 / 9861564, 312 scenes, 0 failures): scored
their way our tuned row is **0.017 / 0.060 / 0.334** (defaults 0.013 / 0.050 / 0.320). Like for
like we are **ahead of both published rows on AP25** (0.334 vs FAST3DIS 0.316 and IGGT 0.287) and
**~1.6–2.2× behind on AP50 and AP**. Collapsing the labels *lowers* our AP/AP50 — it replaces a
mean over 18 classes, which our rare distinctive classes carry (toilet 0.508 AP50 at 1/18 weight),
with one instance-pooled ranking dominated by the numerous weak classes and by `otherfurniture`,
which our 19-class head cannot predict at all. So "in FAST3DIS's ballpark" holds at loose IoU and
not at strict IoU — **for this checkpoint.** On the `--anchor_3d` checkpoint (the row above, and
the one to quote) the collapse goes the other way and we lead on all three, so that sentence
describes the §9.6 model only and must not be carried to the headline row.
Two further provenance facts: IGGT's own paper (arXiv 2510.22706) reports **no ScanNet AP** — only
tracking, reconstruction and open-vocabulary semantics over 10 scenes × 8–10 images; and SegVGGT
states its posed bridge in the paper itself (*"we utilize the ground-truth depth maps and camera
poses during this mapping stage for fair comparison"*), so that point rests on their sentence, not
on our code reading.

**And the numbers a reader may be remembering are a third family.** SegVGGT's Table 1 also lists
Mask3D 55.2 / 73.7 / 85.3, Relation3D 62.5 / 80.2 / 87.0, SegDINO3D 64.0 / 81.5 / 88.9 and ODIN
50.0 / 71.0 / 83.6 — all taking a **point cloud or RGB-D** as input. The single image-only
baseline there, OneFormer3D†, scores **5.4 / 10.2 / 17.4**, below us despite the posed protocol.

**We now have the like-for-like row too, and it does not close the gap** (added 2026-08-04,
docs/MASKDINO.md §9.10; `--transfer_mode gt_projection`, licensed by an oracle whose round-trip
purity is 0.9999 over all 312 scenes). Putting *our* masks through *SegVGGT's* bridge — same
checkpoint, same 17 frames, same queries, same evaluator, only the 2D→3D step swapped — takes us
from 0.067 to 0.156 AP50. So the protocol is worth **2.3×** of the gap and **~4.6× is real**
(their LoRA-adapted backbone, 4–6× the views, 259×196 masks vs our 37×37, 600 kept queries vs
our 100). The right sentence for a supervisor is: *"printed side by side the two numbers measure
different things; measured properly, we are 2.3× closer than the table suggests and still
meaningfully behind."* The unposed 0.023 / 0.067 / 0.268 remains our headline — the posed row is
a decomposition, not a result.

That decomposition also bounds our own roadmap: 0.156 is what our current masks score with a
*perfect* bridge, so all remaining lifting/registration work is worth at most +0.089 AP50, and
anything beyond that has to come from the masks themselves.

Two findings worth carrying:

1. **Data scale beats leakage.** An earlier checkpoint that had *seen* the val scenes during
   training scored 0.052 AP50; this leak-free one, trained on 1201 official scenes, scores
   0.083 — 1.6× better despite the disadvantage. The 3D ruler independently reproduces the 2D
   conclusion that this track is **data-limited, not architecture-limited**.
2. **The bottleneck is the 2D→3D lifting, not the decoder.** AP25 is ~4× AP50: objects are
   found and coarsely placed, but the lifted masks miss the strict-IoU bar. The registration
   diagnostics say why — median camera-centre error after alignment is 0.14 m — and only ~16 %
   of mesh vertices receive any vote. An 8-point sweep of the two lifting hyper-parameters moves
   AP50 from 0.067 to 0.091, more than any decoder ablation in §2 is worth, and the voting
   radius stops helping exactly at 0.15 m — the size of the registration error. This is the price
   of the "no ground-truth geometry at inference" design, and it is now quantified rather than
   assumed. **Amended 2026-08-07:** re-swept on the 3D-anchored checkpoint the same two knobs are
   worth more (class-agnostic AP50 0.138 → 0.185) and the whole grid sits *above* FAST3DIS rather
   than below it, so "the gap is not a tuning artefact" describes the earlier checkpoint only.
   The knobs are checkpoint-dependent — one of them even flips sign — so they must be re-swept per
   checkpoint rather than carried across. Coverage and registration remain the structural limit.

Details, per-class tables and reproduction: `docs/MASKDINO.md` §9, `docs/RESULTS.md` §5.

---

*Sources in the repo: `docs/RESULTS.md` (all numbers, split by protocol) and `docs/MASKDINO.md`
(architecture, §6 protocol, §7.6 COCO equivalence check, §9 the 3D ruler).*
