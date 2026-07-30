# Results — one table per protocol

Two model families, two protocols. **The single most common mistake in this project is comparing
across the horizontal line below.** Read §1 before quoting any number.

**A third ruler exists and is not in this file.** The COCO backbone-swap study
(`docs/MASKDINO_COCO.md`) scores standard COCO mask/box AP with `pycocotools` on a different
dataset, a different task and a different metric implementation. Nothing there is comparable to
anything here — it answers "how much does the backbone swap cost against upstream MaskDINO's own
numbers", not "how good is the ScanNet head".

## 1. The two protocols

| | multi-view (per-bundle) | single-frame (per-frame) |
|---|---|---|
| unit scored | one instance vs its 8-frame GT mask, one IoU over the concatenated frames | each frame separately, averaged over frames then over scenes |
| used by | the D4RT arms A–E (retired) | the MaskDINO track (active) |
| scoring | softmax, "argmax ≠ background" | sigmoid, `max_c sigmoid(logit_c) ≥ 0.25` |
| bridge | — | `scripts/eval_perframe.py` puts a D4RT checkpoint on this protocol |

Per-frame scores **higher** than per-bundle for the same checkpoint: an instance only has to
match in the frames where it is visible, and a prediction that claims no pixels in a frame is
dropped rather than penalised (`train/perframe.py::drop_empty_masks`). That is why arm C reads
0.451/0.294 per-frame and 0.367/0.199 per-bundle — the same model, two rulers.

All numbers below: official ScanNet GT, per-instance masks, val = scenes 0080–0089, held out of
every training set.

### 1.1 The val split, and why it is not the official one (decided 2026-07-28)

Val = scenes **0080–0089** is a *project convention*, not the official ScanNet v2 val split. It
stays that way: it is the one ruler every arm and every MaskDINO scale point was measured on, and
switching would break the continuity of the 50 → 190 → 490 scaling curve for zero real gain in
comparability (§1.2). Alongside it there is now **one comparability read-out**: the official
ScanNet v2 val list (`data/splits/scannetv2_val.txt`, 312 scenes) intersected with our 500-scene
tar gives **77** scenes (`*_00`, id < 490). 74 of those sit inside the usual training range, so
the read-out needs its **own run** rather than a re-scoring of an existing checkpoint —
`VAL_SPLIT=official sbatch slurm/train_maskdino.sh` (413 train / 77 val, job 8900194).

### 1.2 …and none of it is comparable to published ScanNet numbers

Published feed-forward competitors (SegVGGT, FAST3DIS, IGGT, …) unproject their per-view masks
into the scene point cloud and score the **official 3D instance benchmark** (AP/AP50/AP25). We
score **per-view 2D masks on the 37×37 patch grid** with our own metric code. Different task,
different GT, different metric implementation — never put the two in one table. The full
side-by-side of the two protocols is in `docs/RELATED_WORK.md` ("Numbers: what is comparable to
what").

### 1.3 Fixed cached view sets (accepted 2026-07-28)

Frames are drawn once per scene up front and reused for the whole run — that is what makes
head-only training take minutes instead of hours. The risk (the head memorising the cached view
combinations rather than learning view-set robustness) is accepted and stated, and measured once
by the `--bundles_per_scene 2 --color_jitter 0.2` run at N=490 (job 8895565, §2).

### 1.4 The COCO numbers are implementation verification, not a project result

`docs/MASKDINO.md` §7.6 reports **46.133 mask AP / 51.549 box AP on COCO val2017**. That is not a
result of this project and belongs in no table here. It is a *correctness proof for the port*:
our ported decoder + deformable encoder driven by upstream MaskDINO's own released COCO weights,
reproducing that checkpoint's published number (46.1 / 51.5, the README model-zoo row
"MaskDINO (hid 1024)" — *not* a paper table value) to +0.004 AP, and matching an unmodified
upstream run in the same environment to the same tolerance. It says our implementation of MaskDINO is
faithful. It says nothing about VGGT, ScanNet, or 3D consistency — the backbone, the dataset and
the task are all upstream's there. Never quote it next to a ScanNet number.

## 2. Single-frame protocol — the comparison that matters

| Model | Scenes | val mIoU | val AP50 | val AP75 | val mAP |
|---|---|---|---|---|---|
| arm C (best D4RT head) — **the bar** | 190 | 0.451 | 0.294 | 0.141 | 0.154 |
| MaskDINO | 50 | 0.451 | 0.440 | 0.314 | 0.290 |
| MaskDINO | 190 | 0.594 | 0.624 | 0.440 | 0.418 |
| MaskDINO | 490 | 0.669 | 0.699 | 0.506 | 0.475 |
| **MaskDINO + `--bundles_per_scene 2 --color_jitter 0.2`** | **490** | **0.694** | **0.729** | **0.582** | **0.526** |

### One-flag variants at N=490 (2026-07-28, ΔAP50 vs 0.699)

| Change | val mIoU | val AP50 | ΔAP50 | verdict |
|---|---|---|---|---|
| `--bundles_per_scene 2 --color_jitter 0.2` | 0.694 | 0.729 | **+0.030** | best result so far; still data-limited |
| `--bundles_per_scene 4 --color_jitter 0.2` | 0.699 | 0.722 | +0.023 | **saturates**: within noise of 2 draws — the views-per-scene lever is exhausted, more *scenes* is the remaining data lever |
| `--mask_upsample 2` (74×74 masks) | 0.662 | 0.677 | −0.022 | neutral (inside ±0.04 noise) — masks stay on the 37×37 grid |
| `--feature_mode bundle` (multi-view-aware tokens) | 0.622 | 0.651 | −0.048 | **negative result** per frame — but *required* for multi-view consistency (§3) |
| `--multi_frame --feature_mode bundle` | 0.621 | 0.630 | −0.069 | −0.021 against its own control (`bundle`, 0.651) → per-frame neutral, and it buys the multi-view metric below |

Job ids, caveats (the `--bundles_per_scene 2` job got a 2× step budget through an epoch-clamp
bug, since fixed) and the reasoning: `docs/MASKDINO.md` §7.4.

**+48 % mIoU, +138 % AP50 over the best D4RT head.** The curve is still rising at 490 scenes —
all the official-GT tar holds — and overfitting eases with scale (train mIoU 1.000 → 0.994 →
0.947), so the model is still data-limited.

### Ablations at N=190 — no single ingredient carries the win

| Config | val mIoU | val AP50 | ΔAP50 |
|---|---|---|---|
| full recipe | 0.594 | 0.624 | — |
| `--no-two_stage` | 0.592 | 0.578 | −0.046 |
| `--enc_layers 0` | 0.551 | 0.580 | −0.044 |
| `--dn no` | 0.586 | 0.594 | −0.030 |
| `--initialize_box_type no` | 0.610 | 0.608 | −0.016 |

Every crippled variant still beats arm C by ~2×. Credit belongs to the architecture *class*, and
**data scale dominates everything**: +0.26 AP50 from 50→490 scenes vs ≤0.05 from any component.
Details and job ids: `docs/MASKDINO.md` §7.

## 3. Multi-view protocol — the D4RT arms, and now MaskDINO too

Since 2026-07-28 the MaskDINO track can be scored on this ruler as well: with `--multi_frame`
one query is one instance across all 8 views by construction (docs/MASKDINO.md §8.2), so its
mask volume can be scored exactly like an arm's.

| Model (N=490) | mIoU | AP50 | AP75 | mAP |
|---|---|---|---|---|
| arm C — best D4RT head (N=190) | 0.367 | 0.199 | — | — |
| **MaskDINO `--multi_frame`** (job 8900100) | **0.535** | **0.494** | 0.279 | 0.272 |
| … `--no-cross_frame_attn` (job 8950617) | 0.393 | 0.311 | 0.089 | 0.132 |
| … `--feature_mode single` (per-frame features, job 8950613) | 0.429 | 0.347 | 0.154 | 0.181 |

**+46 % mIoU, 2.5× AP50 on the arms' own protocol**, with no post-hoc matching or fusion.

The two ablation rows (2026-07-29, docs/MASKDINO.md §7.4.1) localise the result: **cross-frame
attention is worth 0.183 bundle AP50** — the only individually-decisive component found anywhere
in this track — and **bundle features are worth 0.147**. The same bundle features are a *negative*
for per-frame accuracy (§2), so multi-view consistency has a measured price: 0.729 single-frame
best vs 0.630 per-frame for the best multi-view model.

### 3.1 The retired D4RT arms

Query-initialisation strategies on the same frozen backbone and multi-view supervision. Full
per-arm narrative and verdicts: `docs/ARMS_SUMMARY.md`.

All official GT (`(S)` marks the one SAM3-GT number kept for context — a different ruler, see §4).
Only arm C was run at N=190 on official GT; the other arms were scaled straight to N=490 in the
2026-07-21/22 sweep, all with `--bundles_per_scene 1`.

| Arm | Queries | N=190 mIoU / AP50 | N=490 mIoU / AP50 | Verdict |
|---|---|---|---|---|
| A | point prompts (D4RT style) | 0.216 / 0.105 (S) | 0.264 / 0.102 | superseded by C — plateaued past N=50 |
| B | trained grid queries | — | 0.110 / 0.172 [grid] | closed — AP50 never stable, prompted path regresses |
| **C** | **learned DETR object queries** | **0.367 / 0.199** | 0.350 / 0.177 | **best D4RT arm** |
| D | hybrid (C's slots + A's prompts) | — | NaN @ ep110 (best-before 0.295 / 0.174) | closed — instability recurs at scale |
| E | 3D-anchored (FPS over the point cloud) | — | 0.248 / 0.139 | closed — best E variant (v1 hybrid), still below C |

**Arm C wins at every scale tested**, and the ranking C > E > B ≈ D > A holds at N=490 too, so
data scale does not change the query-strategy verdict. The keeper from arm E is the ablation
story plus a calibration finding (E keeps 0.59–0.86× as many predictions as there are GT
instances, vs C's 1.23× — the 3D-spread prior suppresses duplicates as designed).

**Arm C got *worse* with more data** (0.367@190 → 0.350@490), which at the time read as "the
dataset is not the bottleneck". The MaskDINO scaling curve inverts that conclusion: the D4RT
head was **architecture-limited, not data-limited**. The old scaling result was a property of
that head, not of the task.

## 4. Ground-truth quality — why the older numbers do not transfer

Switching from SAM3-generated GT to official ScanNet GT (2026-07-08) cost about half the honest
AP50 headline. Any pre-2026-07-08 number in `docs/old/` is on the old ruler. Table and reasoning:
`docs/DATASET.md` §1.
