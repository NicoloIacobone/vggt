# Results — one table per protocol

Two model families, two protocols. **The single most common mistake in this project is comparing
across the horizontal line below.** Read §1 before quoting any number.

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

## 2. Single-frame protocol — the comparison that matters

| Model | Scenes | val mIoU | val AP50 | val AP75 | val mAP |
|---|---|---|---|---|---|
| arm C (best D4RT head) — **the bar** | 190 | 0.451 | 0.294 | 0.141 | 0.154 |
| MaskDINO | 50 | 0.451 | 0.440 | 0.314 | 0.290 |
| MaskDINO | 190 | 0.594 | 0.624 | 0.440 | 0.418 |
| **MaskDINO** | **490** | **0.669** | **0.699** | **0.506** | **0.475** |

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

## 3. Multi-view protocol — the retired D4RT arms

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
