# RESULTS — ARCHIVE: the rulers this project no longer reports on

**Nothing in this file is current.** It is the pre-official-split half of
`docs/RESULTS.md`, moved here on 2026-08-27 when the project's reporting was cut back to
the official ScanNet v2 1201/312 split and larger. It is kept for provenance only: the
project-val convention (scenes 0080–0089), the retired baseline head, the SAM3 ground
truth, and the COCO port check. **Never cite a number from this file**, and never place
one next to a competitor's — the split it was measured on overlaps the training scenes of
several of the runs it scores, which is exactly why it was retired.

Section numbers are the ones these sections had in `docs/RESULTS.md`; the live file keeps
its numbering unchanged, so §5 there is still §5.

---

### 1.1 The val split, and why it is not the official one (decided 2026-07-28)

Val = scenes **0080–0089** is a *project convention*, not the official ScanNet v2 val split. It
stays that way: it is the one ruler every retired-baseline and every MaskDINO scale point was measured on, and
switching would break the continuity of the 50 → 190 → 490 scaling curve for zero real gain in
comparability (§1.2). Alongside it there is now **one comparability read-out**: the official
ScanNet v2 val list (`data/splits/scannetv2_val.txt`, 312 scenes) intersected with our 500-scene
tar gives **77** scenes (`*_00`, id < 490). 74 of those sit inside the usual training range, so
the read-out needs its **own run** rather than a re-scoring of an existing checkpoint —
`VAL_SPLIT=official sbatch slurm/train_maskdino.sh` (413 train / 77 val, job 8900194).

**Since 2026-08-02 the full official 1201/312 split has its own runs — §6.** That is a third 2D
ruler: its numbers live only there and are never mixed into the §2/§3 tables.

### 1.2 …and none of it is comparable to published ScanNet numbers

Published feed-forward competitors (SegVGGT, FAST3DIS, IGGT, …) score their masks on the
**official 3D instance benchmark** (AP/AP50/AP25) against the benchmark point clouds. We score
**per-view 2D masks on the 37×37 patch grid** with our own metric code. Different task,
different GT, different metric implementation — never put the two in one table. The full
side-by-side of the two protocols is in `docs/RELATED_WORK.md` ("Numbers: what is comparable to
what"). **The bridge now exists (§5): the 3D ruler runs OUR model on the official benchmark** —
it is the only place in this project where a number may sit next to a published one, and it
lives in its own section for that reason. Read §5's protocol note before doing so: the published
3D numbers are themselves split across **two different protocols**, and only one of them is ours.

**And the training data differs too — see `docs/TRAINING_COMPARABILITY.md`.** SegVGGT trains on the
same official 1201 split we do; FAST3DIS trains on synthetic Aria data *only* and scores ScanNet
zero-shot; IGGT trains on a mixture that includes ScanNet++. A protocol-matched comparison is still
not a training-matched one.

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

**Since 2026-08-12 the port is certified on the *training* path too** (`docs/MASKDINO_COCO.md` §6
row 2): upstream MaskDINO's own code, trained under our recipe, lands at **34.55 segm AP against
our `resnet50` arm's 34.3**. §7.6's certificate covered inference only and explicitly excluded
`matcher.py`, `criterion.py` and DN generation; those three are now corroborated end to end. Still
not a project result, and still belongs in no table here.

## 2. Single-frame protocol — the comparison that matters

| Model | Scenes | val mIoU | val AP50 | val AP75 | val mAP |
|---|---|---|---|---|---|
| the retired baseline head — **the bar** | 190 | 0.451 | 0.294 | 0.141 | 0.154 |
| MaskDINO | 50 | 0.451 | 0.440 | 0.314 | 0.290 |
| MaskDINO | 190 | 0.594 | 0.624 | 0.440 | 0.418 |
| MaskDINO | 490 | 0.669 | 0.699 | 0.506 | 0.475 |
| **MaskDINO + `--bundles_per_scene 2 --color_jitter 0.2`** | **490** | **0.694** | **0.729** | **0.582** | **0.526** |

### One-flag variants at N=490 (2026-07-28, ΔAP50 vs 0.699)

| Change | val mIoU | val AP50 | ΔAP50 | verdict |
|---|---|---|---|---|
| `--bundles_per_scene 2 --color_jitter 0.2` | 0.694 | 0.729 | **+0.030** | best result so far; still data-limited |
| `--bundles_per_scene 4 --color_jitter 0.2` | 0.699 | 0.722 | +0.023 | **saturates**: within noise of 2 draws — the views-per-scene lever is exhausted, more *scenes* is the remaining data lever |
| `--mask_upsample 2` (74×74 masks) | 0.662 | 0.677 | −0.022 | neutral (inside ±0.04 noise) — masks stay on the 37×37 grid. **Confirmed on the full-resolution ruler too** (docs/MASKDINO.md §7.7): 37×37's GT-only ceiling is 0.956 AP50 vs the model's ~0.69 — recognition binds, not resolution |
| `--feature_mode bundle` (multi-view-aware tokens) | 0.622 | 0.651 | −0.048 | **negative result** per frame — but *required* for multi-view consistency (§3) |
| `--multi_frame --feature_mode bundle` | 0.621 | 0.630 | −0.069 | −0.021 against its own control (`bundle`, 0.651) → per-frame neutral, and it buys the multi-view metric below |

Job ids, caveats (the `--bundles_per_scene 2` job got a 2× step budget through an epoch-clamp
bug, since fixed) and the reasoning: `docs/MASKDINO.md` §7.4.

**+48 % mIoU, +138 % AP50 over the retired baseline head** for the plain N=490 recipe (0.669 / 0.699);
**+54 % / +148 %** for the bolded `--bundles_per_scene 2` row (0.694 / 0.729). The curve is still rising at 490 scenes —
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

Every crippled variant still beats the baseline head by ~2×. Credit belongs to the architecture *class*, and
**data scale dominates everything**: +0.26 AP50 from 50→490 scenes vs ≤0.05 from any component.
Details and job ids: `docs/MASKDINO.md` §7.

## 3. Multi-view protocol — the baseline head's own ruler, and now MaskDINO too

Since 2026-07-28 the MaskDINO track can be scored on this ruler as well: with `--multi_frame`
one query is one instance across all 8 views by construction (docs/MASKDINO.md §8.2), so its
mask volume can be scored exactly like the baseline head's.

| Model (N=490) | mIoU | AP50 | AP75 | mAP |
|---|---|---|---|---|
| the retired baseline head (N=190) | 0.367 | 0.199 | — | — |
| MaskDINO `--multi_frame` (job 8900100) | 0.535 | 0.494 | 0.279 | 0.272 |
| … `--no-cross_frame_attn` (job 8950617) | 0.393 | 0.311 | 0.089 | 0.132 |
| … `--feature_mode single` (per-frame features, job 8950613) | 0.429 | 0.347 | 0.154 | 0.181 |
| **… + `--bundles_per_scene 2 --color_jitter 0.2`** (job 9071415) | **0.539** | **0.515** | — | — |

**+47 % mIoU, 2.6× AP50 on the baseline head's own protocol** (the 9071415 row, the current multi-view
best; its per-frame numbers also rise to 0.643 / 0.667), with no post-hoc matching or fusion.

The two ablation rows (2026-07-29, docs/MASKDINO.md §7.4.1) localise the result: **cross-frame
attention is worth 0.183 bundle AP50** — the only individually-decisive component found anywhere
in this track — and **bundle features are worth 0.147**. The same bundle features are a *negative*
for per-frame accuracy (§2), so multi-view consistency has a measured price: 0.729 single-frame
best vs 0.630 per-frame for the best multi-view model.

### 3.1 The retired baseline head

The bar row above (0.367 / 0.199 at N=190; 0.451 / 0.294 per-frame) is the best of a retired
family of hand-rolled DETR-style heads — a query-initialisation study on the same frozen backbone
and the same multi-view supervision. It is kept as the **single historical bar** and nothing else;
the per-variant narrative, tables and verdicts are archived in `docs/old/ARMS_SUMMARY.md` and are
not part of the current story.

One conclusion from it still matters, because MaskDINO **inverts** it: that head got *worse* with
more data (0.367@190 → 0.350@490), which at the time read as "the dataset is not the bottleneck".
The MaskDINO scaling curve says otherwise — the old head was **architecture-limited, not
data-limited**, and the old scaling result was a property of that head, not of the task.

## 4. Ground-truth quality — why the older numbers do not transfer

Switching from SAM3-generated GT to official ScanNet GT (2026-07-08) cost about half the honest
AP50 headline. Any pre-2026-07-08 number in `docs/old/` is on the old ruler. Table and reasoning:
`docs/DATASET.md` §1.