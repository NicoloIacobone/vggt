# Training-setting comparability — what each competitor trains on, and what we can match

**Companion to `docs/RELATED_WORK.md`, which settles the *evaluation* side.** That file answers
"is this number scored the same way as ours" (two 3D protocols, class-aware vs class-agnostic, the
posed/unposed bridge). This file answers the question that was never asked: **is this number
*trained* the same way as ours.** It is not, and the differences are larger than the protocol ones.

Verified 2026-08-07 against the paper full texts and the released repos. Every row cites where it
came from; anything unsourced is marked as such.

## 1. The matrix

| | **trained on** | **evaluated on** | source |
|---|---|---|---|
| **SegVGGT** | **ScanNetv2 train (1201 scenes)**, and separately **ScanNet200 train** — same scenes, 200-class label space, **two checkpoints, no shared pretraining**. 2–24 frames/scene, 518 px long edge, LoRA r=32, lr 2e-4 new / 6e-5 pre-existing, 8×A100 ≈ 2 days per dataset, ≤48 images/batch | ScanNetv2 val + ScanNet200 val (Table 1, full val); **ScanNet++ zero-shot** (Table 2) | §4.1 *"We train the models separately on ScanNetv2 and ScanNet200 using 8 NVIDIA A100 GPUs, taking approximately 2 days per dataset"*; §4.3; `configs/eval/segvggt_scannetv2.yaml` |
| **FAST3DIS** | **Aria Synthetic Environments ONLY.** *"Our model is trained exclusively on the Aria Synthetic Environments Dataset […] we sampled 40% of the scenes to form our training set"* (of 100 000+ scenes). No real data at all. | **ScanNetV2, ScanNet++, Replica — all three zero-shot**, 50 uniformly sampled views/scene, **class-agnostic**, Sim(3)+ICP alignment | §4.1 Datasets; §4.1 *"we uniformly sample 50 views along the camera trajectory to reconstruct and evaluate each scene"*; §4.4 *"In the class-agnostic setting, we ignore the semantic class labels"* |
| **IGGT** | **InsScene-15K**, curated from **Aria (ASE) + Infinigen + RE10K + ScanNet++**. Initialised from VGGT, then finetuned once. 8×A800 × 2 days, 1–12 frames/scene, 24 images/batch, lr 1e-6 backbone / 1e-5 heads | ScanNet + ScanNet++, **10 randomly selected scenes each, 8–10 images per scene** — spatial tracking, reconstruction, open-vocab semantics. **No AP table of its own.** | §2 InsScene-15K; §4 *"we randomly select 10 scenes and sample 8–10 images per scene"*; §A.3 *"initialized with weights from VGGT […] and fine-tuned on the InsScene-15K dataset"* |
| **this project** | ScanNetv2 train, official 1201 split, frozen VGGT-1B, head-only. **Separately labelled extra-data rows** add ScanNet++ and Infinigen from InsScene-15K (arms A/C, `docs/MULTIDATASET.md` §10) and, separately again, RE10K's **SAM2-generated** masks (arm D, §11) | ScanNetv2 val-312 (2D per-frame/per-bundle + 3D benchmark); the 4-benchmark matrix for the extra-data arms | `docs/RESULTS.md` §6, §7.5, `docs/MASKDINO.md` §9 |

### 1.1 Three consequences

1. **We already train on SegVGGT's ScanNetv2 training data.** Their ScanNetv2 checkpoint uses the
   official 1201 split, which is exactly what every official-split run since 2026-08-02 uses. That
   comparison needs **no retraining** — it needs the evaluation matrix completed.
2. **FAST3DIS trains on zero real data**, and every one of its ScanNet/ScanNet++/Replica numbers is
   zero-shot. Our ScanNet-trained model is therefore *advantaged* on ScanNetv2 and *disadvantaged*
   nowhere else — the opposite of the implicit assumption behind quoting the two side by side.
3. **IGGT's training set contains ScanNet++**, which is also one of its evaluation datasets — the
   thing SegVGGT calls out (§4.3: *"while the baseline methods are trained on massive datasets that
   explicitly include ScanNet++ training scenes, our model is trained solely on ScanNet200"*).

## 2. Field practice on "pretrain on everything, then finetune on the target" — from the papers

The question was whether training on many datasets and then finetuning on the evaluation dataset is
accepted practice. **Read from the papers rather than assumed, the answer is no — and one paper
treats it as a defect in its baselines.**

| paper | multi-dataset pretraining? | finetuned on the eval benchmark? |
|---|---|---|
| SegVGGT | **no** — two independent single-dataset trainings | **no**, and it argues the point: ScanNet++ is scored zero-shot *specifically* to contrast with baselines whose training data includes ScanNet++ (§4.3) |
| FAST3DIS | **no** — one synthetic dataset | **no** — all three benchmarks zero-shot (§4.1) |
| IGGT | **yes**, one curated mixture (InsScene-15K) | **no per-benchmark finetuning** — a single VGGT-initialised finetune on the mixture, then evaluated (§A.3) |
| MaskDINO (2D lineage) | **yes**, Objects365 → COCO — but always as a **separately labelled row** ("MaskDINO+O365 data+1.2× larger image"), and the README fences the clean rows: *"we present the clean models that do not use extra detection data or tricks"* | n/a |

**The operative norm: one training run, then evaluate — and when extra data is used, it gets its own
labelled row rather than being folded into the headline.** Our single ScanNet-trained model
evaluated across four benchmarks is therefore *already* the shape the field expects. What is missing
is not a pretrain-then-finetune pipeline; it is the breadth of the evaluation.

## 3. What is already on the cluster

### 3.1 The `.hdf5` packs are depth data, not annotations

`/cluster/work/igp_psr/csakarid/data/3D_datasets` (~3.6 TB) was opened with `h5py` and walked, not
guessed at (our `myenv` has no h5py; `/cluster/work/igp_psr/nedela/litept-env/bin/python` does).

> **Every pack there is RGB + depth only. None carries instance, semantic or pose annotations.**

| pack | GB | structure | usable here? |
|---|---|---|---|
| `scannet.hdf5` | 11.8 | `<split>/<scene>/{color/*.jpg, depth/*.png, intrinsic/*.txt}` | no GT |
| `ScanNetS.hdf5` | 64 | 1616 scans, `color/` + `depth/` only, **no poses** | no GT |
| `ScanNetpp.hdf5` / `_F` / `_viz` | 43.8 / 78.4 / 10.3 | `<scene>/{iphone,dslr}/*.{jpg,png}` | no GT |
| **`ASE.hdf5`** | **534** | 20 002 scenes, `<scene>/*.{jpg,png}` — **no instance maps** | **cannot supply FAST3DIS's supervision** |
| `Matterport3D`, `hypersim/*`, `2D3DS`, `HM3D`, `Taskonomy`, `Gibson`, `ARKit*`, `coco2017`, … | — | same rgb+depth shape | no GT |

This is a depth/MVS training corpus. Treat the whole directory as unusable for instance segmentation.

### 3.2 The usable data is in other users' directories

| what | where | contents | verified |
|---|---|---|---|
| **ScanNet 3D annotations, 1513 scans** | `/cluster/work/igp_psr/nedela/scannet_raw/scans/` | `.aggregation.json` + `_vh_clean_2.0.010000.segs.json` + `_vh_clean_2.ply`, 12 GB | yes — the same three files our `scannet_3d_gt_val312.tar.zst` holds |
| **ScanNet++ v2, all 906 semantic-split scenes** | `/cluster/work/igp_psr/nedela/scannetpp_data/data/` | per scene `scans/{mesh_aligned_0.05.ply, segments.json, segments_anno.json}` + `iphone/{rgb.mkv, depth.bin, pose_intrinsic_imu.json}`, ~1.2 GB/scene | yes — **856/856 train and 50/50 val present, zero missing**; splits + `metadata/semantic_benchmark/top100_instance.txt` alongside |
| COCO 2017 | `/cluster/scratch/niacobone/coco` | train2017 + val2017 + annotations, 20 GB / 123 293 files | present |

> ⚠ **Both ScanNet trees belong to another user and can vanish without notice.** Any build must copy
> what it needs into our own tar and never read that tree at training or evaluation time.

## 4. What is genuinely missing — size AND file count

| dataset | needed for | status | size | file count |
|---|---|---|---|---|
| **ScanNet200 val GT** | SegVGGT's 2nd benchmark | **not missing** — derivable from `scannet_3d_gt_val312.tar.zst` + a raw→200-class label map | **0** | **0** |
| **ScanNet++ val-50 3D GT + frames** | SegVGGT, FAST3DIS, IGGT | buildable from nedela's tree, no download | ~7 GB (2 tars) | 2 inodes on work; ~10 k node-local |
| **Replica (8 scenes: room0-2, office0-4)** | FAST3DIS's 3rd benchmark | **DONE 2026-08-08** — `dataset/replica/{replica_3d_gt_8,replica_frames_8}.tar.zst`; CC-BY-NC-4.0 | 372 MB + 417 MB (789 MB total — the 15–25 GB estimate assumed unsampled frames; 50/scene + zstd is far smaller) | 2 inodes on work; 0 loose on scratch |
| **InsScene-15K** | replicating IGGT's training data | **DONE 2026-08-08** — `dataset/insscene15k/`, mirrored as-is (not unzipped), Apache-2.0. **Still partial**: Aria/ASE not uploaded upstream as of this date (re-checked, unchanged since 2026-08-07) | 522.07 GB | **1565 files** (not ~120 — `processed_infinigen` alone is 1468 small per-scene zips, not one shard per subset) |
| **Aria Synthetic Environments, annotated** | replicating FAST3DIS's training data | **MISSING and out of reach** — 23 TB / 100 000 scenes / 58 M images; their 40 % ≈ 9.2 TB | 9.2 TB | ~23 M frames |
| Infinigen, RE10K (standalone) | only if InsScene-15K's shards prove incomplete | **not needed** — both are inside the mirror and both are annotated. The RE10K row here used to read "missing"; it was **stale from 2026-08-24**, when the masks were found under a *sibling* directory the original survey never looked at (`processed_re10k/sam2_results/<scene>/auto_masks.json`, 5127 of 5138 scenes). See `docs/MULTIDATASET.md` §1.3 | 0 | 0 |

**Storage discipline** (`docs/DATASET.md` §5.1): scratch is quota'd on **file count** (1.0 M soft /
1.5 M hard), currently 250 462 used. Every build above materialises its tree in `$TMPDIR` and lands
**only tars** on work, so the scratch inode cost of this entire programme is **zero**.

## 5. What cannot be resolved — state these wherever the comparison appears

1. **FAST3DIS's training set is not reproducible at any scale.** 9.2 TB, *and* the sampled 40 %
   scene list is unpublished — so even a subset would not be "their data". Every FAST3DIS comparison
   remains a cross-training-set comparison. This is permanent, not a budget problem.
2. **The ASE copy on this cluster has no annotations** (§3.1), so an ASE arm would need a fresh
   download under Project Aria terms, not a local read.
3. **InsScene-15K appears incomplete.** Its HuggingFace tree currently exposes only
   `processed_infinigen`, `processed_re10k`, `processed_scannetpp_v2` — all three of which we now
   train on, with RE10K's rows carrying the **SAM2-supervised** caveat — but the Aria portion is
   absent
   ("datasets are still being uploaded"). Any replication built on it is **partial** and must say so.
   The full 522 GB / 1565-file mirror is on work as of 2026-08-08; **re-checked 2026-08-24 against
   the live HuggingFace tree — still exactly those three folders, still no Aria/ASE directory.**
   This fact does not change once the download completes, only the date it was last confirmed does.
4. **FAST3DIS never states which scenes it evaluates** on any of its three datasets — only "50
   uniformly sampled views". Our numbers will be on official val-312 / `nvs_sem_val`-50 / the
   standard 8 Replica scenes; do not claim identical evaluation sets.
5. **SegVGGT reports two different protocols in one paper.** Table 1 is full-val
   (ScanNetv2 50.4/71.7/87.0); Table 2 is **10 randomly sampled val scenes** (ScanNet++
   13.3/33.9/56.4). Never put a Table 1 and a Table 2 number in the same row.
6. **IGGT's AP triple is FAST3DIS's re-evaluation of IGGT**, not IGGT's own paper — see
   `docs/RELATED_WORK.md`. IGGT publishes no ScanNet AP at all.
7. **Our 19-class head cannot be class-aware** on ScanNet200 (200 classes), ScanNet++ (~84 instance
   classes) or Replica. Those three are class-agnostic-only for us — which is also FAST3DIS's and
   IGGT's own reporting setting, so it is a fair column, not a concession.
8. **Licences.** ScanNet / ScanNet++ require the signed TOS (held). Replica is **CC-BY-NC-4.0** —
   research use fine, redistribution not. InsScene-15K is Apache-2.0. ASE requires Project Aria terms.

## 6. The competitor-matched programme (opened 2026-08-26)

§1–§5 audited the mismatch. This section is what is being **done** about it: every axis on which
a published row and one of ours differ, with the state of each. It is the home of the
"same setting, same epochs, same training and validation data" question — read it with
`docs/todo.md` 6k/6l, which track the jobs.

### 6.1 Two things that were checked before anything was launched

**(a) "For IGGT only Aria/ASE is missing" — TRUE, with one structural addition.** IGGT trains on
InsScene-15K = ASE + Infinigen + RE10K + ScanNet++ (§1). The mirror on work holds three of those
four and the fourth is not published (§4, §5.3). But the gap was never only ASE: **every arm we
have ever trained also contains ScanNet**, which IGGT's does not — so no existing checkpoint is
IGGT-matched, whatever the mirror holds. Matching IGGT means *removing* ScanNet as well as adding
its three sources, which is what arm **I** does (§6.2).

**(b) "Another competitor uses a different setting with geometric GT" — TRUE, and it is SegVGGT.**
Its evaluator projects the GT benchmark cloud into each view with ScanNet's **GT poses and sensor
depth**, quoting their own §: *"We utilize the ground-truth depth maps and camera poses during this
mapping stage for fair comparison."* We already implement exactly that bridge as
`--transfer_mode gt_projection`, oracle-licensed at round-trip purity 0.9999, and report it as its
own column (`docs/RESULTS.md` §5.1, §8.3). **That axis needs no new work** — it is matched.

### 6.2 The zero-shot arms — matching what the competitors TRAIN on

The single largest remaining mismatch is not the protocol, it is that **FAST3DIS and IGGT never
train on ScanNet and we always do**. Every "we lead FAST3DIS/IGGT" row in `docs/RESULTS.md` §8.2
is therefore *favourable to us on the training axis*, and a reviewer sees that before anything
else. Two arms close it, both launched 2026-08-26, both `--class_agnostic --anchor_3d`, lr 5e-5,
step-matched to the arms already in flight, and both scored on the same 4 × 2 matrix:

| arm | train sources | scenes | epochs | steps | job | what it matches |
|---|---|---|---|---|---|---|
| **I** | ScanNet++ + Infinigen + RE10K@1500 | 3819 | 22 | 84 018 | 11839134 | **IGGT's training set minus ASE** — and no target-benchmark data at all |
| **I-gt** | ScanNet++ + Infinigen | 2319 | 36 | 83 484 | 11839135 | the same, minus the SAM2-supervised source: a **GT-only** zero-shot row |

Together with the two arms already running they form a **complete 2 × 2** at ~84 k steps and one
learning rate — `{± ScanNet} × {± RE10K}`:

| | no RE10K | + RE10K@1500 |
|---|---|---|
| **+ ScanNet** | A-long′ 11830142 (3520) | D-long 11830140 (5020) |
| **no ScanNet** | **I-gt 11839135 (2319)** | **I 11839134 (3819)** |

so "how much of our ScanNet lead is ScanNet training data" and "what does SAM2 supervision add"
are each **one variable**, measured twice.

**What arm I is and is not.** It is IGGT's mixture minus ASE, with RE10K capped at 1500 of 5127
scenes for memory (`--cpus-per-task=22`; uncapped it is ~550 GB of feature cache). Two differences
run in *our* favour and must be stated: we drop the 50 `nvs_sem_val` ScanNet++ scenes from training
so our ScanNet++ column is honest, which IGGT does not do (§1.1 consequence 3); and our backbone
stays **frozen** where IGGT finetunes VGGT. One runs against us: IGGT trains on 8 × A800 for 2 days
(~16 GPU-days) against our ~0.8. ASE remains permanently out of reach (§5.1, §5.2).

**Validation data.** The val ruler is the official ScanNet v2 312 for every arm, including the two
that never see ScanNet in training — there it is a **zero-shot** read-out. That required one driver
change (`slurm/train_maskdino_multi.sh` stages the val tar independently of `SOURCES`; 4 new checks
in `tests/test_train_maskdino_multi_sh.sh`). Note that `checkpoint_best_bundle.pth` is *selected* on
that ruler, so for a strictly selection-leak-free row score the final `checkpoint.pth` alongside it.

### 6.3 Views per scene — the last unmatched evaluation axis

| benchmark | their views | ours | state |
|---|---|---|---|
| ScanNet++ | FAST3DIS 50 | **50** | **already matched** — our frames tar is 50/scene by construction |
| Replica | FAST3DIS 50 | **50** | **already matched** |
| ScanNetv2 / ScanNet200 | FAST3DIS 50; SegVGGT every 20th frame (~75–120) | **17.42** | the gap — `scannet_frames_25k` is every 100th frame |
| queries kept | SegVGGT 600 | 100 | **measured neutral** (0.138 → 0.140, `docs/MASKDINO.md` §9.8.1); struck as an explanation |

So the mismatch was confined to the two ScanNet columns, and **it is closed since 2026-08-27**:
at their own 50 views the lead widens (`docs/RESULTS.md` §5.4). The "a third of their views"
caveat is retired. `docs/DATASET.md` §2.3 builds the dense export that
closes it (job 11839821 → 11840376); scored at `--num_frames 50` it is FAST3DIS's budget exactly,
and at full stride it is SegVGGT's sampling. **Run the 17-frame cell on the dense tar too**: its
jpegs are the original `.sens` payloads while the 25k export re-compressed them (~102 KB vs
~260 KB for the same frame), so only a dense-vs-dense comparison isolates view count.

### 6.4 What remains permanently unmatched

Beyond §5's list: **training compute**. SegVGGT and IGGT each spend ~16 GPU-days (8 GPUs × 2 days);
our arms spend ~0.8, head-only on cached frozen features. That is a property of the design, not an
oversight, and the step-budget axis is measured *inside* our own block (A-long ⇄ A-long′ ⇄ C-long′,
`docs/MULTIDATASET.md` §10.5). Do not present our numbers as compute-matched.

### 6.5 Status — every axis, every job (2026-08-26)

| axis | competitor's setting | ours | state |
|---|---|---|---|
| evaluator | official ScanNet 3D instance benchmark | vendored, same options | **matched** since 2026-08-01 |
| bridge, unposed | FAST3DIS / IGGT: predicted geometry + Sim(3)+ICP | same | **matched** |
| bridge, posed ("geometric GT") | SegVGGT: GT poses + sensor depth | `--transfer_mode gt_projection`, oracle 0.9999 | **matched** |
| label setting | FAST3DIS / IGGT class-agnostic; SegVGGT class-aware | both computed per run | **matched** |
| benchmarks | ScanNetv2 / ScanNet200 / ScanNet++ / Replica | all four | **matched** (todo 6d) |
| kept queries | SegVGGT 600 | 100 | **measured neutral** |
| views, ScanNet++ / Replica | 50 | 50 | **matched** |
| views, ScanNetv2 / ScanNet200 | 50 (FAST3DIS) / ~75–120 (SegVGGT) | 17.42 → **50 on demand** | **MATCHED 2026-08-27** — dense export built, 7 cells scored; at 50 views the lead *widens* and the lever saturates (`docs/RESULTS.md` §5.4) |
| train split, SegVGGT | official ScanNetv2 1201 | identical | **matched** since 2026-08-02 |
| train data, IGGT | InsScene-15K (ASE + Infinigen + RE10K + ScanNet++) | **arm I** = the same minus ASE, RE10K@1500 | **IN FLIGHT** — 11839134 → 11839151 |
| train data, FAST3DIS | ASE only → ScanNet zero-shot | **arms I / I-gt** never train on ScanNet | **IN FLIGHT** — 11839134 / 11839135 |
| train data, ASE itself | 9.2 TB, unpublished scene list | — | **permanently out of reach** (§5.1–5.2) |
| ScanNet200 supervision | SegVGGT trains a 200-class checkpoint | our 2D GT is 19-class | **open, costed** — todo 6m |
| training compute | ~16 GPU-days | ~0.8 GPU-days, frozen backbone | **not matchable; state it** (§6.4) |
