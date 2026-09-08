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
| **Aria Synthetic Environments, annotated** | replicating IGGT's training data (and touching FAST3DIS's source) | **HELD — 1 000 scenes fetched and built 2026-08-31 (§6.7).** The public release downloads **by scene range** and its per-scene GT includes 2D instance segmentation; the pilot cost ~223 GB and arm I-ase trains on it. FAST3DIS's own 40 % scene list stays unpublished, so their *training set* remains unreproducible at any size | 223 GB (measured, 1 000 scenes) | **1 764 inodes/scene**, measured by the pilot's gate |
| Infinigen, RE10K (standalone) | only if InsScene-15K's shards prove incomplete | **not needed** — both are inside the mirror and both are annotated. The RE10K row here used to read "missing"; it was **stale from 2026-08-24**, when the masks were found under a *sibling* directory the original survey never looked at (`processed_re10k/sam2_results/<scene>/auto_masks.json`, 5127 of 5138 scenes). See `docs/MULTIDATASET.md` §1.3 | 0 | 0 |

**Storage discipline** (`docs/DATASET.md` §5.1): scratch is quota'd on **file count** (1.0 M soft /
1.5 M hard), currently 250 462 used. Every build above materialises its tree in `$TMPDIR` and lands
**only tars** on work, so the scratch inode cost of this entire programme is **zero**.

## 5. What cannot be resolved — state these wherever the comparison appears

1. **FAST3DIS's training set is not reproducible at any scale.** 9.2 TB, *and* the sampled 40 %
   scene list is unpublished — so even a subset would not be "their data". Every FAST3DIS comparison
   remains a cross-training-set comparison. This is permanent, not a budget problem.
2. **The ASE copy on this cluster has no annotations** (§3.1), so an ASE arm needed a fresh
   download under Project Aria terms, not a local read. **That is DONE and this point is closed**
   (§6.7): 1 000 annotated scenes were fetched 2026-08-31, arm I-ase trained on them and scored
   2026-09-03. Do not repeat "ASE is out of reach" — the permanent item is point 1's scene list,
   not the data. What remains on this axis is **scale**, which is a budget question, not a
   reproducibility one: 1 000 scenes against IGGT's share of ~100 k.
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
| **I-ase** | 〃 **+ ASE@1000** | **4819** | 18 | 86 742 | **12510960** | **IGGT's training set, COMPLETE** — added 2026-09-02, the row to quote (§6.7) |
| **I** | ScanNet++ + Infinigen + RE10K@1500 | 3819 | 22 | 84 018 | 11839134 | IGGT's training set **minus ASE** — and no target-benchmark data at all |
| **I-gt** | ScanNet++ + Infinigen | 2319 | 36 | 83 484 | 11839135 | the same, minus the SAM2-supervised source: a **GT-only** zero-shot row |

Together with the two arms already running they form a **complete 2 × 2** at ~84 k steps and one
learning rate — `{± ScanNet} × {± RE10K}`:

| | no RE10K | + RE10K@1500 |
|---|---|---|
| **+ ScanNet** | A-long′ 11830142 (3520) | D-long 11830140 (5020) |
| **no ScanNet** | **I-gt 11839135 (2319)** | **I 11839134 (3819)** |

so "how much of our ScanNet lead is ScanNet training data" and "what does SAM2 supervision add"
are each **one variable**, measured twice.

**What arm I-ase is and is not.** It is **IGGT's mixture in full** — all four sources — with RE10K
capped at 1500 of 5127 scenes and ASE at 1000 of ~100 k, both for memory (`--cpus-per-task=26`;
uncapped RE10K alone is ~550 GB of feature cache). Two differences run in *our* favour and must be
stated: we drop the 50 `nvs_sem_val` ScanNet++ scenes from training so our ScanNet++ column is
honest, which IGGT does not do (§1.1 consequence 3); and our backbone stays **frozen** where IGGT
finetunes VGGT. Two run against us: IGGT trains on 8 × A800 for 2 days (~16 GPU-days) against our
~0.8, and **on ~100 × the ASE**. The mixture is matched; the **scale is not**, and that is now the
whole of the remaining gap on this axis. FAST3DIS's 40 % scene list is the permanent item (§5.1).

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

### 6.5 Status — every axis, every job (2026-08-26; training axis CLOSED 2026-09-03)

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
| train data, IGGT | InsScene-15K (ASE + Infinigen + RE10K + ScanNet++) | **arm I-ase** = the same four sources, RE10K@1500 + ASE@1000 — **their mixture, COMPLETE** | **MIXTURE MATCHED 2026-09-03, scale NOT** — without ScanNet we score **0.009 / 0.032 / 0.301** against IGGT's 0.028 / 0.112 / 0.287: **~3.5× behind on AP50, AHEAD on AP25**, at 17.42 views to their 50 (`docs/RESULTS.md` §5.6, `docs/MULTIDATASET.md` §12.4). What is left unmatched is **scale** (1 000 ASE scenes vs ~100 k), the frozen backbone and the compute. Adding ASE was worth ×1.4 AP50 over arm I |
| train data, FAST3DIS | ASE only → ScanNet zero-shot | **arm I-ase** never trains on ScanNet and now holds ASE — but as one of four sources, not alone | **MEASURED 2026-09-03** — same numbers as the row above; only the competitor changes. 0.032 vs their 0.096 AP50 = **~3× behind**; **level on AP25** (0.301 vs 0.316). Still not matched: they train on ASE *alone* and their 40 % scene list is unpublished (§5.1) — permanent |
| train data, SegVGGT | official ScanNetv2 1201 | identical | **MATCHED since 2026-08-02 — the one training-matched comparison in the project, and we are ×2.8 behind on it** (§6.6) |
| train data, ASE itself | 9.2 TB / ~100 k scenes, unpublished 40 % scene list | **a 1000-scene pilot is ON DISK and TRAINED ON** — `insscene2d_ase.tar.zst`, 1 000 scenes / 31 897 frames / 78 347 instances | **CLOSED 2026-09-03** (§6.7): fetched and built 2026-08-31 (12262949 + 12264266, zero failures), arm **I-ase** trained on it (12510960) and scored 8/8 cells (12511027). Worth **×1.4 AP50** on the competitor cell. The *data* is public and downloads by scene range; the *scene list* never will be |
| ScanNet200 supervision | SegVGGT trains a 200-class checkpoint | our 2D GT is 19-class | **open, costed** — todo 6m |
| training compute | ~16 GPU-days | ~0.8 GPU-days, frozen backbone | **not matchable; state it** (§6.4) |


### 6.6 The reading this whole section produces — put it before the numbers, not after

§6.2 and §6.5 measured the training axis on all three competitors. Collected, they say one thing:

| competitor | training axis | our arm | result |
|---|---|---|---|
| **SegVGGT** | **matched** — the same official 1201 split | our own headline runs | **×2.8 behind** (posed, after the ×2.3 bridge; `docs/RESULTS.md` §8.3) |
| **FAST3DIS** | approximated — we removed ScanNet and now hold ASE, but they train on ASE *alone* and their scene list is unpublished | **arm I-ase** | **~3× behind** on AP50 (0.032 vs 0.096); **LEVEL on AP25** (0.301 vs 0.316) |
| **IGGT** | **their mixture, COMPLETE** — ScanNet++ + Infinigen + RE10K + ASE, at 1 % of their ASE scale | **arm I-ase** | **~3.5× behind** on AP50 (0.032 vs 0.112); **AHEAD on AP25** (0.301 vs 0.287) |

> **Wherever the training data is matched or approximated, we are behind at 0.5 IoU. The lead in
> `docs/RESULTS.md` §8.2 exists in the one configuration where we train on the evaluation domain
> and the competitor does not.**

**One thing changed on 2026-09-03 and it is worth stating precisely.** Arm I-ase completes IGGT's
mixture, and on **AP25** the gap to both published rows is gone — 0.301 against 0.316 / 0.287, at
**17.42 views to their 50**. That is the first column on which a no-ScanNet arm of this project
reaches a published number. It does **not** generalise: AP50 is still ~3×, and the arm holds 1 000
ASE scenes against ~100 k. Quote it as *"level at the coarse-localisation bar, 3× behind at 0.5
IoU"*, never as *"matched"*.

That sentence is not a retraction of §8.2 — that row is genuinely matched on evaluator, bridge,
label setting and view budget, and a strictly frozen backbone at ~0.8 GPU-days beating two adapted
ones is a result. It is an **ordering** rule: a reviewer forms this sentence unprompted, so it is
stated first and the lead second. The supervisor deck was reordered on 2026-08-31 to do exactly
that (`docs/slides/supervisors_2026-08-27.md`: the training axis is now slide 8, the headline
slide 9, the matched-axes audit slide 10).

⚠ And the counter-statement travels with it, because it is equally true: **none of this shows the
recipe loses at equal data.** Arm I has no ASE at all — 3819 scenes against ~100 k, frozen against
adapted, ~0.8 against ~16 GPU-days. **Arm I-ase (job 12510960, since 2026-09-02) closes the
"no ASE at all" half of that sentence and none of the rest**; until it lands the table above is
unchanged, and after it lands the scale gap (1 000 ASE scenes vs ~100 k), the frozen backbone and
the compute gap all still stand. Supportable: *"we cannot match their training setting, and
without ScanNet we are well behind"*. Not supportable: *"our method loses at equal data"*.

### 6.7 ASE — costed 2026-08-27, downloaded 2026-08-31, MEASURED 2026-09-03 (todo 6n CLOSED)

§5.2 said the ASE copy on this cluster has no annotations, and §4 called a fresh download
"missing and out of reach". The first is still true; the second was too strong, and the correction
matters because ASE is the single missing ingredient of both competitors' training sets.

**What is now in the repo:**

| piece | what it does |
|---|---|
| `slurm/download_ase.py` | the official chunk protocol (10 scenes per `<set>_chunk_<id:07d>.zip`, sha1 from the CDN metadata) **plus** resume via markers on work, a self-imposed time budget, and an **inode** report — the number todo 6n gates scaling on |
| `slurm/fetch_ase.sh` | the driver: fetch → gate → probe → build → pack, **in blocks of 100 scenes** so it never holds 230 GB at once (`--tmp` is 60 GB, not 400). Only one tar leaves the node |
| `--source ase` in `slurm/build_insscene2d.py` | reads `<scene>/{rgb/vignette%07d.jpg, instances/instance%07d.png}` into the same `color/` + `instance/` + `manifest.json` layout every other source writes, ids remapped **once per scene** |
| `tests/test_ase_fetch.py`, `tests/test_insscene2d.py` | 26 + 53 CPU checks, no cluster data |

**Two decisions inside it that are not obvious and must not be silently changed:**

1. **Frames are rotated to upright.** ASE stores them in the Aria sensor's orientation, 90° off;
   the tutorial rotates by −90 to look at them. We never read ASE's calibration here, and every
   other source in the mixture is upright, so the builder rotates rgb and ids through the *same*
   numpy call. `--no-upright` turns it off.
2. **The room-shell cap is NOT inherited.** ASE ships no id→name table, so the shell can only go
   by area, as RE10K's does — but RE10K's 0.30 was *measured on RE10K*. `--probe` reports ASE's
   own area distribution and what each candidate cap would remove; the driver runs it on the
   first block and `PROBE_ONLY=1` stops there. Pick the cap off that table before the first
   training run (docs/MULTIDATASET.md §1.4 is the precedent).

**The one manual step was a signature, and it was given on 2026-08-31.** The per-chunk CDN urls
arrive only after the Project Aria dataset licence is accepted at projectaria.com/datasets/ase —
the account holder's act. It was accepted, the json landed at
`/cluster/work/igp_psr/niacobone/distillation/dataset/ase/ASE_cdn_urls.json`, and the pilot ran the
same afternoon.

**What the pilot actually produced** (jobs **12262949** `PROBE_ONLY=1`, 16 min, and **12264266**
fetch → gate → probe → build → pack, 2 h 41 — both COMPLETED):

| | |
|---|---|
| scenes | **1 000** (ids 0–999), 10 blocks of 100, every block `ok:10 skip:0 fail:0 missing:0` |
| raw download | ~223 GB, **0.223 GB** and **1 764 inodes** per scene — the inode gate the driver prints |
| built | **31 897 frames, 78 347 instances**, 32 frames/scene, `upright: true` |
| tar | `<work>/dataset/insscene2d/insscene2d_ase.tar.zst`, **1.34 GB**, the only thing that left the node |

**The room-shell cap turned out to be a no-op, which is itself the answer to decision 2 above.**
The build used `max_area_frac 0.3`; `PROBE_ase_0_999.json` shows that over the 60 probed scenes the
largest instance covers at most **31.6 %** of a frame (median of the per-scene maxima **0.163**), so
0.3 removes **1 instance in 4 940 (0.02 %)**, 0.2 removes 0.49 %, 0.1 removes 2.29 %. **ASE's
rendered ground truth has no wall/floor mega-instance** — RE10K's SAM2 failure mode does not exist
here — so any cap ≥ 0.2 is equivalent and there is nothing to re-pick or rebuild.

**The arm — job 12510960 (chain 12511027), DONE 2026-09-03.**
`SOURCES='scannetpp infinigen re10k ase'`, `CAP_RE10K=1500` → **4 819 scenes**, 18 epochs =
**86 742 steps**, lr 5e-5, `--anchor_3d`, 26 CPUs / 416 GB. Single variable against arm I
(3 819 scenes, 84 018 steps): **+ASE**. The step budget is **+3.2 %** rather than matched to the
decimal, deliberately in the generous direction — this workstream twice read "more data hurts" off
an under-budgeted larger mixture (`docs/MULTIDATASET.md` §9 reading 1, §10.3 reading 2). It did not
end up mattering: the effects are 40–100 %, an order of magnitude above the differential.

**What ASE was worth** (class-agnostic, final `checkpoint.pth`, 8/8 cells, 0 failed scenes; full
matrix in `docs/MULTIDATASET.md` §12.4):

| cell | arm I | **arm I-ase** | AP50 |
|---|---|---|---|
| ScanNetv2 unposed — **the competitor cell** | 0.005 / 0.023 / 0.251 | **0.009 / 0.032 / 0.301** | **×1.4** |
| ScanNetv2 posed | 0.018 / 0.063 / 0.399 | 0.032 / 0.106 / 0.479 | ×1.7 |
| ScanNet200 posed | 0.030 / 0.086 / 0.364 | 0.061 / 0.149 / 0.445 | ×1.7 |
| Replica posed | 0.005 / 0.023 / 0.278 | 0.021 / 0.082 / 0.371 | **×3.6** |

**ASE helps on every cell with signal, and MORE out of domain than in** — Replica posed ×3.6
against ScanNetv2's ×1.4–1.7. It is the first source in this project whose benefit grows with
distance from the training domain, which is what rendered, layout-diverse, sensor-free synthetic
data should do.

**What the pilot bought, and what it did not.** It turned arm I from "IGGT's mixture minus ASE"
into the **complete** IGGT replication, so §6.6's IGGT row is now a *method* comparison rather than
a data one — and on **AP25 the published gap closed** (0.301 vs FAST3DIS 0.316, IGGT 0.287) at
17.42 views to their 50. It did **not** close AP50 (~3× behind), it did **not** match scale
(1 000 scenes vs ~100 k), and it does **not** reproduce FAST3DIS's training set at any download
size: their sampled 40 % scene list is unpublished (§5.1). Say "ASE scenes 0–999", never
"FAST3DIS's training data".
