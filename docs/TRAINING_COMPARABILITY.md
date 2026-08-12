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
| **this project** | ScanNetv2 train, official 1201 split, frozen VGGT-1B, head-only | ScanNetv2 val-312 (2D per-frame/per-bundle + 3D benchmark) | `docs/RESULTS.md` §6, `docs/MASKDINO.md` §9 |

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
| Infinigen, RE10K (standalone) | only if InsScene-15K's shards prove incomplete | missing | — | — |

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
   `processed_infinigen`, `processed_re10k`, `processed_scannetpp_v2` — the Aria portion is absent
   ("datasets are still being uploaded"). Any replication built on it is **partial** and must say so.
   The full 522 GB / 1565-file mirror is on work as of 2026-08-08; re-checked at mirror time and
   still no Aria/ASE directory — this fact does not change once the download completes, only the
   date it was last confirmed does.
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
