# Commands — the full catalogue

Every runnable recipe in the repo, with the caveats that make each one correct. `CLAUDE.md` keeps
only the handful used daily; everything else lives here. Read the section you need, not the file.

A virtualenv lives in-repo at `myenv/` — use `myenv/bin/python`. Runs on a GPU cluster node;
matplotlib must stay headless (`Agg`). SLURM job logs go to `slurm/logs/` (gitignored) — never let
them accumulate in the repo root.

---

## 1. Tests

Standalone scripts, not pytest. All CPU-only, no backbone weights needed.

```bash
python tests/test_maskdino_model.py   # MSDeformAttn vs naive ref, pixel decoder, decoder configs,
                                      # box ops, head_config round-trip, initialize_box_type guard,
                                      # 3D-anchor geometry (§8.3: soft-nearest-patch projection
                                      # against hand-computed values, pyramid gather, normalisation)
python tests/test_maskdino_loss.py    # matcher, criterion keys + perfect-prediction zero loss,
                                      # out-of-range GT-label guard
python tests/test_maskdino_train.py   # per-frame GT builder (incl. class drop), per-frame metric
                                      # slicing, 60-step synthetic overfit, --anchor_3d position
                                      # cache (14x14 conf-weighted pooling + order-preserving gather)
python tests/test_maskdino_multiframe.py  # shared-query multi-frame path: cross-frame block,
                                      # bundle GT + index expansion, bundle matcher, S=1
                                      # equivalence, multi-frame overfit, bundle batching/scoring,
                                      # + --anchor_3d (MASKDINO.md §8.3): one anchor per bundle
                                      # projected per view, DN rows left on the 2D path, the flag
                                      # inert when off, Δ(xyz,log r) head really in the graph,
                                      # + --eval_num_frames (§8.4): val bundle width pinnable
                                      # independently of train, scoring width-agnostic per scene
python tests/test_maskdino_viz.py     # figure colouring keyed to identity, not per-frame rank
python tests/test_maskdino_consistency.py  # cross-view consistency metrics (MASKDINO.md §6.6):
                                      # planted perfect/switched cases, degenerate inputs,
                                      # additive bundle_* keys
python tests/test_maskdino_fullres.py # --eval_full_res ruler (MASKDINO.md §6.5): helpers,
                                      # grid-vs-full ruler difference, both eval paths
python tests/test_maskdino_eval3d.py  # 3D ruler (MASKDINO.md §9): PLY/GT builders, Umeyama/ICP,
                                      # unprojection round-trip, votes+majority, the vendored
                                      # official evaluator vs hand-computed APs, synthetic E2E,
                                      # + the gt_projection transfer (§9.10): the 518² squash
                                      # intrinsic, sensor-depth visibility, known-scene round
                                      # trip, knob-naming of --transfer_mode
python tests/test_maskdino_viz3d.py   # 3D viewer colour path (MASKDINO.md §9.7): feature-mode
                                      # fidelity, max-over-views selection, colour stable across
                                      # views, end-to-end on a tiny head
python tests/test_demo_gradio_maskdino.py  # the Gradio glue: checkpoint-kind routing, scene
                                      # dropdown, GT/frame ordering, colouring path (imports the
                                      # demo with VGGT_DEMO_SKIP_BACKBONE=1, no weights downloaded)
python tests/test_dualview3d.py       # synced side-by-side 3D (MASKDINO.md §9.7): filtering
                                      # asserted vertex-for-vertex against the GLB path, panels
                                      # share points, payload round-trip, .ply → HTML
bash tests/test_train_maskdino_sh_lists.sh  # slurm scene-list logic via DRY_RUN: numeric-range
                                      # back-compat, TRAIN_LIST/VAL_LIST split files, filtering
bash tests/test_train_maskdino_multi_sh.sh  # the multi-dataset driver's lists + CAP_*. Its
                                      # regression check runs the script under errexit on purpose:
                                      # DRY_RUN skips the sourced `set -e`, so a plain dry run
                                      # cannot see a silent-abort bug (MULTIDATASET.md §7.1)
bash tests/test_eval_3d_matrix_sh.sh   # the cross-dataset eval grid (§4.1): the 4x2 cells, the
                                      # three ways a checkpoint may be named, and the chain job
                                      # run the way SLURM runs it (spooled copy, foreign cwd)
python tests/test_collect_eval3d_matrix.py  # the eval-matrix collector: default-cell filter,
                                      # --run/--only
python tests/test_coco_rle.py         # the COCO compressed-RLE decoder used to read RE10K's SAM2
                                      # masklets (pure numpy, no pycocotools) — NOT the COCO study
python tests/test_maskdino_tracking_metrics.py  # HOTA/AssA/DetA/IDF1: a switch costs association
                                      # where a miss does not, per-view queries collapse AssA
```

**Every test runs under `myenv/` and none needs backbone weights.** The retired COCO arm's two
tests moved to `legacy/coco/tests/` with the rest of that arm; one of them needed the detectron2
reference env and is no longer part of this suite.

---

## 2. Training

```bash
sbatch slurm/train_maskdino.sh                                 # 50 scenes, ~20k steps
sbatch --export=ALL,N_SCENES=490 slurm/train_maskdino.sh       # epochs auto-scale to hold the budget
sbatch --export=ALL,EXTRA_ARGS='--mask_upsample 2' slurm/train_maskdino.sh
python scripts/train_maskdino.py --train_scenes scene0000_00 --val_scenes scene0080_00 \
    --num_epochs 50 --num_queries 300 --scans_root <scans_root>       # local smoke test
```

### Multi-frame

One query set per bundle of 8 frames (`docs/MASKDINO.md` §8.2); reports the per-bundle (multi-view)
metrics as well as the per-frame ones.

```bash
sbatch --export=ALL,N_SCENES=490,EXTRA_ARGS='--multi_frame --feature_mode bundle' slurm/train_maskdino.sh
```

### `--anchor_3d` (MASKDINO.md §8.3, todo 2d)

The decoder's 2D DAB anchor box becomes a 3D anchor (x, y, z, log r) per query per bundle, read off
VGGT's frozen POINT head at cache time (+0.146 % cache) and soft-projected into each view — no
intrinsics/extrinsics. Needs `--feature_mode bundle`. An **ablation** vs the 2D box (FAST3DIS owns
the mechanism), so it is only ever quoted against a control that differs by this flag alone.

```bash
sbatch --export=ALL,N_SCENES=490,EXTRA_ARGS='--multi_frame --feature_mode bundle --anchor_3d' \
    slurm/train_maskdino.sh
```

### Bundle width (MASKDINO.md §8.4, todo 2e)

`--num_frames` is views **per bundle**; it sat at 8 while the 3D ruler runs the head at S≈17.4.
Widening it also widens the per-bundle metric's object, so `--eval_num_frames` pins the **val**
width and keeps `bundle_*` on the ruler the baseline was measured on. Feature cache is linear in
frames: S=16 × b2 ≈ 230 GB → needs an A100 80 GB.

```bash
sbatch --time=16:00:00 --cpus-per-task=26 --mem-per-cpu=13312 --tmp=90000 --gpus=a100_80gb:1 \
    --export=ALL,DATA_TAR="$DS/scannet_official_gt_1201.tar.zst $DS/scannet_official_gt_val312.tar.zst",\
TRAIN_LIST=data/splits/scannetv2_train.txt,VAL_LIST=data/splits/scannetv2_val.txt,EPOCHS=12,WARMUP=2,\
EXTRA_ARGS='--multi_frame --feature_mode bundle --num_frames 16 --eval_num_frames 8 \
--bundles_per_scene 2 --color_jitter 0.2' slurm/train_maskdino.sh
```

### Official 1201/312 split

`TRAIN_LIST`/`VAL_LIST` override the numeric-range scene selection, and `DATA_TAR` takes a
space-separated list of tars staged into one tree. Needs bigger resources than the script header
(fp16 cache ~110 GB at 1201 scenes; ~58 GB staged) — pass them on the command line. `EPOCHS=12` ≈
28.8k steps at 1201 scenes × 2 bundles (≈ the N=490 recipe budget). The scene-list logic is covered
by `tests/test_train_maskdino_sh_lists.sh` (dry-run, CPU-only).

```bash
DS=/cluster/work/igp_psr/niacobone/distillation/dataset/scannet
sbatch --time=24:00:00 --cpus-per-task=12 --mem-per-cpu=14336 --tmp=90000 \
    --export=ALL,DATA_TAR="$DS/scannet_official_gt_1201.tar.zst $DS/scannet_official_gt_val312.tar.zst",\
TRAIN_LIST=data/splits/scannetv2_train.txt,VAL_LIST=data/splits/scannetv2_val.txt,EPOCHS=12,WARMUP=2,\
EXTRA_ARGS='--bundles_per_scene 2 --color_jitter 0.2' slurm/train_maskdino.sh
DRY_RUN=1 TRAIN_LIST=... VAL_LIST=... bash slurm/train_maskdino.sh   # echo lists/schedule, no data
```

### Multi-dataset arms (`slurm/train_maskdino_multi.sh`; docs/MULTIDATASET.md §5, §10, §11)

A different driver, not a flag on the one above: it stages ScanNet **plus** one tar per
instance-map source and passes the scene lists as `@files` (the 128 KB argv cap, §7.2). Val is
always the official ScanNet 312, class-agnostic — **never move it**.

**Two things must be set by hand at submit time**, and both have cost a run in this workstream:

* `--cpus-per-task`, because the feature cache is the binding constraint and the script's default
  is sized for 1201 scenes. Measured: **44 GB + 60 MB × scenes**. 20 → ScanNet+ScanNet++,
  **26** → the 3520-scene mixture and the ~5020-scene four-source arm, 40–44 → all four uncapped.
* `EPOCHS`, because the derived default is `20000/N_TRAIN` clamped to [6, 40] and returns the
  **floor of 6** for any large mixture. One step = one 8-frame bundle = one scene at b1, so
  steps = N_TRAIN × EPOCHS; A-long's budget is 84 480.

…and a third for anything past the 3520-scene mixture:

* `--learning_rate`. The default **1e-4 diverges** on the 5020-scene four-source mixture — job
  11642516 ran 17 h cleanly and produced garbage, training loss rising from the first epoch after
  warmup and `train_AP50` collapsing 0.211 → 0.006. **5e-5 fixes it at the same data and dose**
  (docs/MULTIDATASET.md §11.3). Re-run the control at the same LR too, or the comparison moves
  two variables.

```bash
sbatch slurm/train_maskdino_multi.sh                                   # the 3-source mixture
sbatch --export=ALL,SOURCES='scannet scannetpp',EPOCHS=40 --cpus-per-task=20 \
    slurm/train_maskdino_multi.sh                                      # arm C-long
sbatch --export=ALL,SOURCES='scannet scannetpp infinigen re10k',CAP_RE10K=1500,EPOCHS=17,\
EXTRA_ARGS='--anchor_3d --learning_rate 5e-5' --cpus-per-task=26 \
    slurm/train_maskdino_multi.sh                                      # arm D-long (§11.4)
DRY_RUN=1 SOURCES='scannet scannetpp infinigen re10k' bash slurm/train_maskdino_multi.sh
```

⚠ **Anything trained with `re10k` in `SOURCES` is SAM2-supervised** — its masks are model output,
not ground truth. Label every row it produces and never fold it into the A/A-long row
(docs/MULTIDATASET.md §1.3).

---

### Zero-shot arms — no ScanNet in training (todo 6l, docs/MULTIDATASET.md §12)

The competitor-matched training setting: FAST3DIS and IGGT never train on ScanNet. Drop it from
`SOURCES` and the val-312 ruler is staged and scored anyway — there it is a zero-shot read-out.

```bash
# arm I -- IGGT's mixture MINUS ASE (the ASE portion is not published; docs/…COMPARABILITY §5.3)
SOURCES='scannetpp infinigen re10k' CAP_RE10K=1500 EPOCHS=22 \
    EXTRA_ARGS='--anchor_3d --learning_rate 5e-5' EXP_TAG=_armI_zeroshot \
    sbatch --export=ALL --cpus-per-task=22 slurm/train_maskdino_multi.sh

# arm I-gt -- the same without the SAM2-supervised source (docs/MULTIDATASET.md §1.3)
SOURCES='scannetpp infinigen' EPOCHS=36 \
    EXTRA_ARGS='--anchor_3d --learning_rate 5e-5' EXP_TAG=_armIgt_zeroshot \
    sbatch --export=ALL --cpus-per-task=16 slurm/train_maskdino_multi.sh

sbatch --dependency=afterok:<job> --export=ALL,TRAIN_JOB=<job> slurm/chain_eval3d_matrix.sh
```

`EXTRA_ARGS`/`SOURCES` go through the **environment**, not through `--export`'s comma list —
sbatch splits that list on whitespace and would read the second word as the script name.
`--learning_rate 5e-5` is load-bearing for any mixture at or past A-long's size (§10.5, §11.3),
and the control must run at the same LR or the comparison moves two variables.

### 2.9 Score a finished run on a metric that did not exist when it ran

`--eval_only` loads a checkpoint, runs one validation pass through the same path a periodic eval
uses, appends one `eval_only` row to that run's `metrics.jsonl`, and exits. No training, no
optimizer, and the run's own `config.json` is left alone. Pass the SAME data/protocol flags the
run was trained with — `--eval_only` scores whatever bundle geometry you give it.

`slurm/eval_only_maskdino.sh` is the job form: it stages the **val tar only** (the train split is
never resolved or cached in this mode), scores the official 312-scene val split and appends the
row. Pass the run's own protocol flags in `EXTRA_ARGS` — the eval scores whatever bundle geometry
it is given, so a mismatch there silently changes the ruler.

```bash
OUT=/cluster/work/igp_psr/niacobone/distillation/output
sbatch --export=ALL,CHECKPOINT=$OUT/<run_dir>/checkpoint_best_bundle.pth,\
EXTRA_ARGS='--multi_frame --feature_mode bundle --anchor_3d' slurm/eval_only_maskdino.sh

DRY_RUN=1 CHECKPOINT=... bash slurm/eval_only_maskdino.sh    # echo the command, stage nothing
```

The metrics it prints include `bundle_hota` / `bundle_assa` / `bundle_deta` / `bundle_idf1`
(MASKDINO.md §6.6.1) — the reason the flag exists.

---

## 3. Full-resolution ruler (MASKDINO.md §6.5)

Adds `full_*` metrics scored at the 518×518 GT resolution next to the unchanged grid metrics.

```bash
sbatch --export=ALL,N_SCENES=490,EXTRA_ARGS='--eval_full_res' slurm/train_maskdino.sh
sbatch slurm/scannet_oracle.sh   # GT-only ceiling of the 37/74/148 grids on ScanNet (CPU-only)
```

---

## 4. 3D ruler — official ScanNet 3D instance benchmark (MASKDINO.md §9)

**The third protocol** — the only one placeable next to published numbers; never quote it next to
the 2D tables. And the published 3D numbers are **two** protocols (§9.9): ours is *unposed transfer*
(= FAST3DIS, IGGT); SegVGGT's is *posed transfer* (GT poses + sensor depth, no geometry error), so
its 0.504 / 0.717 / 0.870 is **not** a like-for-like row.

Unprojects a `--multi_frame` checkpoint's masks with VGGT's **own** predicted depth + cameras (no GT
geometry at inference), Sim(3)-registers for scoring only, majority-votes per superpoint, scores
with the vendored official evaluator. Checkpoints trained on scenes 0000–0489 overlap val-312:
their numbers are **diagnostic only** (§9.4).

```bash
sbatch --export=ALL,CHECKPOINT=<run_dir>/checkpoint_best_bundle.pth slurm/eval_3d_maskdino.sh
```

### Two transfer modes = two experiments (§9.10) — printed as two columns, never merged

| mode | what it measures | comparable to | AP / AP50 / AP25 |
|---|---|---|---|
| `--transfer_mode unproject` (default, the headline) | 2D mask quality × feed-forward geometry quality; predicted depth+cameras + Sim(3)/ICP | FAST3DIS, IGGT | 0.023 / 0.067 / 0.268 |
| `--transfer_mode gt_projection` | 2D mask quality **alone**; projects the mesh into each view with GT pose+intrinsics, gates on the ScanNet **sensor** depth, reads the mask there | SegVGGT | 0.060 / 0.156 / 0.408 |

Still image-only in both: GT geometry transfers **finished** masks for scoring, exactly like the
Sim(3) does. `vote_radius` / `depth_conf_percentile` / `icp` are **inert** in `gt_projection` (the
script says so).

```bash
sbatch --export=ALL,CHECKPOINT=<run_dir>/checkpoint_best_bundle.pth,\
EXTRA_ARGS='--transfer_mode gt_projection' slurm/eval_3d_maskdino.sh
```

### The oracle that licenses `gt_projection`

Run it before believing any `gt_projection` number. Renders the 3D GT back through the same
projection; round-trip purity must be ≈1.000 (measured 0.9999).

```bash
sbatch slurm/eval3d_projection_oracle.sh          # CPU-only, no checkpoint, ~15 min for 312
myenv/bin/python scripts/eval3d_projection_oracle.py --frames_root <scans25k> \
    --gt_root <scans3d> --scenes scene0011_00     # local smoke test, seconds
```

### Qualitative 3D (§9.7) — two different pictures, do not confuse them

**(a) What the benchmark scores**: instance-coloured mesh vertices after lifting + voting (grey = no
instance reached that vertex). `--scenes` renames the JSON, so a subset can never overwrite a
full-val result — and a subset is a picture, never a number.

```bash
sbatch --export=ALL,CHECKPOINT=<run_dir>/checkpoint_best_bundle.pth,\
EXTRA_ARGS='--dump_ply --scenes scene0011_00 scene0015_00 --vote_radius 0.1 --depth_conf_percentile 25' \
    slurm/eval_3d_maskdino.sh          # → <run_dir>/eval3d_<scene>.ply (MeshLab/CloudCompare)
```

**(b) What the model predicts**: VGGT's own point cloud coloured by the head, interactively. Needs a
GPU node. Colours are query ids, identical to the 2D panels' palette. The "GT vs Prediction
(synced)" tab shows both under **one** camera (`demos/dualview3d.py`).

```bash
python demos/demo_gradio.py --seg_checkpoint <run_dir>/checkpoint_best_bundle.pth \
    --seg_scans_root /cluster/scratch/niacobone/demo_scans/scans   # 4 val scenes staged there
# Look at a .ply without MeshLab: one self-contained HTML (WebGL inside), scp it and open it.
myenv/bin/python scripts/view_ply.py <run_dir>/eval3d_scene0011_00.ply
myenv/bin/python scripts/view_ply.py a.ply b.ply --out compare.html   # two panels, one camera
```

Needs the two val-312 tars on `work` (built 2026-08-01; rebuild in ~20 min if lost):

```bash
sbatch legacy/dataset_build/slurm/download_3d_gt_val312.sh       # mesh+superpoints+aggregation
sbatch legacy/dataset_build/slurm/download_frames25k_val312.sh   # whole-scan frames + poses
```

### 4.1 The same ruler on the other benchmarks — `--dataset` (todo 6d, RESULTS.md §7)

`--dataset {scannetv2,scannet200,scannetpp,replica}` (`train/datasets3d.py`) swaps the benchmark
and nothing else: same head, same two transfer modes, same vendored evaluator. It **defaults to
`scannetv2`**, so every command above and every published number is unchanged. The SLURM driver
picks the matching tars from `DATASET`.

```bash
sbatch --export=ALL,DATASET=scannetpp,CHECKPOINT=<run_dir>/checkpoint_best_bundle.pth \
    slurm/eval_3d_maskdino.sh
bash slurm/eval_3d_matrix.sh                 # the whole 3 ckpt x 4 dataset x 2 mode grid
DRY_RUN=1 bash slurm/eval_3d_matrix.sh       # …print the sbatch lines instead
CKPTS=<run_dir> bash slurm/eval_3d_matrix.sh # …the same grid for one arbitrary run
```

`CKPTS` takes either one of the three short keys the script defines for the published rows
(`mf`, `anchor3d`, `s16`) or, for anything newer — e.g. the multi-dataset arms of
`docs/MULTIDATASET.md` §10 — a run directory (absolute, or a name under the output root) or an
explicit `.pth`. Checked by `bash tests/test_eval_3d_matrix_sh.sh`, DRY_RUN only, no cluster.

To score a training run that has not finished yet — its directory carries a timestamp minted when
the job *starts*, so it cannot be named at submit time — chain the grid onto it instead:

```bash
sbatch --dependency=afterok:<train job> --export=ALL,TRAIN_JOB=<train job> \
    slurm/chain_eval3d_matrix.sh
```

Collect the finished cells into the §7 table. `--run LABEL=DIR` (repeatable) adds a run the
collector does not name; `--only` drops the three built-in rows so two blocks are never printed as
one table. It keeps **only** cells run at defaults, so a tuned sweep can never leak in.

```bash
myenv/bin/python scripts/collect_eval3d_matrix.py
myenv/bin/python scripts/collect_eval3d_matrix.py --only \
    --run 'B ScanNet=maskdino_multi_scannet_n1201_a3d_e35_20260821_201002'
myenv/bin/python tests/test_collect_eval3d_matrix.py   # its CPU checks
```

**The three non-default datasets are CLASS-AGNOSTIC only** — their taxonomies are not our 19
ScanNet classes, so labels are collapsed on both sides and the class-aware fields are written as
`null` rather than fabricated. Never put such a row next to a class-aware ScanNetv2 one.

### 4.1.1 Matching the competitors' VIEW COUNT (todo 6k, docs/DATASET.md §2.5)

ScanNet++ and Replica already run at exactly **50 frames/scene** = FAST3DIS's budget. ScanNetv2
and ScanNet200 run at **17.42**, because `scannet_frames_25k` is every 100th frame. The dense
export fixes only those two:

```bash
D=/cluster/work/igp_psr/niacobone/distillation/dataset/scannet/scannet_frames_dense_val312.tar.zst
# FAST3DIS's 50 uniformly sampled views
sbatch --export=ALL,CHECKPOINT=<ckpt>,FRAMES_TAR=$D,EXTRA_ARGS='--num_frames 50' \
    slurm/eval_3d_maskdino.sh
# the 17-frame CONTROL -- on the SAME tar, or the jpeg re-compression of the 25k export
# is a second variable (docs/DATASET.md §2.5)
sbatch --export=ALL,CHECKPOINT=<ckpt>,FRAMES_TAR=$D,EXTRA_ARGS='--num_frames 17' \
    slurm/eval_3d_maskdino.sh
# SegVGGT's sampling: every 20th frame, i.e. the tar as built
sbatch --export=ALL,CHECKPOINT=<ckpt>,FRAMES_TAR=$D slurm/eval_3d_maskdino.sh
```

`--tmp` and the GPU may both need raising: the head runs ONE forward pass over every sampled
frame, so 50–120 frames is 3–7× the memory of the 17 the default `rtx_4090:1` was sized for.

### 4.2 The licence gate — run it before believing ANY number from a dataset

Feeds a dataset's own GT back as predictions; the official evaluator must answer exactly
1.000 / 1.000 / 1.000 (MASKDINO.md §9.2), and the sensor depth must land on the mesh. CPU-only,
no checkpoint. `gate_<dataset>.json` lands beside the tars.

```bash
sbatch --export=ALL,DATASET=replica slurm/gate_3d_gt.sh
myenv/bin/python scripts/gate_3d_gt.py --dataset replica --gt_root <scans3d> \
    --frames_root <scans25k> --num_scenes 2 --report_superpoints    # local smoke test
```

---

## 5. Re-render a finished run's figures (MASKDINO.md §6.4)

Colours are keyed to instance identity, so an object keeps its colour across the frames of a bundle.
Changing the drawing code does **not** touch existing runs — re-render them explicitly:

```bash
sbatch --export=ALL,RUNS='<run_dir_1> <run_dir_2>' slurm/visualize_maskdino.sh
python scripts/visualize_maskdino.py --checkpoint <run_dir>/checkpoint_best.pth   # needs GPU+data
```

---

## 6–7. The COCO arm — RETIRED, archived 2026-08-27

The port check is complete and COCO is not a ruler this project reports on. The whole arm — the
upstream-equivalence transplant, `train_maskdino_coco.sh`, the resolution oracle and the
upstream-MaskDINO control (with its `third_party/maskdino_control/` clone glue and both tests) —
now lives under **`legacy/coco/`**, mirroring the layout it had here. The write-ups are
`docs/old/MASKDINO_COCO.md` and `docs/old/MASKDINO_HISTORY.md` §7.6. Nothing in it is quotable
next to a ScanNet number.

---

## 8. Dataset rebuild (only if a tar is lost; docs/DATASET.md §5)

```bash
for t in legacy/dataset_build/tests/test_*.py; do myenv/bin/python "$t"; done   # CPU-only
sbatch legacy/dataset_build/slurm/download_official_gt.sh
sbatch legacy/dataset_build/slurm/extend_dataset_500.sh
sbatch legacy/dataset_build/slurm/pack_official_gt.sh
```

### InsScene-15K 2D training sets (docs/MULTIDATASET.md §2)

Reads the mirror's split zips **without unpacking them**, builds node-local, ships one tar per
source to `dataset/insscene2d/`. `re10k` is deliberately not in the default `SOURCES`.

```bash
sbatch slurm/build_insscene2d.sh                              # scannetpp + infinigen, 1 h 42
sbatch --export=ALL,SOURCES=re10k slurm/build_insscene2d.sh   # +re10k, ~3 h 15, ~10 GB tar
sbatch --export=ALL,SOURCES=re10k,LIMIT=20 slurm/build_insscene2d.sh          # smoke
myenv/bin/python slurm/build_insscene2d.py --source re10k --out $TMPDIR/b --limit 5   # local
```

`--exclude_scenes data/splits/scannetpp_nvs_sem_val.txt` is passed for ScanNet++ **only**, and
that asymmetry is the point: ScanNet++ is the one source that is both trained on and evaluated on.
Infinigen and RE10K are not among the four benchmarks, so there is nothing they can leak.

### 1201-scene official-train extension (todo 1c; separate tar, does not touch the 500-scene one)

**Builds node-local.** `/cluster/scratch` is quota'd on **file count** (1.0M soft / 1.5M hard) and
the 1201-scene tree is ~1.26M files — building it there fails, and did (DATASET.md §5.1). The tree
lives in `$TMPDIR`; only one compressed chunk tar per range lands on scratch.

```bash
sbatch legacy/dataset_build/slurm/extend_dataset_1201.sh <list_start> <list_end>  # one per chunk
sbatch legacy/dataset_build/slurm/pack_official_gt_1201.sh   # after all chunks report COMPLETE
# An incomplete chunk resubmits ITSELF (new job id), so --dependency=afterok on the original id
# never fires — don't chain the pack that way. With a single chunk covering the whole split,
# CHAIN_PACK=1 makes the completing job submit the pack itself:
sbatch --export=ALL,CHAIN_PACK=1 legacy/dataset_build/slurm/extend_dataset_1201.sh 0 1200
# One-off rescue: fold an existing on-scratch build tree into a chunk tar and free the inodes.
sbatch legacy/dataset_build/slurm/snapshot_build_1201.sh
```

### 312-scene official-VAL build — the val ruler for the split above (todo 1c)

Same pipeline and QA gates, `--scene_list data/splits/scannetv2_val.txt`, own chunk dir/tar/README so
neither the 500- nor the 1201-scene tar is touched. One chunk is enough, so the extend job chains the
pack itself: the whole build is this one command (~1 h 20 end to end).

```bash
sbatch --export=ALL,CHAIN_PACK=1 legacy/dataset_build/slurm/extend_dataset_val312.sh 0 311
sbatch legacy/dataset_build/slurm/pack_official_gt_val312.sh   # only if packing separately
```

### ScanNet++ val-50 — the competitor ruler (todo 6c; docs/DATASET.md §2.1, §5.0)

No download: the source is already on `work`. One job, both tars, ~50 min. Node-local, zero
loose files on scratch, resumable from whichever tar (final or `.partial`) exists.

```bash
sbatch legacy/dataset_build/slurm/build_scannetpp_val50.sh
# acceptance test on the SHIPPED tars (unpacks them node-local, then verifies)
sbatch --export=ALL,VERIFY_SCENES=0 legacy/dataset_build/slurm/verify_scannetpp_tars.sh
# or verify an already-unpacked tree by hand
myenv/bin/python scripts/verify_scannetpp_gt.py \
    --gt_root <tree>/scans3d --frames_root <tree>/scans25k --num_scenes 5
```

`build_scannetpp_val50.sh` verifies the tree it just built, before packing. That is not the
same as verifying the deliverable — compression, the count check and the copy to `work` sit
between them — so `verify_scannetpp_tars.sh` unpacks the tars back and re-runs the checks
against what downstream will actually read. Run it after a lost/restored tar, and before a
number depends on this data.

`EXCLUDE` names scenes the upstream release ships broken and defaults to the one that is
(`d755b3d9d8` — docs/DATASET.md §2.1). `EXCLUDE=` builds all 50 and will fail on that scene's
geometry check, which is the intended behaviour: a missing scene is recoverable, a silently
misaligned one is not.

---

### Dense ScanNet val-312 frames (todo 6k; docs/DATASET.md §2.5)

```bash
sbatch legacy/dataset_build/slurm/build_frames_dense_val312.sh          # 16-task array, ~20 s/scene
sbatch --dependency=afterok:<array> legacy/dataset_build/slurm/pack_frames_dense_val312.sh
```

Streams the whole `.sens` per scene (no early abort — a whole-scan sample needs the last frame),
writes only the kept frames, resumable per scene via a `.complete` marker. This is the one build
that writes to **scratch** rather than `$TMPDIR`: ~94 k files, and an array cannot share a
node-local tree. Delete the tree once the tar is on work.

## 9. The retired baseline head (`legacy/`, frozen)

`legacy/d4rt/` holds the previous hand-rolled DETR-style head — the bar every MaskDINO number is
measured against. Frozen on purpose; its published numbers are in `docs/old/ARMS_SUMMARY.md`. Two
live code paths still import it: `scripts/eval_perframe.py` and `demos/demo_gradio.py`.

```bash
# Score a legacy checkpoint on the identical per-frame protocol (the apples-to-apples baseline):
python scripts/eval_perframe.py --checkpoint <legacy_run>/checkpoint_best.pth  # → perframe_eval_*.json
# Still runnable (see legacy/README.md):
python legacy/d4rt/scripts/train_multiscene.py --train_scenes ... --val_scenes ...
python legacy/d4rt/scripts/visualize_masks.py --checkpoint <run_dir>/checkpoint.pth
sbatch legacy/d4rt/slurm/train_full.sh
for t in legacy/d4rt/tests/test_*.py; do python "$t"; done
```
