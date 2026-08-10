# Multi-dataset training — ScanNet v2 + ScanNet++ + Infinigen (todo 6e + 6f)

Opened 2026-08-10. This is the **data-scaling** workstream: every number in `docs/RESULTS.md` was
produced by a head trained on ScanNet v2 and nothing else, and the scaling curve (50 → 190 → 490 →
1201 scenes) was still rising when ScanNet ran out of scenes. This file is the home of what more
data required, what it cost, and what it bought.

> **Nothing here is comparable to a published number in `docs/RESULTS.md` §2/§3/§6.** Those are
> class-aware. Everything trained under this workstream is **class-agnostic** (one class,
> "object"), because the added datasets do not share ScanNet's taxonomy. The honest baseline for
> any run here is a **ScanNet-only run with `--class_agnostic`**, not a published class-aware row.

## 1. What the added data actually is — and what it is not

`docs/TRAINING_COMPARABILITY.md` §4 assumed ScanNet++ 2D supervision would have to be *rendered*
from the mesh. It does not: **InsScene-15K already ships per-frame instance annotations**, and the
mirror has been on work since 2026-08-08 (522 GB, `dataset/insscene15k/`). Read from the archives
themselves, not from the paper:

| subset | what it holds | usable as 2D supervision? |
|---|---|---|
| `processed_scannetpp_v2` | 903 scenes; `images/*.jpg`, `depth/*.png`, `refined_ins_ids/<stem>.jpg.npy` (int16 per-pixel ids at the image's own 920×690) | **yes** |
| `processed_infinigen` | 1466 sub-scene zips (156 scenes); `Image/`, `Depth/`, `ObjectSegmentation/*.npy` (int64 ids), `Objects/*.json` (names + `object_index`), `camview/*.npz` (K, T) | **yes** |
| `processed_re10k` | `rgb/` + `cam/` only | **no** — no instance annotation of any kind, so it is skipped entirely (169 GB never read) |

Two properties were verified before any of it was used, because both are load-bearing:

1. **Ids are global per scene, not per frame.** ScanNet++ scene `00777c41d4`: two adjacent frames
   share **34 of 34** ids. Infinigen: ids are sparse scene-level indices (61, 69, 717, 766 in a
   42-object frame), not a per-frame 1..N relabelling. This is exactly what the multi-frame GT
   needs — it re-links instances across views by global id (CLAUDE.md).
2. **Infinigen's ids index `Objects/*.json`**, so every instance has a name. All 42 ids of a test
   frame resolved (`BedFactory(...)`, `bedroom_0/0.wall`). That is what makes the room-shell drop
   below principled rather than a heuristic.

### 1.1 The two exclusions

- **The 50 `nvs_sem_val` ScanNet++ scenes are dropped from training** (`--exclude_scenes
  data/splits/scannetpp_nvs_sem_val.txt`). The mirror contains **all 49 scenes of our ScanNet++
  evaluation column** (`docs/RESULTS.md` §7); training on it unfiltered would leak the whole
  zero-shot benchmark. 903 − 50 = **853 training scenes**.
- **Infinigen's room shell is dropped by name** (`<room>/N.wall|floor|ceiling|exterior`, measured
  at 21 %, 17 % and 32 % of one frame). The ScanNet benchmark excludes wall/floor and our Replica
  GT excludes the room shell (`docs/DATASET.md` §2.2), so supervising them here would teach the
  head to emit masks that every evaluator counts as false positives.

ScanNet++'s own annotations needed **no** area filter: no instance dominates a frame (largest
median area 0.18–0.32 over the scenes checked), so there is no wall/floor blob to remove.

## 2. The build

`slurm/build_insscene2d.py` (driver `slurm/build_insscene2d.sh`) is a **selection and
re-encoding** pass — nothing is rendered. Per scene it keeps `--frames` evenly spaced frames,
resizes to the trainer's 518×518 squash, remaps the ids **once per scene** onto a dense 1..G, and
writes:

    <source>/<scene>/color/<stem>.jpg        518×518 RGB
    <source>/<scene>/instance/<stem>.png     uint16 id map, 0 = background
    <source>/<scene>/manifest.json           frames, id table, provenance, counters

`slurm/insscene_shards.py` reads the 211 GiB ScanNet++ archive **without unpacking it**: the parts
are a plain `split -b` of one zip, so the central directory sits at the tail of the last part and
any member can be reached by seeking across the concatenation. Concatenating would materialise
211 GiB to read ~0.3 % of it. zip64 is mandatory at this size and is handled explicitly.

Storage discipline per `docs/DATASET.md` §5.1: the tree is built in `$TMPDIR` (≈148 k files) and
only one tar per source lands on work — **scratch inode cost zero**.

## 3. `--class_agnostic` (todo 6e)

One rule, no second flag: **a one-class head means class-agnostic.**

- `scripts/train_maskdino.py --class_agnostic` builds the head with `num_classes=1`.
- `train/maskdino_data.py::build_frame_targets` sees `num_classes == 1` and (a) keeps every
  instance instead of dropping those whose class index the head cannot name, (b) collapses every
  label onto the single class.

Because `head_config` carries `num_classes`, a **checkpoint alone** decides how its GT is built,
and because every scorer (per-frame, per-bundle, full-res) takes its GT classes from those same
targets, the collapse reaches training and evaluation through one code path. Instances with an
invalid class index (< 1) are still dropped — that is corrupt GT, not a foreign taxonomy.

Default off. Every published number in this project stays class-aware.

## 4. The mixed loader

`data/instance_map_dataset.py` reads the instance-map layout and returns **the same sample dict**
as `data/scannet_overfit.py::ScanNetSingleSceneDataset`, so the feature cache, the target builder
and both evaluators consume the two interchangeably. `build_scene_dataset` dispatches **per
directory**: a list of paths may mix all three sources, and a pure-ScanNet list keeps its original
loader untouched, so no existing run changes shape.

`prepare_scenes` **refuses** a mixed scene list against a multi-class head rather than silently
supervising every ScanNet++/Infinigen object as ScanNet class 1.

## 5. Running it

```bash
# 1. the build (CPU, ~3 h; writes two tars to dataset/insscene2d/)
sbatch slurm/build_insscene2d.sh

# 2. the training — class-agnostic by construction, val stays the official ScanNet 312
sbatch slurm/train_maskdino_multi.sh                                   # all three sources
sbatch --export=ALL,SOURCES='scannet scannetpp' slurm/train_maskdino_multi.sh
sbatch --export=ALL,CAP_SCANNETPP=200,CAP_INFINIGEN=200 slurm/train_maskdino_multi.sh
DRY_RUN=1 bash slurm/train_maskdino_multi.sh                           # lists + schedule only

# CPU tests
myenv/bin/python tests/test_insscene2d.py        # the reader, the build's transforms
myenv/bin/python tests/test_class_agnostic.py    # 6e, both directions
myenv/bin/python tests/test_multidata2d.py       # the loader and the dispatcher
```

**Val never moves.** It is the official ScanNet v2 312-scene list in every mixture, scored
class-agnostic — otherwise "more data helped" and "the ruler got easier" are indistinguishable.

**The memory bound is the feature cache**, not the GPU: the trainer caches frozen VGGT features
for every scene up front, ~45 MB per 8-frame bundle, so ~54 GB for ScanNet's 1201 alone and
~160 GB for the full 3520-scene mixture. That is what the job's 16×16 GB request buys; `CAP_*`
shrinks it.

## 6. Status

| step | state |
|---|---|
| shard survey + id-consistency verification | done 2026-08-10 |
| `slurm/insscene_shards.py`, `build_insscene2d.py` + 29 CPU checks | done |
| `--class_agnostic` (6e) + 13 CPU checks | done |
| `data/instance_map_dataset.py` + 29 CPU checks | done |
| the build itself (job 10286143) | running |
| end-to-end smoke run (job 10287385, `--dependency=afterok`) | queued behind the build |
| the full mixture run | **not launched** — waiting on the two above |

Open, in order:

- [ ] Score a class-agnostic **ScanNet-only** run — without it the mixture has no baseline.
- [ ] Run the mixture, then the 3D ruler (`docs/MASKDINO.md` §9) on it in both transfer modes.
      Note `scripts/eval_3d_maskdino.py` maps predicted labels to ScanNet ids; a 1-class
      checkpoint must be scored class-agnostic, which the evaluator already supports but does not
      yet *force* from `head_config`.
- [ ] Per-source ablation: is the gain ScanNet++ (real, same domain) or Infinigen (synthetic,
      512×288 upsampled to 518)? The mixture alone cannot say.
- [ ] Link this file from `docs/todo.md` and `docs/DATASET.md` — both carry other sessions'
      uncommitted work right now, so they were deliberately left untouched.
