# Plan: replace SAM3 GT with official ScanNet instance GT

**Status: PLAN — not yet executed.** Written 2026-07-07. This document is
self-contained: it records the findings that motivate the switch, the verified
state of every relevant path on disk, and a phased execution plan with
verification gates. It is meant to be executed by a fresh Claude agent
(`read this file first, then start at Phase 0`).

---

## 1. Why (context for the executor)

An audit on 2026-07-07 (20 scenes, every 10th of scene0000–0199) found the
SAM3-generated per-instance GT has **systematic cross-class duplicates**:
each class is prompted independently in SAM3, so the same physical object is
often labeled as an instance under two classes.

- 68 instance pairs with cross-frame IoU ≥ 0.5 between *different* classes
  (~3.4/scene; ≈15–20% of instances), mostly IoU 0.98–1.00 (pixel-identical).
- 15.9% of foreground pixels claimed by ≥2 classes.
- Dominant pairs: desk↔table, curtain↔shower_curtain, chair↔sofa,
  cabinet↔door, bookshelf↔cabinet, curtain↔window.
- Effect on training: two GT instances with identical masks and contradictory
  class labels → the Hungarian matcher demands two predictions for one object
  (built-in false positive in honest AP50) and the class head gets
  contradictory supervision on the same pixels.

Decision (project owner): switch supervision to the **official ScanNet 2D
instance annotations** (`_2d-instance-filt` / `_2d-label-filt`), which are
projections of the single human-verified 3D annotation into every RGB frame —
one class per object, cross-view-consistent instance IDs by construction.

## 2. Verified state of the world (checked 2026-07-07)

| Thing | State |
|---|---|
| Official 2D GT on disk | **Not present anywhere reachable.** Must be downloaded. |
| Download tooling (ours) | `/cluster/scratch/niacobone/sam3/scripts/` — the project owner's own SAM3-preprocessing pipeline repo. Contains the official `download-scannet.py` (Python 3, no credentials, supports `--type _2d-instance-filt.zip` / `_2d-label-filt.zip`, but has interactive `input()` prompts and no timeouts) **and** `download_sens.py`, a robust replacement written because the TUM server hangs: non-interactive, per-read socket timeout, retries with backoff, resumable via `.part` files, direct URL construction. **Adapt `download_sens.py` for the zips** (see Phase 0). ScanNet access/ToS is the owner's own (`instructions_scannet.txt` is the access-grant email). |
| Zip URL scheme | `http://kaldir.vc.cit.tum.de/scannet/v2/scans/<scene>/<scene>_2d-instance-filt.zip` (the official script uses `v2/scans` for all types except `.sens`, which lives under `v1/scans` — that swap is specific to `.sens`, do not apply it to the zips). |
| Cluster network | Compute nodes need `module load eth_proxy` for outbound downloads (see `download_split2.sh`, which is also the SLURM job template for long downloads: 24 h, resumable, re-run to heal failures). |
| Our SAM3 dataset | Only as tar: `/cluster/work/igp_psr/niacobone/distillation/dataset/scannet/scannet_instance_dataset_full.tar.zst` (~2.6 GB; unpacks to `scans/<scene>/raw_data/{subset,masks,masks_instance}`). The unpacked `…/scannet/scans/` tree on work was **deliberately deleted** by `sam3/scripts/cleanup_old_dataset.sh` (tar-integrity-gated consolidation) — CLAUDE.md is stale on this. The scratch build trees (`scannet_build`, `scannet_build_split2`) are hollow skeletons (scratch purge ate the PNGs). |
| Subset provenance | `create_subset.sh`: `subset/` = color frames `%05d.jpg` for indices `i*5, i<100` (0–495), copied from `color/` (itself extracted from `.sens`; later pruned by `prune_work_color.sh`). |
| Packing conventions | `sam3/scripts/pack_split2.sh` + `fuse_splits.sh`: tar from the scratch build tree with `zstd -1 -T0`, verify by comparing archive vs source png/jpg counts, copy to work as `.tmp` + atomic `mv`. **Never unpack a tar just to re-tar it** — ~830 K small files risks the scratch **inode quota** (learned the hard way, see `fuse_splits.sh` header). `gen_instance_report.py` is the per-scene README generator to imitate. |
| Label mapping table | `/cluster/work/igp_psr/niacobone/distillation/dataset/scannet/scannetv2-labels.combined.tsv` — present. |
| ScanNet toolkit repo | `/cluster/scratch/niacobone/ScanNet` (official repo clone; `BenchmarkScripts/`, `AnnotationTools/` — useful reference for 2D annotation formats). |
| Subset frame naming | `subset/00000.jpg … 00495.jpg`: 100 frames, stride 5, **original color-frame indices 0–495**. So the official per-frame GT file for subset frame `00375.jpg` is instance-filt frame index `375`. |
| Loader (`data/scannet_overfit.py`) | Discovers segments purely from directory names: `masks/<class>/` (per-class) or `masks_instance/<class>_<k>/` (with `--instance_level`), skipping `_qa`-style dirs. **Skips missing mask files** (`if not mask_path.exists(): continue`) and all-zero masks. Resizes masks with NEAREST to 518. → **Zero loader changes needed if we mirror the layout.** |
| Tests | `tests/test_phase2.py` builds synthetic scenes — no real dataset needed. |
| Class list | `SCANNET_CLASSES` (loader line 14): wall, floor, cabinet, bed, chair, sofa, table, door, window, bookshelf, picture, counter, desk, curtain, refrigerator, "shower curtain", toilet, sink, bathtub, otherfurniture (indices 1..20; 0 = background). SAM3 GT used only the first 19 (no `otherfurniture` dirs; dir naming uses `shower_curtain` with underscore — mirror the SAM3 dir-naming exactly, it is what the loader's parser is proven against). |

NYU40 ids for the 20 benchmark classes, in `SCANNET_CLASSES` order:
`1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39`.

## 3. Strategy

**Build a new dataset tree with the exact same on-disk layout as the SAM3 one**
(`scans/<scene>/raw_data/{subset,masks,masks_instance}`), with masks derived
from official GT instead of SAM3, and ship it as a new tar. This keeps the
loader, the checkpoint round-trip, the visualizers, and the SLURM staging
untouched — the only code added is an offline converter script, and the only
config change is the tar filename.

Rejected alternatives: (a) new loader path reading instance-filt directly —
touches loader + tests + staging for no benefit; (b) projecting the 3D
annotation ourselves via `.sens` poses — much heavier download and code, and
`instance-filt` already *is* that projection, done upstream.

Keep the SAM3 tar untouched — it stays the baseline for "does cleaner GT
improve the numbers" comparisons and is itself a deliverable of the project.

## 4. Execution phases

Work dir for everything below: `/cluster/scratch/niacobone/scannet_official_build/`
(scratch; mind the purge policy — finish with the tar on work storage).
Use `myenv/bin/python` from the repo for all scripts.

### Phase 0 — pilot: download ONE scene and verify every format assumption

For the pilot, the interactive official script is fine (pipe `yes ''` to
answer its ToS/skip prompts, the `download_split2.sh` trick):

```bash
module load eth_proxy 2>/dev/null || true
mkdir -p /cluster/scratch/niacobone/scannet_official_build/pilot
cd /cluster/scratch/niacobone/scannet_official_build
yes '' | python /cluster/scratch/niacobone/sam3/scripts/download-scannet.py \
    -o pilot --id scene0000_00 --type _2d-instance-filt.zip
yes '' | python /cluster/scratch/niacobone/sam3/scripts/download-scannet.py \
    -o pilot --id scene0000_00 --type _2d-label-filt.zip
```

For the full run (Phase 2), instead write `download_2d_gt.py` by adapting
`sam3/scripts/download_sens.py` (keep its timeout/retry/resume structure;
change the URL template to
`http://kaldir.vc.cit.tum.de/scannet/v2/scans/{scene}/{scene}{suffix}` with
suffix in `{_2d-instance-filt.zip, _2d-label-filt.zip}`, and the target path
accordingly). If the kaldir server rejects or the URL scheme changed, stop and
report — do not hunt for mirrors.

Then verify, and **record the answers in this file under Phase-0 results**:

1. Zip member naming: expected `instance-filt/<frameidx>.png` with unpadded
   indices (`0.png`, `5.png`, …). Confirm.
2. Resolution of the PNGs. Expected **1296×968** (color-camera frame) or
   640×480 (depth-registered). Either works (loader resizes with NEAREST),
   but record which.
3. Value semantics: `instance-filt` pixel = per-scene instance id (0 =
   unannotated); `label-filt` pixel = raw ScanNet label id, mapped to NYU40
   via the `id → nyu40id` columns of `scannetv2-labels.combined.tsv`. Confirm
   by decoding one frame and cross-checking a few labels against the RGB.
4. **Alignment check (the one real risk):** overlay instance-filt frame 0 on
   the SAM3 tree's `subset/00000.jpg` for scene0000_00 (unpack just that from
   the SAM3 tar) and save a side-by-side jpg. Object boundaries must sit on
   the objects. If the projection resolution is 640×480, also check for the
   ~0.4% aspect mismatch vs 1296×968 — visually irrelevant at 37×37
   supervision, but confirm.
5. Zip sizes for one scene (extrapolate ×200 for disk/time planning; only
   frames 0–495 step 5 are needed, so extract selectively and delete zips as
   you go).

### Phase 1 — converter script + test

New script `scripts/build_official_masks.py` (follows repo conventions: one
component, standalone CPU test in `tests/`):

- Input: a scene's two zips (or extracted dirs) + the tsv + the subset frame
  list (0..495 step 5).
- For each subset frame: read instance-filt + label-filt; for each instance id
  present, find its class = NYU40 id of its label pixels (majority vote;
  they should be constant per instance), map NYU40 → `SCANNET_CLASSES` index.
- Emit `masks_instance/<class>_<k>/<frame>.png` (uint8 {0,255}, source
  resolution, filename = subset stem + `.png`) and the per-class union
  `masks/<class>/<frame>.png`. Dir naming with underscores
  (`shower_curtain_3`), `<k>` zero-based per class in order of first
  appearance — exactly the SAM3 conventions.
- **Write masks sparsely** (only frames where the instance is visible) — the
  loader verifiably skips missing files; this shrinks the tree a lot.
- Also copy the scene's `subset/` from the SAM3 tar unchanged (images don't
  change — only GT does).
- Per-scene `_qa/stats.json`: instance count per class, plus a
  **cross-class duplicate check** (pairwise cross-frame IoU between instances
  of different classes — port the logic from the audit; must be ~0 by
  construction, this is the acceptance test for the whole migration).

Decisions locked in (change only if Phase 0 contradicts them):
- `otherfurniture` (NYU40 39) and any class outside the 20 → **background**,
  matching the SAM3 GT's 19-class taxonomy and the head's 20 logits.
- No speck filter (SAM3 used 200 px; official GT has no tracking specks —
  keep everything, note it in stats).
- `wall`/`floor`: keep whatever instance ids the official GT has (do NOT force
  single-instance like SAM3 did; multiple wall segments are fine — the loader
  and matcher are instance-based). If this turns out to explode instance
  counts absurdly (>10 wall instances/scene), fall back to merging stuff
  classes to `_0` and record the change here.

Test `tests/test_build_official_masks.py`: synthetic instance-filt/label-filt
PNG pairs + tiny tsv → converter → assert dir naming, sparseness, class
mapping, union consistency, background handling. CPU, no downloads.

### Phase 2 — full download + build + QA (200 scenes)

- Download with the adapted `download_2d_gt.py` (Phase 0), scenes
  `scene0000_00 … scene0199_00`, both zip types, as a SLURM job modeled on
  `sam3/scripts/download_split2.sh` (`module load eth_proxy`, 24 h wall time,
  resumable — re-run to heal failed scenes). Sequential is fine; be polite to
  the server. Download → convert → **delete zips per scene batch**, so peak
  disk stays bounded (record actuals from Phase 0). Mind the scratch **inode
  quota**: extract only the ~100 needed frames per zip, never the full >5500.
- Unpack the SAM3 tar once to get all `subset/` dirs
  (`zstd -dc … | tar x` somewhere on scratch; ~5.4 GB).
- Run the converter over all 200 scenes → build tree
  `scannet_official_build/scans/<scene>/raw_data/{subset,masks,masks_instance}`.
- QA gates (all must pass before packing):
  1. Aggregate cross-class duplicate rate ≈ 0 (vs 15.9% multi-class px in the
     SAM3 GT).
  2. Instance-count sanity vs SAM3 (`INSTANCE_MASKS_README.md` had ≈4195 over
     200 scenes; official GT will differ — often more instances since ScanNet
     annotates exhaustively — but per-scene counts should be same order).
  3. Random visual spot-check: render 5 scenes' overlay strips (one fixed
     color per instance across frames) and eyeball identity consistency.
- Pack following the `pack_split2.sh` conventions: tar from the build tree
  with `zstd -1 -T0` (or `-19` if size matters more than time), verify
  archive-vs-source png/jpg counts, copy to work as `.tmp` + atomic `mv` →
  `/cluster/work/igp_psr/niacobone/distillation/dataset/scannet/scannet_official_gt_full.tar.zst`.
  Generate `OFFICIAL_GT_README.md` next to it in the style of
  `gen_instance_report.py` (provenance, date, converter commit, QA numbers).

### Phase 3 — wire in (config only)

- `slurm/stage_dataset.sh`: make the tar filename an env var
  (`DATASET_TAR`, default = the old SAM3 tar for backward compat) and have the
  train SLURM scripts set it to the new tar. No loader change expected —
  verify by reading `data/scannet_overfit.py` against the built tree, then run
  `python tests/test_phase2.py` (synthetic; must still pass untouched).
- Smoke test on real data: `scripts/train_overfit.py` (~10 epochs is enough)
  pointed at one converted scene with `--instance_level`; assert loss falls
  and prompted mIoU rises — proves the built tree end-to-end.

### Phase 4 — training validation

1. Re-run the current base (arm C: `--query_mode learned
   --num_learned_queries 64 --instance_level`, same hyperparams as the run in
   `docs/MILESTONES.md`) on the official-GT tar, same val scenes 0080–0082
   held out.
2. Compare `metrics.jsonl` vs the SAM3-GT arm-C run: val mIoU (was 0.371) and
   honest val[grid] AP50 (was 0.228 at N=200). Expect honest AP50 to move the
   most — that's where duplicate GT hurt.
3. Cross-eval for the writeup: eval the *old* SAM3-trained checkpoint against
   *official* GT val — quantifies how much of the old score was fitting label
   noise.

### Phase 5 — docs

- Update `CLAUDE.md`: storage section (unpacked trees are gone; add the new
  tar + `DATASET_TAR`), GT provenance (official ScanNet GT is now the default
  supervision; SAM3 tar kept as baseline), and the audit finding.
- Update `docs/MILESTONES.md` + `docs/todo.md`: audit numbers, migration, new
  baseline comparison. Move this plan to `docs/old/` once executed.

## 5. Risks / fallbacks

- **TUM download server slow/down** — a known, recurring problem (it's the
  whole reason `download_sens.py` exists). The adapted downloader fails fast
  and resumes; if the server is down for days, just re-run later. If the URL
  scheme itself changed, stop and report — do not hunt for mirrors.
- **Misalignment between instance-filt and color frames** (Phase 0 gate 4
  fails) → the projections may target the depth camera; then we'd need to
  warp via intrinsics (color↔depth registration from the `.txt` meta file).
  Do not proceed past Phase 0 with visible misalignment.
- **Official instance ids not temporally stable** — they are (ids come from
  the single 3D annotation), but the Phase 2 visual QA catches surprises.
- **Scratch purge** mid-build → keep zips deletable, converter resumable
  per-scene (skip scenes whose `raw_data/masks_instance` already exists,
  SAM3-build style `.complete` markers).
- **Class imbalance shifts** (official GT includes small/occluded instances
  SAM3 missed) → not a blocker; record instance-count distribution in QA and
  mention it when comparing runs.
