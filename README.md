# 3D instance segmentation on a strictly frozen VGGT backbone

> **Research project at ETH Zurich, Photogrammetry & Remote Sensing — closed 2026-09-09.**
> A MaskDINO-family decoder trained for multi-view-consistent 3D instance segmentation on top of a
> **frozen** VGGT-1B backbone, scored with the official ScanNet 3D instance evaluator. This
> repository is a fork of
> [facebookresearch/vggt](https://github.com/facebookresearch/vggt); the upstream model in
> [`vggt/`](vggt/) is untouched and never trained.
>
> **Read [`docs/FINAL_REPORT.md`](docs/FINAL_REPORT.md) first** — it is the complete closing
> account: what was built, what was measured, why it stopped, and what a successor should do.

---

## The question

Every published feed-forward competitor in this space — SegVGGT, FAST3DIS, IGGT — **adapts** a
VGGT-family backbone, typically with LoRA and around 16 GPU-days. Nobody had published the
unadapted case. So: can a **strictly frozen** 3D foundation model support multi-view-consistent 3D
instance segmentation through a small trained decoder, and if not, by how much does it fall short?

Supervision: official **ScanNet v2** 2D instance annotations, official **1201 train / 312 val**
split. Scoring: the **official ScanNet 3D instance evaluator, vendored unmodified**
([`train/benchmark3d.py`](train/benchmark3d.py)), on the official benchmark point clouds.

## The answer

**A frozen backbone is enough to be interesting, and not enough to be competitive.**

In-domain — training on ScanNet, evaluated unposed and class-agnostic at the competitors' own
50-view budget — the model leads both published feed-forward methods on all three columns, at
**~0.8 GPU-days against their ~16**:

| Method | Backbone | Views | AP | AP50 | AP25 |
|---|---|---|---|---|---|
| IGGT *(as re-evaluated by FAST3DIS)* | adapted | 50 | 0.028 | 0.112 | 0.287 |
| FAST3DIS | LoRA-adapted DA3 | 50 | 0.038 | 0.096 | 0.316 |
| **This work — ScanNet-trained** | **frozen VGGT-1B** | 46.7 | **0.053** | **0.170** | **0.542** |
| **This work — ScanNet REMOVED from training**, IGGT's mixture reproduced in full | 〃 | 17.4 | 0.009 | 0.032 | 0.301 |

**The last two rows are one result and must be read together.** Both competitors are *zero-shot on
ScanNet*; the headline row is not. Level the training data and the method is **~3× behind at
AP50** — while staying **level at AP25**, the coarse-localisation bar, on a third of their views.
So the frozen backbone finds and roughly places objects about as well as adapted ones do, and
delineates them considerably worse.

That gap, plus an exhausted compute allocation, is why the project closed. The full reasoning,
the ablation ranking, the posed/unposed decomposition and the transfer results are in
[`docs/FINAL_REPORT.md`](docs/FINAL_REPORT.md).

## How it works

```
frozen VGGT-1B  ──►  aggregated_tokens_list[-1]        cached once per scene, under no_grad
                     F : [B, S, P, 2048]
                              │
                     ┌────────┴──────────────────────────────────┐
                     │  pixel decoder — ViTDet 3-level pyramid    │  models/maskdino/pixel_decoder.py
                     │  + MSDeformAttn encoder                    │
                     └────────┬──────────────────────────────────┘
                              │
                     ┌────────┴──────────────────────────────────┐
                     │  MaskDINO decoder — two-stage selection,   │  models/maskdino/decoder.py
                     │  DAB anchors, denoising, deep supervision  │
                     │  + cross-frame attention (--multi_frame)   │  models/maskdino/multiframe.py
                     │  + optional 3D anchors  (--anchor_3d)      │  models/maskdino/anchor3d.py
                     └────────┬──────────────────────────────────┘
                              │
                     pred_masks : [B, N, S, h, w]
                     one query = one instance across ALL views, by construction
                              │
                     unproject with VGGT's OWN predicted depth + cameras
                     → per-superpoint majority vote → Sim(3)+ICP → 3D instances
```

Three design points define the study:

- **The backbone is never updated.** Features are cached once per scene, so head-only training
  takes minutes per epoch and the headline run costs ~0.8 GPU-days.
- **Multi-view consistency is structural.** A query owns one instance in every frame of a bundle —
  no mask matching, no fusion, no tracking stage.
- **Inference uses no ground-truth geometry.** A second *posed* bridge (GT poses + sensor depth,
  the protocol SegVGGT publishes on) exists only to separate mask quality from geometry quality;
  its oracle returns 99.99 % of annotated vertices to their own instance.

## What the study found

Every Δ read against a measured seed spread of 0.009 per-bundle AP50:

| Lever | Effect on 3D AP50 |
|---|---|
| Training data, 1201 → 3520 scenes | **+0.023** — larger than any decoder ingredient |
| Cross-frame attention (removed) | **−57 %** |
| Bundle features → per-frame features | −24 % class-aware / −49 % class-agnostic |
| 3D anchors instead of 2D DAB boxes | **+66 %** |
| View budget 17 → 50 | +24 %, then saturates |
| Lifting knobs (vote radius, depth confidence) | +0.016 → +0.047 |
| Mask resolution 37² → 74² | −0.022 — **not the bottleneck** |

1. **Data-limited, not architecture-limited.** More data moves the result further than removing any
   single MaskDINO ingredient (+0.023 against ≤0.005).
2. **Recognition and cross-view identity are separate axes.** 3D anchors are AP-neutral in 2D and
   worth +66 % in 3D — a mechanism the 2D ruler is blind to.
3. **The 2D→3D lifting binds, not the decoder.** AP25 ≈ 4× AP50 throughout; the unposed bridge
   costs 2.3× on identical masks; every out-of-domain unposed cell reads 0.000 AP. Since the view
   budget saturates by 50, what remains is **registration**, not coverage.

## Repository layout

| path | what it is |
|---|---|
| [`models/maskdino/`](models/maskdino/) | the model — pixel decoder, MaskDINO decoder, multi-frame and 3D-anchor extensions |
| [`train/`](train/) | data pipeline, feature cache, the 2D protocol, the 3D ruler and its four dataset adapters, the **vendored official ScanNet evaluator** |
| [`scripts/`](scripts/) | entry points — training, 3D evaluation, per-frame baseline scoring, visualisation |
| [`data/`](data/) | dataset loaders and the ScanNet200 taxonomy |
| [`tests/`](tests/) | 22 standalone CPU-only test scripts + 3 shell harnesses; no GPU, no backbone weights |
| [`slurm/`](slurm/) | cluster drivers — training, evaluation matrices, dataset fetch/build |
| [`docs/`](docs/) | the complete measurement record — see the index below |
| [`legacy/`](legacy/) | the retired predecessor head and the one-shot dataset builders; frozen, still imported by active code |
| [`vggt/`](vggt/), [`training/`](training/) | **untouched upstream VGGT** — the frozen backbone and upstream's own finetuning framework (unused here) |

Material retired at project close — the pre-official-split archive, the supervisor decks, the COCO
arm and upstream's demo media — was deleted from the tree and stays in git history;
[`docs/FINAL_REPORT.md`](docs/FINAL_REPORT.md) §8.1 lists it.

## Documentation

Start with the final report; the rest is the underlying record.

| document | what it holds |
|---|---|
| **[`docs/FINAL_REPORT.md`](docs/FINAL_REPORT.md)** | **the closing account — read this first** |
| [`docs/MASKDINO.md`](docs/MASKDINO.md) | architecture, deviations from upstream MaskDINO, the protocols in full |
| [`docs/RESULTS.md`](docs/RESULTS.md) | every number, one home |
| [`docs/MULTIDATASET.md`](docs/MULTIDATASET.md) | the multi-dataset training arms, incl. the no-ScanNet result |
| [`docs/TRAINING_COMPARABILITY.md`](docs/TRAINING_COMPARABILITY.md) | what each competitor trains on vs evaluates on, axis by axis |
| [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md) | competitor landscape and positioning |
| [`docs/SEGVGGT_ANALYSIS.md`](docs/SEGVGGT_ANALYSIS.md) | the closest competitor dissected |
| [`docs/COMMANDS.md`](docs/COMMANDS.md) | the full command catalogue, with the caveat each one needs |
| [`docs/DATASET.md`](docs/DATASET.md) | ground-truth provenance, mask conventions, the tars |
| [`docs/FACTSHEET.md`](docs/FACTSHEET.md) | the frozen outward-facing read-out as it stood at closure |
| [`docs/RESTORE.md`](docs/RESTORE.md) | environment rebuild and cluster-archive layout |
| [`docs/todo.md`](docs/todo.md) | the work ledger, frozen at closure |

## Running it

The project ran on a SLURM GPU cluster whose allocation has since been released; **no checkpoints
or cached features are in this tree**, and the datasets are licence-gated. What is reproducible
from the repository as it stands is the code, the tests and the exact protocols.

```bash
# Environment (torch 2.3.1+cu121 — the versions every published number was produced with).
python -m venv myenv && source myenv/bin/activate
pip install -r requirements.txt -r requirements_demo.txt

# Tests — standalone scripts, not pytest. CPU-only, no backbone weights needed.
for t in tests/test_*.py; do python "$t"; done
bash tests/test_train_maskdino_sh_lists.sh
bash tests/test_train_maskdino_multi_sh.sh
bash tests/test_eval_3d_matrix_sh.sh

# Training (needs the ScanNet tars, see docs/DATASET.md)
python scripts/train_maskdino.py --train_scenes scene0000_00 --val_scenes scene0080_00 \
    --num_epochs 50 --num_queries 300 --scans_root <scans_root>
sbatch --export=ALL,N_SCENES=490 slurm/train_maskdino.sh

# The 3D ruler — the only protocol placeable next to a published number
sbatch --export=ALL,CHECKPOINT=<run_dir>/checkpoint_best_bundle.pth slurm/eval_3d_maskdino.sh
```

The full catalogue — official-split recipes, `--anchor_3d`, view-budget and full-resolution
evaluation, the two 3D transfer modes and their oracle, dataset rebuilds — is in
[`docs/COMMANDS.md`](docs/COMMANDS.md).

## Reading the numbers

Four rules, because they are the errors most likely to be made by someone reading the tables cold.
The complete list is §9 of the final report.

1. **2D and 3D numbers are different rulers.** The 2D figures in `docs/RESULTS.md` come from this
   project's own metric code and may never be placed next to a published figure.
2. **Posed and unposed are different protocols.** SegVGGT publishes posed; FAST3DIS, IGGT and this
   work's headline are unposed. The bridge between them is worth a consistent 2.3×.
3. **Class-aware and class-agnostic are different columns.** FAST3DIS and IGGT publish
   class-agnostic only.
4. **Quote the headline with its no-ScanNet row.** The lead rests on training data the competitors
   never use; the row that prices it is in the same table.

## Context and credits

Research project carried out at **ETH Zurich — Photogrammetry & Remote Sensing**, June–September
2026. It followed, and is independent of, the author's master's thesis.

**Author** — Nicolò Iacobone

**Supervision**

- **Christos Sakaridis** — Lecturer, and Head of the Artificial Visual Intelligence group,
  Photogrammetry and Remote Sensing lab, ETH Zurich
- **Mattia Segù** — Research Scientist, Google Zurich
- **Tatiana Tommasi** — Full Professor, Department of Computer and Control Engineering,
  Politecnico di Torino (Italy); Director of the ELLIS Unit Turin

## Upstream VGGT

This fork does not modify VGGT and does not train it. The backbone
([`vggt/models/vggt.py`](vggt/models/vggt.py)) is loaded frozen, run under `no_grad`, and hooked at
`aggregated_tokens_list[-1]`. Upstream's documentation, demos and finetuning framework are
preserved as they were; upstream's `examples/` demo media has been removed from this fork to keep
the clone small (the demos filter their example gallery by file existence and start without it).

For the original model, its paper and its own README, see
[facebookresearch/vggt](https://github.com/facebookresearch/vggt).

```bibtex
@inproceedings{wang2025vggt,
  title={VGGT: Visual Geometry Grounded Transformer},
  author={Wang, Jianyuan and Chen, Minghao and Karaev, Nikita and Vedaldi, Andrea
          and Rupprecht, Christian and Novotny, David},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```

## License

See [LICENSE.txt](./LICENSE.txt) for the terms this code is made available under — inherited from
upstream VGGT and unchanged. Note that only the
[VGGT-1B-Commercial checkpoint](https://huggingface.co/facebook/VGGT-1B-Commercial) permits
commercial use; the original checkpoint does not.

The ScanNet 3D instance evaluator vendored in [`train/benchmark3d.py`](train/benchmark3d.py) is the
official one, redistributed under its own terms. ScanNet v2, ScanNet++, Replica and Aria Synthetic
Environments are licence-gated and are not redistributed here — this repository contains only the
tooling that fetches and builds them.
