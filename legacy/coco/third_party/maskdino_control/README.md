# `maskdino_control` — official MaskDINO, our recipe

The control arm of `docs/MASKDINO_COCO.md` §6. It trains **upstream's** MaskDINO on COCO with
**our** recipe, so §6's comparison stops leaning on a checkpoint we never trained.

Everything here lives in this repo and points at the pristine clone at
`/cluster/home/niacobone/MaskDINO` from outside. **Do not edit the clone** — `docs/MASKDINO.md`
§7.6 (the weight-transplant equivalence check) depends on it being untouched.

| file | contents |
|---|---|
| `configs/maskdino_upstream_matched.yaml` | the run. `_BASE_` is upstream's own config; every key in it is one of the deviations `docs/MASKDINO_COCO.md` §2 lists |
| `configs/maskdino_upstream_matched_overfit.yaml` | the §4.1 overfit gate: 64 images, train == val |
| `squash_mapper.py` | replaces LSJ@1024 with our 518² squash + hflip 0.5 |
| `lr.py` | our arms' linear-warmup→cosine-to-0.01 lambda, copied verbatim from `scripts/train_maskdino_coco.py` |
| `config.py` | the `CONTROL` config namespace |
| `train_control.py` | the driver: dataset registration, wall-clock self-stop, `summary.json` |
| `make_overfit_root.py` | builds the 64-image COCO root the gate uses (4 inodes) |
| `build_ops.sh` | rebuilds MSDeformAttn for sm_80 **out of tree** into `ops_build/` |

## Two traps worth knowing

**The CUDA op must carry sm_80.** The `.so` in the clone's venv was built on an sm_86 node and
holds sm_86 cubin + sm_86 PTX only. PTX is forward-compatible, so it JITs onto sm_89 but *not*
onto the older sm_80 of an A100 — and upstream's `MSDeformAttn.forward` wraps the CUDA call in a
bare `except:` that falls back to the pure-pytorch core. Wrong arch is therefore a **silent ~10×
slowdown**, not a crash. `build_ops.sh` rebuilds for `8.0;8.6` into `ops_build/` (gitignored,
`*.so`), `train_control.py` puts that ahead of site-packages, and `assert_cuda_msda()` calls the
kernel unwrapped at startup so a missing build fails loudly.

**The resubmit cannot rely on being killed.** At the wall clock SLURM tears down the whole batch
script, so a trailing `sbatch` never runs. `CONTROL.TIME_BUDGET_HOURS` makes python stop itself,
checkpoint, and exit **without** writing `summary.json`; that absence is what
`slurm/train_maskdino_upstream.sh` tests. Same mechanism, same reason as
`slurm/train_maskdino_coco.sh`.

**The reference venv is on scratch, and scratch purges.** It was destroyed mid-run on 2026-08-10
(docs/MASKDINO_COCO.md §6.1). Rebuild:

```bash
rm -rf "$MASKDINO_ROOT/myenv"
export TORCH_CUDA_ARCH_LIST="8.0;8.6"     # REQUIRED: install.sh's op build dies without it on a
bash "$MASKDINO_ROOT/install.sh"          # CPU node — IndexError in _get_cuda_arch_flags
```

Diagnose with the `.py` vs `.pyc` counts (a healthy tree is ~1:1; the damaged one was 27:1848),
never from the error text — a purged PIL reports a perfectly good JPEG as
`UnidentifiedImageError`. Then resume with `sbatch --export=ALL,RUN_DIR=<run_dir>
slurm/train_maskdino_upstream.sh`; checkpoints live on `/cluster/work` and are not purged, so
nothing is lost but the eval scheduled at the crash step.

## Running

```bash
bash third_party/maskdino_control/build_ops.sh                      # once
python third_party/maskdino_control/make_overfit_root.py --n 64     # once
/cluster/home/niacobone/MaskDINO/myenv/bin/python tests/test_maskdino_upstream_control.py

sbatch --time=4:00:00 --export=ALL,GATE=1 slurm/train_maskdino_upstream.sh   # the gate
sbatch slurm/train_maskdino_upstream.sh                                      # the run
```

The tests need the **reference** env (`$MASKDINO_ROOT/myenv`, py3.9 + detectron2 0.6), not the
project's `myenv/`.
