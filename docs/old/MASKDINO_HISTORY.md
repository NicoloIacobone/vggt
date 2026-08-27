# MASKDINO — ARCHIVE: the COCO port check and the pre-official-split diagnostics

**Nothing in this file is current.** Moved out of `docs/MASKDINO.md` on 2026-08-27, when the
project's documentation was cut back to the MaskDINO work on the official ScanNet v2 1201/312
split and larger. Two things live here:

- **§7.6** — the COCO upstream-equivalence check. It proved the port was faithful; that job is
  done, and COCO is not a ruler this project reports on. The rest of that study is in
  `docs/old/MASKDINO_COCO.md`.
- **§9.4 / §9.5** — the 3D diagnostics run on a checkpoint whose training scenes (0000–0489)
  **overlapped the official val-312 split**. Those numbers were never reportable and are not
  quotable now; the rule they produced (val-312 never enters training) is stated in the live
  `docs/MASKDINO.md` §9.4 instead.

Section numbers are the ones these sections had in `docs/MASKDINO.md`.

---

### 7.6 Upstream-equivalence check on COCO (job 8967932, 2026-07-29)

Everything above measures the port against *our own* baselines, which cannot detect a bug that
is faithfully wrong in both. This check closes that loop: it drives **our** ported modules with
**upstream's released COCO weights** and asks whether they reproduce upstream's published COCO
val2017 numbers.

`scripts/coco_transplant_eval.py` loads
`maskdino_r50_50ep_300q_hid1024_3sd1_instance_maskenhanced_mask46.1ap_box51.5ap.pth`
(config `maskdino_R50_bs16_50ep_3s.yaml` — the one our defaults mirror) into upstream's
detectron2 harness, then swaps in our `MaskDINODecoder` and our `MSDeformAttnEncoder`.
`--mode baseline` leaves upstream untouched and is the control.

**The decoder accepts upstream's weights at `strict=True`: 333/333 parameters, names and shapes.**

| COCO val2017, 5000 images | segm AP | segm AP50 | box AP | box AP50 |
|---|---|---|---|---|
| upstream model zoo, row "MaskDINO (hid 1024)" — *this checkpoint* | 46.1 | — | 51.5 | — |
| `--mode baseline` (upstream code, this env) | 46.129 | 69.021 | 51.540 | 70.509 |
| **`--mode ours` (ported modules)** | **46.133** | 69.036 | **51.549** | 70.514 |

**The 46.1 / 51.5 target is the README model-zoo figure for the exact checkpoint used, not a
paper table value** — get this right, the paper's numbers are different. Table 3's 50-epoch /
300-query ResNet-50 rows are **46.0 / 50.5** (plain) and **46.3 / 51.7** (‡, mask-enhanced box
init). The ‡ row corresponds to the *other* released checkpoint, `hid2048` (52 M params,
286 GFLOPs); ours is the narrower `hid1024` variant (47 M, 226 GFLOPs — encoder FFN 1024 instead
of 2048, which is also why `maskdino_R50_bs16_50ep_3s.yaml` is the matching config). Comparing
our run against 46.3 would be comparing against a wider model we never ran.

The comparison that actually carries the verdict is **ours vs `--mode baseline`** — same code
path, same env, same weights, same data — and there the gap is 0.004 AP.

**Δ = +0.004 segm AP / +0.009 box AP.** On CPU the two modes are *bit-identical* to every
printed digit; the ~0.005 AP drift appears only on GPU, because upstream calls the fused CUDA
MSDeformAttn kernel there while our port always uses the `grid_sample` core (§2.1). That is the
one intended difference, and it is worth 0.01 AP.

**Verified as a live path, not a no-op.** `transplant()` asserts the modules are ours
(`models.maskdino.decoder` / `models.maskdino.pixel_decoder`, 6 + 9 ported `MSDeformAttn`
instances), and `--perturb 1.05` scales one weight inside *our* decoder: segm AP moves
55.702 → 55.608 on the 10-image subset. Identical numbers therefore mean equivalence, not a
silent fallback to upstream code.

**One trap this surfaced, worth knowing before touching level plumbing.** Upstream's pixel
decoder returns `multi_scale_features` LOW→HIGH resolution and its decoder walks that list
*backwards* (`idx = num_feature_levels-1-i`); our port takes it HIGH→LOW and walks forwards.
Both flatten the same tensors in the same order only because the adapter reverses the list.
The decoder's own `input_proj` is an empty `nn.Sequential` under this config, so its index
convention carries no weights and the difference is invisible in a state-dict diff.

**Scope — what this does and does not certify.**

| Certified by this check | Not exercised |
|---|---|
| `ms_deform_attn.py` (encoder *and* decoder) | `matcher.py`, `criterion.py` — training only |
| `pixel_decoder.py`: encoder layer/stack, reference points | DN query generation in `decoder.py` |
| `decoder.py`: two-stage selection, DAB anchors, iterative box refinement, mask-enhanced box init, prediction heads | `multiframe.py` — no upstream counterpart |
| `decoder_layers.py`, `utils.py`, `box_ops.masks_to_boxes` | the VGGT ViTDet pyramid (§3) — no COCO counterpart |

The right-hand column is not a known problem, just untested by *this* route; the loss path still
rests on `tests/test_maskdino_loss.py` (perfect-prediction zero loss) and the overfit tests.
Since the multi-frame work edits the matcher and the criterion, that is the column to extend
next if more assurance is wanted.

Reproduce: `sbatch slurm/coco_transplant.sh` (~32 min on one RTX 3090, both modes; results land in
`/cluster/work/igp_psr/niacobone/distillation/output/coco_transplant/`).
COCO val2017 lives at `/cluster/scratch/niacobone/coco` — **global scratch is purged after
15 days**, so re-download (§ the script header) if it has vanished.


### 9.4 Honesty: which checkpoint may quote which number

The official val-312 split overlaps our conventional training range (scenes 0000–0489), so **any
existing checkpoint's 3D numbers are DIAGNOSTIC only** — they verify the pipeline, they are not
reportable. The reportable number needs a checkpoint trained on the official 1201-scene split
(tar built, docs/todo.md 1c) with val-312 never seen. A further caveat for the current diagnostic
checkpoint (`maskdino_sf_n490_mf_b2jit_20260730_105117/checkpoint_best.pth`): it is the epoch-17
mIoU-selected checkpoint — the epoch-19 AP50-selected one that carried the 0.515 bundle headline
did not survive the 2026-07-30 output cleanup (bundle AP50 0.461 at epoch 17).

### 9.5 Diagnostic runs on the leaked checkpoint (2026-08-01, jobs 9327269 / 9327271)

Numbers: `docs/RESULTS.md` §5. Full narrative and per-scene readings:
`docs/old/MASKDINO_RESULTS_HISTORY.md`. Kept here because two of its findings are structural and
the rest of §9 argues from them:

1. **Geometry binds, not recognition.** AP25 ≈ 4–5× AP50: objects are found and coarsely
   localised, but the lifted masks miss the >0.5-IoU bar. Median camera-centre RMS after Sim(3) is
   **0.14 m** and ICP point RMS **~0.10 m** — the same order as the vote radius, and VGGT's own
   depth/pose drift over a whole-scan S≈17 bundle, not a 2D mask-quality problem (the same model
   scores 0.65–0.67 per-frame AP50). This is the price of "no GT geometry at inference".
2. **Coverage is the second cap:** ~15 % of mesh vertices receive any vote, ~63 % of annotated
   vertices get assigned. Every unassigned GT instance is a hard FN.
3. **S-generalisation works as designed:** bundles of 3–55 frames (median 15) through a model
   trained at S=8, no failures — `CrossFrameAttention` has no frame positional encoding, and the
   two-stage top-k just unions over more frames.
