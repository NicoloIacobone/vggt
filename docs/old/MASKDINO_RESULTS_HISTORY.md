# Archive — MaskDINO results narrative (run-by-run)

Cut from `docs/MASKDINO.md` on 2026-08-06 during the doc pruning. **Every number here also lives
in `docs/RESULTS.md`, which is the current home.** This file exists for provenance: job ids, the
reasoning as it was written at the time, and the sanity checks that are no longer worth carrying in
the primary document. Do not cite it as a source of truth.

---

## 7. Results

### 7.1 Machinery check (2026-07-27)

CPU test suite green: the pure-PyTorch deformable attention matches a naive explicit-loop
reference to 1e-5; the decoder produces the right shapes for every two-stage × DN × box-init
combination; the matcher recovers a planted assignment; perfect predictions drive every loss
term to ~0.

GPU smoke test — 4 train scenes / 2 val scenes, 32 training frames, full recipe (300 queries,
6 encoder + 9 decoder layers, two-stage, DN "seg", mask-enhanced box init), 200 epochs in
**6.1 min** on one RTX 4090 (0.46 s/step, 24 M trainable params, backbone cached in 12 s):

| | mIoU | AP50 | AP75 | class_acc |
|---|---|---|---|---|
| train (memorised) | **0.969** | **0.992** | 0.982 | 1.000 |
| val (2 unseen scenes) | 0.095 | 0.095 | 0.059 | 0.29 |

The train row is a *sanity check, not a result*: 32 frames are trivially memorisable. What it
proves is that gradient flow, Hungarian matching, DN, box refinement and the metric path all
work end to end.

### 7.2 Data scaling (jobs 8748952 / 8754527 / 8774050)

All runs: official 500-scene GT, per-instance masks, val = scenes 0080-0089, identical recipe
(300 queries, 6 encoder + 9 decoder layers, two-stage, DN "seg", mask-enhanced box init),
epochs auto-scaled to hold the ~20-29 k gradient-step budget. All COMPLETED cleanly.

| Run | Scenes | val mIoU | val AP50 | val AP75 | val mAP | peak @ | train mIoU |
|---|---|---|---|---|---|---|---|
| **arm C — the bar** | 190 | 0.451 | 0.294 | 0.141 | 0.154 | converged | — |
| job 8748952 | 50 | 0.451 | 0.440 | 0.314 | 0.290 | ep 150/400 | 1.000 |
| job 8754527 | 190 | 0.594 | 0.624 | 0.440 | 0.418 | ep 38/100 | 0.994 |
| **job 8774050** | **490** | **0.669** | **0.699** | **0.506** | **0.475** | ep 31/60 | 0.947 |

**At 490 scenes this head beats arm C by +48 % mIoU, +138 % AP50, 3.6x AP75, 3.1x mAP.** The
curve is still rising at the largest scale available (0.440 -> 0.624 -> 0.699 AP50 for
50 -> 190 -> 490 scenes) and the overfitting eases as data grows (train mIoU 1.000 -> 0.994 ->
0.947), so the model remains data-limited even at 490 scenes. Every run still peaks around
half-way through its schedule; `checkpoint_best.pth` captures it.

**This inverts the project's data-scaling conclusion.** Arm C got *worse* with more data
(0.367@190 -> 0.350@490, `docs/ARMS_SUMMARY.md`), which read as "the dataset is not the
bottleneck". On the same data MaskDINO gains +0.26 AP50 going 50 -> 490. The D4RT head was
**architecture-limited, not data-limited**; the old scaling result was a property of that head,
not of the task.

### 7.2.1 Ablations — no single ingredient carries the win

Each removes ONE MaskDINO component at N=190, everything else identical (jobs 8774052 /
8778736 / 8774056 / 8774065, all COMPLETED). Sorted by cost of removal:

| Config | val mIoU | val AP50 | ΔAP50 vs full | train mIoU |
|---|---|---|---|---|
| full recipe | 0.594 | 0.624 | — | 0.994 |
| `--no-two_stage` (no query selection) | 0.592 | 0.578 | −0.046 | 0.995 |
| `--enc_layers 0` (no deformable encoder) | 0.551 | 0.580 | −0.044 | 0.871 |
| `--dn no` (no denoising) | 0.586 | 0.594 | −0.030 | 0.986 |
| `--initialize_box_type no` (no mask-enhanced box init) | 0.610 | 0.608 | −0.016 | 0.993 |

Two honest readings:

1. **No component is decisive.** Each is worth 0.02-0.05 AP50, and box-init is within the
   ±0.04 eval-to-eval noise (its mIoU is actually *higher* than the full recipe's). The full
   recipe is still the best AP50 of the five, so the pieces are additive rather than redundant —
   but nobody should claim "denoising is what made this work".
2. **Every crippled variant still beats arm C by ~2x on AP50** (0.578-0.608 vs 0.294). The credit
   belongs to the architecture *class* — deformable attention over a multi-scale pyramid,
   per-layer anchor-box refinement, deep supervision, 20.5 M params — not to any one trick.
   And **data scale dominates all of it**: +0.26 AP50 from 50->490 scenes, versus ≤0.05 from any
   single component.

`--enc_layers 0` is the only ablation that also drops train mIoU (0.871 vs ~0.99), i.e. it
removes real capacity rather than just a training aid.

### 7.2.2 Cost note — eval must not scale with the training set

The first N=200 submission (job 8748972) reached only epoch 2 in 30 minutes and was cancelled:
it scored **all 190 train scenes** at every eval, and `_average_precision` loops over every kept
prediction at 10 IoU thresholds, so ~1600 frames x ~180 ms = ~5 min per eval, every 2 epochs.
Two fixes: `--eval_topk 100` (COCO's `test_topk_per_image` — protocol-correct *and* 3x faster per
frame) and `--eval_train_scenes 10` (the train metric is only an overfit read-out). Eval went
~180 s -> ~6 s.

### 7.3 The baseline these beat (measured 2026-07-27)

Arm C (learned object queries, the best D4RT head) scored under **this per-frame protocol** via
`scripts/eval_perframe.py` on `d4rt_full_inst_learned_officialgt_20260708_124452` (the run whose
multi-view numbers are the quotable 0.367 / 0.199), all 10 val scenes (0080–0089), unprompted
learned queries:

| | mIoU | AP50 | AP75 | mAP |
|---|---|---|---|---|
| **arm C, per-frame — the bar** | **0.451** | **0.294** | 0.141 | 0.154 |
| arm C, per-bundle (reference only, NOT comparable) | 0.367 | 0.199 | — | — |

Per-scene spread: mIoU 0.34–0.61, AP50 0.18–0.43. Full JSON:
`<run_dir>/perframe_eval_checkpoint_best.json`.

Per-frame scores *higher* than per-bundle for the same checkpoint, which is expected and worth
stating explicitly: an instance only has to match in the frames where it is visible, and an
empty prediction in a frame where the object is absent is dropped rather than penalised (§6.3).
**This is exactly why the MaskDINO numbers must be read against 0.451 / 0.294 and never against
0.367 / 0.199.**

### 7.4 Multi-view features, mask resolution, view draws, multi-frame (2026-07-28)

Five runs, all at N=490 against the **0.669 / 0.699** single-frame bar, identical recipe
otherwise, official 500-scene GT, peak (`checkpoint_best*`) numbers:

| Job | Change | val mIoU | val AP50 | ΔAP50 | train mIoU @peak |
|---|---|---|---|---|---|
| 8774050 | — (the bar) | 0.669 | 0.699 | — | 0.947 |
| 8895540 | `--feature_mode bundle` | 0.622 | 0.651 | **−0.048** | 0.872 |
| 8895551 | `--mask_upsample 2` | 0.662 | 0.677 | −0.022 | 0.812 |
| **8895565** | **`--bundles_per_scene 2 --color_jitter 0.2`** | **0.694** | **0.729** | **+0.030** | 0.816 |
| 8900100 | `--multi_frame --feature_mode bundle` | 0.621 | 0.630 | −0.069 | 0.867 |

1. **Multi-view-aware tokens are not free — they cost 0.048 AP50** (§8.1). Running the aggregator
   over the bundle mixes the views inside the frozen features, and per-frame segmentation gets
   *worse*, not better. So §8.1 is a **negative result**, and it is the correct control for the
   multi-frame decoder (below), not the bar.
2. **`--mask_upsample 2` is neutral** (−0.022, inside the ±0.04 eval-to-eval noise) — the same
   verdict the D4RT arms reached on a different head. Masks stay on the 37×37 patch grid.
3. **More view draws per scene is the best lever left**: +0.030 AP50 and a *new best* 0.729, with
   train mIoU dropping 0.947 → 0.816 (less memorisation). The model is still data-limited, and
   since the tar holds no more scenes, views-per-scene is the cheapest remaining data. **Honest
   caveat:** the `EPOCHS=30` this job was submitted with was silently clamped back to 60 by
   `slurm/train_maskdino.sh` (fixed 2026-07-28), so it had a 2× larger step budget available;
   it nevertheless *peaked* at epoch 19 = 18.6 k steps, against the bar's peak at 15.2 k, so the
   gain is not simply "trained longer". `--bundles_per_scene 4` **saturates** (§7.4.1).
4. **Multi-frame** (§8.2) costs 0.069 AP50 against the bar — but 0.048 of that is the bundle
   features it is built on. Against its proper control (8895540, 0.651) the shared-query decoder
   is **−0.021 AP50 per frame, i.e. neutral inside the noise band**, and in exchange it produces
   a genuine multi-view result (below). The 2026-07-29 ablations decouple the ingredients
   (§7.4.1).

**The per-bundle number is back.** Job 8900100 scores, on the multi-view protocol of arms A–E:

| | mIoU | AP50 | AP75 | mAP |
|---|---|---|---|---|
| arm C (best D4RT head), per-bundle | 0.367 | 0.199 | — | — |
| **MaskDINO `--multi_frame`, per-bundle** | **0.535** | **0.494** | 0.279 | 0.272 |

**+46 % mIoU and 2.5× AP50 over the best D4RT arm on the arms' own ruler** — one query is one
instance across all 8 views, no post-hoc matching. Read it *only* against 0.367 / 0.199, never
against the per-frame 0.699 (docs/RESULTS.md §1).

### 7.4.1 Multi-frame ablations + bundle saturation (jobs 8950610 / 8950613 / 8950617, 2026-07-29)

All at N=490, otherwise the full recipe. Peak numbers per metric family (the per-frame and
per-bundle peaks can fall on different epochs; `checkpoint_best.pth` / `checkpoint_best_ap50.pth`
select on the *per-frame* metrics — `checkpoint_best_bundle.pth` (§8.2, docs/todo.md 2b) selects
on `bundle_AP50` and is what these runs would have used had it existed at the time):

| Job | Config | per-frame mIoU / AP50 | bundle mIoU / AP50 | Δbundle AP50 |
|---|---|---|---|---|
| 8900100 | `--multi_frame --feature_mode bundle` (full) | 0.621 / 0.630 | **0.535 / 0.494** | — |
| 8950617 | … `--no-cross_frame_attn` | 0.530 / 0.524 | 0.393 / 0.311 | **−0.183** |
| 8950613 | … `--feature_mode single` (per-frame features) | 0.631 / 0.627 | 0.429 / 0.347 | **−0.147** |

Two findings, both load-bearing for the multi-frame story:

1. **Cross-frame attention is the main carrier of the multi-view result** — the single decisive
   component this track has found (the single-frame ablations in §7.2.1 found none). Removing it
   costs −0.183 bundle AP50 *and* −0.106 per-frame AP50: with shared queries but no cross-frame
   communication, the per-frame task gets harder too (one content vector must serve S views it
   can no longer reconcile).
2. **Bundle features are required for multi-view consistency.** §8.1 measured them as a *negative*
   for per-frame accuracy (−0.048 AP50), but swapping them out of the multi-frame model costs
   −0.147 bundle AP50 while leaving per-frame intact (0.627 ≈ the bundle-features control 0.651
   region). Read together: VGGT's global attention writes cross-view correspondence into the
   frozen tokens, the decoder's cross-frame attention consumes it, and the price is per-frame
   accuracy — **consistency is not free, and it is now quantified** (0.729 single-frame best vs
   0.630 per-frame for the best multi-view model).

**Bundle saturation** (job 8950610, `--bundles_per_scene 4 --color_jitter 0.2`, `EPOCHS=15`):
**0.699 mIoU / 0.722 AP50** — mIoU a hair above the b2 run (0.694), AP50 a hair below (0.729),
i.e. inside the noise band. The views-per-scene lever saturates at 2 draws; the remaining data
lever is **more scenes** (the tar holds 500; ScanNet v2 has 1201 official train scenes).

### 7.5 Official-split read-out (job 8900194)

Same recipe, but val = the 77 official ScanNet v2 val scenes inside our tar and train = the other
413 (docs/RESULTS.md §1.1). Peak **val mIoU 0.589 / AP50 0.604** (epoch 27; the job hit its 12 h
wall clock at epoch 39/60, past its peak, so the number stands).

**Our convention split is ~0.10 AP50 "easier" than the official val scenes** — with 67 fewer
training scenes as a partial explanation. Quote 0.699 as *our* split's number and mention 0.604
whenever comparability to the ScanNet literature comes up (it is still a per-view 2D-mask number,
so it is not a leaderboard-comparable figure either — docs/RELATED_WORK.md).

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

### 7.7 The resolution verdict (2026-07-30, jobs 9073136 / 9072738 / 9072749 / 9072761)

Three mutually consistent measurements close the "is the 37×37 grid the bottleneck?" question
on ScanNet. All on the full-resolution ruler of §6.5.

**The GT-only ceiling** (`scripts/scannet_mask_resolution_oracle.py`, job 9073136, val scenes
0080–0089, full JSON in the output dir):

| prediction grid | mIoU | AP50 | AP75 | mAP |
|---|---|---|---|---|
| **37×37** (native patch grid) | 0.910 | **0.956** | 0.919 | 0.863 |
| 74×74 (`--mask_upsample 2`) | 0.963 | 0.992 | 0.982 | 0.953 |
| 148×148 (`--mask_upsample 4`) | 0.983 | 0.997 | 0.995 | 0.988 |
| 259×259 | 0.994 | 1.000 | 1.000 | 0.999 |
| 518×518 (sanity) | 1.000 | 1.000 | 1.000 | 1.000 |

Contrast with COCO, where the same grid caps a perfect model at 44.7 AP
(docs/MASKDINO_COCO.md §1): ScanNet's furniture-scale objects lose only ~0.04 AP50 of ceiling
to the 37×37 quantisation. The model sits at ~0.69 AP50 against a 0.956 ceiling —
**recognition binds, not resolution.**

**The model on both rulers** (N=490, full recipe, `--eval_full_res`, peak epoch):

| Run | grid mIoU / AP50 | full-res mIoU / AP50 | full AP75 / mAP |
|---|---|---|---|
| 9072738 — the bar, re-run | 0.670 / 0.690 | 0.654 / 0.673 | 0.526 / 0.465 |
| 9072749 — `--mask_upsample 2` | 0.650 / 0.685 | 0.648 / 0.680 | 0.505 / 0.460 |

Three readings:

1. **The bar reproduces** (0.670/0.690 vs the original 0.669/0.699) — a free seed-level
   reproducibility check, and the grid→full-res drop is only ~0.017 AP50.
2. **`--mask_upsample 2` stays neutral on the honest ruler too** (full AP50 0.680 vs 0.673,
   inside noise). The §7.4 "neutral" verdict was *not* an artefact of the grid-resolution
   scoring; the oracle explains why (the 37×37 ceiling is far above the model).
3. Job 9072761 (`--mask_upsample 4 --train_num_points 12544`) **OOM'd in backward at 19 min**
   (148² mask tensors, batch 8, 24 GB card; the ScanNet trainer has no gradient accumulation,
   unlike the COCO one). Given 1–2, running it is not worth the engineering — the oracle already
   bounds what it could buy (~0.004 AP50 ceiling over ×2).

**Consequence for the plan:** mask resolution work on ScanNet is de-prioritised; the honest
lever for boundary quality would be the token grid (docs/MASKDINO_COCO.md §1.3, VGGT at higher
input resolution), and even that is bounded by the 0.956→0.99 ceiling gap. Quote §7.7 whenever
resolution comes up.

### 7.8 Official 1201/312 split — first runs (jobs 9329716 / 9386666, 2026-08-01/02)

The full official protocol (todo 1c's last step): train = all 1201 official train scenes
(`scannet_official_gt_1201.tar.zst`), val = all 312 official val scenes
(`scannet_official_gt_val312.tar.zst`), staged into one tree (`DATA_TAR` takes a list;
`TRAIN_LIST`/`VAL_LIST` feed the split files — plumbing covered by
`tests/test_train_maskdino_sh_lists.sh`). Best recipe (`--bundles_per_scene 2 --color_jitter
0.2`), 12 epochs × ~2402 steps/epoch ≈ 28.8k steps ≈ the N=490 recipe budget (29.4k), warmup 2.
12 CPU × 14 GB (fp16 cache ~110 GB at 1201×2 bundles), `--tmp=90000`, 8h16 (SF, incl.
`--eval_full_res`) / 5h42 (MF) on one rtx_4090.

**This is a new ruler** — numbers live in docs/RESULTS.md §6, never next to the 0080–0089-val
tables. Headlines:

- **Single-frame** (job 9329716, `maskdino_sf_list1201_20260801_132724`): val **0.624 mIoU /
  0.662 AP50** (AP75 0.487, mAP 0.459); full-res ruler 0.611 / 0.651 (−0.011 AP50, same gap as
  §7.7 found — recognition still binds). Against the only prior official-val point, job
  8900194's 0.589 / 0.604 (§7.5, val = the 77-scene subset): +0.058 AP50 at ~3× train scenes.
  Train AP50 0.878 vs val 0.662 at epoch 12 — still data-limited.
- **Multi-frame** (job 9386666, `--multi_frame --feature_mode bundle`,
  `maskdino_sf_list1201_mf_20260802_133826`): per-frame 0.623 / 0.650 (peak ep 10); per-bundle
  **0.529 mIoU / 0.525 AP50** (AP75 0.312, mAP 0.311, peak ep 12 — per-frame and per-bundle
  peaks diverge again, vindicating `checkpoint_best_bundle.pth`, §8.2). The multi-view result
  transfers to the honest split (old ruler: 0.539 / 0.515).
- **First cross-view consistency numbers** (§6.6): `bundle_view_consistency` 0.679 → **0.717**
  and `bundle_id_switch` 0.607 → **0.498** over epochs 6→12, ~14.1 matched instances/bundle.
  Roughly: a matched instance is explained by its own query in ~72 % of its visible views, and
  in ~50 % of views some other query still fits better — the headroom the §7.4.1 ablations
  (cross-frame attention) act on.
- Its `checkpoint_best_bundle.pth` is the first checkpoint allowed to quote a reportable 3D
  number (§9.4 — no train/val leakage).

### 7.8.1 What cross-frame attention actually buys: identity (job 9503176, 2026-08-03)

`--no-cross_frame_attn` on the same official split, otherwise identical to 9386666. This is the
cut todo 2c was waiting for — the consistency metrics (§6.6) let it be read as a *mechanism*
claim rather than only as a score drop.

| | with cross-frame attn (9386666) | without (9503176) | Δ |
|---|---|---|---|
| per-frame mIoU / AP50 | 0.623 / 0.650 | 0.576 / 0.588 | −0.062 AP50 |
| per-bundle mIoU / AP50 | 0.529 / 0.525 | 0.471 / 0.389 | **−0.136 AP50** |
| `bundle_view_consistency` ↑ | **0.717** | 0.692 | −0.025 |
| `bundle_id_switch` ↓ | **0.498** | 0.682 | **+0.184** |
| `bundle_num_matched` | 14.1 | 14.0 | ±0 |

**The block's job is identity preservation, and the metrics separate that from recognition.**
The model finds the same number of instances either way (14.0 vs 14.1 matched per bundle) and
its own query still covers most views at IoU ≥ 0.5 (0.692 vs 0.717, a small drop) — but
**`id_switch` jumps from 0.498 to 0.682**: without the block, in 68 % of views some *other*
query fits the object better than the one that owns it. Identity degrades far more than
coverage does, which is precisely what a cross-view communication mechanism is supposed to
prevent, and the −0.136 bundle AP50 follows from it (the multi-view protocol scores one query
against the whole volume, so a switched view is lost mask).

The two metrics differ in strictness by construction: `view_consistency` asks "does my query
explain this view at all (IoU ≥ 0.5)?", `id_switch` asks "is some other query *better*?" — a
view can pass the first and fail the second, which is why the second is the sensitive one and
the one to quote for this mechanism.

This reproduces the N=490 finding (−0.183 bundle AP50, §7.4.1) on the honest split at
−0.136, and closes docs/todo.md 2c.


---

# Archive — 3D ruler, run-by-run (MASKDINO.md §9.5, §9.6)

### 9.5 Results — full val-312 DIAGNOSTIC runs (2026-08-01, jobs 9327269 / 9327271)

Checkpoint: `maskdino_sf_n490_mf_b2jit_20260730_105117/checkpoint_best.pth` (9.4's caveats
apply: **train/val leakage → diagnostic only**, and it is the epoch-17 not the epoch-19
checkpoint). 312/312 scenes, 0 failures, ~45 min/run, ~7.6 s/scene.

| Run | AP / AP50 / AP25 (18-class) | 17-class diagnostic |
|---|---|---|
| defaults (radius 5 cm, no conf filter), job 9327269 | 0.013 / 0.041 / 0.223 | 0.014 / 0.044 / 0.236 |
| `--vote_radius 0.1 --depth_conf_percentile 25`, job 9327271 | **0.016 / 0.052 / 0.238** | 0.016 / 0.055 / 0.253 |

Context (published full-split numbers, all on adapted backbones): under **our** protocol
(unposed transfer, §9.9) FAST3DIS scores 0.038 / 0.096 / 0.316 and IGGT 0.028 / 0.112 / 0.287 —
the same order of magnitude as us. SegVGGT's 0.504 / 0.717 / 0.870 is far above, but under the
**posed-transfer** protocol, so it is not a like-for-like comparison. Per class, `toilet` leads
(AP50 0.28–0.33); `otherfurniture` is 0 by construction (§9.2).

**Reading (from the per-scene diagnostics in the json):**

1. **Geometry binds, not recognition.** AP25 (0.24) is ~5× AP50 (0.05): objects are found and
   coarsely localised, but the lifted masks miss the >0.5-IoU bar. That is what the registration
   numbers predict — median camera-center RMS after Sim(3) is **0.14 m** and ICP point RMS
   **~0.10 m**, the same order as the vote radius (5–10 cm) — VGGT's own depth/pose drift over a
   whole-scan S≈17 bundle, not a 2D mask-quality problem (the same model scores 0.667 per-frame
   AP50). The 2D→3D chain is the price of the "no GT geometry at inference" claim.
2. **Coverage is the second cap:** ~15 % of mesh vertices receive any vote; ~63 % of annotated
   vertices get assigned to some instance. Every unassigned GT instance is a hard FN.
3. Knobs move it a little, in the expected direction (bigger radius + conf filter: +0.011 AP50),
   so the defaults are not at an optimum — but knob-tuning is secondary to geometry quality.
4. **The leakage barely matters at this operating point** — the binding constraints are
   geometric, so the honest 1201-trained number (9.4) will likely land nearby; it is still the
   only quotable one. *(§9.6 shows this prediction was wrong in the useful direction: the
   leak-free checkpoint scores ~1.6× higher.)*

S-generalisation worked as designed: bundles of 3–55 frames (median 15) through a model trained
at S=8, no failures — `CrossFrameAttention` has no frame positional encoding, and the two-stage
top-k just unions over more frames.

### 9.6 The REPORTABLE number (2026-08-03, jobs 9503137 / 9503139)

Checkpoint: `maskdino_sf_list1201_mf_20260802_133826/checkpoint_best_bundle.pth` — trained on
the official 1201-scene train split, **val-312 never seen** (§9.4 satisfied), bundle-selected
(todo 2b). 312/312 scenes, 0 failures, ~46 min/run.

| Run | AP / AP50 / AP25 (18-class) | 17-class diagnostic |
|---|---|---|
| defaults (radius 5 cm, no conf filter), job 9503137 | 0.023 / 0.067 / 0.268 | 0.024 / 0.071 / 0.284 |
| `--vote_radius 0.1 --depth_conf_percentile 25`, job 9503139 | **0.029 / 0.083 / 0.305** | 0.030 / 0.088 / 0.323 |

**Quote the defaults row as the headline** — the second row's knobs were chosen on the leaked
diagnostic runs of §9.5, so it is mildly tuned on data that includes val scenes. Both rows are
otherwise honest.

Three readings:

1. **We land in FAST3DIS's ballpark on a frozen backbone.** AP25 0.305 vs its 0.316, AP50 0.083
   vs 0.096, AP 0.029 vs 0.038 — against a *LoRA-adapted* DA3, while we never touch VGGT. IGGT
   (0.028 / 0.112 / 0.287) sits in the same cluster. SegVGGT (0.504 / 0.717 / 0.870, also
   LoRA-adapted) is **far above but in the other protocol** (§9.9): its masks are transferred
   with ScanNet's GT poses and sensor depth, so its number carries no geometry error, while every
   number in this cluster is 2D mask quality *times* predicted-geometry quality. State the
   protocol difference plainly — and state just as plainly that it is a legitimate evaluation
   choice on their part, not a trick, since their model is as unposed as ours.
2. **The leak-free checkpoint BEATS the leaked one** — 0.083 vs 0.052 AP50 at identical knobs,
   ~1.6×. §9.5's reading 4 predicted "roughly nearby"; the truth is that 1201 official train
   scenes outweigh *having seen the val scenes*. This is the 3D ruler independently reproducing
   §7.2's 2D conclusion: **the track is data-limited, not architecture-limited.** It also means
   every §9.5 number was a pessimistic proxy.
3. **The lifting step is now the binding constraint, not the decoder.** Geometry diagnostics are
   unchanged from §9.5 (median camera-center RMS 0.14 m, ~16 % of vertices voted, ~65 % of
   annotated vertices assigned) and AP25 is still ~4× AP50. Two *lifting* knobs alone bought
   +0.016 AP50 — more than most decoder ablations in §7.2.1 are worth. The next lever on this
   ruler is registration quality / coverage / voting, not query design.

Reproduce: `sbatch --export=ALL,CHECKPOINT=<mf_run_dir>/checkpoint_best_bundle.pth
slurm/eval_3d_maskdino.sh`. Output files now name any non-default result-affecting knob
(`eval3d_<stem>__vote_radius0.1_depth_conf_percentile25.0.json`) — before 2026-08-03 both knob
settings wrote to the same path and the second silently overwrote the first, which is how job
9503137's JSON was lost (its numbers survive only in `slurm/logs/eval3d_9503137.log`; the
defaults run was repeated as job 9532181 and reproduced it exactly — 0.0228 / 0.0672 / 0.2680 —
so the pipeline is deterministic and the headline now has a JSON behind it).
Guarded by `tests/test_maskdino_eval3d.py::test_out_path_names_the_knobs`.

