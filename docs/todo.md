# TODO

Open work only. Everything closed up to 2026-07-28 is in
`docs/old/todo_archive_20260728.md`; the reasoning behind each closed item is in
`docs/old/MILESTONES.md` (D4RT arms) and `docs/MASKDINO.md` §7 (MaskDINO).

## Now — multi-frame MaskDINO

The single-frame question is answered and won (0.669 / 0.699 @ N=490, and **0.694 / 0.729** with
`--bundles_per_scene 2`, vs the arm-C per-frame bar 0.451 / 0.294). Multi-frame is the actual
research goal. Follow `docs/MASKDINO.md` §8 in cost order:

- [x] **`--feature_mode bundle`** — DONE, **negative result** (job 8895540, N=490):
      0.622 / 0.651 vs the 0.669 / 0.699 bar, i.e. **−0.048 AP50**. Multi-view-aware frozen
      tokens make per-frame segmentation worse, not better. It is the right *control* for the
      multi-frame decoder, not a win.
- [x] **Shared queries across frames** — IMPLEMENTED + RUN 2026-07-28 (`--multi_frame`,
      `models/maskdino/multiframe.py`, `tests/test_maskdino_multiframe.py`, docs/MASKDINO.md
      §8.2). Job 8900100 at N=490: per-frame 0.621 / 0.630 = **−0.021 against its own control**
      (bundle features), i.e. neutral inside the noise band, and on the arms' multi-view ruler
      **0.535 mIoU / 0.494 AP50 vs arm C's 0.367 / 0.199** (+46 %, 2.5×). Two follow-ups are
      running: 8950613 (`--multi_frame` on per-frame features, decouples the two changes) and
      8950617 (`--no-cross_frame_attn`, how much the block itself is worth).
- [ ] **3D anchors instead of 2D boxes** — **designed, not implemented** (docs/MASKDINO.md §8.3
      now carries the full sketch: patch-token geometry cache, 3D two-stage anchors, soft
      nearest-patch reference points so no camera math is needed, losses untouched).
      Two things frame the implementation, both deliberate:
      (a) it only makes sense **on top of** the shared-query path — a view-independent anchor is
      meaningless while queries are per-frame. That path now exists and works (job 8900100), so
      this is unblocked; build it on `--multi_frame`, and decide first whether per-frame or
      bundle features are the base (jobs 8950613 / 8950617);
      (b) FAST3DIS already publishes this mechanism, so it is an ablation ("3D anchors vs 2D DAB
      boxes, same backbone / data / protocol"), not a contribution — budget it as such.

## Cheap follow-ups (one flag each) — both answered 2026-07-28

- [x] `--mask_upsample 2` (74×74 masks) — **neutral** (job 8895551: 0.662 / 0.677, −0.022 AP50,
      inside the ±0.04 noise band). Same verdict the D4RT arms reached on a different head.
      Masks stay on the 37×37 patch grid.
- [x] `--bundles_per_scene 2 --color_jitter 0.2` — **the biggest win available** (job 8895565:
      **0.694 / 0.729**, +0.030 AP50, a new best; train mIoU 0.947 → 0.816, so it also reduces
      memorisation). Caveat recorded in docs/MASKDINO.md §7.4: an epoch-clamp bug in
      `slurm/train_maskdino.sh` (now fixed) gave it a 2× step budget, though it peaked at
      18.6 k steps vs the bar's 15.2 k. **`--bundles_per_scene 4` is running (job 8950610).**
- [ ] If 4 bundles also helps, make `--bundles_per_scene` part of the default recipe and re-run
      the headline scale points, so the scaling curve is measured with the better recipe.

## Positioning

- [x] **Read the direct competitors** — DONE 2026-07-28, recorded in `docs/RELATED_WORK.md`.
      SegVGGT full text: VGGT **LoRA-finetuned** (not frozen), 400 plain learned queries in all
      24 aggregator layers, no anchors/DN/two-stage, eval = per-view masks unprojected to the
      benchmark **point cloud** (mAP 50.4 / mAP50 71.7 on ScanNetv2), still **no query-init
      ablation**. FAST3DIS full text: **already owns 3D-anchored queries** (learned 3D anchor
      generator + project-and-sample cross-attention) on a LoRA-adapted Depth-Anything-V3.
      PanSt3R: single forward pass on MUSt3R, *not* the post-hoc-fusion strawman we filed it as.
      EPS3D: open-vocabulary + distillation, different supervision regime.
- [x] **Reframe the contribution around the MaskDINO result** — DONE 2026-07-28 (new "Headline"
      update + "What is already claimed" table in RELATED_WORK.md). Net: shared queries and 3D
      anchors are both published mechanisms; what is unpublished is the **controlled study** —
      one frozen backbone, one dataset, one protocol, decoder ingredients varied one at a time.
- [ ] Consequence to act on: **§8.3 (3D anchors) is an ablation, not a contribution** — frame and
      budget it as "3D anchors vs 2D DAB boxes inside the same decoder", which is also the honest
      re-test of the arm-E negative result.

## Protocol debt — DECIDED 2026-07-28 (see docs/RESULTS.md §1.1–1.3)

- [x] **Scene splits.** Keep val = scenes 0080–0089 as the ruler (continuity of the whole scaling
      curve), plus ONE comparability read-out on the official ScanNet v2 val list intersected
      with our tar (77 scenes; `data/splits/scannetv2_val.txt`, `VAL_SPLIT=official` in
      `slurm/train_maskdino.sh`). It needs its own run — 74 of the 77 are inside the normal train
      range — job 8900194 (413 train / 77 val). **Result: 0.589 mIoU / 0.604 AP50** (peak at
      epoch 27; the job hit its 12 h wall clock at 39/60, past the peak, so the number stands).
      Our convention split is ~0.10 AP50 "easier" — quote both (docs/MASKDINO.md §7.5).
- [x] **Fixed cached view sets.** Accepted and documented as a deliberate tradeoff; measured once
      by the `--bundles_per_scene 2 --color_jitter 0.2` run (job 8895565). Randomising
      `num_frames` per bundle was considered and dropped (bundles in a multi-frame batch must
      share S).

## Data-gated ablations (cheap, low priority)

- [~] Augmentation ablation: 1 vs 2 answered (+0.030 AP50); **4 is running** (job 8950610,
      `EPOCHS=15` to hold the ~29 k step budget, 12 CPUs × 12 GB for the 88 GB feature cache).
      `color_jitter` on/off alone has still never been isolated from the extra bundles.
- [~] `--no-cross_frame_attn` at N=490 (job 8950617): how much of the multi-frame result is the
      cross-frame block rather than shared query init + bundle matching.
- [ ] Longer-term: partial backbone unfreezing, once the train−val gap vs N says data supports it.
