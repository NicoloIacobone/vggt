# TODO

See `docs/MILESTONES.md` for the consolidated summary; `docs/old/` for full per-milestone
detail. This list tracks only what is still open.

## Open — next experiments (GPU)

- [X] **N=100+ scaling point** (DONE 2026-06-22). Arm-A instance-GT curve now runs N ∈ {10, 25,
      50, 100, 200}, wide val (scene0080–0089). **Result: plateau** — val mIoU flattens at
      ~0.21–0.23 past N=50, honest val[grid] AP50 sits ~0.10 (below its N=50 peak of 0.125). More
      scenes is no longer the lever; train−val gap shrank to ~0.05 (no longer overfitting →
      capacity/resolution ceiling). Runs: `d4rt_m2_scale100_inst`, `d4rt_full_inst`. See MILESTONES.
- [X] **Confirm the learned-query win (arm C) at large N** (DONE 2026-06-22 — decisive win).
      At N=200: val mIoU **0.371** (best @ep600, gap 0.086) / 0.326 final, honest AP50 **0.228** —
      vs the plateaued point baseline (0.216 / 0.105): +0.15 mIoU, >2× AP50. Learned queries keep
      scaling (0.259→0.371 from N=50→200) and their N=50 overfitting resolved (gap 0.49→0.086).
      **The ceiling was the head, not the data.** Run: `d4rt_full_inst_learned_20260622_183203`.
      → Arm C learned is now the default base for all further experiments (below).
- [X] **Fix Phase-2 `--train_grid_queries` loss balance** (DONE 2026-07-07 — fix works, arm B
      closed as a loss to arm C). `--no_object_norm matched` fixed the collapse: N=50 rerun
      (job 5647527, `_gridq_fix`) reaches train[grid] mIoU 0.458 (was 0.055) and best val[grid]
      AP50 **0.161** ≥ the 0.125 bar. Scaled to N=190 (job 5658375,
      `d4rt_full_inst_gridq_fix_20260703_184456`): val[grid] mIoU 0.372 @ep1000 (matches arm C)
      but AP50 peaks at only 0.185 and is unstable (0.071 @ep1000) vs arm C's 0.228 →
      **arm C stays the base**. See MILESTONES.
- [X] **Fix Phase-3 hybrid (arm D) NaN** (DONE 2026-07-07 — NaN fixed, no win, arm D closed).
      N=50 rerun (job 5647528, `_hybrid_fix`, lr_scale 0.1, grad_clip 0.5) survived all 1000
      epochs with zero non-finite matcher warnings, but best val mIoU 0.247 / AP50 0.146 only
      ties arm C N=50 (0.259/0.146) and then overfits (val→0.177 while train[grid]→0.75).
      Per the decision rule (scale only on a win): **no N=190 run**. See MILESTONES.
- [X] **Train the MaskDINO pixel decoder** (`--mask_upsample 2`) — DONE 2026-06-30 (SLURM job
      5275027, run `d4rt_full_inst_learned_us2_20260630_161537`; arm-C base: learned, 64 queries,
      instance-level, N=190). **Result: a wash.** Best honest val[grid] AP50 **0.236** (@ep500,
      marginally above the us=1 baseline's 0.228), best val mIoU **0.355** (@ep250, below the
      baseline's 0.371); final @ep1000: 0.200 / 0.311; gap at best 0.098 (vs 0.086). Doubling the
      mask resolution to 74×74 does not move the numbers → **resolution is NOT the current
      bottleneck**; `--mask_upsample 4` is deprioritized (per the decision rule: only on a win).
      The window/door/picture confusion is therefore likely semantic, not resolution-limited —
      points at the score-threshold / no-object levers and arm-D instead.

## Open — data-gated ablations (Phase 6; meaningful now with 200 scenes)

- [ ] No-object-weight sweep (0.05 / 0.1 / 0.4) — tests whether 0.1 drives the under-confidence.
- [ ] Augmentation ablation: `bundles_per_scene` 1 vs 4, `query_jitter` on/off, `color_jitter` on/off.
- [X] Grid-density vs unprompted recall (DONE 2026-07-07, job 6111639 — **negative: density
      is not the lever, learned-vs-grid gap confirmed architectural**). Eval-only sweep
      (`scripts/eval_grid_ablation.py` + `slurm/eval_grid_ablation.sh`, no retraining) of
      `--grid_size` 2/4/6/8/10/12 on the stored val bundles; grid-6 rows reproduce the
      training-time `val_grid_AP50` of the selected epochs exactly (0.134 / 0.185).
      Val AP50 by density —
      arm A `d4rt_full_inst` best_ap50: 0.023 / 0.124 / 0.134 / **0.138** / 0.116 / 0.109
      (flat 6→8, drops beyond); arm B `gridq_fix` best_ap50: 0.018 / 0.063 / **0.185** /
      0.100 / 0.134 / 0.067 (sharp peak exactly at its 6×6 *training* density). Kept
      foreground predictions explode with density (arm A: 58 @6 → 236 @12 vs 14.4 GT/scene)
      → denser grids die by duplicate FPs (no NMS), as predicted. Unprompted mIoU rises
      monotonically with density (0.297→0.336) — the known "unprompted mIoU is optimistic"
      artifact; judge on AP50. Best grid number over all densities/checkpoints (0.185) stays
      well below arm C's 0.228. Results: `<run_dir>/grid_ablation_<ckpt>.json`.
- [X] Score-threshold sweep at viz time (DONE 2026-07-03 — **negative, keep 0.5**). On the
      arm-C best checkpoint (`d4rt_full_inst_learned_20260622_183203`), thr 0.3 surfaces 76
      extra instances over the 10 val scenes of which only 2 have IoU≥0.5 with any GT (1 with
      the right class) — the under-confidence finding was a *point-prompt* phenomenon and does
      not transfer to learned queries. The model already over-predicts at 0.5 (338 kept vs 144
      GT) → the lever is duplicate suppression / no-object weight, not the threshold.
      Renders: `<run>/visualizations_thr03/` (val scenes, vs `visualizations/` at 0.5).
- [ ] Longer-term: partial backbone unfreezing once the train−val gap vs N says data supports it.

## Done (high level — detail in docs/MILESTONES.md)

- [X] M1: prototype, single-scene overfit, 4-scene training (train mIoU 0.967), unseen-scene eval.
- [X] M2: no-object loss, unprompted grid eval, regularization, best-checkpoint + early stop.
- [X] M3 Phase 0: metrics.jsonl, uint8/light checkpoints, noise-robust early stop,
      checkpoint_best_ap50, --schedule_epochs, viz polish, fixed SLURM protocol.
- [X] M3 Phase 4: per-instance loader (`instance_level`) + tests; instance-GT scaling curve
      (monotonic, wide val) N∈{10,25,50}.
- [X] M3 Phases 2/3 CODE + first runs: `--train_grid_queries`, `--query_mode {point,learned,hybrid}`.
      Arms A/B/C/D run at N=50 (C learned = best; B/D need fixes above).
- [X] M3 Phase 5: `models/mask_upsampler.py` + `--mask_upsample`; us=2 trained at N=190 →
      neutral vs us=1 baseline (see MILESTONES) — resolution is not the bottleneck.
- [X] Standardized 2D/3D visualization (2026-07-02): `train/postprocess.py::select_instances`
      is the single instance-selection rule shared by `visualize_masks.py` and the Gradio 3D
      viewer (honest, no-GT, winner-takes-all; + oracle GT-matched panel as diagnostic).
- [X] Per-instance SAM3 dataset: 200 scenes (scene0000–0199) in one `scannet_instance_dataset_full.tar.zst`;
      `stage_dataset.sh` stages it to node-local SSD (`--tmp=16000`).
</content>
