# TODO

See `docs/MILESTONES.md` for the consolidated summary; `docs/old/` for full per-milestone
detail. This list tracks only what is still open.

## Open — next experiments (GPU)

- [X] **N=100+ scaling point** (DONE 2026-06-22). Arm-A instance-GT curve now runs N ∈ {10, 25,
      50, 100, 200}, wide val (scene0080–0089). **Result: plateau** — val mIoU flattens at
      ~0.21–0.23 past N=50, honest val[grid] AP50 sits ~0.10 (below its N=50 peak of 0.125). More
      scenes is no longer the lever; train−val gap shrank to ~0.05 (no longer overfitting →
      capacity/resolution ceiling). Runs: `d4rt_m2_scale100_inst`, `d4rt_full_inst`. See MILESTONES.
- [ ] **Confirm the learned-query win (arm C) at large N** (IN PROGRESS — running at N=200).
      C was best at N=50 (val mIoU 0.259 vs 0.212 baseline) but overfit hard (gap 0.49). With the
      baseline plateaued, the question is whether the best architecture variant keeps climbing.
      `--query_mode learned --num_learned_queries 64`. Metric = val mIoU + honest val[grid] AP50.
- [ ] **Fix Phase-2 `--train_grid_queries` loss balance** (arm B collapsed: train mIoU ~0.05).
      Normalize the no-object term by query count and/or keep GT-centroid queries always matched;
      rerun. Metric = unprompted val AP50.
- [ ] **Fix Phase-3 hybrid (arm D) NaN** — guard the matcher cost (`nan_to_num` + finite assert
      around `linear_sum_assignment`, `train/loss.py:253`), tighter grad-clip / lower LR on the
      learned-embedding params; rerun (it was the most promising arm before crashing).
- [ ] **Train the MaskDINO pixel decoder** (`--mask_upsample 2/4`) — code done, never trained.
      If dense Dice+BCE OOMs, adopt Mask2Former point-sampled mask loss (~3k pts/mask). May fix
      the window/door/picture confusion if it's a resolution (not coverage) problem.

## Open — data-gated ablations (Phase 6; meaningful now with 200 scenes)

- [ ] No-object-weight sweep (0.05 / 0.1 / 0.4) — tests whether 0.1 drives the under-confidence.
- [ ] Augmentation ablation: `bundles_per_scene` 1 vs 4, `query_jitter` on/off, `color_jitter` on/off.
- [ ] Grid-density vs unprompted recall: `--grid_size` 4/6/8.
- [ ] Score-threshold sweep at viz time (no retrain): re-render at `--score_threshold 0.3`.
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
- [X] M3 Phase 5 CODE: `models/mask_upsampler.py` + `--mask_upsample` (not yet trained).
- [X] Per-instance SAM3 dataset: 200 scenes (scene0000–0199) in one `scannet_instance_dataset_full.tar.zst`;
      `stage_dataset.sh` stages it to node-local SSD (`--tmp=16000`).
</content>
