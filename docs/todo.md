# TODO

See `docs/MILESTONES.md` for the consolidated summary; `docs/old/` for full per-milestone
detail. This list tracks only what is still open.

## Open — next experiments (GPU)

- [ ] **N=100+ scaling point** (unblocked: 200 scenes in one full tar, staging handles it).
      Re-run the instance-GT curve at N ∈ {10, 25, 50, 100, 200}, wide val (scene0080–0089),
      arm A point prompts. Target: does val mIoU / honest val[grid] AP50 keep climbing?
- [ ] **Fix Phase-2 `--train_grid_queries` loss balance** (arm B collapsed: train mIoU ~0.05).
      Normalize the no-object term by query count and/or keep GT-centroid queries always matched;
      rerun. Metric = unprompted val AP50.
- [ ] **Fix Phase-3 hybrid (arm D) NaN** — guard the matcher cost (`nan_to_num` + finite assert
      around `linear_sum_assignment`, `train/loss.py:253`), tighter grad-clip / lower LR on the
      learned-embedding params; rerun (it was the most promising arm before crashing).
- [ ] **Confirm the learned-query win (arm C)** holds as N grows — track the point-vs-learned
      crossover toward N=100+ (C overfits hard at N=50: train 0.749 vs val 0.259).
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
