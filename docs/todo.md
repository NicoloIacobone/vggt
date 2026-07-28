# TODO

Open work only. Everything closed up to 2026-07-28 is in
`docs/old/todo_archive_20260728.md`; the reasoning behind each closed item is in
`docs/old/MILESTONES.md` (D4RT arms) and `docs/MASKDINO.md` §7 (MaskDINO).

## Now — multi-frame MaskDINO

The single-frame question is answered and won (0.669 mIoU / 0.699 AP50 @ N=490 vs the arm-C
per-frame bar 0.451 / 0.294). Multi-frame is the actual research goal. Follow `docs/MASKDINO.md`
§8 in cost order:

- [ ] **`--feature_mode bundle`** — already implemented, never run at scale. VGGT's global
      attention makes the per-frame tokens multi-view aware while the decoder stays per-frame.
      Free multi-view signal, no architectural change. Run it at N=490 against the 0.699 bar.
- [ ] **Shared queries across frames** — run one query set against every frame's memory in a
      bundle, with a cross-frame self-attention block between decoder layers, so a query keeps
      one instance id across all views. This is where the multi-view (per-bundle) metric becomes
      meaningful again and the comparison to arms A–E returns to its original ruler.
- [ ] **3D anchors instead of 2D boxes** — replace the DAB 4-d box with a 3D anchor from VGGT's
      point head. Arm E showed 3D anchors alone don't beat 2D queries, but arm E had no box
      refinement, no DN and no deep supervision — the ingredients that make anchors work in DINO.
      `legacy/d4rt/models/anchor_queries.py` has the FPS + kNN construction to reuse.

## Cheap follow-ups (one flag each)

- [ ] `--mask_upsample 2` (74×74 masks). Masks are currently supervised on the 37×37 patch grid.
      Note the D4RT arms found this neutral, but that was a different head.
- [ ] `--bundles_per_scene 2 --color_jitter 0.2` — more frame draws without new scenes. The
      scaling curve had not flattened at 490 scenes, which is all the official-GT tar holds, so
      more views per scene is the cheapest remaining data lever. Costs cache memory; watch the
      NUMA footprint finding in `docs/old/MILESTONES.md`.

## Positioning

- [ ] **Read the direct competitors** (`docs/RELATED_WORK.md`): SegVGGT line-by-line first
      (especially its eval protocol — 3D point-cloud masks vs our per-view 2D patch-grid masks
      are NOT comparable numbers; state the difference explicitly), then EPS3D, FAST3DIS,
      PanSt3R. Check whether any already claims a query-init ablation or 3D-anchored queries.
      Record findings in RELATED_WORK.md.
- [ ] **Reframe the contribution around the MaskDINO result.** RELATED_WORK.md still positions
      the project around the query-strategy study (arms A–E). That study is now the *negative*
      half of the story; the positive half is "a faithful DINO-family decoder on a frozen 3D
      backbone, and the finding that the earlier plateau was architectural, not data".

## Protocol debt (decide before the next round of headline numbers)

- [ ] **Scene splits vs the official ScanNet split.** Val is scenes 0080–0089 by project
      convention, not the official ScanNet val split. Decide: intersect the official split with
      our 500 scenes (comparable to published work, breaks the scaling-curve continuity) or keep
      the convention (self-consistent, not directly comparable).
- [ ] **Fixed cached view sets.** Frames are sampled once per scene up front and reused for the
      whole run, with `num_frames` fixed per run. This is the tradeoff that makes head-only
      training take minutes, but it risks the head memorising the cached view combinations
      rather than learning view-set robustness. Options: raise `--bundles_per_scene`, randomise
      `num_frames` per bundle, or accept it and say so explicitly in the writeup.

## Data-gated ablations (never run; cheap, low priority)

- [ ] Augmentation ablation: `bundles_per_scene` 1 vs 4, `color_jitter` on/off.
- [ ] Longer-term: partial backbone unfreezing, once the train−val gap vs N says data supports it.
