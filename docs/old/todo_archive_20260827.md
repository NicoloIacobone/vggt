# todo — ARCHIVE, closed items (moved 2026-08-27)

Closed work lifted out of `docs/todo.md` so that file holds open work only. Nothing here is
actionable; several of these items were measured on rulers this project no longer reports on
(`docs/old/RESULTS_HISTORY.md`).

---

## 3. Resolution stream — CLOSED 2026-07-30 (docs/MASKDINO.md §7.7)

The mask grid is decoupled from the token grid ("VGGT is not an FPN" is answered,
docs/MASKDINO_COCO.md §1.2), the 37×37 GT-only ceiling on ScanNet is 0.956 AP50 vs the model's
~0.69, and `--mask_upsample 2` is neutral on the full-resolution ruler too. **Recognition
binds, not resolution — nothing left to do here on ScanNet.** The only surviving idea (a
700/1036 px token-grid arm) is bounded by the 0.956→0.99 ceiling gap and stays parked in
"Longer-term".


## Recently closed (2026-07-29/30, 2026-08-01) — details in docs/MASKDINO.md §7.4.1, §7.7, §8.2

- [x] `--bundles_per_scene 4` (job 8950610) — **saturates** (0.699 / 0.722 vs b2's
      0.694 / 0.729, inside noise). Views-per-scene lever exhausted at 2; do NOT fold 4 into
      the default recipe.
- [x] `--no-cross_frame_attn` at N=490 (job 8950617) — bundle AP50 0.494 → 0.311.
      **Cross-frame attention is the main carrier of the multi-view result** — the only
      individually-decisive component found anywhere in this track.
- [x] `--multi_frame` on per-frame features (job 8950613) — bundle AP50 0.494 → 0.347.
      **Bundle features are required for multi-view consistency**, despite costing −0.048
      per-frame as a standalone change (§8.1). Consistency has a measured price.
- [x] 2026-07-30: resolution question closed (§7.7, oracle + two full-res runs); multi-view
      best moved to **0.539 / 0.515** (job 9071415, §8.2); COCO r50/dinov2 arms final
      (MASKDINO_COCO.md §6).
- [x] 2026-08-01: COCO `vggt` arm COMPLETE (job 9262006). **vggt final 37.7 AP vs dinov2 38.8**
      (−1.1 AP at identical geometry); best checkpoint vggt 39.7 @75k vs dinov2 41.3 @85k
      (−1.6 AP). Both trail early (14.1 vs 23.4 at overfit-gate), converge mid-training, diverge
      late. Verdict: 3D pretraining costs ~1–1.6 AP on 2D semantics (docs/MASKDINO_COCO.md §6,
      reading 3).
