---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section { font-size: 25px; }
  section h1 { font-size: 38px; }
  .small { font-size: 18px; }
  .cols { display: flex; gap: 24px; align-items: center; }
  .cols > div { flex: 1; }
  section.compact { font-size: 20px; }
  section.compact h1 { font-size: 34px; }
  section.compact .cols { align-items: flex-start; }
  blockquote { font-size: 20px; color: #555; }
  img { background: transparent; }
  table { font-size: 20px; }
---

<!-- _class: lead -->
<!-- _paginate: false -->

# Multi-View Consistent 3D Instance Segmentation on a Frozen VGGT Backbone

## Progress update: per-instance supervision + scaling & query-mode ablations

**Nico Iacobone — Research update for supervision meeting**
June 15, 2026

<!--
Speaker notes:
This follows the Jun 11 meeting. Two of the supervisor's asks are now closed: (1) the next SAM3 run is per-INSTANCE, and the data has landed; (2) the point-prompt-vs-learned-query ablation has been run. Frame the deck as "what the per-instance data + scaling experiments told us."
-->

---

<!-- _class: compact -->

# 1 · What's new since June 11

<div class="cols">
<div>

**The two blockers from last meeting are resolved**

- **Label format decided → per-INSTANCE** (supervisor call). The SAM3 re-run is **done**: 97 scenes, **2056 instances**, one binary mask per instance, cross-frame identity from SAM3 **video tracking**.
- **Data logistics:** the whole dataset now ships as **one zstd tar** (~1.3 GB); each job copies it to node-local SSD and unpacks once — no more reading thousands of small PNGs off the network filesystem.

</div>
<div>

**Code (all default-off, regression-tested)**

- Loader gained an `instance_level` mode: one ID per *(class, instance)* from `masks_instance/<class>_<k>/`; same-class objects are now **distinct** GT instances.
- Matcher / loss / eval needed **no change** — they were already Hungarian-over-instances.
- Query-mode ablation infra: `--query_mode {point, learned, hybrid}`, `--train_grid_queries`.

</div>
</div>

<!--
Speaker notes:
The loader switch was isolated by design — we predicted in the last meeting that only the dataset class would change, and that held. Stuff classes (wall/floor) remain single instances. The per-class masks are kept alongside, so every earlier baseline is still reproducible.
-->

---

<!-- _class: compact -->

# 2 · Per-instance ground truth: same-class objects now separated

<div class="cols">
<div>

- Last meeting's ceiling: per-class binary masks → at most one instance per class, so "instance" and "category" coloring were **visually indistinguishable**.
- Now each tracked object is its own instance. The GT panel shows e.g. **chair #0 … chair #9** and **table #0 … #6** in distinct colors — exactly the demonstration that was impossible before.
- **Honest caveat for the metrics that follow:** instance GT is *harder* (more, smaller objects), so mIoU/AP are expected to **drop vs the per-class numbers** — this is the cost of a meaningful task, not a regression.

</div>
<div>

![w:470](../visualizations/meeting_jun_15/inst_sep_scene0082_f00.png)
<span class="small">RGB | per-instance GT | prediction — held-out `scene0082_00`. The GT panel separates many same-class chairs/tables; the prediction is still rough on unseen scenes (≈0.2 val mIoU).</span>

</div>
</div>

<!--
Speaker notes:
This directly answers comment 1 from last time ("you're coloring by category"): the GT now proves same-class separation, and one query still emits one mask over all frames, so identity is architectural. The prediction shown is a held-out val scene, deliberately not cherry-picked — it shows where generalization stands.
-->

---

# 3 · Scaling on instance GT (held-out val, 10 scenes 0080–0089)

<div class="cols">
<div>

![w:520](../visualizations/meeting_jun_15/scaling_instance_widefal.png)

</div>
<div>

| N  | val mIoU | val[grid] AP50 |
|----|----------|----------------|
| 10 | 0.152    | 0.089          |
| 25 | 0.174    | 0.111          |
| 50 | 0.212    | 0.125          |

- **Both curves are now monotonic in N** — more scenes help prompted mIoU *and* the honest unprompted AP50.
- Widening val from 3 → 10 scenes removed the noise that made the earlier curve look non-monotonic.
- Absolute mIoU is lower than the per-class baseline — expected for the harder instance task (more, smaller objects).

</div>
</div>

<!--
Speaker notes:
val[grid] AP50 is the honest detection number (uniform grid, no GT prompt). The 3-scene val set previously swung ±0.05–0.08 per eval and showed a spurious N=25 dip; with 10 val scenes the trend is clean. This is the first real instance-level scaling evidence.
-->

---

<!-- _class: compact -->

# 4 · Point prompts vs learned object queries (N=50, instance GT)

<div class="small">

| Arm | Train queries | val mIoU | val[grid] AP50 | train mIoU | Outcome |
|---|---|---|---|---|---|
| **A** point (current) | GT centroids + bg | 0.212 | 0.125 | 0.338 | baseline |
| **B** + grid queries | + random-offset grid | 0.047 | 0.146 | 0.055 | mask learning **collapsed** |
| **C** learned | M learned embeddings | **0.259** | **0.146** | 0.749 | **best val**; overfits (gap 0.49) |
| **D** hybrid | learned + centroids | ~0.27\* | — | 0.54\* | **crashed** (NaN), best before dying |

</div>

- **Learned object queries (C) won** — *against* the prior that DETR-style queries are too data-hungry at ≤50 scenes. Caveat: large train–val gap → tracks as data grows; the crossover toward N=100+ is the thing to watch.
- **`--train_grid_queries` (B)** raised unprompted AP50 (the intended duplicate-suppression effect) but **drowned mask supervision** — a loss-balance bug, fixable (normalize the no-object term by query count).
- **Hybrid (D)** was the strongest before a numerical instability (NaN into the Hungarian cost) killed it ~ep555 — needs a cost guard + tighter grad-clip.

<span class="small">\* D's last eval before the crash.</span>

<!--
Speaker notes:
This is the ablation the supervisor asked for. The headline is positive and a little surprising: learned queries beat point prompts on instance GT at N=50. B and D are not clean negative results — they are bugs (loss balance, numerical stability) that I can fix and rerun cheaply (~50 min/run). I deliberately did not paper over them.
-->

---

<!-- _class: compact -->

# 5 · Where it stands & next steps

<div class="cols">
<div>

**Closed since Jun 11**
- Per-instance SAM3 data (97 scenes / 2056 instances) + single-tar pipeline
- Per-instance loader + tests
- Instance-GT scaling curve (monotonic, wide val)
- Query-mode ablation (learned > point at N=50)

**Open / known issues**
- Masks still at 37×37 (pixel decoder coded, not yet trained)
- `--train_grid_queries` loss balance; hybrid NaN
- Generalization modest (val mIoU ~0.2) — data-bound

</div>
<div>

**Next steps**
1. Fix + rerun arms B and D; confirm learned-query win and the AP50 effect
2. Push C toward **N = 100+** to test the crossover (needs more preprocessed scenes)
3. Train the **MaskDINO-style pixel decoder** for sharper masks
4. No-object-weight / augmentation / grid-density ablations now that val signal > noise

</div>
</div>

<!--
Speaker notes:
Ask: agreement on prioritizing (a) the learned-query direction (it's winning) and (b) more scenes for the N=100+ point. Everything is regression-tested and the per-class baselines remain reproducible. The fixes for B/D are queued and cheap.
-->
