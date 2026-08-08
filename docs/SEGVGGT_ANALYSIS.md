# SegVGGT — the magnitude gap and the conceptual difference (2026-08-07)

> Task 2 of the comparability programme. This document **assembles and attributes** evidence that
> already lives in `docs/RELATED_WORK.md` and `docs/MASKDINO.md` §9.9–§9.10 — it does not restate
> either. One fact, one home; follow the links.

## (a) Does training code exist?

No. `docs/RELATED_WORK.md`'s SegVGGT row and `docs/TRAINING_COMPARABILITY.md` §0.3 already establish
this from a direct grep of `/cluster/scratch/niacobone/SegVGGT`: zero hits for
`optimizer|backward|AdamW|criterion|DataLoader|Dataset(|hungarian|loss` outside comments. The release
is model + LoRA + eval only — `eval/`, `configs/eval/`, `scripts/eval.sh`, `segvggt/` — and
`scripts/eval.sh` → `eval/eval_instance_seg.py` is the entire runnable surface. FADA (their
attention-dispersion mechanism) is only half-present: the aggregator emits `attn_frame_mean`
(`segvggt/models/segvggt.py:150`), the tensor its loss would consume, but the loss itself is absent.

Every claim about SegVGGT's training setting below is therefore **paper-only and unverifiable from
the release** — which is itself a finding worth reporting, not a gap in this analysis.

## (b) Where the ~10.7× ScanNet gap comes from

**Headline: the protocol accounts for ×2.3 of ~10.7×; the residual ~4.6× is real and is bought with
a LoRA-adapted backbone, 4–6× the views and 7× the mask resolution — not with more or different
training data.**

| factor | direction | size | status |
|---|---|---|---|
| **posed vs unposed 2D→3D bridge** — their evaluator projects the GT cloud into each view with ScanNet GT poses, GT intrinsics and **sensor depth**; ours unprojects with VGGT's *predicted* depth+cameras then Sim(3)+ICP | theirs easier | **×2.3 AP50** | **MEASURED** (`--transfer_mode gt_projection`, oracle purity 0.9999 — `docs/MASKDINO.md` §9.10) |
| **LoRA-adapted backbone** (r=32 on frame *and* global attention, all 24 layers) vs our strictly frozen VGGT | theirs stronger | not isolated | paper §4.1 + `configs/eval/segvggt_scannetv2.yaml` (`docs/RELATED_WORK.md`) |
| **views/scene at eval**: ~75–100 (`--downsample_factor 20` over a full `.sens`) vs our ~17 | theirs stronger | bounded — our own oracle ceiling at 17 views is 0.948 AP50, so view count is **not** what binds us | measured (`docs/MASKDINO.md` §9.10 (iii)) |
| **mask resolution** 259×196 (`return_feature_maps_down_ratio: 2`) vs our 37×37 grid | theirs stronger | ScanNet ceiling at 37² is 0.956 → small | measured (`docs/MASKDINO.md` §7.7) |
| **kept queries** 600 vs our 100 | theirs stronger | **neutral** — `--eval_topk` 600 measured at 0.138→0.140 | measured (`docs/MASKDINO.md` §9.8.1) |
| **`otherfurniture`**, which their head predicts and our 19-class head cannot | theirs stronger | ~1/18 of the class-aware mean | structural (`docs/RELATED_WORK.md`) |
| **training set** — identical (official 1201 ScanNetv2 split) | — | **zero** | `docs/TRAINING_COMPARABILITY.md` §0.1 |

One trap to avoid repeating: SegVGGT's paper mixes two protocols inside itself. Table 1 (full val:
50.4/71.7/87.0) and Table 2 (**10 randomly sampled val scenes**, ScanNet++ zero-shot 13.3/33.9/56.4)
are not the same measurement and must never be quoted side by side (`docs/RELATED_WORK.md`).

## (c) The conceptual difference

Both this project and SegVGGT share queries across views — that is table stakes, not a contribution
either side should claim (`docs/RELATED_WORK.md`, "What is already claimed"). The real split:

| | SegVGGT | this project |
|---|---|---|
| where queries live | **inside** the backbone — 400 learned queries cross-attend after global attention in **every one of the 24 aggregator layers**; ablation last-12 30.5 → all-24 31.9 mAP | **outside** — one frozen hook at `aggregated_tokens_list[-1]`, features cached once per scene |
| backbone | LoRA-adapted, frame + global attention | **strictly frozen**, zero trainable backbone parameters |
| query design | plain learned queries — no anchors, no denoising, no two-stage selection | full DINO family: two-stage selection, DAB anchors (or 3D anchors), denoising, deep supervision |
| mask formation | dot product of a query with a per-view DPT instance-feature map at H/2×W/2 | deformable pixel decoder + MaskDINO decoder over a ViTDet pyramid |
| attention dispersion | FADA, an explicit training-time auxiliary loss | implicit — deformable attention samples 4 points/level around an anchor |
| cost | 8×A100 × 2 days per dataset | head-only training in minutes on one GPU (features cached) |

The honest framing: **they buy accuracy by editing the backbone; we buy a controlled study and a
frozen-backbone claim by not touching it.** Their numbers are the price-of-admission for that choice,
and any comparison between the two projects is only meaningful once both the 2D→3D bridge and the
training setting are named — see `docs/RELATED_WORK.md`'s "Two 3D protocols" section and
`docs/TRAINING_COMPARABILITY.md` for the full detail behind each row above.
