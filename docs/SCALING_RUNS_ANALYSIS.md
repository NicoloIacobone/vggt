# Scaling Runs Analysis — scale10 & scale25 (first two §7.1 jobs)

**Date:** 2026-06-12 (analysis of jobs run 2026-06-11)
**Logs:** `slurm/train_scale10_3000973.log`, `slurm/train_scale25_3016559.log`
**Runs:** `d4rt_m2_scale10_20260611_120651`, `d4rt_m2_scale25_20260611_155050` (under
`/cluster/work/igp_psr/niacobone/distillation/output/`)

This document analyzes the first two scaling-experiment jobs (MILESTONE_2.md §7.1) and lays
out what to fix, re-run, and edit **before** launching `train_scale50.sh`. Written offline
(no GPU access); nothing was re-run or modified — everything below is a to-do list for the
next cluster session.

---

## 1. Run summaries

Both jobs exited cleanly (`.err` files contain only the harmless Euler module-stack notice;
scale10 printed `✅ SUCCESS`, scale25 printed the `⚠ Train mIoU below 0.5` warning — explained
below).

| | scale10 (job 3000973) | scale25 (job 3016559) |
|---|---|---|
| Train scenes | 10 (0000–0009) | 25 (0000–0024) |
| Val scenes | 0080–0082 | 0080–0082 |
| `--eval_interval` | **50** | **25** |
| `--early_stop_patience` | 5 evals = **250 epochs** | 5 evals = **125 epochs** |
| Stopped at | epoch 800 / 1000 | epoch **200** / 1000 |
| LR at stop (cosine over 1000) | 2.9e-4 (well into decay) | **1.86e-3 (still ~peak LR)** |
| Best val mIoU (prompted) | **0.278 @ epoch 550** | 0.196 @ epoch 75 |
| Final train mIoU (prompted) | 0.562 | 0.244 (loss still falling: 1.53) |
| Final val AP50 (prompted / grid) | 0.069 / 0.025 | 0.114 / 0.139 |
| Wall-clock training time | **4.3 min** | **3.1 min** |
| Checkpoint size | 444 MB | 809 MB |

Reference point: the Milestone-2 4-scene baseline reached best val mIoU **0.138**
(MILESTONE_2.md §6).

## 2. Main finding: the scale10 vs scale25 comparison is invalid

At face value the scaling curve went **backwards** (val mIoU 0.278 → 0.196 with 2.5× the
data). That is almost certainly an artifact of the experimental setup, not a real result:

1. **scale25 was early-stopped at epoch 200, deep inside warm-up territory.** The cosine
   schedule in `build_scheduler` (scripts/train_multiscene.py:238) spans `--num_epochs`
   (1000), so at epoch 200 the LR was still 1.86e-3 — the model never saw the decay phase.
   In scale10, *all* of the val-mIoU gains beyond 0.23 happened after epoch 300, as the LR
   annealed. scale25 was killed before that phase could start. Its last-epoch loss (1.53)
   was still falling steadily, and the final log line literally warns
   `⚠ Train mIoU below 0.5 — inspect the run`: the model is **underfit**, not worse.

2. **Patience is counted in evals, not epochs** (scripts/train_multiscene.py:529), and the
   two runs used different `--eval_interval` (50 vs 25). So scale25 effectively got *half*
   the patience window of scale10 (125 vs 250 epochs) — on a run that, with more data,
   needs *more* epochs to converge, not fewer. The two runs do not share a protocol, so
   their numbers aren't comparable.

3. **The early stop itself was triggered by val-metric noise.** scale25's val mIoU
   sequence was 0.158, 0.169, **0.196**, 0.179, 0.114, 0.170, 0.183, 0.185 — recovering
   and trending back up when patience ran out. With only 3 val scenes the eval is very
   noisy (scale10 oscillates ±0.04 around its plateau; scale25's val[grid] swung
   0.251 → 0.101 → 0.124 across three consecutive evals). A 5-eval patience on a 3-scene
   val set measures noise, not convergence.

4. **There is no compute pressure justifying aggressive early stopping.** Full training is
   3–4 *minutes* (feature caching dominates the job). The 4–12 h SLURM allocations are
   ~99% idle. Letting every run complete all 1000 epochs costs essentially nothing and
   makes the scaling curve clean.

**Consequence:** `train_scale50.sh` currently has the same configuration as scale25
(`--eval_interval 25 --early_stop_patience 5`) and will almost certainly suffer the same
premature stop. **Do not launch it as-is.**

### Encouraging signal hiding under the artifact

scale10's best val mIoU **0.278 is 2× the 4-scene baseline (0.138)** — the first real
evidence that held-out performance climbs with scene count. At the *same* epoch 200,
scale25 matched scale10 on val (0.185 vs 0.190) while seeing 2.5× more data per epoch and
being much further from convergence. There is no evidence against scaling here; the
experiment just needs to be re-run fairly.

## 3. Secondary observations from the logs

### 3.1 Overfitting gap at N=10 (expected, worth tracking)

scale10 final: train mIoU 0.562 vs val 0.271 — a large memorization gap, but exactly the
gap the scaling curve is supposed to close. Worth plotting train−val gap vs N once the
fair re-runs exist.

### 3.2 Prompted val AP50 is weak and got *worse* as val mIoU improved (scale10)

Val mIoU plateaued ~0.27 from epoch 350, but prompted val AP50 wandered 0.10–0.17 and the
last-epoch value was 0.069 with grid AP50 0.025. Model selection (line 519 of
`train_multiscene.py`) uses **prompted val mIoU only**; the checkpoint chosen at epoch 550
had val[grid] AP50 0.028, while epochs 400/600 had 0.137–0.139. If unprompted detection is
the headline metric (it is, per CLAUDE.md "the honest detection number"), selecting on
prompted mIoU may be picking the wrong checkpoint.

### 3.3 The score threshold (0.5) discards many correct predictions

In the auto-visualizations, a large fraction of matched queries have the **correct class
at score 0.28–0.49** and are dropped as "below score thr / bg — not drawn" (e.g. scale10
scene0003: `refrigerator -> refrigerator (0.28)`, scene0006: `bed -> bed (0.39)`,
`table -> table (0.34)`). The head is systematically under-confident. This depresses AP
(score-ranked) and makes the overlays look emptier than the model deserves.

### 3.4 Class-confusion structure is consistent and interpretable

- Reliable classes: `wall`, `floor`, `door`, `chair`, `desk` (scores often 0.6–0.97).
- Persistent confusion cluster: `window ↔ door ↔ picture ↔ curtain` — all flat, wall-mounted,
  rectangular. With masks evaluated on a 37×37 patch grid these are genuinely hard to
  distinguish geometrically; this is where RGB evidence should matter most.
- Val scene0080 is a bathroom; `toilet -> chair`, `sink -> chair`. Scenes 0000–0009 contain
  few/no bathroom fixtures — a class-coverage problem that more train scenes (25/50) should
  directly fix. Good qualitative metric to track across N.

### 3.5 Unprompted val mIoU consistently *exceeds* prompted (both runs)

E.g. scale10 final: val 0.271 prompted vs 0.316 unprompted; scale25: 0.185 vs 0.246. With
105–150 surviving grid predictions vs ~10–15 prompted ones, the matched-pair mIoU benefits
from having more candidates to match while unmatched false positives go unpunished (they
only hurt AP). Treat unprompted mIoU as optimistic; AP50 is the honest unprompted number —
worth a one-line caveat wherever these numbers get reported (slides included).

### 3.6 Checkpoint size scales linearly with scene count and will bite at N≥50

444 MB (10+3 scenes) → 809 MB (25+3) ≈ **29 MB/scene**, dominated by the float32 images of
bundle 0 of every scene stored in the checkpoint (`save_checkpoint`,
scripts/train_multiscene.py:249 — 8 frames × 3×518×518 float32 ≈ 26 MB/scene). At N=50
that's ~1.6 GB *per checkpoint*, and every new best-val event rewrites the whole thing
(scale10 rewrote 444 MB six times). At N=100+ this becomes slow and storage-heavy.

## 4. Action plan for the next cluster session

### 4.1 Fix the SLURM scripts first (no Python changes needed)

In **`slurm/train_scale25.sh`** and **`slurm/train_scale50.sh`** (and arguably
`train_scale10.sh` for protocol uniformity):

```
--eval_interval 50 --early_stop_patience 0      # was: --eval_interval 25 --early_stop_patience 5
```

`--early_stop_patience 0` disables early stopping entirely (confirmed: the check at
train_multiscene.py:529 is gated on `> 0`); best-checkpoint saving is unaffected. Given
3–4 min runtimes, running all 1000 epochs is the simplest fair protocol. If early stopping
must stay, use `--eval_interval 50 --early_stop_patience 10` (= 500-epoch window) — but for
the scaling *curve* itself, identical fixed-length runs are cleaner.

The SLURM `--time` allocations can also be slashed (the 25-scene job used ~30 min of an 8 h
allocation, almost all of it feature caching + visualization) — e.g. 2 h is generous for
scale50. Shorter requests also schedule faster.

### 4.2 Re-run queue (in order)

1. **Re-run scale25** with the fixed protocol → the real N=25 point.
2. **Re-run scale10** with `--early_stop_patience 0` → full-schedule N=10 point (its best
   was at 550 with LR still at ~9e-4; the tail of the cosine may add a little, and the
   protocol then matches N=25/N=50 exactly).
3. **Launch scale50** only after the fixes.
4. Plot val mIoU (prompted + grid) and val AP50 (grid) vs N ∈ {4, 10, 25, 50} — the §7.1
   deliverable. The N=4 point (0.138) exists in MILESTONE_2.md §6.

### 4.3 Code edits worth making (small, in priority order)

1. **Persist the eval history to disk.** `eval_all` results currently only go to stdout;
   the loss `history` list is built (train_multiscene.py:501) but never saved. Append one
   JSON line per eval (epoch, lr, loss, train/val mIoU & AP50, prompted + grid) to
   `<run_dir>/metrics.jsonl`. The scaling plots then come from files instead of log
   scraping. (~15 lines, no behavior change.)
2. **Shrink checkpoints.** Store `images` as uint8 (`(img.clamp(0,1)*255).to(torch.uint8)`)
   and convert back to float in `scenes_from_checkpoint` / the demo loader — 4× smaller
   (~1.6 GB → ~0.4 GB at N=50) with zero quality impact on visualization. Optionally add
   `--checkpoint_light` that drops per-scene images entirely and stores `frame_names` +
   scene path instead (the visualizer can reload them from `--scans_root`). Keep the
   head-config round-trip intact (CLAUDE.md convention).
3. **Make early stopping noise-robust** (if it's ever re-enabled): add
   `--early_stop_min_delta` (e.g. 0.005) and compare against a 2–3-eval moving average of
   val mIoU rather than single evals; optionally refuse to stop before some fraction of
   the schedule (e.g. epoch ≥ 0.5·num_epochs) so the LR decay phase is always reached.
4. **Track a second "best" checkpoint selected on val[grid] AP50** (or log which epoch
   would have been chosen) — cheap way to test §3.2's hypothesis that prompted-mIoU
   selection picks a poor detection checkpoint. Decide afterwards whether to switch the
   selection metric.
5. **Decouple schedule length from `--num_epochs`** (optional `--schedule_epochs`): with
   early stopping disabled this matters less, but it removes the failure mode of §2.1
   permanently.

### 4.4 Cheap experiments enabled by the tiny runtimes

Each is a ~30 min job dominated by caching; all use the fixed protocol of §4.1:

- **Widen the val set** (e.g. 0080–0089 if preprocessed, else preprocess a few more) to
  denoise model selection — §2.3 showed 3 scenes is not enough signal. This changes the
  selection metric, so do it *before* the final scaling-curve runs, not after.
- **Score-threshold sweep at eval/visualization time** (no retraining): re-run
  `scripts/visualize_masks.py` with `--score_threshold 0.3` on the existing scale10
  checkpoint and see how the overlays change (§3.3). If the no-object head is just
  under-confident, also consider whether `no_object_weight 0.1` is pushing scores down —
  the §7.2 no-object sweep (0.05/0.1/0.4) will answer this properly once N≥25 is trained
  fairly.
- **LR sanity check at N=25**: one run at `--learning_rate 1e-3`. The 2e-3 value was tuned
  in the 4–10-scene regime; with 2.5× more optimizer steps per epoch the larger dataset
  may prefer a lower peak LR (scale25's per-epoch loss curve was noticeably noisier than
  scale10's).
- **Capacity probe (only if the fair N=25/N=50 runs underfit)**: train loss stuck well
  above the N=10 level with val flat would motivate `num_decoder_layers` 4→6 or
  `hidden_dim` 256→384. Note this changes `head_config` — keep the checkpoint round-trip
  intact. Don't do this before the fair re-runs; right now there's no evidence of a
  capacity limit, only of an unfinished run.

### 4.5 Things checked and fine (no action)

- Backbone caching works as designed at both scales; `--cache_device cpu` (scale25) adds no
  measurable per-epoch cost (3.1 min for 200 epochs × 75 bundles).
- Per-epoch `matches` ≈ scenes × ~7.5 GT instances in both runs — the matcher is matching
  essentially all GT instances every epoch, as intended.
- Bundle instance counts fluctuate by ±1–2 across bundles of the same scene (random frame
  subsets see different object subsets) — expected augmentation behavior, not a data bug.
- Initial (untrained) metrics ≈ 0 everywhere — no leakage from initialization.

## 5. Open questions to revisit after the fair re-runs

- Does val mIoU scale with N? (The actual §7.1 question — currently unanswered; only the
  N=4→10 jump of 0.138→0.278 is trustworthy.)
- Is prompted-mIoU model selection the right criterion, or should the grid AP50 drive it
  (§3.2 / §4.3.4)?
- Is the window/door/picture confusion (§3.4) a data-coverage problem (fixed by N) or a
  37×37-resolution problem (needs the FPN-style mask upsampling flagged at the end of
  MILESTONE_2.md §7)?
- Where does the train–val gap (§3.1) go as N grows — and at what N does it justify
  partial backbone unfreezing?
