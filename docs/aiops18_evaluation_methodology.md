# AIOps18 Evaluation Methodology — Episode-Level Metrics

This document fixes the rule for computing detection rate, false-alarm rate, and lead time
on the windowed arrays in `processed_data/aiops18/`, written down now (before any model or
metric code exists) so the rule doesn't drift or get reinvented ad hoc later. It belongs with
the model/evaluation code once that exists, not with preprocessing — see
`docs/aiops18_preprocessing_methodology.md` for how the windows themselves were built.

## Why per-window metrics are not usable as-is

The sliding-window construction uses a 5-minute stride against 15-minute and 60-minute
forecast horizons (`docs/aiops18_preprocessing_methodology.md`, Section 5), which means any
single point in time falls inside the horizon of **3 consecutive windows** at the 15-minute
horizon and **12 consecutive windows** at the 60-minute horizon. A naive per-window
detection-rate or false-alarm-rate calculation would therefore count one real onset event as
up to 12 separate "detections," and one brief false-trigger episode as up to 12 separate
"false alarms" — inflating both metrics and overstating the effective sample size. This was
flagged when the overlap was first quantified; the rule below is the fix.

**All detection-rate, false-alarm-rate, and lead-time metrics are computed at the episode
level, not the per-window level.**

## Detection rate

**Episode** = one true onset event: a single 0→1 transition in a KPI's raw label sequence,
identified independently of any window (this is the same definition used for
`n_onset_segments` in the preprocessing manifest — 2,161 total across the merged dataset).

**Overlapping windows for an episode** = every window (same KPI, same split, same horizon)
whose horizon interval `[window_start_ts + context_steps * modal_interval, window_start_ts +
(context_steps + horizon_steps) * modal_interval)` contains the onset's absolute timestamp.
Given the fixed stride and horizon, this set has exactly `horizon_steps / stride_steps`
members when the onset falls in the interior of a run (3 for the 15-minute horizon, 12 for
the 60-minute horizon) — fewer only if the onset falls near the start/end of a gap-bounded or
split-bounded run, where fewer overlapping windows exist to begin with.

**Collapsing rule**: an episode counts as **detected** if the model predicts a positive
(flags an onset) in **any** of its overlapping windows, and **missed** otherwise. This is a
single existential OR across the overlapping set — one binary outcome per episode, regardless
of how many of the overlapping windows the model actually flags (flagging 1 of 3, 2 of 3, or
3 of 3 all collapse to the same "detected" outcome).

**Lead time** for a detected episode = the time between the onset and the **last context
timestamp of the earliest-starting window among the overlapping windows that correctly
flagged it**. Using the earliest correctly-flagging window gives the maximum lead-time credit
the model actually earned for that episode, consistent with an early-warning framing that
cares about the earliest point a warning could have fired. An episode with no correctly
flagging window has no lead time (it's a miss, not a zero-lead-time detection).

### Worked example (real data, not synthetic)

KPI `02e99bd4f6cfb33f`, validation split, 15-minute horizon, 60s resolution
(`context_steps=288`, `stride_steps=5`, `horizon_steps=15`).

One true onset occurs at **2017-06-29 06:08:00 UTC**. Its overlapping windows (found by
scanning the val split's saved arrays for every window whose horizon contains this timestamp):

| Window | Context ends (= horizon start) | Horizon (wall-clock) | Onset position in horizon |
|---|---|---|---|
| idx 1038 | 2017-06-29 05:57:00 | [05:57:00, 06:12:00) | step 11 (06:08:00) |
| idx 1039 | 2017-06-29 06:02:00 | [06:02:00, 06:17:00) | step 6 (06:08:00) |
| idx 1040 | 2017-06-29 06:07:00 | [06:07:00, 06:22:00) | step 1 (06:08:00) |

Exactly 3 overlapping windows, matching `horizon_steps / stride_steps = 15/5 = 3` as
expected. Confirmed these all track the *same single* onset, not three different ones: the
onset's position within the horizon shifts by exactly 5 steps between consecutive windows
(11 → 6 → 1), matching the 5-step stride, and no window's horizon contains a second,
different onset before or after this one.

**Applying the rule**: this is one episode with 3 overlapping windows. Whatever the model
predicts on windows 1038, 1039, and 1040 individually, the episode-level outcome is:
- **Detected** if the model flags an onset on at least one of {1038, 1039, 1040}.
- **Missed** only if the model flags none of the three.

This produces exactly one clean outcome for this episode — not fractional, not
double-counted across the three windows — confirming the collapsing rule works as intended
before adopting it as final. If detected, lead time is computed from window 1038 (the
earliest of the three, if it was among the windows that correctly flagged the onset), giving
up to 11 minutes (660 seconds) of lead time (06:08:00 onset minus 05:57:00 context end) — the
most credit available for this particular episode.

**Code validation**: `scripts/models/evaluation_metrics.py` was run against this exact
episode (KPI `02e99bd4f6cfb33f`, onset timestamp 1498716480) and reproduces it precisely —
the same three windows (1038, 1039, 1040) are grouped as one episode, and a model that
flags all three returns `lead_time_seconds=660.0`, matching this worked example exactly.
This check is what caught the off-by-one-minute error originally in this table (context-end
was written one minute before horizon-start instead of equal to it) — corrected above.

## False-alarm rate

**False-positive window** = a window where the model predicts a positive but
`onset_in_horizon == 0` for that window (no true onset anywhere in its horizon).

**Collapsing rule**: merge false-positive windows into false-alarm episodes using standard
overlapping-interval merging on their horizon intervals, per KPI per split:

1. Collect every false-positive window's horizon interval `[horizon_start, horizon_end)`.
2. Sort by `horizon_start`.
3. Walk the sorted list, maintaining a current merged interval. If the next interval's
   `horizon_start <= current_merged.horizon_end`, extend the current merged interval's end to
   `max(current_merged.horizon_end, next.horizon_end)`. Otherwise, close the current merged
   interval and start a new one from the next window.
4. Each final merged interval is **one false-alarm episode**, regardless of how many
   individual false-positive windows fed into it.

Illustrative example (synthetic, for clarity — not from real data): if windows with horizons
`[10:00,10:15)`, `[10:05,10:20)`, `[10:10,10:25)`, and separately `[11:00,11:15)` are all
false positives, the first three merge (each overlaps the next) into one episode spanning
`[10:00,10:25)`, and the fourth is its own episode — **2 false-alarm episodes, not 4**.

False-alarm rate is then reported as episodes per unit time (or per KPI-day), not as a
fraction of windows — consistent with treating episodes, not windows, as the unit of
evaluation throughout.

## Summary

| Metric | Unit of counting | Collapsing rule |
|---|---|---|
| Detection rate | one outcome per true onset episode | detected if flagged in *any* overlapping window |
| Lead time | one value per detected episode | measured from the earliest correctly-flagging overlapping window |
| False-alarm rate | one count per merged false-positive interval | overlapping-interval merge on false-positive windows' horizons |

This rule applies uniformly across both horizons (15 min, 60 min) and both resolutions (60s,
300s KPIs) — only the concrete `horizon_steps`/`stride_steps` ratio (and therefore the number
of windows expected to overlap a given episode) differs, not the collapsing logic itself.

## Random seed (LSTM, PatchTST)

**Fixed at `SEED = 42`** in `run_aiops18_phase2.py`. `set_seed()` calls both
`torch.manual_seed(seed)` and `np.random.seed(seed)`, and **must be called immediately
before constructing a model, not just before training** — weight initialization happens in
`__init__`, not in `fit_shared`'s training loop, so seeding only at the start of training
would control the epoch shuffle order but leave initial weights (the larger source of
run-to-run variance for a small dataset of 246 val episodes) uncontrolled. `main()` calls
`set_seed()` once immediately before each of `LSTMModel(...)` and `PatchTSTModel(...)` are
constructed, so every future run of either model — including this run and correction-layer
experiments built on top of it later — is reproducible from the same starting point.

This was added specifically to resolve an open question: an unseeded LSTM rerun on the
z-clipped data showed val detection rate drop from 7.32% (pre-clip) to 3.25% (post-clip),
while test detection stayed flat (4.72%→4.74%). Two explanations were possible — the clip
itself changing val-specific behavior, or ordinary run-to-run initialization variance on a
246-episode val set — and they were not distinguishable without a controlled rerun.

### Resolution: predominantly initialization variance

A seeded (`SEED=42`) LSTM rerun on the same clipped data gave val detection = 6.50%
(16/246) — much closer to the original pre-clip 7.32% than to the unseeded post-clip run's
3.25%. This confirms the swing was mostly random-initialization variance, not the clipping
fix changing val-specific model behavior. Test detection stayed essentially flat across all
three runs (4.72% → 4.74% → 4.74%).

**Asymmetry not fully explained**: test was stable across all three runs while val swung by
more than 2x. A seeded PatchTST rerun on the same data, for comparison, showed much smaller
seed-to-seed movement on both splits (val 7.72%→7.32%, test 5.17%→4.74% — one episode of
difference each), so the volatility is specific to LSTM on this val set, not a general
property of val vs. test here.

**Hypothesis tested and not supported**: one candidate explanation was that val's true
episodes sit structurally closer to their KPIs' calibrated thresholds than test's (small
margins would be more likely to flip between runs). Directly tested by computing, for each
of the 246 val / 232 test episodes (same clean set used throughout, low-precision KPIs
excluded), the true peak |z| across its overlapping windows minus its KPI's calibrated
threshold:

| | val (n=246) | test (n=232) |
|---|---|---|
| Median margin | 1.342 | 1.099 |
| Fraction with \|margin\| < 1.0 | 38.6% | 39.7% |
| Mann-Whitney U test | p = 0.167 | |

This runs opposite to the hypothesis (val's margins are if anything slightly *larger*, not
smaller, than test's) and the difference isn't statistically significant at this sample size.
**There is no structural evidence that val's signal sits closer to the calibration boundary
than test's** — this specific explanation is ruled out, and LSTM's val-specific detection
volatility (absent in PatchTST on the same data) remains unexplained. Flagging as an open
question rather than asserting a mechanism that isn't supported by the data.

## Chronos (zero-shot foundation model) evaluation

**Checkpoint**: `amazon/chronos-t5-tiny` (8,394,496 parameters), the smallest available
Chronos checkpoint. `chronos-forecasting==1.4.1` (the pre-Chronos2, T5-based package line —
the current 2.x release requires `transformers` versions needing PyTorch >= 2.5, which
doesn't exist for this environment's Intel macOS / PyTorch 2.2.2 constraint; see
`baseline_lstm.py`'s docstring for the same platform limitation). Checkpoint size is a
deliberate, stated parameter of this evaluation — local CPU-only hardware, not a compromise
on the result: a zero-shot foundation model is evaluated as-is, off-the-shelf, so which
checkpoint size represents "off-the-shelf performance" is a real methodological choice, made
explicit here rather than silently defaulting to whatever ran fastest.

**No fine-tuning.** `run_aiops18_chronos.py` loads the pretrained checkpoint and calls
`pipeline.predict()` directly on the same clipped, windowed val/test data every other model
uses — no training loop, no gradient updates, nothing analogous to `fit_shared`. This
mirrors realistic deployment (no per-site retraining) and is the correct comparison point
for "can an off-the-shelf foundation model match purpose-built forecasters," not a
handicapped version of Chronos.

**Timing pilot** (100 real val windows, `num_samples=20`, the package's standard/recommended
setting): 0.6438s/window. Batching context windows together did not meaningfully help
(0.466s/window batched vs. 0.469s/window sequential in a separate check) — Chronos generates
each of its `num_samples` autoregressively, so cost scales with sample count, not with how
many windows are batched together. Extrapolated to the full val+test set (350,246 windows):
**62.6 hours raw, 125.3 hours with the mandated 2x caution multiplier** — not viable to run
in full at `num_samples=20`. A `num_samples=1` alternative was measured (0.0426s/window,
~8.3 hours with caution) but was not used, since a single autoregressive sample is a real
deviation from standard Chronos usage, not just a speed optimization.

**Resolution: a stratified subsample, not the full window set.** Two components:
1. Every window overlapping one of the 246 val / 232 test true onset episodes (the same
   clean, non-low-precision-KPI episode set every other model's detection-rate, lead-time,
   and false-alarm metrics are computed on) — 733 val + 690 test = 1,423 windows. Full
   episode coverage means those metrics are **not degraded** by subsampling; every episode
   any other model was scored against, Chronos was too.
2. An equal-per-KPI stratified sample of normal (non-episode) windows — 122 windows per KPI
   per split (all 41 KPIs had at least 122 available, so the allocation came out exactly
   even, no KPI shortfall) — 5,002 val + 5,002 test = 10,004 windows. Stratified by KPI
   rather than a flat pooled random draw specifically so no single series with a
   disproportionately large window count could dominate the MAE/RMSE aggregate.

Total: 11,427 windows, estimated at ~2.04 hours raw / ~4.1 hours with the caution multiplier
at the measured pilot rate — actual run time was consistent with this estimate. **MAE/RMSE
are therefore a representative-subsample estimate (n=5,735 val / 5,692 test), not the
full-population value the other five models report (n=179,684 val / 170,562 test)** — stated
directly in the results table, not left implicit. Detection rate, lead time, and false-alarm
rate are computed on the complete real episode set and are directly comparable to the other
five models without caveat.

**Checkpointing**: `run_aiops18_chronos.py` saves progress to
`results/aiops18/chronos_checkpoint.npz` after every 50-window chunk, via atomic
write-then-rename (never a partially-written checkpoint file). Rerunning the same command
after an interruption loads the checkpoint and skips already-completed windows rather than
restarting — verified directly before the real run: a smoke-scale run was killed mid-chunk
and the rerun correctly resumed from the exact window count where it left off, not from
zero.

**Manual spot-check**: the same onset used for the LSTM/PatchTST spot-checks (KPI
`05f10d3a-239c-3bef-9bdc-a2feeb0037aa`, onset 2017-06-04 03:12:00) was covered by Chronos's
run (guaranteed, since all clean-KPI episode windows are included by design). All 3
overlapping windows' `max|pred|` (2.693, 2.559, 3.034) exceed the KPI's calibrated threshold
(2.539), matching the module's `predicted_positive=True` for all three and hand-traced
`detected=True`, lead time 840s from the earliest flagging window — consistent with every
other model's spot-check on this exact episode.

**Result**: Chronos's zero-shot MAE/RMSE are the worst of all six models on both splits
(0.338/0.894 val, worse than even the trivial persistence baseline's 0.271/0.571), but its
detection rate (6.10% val, 5.17% test) is mid-pack, matching SARIMA/XGBoost/persistence's
test rate exactly. A general-purpose zero-shot model does not match specifically-trained
forecasters on raw accuracy here, but the accuracy gap does not translate one-to-one into a
comparable detection-rate gap.

## Model-agnostic post-hoc correction layer

A lightweight layer inserted between a model's already-saved raw forecast and the flagging
decision — combines residual monitoring with change-point-triggered adaptive thresholding.
Implemented in `scripts/models/correction_layer.py`, applied to all six models' existing
predictions by `scripts/models/run_aiops18_correction.py`. Operates purely on saved
predictions (`kpi_id`, `window_start_ts`, `y_true`, `y_pred`) — no retraining, no access to
raw data beyond what the original windowed arrays provide for cross-referencing episode
labels. Runs in minutes.

### Algorithm

1. **Residual stream**: one scalar residual per window, `mean(|y_true − y_pred|)` across
   that window's 15-step horizon — not per-raw-timestep. This is a window-index-based
   design choice (consistent with every other part of this pipeline treating the window,
   not the raw 60s sample, as the unit of analysis), stated explicitly since "30 steps"
   could otherwise be read as 30 raw timesteps (30 minutes) rather than 30 windows (150
   minutes at 5-minute stride).
2. **Change-point detection**: one-sided (upward-shift) Page's CUSUM per KPI, on that KPI's
   time-ordered residual stream. At window *i*, the local baseline (mean, std) is computed
   from the trailing `ROLLING_WINDOW=30` windows strictly before *i* (no lookahead); the
   residual is standardized against that baseline, accumulated into a cumulative sum with
   slack `k=0.5`, and a change-point is flagged when the cumulative statistic exceeds
   `h=4.0` (both standard Page's CUSUM defaults for detecting an ~1-sigma sustained shift).
   The statistic resets to 0 immediately after flagging, so one regime shift is flagged
   once, not repeatedly while still elevated. This detects a *sustained* shift in the
   residual's statistical behavior, not a single-point spike — a lone large residual with no
   persistence won't accumulate enough CUSUM mass to cross `h`.
3. **Adaptive thresholding, causally gated**: at baseline, the existing per-KPI
   F1-calibrated threshold (`threshold_calibration.py`) is used unchanged. A change-point
   flagged at window *j* is derived from `residual[j]`, which requires *j*'s own realized
   horizon — i.e. it isn't knowable until `matured_ts[j] = context_end_ts[j] +
   horizon_steps×modal_interval`, one full horizon *after* `context_end_ts[j]`, the timestamp
   at which window *j*'s own detection decision is actually made. A change-point may
   therefore only suppress the threshold for windows whose own decision time is `>=` that
   `matured_ts` — found via `searchsorted` on the per-KPI `context_end_ts` array, so it holds
   under irregular stride/gaps, not just uniform spacing. Starting at the first such window,
   the threshold drops to `base × (1 − 0.20)`, then decays linearly back to `base` over
   `POST_CP_WINDOW=15` windows, counted from that first causally-eligible window, not from the
   change-point's own index. Overlapping schedules (a second change-point maturing while still
   inside the first's decay window) take the more sensitive (lower) of the two at each step,
   rather than resetting or stacking. This structurally guarantees a change-point can never
   suppress the threshold for its own triggering window (`matured_ts[j] > context_end_ts[j]`
   always, since `horizon_steps > 0`), and typically excludes the next one or two windows
   immediately after it as well, whichever ones still decide before the change-point matures.
   **An earlier version of this layer applied the decay starting at the change-point's own
   index** (`offset=0` at the triggering window itself), letting a window's own future-realized
   outcome lower its own detection threshold — a look-ahead bug, not a design choice. All
   numbers in this document and in `model_comparison_summary.csv` reflect the corrected,
   causally-gated version; see Result below for the magnitude of the difference this made.
4. **Output**: `flag_onset_from_forecast`'s existing rule (`max|forecast| > threshold`) is
   reapplied per window using this dynamic threshold instead of the static one, producing a
   new `predicted_positive` array. This is fed through `evaluation_metrics.py` completely
   unchanged — same detection/false-alarm/regression functions, same clean 246/232 episode
   set, same low-precision-KPI exclusion from the aggregate — so `{model}_corrected` rows in
   `model_comparison_summary.csv` are directly comparable to the six raw rows already there.

### Parameters: chosen, not tuned

`ROLLING_WINDOW=30`, `POST_CP_WINDOW=15`, `THRESHOLD_REDUCTION=0.20` (plus CUSUM's
`k_slack=0.5`, `h_threshold=4.0`, both standard textbook defaults rather than fit to this
data) are a **principled first pass, explicitly not tuned against val or test** — consistent
with how stride (Section 5 of the preprocessing doc) and the z-clip bound (±10σ) were
handled elsewhere in this pipeline. No sensitivity sweep has been run on any of these three
correction-layer parameters; if this becomes a primary result rather than a demonstration,
that sweep is the natural next step.

### Caveat: not temporally equivalent across models

For the five models evaluated on the complete, contiguous window set (persistence through
PatchTST), 30 trailing windows means 150 minutes of real history — a consistent, meaningful
window. **For Chronos, evaluated on a stratified subsample (Section on Chronos above), the
same "30 windows" can span many hours or days of wall-clock time**, since most of a given
KPI's normal windows were sampled out. The correction layer runs with identical code and
parameters on both, but its temporal meaning is not equivalent — concretely visible in the
results: Chronos flagged only 247–255 change-points total (val/test) versus 5,682–7,813 for
each of the five dense models, an order of magnitude fewer opportunities for the rolling
baseline to detect a shift. This is flagged here rather than presented as if Chronos's
correction-layer behavior were directly comparable to the other five's.

### Manual spot-check

**Note on this section's history**: the walkthrough below (KPI `02e99bd4f6cfb33f`) was
originally used to validate the correction layer, but it was validating the pre-fix,
look-ahead-buggy version — under that version this episode's suppression came from
change-points at windows 3701/3702 applied to those *same* windows (offset=0 from their own
index), which is exactly the leakage the fix removes. **Under the corrected, causally-gated
logic, this specific episode is no longer detected**: windows 3701–3703 keep the static
threshold 1.6013 (unsuppressed) at their own decision times, because the change-points they
themselves trigger don't mature until window 3704 — one window too late to help this episode.
`compute_detection_metrics` on this episode under the fixed code returns
`n_flagged_windows=0, detected=False`. The walkthrough is kept below, annotated, because it's
the clearest concrete illustration of what the bug actually did; a genuine, still-valid
example follows it.

Searched SARIMA's val predictions for episodes where the raw (static-threshold) decision
missed but the corrected decision detected — found 17 under the pre-fix code; **5 remain
under the fixed code** (KPIs `05f10d3a-239c-3bef-9bdc-a2feeb0037aa`,
`6efa3a07-4544-34a0-b921-a155bd1a05e8`, `88cf3a776ba00e7c`, `8bef9af9a922e0b3`,
`affb01ca2b4f0b45`). First, the original (now-invalid) example: KPI `02e99bd4f6cfb33f`,
onset **2017-07-08 12:02:00**, base threshold 1.6013.

Windows are indexed by context-start (`window_start_ts`); with `context_steps=288` at 60s
resolution, a window's horizon — and therefore its `context_end_ts`, the timestamp lead time
is measured from — lands 4.8h after its own start. The three windows whose horizon actually
overlaps this onset (found via `extract_true_onset_timestamps`, not by proximity of
`window_start_ts` to the onset) are:

| Position | Window start | Context end (horizon start) | Residual | Threshold | Change-point? |
|---|---|---|---|---|---|
| ... | 06:39–06:59 | — | 0.001–0.007 | 1.6013 (static) | — |
| 3701 (overlapping) | 07:04 | 11:52 | **0.2306** | 1.2810 | **flagged** |
| 3702 (overlapping) | 07:09 | 11:57 | 0.2306 | 1.2810 | **flagged** |
| 3703 (overlapping) | 07:14 | 12:02 | 0.2342 | 1.3024 | (within decay window) |
| 3704 (not overlapping this onset) | 07:19 | 12:07 | 0.1998 | 1.2810 | **flagged** (third trigger) |

The residual jumps from ~0.001–0.007 to ~0.23 and **stays elevated across multiple
windows**, which is what actually happens here: CUSUM re-triggers twice more after its first
flag at 3701 — at 3702 (07:09) and again at 3704 (07:19) — rather than firing once and
decaying on a single fixed schedule. Each flagged index spawns its own 15-window decay
schedule, and `adaptive_threshold_schedule` takes the more sensitive (lower) of any
overlapping schedules at each step. At 3703, two schedules overlap: CP@3701's offset=2/15
term (`1.6013×0.8 + 1.6013×0.2×2/15 = 1.3237`) and CP@3702's offset=1/15 term
(`1.6013×0.8 + 1.6013×0.2×1/15 = 1.3024`) — **the reported 1.3024 comes from CP@3702's
offset=1/15 term, not CP@3701's offset=2/15 term**, since the more sensitive of the two wins.
(3704 is a genuine third re-trigger of the same sustained shift, but its own horizon starts
at 12:07, after the onset, so it isn't one of this episode's overlapping windows — it's part
of the same residual elevation, evaluated for a different, later forecast target.) The
residual falls back toward baseline after 3704 and the threshold decays fully back to 1.6013
by window-start 08:34, roughly 90 minutes after the first flag — a short, self-terminating
burst, not a plateau held open through the onset.

All three overlapping windows' `max|pred|` (1.5464, 1.5462, 1.5291) sit below the static
threshold (1.6013 — correctly missed by the raw decision, verified against the actual saved
`predicted_positive=False`) but above their respective lowered adaptive thresholds —
correctly caught by the corrected decision. Calling `compute_detection_metrics` directly on
this episode confirms `n_flagged_windows=3`, `detected=True`, and — using the actual
production lead-time convention (`onset_ts − context_end_ts` of the earliest-firing window,
i.e. 3701's context end at 11:52) — **`lead_time_seconds=600.0`, a 10-minute lead time**,
consistent with the 4–14 minute range seen elsewhere in this validation. (An earlier version
of this note measured lead time from window *start* rather than context *end*, which
overstated it by the full 4.8h context span; that was an error in ad hoc analysis, not in
`evaluation_metrics.py` or in any number in `model_comparison_summary.csv`, which has always
used `context_end_ts` per the function definition above. This is separate from the
threshold-suppression look-ahead bug described above — as just shown, this episode is no
longer detected once that bug is fixed.)

**Valid example under the fixed logic**: KPI `05f10d3a-239c-3bef-9bdc-a2feeb0037aa`, onset
**2017-06-17 08:23:00**, base threshold 2.5386 (train precision 0.4508, well above the 0.30
exclusion bar). The three onset-overlapping windows:

| Window start | Context end | Residual | Own change-point? | Threshold | max\|pred\| |
|---|---|---|---|---|---|
| 03:24 | 08:12 | 1.7217 | yes | **2.0309** | 2.4417 |
| 03:29 | 08:17 | 3.8522 | yes | **2.0648** | 2.3219 |
| 03:34 | 08:22 | 5.9285 | yes | **2.0986** | 2.4458 |

All three are themselves flagged as change-points, but none is suppressed by its *own* flag —
each window's applied threshold (2.03–2.10) instead traces back to a change-point at window
**03:09** (residual 0.6305, its own trigger), whose `matured_ts` (context end 07:57 + 15min
horizon = 08:12) lands exactly at the first window's context end, making it the earliest
eligible suppression source. All three `max|pred|` values (2.32–2.45) sit below the static
threshold (2.5386 — correctly missed raw) but above the causally-valid adaptive threshold —
correctly caught corrected. `compute_detection_metrics` confirms `n_flagged_windows=3,
detected=True, lead_time_seconds=660.0` (11.0 minutes), in the same 4–14 minute range as
before — this example shows the mechanism still works correctly once look-ahead is removed,
just via a genuinely earlier change-point rather than the triggering window's own future.

### Result

**These numbers reflect the causally-gated (fixed) implementation.** An earlier version of
this section reported numbers from a buggy implementation in which a change-point could
suppress the threshold for its own triggering window — a look-ahead leak (see Algorithm §3
and the spot-check above). That bug inflated the apparent improvement; the honest numbers
below are smaller, and for four of twelve model/split combinations the correction layer no
longer beats the raw baseline at all. This is reported as-is, not smoothed into "smaller but
still consistently positive," because it isn't consistently positive.

| model | split | precision: raw→corrected | recall: raw→corrected | F1: raw→corrected |
|---|---|---|---|---|
| Persistence | val | 0.0287→0.0277 | 0.077→0.098 | 0.0418→0.0431 |
| Persistence | test | 0.0174→0.0208 | 0.052→0.069 | 0.0260→0.0319 |
| SARIMA | val | 0.0484→**0.0389** | 0.069→0.089 | 0.0570→**0.0543** |
| SARIMA | test | 0.0262→**0.0220** | 0.052→0.056 | 0.0348→**0.0316** |
| XGBoost | val | 0.0415→0.0479 | 0.098→0.130 | 0.0583→0.0700 |
| XGBoost | test | 0.0201→0.0257 | 0.052→0.078 | 0.0290→0.0386 |
| LSTM | val | 0.0449→**0.0426** | 0.065→0.089 | 0.0532→0.0577 |
| LSTM | test | 0.0234→**0.0215** | 0.047→0.056 | 0.0313→**0.0311** |
| PatchTST | val | 0.0513→**0.0391** | 0.073→0.089 | 0.0603→**0.0544** |
| PatchTST | test | 0.0276→0.0331 | 0.047→0.078 | 0.0349→0.0465 |
| Chronos | val | 0.1579→**0.1409** | 0.061→0.085 | 0.0880→0.1063 |
| Chronos | test | 0.0750→**0.0686** | 0.052→0.060 | 0.0612→0.0642 |

(Bold = corrected value is worse than raw on that metric.)

MAE/RMSE are unchanged for every model (the correction layer never touches the forecast
itself, only the flagging threshold). **F1 improves for 8 of 12 model/split combinations**
(persistence both splits, XGBoost both splits, LSTM/val, PatchTST/test, Chronos both splits)
and **regresses for 4** — SARIMA val (0.0570→0.0543) and test (0.0348→0.0316), PatchTST/val
(0.0603→0.0544), and LSTM/test, which is flat to a hair negative (0.0313→0.0311). Compared to
the pre-fix numbers, every combination's F1 improvement shrank by 12–41% in relative terms
(median ≈−25%), consistent with the expected consequence of removing look-ahead: a real
correction-layer effect remains for most models, but it is smaller and, for two models
(SARIMA on both splits, and PatchTST on val), it no longer clears raw at all.

Precision moved from "improves or holds flat in 11/12" (the pre-fix finding) to **worse than
raw in 8 of 12** under the fixed logic. This reverses the earlier claim that the trade wasn't
lopsided — under honest causal ordering, the correction layer trades precision for recall in
the majority of cases, and for SARIMA/PatchTST-val/LSTM-test that trade isn't favorable enough
to net out positive on F1 either. The mechanism proposed earlier (suppression deployed
selectively where true positives are more likely) still operates, since real change-points
still correlate with real onsets — it's just weaker than the pre-fix numbers suggested,
because part of that apparent selectivity came from a change-point "knowing" about its own
window's future residual, which is not information the layer can legitimately use online.

Absolute precision remains low throughout regardless of the fix, 0.021–0.158 even after
correction — the large majority of flagged episodes are still false alarms in absolute terms
for every model, before or after correction.

Chronos still has the highest precision of the six, both raw and corrected, on both splits
(0.069–0.158 vs. 0.021–0.048 for the other five), consistent with its much smaller raw
false-alarm counts. Its detection-rate gain is smaller under the fix than previously reported
(raw→corrected 6.10%→**8.54%** val, 5.17%→**6.03%** test — down from the pre-fix 6.10%→9.76%
and 5.17%→9.48%), and its change-point count is unchanged by the fix (247–255 flaggable
change-points val/test combined, identical to before, since the fix only changes *when* a
flag can be used, not *whether* one fires) versus 5,682–7,813 for the five densely-evaluated
models. The stratified-subsample caveat from before still applies and, if anything, is
sharper now: with fewer flaggable change-points to begin with, the additional causal-maturity
gating leaves even less room for the layer to act within Chronos's sparse sample. As before,
this is a property of the evaluation design (sparse sampling), not evidence that regime-aware
correction is inherently weaker for foundation models — but the magnitude of Chronos's
remaining gain should be read as a lower bound shaped by that design, not a clean estimate of
the method's ceiling.
