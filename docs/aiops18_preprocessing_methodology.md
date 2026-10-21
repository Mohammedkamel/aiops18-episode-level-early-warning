# AIOps18 Preprocessing — Methods and Provenance

This document describes the preprocessing pipeline for AIOps18 (NetManAIOps
KPI-Anomaly-Detection), covering the merge, gap handling, temporal split, normalization,
and sliding-window construction that produced the arrays in `processed_data/aiops18/`. It
follows the same style as `docs/aiops2022_detectability_methodology.md` — detailed enough
to lift into a paper's Methods section, precise enough to reproduce exactly.

## 1. Dataset scope

Downloaded via `git clone https://github.com/NetManAIOps/KPI-Anomaly-Detection.git` into
`data/raw/aiops18/`. Two labeled subsets, verified against expected line counts before use:

| File | Rows (excl. header) | KPIs |
|---|---|---|
| `Preliminary_dataset/train.csv` | 2,476,315 | 26 |
| `Finals_dataset/phase2_train.csv` | 3,004,066 | 29 |

(`wc -l` on `phase2_train.csv` reports 3,004,067 — one more than the nominal 3,004,066 — a
trailing-newline artifact in how `wc -l` counts, not a data issue.)

**Merged**: 5,480,381 rows, 55 unique KPI IDs, **zero ID overlap** between Preliminary and
Finals (confirmed by set intersection) — 55 genuinely independent series, as expected.

### `phase2_ground_truth.hdf.zip` — confirmed paired with Finals train, not used in this pipeline

The Finals dataset folder also ships `phase2_ground_truth.hdf.zip` (100MB, a pandas/pytables
HDF5 store with the same `timestamp, value, label, KPI ID` schema). It was extracted and
inspected before deciding whether to use it:

- 2,918,847 rows, 29 unique KPI IDs (same count as Finals train, not Preliminary)
- Timestamp range 1471044900–1509383340 (~Aug 2016–Oct 2017) vs. `phase2_train.csv`'s
  1467302400–1502342760 (~Jun 2016–Aug 2017) — a later, shifted window, structurally
  consistent with a subsequent held-out test period
- Anomaly rate 1.87% (54,560/2,918,847) vs. Finals train's 2.65% — same order of magnitude

An initial check found zero KPI ID overlap with `phase2_train.csv` and concluded this file
might be an orphaned/unrelated artifact. **That check was wrong**: the HDF store's `KPI ID`
column is loaded as Python `uuid.UUID` objects (a pytables storage quirk), not strings, so a
naive set intersection against the CSV's string IDs never matches even for identical values.
After normalizing both to strings, **all 29 KPI IDs match exactly** between the two files,
zero mismatches either direction. Combined with the later/shifted timestamp range and
comparable anomaly rate, this is conclusive: `phase2_ground_truth.hdf` **is the ground truth
for the same 29 Finals series over a later (held-out) time period** — almost certainly the
labels for the competition's hidden test window, released after judging. It is a genuinely
paired artifact, not an orphaned one.

It is still **not incorporated into the outputs described below** — the pipeline as run
covers only the 55-series Preliminary+Finals training merge (Section 1 above), and folding
in a 29-series later-time held-out set would be a deliberate scope extension (e.g. as a
second, independent test period) rather than a preprocessing detail, so it's left as a
flagged opportunity rather than silently added. Extracted at
`data/raw/aiops18/Finals_dataset/phase2_ground_truth.hdf`.

## 2. Gap detection and handling

Per KPI, the modal sampling interval was computed from the mode of consecutive timestamp
differences. **41 of 55 KPIs run at 60s resolution, 14 at 300s** — both are represented in
the merged set, no KPI has any other resolution.

### The literal "any deviation is a gap" reading was tested and rejected

A first pass treated any interval deviation from the modal value as a gap that no window may
span. This is enormously costly for a subset of KPIs: **19 of 55 KPIs lost over 90% of their
naive window yield**, and 23 lost over half. Inspecting the actual gap sizes on the
worst-affected KPIs showed **79–96% of all gap transitions were exactly one missed sample**
(interval = 2× modal — a single dropped 60s reading, not a real outage); genuine multi-sample
outages exist (the largest observed gap was ~22 hours) but are rare by comparison.

**Resolution adopted**: single-sample gaps (interval == 2× modal interval) are linearly
interpolated rather than treated as breaks:
- interpolated **value** = midpoint linear interpolation between the two surrounding samples
- interpolated **label** = 1 only if *both* surrounding samples are labeled 1, else 0 — this
  preserves segment continuity (a single dropped sample in the middle of a genuine anomaly
  doesn't spuriously split one segment into two) without inventing an anomaly where neither
  neighbor was anomalous

Only gaps of **3× modal interval or larger** (2+ consecutive missed samples) are treated as
hard breaks that no sliding window may span. This reduced the worst-case yield loss from
>92% to 25%, and **zero KPIs now lose more than 50%** of naive window yield. 61,893
single-sample points were interpolated in total across all 55 KPIs (exact per-KPI counts in
the manifest, `n_single_sample_gaps_interpolated`).

## 3. Temporal split

**60/20/20, chronological by row count within each KPI** (not random), matching Maat's
convention for direct comparability. Verified per-KPI: no KPI has fewer than 100 normal
training points, and no KPI's training-normal standard deviation is zero or NaN (both would
have indicated an unusable split).

**Split boundaries are treated as additional hard breaks, on top of real gaps** — no sliding
window may span from one split into another, exactly as a window may not span a data gap.
This is necessary to prevent leakage: a window whose context ends in the train portion but
whose forecast horizon dips into the val portion would let training implicitly see
validation-period information.

**Known limitation, flagged rather than silently accepted**: because anomaly segments are
sparse and chronologically clustered (a burst of related incidents can fall entirely within
one split), a strict per-KPI temporal split leaves **13 of 55 KPIs with zero anomalous points
in at least one of val/test**: 8 have zero in val, 7 have zero in test, and **2 of those have
zero anomalies in *both*** (`046ec29ddf80d62e`, `9bd90500bfd11edb`). This is an inherent
consequence of combining sparse, bursty anomaly occurrence with a strict chronological
(non-random, non-stratified) split, not a bug in the split logic.

### Explicit rule for zero-anomaly KPI x split cells

Rather than leave the handling of these 13 KPIs implicit, downstream evaluation must follow
one rule, consistently applied per metric family:

| Metric family | Needs anomaly labels? | Rule for a KPI x split with zero anomalous points |
|---|---|---|
| Onset-based (detection rate, recall, lead time) | Yes — needs actual onset events to measure whether they were caught | **Excluded.** The split contributes neither to the numerator nor the denominator for that KPI — it is not scored as 0% (which would misrepresent "nothing to detect" as "failed to detect"), and it is not silently dropped from other metrics. |
| Aggregate forecasting accuracy (MAE, RMSE on `horizon_value`) | No — compares predicted vs. actual values regardless of label | **Retained.** These metrics need no anomaly labels and are computed normally on every window. |
| False-positive-rate / precision | No — only needs negative (label=0) windows and the model's predictions on them | **Retained.** A KPI x split with zero true anomalies still has plenty of normal windows, and any false alarm the model raises on them is still valid signal. |

This is encoded machine-readably in the manifest, not just as prose here: every KPI entry in
`aiops18_manifest.json` carries `train_anomaly_points`, `val_anomaly_points`,
`test_anomaly_points`, and `onset_metrics_excluded_splits` (a list, e.g. `["val", "test"]`
for the two fully-affected KPIs, `["val"]` or `["test"]` for the other 11). The manifest also
has top-level `kpis_excluded_from_val_onset_metrics` and
`kpis_excluded_from_test_onset_metrics` lists for convenience, plus a
`zero_anomaly_split_policy` field stating this same rule in the manifest itself so it travels
with the data even without this document.

## 4. Normalization

Z-score per KPI: mean and standard deviation are fit on the **training split's label==0 rows
only**, then applied to all three splits' values (`value_z = (value - train_mean) /
train_std`). Stats are saved per KPI in the manifest (`train_mean`, `train_std`) — needed to
un-normalize predictions later.

Verified post-hoc on the assembled windows: restricting to context points labeled 0 gives
mean ≈ -0.019, std ≈ 1.004 (correct, as expected for a normalization fit on exactly that
subset). Points labeled 1 show mean ≈ 2.47, std ≈ 48.9 — anomalies are genuinely extreme
relative to the normal baseline, which is why the *unconditional* mean/std across all context
points (≈0.015 / ≈5.85) looks inflated; that's the anomalous points pulling the aggregate, not
a normalization error.

### Post-normalization clipping (±10σ)

Added after diagnosing a Phase 2 (evaluation-methodology) training failure: PatchTST's
training loss stayed pinned near 31.0 across 15 epochs (31.74 → 31.08, a 2.1% reduction, not
convergence) while LSTM trained normally on the same data. Traced to genuinely unbounded
outliers in `value_z`, not a shape or feed-order bug — verified directly on the windowed
arrays: **0.35% of training windows contain a value with |z| > 100**, and the single global
max was **z = 2540.68** (KPI `57051487-...`). These are concentrated in just 6 of 41 KPIs,
almost certainly ones whose training-normal standard deviation is very small, so an ordinary
raw-value fluctuation gets amplified into an enormous apparent z-score by the per-KPI
normalization — correct in aggregate, but exactly the failure mode small-std normalization is
prone to.

**Mechanism**: under MSE loss, a single z≈2540 target contributes squared error on the order
of millions versus ~1–10 for a normal point. With ~3.2% of windows containing at least one
|z|>10 value, most 256-window training batches contain at least one such point — its loss and
gradient can dominate the entire batch's update. PatchTST's architecture has no
LayerNorm/clipping between the raw input and its first attention computation (`patch_embed`
feeds straight into `TransformerEncoderLayer`, which only normalizes *after* attention has
already seen the raw-scale embedding), so it has no structural defense against this. LSTM's
sigmoid/tanh gates plausibly provide some inherent robustness by squashing extreme values as
they enter the hidden state, which is consistent with LSTM training normally on the same
outlier-contaminated data while PatchTST did not.

**Fix**: `value_z` is clipped to **±10** immediately after normalization
(`Z_CLIP = 10.0` in `aiops18_preprocessing.py`), before windowing. Rationale for the
threshold: a value 10 standard deviations from the training-normal mean is already extreme
enough for any model to treat it as anomalous — clipping further doesn't remove meaningful
signal (nothing distinguishes "10σ" from "2540σ" in terms of what a forecaster should learn
to do about it), it only prevents unbounded outliers from dominating the loss. Applied at the
normalization stage, upstream of all consumers (the classical models' raw per-KPI series via
`process_kpi`, and the windowed arrays every model reads from) — not a PatchTST-specific
patch. 0.31% of context points sit at the clip boundary post-fix, consistent with the
pre-fix outlier prevalence measured above. Window counts are unaffected (window construction
depends on gaps/splits, not values), but every KPI's `train_mean`/`train_std` and every
window's stored values changed; **processed_data/aiops18/ was regenerated and the prior
version preserved at `processed_data/aiops18/v2_before_z_clip/`.**

## 5. Sliding window construction

- **Context length**: 288 steps at 60s resolution (4.8 hours), scaled proportionally to 58
  steps at 300s resolution (288 × 60/300 = 57.6, rounded to 58, ≈4.83 hours — the closest
  achievable match to "4.8 hours" at this resolution).
- **Forecast horizons**: 15 minutes and 60 minutes, both produced (15/60 steps at 60s
  resolution; 3/12 steps at 300s resolution) — two separate sets of window files rather than
  one, per the "multiple horizons if feasible" instruction.
- **Stride**: 5 minutes of wall-clock time (5 steps at 60s resolution, 1 step at 300s
  resolution). This was **not specified** in the original task and was chosen deliberately:
  stride=1 (every step) would have produced windows totaling an estimated ~12.8GB per horizon
  (~25.5GB for both horizons combined) against 21–23GB of free disk at the time — too tight.
  A 5-minute stride keeps meaningful window diversity while landing at a safe, much smaller
  footprint (below, "8. Outputs").
- **60s-resolution and 300s-resolution KPIs are written to separate files**, not merged into
  one array — their context length differs (288 vs. 58 steps), so they cannot share a single
  fixed-width array. Any pipeline consuming these outputs needs to either train separate
  models per resolution or handle the two context lengths explicitly.
- **No window spans a hard gap or a split boundary** (Sections 2–3).

## 6. Onset labeling and segment-count reconciliation

Onset = the first sample of a contiguous `label==1` segment (a 0→1 transition). Per-window,
`onset_in_horizon` = 1 if any point within the forecast horizon is such a transition, else 0
— this is the label most directly useful for early-warning evaluation (did a new anomaly
begin within the lookahead window), and is provided alongside the raw `horizon_label`
sequence so other labeling conventions (e.g. "any anomalous point in horizon") can be derived
downstream without re-deriving from the raw data.

**Segment count reconciliation**: an initial full count on the merged 55-KPI set gave 2,161
segments, not the ~933 expected from earlier exploratory work. Rather than assume either
number was wrong, the discrepancy was traced directly: **933 is the exact segment count for
Preliminary alone (26 KPIs)**; **Finals alone (29 KPIs) has 1,228**; **933 + 1,228 = 2,161**,
exactly matching the merged-set count. The earlier "~933" figure was scoped to Preliminary
only and was never meant to describe the full 55-KPI set used here — there is no actual
inconsistency, just a difference in scope between the two figures. **2,161 is the confirmed,
correct total for the merged dataset used in this pipeline.**

## 7. Outputs

All under `processed_data/aiops18/`, 78MB total:

| Pattern | Count | Description |
|---|---|---|
| `aiops18_{split}_windows_h{15,60}min_{60,300}s.npz` | 12 files (3 splits × 2 horizons × 2 resolutions) | Windowed arrays |
| `aiops18_manifest.json` | 1 | Per-KPI split sizes, gap locations, normalization stats, window-construction parameters |

Each `.npz` contains: `context` (N × context_steps, float32, z-scored), `context_label` (N ×
context_steps, int8, raw 0/1), `horizon_value` (N × horizon_steps, float32, z-scored),
`horizon_label` (N × horizon_steps, int8), `onset_in_horizon` (N, int8), `window_start_ts`
(N, int64, first context timestamp), `kpi_id` (N, object/string).

Window counts:

| Split | h=15min, 60s-res | h=15min, 300s-res | h=60min, 60s-res | h=60min, 300s-res |
|---|---|---|---|---|
| train | 578,235 | 73,538 | 569,229 | 73,160 |
| val | 179,684 | 24,143 | 174,866 | 23,954 |
| test | 170,562 | 23,631 | 165,148 | 23,361 |

`onset_in_horizon` positive rate (train, 60s-resolution): 0.56% at 15-minute horizon, 2.08%
at 60-minute horizon — increasing with horizon length as expected (more time for a new onset
to occur within the lookahead window).

**Scripts**: `scripts/preprocessing/aiops18_preprocessing.py` (the entire pipeline above, one
script, run once end-to-end).

## 8. Known limitations

- **`phase2_ground_truth.hdf.zip` is confirmed as the same 29 Finals series over a later
  held-out period (Section 1) but is not incorporated into any output here.** Using it as a
  second, independent test period is a real opportunity for a stronger held-out evaluation
  later — revisit deliberately rather than leaving it unused indefinitely now that its
  identity is resolved.
- **13 of 55 KPIs have zero anomalies in at least one of val/test** (2 in both). This is
  handled by an explicit, machine-readable rule rather than left implicit — see Section 3's
  "Explicit rule for zero-anomaly KPI x split cells" and the manifest's
  `onset_metrics_excluded_splits` field per KPI.
- **Stride (5 minutes) was a disk-budget-driven choice, not a methodologically derived one.**
  A denser stride (more overlapping windows) was ruled out primarily because of disk space at
  the time of this run, not because 5 minutes is known to be the right granularity for this
  forecasting task. If disk constraints ease, revisiting with a denser stride (or generating
  windows on-the-fly during training rather than as pre-materialized arrays) would give more
  training diversity.
- **60s- and 300s-resolution KPIs require separate handling downstream** (Section 5) — any
  model architecture spanning all 55 series needs to either accept variable context length or
  be trained/evaluated per resolution group separately.
- **Single-sample-gap interpolation (Section 2) synthesizes both a value and a label that
  were never actually observed.** This was deliberately scoped to the narrowest case (exactly
  one missed sample, in the middle of an otherwise-regular series) precisely to keep the
  synthesized fraction small (61,893 points, a small fraction of the ~5.5M-row merged
  dataset) — but it does mean a small number of "observations" in the training data are
  interpolated rather than real, which should be disclosed if this dataset's provenance is
  discussed in the paper.
