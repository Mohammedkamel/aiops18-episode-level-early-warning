# Figure data and reconstruction notes

Five CSV files, one per figure, all pulled from values already verified in the paper draft and
the production pipeline (not re-derived or estimated for this export).

## Figure 1 — Forecast accuracy vs. degradation-onset detection
File: `fig1_accuracy_vs_detection.csv`

- Scatter plot: x = MAE, y = detection rate (%).
- **Design constraint: must remain interpretable in grayscale/print.** Each model has both a
  unique marker *shape* (not just color) and validation vs. test is distinguished by *fill*
  (filled = validation, hollow = test), not color or shape. Verify any reconstruction by
  converting to grayscale and confirming all 6 models are still distinguishable.
- Marker shapes used: Persistence = circle, SARIMA = square, XGBoost = triangle-up,
  LSTM = diamond, PatchTST = plus (filled cross), Chronos = star.
- Chronos marker was drawn larger (star shapes read smaller at equal point-size than filled
  polygons) — roughly 1.7x the area of the other markers.
- Each model's name is labeled directly next to its validation-split point in the original.

## Figure 2 — PatchTST training loss, before vs. after clipping
File: `fig2_patchtst_clipping.csv`

- Two side-by-side panels (not one overlaid dual-axis plot), since the two loss scales differ
  by ~150x (≈31 vs. ≈0.2) and overlaying them on one axis would be misleading.
- Left panel y-range: roughly 30.5–32.0. Right panel y-range: roughly 0.18–0.26.
- Both panels share the same x-axis (epoch, 1–15).

## Figure 3 — Correction layer effect on episode-level F1
File: `fig3_correction_layer_f1.csv`

- Dumbbell / slope chart: one horizontal line per model per split, connecting the raw F1 value
  (hollow marker) to the corrected F1 value (filled marker), same shape convention as Figure 1
  (filled = corrected/"after", hollow = raw/"before" — inverted intent from Fig. 1's val/test
  meaning, so don't reuse the same legend wording across figures without checking).
  ⚠ Note: in the original, marker *shape* was uniform (circle) for this figure since the
  6 models are already separated by row position and color; only fill state carries meaning here.
- Two panels: validation (left), test (right), sharing the same x-axis scale (0 to ~0.135) for
  direct visual comparability between splits.
- Row order (top to bottom in the original): Chronos, PatchTST, LSTM, XGBoost, SARIMA,
  Persistence.
- Precision/recall columns are included in the CSV for reference/annotation but were not
  separately plotted in the original F1-only version.

## Figure 4 — Episode-level window-overlap schematic
File: `fig4_episode_overlap_schematic.csv`

- **This is a conceptual diagram, not real per-KPI data** — it illustrates the general
  windowing/stride mechanism (5-minute stride, 15-minute horizon → 3 overlapping windows per
  onset), not one specific observed episode.
- Rendered as three horizontal rounded bars (one per window), each spanning its horizon
  [horizon_start, horizon_end], with a vertical dashed red line marking the onset at t = 0.
- An arrow + boxed annotation on the right states the collapse rule in plain text: "Detected if
  ANY window flags the onset."
- Colors used per window: Window A = #1f77b4 (blue), Window B = #2ca02c (green),
  Window C = #9467bd (purple) — arbitrary, chosen only for visual separation.

## Figure 5 — Correction-layer mechanism on a verified real detection
File: `fig5_correction_mechanism.csv`

- **All values are real, verified production-pipeline output** (via direct inspection of
  `correction_layer.py` and `evaluation_metrics.py`, cross-checked against
  `compute_detection_metrics()`), not illustrative or estimated — this is the one figure where
  data fidelity was specifically audited after an earlier draft used an incorrect timestamp
  convention (see the paper's spot-check discussion for the full story of that correction).
- Single panel. X-axis: time relative to onset, in minutes (NOT relative to change-point
  detection — an earlier draft used that axis and it was misleading; use time-relative-to-onset).
- Two horizontal reference lines: static threshold (flat, constant at 1.6013) and adaptive
  threshold (step function: drops to 1.2810 at t = −10, steps up slightly to 1.3024 at t = 0).
- Three real data points (red) plotted at their actual (time, predicted |z|) values — see CSV.
- Shaded vertical band from t = −10 to the right edge marks the period during which the
  adaptive threshold is below the static threshold ("correction active").
- Two short text annotations with arrows: "Missed by the raw forecast (below the static
  threshold)" pointing at the gap between the points and the static line; "Caught by the
  correction layer (above the adaptive threshold)" pointing at the gap between the points and
  the adaptive line.
- Caption/subtitle states the lead time (10 minutes) and the residual values that triggered
  the change-point detection (0.001–0.007 baseline → 0.231–0.234 held), since these drive the
  mechanism but aren't directly plotted as a second panel in the final version (an earlier
  two-panel draft with a separate residual-stream subplot was replaced with this simpler
  single-panel version after it proved confusing).

## Shared style notes (all figures)
- Font family: serif (matplotlib default serif, e.g. DejaVu Serif / Times-like).
- No top/right axis spines (open-box style).
- Light gridlines only, low alpha (~0.35–0.4).
- Figure export: vector PDF as primary format, PNG (150 dpi) for quick preview.
