"""Model-agnostic post-hoc correction layer: sits between a model's already-saved raw
forecast and the flagging decision. Operates purely on saved predictions (kpi_id,
window_start_ts, y_true, y_pred) -- no retraining, no access to raw data beyond what's
already in the prediction .npz files.

Three chosen-but-not-tuned parameters (documented, not fit against val/test -- consistent
with how stride and the z-clip bound were handled: a principled first pass, stated as such):
  ROLLING_WINDOW = 30    (trailing windows used to estimate each KPI's local residual baseline)
  POST_CP_WINDOW = 15    (windows after a flagged change-point during which the threshold is lowered)
  THRESHOLD_REDUCTION = 0.20  (20% lower immediately after a change-point, linearly decaying to 0)

Caveat for models evaluated on a stratified subsample (Chronos): "30 trailing windows" is a
window-INDEX count, not a wall-clock duration. For the five models evaluated on the complete,
contiguous window set, 30 windows = 150 minutes of real history. For Chronos, whose per-KPI
windows are a sparse stratified sample, 30 included windows can span many hours or days of
wall-clock time. The correction layer runs identically on both (same code, same parameters),
but its behavior is not temporally equivalent across the two cases -- flagged explicitly here
and in the methodology doc rather than presented as if it were.

Causal-maturation ordering (required for correctness, not optional): window j's residual is
computed from y_true[j], the REALIZED horizon -- values that are only knowable once window
j's own horizon has actually played out, at matured_ts[j] = context_end_ts[j] +
horizon_steps*modal_interval. Window j's own detection decision, however, is made earlier, at
context_end_ts[j] (when the forecast is issued, before the horizon has played out). A change-
point flagged AT window j must therefore never inform window j's own threshold (nor any other
window whose decision time precedes matured_ts[j]) -- doing so lets a window's own future
outcome lower its own detection threshold. `adaptive_threshold_schedule` enforces this by only
applying a change-point's suppression starting at the first window whose context_end_ts is >=
that change-point's matured_ts, found via np.searchsorted on the per-KPI context_end_ts array
(robust to irregular stride/gaps, no assumption of uniform spacing). Since
matured_ts[j] > context_end_ts[j] always (horizon_steps > 0), this structurally excludes a
change-point from ever suppressing its own triggering window, and typically also excludes the
next one or two windows immediately after it (whichever windows' own decision times still fall
before the change-point has matured).
"""

import numpy as np

ROLLING_WINDOW = 30
POST_CP_WINDOW = 15
THRESHOLD_REDUCTION = 0.20
CUSUM_SLACK = 0.5
CUSUM_THRESHOLD = 4.0
MIN_BASELINE_POINTS = 5


def compute_window_residuals(y_true_flat: np.ndarray, y_pred_flat: np.ndarray, horizon_steps: int) -> np.ndarray:
    """One scalar residual per window: mean absolute error across its horizon."""
    y_true_2d = y_true_flat.reshape(-1, horizon_steps)
    y_pred_2d = y_pred_flat.reshape(-1, horizon_steps)
    return np.mean(np.abs(y_true_2d - y_pred_2d), axis=1)


def cusum_changepoints(
    residuals_ordered: np.ndarray,
    rolling_window: int = ROLLING_WINDOW,
    k_slack: float = CUSUM_SLACK,
    h_threshold: float = CUSUM_THRESHOLD,
) -> np.ndarray:
    """One-sided (upward-shift) Page's CUSUM on a time-ordered per-KPI residual stream.
    Detects a SUSTAINED rise in residual level relative to a trailing rolling baseline, not
    a single-point spike -- the baseline (mean, std) at position i uses only residuals
    strictly before i (no lookahead). Resets the cumulative statistic to 0 immediately after
    flagging a change-point, standard CUSUM practice, so a single regime shift is flagged
    once, not on every subsequent step while still elevated.
    """
    n = len(residuals_ordered)
    is_cp = np.zeros(n, dtype=bool)
    cusum = 0.0
    for i in range(n):
        lo = max(0, i - rolling_window)
        baseline = residuals_ordered[lo:i]
        if len(baseline) < MIN_BASELINE_POINTS:
            continue
        mu, sigma = baseline.mean(), baseline.std()
        sigma_eff = max(sigma, 1e-6)
        z = (residuals_ordered[i] - mu) / sigma_eff
        cusum = max(0.0, cusum + z - k_slack)
        if cusum > h_threshold:
            is_cp[i] = True
            cusum = 0.0
    return is_cp


def adaptive_threshold_schedule(
    context_end_ts_ordered: np.ndarray,
    changepoint_flags: np.ndarray,
    base_threshold: float,
    horizon_steps: int,
    modal_interval: int,
    post_cp_window: int = POST_CP_WINDOW,
    reduction_factor: float = THRESHOLD_REDUCTION,
) -> np.ndarray:
    """base_threshold at baseline; starting at the first window whose OWN decision time
    (context_end_ts) is >= the change-point's matured_ts (context_end_ts[cp] +
    horizon_steps*modal_interval -- when the change-point's triggering residual actually
    became knowable), the threshold drops to base_threshold*(1-reduction_factor), then decays
    linearly back to base_threshold over post_cp_window steps counted from that first usable
    window (NOT from the change-point's own index). This is what prevents a window's own
    future-realized residual from lowering its own (or an immediately-adjacent window's)
    detection threshold -- see module docstring. Overlapping schedules take the more
    sensitive (lower) threshold at each step.

    context_end_ts_ordered must be sorted ascending (same per-KPI time order used to build
    changepoint_flags).
    """
    n = len(context_end_ts_ordered)
    thresholds = np.full(n, base_threshold, dtype=np.float64)
    horizon_span = horizon_steps * modal_interval
    for cp_idx in np.where(changepoint_flags)[0]:
        matured_ts = context_end_ts_ordered[cp_idx] + horizon_span
        q = int(np.searchsorted(context_end_ts_ordered, matured_ts, side="left"))
        for offset in range(post_cp_window):
            idx = q + offset
            if idx >= n:
                break
            frac = offset / post_cp_window
            adjusted = base_threshold * (1 - reduction_factor) + base_threshold * reduction_factor * frac
            thresholds[idx] = min(thresholds[idx], adjusted)
    return thresholds


def apply_correction_layer(
    kpi_id: np.ndarray,
    window_start_ts: np.ndarray,
    y_true_flat: np.ndarray,
    y_pred_flat: np.ndarray,
    horizon_steps: int,
    context_steps: int,
    modal_interval: int,
    base_thresholds: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (corrected_predicted_positive, changepoint_flags), both aligned to the input
    arrays' original order (not the per-KPI time-sorted order used internally).
    """
    y_pred_2d = y_pred_flat.reshape(-1, horizon_steps)
    n = len(y_pred_2d)
    residuals = compute_window_residuals(y_true_flat, y_pred_flat, horizon_steps)
    context_end_ts = window_start_ts + context_steps * modal_interval

    corrected_positive = np.zeros(n, dtype=bool)
    changepoint_flags_out = np.zeros(n, dtype=bool)

    for kpi in np.unique(kpi_id):
        idx = np.where(kpi_id == kpi)[0]
        order = np.argsort(window_start_ts[idx])
        ordered_idx = idx[order]

        cps = cusum_changepoints(residuals[ordered_idx])
        thresholds = adaptive_threshold_schedule(
            context_end_ts[ordered_idx], cps, base_thresholds[kpi], horizon_steps, modal_interval
        )

        for j, global_idx in enumerate(ordered_idx):
            corrected_positive[global_idx] = bool(np.any(np.abs(y_pred_2d[global_idx]) > thresholds[j]))
        changepoint_flags_out[ordered_idx] = cps

    return corrected_positive, changepoint_flags_out
