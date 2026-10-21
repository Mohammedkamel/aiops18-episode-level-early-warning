"""Dataset-agnostic episode-level evaluation metrics, implementing exactly the rules locked
in docs/aiops18_evaluation_methodology.md: existential-OR detection across an onset's
overlapping windows, overlapping-interval merging for false-alarm episodes, lead time from
the earliest correctly-flagging window. Plus standard aggregate MAE/RMSE.

Nothing here is specific to AIOps18 or aiops2022, or to any particular resolution/horizon --
every function takes plain arrays (predictions, ground truth, timestamps) and a kpi_id array
to scope episode grouping to one series at a time. Works identically for both datasets and
any split/horizon/resolution combination.
"""

from dataclasses import dataclass, field

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE/RMSE over any array of predicted vs. actual continuous values (e.g. horizon_value)."""
    y_true, y_pred = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    return dict(mae=float(np.abs(err).mean()), rmse=float(np.sqrt((err**2).mean())), n=len(y_true))


def merge_overlapping_intervals(starts: np.ndarray, ends: np.ndarray) -> list[dict]:
    """Standard overlapping-interval merge. starts/ends are wall-clock timestamps (or steps),
    ends exclusive. Returns one dict per merged group: {start, end, member_indices} where
    member_indices are positions into the original starts/ends arrays, sorted by start.
    """
    starts, ends = np.asarray(starts), np.asarray(ends)
    if len(starts) == 0:
        return []
    order = np.argsort(starts, kind="stable")
    groups = []
    cur_start, cur_end = starts[order[0]], ends[order[0]]
    cur_members = [int(order[0])]
    for idx in order[1:]:
        s, e = starts[idx], ends[idx]
        if s <= cur_end:  # overlaps (or touches) the current merged interval
            cur_end = max(cur_end, e)
            cur_members.append(int(idx))
        else:
            groups.append(dict(start=cur_start, end=cur_end, member_indices=cur_members))
            cur_start, cur_end, cur_members = s, e, [int(idx)]
    groups.append(dict(start=cur_start, end=cur_end, member_indices=cur_members))
    return groups


def extract_true_onset_timestamps(
    window_start_ts: np.ndarray,
    context_steps: int,
    horizon_steps: int,
    modal_interval: int,
    context_label_last: np.ndarray,
    horizon_label: np.ndarray,
) -> np.ndarray:
    """Per window, the absolute timestamp of the onset in its horizon, or NaN if none.

    An onset is the first 0->1 transition; context_label_last (the window's last context-step
    label) is needed to correctly detect a transition landing on the horizon's first step.
    horizon_label is (N, horizon_steps); only the first onset per window is used, matching the
    dataset's segment definition (a window's horizon should contain at most one onset for any
    horizon short enough not to span two full segments back-to-back -- this holds for the
    15/60-minute horizons used here given typical segment spacing, but is not separately
    re-verified inside this function).
    """
    n = horizon_label.shape[0]
    full = np.concatenate([context_label_last.reshape(n, 1), horizon_label], axis=1)
    is_onset = (full[:, 1:] == 1) & (full[:, :-1] == 0)  # (n, horizon_steps)
    out = np.full(n, np.nan)
    for i in range(n):
        pos = np.argmax(is_onset[i]) if is_onset[i].any() else None
        if pos is not None:
            out[i] = window_start_ts[i] + (context_steps + pos) * modal_interval
    return out


@dataclass
class DetectionResult:
    n_episodes: int
    n_detected: int
    detection_rate: float
    missed_degradation_rate: float
    lead_times_seconds: list[float] = field(default_factory=list)
    lead_time_median_seconds: float | None = None
    lead_time_iqr_seconds: tuple[float, float] | None = None
    episodes: list[dict] = field(default_factory=list)  # per-episode detail, for spot-checking


def compute_detection_metrics(
    kpi_id: np.ndarray,
    window_start_ts: np.ndarray,
    context_steps: int,
    horizon_steps: int,
    modal_interval: int,
    true_onset_timestamp: np.ndarray,  # NaN if this window's horizon has no onset
    predicted_positive: np.ndarray,  # bool: did the model flag an onset in this window's horizon
) -> DetectionResult:
    """Episode-level detection rate and lead time (docs/aiops18_evaluation_methodology.md).

    Episodes are grouped by (kpi_id, exact true_onset_timestamp) -- NOT by transitive interval
    overlap -- so two distinct onsets close enough for their windows to chain together are
    never merged into one episode. Windows genuinely overlapping the same onset always share
    the identical onset timestamp by construction, so this grouping is exact, not approximate.
    """
    context_end_ts = window_start_ts + context_steps * modal_interval
    has_onset = ~np.isnan(true_onset_timestamp)

    episodes = []
    for kid in np.unique(kpi_id[has_onset]):
        kmask = has_onset & (kpi_id == kid)
        for onset_ts in np.unique(true_onset_timestamp[kmask]):
            member_mask = kmask & (true_onset_timestamp == onset_ts)
            member_idx = np.where(member_mask)[0]
            flagged_idx = member_idx[predicted_positive[member_idx].astype(bool)]
            detected = len(flagged_idx) > 0
            lead_time = None
            if detected:
                earliest = flagged_idx[np.argmin(context_end_ts[flagged_idx])]
                lead_time = float(onset_ts - context_end_ts[earliest])
            episodes.append(
                dict(
                    kpi_id=kid,
                    onset_timestamp=float(onset_ts),
                    n_overlapping_windows=len(member_idx),
                    n_flagged_windows=len(flagged_idx),
                    detected=detected,
                    lead_time_seconds=lead_time,
                )
            )

    n_episodes = len(episodes)
    n_detected = sum(e["detected"] for e in episodes)
    detection_rate = n_detected / n_episodes if n_episodes else float("nan")
    lead_times = [e["lead_time_seconds"] for e in episodes if e["detected"]]

    lead_median, lead_iqr = None, None
    if lead_times:
        lead_median = float(np.median(lead_times))
        lead_iqr = (float(np.percentile(lead_times, 25)), float(np.percentile(lead_times, 75)))

    return DetectionResult(
        n_episodes=n_episodes,
        n_detected=n_detected,
        detection_rate=detection_rate,
        missed_degradation_rate=(1 - detection_rate) if n_episodes else float("nan"),
        lead_times_seconds=lead_times,
        lead_time_median_seconds=lead_median,
        lead_time_iqr_seconds=lead_iqr,
        episodes=episodes,
    )


@dataclass
class FalseAlarmResult:
    n_false_positive_windows: int
    n_false_alarm_episodes: int
    episodes: list[dict] = field(default_factory=list)


def compute_false_alarm_metrics(
    kpi_id: np.ndarray,
    window_start_ts: np.ndarray,
    context_steps: int,
    horizon_steps: int,
    modal_interval: int,
    true_onset_in_horizon: np.ndarray,  # bool: any true onset anywhere in this window's horizon
    predicted_positive: np.ndarray,  # bool
) -> FalseAlarmResult:
    """False-positive windows (flagged but no true onset in horizon) collapsed into false-alarm
    episodes via overlapping-interval merge on their horizon intervals, per KPI.
    """
    horizon_start = window_start_ts + context_steps * modal_interval
    horizon_end = horizon_start + horizon_steps * modal_interval
    is_fp = predicted_positive.astype(bool) & ~true_onset_in_horizon.astype(bool)

    episodes = []
    for kid in np.unique(kpi_id[is_fp]):
        kmask = is_fp & (kpi_id == kid)
        idx = np.where(kmask)[0]
        merged = merge_overlapping_intervals(horizon_start[idx], horizon_end[idx])
        for g in merged:
            episodes.append(
                dict(
                    kpi_id=kid,
                    start=float(g["start"]),
                    end=float(g["end"]),
                    n_member_windows=len(g["member_indices"]),
                )
            )

    return FalseAlarmResult(
        n_false_positive_windows=int(is_fp.sum()),
        n_false_alarm_episodes=len(episodes),
        episodes=episodes,
    )
