"""Per-KPI detection threshold calibration, replacing the miscalibrated fixed 3-sigma default
in model_interface.flag_onset_from_forecast (see completion-gate investigation: for at least
one KPI, only 1% of true anomalous points ever reach |z|>3, making 3-sigma an unusable
operating point for that KPI regardless of forecast quality).

Calibration rule: sweep candidate |z| thresholds against the TRAINING split's actual observed
(value_z, label) pairs -- not model forecasts -- and pick the threshold maximizing F1 between
"|value_z| > threshold" and the true label. This is a property of the KPI's own label/magnitude
relationship, computed once, fixed, and reused unchanged for val and test -- never re-tuned per
split, to avoid leakage from evaluation data into the operating threshold.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class ThresholdCalibration:
    kpi_id: str
    threshold: float
    precision: float
    recall: float
    f1: float
    low_precision_flag: bool  # True if precision < 0.30 even at the F1-optimal threshold


def calibrate_threshold_f1(kpi_id: str, values_z: np.ndarray, labels: np.ndarray, n_candidates: int = 200) -> ThresholdCalibration:
    """values_z, labels: TRAINING split only, one entry per raw timestep (not per window)."""
    abs_z = np.abs(values_z)
    labels = labels.astype(bool)
    candidates = np.unique(np.quantile(abs_z, np.linspace(0.0, 0.999, n_candidates)))

    best = None
    for tau in candidates:
        pred_pos = abs_z > tau
        tp = int(np.sum(pred_pos & labels))
        fp = int(np.sum(pred_pos & ~labels))
        fn = int(np.sum(~pred_pos & labels))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        if best is None or f1 > best["f1"]:
            best = dict(threshold=float(tau), precision=precision, recall=recall, f1=f1)

    return ThresholdCalibration(
        kpi_id=kpi_id,
        threshold=best["threshold"],
        precision=best["precision"],
        recall=best["recall"],
        f1=best["f1"],
        low_precision_flag=best["precision"] < 0.30,
    )
