"""Common model interface so the training/evaluation loop is model-agnostic across classical
(SARIMA), gradient-boosted (XGBoost), and future deep-learning/foundation models.

Two fit patterns exist and both must fit through this same interface:
  - per_series: one model instance fit per KPI/service (SARIMA, XGBoost -- standard practice
    for these model classes; used for Phase 1).
  - shared: one model instance fit once across every series in the dataset, relying on the
    existing per-KPI z-score normalization so scale differences don't confound a shared
    representation (LSTM, PatchTST -- Phase 2, NOT implemented yet; this interface just needs
    to accommodate it later without a redesign).

A model's fit_mode attribute tells the training harness which loop to use: loop over KPIs and
fit one instance each (per_series), or fit a single instance on the concatenated data of all
KPIs at once (shared).

predict() returns the continuous forecast for the horizon (not a binary onset flag) -- see
flag_onset_from_forecast() below for how a forecast becomes a detection input for
evaluation_metrics.py. Keeping models as pure forecasters (not classifiers) keeps every model
class comparable through the same MAE/RMSE and the same downstream flagging rule.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class TrainData:
    """What a model's fit() receives.

    For fit_mode="per_series": kpi_id is set, values/timestamps/labels cover only that one
    series' training split.
    For fit_mode="shared": kpi_id is None, values/timestamps/labels are concatenated across
    every series, and kpi_ids (plural, one entry per row) identifies which series each row
    belongs to -- needed because temporal patterns differ by series even after normalization
    puts them on a common scale.
    """

    values: np.ndarray  # (n,) z-scored
    timestamps: np.ndarray  # (n,) unix seconds
    labels: np.ndarray  # (n,) 0/1
    kpi_id: str | None = None
    kpi_ids: np.ndarray | None = None  # only used when kpi_id is None


class ForecastModel(ABC):
    fit_mode: str  # "per_series" or "shared" -- set by subclasses

    @abstractmethod
    def fit(self, train_data: TrainData) -> None: ...

    @abstractmethod
    def predict(
        self,
        context_window: np.ndarray,
        horizon_steps: int,
        kpi_id: str | None = None,
        context_end_index: int | None = None,
    ) -> np.ndarray:
        """context_window: (context_steps,) z-scored values ending right before the forecast
        horizon. Returns (horizon_steps,) predicted z-scored values.

        kpi_id is passed through even for per_series models (which can ignore it, already
        being scoped to one series) so shared models have it available for series-conditioning
        without a different call signature between the two fit_mode patterns.

        context_end_index is the position, within the single chronologically-ordered
        train+val+test series a per_series model was fit/extended on, that this window's
        context ends at. Stateless models (persistence, XGBoost using only the passed-in
        context_window's own lag features) can ignore it. Models that maintain global fitted
        state across predict() calls (SARIMA: fit once on train, then extend that one state
        with the val/test series) need it to do an O(horizon_steps) indexed lookup into
        already-computed state instead of re-fitting/re-filtering from scratch per window --
        the difference between ~2.5ms and ~80ms per window (see compute-budget pilot notes).
        """
        ...


def flag_onset_from_forecast(predicted_horizon: np.ndarray, sigma_threshold: float = 3.0) -> bool:
    """Converts a continuous forecast into the binary 'predicted_positive' flag that
    evaluation_metrics.py's detection/false-alarm functions take as input.

    Rule: flag if any predicted horizon step deviates from 0 (the z-scored normal baseline)
    by more than sigma_threshold -- consistent with the 3sigma detection convention used
    throughout the aiops2022 analysis. Values are already z-scored per KPI on the training
    split's normal (label=0) samples (docs/aiops18_preprocessing_methodology.md, Section 4),
    so "0" is the normal baseline and this threshold is directly comparable across KPIs.
    """
    return bool(np.any(np.abs(predicted_horizon) > sigma_threshold))
