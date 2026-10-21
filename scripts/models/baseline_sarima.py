"""SARIMA baseline: classical statistical model, fit once per KPI on the training series.

Prediction uses the efficient approach found during the compute-budget pilot: extend the
fitted model's state with the val/test series ONCE (`append(..., refit=False)`), then get
each window's forecast via an indexed `get_prediction(start, end, dynamic=True)` against that
single extended object. A naive per-window re-fit/re-filter loop measured 79ms/window; this
approach measures 2.5ms/window (~31x faster) because it reuses already-computed filtered
state instead of recomputing it from scratch for every window.
"""

import warnings

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

from model_interface import ForecastModel, TrainData

warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")


class SarimaModel(ForecastModel):
    fit_mode = "per_series"

    def __init__(self, order: tuple[int, int, int] = (2, 1, 2)):
        self.order = order
        self.fit_result = None
        self.extended_result = None

    def fit(self, train_data: TrainData) -> None:
        model = SARIMAX(train_data.values, order=self.order, enforce_stationarity=False, enforce_invertibility=False)
        self.fit_result = model.fit(disp=False)
        self.extended_result = self.fit_result  # no extension yet; extend_state() adds val/test

    def extend_state(self, additional_values: np.ndarray) -> None:
        """Call once with the full val (then again with val+test, or separately) series before
        any predict() calls that need context_end_index to reach into that range. Must be
        called after fit(); each call extends from the CURRENT extended_result, so call once
        with the complete additional series rather than incrementally per window.
        """
        self.extended_result = self.extended_result.append(additional_values, refit=False)

    def predict(self, context_window, horizon_steps, kpi_id=None, context_end_index=None) -> np.ndarray:
        if context_end_index is None:
            raise ValueError("SarimaModel.predict requires context_end_index (see model_interface.py)")
        pred = self.extended_result.get_prediction(
            start=context_end_index, end=context_end_index + horizon_steps - 1, dynamic=True
        )
        return np.asarray(pred.predicted_mean)
