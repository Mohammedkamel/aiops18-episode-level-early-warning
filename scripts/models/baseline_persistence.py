"""Persistence baseline: forecast = last observed value, held flat across the horizon."""

import numpy as np

from model_interface import ForecastModel, TrainData


class PersistenceModel(ForecastModel):
    fit_mode = "per_series"

    def fit(self, train_data: TrainData) -> None:
        pass  # nothing to fit

    def predict(self, context_window, horizon_steps, kpi_id=None, context_end_index=None) -> np.ndarray:
        return np.full(horizon_steps, context_window[-1], dtype=np.float64)
