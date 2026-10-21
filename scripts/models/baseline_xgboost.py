"""XGBoost baseline: feature-engineered (lags, rolling mean/std) gradient-boosted regressor,
fit per-KPI on the training series. Forecasts the horizon recursively -- one 1-step-ahead
model, applied repeatedly with each prediction fed back in as a new lag -- rather than
training a separate direct model per horizon step, to keep the per-KPI cost low (see
compute-budget pilot notes: fit is ~0.7s regardless of horizon length under this strategy).

Unlike SarimaModel, this needs no external state across predict() calls -- every feature is
derived purely from the passed-in context_window, so context_end_index is unused.
"""

import numpy as np
import xgboost as xgb

from model_interface import ForecastModel, TrainData

N_LAGS = 10
ROLL_WINDOW = 30


def _make_features(history: np.ndarray) -> np.ndarray:
    """history: at least ROLL_WINDOW values, most recent last. Returns one feature row."""
    lags = history[-N_LAGS:][::-1]
    roll = history[-ROLL_WINDOW:]
    return np.concatenate([lags, [roll.mean(), roll.std()]])


class XGBoostModel(ForecastModel):
    fit_mode = "per_series"

    def __init__(self, n_estimators: int = 200, max_depth: int = 6):
        self.model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, tree_method="hist")

    def fit(self, train_data: TrainData) -> None:
        vals = train_data.values
        X, y = [], []
        for i in range(ROLL_WINDOW, len(vals)):
            X.append(_make_features(vals[:i]))
            y.append(vals[i])
        self.model.fit(np.array(X), np.array(y))

    def predict(self, context_window, horizon_steps, kpi_id=None, context_end_index=None) -> np.ndarray:
        history = list(context_window[-max(N_LAGS, ROLL_WINDOW):])
        preds = []
        for _ in range(horizon_steps):
            feat = _make_features(np.array(history))
            pred = float(self.model.predict(feat.reshape(1, -1))[0])
            preds.append(pred)
            history.append(pred)
        return np.array(preds)
