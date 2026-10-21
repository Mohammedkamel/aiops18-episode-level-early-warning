"""Generates Persistence and SARIMA forecasts on ALL train windows (all 41 KPIs), for the
model-specific threshold-calibration sensitivity check: calibrating on each model's own
train-set forecast behavior instead of the raw observed value_z used by calibrate_threshold_f1.

Mirrors run_aiops18_baselines.py's per-window prediction loop exactly (same fit call, same
ts_to_pos indexing, same efficient SARIMA get_prediction(dynamic=True) approach) but scoped to
the train split's own windows instead of val/test.

Writes results/aiops18/train_forecasts_persistence_sarima.npz:
  kpi_id, window_start_ts, onset_in_horizon (bool), y_pred_persistence, y_pred_sarima
  (each (n_windows, horizon_steps)), y_true (horizon_value, for reference).
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))
from aiops18_preprocessing import load_merged, process_kpi  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_persistence import PersistenceModel  # noqa: E402
from baseline_sarima import SarimaModel  # noqa: E402
from model_interface import TrainData  # noqa: E402

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "processed_data" / "aiops18"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "aiops18"
HORIZON_MIN, RESOLUTION_S, CONTEXT_STEPS, HORIZON_STEPS = 15, 60, 288, 15


def main():
    train_windows = np.load(PROCESSED_DIR / f"aiops18_train_windows_h{HORIZON_MIN}min_{RESOLUTION_S}s.npz", allow_pickle=True)
    train_windows = {k: train_windows[k] for k in train_windows.files}
    all_kpis = sorted(set(train_windows["kpi_id"]))
    merged = load_merged()

    n_total = len(train_windows["kpi_id"])
    y_pred_persistence = np.zeros((n_total, HORIZON_STEPS), dtype=np.float32)
    y_pred_sarima = np.zeros((n_total, HORIZON_STEPS), dtype=np.float32)

    t_start = time.time()
    for ki, kpi_id in enumerate(all_kpis):
        g = merged[merged["KPI ID"] == kpi_id]
        result = process_kpi(kpi_id, g)
        frame = result["frame"]
        ts_to_pos = {ts: i for i, ts in enumerate(frame["timestamp"].to_numpy())}
        train_vals = frame.loc[frame.split == "train", "value_z"].to_numpy()

        mask = train_windows["kpi_id"] == kpi_id
        idx = np.where(mask)[0]
        n = len(idx)
        if n == 0:
            continue
        contexts = train_windows["context"][idx]
        window_start_ts = train_windows["window_start_ts"][idx]

        persist_model = PersistenceModel()
        persist_model.fit(TrainData(values=train_vals, timestamps=None, labels=None, kpi_id=kpi_id))

        sarima_model = SarimaModel()
        t0 = time.time()
        sarima_model.fit(TrainData(values=train_vals, timestamps=None, labels=None, kpi_id=kpi_id))
        fit_t = time.time() - t0

        t0 = time.time()
        for j, gi in enumerate(idx):
            ctx_end_idx = ts_to_pos[int(window_start_ts[j])] + CONTEXT_STEPS
            y_pred_persistence[gi] = persist_model.predict(contexts[j], HORIZON_STEPS, kpi_id=kpi_id, context_end_index=ctx_end_idx)
            y_pred_sarima[gi] = sarima_model.predict(contexts[j], HORIZON_STEPS, kpi_id=kpi_id, context_end_index=ctx_end_idx)
        pred_t = time.time() - t0

        elapsed = time.time() - t_start
        eta_min = elapsed / (ki + 1) * (len(all_kpis) - ki - 1) / 60
        print(f"[{ki+1}/{len(all_kpis)}] {kpi_id}: {n} windows, sarima_fit={fit_t:.1f}s pred={pred_t:.1f}s, "
              f"elapsed={elapsed/60:.1f}min ETA={eta_min:.1f}min", flush=True)

    np.savez(
        RESULTS_DIR / "train_forecasts_persistence_sarima.npz",
        kpi_id=train_windows["kpi_id"], window_start_ts=train_windows["window_start_ts"],
        onset_in_horizon=train_windows["onset_in_horizon"],
        y_pred_persistence=y_pred_persistence, y_pred_sarima=y_pred_sarima,
        y_true=train_windows["horizon_value"],
    )
    print(f"\nsaved to {RESULTS_DIR / 'train_forecasts_persistence_sarima.npz'}")


if __name__ == "__main__":
    main()
