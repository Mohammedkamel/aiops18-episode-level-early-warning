"""Result-logging convention shared by every model/dataset combination.

Predictions: results/{dataset}/{model_name}_{horizon}_{resolution}_predictions.npz
Running summary: results/{dataset}/model_comparison_summary.csv, one row per
(dataset, model_name, horizon, resolution) -- reruns upsert (replace the matching row) rather
than duplicate or silently append twice.
"""

import csv
from pathlib import Path

import numpy as np

RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results"

SUMMARY_COLUMNS = [
    "dataset",
    "model_name",
    "split",
    "horizon_minutes",
    "resolution_seconds",
    "n_windows",
    "mae",
    "rmse",
    "n_episodes",
    "detection_rate",
    "missed_degradation_rate",
    "lead_time_median_seconds",
    "lead_time_iqr_low_seconds",
    "lead_time_iqr_high_seconds",
    "n_false_alarm_episodes",
    "n_kpis_total",
    "n_kpis_excluded_low_precision",
    "excluded_low_precision_kpi_ids",
]


def predictions_path(dataset: str, model_name: str, split: str, horizon_minutes: int, resolution_seconds: int) -> Path:
    return RESULTS_ROOT / dataset / f"{model_name}_{split}_{horizon_minutes}min_{resolution_seconds}s_predictions.npz"


def save_predictions(
    dataset: str,
    model_name: str,
    split: str,
    horizon_minutes: int,
    resolution_seconds: int,
    *,
    kpi_id: np.ndarray,
    window_start_ts: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    predicted_positive: np.ndarray,
    overwrite: bool = False,
) -> Path:
    out_path = predictions_path(dataset, model_name, split, horizon_minutes, resolution_seconds)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} already exists — pass overwrite=True to confirm replacing it")
    np.savez_compressed(
        out_path,
        kpi_id=kpi_id,
        window_start_ts=window_start_ts,
        y_true=y_true,
        y_pred=y_pred,
        predicted_positive=predicted_positive,
    )
    return out_path


def append_summary_row(dataset: str, model_name: str, split: str, horizon_minutes: int, resolution_seconds: int, metrics: dict) -> Path:
    """Upserts one row into results/{dataset}/model_comparison_summary.csv keyed on
    (dataset, model_name, split, horizon_minutes, resolution_seconds). metrics should contain
    whatever subset of SUMMARY_COLUMNS[5:] is available (missing keys become blank cells).
    """
    summary_path = RESULTS_ROOT / dataset / "model_comparison_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    row = {c: "" for c in SUMMARY_COLUMNS}
    row.update(dataset=dataset, model_name=model_name, split=split, horizon_minutes=horizon_minutes, resolution_seconds=resolution_seconds)
    row.update({k: v for k, v in metrics.items() if k in SUMMARY_COLUMNS})

    existing_rows = []
    if summary_path.exists():
        with open(summary_path, newline="") as f:
            existing_rows = list(csv.DictReader(f))

    key = (dataset, str(model_name), str(split), str(horizon_minutes), str(resolution_seconds))
    existing_rows = [
        r for r in existing_rows
        if (r["dataset"], r["model_name"], r["split"], r["horizon_minutes"], r["resolution_seconds"]) != key
    ]
    existing_rows.append({k: str(v) for k, v in row.items()})

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(existing_rows)

    return summary_path
