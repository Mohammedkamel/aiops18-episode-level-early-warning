"""Applies correction_layer.py to all six models' already-saved predictions (post-hoc, no
retraining) and appends {model}_corrected rows to model_comparison_summary.csv, same columns
as the raw rows, for direct before/after comparison.

Prediction files (kpi_id, window_start_ts, y_true, y_pred, predicted_positive) don't carry
context_label/horizon_label/onset_in_horizon needed for detection/false-alarm metrics, so
this joins back to the original windowed .npz files by (kpi_id, window_start_ts) rather than
assume row-order alignment -- robust regardless of how a given model's prediction file was
ordered, and necessary for Chronos's stratified-subsample predictions specifically.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))
from aiops18_preprocessing import load_merged, process_kpi  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from correction_layer import apply_correction_layer  # noqa: E402
from evaluation_metrics import (  # noqa: E402
    compute_detection_metrics,
    compute_false_alarm_metrics,
    extract_true_onset_timestamps,
    regression_metrics,
)
from results_logging import RESULTS_ROOT, append_summary_row, save_predictions  # noqa: E402
from threshold_calibration import calibrate_threshold_f1  # noqa: E402

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "processed_data" / "aiops18"
HORIZON_MIN, RESOLUTION_S, CONTEXT_STEPS, HORIZON_STEPS = 15, 60, 288, 15
MODELS = ["persistence", "sarima", "xgboost", "lstm", "patchtst", "chronos"]


def load_windows(split: str) -> dict:
    d = np.load(PROCESSED_DIR / f"aiops18_{split}_windows_h{HORIZON_MIN}min_{RESOLUTION_S}s.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def build_lookup(windows: dict) -> dict:
    return {(kpi, int(ts)): i for i, (kpi, ts) in enumerate(zip(windows["kpi_id"], windows["window_start_ts"]))}


def calibrate_all_kpis(merged, kpi_ids) -> dict:
    calibs = {}
    for kpi_id in kpi_ids:
        g = merged[merged["KPI ID"] == kpi_id]
        result = process_kpi(kpi_id, g)
        frame = result["frame"]
        train_vals = frame.loc[frame.split == "train", "value_z"].to_numpy()
        train_labels = frame.loc[frame.split == "train", "label"].to_numpy()
        calibs[kpi_id] = calibrate_threshold_f1(kpi_id, train_vals, train_labels)
    return calibs


def main():
    merged = load_merged()
    val_orig = load_windows("val")
    test_orig = load_windows("test")
    all_kpis = sorted(set(val_orig["kpi_id"]))

    calibs = calibrate_all_kpis(merged, all_kpis)
    low_precision_kpis = {k for k, c in calibs.items() if c.low_precision_flag}
    base_thresholds = {k: c.threshold for k, c in calibs.items()}
    print(f"{len(low_precision_kpis)}/{len(all_kpis)} KPIs low-precision (excluded from aggregate, same as raw runs)")

    val_lookup = build_lookup(val_orig)
    test_lookup = build_lookup(test_orig)
    origs = {"val": (val_orig, val_lookup), "test": (test_orig, test_lookup)}

    for model_name in MODELS:
        for split_name, (orig, lookup) in origs.items():
            pred_path = RESULTS_ROOT / "aiops18" / f"{model_name}_{split_name}_{HORIZON_MIN}min_{RESOLUTION_S}s_predictions.npz"
            if not pred_path.exists():
                print(f"skipping {model_name}/{split_name}: {pred_path} not found")
                continue
            d = np.load(pred_path, allow_pickle=True)
            kpi_id, window_start_ts = d["kpi_id"], d["window_start_ts"]
            y_true, y_pred = d["y_true"], d["y_pred"]
            n = len(kpi_id)

            corrected_positive, cp_flags = apply_correction_layer(
                kpi_id, window_start_ts, y_true, y_pred, HORIZON_STEPS, CONTEXT_STEPS, RESOLUTION_S, base_thresholds
            )

            orig_idx = np.array([lookup[(kpi_id[i], int(window_start_ts[i]))] for i in range(n)])
            context_label_last = orig["context_label"][orig_idx, -1]
            horizon_label = orig["horizon_label"][orig_idx]
            onset_in_horizon = orig["onset_in_horizon"][orig_idx]

            true_onset_ts = extract_true_onset_timestamps(
                window_start_ts, CONTEXT_STEPS, HORIZON_STEPS, RESOLUTION_S, context_label_last, horizon_label
            )
            clean_mask = ~np.isin(kpi_id, list(low_precision_kpis))

            reg = regression_metrics(y_true, y_pred)
            det = compute_detection_metrics(
                kpi_id[clean_mask], window_start_ts[clean_mask], CONTEXT_STEPS, HORIZON_STEPS,
                RESOLUTION_S, true_onset_ts[clean_mask], corrected_positive[clean_mask],
            )
            fa = compute_false_alarm_metrics(
                kpi_id[clean_mask], window_start_ts[clean_mask], CONTEXT_STEPS, HORIZON_STEPS,
                RESOLUTION_S, onset_in_horizon[clean_mask].astype(bool), corrected_positive[clean_mask],
            )

            metrics = dict(
                n_windows=n, mae=reg["mae"], rmse=reg["rmse"],
                n_episodes=det.n_episodes, detection_rate=det.detection_rate,
                missed_degradation_rate=det.missed_degradation_rate,
                lead_time_median_seconds=det.lead_time_median_seconds or "",
                lead_time_iqr_low_seconds=det.lead_time_iqr_seconds[0] if det.lead_time_iqr_seconds else "",
                lead_time_iqr_high_seconds=det.lead_time_iqr_seconds[1] if det.lead_time_iqr_seconds else "",
                n_false_alarm_episodes=fa.n_false_alarm_episodes,
                n_kpis_total=len(all_kpis), n_kpis_excluded_low_precision=len(low_precision_kpis),
                excluded_low_precision_kpi_ids=";".join(sorted(low_precision_kpis)),
            )
            corrected_name = f"{model_name}_corrected"
            print(f"\n[{corrected_name} / {split_name}] n_changepoints_flagged={int(cp_flags.sum())} {metrics}")

            save_predictions(
                "aiops18", corrected_name, split_name, HORIZON_MIN, RESOLUTION_S,
                kpi_id=kpi_id, window_start_ts=window_start_ts, y_true=y_true, y_pred=y_pred,
                predicted_positive=corrected_positive, overwrite=True,
            )
            append_summary_row("aiops18", corrected_name, split_name, HORIZON_MIN, RESOLUTION_S, metrics)


if __name__ == "__main__":
    main()
