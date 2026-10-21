"""Table 1 cohort-matching analysis (raw models only -- correction layer untouched).

Two paired-comparison recomputations of MAE/RMSE, both against the raw six-model
predictions already saved by run_aiops18_baselines.py / run_aiops18_phase2.py /
run_aiops18_chronos.py:

Issue 2 -- cohort mismatch: Table 1's MAE/RMSE is computed on all 41 series while
detection-rate metrics are computed only on the 26 series with low_precision_flag=False.
This recomputes MAE/RMSE restricted to exactly that same 26-series set, for direct
comparability with the detection metrics already reported.

Issue 3 -- Chronos enrichment: Chronos is evaluated on a stratified, ~30x anomaly-enriched
sample (all onset-overlapping windows + 122 sampled normal windows per series), not the full
population the other five models see. Using Chronos's exact (kpi_id, window_start_ts)
window set per split, this computes MAE/RMSE for the other five models on those identical
windows -- a genuine paired six-way comparison without rerunning Chronos.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))
from aiops18_preprocessing import load_merged, process_kpi  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation_metrics import regression_metrics  # noqa: E402
from threshold_calibration import calibrate_threshold_f1  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "aiops18"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "processed_data" / "aiops18"
HORIZON_MIN, RESOLUTION_S = 15, 60
MODELS = ["persistence", "sarima", "xgboost", "lstm", "patchtst", "chronos"]


def load_windows(split):
    d = np.load(PROCESSED_DIR / f"aiops18_{split}_windows_h{HORIZON_MIN}min_{RESOLUTION_S}s.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def load_pred(model_name, split):
    path = RESULTS_DIR / f"{model_name}_{split}_{HORIZON_MIN}min_{RESOLUTION_S}s_predictions.npz"
    d = np.load(path, allow_pickle=True)
    return d["kpi_id"], d["window_start_ts"], d["y_true"], d["y_pred"]


def main():
    val_windows = load_windows("val")
    all_kpis = sorted(set(val_windows["kpi_id"]))
    merged = load_merged()
    calibs = {}
    for kpi_id in all_kpis:
        g = merged[merged["KPI ID"] == kpi_id]
        result = process_kpi(kpi_id, g)
        frame = result["frame"]
        train_vals = frame.loc[frame.split == "train", "value_z"].to_numpy()
        train_labels = frame.loc[frame.split == "train", "label"].to_numpy()
        calibs[kpi_id] = calibrate_threshold_f1(kpi_id, train_vals, train_labels)
    low_precision_kpis = {k for k, c in calibs.items() if c.low_precision_flag}
    clean_kpis = set(all_kpis) - low_precision_kpis
    print(f"{len(low_precision_kpis)}/{len(all_kpis)} KPIs low-precision -> {len(clean_kpis)}-series clean cohort")

    # Chronos's exact (kpi_id, window_start_ts) window sets, per split
    chronos_keys = {}
    for split in ["val", "test"]:
        kpi_id, ts, _, _ = load_pred("chronos", split)
        chronos_keys[split] = set(zip(kpi_id.tolist(), ts.astype(np.int64).tolist()))
        print(f"chronos {split}: {len(chronos_keys[split])} windows in its evaluated set")

    rows = []
    for model in MODELS:
        for split in ["val", "test"]:
            kpi_id, ts, y_true, y_pred = load_pred(model, split)
            n_full = len(kpi_id)
            reg_full = regression_metrics(y_true, y_pred)

            clean_mask = np.isin(kpi_id, list(clean_kpis))
            n_clean = int(clean_mask.sum())
            # y_true/y_pred are flat (n_windows*horizon_steps,); reshape via horizon inferred from full length
            horizon_steps = y_true.size // n_full
            y_true_2d = y_true.reshape(n_full, horizon_steps)
            y_pred_2d = y_pred.reshape(n_full, horizon_steps)
            reg_clean = regression_metrics(y_true_2d[clean_mask].flatten(), y_pred_2d[clean_mask].flatten())

            keys = list(zip(kpi_id.tolist(), ts.astype(np.int64).tolist()))
            chronos_mask = np.array([k in chronos_keys[split] for k in keys])
            n_chronos_matched = int(chronos_mask.sum())
            reg_chronos_matched = regression_metrics(y_true_2d[chronos_mask].flatten(), y_pred_2d[chronos_mask].flatten())

            rows.append(dict(
                model=model, split=split,
                n_full=n_full, mae_full=reg_full["mae"], rmse_full=reg_full["rmse"],
                n_clean26=n_clean, mae_clean26=reg_clean["mae"], rmse_clean26=reg_clean["rmse"],
                n_chronos_matched=n_chronos_matched,
                mae_chronos_matched=reg_chronos_matched["mae"], rmse_chronos_matched=reg_chronos_matched["rmse"],
            ))
            if model != "chronos" and n_chronos_matched != len(chronos_keys[split]):
                print(f"  WARNING: {model}/{split} matched only {n_chronos_matched}/{len(chronos_keys[split])} of chronos's windows")

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "table1_cohort_comparison.csv", index=False)
    pd.set_option("display.width", 250)
    print()
    print(out.to_string(index=False))

    print("\n=== Rankings per split (lower MAE/RMSE = better) ===")
    for split in ["val", "test"]:
        sub = out[out.split == split]
        for metric, col in [("MAE", "mae"), ("RMSE", "rmse")]:
            for cohort in ["full", "clean26", "chronos_matched"]:
                ranked = sub.sort_values(f"{metric.lower()}_{cohort}")[["model", f"{metric.lower()}_{cohort}"]]
                order = ", ".join(ranked["model"].tolist())
                print(f"{split} {metric} [{cohort}]: {order}")


if __name__ == "__main__":
    main()
