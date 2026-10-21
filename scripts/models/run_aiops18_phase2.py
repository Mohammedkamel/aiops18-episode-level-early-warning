"""Phase 2: LSTM and PatchTST, single shared-weights models trained across all 41
60s-resolution AIOps18 KPIs, evaluated through the same episode-level pipeline as Phase 1
(same calibrate-on-train-only threshold rule, same low-precision-KPI exclusion from the
aggregate, same result-logging convention). Must run under .venv-torch (see
baseline_lstm.py/baseline_patchtst.py docstrings for why).

Training data is the precomputed windowed arrays (processed_data/aiops18/aiops18_train_windows_h15min_60s.npz),
pooled across all 41 KPIs -- NOT the raw per-KPI series Phase 1's classical models used,
since shared models need one pooled dataset rather than one series per fit() call.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))
from aiops18_preprocessing import load_merged, process_kpi  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_lstm import LSTMModel  # noqa: E402
from baseline_patchtst import PatchTSTModel  # noqa: E402
from evaluation_metrics import (  # noqa: E402
    compute_detection_metrics,
    compute_false_alarm_metrics,
    extract_true_onset_timestamps,
    regression_metrics,
)
from model_interface import flag_onset_from_forecast  # noqa: E402
from results_logging import append_summary_row, save_predictions  # noqa: E402
from threshold_calibration import calibrate_threshold_f1  # noqa: E402

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "processed_data" / "aiops18"
HORIZON_MIN, RESOLUTION_S, CONTEXT_STEPS, HORIZON_STEPS = 15, 60, 288, 15
SEED = 42  # fixed for reproducibility across LSTM/PatchTST runs -- see aiops18_evaluation_methodology.md


def set_seed(seed: int = SEED) -> None:
    """Must be called immediately before constructing a model -- weight initialization
    happens in __init__, not fit_shared, so seeding only inside fit_shared would leave
    initial weights uncontrolled (the larger source of run-to-run variance) and only fix
    the epoch shuffle order.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_windows(split: str) -> dict:
    d = np.load(PROCESSED_DIR / f"aiops18_{split}_windows_h{HORIZON_MIN}min_{RESOLUTION_S}s.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


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


def evaluate_model(model, model_name, val_windows, test_windows, calibs, low_precision_kpis):
    for split_name, windows in (("val", val_windows), ("test", test_windows)):
        n = len(windows["context"])
        y_pred = model.predict_batch(windows["context"], windows["kpi_id"])
        predicted_positive = np.array(
            [flag_onset_from_forecast(y_pred[i], sigma_threshold=calibs[windows["kpi_id"][i]].threshold) for i in range(n)]
        )

        true_onset_ts = extract_true_onset_timestamps(
            windows["window_start_ts"], CONTEXT_STEPS, HORIZON_STEPS, RESOLUTION_S,
            windows["context_label"][:, -1], windows["horizon_label"],
        )

        clean_mask = ~np.isin(windows["kpi_id"], list(low_precision_kpis))

        reg = regression_metrics(windows["horizon_value"].flatten(), y_pred.flatten())
        det = compute_detection_metrics(
            windows["kpi_id"][clean_mask], windows["window_start_ts"][clean_mask], CONTEXT_STEPS, HORIZON_STEPS,
            RESOLUTION_S, true_onset_ts[clean_mask], predicted_positive[clean_mask],
        )
        fa = compute_false_alarm_metrics(
            windows["kpi_id"][clean_mask], windows["window_start_ts"][clean_mask], CONTEXT_STEPS, HORIZON_STEPS,
            RESOLUTION_S, windows["onset_in_horizon"][clean_mask].astype(bool), predicted_positive[clean_mask],
        )

        metrics = dict(
            n_windows=n, mae=reg["mae"], rmse=reg["rmse"],
            n_episodes=det.n_episodes, detection_rate=det.detection_rate,
            missed_degradation_rate=det.missed_degradation_rate,
            lead_time_median_seconds=det.lead_time_median_seconds or "",
            lead_time_iqr_low_seconds=det.lead_time_iqr_seconds[0] if det.lead_time_iqr_seconds else "",
            lead_time_iqr_high_seconds=det.lead_time_iqr_seconds[1] if det.lead_time_iqr_seconds else "",
            n_false_alarm_episodes=fa.n_false_alarm_episodes,
            n_kpis_total=len(set(windows["kpi_id"])),
            n_kpis_excluded_low_precision=len(low_precision_kpis),
            excluded_low_precision_kpi_ids=";".join(sorted(low_precision_kpis)),
        )
        print(f"\n[{model_name} / {split_name}] {metrics}")

        save_predictions(
            "aiops18", model_name, split_name, HORIZON_MIN, RESOLUTION_S,
            kpi_id=windows["kpi_id"], window_start_ts=windows["window_start_ts"],
            y_true=windows["horizon_value"].flatten(), y_pred=y_pred.flatten(),
            predicted_positive=predicted_positive, overwrite=True,
        )
        append_summary_row("aiops18", model_name, split_name, HORIZON_MIN, RESOLUTION_S, metrics)


def main(lstm_epochs=5, patchtst_epochs=15, run_lstm=True, run_patchtst=True):
    checkpoint_dir = Path(__file__).resolve().parents[2] / "results" / "aiops18" / "checkpoints"

    train_windows = load_windows("train")
    val_windows = load_windows("val")
    test_windows = load_windows("test")

    all_kpis = sorted(set(train_windows["kpi_id"]))
    kpi_to_idx = {k: i for i, k in enumerate(all_kpis)}
    print(f"pooling {len(all_kpis)} KPIs, {len(train_windows['context'])} train windows")

    merged = load_merged()
    calibs = calibrate_all_kpis(merged, all_kpis)
    low_precision_kpis = {k for k, c in calibs.items() if c.low_precision_flag}
    print(f"{len(low_precision_kpis)}/{len(all_kpis)} KPIs low-precision (excluded from aggregate)")

    if run_lstm:
        print("\n=== training LSTM ===")
        t0 = time.time()
        set_seed()
        lstm = LSTMModel(n_kpis=len(all_kpis), kpi_to_idx=kpi_to_idx, horizon_steps=HORIZON_STEPS)
        lstm.fit_shared(
            train_windows["context"], train_windows["horizon_value"], train_windows["kpi_id"],
            n_epochs=lstm_epochs, checkpoint_dir=checkpoint_dir,
        )
        print(f"LSTM training took {(time.time()-t0)/60:.1f} min")
        evaluate_model(lstm, "lstm", val_windows, test_windows, calibs, low_precision_kpis)

    if not run_patchtst:
        return
    print("\n=== training PatchTST ===")
    t0 = time.time()
    set_seed()
    patchtst = PatchTSTModel(n_kpis=len(all_kpis), kpi_to_idx=kpi_to_idx, context_steps=CONTEXT_STEPS, horizon_steps=HORIZON_STEPS)
    patchtst.fit_shared(
        train_windows["context"], train_windows["horizon_value"], train_windows["kpi_id"],
        n_epochs=patchtst_epochs, checkpoint_dir=checkpoint_dir,
    )
    print(f"PatchTST training took {(time.time()-t0)/60:.1f} min")
    evaluate_model(patchtst, "patchtst", val_windows, test_windows, calibs, low_precision_kpis)


if __name__ == "__main__":
    main()
