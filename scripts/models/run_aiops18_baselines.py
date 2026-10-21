"""Phase 1: run Persistence, SARIMA, and XGBoost baselines on AIOps18's 15-minute horizon,
60s-resolution KPIs, through the shared episode-level evaluation module.

Fits each model on the per-KPI raw (gap-interpolated, z-scored) train series reconstructed
via the same aiops18_preprocessing.process_kpi() used to build the saved windows, so training
data is guaranteed consistent with what the windows were built from. Known simplification:
the fitted series concatenates a split's rows in order including any hard-gap boundaries
within that split (only window *construction*, not classical-model *fitting*, is gap-aware --
SARIMA/XGBoost see the gap-adjacent values as if temporally contiguous). Reasonable for a
Phase 1 baseline; flagged here rather than silently assumed away.

Usage: main(kpi_ids=[...]) for a quick subset (completion-gate check), or main() for all 41
60s-resolution KPIs (the full compute-budget-pilot-estimated ~1.75-2hr run).
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))
from aiops18_preprocessing import load_merged, process_kpi  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_persistence import PersistenceModel  # noqa: E402
from baseline_sarima import SarimaModel  # noqa: E402
from baseline_xgboost import XGBoostModel  # noqa: E402
from evaluation_metrics import (  # noqa: E402
    compute_detection_metrics,
    compute_false_alarm_metrics,
    extract_true_onset_timestamps,
    regression_metrics,
)
from model_interface import TrainData, flag_onset_from_forecast  # noqa: E402
from results_logging import append_summary_row, save_predictions  # noqa: E402
from threshold_calibration import calibrate_threshold_f1  # noqa: E402

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "processed_data" / "aiops18"
HORIZON_MIN = 15
RESOLUTION_S = 60


def load_windows(split: str) -> dict:
    d = np.load(PROCESSED_DIR / f"aiops18_{split}_windows_h{HORIZON_MIN}min_{RESOLUTION_S}s.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def run_one_kpi(kpi_id: str, frame: pd.DataFrame, val_windows: dict, test_windows: dict, context_steps: int, model_names: tuple[str, ...] = ("persistence", "sarima", "xgboost")) -> dict:
    ts_to_pos = {ts: i for i, ts in enumerate(frame["timestamp"].to_numpy())}

    train_vals = frame.loc[frame.split == "train", "value_z"].to_numpy()
    train_labels = frame.loc[frame.split == "train", "label"].to_numpy()
    val_vals = frame.loc[frame.split == "val", "value_z"].to_numpy()
    test_vals = frame.loc[frame.split == "test", "value_z"].to_numpy()

    # Calibrated ONCE from train (value_z, label) only; the same threshold is reused for both
    # val and test below with no re-tuning per split (threshold_calibration.py docstring).
    calib = calibrate_threshold_f1(kpi_id, train_vals, train_labels)
    print(
        f"    [{kpi_id}] calibrated threshold={calib.threshold:.3f} "
        f"(train precision={calib.precision:.3f}, recall={calib.recall:.3f}, f1={calib.f1:.3f})"
        + (" *** LOW PRECISION (<30%) even at best threshold ***" if calib.low_precision_flag else "")
    )

    available = {"persistence": PersistenceModel, "sarima": SarimaModel, "xgboost": XGBoostModel}
    models = {name: available[name]() for name in model_names}
    fit_times = {}
    for name, model in models.items():
        t0 = time.time()
        model.fit(TrainData(values=train_vals, timestamps=np.arange(len(train_vals)), labels=np.zeros(len(train_vals)), kpi_id=kpi_id))
        if name == "sarima":
            model.extend_state(val_vals)
            model.extend_state(test_vals)
        fit_times[name] = time.time() - t0

    results = {}
    for split_name, windows in (("val", val_windows), ("test", test_windows)):
        mask = windows["kpi_id"] == kpi_id
        n = mask.sum()
        if n == 0:
            continue
        contexts = windows["context"][mask]
        horizon_values = windows["horizon_value"][mask]
        horizon_labels = windows["horizon_label"][mask]
        context_labels = windows["context_label"][mask]
        onset_in_horizon = windows["onset_in_horizon"][mask]
        window_start_ts = windows["window_start_ts"][mask]

        true_onset_ts = extract_true_onset_timestamps(
            window_start_ts, context_steps, horizon_values.shape[1], RESOLUTION_S, context_labels[:, -1], horizon_labels
        )

        for name, model in models.items():
            y_pred = np.zeros_like(horizon_values)
            for i in range(n):
                ctx_end_idx = ts_to_pos[int(window_start_ts[i])] + context_steps
                y_pred[i] = model.predict(
                    contexts[i], horizon_values.shape[1], kpi_id=kpi_id, context_end_index=ctx_end_idx
                )
            predicted_positive = np.array(
                [flag_onset_from_forecast(y_pred[i], sigma_threshold=calib.threshold) for i in range(n)]
            )

            reg = regression_metrics(horizon_values.flatten(), y_pred.flatten())
            det = compute_detection_metrics(
                windows["kpi_id"][mask], window_start_ts, context_steps, horizon_values.shape[1],
                RESOLUTION_S, true_onset_ts, predicted_positive,
            )
            fa = compute_false_alarm_metrics(
                windows["kpi_id"][mask], window_start_ts, context_steps, horizon_values.shape[1],
                RESOLUTION_S, onset_in_horizon.astype(bool), predicted_positive,
            )
            results[(name, split_name)] = dict(
                kpi_id=kpi_id, reg=reg, det=det, fa=fa, y_pred=y_pred, predicted_positive=predicted_positive,
                window_start_ts=window_start_ts, y_true=horizon_values, fit_time=fit_times[name],
            )
    return results, calib


def main(kpi_ids: list[str] | None = None, save: bool = True, model_names: tuple[str, ...] = ("persistence", "sarima", "xgboost")):
    merged = load_merged()
    val_windows = load_windows("val")
    test_windows = load_windows("test")

    all_kpis_60s = sorted(set(val_windows["kpi_id"]) & set(test_windows["kpi_id"]))
    targets = kpi_ids if kpi_ids is not None else all_kpis_60s
    print(f"running on {len(targets)} KPI(s): {targets if len(targets) <= 5 else targets[:5] + ['...']}")

    accum = {}  # (model_name, split) -> list of per-kpi results, for aggregation
    calibrations = []
    for kpi_id in targets:
        t0 = time.time()
        g = merged[merged["KPI ID"] == kpi_id]
        result = process_kpi(kpi_id, g)
        frame = result["frame"]
        context_steps = result["context_steps"]

        kpi_results, calib = run_one_kpi(kpi_id, frame, val_windows, test_windows, context_steps, model_names=model_names)
        calibrations.append(calib)
        for key, val in kpi_results.items():
            accum.setdefault(key, []).append(val)
        print(f"  {kpi_id}: done in {time.time()-t0:.1f}s")

    low_precision_kpis = [c.kpi_id for c in calibrations if c.low_precision_flag]
    print(f"\n--- threshold calibration summary ({len(calibrations)} KPIs) ---")
    for c in calibrations:
        flag = " *** LOW PRECISION ***" if c.low_precision_flag else ""
        print(f"  {c.kpi_id}: threshold={c.threshold:.3f} precision={c.precision:.3f} recall={c.recall:.3f} f1={c.f1:.3f}{flag}")
    if low_precision_kpis:
        print(f"\n  {len(low_precision_kpis)}/{len(calibrations)} KPIs have <30% precision even at their F1-optimal threshold: {low_precision_kpis}")

    for (model_name, split_name), kpi_results_list in accum.items():
        # MAE/RMSE don't depend on the flagging threshold -- computed over ALL KPIs.
        y_true = np.concatenate([r["y_true"].flatten() for r in kpi_results_list])
        y_pred = np.concatenate([r["y_pred"].flatten() for r in kpi_results_list])
        reg = regression_metrics(y_true, y_pred)

        # Detection rate / false-alarm rate / lead time: EXCLUDE low-precision-flagged KPIs
        # from the aggregate (their F1-optimal threshold is "flag almost everything," which
        # saturates every model to ~100% detection regardless of forecast quality and masks
        # real model differences -- confirmed concretely during the completion-gate check).
        # Still reported per-KPI below, just not pooled into the headline number.
        clean = [r for r in kpi_results_list if r["kpi_id"] not in low_precision_kpis]
        excluded = [r for r in kpi_results_list if r["kpi_id"] in low_precision_kpis]

        n_episodes = sum(r["det"].n_episodes for r in clean)
        n_detected = sum(r["det"].n_detected for r in clean)
        detection_rate = n_detected / n_episodes if n_episodes else float("nan")
        all_lead_times = [lt for r in clean for lt in r["det"].lead_times_seconds]
        n_fa = sum(r["fa"].n_false_alarm_episodes for r in clean)

        metrics = dict(
            n_windows=sum(len(r["y_pred"]) for r in kpi_results_list),
            mae=reg["mae"], rmse=reg["rmse"],
            n_episodes=n_episodes, detection_rate=detection_rate,
            missed_degradation_rate=(1 - detection_rate) if n_episodes else float("nan"),
            lead_time_median_seconds=float(np.median(all_lead_times)) if all_lead_times else "",
            lead_time_iqr_low_seconds=float(np.percentile(all_lead_times, 25)) if all_lead_times else "",
            lead_time_iqr_high_seconds=float(np.percentile(all_lead_times, 75)) if all_lead_times else "",
            n_false_alarm_episodes=n_fa,
            n_kpis_total=len(kpi_results_list),
            n_kpis_excluded_low_precision=len(excluded),
            excluded_low_precision_kpi_ids=";".join(sorted({r["kpi_id"] for r in excluded})),
        )
        print(f"\n[{model_name} / {split_name}] {metrics}")
        for r in excluded:
            print(
                f"    (excluded from aggregate) {r['kpi_id']}: n_episodes={r['det'].n_episodes}, "
                f"detected={r['det'].n_detected}, rate={r['det'].detection_rate:.3f}"
            )

        if save:
            kpi_id_concat = np.concatenate([[r["kpi_id"]] * len(r["window_start_ts"]) for r in kpi_results_list])
            window_start_ts_concat = np.concatenate([r["window_start_ts"] for r in kpi_results_list])
            predicted_positive_concat = np.concatenate([r["predicted_positive"] for r in kpi_results_list])
            save_predictions(
                "aiops18", model_name, split_name, HORIZON_MIN, RESOLUTION_S,
                kpi_id=kpi_id_concat,
                window_start_ts=window_start_ts_concat,
                y_true=y_true, y_pred=y_pred,
                predicted_positive=predicted_positive_concat,
                overwrite=True,
            )
            append_summary_row("aiops18", model_name, split_name, HORIZON_MIN, RESOLUTION_S, metrics)

    return accum


if __name__ == "__main__":
    import sys as _sys
    kpi_arg = _sys.argv[1:] if len(_sys.argv) > 1 else None
    main(kpi_ids=kpi_arg)
