"""Chronos (amazon/chronos-t5-tiny) zero-shot evaluation on AIOps18's clipped, windowed
data. No fine-tuning -- pretrained checkpoint, direct inference only.

Compute-driven design: the full 350k-window val+test set at num_samples=20 (standard
Chronos usage) was timed at ~125 hours with the mandated 2x caution multiplier -- not
viable. Instead this runs on a stratified subsample: (1) every window overlapping one of
the 246 val / 232 test true onset episodes (clean, non-low-precision KPI set -- same
episode set every other model's detection-rate/lead-time/false-alarm metrics are computed
on, so those metrics are NOT degraded by subsampling), plus (2) an equal-per-KPI stratified
sample of ~122 normal windows per KPI per split (~5,000/split, ~10,000 total) for a
representative MAE/RMSE estimate -- stratified by KPI, not a flat pooled random draw, so no
single series with a disproportionate window count can dominate the aggregate.

Checkpoints incrementally (results/aiops18/chronos_checkpoint.npz, atomic write via
tmp-then-rename) so a multi-hour run can be interrupted and resumed without recomputing
already-processed windows.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))
from aiops18_preprocessing import load_merged, process_kpi  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
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
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "aiops18"
CHECKPOINT_PATH = RESULTS_DIR / "chronos_checkpoint.npz"
HORIZON_MIN, RESOLUTION_S, CONTEXT_STEPS, HORIZON_STEPS = 15, 60, 288, 15
TARGET_NORMAL_PER_KPI = 122
NUM_SAMPLES = 20
CHUNK_SIZE = 50
CHECKPOINT_SEED = 42


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


def build_stratified_sample(windows: dict, all_kpis: list[str], low_precision_kpis: set[str], rng: np.random.Generator) -> np.ndarray:
    """Returns sorted array of window indices: all clean-KPI episode windows + an equal-
    per-KPI stratified sample of normal windows."""
    true_onset_ts = extract_true_onset_timestamps(
        windows["window_start_ts"], CONTEXT_STEPS, HORIZON_STEPS, RESOLUTION_S,
        windows["context_label"][:, -1], windows["horizon_label"],
    )
    clean_mask = ~np.isin(windows["kpi_id"], list(low_precision_kpis))
    is_episode_window = (~np.isnan(true_onset_ts)) & clean_mask
    episode_idx = np.where(is_episode_window)[0]

    normal_idx_all = np.where(~is_episode_window)[0]
    sampled = [episode_idx]
    for kpi in all_kpis:
        kpi_normal_idx = normal_idx_all[windows["kpi_id"][normal_idx_all] == kpi]
        n_take = min(TARGET_NORMAL_PER_KPI, len(kpi_normal_idx))
        if n_take > 0:
            sampled.append(rng.choice(kpi_normal_idx, size=n_take, replace=False))
    return np.sort(np.concatenate(sampled))


def save_checkpoint(y_preds: dict) -> None:
    keys = np.array(list(y_preds.keys()), dtype=object)
    preds = np.array(list(y_preds.values()), dtype=np.float32)
    tmp_path = CHECKPOINT_PATH.with_suffix(".tmp.npz")
    np.savez(tmp_path, keys=keys, preds=preds)
    tmp_path.rename(CHECKPOINT_PATH)  # atomic on the same filesystem


def load_checkpoint() -> dict:
    if not CHECKPOINT_PATH.exists():
        return {}
    d = np.load(CHECKPOINT_PATH, allow_pickle=True)
    return dict(zip(d["keys"].tolist(), d["preds"]))


def main():
    val_windows = load_windows("val")
    test_windows = load_windows("test")
    all_kpis = sorted(set(val_windows["kpi_id"]))

    merged = load_merged()
    calibs = calibrate_all_kpis(merged, all_kpis)
    low_precision_kpis = {k for k, c in calibs.items() if c.low_precision_flag}
    print(f"{len(low_precision_kpis)}/{len(all_kpis)} KPIs low-precision (excluded from episode aggregate)")

    rng = np.random.default_rng(CHECKPOINT_SEED)
    val_idx = build_stratified_sample(val_windows, all_kpis, low_precision_kpis, rng)
    test_idx = build_stratified_sample(test_windows, all_kpis, low_precision_kpis, rng)
    print(f"val sample: {len(val_idx)} windows, test sample: {len(test_idx)} windows, total: {len(val_idx)+len(test_idx)}")

    tasks = [("val", int(i)) for i in val_idx] + [("test", int(i)) for i in test_idx]
    y_preds = load_checkpoint()
    print(f"resuming from checkpoint: {len(y_preds)} windows already done" if y_preds else "no checkpoint found, starting fresh")

    remaining = [(s, i) for (s, i) in tasks if f"{s}_{i}" not in y_preds]
    print(f"{len(remaining)} windows remaining out of {len(tasks)}")

    if remaining:
        from chronos import ChronosPipeline

        pipeline = ChronosPipeline.from_pretrained("amazon/chronos-t5-tiny", device_map="cpu", torch_dtype=torch.float32)
        windows_by_split = {"val": val_windows, "test": test_windows}

        t_start = time.time()
        for chunk_start in range(0, len(remaining), CHUNK_SIZE):
            t0 = time.time()
            chunk = remaining[chunk_start : chunk_start + CHUNK_SIZE]
            contexts = [torch.tensor(windows_by_split[s]["context"][i], dtype=torch.float32) for s, i in chunk]
            forecast = pipeline.predict(context=contexts, prediction_length=HORIZON_STEPS, num_samples=NUM_SAMPLES)
            median_forecast = np.median(forecast.numpy(), axis=1)  # (batch, horizon_steps)
            for (s, i), pred in zip(chunk, median_forecast):
                y_preds[f"{s}_{i}"] = pred
            save_checkpoint(y_preds)
            done = len(y_preds)
            elapsed_total = time.time() - t_start
            rate = elapsed_total / (chunk_start + len(chunk))
            eta_min = rate * (len(remaining) - chunk_start - len(chunk)) / 60
            print(
                f"  chunk {chunk_start // CHUNK_SIZE + 1}: {done}/{len(tasks)} done "
                f"({time.time()-t0:.1f}s this chunk, ETA {eta_min:.1f} min)"
            )

    print("\nall windows processed, computing final metrics...")

    for split_name, windows, idx in (("val", val_windows, val_idx), ("test", test_windows, test_idx)):
        y_pred = np.array([y_preds[f"{split_name}_{i}"] for i in idx])
        sub = {k: v[idx] for k, v in windows.items()}
        y_true = sub["horizon_value"]

        true_onset_ts = extract_true_onset_timestamps(
            sub["window_start_ts"], CONTEXT_STEPS, HORIZON_STEPS, RESOLUTION_S,
            sub["context_label"][:, -1], sub["horizon_label"],
        )
        predicted_positive = np.array(
            [flag_onset_from_forecast(y_pred[i], sigma_threshold=calibs[sub["kpi_id"][i]].threshold) for i in range(len(idx))]
        )

        clean_mask = ~np.isin(sub["kpi_id"], list(low_precision_kpis))
        reg = regression_metrics(y_true.flatten(), y_pred.flatten())
        det = compute_detection_metrics(
            sub["kpi_id"][clean_mask], sub["window_start_ts"][clean_mask], CONTEXT_STEPS, HORIZON_STEPS,
            RESOLUTION_S, true_onset_ts[clean_mask], predicted_positive[clean_mask],
        )
        fa = compute_false_alarm_metrics(
            sub["kpi_id"][clean_mask], sub["window_start_ts"][clean_mask], CONTEXT_STEPS, HORIZON_STEPS,
            RESOLUTION_S, sub["onset_in_horizon"][clean_mask].astype(bool), predicted_positive[clean_mask],
        )

        metrics = dict(
            n_windows=len(idx), mae=reg["mae"], rmse=reg["rmse"],
            n_episodes=det.n_episodes, detection_rate=det.detection_rate,
            missed_degradation_rate=det.missed_degradation_rate,
            lead_time_median_seconds=det.lead_time_median_seconds or "",
            lead_time_iqr_low_seconds=det.lead_time_iqr_seconds[0] if det.lead_time_iqr_seconds else "",
            lead_time_iqr_high_seconds=det.lead_time_iqr_seconds[1] if det.lead_time_iqr_seconds else "",
            n_false_alarm_episodes=fa.n_false_alarm_episodes,
            n_kpis_total=len(all_kpis), n_kpis_excluded_low_precision=len(low_precision_kpis),
            excluded_low_precision_kpi_ids=";".join(sorted(low_precision_kpis)),
        )
        print(f"\n[chronos / {split_name}] {metrics}")

        save_predictions(
            "aiops18", "chronos", split_name, HORIZON_MIN, RESOLUTION_S,
            kpi_id=sub["kpi_id"], window_start_ts=sub["window_start_ts"],
            y_true=y_true.flatten(), y_pred=y_pred.flatten(),
            predicted_positive=predicted_positive, overwrite=True,
        )
        append_summary_row("aiops18", "chronos", split_name, HORIZON_MIN, RESOLUTION_S, metrics)

    print(f"\ndone. checkpoint preserved at {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
