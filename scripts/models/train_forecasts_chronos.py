"""Chronos zero-shot forecasts on a stratified sample of TRAIN windows (all 41 KPIs), for the
model-specific threshold-calibration sensitivity check. Same stratified-sampling design as
run_aiops18_chronos.py's val/test evaluation (all onset-overlapping windows + up to 122
per-KPI normal windows), applied to the train split instead -- this is deliberately a SAMPLE,
not the full 578,235-window train population, because Chronos inference is far too slow
(~0.64s/window) to run at that scale; a stratified sample is the same tradeoff already made
for its production val/test evaluation.

Writes results/aiops18/train_forecasts_chronos.npz (only for the sampled windows):
  kpi_id, window_start_ts, onset_in_horizon, y_pred (median of 20 samples), y_true.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "processed_data" / "aiops18"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "aiops18"
CHECKPOINT_PATH = RESULTS_DIR / "chronos_train_checkpoint.npz"
HORIZON_MIN, RESOLUTION_S, CONTEXT_STEPS, HORIZON_STEPS = 15, 60, 288, 15
TARGET_NORMAL_PER_KPI = 122
NUM_SAMPLES = 20
CHUNK_SIZE = 50
SEED = 42


def build_stratified_train_sample(windows: dict, all_kpis: list[str], rng: np.random.Generator) -> np.ndarray:
    onset = windows["onset_in_horizon"].astype(bool)
    episode_idx = np.where(onset)[0]
    normal_idx_all = np.where(~onset)[0]
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
    tmp_path.rename(CHECKPOINT_PATH)


def load_checkpoint() -> dict:
    if not CHECKPOINT_PATH.exists():
        return {}
    d = np.load(CHECKPOINT_PATH, allow_pickle=True)
    return dict(zip(d["keys"].tolist(), d["preds"]))


def main():
    train_windows = np.load(PROCESSED_DIR / f"aiops18_train_windows_h{HORIZON_MIN}min_{RESOLUTION_S}s.npz", allow_pickle=True)
    train_windows = {k: train_windows[k] for k in train_windows.files}
    all_kpis = sorted(set(train_windows["kpi_id"]))

    rng = np.random.default_rng(SEED)
    idx = build_stratified_train_sample(train_windows, all_kpis, rng)
    print(f"train sample: {len(idx)} windows ({train_windows['onset_in_horizon'].astype(bool)[idx].sum()} onset-overlapping)")

    y_preds = load_checkpoint()
    print(f"resuming: {len(y_preds)} already done" if y_preds else "starting fresh")
    remaining = [int(i) for i in idx if str(int(i)) not in y_preds]
    print(f"{len(remaining)} remaining of {len(idx)}")

    if remaining:
        from chronos import ChronosPipeline

        pipeline = ChronosPipeline.from_pretrained("amazon/chronos-t5-tiny", device_map="cpu", torch_dtype=torch.float32)
        t_start = time.time()
        for chunk_start in range(0, len(remaining), CHUNK_SIZE):
            t0 = time.time()
            chunk = remaining[chunk_start: chunk_start + CHUNK_SIZE]
            contexts = [torch.tensor(train_windows["context"][i], dtype=torch.float32) for i in chunk]
            forecast = pipeline.predict(context=contexts, prediction_length=HORIZON_STEPS, num_samples=NUM_SAMPLES)
            median_forecast = np.median(forecast.numpy(), axis=1)
            for i, pred in zip(chunk, median_forecast):
                y_preds[str(i)] = pred
            save_checkpoint(y_preds)
            done = len(y_preds)
            elapsed_total = time.time() - t_start
            rate = elapsed_total / (chunk_start + len(chunk))
            eta_min = rate * (len(remaining) - chunk_start - len(chunk)) / 60
            print(f"  chunk {chunk_start // CHUNK_SIZE + 1}: {done}/{len(idx)} done "
                  f"({time.time()-t0:.1f}s this chunk, ETA {eta_min:.1f} min)", flush=True)

    print("\nall windows processed, saving final output...")
    y_pred = np.array([y_preds[str(i)] for i in idx])
    np.savez(
        RESULTS_DIR / "train_forecasts_chronos.npz",
        kpi_id=train_windows["kpi_id"][idx], window_start_ts=train_windows["window_start_ts"][idx],
        onset_in_horizon=train_windows["onset_in_horizon"][idx],
        y_pred=y_pred, y_true=train_windows["horizon_value"][idx],
    )
    print(f"saved to {RESULTS_DIR / 'train_forecasts_chronos.npz'}")


if __name__ == "__main__":
    main()
