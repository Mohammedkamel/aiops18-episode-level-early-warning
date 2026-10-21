"""Model-specific threshold calibration sensitivity check: does the episode-identical /
accuracy-detection-decoupling finding survive replacing the shared observed-value threshold
(calibrate_threshold_f1 on raw |value_z| vs. train labels, same threshold reused for all six
models) with a threshold calibrated separately per (model x KPI) on that model's OWN
train-set forecast score (max|forecast| over the horizon, exactly the detection score used in
production) vs. train labels?

Scoped to Persistence, SARIMA, Chronos -- the three models in the episode-identical check.

Reads: results/aiops18/train_forecasts_persistence_sarima.npz (all 41 KPIs, full train
population) and train_forecasts_chronos.npz (stratified train sample, same design as the
production val/test evaluation). Re-flags the ALREADY-SAVED val/test raw predictions under
the new per-model thresholds -- no re-running of val/test inference, only the threshold and
(consequently) the low-precision exclusion cohort change.

Writes results/aiops18/model_specific_calibration_results.csv (per-model per-KPI calibration)
and prints the Table-1-style detection-rate comparison and the episode-identical check.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap_significance_tests import CONTEXT_STEPS, HORIZON_STEPS, RESOLUTION_S, build_lookup, load_raw_pred, load_windows  # noqa: E402
from evaluation_metrics import compute_detection_metrics, extract_true_onset_timestamps  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "aiops18"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "processed_data" / "aiops18"
MODELS = ["persistence", "sarima", "chronos"]


def calibrate_model_specific(kpi_id: str, scores: np.ndarray, onset_labels: np.ndarray, n_candidates: int = 200) -> dict:
    """Mirrors threshold_calibration.calibrate_threshold_f1's sweep/tie-break exactly, scoring
    max|forecast| per window against onset_in_horizon instead of |value_z| against the raw
    per-timestep label."""
    labels = onset_labels.astype(bool)
    candidates = np.unique(np.quantile(scores, np.linspace(0.0, 0.999, n_candidates)))
    best = None
    for tau in candidates:
        pred_pos = scores > tau
        tp = int(np.sum(pred_pos & labels))
        fp = int(np.sum(pred_pos & ~labels))
        fn = int(np.sum(~pred_pos & labels))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        if best is None or f1 > best["f1"]:
            best = dict(threshold=float(tau), precision=precision, recall=recall, f1=f1, n_windows=len(scores))
    return dict(kpi_id=kpi_id, model=None, **best, low_precision_flag=best["precision"] < 0.30)


def main():
    ps = np.load(RESULTS_DIR / "train_forecasts_persistence_sarima.npz", allow_pickle=True)
    chr_ = np.load(RESULTS_DIR / "train_forecasts_chronos.npz", allow_pickle=True)

    # Calibration ground truth: calibrate_threshold_f1 (the original, observed-value version)
    # scores every raw TRAIN timestep against the dense per-timestep `label` (any anomalous
    # point), not just the sparse onset marker -- onset_in_horizon is ~3.4x sparser (0.56% vs
    # 1.90% positive rate) and is the wrong ground truth to hold the comparison fair against.
    # Reconstruct "any true anomalous point in this window's horizon" per window by joining
    # back to the original train windows' horizon_label field.
    train_windows = np.load(PROCESSED_DIR / "aiops18_train_windows_h15min_60s.npz", allow_pickle=True)
    any_label_lookup = {}
    tw_kpi, tw_ts, tw_horizon_label = train_windows["kpi_id"], train_windows["window_start_ts"], train_windows["horizon_label"]
    tw_any_label = tw_horizon_label.any(axis=1)
    for k, t, a in zip(tw_kpi.tolist(), tw_ts.astype(np.int64).tolist(), tw_any_label.tolist()):
        any_label_lookup[(k, t)] = a

    def any_label_for(kpi_arr, ts_arr):
        return np.array([any_label_lookup[(k, int(t))] for k, t in zip(kpi_arr.tolist(), ts_arr.tolist())])

    ps_any_label = any_label_for(ps["kpi_id"], ps["window_start_ts"])
    chr_any_label = any_label_for(chr_["kpi_id"], chr_["window_start_ts"])
    print(f"[sanity] persistence/sarima pop: {ps_any_label.mean()*100:.2f}% any-label positive "
          f"(vs {ps['onset_in_horizon'].astype(bool).mean()*100:.2f}% onset-only)")
    print(f"[sanity] chronos sample: {chr_any_label.mean()*100:.2f}% any-label positive "
          f"(vs {chr_['onset_in_horizon'].astype(bool).mean()*100:.2f}% onset-only)")

    scores = {
        "persistence": (ps["kpi_id"], np.max(np.abs(ps["y_pred_persistence"]), axis=1), ps_any_label),
        "sarima": (ps["kpi_id"], np.max(np.abs(ps["y_pred_sarima"]), axis=1), ps_any_label),
        "chronos": (chr_["kpi_id"], np.max(np.abs(chr_["y_pred"]), axis=1), chr_any_label),
    }

    all_kpis = sorted(set(ps["kpi_id"]))
    calib_rows = []
    calibs_by_model = {}
    for model in MODELS:
        kpi_arr, score_arr, onset_arr = scores[model]
        calibs = {}
        for kpi in all_kpis:
            mask = kpi_arr == kpi
            if mask.sum() < 5:
                continue
            c = calibrate_model_specific(kpi, score_arr[mask], onset_arr[mask])
            c["model"] = model
            calibs[kpi] = c
            calib_rows.append(c)
        calibs_by_model[model] = calibs

    calib_df = pd.DataFrame(calib_rows)
    calib_df.to_csv(RESULTS_DIR / "model_specific_calibration_results.csv", index=False)

    print("=== Model-specific clean cohorts ===")
    clean_by_model = {}
    for model in MODELS:
        low_prec = {k for k, c in calibs_by_model[model].items() if c["low_precision_flag"]}
        clean = sorted(set(all_kpis) - low_prec)
        clean_by_model[model] = set(clean)
        print(f"{model}: {len(low_prec)}/{len(all_kpis)} low-precision -> {len(clean)}-series clean cohort")

    intersection_clean = clean_by_model["persistence"] & clean_by_model["sarima"] & clean_by_model["chronos"]
    print(f"\nintersection of all three models' clean cohorts: {len(intersection_clean)} series")

    print("\n=== Table 1 (model-specific calibration): detection rate, own clean cohort per model ===")
    windows_by_split = {"val": load_windows("val"), "test": load_windows("test")}
    lookup_by_split = {s: build_lookup(w) for s, w in windows_by_split.items()}

    det_episodes_by_model_split = {}
    rows = []
    for model in MODELS:
        base_thresholds = {k: c["threshold"] for k, c in calibs_by_model[model].items()}
        own_clean = clean_by_model[model]
        for split in ["val", "test"]:
            windows, lookup = windows_by_split[split], lookup_by_split[split]
            d = load_raw_pred(model, split)
            kpi_id, ts, y_pred = d["kpi_id"], d["window_start_ts"], d["y_pred"]
            n = len(kpi_id)
            max_abs_pred = np.max(np.abs(y_pred.reshape(n, HORIZON_STEPS)), axis=1)
            thr_arr = np.array([base_thresholds.get(k, np.inf) for k in kpi_id])
            predicted_positive = max_abs_pred > thr_arr

            orig_idx = np.array([lookup[(kpi_id[i], int(ts[i]))] for i in range(n)])
            context_label_last = windows["context_label"][orig_idx, -1]
            horizon_label = windows["horizon_label"][orig_idx]
            true_onset_ts = extract_true_onset_timestamps(ts, CONTEXT_STEPS, HORIZON_STEPS, RESOLUTION_S, context_label_last, horizon_label)

            clean_mask = np.isin(kpi_id, list(own_clean))
            det = compute_detection_metrics(kpi_id[clean_mask], ts[clean_mask], CONTEXT_STEPS, HORIZON_STEPS, RESOLUTION_S,
                                             true_onset_ts[clean_mask], predicted_positive[clean_mask])
            rows.append(dict(model=model, split=split, n_clean_series=len(own_clean),
                              n_episodes=det.n_episodes, n_detected=det.n_detected, detection_rate=det.detection_rate))

            # also compute on the INTERSECTION cohort for a directly comparable cross-model number
            clean_mask_int = np.isin(kpi_id, list(intersection_clean))
            det_int = compute_detection_metrics(kpi_id[clean_mask_int], ts[clean_mask_int], CONTEXT_STEPS, HORIZON_STEPS, RESOLUTION_S,
                                                  true_onset_ts[clean_mask_int], predicted_positive[clean_mask_int])
            rows[-1]["n_episodes_intersection_cohort"] = det_int.n_episodes
            rows[-1]["n_detected_intersection_cohort"] = det_int.n_detected
            rows[-1]["detection_rate_intersection_cohort"] = det_int.detection_rate

            det_episodes_by_model_split[(model, split)] = {(e["kpi_id"], e["onset_timestamp"]): e["detected"] for e in det_int.episodes}

    table1 = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(table1.to_string(index=False))
    table1.to_csv(RESULTS_DIR / "model_specific_calibration_table1.csv", index=False)

    print("\n=== Episode-identical check (test split, intersection clean cohort, model-specific calibration) ===")
    ep_p = det_episodes_by_model_split[("persistence", "test")]
    ep_s = det_episodes_by_model_split[("sarima", "test")]
    ep_c = det_episodes_by_model_split[("chronos", "test")]
    keys = sorted(set(ep_p) & set(ep_s) & set(ep_c))
    print(f"n episodes in intersection cohort (test): {len(keys)}")
    for name_a, ep_a, name_b, ep_b in [("chronos", ep_c, "persistence", ep_p), ("chronos", ep_c, "sarima", ep_s), ("sarima", ep_s, "persistence", ep_p)]:
        mismatches = [k for k in keys if ep_a[k] != ep_b[k]]
        n_det_a = sum(ep_a[k] for k in keys)
        n_det_b = sum(ep_b[k] for k in keys)
        print(f"{name_a} vs {name_b}: {len(mismatches)} mismatched of {len(keys)} (n_detected: {name_a}={n_det_a}, {name_b}={n_det_b})")

    print(f"\nsaved calibration to {RESULTS_DIR / 'model_specific_calibration_results.csv'}")
    print(f"saved table1 to {RESULTS_DIR / 'model_specific_calibration_table1.csv'}")


if __name__ == "__main__":
    main()
