"""Ablation: does CUSUM change-point *timing* matter, or would any uniform threshold
reduction produce the same effect as the correction layer?

Control condition: apply base_threshold*(1-THRESHOLD_REDUCTION) to EVERY window,
unconditionally -- no CUSUM, no change-point detection, no decay schedule, no maturation
gating. Same 20% reduction constant as the real correction layer, applied everywhere instead
of only near a detected sustained shift. Run through the same episode-level evaluation
pipeline and the same series-clustered paired bootstrap (this time on flat-control vs raw)
used for the real correction layer, so the two can be compared side by side.

Read-only relative to existing results: reuses shared helpers from
bootstrap_significance_tests.py, reads the already-saved raw prediction .npz files and the
already-computed CUSUM-corrected numbers (precision_f1_table.csv, bootstrap_significance_tests.csv)
for comparison, and writes exactly one new file: results/aiops18/ablation_flat_threshold.csv.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap_significance_tests import (  # noqa: E402
    CONTEXT_STEPS,
    HORIZON_STEPS,
    MODELS,
    RESOLUTION_S,
    build_lookup,
    ci_and_flag,
    episode_lists,
    f1_from_pooled,
    get_calibration,
    index_by_kpi_bool,
    index_by_kpi_count,
    load_raw_pred,
    load_windows,
)
from correction_layer import THRESHOLD_REDUCTION  # noqa: E402
from evaluation_metrics import (  # noqa: E402
    compute_detection_metrics,
    compute_false_alarm_metrics,
    extract_true_onset_timestamps,
)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "aiops18"
N_BOOT = 10000  # standardized across every bootstrap analysis in this project
SEED = 42


def flat_predicted_positive(y_pred, horizon_steps, kpi_id, base_thresholds, reduction=THRESHOLD_REDUCTION):
    """Unconditional flat threshold: base*(1-reduction) applied to every window, no CUSUM,
    no time structure at all -- the control condition."""
    n = len(kpi_id)
    y_pred_2d = y_pred.reshape(n, horizon_steps)
    max_abs_pred = np.max(np.abs(y_pred_2d), axis=1)
    flat_thresholds = np.array([base_thresholds[k] * (1 - reduction) for k in kpi_id])
    return max_abs_pred > flat_thresholds


def flat_episode_lists(model, split, calibs, low_precision_kpis, windows, lookup, base_thresholds):
    d = load_raw_pred(model, split)
    kpi_id, ts, y_true, y_pred = d["kpi_id"], d["window_start_ts"], d["y_true"], d["y_pred"]

    predicted_positive = flat_predicted_positive(y_pred, HORIZON_STEPS, kpi_id, base_thresholds)

    orig_idx = np.array([lookup[(kpi_id[i], int(ts[i]))] for i in range(len(kpi_id))])
    context_label_last = windows["context_label"][orig_idx, -1]
    horizon_label = windows["horizon_label"][orig_idx]
    onset_in_horizon = windows["onset_in_horizon"][orig_idx]

    true_onset_ts = extract_true_onset_timestamps(
        ts, CONTEXT_STEPS, HORIZON_STEPS, RESOLUTION_S, context_label_last, horizon_label
    )
    clean_mask = ~np.isin(kpi_id, list(low_precision_kpis))

    det = compute_detection_metrics(
        kpi_id[clean_mask], ts[clean_mask], CONTEXT_STEPS, HORIZON_STEPS, RESOLUTION_S,
        true_onset_ts[clean_mask], predicted_positive[clean_mask],
    )
    fa = compute_false_alarm_metrics(
        kpi_id[clean_mask], ts[clean_mask], CONTEXT_STEPS, HORIZON_STEPS, RESOLUTION_S,
        onset_in_horizon[clean_mask].astype(bool), predicted_positive[clean_mask],
    )
    return det.episodes, fa.episodes


def bootstrap_f1_delta_flat_vs_raw(det_raw_by_kpi, fa_raw_by_kpi, det_flat_by_kpi, fa_flat_by_kpi, series_pool, rng, n_boot=N_BOOT):
    n_series = len(series_pool)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.choice(series_pool, size=n_series, replace=True)
        det_raw, det_flat, fp_raw, fp_flat = [], [], 0, 0
        for k in draw:
            det_raw.extend(det_raw_by_kpi.get(k, []))
            det_flat.extend(det_flat_by_kpi.get(k, []))
            fp_raw += fa_raw_by_kpi.get(k, 0)
            fp_flat += fa_flat_by_kpi.get(k, 0)
        deltas[b] = f1_from_pooled(det_flat, fp_flat) - f1_from_pooled(det_raw, fp_raw)
    return deltas


def main():
    rng = np.random.default_rng(SEED)

    windows_by_split = {"val": load_windows("val"), "test": load_windows("test")}
    lookup_by_split = {s: build_lookup(w) for s, w in windows_by_split.items()}
    all_kpis = sorted(set(windows_by_split["val"]["kpi_id"]))
    calibs = get_calibration(all_kpis)
    low_precision_kpis = {k for k, c in calibs.items() if c.low_precision_flag}
    clean_kpis = sorted(set(all_kpis) - low_precision_kpis)
    base_thresholds = {k: c.threshold for k, c in calibs.items()}
    clean_pool = np.array(clean_kpis, dtype=object)
    print(f"{len(low_precision_kpis)}/{len(all_kpis)} low-precision -> {len(clean_kpis)}-series clean cohort, "
          f"flat reduction factor={THRESHOLD_REDUCTION}")

    precision_f1 = pd.read_csv(RESULTS_DIR / "precision_f1_table.csv")
    bootstrap_existing = pd.read_csv(RESULTS_DIR / "bootstrap_significance_tests.csv")

    def pf1_row(model, split, corrected):
        return precision_f1[
            (precision_f1.model == model) & (precision_f1.split == split) & (precision_f1.corrected == corrected)
        ].iloc[0]

    def cusum_ci_row(model, split):
        comp = f"F1(corrected) - F1(raw), {model}, {split}"
        return bootstrap_existing[
            (bootstrap_existing.task == "3_f1_delta_corrected_minus_raw") & (bootstrap_existing.comparison == comp)
        ].iloc[0]

    rows = []
    for model in MODELS:
        for split in ["val", "test"]:
            windows, lookup = windows_by_split[split], lookup_by_split[split]

            det_raw, fa_raw = episode_lists(model, split, False, calibs, low_precision_kpis, windows, lookup, base_thresholds)
            det_flat, fa_flat = flat_episode_lists(model, split, calibs, low_precision_kpis, windows, lookup, base_thresholds)

            det_raw_by_kpi = index_by_kpi_bool(det_raw)
            det_flat_by_kpi = index_by_kpi_bool(det_flat)
            fa_raw_by_kpi = index_by_kpi_count(fa_raw)
            fa_flat_by_kpi = index_by_kpi_count(fa_flat)

            det_raw_flat_list = [v for vs in det_raw_by_kpi.values() for v in vs]
            det_flat_flat_list = [v for vs in det_flat_by_kpi.values() for v in vs]
            n_fp_raw = sum(fa_raw_by_kpi.values())
            n_fp_flat = sum(fa_flat_by_kpi.values())
            f1_raw_point = f1_from_pooled(det_raw_flat_list, n_fp_raw)
            f1_flat_point = f1_from_pooled(det_flat_flat_list, n_fp_flat)

            deltas = bootstrap_f1_delta_flat_vs_raw(det_raw_by_kpi, fa_raw_by_kpi, det_flat_by_kpi, fa_flat_by_kpi, clean_pool, rng)
            # ci_and_flag's returned "point" is the mean of the bootstrap resample distribution,
            # which is NOT the same quantity as the naive full-sample delta (F1 is nonlinear in
            # TP/FP/FN, and undefined resamples are excluded via nanmean) -- they can differ by a
            # meaningful amount, not just rounding. bootstrap_significance_tests.py's Task 3 (the
            # CUSUM numbers this is compared against) reports the naive delta as point_estimate,
            # so this uses the same convention for a fair side-by-side comparison; only the CI
            # bounds come from the bootstrap distribution itself.
            _, flat_lo, flat_hi, flat_flag, flat_n_undef, flat_p_value = ci_and_flag(deltas)
            flat_point_bootstrap_mean = float(np.nanmean(deltas))

            pf1_raw = pf1_row(model, split, False)
            pf1_cusum = pf1_row(model, split, True)
            cusum_ci = cusum_ci_row(model, split)

            d_raw = load_raw_pred(model, split)
            clean_win_mask = ~np.isin(d_raw["kpi_id"], list(low_precision_kpis))
            n_windows_clean = int(clean_win_mask.sum())

            rows.append(dict(
                model=model, split=split,
                n_episodes=int(pf1_raw.n_episodes),
                raw_f1=float(pf1_raw.f1), raw_precision=float(pf1_raw.precision), raw_recall=float(pf1_raw.recall),
                raw_n_false_alarm_episodes=int(pf1_raw.fp),
                cusum_f1=float(pf1_cusum.f1), cusum_precision=float(pf1_cusum.precision), cusum_recall=float(pf1_cusum.recall),
                cusum_n_false_alarm_episodes=int(pf1_cusum.fp),
                cusum_f1_delta_point=float(cusum_ci.point_estimate), cusum_f1_delta_ci_low=float(cusum_ci.ci_low),
                cusum_f1_delta_ci_high=float(cusum_ci.ci_high), cusum_p_value=float(cusum_ci.p_value), cusum_flag=cusum_ci.flag,
                flat_f1=f1_flat_point, flat_n_false_alarm_episodes=n_fp_flat,
                flat_f1_delta_point=f1_flat_point - f1_raw_point,
                flat_f1_delta_point_bootstrap_mean=flat_point_bootstrap_mean,
                flat_f1_delta_ci_low=flat_lo, flat_f1_delta_ci_high=flat_hi, flat_p_value=flat_p_value,
                flat_flag=flat_flag, flat_n_undefined_resamples=flat_n_undef,
                n_windows_clean_cohort=n_windows_clean,
                false_alarm_episodes_flat_minus_cusum=int(n_fp_flat - pf1_cusum.fp),
                false_alarm_rate_per_1k_windows_raw=1000 * pf1_raw.fp / n_windows_clean,
                false_alarm_rate_per_1k_windows_cusum=1000 * pf1_cusum.fp / n_windows_clean,
                false_alarm_rate_per_1k_windows_flat=1000 * n_fp_flat / n_windows_clean,
            ))

    out = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "ablation_flat_threshold.csv"
    out.to_csv(out_path, index=False)

    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", None)
    print()
    print(out[["model", "split", "raw_f1", "cusum_f1", "cusum_f1_delta_point", "cusum_flag",
               "flat_f1", "flat_f1_delta_point", "flat_flag"]].to_string(index=False))
    print()
    print(out[["model", "split", "raw_n_false_alarm_episodes", "cusum_n_false_alarm_episodes",
               "flat_n_false_alarm_episodes", "false_alarm_episodes_flat_minus_cusum",
               "false_alarm_rate_per_1k_windows_cusum", "false_alarm_rate_per_1k_windows_flat"]].to_string(index=False))

    xgb_val = out[(out.model == "xgboost") & (out.split == "val")].iloc[0]
    print("\n=== XGBoost/val flag ===")
    print(f"CUSUM: delta={xgb_val.cusum_f1_delta_point:.4f} CI=[{xgb_val.cusum_f1_delta_ci_low:.4f}, "
          f"{xgb_val.cusum_f1_delta_ci_high:.4f}] -> {xgb_val.cusum_flag}")
    print(f"Flat:  delta={xgb_val.flat_f1_delta_point:.4f} CI=[{xgb_val.flat_f1_delta_ci_low:.4f}, "
          f"{xgb_val.flat_f1_delta_ci_high:.4f}] -> {xgb_val.flat_flag}")

    print(f"\nsaved to {out_path}")


if __name__ == "__main__":
    main()
