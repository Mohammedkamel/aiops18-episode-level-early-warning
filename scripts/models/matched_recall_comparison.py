"""Matched-recall CUSUM vs. static-threshold comparison: the test of whether CUSUM's
change-point *timing* is more efficient than a static threshold, not just less aggressive.

The earlier ablation (ablation_flat_threshold.py) compared CUSUM against a flat reduction
using the SAME 20% factor, which conflates "how much more sensitive" with "when it's applied."
This instead finds, per model/split, the flat reduction factor that gives static thresholding
approximately the SAME recall CUSUM achieves, then compares false-alarm rates at that matched
operating point -- an apples-to-apples efficiency comparison.

Read-only: reuses shared helpers from bootstrap_significance_tests.py, reads the already-saved
raw prediction .npz files, and writes exactly one new file:
results/aiops18/matched_recall_comparison.csv.
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
    get_calibration,
    index_by_kpi_count,
    load_raw_pred,
    load_windows,
)
from evaluation_metrics import compute_detection_metrics, compute_false_alarm_metrics, extract_true_onset_timestamps  # noqa: E402
from lead_time_false_alarm_analysis import load_manifest_stride_seconds  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "aiops18"
N_BOOT = 10000
SEED = 42
REDUCTION_GRID = np.round(np.arange(0.0, 0.96, 0.01), 4)  # 0% to 95% reduction, 1% steps


def prepare_flat_sweep_inputs(model, split, calibs, low_precision_kpis, windows, lookup, base_thresholds):
    """Everything about a model/split that does NOT depend on the reduction factor r --
    computed once and reused across all REDUCTION_GRID points, instead of redone from scratch
    per grid point (a ~96x reduction in redundant window-join work for the sweep below).
    """
    d = load_raw_pred(model, split)
    kpi_id, ts, y_pred = d["kpi_id"], d["window_start_ts"], d["y_pred"]
    n = len(kpi_id)
    max_abs_pred = np.max(np.abs(y_pred.reshape(n, HORIZON_STEPS)), axis=1)
    base_threshold_arr = np.array([base_thresholds[k] for k in kpi_id])

    orig_idx = np.array([lookup[(kpi_id[i], int(ts[i]))] for i in range(n)])
    context_label_last = windows["context_label"][orig_idx, -1]
    horizon_label = windows["horizon_label"][orig_idx]
    onset_in_horizon = windows["onset_in_horizon"][orig_idx].astype(bool)
    true_onset_ts = extract_true_onset_timestamps(ts, CONTEXT_STEPS, HORIZON_STEPS, RESOLUTION_S, context_label_last, horizon_label)
    clean_mask = ~np.isin(kpi_id, list(low_precision_kpis))

    return dict(
        kpi_id=kpi_id[clean_mask], ts=ts[clean_mask], max_abs_pred=max_abs_pred[clean_mask],
        base_threshold_arr=base_threshold_arr[clean_mask], true_onset_ts=true_onset_ts[clean_mask],
        onset_in_horizon=onset_in_horizon[clean_mask],
    )


def flat_episodes_at_reduction(prepared, reduction):
    predicted_positive = prepared["max_abs_pred"] > prepared["base_threshold_arr"] * (1 - reduction)
    det = compute_detection_metrics(prepared["kpi_id"], prepared["ts"], CONTEXT_STEPS, HORIZON_STEPS, RESOLUTION_S,
                                     prepared["true_onset_ts"], predicted_positive)
    fa = compute_false_alarm_metrics(prepared["kpi_id"], prepared["ts"], CONTEXT_STEPS, HORIZON_STEPS, RESOLUTION_S,
                                      prepared["onset_in_horizon"], predicted_positive)
    return det.episodes, fa.episodes


def bootstrap_false_alarm_rate_delta(fa_flat_by_kpi, fa_cusum_by_kpi, kpi_days_per_series, series_pool, rng, n_boot=N_BOOT):
    """Paired series-clustered bootstrap on false-alarm RATE (per KPI-day) delta =
    flat_matched - cusum, at the already-matched (fixed) operating point. Each resample
    recomputes both the false-alarm counts AND the kpi-days denominator from the SAME drawn
    series set, so duplicated series correctly inflate both numerator and denominator together.

    kpi_days_per_series[k] must already be that series' OWN total monitored days within this
    model/split's evaluated window population (window_count_for_k * stride_seconds_for_k /
    86400) -- i.e. computed the same way, and in the same units, as the naive full-sample
    kpi_days_total this is meant to reproduce under resampling with no duplication. Passing a
    bare per-KPI stride constant here (seconds, not weighted by that series' window count) was
    the bug this replaces: it made each resample's denominator ~4-5 orders of magnitude smaller
    than the true one, since it neither multiplied by window count nor converted seconds->days.
    """
    n_series = len(series_pool)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.choice(series_pool, size=n_series, replace=True)
        fp_flat = fp_cusum = 0
        kpi_days = 0.0
        for k in draw:
            fp_flat += fa_flat_by_kpi.get(k, 0)
            fp_cusum += fa_cusum_by_kpi.get(k, 0)
            kpi_days += kpi_days_per_series.get(k, 0.0)
        if kpi_days <= 0:
            deltas[b] = np.nan
            continue
        deltas[b] = (fp_flat - fp_cusum) / kpi_days
    return deltas


def main():
    rng = np.random.default_rng(SEED)
    stride_seconds_by_kpi = load_manifest_stride_seconds()

    windows_by_split = {"val": load_windows("val"), "test": load_windows("test")}
    lookup_by_split = {s: build_lookup(w) for s, w in windows_by_split.items()}
    all_kpis = sorted(set(windows_by_split["val"]["kpi_id"]))
    calibs = get_calibration(all_kpis)
    low_precision_kpis = {k for k, c in calibs.items() if c.low_precision_flag}
    clean_kpis = sorted(set(all_kpis) - low_precision_kpis)
    base_thresholds = {k: c.threshold for k, c in calibs.items()}
    clean_pool = np.array(clean_kpis, dtype=object)
    print(f"{len(low_precision_kpis)}/{len(all_kpis)} low-precision -> {len(clean_kpis)}-series clean cohort")

    rows = []
    for model in MODELS:
        for split in ["val", "test"]:
            windows, lookup = windows_by_split[split], lookup_by_split[split]

            det_cusum, fa_cusum = episode_lists(model, split, True, calibs, low_precision_kpis, windows, lookup, base_thresholds)
            n_episodes = len(det_cusum)
            recall_cusum = sum(e["detected"] for e in det_cusum) / n_episodes
            fp_cusum_total = len(fa_cusum)

            # sweep the flat reduction factor to match CUSUM's recall
            prepared = prepare_flat_sweep_inputs(model, split, calibs, low_precision_kpis, windows, lookup, base_thresholds)
            best_r, best_diff, best_recall_flat, best_det_flat, best_fa_flat = None, np.inf, None, None, None
            for r in REDUCTION_GRID:
                det_flat, fa_flat = flat_episodes_at_reduction(prepared, r)
                recall_flat = sum(e["detected"] for e in det_flat) / n_episodes
                diff = abs(recall_flat - recall_cusum)
                if diff < best_diff:
                    best_r, best_diff, best_recall_flat = float(r), diff, recall_flat
                    best_det_flat, best_fa_flat = det_flat, fa_flat

            fp_flat_matched = len(best_fa_flat)
            print(f"{model}/{split}: recall_cusum={recall_cusum:.4f} -> matched r={best_r:.2f} "
                  f"recall_flat={best_recall_flat:.4f} (|diff|={best_diff:.4f}), "
                  f"fp_cusum={fp_cusum_total} fp_flat_matched={fp_flat_matched}")

            fa_cusum_by_kpi = index_by_kpi_count(fa_cusum)
            fa_flat_by_kpi = index_by_kpi_count(best_fa_flat)

            d = load_raw_pred(model, split)
            clean_win_mask = ~np.isin(d["kpi_id"], list(low_precision_kpis))
            clean_kpi_ids_this_pop = d["kpi_id"][clean_win_mask]
            kpi_window_counts = pd.Series(clean_kpi_ids_this_pop).value_counts().to_dict()
            kpi_days_per_series = {
                k: kpi_window_counts.get(k, 0) * stride_seconds_by_kpi[k] / 86400.0 for k in clean_kpis
            }
            kpi_days_total = sum(kpi_days_per_series.values())
            assert abs(kpi_days_total - sum(stride_seconds_by_kpi[k] for k in clean_kpi_ids_this_pop) / 86400.0) < 1e-6, \
                "kpi_days_per_series must reproduce the naive full-sample total exactly"

            fa_rate_cusum = fp_cusum_total / kpi_days_total
            fa_rate_flat_matched = fp_flat_matched / kpi_days_total

            deltas = bootstrap_false_alarm_rate_delta(fa_flat_by_kpi, fa_cusum_by_kpi, kpi_days_per_series, clean_pool, rng)
            point, lo, hi, flag, n_undef, p_value = ci_and_flag(deltas)

            rows.append(dict(
                model=model, split=split, n_episodes=n_episodes,
                recall_cusum=recall_cusum, matched_reduction_factor=best_r, recall_flat_matched=best_recall_flat,
                recall_match_abs_diff=best_diff,
                fp_cusum=fp_cusum_total, fp_flat_matched=fp_flat_matched,
                kpi_days_monitored=kpi_days_total,
                false_alarm_rate_per_kpi_day_cusum=fa_rate_cusum,
                false_alarm_rate_per_kpi_day_flat_matched=fa_rate_flat_matched,
                false_alarm_rate_delta_point=fa_rate_flat_matched - fa_rate_cusum,
                false_alarm_rate_delta_bootstrap_mean=point,
                false_alarm_rate_delta_ci_low=lo, false_alarm_rate_delta_ci_high=hi,
                p_value=p_value, n_undefined_resamples=n_undef, n_boot=N_BOOT, flag=flag,
            ))

    out = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "matched_recall_comparison.csv"
    out.to_csv(out_path, index=False)

    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", None)
    print()
    print(out[["model", "split", "recall_cusum", "matched_reduction_factor", "recall_flat_matched",
               "false_alarm_rate_per_kpi_day_cusum", "false_alarm_rate_per_kpi_day_flat_matched",
               "false_alarm_rate_delta_point", "false_alarm_rate_delta_ci_low", "false_alarm_rate_delta_ci_high", "flag"]].to_string(index=False))
    print(f"\nsaved to {out_path}")


if __name__ == "__main__":
    main()
