"""Paired, series-clustered bootstrap significance tests for claims made elsewhere in this
project ("Chronos and PatchTST MAE are ~tied", "detection rates are identical",
"SARIMA is not better than Persistence despite lower MAE", "the correction layer's F1 change
is a real improvement/regression").

Read-only: loads already-saved raw prediction .npz files and recomputes downstream via the
existing evaluation_metrics.py / correction_layer.py functions -- never overwrites any
existing result file. Writes exactly one new file: results/aiops18/bootstrap_significance_tests.csv.

Clustering unit is the KPI SERIES, not the window or episode: episodes/windows within one
series are not independent (they share the series' own noise characteristics, anomaly
pattern, and model fit quality), so treating them as i.i.d. would understate the true
sampling variance. Each bootstrap resample draws series with replacement (same size as the
original series pool) and recomputes the target statistic by pooling every episode/window
belonging to the resampled series (duplicated if a series is drawn more than once). All
paired comparisons (e.g. PatchTST vs Chronos, corrected vs raw) use the SAME per-iteration
series draw for both arms, which is what makes it a *paired* bootstrap rather than two
independent ones.
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))
from aiops18_preprocessing import load_merged, process_kpi  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from correction_layer import apply_correction_layer  # noqa: E402
from evaluation_metrics import (  # noqa: E402
    compute_detection_metrics,
    compute_false_alarm_metrics,
    extract_true_onset_timestamps,
)
from threshold_calibration import calibrate_threshold_f1  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "aiops18"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "processed_data" / "aiops18"
HORIZON_MIN, RESOLUTION_S, CONTEXT_STEPS, HORIZON_STEPS = 15, 60, 288, 15
MODELS = ["persistence", "sarima", "xgboost", "lstm", "patchtst", "chronos"]
N_BOOT = 10000  # standardized across every bootstrap analysis in this project so all reported CIs are comparable in precision
SEED = 42


def load_windows(split):
    d = np.load(PROCESSED_DIR / f"aiops18_{split}_windows_h{HORIZON_MIN}min_{RESOLUTION_S}s.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def load_raw_pred(model, split):
    path = RESULTS_DIR / f"{model}_{split}_{HORIZON_MIN}min_{RESOLUTION_S}s_predictions.npz"
    d = np.load(path, allow_pickle=True)
    return d


def get_calibration(all_kpis):
    merged = load_merged()
    calibs = {}
    for kpi_id in all_kpis:
        g = merged[merged["KPI ID"] == kpi_id]
        result = process_kpi(kpi_id, g)
        frame = result["frame"]
        train_vals = frame.loc[frame.split == "train", "value_z"].to_numpy()
        train_labels = frame.loc[frame.split == "train", "label"].to_numpy()
        calibs[kpi_id] = calibrate_threshold_f1(kpi_id, train_vals, train_labels)
    return calibs


def build_lookup(windows):
    return {(kpi, int(ts)): i for i, (kpi, ts) in enumerate(zip(windows["kpi_id"], windows["window_start_ts"]))}


def episode_lists(model, split, corrected, calibs, low_precision_kpis, windows, lookup, base_thresholds):
    """Returns (detection_episodes, false_alarm_episodes) restricted to the clean cohort,
    exactly matching the logic in run_aiops18_baselines.py / run_aiops18_correction.py.
    """
    d = load_raw_pred(model, split)
    kpi_id, ts, y_true, y_pred = d["kpi_id"], d["window_start_ts"], d["y_true"], d["y_pred"]

    if corrected:
        predicted_positive, _ = apply_correction_layer(
            kpi_id, ts, y_true, y_pred, HORIZON_STEPS, CONTEXT_STEPS, RESOLUTION_S, base_thresholds
        )
    else:
        predicted_positive = d["predicted_positive"]

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


def index_by_kpi_bool(episodes):
    by_kpi = defaultdict(list)
    for e in episodes:
        by_kpi[e["kpi_id"]].append(bool(e["detected"]))
    return by_kpi


def index_by_kpi_count(episodes):
    by_kpi = defaultdict(int)
    for e in episodes:
        by_kpi[e["kpi_id"]] += 1
    return by_kpi


def bootstrap_detection_rate_delta(det_by_kpi_A, det_by_kpi_B, series_pool, rng, n_boot=N_BOOT):
    n_series = len(series_pool)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.choice(series_pool, size=n_series, replace=True)
        detA, detB = [], []
        for k in draw:
            detA.extend(det_by_kpi_A.get(k, []))
            detB.extend(det_by_kpi_B.get(k, []))
        rateA = np.mean(detA) if detA else np.nan
        rateB = np.mean(detB) if detB else np.nan
        deltas[b] = rateA - rateB
    return deltas


def f1_from_pooled(det_list, fp_count):
    n = len(det_list)
    if n == 0:
        return np.nan
    tp = sum(det_list)
    recall = tp / n
    precision = tp / (tp + fp_count) if (tp + fp_count) > 0 else np.nan
    if np.isnan(precision) or (precision + recall) == 0:
        return np.nan
    return 2 * precision * recall / (precision + recall)


def bootstrap_f1_delta(det_raw_by_kpi, fa_raw_by_kpi, det_corr_by_kpi, fa_corr_by_kpi, series_pool, rng, n_boot=N_BOOT):
    n_series = len(series_pool)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.choice(series_pool, size=n_series, replace=True)
        det_raw, det_corr, fp_raw, fp_corr = [], [], 0, 0
        for k in draw:
            det_raw.extend(det_raw_by_kpi.get(k, []))
            det_corr.extend(det_corr_by_kpi.get(k, []))
            fp_raw += fa_raw_by_kpi.get(k, 0)
            fp_corr += fa_corr_by_kpi.get(k, 0)
        deltas[b] = f1_from_pooled(det_corr, fp_corr) - f1_from_pooled(det_raw, fp_raw)
    return deltas


def bootstrap_mae_delta(errs_by_kpi_A, errs_by_kpi_B, series_pool, rng, n_boot=N_BOOT):
    n_series = len(series_pool)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.choice(series_pool, size=n_series, replace=True)
        errsA = [errs_by_kpi_A[k] for k in draw]
        errsB = [errs_by_kpi_B[k] for k in draw]
        maeA = np.concatenate(errsA).mean()
        maeB = np.concatenate(errsB).mean()
        deltas[b] = maeA - maeB
    return deltas


def bootstrap_p_value(deltas):
    """Two-sided percentile-bootstrap p-value: twice the smaller tail proportion on either
    side of zero (the standard bootstrap analogue of a two-sided test), excluding undefined
    resamples. Used only for Holm-Bonferroni correction across a family of tests -- the CI
    itself (from ci_and_flag) is the primary reported result per test.
    """
    valid = deltas[~np.isnan(deltas)]
    if len(valid) == 0:
        return float("nan")
    p_le = np.mean(valid <= 0)
    p_ge = np.mean(valid >= 0)
    return float(min(2 * min(p_le, p_ge), 1.0))


def ci_and_flag(deltas):
    """Percentile CI, robust to resamples where the statistic is undefined (e.g. a resample
    with zero predicted positives, making precision -- and hence F1 -- undefined: NaN by the
    same convention used everywhere else in this analysis). Undefined resamples are excluded
    from the percentile/mean computation via nan* variants rather than silently corrupting the
    bound (plain np.percentile treats NaN as larger than every real value during sorting, which
    would push the upper percentile to NaN once >=2.5% of resamples are undefined, and comparing
    NaN <= 0 <= NaN is always False -- silently flipping the verdict to "distinguishable").
    n_undefined is reported so a CI built from too small a valid fraction is visibly suspect.
    Also returns a bootstrap p-value for later Holm-Bonferroni correction across a test family.
    """
    n_total = len(deltas)
    n_undefined = int(np.isnan(deltas).sum())
    n_valid = n_total - n_undefined
    if n_valid == 0:
        return float("nan"), float("nan"), float("nan"), "undefined (all resamples produced an undefined statistic)", n_undefined, float("nan")
    lo, hi = np.nanpercentile(deltas, [2.5, 97.5])
    point = float(np.nanmean(deltas))
    p_value = bootstrap_p_value(deltas)
    distinguishable = not (lo <= 0 <= hi)
    flag = "statistically distinguishable" if distinguishable else "not distinguishable from zero"
    if n_undefined / n_total > 0.05:
        flag += f" (CAUTION: {n_undefined}/{n_total} resamples undefined -- CI reliability reduced)"
    return point, lo, hi, flag, n_undefined, p_value


def main():
    rng = np.random.default_rng(SEED)
    rows = []

    windows_by_split = {"val": load_windows("val"), "test": load_windows("test")}
    lookup_by_split = {s: build_lookup(w) for s, w in windows_by_split.items()}
    all_kpis = sorted(set(windows_by_split["val"]["kpi_id"]))
    calibs = get_calibration(all_kpis)
    low_precision_kpis = {k for k, c in calibs.items() if c.low_precision_flag}
    clean_kpis = sorted(set(all_kpis) - low_precision_kpis)
    base_thresholds = {k: c.threshold for k, c in calibs.items()}
    print(f"{len(low_precision_kpis)}/{len(all_kpis)} low-precision -> {len(clean_kpis)}-series clean cohort")
    clean_pool = np.array(clean_kpis, dtype=object)

    # ---------------- Task 1: MAE(PatchTST) - MAE(Chronos), Chronos's matched cohort ----------------
    for split in ["val", "test"]:
        d_chr = load_raw_pred("chronos", split)
        chr_kpi, chr_ts, chr_true, chr_pred = d_chr["kpi_id"], d_chr["window_start_ts"], d_chr["y_true"], d_chr["y_pred"]
        n_chr = len(chr_kpi)
        chr_abs_err_2d = np.abs(chr_pred.reshape(n_chr, HORIZON_STEPS) - chr_true.reshape(n_chr, HORIZON_STEPS))
        chr_keys = set(zip(chr_kpi.tolist(), chr_ts.astype(np.int64).tolist()))

        d_pt = load_raw_pred("patchtst", split)
        pt_kpi, pt_ts, pt_true, pt_pred = d_pt["kpi_id"], d_pt["window_start_ts"], d_pt["y_true"], d_pt["y_pred"]
        n_pt_full = len(pt_kpi)
        pt_keys = list(zip(pt_kpi.tolist(), pt_ts.astype(np.int64).tolist()))
        pt_mask = np.array([k in chr_keys for k in pt_keys])
        pt_abs_err_2d_full = np.abs(pt_pred.reshape(n_pt_full, HORIZON_STEPS) - pt_true.reshape(n_pt_full, HORIZON_STEPS))
        pt_abs_err_2d = pt_abs_err_2d_full[pt_mask]
        pt_kpi_matched = pt_kpi[pt_mask]
        assert len(pt_kpi_matched) == n_chr, "patchtst did not match every chronos window"

        chr_errs_by_kpi = defaultdict(list)
        for k, row in zip(chr_kpi.tolist(), chr_abs_err_2d):
            chr_errs_by_kpi[k].append(row)
        chr_errs_by_kpi = {k: np.concatenate(v) for k, v in chr_errs_by_kpi.items()}

        pt_errs_by_kpi = defaultdict(list)
        for k, row in zip(pt_kpi_matched.tolist(), pt_abs_err_2d):
            pt_errs_by_kpi[k].append(row)
        pt_errs_by_kpi = {k: np.concatenate(v) for k, v in pt_errs_by_kpi.items()}

        series_pool = np.array(sorted(chr_errs_by_kpi.keys()), dtype=object)
        point_mae_pt = np.concatenate(list(pt_errs_by_kpi.values())).mean()
        point_mae_chr = np.concatenate(list(chr_errs_by_kpi.values())).mean()
        print(f"[sanity] {split}: point MAE patchtst(chronos-cohort)={point_mae_pt:.4f} chronos={point_mae_chr:.4f} "
              f"diff={point_mae_pt-point_mae_chr:.4f} ({len(series_pool)} series)")

        deltas = bootstrap_mae_delta(pt_errs_by_kpi, chr_errs_by_kpi, series_pool, rng)
        point, lo, hi, flag, n_undef, p_value = ci_and_flag(deltas)
        rows.append(dict(
            task="1_mae_patchtst_vs_chronos", comparison=f"MAE(PatchTST) - MAE(Chronos), Chronos-matched cohort, {split}",
            split=split, point_estimate=point, ci_low=lo, ci_high=hi, n_boot=N_BOOT, n_clusters=len(series_pool),
            n_undefined_resamples=n_undef, p_value=p_value, flag=flag,
        ))

    # ---------------- Task 2: detection-rate deltas ----------------
    det_episodes_cache = {}
    for model in MODELS:
        for split in ["val", "test"]:
            det, _ = episode_lists(model, split, False, calibs, low_precision_kpis,
                                    windows_by_split[split], lookup_by_split[split], base_thresholds)
            det_episodes_cache[(model, split, False)] = index_by_kpi_bool(det)

    pairs_task2 = [
        ("chronos", "persistence", "test"),
        ("chronos", "sarima", "test"),
        ("chronos", "xgboost", "test"),
        ("patchtst", "persistence", "test"),
        ("sarima", "persistence", "val"),
        ("sarima", "persistence", "test"),
    ]
    for model_a, model_b, split in pairs_task2:
        detA = det_episodes_cache[(model_a, split, False)]
        detB = det_episodes_cache[(model_b, split, False)]
        rateA_point = np.mean([v for vs in detA.values() for v in vs])
        rateB_point = np.mean([v for vs in detB.values() for v in vs])
        deltas = bootstrap_detection_rate_delta(detA, detB, clean_pool, rng)
        point, lo, hi, flag, n_undef, p_value = ci_and_flag(deltas)
        rows.append(dict(
            task="2_detection_rate_delta",
            comparison=f"detection_rate({model_a}) - detection_rate({model_b}), {split}",
            split=split, point_estimate=rateA_point - rateB_point, ci_low=lo, ci_high=hi,
            n_boot=N_BOOT, n_clusters=len(clean_pool), n_undefined_resamples=n_undef, p_value=p_value, flag=flag,
        ))

    # ---------------- Task 3: F1 delta (corrected - raw), 12 combos ----------------
    for model in MODELS:
        for split in ["val", "test"]:
            det_raw, fa_raw = episode_lists(model, split, False, calibs, low_precision_kpis,
                                              windows_by_split[split], lookup_by_split[split], base_thresholds)
            det_corr, fa_corr = episode_lists(model, split, True, calibs, low_precision_kpis,
                                               windows_by_split[split], lookup_by_split[split], base_thresholds)
            det_raw_by_kpi = index_by_kpi_bool(det_raw)
            det_corr_by_kpi = index_by_kpi_bool(det_corr)
            fa_raw_by_kpi = index_by_kpi_count(fa_raw)
            fa_corr_by_kpi = index_by_kpi_count(fa_corr)

            det_raw_flat = [v for vs in det_raw_by_kpi.values() for v in vs]
            det_corr_flat = [v for vs in det_corr_by_kpi.values() for v in vs]
            f1_raw_point = f1_from_pooled(det_raw_flat, sum(fa_raw_by_kpi.values()))
            f1_corr_point = f1_from_pooled(det_corr_flat, sum(fa_corr_by_kpi.values()))

            deltas = bootstrap_f1_delta(det_raw_by_kpi, fa_raw_by_kpi, det_corr_by_kpi, fa_corr_by_kpi, clean_pool, rng)
            point, lo, hi, flag, n_undef, p_value = ci_and_flag(deltas)
            rows.append(dict(
                task="3_f1_delta_corrected_minus_raw",
                comparison=f"F1(corrected) - F1(raw), {model}, {split}",
                split=split, point_estimate=f1_corr_point - f1_raw_point, ci_low=lo, ci_high=hi,
                n_boot=N_BOOT, n_clusters=len(clean_pool), n_undefined_resamples=n_undef, p_value=p_value, flag=flag,
            ))

    out = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "bootstrap_significance_tests.csv"
    out.to_csv(out_path, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 80)
    print()
    print(out.to_string(index=False))
    print(f"\nsaved to {out_path}")


if __name__ == "__main__":
    main()
