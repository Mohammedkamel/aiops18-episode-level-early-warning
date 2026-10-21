"""Aggregate lead-time distribution and normalized false-alarm-rate table, raw vs.
CUSUM-corrected (and flat-control, for the false-alarm normalization) across all six models
and both splits.

Read-only: reuses shared helpers from bootstrap_significance_tests.py, reads the already-saved
raw prediction .npz files and the already-computed ablation_flat_threshold.csv, and writes
exactly one new file: results/aiops18/lead_time_and_false_alarm_rate_table.csv.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap_significance_tests import (  # noqa: E402
    HORIZON_STEPS,
    MODELS,
    build_lookup,
    ci_and_flag,
    episode_lists,
    get_calibration,
    load_raw_pred,
    load_windows,
)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "aiops18"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "processed_data" / "aiops18"
MANIFEST_PATH = PROCESSED_DIR / "aiops18_manifest.json"
N_BOOT = 10000  # standardized across every bootstrap analysis in this project
SEED = 42

LEAD_THRESHOLDS = {"ge_15min": 900, "ge_10min": 600, "ge_5min": 300}


def load_manifest_stride_seconds():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    out = {}
    for kpi_id, info in manifest["kpis"].items():
        out[kpi_id] = info["stride_steps"] * info["modal_interval_seconds"]
    return out


def lead_time_stats(episodes):
    detected = [e for e in episodes if e["detected"]]
    n_episodes = len(episodes)
    n_detected = len(detected)
    leads = [e["lead_time_seconds"] for e in detected]
    out = dict(n_episodes=n_episodes, n_detected=n_detected)
    if leads:
        out["lead_time_median_seconds"] = float(np.median(leads))
        out["lead_time_iqr_low_seconds"] = float(np.percentile(leads, 25))
        out["lead_time_iqr_high_seconds"] = float(np.percentile(leads, 75))
    else:
        out["lead_time_median_seconds"] = None
        out["lead_time_iqr_low_seconds"] = None
        out["lead_time_iqr_high_seconds"] = None
    for name, thresh in LEAD_THRESHOLDS.items():
        n_ge = sum(1 for lt in leads if lt >= thresh)
        out[f"frac_of_detected_{name}"] = n_ge / n_detected if n_detected else float("nan")
        out[f"n_episodes_{name}"] = n_ge
        out[f"detection_rate_at_{name}"] = n_ge / n_episodes if n_episodes else float("nan")
    n_late = sum(1 for lt in leads if lt <= 0)
    out["frac_of_detected_at_or_after_onset"] = n_late / n_detected if n_detected else float("nan")
    out["n_episodes_at_or_after_onset"] = n_late
    return out


def index_by_kpi_lead_ge(episodes, threshold):
    by_kpi = defaultdict(list)
    for e in episodes:
        flag = bool(e["detected"] and e["lead_time_seconds"] is not None and e["lead_time_seconds"] >= threshold)
        by_kpi[e["kpi_id"]].append(flag)
    return by_kpi


def bootstrap_rate_delta(by_kpi_A, by_kpi_B, series_pool, rng, n_boot=N_BOOT):
    n_series = len(series_pool)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.choice(series_pool, size=n_series, replace=True)
        listA, listB = [], []
        for k in draw:
            listA.extend(by_kpi_A.get(k, []))
            listB.extend(by_kpi_B.get(k, []))
        rateA = np.mean(listA) if listA else np.nan
        rateB = np.mean(listB) if listB else np.nan
        deltas[b] = rateA - rateB
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

    summary_csv = pd.read_csv(RESULTS_DIR / "model_comparison_summary.csv")
    ablation_csv = pd.read_csv(RESULTS_DIR / "ablation_flat_threshold.csv")

    def existing_lead_row(model_name, split):
        r = summary_csv[(summary_csv.model_name == model_name) & (summary_csv.split == split)]
        return r.iloc[0] if len(r) else None

    rows = []
    for model in MODELS:
        for split in ["val", "test"]:
            windows, lookup = windows_by_split[split], lookup_by_split[split]

            det_raw, fa_raw = episode_lists(model, split, False, calibs, low_precision_kpis, windows, lookup, base_thresholds)
            det_cusum, fa_cusum = episode_lists(model, split, True, calibs, low_precision_kpis, windows, lookup, base_thresholds)

            stats_raw = lead_time_stats(det_raw)
            stats_cusum = lead_time_stats(det_cusum)

            # confirm against existing summary CSV fields (raw + corrected rows)
            existing_raw = existing_lead_row(model, split)
            existing_cusum = existing_lead_row(f"{model}_corrected", split)

            def close(a, b, tol=1e-6):
                if a is None or b is None or (isinstance(b, float) and np.isnan(b)):
                    return a is None and (b is None or (isinstance(b, float) and np.isnan(b)))
                return abs(a - b) < tol

            raw_match = close(stats_raw["lead_time_median_seconds"], existing_raw.lead_time_median_seconds if existing_raw is not None else None)
            cusum_match = close(stats_cusum["lead_time_median_seconds"], existing_cusum.lead_time_median_seconds if existing_cusum is not None else None)
            if not raw_match:
                print(f"  MISMATCH raw {model}/{split}: recomputed={stats_raw['lead_time_median_seconds']} "
                      f"existing={existing_raw.lead_time_median_seconds if existing_raw is not None else 'N/A'}")
            if not cusum_match:
                print(f"  MISMATCH cusum {model}/{split}: recomputed={stats_cusum['lead_time_median_seconds']} "
                      f"existing={existing_cusum.lead_time_median_seconds if existing_cusum is not None else 'N/A'}")

            # KPI-days monitored: sum over evaluated (clean-cohort) windows of that window's
            # own KPI's stride, in day units. Same window population for raw/cusum (correction
            # layer doesn't change which windows are evaluated), so this denominator is shared.
            d = load_raw_pred(model, split)
            kpi_id_all = d["kpi_id"]
            clean_win_mask = ~np.isin(kpi_id_all, list(low_precision_kpis))
            kpi_id_clean = kpi_id_all[clean_win_mask]
            kpi_days = sum(stride_seconds_by_kpi[k] for k in kpi_id_clean) / 86400.0
            n_windows_clean = int(clean_win_mask.sum())

            ablation_row = ablation_csv[(ablation_csv.model == model) & (ablation_csv.split == split)].iloc[0]
            fp_raw, fp_cusum, fp_flat = int(ablation_row.raw_n_false_alarm_episodes), int(ablation_row.cusum_n_false_alarm_episodes), int(ablation_row.flat_n_false_alarm_episodes)

            row = dict(model=model, split=split)
            for cond, stats in [("raw", stats_raw), ("cusum", stats_cusum)]:
                for k, v in stats.items():
                    row[f"{cond}_{k}"] = v
            row["lead_time_median_raw_matches_existing_csv"] = raw_match
            row["lead_time_median_cusum_matches_existing_csv"] = cusum_match
            row["n_windows_clean_cohort"] = n_windows_clean
            row["kpi_days_monitored"] = kpi_days
            row["fp_raw"] = fp_raw
            row["fp_cusum"] = fp_cusum
            row["fp_flat"] = fp_flat
            row["false_alarms_per_kpi_day_raw"] = fp_raw / kpi_days
            row["false_alarms_per_kpi_day_cusum"] = fp_cusum / kpi_days
            row["false_alarms_per_kpi_day_flat"] = fp_flat / kpi_days

            # bootstrap: detection_rate_at_ge_10min, raw vs cusum
            by_kpi_raw_ge10 = index_by_kpi_lead_ge(det_raw, 600)
            by_kpi_cusum_ge10 = index_by_kpi_lead_ge(det_cusum, 600)
            point_raw_ge10 = np.mean([v for vs in by_kpi_raw_ge10.values() for v in vs])
            point_cusum_ge10 = np.mean([v for vs in by_kpi_cusum_ge10.values() for v in vs])
            deltas = bootstrap_rate_delta(by_kpi_cusum_ge10, by_kpi_raw_ge10, clean_pool, rng)
            boot_point, boot_lo, boot_hi, boot_flag, n_undef, boot_p = ci_and_flag(deltas)
            row["detection_rate_at_ge_10min_raw"] = point_raw_ge10
            row["detection_rate_at_ge_10min_cusum"] = point_cusum_ge10
            row["detection_rate_at_ge_10min_delta_point"] = point_cusum_ge10 - point_raw_ge10
            row["detection_rate_at_ge_10min_delta_ci_low"] = boot_lo
            row["detection_rate_at_ge_10min_delta_ci_high"] = boot_hi
            row["detection_rate_at_ge_10min_p_value"] = boot_p
            row["detection_rate_at_ge_10min_flag"] = boot_flag
            row["detection_rate_at_ge_10min_n_undefined_resamples"] = n_undef

            rows.append(row)

    out = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "lead_time_and_false_alarm_rate_table.csv"
    out.to_csv(out_path, index=False)

    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", None)
    print("\n=== Lead-time distribution ===")
    print(out[["model", "split",
                "raw_lead_time_median_seconds", "raw_frac_of_detected_ge_10min", "raw_frac_of_detected_at_or_after_onset",
                "cusum_lead_time_median_seconds", "cusum_frac_of_detected_ge_10min", "cusum_frac_of_detected_at_or_after_onset"]].to_string(index=False))

    print("\n=== False alarms per KPI-day ===")
    print(out[["model", "split", "kpi_days_monitored", "fp_raw", "fp_cusum", "fp_flat",
                "false_alarms_per_kpi_day_raw", "false_alarms_per_kpi_day_cusum", "false_alarms_per_kpi_day_flat"]].to_string(index=False))

    print("\n=== Bootstrap: detection_rate_at_ge_10min, cusum - raw ===")
    print(out[["model", "split", "detection_rate_at_ge_10min_raw", "detection_rate_at_ge_10min_cusum",
                "detection_rate_at_ge_10min_delta_point", "detection_rate_at_ge_10min_delta_ci_low",
                "detection_rate_at_ge_10min_delta_ci_high", "detection_rate_at_ge_10min_flag"]].to_string(index=False))

    print(f"\nsaved to {out_path}")


if __name__ == "__main__":
    main()
