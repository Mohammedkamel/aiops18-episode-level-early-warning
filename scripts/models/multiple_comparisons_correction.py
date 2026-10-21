"""Holm-Bonferroni correction within each family of bootstrap tests reported across this
project's analyses. Each bootstrap analysis (one script's set of comparisons) is treated as
its own family, per the standard practice of correcting within a coherent set of related
hypotheses rather than across unrelated ones.

Read-only: reads the (now-standardized-to-10,000-resample) CSVs already produced by
bootstrap_significance_tests.py, ablation_flat_threshold.py, lead_time_false_alarm_analysis.py,
and matched_recall_comparison.py. Writes exactly one new file:
results/aiops18/multiple_comparisons_correction.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "aiops18"
ALPHA = 0.05


def holm_bonferroni(pvals: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """Standard step-down Holm procedure. Returns a boolean array (same order as input):
    True = rejects H0 (significant) after correction. Once a sorted p-value fails its
    threshold, it and every larger p-value in the family are marked not-significant, even if
    a later one would individually pass its own per-rank threshold -- that's the defining
    step-down property that makes Holm valid (not just per-test Bonferroni).
    """
    m = len(pvals)
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    reject_sorted = np.zeros(m, dtype=bool)
    for i, p in enumerate(sorted_p):
        if np.isnan(p):
            continue
        thresh = alpha / (m - i)
        if p <= thresh:
            reject_sorted[i] = True
        else:
            break
    reject = np.zeros(m, dtype=bool)
    reject[order] = reject_sorted
    return reject


def main():
    families = []

    bst = pd.read_csv(RESULTS_DIR / "bootstrap_significance_tests.csv")
    for task_name, family_label in [
        ("1_mae_patchtst_vs_chronos", "F1_MAE_patchtst_vs_chronos"),
        ("2_detection_rate_delta", "F2_detection_rate_identical_pairs"),
        ("3_f1_delta_corrected_minus_raw", "F3_cusum_f1_delta"),
    ]:
        sub = bst[bst.task == task_name].copy()
        sub["family"] = family_label
        sub["comparison_label"] = sub["comparison"]
        sub["n_boot"] = sub["n_boot"]
        families.append(sub[["family", "comparison_label", "split", "point_estimate", "ci_low", "ci_high", "p_value", "n_boot", "flag"]])

    abl = pd.read_csv(RESULTS_DIR / "ablation_flat_threshold.csv")
    abl_rows = abl.copy()
    abl_rows["family"] = "F4_flat_control_f1_delta"
    abl_rows["comparison_label"] = "F1(flat_control) - F1(raw), " + abl_rows["model"] + ", " + abl_rows["split"]
    abl_rows["point_estimate"] = abl_rows["flat_f1_delta_point"]
    abl_rows["ci_low"] = abl_rows["flat_f1_delta_ci_low"]
    abl_rows["ci_high"] = abl_rows["flat_f1_delta_ci_high"]
    abl_rows["p_value"] = abl_rows["flat_p_value"]
    abl_rows["n_boot"] = 10000
    abl_rows["flag"] = abl_rows["flat_flag"]
    families.append(abl_rows[["family", "comparison_label", "split", "point_estimate", "ci_low", "ci_high", "p_value", "n_boot", "flag"]])

    lt = pd.read_csv(RESULTS_DIR / "lead_time_and_false_alarm_rate_table.csv")
    lt_rows = lt.copy()
    lt_rows["family"] = "F5_ge10min_early_warning_recall_delta"
    lt_rows["comparison_label"] = "detection_rate_at_ge_10min(cusum) - (raw), " + lt_rows["model"] + ", " + lt_rows["split"]
    lt_rows["point_estimate"] = lt_rows["detection_rate_at_ge_10min_delta_point"]
    lt_rows["ci_low"] = lt_rows["detection_rate_at_ge_10min_delta_ci_low"]
    lt_rows["ci_high"] = lt_rows["detection_rate_at_ge_10min_delta_ci_high"]
    lt_rows["p_value"] = lt_rows["detection_rate_at_ge_10min_p_value"]
    lt_rows["n_boot"] = 10000
    lt_rows["flag"] = lt_rows["detection_rate_at_ge_10min_flag"]
    families.append(lt_rows[["family", "comparison_label", "split", "point_estimate", "ci_low", "ci_high", "p_value", "n_boot", "flag"]])

    mr = pd.read_csv(RESULTS_DIR / "matched_recall_comparison.csv")
    mr_rows = mr.copy()
    mr_rows["family"] = "F6_matched_recall_false_alarm_rate_delta"
    mr_rows["comparison_label"] = "false_alarm_rate(flat_matched) - (cusum), " + mr_rows["model"] + ", " + mr_rows["split"]
    mr_rows["point_estimate"] = mr_rows["false_alarm_rate_delta_point"]
    mr_rows["ci_low"] = mr_rows["false_alarm_rate_delta_ci_low"]
    mr_rows["ci_high"] = mr_rows["false_alarm_rate_delta_ci_high"]
    mr_rows["p_value"] = mr_rows["p_value"]
    mr_rows["n_boot"] = mr_rows["n_boot"]
    mr_rows["flag"] = mr_rows["flag"]
    families.append(mr_rows[["family", "comparison_label", "split", "point_estimate", "ci_low", "ci_high", "p_value", "n_boot", "flag"]])

    all_rows = pd.concat(families, ignore_index=True)
    all_rows["significant_before_correction"] = all_rows["flag"].str.startswith("statistically distinguishable")

    all_rows["holm_significant"] = False
    all_rows["holm_alpha_threshold"] = np.nan
    for family_label, group in all_rows.groupby("family"):
        idx = group.index
        pvals = group["p_value"].to_numpy()
        reject = holm_bonferroni(pvals)
        all_rows.loc[idx, "holm_significant"] = reject
        m = len(pvals)
        order = np.argsort(pvals)
        rank_of = {orig_pos: rank for rank, orig_pos in enumerate(order)}
        thresholds = np.array([ALPHA / (m - rank_of[i]) for i in range(m)])
        all_rows.loc[idx, "holm_alpha_threshold"] = thresholds

    out_path = RESULTS_DIR / "multiple_comparisons_correction.csv"
    all_rows.to_csv(out_path, index=False)

    pd.set_option("display.width", 300)
    pd.set_option("display.max_colwidth", 70)
    print("\n=== Per-family summary ===")
    summary = all_rows.groupby("family").agg(
        n_tests=("p_value", "size"),
        n_sig_before=("significant_before_correction", "sum"),
        n_sig_after_holm=("holm_significant", "sum"),
    )
    print(summary.to_string())

    print("\n=== Tests significant before correction but NOT after Holm ===")
    lost = all_rows[all_rows.significant_before_correction & ~all_rows.holm_significant]
    if len(lost):
        print(lost[["family", "comparison_label", "split", "p_value", "holm_alpha_threshold"]].to_string(index=False))
    else:
        print("none")

    print("\n=== Tests surviving Holm correction ===")
    survived = all_rows[all_rows.holm_significant]
    if len(survived):
        print(survived[["family", "comparison_label", "split", "point_estimate", "p_value", "holm_alpha_threshold"]].to_string(index=False))
    else:
        print("none")

    print(f"\nsaved to {out_path}")


if __name__ == "__main__":
    main()
