"""Diagnostic: does node_disk_write_io_consumption have unusually noisy/high-variance
baselines compared to other node-level fault types, which would explain why its detection
rate drops more sharply than others as the sigma threshold tightens (see Section 7 of
docs/aiops2022_detectability_methodology.md)?

For each event of the fault types under comparison, computes the baseline (10-min
pre-injection) mean and standard deviation per relevant KPI, plus the coefficient of
variation (std/mean) to compare noisiness on a scale that's comparable across KPIs with
very different units.

Output: results/aiops2022/aiops2022_node_baseline_variance_by_faulttype.csv
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))
from aiops2022_archive_readers import read_metric_node_from_faults_zip
from aiops2022_failure_taxonomy import translate
from aiops2022_groundtruth_loader import load_train_events

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aiops2022_node_detectability import NODE_FAULT_KPIS

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "aiops2022"
BASELINE_WINDOW_S = 600

COMPARISON_FAULT_TYPES = [
    "node_disk_write_io_consumption",
    "node_disk_read_io_consumption",
    "node_cpu_fault",
]


def main():
    events = load_train_events()
    events["failure_type"] = events["failure_type"].apply(translate)
    node_events = events[events["failure_type"].isin(COMPARISON_FAULT_TYPES)].copy()

    rows = []
    for (cloudbed, date_mmdd), group in node_events.groupby(["cloudbed", "date_mmdd"]):
        df_node = read_metric_node_from_faults_zip(int(cloudbed), date_mmdd)
        for _, ev in group.iterrows():
            kpis = NODE_FAULT_KPIS[ev["failure_type"]]
            df_this_node = df_node[df_node["cmdb_id"] == ev["cmdb_id"]]
            ts0 = int(ev["timestamp"])
            for kpi in kpis:
                df_kpi = df_this_node[df_this_node["kpi_name"] == kpi]
                baseline = df_kpi[
                    (df_kpi.timestamp >= ts0 - BASELINE_WINDOW_S) & (df_kpi.timestamp < ts0)
                ]
                if len(baseline) < 3:
                    continue
                mean, std = baseline["value"].mean(), baseline["value"].std()
                cv = std / mean if mean not in (0, None) and pd.notna(mean) and mean != 0 else float("nan")
                rows.append(
                    dict(
                        event_id=ev["event_id"],
                        failure_type=ev["failure_type"],
                        cmdb_id=ev["cmdb_id"],
                        kpi_name=kpi,
                        baseline_mean=mean,
                        baseline_std=std,
                        coefficient_of_variation=cv,
                    )
                )
        del df_node

    df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "aiops2022_node_baseline_variance_by_faulttype.csv"
    if out_path.exists():
        raise FileExistsError(f"{out_path} already exists — confirm with user before overwriting")
    df.to_csv(out_path, index=False)
    print(f"wrote {len(df)} rows -> {out_path}\n")

    summary = (
        df.groupby(["failure_type", "kpi_name"])
        .agg(
            n=("baseline_std", "count"),
            median_baseline_mean=("baseline_mean", "median"),
            median_baseline_std=("baseline_std", "median"),
            median_cv=("coefficient_of_variation", "median"),
            p75_cv=("coefficient_of_variation", lambda x: x.quantile(0.75)),
        )
        .reset_index()
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
