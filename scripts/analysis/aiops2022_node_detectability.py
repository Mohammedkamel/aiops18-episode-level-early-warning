"""Node-level counterpart to aiops2022_detectability_recovery.py: detectability/recovery
for the 85 node-level fault events, using metric/node (per-node system KPIs) as the signal
instead of metric/service — these are reported separately since they measure detectability
against a different signal type, not folded into the service-level table.

Outputs:
  results/aiops2022/aiops2022_train_node_event_detectability.csv   (one row per node event)
  results/aiops2022/node_detectability_by_faulttype.csv            (aggregated by failure type)
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))
from aiops2022_archive_readers import read_metric_node_from_faults_zip
from aiops2022_failure_taxonomy import translate
from aiops2022_groundtruth_loader import load_train_events
from aiops2022_sigma_detection import detect_and_recover_sigma, flag_compound_events

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "aiops2022"

# node failure_type -> relevant metric/node kpi_name(s), verified present in
# kpi_cloudbed1_metric_0320.csv's kpi_name column
NODE_FAULT_KPIS = {
    "node_cpu_fault": ["system.cpu.pct_usage", "system.load.1"],
    "node_cpu_climb": ["system.cpu.pct_usage", "system.load.1"],
    "node_disk_read_io_consumption": ["system.io.r_s", "system.io.rkb_s", "system.io.r_await"],
    "node_disk_write_io_consumption": ["system.io.w_s", "system.io.w_await"],
    "node_disk_space_consumption": ["system.disk.used", "system.disk.pct_usage"],
    "node_memory_consumption": ["system.mem.used", "system.mem.pct_usage"],
}


def analyze_group(cloudbed: int, date_mmdd: str, group_events: pd.DataFrame) -> list[dict]:
    df_node = read_metric_node_from_faults_zip(cloudbed, date_mmdd)

    results = []
    for _, ev in group_events.iterrows():
        failure_type = translate(ev["failure_type"])
        row = {
            "event_id": ev["event_id"],
            "cloudbed": cloudbed,
            "date_mmdd": date_mmdd,
            "timestamp": ev["timestamp"],
            "cmdb_id": ev["cmdb_id"],
            "failure_type_raw": ev["failure_type"],
            "failure_type": failure_type,
        }

        kpis = NODE_FAULT_KPIS.get(failure_type)
        if kpis is None:
            row.update(mapping_status="no_kpi_mapping_defined")
            results.append(row)
            continue

        df_this_node = df_node[df_node["cmdb_id"] == ev["cmdb_id"]]
        per_kpi = []
        used_kpis = []
        for kpi in kpis:
            df_kpi = df_this_node[df_this_node["kpi_name"] == kpi]
            if df_kpi.empty:
                continue
            outcome = detect_and_recover_sigma(df_kpi, int(ev["timestamp"]), ["value"])
            if outcome is not None:
                per_kpi.append(outcome)
                used_kpis.append(kpi)

        if not per_kpi:
            row.update(mapping_status="no_baseline_data", matched_kpis="")
            results.append(row)
            continue

        detected_any = any(o["detected"] for o in per_kpi)
        resolved_recoveries = [o["recovery_seconds"] for o in per_kpi if o["detected"] and pd.notna(o["recovery_seconds"])]
        row.update(
            mapping_status="mapped",
            matched_kpis=";".join(used_kpis),
            detected_any=detected_any,
            recovery_seconds=max(resolved_recoveries) if resolved_recoveries else float("nan"),
            recovery_exceeded_scan_window=detected_any and not resolved_recoveries,
        )
        results.append(row)

    del df_node
    return results


def main():
    events = load_train_events()
    node_events = events[events["level"] == "node"].copy()
    assert len(node_events) == 85, f"expected 85 node-level events, got {len(node_events)}"

    all_rows = []
    for (cloudbed, date_mmdd), group in node_events.groupby(["cloudbed", "date_mmdd"]):
        print(f"processing cloudbed{cloudbed} 2022-{date_mmdd} ({len(group)} node events)...")
        all_rows.extend(analyze_group(int(cloudbed), date_mmdd, group))

    results = pd.DataFrame(all_rows)

    mapped_mask = results["mapping_status"] == "mapped"
    results["compound"] = False
    for (cloudbed, date_mmdd), sub in results[mapped_mask].groupby(["cloudbed", "date_mmdd"]):
        compound_flags = flag_compound_events(sub, group_cols=["cmdb_id"])
        results.loc[compound_flags.index, "compound"] = compound_flags

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    events_path = RESULTS_DIR / "aiops2022_train_node_event_detectability.csv"
    if events_path.exists():
        raise FileExistsError(f"{events_path} already exists — confirm with user before overwriting")
    results.to_csv(events_path, index=False)
    print(f"\nwrote {len(results)} rows -> {events_path}")

    mapped = results[mapped_mask]
    summary = (
        mapped.groupby("failure_type")
        .agg(
            n_events=("event_id", "count"),
            n_detected=("detected_any", "sum"),
            detection_rate=("detected_any", "mean"),
            median_recovery_seconds=("recovery_seconds", "median"),
            p25_recovery_seconds=("recovery_seconds", lambda x: x.quantile(0.25)),
            p75_recovery_seconds=("recovery_seconds", lambda x: x.quantile(0.75)),
            n_recovery_exceeded_scan_window=("recovery_exceeded_scan_window", "sum"),
            n_compound=("compound", "sum"),
        )
        .reset_index()
        .sort_values("failure_type")
    )

    summary_path = RESULTS_DIR / "node_detectability_by_faulttype.csv"
    if summary_path.exists():
        raise FileExistsError(f"{summary_path} already exists — confirm with user before overwriting")
    summary.to_csv(summary_path, index=False)
    print(f"wrote {len(summary)} rows -> {summary_path}")

    print("\n--- mapping status breakdown ---")
    print(results["mapping_status"].value_counts())
    print("\n--- node-level detectability/recovery summary ---")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
