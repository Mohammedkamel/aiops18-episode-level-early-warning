"""Node-level detectability/recovery for the CCF AIOps 2022 test-set node-level events,
streamed from test_data.7z. Same sigma-based method as aiops2022_test_detectability_recovery.py
(mean+/-3sigma detect within 5 min, mean+/-2sigma recovery hysteresis sustained 3+ samples).

Outputs:
  results/aiops2022/aiops2022_test_node_event_detectability.csv
  results/aiops2022/test_node_detectability_by_faulttype.csv
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))
from aiops2022_archive_readers import read_metric_node_from_test_7z
from aiops2022_failure_taxonomy import translate
from aiops2022_groundtruth_loader import load_test_events
from aiops2022_sigma_detection import detect_and_recover_sigma, flag_compound_events

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aiops2022_node_detectability import NODE_FAULT_KPIS

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "aiops2022"


def analyze_date(date_mmdd: str, group_events: pd.DataFrame) -> list[dict]:
    df_node = read_metric_node_from_test_7z(date_mmdd)

    results = []
    for _, ev in group_events.iterrows():
        failure_type = translate(ev["failure_type"])
        row = {
            "event_id": ev["event_id"],
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
            df_kpi = df_this_node[df_this_node["kpi_name"] == kpi][["timestamp", "value"]]
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
        resolved = [o["recovery_seconds"] for o in per_kpi if o["detected"] and pd.notna(o["recovery_seconds"])]
        row.update(
            mapping_status="mapped",
            matched_kpis=";".join(used_kpis),
            detected_any=detected_any,
            recovery_seconds=max(resolved) if resolved else float("nan"),
            recovery_exceeded_scan_window=detected_any and not resolved,
        )
        results.append(row)

    del df_node
    return results


def main():
    events = load_test_events()
    node_events = events[events["level"] == "node"].copy()

    all_rows = []
    for date_mmdd, group in node_events.groupby("date_mmdd"):
        print(f"processing test date 2022-{date_mmdd} ({len(group)} node events)...")
        all_rows.extend(analyze_date(date_mmdd, group))

    results = pd.DataFrame(all_rows)

    mapped_mask = results["mapping_status"] == "mapped"
    results["compound"] = False
    for date_mmdd, sub in results[mapped_mask].groupby("date_mmdd"):
        compound_flags = flag_compound_events(sub, group_cols=["cmdb_id"])
        results.loc[compound_flags.index, "compound"] = compound_flags

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    events_path = RESULTS_DIR / "aiops2022_test_node_event_detectability.csv"
    if events_path.exists():
        raise FileExistsError(f"{events_path} already exists — confirm with user before overwriting")
    results.to_csv(events_path, index=False)
    print(f"\nwrote {len(results)} rows -> {events_path}")

    mapped = results[mapped_mask].copy()
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

    summary_path = RESULTS_DIR / "test_node_detectability_by_faulttype.csv"
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
