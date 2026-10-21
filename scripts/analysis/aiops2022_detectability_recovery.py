"""Task 1: per-event detectability and recovery-time analysis for the 300 CCF AIOps 2022
training fault-injection events, using metric/service/metric_service.csv (rr/sr/mrt/count)
streamed directly out of training_data_with_faults.zip (no full extraction).

Outputs:
  results/aiops2022/aiops2022_train_event_detectability.csv   (one row per event)
  results/aiops2022/cloudbed_detectability_by_faulttype.csv   (aggregated by failure type)
"""

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))
from aiops2022_archive_readers import read_metric_service_from_faults_zip
from aiops2022_failure_taxonomy import translate
from aiops2022_groundtruth_loader import load_train_events
from aiops2022_sigma_detection import detect_and_recover_sigma, flag_compound_events

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "aiops2022"

SERVICE_VALUE_COLS = ["rr", "sr", "mrt"]

POD_SUFFIX_RE = re.compile(r"-\d+$")


def base_name_of_service_entry(service_entry: str) -> str:
    for suffix in ("-grpc", "-http"):
        if service_entry.endswith(suffix):
            return service_entry[: -len(suffix)]
    return service_entry


def resolve_service_base_name(level: str, cmdb_id: str, known_base_names: set[str]) -> str | None:
    if level == "service":
        return cmdb_id
    if level == "pod":
        candidate = POD_SUFFIX_RE.sub("", cmdb_id)
        if candidate in known_base_names:
            return candidate
        # second-replica naming, e.g. 'adservice2-0' -> base 'adservice2' -> real service 'adservice'
        if candidate.endswith("2") and candidate[:-1] in known_base_names:
            return candidate[:-1]
        return candidate  # unresolvable; caller reports as unmapped_unknown_service_prefix
    return None  # level == "node": no pod-to-node topology available


def analyze_group(cloudbed: int, date_mmdd: str, group_events: pd.DataFrame) -> list[dict]:
    df_service = read_metric_service_from_faults_zip(cloudbed, date_mmdd)
    known_base_names = {base_name_of_service_entry(s) for s in df_service["service"].unique()}

    results = []
    for _, ev in group_events.iterrows():
        base_name = resolve_service_base_name(ev["level"], ev["cmdb_id"], known_base_names)
        row = {
            "event_id": ev["event_id"],
            "cloudbed": cloudbed,
            "date_mmdd": date_mmdd,
            "timestamp": ev["timestamp"],
            "level": ev["level"],
            "cmdb_id": ev["cmdb_id"],
            "failure_type_raw": ev["failure_type"],
            "failure_type": translate(ev["failure_type"]),
        }

        if base_name is None:
            row.update(mapping_status="unmapped_node_no_topology", matched_services="")
            results.append(row)
            continue
        if base_name not in known_base_names:
            row.update(mapping_status="unmapped_unknown_service_prefix", matched_services="")
            results.append(row)
            continue

        matched = [s for s in df_service["service"].unique() if base_name_of_service_entry(s) == base_name]
        per_service = []
        for svc in matched:
            df_svc = df_service[df_service["service"] == svc]
            outcome = detect_and_recover_sigma(df_svc, int(ev["timestamp"]), SERVICE_VALUE_COLS)
            if outcome is not None:
                per_service.append(outcome)

        if not per_service:
            row.update(
                mapping_status="mapped_but_no_baseline_data", matched_services=";".join(matched), base_name=base_name
            )
            results.append(row)
            continue

        detected_any = any(o["detected"] for o in per_service)
        resolved_recoveries = [
            o["recovery_seconds"] for o in per_service if o["detected"] and pd.notna(o["recovery_seconds"])
        ]
        row.update(
            mapping_status="mapped",
            matched_services=";".join(matched),
            base_name=base_name,
            detected_any=detected_any,
            detected_rr=any(o["detected_rr"] for o in per_service),
            detected_sr=any(o["detected_sr"] for o in per_service),
            detected_mrt=any(o["detected_mrt"] for o in per_service),
            recovery_seconds=max(resolved_recoveries) if resolved_recoveries else float("nan"),
            recovery_exceeded_scan_window=detected_any and not resolved_recoveries,
        )
        results.append(row)

    del df_service
    return results


def main():
    events = load_train_events()
    assert len(events) == 300, f"expected 300 training events, got {len(events)}"

    all_rows = []
    for (cloudbed, date_mmdd), group in events.groupby(["cloudbed", "date_mmdd"]):
        print(f"processing cloudbed{cloudbed} 2022-{date_mmdd} ({len(group)} events)...")
        all_rows.extend(analyze_group(int(cloudbed), date_mmdd, group))

    results = pd.DataFrame(all_rows)

    mapped_mask = results["mapping_status"] == "mapped"
    results["compound"] = False
    for (cloudbed, date_mmdd), sub in results[mapped_mask].groupby(["cloudbed", "date_mmdd"]):
        compound_flags = flag_compound_events(sub, group_cols=["base_name"])
        results.loc[compound_flags.index, "compound"] = compound_flags

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    events_path = RESULTS_DIR / "aiops2022_train_event_detectability.csv"
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
    unmapped_counts = (
        results[results["mapping_status"] != "mapped"]
        .groupby(["failure_type", "mapping_status"])
        .size()
        .rename("n_unmapped_events")
        .reset_index()
    )

    summary_path = RESULTS_DIR / "cloudbed_detectability_by_faulttype.csv"
    if summary_path.exists():
        raise FileExistsError(f"{summary_path} already exists — confirm with user before overwriting")
    summary.to_csv(summary_path, index=False)
    print(f"wrote {len(summary)} rows -> {summary_path}")

    print("\n--- mapping status breakdown ---")
    print(results["mapping_status"].value_counts())
    print("\n--- unmapped events by failure type ---")
    print(unmapped_counts.to_string(index=False))
    print("\n--- detectability/recovery summary (mapped events only) ---")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
