"""Supplementary robustness check (training set only): does the fault-type detection-rate
ranking change if the detection threshold is 2.5 sigma or 4 sigma instead of 3 sigma?
Detection rate only (no recovery time) — cheap, and doesn't touch the four primary result
files. Reuses the same mapping/reading pipeline as aiops2022_detectability_recovery.py and
aiops2022_node_detectability.py.

Output: results/aiops2022/aiops2022_train_sigma_robustness.csv
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))
from aiops2022_archive_readers import read_metric_node_from_faults_zip, read_metric_service_from_faults_zip
from aiops2022_failure_taxonomy import translate
from aiops2022_groundtruth_loader import load_train_events
from aiops2022_sigma_detection import detect_and_recover_sigma

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aiops2022_detectability_recovery import base_name_of_service_entry, resolve_service_base_name
from aiops2022_node_detectability import NODE_FAULT_KPIS

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "aiops2022"
SIGMAS = [2.5, 3.0, 4.0]
SERVICE_VALUE_COLS = ["rr", "sr", "mrt"]


def detected_service(df_service, known_base_names, ev, sigma) -> bool | None:
    base_name = resolve_service_base_name(ev["level"], ev["cmdb_id"], known_base_names)
    if base_name is None or base_name not in known_base_names:
        return None
    matched = [s for s in df_service["service"].unique() if base_name_of_service_entry(s) == base_name]
    detected_any, any_data = False, False
    for svc in matched:
        df_svc = df_service[df_service["service"] == svc]
        outcome = detect_and_recover_sigma(df_svc, int(ev["timestamp"]), SERVICE_VALUE_COLS, detect_sigma=sigma)
        if outcome is not None:
            any_data = True
            detected_any = detected_any or outcome["detected"]
    return detected_any if any_data else None


def detected_node(df_node, ev, sigma) -> bool | None:
    failure_type = translate(ev["failure_type"])
    kpis = NODE_FAULT_KPIS.get(failure_type)
    if kpis is None:
        return None
    df_this_node = df_node[df_node["cmdb_id"] == ev["cmdb_id"]]
    detected_any, any_data = False, False
    for kpi in kpis:
        df_kpi = df_this_node[df_this_node["kpi_name"] == kpi][["timestamp", "value"]]
        if df_kpi.empty:
            continue
        outcome = detect_and_recover_sigma(df_kpi, int(ev["timestamp"]), ["value"], detect_sigma=sigma)
        if outcome is not None:
            any_data = True
            detected_any = detected_any or outcome["detected"]
    return detected_any if any_data else None


def main():
    events = load_train_events()

    rows = []
    for (cloudbed, date_mmdd), group in events.groupby(["cloudbed", "date_mmdd"]):
        print(f"processing cloudbed{cloudbed} 2022-{date_mmdd} ({len(group)} events)...")
        df_service = read_metric_service_from_faults_zip(int(cloudbed), date_mmdd)
        df_node = read_metric_node_from_faults_zip(int(cloudbed), date_mmdd)
        known_base_names = {base_name_of_service_entry(s) for s in df_service["service"].unique()}

        for _, ev in group.iterrows():
            failure_type = translate(ev["failure_type"])
            for sigma in SIGMAS:
                if ev["level"] == "node":
                    det = detected_node(df_node, ev, sigma)
                else:
                    det = detected_service(df_service, known_base_names, ev, sigma)
                if det is None:
                    continue
                rows.append(dict(sigma=sigma, event_id=ev["event_id"], failure_type=failure_type, detected=det))
        del df_service, df_node

    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["sigma", "failure_type"])
        .agg(n_events=("detected", "count"), detection_rate=("detected", "mean"))
        .reset_index()
        .sort_values(["failure_type", "sigma"])
    )

    out_path = RESULTS_DIR / "aiops2022_train_sigma_robustness.csv"
    if out_path.exists():
        raise FileExistsError(f"{out_path} already exists — confirm with user before overwriting")
    summary.to_csv(out_path, index=False)
    print(f"\nwrote {len(summary)} rows -> {out_path}")

    pivot = summary.pivot(index="failure_type", columns="sigma", values="detection_rate")
    pivot.columns = [f"det_rate_sigma_{c}" for c in pivot.columns]
    for sigma in SIGMAS:
        pivot[f"rank_sigma_{sigma}"] = pivot[f"det_rate_sigma_{sigma}"].rank(ascending=False, method="min")
    pivot = pivot.sort_values("rank_sigma_3.0")

    print("\n--- detection rate + rank by fault type, across sigma thresholds ---")
    print(pivot.to_string())

    # Spearman correlation == Pearson correlation of the already-computed ranks (no scipy needed)
    rank_cols = [f"rank_sigma_{s}" for s in SIGMAS]
    spearman_2_5_vs_3 = pivot[rank_cols[0]].corr(pivot[rank_cols[1]], method="pearson")
    spearman_3_vs_4 = pivot[rank_cols[1]].corr(pivot[rank_cols[2]], method="pearson")
    spearman_2_5_vs_4 = pivot[rank_cols[0]].corr(pivot[rank_cols[2]], method="pearson")
    print("\n--- rank correlation (Spearman) between threshold choices ---")
    print(f"2.5sigma vs 3sigma: {spearman_2_5_vs_3:.3f}")
    print(f"3sigma vs 4sigma:   {spearman_3_vs_4:.3f}")
    print(f"2.5sigma vs 4sigma: {spearman_2_5_vs_4:.3f}")


if __name__ == "__main__":
    main()
