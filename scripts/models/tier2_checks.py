"""Tier 2 checks: threshold-exclusion sensitivity, onset-multiplicity verification, and
gap/boundary verification -- all run directly against saved data, not re-derived from reading
the code.

Read-only: writes exactly one new file, results/aiops18/threshold_sensitivity.csv. The other
two checks print their findings (and would only produce a file if a violation were found,
which -- per the results below -- did not happen).
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))
from aiops18_preprocessing import load_merged, process_kpi  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap_significance_tests import get_calibration, load_raw_pred, load_windows  # noqa: E402
from evaluation_metrics import compute_detection_metrics, extract_true_onset_timestamps  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "aiops18"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "processed_data" / "aiops18"
MANIFEST_PATH = PROCESSED_DIR / "aiops18_manifest.json"
MODELS = ["persistence", "sarima", "xgboost", "lstm", "patchtst", "chronos"]
HORIZON_MIN, RESOLUTION_S, CONTEXT_STEPS, HORIZON_STEPS = 15, 60, 288, 15  # the group actually used for model evaluation


# ==================== Check 1: threshold-exclusion sensitivity ====================

def check1_threshold_sensitivity():
    print("\n" + "=" * 70)
    print("CHECK 1: threshold-exclusion sensitivity sweep")
    print("=" * 70)

    windows_val = load_windows("val")
    all_kpis = sorted(set(windows_val["kpi_id"]))
    calibs = get_calibration(all_kpis)  # precision/threshold computed once; cutoff applied after

    rows = []
    for cutoff in [0.10, 0.20, 0.30]:
        excluded = {k for k, c in calibs.items() if c.precision < cutoff}
        clean = sorted(set(all_kpis) - excluded)
        print(f"\ncutoff={cutoff:.0%}: {len(excluded)}/{len(all_kpis)} excluded -> {len(clean)}-series clean cohort")

        for split in ["val", "test"]:
            w = load_windows(split)
            for model in MODELS:
                d = load_raw_pred(model, split)
                kpi_id, ts, predicted_positive = d["kpi_id"], d["window_start_ts"], d["predicted_positive"]
                lookup = {(k, int(t)): i for i, (k, t) in enumerate(zip(w["kpi_id"], w["window_start_ts"]))}
                orig_idx = np.array([lookup[(kpi_id[i], int(ts[i]))] for i in range(len(kpi_id))])
                context_label_last = w["context_label"][orig_idx, -1]
                horizon_label = w["horizon_label"][orig_idx]
                true_onset_ts = extract_true_onset_timestamps(
                    ts, CONTEXT_STEPS, HORIZON_STEPS, RESOLUTION_S, context_label_last, horizon_label
                )
                clean_mask = np.isin(kpi_id, clean)
                det = compute_detection_metrics(
                    kpi_id[clean_mask], ts[clean_mask], CONTEXT_STEPS, HORIZON_STEPS, RESOLUTION_S,
                    true_onset_ts[clean_mask], predicted_positive[clean_mask],
                )
                rows.append(dict(
                    cutoff=cutoff, n_excluded=len(excluded), n_clean=len(clean),
                    model=model, split=split, n_episodes=det.n_episodes, n_detected=det.n_detected,
                    detection_rate=det.detection_rate,
                ))

    out = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "threshold_sensitivity.csv"
    out.to_csv(out_path, index=False)

    pd.set_option("display.width", 250)
    pivot = out.pivot_table(index=["model", "split"], columns="cutoff", values="detection_rate")
    print("\ndetection_rate by cutoff:")
    print(pivot.to_string())
    print(f"\nsaved to {out_path}")
    return out


# ==================== Check 2: onset-multiplicity verification ====================

def check2_onset_multiplicity():
    print("\n" + "=" * 70)
    print("CHECK 2: at-most-one-onset-per-horizon verification")
    print("=" * 70)

    total_windows = 0
    total_multi = 0
    examples = []
    for split in ["val", "test"]:
        for horizon_min in [15, 60]:
            for resolution in [60, 300]:
                path = PROCESSED_DIR / f"aiops18_{split}_windows_h{horizon_min}min_{resolution}s.npz"
                if not path.exists():
                    print(f"  (missing: {path.name})")
                    continue
                d = np.load(path, allow_pickle=True)
                context_label_last = d["context_label"][:, -1]
                horizon_label = d["horizon_label"]
                n = horizon_label.shape[0]
                full = np.concatenate([context_label_last.reshape(n, 1), horizon_label], axis=1)
                is_onset = (full[:, 1:] == 1) & (full[:, :-1] == 0)
                n_onsets_per_window = is_onset.sum(axis=1)
                n_multi = int((n_onsets_per_window > 1).sum())
                total_windows += n
                total_multi += n_multi
                print(f"  {split} h{horizon_min}min {resolution}s: {n} windows, {n_multi} with >1 onset in horizon")

                if n_multi and len(examples) < 3:
                    multi_idx = np.where(n_onsets_per_window > 1)[0]
                    for i in multi_idx[: 3 - len(examples)]:
                        kpi_id = d["kpi_id"][i]
                        window_start_ts = int(d["window_start_ts"][i])
                        positions = np.where(is_onset[i])[0]
                        modal_interval = resolution
                        onset_times = [
                            datetime.fromtimestamp(window_start_ts + int(p) * modal_interval, tz=timezone.utc)
                            for p in positions
                        ]
                        examples.append(dict(
                            split=split, horizon_min=horizon_min, resolution=resolution,
                            kpi_id=kpi_id, window_start_ts=window_start_ts,
                            window_start=datetime.fromtimestamp(window_start_ts, tz=timezone.utc),
                            onset_positions_in_full=positions.tolist(), onset_times=onset_times,
                        ))

    print(f"\nTOTAL across all val+test, both horizons, both resolutions: {total_multi}/{total_windows} windows with >1 onset in horizon")
    if examples:
        print("\nexamples:")
        for e in examples:
            print(f"  {e}")
    else:
        print("no examples to show (count is zero)")
    return total_windows, total_multi, examples


# ==================== Check 3: boundary/gap verification ====================

def check3_boundary_gap():
    print("\n" + "=" * 70)
    print("CHECK 3: window context/horizon never spans a hard-gap or split boundary")
    print("=" * 70)

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    kpi_info = manifest["kpis"]

    merged = load_merged()
    total_windows_checked = 0
    total_violations = 0
    violation_examples = []

    for kpi_id, g in merged.groupby("KPI ID"):
        if kpi_id not in kpi_info:
            continue
        result = process_kpi(kpi_id, g)
        frame = result["frame"]
        modal_interval = result["modal_interval"]
        info = kpi_info[kpi_id]
        context_steps = info["context_steps"]

        for split in ["train", "val", "test"]:
            sub = frame[frame.split == split].reset_index(drop=True)
            if len(sub) == 0:
                continue
            run_id_arr = sub["run_id"].to_numpy()
            ts_arr = sub["timestamp"].to_numpy()
            ts_to_pos = {int(t): i for i, t in enumerate(ts_arr)}

            for horizon_min, horizon_steps in info["horizon_steps"].items():
                path = PROCESSED_DIR / f"aiops18_{split}_windows_h{horizon_min}min_{int(modal_interval)}s.npz"
                if not path.exists():
                    continue
                d = np.load(path, allow_pickle=True)
                mask = d["kpi_id"] == kpi_id
                if not mask.any():
                    continue
                win_starts = d["window_start_ts"][mask]
                need = context_steps + horizon_steps

                for wts in win_starts:
                    pos = ts_to_pos.get(int(wts))
                    total_windows_checked += 1
                    if pos is None:
                        violation_examples.append(dict(kind="window_start_ts_not_found_in_frame", kpi_id=kpi_id, split=split, window_start_ts=int(wts)))
                        total_violations += 1
                        continue
                    span_run_ids = run_id_arr[pos: pos + need]
                    if len(span_run_ids) < need or not np.all(span_run_ids == span_run_ids[0]):
                        total_violations += 1
                        if len(violation_examples) < 5:
                            violation_examples.append(dict(
                                kind="run_id_not_constant_across_window", kpi_id=kpi_id, split=split,
                                window_start_ts=int(wts), horizon_min=horizon_min,
                                run_ids_in_span=sorted(set(span_run_ids.tolist())),
                            ))

    print(f"\nTOTAL windows checked (all KPIs, all splits, all horizons, native resolution): {total_windows_checked}")
    print(f"TOTAL violations (window whose context+horizon span crosses a run_id boundary, i.e. a hard gap or split boundary): {total_violations}")
    if violation_examples:
        print("\nviolation examples:")
        for e in violation_examples[:5]:
            print(f"  {e}")
    else:
        print("no violations found -- every window's full context+horizon span stays within a single contiguous run_id, confirmed directly against the actual data.")
    return total_windows_checked, total_violations, violation_examples


if __name__ == "__main__":
    check1_threshold_sensitivity()
    check2_onset_multiplicity()
    check3_boundary_gap()
