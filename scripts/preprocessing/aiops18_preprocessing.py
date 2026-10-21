"""AIOps18 (NetManAIOps KPI-Anomaly-Detection) preprocessing: merges Preliminary + Finals,
does gap-aware per-KPI temporal splitting, z-score normalization, and sliding-window
construction with onset labeling.

See docs/aiops18_preprocessing_methodology.md for the full rationale behind every choice
below (gap-interpolation threshold, split-boundary handling, stride, etc.) — this docstring
only summarizes.

Pipeline per KPI:
  1. Merge Preliminary (26 KPIs) + Finals (29 KPIs) -> 55 independent series, zero ID overlap.
  2. Detect modal sampling interval (60s or 300s). Interpolate single-sample gaps (interval
     == 2x modal) linearly; value at the midpoint; the label is 1 only if both neighbors are
     1 (preserves segment continuity through a single dropped sample). Gaps >= 3x modal are
     treated as hard breaks that no window may span.
  3. Chronological 60/20/20 split by row count (not random). Split boundaries are additional
     hard breaks alongside real gaps -- no window spans a split boundary either.
  4. Z-score normalize using mean/std fit on the train split's label==0 rows only. Same
     transform applied to val/test.
  5. Sliding windows: context_length steps of history, then a forecast horizon. Two
     horizons: 15 min and 60 min. Stride = 5 minutes of wall-clock time (5 steps at 60s
     resolution, 1 step at 300s resolution) to keep output size within the ~5GB budget
     agreed for this run.
  6. Onset label per window = 1 if any point in the forecast horizon is the first sample of
     a contiguous label==1 segment (a 0->1 transition), else 0.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "aiops18"
OUT_DIR = Path(__file__).resolve().parents[2] / "processed_data" / "aiops18"

TRAIN_FRAC, VAL_FRAC = 0.6, 0.2
HORIZONS_MIN = [15, 60]
CONTEXT_MINUTES = 288  # 288 steps at 60s = 4.8 hours; scaled proportionally at 300s
STRIDE_MINUTES = 5
Z_CLIP = 10.0  # clip post-normalization values to +/-10 sigma; see methodology doc Section 4


def load_merged() -> pd.DataFrame:
    prelim = pd.read_csv(RAW_DIR / "Preliminary_dataset" / "train.csv")
    finals = pd.read_csv(RAW_DIR / "Finals_dataset" / "phase2_train.csv")
    assert prelim["KPI ID"].nunique() == 26
    assert finals["KPI ID"].nunique() == 29
    assert len(set(prelim["KPI ID"]) & set(finals["KPI ID"])) == 0
    merged = pd.concat([prelim, finals], ignore_index=True)
    assert merged["KPI ID"].nunique() == 55
    return merged


def interpolate_single_gaps(g: pd.DataFrame, modal_interval: int) -> pd.DataFrame:
    g = g.sort_values("timestamp").reset_index(drop=True)
    ts, val, lab = g["timestamp"].to_numpy(), g["value"].to_numpy(), g["label"].to_numpy()
    diffs = np.diff(ts)
    single_gap = diffs == 2 * modal_interval
    if not single_gap.any():
        return g
    idx = np.where(single_gap)[0]
    mid_ts = ts[idx] + modal_interval
    mid_val = (val[idx] + val[idx + 1]) / 2
    mid_lab = np.where((lab[idx] == 1) & (lab[idx + 1] == 1), 1, 0)
    new_rows = pd.DataFrame(
        {"timestamp": mid_ts, "value": mid_val, "label": mid_lab, "KPI ID": g["KPI ID"].iloc[0]}
    )
    return pd.concat([g, new_rows], ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def process_kpi(kpi_id: str, g: pd.DataFrame) -> dict:
    g = g.sort_values("timestamp").reset_index(drop=True)
    diffs = g["timestamp"].diff()
    modal_interval = int(diffs.mode().iloc[0])
    assert modal_interval in (60, 300), f"unexpected modal interval {modal_interval} for {kpi_id}"

    n_before = len(g)
    g = interpolate_single_gaps(g, modal_interval)
    n = len(g)
    n_interpolated = n - n_before

    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    split = np.array(["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val))
    g["split"] = split

    # hard breaks: real gaps (>=3x modal) OR a split boundary -- no window may cross either
    ts = g["timestamp"].to_numpy()
    is_hard_gap = np.zeros(n, dtype=bool)
    is_hard_gap[1:] = np.diff(ts) >= 3 * modal_interval
    is_split_boundary = np.zeros(n, dtype=bool)
    is_split_boundary[1:] = split[1:] != split[:-1]
    run_id = (is_hard_gap | is_split_boundary).cumsum()
    g["run_id"] = run_id

    train_normal = g[(g.split == "train") & (g.label == 0)]
    mean, std = float(train_normal["value"].mean()), float(train_normal["value"].std())
    if std == 0 or pd.isna(std):
        std = 1e-6  # degenerate flat-normal-baseline guard; none observed in this dataset

    g["value_z"] = ((g["value"] - mean) / std).clip(-Z_CLIP, Z_CLIP)

    context_steps = round(CONTEXT_MINUTES * 60 / modal_interval)
    stride_steps = max(1, round(STRIDE_MINUTES * 60 / modal_interval))
    horizon_steps = {h: round(h * 60 / modal_interval) for h in HORIZONS_MIN}

    label = g["label"].to_numpy()
    onset = ((label == 1) & (np.concatenate([[0], label[:-1]]) == 0)).astype(np.int8)

    return dict(
        kpi_id=kpi_id,
        modal_interval=modal_interval,
        n_rows=n,
        n_interpolated=n_interpolated,
        mean=mean,
        std=std,
        context_steps=context_steps,
        stride_steps=stride_steps,
        horizon_steps=horizon_steps,
        frame=g.assign(onset=onset),
    )


def build_windows(frame: pd.DataFrame, kpi_id: str, context_steps: int, stride_steps: int, horizon_steps: int, split_name: str):
    sub = frame[frame.split == split_name]
    value_z = sub["value_z"].to_numpy()
    label = sub["label"].to_numpy()
    onset = sub["onset"].to_numpy()
    ts = sub["timestamp"].to_numpy()
    run_id = sub["run_id"].to_numpy()

    need = context_steps + horizon_steps
    contexts, ctx_labels, horizon_vals, horizon_labels, onset_flags, starts = [], [], [], [], [], []

    for _, idx in pd.Series(np.arange(len(sub))).groupby(run_id):
        idx = idx.to_numpy()
        if len(idx) < need:
            continue
        run_start = idx[0]
        last_start = idx[-1] - need + 1
        for start in range(run_start, run_start + (last_start - run_start) + 1, stride_steps):
            ctx_slice = slice(start, start + context_steps)
            hor_slice = slice(start + context_steps, start + need)
            contexts.append(value_z[ctx_slice])
            ctx_labels.append(label[ctx_slice])
            horizon_vals.append(value_z[hor_slice])
            horizon_labels.append(label[hor_slice])
            onset_flags.append(int(onset[hor_slice].any()))
            starts.append(ts[start])

    if not contexts:
        return None
    return dict(
        context=np.array(contexts, dtype=np.float32),
        context_label=np.array(ctx_labels, dtype=np.int8),
        horizon_value=np.array(horizon_vals, dtype=np.float32),
        horizon_label=np.array(horizon_labels, dtype=np.int8),
        onset_in_horizon=np.array(onset_flags, dtype=np.int8),
        window_start_ts=np.array(starts, dtype=np.int64),
        kpi_id=np.array([kpi_id] * len(contexts), dtype=object),
    )


def main():
    merged = load_merged()
    print(f"merged: {len(merged)} rows, {merged['KPI ID'].nunique()} KPIs")

    total_segments = 0
    manifest_kpis = {}
    # keyed by (split, horizon_minutes, modal_interval_seconds) -- 60s and 300s KPIs have
    # different context_steps (288 vs 58, since context length is scaled proportionally per
    # the spec) and so cannot share one array; each resolution gets its own output file.
    per_split_per_horizon = {
        (split, h, resolution): {"context": [], "context_label": [], "horizon_value": [], "horizon_label": [], "onset_in_horizon": [], "window_start_ts": [], "kpi_id": []}
        for split in ("train", "val", "test") for h in HORIZONS_MIN for resolution in (60, 300)
    }

    for kpi_id, g in merged.groupby("KPI ID"):
        result = process_kpi(kpi_id, g)
        frame = result["frame"]
        total_segments += int(frame["onset"].sum())

        # timestamp is the first sample AFTER the gap (resumption point), not the gap's start
        gap_resumption_timestamps = frame.loc[
            frame["timestamp"].diff() >= 3 * result["modal_interval"], "timestamp"
        ].tolist()

        train_anomaly_points = int((frame.split == "train").values @ (frame.label == 1).values)
        val_anomaly_points = int((frame.split == "val").values @ (frame.label == 1).values)
        test_anomaly_points = int((frame.split == "test").values @ (frame.label == 1).values)
        # onset-based metrics (detection rate, recall, lead time) are undefined for a split
        # with zero true anomalous points -- there is nothing to detect. MAE/RMSE and
        # false-positive-rate/precision remain valid regardless (see Section 3 of
        # docs/aiops18_preprocessing_methodology.md for the full rule).
        onset_metrics_excluded_splits = [
            split_name
            for split_name, count in (("val", val_anomaly_points), ("test", test_anomaly_points))
            if count == 0
        ]

        manifest_kpis[kpi_id] = dict(
            modal_interval_seconds=result["modal_interval"],
            n_rows_after_interpolation=result["n_rows"],
            n_single_sample_gaps_interpolated=result["n_interpolated"],
            n_train=int((frame.split == "train").sum()),
            n_val=int((frame.split == "val").sum()),
            n_test=int((frame.split == "test").sum()),
            train_anomaly_points=train_anomaly_points,
            val_anomaly_points=val_anomaly_points,
            test_anomaly_points=test_anomaly_points,
            onset_metrics_excluded_splits=onset_metrics_excluded_splits,
            train_mean=result["mean"],
            train_std=result["std"],
            context_steps=result["context_steps"],
            stride_steps=result["stride_steps"],
            horizon_steps=result["horizon_steps"],
            n_hard_gaps=len(gap_resumption_timestamps),
            gap_resumption_timestamps=gap_resumption_timestamps,
            n_onset_segments=int(frame["onset"].sum()),
        )

        for h_min in HORIZONS_MIN:
            h_steps = result["horizon_steps"][h_min]
            for split_name in ("train", "val", "test"):
                windows = build_windows(
                    frame, kpi_id, result["context_steps"], result["stride_steps"], h_steps, split_name
                )
                if windows is None:
                    continue
                for k, v in windows.items():
                    per_split_per_horizon[(split_name, h_min, result["modal_interval"])][k].append(v)

        print(f"  {kpi_id}: {result['n_rows']} rows, modal={result['modal_interval']}s, "
              f"{len(gap_resumption_timestamps)} hard gaps, {int(frame['onset'].sum())} onset segments")

    print(f"\ntotal onset segments across all 55 KPIs: {total_segments}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for (split_name, h_min, resolution), arrays in per_split_per_horizon.items():
        if not arrays["context"]:
            continue
        combined = {k: np.concatenate(v, axis=0) for k, v in arrays.items()}
        out_path = OUT_DIR / f"aiops18_{split_name}_windows_h{h_min}min_{resolution}s.npz"
        if out_path.exists():
            raise FileExistsError(f"{out_path} already exists — confirm with user before overwriting")
        np.savez_compressed(out_path, **combined)
        print(f"wrote {len(combined['context'])} windows -> {out_path}")

    kpis_excluded_from_val_onset_metrics = sorted(
        k for k, v in manifest_kpis.items() if "val" in v["onset_metrics_excluded_splits"]
    )
    kpis_excluded_from_test_onset_metrics = sorted(
        k for k, v in manifest_kpis.items() if "test" in v["onset_metrics_excluded_splits"]
    )

    manifest_path = OUT_DIR / "aiops18_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"{manifest_path} already exists — confirm with user before overwriting")
    manifest = dict(
        n_kpis=55,
        n_preliminary_kpis=26,
        n_finals_kpis=29,
        train_frac=TRAIN_FRAC,
        val_frac=VAL_FRAC,
        context_minutes=CONTEXT_MINUTES,
        stride_minutes=STRIDE_MINUTES,
        horizons_minutes=HORIZONS_MIN,
        total_onset_segments=total_segments,
        preliminary_only_segments=933,
        finals_only_segments=1228,
        zero_anomaly_split_policy=(
            "A KPI x split with zero true anomalous points is EXCLUDED from onset-based "
            "metrics (detection rate, recall, lead time) for that split -- there is nothing "
            "to detect, so the split contributes neither to the numerator nor the "
            "denominator, it is not scored as 0% and not silently dropped from other "
            "metrics. The same KPI x split IS RETAINED for (a) aggregate forecasting "
            "accuracy (MAE/RMSE on horizon_value), which needs no anomaly labels, and (b) "
            "false-positive-rate/precision, which only needs negative (label=0) windows and "
            "the model's predictions on them, both of which these splits still have. See "
            "onset_metrics_excluded_splits per KPI below, and Section 3 of "
            "docs/aiops18_preprocessing_methodology.md."
        ),
        kpis_excluded_from_val_onset_metrics=kpis_excluded_from_val_onset_metrics,
        kpis_excluded_from_test_onset_metrics=kpis_excluded_from_test_onset_metrics,
        kpis=manifest_kpis,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
