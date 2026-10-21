"""Single shared detect/recover implementation used by all four CCF AIOps 2022 analysis
scripts (train/test x service/node level) — aiops2022_detectability_recovery.py,
aiops2022_node_detectability.py, aiops2022_test_detectability_recovery.py, and
aiops2022_test_node_detectability.py all call detect_and_recover_sigma() below rather than
maintaining their own detection math, specifically to prevent the methodology from
silently diverging between them as the pipeline evolves.

Spec: detectable = signal deviates beyond baseline mean +/- 3 sigma within 5 minutes of
injection; recovery = first return to baseline +/- 2 sigma sustained for 3+ samples.
Baseline is the 10 minutes immediately pre-injection.
"""

import numpy as np
import pandas as pd

BASELINE_WINDOW_S = 600
DETECT_WINDOW_S = 300
RECOVERY_SCAN_WINDOW_S = 3600
SUSTAINED_POINTS = 3
DETECT_SIGMA = 3
RECOVER_SIGMA = 2
STD_FLOOR = 1e-6


def _deviation_mask(df_slice: pd.DataFrame, value_cols: list[str], means, stds, sigma: float) -> pd.Series:
    mask = pd.Series(False, index=df_slice.index)
    for col in value_cols:
        std_eff = max(stds[col], STD_FLOOR) if pd.notna(stds[col]) else STD_FLOOR
        mask = mask | ((df_slice[col] - means[col]).abs() > sigma * std_eff)
    return mask


def detect_and_recover_sigma(
    df: pd.DataFrame, ts0: int, value_cols: list[str], detect_sigma: float = DETECT_SIGMA
) -> dict | None:
    """df must have a 'timestamp' column plus every column in value_cols.

    detect_sigma overrides the module default DETECT_SIGMA (3) — used by the sigma
    robustness check (aiops2022_train_sigma_robustness.py) to rerun detection at 2.5/4.0
    without touching the primary pipeline's behavior (every other caller omits this arg).
    """
    df = df.sort_values("timestamp")
    baseline = df[(df.timestamp >= ts0 - BASELINE_WINDOW_S) & (df.timestamp < ts0)]
    if len(baseline) < 3:
        return None
    means, stds = baseline[value_cols].mean(), baseline[value_cols].std()

    detect_slice = df[(df.timestamp >= ts0) & (df.timestamp <= ts0 + DETECT_WINDOW_S)]
    if detect_slice.empty:
        return None
    detect_dev = _deviation_mask(detect_slice, value_cols, means, stds, detect_sigma)
    detected = bool(detect_dev.any())

    result = {"detected": detected}
    for col in value_cols:
        std_eff = max(stds[col], STD_FLOOR) if pd.notna(stds[col]) else STD_FLOOR
        result[f"detected_{col}"] = bool(
            ((detect_slice[col] - means[col]).abs() > detect_sigma * std_eff).any()
        )

    if not detected:
        result["recovery_seconds"] = np.nan
        result["recovery_exceeded_scan_window"] = False
        return result

    post = df[(df.timestamp >= ts0) & (df.timestamp <= ts0 + RECOVERY_SCAN_WINDOW_S)].reset_index(drop=True)
    recover_dev = _deviation_mask(post, value_cols, means, stds, RECOVER_SIGMA)
    recovery_seconds = None
    n = len(recover_dev)
    for i in range(n):
        window = recover_dev.iloc[i : i + SUSTAINED_POINTS]
        if len(window) == SUSTAINED_POINTS and not window.any():
            recovery_seconds = float(post["timestamp"].iloc[i] - ts0)
            break

    result["recovery_seconds"] = recovery_seconds if recovery_seconds is not None else np.nan
    result["recovery_exceeded_scan_window"] = recovery_seconds is None
    return result


def flag_compound_events(df: pd.DataFrame, group_cols: list[str], window_s: int = RECOVERY_SCAN_WINDOW_S) -> pd.Series:
    """An event is 'compound' if another event sharing group_cols (e.g. same resolved
    base_name, or same node cmdb_id) has an injection timestamp within window_s of it.
    Scoped to rows already restricted to one (cloudbed, date_mmdd) group by the caller.
    """
    is_compound = pd.Series(False, index=df.index)
    for _, sub in df.groupby(group_cols):
        if len(sub) < 2:
            continue
        ts = sub["timestamp"].to_numpy()
        for i, idx_i in enumerate(sub.index):
            others = np.delete(ts, i)
            if len(others) and (np.abs(others - ts[i]) <= window_s).any():
                is_compound[idx_i] = True
    return is_compound
