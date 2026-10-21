"""Figure 5 -- correction-layer mechanism on a verified real detection.

The original Figures/files/fig5_correction_mechanism.csv was found (by directly re-running
correction_layer.py's own functions against the saved SARIMA predictions) to depict a window
whose own just-flagged change-point was suppressing its own threshold immediately -- the
pre-causal-fix look-ahead bug the manuscript (Sec. 4.4/5.3) explicitly describes fixing. Under
the current, causally-gated correction_layer.py, that specific episode never produces an early
catch at all.

This script instead locates and plots a real episode that (a) is missed by the raw threshold,
(b) is caught by the current, causally-correct correction layer, and (c) matches the
manuscript's own Figure 5 caption exactly: SARIMA, onset 2017-06-17 08:23:00 UTC, 11-minute
lead time. The active change-point here was flagged 15 minutes earlier (at t=-26min, maturing
at t=-11min = -26+horizon), so the figure also directly illustrates the causal-maturity
mechanism: an earlier, already-matured change-point sets the threshold, never the triggering
window's own future-realized residual.

All values below are computed directly from results/aiops18/sarima_val_15min_60s_predictions.npz
and the live correction_layer.py functions, not read from any pre-existing figure CSV.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))
from bootstrap_significance_tests import load_windows, load_raw_pred, get_calibration, CONTEXT_STEPS, HORIZON_STEPS, RESOLUTION_S  # noqa: E402
from correction_layer import compute_window_residuals, cusum_changepoints, adaptive_threshold_schedule  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[2] / "Figures" / "rendered"

MODEL = "sarima"
SPLIT = "val"
KPI = "05f10d3a-239c-3bef-9bdc-a2feeb0037aa"
ONSET_STR = "2017-06-17 08:23:00"


def load_episode_trace():
    windows = load_windows(SPLIT)
    all_kpis = sorted(set(windows["kpi_id"]))
    calibs = get_calibration(all_kpis)
    base_threshold = calibs[KPI].threshold
    assert not calibs[KPI].low_precision_flag

    d = load_raw_pred(MODEL, SPLIT)
    kpi_id_all, ts_all, y_true_all, y_pred_all = d["kpi_id"], d["window_start_ts"], d["y_true"], d["y_pred"]
    n_all = len(kpi_id_all)
    y_pred_2d_all = y_pred_all.reshape(n_all, HORIZON_STEPS)
    y_true_2d_all = y_true_all.reshape(n_all, HORIZON_STEPS)

    idx = np.where(kpi_id_all == KPI)[0]
    order = np.argsort(ts_all[idx])
    oidx = idx[order]
    ts_k = ts_all[oidx].astype(np.int64)
    context_end_ts = ts_k + CONTEXT_STEPS * RESOLUTION_S
    y_pred_k, y_true_k = y_pred_2d_all[oidx], y_true_2d_all[oidx]
    max_abs_pred = np.max(np.abs(y_pred_k), axis=1)

    residuals = compute_window_residuals(y_true_k.reshape(-1), y_pred_k.reshape(-1), HORIZON_STEPS)
    cps = cusum_changepoints(residuals)
    thresholds = adaptive_threshold_schedule(context_end_ts, cps, base_threshold, HORIZON_STEPS, RESOLUTION_S)
    raw_flag = max_abs_pred > base_threshold
    corrected_flag = max_abs_pred > thresholds

    onset_epoch = int(pd.Timestamp(ONSET_STR).timestamp())
    t_min = (context_end_ts - onset_epoch) / 60.0

    trace = pd.DataFrame(dict(
        t_min=t_min, max_abs_pred=max_abs_pred, static_threshold=base_threshold,
        adaptive_threshold=thresholds, own_changepoint=cps, raw_flag=raw_flag, corrected_flag=corrected_flag,
    ))

    episode_mask = (trace.t_min >= -15) & (trace.t_min <= 0)
    episode = trace[episode_mask].sort_values("t_min").reset_index(drop=True)
    assert len(episode) == 3, f"expected 3 member windows, got {len(episode)}"
    assert not episode.raw_flag.any(), "expected raw to miss this episode"
    assert episode.corrected_flag.all(), "expected correction layer to catch all 3 member windows"
    lead_time_min = -episode.t_min.iloc[0]

    trigger_mask = trace.own_changepoint & (trace.t_min < episode.t_min.min())
    trigger_t = trace[trigger_mask].t_min.max() if trigger_mask.any() else None

    return trace, episode, lead_time_min, base_threshold, trigger_t


def plot(trace, episode, lead_time_min, base_threshold, trigger_t):
    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(figsize=(8.5, 5))

    window = trace[(trace.t_min >= -32) & (trace.t_min <= 20)].sort_values("t_min")
    x_lo, x_hi = window.t_min.min() - 1, window.t_min.max() + 1
    y_lo, y_hi = 1.85, 3.05

    # Static vs. adaptive threshold are distinguished by LINESTYLE (solid vs. dashed), not just
    # color -- in grayscale, "#444444" and "#1f77b4" render as near-identical mid-grays and
    # become impossible to tell apart (confirmed by rendering a desaturated check copy).
    ax.axvspan(episode.t_min.min(), x_hi, color="#2ca02c", alpha=0.07, zorder=0, edgecolor="#2ca02c", linewidth=0.8)
    ax.axhline(base_threshold, color="black", linestyle="-", linewidth=1.5, label="Static threshold", zorder=2)
    ax.step(window.t_min, window.adaptive_threshold, where="post", color="#1f77b4", linestyle="--",
            linewidth=1.8, label="Adaptive threshold", zorder=2)
    ax.axvline(0, color="#d62728", linestyle=":", linewidth=1.4, zorder=1)
    ax.scatter(episode.t_min, episode.max_abs_pred, color="#d62728", marker="o", s=60, zorder=4,
               edgecolors="black", linewidths=0.6, label="Real forecast score")

    if trigger_t is not None:
        trigger_val = trace.set_index("t_min").loc[trigger_t, "max_abs_pred"]
        ax.scatter([trigger_t], [trigger_val], facecolors="none", edgecolors="#1f77b4", s=70, zorder=4, linewidths=1.8)
        ax.annotate(
            "change-point\nflagged here",
            xy=(trigger_t, trigger_val), xytext=(x_lo + 1, 2.45),
            fontsize=8.5, ha="left", va="center",
            arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=0.9, connectionstyle="arc3,rad=0.2"),
        )

    ax.text(0.6, y_hi - 0.08, "onset", color="#d62728", fontsize=9, va="top")
    ax.annotate(
        "Missed by the raw forecast\n(below the static threshold)",
        xy=(episode.t_min.iloc[0], episode.max_abs_pred.iloc[0]), xytext=(-20, y_hi - 0.12),
        fontsize=9, ha="center", va="top",
        arrowprops=dict(arrowstyle="->", color="#444444", lw=0.9, connectionstyle="arc3,rad=-0.2"),
    )
    ax.annotate(
        "Caught by the correction layer\n(above the adaptive threshold)",
        xy=(episode.t_min.iloc[-1], episode.adaptive_threshold.iloc[-1]), xytext=(12, 2.35),
        fontsize=9, ha="center", va="center",
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=0.9, connectionstyle="arc3,rad=-0.2"),
    )

    ax.set_xlabel("Time relative to onset (minutes)")
    ax.set_ylabel("Predicted |z| (forecast score)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.35, linewidth=0.7)
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    fig.subplots_adjust(bottom=0.18)
    fig.text(0.5, 0.03,
              f"SARIMA, onset {ONSET_STR} UTC, KPI {KPI} -- lead time {lead_time_min:.0f} minutes.",
              ha="center", fontsize=8.5, style="italic")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig5_correction_mechanism.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig5_correction_mechanism.png", dpi=150, bbox_inches="tight")
    print(f"saved to {OUT_DIR / 'fig5_correction_mechanism.pdf'} and .png")


def main():
    trace, episode, lead_time_min, base_threshold, trigger_t = load_episode_trace()
    pd.set_option("display.width", 200)
    print(f"static threshold: {base_threshold:.4f}")
    print(f"triggering change-point at t={trigger_t:.1f} min (matures at t={trigger_t + 15:.1f} min)")
    print(episode.round(4).to_string(index=False))
    print(f"lead time: {lead_time_min:.1f} minutes")
    plot(trace, episode, lead_time_min, base_threshold, trigger_t)


if __name__ == "__main__":
    main()
