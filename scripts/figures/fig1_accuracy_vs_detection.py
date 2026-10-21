"""Figure 1 -- paired-window forecast accuracy (MAE) vs. episode-level detection rate.

Design constraint (Figures/files/README.md): must stay interpretable in grayscale/print.
Model identity is carried by marker SHAPE (not color), and validation vs. test by fill
state (filled = validation, hollow = test) -- neither distinction depends on color.

Data source: derived directly from the result files, NOT from a hand-maintained figure CSV.
  - MAE: results/aiops18/table1_cohort_comparison.csv, column `mae_chronos_matched`
        (all six models on Chronos's exact 5,735 val / 5,692 test window subset)
  - Detection rate: results/aiops18/model_comparison_summary.csv, column `detection_rate`
        (common full episode set: 246 val / 232 test)

The superseded Figures/files/fig1_accuracy_vs_detection.csv mixed populations: it carried
full-population MAE for the five densely evaluated models but subset MAE for Chronos. That
reproduced the exact population mismatch the paper's paired-window analysis exists to
remove, and contradicted both Table 1 and this figure's own caption. EXPECTED below pins
the values to the manuscript's Table 1 so the mismatch cannot recur silently.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "aiops18"
OUT_DIR = Path(__file__).resolve().parents[2] / "Figures" / "rendered"

# Marker shape per model (grayscale-safe); color is decorative only.
STYLE = {
    "Persistence": ("o", "#7f7f7f"),
    "SARIMA":      ("s", "#1f77b4"),
    "XGBoost":     ("^", "#ff7f0e"),
    "LSTM":        ("D", "#2ca02c"),
    "PatchTST":    ("P", "#9467bd"),
    "Chronos":     ("*", "#d62728"),
}
ORDER = list(STYLE)
BASE_SIZE = 90
CHRONOS_SIZE_MULT = 1.7

CANON = {"persistence": "Persistence", "sarima": "SARIMA", "xgboost": "XGBoost",
         "lstm": "LSTM", "patchtst": "PatchTST", "chronos": "Chronos"}

# Manuscript Table 1: (paired MAE, detection rate %) keyed by (model, split).
EXPECTED = {
    ("Persistence", "val"):  (0.382, 7.72), ("Persistence", "test"): (0.361, 5.17),
    ("SARIMA", "val"):       (0.340, 6.91), ("SARIMA", "test"):      (0.323, 5.17),
    ("XGBoost", "val"):      (0.417, 9.76), ("XGBoost", "test"):     (0.358, 5.17),
    ("LSTM", "val"):         (0.347, 6.50), ("LSTM", "test"):        (0.325, 4.74),
    ("PatchTST", "val"):     (0.339, 7.32), ("PatchTST", "test"):    (0.319, 4.74),
    ("Chronos", "val"):      (0.338, 6.10), ("Chronos", "test"):     (0.318, 5.17),
}


def load_data():
    acc = pd.read_csv(RESULTS_DIR / "table1_cohort_comparison.csv")
    acc["model"] = acc["model"].map(CANON)
    acc = acc[["model", "split", "mae_chronos_matched"]].rename(
        columns={"mae_chronos_matched": "mae"})

    det = pd.read_csv(RESULTS_DIR / "model_comparison_summary.csv")
    det = det[~det["model_name"].astype(str).str.contains("corrected|smoketest")]
    det["model"] = det["model_name"].map(CANON)
    det = det.dropna(subset=["model"])
    det["detection_rate_pct"] = det["detection_rate"] * 100.0
    det = det[["model", "split", "detection_rate_pct", "n_episodes"]]

    df = acc.merge(det, on=["model", "split"], how="inner")
    df["model"] = pd.Categorical(df["model"], categories=ORDER, ordered=True)
    return df.sort_values(["model", "split"]).reset_index(drop=True)


def sanity_check(df):
    """Fail loudly if the derived values drift from the manuscript's Table 1."""
    assert len(df) == 12, f"expected 12 model-split rows, got {len(df)}"

    for n in df[df.split == "val"].n_episodes.unique():
        assert n == 246, f"validation episode count is {n}, expected 246"
    for n in df[df.split == "test"].n_episodes.unique():
        assert n == 232, f"test episode count is {n}, expected 232"

    for _, r in df.iterrows():
        exp_mae, exp_det = EXPECTED[(r.model, r.split)]
        assert abs(r.mae - exp_mae) < 5e-4, (
            f"{r.model}/{r.split} paired MAE {r.mae:.4f} != Table 1 {exp_mae}")
        assert abs(r.detection_rate_pct - exp_det) < 5e-3, (
            f"{r.model}/{r.split} detection {r.detection_rate_pct:.2f}% != Table 1 {exp_det}%")

    # The paper's central claim: MAE rank and detection rank must not coincide.
    for split in ("val", "test"):
        s = df[df.split == split]
        assert s.sort_values("mae").model.tolist() != \
               s.sort_values("detection_rate_pct", ascending=False).model.tolist(), \
               f"{split}: MAE ordering matches detection ordering, contradicting the paper"

    print("sanity_check: all 12 values match manuscript Table 1")


def plot(df):
    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(figsize=(7, 5.5))

    for model, g in df.groupby("model", sort=True, observed=True):
        marker, color = STYLE[model]
        size = BASE_SIZE * (CHRONOS_SIZE_MULT if model == "Chronos" else 1.0)

        val_row = g[g.split == "val"].iloc[0]
        test_row = g[g.split == "test"].iloc[0]

        ax.scatter(val_row.mae, val_row.detection_rate_pct, marker=marker, s=size,
                   facecolors=color, edgecolors="black", linewidths=0.7, zorder=3)
        ax.scatter(test_row.mae, test_row.detection_rate_pct, marker=marker, s=size,
                   facecolors="none", edgecolors=color, linewidths=1.6, zorder=3)

        ax.annotate(model, (val_row.mae, val_row.detection_rate_pct),
                    xytext=(6, 4), textcoords="offset points", fontsize=9)

    ax.set_xlabel("Paired-window mean absolute error (MAE)")
    ax.set_ylabel("Episode-level detection rate (%)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.35, linewidth=0.7)

    legend_elems = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="black",
                   markeredgecolor="black", markersize=8, label="Validation (filled)"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
                   markeredgecolor="black", markersize=8, markeredgewidth=1.6,
                   label="Test (hollow)"),
    ]
    ax.legend(handles=legend_elems, loc="upper left", frameon=False, fontsize=9)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig1_accuracy_vs_detection.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig1_accuracy_vs_detection.png", dpi=600, bbox_inches="tight")
    print(f"saved to {OUT_DIR / 'fig1_accuracy_vs_detection.pdf'} and .png")


def main():
    df = load_data()
    print(df.to_string(index=False))
    sanity_check(df)
    plot(df)


if __name__ == "__main__":
    main()
