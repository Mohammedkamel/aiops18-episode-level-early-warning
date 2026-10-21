"""Figure 3 -- correction layer effect on episode-level F1 (dumbbell/slope chart).

Reads results/aiops18/model_comparison_summary.csv (raw and *_corrected rows, written by
run_aiops18_baselines.py / run_aiops18_correction.py -- the current, causal-fix-aware pipeline)
rather than the stale Figures/files/fig3_correction_layer_f1.csv, which was confirmed to still
hold pre-causal-fix corrected values (see audit). Precision/recall/F1 are derived here from
n_episodes, detection_rate (=recall), and n_false_alarm_episodes (=fp), then cross-checked
against results/aiops18/causal_fix_comparison.csv's NEW_corrected_* columns as a guard against
silently plotting the wrong numbers again.

Two panels (validation left, test right), one horizontal dumbbell per model: hollow marker =
raw F1, filled marker = corrected F1, same color per model. Row order top-to-bottom: Chronos,
PatchTST, LSTM, XGBoost, SARIMA, Persistence (matches Figures/files/README.md's description of
the original figure).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "aiops18"
OUT_DIR = Path(__file__).resolve().parents[2] / "Figures" / "rendered"

MODEL_ORDER = ["chronos", "patchtst", "lstm", "xgboost", "sarima", "persistence"]
MODEL_LABEL = {
    "persistence": "Persistence", "sarima": "SARIMA", "xgboost": "XGBoost",
    "lstm": "LSTM", "patchtst": "PatchTST", "chronos": "Chronos",
}
MODEL_COLOR = {
    "persistence": "#7f7f7f", "sarima": "#1f77b4", "xgboost": "#ff7f0e",
    "lstm": "#2ca02c", "patchtst": "#9467bd", "chronos": "#d62728",
}
SPLIT_LABEL = {"val": "Validation", "test": "Test"}


def precision_recall_f1(n_episodes, detection_rate, n_false_alarm_episodes):
    tp = round(detection_rate * n_episodes)
    fp = n_false_alarm_episodes
    recall = detection_rate
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan")
    return precision, recall, f1


def load_f1_table():
    df = pd.read_csv(RESULTS_DIR / "model_comparison_summary.csv")
    df = df[df.dataset == "aiops18"]

    rows = []
    for model in MODEL_ORDER:
        for split in ["val", "test"]:
            raw = df[(df.model_name == model) & (df.split == split)].iloc[0]
            corr = df[(df.model_name == model + "_corrected") & (df.split == split)].iloc[0]
            _, _, f1_raw = precision_recall_f1(raw.n_episodes, raw.detection_rate, raw.n_false_alarm_episodes)
            _, _, f1_corr = precision_recall_f1(corr.n_episodes, corr.detection_rate, corr.n_false_alarm_episodes)
            rows.append(dict(model=model, split=split, f1_raw=f1_raw, f1_corrected=f1_corr))
    return pd.DataFrame(rows)


def sanity_check_against_causal_fix_comparison(f1_table):
    """model_comparison_summary.csv has no F1 column directly -- it's derived here from
    n_episodes/detection_rate/n_false_alarm_episodes above. Cross-check that derivation against
    causal_fix_comparison.csv's NEW_corrected_f1 (the file documenting the post-causal-fix
    values) so a silent unit/derivation bug can't reintroduce a fig3-style staleness problem.
    """
    ref = pd.read_csv(RESULTS_DIR / "causal_fix_comparison.csv")
    merged = f1_table.merge(ref, on=["model", "split"])
    max_abs_err = (merged.f1_corrected - merged.NEW_corrected_f1).abs().max()
    assert max_abs_err < 1e-3, f"derived corrected F1 diverges from causal_fix_comparison.csv by {max_abs_err}"
    print(f"sanity check OK: derived corrected F1 matches causal_fix_comparison.csv (max abs err {max_abs_err:.2e})")


def plot(f1_table):
    plt.rcParams["font.family"] = "serif"
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharex=True, sharey=True)
    x_max = 0.135
    y_positions = {m: i for i, m in enumerate(reversed(MODEL_ORDER))}

    for ax, split in zip(axes, ["val", "test"]):
        sub = f1_table[f1_table.split == split]
        for _, row in sub.iterrows():
            y = y_positions[row.model]
            color = MODEL_COLOR[row.model]
            ax.plot([row.f1_raw, row.f1_corrected], [y, y], color=color, lw=1.5, zorder=1, alpha=0.8)
            ax.scatter(row.f1_raw, y, facecolors="none", edgecolors=color, s=55, zorder=2, linewidths=1.5)
            ax.scatter(row.f1_corrected, y, facecolors=color, edgecolors=color, s=55, zorder=2)

        ax.set_title(SPLIT_LABEL[split])
        ax.set_xlim(0, x_max)
        ax.set_xlabel("Episode-level F1")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.35, linewidth=0.7)

    axes[0].set_yticks(list(y_positions.values()))
    axes[0].set_yticklabels([MODEL_LABEL[m] for m in y_positions])
    axes[1].tick_params(axis="y", left=False)

    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor="black",
                   markersize=7, markeredgewidth=1.5, label="Raw (before)"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="black", markeredgecolor="black",
                   markersize=7, label="Corrected (after)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig3_correction_layer_f1.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig3_correction_layer_f1.png", dpi=150, bbox_inches="tight")
    print(f"saved to {OUT_DIR / 'fig3_correction_layer_f1.pdf'} and .png")


def main():
    f1_table = load_f1_table()
    sanity_check_against_causal_fix_comparison(f1_table)
    pd.set_option("display.width", 200)
    print(f1_table.round(4).to_string(index=False))
    plot(f1_table)


if __name__ == "__main__":
    main()
