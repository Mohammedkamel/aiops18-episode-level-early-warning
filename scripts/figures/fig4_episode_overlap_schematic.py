"""Figure 4 -- episode-level window-overlap schematic.

Conceptual diagram, not derived from real per-KPI data (see Figures/files/fig4_episode_overlap_schematic.csv
and its notes): illustrates why a 5-minute stride and 15-minute forecast horizon put every onset
inside three overlapping windows' horizons, motivating the existential-OR episode collapse rule
used throughout the evaluation (Section 3.5 of the manuscript).
"""

import io
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

FIG_DATA = Path(__file__).resolve().parents[2] / "Figures" / "files" / "fig4_episode_overlap_schematic.csv"
OUT_DIR = Path(__file__).resolve().parents[2] / "Figures" / "rendered"


def load_windows():
    # Trailing "# Notes for reconstruction" block uses '#' only at line-start, but color_hex
    # values (e.g. "#1f77b4") also contain '#' mid-line -- pd.read_csv(comment="#") would
    # truncate those cells, so filter out full-comment lines manually instead.
    lines = FIG_DATA.read_text().splitlines()
    data_lines = [ln for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]
    return pd.read_csv(io.StringIO("\n".join(data_lines)))


def plot(windows):
    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(figsize=(8, 3.2))

    bar_height = 0.6
    for i, row in windows.iterrows():
        y = len(windows) - 1 - i
        width = row.horizon_end_min - row.horizon_start_min
        ax.add_patch(FancyBboxPatch(
            (row.horizon_start_min, y - bar_height / 2), width, bar_height,
            boxstyle=f"round,pad=0,rounding_size={bar_height / 2}",
            facecolor=row.color_hex, edgecolor="none", zorder=2, mutation_aspect=1 / 3.2,
        ))
        ax.text(
            row.horizon_start_min - 0.8, y, row.window_label, ha="right", va="center", fontsize=10,
        )

    ax.axvline(0, color="#d62728", linestyle="--", linewidth=1.5, zorder=3)
    ax.text(0, len(windows) - 0.3, "onset (t = 0)", color="#d62728", ha="center", va="bottom", fontsize=9)

    ax.annotate(
        'Detected if ANY\nwindow flags the onset',
        xy=(16, 1), xytext=(20, 1),
        fontsize=9, va="center", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#444444", lw=0.8),
        arrowprops=dict(arrowstyle="->", color="#444444", lw=1.0),
    )

    ax.set_xlim(-14, 32)
    ax.set_ylim(-0.8, len(windows) - 0.2)
    ax.set_xlabel("Time relative to onset (minutes)")
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", alpha=0.35, linewidth=0.7)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig4_episode_overlap_schematic.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig4_episode_overlap_schematic.png", dpi=150, bbox_inches="tight")
    print(f"saved to {OUT_DIR / 'fig4_episode_overlap_schematic.pdf'} and .png")


def main():
    windows = load_windows()
    print(windows.to_string(index=False))
    plot(windows)


if __name__ == "__main__":
    main()
