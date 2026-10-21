"""Figure 2 -- PatchTST training loss, before vs. after clipping inputs to +/-10 sigma.

Two side-by-side panels, not one overlaid dual-axis plot: the loss scales differ by ~150x
(~31 vs. ~0.2), so sharing one y-axis would flatten the after-clipping curve to a near-flat
line. Both panels share the same x-axis (epoch, 1-15).

Source: Figures/files/fig2_patchtst_clipping.csv. Verified directly against two real training
logs rather than trusted as-is: loss_before_clipping matches results/aiops18/phase2_run.log
(the original pre-clip PatchTST run) and loss_after_clipping matches
results/aiops18/patchtst_reclip_run.log, both to 4 decimal places across all 15 epochs.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

FIG_DATA = Path(__file__).resolve().parents[2] / "Figures" / "files" / "fig2_patchtst_clipping.csv"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "aiops18"
OUT_DIR = Path(__file__).resolve().parents[2] / "Figures" / "rendered"


def load_data():
    return pd.read_csv(FIG_DATA)


def parse_log_losses(log_path, n_epochs=15):
    losses = []
    for line in log_path.read_text().splitlines():
        if "[patchtst] epoch" in line and "mean MSE loss" in line:
            losses.append(float(line.strip().rsplit("=", 1)[1]))
    assert len(losses) == n_epochs, f"{log_path}: expected {n_epochs} epochs, found {len(losses)}"
    return losses


def sanity_check_against_logs(df):
    before_log = parse_log_losses(RESULTS_DIR / "phase2_run.log")
    after_log = parse_log_losses(RESULTS_DIR / "patchtst_reclip_run.log")
    max_err_before = max(abs(a - b) for a, b in zip(df.loss_before_clipping, before_log))
    max_err_after = max(abs(a - b) for a, b in zip(df.loss_after_clipping, after_log))
    assert max_err_before < 1e-3, f"loss_before_clipping diverges from phase2_run.log by {max_err_before}"
    assert max_err_after < 1e-3, f"loss_after_clipping diverges from patchtst_reclip_run.log by {max_err_after}"
    print(f"sanity check OK: fig2 CSV matches training logs "
          f"(max abs err before={max_err_before:.2e}, after={max_err_after:.2e})")


def plot(df):
    plt.rcParams["font.family"] = "serif"
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(9.5, 4))

    for ax, col, title, y_range in [
        (ax_l, "loss_before_clipping", "Before clipping", (30.5, 32.0)),
        (ax_r, "loss_after_clipping", "After clipping (±10σ)", (0.18, 0.26)),
    ]:
        ax.plot(df.epoch, df[col], color="black", marker="o", markersize=4, linewidth=1.3)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_xticks(range(1, 16, 2))
        ax.set_ylim(*y_range)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.35, linewidth=0.7)

    ax_l.set_ylabel("Mean MSE loss")
    fig.suptitle("")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig2_patchtst_clipping.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig2_patchtst_clipping.png", dpi=600, bbox_inches="tight")
    print(f"saved to {OUT_DIR / 'fig2_patchtst_clipping.pdf'} and .png")


def main():
    df = load_data()
    sanity_check_against_logs(df)
    print(df.to_string(index=False))
    plot(df)


if __name__ == "__main__":
    main()
