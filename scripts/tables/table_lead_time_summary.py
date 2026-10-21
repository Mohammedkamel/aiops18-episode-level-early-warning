"""Lead-time summary table -- currently only reported as narrative ranges in the manuscript
(Sec. 4.3, last paragraph: "median lead time is consistently 10-13 minutes... between 58% and
94%... 0-9%"). This tabulates the same raw-model numbers per model x split so a reader can
check the claim directly instead of trusting a prose range.

Source: results/aiops18/lead_time_and_false_alarm_rate_table.csv (raw_* columns -- the
manuscript paragraph this supports explicitly says "across all six raw models", not corrected),
written by scripts/models/lead_time_false_alarm_analysis.py. Already verified in the earlier
audit: every raw_lead_time_median_seconds falls in 600-780s (10-13min), raw_frac_of_detected_ge_10min
ranges 58.3-94.1%, raw_frac_of_detected_at_or_after_onset ranges 0-9.09% -- matching the
manuscript's prose exactly.
"""

from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "aiops18"
OUT_DIR = Path(__file__).resolve().parents[2] / "Tables" / "rendered"

MODEL_ORDER = ["persistence", "sarima", "xgboost", "lstm", "patchtst", "chronos"]
MODEL_LABEL = {
    "persistence": "Persistence", "sarima": "SARIMA", "xgboost": "XGBoost",
    "lstm": "LSTM (seeded)", "patchtst": "PatchTST (seeded)", "chronos": "Chronos (zero-shot)",
}
SPLIT_LABEL = {"val": "Val", "test": "Test"}


def load_table():
    df = pd.read_csv(RESULTS_DIR / "lead_time_and_false_alarm_rate_table.csv")
    rows = []
    for model in MODEL_ORDER:
        for split in ["val", "test"]:
            r = df[(df.model == model) & (df.split == split)].iloc[0]
            rows.append(dict(
                model=MODEL_LABEL[model],
                split=SPLIT_LABEL[split],
                n_detected=int(r.raw_n_detected),
                n_episodes=int(r.raw_n_episodes),
                median_min=r.raw_lead_time_median_seconds / 60,
                iqr_low_min=r.raw_lead_time_iqr_low_seconds / 60,
                iqr_high_min=r.raw_lead_time_iqr_high_seconds / 60,
                pct_ge_10min=100 * r.raw_frac_of_detected_ge_10min,
                pct_at_or_after_onset=100 * r.raw_frac_of_detected_at_or_after_onset,
            ))
    return pd.DataFrame(rows)


def to_markdown(table):
    lines = [
        "**Table 3.** Lead-time distribution for raw (uncorrected) detections, by model and split. "
        "\"$\\geq$10 min early\" is the share of detected episodes flagged at least 10 minutes "
        "before onset; \"At/after onset\" is the share flagged at or after the onset itself.",
        "",
        "| Model | Split | N detected | Median lead time | IQR | % $\\geq$10 min early | % at/after onset |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in table.iterrows():
        lines.append(
            f"| {r.model} | {r.split} | {r.n_detected}/{r.n_episodes} | "
            f"{r.median_min:.0f} min | [{r.iqr_low_min:.0f}, {r.iqr_high_min:.0f}] min | "
            f"{r.pct_ge_10min:.0f}% | {r.pct_at_or_after_onset:.0f}% |"
        )
    return "\n".join(lines)


def sanity_check(table):
    assert table.median_min.between(10, 13).all(), "median lead time outside manuscript's stated 10-13min range"
    assert table.pct_ge_10min.between(58, 94.2).all(), "%>=10min outside manuscript's stated 58-94% range"
    assert table.pct_at_or_after_onset.between(0, 9.1).all(), "%at/after-onset outside manuscript's stated 0-9% range"
    print("sanity check OK: every row falls within the manuscript's stated ranges")


def main():
    table = load_table()
    sanity_check(table)
    pd.set_option("display.width", 200)
    print(table.round(2).to_string(index=False))

    md = to_markdown(table)
    print()
    print(md)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "table3_lead_time_summary.md"
    out_path.write_text(md + "\n")
    print(f"\nsaved to {out_path}")


if __name__ == "__main__":
    main()
