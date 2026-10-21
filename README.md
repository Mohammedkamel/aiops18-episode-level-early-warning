# Episode-Level Evaluation of Forecasting-Based Early Warning in Operational Time Series

**Author:** Mohammed Aleinzi ([ORCID 0000-0003-2595-0611](https://orcid.org/0000-0003-2595-0611)),
Department of Information Systems, Faculty of Computing and Information Technology,
Northern Border University, Rafha 91431, Saudi Arabia.

Analysis code for the paper *"Beyond Forecast Accuracy: Episode-Level Evaluation of
Forecasting-Based Early Warning in Operational Time Series"* (AIOps18 KPI anomaly
anticipation).

The study compares six forecasters (Persistence, SARIMA, XGBoost, LSTM, PatchTST,
zero-shot Chronos) at a 15-minute horizon, routing all of them through one shared
detection pipeline, and evaluates them on episode-level detection rate, lead time, and
false-alarm burden rather than on aggregate forecast error alone.

## 1. Data

This repository contains **no data**. The AIOps18 benchmark is public; download it and
place it as shown below.

- Source: https://github.com/NetManAIOps/KPI-Anomaly-Detection
- Dataset reference: Li et al., *Constructing Large-Scale Real-World Benchmark Datasets
  for AIOps*, arXiv:2208.03938.

```
data/raw/aiops18/     <- Preliminary + Finals releases, unmodified
```

The merged releases give 55 independent KPI series (41 at 60 s, 14 at 300 s). This study
uses the 41 series at 60 s resolution.

## 2. Environments

Two environments are needed, because `chronos-forecasting` pins an older NumPy than the
main analysis stack:

```bash
# Main: preprocessing, Persistence, SARIMA, XGBoost, evaluation, stats, figures, tables
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Torch: LSTM, PatchTST, Chronos only
python3.11 -m venv .venv-torch
.venv-torch/bin/pip install -r requirements-torch.txt
```

## 3. Run order

Steps 2 and 3 are the expensive ones. Everything downstream reads the saved outputs in
`results/aiops18/`, so analysis, figures, and tables can be re-run cheaply on their own.

| # | Command | Env | Produces |
|---|---|---|---|
| 1 | `python scripts/preprocessing/aiops18_preprocessing.py` | main | `processed_data/aiops18/*_windows_h15min_60s.npz` |
| 2 | `python scripts/models/run_aiops18_baselines.py` | main | Persistence, SARIMA, XGBoost predictions + summary rows (~2 h) |
| 3 | `python scripts/models/run_aiops18_phase2.py` | torch | LSTM, PatchTST predictions + summary rows |
| 4 | `python scripts/models/run_aiops18_chronos.py` | torch | Chronos zero-shot predictions (stratified subset; checkpoints and resumes) |
| 5 | `python scripts/models/run_aiops18_correction.py` | main | `{model}_corrected` rows (CUSUM correction layer, post-hoc) |
| 6 | `python scripts/models/table1_cohort_analysis.py` | main | `table1_cohort_comparison.csv` |
| 7 | `python scripts/models/lead_time_false_alarm_analysis.py` | main | `lead_time_and_false_alarm_rate_table.csv` |
| 8 | `python scripts/models/model_specific_calibration_analysis.py` | main | `model_specific_calibration_results.csv` |
| 9 | `python scripts/models/ablation_flat_threshold.py` | main | `ablation_flat_threshold.csv`, `precision_f1_table.csv` |
| 10 | `python scripts/models/matched_recall_comparison.py` | main | `matched_recall_comparison.csv` |
| 11 | `python scripts/models/tier2_checks.py` | main | `threshold_sensitivity.csv` |
| 12 | `python scripts/models/bootstrap_significance_tests.py` | main | `bootstrap_significance_tests.csv` (10,000 series-clustered resamples) |
| 13 | `python scripts/models/multiple_comparisons_correction.py` | main | `multiple_comparisons_correction.csv` (56 tests, within-family Holm) |

## 4. Which script produces which table or figure

### Main paper

| Paper item | Script | Reads |
|---|---|---|
| Table 1 (paired MAE/RMSE, detection rate) | `scripts/models/table1_cohort_analysis.py` | `table1_cohort_comparison.csv`, `model_comparison_summary.csv` |
| Table 2 (test lead time) | `scripts/tables/table_lead_time_summary.py` | `lead_time_and_false_alarm_rate_table.csv` |
| Figure 1 (MAE vs. detection rate) | `scripts/figures/fig1_accuracy_vs_detection.py` | `table1_cohort_comparison.csv`, `model_comparison_summary.csv` |
| Section 4.3 (CUSUM, matched recall, Holm) | `scripts/models/multiple_comparisons_correction.py` | `matched_recall_comparison.csv`, `ablation_flat_threshold.csv` |

### Supplementary

| Paper item | Script | Reads |
|---|---|---|
| Table S1 (configuration) | n/a, documented in `docs/` | |
| Table S2 (threshold-cutoff sensitivity) | `scripts/models/tier2_checks.py` | `threshold_sensitivity.csv` |
| Table S3 (full-population accuracy) | `scripts/models/table1_cohort_analysis.py` | `table1_cohort_comparison.csv` |
| Table S4 (validation lead time) | `scripts/tables/table_lead_time_summary.py` | `lead_time_and_false_alarm_rate_table.csv` |
| Table S5 (model-specific calibration) | `scripts/models/model_specific_calibration_analysis.py` | `model_specific_calibration_results.csv` |
| Tables S6, S7 (correction layer, flat-threshold ablation) | `scripts/models/ablation_flat_threshold.py` | `model_comparison_summary.csv`, `ablation_flat_threshold.csv` |
| Table S8 (56 inferential tests) | `scripts/models/multiple_comparisons_correction.py` | `multiple_comparisons_correction.csv` |
| Figure S1 (PatchTST clipping) | `scripts/figures/fig2_patchtst_clipping.py` | `phase2_run.log`, `patchtst_reclip_run.log` |
| Figure S2 (correction-layer trace) | `scripts/figures/fig5_correction_mechanism.py` | `correction_layer.py` re-run on one episode |

`scripts/figures/fig3_correction_layer_f1.py` and `fig4_episode_overlap_schematic.py`
produce supporting figures retained from earlier drafts; they are not referenced in the
submitted version.

## 5. Key methodological points

- **Threshold calibration** is per-KPI, on the training split only (200 quantile
  candidates, F1-optimal), then frozen for validation and test and shared across all six
  models. See `scripts/models/threshold_calibration.py`.
- **Low-precision exclusion**: KPI series whose F1-optimal threshold reaches under 30%
  training precision are dropped from detection metrics. Defined on training data only
  and applied identically to every model. Retains 26 series, 246 validation and 232 test
  episodes.
- **Episode-level collapse**: overlapping forecast windows referring to one onset collapse
  to a single binary outcome; lead time runs from the earliest qualifying alarm to onset.
  Consecutive overlapping false positives merge into one false-alarm episode. See
  `scripts/models/evaluation_metrics.py`.
- **Causal residual maturity**: a forecast residual only becomes available at `t + H`, so
  the CUSUM correction layer processes residuals by maturity timestamp, never by forecast
  origin. A residual may only affect alarms whose origins fall at or after its maturity.
  See `scripts/models/correction_layer.py`.
- **Uncertainty**: series-clustered bootstrap, 10,000 resamples, resampling KPI series
  rather than windows or episodes.
- **Multiplicity**: 56 tests across 6 pre-specified families, Holm within family. Only
  within-family adjusted results are treated as confirmatory.
- **Chronos population**: evaluated on a stratified subset (5,735 validation / 5,692 test
  windows) for compute reasons. Episode-level detection uses the common full episode set;
  MAE/RMSE comparisons are made on Chronos's exact window subset for all six models.

Fuller rationale is in `docs/aiops18_preprocessing_methodology.md` and
`docs/aiops18_evaluation_methodology.md`.

## 6. Reproducibility notes

- The LSTM and PatchTST runs use fixed seeds. The LSTM showed some initialization
  sensitivity on validation; this is reported in the paper's limitations.
- Per-episode detection outcomes are computed in memory by
  `evaluation_metrics.compute_detection_metrics` and aggregated before being written, so
  no per-episode file is checked in. Reproducing the episode-level agreement figures
  quoted in the paper requires re-running steps 2 to 5.
- Superseded result files from before three bugfixes (causal residual maturity, the
  bootstrap resample count, and the onset ground-truth definition) are excluded by
  `.gitignore`. Only post-fix results back the published numbers.
- The `aiops2022_*` scripts under `scripts/analysis/` and `scripts/preprocessing/` are
  exploratory work on a different dataset and are not part of this paper; their outputs are
  not included here.

## 7. Citation

Aleinzi, M. Beyond Forecast Accuracy: Episode-Level Evaluation of Forecasting-Based Early
Warning in Operational Time Series. (Under review.)

## 8. License

Code released under the MIT License. The AIOps18 dataset remains under its original
license from the NetMan Laboratory, Tsinghua University.
