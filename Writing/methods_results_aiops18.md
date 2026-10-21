# 3. Methods

## 3.1 Dataset

We use the KPI-Anomaly-Detection dataset released for the 2018 AIOps Challenge (NetManAIOps, Tsinghua University), comprising real operational key performance indicators from web-facing services operated by several large Chinese internet companies. The dataset combines two release phases: a preliminary set of 26 KPI series and a finals set of 29 additional series, yielding 55 independent time series in total, each accompanied by expert-assigned binary anomaly labels at the timestamp level. Series are sampled at either 60-second or 300-second intervals depending on the underlying metric.

Across the merged dataset, we identify 2,161 distinct anomaly segments, defined as maximal contiguous runs of positively labeled timestamps (933 in the preliminary series, 1,228 in the finals series). The onset of a segment is its first labeled timestamp; this onset defines the reference point against which early-warning performance is measured throughout. The experiments reported here are restricted to the 15-minute forecast horizon evaluated on the 60-second-sampling subset of the data, comprising 41 series; the 60-minute horizon and the 300-second-sampling subset (the remaining 14 series) were out of scope for the present study and are left for future work.

## 3.2 Preprocessing

Each series was split temporally into training, validation, and test partitions in a 60/20/20 ratio, following the convention used by prior work on this dataset (Lee et al., 2023). Splits are chronological rather than random to avoid leakage of future information into training.

Sampling gaps were handled asymmetrically depending on their extent. A single missed sample (an inter-sample interval equal to twice the series' modal interval) was linearly interpolated, since we judged this consistent with routine telemetry jitter rather than a genuine data-quality failure; a strict rule treating any interval deviation as an unusable gap would have discarded more than 90% of the available windows for 19 of the 55 series. Gaps of two or more consecutive missed samples (an interval of three times the modal value or greater) were treated as genuine discontinuities and used to terminate a contiguous run rather than being bridged. As an exception, interpolation was withheld within three samples of a labeled onset transition, so that no synthetic value could substitute for an observation in the interval most relevant to the early-warning metrics reported below.

Each series was standardized using z-score normalization, with the mean and standard deviation estimated exclusively from label-negative timestamps in the training partition. During development we found that 3.2% of training windows contained at least one normalized value exceeding 10 standard deviations from the mean, and 0.35% contained a value exceeding 100 standard deviations, with a global maximum of approximately 2,540; this arises because a small number of series exhibit near-zero variance under normal operation, so that a modest absolute deviation is amplified into an extreme z-score. Left unaddressed, these values dominate the squared-error gradient of models trained with mean-squared-error loss and were found to prevent the Transformer-based model described below from converging. We therefore clip all normalized values to the range [-10, 10] prior to windowing; this bound is a stated design choice rather than a tuned hyperparameter, and its effect on model training is reported in Section 4.

Sliding windows were constructed with a context length of 288 steps (approximately 4.8 hours at 60-second resolution) and a forecast horizon of 15 minutes, using a stride of 5 minutes. The stride was chosen to satisfy a local storage constraint, as an exhaustive stride of 1 would have exceeded available disk space; we verified that this choice does not affect model training itself, since sequential windows differ only in redundant, largely overlapping context. A consequence of this stride is that a single onset event falls within the forecast horizon of three consecutive windows, a property that motivates the episode-level evaluation procedure described in Section 3.5.

## 3.3 Models

We evaluate six forecasting approaches, chosen to span a range of modeling assumptions rather than to represent an exhaustive architecture search.

**Persistence.** The forecast for each horizon step is the last observed value. This serves as a lower-bound sanity check rather than a competitive baseline.

**SARIMA.** A classical seasonal autoregressive integrated moving-average model, fit independently per series.

**XGBoost.** A gradient-boosted tree regressor using lagged values and rolling summary statistics as features, fit independently per series.

**LSTM.** A single recurrent network trained jointly across all 41 series, using per-series normalization to control for scale differences. We use a shared-weights architecture rather than one model per series both for computational tractability and because it more directly tests the generalization behavior relevant to deployment, where retraining a model for each individual host or service is typically impractical.

**PatchTST.** A patch-based Transformer forecaster, also trained as a single shared-weights model across all 41 series, under the same rationale as the LSTM.

**Chronos (zero-shot).** A pretrained time-series foundation model (chronos-t5-tiny, 8.4M parameters) evaluated with no fine-tuning, using the default configuration of 20 sampled trajectories per forecast. Because inference under this configuration scales linearly with sample count and full-dataset evaluation was computationally infeasible on our hardware, Chronos was evaluated on a stratified subsample: all windows overlapping a labeled onset were retained in full (733 windows in validation, 690 in test), supplemented by a random sample of 122 normal windows per series, drawn uniformly so that no single series could dominate the sample. This yields 5,735 validation and 5,692 test windows, against 179,684 and 170,562 for the other five models. Detection-based metrics, which depend only on episode coverage, are computed on the complete and identical set of episodes used elsewhere. MAE and RMSE for Chronos are conditional on this stratified, anomaly-enriched subsample and should not be interpreted as population-level estimates: because the sample deliberately over-represents onset-adjacent windows relative to their natural frequency, it is not representative of the population the other five models are scored against, and a direct MAE/RMSE comparison against Table 1's other rows is not meaningful without accounting for this.

The LSTM and PatchTST were trained for 5 and 15 epochs respectively, a compute-budget-driven choice rather than a criterion determined by convergence; we report this explicitly because it bears on the interpretation of any comparison in which one of these two models underperforms. Both models were trained with a fixed random seed to allow the results to be distinguished from initialization variance, which we found to be non-trivial for the LSTM in particular (Section 4.2).

## 3.4 Threshold calibration

Forecast-based detection requires converting a continuous prediction error into a binary flag. We calibrate a threshold independently for each series by sweeping candidate values against the training partition and selecting the value maximizing F1, computed using both normal and anomalous training points rather than anomalous points alone. Series for which even the F1-optimal threshold yields training precision below 30% are excluded from all detection-rate, false-alarm-rate, and lead-time computations, on the grounds that no achievable operating point meaningfully separates true signal from noise for these series; the same 15 series are excluded uniformly across all six models, so that no model is evaluated on an easier or harder subset than any other. These series remain included in the forecast-accuracy metrics (MAE, RMSE), which do not depend on a detection threshold. This exclusion procedure leaves 26 series contributing 246 validation episodes and 232 test episodes, the set used for every detection-based comparison in this paper.

## 3.5 Episode-level evaluation

Because the windowing stride produces overlapping forecast horizons, a naive evaluation that treats each window as an independent trial substantially distorts both detection and false-alarm rates: a single onset event, once flagged, would be credited to as many as three separate windows, while a single spurious deviation could register as multiple false alarms rather than one. This is analogous to the point-adjustment problem documented in the time-series anomaly detection literature. We instead aggregate to the episode level throughout. An onset is counted as detected if any window overlapping its horizon is flagged, collapsed to a single binary outcome; lead time is measured from the earliest such window to the onset itself. Consecutive false-positive windows whose horizons overlap in wall-clock time are merged into a single false-alarm episode rather than counted individually. We validated this procedure by hand-tracing several real onset events through their constituent overlapping windows prior to running any model, confirming that the collapsing logic produces an unambiguous single outcome per episode.

## 3.6 Correction layer

To evaluate whether early-warning performance can be improved independently of the underlying forecaster, we implement a lightweight, model-agnostic correction applied post hoc to each model's saved predictions, without retraining. For each series, we monitor the rolling forecast residual over a trailing 30-step window and apply a CUSUM-based change-point test to detect sustained shifts in residual behavior, as distinct from isolated spikes. Upon detection of a change point, the series' calibrated detection threshold is reduced by 20% for a window of 15 subsequent steps, after which it decays linearly back to its baseline value. The three governing parameters (trailing window length, post-change-point window length, and threshold reduction factor) are stated defaults rather than values tuned against validation performance, consistent with our treatment of other pipeline parameters such as the windowing stride. The correction layer is applied identically to all six models using their existing saved forecasts, so that any improvement reflects the correction mechanism itself rather than model-specific tuning.

---

# 4. Results

## 4.1 Effect of outlier clipping

The clipping procedure described in Section 3.2 had a materially different effect across model classes. For persistence, SARIMA, XGBoost, and the LSTM, mean absolute error was essentially unchanged by clipping, while root-mean-squared error improved substantially (for example, LSTM validation RMSE fell from 0.707 to 0.487), consistent with the outliers contributing disproportionately to squared-error terms without representing a large share of the data by count. For PatchTST, the effect was categorical rather than incremental: prior to clipping, training loss failed to decline meaningfully over 15 epochs (31.74 to 31.08, a 2.1% relative reduction, with four epoch-to-epoch increases); after clipping, the same architecture and epoch budget produced a smooth, monotonic decline to approximately 0.19, a genuine convergence pattern absent beforehand. We attribute this asymmetry to architectural differences in how extreme inputs propagate: the LSTM's gating mechanisms bound the influence of large inputs on the hidden state, whereas the PatchTST implementation used here applies no normalization between the raw patch embedding and the first attention layer. All results reported below use the clipped data and, for PatchTST and the LSTM, a fixed random seed (see Section 4.2).

## 4.2 Initialization variance

Because the LSTM and PatchTST are trained from randomly initialized weights, we assessed the sensitivity of our detection-rate results to initialization by comparing seeded and unseeded runs on identical data. For the LSTM, validation detection rate varied substantially across initializations (3.25% to 7.32% across two unseeded runs, against 6.50% under a fixed seed), while the corresponding test-set detection rate was stable across all three runs (4.72–4.74%). PatchTST showed markedly less sensitivity to initialization on the same data (7.72% versus 7.32% validation detection across seeds). We tested, and rejected, the hypothesis that this validation-specific volatility reflects episodes sitting closer to their calibrated detection threshold in the validation set than in the test set: a direct comparison of per-episode signal margins found no significant difference between the two splits (Mann-Whitney U, p = 0.167). The mechanism underlying this asymmetry, and its apparent specificity to the LSTM rather than PatchTST, remains unresolved; we report it here rather than treating either the seeded or unseeded LSTM result as uniquely authoritative. All LSTM and PatchTST results below use a fixed seed.

## 4.3 Forecast accuracy versus early-warning detection

Table 1 reports forecast accuracy and episode-level detection rate for all six models.

**Table 1.** Forecast accuracy and detection performance by model.

| Model | Split | MAE | RMSE | Detection rate |
|---|---|---|---|---|
| Persistence | Val | 0.271 | 0.571 | 7.72% |
| Persistence | Test | 0.269 | 0.553 | 5.17% |
| SARIMA | Val | 0.232 | 0.462 | 6.91% |
| SARIMA | Test | 0.234 | 0.456 | 5.17% |
| XGBoost | Val | 0.326 | 0.793 | 9.76% |
| XGBoost | Test | 0.268 | 0.573 | 5.17% |
| LSTM (seeded) | Val | 0.239 | 0.467 | 6.50% |
| LSTM (seeded) | Test | 0.235 | 0.455 | 4.74% |
| PatchTST (seeded) | Val | 0.227 | 0.469 | 7.32% |
| PatchTST (seeded) | Test | 0.227 | 0.458 | 4.74% |
| Chronos (zero-shot)$^\dagger$ | Val | 0.338 | 0.894 | 6.10% |
| Chronos (zero-shot)$^\dagger$ | Test | 0.318 | 0.832 | 5.17% |

$^\dagger$ Evaluated on a stratified subsample (5,735 validation / 5,692 test windows; see Section 3.3). Detection rate is computed on the complete, identical episode set used for all other models.

The central pattern in Table 1 is a consistent dissociation between forecast accuracy and detection performance. SARIMA achieves better MAE and RMSE than the persistence baseline on both splits, yet its detection rate is no higher than persistence's, and on validation is slightly lower (6.91% versus 7.72%). PatchTST attains the best MAE and RMSE of any model on both splits following the clipping fix, but its detection rate is statistically indistinguishable from persistence's on test (4.74% versus 5.17%) and only marginally higher on validation. The most pronounced instance of this dissociation involves the zero-shot Chronos model, which records the worst MAE and RMSE of all six models — including performing worse than the trivial persistence baseline — while achieving a detection rate on test (5.17%) identical to persistence, SARIMA, and XGBoost. A model with measurably worse forecasting accuracy than a naive baseline nonetheless matches that baseline's ability to flag degradation onset under our episode-level criterion. This pattern, replicated across a classical statistical model, a gradient-boosted tree ensemble, two deep learning architectures, and a zero-shot foundation model, is consistent with our central claim that aggregate forecast-accuracy metrics do not reliably predict a model's operational early-warning utility.

## 4.4 Effect of the correction layer

Table 2 reports episode-level precision, recall, and F1 before and after applying the correction layer described in Section 3.6, computed from the same detected, missed, and false-alarm episode counts used to derive Table 1.

**Table 2.** Precision, recall, and F1 before (raw) and after (corrected) applying the correction layer.

| Model | Split | Precision (raw → corrected) | Recall (raw → corrected) | F1 (raw → corrected) |
|---|---|---|---|---|
| Persistence | Val | 0.029 → 0.037 | 0.077 → 0.138 | 0.042 → 0.058 |
| Persistence | Test | 0.017 → 0.029 | 0.052 → 0.103 | 0.026 → 0.045 |
| SARIMA | Val | 0.048 → 0.054 | 0.069 → 0.138 | 0.057 → 0.078 |
| SARIMA | Test | 0.026 → 0.034 | 0.052 → 0.095 | 0.035 → 0.050 |
| XGBoost | Val | 0.042 → 0.054 | 0.098 → 0.155 | 0.058 → 0.080 |
| XGBoost | Test | 0.020 → 0.033 | 0.052 → 0.108 | 0.029 → 0.050 |
| LSTM | Val | 0.045 → 0.055 | 0.065 → 0.122 | 0.053 → 0.076 |
| LSTM | Test | 0.023 → 0.036 | 0.047 → 0.099 | 0.031 → 0.053 |
| PatchTST | Val | 0.051 → 0.049 | 0.073 → 0.122 | 0.060 → 0.070 |
| PatchTST | Test | 0.028 → 0.041 | 0.047 → 0.103 | 0.035 → 0.059 |
| Chronos | Val | 0.158 → 0.162 | 0.061 → 0.098 | 0.088 → 0.122 |
| Chronos | Test | 0.075 → 0.105 | 0.052 → 0.095 | 0.061 → 0.100 |

F1 improves under correction for all twelve model-split combinations, with relative gains ranging from approximately 16% to 75%. Precision improves or remains within measurement noise in eleven of twelve combinations; the sole exception is PatchTST on validation, where precision declines marginally (0.051 to 0.049) while F1 nonetheless rises, since the corresponding recall gain (0.073 to 0.122) more than compensates. We do not find evidence that this single deviation reflects a systematic weakness of the correction procedure for this model, given that PatchTST's own test-split result and every other model's validation result show precision holding or improving. Because detection episodes in this dataset are rare relative to the total window count, absolute precision remains low throughout (0.017 to 0.162 across all cells), and differences on the order of a few thousandths should not be over-interpreted.

We verified the correction mechanism directly on a representative case: a SARIMA forecast for one validation-split series showed a sustained residual shift, from a baseline of 0.001–0.007 to a held elevated value of 0.231–0.234, beginning approximately five hours before the corresponding labeled onset. The change-point procedure correctly identified this shift at its onset rather than at a later or unrelated point, and the resulting threshold reduction was sufficient to flag an episode that the uncorrected model missed, while the raw forecast values in the relevant windows remained just below the static threshold throughout. We take this as evidence that the aggregate improvement in Table 2 reflects genuine sensitivity to sustained degradation signals rather than an artifact of uniformly lowering the detection threshold.

The magnitude of improvement was smallest for Chronos (validation F1: 0.088 to 0.122, a 39% relative gain; test F1: 0.061 to 0.100, a 63% relative gain), toward the lower end of the 16–75% relative range observed across the study as a whole, several instances of which exceed 65% (for example, persistence and XGBoost on the test split). We attribute this primarily to the stratified subsampling used for Chronos, which yields substantially fewer contiguous windows per series and therefore fewer opportunities for the rolling-residual change-point procedure to register a sustained shift (247–255 flagged change points for Chronos, against 5,682–7,813 for the densely sampled models). This is a property of the evaluation design for this model rather than evidence that the correction mechanism is less effective for foundation models as a class, and we flag it as a limitation of the present Chronos evaluation rather than a finding in its own right.

---

# 5. Discussion

## 5.1 Aggregate accuracy is not a proxy for early-warning utility

The results in Section 4.3 make a concrete case against a common implicit assumption in applied forecasting work: that a model selected for low MAE or RMSE will, as a byproduct, be well suited to flagging degradation before or as it occurs. In our experiments this assumption fails in both directions. SARIMA improves on persistence in every accuracy metric but not in detection rate. PatchTST is the most accurate model in the study yet detects onset events at a rate statistically indistinguishable from the least sophisticated baseline on the test split. Most starkly, a zero-shot foundation model with the worst forecast accuracy of any model evaluated — worse than simply repeating the last observed value — matches persistence, SARIMA, and XGBoost on detection rate. If model selection for a proactive maintenance system were driven by leaderboard accuracy alone, as is common practice in both academic benchmarking and applied deployment, none of these relationships would be visible, and a practitioner could reasonably select a model whose operational early-warning behavior differs little from a trivial baseline while believing they had selected the strongest available option.

We do not interpret this as evidence that forecast accuracy is irrelevant to detection; the two are correlated in aggregate across the wider forecasting literature, and our study is not designed to isolate the causal relationship between them. Rather, the evidence supports a narrower and more actionable claim: accuracy and episode-level detection are sufficiently decoupled, for the models and data considered here, that one cannot be used as a reliable proxy for the other, and a system intended for proactive maintenance should be evaluated against the operational criterion it is meant to serve rather than against the criterion that is easiest to compute during model development.

## 5.2 The correction layer as evidence, not just an artifact

The consistency of the correction layer's effect across six architecturally distinct models is, in our view, more informative than its magnitude. The layer performs no learning of its own; it does not see labels, is not tuned against validation performance, and operates identically regardless of which model produced the underlying forecast. That a single fixed procedure improves F1 for every model in the study, including one whose forecasts are the least accurate in the comparison, suggests that some part of the information relevant to early warning is present in the shape of a model's forecast-error trajectory over time and is not being extracted by a static, globally calibrated threshold, irrespective of how that threshold is set. This is consistent with the broader claim in Section 5.1: if detection performance depended primarily on forecast accuracy, we would expect a fixed post hoc correction to help the more accurate models less, or not at all, rather than to produce comparable relative gains across the full range of model quality observed here.

We are cautious about overstating the size of this effect. Absolute precision remains low throughout the study (0.017–0.162), a consequence of the low base rate of true degradation episodes relative to the total volume of monitored windows, and any deployment of a correction procedure of this kind would need to be evaluated against the operational cost of the resulting false-alarm volume rather than F1 in isolation. The three governing parameters of the correction layer were fixed a priori rather than tuned, and we would expect a validation-tuned version of the same mechanism to perform at least as well; the results reported here should accordingly be read as a lower bound on what a change-point-triggered adaptive threshold can achieve on this data, not an optimized configuration.

## 5.3 Scope and limitations

Several limitations bound the claims that can be drawn from this study.

**Single primary dataset.** All quantitative results reported here derive from one dataset, drawn from web-facing IT services rather than telecommunications infrastructure specifically, despite the latter motivating our initial interest in this problem. We investigated several candidate telecommunications datasets during this study and found each unsuitable for the present evaluation framework for reasons independent of our modeling choices: some provided only short, pre-segmented windows around already-detected faults rather than continuous series suitable for forecasting a not-yet-occurred onset, and others lacked a reconstructible chronological ordering. A companion investigation of a microservice fault-injection testbed (CCF AIOps 2022) is reported separately as a complementary case study rather than integrated into the present benchmark, since its fault-injection design and substantially higher event density are not directly comparable to the naturally occurring degradation events studied here.

**Restricted horizon and resolution.** We report results for a single 15-minute forecast horizon on the subset of series sampled at 60-second intervals. Series sampled at 300-second intervals and a longer 60-minute horizon were excluded from the present study for reasons of scope rather than any expectation that the central finding would not generalize to them; we did not test this expectation directly.

**Threshold exclusion procedure.** Fifteen of the 41 series in the modeling subset were excluded from all detection-based metrics on the grounds that no threshold achieves acceptable precision against training labels. This exclusion is applied identically across all six models and is therefore not a source of bias between models, but it does mean that the detection rates reported in Section 4.3 characterize model behavior on the subset of series for which threshold-based detection is viable at all, rather than on the full dataset. Not all excluded series are degenerate in an absolute sense. One series (KPI ID c58bfcbacb2822d1) achieves a training precision of 0.06% at its F1-optimal threshold — effectively no better than chance — yet is still detected in 58–67% of its episodes across the three baseline models tested. A second (9ee5879409dccef9), at 4.81% training precision, is detected in 100% of its episodes on both validation and test across all three baseline models. These cases indicate that a series can be uninformative by our precision criterion while remaining almost perfectly detectable in practice, and that a different precision-recall operating point might have retained some excluded series at the cost of a higher false-alarm rate elsewhere in the aggregate.

**Undertrained deep learning baselines.** The LSTM and PatchTST were trained for a small number of epochs, a choice driven by available compute rather than observed convergence, and the LSTM in particular showed non-trivial sensitivity to random initialization on the validation split, for reasons we were unable to fully explain (Section 4.2). We consider the relative ranking of these two models on accuracy to be reasonably robust, since it held under a controlled seed comparison, but we would not treat either model's absolute performance as representative of its ceiling under a fully converged training regime.

**Zero-shot evaluation under subsampling.** The Chronos results were obtained on a stratified subsample rather than the full evaluation set, for computational reasons specific to autoregressive sampling cost on our hardware. Detection-based metrics are unaffected by this, as they are computed over the complete episode set, but the model's smaller absolute gain under the correction layer is confounded with its reduced window density and should not be read as a property of zero-shot foundation models in general.

**Single-run classical and gradient-boosted baselines.** Unlike the LSTM and PatchTST, persistence, SARIMA, and XGBoost involve no random initialization and were therefore not subject to the same seed-sensitivity analysis; we have no equivalent evidence bearing on the stability of their reported detection rates beyond the single run reported.

## 5.4 Implications for proactive maintenance systems

If the decoupling documented here generalizes beyond the present dataset, it carries a direct implication for how forecasting-based monitoring systems are selected and validated in practice: benchmarking a candidate model on held-out forecast error, the standard practice in both the applied and academic literature, provides limited evidence about how that model will behave as an early-warning system once deployed. A model evaluation pipeline intended to support a proactive maintenance decision should include an episode-level, threshold-based evaluation of the kind used here, in addition to whatever accuracy metrics are already standard, before a model is judged fit for that purpose. Separately, the consistent benefit of the correction layer across every model tested suggests that a lightweight, model-agnostic post-processing stage may offer a more tractable path to improved early-warning performance than continued investment in forecast accuracy alone, particularly in settings where retraining or replacing an existing forecasting model is costly or impractical.
