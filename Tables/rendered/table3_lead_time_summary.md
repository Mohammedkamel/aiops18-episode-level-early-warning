**Table 3.** Lead-time distribution for raw (uncorrected) detections, by model and split. "$\geq$10 min early" is the share of detected episodes flagged at least 10 minutes before onset; "At/after onset" is the share flagged at or after the onset itself.

| Model | Split | N detected | Median lead time | IQR | % $\geq$10 min early | % at/after onset |
|---|---|---|---|---|---|---|
| Persistence | Val | 19/246 | 11.0 min | [9.5, 12.0] min | 74% | 5% |
| Persistence | Test | 12/232 | 12.5 min | [10.8, 14.0] min | 83% | 8% |
| SARIMA | Val | 17/246 | 12.0 min | [11.0, 13.0] min | 94% | 0% |
| SARIMA | Test | 12/232 | 12.5 min | [10.8, 14.0] min | 83% | 8% |
| XGBoost | Val | 24/246 | 10.0 min | [7.0, 12.0] min | 58% | 4% |
| XGBoost | Test | 12/232 | 12.0 min | [4.3, 14.0] min | 67% | 8% |
| LSTM (seeded) | Val | 16/246 | 11.5 min | [10.0, 13.0] min | 81% | 0% |
| LSTM (seeded) | Test | 11/232 | 12.0 min | [10.5, 13.5] min | 82% | 9% |
| PatchTST (seeded) | Val | 18/246 | 11.5 min | [10.0, 12.8] min | 89% | 0% |
| PatchTST (seeded) | Test | 11/232 | 12.0 min | [10.5, 13.5] min | 82% | 0% |
| Chronos (zero-shot) | Val | 15/246 | 12.0 min | [10.0, 12.5] min | 87% | 0% |
| Chronos (zero-shot) | Test | 12/232 | 12.5 min | [10.8, 14.0] min | 83% | 8% |
