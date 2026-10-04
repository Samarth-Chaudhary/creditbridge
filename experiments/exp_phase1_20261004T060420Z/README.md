# Experiment Report: exp_phase1_20261004T060420Z

- **Model Name**: XGBoost Classifier
- **Status**: NO_CHAMPION
- **Timestamp**: 20261004T060420Z
- **Git Commit**: `5c7fd7624fc73896b50f69391c94cb145c165516`
- **Random Seed**: 42

## Headline Performance Summary
| Metric | Out-of-Time (OOT) | Test Set | Validation | Train |
| :--- | :--- | :--- | :--- | :--- |
| **ROC-AUC** | **0.9587** | 0.9731 | 0.9688 | 0.9939 |
| **PR-AUC** | **0.7799** | 0.8786 | 0.8613 | 0.9391 |
| **KS-Statistic (%)** | **81.86%** | 82.42% | 80.49% | 95.20% |
| **Gini** | **0.9175** | 0.9463 | 0.9375 | 0.9878 |
| **Brier Score** | **0.0511** | 0.0472 | 0.0444 | 0.0316 |
| **Precision** | **0.6436** | 0.6783 | 0.6869 | 0.7263 |
| **Recall** | **0.7658** | 0.8667 | 0.8293 | 0.9919 |
| **F1-Score** | **0.6994** | 0.7610 | 0.7514 | 0.8386 |

## Champion Selection Status
- Decision: `NO-CHAMPION`
- Rationale: No candidate satisfied all mandatory model-risk gates. Production promotion blocked.

All figures generated deterministically and verifiable against `artifact_hash.txt`.
