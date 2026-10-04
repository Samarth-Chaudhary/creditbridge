# Experiment Report: exp_phase1_20261004T061033Z

- **Model Name**: Logistic Regression
- **Status**: CHAMPION
- **Timestamp**: 20261004T061033Z
- **Git Commit**: `5c7fd7624fc73896b50f69391c94cb145c165516`
- **Random Seed**: 42

## Headline Performance Summary
| Metric | Out-of-Time (OOT) | Test Set | Validation | Train |
| :--- | :--- | :--- | :--- | :--- |
| **ROC-AUC** | **0.9669** | 0.9802 | 0.9772 | 0.9820 |
| **PR-AUC** | **0.7823** | 0.8972 | 0.8573 | 0.8246 |
| **KS-Statistic (%)** | **82.30%** | 86.26% | 85.23% | 88.02% |
| **Gini** | **0.9338** | 0.9605 | 0.9544 | 0.9641 |
| **Brier Score** | **0.0595** | 0.0565 | 0.0524 | 0.0528 |
| **Precision** | **0.5862** | 0.6087 | 0.6229 | 0.5902 |
| **Recall** | **0.8608** | 0.9333 | 0.8963 | 0.9455 |
| **F1-Score** | **0.6974** | 0.7368 | 0.7350 | 0.7267 |

## Champion Selection Status
- Decision: `CHAMPION_SELECTED`
- Rationale: Selected Logistic Regression as champion under regulatory parsimony rule. XGBoost performance lead (+-0.0082 AUC) is within 0.03 tolerance, making linear monotonicity and exact log-odds regulatory attribution preferred for underwriting.

All figures generated deterministically and verifiable against `artifact_hash.txt`.
