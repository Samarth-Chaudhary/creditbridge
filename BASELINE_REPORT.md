# CreditBridge Baseline Audit & Pre-Rebuild Discrepancy Report

**Date**: 2026-10-04  
**Audit Author**: Principal ML Engineer, Model-Risk Engineer, Technical Reviewer  
**Status**: PERMANENT BASELINE — PRESERVE UNMODIFIED  

---

## 1. Environment & Exact Dependency Inventory

### 1.1 Python Runtimes
- **System Global Python**: Python 3.11.6 (`C:\Users\Samarth Chaudhary\AppData\Local\Programs\Python\Python311\python.exe`)
- **Isolated Virtual Environment (`.venv`)**: Python 3.11.6 (`c:\Users\Samarth Chaudhary\Downloads\creditbridge\.venv\Scripts\python.exe`)

### 1.2 Dependency Versions (Pinned in `.venv` vs Global)
| Package | Global Environment | Project `.venv` | `requirements.txt` Specification | Status / Incompatibility Note |
| :--- | :--- | :--- | :--- | :--- |
| **numpy** | 2.4.6 | 1.26.4 | `>=1.24.0` | Global numpy 2.4.6 caused ABI C-extension `ValueError` in pandas/sklearn. Pinned to `<2.0.0` (1.26.4) in `.venv`. |
| **pandas** | 2.2.1 | 3.0.6 | `>=2.0.0` | Operational in `.venv`. |
| **scikit-learn** | 1.4.1.post1 | 1.9.1 | `>=1.3.0` | Unpickling older model generated `InconsistentVersionWarning` (trained with 1.8.0). |
| **scipy** | 1.17.1 | 1.17.1 | `>=1.11.0` | Operational in `.venv`. |
| **xgboost** | 3.2.0 | 3.2.0 | `>=2.0.0` | Operational in `.venv`. |
| **shap** | 0.51.0 | 0.49.1 | `>=0.44.0` | Operational in `.venv`. |
| **joblib** | 1.5.3 | 1.6.0 | `>=1.3.0` | Operational in `.venv`. |
| **matplotlib** | 3.10.8 | 3.11.2 | `>=3.7.0` | Operational in `.venv`. |
| **streamlit** | 1.64.0 | 1.65.0 | `>=1.30.0` | Operational in `.venv`. |
| **seaborn** | 0.13.2 | 0.13.2 | `>=0.12.0` | Operational in `.venv`. |
| **plotly** | 7.1.0 | 7.1.0 | `>=5.18.0` | Operational in `.venv`. |
| **faker** | 40.19.1 | 40.40.0 | `>=19.0.0` | Operational in `.venv`. |
| **pytest** | 8.1.1 | 9.1.1 | N/A (test runner) | Operational in `.venv`. |

---

## 2. Baseline Test Execution Results

- **Command Executed**: `.\.venv\Scripts\pytest -v`
- **Result**: `111 passed, 3 warnings in 159.56s (0:02:39)`
- **Test Inventory Breakdown**:
  - `tests/test_adversarial_and_e2e_fixtures.py`: 28 passed
  - `tests/test_csv_parser.py`: 15 passed
  - `tests/test_dashboard_smoke.py`: 4 passed
  - `tests/test_data_quality.py`: 12 passed
  - `tests/test_model_integration.py`: 17 passed
  - `tests/test_privacy_security.py`: 12 passed (Note: `test_raw_statement_is_never_written_to_disk` walked all files including `.venv`, causing a 2-minute stall)
  - `tests/test_real_data_features.py`: 15 passed
  - `tests/test_synthetic_baseline_regression.py`: 8 passed
- **Static Typing**: `npx --yes pyright`
  - Result: `0 errors, 0 warnings, 0 informations`

---

## 3. Current Dataset Dimensions & Class Prevalence

- **Dataset File**: `data/synthetic_borrowers.csv`
- **File Size**: 1,193,374 bytes
- **Dimensions**: 8,000 rows × 23 columns
- **Class Balance (`defaulted`)**:
  - `0` (Non-default): 6,883 (86.04%)
  - `1` (Default): 1,117 (13.96%)
  - Class prevalence: **13.96%** (calibrated prior in legacy code is 14.0%)
- **Raw Features in Dataset (22 features + 1 target)**:
  - Identifiers: `borrower_id`
  - Demographics: `age`, `occupation_type`, `city_tier`
  - Proxy financials & transactions: `monthly_income_estimate`, `electricity_bill_ontime_rate`, `electricity_bill_avg_delay_days`, `recharge_frequency_per_month`, `avg_recharge_amount`, `recharge_amount_volatility`, `days_since_last_recharge_lapse`, `monthly_upi_transaction_count`, `monthly_upi_inflow_avg`, `monthly_upi_outflow_avg`, `upi_inflow_volatility_coefficient`, `p2p_vs_merchant_txn_ratio`
  - Gig-worker specific (imputed for non-gig): `avg_weekly_gig_hours`, `gig_platform_rating`, `active_weeks_last_6_months`, `earnings_coefficient_of_variation`
  - Ecosystem tenure: `phone_number_tenure_months`, `app_account_age_months`
  - Target: `defaulted` (binary 0/1)

---

## 4. Current Persisted Model Artifact Metadata

- **Artifact Path**: `models/credit_model.pkl` (23,277 bytes)
- **Bundle Dictionary Keys**: `['model', 'model_name', 'pipeline', 'feature_names', 'test_metrics', 'all_metrics']`
- **Champion Model Name**: `Logistic Regression`
- **Champion Architecture**: `sklearn.pipeline.Pipeline(StandardScaler, LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42))`
- **Preprocessing Pipeline**: `FeaturePipeline(SimpleImputer(strategy='median'), OneHotEncoder(drop=None, sparse_output=False))`
- **Transformed Feature Count**: 30 numeric & one-hot columns

### 4.1 Actual Reproducible Evaluation Metrics on Test Set (1,200 held-out samples: 1,032 goods, 168 bads)
| Evaluation Metric | Actual Persisted Logistic Regression | Actual Persisted XGBoost Challenger |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.6240** | 0.6075 |
| **KS-Statistic (%)** | **21.08%** | 17.36% |
| **Precision (Default)** | **0.1958** | 0.2000 |
| **Recall (Default)** | **0.5595** | 0.3393 |
| **F1-Score** | **0.2901** | 0.2517 |
| **Confusion Matrix** | `[[646, 386], [74, 94]]` | `[[804, 228], [111, 57]]` |

---

## 5. Current Fairness & Demographic Disparity Diagnostics

Evaluated across the 8,000 synthetic records with `src/fairness_diagnostics.py`:
- **Overall Mean Credit Score**: 669.09
- **Overall Approval Proxy Rate**: 69.08% (scores >= 650)
- **Adverse Impact Ratios (AIR) and Disparities**:
  - `occupation_type:daily_wage_labor`: Approval Rate = 12.79%, **AIR = 0.1382** (Severe disparate impact vs gig_rideshare 92.57%)
  - `occupation_type:gig_delivery`: Approval Rate = 68.80%, **AIR = 0.7432** (Below 0.80 four-fifths rule threshold)
  - `occupation_type:informal_retail`: Approval Rate = 75.33%, AIR = 0.8137
  - `occupation_type:small_trader`: Approval Rate = 80.35%, AIR = 0.8680
  - `occupation_type:freelance_digital`: Approval Rate = 79.39%, AIR = 0.8575
  - `city_tier:tier_3`: Approval Rate = 54.64%, **AIR = 0.7068** (Below 0.80 benchmark vs tier_1 77.30%)
  - `city_tier:tier_2`: Approval Rate = 70.08%, AIR = 0.9066

---

## 6. Comprehensive Documentation vs. Code Discrepancy Table

| Item / Claim | Documented Claim in README / Docs | Actual Code / Artifact Reality | Severity | Risk & Impact Analysis |
| :--- | :--- | :--- | :--- | :--- |
| **Champion Model ROC-AUC** | `AUC: 0.7552` (README line 64, MODEL_CARD line 50) | `AUC: 0.6240` (saved in `models/credit_model.pkl`) | **CRITICAL** | Severe credibility gap. Documentation quotes numbers from a stale or untracked run. |
| **Champion Model KS-Statistic** | `KS: 38.82%` (README line 64, MODEL_CARD line 51) | `KS: 21.08%` (saved in `models/credit_model.pkl`) | **CRITICAL** | KS 21% is considered marginal/weak for commercial credit risk models. |
| **Challenger XGBoost Metrics** | `AUC: 0.7410`, `KS: 36.45%` | `AUC: 0.6075`, `KS: 17.36%` | **HIGH** | Challenger is substantially weaker than documented. |
| **Precision / Recall / F1** | Precision 0.6840, Recall 0.7120, F1 0.6977 | Precision 0.1958, Recall 0.5595, F1 0.2901 | **CRITICAL** | Model was evaluated under imbalanced default rate (14%), whereas docs appear to cite balanced pseudo-metrics. |
| **Champion Selection Rule** | "Selected based on regulatory interpretability, monotonicity, and exact SHAP" (MODEL_CARD §4) | `if xgb_metrics["auc"] >= lr_metrics["auc"]:` in `src/train_model.py` | **HIGH** | Purely AUC-driven code branch. No gates for calibration, stability, fairness, or complexity. |
| **Temporal Horizon & Leakage** | Claims observation and prediction windows (DATA_CARD, README) | Target `defaulted` in `data/generate_synthetic_data.py` is calculated directly from the same instantaneous summary statistics used as features. | **CRITICAL** | No separate observation ($T_{-12..-1}$) and prediction ($T_{0..+3}$) windows; instantaneous feature-target leakage. |
| **Validation Methodology** | Claims rigorous train/val/test evaluation | Standard random stratified `train_test_split`. No Out-Of-Time (OOT) evaluation split exists. | **HIGH** | Random splits overestimate performance and fail to test temporal degradation. |
| **Calibration Architecture** | Bayesian odds multiplier adjustment with $\pi = 0.14$ | Hard-coded heuristic formula in `src/scoring_utils.py`. No fitted calibration model (Platt/isotonic) or calibration curve/Brier tracking. | **MEDIUM** | Assumes post-hoc Bayesian shift is calibrated without empirical calibration verification. |
| **Economic Decisioning** | Claims underwriting decision support | Hard-coded tiers (Low, Moderate, High, Very High) with no configurable PD, LGD, EAD, or Expected Loss calculation. | **MEDIUM** | Lacks economic loss decisioning engine. |
| **Score Naming & Transformation** | "CIBIL-style 300–900 score" (README line 29, 90) | Explicitly prohibited by Phase 1 Prompt from referencing CIBIL or bureau equivalence. Must be "CreditBridge Risk Score". | **MEDIUM** | Regulatory and positioning compliance requirement. |

---

## 7. Inventory of Codebase Components

### 7.1 Source Modules (`src/`)
- `explain.py`: Local SHAP attribution calculator and regulatory narrative generator. Uses linear SHAP for Logistic Regression and TreeExplainer for XGBoost.
- `fairness_diagnostics.py`: Evaluates subgroup approval rates, AIR, and default rates across age, occupation, and city tier.
- `feature_engineering.py`: `FeaturePipeline` implementing `SimpleImputer(strategy='median')`, `StandardScaler`, and `OneHotEncoder`.
- `feature_provenance.py`: Provenance taxonomy enums: `DERIVED`, `SELF_REPORTED`, `UNAVAILABLE_IMPUTED`.
- `privacy_security.py`: File upload security firewall (magic bytes, path traversal, row limits) and ephemeral `AuditManifest`.
- `real_data_contracts.py`: Dataclasses defining manual inputs, transaction models, and assessment results.
- `real_data_features.py`: Feature derivation rules from raw statement transactions.
- `real_data_parser.py`: CSV transaction parser supporting multiple header dialects and ledger styles.
- `real_data_quality.py`: Statement sufficiency checks (90 days / 15 transactions) and distribution shift flags.
- `real_data_scoring.py`: End-to-end evaluation orchestrator binding parser, classifier, feature mapper, and model inference.
- `scoring_utils.py`: Log-odds to 300–900 score transformation and Bayesian prior calibration math.
- `train_model.py`: Training script for Logistic Regression and XGBoost.
- `transaction_classifier.py`: Rule-based transaction classification (reversals, utilities, gig inflows, P2P).

### 7.2 Tests (`tests/`)
- 8 test files with 111 tests. Tests verify contracts, adversarial inputs, data quality gates, and legacy baseline invariance.

### 7.3 Data & Models
- `data/generate_synthetic_data.py`: Generator producing `data/synthetic_borrowers.csv` (8,000 samples).
- `data/reference_distributions.json`: 5th and 95th percentiles of features used for outlier warnings.
- `models/credit_model.pkl`: Serialized joblib bundle.

### 7.4 Dashboard & Documentation
- `dashboard/app.py`: Streamlit interactive underwriting app.
- `docs/`: 8 markdown files documenting architecture, model card, data card, security, and interview defenses.

---

## 8. Preservation Commitment

This document is preserved permanently. All subsequent modifications in Phase 1 rebuild must strictly trace back to reproducible experiment runs and verifiable artifacts.
