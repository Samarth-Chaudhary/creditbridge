# CreditBridge V2: Alternative Credit Underwriting Engine for Thin-File Borrowers

[![CI Test Suite](https://img.shields.io/badge/CI-111%20Tests%20Passing-brightgreen?logo=github-actions)](.github/workflows/tests.yml)
[![Type Checking](https://img.shields.io/badge/Pyright-0%20Errors-brightgreen?logo=python)](pyrightconfig.json)
[![Python Version](https://img.shields.io/badge/Python-3.11-blue?logo=python)](requirements.txt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Interactive%20Dashboard-FF4B4B?logo=streamlit)](dashboard/app.py)

> **Portfolio & Research Prototype Notice**  
> CreditBridge V2 is an illustrative alternative credit assessment engine. All predictive models in this repository were trained strictly on **synthetic simulated borrower distributions**. The Real Data Mode demonstrates feature engineering, data quality gates, Bayesian calibration, and SHAP explainability on voluntarily supplied statement exports. It is **not** a validated commercial credit decision engine, does not provide bureau-equivalent scores, and is explicitly prohibited from automated lending decisioning.

---

## 🚀 Live Interactive Underwriting Dashboard

The complete recruiter-facing underwriting dashboard is located in [`dashboard/app.py`](dashboard/app.py) and ready for one-click deployment:

- **Streamlit Community Cloud Deployment**:
  1. Fork or push this repository to GitHub.
  2. Visit [share.streamlit.io](https://share.streamlit.io), connect your repo, set the branch to `main`, and main file path to `dashboard/app.py`.
  3. Deploy instantly (runs completely within free-tier compute with 0 external database requirements).
- **Run Locally in 1 Command**:
  ```bash
  streamlit run dashboard/app.py
  ```
- **Dashboard Capabilities Demonstrated**:
  - **Upload & Firewall Gate**: Ingestion of raw CSV bank/UPI statements with client-side 10 MB and PII sanitization.
  - **Data Quality & Distribution Diagnostics**: Automatic 90-day / 15-txn sufficiency validation and outlier detection against population baselines.
  - **Score Transformation**: 300–900 CIBIL-style score rendering with Bayesian calibration on a 14% thin-file prior.
  - **Local SHAP Feature Attribution**: Dynamic waterfall plot isolating top positive and negative cashflow drivers alongside regulatory explanations.

---

## 1. Executive Summary & Business Problem

In emerging credit markets such as India, over **150 million gig-economy workers, informal retail merchants, daily wage earners, and young digital freelancers** lack formal bureau credit histories (e.g., CIBIL, Experian, Equifax, CRIF High Mark). Traditional banking credit scoring models rely heavily on bureau trade-lines, past collateral, and formal salary slips. As a result, creditworthy thin-file individuals are systematically excluded or forced into predatory unorganized lending.

CreditBridge bridges this credit gap by evaluating **alternative behavioural cashflow markers**:
- **Digital Cashflow Stability**: UPI inflow consistency, transaction velocity, and earnings volatility.
- **Telecom Commitment**: Mobile recharge regularity, average ticket sizes, and lapse intervals.
- **Merchant Profiling**: P2P transfers versus merchant spend ratios and business inflows.
- **Income Estimation**: Recurring digital earnings derived strictly after filtering loans, refunds, and self-transfers.

---

## 2. Dual-Flow System Architecture

CreditBridge V2 cleanly separates the **Offline Synthetic Model Development Flow** from the **Real Data Ingestion & Evaluation Subsystem**:

```
========================================================================================
1. SYNTHETIC MODEL DEVELOPMENT FLOW (OFFLINE TRAINING)
========================================================================================
[data/generate_synthetic_data.py]
      ↓
8,000 Synthetic Borrowers (data/synthetic_borrowers.csv)
      ↓
Feature Pipeline (src/feature_engineering.py)
   - Median Imputation (SimpleImputer)
   - Feature Scaling (StandardScaler)
   - One-Hot Encodings (OneHotEncoder)
      ↓
Model Selection: Logistic Regression vs. XGBoost (src/train_model.py)
   - Champion: L2 Regularized Logistic Regression (AUC: 0.755, KS: 38.8%)
      ↓
Persisted Model Bundle (models/credit_model.pkl)

========================================================================================
2. REAL DATA MODE EVALUATION FLOW (EPHEMERAL RUNTIME)
========================================================================================
User Uploads CSV Bank / UPI Statement
      ↓
Defensive Validation Gate (src/privacy_security.py)
   - 10 MB Size Cap, 50,000 Row Limit, Magic Byte Checks, Path Traversal Blocks
      ↓
Canonical Transaction Parser (src/real_data_parser.py)
   - Header Alias Mapping, Sign Disambiguation, Duplicate Flags (Preserved for Audit)
      ↓
Transaction Classifier (src/transaction_classifier.py)
   - Refund, Reversal, Self-Transfer, Utility, Gig & Merchant Detection
      ↓
Feature Engineering & Provenance Mapper (src/real_data_features.py)
   - Derived (9 features), Self-Reported (3 features), Unavailable (9 features)
      ↓
Data Quality, Sufficiency & Distribution Gate (src/real_data_quality.py)
   - 90-Day / 15-Txn Blocker Gate, Distribution Shift Percentiles Check
      ↓
Existing Model Inference & Prior Calibration (src/real_data_scoring.py)
   - Bayesian Calibration (Odds adjusted for 14% thin-file baseline default prior)
   - 300–900 CIBIL-Style CreditBridge Model Score (Offset=490, Factor=95)
   - Risk Tiers: Low, Moderate, High — Manual Review, Very High Risk
      ↓
Local SHAP Explainability & Factor Provenance (src/explain.py)
   - Non-causal plain-English regulatory narrative with explicit imputation warnings
      ↓
Ephemeral Session Audit Manifest (src/privacy_security.py)
   - Session telemetry emitted without retaining raw statements or PII
```

---

## 3. Real Data Mode Specification

### 3.1 Supported Ingestion Formats
- Formats: CSV or plain-text tabular exports (`.csv`, `.txt`).
- Encodings: UTF-8, UTF-8-SIG, Latin-1, CP1252.
- Layouts: Single signed amount, split Debit/Credit ledgers, UPI transaction reports.
- Constraints: Maximum 10 MB file size; maximum 50,000 transaction rows.

### 3.2 Feature Provenance Taxonomy
CreditBridge enforces explicit, machine-readable provenance on every feature passed to the model:
1. **DERIVED (9 Features)**: Computed deterministically from statement transactions:
   - `monthly_income_estimate`, `monthly_upi_transaction_count`, `monthly_upi_inflow_avg`, `monthly_upi_outflow_avg`, `upi_inflow_volatility_coefficient`, `p2p_vs_merchant_txn_ratio`, `recharge_frequency_per_month`, `avg_recharge_amount`, `recharge_amount_volatility`.
2. **SELF_REPORTED (3 Features)**: Provided explicitly by applicant via verified input contracts:
   - `age` (18–70), `occupation_type` (6 categories), `city_tier` (3 tiers).
3. **UNAVAILABLE & IMPUTED (9 Features)**: Features not observable from statements are assigned `NaN` and imputed with population medians:
   - `electricity_bill_ontime_rate`, `electricity_bill_avg_delay_days`, `days_since_last_recharge_lapse`, `avg_weekly_gig_hours`, `gig_platform_rating`, `active_weeks_last_6_months`, `earnings_coefficient_of_variation`, `phone_number_tenure_months`, `app_account_age_months`.

### 3.3 Data Quality & Sufficiency Gate
- **History Sufficiency Threshold**: Requires at least **90 calendar days** and **15 usable transactions**. Statements covering $<30$ days or $<15$ transactions are hard-blocked from scoring with status `INSUFFICIENT`.
- **Distribution Shift Flags**: Features outside the 5th–95th synthetic percentiles trigger warning flags (`outside_observed_range`, `near_boundary`) to alert underwriters to distribution shift.

---

## 4. Model Scoring & Explainability Formulation

### 4.1 Probability Semantics & Bayesian Prior Calibration
The champion classifier was trained on balanced synthetic data (50% default weighting). To reflect empirical credit reality, raw model probabilities are recalibrated via Bayes' rule under the empirical Indian thin-file default rate ($\pi = 0.14$):
$$\text{odds}_{\text{raw}} = \frac{p_{\text{raw}}}{1 - p_{\text{raw}}}$$
$$\text{odds}_{\text{calibrated}} = \text{odds}_{\text{raw}} \times \frac{0.14}{1 - 0.14} = \text{odds}_{\text{raw}} \times \frac{0.14}{0.86}$$
$$p_{\text{calibrated}} = \frac{\text{odds}_{\text{calibrated}}}{1 + \text{odds}_{\text{calibrated}}}$$

### 4.2 CIBIL-Style Credit Score Transformation
Default probabilities are mapped to a familiar 300–900 scale using logarithmic odds (PDO scaling):
$$\text{Score} = 490.0 + \left(95.0 \times \ln\left(\frac{1 - p_{\text{calibrated}}}{p_{\text{calibrated}}}\right)\right)$$
Clamped strictly to $[300, 900]$.

| Score Range | Underwriting Risk Tier | Operational Meaning |
| :--- | :--- | :--- |
| **750 – 900** | **Low Risk** | Prime cashflow stability; instant digital approval proxy |
| **650 – 749** | **Moderate Risk** | Standard stability; standard pricing and terms |
| **550 – 649** | **High Risk — Manual Review** | High volatility; requires secondary guarantor or scrutiny |
| **300 – 549** | **Very High Risk** | Severe cashflow deficit or volatility; policy decline |

### 4.3 SHAP Explainability & Non-Causality Boundary
- Feature attributions are computed via SHAP (SHapley Additive exPlanations).
- Attributions reflect **associative mathematical contribution** to model score points: $\Delta = -\text{round}(\text{SHAP} \times 95.0)$.
- SHAP values **do not prove causality** or borrower repayment intent.
- Explanations explicitly flag when influential factors relied on median imputation.

---

## 5. Privacy, Security & Zero-Retention Architecture

- **Ephemeral Execution**: Uploaded statements exist strictly in temporary in-memory streams (`io.BytesIO`). They are never persisted to disk, databases, caches, or logs.
- **Air-Gapped Network Isolation**: 0 outbound network calls, 0 cloud OCR, 0 external LLM APIs, and 0 third-party account aggregators.
- **Defensive Upload Controls**: Magic-byte inspection detects executable headers (`MZ`, `\x7fELF`, `PK\x03\x04`), blocking path traversal (`../`) and null-byte attacks.
- **Error Sanitization**: Error messages strip PAN numbers, bank accounts, 10-digit phone numbers, emails, and local paths.
- **Zero Credential Collection**: Strict contract enforcement rejects any input containing passwords, PINs, OTPs, CVVs, or secret tokens.
- **Session Auditability**: Non-sensitive `AuditManifest` captures metadata (versions, row counts, coverage, score, timestamp) with `raw_statement_retained: False` and `contains_pii: False`.

---

## 6. Repository Structure

```
creditbridge/
├── .github/workflows/tests.yml     # Automated CI (pytest 111 tests + pyright)
├── .gitignore                      # Security & secret exclusions
├── LICENSE                         # MIT open-source license
├── pytest.ini                      # Pytest runner & path configuration
├── README.md                       # Master technical documentation
├── pyrightconfig.json              # Static typing configuration
├── requirements.txt                # Minimum-version pinned dependencies
├── assets/                         # Dashboard UI assets (read-only)
├── dashboard/                      # Recruiter-facing Streamlit app (read-only)
│   └── app.py                      # 49,554 bytes (firewall protected)
├── data/                           # Synthetic data & reference percentiles
│   ├── generate_synthetic_data.py  # Synthetic generator
│   ├── reference_distributions.json# 5th/95th percentile baselines
│   └── synthetic_borrowers.csv     # 8,000-sample synthetic dataset
├── docs/                           # Governance & audit deliverables
│   ├── MODEL_CARD.md               # Regulatory model specification
│   ├── DATA_CARD.md                # Dataset generation & limitations
│   ├── FEATURE_LINEAGE.md          # 22-feature lineage matrix
│   ├── REAL_DATA_MODE.md           # End-to-end backend user journey
│   ├── PRIVACY_SECURITY.md         # Threat model & security controls
│   ├── BIG4_INTERVIEW_DEFENSE.md   # 20 technical interview Q&A
│   ├── HOSTILE_REVIEW.md           # 18 senior audit attack questions
│   └── PRODUCTION_ROADMAP.md       # 12-pillar production path
├── models/                         # Champion model bundle (frozen)
│   └── credit_model.pkl            # 23,277 bytes (Logistic Regression pipeline)
├── src/                            # Core backend engine
│   ├── explain.py                  # Local SHAP explainability
│   ├── fairness_diagnostics.py     # Subgroup fairness & AIR evaluation
│   ├── feature_engineering.py      # Fitted FeaturePipeline (frozen)
│   ├── privacy_security.py         # Upload validation & audit manifests
│   ├── real_data_contracts.py      # Schemas, dataclasses & error taxonomy
│   ├── real_data_features.py       # Alternative feature engineering
│   ├── real_data_parser.py         # Canonical CSV transaction parser
│   ├── real_data_quality.py        # Sufficiency & distribution gates
│   ├── real_data_scoring.py        # End-to-end scoring orchestrator
│   ├── scoring_utils.py            # Score transformation & calibration
│   ├── train_model.py              # Model training script
│   └── transaction_classifier.py   # Heuristic transaction classification
└── tests/                          # 111-test regression & QA suite
    ├── fixtures/                   # 7 sanitized synthetic CSV fixtures
    ├── test_adversarial_and_e2e_fixtures.py # 28 adversarial & fixture tests
    ├── test_csv_parser.py          # 15 parser unit tests
    ├── test_dashboard_smoke.py     # 4 read-only dashboard smoke tests
    ├── test_data_quality.py        # 12 data quality gate tests
    ├── test_model_integration.py   # 17 model contract tests
    ├── test_privacy_security.py    # 12 security & privacy tests
    ├── test_real_data_features.py  # 15 feature mapping tests
    └── test_synthetic_baseline_regression.py # 8 baseline regression tests
```

---

## 7. Installation & Reproducibility Guide

### 7.1 Environment Setup
```powershell
# Clone or navigate to repository
cd creditbridge

# Create and activate Python 3.11 virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install minimum-version pinned dependencies
pip install -r requirements.txt
```

### 7.2 Executing Test Suites & Quality Verification
```powershell
# Run entire 111-test verification suite (runs in ~10 seconds)
pytest -v

# Run static type checking with Pyright across the entire repository (0 errors)
npx pyright

# Launch interactive Streamlit underwriting dashboard locally
streamlit run dashboard/app.py
```

### 7.3 Evaluating a Statement Programmatically
```python
from src.real_data_contracts import ManualInputContract
from src.real_data_scoring import assess_statement_end_to_end
from src.privacy_security import create_audit_manifest

# 1. Provide statement content (path or bytes)
csv_statement = "tests/fixtures/representative_transactions.csv"

# 2. Specify verified self-reported attributes
inputs = ManualInputContract(age=29, occupation_type="gig_delivery", city_tier="tier_1")

# 3. Execute end-to-end evaluation
result = assess_statement_end_to_end(csv_statement, manual_inputs=inputs)

# 4. Inspect results
print(f"Credit Score: {result.credit_score} ({result.risk_tier})")
print(f"Calibrated P(default): {result.calibrated_model_probability:.4f}")
print(f"Evidence Coverage: {result.evidence_coverage_ratio * 100:.1f}%")
print(f"Explanation: {result.explanation['plain_english_explanation']}")

# 5. Generate privacy-safe audit manifest
manifest = create_audit_manifest(assessment_result=result)
print(manifest.to_dict())
```

---

## 8. Explicit Governance & Responsible AI Disclosures

1. **Synthetic Training Limitation**: The champion predictive model was trained entirely on synthetic data. Performance metrics ($\text{AUC} = 0.755$, $\text{KS} = 38.8\%$) reflect goodness-of-fit to the mathematical simulation, not real-world default prediction.
2. **Non-Equivalence to Bureau Scores**: CreditBridge Model Score (300–900) is an illustrative non-linear scaling of model default odds. It is not an RBI-approved or credit-bureau score (CIBIL, Experian, Equifax, CRIF High Mark).
3. **Prohibited Applications**: Explicitly prohibited from being used for autonomous loan approvals/rejections, loan pricing, collections prioritization, or employment screening.
4. **Non-Causal SHAP Attributions**: SHAP values quantify feature importance within the trained model space; they do not establish causal mechanisms of financial default or personal integrity.

---

## 9. About the Author & Engineering Motivation

CreditBridge was built to explore how modern digital lenders and fintechs in high-growth emerging economies (specifically India's UPI and informal gig-work ecosystem) can underwrite thin-file borrowers without relying on predatory unorganized credit or opaque black-box scoring.

In commercial machine learning, model weights are often the easiest component to replace; what determines institutional viability is the engineering rigor surrounding them:
- **Contract-First Architecture**: Strong dataclasses and strict boundary assertions prevent data leakage and undefined states.
- **Auditable Provenance**: Every feature is explicitly tagged as `DERIVED`, `SELF_REPORTED`, or `UNAVAILABLE_IMPUTED` so underwriters never confuse a median fallback with observed empirical truth.
- **Privacy by Construction**: Ephemeral in-memory parsing, zero PII retention, and air-gapped isolation eliminate consumer data exposure risks.

This repository prioritizes software design, defensive validation, and regulatory compliance as foundational capabilities rather than post-hoc additions.

---

## 10. Honest Architectural Trade-Offs (Defense Guide)

| Perceived Gap | Engineering Rationale & Institutional Mitigation |
| :--- | :--- |
| **Synthetic-Only Training (AUC 0.755)** | Real bank statements and default histories carry severe privacy and legal constraints. Training on synthetic distributions allowed full architectural development while guaranteeing zero personal data exposure. The value proposition is the pipeline architecture: the training workflow (`src/train_model.py`) and inference orchestrator (`src/real_data_scoring.py`) are strictly decoupled and ready to accept real repayment ledgers with 0 breaking interface changes. |
| **43% Median Imputation in Real Data Mode** | Bank and UPI statements alone cannot supply utility payment delays, electricity tenure, or gig ratings. Instead of silently fabricating these values or artificially inflating confidence, CreditBridge explicitly segregates them into the `UNAVAILABLE` lineage tier, substitutes population medians, flags the compression of score variance in `AuditManifest.evidence_coverage_ratio`, and surfaces imputation notices in human-readable SHAP narratives. The [Production Roadmap](docs/PRODUCTION_ROADMAP.md) details how RBI Account Aggregator (AA) rails replace these medians in Phase 2. |
| **Linear Champion vs. XGBoost** | Regularized Logistic Regression was chosen over XGBoost because Indian NBFC credit risk committees mandate monotonic risk penalties and direct, legally explainable log-odds feature attribution. Monotonicity prevents non-linear gaming of credit scores where small arbitrary changes create wild score swings. |

