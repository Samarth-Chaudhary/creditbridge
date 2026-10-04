# CreditBridge: Alternative Credit Underwriting Engine for Thin-File Borrowers

[![CI Test Suite](https://img.shields.io/badge/CI-157%20Tests%20Passing-brightgreen?logo=github-actions)](.github/workflows/ci.yml)
[![Type Checking](https://img.shields.io/badge/Pyright-0%20Errors-brightgreen?logo=python)](pyrightconfig.json)
[![Code Style](https://img.shields.io/badge/Ruff-Passed-brightgreen?logo=python)](ruff.toml)
[![Python Version](https://img.shields.io/badge/Python-3.11-blue?logo=python)](requirements.lock)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit Dashboard](https://img.shields.io/badge/Streamlit-9--Tab%20Interactive%20Console-FF4B4B?logo=streamlit)](dashboard/app.py)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live%20Underwriting%20Portal-10B981?logo=githubpages)](https://samarth-chaudhary.github.io/creditbridge/)
[![Hosted App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://samarth-chaudhary-creditbridge-dashboardapp-vzrsrp.streamlit.app)

> [!IMPORTANT]
> **RESEARCH & ENGINEERING PROTOTYPE DISCLOSURE**:
> CreditBridge is an end-to-end alternative-credit underwriting **research and engineering prototype** designed to demonstrate data ingestion, provenance tracking, data-quality gating, feature engineering, model development, temporal validation, calibration, fairness auditing/mitigation, explainability, decisioning, drift monitoring, model governance, security controls, reproducibility, testing, and auditability.
>
> It is **NOT** a production-approved lending system, does not replace statutory credit bureaus (e.g., CIBIL, Experian), and is strictly prohibited from autonomous loan approval. All reported performance numbers are generated from reproducible artifacts traceable to a specific dataset version, seed, model version, and evaluation run:
> - **Reproducible Baseline (Champion)**: **ROC-AUC: 0.6240** and **KS: 21.08%** on `data/synthetic_borrowers.csv` (`models/credit_model.pkl`).
> - **Historical Runs**: Legacy documentation citing ~0.755 AUC and 38.8% KS represents earlier untracked runs, now labeled **HISTORICAL** and superseded by the verifiable frozen artifacts documented in [`RESULTS.md`](RESULTS.md).

---

## 🧭 Project Navigation & Core Documentation

| Document | Primary Focus & Regulatory Scope |
| :--- | :--- |
| 📊 [**`RESULTS.md`**](RESULTS.md) | Comprehensive empirical results: Multi-split metrics, Platt calibration, 10-decile lift table, subgroup fairness, and drift surveillance. |
| 🪪 [**`MODEL_CARD.md`**](MODEL_CARD.md) | Formal model documentation: Intended use, out-of-scope applications, feature schema, calibration, and security boundaries. |
| 📋 [**`DATA_CARD.md`**](DATA_CARD.md) | Data lineage: Two-stage synthetic simulation, latent drivers, observable features, missingness mechanisms, and temporal windows. |
| 🛡️ [**`THREAT_MODEL.md`**](THREAT_MODEL.md) | AppSec & model security: STRIDE threat matrix, CWE-502 pickle defense, CSV injection neutralization, and PII masking. |
| 🏛️ [**`GOVERNANCE.md`**](GOVERNANCE.md) | Institutional MLOps: Single Champion Invariant, 5 promotion gates, health state machine, and incident rollback runbook. |
| 🔁 [**`REPRODUCE.md`**](REPRODUCE.md) | Clean-environment reproduction runbook: Exact commands, environment requirements, and verification tolerances. |
| ⚠️ [**`LIMITATIONS.md`**](LIMITATIONS.md) | Deep structural limitations: Synthetic data boundaries, lack of macro shocks, Goodhart's Law gaming, and Kleinberg impossibility. |
| 🎯 [**`FINAL_REVIEW.md`**](FINAL_REVIEW.md) | 3-persona adversarial review (EY Recruiter, Senior ML Engineer, Model-Risk Auditor) with 30 rejection points and 15-dimension score table. |

---

## 1. Executive Summary & Why Alternative Credit

In emerging credit economies like India, over **150 million gig-economy workers, informal retail merchants, daily wage earners, and young digital freelancers** lack formal credit bureau files (thin-file borrowers). Traditional banking underwriting relies on bureau trade-lines, past collateral, and formal salary slips. Consequently, creditworthy individuals are systematically excluded from formal credit or forced into unorganized lending.

CreditBridge evaluates **verifiable alternative cashflow behaviors**:
1. **Digital Cashflow Stability**: UPI inflow velocity, earnings variance, and active transaction days.
2. **Telecom Commitment**: Mobile recharge regularity, ticket size, and maximum disconnection lapses.
3. **Utility Discipline**: Electricity (DISCOM) bill on-time payment track records.
4. **Platform Gig Telemetry**: Aggregator customer ratings, active weekly hours, and tenure.

---

## 2. End-to-End System Architecture

CreditBridge unifies offline temporal research and real-time bank statement ingestion into a single, cohesive architecture:

```
[Raw Input Data: CSV Bank / UPI Statement or Synthetic Portfolio]
                             │
                             ▼
┌────────────────────────────────────────────────────────┐
│ 1. SECURITY & SCHEMA INGESTION GATE                    │
│    - 10 MB File Size Cap & 50,000 Transaction Row Limit│
│    - Binary Magic Byte Inspection (CWE-434 Defense)    │
│    - Path Traversal Sanitization (CWE-22)              │
│    - CSV Formula Injection Neutralization (CWE-1236)   │
│    - Regex-Based PII Masking (Aadhaar, PAN, Phone)     │
└────────────────────────────┬───────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────┐
│ 2. 9-GATE DATA QUALITY STATE MACHINE                   │
│    - Locked Institutional History Policy:              │
│      • < 30 Days History or < 15 Txns: STRICT BLOCK    │
│      • 30 – 89 Days History: WARN & MANUAL REVIEW      │
│      • >= 90 Days History: CLEAN PASS                  │
│    - Completeness, Reconciliation & Anomaly Checks     │
└────────────────────────────┬───────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────┐
│ 3. CANONICAL CLASSIFICATION & FEATURE PROVENANCE       │
│    - Deterministic Transaction Categorization          │
│      (Refunds, Reversals, Self-Transfers, Gig Inflows) │
│    - Feature Provenance Attribution Taxonomy:          │
│      [OBSERVED, DERIVED, SELF_REPORTED, IMPUTED]       │
└────────────────────────────┬───────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────┐
│ 4. STATISTICAL & DISTRIBUTION CHECKS                   │
│    - Out-of-Fold Median Imputation (Leak-Free)         │
│    - Reference Distribution Alignment vs Synthetic Base│
└────────────────────────────┬───────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────┐
│ 5. CHAMPION INFERENCE & CALIBRATION (SINGLE CHAMPION)  │
│    - L2-Regularized Logistic Regression Champion       │
│    - Platt Sigmoid Calibration (Brier: 0.1654 -> 0.117)│
│    - CreditBridge Risk Score (300 – 900 Presentation)  │
└────────────────────────────┬───────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────┐
│ 6. ECONOMIC DECISIONING & POLICY ALLOCATION            │
│    - Expected Loss: EL = PD * LGD * EAD                │
│    - Policy Tiers: AUTO-APPROVE / MANUAL-REVIEW / DECLINE
└────────────────────────────┬───────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────┐
│ 7. RESPONSIBLE AI, FAIRNESS & EXPLAINABILITY           │
│    - Subgroup Fairness Audit (Age, Occupation, Tier)   │
│    - Adverse Impact Ratio (AIR >= 0.80) & Wilson 95% CI│
│    - Local SHAP Attribution (Explicit Model Attribution)│
│    - Kleinberg Impossibility Theorem Disclosure        │
└────────────────────────────┬───────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────┐
│ 8. AUDIT MANIFEST & PRODUCTION DRIFT SURVEILLANCE      │
│    - Cryptographic Request Audit Manifest Persistence  │
│    - Longitudinal PSI Tracking (6 Production Cohorts)  │
│    - Health State Machine (HEALTHY / MONITOR / BLOCK)  │
└────────────────────────────────────────────────────────┘
```

---

## 3. Interactive Risk & Governance Dashboard (`dashboard/app.py`)

CreditBridge features a single, professional **9-tab dark-fintech underwriting and model governance console**:

```bash
streamlit run dashboard/app.py
```

### Dashboard Tabs Overview
1. **📊 Executive Overview**: Dynamic portfolio KPIs (8,000 borrowers), calibrated default prevalence, approval distribution, portfolio Expected Loss (₹35.98M), and champion governance state.
2. **👤 Borrower Assessment**: Single applicant underwriting console with 300–900 score gauge, calibrated PD, risk tier, local model attribution waterfall, and RBI-compliant adverse action notices.
3. **📈 Model Performance & Deciles**: Multi-split metrics (In-Time Train, Validation, OOT), 10-bin monotonic risk decile table, ROC curve, and KS distribution separation.
4. **⚖️ Fairness & Bias Mitigation**: Disparate impact audits across age, occupation, and city tier; 80% Four-Fifths rule evaluation; Wilson 95% CIs; and before/after mitigation comparison.
5. **🛡️ Data Quality & Feature Provenance**: 9-gate state machine visualization, locked institutional history policy enforcement, and 4-tier provenance breakdown.
6. **📡 Drift & Production Surveillance**: 6-month simulated production cohorts, feature PSI, score PSI, and automated model health state tracking (`HEALTHY` vs `MONITOR`).
7. **🎛️ Policy Simulator**: Interactive underwriter sliders for approval thresholds, review bands, EAD, and LGD with real-time recalculation of portfolio loss and demographic tradeoffs.
8. **🏛️ Model Registry & Audit Trail**: Single Champion Invariant enforcement (`v1.0.0-lr-baseline`), cryptographic SHA-256 integrity verification, and production audit manifests.
9. **📜 Methodology, Ethics & Limitations**: Full prototype disclosure, synthetic data limitations, and RBI Account Aggregator roadmap.

---

## 4. Key Empirical Benchmark Metrics

All metrics reflect verifiable runs documented in [`RESULTS.md`](RESULTS.md):

| Split / Benchmark | Sample Size ($N$) | ROC-AUC | KS-Statistic | Brier Score | Gini Index |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline Champion (`models/credit_model.pkl`)** | 8,000 | **0.6240** | **21.08%** | **0.1654** | **0.2480** |
| **Temporal In-Time Train Split** | 5,000 | 0.6312 | 22.14% | 0.1621 | 0.2624 |
| **Temporal In-Time Validation Split** | 1,500 | 0.6258 | 21.42% | 0.1662 | 0.2516 |
| **Temporal Out-of-Time (OOT) Split** | 1,500 | 0.6204 | 20.89% | 0.1685 | 0.2408 |
| **Calibrated Champion (Platt Sigmoid)** | 8,000 | **0.6281** | **21.50%** | **0.1170** | **0.2562** |

---

## 5. Verification & Quickstart

### 5.1 Quick Setup
```bash
# Clone the repository
git clone https://github.com/Samarth-Chaudhary/creditbridge.git
cd creditbridge

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1

# Install exact pinned dependencies
pip install -r requirements.lock
```

### 5.2 Execute the Test Suite (157 Tests)
```bash
pytest
```
*Expected: 157 passed in ~7s, 0 failures, 0 warnings.*

### 5.3 Static Analysis & Linting
```bash
# Type check (0 errors)
npx pyright

# Style check (0 errors)
ruff check .
```

### 5.4 Execute Governance Runner
```bash
python src/phase2_governance_runner.py
```

### 5.5 Dashboard Hosting & Live Access
The official 9-tab **Streamlit Master Underwriting & Governance Dashboard** (`dashboard/app.py`) is hosted on GitHub:
1. **GitHub Pages Streamlit Portal**:
   - **URL**: [https://samarth-chaudhary.github.io/creditbridge/](https://samarth-chaudhary.github.io/creditbridge/)
   - **Behavior**: Seamlessly displays the full Streamlit dashboard directly on GitHub Pages via automated CI/CD ([`.github/workflows/pages.yml`](.github/workflows/pages.yml)).
2. **Streamlit Community Cloud Direct Launch**:
   - **Live App URL**: [https://samarth-chaudhary-creditbridge-dashboardapp-vzrsrp.streamlit.app](https://samarth-chaudhary-creditbridge-dashboardapp-vzrsrp.streamlit.app)
   - **Architecture**: Runs the complete 9-tab operational suite (Executive Overview, Borrower Assessment with SHAP explainability, Decile Calibration, Fair Lending Disparate Impact Auditing, Data Quality Provenance, Population Drift PSI Surveillance, Economic Policy Simulator, Model Registry Integrity, and Limitations).
   - **Theme**: Pinned to dark fintech tokens in [`.streamlit/config.toml`](.streamlit/config.toml).

---

## 6. License & Authorship

- **License**: MIT License ([`LICENSE`](LICENSE))
- **Author**: Samarth Chaudhary & CreditBridge Engineering Team  
- **Positioning**: Alternative Credit Underwriting Research & Engineering Prototype
