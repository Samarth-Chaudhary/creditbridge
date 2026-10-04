# Model Card: CreditBridge Alternative Credit Underwriting Engine

**Model Name**: CreditBridge Alternative Credit Risk Classifier  
**Model Version**: `v1.0.0-lr-baseline` (Single Champion Invariant)  
**Model Type**: L2-Regularized Logistic Regression Pipeline (`StandardScaler` + `SimpleImputer` + `OneHotEncoder` + `LogisticRegression`)  
**Persistent Artifact**: [`models/credit_model.pkl`](models/credit_model.pkl) (23,277 bytes)  
**Cryptographic Hash (SHA-256)**: `bfbabaa4e80accc24495fe9f1ef4c0d1d2bb95b8bf9b835b0744ef6c0729615c`  
**License**: MIT  
**Date of Release**: October 2026  

---

## 1. Executive Model Details & Architecture

CreditBridge is an alternative-credit underwriting scoring engine engineered specifically for thin-file consumers, gig-economy workers, and informal micro-entrepreneurs in emerging credit markets who lack traditional credit bureau trade-lines (e.g., CIBIL, Experian, CRIF High Mark).

### 1.1 Architecture & Pipeline Specification
The production champion pipeline consists of a deterministic, leak-free scikit-learn pipeline:
1. **Numerical Imputation & Scaling**:
   - `SimpleImputer(strategy='median')`: Handles missingness in optional utility metrics without information leakage.
   - `StandardScaler()`: Normalizes cashflow velocity, income stability, and recharge lapse variables.
2. **Categorical Encoding**:
   - `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`: Encodes occupation categories (`gig_delivery`, `gig_rideshare`, `daily_wage_labor`, `small_trader`, `informal_retail`, `freelance_digital`) and city tiers (`tier_1`, `tier_2`, `tier_3`).
3. **Core Classifier**:
   - `LogisticRegression(penalty='l2', C=1.0, class_weight='balanced', solver='lbfgs', random_state=42)`
   - Chosen deliberately over gradient-boosted decision trees for **strict monotonicity**, regulatory interpretability, auditable log-odds contributions, and robust resistance to synthetic noise memorization.
4. **Calibration Layer**:
   - Platt Sigmoid Scaling transforming log-odds into well-calibrated posterior default probabilities ($P(\text{Default}|X)$) aligned with the 14.0% population base default rate.
5. **Presentation Layer**:
   - Scaled 300–900 CreditBridge Risk Score: $\text{Score} = 600 + 50 \times \log_2\left(\frac{1 - P(Y=1)}{P(Y=1)} \times \frac{0.14}{0.86}\right)$, clamped strictly to $[300, 900]$.

---

## 2. Intended Use vs. Prohibited Out-of-Scope Use

### 2.1 Intended Applications
- **Research & Engineering Demonstration**: End-to-end prototyping of alternative-credit underwriting pipelines, data-quality state machines, and algorithmic fairness audits.
- **Underwriter Decision-Support**: Assist human loan officers at non-banking financial companies (NBFCs) and microfinance institutions (MFIs) by highlighting verifiable cashflow patterns.
- **Adverse Action Explanations**: Generate model-attribution waterfall notices compliant with RBI Fair Practices Code.

### 2.2 Explicitly Prohibited Out-of-Scope Applications
- **Autonomous Credit Decisioning**: The model must NEVER autonomously approve, reject, or disburse loans without human underwriter concurrence.
- **Sole Source of Truth**: The model must NEVER replace formal KYC, AML/CFT screenings, or legally mandated credit bureau checks when available.
- **Automated Risk-Based Pricing**: Prohibited from automatically setting predatory interest rates or adjusting loan APRs based on thin-file score proxies.
- **Employment or Tenant Screening**: Prohibited from general consumer evaluation outside explicit micro-credit underwriting contexts.
- **CIBIL / Credit Bureau Replacement**: Prohibited from claiming equivalence to or replacement of statutory Indian credit information companies (CICs).

---

## 3. Training & Validation Data Summary

- **Source Data**: 8,000 synthetic borrower profiles generated via `data/generate_synthetic_data.py` and `src/temporal_data_generator.py`.
- **Target Definition (`defaulted`)**: Binary indicator ($1 = \text{default}, 0 = \text{non-default}$) modeling 90+ Days Past Due (DPD) within a 90-day forward prediction horizon.
- **Population Default Prevalence**: 14.05% unweighted base rate.
- **Synthetic Ground Truth Disclosure**: All training records are simulated. While generated with realistic Indian fintech cashflow distributions (UPI inflows, mobile recharges, electricity bill histories, gig platform ratings), they contain no real consumer data.

---

## 4. Empirical Performance & Metrics

All metrics are verifiable against versioned artifacts tracked in `RESULTS.md` and `BASELINE_REPORT.md`:

| Split / Benchmark | Sample Size | ROC-AUC | PR-AUC | KS-Statistic | Brier Score | Gini Index |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Preserved Baseline (`models/credit_model.pkl`)** | 8,000 | **0.6240** | **0.2116** | **21.08%** | **0.1654** | **0.2480** |
| **In-Time Train Split** | 5,000 | 0.6312 | 0.2195 | 22.14% | 0.1621 | 0.2624 |
| **In-Time Validation Split** | 1,500 | 0.6258 | 0.2084 | 21.42% | 0.1662 | 0.2516 |
| **Out-of-Time (OOT) Split** | 1,500 | 0.6204 | 0.2031 | 20.89% | 0.1685 | 0.2408 |

> [!WARNING]
> **Historical Metric Reconciliation**: Older documentation iterations cited ~0.755 AUC and 38.8% KS from legacy untracked runs. Those numbers are obsolete. The verified, reproducible baseline is **0.6240 ROC-AUC** and **21.08% KS**.

---

## 5. Demographic Fairness & Disparate Impact Auditing

Audited across 8,000 borrowers across age groups, occupation categories, and city tiers under the EEOC Four-Fifths (80%) Rule and True Positive Rate (TPR) parity standards.

- **Adverse Impact Ratio (AIR)**: All evaluated demographic subgroups achieve $\text{AIR} \ge 0.80$ (overall minimum subgroup AIR is 0.97 for `daily_wage_labor`).
- **Demographic Parity**: Selection rates vary between 97.5% (`daily_wage_labor`) and 100.0% (`small_trader`, `gig_rideshare`), reflecting baseline income differences.
- **Kleinberg Impossibility Theorem**: Per Kleinberg et al. (2016), when base default rates vary across groups, no underwriting system can simultaneously achieve Demographic Parity, Equal Opportunity, and Predictive Calibration. CreditBridge prioritizes well-calibrated risk estimation and explicit disclosure over deceptive optimization.

---

## 6. Calibration & Goodness-of-Fit

- **Uncalibrated Output**: Raw logistic probabilities overestimate defaults due to `class_weight='balanced'`.
- **Platt Sigmoid Scaling**: Reduces Brier score from 0.1654 to **0.1170** and Expected Calibration Error (ECE) to **0.0185**, ensuring posterior default probabilities accurately represent portfolio loss probabilities.

---

## 7. Security Assumptions & Threat Mitigations

1. **Pickle Serialization Security**: Python `pickle` files are not security boundaries. `models/credit_model.pkl` is verified against an immutable SHA-256 hash prior to deserialization (`bfbabaa4e80accc24495fe9f1ef4c0d1d2bb95b8bf9b835b0744ef6c0729615c`).
2. **Defensive Input Ingestion**: Maximum upload file size is capped at 10 MB and 50,000 transactions; binary magic bytes are inspected to prevent executable upload attacks.
3. **CSV Formula Injection Neutralization**: Cells starting with `=`, `+`, `-`, `@`, `\t`, or `\r` are neutralized with single quotes.
4. **PII Sanitization**: Account numbers, customer names, and phone numbers are stripped before feature extraction.

---

## 8. Continuous Drift & Production Monitoring

The champion model is continuously monitored in production via Population Stability Index (PSI):
- $\text{PSI} < 0.10$: **HEALTHY / STABLE** (Continue normal scoring)
- $0.10 \le \text{PSI} < 0.25$: **MONITOR / WARNING** (Flag for data science review)
- $\text{PSI} \ge 0.25$: **BLOCK / CRITICAL** (Halt automated scoring and trigger incident rollback)
