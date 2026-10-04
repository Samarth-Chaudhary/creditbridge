# Model Card: CreditBridge Alternative Credit Scoring Model (v1.0-synthetic)

## 1. Model Details
- **Model Name**: CreditBridge Alternative Credit Scoring Classifier
- **Model Version**: `v1_synthetic_lr_pipeline`
- **Model Type**: L2-Regularized Logistic Regression Pipeline (`StandardScaler` + `SimpleImputer` + `OneHotEncoder` + `LogisticRegression`)
- **Persisted Artifact**: [`models/credit_model.pkl`](../models/credit_model.pkl) (23,277 bytes)
- **Primary Objective**: Estimate the probability of credit default for thin-file and gig-economy borrowers lacking traditional credit bureau histories, based on alternative behavioural and cashflow features.

---

## 2. Intended Use & Prohibited Applications
- **Intended Use**: Research prototype, methodology demonstration, and underwriter decision-support tool for evaluating thin-file cashflow indicators.
- **Prohibited Use**:
  - Autonomous loan approval or decline decisions.
  - Automated risk-based interest rate pricing.
  - Collections queue prioritization.
  - Employment screening or general consumer profiling.
  - Legal or regulatory credit scoring without independent validation on real historical repayments.

---

## 3. Training Data & Assumptions
- **Training Population**: 8,000 synthetic borrower profiles generated via `data/generate_synthetic_data.py`.
- **Target Variable (`defaulted`)**: Binary indicator ($1 = \text{default}, 0 = \text{non-default}$) representing a 90+ Days Past Due (DPD) proxy.
- **Target Generation Logic**: Latent score derived via non-linear combination of cashflow stability, recharge lapses, utility payment delays, and gig engagement metrics, calibrated with logistic noise to achieve a 14% unweighted population default rate.
- **Critical Disclosure**: The model was trained **entirely on simulated data**. No real borrower performance, bank statements, or bureau repayment records were used in model fitting.

---

## 4. Model Architecture & Selection Rationale
Two candidate architectures were evaluated on a 70/15/15 stratified train/validation/test split:
1. **Regularized Logistic Regression** (Cost-sensitive, balanced class weights)
2. **Gradient Boosted Decision Trees** (`XGBClassifier`)

**Selection Rationale**:
While XGBoost achieved marginal non-linear fit on synthetic patterns, **Regularized Logistic Regression was selected as champion** for the following institutional reasons:
- **Strict Monotonicity & Regulatory Interpretability**: Linear log-odds coefficients guarantee monotonic score degradation as default risk indicators worsen, preventing non-monotonic anomalies common in unconstrained trees.
- **Exact SHAP Attribution**: Linear SHAP values map directly and deterministically to score points ($\Delta = -\text{round}(\text{SHAP} \times 95.0)$).
- **Overfitting Resistance**: Less susceptible to memorizing synthetic generation artifacts.

---

## 5. Performance Metrics & Multi-Split Evaluation

All reported metrics are deterministically produced and verifiable against versioned experiment runs tracked in `experiments/` and `BASELINE_REPORT.md`.

### 5.1 Preserved Baseline Reference (Legacy Random Split, N=8,000)
Evaluated on the preserved baseline model artifact (`models/credit_model.pkl`, 23,277 bytes):

| Metric | Logistic Regression (Baseline) | XGBoost (Baseline) | Documentation Reality |
| :--- | :--- | :--- | :--- |
| **AUC-ROC** | **0.6240** | 0.6075 | Discrepancy vs legacy claimed 0.755 documented in `BASELINE_REPORT.md` |
| **KS-Statistic** | **21.08%** | 17.36% | Baseline separation on balanced random train/test split |
| **Brier Score** | **0.1654** | 0.1702 | Mean squared probability error |
| **Precision** | **0.1958** | 0.2000 | Default class precision under 14% population prevalence |
| **Recall** | **0.5595** | 0.3393 | Sensitivity to default events |
| **F1-Score** | **0.2901** | 0.2517 | Harmonic mean |

### 5.2 Phase 1 Temporal Out-of-Time (OOT) Benchmark (Experiment: `exp_phase1_20261004T061033Z`)
Evaluated strictly on held-out Out-of-Time cohort (1,500 samples, forward 90-day prediction horizon, zero temporal leakage):

| Metric | Logistic Regression (Champion) | XGBoost (Challenger) | 95% Bootstrap CI (Champion) |
| :--- | :--- | :--- | :--- |
| **AUC-ROC** | **0.9669** | 0.9587 | **[0.9564, 0.9766]** |
| **KS-Statistic** | **82.30%** | 81.18% | **[80.08%, 87.09%]** |
| **PR-AUC** | **0.7823** | 0.7712 | **[0.7131, 0.8438]** |
| **Gini Coefficient** | **0.9338** | 0.9174 | Derived ($2 \times \text{AUC} - 1$) |
| **Brier Score (Raw)** | **0.0595** | 0.0612 | Uncalibrated probability error |
| **Brier Score (Calibrated)** | **0.0437** | 0.0461 | Improved post-Platt calibration |
| **Precision** | **0.5862** | 0.5714 | Threshold at 0.50 |
| **Recall** | **0.8608** | 0.8354 | NPA capture rate |
| **F1-Score** | **0.6974** | 0.6780 | Balanced discrimination |
| **Decile 10 Capture** | **100.0%** | 98.7% | Top decile lift = 6.77x |

---

## 6. Probability Calibration & Score Formulation

### 6.1 Calibration Engine
Raw model probabilities are calibrated using Platt Sigmoid scaling (`src/evaluation_engine.py`), reducing Brier score from 0.0595 to 0.0437 on OOT data while preserving rank-ordering.

### 6.2 CreditBridge Risk Score Presentation Layer (300–900 Scale)
Calibrated probabilities are transformed into a 300–900 presentation score using standard logarithmic odds (PDO scaling):
$$\text{Score} = 490.0 + \left(95.0 \times \ln\left(\frac{1 - p_{\text{calibrated}}}{p_{\text{calibrated}}}\right)\right)$$
Clamped strictly to $[300, 900]$.

> [!NOTE]
> **Strict Bureau Non-Equivalence**: The 300–900 score is solely a presentation layer called **CreditBridge Risk Score**. It is **NOT** a CIBIL, Experian, Equifax, or CRIF High Mark score, does not imply equivalence to any credit bureau score, and is not approved by the Reserve Bank of India (RBI) for autonomous credit decisioning.

### 6.3 Risk Tier Partitions
- **750 – 900**: **Low Risk** (Instant digital approval proxy)
- **650 – 749**: **Moderate Risk** (Standard terms and pricing)
- **550 – 649**: **High Risk — Manual Review** (Requires secondary review / guarantor)
- **300 – 549**: **Very High Risk** (Policy decline)

---

## 7. Explainability & SHAP Governance
- **Mechanism**: Local SHAP attributions decomposed into point impacts: $\Delta = -\text{round}(\text{SHAP} \times 95.0)$.
- **Non-Causality Boundary**: SHAP values indicate associative model sensitivity, **not causality**. The narrative explicitly uses non-causal phrasing ("primarily constrained by", "positively supported by") and avoids deterministic claims ("proved", "will default because").
- **Provenance Transparency**: Explanations explicitly flag when influential factors relied on population median imputation.

---

## 8. Limitations & Known Weaknesses
1. **Synthetic-to-Real Generalization**: High test performance on synthetic data does **not** prove real-world accuracy. Real borrower behaviour exhibits unmodelled macro shocks, seasonality, and strategic repayment defaults.
2. **Missing Feature Imputation**: In Real Data Mode, 9 of the 22 model features (such as utility punctuality and platform ratings) cannot be observed from bank statements and are imputed using population medians.
3. **Non-Equivalence to Bureau Scores**: The CreditBridge score is an illustrative mathematical score and must never be cited as an authorized credit bureau score.
4. **Subgroup Disparities**: Synthetic fairness analysis shows Adverse Impact Ratios (AIR) $< 0.80$ for daily wage laborers relative to freelance digital workers. These reflect synthetic modeling assumptions and must be validated against empirical default rates before production deployment.
