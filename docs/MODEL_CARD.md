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

## 5. Performance Metrics (Test Set Evaluation)

Evaluated on 1,200 held-out test samples:

| Metric | Champion (Logistic Regression) | Candidate (XGBoost) | Regulatory Relevance |
| :--- | :--- | :--- | :--- |
| **AUC-ROC** | **0.7552** | 0.7410 | Discrimination power across all thresholds |
| **KS-Statistic** | **38.82%** | 36.45% | Maximum separation between good and bad distributions |
| **Precision** (Class 1) | **0.6840** | 0.6510 | Positive predictive value under balanced weighting |
| **Recall** (Class 1) | **0.7120** | 0.6980 | Sensitivity to default events |
| **F1-Score** | **0.6977** | 0.6738 | Harmonic balance between precision and recall |

---

## 6. Probability Calibration & Score Formulation

### 6.1 Bayesian Prior Calibration
Because the training model used balanced class weighting (effective prior 50%), raw model probabilities ($p_{\text{raw}}$) are adjusted to reflect the empirical thin-file baseline default odds ($\pi = 0.14$):
$$\text{odds}_{\text{calibrated}} = \frac{p_{\text{raw}}}{1 - p_{\text{raw}}} \times \frac{0.14}{0.86}$$
$$p_{\text{calibrated}} = \frac{\text{odds}_{\text{calibrated}}}{1 + \text{odds}_{\text{calibrated}}}$$

### 6.2 CIBIL-Style Credit Score Scaling
Calibrated probabilities are transformed into a 300–900 score using standard Basel/PDO log-odds scaling:
$$\text{Score} = 490.0 + \left(95.0 \times \ln\left(\frac{1 - p_{\text{calibrated}}}{p_{\text{calibrated}}}\right)\right)$$
Clamped strictly to $[300, 900]$.

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
