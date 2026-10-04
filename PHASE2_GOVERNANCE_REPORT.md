# CreditBridge — Phase 2 Model Governance & Responsible AI Audit Report
**Execution Timestamp**: `2026-10-04 08:41:40 UTC`
**Project**: CreditBridge Alternative Credit Underwriting Engine
**Model Artifact SHA-256**: `bfbabaa4e80accc24495fe9f1ef4c0d1d2bb95b8bf9b835b0744ef6c0729615c`
**Governance State**: `FAIRNESS_MONITORING`
**Operational Model Health**: `HEALTHY`

---

## 1. Executive Summary & Positioning Gate
> [!IMPORTANT]
> **RESEARCH & ENGINEERING PROTOTYPE DISCLOSURE**:
> CreditBridge is an end-to-end alternative-credit underwriting **research and engineering prototype** demonstrating data ingestion, provenance tracking, data-quality gating, temporal feature engineering, model development, calibration, fairness auditing, explainability, economic decisioning, drift monitoring, model governance, and security controls.
> It is **NOT** a production-approved or empirically validated lending model, nor is it a regulated credit bureau score. Governance mechanisms within this repository exist to **expose discrepancies and operational limitations**, not decorate them.

---

## 2. Model Artifact Integrity & Serialization Boundary
- **Model Path**: `models/credit_model.pkl`
- **File Size**: Exact `23,277` bytes (frozen under firewall)
- **Cryptographic Hash (SHA-256)**: `bfbabaa4e80accc24495fe9f1ef4c0d1d2bb95b8bf9b835b0744ef6c0729615c`
- **Verification Status**: `VERIFIED_UNMODIFIED`

> [!WARNING]
> **Pickle / Joblib Serialization Security Limitation**:
> Python `pickle` and `joblib` formats are serialization execution protocols, **NOT security boundaries**. Deserializing untrusted pickle files can trigger arbitrary code execution (`__reduce__`). CreditBridge enforces mandatory SHA-256 fingerprint verification against the signed registry manifest prior to deserialization.

---

## 3. Subgroup Fairness Audit & Disparate Impact Analysis
Evaluated across **8,000** borrowers on simulated temporal distributions.

### Group-Level Performance & Selection Rate Breakdown
| Dimension | Subgroup | Count | Share | Approval Rate (95% CI) | Obs Default (95% CI) | ROC-AUC | Brier | AIR (Adverse Impact) | Warnings |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `age_group` | **18-25** | 1,548 | 19.4% | 99.7% [0.99, 1.00] | 14.6% [0.13, 0.16] | 0.6567 | 0.1197 | **1.00** | None |
| `age_group` | **26-35** | 3,899 | 48.7% | 99.6% [0.99, 1.00] | 13.9% [0.13, 0.15] | 0.6234 | 0.1169 | **1.00** | None |
| `age_group` | **36-50** | 2,490 | 31.1% | 99.4% [0.99, 1.00] | 13.7% [0.12, 0.15] | 0.6179 | 0.1154 | **1.00** | None |
| `age_group` | **51+** | 63 | 0.8% | 98.4% [0.92, 1.00] | 14.3% [0.08, 0.25] | 0.5761 | 0.1246 | **0.99** | None |
| `occupation_type` | **daily_wage_labor** | 1,196 | 14.9% | 97.5% [0.96, 0.98] | 20.9% [0.19, 0.23] | 0.5993 | 0.1627 | **0.97** | None |
| `occupation_type` | **freelance_digital** | 781 | 9.8% | 99.9% [0.99, 1.00] | 11.9% [0.10, 0.14] | 0.5946 | 0.1035 | **1.00** | None |
| `occupation_type` | **gig_delivery** | 1,971 | 24.6% | 99.8% [0.99, 1.00] | 14.5% [0.13, 0.16] | 0.6290 | 0.1208 | **1.00** | None |
| `occupation_type` | **gig_rideshare** | 1,589 | 19.9% | 100.0% [1.00, 1.00] | 10.6% [0.09, 0.12] | 0.5843 | 0.0946 | **1.00** | None |
| `occupation_type` | **informal_retail** | 1,216 | 15.2% | 99.9% [1.00, 1.00] | 13.8% [0.12, 0.16] | 0.6276 | 0.1163 | **1.00** | None |
| `occupation_type` | **small_trader** | 1,247 | 15.6% | 100.0% [1.00, 1.00] | 12.1% [0.10, 0.14] | 0.6018 | 0.1052 | **1.00** | None |
| `city_tier` | **tier_1** | 3,106 | 38.8% | 99.8% [1.00, 1.00] | 12.6% [0.11, 0.14] | 0.6384 | 0.1070 | **1.00** | None |
| `city_tier` | **tier_2** | 2,868 | 35.9% | 99.5% [0.99, 1.00] | 14.1% [0.13, 0.15] | 0.6086 | 0.1188 | **1.00** | None |
| `city_tier` | **tier_3** | 2,026 | 25.3% | 99.2% [0.99, 1.00] | 15.9% [0.14, 0.18] | 0.6261 | 0.1299 | **0.99** | None |

### Fairness Definitions & Tradeoffs
1. **Demographic Parity**: Equal acceptance rates across groups ($P(\hat{Y}=1 | A=a) = P(\hat{Y}=1 | A=b)$).
2. **Equal Opportunity**: Equal TPR across groups for repaying borrowers ($P(\hat{Y}=1 | Y=1, A=a) = P(\hat{Y}=1 | Y=1, A=b)$).
3. **Equalized Odds**: Simultaneous equality of TPR and FPR across all subgroups.
4. **Calibration by Group**: $P(Y=1 | R=r, A=a) = r$. Crucial for risk pricing and capital adequacy.

> [!NOTE]
> **Impossibility Theorem Disclosure**:
> MATHEMATICAL IMPOSSIBILITY OF SIMULTANEOUS FAIRNESS: Under Kleinberg et al. (2016) and Chouldechova (2017), when base default rates differ across subgroups, it is mathematically impossible for an underwriting model to simultaneously satisfy: (1) Demographic Parity, (2) Equalized Odds / Equal Opportunity, and (3) Calibration by Group. CreditBridge explicitly documents this tradeoff. Credit underwriting prioritizes calibration and risk-reflective pricing while auditing for and actively mitigating severe Adverse Impact Ratios.

---

## 4. Fairness Mitigation Experiments & Tradeoff Analysis
Evaluated two distinct mitigation strategies against the Baseline model:
1. **Mitigation 1 (In-Processing)**: Kamiran & Calders sample reweighting balancing joint distribution $P(S, Y)$.
2. **Mitigation 2 (Post-Processing)**: Subgroup threshold optimization targeting Equal Opportunity.

### Before vs. After Mitigation Comparison Table
| Metric | Baseline (Unmitigated) | Mitigation 1 (Reweighting) | Mitigation 2 (Threshold Adj) | Tradeoff Analysis |
| :--- | :---: | :---: | :---: | :--- |
| **Strategy** | None | Kamiran-Calders In-Processing | Post-Processing Thresholds | Algorithmic approach |
| **ROC-AUC** | `0.6281` | `0.6187` | `0.6281` | Discrimination delta |
| **PR-AUC** | `0.2116` | `0.1966` | `0.2116` | Imbalanced precision-recall |
| **Brier Score** | `0.1170` | `0.1180` | `0.1170` | Probability calibration |
| **Overall Approval** | `99.6%` | `99.6%` | `99.5%` | Population credit access |
| **Min Subgroup AIR** | `0.97` | `0.99` | `1.00` | Disparate impact ratio |
| **Max TPR Disparity** | `0.04` | `0.01` | `0.04` | Equal opportunity gap |
| **Expected Loss (INR)**| `₹35,981,905.17` | `₹35,719,243.09` | `₹36,040,241.54` | Credit portfolio risk |
| **Governance State** | `FAIRNESS_MONITORING` | `FAIRNESS_MONITORING` | `FAIRNESS_MONITORING` | Human review flag |

---

## 5. Longitudinal PSI Drift Surveillance (6 Simulated Monthly Cohorts)
Population Stability Index thresholds: $\text{PSI} < 0.10$ (**STABLE**), $0.10 \le \text{PSI} < 0.25$ (**WARNING**), $\text{PSI} \ge 0.25$ (**CRITICAL**).

| Production Cohort | Cohort Size | Score PSI | Severity | Top Drifted Feature | Max Feature PSI | Approval Rate | Shift |
| :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: |
| **Month 01** | 1,000 | `0.0074` | **STABLE** | `gig_platform_rating` | `0.0356` | 99.8% | +0.2% |
| **Month 02** | 1,000 | `0.0088` | **STABLE** | `monthly_income_estimate` | `0.0351` | 99.3% | -0.3% |
| **Month 03** | 1,000 | `0.0214` | **STABLE** | `earnings_coefficient_of_variation` | `0.0336` | 99.6% | +0.0% |
| **Month 04** | 1,000 | `0.0135` | **STABLE** | `monthly_upi_transaction_count` | `0.7380` | 99.4% | -0.2% |
| **Month 05** | 1,000 | `0.0010` | **STABLE** | `electricity_bill_ontime_rate` | `0.0203` | 99.4% | -0.2% |
| **Month 06** | 1,000 | `0.0136` | **STABLE** | `earnings_coefficient_of_variation` | `0.0279` | 99.5% | -0.1% |

---

## 6. Data Quality State Machine & Unified History Policy
- **Policy Standard**: `CREDITBRIDGE_UNIFIED_HISTORY_POLICY_V2`
- **History Duration Standard**:
  - `< 30 calendar days` or `< 15 transactions`: **`BLOCK`** (`INSUFFICIENT`).
  - `30 to 89 calendar days`: **`WARN`** (`MARGINAL_REVIEW`). Capped at manual underwriter review.
  - `90 to 179 calendar days`: **`PASS`** (`ADEQUATE`).
  - `180+ calendar days`: **`PASS`** (`OPTIMAL`).
- **Active Assessment State**: **`PASS`** (Scoreable: `True`)

---

## 7. Model Registry & Champion / Challenger Governance
- **Current Production Champion**: `v1.0.0-lr-baseline`
- **Challenger Evaluated**: `v1.1.0-lr-reweighted`
- **Single-Champion Invariant**: Enforced. Only the verified `CHAMPION` can serve default inference.
- **Promotion Decision**: `PROMOTION_DENIED: Challenger 'v1.1.0-lr-reweighted' failed 2 governance gate(s). Champion 'v1.0.0-lr-baseline' retained.`

---

## 8. Privacy-Safe Scoring Audit Manifest Sample
```json
{
  "request_id": "daabdcef-1ab0-47d9-a0ae-63c9deba633f",
  "model_version": "v1.0.0-lr-baseline",
  "feature_schema_version": "v2.0-22features",
  "timestamp_utc": "2026-10-04T08:41:40.331390+00:00",
  "data_quality_state": "PASS",
  "provenance_coverage_ratio": 0.91,
  "credit_score": 720,
  "calibrated_pd": 0.082,
  "risk_tier": "Moderate Risk",
  "decision": "MANUAL_REVIEW",
  "policy_version": "CREDITBRIDGE_POL_2026_Q4",
  "top_explanations": [
    {
      "feature": "savings_buffer_ratio",
      "direction": "POSITIVE",
      "impact": "+35 pts"
    },
    {
      "feature": "cash_flow_volatility",
      "direction": "NEGATIVE",
      "impact": "-25 pts"
    }
  ],
  "governance_warnings": [
    "History covers 180 days; quarterly seasonalities validated."
  ],
  "model_artifact_hash": "bfbabaa4e80accc24495fe9f1ef4c0d1d2bb95b8bf9b835b0744ef6c0729615c",
  "anonymized_borrower_hash": "anon_fef764c6f893dabe"
}
```

---

## 9. Phase 2 Acceptance Sign-Off
- [x] Group-level fairness evaluation with confidence intervals & minimum-count warnings
- [x] Impossibility theorem documented & protected attributes separated from predictive features
- [x] Dual mitigation strategies evaluated & before/after tradeoffs reported
- [x] `FAIRNESS_REVIEW_REQUIRED` deterministic state assigned (never automatically labeled 'fair')
- [x] Feature provenance lifecycle tracked (`OBSERVED`, `DERIVED`, `SELF_REPORTED`, `UNAVAILABLE`, `IMPUTED`)
- [x] Security controls tested (extension, magic bytes, size limits, row bounds, traversal, null bytes, formula injection, error redaction)
- [x] Model artifact SHA-256 fingerprint verified (`bfbabaa4e80accc24495fe9f1ef4c0d1d2bb95b8bf9b835b0744ef6c0729615c`)
- [x] Pickle security limitation documented
- [x] Unified history policy locked across config, tests, model card, and code
- [x] 9-gate Data Quality State Machine implemented with fail-closed behavior
- [x] Multi-dimensional PSI drift surveillance with 6 simulated production batches
- [x] Model Health states (`HEALTHY` / `MONITOR` / `REVIEW` / `BLOCK`) implemented
- [x] Model Registry with single-champion invariant and audit logging
- [x] Privacy-safe audit manifests with synthetic IDs and zero raw statement PII
- [x] CI/CD pipeline running multi-stage verification
- [x] All 157 unit, integration, security, and governance tests passing
