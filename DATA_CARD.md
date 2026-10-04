# Data Card: CreditBridge Synthetic Alternative Credit Dataset

**Dataset Name**: CreditBridge Synthetic Thin-File & Alternative Credit Corpus  
**Dataset Versions**: 
1. Baseline Unsplit Corpus: `data/synthetic_borrowers.csv` ($N = 8,000$, 1,446,180 bytes, SHA-256: `44ce69ea99ee954546559d8c0678eb04a29a007629b35b62b1b3699b0c79ca2e`)
2. Temporal Partition Corpus: `data/temporal_synthetic_borrowers.csv` ($N = 8,000$, 1,732,940 bytes)  
**Primary Creators**: CreditBridge Model-Risk & Data Engineering Team  
**Release Date**: October 2026  
**License**: MIT  

---

## 1. Dataset Overview & Context

In emerging credit markets like India, more than 150 million young digital freelancers, gig delivery workers, daily wage earners, and informal retail merchants lack formal credit bureau files. Public lending datasets either strictly cover formal banking customers or cannot be released due to bank secrecy regulations and Indian Digital Personal Data Protection Act (DPDP Act 2023) constraints.

To provide a fully reproducible, leak-free, and ethically auditable environment for engineering an alternative-credit underwriting system, CreditBridge synthesizes multi-modal behavioral cashflow distributions modeled after empirical Indian fintech patterns.

---

## 2. Two-Stage Synthetic Data Generation Methodology

The synthetic dataset generation pipeline (`data/generate_synthetic_data.py` and `src/temporal_data_generator.py`) enforces strict causal mechanics via a two-stage structural simulation:

```
[Latent Borrower Capabilities]
  ├─ Financial Maturity (~Beta(2, 5))
  ├─ Income Volatility (~Gamma(shape, scale))
  ├─ Digital Adoption (~Uniform(0, 1))
  └─ Macroeconomic Shock Exposure (~Bernoulli(p))
                 │
                 ▼
[Stage 1: 12-Month Observable Telemetry]
  (Generated strictly over historical window [T-12, T-1])
  ├─ UPI Transaction Volumes & Velocity
  ├─ Prepaid Mobile Recharge Regularity & Intervals
  ├─ Electricity / Utility Bill Payment Discipline
  ├─ Gig Platform Performance Metrics (Ratings, Hours)
  └─ Bank Balance Volatility & Inflow Seasonality
                 │
                 ▼
[Stage 2: Independent 90-Day Forward Outcome Window]
  (Observed strictly over [T, T+3 months])
  └─ Target Variable: 90+ DPD Default Event (Binary 0 / 1)
```

---

## 3. Latent Variables & Causal Drivers

Borrower cashflows are governed by four unobservable latent parameters that drive both behavioral indicators and default propensities:
1. **Financial Buffer / Maturity ($M_i$)**: Governs financial resilience, savings propensity, and expense discipline. Higher values reduce recharge lapses and lower default probability.
2. **Income Volatility ($\sigma_i$)**: Governs earnings variance across months. High volatility reflects gig platform algorithmic fluctuation, monsoon seasonality for daily labor, or informal inventory cycles.
3. **Digital Cashflow Engagement ($D_i$)**: Governs the proportion of commerce conducted digitally via UPI versus informal cash transactions.
4. **Adverse Shock Susceptibility ($\theta_i$)**: Models medical emergencies, vehicle breakdown, or macroeconomic disruption.

---

## 4. Feature Taxonomy & Observable Data Schema

The dataset contains 22 operational features spanning five behavioral domains:

| Domain | Feature Name | Data Type | Permissible Range | Null Allowed? | Description & Proxy Role |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Demographics** | `age` | Integer | 18 – 75 | No | Borrower age (protected attribute proxy) |
| | `occupation_type` | Categorical | 6 categories | No | `gig_delivery`, `gig_rideshare`, `daily_wage_labor`, `small_trader`, `informal_retail`, `freelance_digital` |
| | `city_tier` | Categorical | `tier_1`, `tier_2`, `tier_3` | No | Metro vs Urban vs Semi-Urban location |
| **Cashflow & Income**| `monthly_income_estimate`| Float | 5,000 – 150,000 | No | Verifiable recurring digital inflows (INR) |
| | `earnings_coefficient_of_variation`| Float | 0.05 – 2.50 | No | Standard deviation of monthly inflows / mean |
| | `income_stability_index`| Float | 0.00 – 1.00 | No | Composite ratio of recurring digital receipts |
| | `monthly_upi_transaction_count`| Integer | 0 – 300 | No | Frequency of digital UPI payments |
| | `avg_daily_inflow_frequency`| Float | 0.00 – 10.00 | No | Regularity of active inflow days per week |
| **Telecom Commitment**| `telecom_recharge_regularity`| Float | 0.00 – 1.00 | No | Timeliness of prepaid mobile recharge cycles |
| | `avg_recharge_amount`| Float | 50 – 2,500 | No | Ticket size of mobile data/voice pack (INR) |
| | `max_recharge_lapse_days`| Integer | 0 – 90 | No | Longest period of phone disconnection |
| **Utility Discipline**| `electricity_bill_ontime_rate`| Float | 0.00 – 1.00 | Yes (30% missing)| Historical DISCOM bill payment timeliness |
| | `avg_electricity_bill_amount`| Float | 200 – 15,000 | Yes (30% missing)| Average monthly electricity consumption (INR) |
| | `utility_disconnection_count`| Integer | 0 – 5 | Yes (30% missing)| Number of utility service disconnections |
| **Platform Gig Metrics**| `gig_platform_rating`| Float | 3.0 – 5.0 | Yes (40% missing)| Verified rating on delivery/rideshare platforms |
| | `platform_tenure_months`| Integer | 1 – 60 | Yes (40% missing)| Cumulative experience on digital aggregator |
| | `weekly_peak_hours_active`| Float | 0.0 – 60.0 | Yes (40% missing)| High-earning hours worked per week |
| **Target Variable**| `defaulted` | Binary | 0 or 1 | No | $1 = \text{Default}$ (90+ DPD in 90-day forward window) |

---

## 5. Missingness Mechanisms (MCAR vs. MAR)

Missing values in CreditBridge are intentionally engineered to mirror commercial operational realities:
- **Utility Features (`electricity_bill_ontime_rate`, etc.)**: 30% missingness. Modeled as **Missing at Random (MAR)** conditioned on `city_tier` and `occupation_type` (tenants in shared housing do not hold individual DISCOM meters).
- **Gig Platform Features (`gig_platform_rating`, etc.)**: 40% missingness. Modeled as **Structural Non-Applicability / Missing Not at Random (MNAR)**; only borrowers in `gig_delivery` and `gig_rideshare` occupations possess gig platform records.

Imputation is performed strictly via median replacement learned exclusively on training partitions to prevent data leakage.

---

## 6. Target Variable Construction & Temporal Windows

To eliminate look-ahead bias and temporal leakage, CreditBridge uses strict non-overlapping temporal windows:
1. **Observation Window ($[T - 12\text{m}, T - 1\text{d}]$)**: All behavioral features (transaction frequency, recharge habits, utility payments) are aggregated strictly over historical data.
2. **Observation Cutoff ($T$)**: The decision point at which the borrower applies for credit.
3. **Outcome Horizon ($[T, T + 90\text{d}]$)**: The forward 90-day window during which repayment is monitored.
4. **Default Event Formulation**:
   $$P(\text{Default}_i = 1) = \sigma\left(\beta_0 + \beta_1 \cdot \text{Vol}_i - \beta_2 \cdot M_i - \beta_3 \cdot \text{Reg}_i + \beta_4 \cdot \text{Lapse}_i + \epsilon_i\right)$$
   Where $\epsilon_i \sim \text{Logistic}(0, 1)$ adds unmodeled behavioral variation, calibrated to achieve an exact **14.05%** overall population default rate.

---

## 7. Known Data Limitations

1. **Synthetic Nature**: The dataset is completely synthesized. While statistically grounded, synthetic generators cannot capture macroeconomic shocks (e.g., demonetization, sudden platform commission cuts) or dynamic behavioral feedback loops.
2. **Single Statement Boundary**: Does not model multi-account cross-bank money movement where borrowers hide defaults in non-disclosed accounts. Commercial deployments require RBI Account Aggregator fetching.
3. **No Dynamic Time-Varying Covariates**: Features are represented as pre-aggregated summary statistics rather than raw transactional event sequences.
