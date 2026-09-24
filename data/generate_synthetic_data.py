"""
CreditBridge - Alternative Credit Scoring Engine
Stage 1: Synthetic Data Generation Module
Path: data/generate_synthetic_data.py

Generates 8,000 realistic synthetic borrower records for gig workers and
thin-file borrowers in India. Encodes realistic correlations between non-traditional
behavioral/transactional proxy signals and latent creditworthiness.
"""

import os
import uuid
import numpy as np
import pandas as pd
from faker import Faker

# -----------------------------------------------------------------------------
# REPRODUCIBILITY & SETUP
# -----------------------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
fake = Faker("en_IN")
Faker.seed(RANDOM_SEED)

NUM_BORROWERS = 8000

# -----------------------------------------------------------------------------
# DATA DICTIONARY DEFINITION
# -----------------------------------------------------------------------------
DATA_DICTIONARY = [
    {
        "column": "borrower_id",
        "role": "identifier",
        "data_type": "string (UUID)",
        "description": "Unique identifier for each borrower"
    },
    {
        "column": "age",
        "role": "demographic",
        "data_type": "integer",
        "description": "Borrower age (18-60, normally distributed around ~32)"
    },
    {
        "column": "occupation_type",
        "role": "raw signal",
        "data_type": "categorical",
        "description": "Borrower occupation category across gig and informal sectors"
    },
    {
        "column": "city_tier",
        "role": "raw signal",
        "data_type": "categorical",
        "description": "Indian urban classification (tier_1, tier_2, tier_3)"
    },
    {
        "column": "monthly_income_estimate",
        "role": "derived proxy",
        "data_type": "float",
        "description": "Underwriter-inferred monthly income (INR) based on proxy signals"
    },
    {
        "column": "electricity_bill_ontime_rate",
        "role": "raw signal",
        "data_type": "float (0-1)",
        "description": "Proportion of last 12 utility bills paid on or before due date"
    },
    {
        "column": "electricity_bill_avg_delay_days",
        "role": "raw signal",
        "data_type": "float",
        "description": "Average delay in days past due date for utility payments"
    },
    {
        "column": "recharge_frequency_per_month",
        "role": "raw signal",
        "data_type": "float",
        "description": "Average number of mobile telecom recharges per month"
    },
    {
        "column": "avg_recharge_amount",
        "role": "raw signal",
        "data_type": "float",
        "description": "Average mobile recharge ticket size in INR"
    },
    {
        "column": "recharge_amount_volatility",
        "role": "raw signal",
        "data_type": "float",
        "description": "Coefficient of variation (CV) of recharge amounts over 6 months"
    },
    {
        "column": "days_since_last_recharge_lapse",
        "role": "raw signal",
        "data_type": "float",
        "description": "Days since last telecom balance lapse (3+ days without balance/data)"
    },
    {
        "column": "monthly_upi_transaction_count",
        "role": "raw signal",
        "data_type": "integer",
        "description": "Monthly digital transaction frequency (UPI)"
    },
    {
        "column": "monthly_upi_inflow_avg",
        "role": "raw signal",
        "data_type": "float",
        "description": "Average monthly total credits/inflows via UPI (INR)"
    },
    {
        "column": "monthly_upi_outflow_avg",
        "role": "raw signal",
        "data_type": "float",
        "description": "Average monthly total debits/outflows via UPI (INR)"
    },
    {
        "column": "upi_inflow_volatility_coefficient",
        "role": "raw signal",
        "data_type": "float",
        "description": "Lumpiness coefficient of UPI income inflows (modeled via Gamma/Lognormal)"
    },
    {
        "column": "p2p_vs_merchant_txn_ratio",
        "role": "raw signal",
        "data_type": "float",
        "description": "Ratio of peer-to-peer transfers to merchant payments (informal credit proxy)"
    },
    {
        "column": "avg_weekly_gig_hours",
        "role": "raw signal (gig only)",
        "data_type": "float",
        "description": "Weekly platform active logged-in hours (delivery/rideshare only, NaN otherwise)"
    },
    {
        "column": "gig_platform_rating",
        "role": "raw signal (gig only)",
        "data_type": "float (3.0-5.0)",
        "description": "Customer/service feedback rating on gig platforms (skewed high)"
    },
    {
        "column": "active_weeks_last_6_months",
        "role": "raw signal (gig only)",
        "data_type": "integer (0-26)",
        "description": "Number of weeks with active order completion in the last 26 weeks"
    },
    {
        "column": "earnings_coefficient_of_variation",
        "role": "raw signal (gig only)",
        "data_type": "float",
        "description": "Weekly platform payout variation coefficient (CV)"
    },
    {
        "column": "phone_number_tenure_months",
        "role": "raw signal",
        "data_type": "integer",
        "description": "Duration in months the primary mobile number has been active (SIM stability)"
    },
    {
        "column": "app_account_age_months",
        "role": "raw signal",
        "data_type": "integer",
        "description": "Tenure in months of the borrower's fintech/ecosystem account"
    },
    {
        "column": "defaulted",
        "role": "target variable",
        "data_type": "binary (0/1)",
        "description": "Loan default outcome (1 = Defaulted, 0 = Non-default / fully repaid)"
    }
]


def print_data_dictionary():
    """Prints a structured data dictionary to the console."""
    print("=" * 100)
    print("CREDITBRIDGE - DATA DICTIONARY (STAGE 1 SYNTHETIC BORROWER DATASET)")
    print("=" * 100)
    fmt = "{:<34} | {:<20} | {:<16} | {:<24}"
    print(fmt.format("Column Name", "Role", "Data Type", "Description Summary"))
    print("-" * 100)
    for entry in DATA_DICTIONARY:
        desc = (entry['description'][:35] + '...') if len(entry['description']) > 38 else entry['description']
        print(fmt.format(entry["column"], entry["role"], entry["data_type"], desc))
    print("=" * 100)


def generate_synthetic_data(num_samples: int = NUM_BORROWERS) -> pd.DataFrame:
    """
    Generates realistic synthetic alternative credit data for gig & thin-file borrowers.
    
    Generative Design Architecture:
    -------------------------------
    1. Demographics & Stratification:
       - Age ~ N(32, 7.5) bounded within [18, 60].
       - Occupations partitioned across gig and informal micro-sectors.
       - City tiers (Tier 1: 40%, Tier 2: 35%, Tier 3: 25%) modulating cost of living.
    2. Income & Cashflow Generation:
       - Monthly income is non-uniform and tied to occupation baseline and city tier.
       - UPI inflows and outflows reflect real banking behavior with liquidity margins.
    3. Behavioral Proxy Modeling:
       - Utility payment consistency correlates with age and income, with injected 18% noise.
       - Volatility measures (recharge CV, UPI inflow CV, gig earnings CV) explicitly capture
         income lumpiness via heavy-tailed gamma/lognormal distributions.
       - P2P vs merchant ratio serves as an informal peer debt distress signal.
    4. Target Variable Calibration:
       - Latent credit risk is modeled as a weighted linear combination:
         * Income stability (40%)
         * Payment consistency (30%)
         * Earnings volatility (20%)
         * Digital footprint stability (10%)
       - Transformed via standard sigmoid with irreducible noise term epsilon ~ N(0, 0.65).
       - Intercept calibrated to yield ~12-15% default rate reflecting Indian thin-file segments.
    """
    # 1. Demographic & baseline fields
    borrower_ids = [str(uuid.uuid4()) for _ in range(num_samples)]

    # Age: Normal distribution centered ~32, bounded [18, 60]
    raw_age = np.random.normal(loc=32.0, scale=7.5, size=num_samples)
    age = np.clip(np.round(raw_age), 18, 60).astype(int)

    # Occupation types in India's informal & gig economy
    occupations = [
        "gig_delivery",       # Zomato, Swiggy, Zepto, Blinkit
        "gig_rideshare",      # Uber, Ola, Rapido
        "informal_retail",    # Kirana store assistants, local bazaar vendors
        "freelance_digital",  # Content creators, graphic design, remote tech gigs
        "daily_wage_labor",   # Construction, loading, casual contract labor
        "small_trader"        # Hawkers, micro-merchants, street cart operators
    ]
    occ_weights = [0.25, 0.20, 0.15, 0.10, 0.15, 0.15]
    occupation_type = np.random.choice(occupations, size=num_samples, p=occ_weights)

    # City Tier: Tier 1 (Metros: Bengaluru, Mumbai, Delhi-NCR), Tier 2 (Pune, Jaipur, etc.), Tier 3 (Semi-urban)
    city_tiers = ["tier_1", "tier_2", "tier_3"]
    city_tier_weights = [0.40, 0.35, 0.25]
    city_tier = np.random.choice(city_tiers, size=num_samples, p=city_tier_weights)

    # City tier income multiplier: higher living expense & earning potential in Tier 1
    tier_income_multiplier = {
        "tier_1": 1.25,
        "tier_2": 1.00,
        "tier_3": 0.80
    }
    multipliers = np.array([tier_income_multiplier[t] for t in city_tier])

    # Base income distributions per occupation (in INR)
    # Reflecting typical informal and gig wage realities in India
    base_incomes = np.zeros(num_samples)
    for i, occ in enumerate(occupation_type):
        if occ == "daily_wage_labor":
            base = np.random.uniform(10000, 17000)
        elif occ == "informal_retail":
            base = np.random.uniform(14000, 24000)
        elif occ == "gig_delivery":
            base = np.random.uniform(18000, 28000)
        elif occ == "gig_rideshare":
            base = np.random.uniform(22000, 36000)
        elif occ == "small_trader":
            base = np.random.uniform(18000, 38000)
        elif occ == "freelance_digital":
            base = np.random.uniform(25000, 60000)
        else:
            base = 20000.0
        base_incomes[i] = base

    # Derived monthly income estimate (incurred by living tier + personal baseline)
    monthly_income_estimate = np.round(base_incomes * multipliers, 2)

    # -------------------------------------------------------------------------
    # 2. Non-traditional proxy signals
    # -------------------------------------------------------------------------

    # 2.1 Utility bill payment consistency
    # Correlation: higher income + older age -> higher on-time rate
    # Realistic noise: 15-20% of high-income borrowers still show payment volatility (forgetfulness, travel, disputes)
    norm_income = (monthly_income_estimate - monthly_income_estimate.min()) / (monthly_income_estimate.max() - monthly_income_estimate.min())
    norm_age = (age - 18.0) / (60.0 - 18.0)
    
    # Base on-time propensity (0 to 1)
    base_ontime_propensity = 0.40 + 0.35 * norm_income + 0.25 * norm_age
    
    # Inject 18% random behavioral noise
    noise_mask = np.random.rand(num_samples) < 0.18
    bill_noise = np.random.uniform(-0.35, 0.20, size=num_samples)
    base_ontime_propensity = np.where(noise_mask, np.clip(base_ontime_propensity + bill_noise, 0.05, 0.95), base_ontime_propensity)

    # electricity_bill_ontime_rate: % of last 12 months paid on/before due date
    # Simulated as binomial draws out of 12 months, expressed as proportion
    bills_ontime_count = np.random.binomial(n=12, p=np.clip(base_ontime_propensity, 0.05, 0.98))
    electricity_bill_ontime_rate = np.round(bills_ontime_count / 12.0, 3)

    # electricity_bill_avg_delay_days: inversely proportional to on-time rate
    # Higher delay days indicate recurring liquidity crunches
    avg_delay_raw = (1.0 - electricity_bill_ontime_rate) * np.random.gamma(shape=3.0, scale=3.5, size=num_samples)
    electricity_bill_avg_delay_days = np.round(np.clip(avg_delay_raw, 0.0, 45.0), 1)

    # 2.2 Mobile recharge / telecom behavior
    # In India, low-income/informal workers frequently buy smaller sachets (daily/weekly data vouchers)
    # Higher frequency + small amount + high volatility = hand-to-mouth cashflow
    recharge_frequency_per_month = np.zeros(num_samples)
    avg_recharge_amount = np.zeros(num_samples)
    recharge_amount_volatility = np.zeros(num_samples)
    days_since_last_recharge_lapse = np.zeros(num_samples)

    for i in range(num_samples):
        inc = monthly_income_estimate[i]
        if inc < 18000:
            # Low income: frequent small top-ups (INR 49 - 179)
            recharge_frequency_per_month[i] = np.random.choice([3, 4, 5, 6, 7], p=[0.2, 0.3, 0.25, 0.15, 0.1])
            avg_recharge_amount[i] = np.round(np.random.uniform(65, 160), 2)
            # High volatility in recharge tickets
            recharge_amount_volatility[i] = np.round(np.random.uniform(0.25, 0.75), 3)
            # Recent recharge lapses are more frequent (0-45 days ago)
            days_since_last_recharge_lapse[i] = np.random.exponential(scale=35.0)
        elif inc < 32000:
            # Moderate income: monthly packs (INR 239 - 349) with occasional top-ups
            recharge_frequency_per_month[i] = np.random.choice([1, 2, 3, 4], p=[0.4, 0.35, 0.15, 0.1])
            avg_recharge_amount[i] = np.round(np.random.uniform(199, 399), 2)
            recharge_amount_volatility[i] = np.round(np.random.uniform(0.10, 0.40), 3)
            days_since_last_recharge_lapse[i] = np.random.exponential(scale=90.0) + 15
        else:
            # Higher income: periodic quarterly or annual packs (INR 479 - 899)
            recharge_frequency_per_month[i] = np.random.choice([1, 2], p=[0.75, 0.25])
            avg_recharge_amount[i] = np.round(np.random.uniform(349, 799), 2)
            recharge_amount_volatility[i] = np.round(np.random.uniform(0.02, 0.20), 3)
            days_since_last_recharge_lapse[i] = np.random.exponential(scale=180.0) + 60

    days_since_last_recharge_lapse = np.round(np.clip(days_since_last_recharge_lapse, 1.0, 365.0), 1)

    # 2.3 UPI / digital transaction patterns
    # Monthly UPI transaction count correlates with digital comfort and occupation
    upi_txn_base = np.random.poisson(lam=45, size=num_samples)
    # Daily wage labor has lower digital adoption; gig delivery and freelancers have very high
    occ_upi_boost = {
        "daily_wage_labor": -20,
        "informal_retail": 10,
        "small_trader": 25,
        "gig_delivery": 35,
        "gig_rideshare": 30,
        "freelance_digital": 40
    }
    boosts = np.array([occ_upi_boost[o] for o in occupation_type])
    monthly_upi_transaction_count = np.clip(upi_txn_base + boosts, 3, 200).astype(int)

    # Monthly UPI Inflow: closely tracks monthly income + small noise
    inflow_ratio = np.random.normal(loc=0.92, scale=0.08, size=num_samples)
    monthly_upi_inflow_avg = np.round(np.maximum(monthly_income_estimate * np.clip(inflow_ratio, 0.50, 1.30), 3000.0), 2)

    # Monthly UPI Outflow: typically 85-98% of inflow; cash-strapped borrowers spend nearly 100% or more
    outflow_ratio = np.random.beta(a=12, b=1.5, size=num_samples)  # Mean ~0.89, skewed towards high burn
    # Higher income individuals retain a higher savings buffer (lower outflow ratio)
    outflow_ratio = np.clip(outflow_ratio - 0.12 * norm_income + np.random.normal(0, 0.04, num_samples), 0.60, 1.08)
    monthly_upi_outflow_avg = np.round(monthly_upi_inflow_avg * outflow_ratio, 2)

    # upi_inflow_volatility_coefficient:
    # Gig income is famously lumpy. Modeled via Gamma distribution per occupation
    upi_volatility = np.zeros(num_samples)
    for i, occ in enumerate(occupation_type):
        if occ in ["daily_wage_labor", "freelance_digital"]:
            # Highest lumpiness / erratic timing
            upi_volatility[i] = np.random.gamma(shape=3.5, scale=0.14)
        elif occ in ["gig_delivery", "gig_rideshare"]:
            # Weekly payout cycles but seasonal surges / rain incentives
            upi_volatility[i] = np.random.gamma(shape=2.5, scale=0.12)
        else:
            # Traders and informal retail have steadier daily cash intake
            upi_volatility[i] = np.random.gamma(shape=2.0, scale=0.10)
    upi_inflow_volatility_coefficient = np.round(np.clip(upi_volatility, 0.08, 0.95), 3)

    # p2p_vs_merchant_txn_ratio:
    # High P2P ratio indicates informal borrowing/lending from acquaintances or local moneylenders
    # Merchant transactions (kirana QR scans, utility bills) represent healthy consumption
    raw_p2p_ratio = np.random.lognormal(mean=-0.2, sigma=0.6, size=num_samples)
    p2p_vs_merchant_txn_ratio = np.round(np.clip(raw_p2p_ratio, 0.05, 5.0), 2)

    # 2.4 Gig-platform earnings stability
    # Populated strictly for gig_delivery and gig_rideshare, NaN / null for other professions
    is_gig_worker = np.isin(occupation_type, ["gig_delivery", "gig_rideshare"])
    
    avg_weekly_gig_hours = np.full(num_samples, np.nan)
    gig_platform_rating = np.full(num_samples, np.nan)
    active_weeks_last_6_months = np.full(num_samples, np.nan)
    earnings_coefficient_of_variation = np.full(num_samples, np.nan)

    for i in range(num_samples):
        if is_gig_worker[i]:
            # Platform hours: 15 to 65 hours per week
            avg_weekly_gig_hours[i] = np.round(np.random.normal(loc=42.0, scale=10.0), 1)
            # Platform ratings: skewed heavily toward 4.2 - 4.95 (platforms deactivate below ~4.3)
            raw_rating = 5.0 - np.random.exponential(scale=0.25)
            gig_platform_rating[i] = np.round(np.clip(raw_rating, 3.0, 5.0), 2)
            # Consistency: active weeks out of 26
            active_weeks = np.random.binomial(n=26, p=0.82)
            active_weeks_last_6_months[i] = int(np.clip(active_weeks, 2, 26))
            # Weekly earnings CV
            earnings_coefficient_of_variation[i] = np.round(np.random.uniform(0.12, 0.58), 3)

    avg_weekly_gig_hours = np.clip(avg_weekly_gig_hours, 10.0, 75.0)

    # 2.5 Digital footprint stability (secondary signal)
    # phone_number_tenure_months: SIM stability as an anti-fraud proxy (informal fraud frequently rotates SIMs)
    # App account age: tenure in ecosystem
    tenure_base = np.random.exponential(scale=30.0, size=num_samples) + (age - 18) * 0.8
    phone_number_tenure_months = np.clip(np.round(tenure_base), 3, 180).astype(int)

    app_account_age_base = np.random.exponential(scale=14.0, size=num_samples) + 2.0
    app_account_age_months = np.clip(np.round(app_account_age_base), 1, 60).astype(int)

    # -------------------------------------------------------------------------
    # 3. TARGET VARIABLE GENERATION (`defaulted`)
    # -------------------------------------------------------------------------
    # The target `defaulted` is binary (0/1), modeled via latent creditworthiness.
    # Formula weights:
    # - Income stability (40% weight): Buffer between inflow & outflow + low UPI volatility
    # - Payment consistency (30% weight): Utility bill ontime rate + telecom lapse avoidance
    # - Earnings volatility (20% weight): Low recharge & gig earnings volatility
    # - Digital footprint stability (10% weight): Long SIM tenure + established app account age
    # 
    # All sub-scores are normalized to [0, 1] where 1.0 indicates highest creditworthiness.

    # 3.1 Income stability subscore (40%)
    # Net inflow margin = (inflow - outflow) / inflow
    net_margin = (monthly_upi_inflow_avg - monthly_upi_outflow_avg) / np.maximum(monthly_upi_inflow_avg, 1.0)
    norm_margin = np.clip((net_margin - 0.0) / 0.35, 0.0, 1.0)
    norm_upi_stability = 1.0 - np.clip(upi_inflow_volatility_coefficient / 0.85, 0.0, 1.0)
    subscore_income_stability = 0.60 * norm_margin + 0.40 * norm_upi_stability

    # 3.2 Payment consistency subscore (30%)
    norm_utility_ontime = electricity_bill_ontime_rate  # already 0 to 1
    norm_delay = 1.0 - np.clip(electricity_bill_avg_delay_days / 30.0, 0.0, 1.0)
    norm_lapse = np.clip(days_since_last_recharge_lapse / 180.0, 0.0, 1.0)
    
    # Incorporate gig activity for gig workers if present
    gig_bonus = np.zeros(num_samples)
    for i in range(num_samples):
        if is_gig_worker[i]:
            gig_bonus[i] = (active_weeks_last_6_months[i] / 26.0) * 0.5 + ((gig_platform_rating[i] - 3.0) / 2.0) * 0.5
        else:
            gig_bonus[i] = 0.70  # Baseline neutral for non-gig borrowers

    subscore_payment_consistency = (
        0.40 * norm_utility_ontime +
        0.25 * norm_delay +
        0.20 * norm_lapse +
        0.15 * gig_bonus
    )

    # 3.3 Earnings volatility subscore (20%) (Higher = more stable)
    norm_recharge_stability = 1.0 - np.clip(recharge_amount_volatility / 0.70, 0.0, 1.0)
    norm_p2p_prudence = 1.0 - np.clip((p2p_vs_merchant_txn_ratio - 0.2) / 2.5, 0.0, 1.0)
    subscore_earnings_volatility = 0.60 * norm_recharge_stability + 0.40 * norm_p2p_prudence

    # 3.4 Digital footprint stability subscore (10%)
    norm_sim_tenure = np.clip(phone_number_tenure_months / 72.0, 0.0, 1.0)
    norm_app_tenure = np.clip(app_account_age_months / 36.0, 0.0, 1.0)
    subscore_digital_footprint = 0.60 * norm_sim_tenure + 0.40 * norm_app_tenure

    # Composite latent creditworthiness (0 to 1, higher = more creditworthy)
    latent_creditworthiness = (
        0.40 * subscore_income_stability +
        0.30 * subscore_payment_consistency +
        0.20 * subscore_earnings_volatility +
        0.10 * subscore_digital_footprint
    )

    # Logistic transformation to default probability:
    # log_odds = beta_0 - beta_1 * latent_creditworthiness + noise
    # Calibrate beta_0 and beta_1 so default rate lands accurately in the ~12-15% target range.
    beta_0 = 1.45
    beta_1 = 6.20
    irreducible_noise = np.random.normal(loc=0.0, scale=0.75, size=num_samples)
    
    log_odds = beta_0 - (beta_1 * latent_creditworthiness) + irreducible_noise
    prob_default = 1.0 / (1.0 + np.exp(-log_odds))

    # Bernoulli draw for binary default outcome
    defaulted = (np.random.rand(num_samples) < prob_default).astype(int)

    # -------------------------------------------------------------------------
    # 4. DATA QUALITY REALISM: MISSING VALUES & OUTLIERS
    # -------------------------------------------------------------------------
    df = pd.DataFrame({
        "borrower_id": borrower_ids,
        "age": age,
        "occupation_type": occupation_type,
        "city_tier": city_tier,
        "monthly_income_estimate": monthly_income_estimate,
        "electricity_bill_ontime_rate": electricity_bill_ontime_rate,
        "electricity_bill_avg_delay_days": electricity_bill_avg_delay_days,
        "recharge_frequency_per_month": recharge_frequency_per_month,
        "avg_recharge_amount": avg_recharge_amount,
        "recharge_amount_volatility": recharge_amount_volatility,
        "days_since_last_recharge_lapse": days_since_last_recharge_lapse,
        "monthly_upi_transaction_count": monthly_upi_transaction_count,
        "monthly_upi_inflow_avg": monthly_upi_inflow_avg,
        "monthly_upi_outflow_avg": monthly_upi_outflow_avg,
        "upi_inflow_volatility_coefficient": upi_inflow_volatility_coefficient,
        "p2p_vs_merchant_txn_ratio": p2p_vs_merchant_txn_ratio,
        "avg_weekly_gig_hours": avg_weekly_gig_hours,
        "gig_platform_rating": gig_platform_rating,
        "active_weeks_last_6_months": active_weeks_last_6_months,
        "earnings_coefficient_of_variation": earnings_coefficient_of_variation,
        "phone_number_tenure_months": phone_number_tenure_months,
        "app_account_age_months": app_account_age_months,
        "defaulted": defaulted
    })

    # 4.1 Inject 3-5% realistic missing values into select digital & utility fields
    # Represents borrowers who conduct cash-only transactions or have incomplete bill scrape records
    missing_cols_rates = {
        "monthly_upi_transaction_count": 0.04,
        "monthly_upi_inflow_avg": 0.035,
        "monthly_upi_outflow_avg": 0.035,
        "upi_inflow_volatility_coefficient": 0.045,
        "days_since_last_recharge_lapse": 0.03,
        "electricity_bill_avg_delay_days": 0.025
    }
    
    for col, rate in missing_cols_rates.items():
        missing_indices = np.random.choice(num_samples, size=int(num_samples * rate), replace=False)
        df.loc[missing_indices, col] = np.nan

    # 4.2 Inject ~1% extreme outliers to stress-test feature engineering and downstream models
    outlier_count = int(num_samples * 0.01)
    outlier_indices = np.random.choice(num_samples, size=outlier_count, replace=False)

    # Half get abnormally high UPI transaction volume (e.g. bulk reseller), half get extreme delays
    half_outliers = outlier_count // 2
    df.loc[outlier_indices[:half_outliers], "monthly_upi_transaction_count"] = np.random.randint(450, 900, size=half_outliers)
    df.loc[outlier_indices[:half_outliers], "monthly_upi_inflow_avg"] = df.loc[outlier_indices[:half_outliers], "monthly_upi_inflow_avg"] * 6.0
    df.loc[outlier_indices[half_outliers:], "electricity_bill_avg_delay_days"] = np.random.uniform(75.0, 120.0, size=outlier_count - half_outliers)

    return df


def main():
    print_data_dictionary()
    print("\nGenerating 8,000 synthetic borrower profiles with realistic alt-data proxy signals...")
    df = generate_synthetic_data(num_samples=NUM_BORROWERS)

    default_rate = df["defaulted"].mean() * 100
    print(f"\n[Generated Dataset Summary]")
    print(f"Total records: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Target 'defaulted' rate: {default_rate:.2f}% (Target specification: ~12-15%)")
    
    # Verify non-trivial correlations
    high_vol_def = df[df["upi_inflow_volatility_coefficient"] > df["upi_inflow_volatility_coefficient"].median()]["defaulted"].mean() * 100
    low_vol_def = df[df["upi_inflow_volatility_coefficient"] <= df["upi_inflow_volatility_coefficient"].median()]["defaulted"].mean() * 100
    print(f"Default rate for High UPI Volatility borrowers: {high_vol_def:.2f}%")
    print(f"Default rate for Low UPI Volatility borrowers:  {low_vol_def:.2f}%")
    print(f"Risk gradient demonstrated: {(high_vol_def - low_vol_def):+.2f}% delta between high/low volatility segments.")

    # Save relative to project root
    data_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(data_dir, "synthetic_borrowers.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSaved synthetic dataset to: {output_path}")


if __name__ == "__main__":
    main()
