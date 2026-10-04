"""
CreditBridge - Defensible Temporal Synthetic Data Generator
Path: src/temporal_data_generator.py

Reconstructs the synthetic data generator to be scientifically defensible:
1. Two-stage generative architecture:
   - Stage 1: Latent borrower state (financial capacity, expense burden, volatility, resilience).
   - Stage 2: Observable transaction history across 12-month observation window (T-12 to T-1).
2. Separate forward prediction horizon (T to T+3 = 90 days):
   - Future default event is simulated forward from liquidity shocks and cumulative cashflow solvency.
   - The target is NOT a linear/deterministic combination of observation features.
3. Deterministic temporal partitions:
   - Train cohort (T_obs_end: 2023-06-30, Target window: 2023-07 to 2023-09) -> 5,000 rows
   - Validation cohort (T_obs_end: 2023-09-30, Target window: 2023-10 to 2023-12) -> 1,500 rows
   - Out-of-Time (OOT) cohort (T_obs_end: 2023-12-31, Target window: 2024-01 to 2024-03) -> 1,500 rows
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# SIMULATION ASSUMPTIONS & ARCHETYPES
# -----------------------------------------------------------------------------
OCCUPATION_CONFIGS: Dict[str, Dict[str, Any]] = {
    "daily_wage_labor": {
        "share": 0.15,
        "base_income_range": (14000, 22000),
        "expense_ratio_mean": 0.88,
        "volatility_base": 0.35,
        "shock_prob_monthly": 0.22,
        "is_gig": False,
        "p2p_ratio_mean": 1.4,
    },
    "gig_delivery": {
        "share": 0.25,
        "base_income_range": (18000, 32000),
        "expense_ratio_mean": 0.82,
        "volatility_base": 0.22,
        "shock_prob_monthly": 0.16,
        "is_gig": True,
        "p2p_ratio_mean": 0.9,
    },
    "gig_rideshare": {
        "share": 0.20,
        "base_income_range": (24000, 42000),
        "expense_ratio_mean": 0.80,
        "volatility_base": 0.19,
        "shock_prob_monthly": 0.18,  # vehicle repair risk
        "is_gig": True,
        "p2p_ratio_mean": 0.8,
    },
    "informal_retail": {
        "share": 0.15,
        "base_income_range": (20000, 45000),
        "expense_ratio_mean": 0.78,
        "volatility_base": 0.18,
        "shock_prob_monthly": 0.14,
        "is_gig": False,
        "p2p_ratio_mean": 0.7,
    },
    "small_trader": {
        "share": 0.15,
        "base_income_range": (25000, 60000),
        "expense_ratio_mean": 0.74,
        "volatility_base": 0.16,
        "shock_prob_monthly": 0.12,
        "is_gig": False,
        "p2p_ratio_mean": 0.6,
    },
    "freelance_digital": {
        "share": 0.10,
        "base_income_range": (22000, 65000),
        "expense_ratio_mean": 0.72,
        "volatility_base": 0.30,
        "shock_prob_monthly": 0.13,
        "is_gig": False,
        "p2p_ratio_mean": 0.5,
    },
}

CITY_TIER_CONFIGS: Dict[str, Dict[str, Any]] = {
    "tier_1": {"share": 0.40, "income_mult": 1.25, "cost_mult": 1.25},
    "tier_2": {"share": 0.35, "income_mult": 1.00, "cost_mult": 1.00},
    "tier_3": {"share": 0.25, "income_mult": 0.82, "cost_mult": 0.85},
}


@dataclass
class LatentBorrower:
    """Represents the unobservable true economic state and behavioral propensity."""
    borrower_id: str
    age: int
    occupation_type: str
    city_tier: str
    base_earning_capacity: float
    base_expense_burden: float
    inherent_volatility: float
    shock_susceptibility: float
    bill_payment_discipline: float  # 0 to 1
    savings_cushion_months: float   # 0 to 3 months of buffer
    is_gig: bool


def sample_latent_borrowers(num_borrowers: int, seed: int = 42) -> List[LatentBorrower]:
    """Generates the latent population of borrowers."""
    rng = np.random.default_rng(seed)
    borrowers: List[LatentBorrower] = []

    occupations = list(OCCUPATION_CONFIGS.keys())
    occ_probs = [OCCUPATION_CONFIGS[o]["share"] for o in occupations]

    city_tiers = list(CITY_TIER_CONFIGS.keys())
    tier_probs = [CITY_TIER_CONFIGS[t]["share"] for t in city_tiers]

    sampled_occs = rng.choice(occupations, size=num_borrowers, p=occ_probs)
    sampled_tiers = rng.choice(city_tiers, size=num_borrowers, p=tier_probs)

    for i in range(num_borrowers):
        occ = str(sampled_occs[i])
        tier = str(sampled_tiers[i])
        occ_cfg = OCCUPATION_CONFIGS[occ]
        tier_cfg = CITY_TIER_CONFIGS[tier]

        # Age distribution: skewed toward working age 21 to 52
        age = int(np.clip(rng.normal(loc=31.5, scale=7.5), 19, 58))

        # Base earning capacity (INR/month)
        min_inc, max_inc = occ_cfg["base_income_range"]
        raw_income = rng.uniform(min_inc, max_inc) * tier_cfg["income_mult"]
        # Experience bump with age
        income_exp_factor = 1.0 + min((age - 20) * 0.012, 0.35)
        base_income = round(raw_income * income_exp_factor, 2)

        # Expense burden
        exp_mean = occ_cfg["expense_ratio_mean"]
        expense_ratio = float(np.clip(rng.normal(exp_mean, 0.06), 0.55, 0.98))
        base_expense = round(base_income * expense_ratio * (tier_cfg["cost_mult"] / tier_cfg["income_mult"]), 2)

        # Volatility
        vol = float(np.clip(rng.normal(occ_cfg["volatility_base"], 0.05), 0.08, 0.55))

        # Shock susceptibility
        shock_p = float(np.clip(rng.normal(occ_cfg["shock_prob_monthly"], 0.04), 0.04, 0.38))

        # Discipline (drives timely bill payment)
        discipline = float(np.clip(rng.beta(a=3.5, b=1.5), 0.05, 0.99))

        # Initial savings cushion in months of expenses (thin-file: mostly 0.2 to 2.0 months)
        cushion = float(np.clip(rng.exponential(scale=0.8), 0.05, 3.5))

        borrowers.append(
            LatentBorrower(
                borrower_id=str(uuid.UUID(bytes=bytes(rng.integers(0, 256, size=16, dtype=np.uint8)))),
                age=age,
                occupation_type=occ,
                city_tier=tier,
                base_earning_capacity=base_income,
                base_expense_burden=base_expense,
                inherent_volatility=vol,
                shock_susceptibility=shock_p,
                bill_payment_discipline=discipline,
                savings_cushion_months=cushion,
                is_gig=occ_cfg["is_gig"],
            )
        )

    return borrowers


def simulate_borrower_history_and_default(
    borrower: LatentBorrower,
    cohort_split: str,
    cutoff_date: date,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """
    Simulates:
    1. 12 months of observation history (months -12 to -1).
    2. Derived observation features matching CreditBridge FeaturePipeline schema.
    3. 3 months of forward prediction horizon (months 0 to +2 = 90 days) determining default.
    """
    # -------------------------------------------------------------------------
    # PART 1: 12-Month Observation Window
    # -------------------------------------------------------------------------
    monthly_inflows: List[float] = []
    monthly_outflows: List[float] = []
    monthly_bills_ontime: List[int] = []
    monthly_bill_delays: List[float] = []
    monthly_recharges: List[float] = []
    recharge_counts: List[int] = []
    lapses_in_window = 0
    days_since_lapse = 180.0

    current_cushion_inr = borrower.savings_cushion_months * borrower.base_expense_burden

    for m in range(12):
        # Seasonality: Diwali festival surge (months 3-4 from start or Oct-Nov), monsoon dip for labor
        month_idx = (cutoff_date.month - 12 + m) % 12 + 1
        season_inflow_mult = 1.0
        if month_idx in [10, 11]:  # Festival season
            season_inflow_mult = 1.15 if borrower.is_gig or borrower.occupation_type in ["informal_retail", "small_trader"] else 1.05
        elif month_idx in [7, 8] and borrower.occupation_type == "daily_wage_labor":  # Monsoon slump
            season_inflow_mult = 0.82

        # Stochastic monthly inflow
        inflow_noise = rng.normal(0, borrower.inherent_volatility)
        inflow = max(borrower.base_earning_capacity * (season_inflow_mult + inflow_noise), 3000.0)

        # Stochastic monthly outflow (living expenses, essentials)
        outflow_noise = rng.normal(0, 0.08)
        outflow = max(borrower.base_expense_burden * (1.0 + outflow_noise), 2500.0)

        # Liquidity shock check
        if rng.random() < borrower.shock_susceptibility:
            shock_amt = rng.uniform(4000.0, 20000.0)
            outflow += shock_amt

        net_flow = inflow - outflow
        current_cushion_inr += net_flow

        # Utility bill behavior
        # Probability of paying on time is driven by discipline and liquidity cushion
        ontime_prob = borrower.bill_payment_discipline * (1.0 if current_cushion_inr > 0 else 0.45)
        if rng.random() < ontime_prob:
            monthly_bills_ontime.append(1)
            monthly_bill_delays.append(0.0)
        else:
            monthly_bills_ontime.append(0)
            delay = rng.exponential(scale=12.0) + (0.0 if current_cushion_inr > 0 else 14.0)
            monthly_bill_delays.append(min(delay, 60.0))

        # Telecom recharge behavior
        num_rc = int(np.clip(rng.poisson(lam=2.8), 1, 8))
        recharge_counts.append(num_rc)
        # Average ticket size
        rc_amt = float(np.clip(rng.normal(249.0, 60.0), 99.0, 799.0))
        monthly_recharges.append(rc_amt)

        # Telecom balance lapse event (when cushion is critically low)
        if current_cushion_inr < -5000.0 and rng.random() < 0.25:
            lapses_in_window += 1
            days_since_lapse = rng.uniform(5.0, 45.0)
        else:
            days_since_lapse = min(days_since_lapse + 30.0, 180.0)

        monthly_inflows.append(inflow)
        monthly_outflows.append(outflow)

    # -------------------------------------------------------------------------
    # PART 2: Aggregate Observation Features (strictly matching schema)
    # -------------------------------------------------------------------------
    inflow_arr = np.array(monthly_inflows)
    outflow_arr = np.array(monthly_outflows)

    monthly_inflow_avg = float(np.mean(inflow_arr))
    monthly_outflow_avg = float(np.mean(outflow_arr))
    inflow_std = float(np.std(inflow_arr))
    inflow_cv = float(inflow_std / max(monthly_inflow_avg, 1.0))

    # UPI transaction count per month (correlated with income frequency)
    tx_count_base = int(monthly_inflow_avg / 850.0)
    monthly_upi_tx_count = int(np.clip(rng.poisson(lam=max(tx_count_base, 10)), 12, 180))

    # P2P ratio: increases if cushion is negative (borrowing from peers/family)
    p2p_mean = OCCUPATION_CONFIGS[borrower.occupation_type]["p2p_ratio_mean"]
    if current_cushion_inr < 0:
        p2p_mean *= 1.45
    p2p_ratio = float(np.clip(rng.lognormal(mean=np.log(p2p_mean), sigma=0.35), 0.1, 4.8))

    # Electricity bill metrics
    bill_ontime_rate = float(np.mean(monthly_bills_ontime))
    bill_avg_delay = float(np.mean(monthly_bill_delays))

    # Telecom metrics
    recharge_freq = float(np.mean(recharge_counts))
    avg_rc_amt = float(np.mean(monthly_recharges))
    rc_volatility = float(np.std(monthly_recharges) / max(avg_rc_amt, 1.0))

    # Gig specific features (strictly NaN for non-gig borrowers)
    if borrower.is_gig:
        gig_hours = float(np.clip(rng.normal(40.0, 9.0), 15.0, 70.0))
        # Platform rating (mean 4.6, range 3.5 to 5.0)
        gig_rating = float(np.clip(5.0 - rng.exponential(scale=0.22), 3.2, 5.0))
        active_weeks = int(np.clip(rng.binomial(26, p=0.86), 8, 26))
        earnings_cv = float(np.clip(rng.uniform(0.12, 0.48), 0.08, 0.65))
    else:
        gig_hours = np.nan
        gig_rating = np.nan
        active_weeks = np.nan
        earnings_cv = np.nan

    # Tenure metrics
    tenure_sim = int(np.clip(rng.exponential(scale=32.0) + (borrower.age - 18) * 0.75, 4, 180))
    app_age_sim = int(np.clip(rng.exponential(scale=15.0) + 3.0, 1, 60))

    # -------------------------------------------------------------------------
    # PART 3: Forward 90-Day Prediction Window (Months 0, 1, 2)
    # -------------------------------------------------------------------------
    # Forward simulation determines default. Default occurs if:
    # Forward cashflows and shocks cause a severe solvency deficit that cannot service obligations.
    forward_cushion = current_cushion_inr
    default_occurred = 0

    # Macro shock scenario for OOT split (e.g. slight inflation shock in early 2024)
    macro_expense_mult = 1.04 if cohort_split == "oot" else 1.0

    for fwd_m in range(3):
        fwd_inflow = max(borrower.base_earning_capacity * (1.0 + rng.normal(0, borrower.inherent_volatility)), 2500.0)
        fwd_outflow = max(borrower.base_expense_burden * macro_expense_mult * (1.0 + rng.normal(0, 0.08)), 2500.0)

        # Forward shock
        if rng.random() < borrower.shock_susceptibility:
            fwd_shock = rng.uniform(4000.0, 18000.0)
            fwd_outflow += fwd_shock

        # Forward debt repayment obligation (simulated ~15% of earning capacity)
        debt_obligation = borrower.base_earning_capacity * 0.15
        fwd_outflow += debt_obligation

        forward_cushion += (fwd_inflow - fwd_outflow)

        # Solvency shortfall condition: cushion drops negative past buffer tolerance
        if forward_cushion < -0.08 * borrower.base_expense_burden:
            deficit_ratio = min(max(0.0, -forward_cushion) / max(borrower.base_expense_burden, 1.0), 1.0)
            # Default risk increases with deficit depth and lack of payment discipline
            p_default = 0.22 + 0.45 * (1.0 - borrower.bill_payment_discipline) + 0.33 * deficit_ratio
            if rng.random() < p_default:
                default_occurred = 1
                break

    return {
        "borrower_id": borrower.borrower_id,
        "age": borrower.age,
        "occupation_type": borrower.occupation_type,
        "city_tier": borrower.city_tier,
        "monthly_income_estimate": round(monthly_inflow_avg, 2),
        "electricity_bill_ontime_rate": round(bill_ontime_rate, 4),
        "electricity_bill_avg_delay_days": round(bill_avg_delay, 2),
        "recharge_frequency_per_month": round(recharge_freq, 2),
        "avg_recharge_amount": round(avg_rc_amt, 2),
        "recharge_amount_volatility": round(rc_volatility, 4),
        "days_since_last_recharge_lapse": round(days_since_lapse, 1),
        "monthly_upi_transaction_count": monthly_upi_tx_count,
        "monthly_upi_inflow_avg": round(monthly_inflow_avg, 2),
        "monthly_upi_outflow_avg": round(monthly_outflow_avg, 2),
        "upi_inflow_volatility_coefficient": round(inflow_cv, 4),
        "p2p_vs_merchant_txn_ratio": round(p2p_ratio, 2),
        "avg_weekly_gig_hours": gig_hours,
        "gig_platform_rating": gig_rating,
        "active_weeks_last_6_months": active_weeks,
        "earnings_coefficient_of_variation": earnings_cv,
        "phone_number_tenure_months": tenure_sim,
        "app_account_age_months": app_age_sim,
        "cohort_split": cohort_split,
        "observation_cutoff": cutoff_date.isoformat(),
        "defaulted": int(default_occurred),
    }


def generate_full_temporal_dataset(
    seed: int = 42,
    train_count: int = 5000,
    val_count: int = 1500,
    oot_count: int = 1500,
) -> pd.DataFrame:
    """
    Generates the complete temporal dataset with train, val, and out-of-time (OOT) cohorts.
    Total default rate is naturally calibrated around 13-16% without synthetic circularity.
    """
    rng = np.random.default_rng(seed)
    total_count = train_count + val_count + oot_count

    # Sample latent population
    latent_population = sample_latent_borrowers(total_count, seed=seed)

    records: List[Dict[str, Any]] = []

    # 1. Train Cohort
    train_cutoff = date(2023, 6, 30)
    for b in latent_population[:train_count]:
        rec = simulate_borrower_history_and_default(b, cohort_split="train", cutoff_date=train_cutoff, rng=rng)
        records.append(rec)

    # 2. Validation Cohort
    val_cutoff = date(2023, 9, 30)
    for b in latent_population[train_count : train_count + val_count]:
        rec = simulate_borrower_history_and_default(b, cohort_split="val", cutoff_date=val_cutoff, rng=rng)
        records.append(rec)

    # 3. Out-Of-Time (OOT) Cohort
    oot_cutoff = date(2023, 12, 31)
    for b in latent_population[train_count + val_count :]:
        rec = simulate_borrower_history_and_default(b, cohort_split="oot", cutoff_date=oot_cutoff, rng=rng)
        records.append(rec)

    df = pd.DataFrame(records)

    # Inject realistic missing values (3-5%) into select digital & utility fields
    # Represents borrowers who conduct cash-only transactions or have incomplete bill scrape records
    missing_cols_rates = {
        "monthly_upi_transaction_count": 0.04,
        "monthly_upi_inflow_avg": 0.035,
        "monthly_upi_outflow_avg": 0.035,
        "upi_inflow_volatility_coefficient": 0.045,
        "days_since_last_recharge_lapse": 0.03,
        "electricity_bill_avg_delay_days": 0.025,
    }
    for col, rate in missing_cols_rates.items():
        if col in df.columns:
            missing_idx = rng.choice(len(df), size=int(len(df) * rate), replace=False)
            df.loc[missing_idx, col] = np.nan

    # Inject ~1% extreme outliers to test feature pipeline clipping and robustness
    outlier_count = int(len(df) * 0.01)
    outlier_idx = rng.choice(len(df), size=outlier_count, replace=False)
    half = outlier_count // 2
    df.loc[outlier_idx[:half], "monthly_upi_transaction_count"] = rng.integers(450, 900, size=half)
    df.loc[outlier_idx[:half], "monthly_upi_inflow_avg"] = df.loc[outlier_idx[:half], "monthly_upi_inflow_avg"] * 5.0
    df.loc[outlier_idx[half:], "electricity_bill_avg_delay_days"] = rng.uniform(75.0, 120.0, size=outlier_count - half)

    return df


if __name__ == "__main__":
    df = generate_full_temporal_dataset()
    print("Generated temporal dataset shape:", df.shape)
    print("Class prevalence by cohort:")
    print(df.groupby("cohort_split")["defaulted"].agg(["count", "mean"]))
