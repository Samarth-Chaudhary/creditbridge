"""
CreditBridge - Alternative Credit Scoring Engine
Real Data Mode: Transaction Intelligence & Feature Mapping
Path: src/real_data_features.py

Transforms validated canonical transactions into a model-compatible single-borrower
feature DataFrame (exact 22-column schema) with machine-readable provenance and
history sufficiency metrics.
"""

import uuid
from typing import Any, Dict, cast

import numpy as np
import pandas as pd

from src.real_data_contracts import (
    RAW_BORROWER_COLUMNS,
    UNAVAILABLE_MODEL_COLUMNS,
    FeatureProvenanceRecord,
    HistorySufficiencyReport,
    InsufficientHistoryError,
    InvalidDataError,
    ManualInputContract,
    NormalizedCategory,
    ParsedStatement,
    ProvenanceState,
    RealBorrowerPayload,
    SufficiencyTier,
    TransactionType,
)
from src.real_data_validation import validate_borrower_row, validate_manual_inputs


def evaluate_history_sufficiency(
    dates: Any,
    txn_count: int
) -> HistorySufficiencyReport:
    """
    Evaluates statement historical coverage according to locked institutional risk policy.

    Tiers:
    - <30 days or <15 txns: INSUFFICIENT (Block scoring)
    - 30-89 days: MARGINAL (Allow with manual review cap)
    - 90-179 days: ADEQUATE (Allow scoring)
    - 180+ days: OPTIMAL (Full 6-month baseline)
    """
    dt_series = pd.to_datetime(dates)
    start_date = dt_series.min().date()
    end_date = dt_series.max().date()

    total_days = max((end_date - start_date).days + 1, 1)
    approx_months = max(round(total_days / 30.0, 2), 1.0)

    if total_days < 30 or txn_count < 15:
        tier = SufficiencyTier.INSUFFICIENT.value
        is_scoreable = False
        action = "BLOCK_SCORING"
        warning = (
            f"Statement covers {total_days} calendar days and {txn_count} transactions. "
            "Underwriting policy strictly blocks scoring for histories under 30 days or <15 transactions."
        )
    elif total_days < 90:
        tier = SufficiencyTier.MARGINAL.value
        is_scoreable = True
        action = "ALLOW_WITH_MANUAL_REVIEW"
        warning = (
            f"Statement covers {total_days} calendar days ({approx_months} months). "
            "Short-window monthly volatility and recharge pacing are extrapolated; tier capped at Manual Review."
        )
    elif total_days < 180:
        tier = SufficiencyTier.ADEQUATE.value
        is_scoreable = True
        action = "ALLOW_SCORING"
        warning = (
            f"Statement covers {total_days} calendar days ({approx_months} months). "
            "Adequate historical coverage; quarterly seasonalities may be under-represented."
        )
    else:
        tier = SufficiencyTier.OPTIMAL.value
        is_scoreable = True
        action = "ALLOW_SCORING"
        warning = None

    return HistorySufficiencyReport(
        start_date=start_date,
        end_date=end_date,
        total_calendar_days=total_days,
        approx_months=approx_months,
        total_usable_transactions=txn_count,
        sufficiency_tier=tier,
        is_scoreable=is_scoreable,
        underwriting_action=action,
        warning_message=warning,
    )


def build_real_borrower_payload(
    statement_or_df: Any,
    manual_inputs: ManualInputContract,
    allow_insufficient_history: bool = False,
) -> RealBorrowerPayload:
    """
    Transforms canonical transactions and self-reported inputs into a model-compatible
    single-borrower feature DataFrame with parallel provenance records.

    Parameters:
    -----------
    statement_or_df : ParsedStatement or pd.DataFrame
        Canonical transactions table produced by Part 4.
    manual_inputs : ManualInputContract
        Validated user inputs (age, occupation_type, city_tier).
    allow_insufficient_history : bool
        If True, allows feature calculation even for <30 days history (used for diagnostic testing).

    Returns:
    --------
    RealBorrowerPayload:
        Container with 1-row DataFrame (22 columns), provenance records, and history report.
    """
    # 1. Extract canonical transactions DataFrame
    if isinstance(statement_or_df, ParsedStatement):
        tx_df = statement_or_df.transactions.copy()
    elif isinstance(statement_or_df, pd.DataFrame):
        tx_df = statement_or_df.copy()
    else:
        raise InvalidDataError(f"Expected ParsedStatement or DataFrame. Got: {type(statement_or_df)}")

    if tx_df.empty or len(tx_df) == 0:
        raise InvalidDataError("Cannot map features from an empty transaction dataset.")

    # 2. Validate Manual Inputs
    validate_manual_inputs(manual_inputs)

    # 3. Evaluate Statement History Sufficiency
    history_report = evaluate_history_sufficiency(tx_df["transaction_date"], len(tx_df))
    if not history_report.is_scoreable and not allow_insufficient_history:
        raise InsufficientHistoryError(history_report.warning_message)

    history_months = history_report.approx_months
    history_days = history_report.total_calendar_days

    provenance_records: Dict[str, FeatureProvenanceRecord] = {}

    # -------------------------------------------------------------------------
    # 4. INCOME CLASSIFICATION & DERIVATION
    # -------------------------------------------------------------------------
    # Filter valid credits:
    # Exclude: refunds, reversals, internal transfers, and loan proceeds
    is_credit = tx_df["transaction_type"] == TransactionType.CREDIT.value
    is_excluded = (
        tx_df["is_refund"]
        | tx_df["is_reversal"]
        | tx_df["is_internal_transfer"]
        | tx_df["normalized_category"].isin([
            NormalizedCategory.REFUND.value,
            NormalizedCategory.LOAN_RELATED.value,
            NormalizedCategory.TRANSFER.value,
            NormalizedCategory.INTERNAL_TRANSFER.value,
        ])
    )

    # Check for internal transfer keywords if account_holder_name was provided
    if manual_inputs.account_holder_name:
        name_clean = manual_inputs.account_holder_name.upper()
        self_xfer = tx_df["counterparty"].astype(str).str.upper().str.contains(name_clean, na=False)
        is_excluded = is_excluded | self_xfer

    eligible_credits = tx_df[is_credit & (~is_excluded)]

    # Apply qualification weights:
    # - 100% for recognized Salary, Gig payout, Business QR inflows
    # - 50% for P2P personal transfers (discounted for informal debt/reimbursements)
    # - 0% for Unknown credits (conservative BFSI posture)
    norm_cats = pd.Series(eligible_credits["normalized_category"])
    tier_a_mask = norm_cats.isin([
        NormalizedCategory.SALARY_LIKE.value,
        NormalizedCategory.GIG_INCOME_LIKE.value,
        NormalizedCategory.BUSINESS_INFLOW.value,
    ])
    tier_b_mask = norm_cats.isin([
        NormalizedCategory.PERSON_LIKE.value,
    ])

    tier_a_inflows = eligible_credits[tier_a_mask]["amount"].sum()
    tier_b_inflows = eligible_credits[tier_b_mask]["amount"].sum() * 0.50

    qualified_monthly_inflow = round(float((tier_a_inflows + tier_b_inflows) / history_months), 2)

    provenance_records["monthly_income_estimate"] = FeatureProvenanceRecord(
        feature_name="monthly_income_estimate",
        value=qualified_monthly_inflow,
        provenance_state=ProvenanceState.DERIVED.value,
        source="statement",
        transformation=f"Sum of qualified credits (Tier A 100% + Tier B P2P 50%) / {history_months} months",
        confidence="medium",
        is_imputed=False,
    )

    # -------------------------------------------------------------------------
    # 5. TELECOM RECHARGE FEATURES
    # -------------------------------------------------------------------------
    is_telco = (
        (tx_df["normalized_category"] == NormalizedCategory.TELECOM_RECHARGE.value)
        | tx_df["counterparty"].astype(str).str.contains(
            r"JIO|AIRTEL|VI |VODAFONE|BSNL|RECHARGE|PREPAID", case=False, na=False, regex=True
        )
    )
    telco_txns = tx_df[is_telco & (tx_df["transaction_type"] != TransactionType.REFUND.value)]
    telco_count = len(telco_txns)

    recharge_freq_mo = round(float(telco_count / history_months), 2)

    if telco_count > 0:
        avg_recharge_amt = round(float(cast(Any, telco_txns["amount"].mean())), 2)
    else:
        avg_recharge_amt = 0.0

    if telco_count >= 2 and avg_recharge_amt > 0.0:
        recharge_vol = round(float(cast(Any, telco_txns["amount"].std(ddof=1)) / avg_recharge_amt), 3)
    else:
        recharge_vol = 0.0

    provenance_records["recharge_frequency_per_month"] = FeatureProvenanceRecord(
        feature_name="recharge_frequency_per_month",
        value=recharge_freq_mo,
        provenance_state=ProvenanceState.DERIVED.value,
        source="statement",
        transformation=f"{telco_count} telco transactions / {history_months} months",
        confidence="high" if telco_count > 0 else "neutral",
        is_imputed=False,
    )
    provenance_records["avg_recharge_amount"] = FeatureProvenanceRecord(
        feature_name="avg_recharge_amount",
        value=avg_recharge_amt,
        provenance_state=ProvenanceState.DERIVED.value,
        source="statement",
        transformation=f"Mean ticket size of {telco_count} identified recharges",
        confidence="high" if telco_count > 0 else "neutral",
        is_imputed=False,
    )
    provenance_records["recharge_amount_volatility"] = FeatureProvenanceRecord(
        feature_name="recharge_amount_volatility",
        value=recharge_vol,
        provenance_state=ProvenanceState.DERIVED.value,
        source="statement",
        transformation="Coefficient of variation (std/mean) of recharge amounts",
        confidence="high" if telco_count >= 2 else "neutral",
        is_imputed=False,
    )

    # -------------------------------------------------------------------------
    # 6. UPI & DIGITAL CASHFLOW FEATURES
    # -------------------------------------------------------------------------
    # Identify UPI transactions via source_type, VPA syntax, or payment narration
    is_upi = (
        (tx_df["source_type"] == "csv_upi")
        | tx_df["counterparty"].astype(str).str.contains(r"UPI|VPA|@|IMPS|GPAY|PHONEPE|PAYTM", case=False, na=False, regex=True)
        | tx_df["normalized_category"].isin([
            NormalizedCategory.PERSON_LIKE.value,
            NormalizedCategory.MERCHANT_SPEND.value,
            NormalizedCategory.BUSINESS_INFLOW.value,
        ])
    )
    upi_txns = tx_df[is_upi]
    upi_count = len(upi_txns)
    monthly_upi_count = int(round(upi_count / history_months))

    upi_credits = tx_df[is_upi & (tx_df["transaction_type"] == TransactionType.CREDIT.value) & (~tx_df["is_refund"])]
    upi_debits = tx_df[is_upi & (tx_df["transaction_type"] == TransactionType.DEBIT.value)]

    monthly_upi_inflow = round(float(upi_credits["amount"].sum() / history_months), 2)
    monthly_upi_outflow = round(float(upi_debits["amount"].sum() / history_months), 2)

    # Compute UPI Inflow Volatility (Coefficient of Variation of weekly inflows)
    if len(upi_credits) > 0 and history_days >= 14:
        credit_dates = pd.to_datetime(upi_credits["transaction_date"])
        # Group into 7-day relative periods from statement start
        days_from_start = (credit_dates - pd.to_datetime(history_report.start_date)).dt.days
        week_buckets = days_from_start // 7
        weekly_inflows = upi_credits.groupby(week_buckets)["amount"].sum()

        # Ensure we evaluate over all passed weeks including zero-inflow weeks
        num_weeks = max(int(np.ceil(history_days / 7.0)), 2)
        full_weekly_series = pd.Series(0.0, index=range(num_weeks))
        full_weekly_series.update(pd.Series(weekly_inflows))

        mean_w = full_weekly_series.mean()
        if mean_w > 0.0:
            upi_vol = round(float(full_weekly_series.std(ddof=1) / mean_w), 3)
            # Bound CV between 0.0 and 1.0 matching synthetic scale
            upi_vol = float(np.clip(upi_vol, 0.0, 1.0))
        else:
            upi_vol = 0.0
    else:
        upi_vol = 0.0

    provenance_records["monthly_upi_transaction_count"] = FeatureProvenanceRecord(
        feature_name="monthly_upi_transaction_count",
        value=monthly_upi_count,
        provenance_state=ProvenanceState.DERIVED.value,
        source="statement",
        transformation=f"{upi_count} UPI events / {history_months} months",
        confidence="high",
        is_imputed=False,
    )
    provenance_records["monthly_upi_inflow_avg"] = FeatureProvenanceRecord(
        feature_name="monthly_upi_inflow_avg",
        value=monthly_upi_inflow,
        provenance_state=ProvenanceState.DERIVED.value,
        source="statement",
        transformation=f"Total UPI credits INR {upi_credits['amount'].sum():.2f} / {history_months} months",
        confidence="high",
        is_imputed=False,
    )
    provenance_records["monthly_upi_outflow_avg"] = FeatureProvenanceRecord(
        feature_name="monthly_upi_outflow_avg",
        value=monthly_upi_outflow,
        provenance_state=ProvenanceState.DERIVED.value,
        source="statement",
        transformation=f"Total UPI debits INR {upi_debits['amount'].sum():.2f} / {history_months} months",
        confidence="high",
        is_imputed=False,
    )
    provenance_records["upi_inflow_volatility_coefficient"] = FeatureProvenanceRecord(
        feature_name="upi_inflow_volatility_coefficient",
        value=upi_vol,
        provenance_state=ProvenanceState.DERIVED.value,
        source="statement",
        transformation="Weekly credit inflow coefficient of variation (std/mean)",
        confidence="medium" if history_days >= 30 else "low",
        is_imputed=False,
    )

    # -------------------------------------------------------------------------
    # 7. P2P VS MERCHANT RATIO
    # -------------------------------------------------------------------------
    # Filter valid P2P and Merchant volumes (exclude reversals and internal transfers)
    is_internal = (
        tx_df["is_internal_transfer"]
        | (tx_df["normalized_category"] == NormalizedCategory.INTERNAL_TRANSFER.value)
    )
    if manual_inputs.account_holder_name:
        name_clean = manual_inputs.account_holder_name.upper()
        self_xfer = tx_df["counterparty"].astype(str).str.upper().str.contains(name_clean, na=False)
        is_internal = is_internal | self_xfer

    p2p_mask = (
        tx_df["normalized_category"].isin([
            NormalizedCategory.PERSON_LIKE.value,
            NormalizedCategory.TRANSFER.value,
        ])
        & (~is_internal)
        & (~tx_df["is_reversal"])
    )
    merchant_mask = (
        tx_df["normalized_category"].isin([
            NormalizedCategory.MERCHANT_SPEND.value,
            NormalizedCategory.BUSINESS_INFLOW.value,
            NormalizedCategory.TELECOM_RECHARGE.value,
            NormalizedCategory.UTILITY.value,
        ])
        & (~tx_df["is_reversal"])
    )

    p2p_vol = float(cast(Any, tx_df[p2p_mask]["amount"].sum()))
    merchant_vol = float(cast(Any, tx_df[merchant_mask]["amount"].sum()))

    if p2p_vol == 0.0 and merchant_vol == 0.0:
        p2p_ratio = 1.0  # Neutral baseline
        ratio_conf = "low"
    else:
        p2p_ratio = round(float(p2p_vol / max(merchant_vol, 1.0)), 2)
        p2p_ratio = float(np.clip(p2p_ratio, 0.0, 10.0))
        ratio_conf = "high" if merchant_vol > 0.0 else "medium"

    provenance_records["p2p_vs_merchant_txn_ratio"] = FeatureProvenanceRecord(
        feature_name="p2p_vs_merchant_txn_ratio",
        value=p2p_ratio,
        provenance_state=ProvenanceState.DERIVED.value,
        source="statement",
        transformation=f"P2P Volume (INR {p2p_vol:.2f}) / Merchant Volume (INR {merchant_vol:.2f})",
        confidence=ratio_conf,
        is_imputed=False,
    )

    # -------------------------------------------------------------------------
    # 8. SELF-REPORTED DEMOGRAPHIC FEATURES
    # -------------------------------------------------------------------------
    borrower_id_val = manual_inputs.borrower_id or f"real_{uuid.uuid4().hex[:10]}"

    provenance_records["borrower_id"] = FeatureProvenanceRecord(
        feature_name="borrower_id",
        value=borrower_id_val,
        provenance_state=ProvenanceState.DERIVED.value,
        source="session",
        transformation="Unique session borrower identifier",
        confidence="high",
        is_imputed=False,
    )
    provenance_records["age"] = FeatureProvenanceRecord(
        feature_name="age",
        value=int(manual_inputs.age),
        provenance_state=ProvenanceState.SELF_REPORTED.value,
        source="user_input",
        transformation="Direct user manual form selection",
        confidence="high",
        is_imputed=False,
    )
    provenance_records["occupation_type"] = FeatureProvenanceRecord(
        feature_name="occupation_type",
        value=str(manual_inputs.occupation_type),
        provenance_state=ProvenanceState.SELF_REPORTED.value,
        source="user_input",
        transformation="Direct user manual form selection",
        confidence="high",
        is_imputed=False,
    )
    provenance_records["city_tier"] = FeatureProvenanceRecord(
        feature_name="city_tier",
        value=str(manual_inputs.city_tier),
        provenance_state=ProvenanceState.SELF_REPORTED.value,
        source="user_input",
        transformation="Direct user manual form selection",
        confidence="high",
        is_imputed=False,
    )

    # -------------------------------------------------------------------------
    # 9. UNAVAILABLE FEATURES (PERMANENT NULLS FOR PIPELINE MEDIAN IMPUTATION)
    # -------------------------------------------------------------------------
    unavailable_reasons = {
        "electricity_bill_ontime_rate": "Utility due dates are absent from banking transaction records.",
        "electricity_bill_avg_delay_days": "Payment delay duration requires bill due-date evidence from DISCOM.",
        "days_since_last_recharge_lapse": "Telecom balance exhaustion/lapse telemetry is not visible on bank statements.",
        "avg_weekly_gig_hours": "Weekly platform login hours reside strictly in gig aggregator driver portals.",
        "gig_platform_rating": "Customer feedback ratings reside strictly in gig aggregator platforms.",
        "active_weeks_last_6_months": "Platform order completion activity cannot be proven from generic statements.",
        "earnings_coefficient_of_variation": "Specific platform payout pacing is absent from generic non-gig ledgers.",
        "phone_number_tenure_months": "SIM subscriber vintage is telecom KYC data, absent from bank accounts.",
        "app_account_age_months": "Ecosystem app registration vintage is external metadata.",
    }

    for col_name in UNAVAILABLE_MODEL_COLUMNS:
        reason = unavailable_reasons.get(col_name, "Signal not supportable from transaction statement evidence.")
        provenance_records[col_name] = FeatureProvenanceRecord(
            feature_name=col_name,
            value=np.nan,
            provenance_state=ProvenanceState.UNAVAILABLE.value,
            source="synthetic_imputation",
            transformation="Preserved as NaN; filled by FeaturePipeline fitted median imputer",
            confidence="neutral",
            is_imputed=True,
            fallback_reason=reason,
        )

    # -------------------------------------------------------------------------
    # 10. ASSEMBLE MODEL-COMPATIBLE ROW DATAFRAME
    # -------------------------------------------------------------------------
    row_data: Dict[str, Any] = {
        col: provenance_records[col].value for col in RAW_BORROWER_COLUMNS
    }
    borrower_df: pd.DataFrame = pd.DataFrame([row_data])

    # 11. Enforce Feature Validation Contract Before Handoff
    validate_borrower_row(borrower_df, provenance_records)

    # 12. Compile Evidence Coverage Counts
    evidence_counts = {
        "total_model_features": len(RAW_BORROWER_COLUMNS),
        "derived_feature_count": sum(1 for r in provenance_records.values() if r.provenance_state == ProvenanceState.DERIVED.value),
        "self_reported_feature_count": sum(1 for r in provenance_records.values() if r.provenance_state == ProvenanceState.SELF_REPORTED.value),
        "unavailable_feature_count": sum(1 for r in provenance_records.values() if r.provenance_state == ProvenanceState.UNAVAILABLE.value),
        "imputed_feature_count": sum(1 for r in provenance_records.values() if r.is_imputed),
    }

    return RealBorrowerPayload(
        borrower_df=borrower_df,
        provenance_records=provenance_records,
        history_report=history_report,
        evidence_counts=evidence_counts,
    )
