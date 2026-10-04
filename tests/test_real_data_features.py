"""
CreditBridge - Real Data Mode: Comprehensive Feature Mapping & Transaction Intelligence Test Suite
Path: tests/test_real_data_features.py

Tests all 15 mandated testing requirements from Part 5 specification (Sections 20 & 26).
"""

import numpy as np
import pandas as pd
import pytest

from src.feature_provenance import summarize_provenance
from src.real_data_contracts import (
    RAW_BORROWER_COLUMNS,
    UNAVAILABLE_MODEL_COLUMNS,
    FeatureProvenanceRecord,
    InsufficientHistoryError,
    InvalidDataError,
    ManualInputContract,
    ManualInputContractError,
    ModelInputContractError,
    NormalizedCategory,
    ProvenanceState,
    SufficiencyTier,
)
from src.real_data_features import (
    build_real_borrower_payload,
    evaluate_history_sufficiency,
)
from src.real_data_parser import parse_csv_statement
from src.real_data_validation import validate_borrower_row
from src.scoring_utils import load_model_bundle, score_borrower
from src.transaction_classifier import (
    classify_transaction,
)

# -----------------------------------------------------------------------------
# FIXTURES
# -----------------------------------------------------------------------------

@pytest.fixture
def valid_manual_inputs():
    return ManualInputContract(
        age=28,
        occupation_type="gig_delivery",
        city_tier="tier_1",
        account_holder_name="Rahul Sharma",
        borrower_id="test_borrower_001",
    )


@pytest.fixture
def multi_month_statement_csv():
    """Generates a realistic 6-month statement CSV (180 days, ~30 transactions)."""
    rows = [
        "Date,Transaction Amount,Type,Counterparty,Category",
        # Month 1 (Nov 2023)
        "2023-11-01,25000.00,Credit,UPI-Zomato Payout,Salary",
        "2023-11-03,299.00,Debit,Jio Prepaid Recharge,Telecom",
        "2023-11-05,1200.00,Debit,BESCOM Electricity,Utility",
        "2023-11-10,450.00,Debit,Swiggy Order,Food",
        "2023-11-15,1000.00,Credit,Transfer from Priya,P2P",
        # Month 2 (Dec 2023)
        "2023-12-01,26000.00,Credit,UPI-Zomato Payout,Salary",
        "2023-12-04,299.00,Debit,Jio Prepaid Recharge,Telecom",
        "2023-12-08,1150.00,Debit,BESCOM Electricity,Utility",
        "2023-12-15,350.00,Debit,Blinkit Grocery,Groceries",
        # Month 3 (Jan 2024)
        "2024-01-02,24500.00,Credit,UPI-Zomato Payout,Salary",
        "2024-01-05,299.00,Debit,Jio Prepaid Recharge,Telecom",
        "2024-01-10,1300.00,Debit,BESCOM Electricity,Utility",
        "2024-01-18,500.00,Debit,Amazon India,Shopping",
        # Month 4 (Feb 2024)
        "2024-02-01,27000.00,Credit,UPI-Zomato Payout,Salary",
        "2024-02-06,299.00,Debit,Jio Prepaid Recharge,Telecom",
        "2024-02-12,1250.00,Debit,BESCOM Electricity,Utility",
        "2024-02-20,200.00,Credit,Refund for order,Refund",
        # Month 5 (Mar 2024)
        "2024-03-01,25500.00,Credit,UPI-Zomato Payout,Salary",
        "2024-03-05,299.00,Debit,Jio Prepaid Recharge,Telecom",
        "2024-03-10,1100.00,Debit,BESCOM Electricity,Utility",
        "2024-03-22,5000.00,Credit,Transfer to own A/C Rahul Sharma,Transfer",  # Self transfer
        # Month 6 (Apr 2024)
        "2024-04-01,28000.00,Credit,UPI-Zomato Payout,Salary",
        "2024-04-05,299.00,Debit,Jio Prepaid Recharge,Telecom",
        "2024-04-12,1400.00,Debit,BESCOM Electricity,Utility",
        "2024-04-28,10000.00,Credit,KreditBee Loan Disbursal,Loan",  # Loan credit
        "2024-04-30,500.00,Debit,Dmart Retail Store,Groceries",
    ]
    return "\n".join(rows)


# -----------------------------------------------------------------------------
# 1. TRANSACTION CLASSIFIER TESTS
# -----------------------------------------------------------------------------

def test_classification_rules_known_examples():
    """Verifies deterministic classification of known Indian banking entities."""
    examples = [
        ("CMS/ CORPORATE SALARY CR", "", 50000.0, "credit", NormalizedCategory.SALARY_LIKE.value),
        ("UPI-Zomato Driver Payout", "", 12000.0, "credit", NormalizedCategory.GIG_INCOME_LIKE.value),
        ("Swiggy Delivery Payout", "", 8500.0, "credit", NormalizedCategory.GIG_INCOME_LIKE.value),
        ("BHARATPE QR SETTLEMENT", "", 4500.0, "credit", NormalizedCategory.BUSINESS_INFLOW.value),
        ("JIO PREPAID RECHARGE 299", "", 299.0, "debit", NormalizedCategory.TELECOM_RECHARGE.value),
        ("AIRTEL 4G RECHARGE", "", 479.0, "debit", NormalizedCategory.TELECOM_RECHARGE.value),
        ("BESCOM BANGALORE ELEC BILL", "", 1450.0, "debit", NormalizedCategory.UTILITY.value),
        ("TATA POWER MUMBAI", "", 2100.0, "debit", NormalizedCategory.UTILITY.value),
        ("AMAZON INDIA SELLER", "", 1299.0, "debit", NormalizedCategory.MERCHANT_SPEND.value),
        ("FLIPKART INTERNET PVT LTD", "", 899.0, "debit", NormalizedCategory.MERCHANT_SPEND.value),
        ("KREDITBEE LOAN DISBURSAL", "", 25000.0, "credit", NormalizedCategory.LOAN_RELATED.value),
        ("NAVI FINSERV EMI DEBIT", "", 3200.0, "debit", NormalizedCategory.LOAN_RELATED.value),
        ("AUTOSWEEP TO FIXED DEPOSIT", "", 15000.0, "debit", NormalizedCategory.TRANSFER.value),
        ("REFUND FOR CANCELLED TRIP", "", 350.0, "credit", NormalizedCategory.REFUND.value),
        ("REV-UPI FAILED REVERSAL", "", 500.0, "credit", NormalizedCategory.REFUND.value),
    ]

    for narr, cparty, amt, ttype, expected_cat in examples:
        res = classify_transaction(narration=narr, counterparty=cparty, amount=amt, txn_type=ttype)
        assert res.category == expected_cat, f"Failed for {narr}: got {res.category}, expected {expected_cat}"
        assert res.confidence > 0.70
        assert res.rule_name.startswith("RULE_")


def test_unknown_transaction_handling():
    """Ensures unrecognized narrations safely map to UNKNOWN with confidence 0.0."""
    res = classify_transaction(
        narration="TXN REF 987123456789 RANDOM STRING ABC",
        counterparty="",
        amount=150.0,
        txn_type="debit",
    )
    assert res.category == NormalizedCategory.UNKNOWN.value
    assert res.confidence == 0.0
    assert res.rule_name == "RULE_UNKNOWN_NO_MATCH"


def test_self_transfer_detection_by_account_holder_name():
    """Ensures self-transfers matching user's name are classified as TRANSFER."""
    res = classify_transaction(
        narration="UPI-TRF TO RAHUL SHARMA",
        counterparty="Rahul Sharma",
        amount=10000.0,
        txn_type="credit",
        account_holder_name="Rahul Sharma",
    )
    assert res.category == NormalizedCategory.TRANSFER.value
    assert "SELF" in res.rule_name or "INTERNAL" in res.rule_name


# -----------------------------------------------------------------------------
# 2. HISTORY SUFFICIENCY TESTS
# -----------------------------------------------------------------------------

def test_history_sufficiency_tiers():
    """Verifies underwriting policy tiers: <30d block, 30-89d marginal, 90-179d adequate, 180d+ optimal."""
    # < 30 days
    dates_short = pd.Series(pd.date_range("2024-01-01", "2024-01-15", freq="D"))
    rep_short = evaluate_history_sufficiency(dates_short, len(dates_short))
    assert rep_short.sufficiency_tier == SufficiencyTier.INSUFFICIENT.value
    assert not rep_short.is_scoreable
    assert rep_short.underwriting_action == "BLOCK_SCORING"

    # 45 days
    dates_marginal = pd.Series(pd.date_range("2024-01-01", "2024-02-15", freq="D"))
    rep_marginal = evaluate_history_sufficiency(dates_marginal, len(dates_marginal))
    assert rep_marginal.sufficiency_tier == SufficiencyTier.MARGINAL.value
    assert rep_marginal.is_scoreable
    assert rep_marginal.underwriting_action == "ALLOW_WITH_MANUAL_REVIEW"

    # 120 days
    dates_adequate = pd.Series(pd.date_range("2024-01-01", "2024-05-01", freq="D"))
    rep_adequate = evaluate_history_sufficiency(dates_adequate, len(dates_adequate))
    assert rep_adequate.sufficiency_tier == SufficiencyTier.ADEQUATE.value
    assert rep_adequate.is_scoreable
    assert rep_adequate.underwriting_action == "ALLOW_SCORING"

    # 185 days
    dates_optimal = pd.Series(pd.date_range("2024-01-01", "2024-07-05", freq="D"))
    rep_optimal = evaluate_history_sufficiency(dates_optimal, len(dates_optimal))
    assert rep_optimal.sufficiency_tier == SufficiencyTier.OPTIMAL.value
    assert rep_optimal.is_scoreable
    assert rep_optimal.underwriting_action == "ALLOW_SCORING"


def test_insufficient_history_blocks_scoring_by_default(valid_manual_inputs):
    """Verifies build_real_borrower_payload raises InsufficientHistoryError when <30 days."""
    short_csv = """Date,Amount,Type,Counterparty
2024-01-01,500.00,Credit,Salary
2024-01-05,200.00,Debit,Food
"""
    parsed = parse_csv_statement(short_csv)
    with pytest.raises(InsufficientHistoryError):
        build_real_borrower_payload(parsed, valid_manual_inputs)

    # Diagnostic override flag allows feature calculation
    payload = build_real_borrower_payload(parsed, valid_manual_inputs, allow_insufficient_history=True)
    assert payload.borrower_df is not None


# -----------------------------------------------------------------------------
# 3. INCOME CLASSIFICATION HIERARCHY TESTS
# -----------------------------------------------------------------------------

def test_income_derivation_excludes_loans_refunds_and_self_transfers(multi_month_statement_csv, valid_manual_inputs):
    """
    Verifies that income logic strictly discounts:
    - KreditBee loan credit (INR 10,000) excluded
    - Self transfer to Rahul Sharma (INR 5,000) excluded
    - Order refund (INR 200) excluded
    - Salary credits (6 x ~26,000) included at 100%
    - P2P transfer (INR 1,000) included at 50% (INR 500)
    """
    parsed = parse_csv_statement(multi_month_statement_csv, user_name=valid_manual_inputs.account_holder_name)
    payload = build_real_borrower_payload(parsed, valid_manual_inputs)

    row = payload.borrower_df.iloc[0]
    income_val = row["monthly_income_estimate"]

    # 6 months, ~156,000 salary + 500 P2P = ~156,500 / 6 = ~26,000 / month
    # Must NOT include the 10,000 loan or 5,000 self-transfer
    assert 24000.0 <= income_val <= 28000.0

    income_prov = payload.provenance_records["monthly_income_estimate"]
    assert income_prov.provenance_state == ProvenanceState.DERIVED.value
    assert income_prov.source == "statement"
    assert "Tier A 100%" in income_prov.transformation


# -----------------------------------------------------------------------------
# 4. TELECOM RECHARGE FEATURE TESTS
# -----------------------------------------------------------------------------

def test_recharge_feature_derivation(multi_month_statement_csv, valid_manual_inputs):
    """Verifies recharge frequency, average ticket, and amount volatility calculation."""
    parsed = parse_csv_statement(multi_month_statement_csv)
    payload = build_real_borrower_payload(parsed, valid_manual_inputs)

    row = payload.borrower_df.iloc[0]
    freq = row["recharge_frequency_per_month"]
    avg_amt = row["avg_recharge_amount"]
    vol = row["recharge_amount_volatility"]

    # 6 recharges of INR 299 over 6 months -> ~1.0/mo, avg 299, vol 0.0
    assert 0.8 <= freq <= 1.2
    assert avg_amt == 299.0
    assert vol == 0.0  # Constant recharge amount has 0 std dev

    # Check days_since_last_recharge_lapse is UNAVAILABLE (np.nan)
    assert pd.isna(row["days_since_last_recharge_lapse"])


# -----------------------------------------------------------------------------
# 5. UTILITY PAYMENT & UNAVAILABLE FEATURES TESTS
# -----------------------------------------------------------------------------

def test_utility_punctuality_strictly_unavailable(multi_month_statement_csv, valid_manual_inputs):
    """
    HARD STOP VERIFICATION:
    Verifies that electricity_bill_ontime_rate and electricity_bill_avg_delay_days
    are NEVER fabricated from transaction dates alone and remain np.nan.
    """
    parsed = parse_csv_statement(multi_month_statement_csv)
    payload = build_real_borrower_payload(parsed, valid_manual_inputs)

    row = payload.borrower_df.iloc[0]
    assert pd.isna(row["electricity_bill_ontime_rate"])
    assert pd.isna(row["electricity_bill_avg_delay_days"])

    prov_ontime = payload.provenance_records["electricity_bill_ontime_rate"]
    prov_delay = payload.provenance_records["electricity_bill_avg_delay_days"]

    assert prov_ontime.provenance_state == ProvenanceState.UNAVAILABLE.value
    assert prov_ontime.is_imputed is True
    assert "due date" in (prov_ontime.fallback_reason or "").lower()

    assert prov_delay.provenance_state == ProvenanceState.UNAVAILABLE.value
    assert prov_delay.is_imputed is True


def test_all_unavailable_features_are_nan(multi_month_statement_csv, valid_manual_inputs):
    """Verifies all 9 designated unavailable features are strictly np.nan."""
    parsed = parse_csv_statement(multi_month_statement_csv)
    payload = build_real_borrower_payload(parsed, valid_manual_inputs)

    row = payload.borrower_df.iloc[0]
    for col in UNAVAILABLE_MODEL_COLUMNS:
        assert pd.isna(row[col]), f"Feature {col} should be NaN but got {row[col]}"
        prov = payload.provenance_records[col]
        assert prov.provenance_state == ProvenanceState.UNAVAILABLE.value
        assert prov.is_imputed is True


# -----------------------------------------------------------------------------
# 6. MANUAL INPUTS & DEMOGRAPHICS TESTS
# -----------------------------------------------------------------------------

def test_manual_inputs_precedence_and_validation(valid_manual_inputs):
    """Verifies applicant self-reported inputs are strictly validated and preserved."""
    assert valid_manual_inputs.age == 28
    assert valid_manual_inputs.occupation_type == "gig_delivery"
    assert valid_manual_inputs.city_tier == "tier_1"

    # Invalid age
    with pytest.raises(ManualInputContractError):
        ManualInputContract(age=15, occupation_type="gig_delivery", city_tier="tier_1").validate()

    with pytest.raises(ManualInputContractError):
        ManualInputContract(age=75, occupation_type="gig_delivery", city_tier="tier_1").validate()

    # Invalid occupation
    with pytest.raises(ManualInputContractError):
        ManualInputContract(age=30, occupation_type="astronaut", city_tier="tier_1").validate()

    # Invalid city tier
    with pytest.raises(ManualInputContractError):
        ManualInputContract(age=30, occupation_type="gig_delivery", city_tier="tier_5").validate()


# -----------------------------------------------------------------------------
# 7. SCHEMA COMPATIBILITY & PIPELINE INTEGRATION TESTS
# -----------------------------------------------------------------------------

def test_model_compatible_borrower_dataframe_structure(multi_month_statement_csv, valid_manual_inputs):
    """Verifies output DataFrame has exactly 22 columns in the exact legacy order."""
    parsed = parse_csv_statement(multi_month_statement_csv)
    payload = build_real_borrower_payload(parsed, valid_manual_inputs)

    df = payload.borrower_df
    assert len(df) == 1
    assert list(df.columns) == RAW_BORROWER_COLUMNS


def test_handoff_into_feature_pipeline_and_scoring(multi_month_statement_csv, valid_manual_inputs):
    """
    REGRESSION TEST:
    Feeds the mapped real-data borrower row directly into the existing fitted
    FeaturePipeline and verifies it transforms cleanly into 30 numerical features
    with 0 missing values, and produces a valid 300-900 credit score.
    """
    parsed = parse_csv_statement(multi_month_statement_csv)
    payload = build_real_borrower_payload(parsed, valid_manual_inputs)

    model_bundle = load_model_bundle()
    pipeline = model_bundle["pipeline"]
    model = model_bundle["model"]

    # In scikit-learn 1.4.1 runtime, ensure multi_class attribute compatibility on LogisticRegression step
    clf = model.named_steps.get("classifier", model) if hasattr(model, "named_steps") else model
    if not hasattr(clf, "multi_class"):
        clf.multi_class = "auto"

    # Transform through pipeline
    X_transformed = pipeline.transform(payload.borrower_df)
    assert X_transformed.shape == (1, 30), f"Expected shape (1, 30), got {X_transformed.shape}"
    assert not X_transformed.isna().any().any(), "Transformed feature matrix contains unexpected NaNs!"

    # Score borrower
    score, tier = score_borrower(payload.borrower_df, model_bundle)
    assert 300 <= score <= 900
    assert tier in ["Low Risk", "Moderate Risk", "High Risk", "Very High Risk"]


# -----------------------------------------------------------------------------
# 8. VALIDATION & ERROR HANDLING TESTS
# -----------------------------------------------------------------------------

def test_empty_transaction_set_rejected(valid_manual_inputs):
    """Verifies passing an empty DataFrame raises InvalidDataError."""
    empty_df = pd.DataFrame({"transaction_date": [], "amount": [], "transaction_type": []})
    with pytest.raises(InvalidDataError):
        build_real_borrower_payload(empty_df, valid_manual_inputs)


def test_invalid_numeric_ranges_rejected(valid_manual_inputs):
    """Verifies validate_borrower_row catches negative or invalid calculated values."""
    corrupted_records = {
        col: FeatureProvenanceRecord(
            feature_name=col,
            value=0.0 if col not in UNAVAILABLE_MODEL_COLUMNS else np.nan,
            provenance_state=ProvenanceState.DERIVED.value if col not in UNAVAILABLE_MODEL_COLUMNS else ProvenanceState.UNAVAILABLE.value,
            source="test",
            transformation="test",
        )
        for col in RAW_BORROWER_COLUMNS
    }
    # Corrupt monthly_income_estimate to negative
    corrupted_records["monthly_income_estimate"] = FeatureProvenanceRecord(
        feature_name="monthly_income_estimate",
        value=-500.0,
        provenance_state=ProvenanceState.DERIVED.value,
        source="test",
        transformation="test",
    )
    corrupted_records["borrower_id"] = FeatureProvenanceRecord(
        feature_name="borrower_id", value="b1", provenance_state=ProvenanceState.DERIVED.value, source="test", transformation="test"
    )
    corrupted_records["age"] = FeatureProvenanceRecord(
        feature_name="age", value=25, provenance_state=ProvenanceState.SELF_REPORTED.value, source="test", transformation="test"
    )
    corrupted_records["occupation_type"] = FeatureProvenanceRecord(
        feature_name="occupation_type", value="gig_delivery", provenance_state=ProvenanceState.SELF_REPORTED.value, source="test", transformation="test"
    )
    corrupted_records["city_tier"] = FeatureProvenanceRecord(
        feature_name="city_tier", value="tier_1", provenance_state=ProvenanceState.SELF_REPORTED.value, source="test", transformation="test"
    )

    row_data = {c: corrupted_records[c].value for c in RAW_BORROWER_COLUMNS}
    df = pd.DataFrame([row_data])

    with pytest.raises(ModelInputContractError) as excinfo:
        validate_borrower_row(df, corrupted_records)
    assert "must be >= 0.0" in str(excinfo.value)


# -----------------------------------------------------------------------------
# 9. PROVENANCE MANIFEST AND SUMMARY TESTS
# -----------------------------------------------------------------------------

def test_provenance_manifest_and_summary(multi_month_statement_csv, valid_manual_inputs):
    """Verifies provenance summary accurately counts derived, self-reported, and unavailable fields."""
    parsed = parse_csv_statement(multi_month_statement_csv)
    payload = build_real_borrower_payload(parsed, valid_manual_inputs)

    summary = summarize_provenance(payload.provenance_records)
    assert summary["total_features"] == 22
    assert summary["state_breakdown"][ProvenanceState.UNAVAILABLE.value] == 9
    assert summary["state_breakdown"][ProvenanceState.SELF_REPORTED.value] == 3
    assert summary["state_breakdown"][ProvenanceState.DERIVED.value] == 10
    assert len(summary["manifest"]) == 22
