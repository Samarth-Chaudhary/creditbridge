"""
CreditBridge - Real Data Mode: Data Quality & Evidence Diagnostics Test Suite
Path: tests/test_data_quality.py

Tests all mandated quality metrics, boundary conditions, evidence coverage formulas,
and distribution diagnostics from Part 6 specification (Sections 17 & 21).
"""

import io
import datetime
import pytest
import pandas as pd
import numpy as np

from src.real_data_contracts import (
    RAW_BORROWER_COLUMNS,
    UNAVAILABLE_MODEL_COLUMNS,
    ManualInputContract,
    FeatureProvenanceRecord,
    ProvenanceState,
    SufficiencyTier,
    NormalizedCategory,
    TransactionType,
    DistributionStatus,
    AssessmentQualityStatus,
    RealDataQualityReport,
    FieldQualityDiagnostic,
)
from src.real_data_parser import parse_csv_statement
from src.real_data_features import build_real_borrower_payload
from src.real_data_quality import (
    load_reference_distributions,
    evaluate_field_distribution,
    build_data_quality_report,
)


# -----------------------------------------------------------------------------
# FIXTURES
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_manual_inputs():
    return ManualInputContract(
        age=32,
        occupation_type="gig_delivery",
        city_tier="tier_1",
        account_holder_name="Arjun Patel",
        borrower_id="audit_borrower_101",
    )


@pytest.fixture
def clean_six_month_csv():
    rows = [
        "Date,Transaction Amount,Type,Counterparty,Category",
        "2024-01-02,28000.00,Credit,UPI-Zomato Payout,Salary",
        "2024-01-05,299.00,Debit,Jio Prepaid Recharge,Telecom",
        "2024-01-10,1200.00,Debit,BESCOM Electricity,Utility",
        "2024-01-15,450.00,Debit,Swiggy Bangalore,Food",
        "2024-01-20,1000.00,Credit,Transfer from Priya,P2P",
        "2024-02-02,29000.00,Credit,UPI-Zomato Payout,Salary",
        "2024-02-05,299.00,Debit,Jio Prepaid Recharge,Telecom",
        "2024-02-12,1150.00,Debit,BESCOM Electricity,Utility",
        "2024-02-25,350.00,Debit,Blinkit Grocery,Groceries",
        "2024-03-02,27500.00,Credit,UPI-Zomato Payout,Salary",
        "2024-03-05,299.00,Debit,Jio Prepaid Recharge,Telecom",
        "2024-03-12,1300.00,Debit,BESCOM Electricity,Utility",
        "2024-04-02,28500.00,Credit,UPI-Zomato Payout,Salary",
        "2024-04-05,299.00,Debit,Jio Prepaid Recharge,Telecom",
        "2024-04-12,1250.00,Debit,BESCOM Electricity,Utility",
        "2024-05-02,30000.00,Credit,UPI-Zomato Payout,Salary",
        "2024-05-05,299.00,Debit,Jio Prepaid Recharge,Telecom",
        "2024-05-12,1100.00,Debit,BESCOM Electricity,Utility",
        "2024-06-02,29500.00,Credit,UPI-Zomato Payout,Salary",
        "2024-06-05,299.00,Debit,Jio Prepaid Recharge,Telecom",
        "2024-06-12,1400.00,Debit,BESCOM Electricity,Utility",
        "2024-07-05,500.00,Debit,Dmart Retail,Groceries",
    ]
    return "\n".join(rows)


# -----------------------------------------------------------------------------
# 1. PARSE & TRANSACTION QUALITY TESTS
# -----------------------------------------------------------------------------

def test_parse_and_transaction_quality_metrics(sample_manual_inputs):
    """Verifies rows received, accepted, rejected, and duplicate rates in quality report."""
    csv_with_errors = """Date,Amount,Type,Counterparty
2024-01-01,500.00,Debit,Swiggy
2024-01-02,600.00,Credit,Salary
2024-01-02,600.00,Credit,Salary
INVALID-DATE,700.00,Debit,Food
2024-01-04,NOT_A_NUM,Debit,Telecom
"""
    parsed = parse_csv_statement(csv_with_errors)
    payload = build_real_borrower_payload(parsed, sample_manual_inputs, allow_insufficient_history=True)
    report = build_data_quality_report(parsed, payload)

    assert report.rows_received == 5
    assert report.rows_accepted == 3
    assert report.rows_rejected == 2
    assert report.duplicate_count == 1
    assert report.duplicate_rate == 0.20
    assert report.invalid_date_count >= 1
    assert report.missing_amount_count >= 1


def test_classification_coverage_and_sparsity_detection(sample_manual_inputs):
    """Verifies that high unknown-category share triggers sparse classification warning."""
    mostly_unknown_csv = """Date,Amount,Type,Description
2024-01-01,100.00,Debit,RANDOM NARRATION ABC 123
2024-01-05,200.00,Debit,XYZ UNSPECIFIED PAYMENT
2024-01-10,300.00,Debit,UNRECOGNIZED WIRE
2024-01-15,400.00,Debit,BESCOM ELECTRICITY
2024-01-20,500.00,Credit,RANDOM INFLOW 999
"""
    parsed = parse_csv_statement(mostly_unknown_csv)
    payload = build_real_borrower_payload(parsed, sample_manual_inputs, allow_insufficient_history=True)
    report = build_data_quality_report(parsed, payload)

    # 4 out of 5 transactions are unknown -> 80% unknown share
    assert report.unknown_category_share >= 0.80
    assert report.is_classification_sparse is True
    assert any("High unknown transaction share" in w for w in report.quality_warnings)


# -----------------------------------------------------------------------------
# 2. EVIDENCE COVERAGE & COMPOSITION TESTS
# -----------------------------------------------------------------------------

def test_feature_evidence_composition_and_denominators(clean_six_month_csv, sample_manual_inputs):
    """
    Verifies that feature evidence coverage strictly evaluates the 21 model features:
    10 derived + 3 self-reported = 13 / 21 = 61.9%
    9 unavailable/imputed = 9 / 21 = 42.9%
    """
    parsed = parse_csv_statement(clean_six_month_csv)
    payload = build_real_borrower_payload(parsed, sample_manual_inputs)
    report = build_data_quality_report(parsed, payload)

    assert report.total_model_features == 21
    assert report.derived_count == 9
    assert report.self_reported_count == 3
    assert report.unavailable_count == 9
    assert report.imputed_count == 9

    expected_coverage = round(12 / 21.0, 4)
    assert report.evidence_coverage_ratio == expected_coverage
    assert report.derived_coverage_ratio == round(9 / 21.0, 4)
    assert report.self_reported_coverage_ratio == round(3 / 21.0, 4)
    assert report.imputed_ratio == round(9 / 21.0, 4)


def test_imputed_vs_unavailable_distinction(clean_six_month_csv, sample_manual_inputs):
    """Verifies that unavailable and imputed statuses are distinguished and not collapsed."""
    parsed = parse_csv_statement(clean_six_month_csv)
    payload = build_real_borrower_payload(parsed, sample_manual_inputs)
    report = build_data_quality_report(parsed, payload)

    for diag in report.field_diagnostics:
        if diag.feature_name in UNAVAILABLE_MODEL_COLUMNS:
            assert diag.provenance_state == ProvenanceState.UNAVAILABLE.value
            assert diag.distribution_status == DistributionStatus.NOT_ASSESSABLE.value


# -----------------------------------------------------------------------------
# 3. DISTRIBUTION DIAGNOSTICS & BOUNDARY TESTS
# -----------------------------------------------------------------------------

def test_distribution_diagnostics_boundaries():
    """
    Verifies exact boundary logic against synthetic training percentiles:
    - Within range: [p5, p95]
    - Near boundary: [min, p5) or (p95, max]
    - Outside observed range: < min or > max
    """
    ref = load_reference_distributions()
    stat = ref["numeric_features"]["monthly_income_estimate"]
    p_min = stat["min"]
    p5 = stat["p5"]
    p95 = stat["p95"]
    p_max = stat["max"]

    # 1. Comfortable median value -> WITHIN_RANGE
    diag_median = evaluate_field_distribution("monthly_income_estimate", stat["median"], "DERIVED", ref)
    assert diag_median.distribution_status == DistributionStatus.WITHIN_RANGE.value
    assert diag_median.warning_message is None

    # 2. Exact p5 and p95 boundary points -> WITHIN_RANGE
    diag_p5 = evaluate_field_distribution("monthly_income_estimate", p5, "DERIVED", ref)
    assert diag_p5.distribution_status == DistributionStatus.WITHIN_RANGE.value

    diag_p95 = evaluate_field_distribution("monthly_income_estimate", p95, "DERIVED", ref)
    assert diag_p95.distribution_status == DistributionStatus.WITHIN_RANGE.value

    # 3. Just below p5 -> NEAR_BOUNDARY
    diag_below_p5 = evaluate_field_distribution("monthly_income_estimate", p5 - 1.0, "DERIVED", ref)
    assert diag_below_p5.distribution_status == DistributionStatus.NEAR_BOUNDARY.value
    assert "near distribution boundary" in str(diag_below_p5.warning_message)

    # 4. Just above p95 -> NEAR_BOUNDARY
    diag_above_p95 = evaluate_field_distribution("monthly_income_estimate", p95 + 1.0, "DERIVED", ref)
    assert diag_above_p95.distribution_status == DistributionStatus.NEAR_BOUNDARY.value
    assert "near distribution boundary" in str(diag_above_p95.warning_message)

    # 5. Just below min -> OUTSIDE_OBSERVED_RANGE
    diag_below_min = evaluate_field_distribution("monthly_income_estimate", p_min - 100.0, "DERIVED", ref)
    assert diag_below_min.distribution_status == DistributionStatus.OUTSIDE_OBSERVED_RANGE.value
    assert "outside observed synthetic development range" in str(diag_below_min.warning_message)

    # 6. Just above max -> OUTSIDE_OBSERVED_RANGE
    diag_above_max = evaluate_field_distribution("monthly_income_estimate", p_max + 1000.0, "DERIVED", ref)
    assert diag_above_max.distribution_status == DistributionStatus.OUTSIDE_OBSERVED_RANGE.value
    assert "outside observed synthetic development range" in str(diag_above_max.warning_message)


def test_categorical_distribution_diagnostics():
    """Verifies detection of known training categories versus unseen categories."""
    ref = load_reference_distributions()

    # Known category
    diag_known = evaluate_field_distribution("occupation_type", "gig_delivery", "SELF_REPORTED", ref)
    assert diag_known.distribution_status == DistributionStatus.KNOWN_IN_TRAINING.value
    assert diag_known.warning_message is None

    # Unseen category
    diag_unseen = evaluate_field_distribution("occupation_type", "astronaut", "SELF_REPORTED", ref)
    assert diag_unseen.distribution_status == DistributionStatus.UNSEEN_IN_TRAINING.value
    assert "unseen in synthetic development baseline" in str(diag_unseen.warning_message)


# -----------------------------------------------------------------------------
# 4. OVERALL ASSESSMENT STATUS & FAILURE MODES
# -----------------------------------------------------------------------------

def test_overall_assessment_status_sufficient(clean_six_month_csv, sample_manual_inputs):
    """Verifies that high-quality 6-month statement produces SUFFICIENT status."""
    parsed = parse_csv_statement(clean_six_month_csv)
    payload = build_real_borrower_payload(parsed, sample_manual_inputs)
    report = build_data_quality_report(parsed, payload)

    assert report.history_tier == SufficiencyTier.OPTIMAL.value
    assert report.overall_status == AssessmentQualityStatus.SUFFICIENT.value
    assert report.is_eligible_for_scoring is True
    assert len(report.blocker_reasons) == 0


def test_overall_assessment_status_limited_for_marginal_history(sample_manual_inputs):
    """Verifies that marginal 45-day history yields LIMITED status."""
    # 45 days statement
    dates = pd.date_range("2024-01-01", "2024-02-15", periods=20)
    rows = ["Date,Amount,Type,Counterparty"]
    for d in dates:
        rows.append(f"{d.strftime('%Y-%m-%d')},500.00,Debit,Swiggy Bangalore")
    csv_content = "\n".join(rows)

    parsed = parse_csv_statement(csv_content)
    payload = build_real_borrower_payload(parsed, sample_manual_inputs)
    report = build_data_quality_report(parsed, payload)

    assert report.history_tier == SufficiencyTier.MARGINAL.value
    assert report.overall_status == AssessmentQualityStatus.LIMITED.value
    assert report.is_eligible_for_scoring is True
    assert any("History is limited" in w for w in report.quality_warnings)


def test_overall_assessment_status_insufficient_blocks_scoring(sample_manual_inputs):
    """Verifies that insufficient history (<30 days) sets status INSUFFICIENT and blocks scoring."""
    short_csv = """Date,Amount,Type,Counterparty
2024-01-01,1000.00,Debit,Airtel Recharge
2024-01-05,500.00,Debit,BESCOM Electricity
"""
    parsed = parse_csv_statement(short_csv)
    payload = build_real_borrower_payload(parsed, sample_manual_inputs, allow_insufficient_history=True)
    report = build_data_quality_report(parsed, payload)

    assert report.history_tier == SufficiencyTier.INSUFFICIENT.value
    assert report.overall_status == AssessmentQualityStatus.INSUFFICIENT.value
    assert report.is_eligible_for_scoring is False
    assert len(report.blocker_reasons) > 0


# -----------------------------------------------------------------------------
# 5. DETERMINISM & IMMUTABILITY TESTS
# -----------------------------------------------------------------------------

def test_deterministic_report_generation(clean_six_month_csv, sample_manual_inputs):
    """Verifies that repeated report generation yields identical numbers without drift."""
    parsed = parse_csv_statement(clean_six_month_csv)
    payload = build_real_borrower_payload(parsed, sample_manual_inputs)

    rep1 = build_data_quality_report(parsed, payload)
    rep2 = build_data_quality_report(parsed, payload)

    assert rep1.to_dict() == rep2.to_dict()


def test_immutability_of_inputs(clean_six_month_csv, sample_manual_inputs):
    """Verifies quality engine does not mutate transactions DataFrame or borrower row."""
    parsed = parse_csv_statement(clean_six_month_csv)
    payload = build_real_borrower_payload(parsed, sample_manual_inputs)

    tx_orig = parsed.transactions.copy()
    b_orig = payload.borrower_df.copy()

    _ = build_data_quality_report(parsed, payload)

    pd.testing.assert_frame_equal(parsed.transactions, tx_orig)
    pd.testing.assert_frame_equal(payload.borrower_df, b_orig)


def test_serialized_report_dictionary(clean_six_month_csv, sample_manual_inputs):
    """Verifies report serializes cleanly to JSON-compatible dictionary."""
    parsed = parse_csv_statement(clean_six_month_csv)
    payload = build_real_borrower_payload(parsed, sample_manual_inputs)
    report = build_data_quality_report(parsed, payload)

    d = report.to_dict()
    assert isinstance(d, dict)
    assert "field_diagnostics" in d
    assert len(d["field_diagnostics"]) == 21
    assert d["overall_status"] == "SUFFICIENT"
    assert d["evidence_coverage_ratio"] > 0.50
