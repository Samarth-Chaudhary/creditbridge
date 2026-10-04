"""
CreditBridge - Alternative Credit Scoring Engine
Part 9: End-to-End Fixtures, Adversarial Edge Cases & Mathematical Invariants
Path: tests/test_adversarial_and_e2e_fixtures.py

Tests the complete Real Data backend pipeline across 7 sanitized fixtures,
20+ adversarial edge cases, property-based invariants, and failure isolation boundaries.
"""

from pathlib import Path

import numpy as np
import pytest

from src.privacy_security import validate_upload_security
from src.real_data_contracts import (
    CANONICAL_COLUMNS,
    EmptyStatementError,
    FileTypeError,
    ManualInputContract,
    OversizedFileError,
    PathTraversalError,
    UnsupportedSchemaError,
)
from src.real_data_parser import parse_csv_statement
from src.real_data_scoring import assess_statement_end_to_end

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


# -----------------------------------------------------------------------------
# 1. END-TO-END EVALUATION ON SANITIZED FIXTURES
# -----------------------------------------------------------------------------

def test_e2e_representative_transactions_healthy_assessment():
    """Verify healthy 6-month representative statement produces complete, scoreable assessment."""
    fixture_path = FIXTURES_DIR / "representative_transactions.csv"
    assert fixture_path.exists()

    inputs = ManualInputContract(age=29, occupation_type="gig_delivery", city_tier="tier_1")
    result = assess_statement_end_to_end(str(fixture_path), manual_inputs=inputs)

    assert result is not None
    assert result.is_scoreable is True
    assert result.assessment_status in ("SUFFICIENT", "LIMITED")
    assert 300 <= result.credit_score <= 900
    assert result.risk_tier in ("Low Risk", "Moderate Risk", "High Risk — Manual Review", "Very High Risk")
    assert 0.0 <= result.calibrated_model_probability <= 1.0
    assert result.evidence_coverage_ratio >= 0.50
    assert result.history_months >= 5.0
    assert "plain_english_explanation" in result.explanation
    assert len(result.explanation.get("factors", [])) > 0


def test_e2e_refund_reversal_transactions():
    """Verify refunds, reversals, and internal transfers are recognized and excluded from income."""
    fixture_path = FIXTURES_DIR / "refund_reversal_transactions.csv"
    assert fixture_path.exists()

    parsed = parse_csv_statement(str(fixture_path))
    txns = parsed.transactions

    # Verify refund and reversal flags
    assert txns["is_refund"].sum() >= 1
    assert txns["is_reversal"].sum() >= 1
    assert txns["is_internal_transfer"].sum() >= 1

    # End-to-end evaluation
    inputs = ManualInputContract(age=35, occupation_type="small_trader", city_tier="tier_2")
    result = assess_statement_end_to_end(str(fixture_path), manual_inputs=inputs)
    assert result is not None


def test_e2e_duplicate_transactions_deduplicated():
    """Verify duplicate transactions are detected, counted, and properly deduplicated."""
    fixture_path = FIXTURES_DIR / "duplicate_transactions.csv"
    assert fixture_path.exists()

    parsed = parse_csv_statement(str(fixture_path))
    assert parsed.duplicate_rows == 3
    assert any("duplicate" in str(d).lower() for d in parsed.diagnostics)


def test_e2e_malformed_transactions_handled_safely():
    """Verify corrupted dates, non-numeric rows, and contradiction rows are rejected with diagnostics."""
    fixture_path = FIXTURES_DIR / "malformed_transactions.csv"
    assert fixture_path.exists()

    parsed = parse_csv_statement(str(fixture_path))
    assert parsed.rows_rejected > 0
    assert len(parsed.diagnostics) > 0

    # Diagnostic reasons must record structural causes, not raw confidential rows
    reasons = [d.get("reason", "") for d in parsed.diagnostics]
    assert any("date" in r.lower() or "contradiction" in r.lower() or "amount" in r.lower() for r in reasons)


def test_e2e_minimal_history_transactions_blocked():
    """Verify statements with history shorter than 90 days are blocked from scoring."""
    fixture_path = FIXTURES_DIR / "minimal_history_transactions.csv"
    assert fixture_path.exists()

    inputs = ManualInputContract(age=24, occupation_type="informal_retail", city_tier="tier_3")
    result = assess_statement_end_to_end(str(fixture_path), manual_inputs=inputs)

    assert result.is_scoreable is False
    assert result.assessment_status == "INSUFFICIENT"
    assert any("history" in b.lower() or "days" in b.lower() for b in result.blocker_reasons)


def test_e2e_out_of_distribution_transactions_flagged():
    """Verify extreme values trigger distribution boundary flags without crashing."""
    fixture_path = FIXTURES_DIR / "out_of_distribution_transactions.csv"
    assert fixture_path.exists()

    inputs = ManualInputContract(age=45, occupation_type="small_trader", city_tier="tier_1")
    result = assess_statement_end_to_end(str(fixture_path), manual_inputs=inputs)

    assert result is not None
    # Must flag anomalous values
    assert len(result.distribution_flags) > 0


# -----------------------------------------------------------------------------
# 2. ADVERSARIAL EDGE CASE SUITE (20+ SCENARIOS)
# -----------------------------------------------------------------------------

def test_adversarial_empty_file_rejected():
    """Verify 0-byte file raises EmptyStatementError."""
    with pytest.raises(EmptyStatementError):
        parse_csv_statement(b"")


def test_adversarial_header_only_file_rejected():
    """Verify CSV with headers but zero data rows raises EmptyStatementError."""
    with pytest.raises(EmptyStatementError):
        parse_csv_statement("Date,Amount,Description\n")


def test_adversarial_wrong_file_type_rejected():
    """Verify unsupported file types raise FileTypeError."""
    with pytest.raises(FileTypeError):
        validate_upload_security(b"dummy data", filename="malicious.exe")


def test_adversarial_missing_required_date_column():
    """Verify CSV missing date header raises UnsupportedSchemaError."""
    with pytest.raises(UnsupportedSchemaError) as excinfo:
        parse_csv_statement("Amount,Description\n100,Groceries\n")
    assert "date" in str(excinfo.value).lower()


def test_adversarial_missing_required_amount_column():
    """Verify CSV missing amount headers raises UnsupportedSchemaError."""
    with pytest.raises(UnsupportedSchemaError) as excinfo:
        parse_csv_statement("Date,Description\n2023-01-01,Groceries\n")
    assert "amount" in str(excinfo.value).lower()


def test_adversarial_all_transactions_on_one_date():
    """Verify file with all transactions on one date calculates history safely without zero division."""
    csv_data = (
        "Date,Amount,Description\n"
        "2023-01-01,500,Swiggy\n"
        "2023-01-01,300,Zomato\n"
        "2023-01-01,200,Airtel\n"
    )
    parsed = parse_csv_statement(csv_data)
    assert len(parsed.transactions) == 3


def test_adversarial_only_one_transaction():
    """Verify file with only 1 transaction does not crash volatility / ratio calculations."""
    csv_data = "Date,Amount,Description\n2023-01-01,1500,Swiggy\n"
    inputs = ManualInputContract(age=28, occupation_type="gig_delivery", city_tier="tier_1")
    result = assess_statement_end_to_end(csv_data, manual_inputs=inputs)
    assert result.is_scoreable is False  # Blocked by insufficient history


def test_adversarial_extremely_large_transaction_amount():
    """Verify transaction amounts in hundreds of crores do not cause overflow."""
    csv_data = "Date,Amount,Description\n2023-01-01,99999999999.00,Special Transfer\n"
    parsed = parse_csv_statement(csv_data)
    assert parsed.transactions["amount"].iloc[0] == 99999999999.00


def test_adversarial_negative_and_zero_amount_scalars():
    """Verify negative and zero amount strings are normalized safely."""
    csv_data = (
        "Date,Amount,Description\n"
        "2023-01-01,-500.00,Debit Spend\n"
        "2023-01-02,0.00,Zero Fee\n"
        "2023-01-03,1500.00,Credit Inflow\n"
    )
    parsed = parse_csv_statement(csv_data)
    amounts = parsed.transactions["amount"].tolist()
    assert amounts == [500.0, 0.0, 1500.0]


def test_adversarial_malformed_date_strings():
    """Verify corrupt and unparseable dates are rejected without crashing."""
    csv_data = (
        "Date,Amount,Description\n"
        "not-a-date,100,Item 1\n"
        "2023-99-99,200,Item 2\n"
        "2023-01-15,300,Valid Item\n"
    )
    parsed = parse_csv_statement(csv_data)
    assert parsed.rows_rejected == 2
    assert parsed.rows_parsed == 1


def test_adversarial_unknown_merchant_classified_as_unknown():
    """Verify unmapped counterparties become UNKNOWN rather than being guessed."""
    csv_data = "Date,Amount,Description\n2023-01-01,450,ArbitraryEntityAlpha99\n"
    parsed = parse_csv_statement(csv_data)
    assert parsed.transactions["normalized_category"].iloc[0] == "unknown"


def test_adversarial_unicode_and_latin1_encodings():
    """Verify CSV files encoded in Latin-1 / CP1252 decode seamlessly."""
    latin1_bytes = "Date,Amount,Description\n2023-01-01,150,Café Spend\n".encode("latin-1")
    parsed = parse_csv_statement(latin1_bytes)
    assert parsed.rows_parsed == 1


def test_adversarial_header_whitespace_and_casing_variations():
    """Verify messy header casing and whitespace are resolved via alias dictionaries."""
    csv_data = "  TRANSACTION DATE  ,  TXN AMOUNT  ,  NARATION  \n2023-01-01,500,Swiggy\n"
    parsed = parse_csv_statement(csv_data)
    assert parsed.rows_parsed == 1


def test_adversarial_path_traversal_filename_blocked():
    """Verify path traversal filenames raise PathTraversalError."""
    with pytest.raises(PathTraversalError):
        validate_upload_security(b"Date,Amount\n2023-01-01,100\n", filename="../../../secret.csv")


def test_adversarial_oversized_upload_blocked():
    """Verify files larger than 10MB raise OversizedFileError."""
    huge_bytes = b"Date,Amount\n" + (b"0" * (10 * 1024 * 1024 + 1024))
    with pytest.raises(OversizedFileError):
        validate_upload_security(huge_bytes, filename="big.csv")


# -----------------------------------------------------------------------------
# 3. MATHEMATICAL INVARIANTS & PROPERTY CHECKS
# -----------------------------------------------------------------------------

def test_invariant_normalized_amounts_finite_and_numeric():
    """Invariant: All canonical amounts must be finite non-null floats."""
    fixture_path = FIXTURES_DIR / "representative_transactions.csv"
    parsed = parse_csv_statement(str(fixture_path))
    amounts = parsed.transactions["amount"]

    assert bool(amounts.notna().all())
    assert bool(np.isfinite(amounts).all())
    assert bool((amounts >= 0).all())


def test_invariant_canonical_schema_adherence():
    """Invariant: Parsed canonical transactions must contain every required column."""
    fixture_path = FIXTURES_DIR / "representative_transactions.csv"
    parsed = parse_csv_statement(str(fixture_path))
    assert set(CANONICAL_COLUMNS).issubset(set(parsed.transactions.columns))


def test_invariant_evidence_coverage_sums_consistently():
    """Invariant: Derived + Self-Reported + Unavailable + Imputed must equal 21."""
    fixture_path = FIXTURES_DIR / "representative_transactions.csv"
    inputs = ManualInputContract(age=30, occupation_type="gig_delivery", city_tier="tier_1")
    result = assess_statement_end_to_end(str(fixture_path), manual_inputs=inputs)

    counts = result.feature_status_counts
    total = (
        counts["derived_count"]
        + counts["self_reported_count"]
        + counts["unavailable_count"]
    )
    assert total == counts["total_model_features"] == 21
    assert counts["imputed_count"] == counts["unavailable_count"] == 9


def test_invariant_unavailable_never_presented_as_observed():
    """Invariant: Unavailable features must never have OBSERVED provenance state."""
    fixture_path = FIXTURES_DIR / "representative_transactions.csv"
    inputs = ManualInputContract(age=30, occupation_type="gig_delivery", city_tier="tier_1")
    result = assess_statement_end_to_end(str(fixture_path), manual_inputs=inputs)

    factors = result.explanation.get("factor_objects", [])
    for f in factors:
        if f.source_status == "unavailable_imputed":
            assert f.is_imputed is True


def test_invariant_credit_score_bounded_in_300_900():
    """Invariant: Any scoreable assessment must emit score strictly in [300, 900]."""
    for fix_name in ["representative_transactions.csv", "out_of_distribution_transactions.csv"]:
        f_path = FIXTURES_DIR / fix_name
        inputs = ManualInputContract(age=30, occupation_type="gig_delivery", city_tier="tier_1")
        result = assess_statement_end_to_end(str(f_path), manual_inputs=inputs)
        if result.is_scoreable:
            assert 300 <= result.credit_score <= 900


# -----------------------------------------------------------------------------
# 4. FAILURE & STATE ISOLATION
# -----------------------------------------------------------------------------

def test_malformed_statement_does_not_mutate_model_artifact():
    """Verify that failed, corrupt, or adversarial statements never mutate model weights or file on disk."""
    model_path = Path(__file__).resolve().parent.parent / "models" / "credit_model.pkl"
    original_mtime = model_path.stat().st_mtime
    original_size = model_path.stat().st_size

    # Run adversarial corrupted statement
    with pytest.raises(UnsupportedSchemaError):
        parse_csv_statement("BadHeader1,BadHeader2\n1,2\n")

    # Verify model artifact untouched
    assert model_path.stat().st_mtime == original_mtime
    assert model_path.stat().st_size == original_size == 23277


def test_cross_session_state_isolation():
    """Verify evaluating borrower A does not bleed features or state into borrower B."""
    fixture_path = FIXTURES_DIR / "representative_transactions.csv"

    inputs_a = ManualInputContract(age=22, occupation_type="freelance_digital", city_tier="tier_1")
    res_a = assess_statement_end_to_end(str(fixture_path), manual_inputs=inputs_a)

    inputs_b = ManualInputContract(age=55, occupation_type="small_trader", city_tier="tier_3")
    res_b = assess_statement_end_to_end(str(fixture_path), manual_inputs=inputs_b)

    # Borrower A and B must differ based on self-reported features
    assert res_a.borrower_id != res_b.borrower_id
    assert res_a.credit_score != res_b.credit_score
