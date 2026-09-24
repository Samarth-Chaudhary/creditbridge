"""
CreditBridge - Real Data Mode: Comprehensive CSV Parser Test Suite
Path: tests/test_csv_parser.py

Tests all 15 mandated fixtures and verification requirements from Part 4 specification.
"""

import io
import datetime
import pytest
import pandas as pd
import numpy as np

from src.real_data_contracts import (
    CANONICAL_COLUMNS,
    ParsedStatement,
    TransactionType,
    NormalizedCategory,
    SourceType,
    UnsupportedSchemaError,
    AmbiguousColumnError,
    AmbiguousDateError,
    EmptyStatementError,
    FileTypeError,
    InvalidDataError,
)
from src.real_data_parser import parse_csv_statement, parse_statement


# -----------------------------------------------------------------------------
# FIXTURE GENERATORS & RAW STRINGS
# -----------------------------------------------------------------------------

CLEAN_UPI_CSV = """Date,Transaction Amount,Type,Counterparty,Category
2024-04-15,450.00,Debit,UPI-Swiggy Bangalore,Food
2024-04-18,25000.00,Credit,UPI-Zomato Payout,Salary
2024-04-20,150.00,Debit,UPI-Airtel Prepaid Recharge,Telecom
2024-04-22,1200.00,Debit,UPI-Bescom Electricity,Utility
2024-04-25,500.00,Credit,UPI-Transfer from Rahul,P2P
"""

CLEAN_BANK_CSV = """Txn Date,Description,Withdrawal,Deposit,Balance
15/04/2024,UPI-SWIGGY-BANGALORE,350.00,,45000.00
18/04/2024,NEFT-CR-CMS-ZOMATO-PAYOUT,,24500.00,69500.00
20/04/2024,BILLDESK-BESCOM-BLR,1450.00,,68050.00
25/04/2024,UPI-AIRTEL-RECHARGE,299.00,,67751.00
"""

ALTERNATE_ALIASES_CSV = """Timestamp,Txn Val,Dr/Cr,Narration
2024-05-01,1500.00,DR,Local Kirana Store
2024-05-05,32000.00,CR,Monthly Wage Payment
2024-05-10,666.00,DR,Jio Recharge Pack
"""

CURRENCY_SYMBOLS_COMMAS_CSV = """Date,Amount,Type,Details
2024-06-01,"₹1,50,000.00",Credit,Business Settlement
2024-06-05,"Rs. 2,450.50",Debit,Utility Payment
2024-06-10,"INR 199.00",Debit,Mobile Recharge
"""

DUPLICATE_ROWS_CSV = """Date,Amount,Type,Description
2024-05-01,250.00,Debit,Coffee Shop
2024-05-01,250.00,Debit,Coffee Shop
2024-05-02,500.00,Credit,UPI Inflow
"""

REFUNDS_CSV = """Date,Amount,Type,Description
2024-05-01,1200.00,Debit,Online Purchase
2024-05-03,1200.00,Credit,REV-REFUND Online Purchase
2024-05-05,50.00,Credit,Cashback Reward
"""

REVERSALS_CSV = """Date,Amount,Type,Description
2024-05-01,5000.00,Debit,ATM Cash Withdrawal
2024-05-02,5000.00,Credit,FAILED TXN REV ATM
"""

MALFORMED_DATES_CSV = """Date,Amount,Type,Description
2024-05-01,100.00,Debit,Valid Day
INVALID-DATE,200.00,Debit,Broken Date
2024-99-99,300.00,Debit,Impossible Date
2024-05-04,400.00,Credit,Another Valid Day
"""

MALFORMED_AMOUNTS_CSV = """Date,Amount,Type,Description
2024-05-01,100.00,Debit,Valid Amount
2024-05-02,NOT_A_NUMBER,Debit,Invalid Amount
2024-05-03,-,Debit,Dash Amount
2024-05-04,400.00,Credit,Valid Amount
"""

MISSING_CATEGORY_COUNTERPARTY_CSV = """Date,Amount,Type
2024-05-01,100.00,Debit
2024-05-02,200.00,Credit
"""

AMBIGUOUS_DATE_CSV = """Date,Amount,Type,Description
01/02/2024,100.00,Debit,Could be Jan 2 or Feb 1
02/03/2024,200.00,Debit,Could be Feb 3 or Mar 2
03/04/2024,300.00,Credit,Could be Mar 4 or Apr 3
"""

EMPTY_CSV = ""

MISSING_MANDATORY_FIELDS_CSV = """Transaction ID,Counterparty,Category
txn_123,Swiggy,Food
txn_124,Bescom,Utility
"""

UNKNOWN_MERCHANT_CSV = """Date,Amount,Type,Description
2024-05-01,150.00,Debit,MISC UNKNOWN TRANSACTION 847291
2024-05-02,250.00,Credit,UNSPECIFIED INFLOW
"""


# -----------------------------------------------------------------------------
# 2. TEST CASES
# -----------------------------------------------------------------------------

def test_clean_upi_csv():
    """Verifies that standard clean UPI export parses cleanly into canonical schema."""
    res = parse_csv_statement(CLEAN_UPI_CSV.encode("utf-8"))
    assert isinstance(res, ParsedStatement)
    assert res.source_type == SourceType.CSV_UPI.value
    assert res.rows_received == 5
    assert res.rows_parsed == 5
    assert res.rows_rejected == 0
    assert list(res.transactions.columns) == CANONICAL_COLUMNS
    
    # Check specific row attributes
    swiggy_row = res.transactions[res.transactions["counterparty"].str.contains("Swiggy")].iloc[0]
    assert swiggy_row["amount"] == 450.0
    assert swiggy_row["transaction_type"] == TransactionType.DEBIT.value
    assert swiggy_row["normalized_category"] == NormalizedCategory.GIG_INCOME_LIKE.value or swiggy_row["normalized_category"] == NormalizedCategory.MERCHANT_SPEND.value


def test_clean_bank_csv_with_dr_cr_split():
    """Verifies that separate debit/credit column bank exports are parsed with correct types."""
    res = parse_csv_statement(CLEAN_BANK_CSV.encode("utf-8"))
    assert res.source_type == SourceType.CSV_BANK.value
    assert res.rows_parsed == 4
    
    # Verify withdrawal row
    bescom_row = res.transactions[res.transactions["counterparty"].str.contains("BESCOM")].iloc[0]
    assert bescom_row["amount"] == 1450.0
    assert bescom_row["transaction_type"] == TransactionType.DEBIT.value
    assert bescom_row["normalized_category"] == NormalizedCategory.UTILITY.value

    # Verify deposit row
    zomato_row = res.transactions[res.transactions["counterparty"].str.contains("ZOMATO")].iloc[0]
    assert zomato_row["amount"] == 24500.0
    assert zomato_row["transaction_type"] == TransactionType.CREDIT.value
    assert zomato_row["normalized_category"] == NormalizedCategory.GIG_INCOME_LIKE.value


def test_alternate_header_aliases():
    """Verifies that alternate header synonyms (Timestamp, Txn Val, Dr/Cr) are matched."""
    res = parse_csv_statement(ALTERNATE_ALIASES_CSV.encode("utf-8"))
    assert res.rows_parsed == 3
    assert res.columns_detected["date"] == "Timestamp"
    assert res.columns_detected["amount"] == "Txn Val"
    assert res.columns_detected["transaction_type"] == "Dr/Cr"
    
    wage_row = res.transactions[res.transactions["counterparty"].str.contains("Wage")].iloc[0]
    assert wage_row["amount"] == 32000.0
    assert wage_row["transaction_type"] == TransactionType.CREDIT.value
    assert wage_row["normalized_category"] == NormalizedCategory.SALARY_LIKE.value


def test_currency_symbols_and_commas():
    """Verifies that ₹, Rs., INR and thousand commas (1,50,000.00) are normalized without loss."""
    res = parse_csv_statement(CURRENCY_SYMBOLS_COMMAS_CSV.encode("utf-8"))
    assert res.rows_parsed == 3
    amounts = res.transactions["amount"].tolist()
    assert amounts == [150000.0, 2450.50, 199.0]
    assert res.transactions["currency"].tolist() == ["INR", "INR", "INR"]


def test_duplicate_detection():
    """Verifies that duplicate rows are detected, counted, and recorded in diagnostics."""
    res = parse_csv_statement(DUPLICATE_ROWS_CSV.encode("utf-8"))
    assert res.rows_received == 3
    assert res.duplicate_rows == 1
    # Both preserved in in-memory table for auditability
    assert res.rows_parsed == 3
    diag_infos = [d.get("info", "") for d in res.diagnostics]
    assert any("Duplicate transaction detected" in msg for msg in diag_infos)


def test_refunds_and_reversals():
    """Verifies that refunds and reversals are flagged deterministically."""
    res_refund = parse_csv_statement(REFUNDS_CSV.encode("utf-8"))
    assert res_refund.rows_parsed == 3
    refund_row = res_refund.transactions[res_refund.transactions["counterparty"].str.contains("REV-REFUND")].iloc[0]
    assert bool(refund_row["is_refund"]) is True
    assert refund_row["transaction_type"] == TransactionType.REFUND.value

    res_reversal = parse_csv_statement(REVERSALS_CSV.encode("utf-8"))
    assert res_reversal.rows_parsed == 2
    rev_row = res_reversal.transactions[res_reversal.transactions["counterparty"].str.contains("FAILED TXN REV")].iloc[0]
    assert bool(rev_row["is_reversal"]) is True
    assert rev_row["transaction_type"] == TransactionType.REVERSAL.value


def test_malformed_dates_reported():
    """Verifies that malformed date rows are rejected with diagnostics and not silently lost."""
    res = parse_csv_statement(MALFORMED_DATES_CSV.encode("utf-8"))
    assert res.rows_received == 4
    assert res.rows_parsed == 2
    assert res.rows_rejected == 2
    reasons = [d.get("reason", "") for d in res.diagnostics]
    assert "Unparseable date format" in reasons


def test_malformed_amounts_reported():
    """Verifies that unparseable amounts are rejected with diagnostics."""
    res = parse_csv_statement(MALFORMED_AMOUNTS_CSV.encode("utf-8"))
    assert res.rows_received == 4
    assert res.rows_parsed == 2
    assert res.rows_rejected == 2
    reasons = [d.get("reason", "") for d in res.diagnostics]
    assert "Malformed or missing amount value" in reasons


def test_missing_category_and_counterparty():
    """Verifies that statements without category/counterparty columns parse safely with None."""
    res = parse_csv_statement(MISSING_CATEGORY_COUNTERPARTY_CSV.encode("utf-8"))
    assert res.rows_parsed == 2
    assert bool(res.transactions["counterparty"].isnull().all())
    assert bool(res.transactions["raw_category"].isnull().all())
    assert bool((res.transactions["normalized_category"] == NormalizedCategory.UNKNOWN.value).all())


def test_ambiguous_date_warning():
    """Verifies that dates where all day/month numbers <= 12 emit an explicit warning."""
    res = parse_csv_statement(AMBIGUOUS_DATE_CSV.encode("utf-8"))
    assert res.rows_parsed == 3
    assert len(res.warnings) > 0
    assert any("lacks numbers > 12" in w for w in res.warnings)


def test_empty_csv_raises_error():
    """Verifies that an empty CSV raises EmptyStatementError."""
    with pytest.raises(EmptyStatementError):
        parse_csv_statement(EMPTY_CSV.encode("utf-8"))


def test_missing_mandatory_fields_raises_error():
    """Verifies that CSV missing date/amount raises UnsupportedSchemaError."""
    with pytest.raises(UnsupportedSchemaError):
        parse_csv_statement(MISSING_MANDATORY_FIELDS_CSV.encode("utf-8"))


def test_unknown_merchant_preserves_unknown():
    """Verifies that ambiguous merchants are preserved as UNKNOWN, never over-classified."""
    res = parse_csv_statement(UNKNOWN_MERCHANT_CSV.encode("utf-8"))
    assert res.rows_parsed == 2
    assert (res.transactions["normalized_category"] == NormalizedCategory.UNKNOWN.value).all()
    assert (res.transactions["classification_confidence"] == 0.0).all()


def test_large_synthetic_csv_performance():
    """Performance smoke test: parses 5,000 synthetic transaction rows efficiently."""
    n_rows = 5000
    dates = [datetime.date(2024, 1, 1) + datetime.timedelta(days=i % 180) for i in range(n_rows)]
    amounts = np.random.uniform(50.0, 5000.0, size=n_rows).round(2)
    types = ["Debit" if i % 3 != 0 else "Credit" for i in range(n_rows)]
    narrations = ["UPI-TEST-MERCHANT" if i % 2 == 0 else "SWIGGY-BANGALORE" for i in range(n_rows)]
    
    df = pd.DataFrame({
        "Date": [d.strftime("%Y-%m-%d") for d in dates],
        "Amount": amounts,
        "Type": types,
        "Narration": narrations
    })
    
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    
    start_time = datetime.datetime.now()
    res = parse_csv_statement(csv_bytes)
    duration = (datetime.datetime.now() - start_time).total_seconds()
    
    assert res.rows_parsed == n_rows
    assert duration < 5.0, f"Parsing 5,000 rows took {duration:.2f}s, expected < 5s."


def test_privacy_no_sensitive_data_in_diagnostics():
    """Verifies that PII, full descriptions, and account numbers do not leak into diagnostics."""
    sensitive_csv = """Date,Amount,Type,Narration
2024-05-01,1000.00,Debit,PAYMENT TO A/C 987654321012 IFSC HDFC0001234 SECRET_PIN_XYZ
"""
    res = parse_csv_statement(sensitive_csv.encode("utf-8"))
    diag_str = str(res.diagnostics) + " ".join(res.warnings)
    assert "987654321012" not in diag_str
    assert "SECRET_PIN_XYZ" not in diag_str
