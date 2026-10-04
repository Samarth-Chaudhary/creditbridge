"""
CreditBridge - Alternative Credit Scoring Engine
Real Data Mode: Production CSV Transaction Parser
Path: src/real_data_parser.py

Converts voluntarily supplied transaction exports (CSV format) into a validated
canonical transaction DataFrame and structured parse/quality report.
Adheres strictly to the Part 3 & Part 4 canonical schema and privacy constraints.
"""

import datetime
import io
import re
import uuid
import warnings as py_warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from src.privacy_security import (
    MAX_CSV_ROWS,
    validate_upload_security,
)
from src.real_data_contracts import (
    AmbiguousColumnError,
    CorruptFileError,
    EmptyStatementError,
    FileTypeError,
    InvalidDataError,
    NormalizedCategory,
    ParsedStatement,
    SourceType,
    TransactionType,
    UnsupportedSchemaError,
)

# -----------------------------------------------------------------------------
# 1. HEADER ALIAS FAMILIES & RECOGNITION DICTIONARIES
# -----------------------------------------------------------------------------

HEADER_ALIAS_MAP: Dict[str, List[str]] = {
    "date": [
        "date", "transaction date", "txn date", "transaction date",
        "transaction datetime", "timestamp", "trans date",
        "txndate", "value date", "posting date", "booking date", "entry date"
    ],
    "amount": [
        "amount", "transaction amount", "txn amount", "value",
        "total amount", "inr amount", "txn val"
    ],
    "debit": [
        "debit", "withdrawal", "debit amount", "dr amount", "withdrawals",
        "dr", "withdrawal amount", "debits", "withdrawal dr"
    ],
    "credit": [
        "credit", "deposit", "credit amount", "cr amount", "deposits",
        "cr", "deposit amount", "credits", "deposit cr"
    ],
    "transaction_type": [
        "type", "transaction type", "txn type", "debit credit", "dr cr",
        "cr dr", "transaction mode", "mode", "drcr", "crdr", "type of transaction"
    ],
    "counterparty": [
        "counterparty", "merchant", "payee", "payer", "description",
        "narration", "particulars", "remarks", "transaction remarks",
        "details", "note", "beneficiary", "party", "beneficiary name",
        "account description"
    ],
    "category": [
        "category", "transaction category", "merchant category", "tag",
        "expense category", "sub category"
    ],
    "currency": [
        "currency", "curr", "ccy"
    ],
    "transaction_id": [
        "transaction id", "txn id", "ref no", "reference no",
        "reference number", "utr", "rrn", "chq ref no",
        "cheque no", "id", "trans id", "journal no"
    ],
    "balance": [
        "balance", "closing balance", "account balance", "running balance",
        "available balance", "net balance"
    ]
}

# Currency symbol cleanup regex
CURRENCY_SYMBOLS_PATTERN = re.compile(r"[₹$€£\s]|INR|Rs\.?|USD|EUR|GBP", re.IGNORECASE)

# Refund & Reversal keyword patterns
REFUND_PATTERN = re.compile(r"\b(REFUND|CASHBACK|REBATE|RTN-REFUND|REV-REFUND)\b", re.IGNORECASE)
REVERSAL_PATTERN = re.compile(r"\b(REVERSAL|REV-|REVERSED|CANCELLED|CANCELED|BOUNCE|FAILED TXN REV)\b", re.IGNORECASE)
INTERNAL_TRANSFER_PATTERN = re.compile(r"\b(SELF|SWEEP|OWN A/C|TO OWN|TRANSFER TO SELF|INTERNAL TRANSFER)\b", re.IGNORECASE)

# Known merchant entity keyword classifiers
GIG_ENTITIES = ["SWIGGY", "ZOMATO", "UBER", "OLA", "ZEPTO", "BLINKIT", "SHADOWFAX", "PORTER", "DUNZO"]
TELCO_ENTITIES = ["JIO", "AIRTEL", "VODAFONE", "VI ", "BSNL", "RECHARGE", "AIRTEL PREPAID", "JIO PREPAID"]
UTILITY_ENTITIES = ["BESCOM", "TATA POWER", "MSEDCL", "UPPCL", "DISCOM", "ELECTRICITY", "WATER BOARD", "GAS BILL"]
SALARY_ENTITIES = ["SALARY", "PAYROLL", "MONTHLY WAGE", "NEFT-CR-CMS"]
BUSINESS_ENTITIES = ["PAYTM QR", "BHARATPE", "PHONEPE QR", "SETTLEMENT", "MERCHANT CR", "POS SETTLEMENT"]


# -----------------------------------------------------------------------------
# 2. INTERNAL NORMALIZATION UTILITIES
# -----------------------------------------------------------------------------

def _clean_header_string(header: str) -> str:
    """Normalizes header string for deterministic alias matching."""
    cleaned = str(header).lower().strip()
    cleaned = re.sub(r"[_\-\/\\]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _detect_source_type(normalized_cols: Dict[str, str], raw_df: pd.DataFrame) -> str:
    """Infers whether statement resembles UPI or standard banking ledger."""
    # Check if dual debit/credit ledger columns exist first (structural bank format)
    if "debit" in normalized_cols and "credit" in normalized_cols:
        return SourceType.CSV_BANK.value

    # Check column names for UPI markers
    col_names_joined = " ".join(normalized_cols.values()).lower()
    if any(k in col_names_joined for k in ["upi", "vpa", "gpay", "phonepe", "paytm"]):
        return SourceType.CSV_UPI.value

    # Check counterparty and category content for UPI / P2P signals
    text_samples = []
    for role in ["counterparty", "category"]:
        col = normalized_cols.get(role)
        if col and col in raw_df.columns:
            text_samples.extend(raw_df[col].dropna().head(30).astype(str).tolist())

    sample_text = " ".join(text_samples).upper()
    if any(k in sample_text for k in ["UPI/", "UPI-", "@OK", "@YBL", "@PAYTM", "GPAY", "PHONEPE", "P2P"]):
        return SourceType.CSV_UPI.value

    return SourceType.CSV_GENERIC.value


def _map_columns(raw_columns: List[str]) -> Dict[str, str]:
    """
    Maps raw CSV column names to canonical roles using explicit alias dictionaries.
    Guarantees deterministic, non-ambiguous resolution.
    """
    detected: Dict[str, str] = {}
    cleaned_to_raw = {_clean_header_string(c): c for c in raw_columns}
    cleaned_headers = list(cleaned_to_raw.keys())

    for canonical_role, aliases in HEADER_ALIAS_MAP.items():
        matched_candidates = []
        for alias in aliases:
            for cleaned_hdr in cleaned_headers:
                if cleaned_hdr == alias:
                    matched_candidates.append(cleaned_to_raw[cleaned_hdr])

        # Deduplicate candidates while preserving order
        unique_matches = list(dict.fromkeys(matched_candidates))

        if len(unique_matches) == 1:
            detected[canonical_role] = unique_matches[0]
        elif len(unique_matches) > 1:
            # Deterministic resolution: prefer exact alias match over loose match
            exact_match = None
            for alias in aliases[:3]:  # Top priority aliases
                for match in unique_matches:
                    if _clean_header_string(match) == alias:
                        exact_match = match
                        break
                if exact_match:
                    break

            if exact_match:
                detected[canonical_role] = exact_match
            else:
                # Ambiguous collision between competing columns
                raise AmbiguousColumnError(
                    f"Ambiguous columns detected for role '{canonical_role}': {unique_matches}. "
                    "Cannot silently select without risking financial misinterpretation."
                )

    # Validate essential schema requirements
    if "date" not in detected:
        raise UnsupportedSchemaError(
            f"Could not identify a mandatory transaction date column. Available headers: {raw_columns}"
        )

    has_amount = "amount" in detected
    has_dr_cr_split = ("debit" in detected and "credit" in detected)

    if not has_amount and not has_dr_cr_split:
        raise UnsupportedSchemaError(
            f"Could not identify mandatory amount columns (either 'amount' or both 'debit' and 'credit'). "
            f"Available headers: {raw_columns}"
        )

    return detected


def _resolve_date_disambiguation(date_series: Any) -> Tuple[bool, bool]:
    """
    Inspects non-null date strings across the statement to safely disambiguate
    day-first (DD/MM/YYYY) vs month-first (MM/DD/YYYY) without silent guessing.

    Returns:
    --------
    (is_ambiguous, dayfirst_flag)
    """
    clean_dates = date_series.dropna().astype(str).str.strip().tolist()
    if not clean_dates:
        return True, True

    has_first_gt_12 = False
    has_second_gt_12 = False
    has_named_month = False

    for d_str in clean_dates:
        # Check for named month (e.g. 12-Aug-2024, Aug 12 2024)
        if re.search(r"[A-Za-z]{3,}", d_str):
            has_named_month = True
            break

        parts = re.findall(r"\d+", d_str)
        if len(parts) >= 3:
            # If standard YYYY-MM-DD format (first part is 4 digits)
            if len(parts[0]) == 4:
                return False, False  # ISO format is inherently unambiguous

            p1, p2 = int(parts[0]), int(parts[1])
            if p1 > 12 and p2 <= 12:
                has_first_gt_12 = True
            elif p2 > 12 and p1 <= 12:
                has_second_gt_12 = True

    if has_named_month:
        return False, True

    if has_first_gt_12 and not has_second_gt_12:
        return False, True   # Unambiguously Day-First (e.g. 25/04/2024)
    elif has_second_gt_12 and not has_first_gt_12:
        return False, False  # Unambiguously Month-First (e.g. 04/25/2024)
    elif has_first_gt_12 and has_second_gt_12:
        # Contradictory date formatting within the same file!
        return True, True
    else:
        # All day and month numbers are <= 12 (e.g. 01/02/2024 to 05/02/2024)
        # In Indian banking context, DD/MM/YYYY is standard, but if context cannot confirm,
        # flag ambiguity.
        return True, True


def _clean_amount_scalar(val: Any) -> Optional[float]:
    """Safely parses currency amounts with commas and symbols into clean positive float."""
    if val is None or pd.isna(val):
        return None

    if isinstance(val, (int, float)):
        return float(val) if not np.isnan(val) else None

    s = str(val).strip()
    if not s or s == "-" or s.lower() == "nan" or s.lower() == "null":
        return None

    # Strip currency codes and symbols
    s = CURRENCY_SYMBOLS_PATTERN.sub("", s)
    # Remove thousand separators
    s = s.replace(",", "").strip()

    # Handle accounting parentheses: (500.00) -> -500.00
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1].strip()

    try:
        return float(s)
    except ValueError:
        return None


def _classify_low_level_category(narration: str, txn_type: str) -> Tuple[str, float]:
    """Assigns documented canonical category and rule confidence without high-level feature bias."""
    text = (narration or "").upper()
    if not text:
        return NormalizedCategory.UNKNOWN.value, 0.0

    if txn_type == TransactionType.REFUND.value or REFUND_PATTERN.search(text):
        return NormalizedCategory.REFUND.value, 0.95
    if txn_type == TransactionType.REVERSAL.value or REVERSAL_PATTERN.search(text):
        return NormalizedCategory.REFUND.value, 0.95
    if INTERNAL_TRANSFER_PATTERN.search(text):
        return NormalizedCategory.TRANSFER.value, 0.90

    # Telecom check
    if any(telco in text for telco in TELCO_ENTITIES):
        return NormalizedCategory.TELECOM_RECHARGE.value, 0.95

    # Utility check
    if any(util in text for util in UTILITY_ENTITIES):
        return NormalizedCategory.UTILITY.value, 0.95

    # Gig check
    if any(gig in text for gig in GIG_ENTITIES):
        return NormalizedCategory.GIG_INCOME_LIKE.value, 0.90

    # Salary check
    if any(sal in text for sal in SALARY_ENTITIES):
        return NormalizedCategory.SALARY_LIKE.value, 0.90

    # Merchant check
    if any(biz in text for biz in BUSINESS_ENTITIES):
        return NormalizedCategory.BUSINESS_INFLOW.value, 0.85
    if "MERCHANT" in text or "STORE" in text or "RETAIL" in text:
        return NormalizedCategory.MERCHANT_SPEND.value, 0.80

    if "P2P" in text or "TRANSFER" in text:
        return NormalizedCategory.PERSON_LIKE.value, 0.70

    return NormalizedCategory.UNKNOWN.value, 0.0


# -----------------------------------------------------------------------------
# 3. PRIMARY PARSER IMPLEMENTATION
# -----------------------------------------------------------------------------

def parse_csv_statement(
    file_or_path: Union[str, Path, io.BytesIO, io.StringIO, bytes],
    user_name: Optional[str] = None,
    filename: Optional[str] = None,
) -> ParsedStatement:
    """
    Parses a CSV transaction export into a canonical transaction DataFrame and
    structured quality report.

    Parameters:
    -----------
    file_or_path : str, Path, BytesIO, StringIO, or bytes
        Input statement content or path.
    user_name : Optional[str]
        Optional account holder name to identify self-transfers safely.
    filename : Optional[str]
        Optional original upload filename for security & extension checks.

    Returns:
    --------
    ParsedStatement:
        Container with canonical transactions DataFrame and structured diagnostics.
    """
    diagnostics: List[Dict[str, Any]] = []
    warnings: List[str] = []

    # 0. Defensive Security & Upload Validation
    validate_upload_security(file_or_path, filename=filename)

    # 1. Ingest into in-memory buffer
    if isinstance(file_or_path, bytes):
        if len(file_or_path) == 0:
            raise EmptyStatementError("CSV file is empty (0 bytes).")
        stream = io.BytesIO(file_or_path)
    elif isinstance(file_or_path, (io.BytesIO, io.StringIO)):
        stream = file_or_path
    elif isinstance(file_or_path, str) and ("\n" in file_or_path or "," in file_or_path):
        if len(file_or_path.strip()) == 0:
            raise EmptyStatementError("CSV file is empty (0 bytes).")
        stream = io.StringIO(file_or_path)
    elif isinstance(file_or_path, (str, Path)):
        p = Path(file_or_path)
        if not p.exists():
            raise CorruptFileError(f"File not found: {p}")
        if p.stat().st_size == 0:
            raise EmptyStatementError(f"CSV file at {p} is empty (0 bytes).")
        try:
            with open(p, "rb") as f:
                content = f.read()
            stream = io.BytesIO(content)
        except Exception as e:
            raise CorruptFileError(f"Could not read CSV file at {p}: {str(e)}")
    else:
        raise FileTypeError(f"Unsupported file_or_path type: {type(file_or_path)}")

    # 2. Decode and parse raw CSV with encoding fallbacks
    raw_df = None
    for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            if isinstance(stream, io.BytesIO):
                stream.seek(0)
            raw_df = pd.read_csv(stream, encoding=encoding, skipinitialspace=True)
            break
        except Exception:
            continue

    if raw_df is None:
        raise CorruptFileError("Failed to decode CSV statement with supported encodings (UTF-8, Latin-1).")

    if raw_df.empty or len(raw_df) == 0:
        raise EmptyStatementError("CSV statement contains 0 data rows.")

    if len(raw_df) > MAX_CSV_ROWS:
        raise InvalidDataError(
            f"Uploaded statement exceeds maximum allowed limit of {MAX_CSV_ROWS:,} rows "
            f"(received {len(raw_df):,} rows). File rejected to prevent resource exhaustion."
        )

    rows_received = len(raw_df)

    # 3. Detect Schema and Map Columns
    raw_headers = list(raw_df.columns)
    col_map = _map_columns(raw_headers)
    source_type = _detect_source_type(col_map, raw_df)

    # 4. Date Normalization & Disambiguation
    date_col = col_map["date"]
    date_series = raw_df[date_col]
    is_date_ambiguous, dayfirst_flag = _resolve_date_disambiguation(date_series)

    if is_date_ambiguous:
        # Check if the date format is completely unresolvable or if all days/months <= 12
        clean_sample = date_series.dropna().head(5).tolist()
        warnings.append(
            f"Date format in column '{date_col}' lacks numbers > 12 to prove day/month order. "
            f"Standard day-first convention applied. Sample: {clean_sample}"
        )

    # 5. Row-by-Row Normalization Loop (Vectorized / Fast Iteration)
    canonical_rows: List[Dict[str, Any]] = []
    rejected_rows_count = 0
    duplicate_rows_count = 0
    seen_signatures: Set[Tuple[Any, ...]] = set()

    records = raw_df.to_dict("records")
    for idx, row in enumerate(records):
        # --- A. Date Parsing ---
        raw_date_val = row[date_col]
        if pd.isna(raw_date_val):
            rejected_rows_count += 1
            diagnostics.append({"row_index": idx, "reason": "Missing transaction date"})
            continue

        val_str = str(raw_date_val).strip()
        if not val_str:
            rejected_rows_count += 1
            diagnostics.append({"row_index": idx, "reason": "Missing transaction date"})
            continue

        try:
            # Fast path for standard ISO format YYYY-MM-DD
            if len(val_str) == 10 and val_str[4] == '-' and val_str[7] == '-':
                try:
                    txn_date = datetime.date.fromisoformat(val_str)
                except ValueError:
                    with py_warnings.catch_warnings():
                        py_warnings.filterwarnings("ignore", category=UserWarning)
                        parsed_dt = pd.to_datetime(val_str, dayfirst=dayfirst_flag, errors="coerce")
                    if pd.isna(parsed_dt):
                        rejected_rows_count += 1
                        diagnostics.append({"row_index": idx, "reason": "Unparseable date format"})
                        continue
                    txn_date = parsed_dt.date()
            else:
                with py_warnings.catch_warnings():
                    py_warnings.filterwarnings("ignore", category=UserWarning)
                    parsed_dt = pd.to_datetime(val_str, dayfirst=dayfirst_flag, errors="coerce")
                if pd.isna(parsed_dt):
                    rejected_rows_count += 1
                    diagnostics.append({"row_index": idx, "reason": "Unparseable date format"})
                    continue
                txn_date = parsed_dt.date()
        except Exception:
            rejected_rows_count += 1
            diagnostics.append({"row_index": idx, "reason": "Date parsing exception"})
            continue

        # --- B. Amount & Transaction Type Normalization ---
        raw_amount_val = None
        txn_type = TransactionType.UNKNOWN.value

        # Scenario 1: Separate Debit / Credit columns exist
        if "debit" in col_map and "credit" in col_map:
            dr_val = _clean_amount_scalar(row[col_map["debit"]])
            cr_val = _clean_amount_scalar(row[col_map["credit"]])

            has_dr = (dr_val is not None and dr_val > 0.0)
            has_cr = (cr_val is not None and cr_val > 0.0)

            if has_dr and has_cr:
                # Contradiction: Row has non-zero debit AND non-zero credit
                rejected_rows_count += 1
                diagnostics.append({"row_index": idx, "reason": "Contradiction: simultaneous non-zero debit and credit"})
                continue
            elif has_dr:
                txn_type = TransactionType.DEBIT.value
                raw_amount_val = dr_val
            elif has_cr:
                txn_type = TransactionType.CREDIT.value
                raw_amount_val = cr_val
            else:
                # Both zero or null
                raw_amount_val = 0.0
                txn_type = TransactionType.UNKNOWN.value

        # Scenario 2: Single Amount column
        elif "amount" in col_map:
            cleaned_amt = _clean_amount_scalar(row[col_map["amount"]])
            if cleaned_amt is None:
                rejected_rows_count += 1
                diagnostics.append({"row_index": idx, "reason": "Malformed or missing amount value"})
                continue

            # Determine type from explicit transaction_type column if present
            if "transaction_type" in col_map:
                type_indicator = str(row[col_map["transaction_type"]]).lower().strip()
                if type_indicator in ["dr", "debit", "withdrawal", "dr.", "wdl"]:
                    # Contradiction check: signed amount indicates positive credit, but type says debit
                    if cleaned_amt < 0:
                        # Negative sign + debit flag is consistent
                        raw_amount_val = abs(cleaned_amt)
                        txn_type = TransactionType.DEBIT.value
                    else:
                        raw_amount_val = cleaned_amt
                        txn_type = TransactionType.DEBIT.value
                elif type_indicator in ["cr", "credit", "deposit", "cr.", "dep"]:
                    if cleaned_amt < 0:
                        # Contradiction: negative amount labeled as credit!
                        diagnostics.append({"row_index": idx, "warning": "Negative amount marked as credit; treated as refund/reversal"})
                        raw_amount_val = abs(cleaned_amt)
                        txn_type = TransactionType.REFUND.value
                    else:
                        raw_amount_val = cleaned_amt
                        txn_type = TransactionType.CREDIT.value
                else:
                    # Unrecognized type flag, fall back to sign
                    raw_amount_val = abs(cleaned_amt)
                    txn_type = TransactionType.DEBIT.value if cleaned_amt < 0 else TransactionType.CREDIT.value
            else:
                # Sourced strictly from signed amount
                if cleaned_amt < 0:
                    raw_amount_val = abs(cleaned_amt)
                    txn_type = TransactionType.DEBIT.value
                elif cleaned_amt > 0:
                    raw_amount_val = cleaned_amt
                    txn_type = TransactionType.CREDIT.value
                else:
                    raw_amount_val = 0.0
                    txn_type = TransactionType.UNKNOWN.value

        if raw_amount_val is None:
            rejected_rows_count += 1
            diagnostics.append({"row_index": idx, "reason": "Unresolvable transaction amount"})
            continue

        final_amount = float(raw_amount_val)

        # --- C. Counterparty, Narration & Metadata ---
        counterparty_str = ""
        if "counterparty" in col_map:
            raw_cparty = row[col_map["counterparty"]]
            if not pd.isna(raw_cparty):
                counterparty_str = str(raw_cparty).strip()

        raw_cat_str = ""
        if "category" in col_map:
            raw_cat = row[col_map["category"]]
            if not pd.isna(raw_cat):
                raw_cat_str = str(raw_cat).strip()

        currency_val = "INR"
        if "currency" in col_map:
            raw_curr = row[col_map["currency"]]
            if not pd.isna(raw_curr) and str(raw_curr).strip():
                currency_val = str(raw_curr).strip().upper()

        txn_id_val = f"txn_{uuid.uuid4().hex[:12]}"
        if "transaction_id" in col_map:
            raw_id = row[col_map["transaction_id"]]
            if not pd.isna(raw_id) and str(raw_id).strip():
                txn_id_val = str(raw_id).strip()

        # --- D. Refund, Reversal & Internal Transfer Rule Flags ---
        text_for_rules = f"{counterparty_str} {raw_cat_str}".upper()

        is_refund_flag = bool(REFUND_PATTERN.search(text_for_rules))
        is_reversal_flag = bool(REVERSAL_PATTERN.search(text_for_rules))
        is_internal_flag = bool(INTERNAL_TRANSFER_PATTERN.search(text_for_rules))
        if user_name and user_name.upper() in text_for_rules:
            is_internal_flag = True

        if is_refund_flag:
            txn_type = TransactionType.REFUND.value
        elif is_reversal_flag:
            txn_type = TransactionType.REVERSAL.value

        # --- E. Low-Level Normalized Category Assignment ---
        norm_cat, cat_conf = _classify_low_level_category(text_for_rules, txn_type)

        # --- F. Duplicate Detection ---
        signature = (txn_date, final_amount, txn_type, counterparty_str[:30])
        if signature in seen_signatures:
            duplicate_rows_count += 1
            # Exact duplicate detected
            diagnostics.append({"row_index": idx, "info": "Duplicate transaction detected; preserved for auditability"})
        else:
            seen_signatures.add(signature)

        canonical_rows.append({
            "transaction_id": txn_id_val,
            "transaction_date": txn_date,
            "transaction_type": txn_type,
            "amount": final_amount,
            "currency": currency_val,
            "counterparty": counterparty_str if counterparty_str else None,
            "raw_category": raw_cat_str if raw_cat_str else None,
            "normalized_category": norm_cat,
            "source_type": source_type,
            "is_refund": is_refund_flag,
            "is_reversal": is_reversal_flag,
            "is_internal_transfer": is_internal_flag,
            "classification_confidence": cat_conf
        })

    if len(canonical_rows) == 0:
        raise InvalidDataError(
            f"All {rows_received} rows in the CSV file failed validation. "
            "Please check date formatting, amount formatting, and debit/credit columns."
        )

    # 6. Construct Final Canonical DataFrame
    canonical_df: pd.DataFrame = pd.DataFrame(canonical_rows)
    # Sort chronologically by date
    canonical_df = canonical_df.sort_values(by="transaction_date")
    canonical_df = canonical_df.reset_index(drop=True)

    rows_parsed = len(canonical_df)

    statement_result = ParsedStatement(
        transactions=canonical_df,
        source_type=source_type,
        columns_detected=col_map,
        rows_received=rows_received,
        rows_parsed=rows_parsed,
        rows_rejected=rejected_rows_count,
        duplicate_rows=duplicate_rows_count,
        diagnostics=diagnostics,
        warnings=warnings
    )

    statement_result.validate_schema()
    return statement_result


# Public aliases
parse_statement = parse_csv_statement
