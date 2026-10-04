"""
CreditBridge - Alternative Credit Scoring Engine
Real Data Mode: Feature & Schema Validation Layer
Path: src/real_data_validation.py

Enforces strict verification on the mapped single-borrower DataFrame before handoff
to downstream model execution, preventing malformed or invalid records from entering
the frozen FeaturePipeline.
"""

from typing import Dict

import numpy as np
import pandas as pd

from src.real_data_contracts import (
    ALLOWED_CITY_TIERS,
    ALLOWED_OCCUPATIONS,
    RAW_BORROWER_COLUMNS,
    UNAVAILABLE_MODEL_COLUMNS,
    FeatureProvenanceRecord,
    ManualInputContract,
    ManualInputContractError,
    ModelInputContractError,
)


def validate_manual_inputs(manual_inputs: ManualInputContract) -> None:
    """Validates applicant-supplied self-reported fields."""
    if not isinstance(manual_inputs, ManualInputContract):
        raise ManualInputContractError(f"manual_inputs must be ManualInputContract instance. Got: {type(manual_inputs)}")
    manual_inputs.validate()


def validate_borrower_row(
    borrower_df: pd.DataFrame,
    provenance_records: Dict[str, FeatureProvenanceRecord]
) -> None:
    """
    Performs comprehensive schema and value verification on the model-facing borrower DataFrame.

    Raises:
    -------
    ModelInputContractError:
        If any column is missing, unexpected, has wrong dtype, or contains invalid/infinite values.
    """
    if not isinstance(borrower_df, pd.DataFrame):
        raise ModelInputContractError(f"borrower_df must be a pandas DataFrame. Got: {type(borrower_df)}")

    if len(borrower_df) != 1:
        raise ModelInputContractError(f"borrower_df must contain exactly 1 row. Found: {len(borrower_df)}")

    # 1. Exact Column Contract Verification
    df_cols = list(borrower_df.columns)
    if df_cols != RAW_BORROWER_COLUMNS:
        missing = [c for c in RAW_BORROWER_COLUMNS if c not in df_cols]
        unexpected = [c for c in df_cols if c not in RAW_BORROWER_COLUMNS]
        err_parts = []
        if missing:
            err_parts.append(f"Missing mandatory columns: {missing}")
        if unexpected:
            err_parts.append(f"Unexpected extra columns: {unexpected}")
        if not missing and not unexpected and df_cols != RAW_BORROWER_COLUMNS:
            err_parts.append(f"Column order mismatch. Expected: {RAW_BORROWER_COLUMNS}, Got: {df_cols}")
        raise ModelInputContractError("; ".join(err_parts))

    row = borrower_df.iloc[0]

    # 2. Borrower ID Check
    b_id = row["borrower_id"]
    if pd.isna(b_id) or not str(b_id).strip():
        raise ModelInputContractError("borrower_id must be a non-empty string.")

    # 3. Categorical Domain Checks
    occ = row["occupation_type"]
    if occ not in ALLOWED_OCCUPATIONS:
        raise ModelInputContractError(f"Invalid occupation_type '{occ}'. Allowed: {ALLOWED_OCCUPATIONS}")

    tier = row["city_tier"]
    if tier not in ALLOWED_CITY_TIERS:
        raise ModelInputContractError(f"Invalid city_tier '{tier}'. Allowed: {ALLOWED_CITY_TIERS}")

    # 4. Age Domain Check
    age_val = row["age"]
    if pd.isna(age_val) or not (18 <= age_val <= 70):
        raise ModelInputContractError(f"age must be between 18 and 70. Found: {age_val}")

    # 5. Non-Unavailable Numeric Features Check
    # These must be valid, finite non-negative numbers
    numeric_checks = [
        ("monthly_income_estimate", 0.0),
        ("recharge_frequency_per_month", 0.0),
        ("avg_recharge_amount", 0.0),
        ("recharge_amount_volatility", 0.0),
        ("monthly_upi_transaction_count", 0),
        ("monthly_upi_inflow_avg", 0.0),
        ("monthly_upi_outflow_avg", 0.0),
        ("upi_inflow_volatility_coefficient", 0.0),
        ("p2p_vs_merchant_txn_ratio", 0.0),
    ]

    for col_name, min_val in numeric_checks:
        val = row[col_name]
        if pd.isna(val):
            raise ModelInputContractError(f"Feature '{col_name}' cannot be NaN. Must be a calculated numerical value.")
        if not np.isfinite(val):
            raise ModelInputContractError(f"Feature '{col_name}' must be finite. Found: {val}")
        if val < min_val:
            raise ModelInputContractError(f"Feature '{col_name}' must be >= {min_val}. Found: {val}")

    # 6. Unavailable Columns Check
    # These must be preserved as NaN for the downstream pipeline's median imputer
    for col_name in UNAVAILABLE_MODEL_COLUMNS:
        val = row[col_name]
        if not pd.isna(val):
            raise ModelInputContractError(
                f"Feature '{col_name}' is classified as UNAVAILABLE in Real Data Mode, "
                f"but received non-NaN value {val}. Must be np.nan to trigger pipeline median imputation."
            )

    # 7. Provenance Verification
    for col_name in RAW_BORROWER_COLUMNS:
        if col_name not in provenance_records:
            raise ModelInputContractError(f"Missing provenance record for model feature: '{col_name}'")
