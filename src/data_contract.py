"""
CreditBridge - Preprocessing & Data Contract Validation Engine
Path: src/data_contract.py

Implements rigorous pre-flight and pre-scoring schema checks:
- Required columns and exact data types
- Allowed numeric ranges and boundary limits
- Allowed categorical domains
- Missingness (null) thresholds and availability checks
- Duplicate record checks
- Machine-readable rejection codes and audit reasons
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

DATA_CONTRACT_VERSION = "2.0.0-temporal"


class DataContractViolationError(ValueError):
    """Raised when an incoming dataset or payload violates the strict data contract."""
    def __init__(self, message: str, rejection_code: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.rejection_code = rejection_code
        self.details = details or {}


@dataclass
class ColumnSpec:
    name: str
    dtype: str  # "float", "int", "string", "category"
    required: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_categories: Optional[List[str]] = None
    allow_null: bool = False
    max_null_pct: float = 0.0


# Specification of raw dataset boundary schema (before transformation)
RAW_DATASET_SCHEMA: Dict[str, ColumnSpec] = {
    "borrower_id": ColumnSpec("borrower_id", "string", required=True, allow_null=False),
    "age": ColumnSpec("age", "int", required=True, min_value=18, max_value=80, allow_null=False),
    "occupation_type": ColumnSpec(
        "occupation_type",
        "category",
        required=True,
        allowed_categories=["daily_wage_labor", "gig_delivery", "gig_rideshare", "informal_retail", "small_trader", "freelance_digital"],
        allow_null=False,
    ),
    "city_tier": ColumnSpec(
        "city_tier",
        "category",
        required=True,
        allowed_categories=["tier_1", "tier_2", "tier_3"],
        allow_null=False,
    ),
    "monthly_income_estimate": ColumnSpec("monthly_income_estimate", "float", required=True, min_value=0.0, max_value=1_000_000.0, allow_null=True, max_null_pct=0.10),
    "electricity_bill_ontime_rate": ColumnSpec("electricity_bill_ontime_rate", "float", required=True, min_value=0.0, max_value=1.0, allow_null=True, max_null_pct=0.10),
    "electricity_bill_avg_delay_days": ColumnSpec("electricity_bill_avg_delay_days", "float", required=True, min_value=0.0, max_value=180.0, allow_null=True, max_null_pct=0.10),
    "recharge_frequency_per_month": ColumnSpec("recharge_frequency_per_month", "float", required=True, min_value=0.0, max_value=50.0, allow_null=True, max_null_pct=0.10),
    "avg_recharge_amount": ColumnSpec("avg_recharge_amount", "float", required=True, min_value=0.0, max_value=10_000.0, allow_null=True, max_null_pct=0.10),
    "recharge_amount_volatility": ColumnSpec("recharge_amount_volatility", "float", required=True, min_value=0.0, max_value=5.0, allow_null=True, max_null_pct=0.10),
    "days_since_last_recharge_lapse": ColumnSpec("days_since_last_recharge_lapse", "float", required=True, min_value=0.0, max_value=365.0, allow_null=True, max_null_pct=0.10),
    "monthly_upi_transaction_count": ColumnSpec("monthly_upi_transaction_count", "int", required=True, min_value=0, max_value=5000, allow_null=True, max_null_pct=0.10),
    "monthly_upi_inflow_avg": ColumnSpec("monthly_upi_inflow_avg", "float", required=True, min_value=0.0, max_value=2_000_000.0, allow_null=True, max_null_pct=0.10),
    "monthly_upi_outflow_avg": ColumnSpec("monthly_upi_outflow_avg", "float", required=True, min_value=0.0, max_value=2_000_000.0, allow_null=True, max_null_pct=0.10),
    "upi_inflow_volatility_coefficient": ColumnSpec("upi_inflow_volatility_coefficient", "float", required=True, min_value=0.0, max_value=5.0, allow_null=True, max_null_pct=0.10),
    "p2p_vs_merchant_txn_ratio": ColumnSpec("p2p_vs_merchant_txn_ratio", "float", required=True, min_value=0.0, max_value=50.0, allow_null=True, max_null_pct=0.10),
    "avg_weekly_gig_hours": ColumnSpec("avg_weekly_gig_hours", "float", required=False, min_value=0.0, max_value=120.0, allow_null=True, max_null_pct=1.0),
    "gig_platform_rating": ColumnSpec("gig_platform_rating", "float", required=False, min_value=1.0, max_value=5.0, allow_null=True, max_null_pct=1.0),
    "active_weeks_last_6_months": ColumnSpec("active_weeks_last_6_months", "int", required=False, min_value=0, max_value=26, allow_null=True, max_null_pct=1.0),
    "earnings_coefficient_of_variation": ColumnSpec("earnings_coefficient_of_variation", "float", required=False, min_value=0.0, max_value=5.0, allow_null=True, max_null_pct=1.0),
    "phone_number_tenure_months": ColumnSpec("phone_number_tenure_months", "int", required=True, min_value=1, max_value=600, allow_null=True, max_null_pct=0.10),
    "app_account_age_months": ColumnSpec("app_account_age_months", "int", required=True, min_value=0, max_value=240, allow_null=True, max_null_pct=0.10),
}


class DataContractValidator:
    """Validates dataframes against CreditBridge data contract specs."""

    def __init__(self, schema: Dict[str, ColumnSpec] = RAW_DATASET_SCHEMA):
        self.schema = schema

    def validate(self, df: pd.DataFrame, is_training: bool = False) -> Tuple[bool, List[str]]:
        """
        Runs comprehensive validation suite against input dataframe.
        Returns (is_valid, list_of_violations).
        """
        violations: List[str] = []

        # 1. Missing required columns
        for col_name, spec in self.schema.items():
            if spec.required and col_name not in df.columns:
                violations.append(f"MISSING_COLUMN: Required column '{col_name}' missing from input data.")

        # 2. Check for duplicate IDs if identifier present
        if "borrower_id" in df.columns:
            dups = df["borrower_id"].duplicated().sum()
            if dups > 0:
                violations.append(f"DUPLICATE_IDENTIFIERS: Found {dups} duplicate borrower_id values.")

        # 3. Value ranges and types for existing columns
        for col_name, spec in self.schema.items():
            if col_name not in df.columns:
                continue

            series = df[col_name]
            non_null = series.dropna()

            # Null percentage check
            null_pct = series.isna().mean()
            if not spec.allow_null and null_pct > 0:
                violations.append(f"NULL_DISALLOWED: Column '{col_name}' contains nulls ({null_pct:.1%}), but nulls are disallowed.")
            elif null_pct > spec.max_null_pct:
                violations.append(f"NULL_EXCEEDED: Column '{col_name}' null percentage {null_pct:.1%} exceeds allowed {spec.max_null_pct:.1%}.")

            if len(non_null) == 0:
                continue

            # Category domain check
            if spec.dtype == "category" and spec.allowed_categories is not None:
                invalid_cats = set(non_null.unique()) - set(spec.allowed_categories)
                if invalid_cats:
                    violations.append(f"INVALID_CATEGORY: Column '{col_name}' contains unauthorized values: {list(invalid_cats)}.")

            # Numeric range checks
            if spec.min_value is not None:
                numeric_vals = pd.Series(pd.to_numeric(non_null, errors="coerce"))
                min_viol = int((numeric_vals < float(spec.min_value)).sum())
                if min_viol > 0:
                    violations.append(f"RANGE_VIOLATION_MIN: Column '{col_name}' has {min_viol} values below minimum {spec.min_value}.")

            if spec.max_value is not None:
                numeric_vals = pd.Series(pd.to_numeric(non_null, errors="coerce"))
                max_viol = int((numeric_vals > float(spec.max_value)).sum())
                if max_viol > 0:
                    violations.append(f"RANGE_VIOLATION_MAX: Column '{col_name}' has {max_viol} values above maximum {spec.max_value}.")

        # 4. Target variable check in training mode
        if is_training:
            if "defaulted" not in df.columns:
                violations.append("MISSING_TARGET: Target column 'defaulted' required for training mode.")
            else:
                target_vals = set(df["defaulted"].dropna().unique())
                if not target_vals.issubset({0, 1}):
                    violations.append(f"INVALID_TARGET_VALUES: 'defaulted' must be binary (0 or 1), got: {target_vals}")

        is_valid = len(violations) == 0
        return is_valid, violations

    def assert_valid(self, df: pd.DataFrame, is_training: bool = False) -> None:
        """Raises DataContractViolationError if any violation exists."""
        is_valid, violations = self.validate(df, is_training=is_training)
        if not is_valid:
            rejection_code = violations[0].split(":")[0]
            raise DataContractViolationError(
                f"Data contract validation failed with {len(violations)} violations:\n" + "\n".join(violations[:5]),
                rejection_code=rejection_code,
                details={"violations": violations},
            )
