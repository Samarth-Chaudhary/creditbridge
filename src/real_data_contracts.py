"""
CreditBridge - Alternative Credit Scoring Engine
Real Data Mode: Contracts, Schemas, Enums & Error Taxonomy
Path: src/real_data_contracts.py

Defines the source-independent canonical schema, data contracts, provenance
enums, and structured error hierarchy for Real Data Mode.
"""

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# 1. CONTROLLED ENUMS & VOCABULARIES
# -----------------------------------------------------------------------------

class TransactionType(str, Enum):
    CREDIT = "credit"
    DEBIT = "debit"
    REFUND = "refund"
    REVERSAL = "reversal"
    TRANSFER = "transfer"
    UNKNOWN = "unknown"


class NormalizedCategory(str, Enum):
    SALARY_LIKE = "salary_like"
    GIG_INCOME_LIKE = "gig_income_like"
    BUSINESS_INFLOW = "business_inflow"
    TRANSFER = "transfer"
    INTERNAL_TRANSFER = "internal_transfer"
    REFUND = "refund"
    LOAN_RELATED = "loan_related"
    TELECOM_RECHARGE = "telecom_recharge"
    UTILITY = "utility"
    MERCHANT_SPEND = "merchant_spend"
    PERSON_LIKE = "person_like"
    UNKNOWN = "unknown"


class SourceType(str, Enum):
    CSV_UPI = "csv_upi"
    CSV_BANK = "csv_bank"
    CSV_GENERIC = "csv_generic"
    UNKNOWN = "unknown"


class ProvenanceState(str, Enum):
    OBSERVED = "OBSERVED"
    DERIVED = "DERIVED"
    SELF_REPORTED = "SELF_REPORTED"
    UNAVAILABLE = "UNAVAILABLE"
    IMPUTED = "IMPUTED"


class SufficiencyTier(str, Enum):
    INSUFFICIENT = "INSUFFICIENT"
    MARGINAL = "MARGINAL"
    ADEQUATE = "ADEQUATE"
    OPTIMAL = "OPTIMAL"


class DistributionStatus(str, Enum):
    WITHIN_RANGE = "within_range"
    NEAR_BOUNDARY = "near_boundary"
    OUTSIDE_OBSERVED_RANGE = "outside_observed_range"
    NOT_ASSESSABLE = "not_assessable"
    KNOWN_IN_TRAINING = "known_in_training"
    UNSEEN_IN_TRAINING = "unseen_in_training"


class AssessmentQualityStatus(str, Enum):
    SUFFICIENT = "SUFFICIENT"
    LIMITED = "LIMITED"
    INSUFFICIENT = "INSUFFICIENT"


# -----------------------------------------------------------------------------
# 2. CANONICAL TRANSACTION CONTRACT
# -----------------------------------------------------------------------------

CANONICAL_COLUMNS = [
    "transaction_id",
    "transaction_date",
    "transaction_type",
    "amount",
    "currency",
    "counterparty",
    "raw_category",
    "normalized_category",
    "source_type",
    "is_refund",
    "is_reversal",
    "is_internal_transfer",
    "classification_confidence",
]


@dataclass(frozen=True)
class CanonicalTransaction:
    """Represents a single normalized transaction in the source-independent canonical schema."""
    transaction_id: str
    transaction_date: date
    transaction_type: str
    amount: float
    currency: Optional[str] = "INR"
    counterparty: Optional[str] = None
    raw_category: Optional[str] = None
    normalized_category: str = NormalizedCategory.UNKNOWN.value
    source_type: str = SourceType.CSV_GENERIC.value
    is_refund: bool = False
    is_reversal: bool = False
    is_internal_transfer: bool = False
    classification_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "transaction_date": self.transaction_date,
            "transaction_type": self.transaction_type,
            "amount": self.amount,
            "currency": self.currency,
            "counterparty": self.counterparty,
            "raw_category": self.raw_category,
            "normalized_category": self.normalized_category,
            "source_type": self.source_type,
            "is_refund": self.is_refund,
            "is_reversal": self.is_reversal,
            "is_internal_transfer": self.is_internal_transfer,
            "classification_confidence": self.classification_confidence,
        }


# -----------------------------------------------------------------------------
# 3. BORROWER FEATURE CONTRACT & DOMAINS
# -----------------------------------------------------------------------------

RAW_BORROWER_COLUMNS: List[str] = [
    "borrower_id",
    "age",
    "occupation_type",
    "city_tier",
    "monthly_income_estimate",
    "electricity_bill_ontime_rate",
    "electricity_bill_avg_delay_days",
    "recharge_frequency_per_month",
    "avg_recharge_amount",
    "recharge_amount_volatility",
    "days_since_last_recharge_lapse",
    "monthly_upi_transaction_count",
    "monthly_upi_inflow_avg",
    "monthly_upi_outflow_avg",
    "upi_inflow_volatility_coefficient",
    "p2p_vs_merchant_txn_ratio",
    "avg_weekly_gig_hours",
    "gig_platform_rating",
    "active_weeks_last_6_months",
    "earnings_coefficient_of_variation",
    "phone_number_tenure_months",
    "app_account_age_months",
]

ALLOWED_OCCUPATIONS: List[str] = [
    "daily_wage_labor",
    "freelance_digital",
    "gig_delivery",
    "gig_rideshare",
    "informal_retail",
    "small_trader",
]

ALLOWED_CITY_TIERS: List[str] = [
    "tier_1",
    "tier_2",
    "tier_3",
]

UNAVAILABLE_MODEL_COLUMNS: List[str] = [
    "electricity_bill_ontime_rate",
    "electricity_bill_avg_delay_days",
    "days_since_last_recharge_lapse",
    "avg_weekly_gig_hours",
    "gig_platform_rating",
    "active_weeks_last_6_months",
    "earnings_coefficient_of_variation",
    "phone_number_tenure_months",
    "app_account_age_months",
]


@dataclass(frozen=True)
class ManualInputContract:
    """Explicit minimal manual-input contract for attributes that cannot be observed from statements."""
    age: int
    occupation_type: str
    city_tier: str
    borrower_id: Optional[str] = None
    account_holder_name: Optional[str] = None

    def validate(self) -> None:
        if not isinstance(self.age, (int, np.integer)) or not (18 <= self.age <= 70):
            raise ManualInputContractError(f"Age must be an integer between 18 and 70. Received: {self.age}")
        if self.occupation_type not in ALLOWED_OCCUPATIONS:
            raise ManualInputContractError(
                f"Invalid occupation_type '{self.occupation_type}'. Allowed: {ALLOWED_OCCUPATIONS}"
            )
        if self.city_tier not in ALLOWED_CITY_TIERS:
            raise ManualInputContractError(
                f"Invalid city_tier '{self.city_tier}'. Allowed: {ALLOWED_CITY_TIERS}"
            )


@dataclass(frozen=True)
class FeatureProvenanceRecord:
    """Machine-readable provenance record for an individual model-facing feature."""
    feature_name: str
    value: Any
    provenance_state: str  # OBSERVED, DERIVED, SELF_REPORTED, UNAVAILABLE, IMPUTED
    source: str            # statement, user_input, synthetic_imputation
    transformation: str
    confidence: str = "high"        # high, medium, low, neutral
    is_imputed: bool = False
    fallback_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "value": None if pd.isna(self.value) else self.value,
            "provenance_state": self.provenance_state,
            "source": self.source,
            "transformation": self.transformation,
            "confidence": self.confidence,
            "is_imputed": self.is_imputed,
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True)
class HistorySufficiencyReport:
    """Detailed statement history and coverage evaluation report."""
    start_date: date
    end_date: date
    total_calendar_days: int
    approx_months: float
    total_usable_transactions: int
    sufficiency_tier: str  # INSUFFICIENT, MARGINAL, ADEQUATE, OPTIMAL
    is_scoreable: bool
    underwriting_action: str
    warning_message: Optional[str] = None


@dataclass
class RealBorrowerPayload:
    """Complete container emitted by feature mapper for downstream pipeline handoff."""
    borrower_df: pd.DataFrame
    provenance_records: Dict[str, FeatureProvenanceRecord]
    history_report: HistorySufficiencyReport
    evidence_counts: Dict[str, int]


# -----------------------------------------------------------------------------
# 4. PARSER & VALIDATION OUTPUT CONTRACTS
# -----------------------------------------------------------------------------

@dataclass
class ParsedStatement:
    """Primary output container emitted by parse_csv_statement."""
    transactions: pd.DataFrame
    source_type: str
    columns_detected: Dict[str, str]
    rows_received: int
    rows_parsed: int
    rows_rejected: int
    duplicate_rows: int
    diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def validate_schema(self) -> bool:
        """Verifies that the transactions DataFrame matches the exact canonical columns."""
        if not set(CANONICAL_COLUMNS).issubset(set(self.transactions.columns)):
            missing = set(CANONICAL_COLUMNS) - set(self.transactions.columns)
            raise SchemaMismatchError(f"Canonical DataFrame missing required columns: {missing}")
        return True


@dataclass
class FieldQualityDiagnostic:
    """Field-level quality and distribution diagnostic record."""
    feature_name: str
    value: Any
    provenance_state: str
    distribution_status: str  # within_range, near_boundary, outside_observed_range, not_assessable, known_in_training, unseen_in_training
    training_min: Optional[float] = None
    training_p5: Optional[float] = None
    training_p95: Optional[float] = None
    training_max: Optional[float] = None
    warning_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "value": None if pd.isna(self.value) else self.value,
            "provenance_state": self.provenance_state,
            "distribution_status": self.distribution_status,
            "training_min": self.training_min,
            "training_p5": self.training_p5,
            "training_p95": self.training_p95,
            "training_max": self.training_max,
            "warning_message": self.warning_message,
        }


@dataclass
class RealDataQualityReport:
    """
    Authoritative data quality, evidence coverage, and distribution report.
    Consolidated assessment artifact produced by Part 6 for Part 7 model scoring handoff.
    """
    # 1. Parse and Transaction Quality Metrics
    rows_received: int
    rows_accepted: int
    rows_rejected: int
    duplicate_count: int
    duplicate_rate: float
    invalid_date_count: int
    missing_amount_count: int
    unknown_transaction_type_count: int
    unknown_category_count: int

    # 2. History Sufficiency Metrics
    history_start_date: Optional[date]
    history_end_date: Optional[date]
    history_days: int
    history_months: float
    transaction_count: int
    history_tier: str

    # 3. Classification Metrics
    category_coverage: float
    unknown_category_share: float
    is_classification_sparse: bool

    # 4. Feature Evidence Composition (Denominator: 21 model features, excluding borrower_id)
    total_model_features: int
    observed_count: int
    derived_count: int
    self_reported_count: int
    unavailable_count: int
    imputed_count: int
    evidence_coverage_ratio: float
    derived_coverage_ratio: float
    self_reported_coverage_ratio: float
    imputed_ratio: float

    # 5. Distribution Diagnostics
    field_diagnostics: List[FieldQualityDiagnostic]
    outside_range_count: int
    near_boundary_count: int
    within_range_count: int

    # 6. Overall Assessment Summary
    overall_status: str  # SUFFICIENT, LIMITED, INSUFFICIENT
    is_eligible_for_scoring: bool
    quality_warnings: List[str] = field(default_factory=list)
    blocker_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rows_received": self.rows_received,
            "rows_accepted": self.rows_accepted,
            "rows_rejected": self.rows_rejected,
            "duplicate_count": self.duplicate_count,
            "duplicate_rate": self.duplicate_rate,
            "invalid_date_count": self.invalid_date_count,
            "missing_amount_count": self.missing_amount_count,
            "unknown_transaction_type_count": self.unknown_transaction_type_count,
            "unknown_category_count": self.unknown_category_count,
            "history_start_date": str(self.history_start_date) if self.history_start_date else None,
            "history_end_date": str(self.history_end_date) if self.history_end_date else None,
            "history_days": self.history_days,
            "history_months": self.history_months,
            "transaction_count": self.transaction_count,
            "history_tier": self.history_tier,
            "category_coverage": self.category_coverage,
            "unknown_category_share": self.unknown_category_share,
            "is_classification_sparse": self.is_classification_sparse,
            "total_model_features": self.total_model_features,
            "observed_count": self.observed_count,
            "derived_count": self.derived_count,
            "self_reported_count": self.self_reported_count,
            "unavailable_count": self.unavailable_count,
            "imputed_count": self.imputed_count,
            "evidence_coverage_ratio": self.evidence_coverage_ratio,
            "derived_coverage_ratio": self.derived_coverage_ratio,
            "self_reported_coverage_ratio": self.self_reported_coverage_ratio,
            "imputed_ratio": self.imputed_ratio,
            "outside_range_count": self.outside_range_count,
            "near_boundary_count": self.near_boundary_count,
            "within_range_count": self.within_range_count,
            "overall_status": self.overall_status,
            "is_eligible_for_scoring": self.is_eligible_for_scoring,
            "quality_warnings": self.quality_warnings,
            "blocker_reasons": self.blocker_reasons,
            "field_diagnostics": [d.to_dict() for d in self.field_diagnostics],
        }


# -----------------------------------------------------------------------------
# 5. STRUCTURED ERROR TAXONOMY
# -----------------------------------------------------------------------------

class CreditBridgeError(Exception):
    """Root exception for all CreditBridge operations."""
    pass


class ParsingError(CreditBridgeError):
    """Base class for statement parsing and format errors."""
    pass


class UnsupportedSchemaError(ParsingError):
    """Raised when mandatory transaction columns (date, amount) cannot be identified."""
    pass


class AmbiguousColumnError(ParsingError):
    """Raised when multiple candidate columns equally match a mandatory canonical role."""
    pass


class AmbiguousDateError(ParsingError):
    """Raised when date format (dayfirst vs monthfirst) cannot be resolved safely."""
    pass


class EmptyStatementError(ParsingError):
    """Raised when the uploaded file contains 0 valid transaction rows."""
    pass


class FileTypeError(ParsingError):
    """Raised when file extension or MIME type is not supported."""
    pass


class OversizedFileError(ParsingError):
    """Raised when the uploaded file exceeds maximum allowed size (e.g. 10 MB)."""
    pass


class SecurityViolationError(CreditBridgeError):
    """Base class for security, privacy, or integrity constraint violations."""
    pass


class PathTraversalError(SecurityViolationError):
    """Raised when an untrusted path or filename contains traversal sequences (e.g. ../)."""
    pass


class CorruptFileError(ParsingError):
    """Raised when the statement file cannot be decoded or read."""
    pass


class ParserInternalError(ParsingError):
    """Raised when an unexpected coding/runtime failure occurs inside the parser."""
    pass


class ValidationError(CreditBridgeError):
    """Base class for data validation failures."""
    pass


class InvalidDataError(ValidationError):
    """Raised when rows contain unrecoverable date/amount contradictions."""
    pass


class InvalidTransactionError(ValidationError):
    """Raised when a transaction row violates fundamental financial constraints."""
    pass


class InsufficientHistoryError(ValidationError):
    """Raised when statement history is too short for alternative credit evaluation."""
    pass


class SchemaMismatchError(ValidationError):
    """Raised when output DataFrame fails the canonical schema contract."""
    pass


class MappingError(CreditBridgeError):
    """Base class for feature mapping and borrower row translation failures."""
    pass


class ManualInputContractError(MappingError):
    """Raised when user-supplied inputs (age, occupation, city tier) fail contract checks."""
    pass


class ModelInputContractError(MappingError):
    """Raised when mapped borrower row fails the exact 22-column legacy model schema."""
    pass


class IntegrationError(CreditBridgeError):
    """Base class for downstream model and pipeline integration failures."""
    error_code: str = "INTEGRATION_ERROR"


class ModelLoadError(IntegrationError):
    """Raised when models/credit_model.pkl cannot be located or unpickled."""
    error_code: str = "MODEL_LOAD_FAILURE"


class MissingRequiredModelFeatureError(IntegrationError):
    """Raised when one or more required model features are absent from the input DataFrame."""
    error_code: str = "MISSING_REQUIRED_MODEL_FEATURE"


class ModelSchemaMismatchError(IntegrationError):
    """Raised when columns, dtypes, or ordering fail the validated model schema contract."""
    error_code: str = "SCHEMA_MISMATCH"


class UnknownCategoryError(IntegrationError):
    """Raised when a categorical input (occupation_type or city_tier) violates trained vocabulary."""
    error_code: str = "UNKNOWN_CATEGORY"


class PreprocessingFailureError(IntegrationError):
    """Raised when feature transformation through the fitted FeaturePipeline fails."""
    error_code: str = "PREPROCESSING_FAILURE"


class PredictionFailureError(IntegrationError):
    """Raised when classifier model.predict_proba() fails during inference."""
    error_code: str = "PREDICTION_FAILURE"


class ScoreTransformationFailureError(IntegrationError):
    """Raised when probability-to-score or tier mapping fails numerical sanity bounds."""
    error_code: str = "SCORE_TRANSFORMATION_FAILURE"


class ExplanationFailureError(IntegrationError):
    """Raised when SHAP attribution or plain-English narrative generation fails."""
    error_code: str = "EXPLANATION_FAILURE"


# -----------------------------------------------------------------------------
# 6. MODEL ASSESSMENT & EXPLANATION CONTRACTS
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class FactorContribution:
    """Represents an individual feature's attribution to the CreditBridge model score."""
    feature_name: str
    human_name: str
    point_impact: int
    contribution: str  # "positive" (improves credit score) or "negative" (drags score down)
    shap_value: float
    feature_value: Any
    source_status: str  # derived_from_statement, self_reported, unavailable_imputed, composite_derived, composite_partially_imputed
    is_imputed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "human_name": self.human_name,
            "point_impact": self.point_impact,
            "contribution": self.contribution,
            "shap_value": round(float(self.shap_value), 6),
            "feature_value": None if pd.isna(self.feature_value) else self.feature_value,
            "source_status": self.source_status,
            "is_imputed": self.is_imputed,
        }


@dataclass(frozen=True)
class RealDataAssessmentResult:
    """
    Authoritative end-to-end result emitted by the Real Data Mode scoring engine.
    Integrates parser metrics, quality checks, model inference, calibration,
    credit scoring, risk tiers, and SHAP explainability.
    """
    borrower_id: str
    raw_model_probability: float         # Raw P(default) from balanced logistic classifier
    calibrated_model_probability: float    # Calibrated P(default) under thin-file 14% prior
    credit_score: int                    # 300-900 CreditBridge Model Score
    risk_tier: str                       # Low Risk, Moderate Risk, High Risk — Manual Review, Very High Risk
    assessment_status: str               # SUFFICIENT, LIMITED, INSUFFICIENT
    is_scoreable: bool
    evidence_coverage_ratio: float       # Exact ratio against 21 features
    history_months: float
    feature_status_counts: Dict[str, int]
    distribution_flags: List[Dict[str, Any]]
    parser_quality: Dict[str, Any]
    explanation: Dict[str, Any]
    quality_warnings: List[str]
    blocker_reasons: List[str]

    def to_dict(self) -> Dict[str, Any]:
        expl_dict = dict(self.explanation)
        if "factor_objects" in expl_dict:
            del expl_dict["factor_objects"]
        return {
            "borrower_id": self.borrower_id,
            "raw_model_probability": round(float(self.raw_model_probability), 4),
            "calibrated_model_probability": round(float(self.calibrated_model_probability), 4),
            "credit_score": self.credit_score,
            "risk_tier": self.risk_tier,
            "assessment_status": self.assessment_status,
            "is_scoreable": self.is_scoreable,
            "evidence_coverage_ratio": round(float(self.evidence_coverage_ratio), 4),
            "history_months": round(float(self.history_months), 2),
            "feature_status_counts": dict(self.feature_status_counts),
            "distribution_flags": self.distribution_flags,
            "parser_quality": self.parser_quality,
            "explanation": expl_dict,
            "quality_warnings": list(self.quality_warnings),
            "blocker_reasons": list(self.blocker_reasons),
        }

