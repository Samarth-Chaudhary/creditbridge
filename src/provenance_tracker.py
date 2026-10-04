"""
CreditBridge - Alternative Credit Scoring Engine
Phase 2: Provenance Tracking, Evidence Quality & Provenance Gates
Path: src/provenance_tracker.py

Tracks and persists the regulatory provenance lifecycle of every model feature:
- OBSERVED: Directly recorded in primary verifiable transaction / banking data.
- DERIVED: Formulated mathematically from verifiable transaction history.
- SELF_REPORTED: User-provided via manual input form without documentary proof.
- UNAVAILABLE: Unobtainable from raw inputs; preserved as NaN.
- IMPUTED: Replaced downstream by median or statistical imputer.

Computes:
- evidence_coverage: (observed + derived) / total_features
- imputation_rate: imputed / total_features
- unavailable_rate: unavailable / total_features
- self_reported_rate: self_reported / total_features
- Provenance Data Quality Gate: PASS / WARN / BLOCK
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class FeatureProvenanceState(str, Enum):
    OBSERVED = "OBSERVED"
    DERIVED = "DERIVED"
    SELF_REPORTED = "SELF_REPORTED"
    UNAVAILABLE = "UNAVAILABLE"
    IMPUTED = "IMPUTED"


@dataclass(frozen=True)
class ProvenanceEntry:
    """Individual feature-level provenance metadata record."""
    feature_name: str
    value: Any
    provenance_state: FeatureProvenanceState
    source_system: str
    transformation_applied: str
    confidence_level: str  # "high", "medium", "low"
    is_imputed: bool = False
    imputation_method: Optional[str] = None
    fallback_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "value": str(self.value) if self.value is not None else None,
            "provenance_state": self.provenance_state.value,
            "source_system": self.source_system,
            "transformation_applied": self.transformation_applied,
            "confidence_level": self.confidence_level,
            "is_imputed": self.is_imputed,
            "imputation_method": self.imputation_method,
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True)
class ProvenanceAssessmentSummary:
    """Session or dataset level provenance summary and data-quality gate."""
    total_features: int
    observed_count: int
    derived_count: int
    self_reported_count: int
    unavailable_count: int
    imputed_count: int
    evidence_coverage: float
    imputation_rate: float
    unavailable_rate: float
    self_reported_rate: float
    provenance_gate: str  # PASS / WARN / BLOCK
    gate_rationale: str
    feature_manifest: List[ProvenanceEntry]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_features": self.total_features,
            "observed_count": self.observed_count,
            "derived_count": self.derived_count,
            "self_reported_count": self.self_reported_count,
            "unavailable_count": self.unavailable_count,
            "imputed_count": self.imputed_count,
            "evidence_coverage": round(self.evidence_coverage, 4),
            "imputation_rate": round(self.imputation_rate, 4),
            "unavailable_rate": round(self.unavailable_rate, 4),
            "self_reported_rate": round(self.self_reported_rate, 4),
            "provenance_gate": self.provenance_gate,
            "gate_rationale": self.gate_rationale,
            "manifest_sample": [e.to_dict() for e in self.feature_manifest[:10]],
        }


# Default feature categorization for CreditBridge 22-feature underwriting schema
DEFAULT_PROVENANCE_SCHEMA: Dict[str, Tuple[FeatureProvenanceState, str]] = {
    "monthly_income_estimate": (FeatureProvenanceState.DERIVED, "Sum of qualified salary and P2P credits / months"),
    "cash_flow_volatility": (FeatureProvenanceState.DERIVED, "Standard deviation of weekly net cash flows"),
    "recharge_frequency_per_month": (FeatureProvenanceState.DERIVED, "Count of telecom and utility recharges / months"),
    "avg_recharge_amount": (FeatureProvenanceState.DERIVED, "Mean transaction amount for utility recharges"),
    "discretionary_spend_ratio": (FeatureProvenanceState.DERIVED, "Discretionary debits / total outflows"),
    "savings_buffer_ratio": (FeatureProvenanceState.DERIVED, "Average monthly minimum balance / monthly inflows"),
    "inflow_outflow_ratio": (FeatureProvenanceState.DERIVED, "Total credits / total debits"),
    "utility_bill_payment_regularity": (FeatureProvenanceState.DERIVED, "Proportion of on-time recurring utility payments"),
    "monthly_upi_transaction_count": (FeatureProvenanceState.DERIVED, "Count of UPI debit and credit events / months"),
    "monthly_upi_inflow_amount": (FeatureProvenanceState.DERIVED, "Total UPI credits / months"),
    "monthly_upi_outflow_amount": (FeatureProvenanceState.DERIVED, "Total UPI debits / months"),
    "upi_inflow_consistency_score": (FeatureProvenanceState.DERIVED, "Weekly regularity score of UPI receipts"),
    "late_night_transaction_ratio": (FeatureProvenanceState.DERIVED, "Transactions between 23:00 and 05:00 / total transactions"),
    "age": (FeatureProvenanceState.SELF_REPORTED, "User provided stated age on application form"),
    "occupation_type": (FeatureProvenanceState.SELF_REPORTED, "User declared employment category"),
    "city_tier": (FeatureProvenanceState.SELF_REPORTED, "User declared residential city tier"),
    "total_credit_events": (FeatureProvenanceState.OBSERVED, "Direct count of incoming bank credits"),
    "total_debit_events": (FeatureProvenanceState.OBSERVED, "Direct count of outgoing bank debits"),
    "statement_history_days": (FeatureProvenanceState.OBSERVED, "Calendar duration between first and last statement dates"),
    "zero_balance_days_ratio": (FeatureProvenanceState.DERIVED, "Days with end-of-day balance <= 0 / history days"),
    "bounced_cheque_count": (FeatureProvenanceState.OBSERVED, "Direct count of clearing returns / bounce entries"),
    "loan_repayment_debits": (FeatureProvenanceState.DERIVED, "Sum of EMI, NACH, and loan payment debits"),
}


def build_provenance_manifest(
    feature_values: Dict[str, Any],
    imputed_features: Optional[List[str]] = None,
    custom_schema: Optional[Dict[str, Tuple[FeatureProvenanceState, str]]] = None,
) -> ProvenanceAssessmentSummary:
    """
    Evaluates feature provenance, calculates coverage/imputation metrics, and executes provenance gate.
    """
    schema = custom_schema or DEFAULT_PROVENANCE_SCHEMA
    imputed_set = set(imputed_features or [])

    entries: List[ProvenanceEntry] = []
    counts = {
        FeatureProvenanceState.OBSERVED: 0,
        FeatureProvenanceState.DERIVED: 0,
        FeatureProvenanceState.SELF_REPORTED: 0,
        FeatureProvenanceState.UNAVAILABLE: 0,
        FeatureProvenanceState.IMPUTED: 0,
    }

    total_features = len(schema)

    for feat_name, (default_state, desc) in schema.items():
        val = feature_values.get(feat_name, None)

        if feat_name in imputed_set:
            state = FeatureProvenanceState.IMPUTED
            is_imp = True
            method = "Median Imputation"
            fallback = "Raw feature was missing or unparseable"
        elif val is None or (isinstance(val, float) and np.isnan(val)):
            state = FeatureProvenanceState.UNAVAILABLE
            is_imp = False
            method = None
            fallback = "Not present in transaction source"
        else:
            state = default_state
            is_imp = False
            method = None
            fallback = None

        counts[state] += 1
        entries.append(
            ProvenanceEntry(
                feature_name=feat_name,
                value=val,
                provenance_state=state,
                source_system="BankStatement/Parser" if state in (FeatureProvenanceState.OBSERVED, FeatureProvenanceState.DERIVED) else "UserInput",
                transformation_applied=desc,
                confidence_level="high" if state in (FeatureProvenanceState.OBSERVED, FeatureProvenanceState.DERIVED) else "medium",
                is_imputed=is_imp,
                imputation_method=method,
                fallback_reason=fallback,
            )
        )

    evidence_coverage = (counts[FeatureProvenanceState.OBSERVED] + counts[FeatureProvenanceState.DERIVED]) / total_features
    imputation_rate = counts[FeatureProvenanceState.IMPUTED] / total_features
    unavailable_rate = counts[FeatureProvenanceState.UNAVAILABLE] / total_features
    self_reported_rate = counts[FeatureProvenanceState.SELF_REPORTED] / total_features

    # Deterministic Provenance-Aware Data-Quality Gate
    if evidence_coverage < 0.50 or unavailable_rate > 0.35 or imputation_rate > 0.35:
        gate = "BLOCK"
        rationale = (
            f"PROVENANCE_BLOCK: Verifiable evidence coverage ({evidence_coverage:.1%}) is below 50.0% "
            f"or excessive missingness/imputation ({imputation_rate:.1%}). Underwriting cannot proceed on unverified signals."
        )
    elif evidence_coverage < 0.70 or imputation_rate > 0.15:
        gate = "WARN"
        rationale = (
            f"PROVENANCE_WARN: Moderate evidence coverage ({evidence_coverage:.1%}) or imputation ({imputation_rate:.1%}). "
            "Downstream decision policy requires elevated manual review."
        )
    else:
        gate = "PASS"
        rationale = (
            f"PROVENANCE_PASS: Strong evidence coverage ({evidence_coverage:.1%}) with low imputation ({imputation_rate:.1%}). "
            "Sufficient objective transaction signals present."
        )

    return ProvenanceAssessmentSummary(
        total_features=total_features,
        observed_count=counts[FeatureProvenanceState.OBSERVED],
        derived_count=counts[FeatureProvenanceState.DERIVED],
        self_reported_count=counts[FeatureProvenanceState.SELF_REPORTED],
        unavailable_count=counts[FeatureProvenanceState.UNAVAILABLE],
        imputed_count=counts[FeatureProvenanceState.IMPUTED],
        evidence_coverage=evidence_coverage,
        imputation_rate=imputation_rate,
        unavailable_rate=unavailable_rate,
        self_reported_rate=self_reported_rate,
        provenance_gate=gate,
        gate_rationale=rationale,
        feature_manifest=entries,
    )
