"""
CreditBridge - Alternative Credit Scoring Engine
Real Data Mode: Feature Provenance Tracking & Reporting
Path: src/feature_provenance.py

Maintains machine-readable provenance metadata for every feature fed into downstream models.
Guarantees full regulatory transparency:
- OBSERVED: Directly recorded in primary transaction data
- DERIVED: Formulated mathematically from valid transaction evidence
- SELF_REPORTED: User-provided via manual form inputs
- UNAVAILABLE: Unobtainable from statement; preserved as NaN
- IMPUTED: Replaced downstream by synthetic pipeline median imputer
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from src.real_data_contracts import (
    FeatureProvenanceRecord,
    ProvenanceState,
    RAW_BORROWER_COLUMNS,
)


def create_provenance_record(
    feature_name: str,
    value: Any,
    provenance_state: str,
    source: str,
    transformation: str,
    confidence: str = "high",
    is_imputed: bool = False,
    fallback_reason: Optional[str] = None,
) -> FeatureProvenanceRecord:
    """Factory helper to build a validated FeatureProvenanceRecord."""
    return FeatureProvenanceRecord(
        feature_name=feature_name,
        value=value,
        provenance_state=provenance_state,
        source=source,
        transformation=transformation,
        confidence=confidence,
        is_imputed=is_imputed,
        fallback_reason=fallback_reason,
    )


def summarize_provenance(
    provenance_records: Dict[str, FeatureProvenanceRecord]
) -> Dict[str, Any]:
    """
    Summarizes feature distribution across provenance states for underwriting audits.
    """
    state_counts = {
        ProvenanceState.OBSERVED.value: 0,
        ProvenanceState.DERIVED.value: 0,
        ProvenanceState.SELF_REPORTED.value: 0,
        ProvenanceState.UNAVAILABLE.value: 0,
        ProvenanceState.IMPUTED.value: 0,
    }

    feature_manifest = []

    for name in RAW_BORROWER_COLUMNS:
        record = provenance_records.get(name)
        if record:
            state = record.provenance_state
            if state in state_counts:
                state_counts[state] += 1
            feature_manifest.append(record.to_dict())

    total_features = len(RAW_BORROWER_COLUMNS)
    derivable_ratio = (
        (state_counts[ProvenanceState.DERIVED.value] + state_counts[ProvenanceState.OBSERVED.value])
        / total_features
    )

    return {
        "total_features": total_features,
        "state_breakdown": state_counts,
        "derivable_signal_coverage": round(derivable_ratio, 3),
        "manifest": feature_manifest,
    }
