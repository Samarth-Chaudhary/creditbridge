"""
CreditBridge - Alternative Credit Scoring Engine
Phase 2: Deterministic Data Quality State Machine & Unified History Policy
Path: src/data_quality_state_machine.py

Implements a deterministic PASS / WARN / BLOCK state machine evaluating 9 core data health gates:
1. Transaction count
2. History duration (Unified Institutional Policy: <30d BLOCK, 30-89d WARN, >=90d PASS)
3. Duplicate transaction rate
4. Unknown category share
5. Feature coverage ratio
6. Field missingness
7. Imputation ratio
8. Invalid transaction rate (syntax/balance contradictions)
9. Distribution bounds and outlier flags

Guarantees fail-closed governance: ANY critical BLOCK gate immediately triggers overall BLOCK.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class DataQualityState(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    BLOCK = "BLOCK"


@dataclass(frozen=True)
class HistoryPolicyConfig:
    """Locked institutional history policy encoded in configuration."""
    min_days_block_threshold: int = 30
    min_txns_block_threshold: int = 15
    marginal_days_threshold: int = 90
    optimal_days_threshold: int = 180
    policy_name: str = "CREDITBRIDGE_UNIFIED_HISTORY_POLICY_V2"
    policy_description: str = (
        "Underwriting policy strictly blocks automated scoring for statements under 30 calendar days "
        "or fewer than 15 usable transactions. Statements with 30-89 days history are admitted under "
        "WARN (Marginal Review) status with risk tier capped at Moderate/Review. Statements >= 90 days "
        "are fully scoreable."
    )


LOCKED_HISTORY_POLICY = HistoryPolicyConfig()


@dataclass(frozen=True)
class GateEvaluationResult:
    """Individual data quality gate assessment record."""
    gate_name: str
    observed_value: Any
    unit: str
    status: DataQualityState
    warning_threshold: str
    block_threshold: str
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_name": self.gate_name,
            "observed_value": self.observed_value,
            "unit": self.unit,
            "status": self.status.value,
            "warning_threshold": self.warning_threshold,
            "block_threshold": self.block_threshold,
            "message": self.message,
        }


@dataclass(frozen=True)
class OverallDataQualityAssessment:
    """Comprehensive Data Quality State Machine assessment report."""
    overall_status: DataQualityState
    is_scoreable: bool
    requires_manual_review: bool
    gate_results: Dict[str, GateEvaluationResult]
    blocking_reasons: List[str]
    warning_reasons: List[str]
    history_policy: HistoryPolicyConfig = LOCKED_HISTORY_POLICY

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_status": self.overall_status.value,
            "is_scoreable": self.is_scoreable,
            "requires_manual_review": self.requires_manual_review,
            "gate_results": {k: v.to_dict() for k, v in self.gate_results.items()},
            "blocking_reasons": self.blocking_reasons,
            "warning_reasons": self.warning_reasons,
            "history_policy_name": self.history_policy.policy_name,
            "history_policy_description": self.history_policy.policy_description,
        }


class DataQualityStateMachine:
    """
    Deterministic rule-engine evaluating statement and feature payload quality.
    """

    def __init__(self, history_policy: Optional[HistoryPolicyConfig] = None):
        self.policy = history_policy or LOCKED_HISTORY_POLICY

    def evaluate(
        self,
        transaction_count: int,
        history_days: int,
        duplicate_rate: float,
        unknown_category_share: float,
        feature_coverage: float,
        missingness_rate: float,
        imputation_ratio: float,
        invalid_transaction_rate: float,
        outlier_flags_count: int = 0,
    ) -> OverallDataQualityAssessment:
        """
        Executes all 9 gates and computes the deterministic final state.
        """
        gates: Dict[str, GateEvaluationResult] = {}
        blockers: List[str] = []
        warnings: List[str] = []

        # 1. Transaction Count Gate
        if transaction_count < self.policy.min_txns_block_threshold:
            s1 = DataQualityState.BLOCK
            m1 = f"Transaction count ({transaction_count}) < minimum threshold ({self.policy.min_txns_block_threshold})."
            blockers.append(m1)
        elif transaction_count < 30:
            s1 = DataQualityState.WARN
            m1 = f"Transaction count ({transaction_count}) is modest (30 recommended)."
            warnings.append(m1)
        else:
            s1 = DataQualityState.PASS
            m1 = f"Transaction count ({transaction_count}) is robust."
        gates["transaction_count"] = GateEvaluationResult(
            "transaction_count", transaction_count, "count", s1, "< 30", f"< {self.policy.min_txns_block_threshold}", m1
        )

        # 2. History Duration Gate (Unified Policy)
        if history_days < self.policy.min_days_block_threshold:
            s2 = DataQualityState.BLOCK
            m2 = (
                f"Statement history ({history_days} days) < mandatory minimum ({self.policy.min_days_block_threshold} days). "
                f"Policy: {self.policy.policy_name} strictly blocks scoring."
            )
            blockers.append(m2)
        elif history_days < self.policy.marginal_days_threshold:
            s2 = DataQualityState.WARN
            m2 = (
                f"Statement history ({history_days} days) is marginal (30-89 days). "
                "Admitted with mandatory manual review cap."
            )
            warnings.append(m2)
        else:
            s2 = DataQualityState.PASS
            m2 = f"Statement history ({history_days} days) is adequate (>= {self.policy.marginal_days_threshold} days)."
        gates["history_duration"] = GateEvaluationResult(
            "history_duration", history_days, "days", s2, f"< {self.policy.marginal_days_threshold}", f"< {self.policy.min_days_block_threshold}", m2
        )

        # 3. Duplicate Rate Gate
        if duplicate_rate > 0.05:
            s3 = DataQualityState.BLOCK
            m3 = f"Duplicate transaction rate ({duplicate_rate:.1%}) exceeds critical threshold (5.0%)."
            blockers.append(m3)
        elif duplicate_rate > 0.01:
            s3 = DataQualityState.WARN
            m3 = f"Duplicate transaction rate ({duplicate_rate:.1%}) is elevated (1.0% - 5.0%)."
            warnings.append(m3)
        else:
            s3 = DataQualityState.PASS
            m3 = f"Duplicate transaction rate ({duplicate_rate:.1%}) within acceptable limits (<= 1.0%)."
        gates["duplicate_rate"] = GateEvaluationResult(
            "duplicate_rate", round(duplicate_rate, 4), "ratio", s3, "> 0.01", "> 0.05", m3
        )

        # 4. Unknown Category Share Gate
        if unknown_category_share > 0.20:
            s4 = DataQualityState.BLOCK
            m4 = f"Unclassified transaction share ({unknown_category_share:.1%}) exceeds 20.0% critical bound."
            blockers.append(m4)
        elif unknown_category_share > 0.05:
            s4 = DataQualityState.WARN
            m4 = f"Unclassified transaction share ({unknown_category_share:.1%}) is elevated (5.0% - 20.0%)."
            warnings.append(m4)
        else:
            s4 = DataQualityState.PASS
            m4 = f"Unclassified transaction share ({unknown_category_share:.1%}) is low (<= 5.0%)."
        gates["unknown_category_share"] = GateEvaluationResult(
            "unknown_category_share", round(unknown_category_share, 4), "ratio", s4, "> 0.05", "> 0.20", m4
        )

        # 5. Feature Coverage Gate
        if feature_coverage < 0.50:
            s5 = DataQualityState.BLOCK
            m5 = f"Verifiable feature coverage ({feature_coverage:.1%}) < 50.0% minimum underwriting standard."
            blockers.append(m5)
        elif feature_coverage < 0.70:
            s5 = DataQualityState.WARN
            m5 = f"Verifiable feature coverage ({feature_coverage:.1%}) is moderate (50.0% - 70.0%)."
            warnings.append(m5)
        else:
            s5 = DataQualityState.PASS
            m5 = f"Verifiable feature coverage ({feature_coverage:.1%}) is strong (>= 70.0%)."
        gates["feature_coverage"] = GateEvaluationResult(
            "feature_coverage", round(feature_coverage, 4), "ratio", s5, "< 0.70", "< 0.50", m5
        )

        # 6. Missingness Rate Gate
        if missingness_rate > 0.30:
            s6 = DataQualityState.BLOCK
            m6 = f"Field missingness ({missingness_rate:.1%}) exceeds 30.0% critical bound."
            blockers.append(m6)
        elif missingness_rate > 0.10:
            s6 = DataQualityState.WARN
            m6 = f"Field missingness ({missingness_rate:.1%}) is elevated (10.0% - 30.0%)."
            warnings.append(m6)
        else:
            s6 = DataQualityState.PASS
            m6 = f"Field missingness ({missingness_rate:.1%}) is minimal (<= 10.0%)."
        gates["missingness_rate"] = GateEvaluationResult(
            "missingness_rate", round(missingness_rate, 4), "ratio", s6, "> 0.10", "> 0.30", m6
        )

        # 7. Imputation Ratio Gate
        if imputation_ratio > 0.30:
            s7 = DataQualityState.BLOCK
            m7 = f"Imputation ratio ({imputation_ratio:.1%}) exceeds 30.0% allowable bound."
            blockers.append(m7)
        elif imputation_ratio > 0.10:
            s7 = DataQualityState.WARN
            m7 = f"Imputation ratio ({imputation_ratio:.1%}) is elevated (10.0% - 30.0%)."
            warnings.append(m7)
        else:
            s7 = DataQualityState.PASS
            m7 = f"Imputation ratio ({imputation_ratio:.1%}) is low (<= 10.0%)."
        gates["imputation_ratio"] = GateEvaluationResult(
            "imputation_ratio", round(imputation_ratio, 4), "ratio", s7, "> 0.10", "> 0.30", m7
        )

        # 8. Invalid Transaction Rate Gate
        if invalid_transaction_rate > 0.10:
            s8 = DataQualityState.BLOCK
            m8 = f"Invalid/contradictory transaction rate ({invalid_transaction_rate:.1%}) exceeds 10.0% critical bound."
            blockers.append(m8)
        elif invalid_transaction_rate > 0.02:
            s8 = DataQualityState.WARN
            m8 = f"Invalid/contradictory transaction rate ({invalid_transaction_rate:.1%}) is elevated (2.0% - 10.0%)."
            warnings.append(m8)
        else:
            s8 = DataQualityState.PASS
            m8 = f"Invalid transaction rate ({invalid_transaction_rate:.1%}) is low (<= 2.0%)."
        gates["invalid_transaction_rate"] = GateEvaluationResult(
            "invalid_transaction_rate", round(invalid_transaction_rate, 4), "ratio", s8, "> 0.02", "> 0.10", m8
        )

        # 9. Distribution Checks Gate
        if outlier_flags_count >= 6:
            s9 = DataQualityState.BLOCK
            m9 = f"Extreme out-of-distribution anomaly flags ({outlier_flags_count}) exceed severe risk limit (>=6)."
            blockers.append(m9)
        elif outlier_flags_count >= 3:
            s9 = DataQualityState.WARN
            m9 = f"Multiple distribution anomaly flags ({outlier_flags_count}) detected (3-5)."
            warnings.append(m9)
        else:
            s9 = DataQualityState.PASS
            m9 = f"Distribution anomaly flags ({outlier_flags_count}) within normal variance (< 3)."
        gates["distribution_checks"] = GateEvaluationResult(
            "distribution_checks", outlier_flags_count, "count", s9, ">= 3", ">= 6", m9
        )

        # Deterministic State Aggregation (Fail-Closed)
        if len(blockers) > 0:
            overall_status = DataQualityState.BLOCK
            is_scoreable = False
            requires_manual_review = True
        elif len(warnings) > 0:
            overall_status = DataQualityState.WARN
            is_scoreable = True
            requires_manual_review = True
        else:
            overall_status = DataQualityState.PASS
            is_scoreable = True
            requires_manual_review = False

        return OverallDataQualityAssessment(
            overall_status=overall_status,
            is_scoreable=is_scoreable,
            requires_manual_review=requires_manual_review,
            gate_results=gates,
            blocking_reasons=blockers,
            warning_reasons=warnings,
            history_policy=self.policy,
        )
