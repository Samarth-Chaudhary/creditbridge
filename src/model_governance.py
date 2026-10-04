"""
CreditBridge - Alternative Credit Scoring Engine
Phase 2: Model Governance, Health States, Model Registry & Audit Manifests
Path: src/model_governance.py

Provides:
1. Model Health State Machine:
   - Evaluates Data Quality, Drift, Fairness, and Performance gates.
   - Computes deterministic states: HEALTHY / MONITOR / REVIEW / BLOCK.
   - Fail-closed enforcement on any critical governance failure.
2. Lightweight Model Registry:
   - Stores versioned model records with SHA-256 artifact hashes, schemas, and metrics.
   - Enforces lifecycle states: CANDIDATE / CHALLENGER / CHAMPION / RETIRED / BLOCKED.
   - Strict rule: ONLY 'CHAMPION' can serve default production inference.
3. Champion / Challenger Deterministic Selection Engine:
   - Objective multi-criteria evaluation (AUC, KS, Brier, Fairness, PSI stability, Complexity).
   - Prevents silent replacement; mandates formal decision log and approval rationale.
4. Privacy-Safe Scoring Audit Manifest:
   - Produces comprehensive per-request audit records with synthetic IDs and zero raw PII.
"""

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.data_quality_state_machine import DataQualityState
from src.drift_monitor import DriftSeverity

# -----------------------------------------------------------------------------
# 1. MODEL HEALTH STATE MACHINE
# -----------------------------------------------------------------------------

class ModelHealthState(str, Enum):
    HEALTHY = "HEALTHY"
    MONITOR = "MONITOR"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"


@dataclass(frozen=True)
class ModelHealthAssessment:
    """Consolidated model operational health status across 4 governance gates."""
    overall_health: ModelHealthState
    can_serve_inference: bool
    data_quality_gate: str
    drift_gate: str
    fairness_gate: str
    performance_gate: str
    failure_reasons: List[str]
    monitoring_warnings: List[str]
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_health": self.overall_health.value,
            "can_serve_inference": self.can_serve_inference,
            "data_quality_gate": self.data_quality_gate,
            "drift_gate": self.drift_gate,
            "fairness_gate": self.fairness_gate,
            "performance_gate": self.performance_gate,
            "failure_reasons": self.failure_reasons,
            "monitoring_warnings": self.monitoring_warnings,
            "timestamp_utc": self.timestamp_utc,
        }


def evaluate_model_health(
    dq_state: DataQualityState,
    drift_severity: DriftSeverity,
    fairness_state: str,  # "FAIRNESS_MONITORING" or "FAIRNESS_REVIEW_REQUIRED"
    roc_auc: Optional[float] = None,
    ks_stat: Optional[float] = None,
    min_air_ratio: Optional[float] = None,
) -> ModelHealthAssessment:
    """
    Evaluates operational health across Data Quality, Drift, Fairness, and Performance.
    Enforces fail-closed: ANY critical gate failure blocks inference.
    """
    failures: List[str] = []
    warnings: List[str] = []

    # 1. Data Quality Gate
    if dq_state == DataQualityState.BLOCK:
        failures.append("DATA_QUALITY_GATE_BLOCK: Input data breached critical quality thresholds.")
    elif dq_state == DataQualityState.WARN:
        warnings.append("DATA_QUALITY_GATE_WARN: Input data exhibits marginal history or elevated missingness.")

    # 2. Drift Gate
    if drift_severity == DriftSeverity.CRITICAL:
        failures.append("DRIFT_GATE_CRITICAL: Population or score drift (PSI >= 0.25) exceeds stability limits.")
    elif drift_severity == DriftSeverity.WARNING:
        warnings.append("DRIFT_GATE_WARNING: Moderate drift (0.10 <= PSI < 0.25) observed; monitor distribution shifts.")

    # 3. Fairness Gate
    if fairness_state == "FAIRNESS_REVIEW_REQUIRED":
        warnings.append("FAIRNESS_GATE_REVIEW: Subgroup AIR or TPR disparity breached thresholds. Human review required.")
    if min_air_ratio is not None and min_air_ratio < 0.60:
        failures.append(f"FAIRNESS_GATE_BLOCK: Severe Adverse Impact Ratio ({min_air_ratio:.2f} < 0.60) detected.")

    # 4. Performance Gate
    if roc_auc is not None:
        if roc_auc < 0.55:
            failures.append(f"PERFORMANCE_GATE_BLOCK: Model ROC-AUC ({roc_auc:.4f}) fell below discriminatory floor (0.55).")
        elif roc_auc < 0.60:
            warnings.append(f"PERFORMANCE_GATE_WARN: Model ROC-AUC ({roc_auc:.4f}) is degraded (< 0.60).")

    if ks_stat is not None and ks_stat < 15.0:
        warnings.append(f"PERFORMANCE_GATE_WARN: KS separation ({ks_stat:.2f}%) below institutional target (18.0%).")

    # State Aggregation
    if len(failures) > 0:
        overall = ModelHealthState.BLOCK
        can_serve = False
    elif any("FAIRNESS_GATE_REVIEW" in w or drift_severity == DriftSeverity.WARNING for w in warnings):
        overall = ModelHealthState.REVIEW if "FAIRNESS_GATE_REVIEW" in "".join(warnings) else ModelHealthState.MONITOR
        can_serve = True
    elif len(warnings) > 0:
        overall = ModelHealthState.MONITOR
        can_serve = True
    else:
        overall = ModelHealthState.HEALTHY
        can_serve = True

    return ModelHealthAssessment(
        overall_health=overall,
        can_serve_inference=can_serve,
        data_quality_gate=dq_state.value,
        drift_gate=drift_severity.value,
        fairness_gate=fairness_state,
        performance_gate="PASS" if (roc_auc is None or roc_auc >= 0.60) else ("WARN" if roc_auc >= 0.55 else "FAIL"),
        failure_reasons=failures,
        monitoring_warnings=warnings,
    )


# -----------------------------------------------------------------------------
# 2. MODEL REGISTRY
# -----------------------------------------------------------------------------

class ModelLifecycleStatus(str, Enum):
    CANDIDATE = "CANDIDATE"
    CHALLENGER = "CHALLENGER"
    CHAMPION = "CHAMPION"
    RETIRED = "RETIRED"
    BLOCKED = "BLOCKED"


@dataclass
class ModelRegistryRecord:
    """Standardized institutional metadata record for a trained model in the registry."""
    model_version: str
    experiment_id: str
    dataset_version: str
    feature_schema_version: str
    metrics: Dict[str, float]
    fairness_summary: Dict[str, Any]
    calibration_summary: Dict[str, Any]
    artifact_hash_sha256: str
    lifecycle_status: ModelLifecycleStatus
    approval_state: str  # e.g., "APPROVED_FOR_PRODUCTION", "PENDING_VALIDATION", "REJECTED"
    created_timestamp: str
    model_type: str = "LogisticRegression"
    decision_rationale: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_version": self.model_version,
            "experiment_id": self.experiment_id,
            "dataset_version": self.dataset_version,
            "feature_schema_version": self.feature_schema_version,
            "model_type": self.model_type,
            "metrics": self.metrics,
            "fairness_summary": self.fairness_summary,
            "calibration_summary": self.calibration_summary,
            "artifact_hash_sha256": self.artifact_hash_sha256,
            "lifecycle_status": self.lifecycle_status.value,
            "approval_state": self.approval_state,
            "decision_rationale": self.decision_rationale,
            "created_timestamp": self.created_timestamp,
        }


class ModelRegistry:
    """
    Lightweight, filesystem-backed institutional model registry.
    Enforces the single-champion invariant: ONLY the model marked CHAMPION can serve default inference.
    """

    def __init__(self, registry_file: Optional[Union[str, Path]] = None):
        if registry_file is None:
            project_root = Path(__file__).resolve().parent.parent
            self.registry_path = project_root / "registry" / "model_registry.json"
        else:
            self.registry_path = Path(registry_file)

        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.records: Dict[str, ModelRegistryRecord] = {}
        self._load()

    def _load(self) -> None:
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for k, v in data.items():
                        self.records[k] = ModelRegistryRecord(
                            model_version=v["model_version"],
                            experiment_id=v["experiment_id"],
                            dataset_version=v["dataset_version"],
                            feature_schema_version=v["feature_schema_version"],
                            metrics=v.get("metrics", {}),
                            fairness_summary=v.get("fairness_summary", {}),
                            calibration_summary=v.get("calibration_summary", {}),
                            artifact_hash_sha256=v["artifact_hash_sha256"],
                            lifecycle_status=ModelLifecycleStatus(v["lifecycle_status"]),
                            approval_state=v["approval_state"],
                            created_timestamp=v["created_timestamp"],
                            model_type=v.get("model_type", "LogisticRegression"),
                            decision_rationale=v.get("decision_rationale"),
                        )
            except Exception:
                self.records = {}

    def save(self) -> None:
        data = {k: v.to_dict() for k, v in self.records.items()}
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def register_model(self, record: ModelRegistryRecord) -> None:
        """Registers a new model version. Enforces single-champion rule."""
        if record.lifecycle_status == ModelLifecycleStatus.CHAMPION:
            # Demote any existing champion to retired
            for v_name, rec in self.records.items():
                if rec.lifecycle_status == ModelLifecycleStatus.CHAMPION and v_name != record.model_version:
                    self.records[v_name] = ModelRegistryRecord(
                        model_version=rec.model_version,
                        experiment_id=rec.experiment_id,
                        dataset_version=rec.dataset_version,
                        feature_schema_version=rec.feature_schema_version,
                        metrics=rec.metrics,
                        fairness_summary=rec.fairness_summary,
                        calibration_summary=rec.calibration_summary,
                        artifact_hash_sha256=rec.artifact_hash_sha256,
                        lifecycle_status=ModelLifecycleStatus.RETIRED,
                        approval_state=rec.approval_state,
                        created_timestamp=rec.created_timestamp,
                        model_type=rec.model_type,
                        decision_rationale=f"Superseded by new champion: {record.model_version}",
                    )
        self.records[record.model_version] = record
        self.save()

    def get_champion(self) -> Optional[ModelRegistryRecord]:
        """Returns the currently active champion model record."""
        for rec in self.records.values():
            if rec.lifecycle_status == ModelLifecycleStatus.CHAMPION:
                return rec
        return None

    def get_model(self, model_version: str) -> Optional[ModelRegistryRecord]:
        return self.records.get(model_version)

    def list_models(self) -> List[ModelRegistryRecord]:
        return list(self.records.values())


# -----------------------------------------------------------------------------
# 3. CHAMPION / CHALLENGER SELECTION ENGINE
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ChampionChallengerComparison:
    champion_version: str
    challenger_version: str
    auc_diff: float
    ks_diff: float
    brier_diff: float
    fairness_air_diff: float
    is_promotable: bool
    rejection_reasons: List[str]
    selection_rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "champion_version": self.champion_version,
            "challenger_version": self.challenger_version,
            "auc_diff": round(self.auc_diff, 4),
            "ks_diff": round(self.ks_diff, 4),
            "brier_diff": round(self.brier_diff, 4),
            "fairness_air_diff": round(self.fairness_air_diff, 4),
            "is_promotable": self.is_promotable,
            "rejection_reasons": self.rejection_reasons,
            "selection_rationale": self.selection_rationale,
        }


def evaluate_challenger_promotion(
    champion: ModelRegistryRecord,
    challenger: ModelRegistryRecord,
    min_auc_gain: float = 0.010,
    max_brier_degradation: float = 0.005,
    max_air_degradation: float = 0.050,
) -> ChampionChallengerComparison:
    """
    Executes deterministic champion/challenger comparison:
    1. Challenger must achieve at least min_auc_gain (e.g. +0.01 AUC)
    2. Challenger cannot degrade Brier score by more than max_brier_degradation
    3. Challenger cannot degrade minimum subgroup AIR by more than max_air_degradation
    4. Records exact rationale for governance transparency.
    """
    champ_auc = champion.metrics.get("roc_auc", 0.0)
    chal_auc = challenger.metrics.get("roc_auc", 0.0)
    auc_diff = chal_auc - champ_auc

    champ_ks = champion.metrics.get("ks_statistic", 0.0)
    chal_ks = challenger.metrics.get("ks_statistic", 0.0)
    ks_diff = chal_ks - champ_ks

    champ_brier = champion.metrics.get("brier_score", 1.0)
    chal_brier = challenger.metrics.get("brier_score", 1.0)
    brier_diff = chal_brier - champ_brier  # Negative is better

    champ_air = champion.fairness_summary.get("min_air_ratio", 1.0)
    chal_air = challenger.fairness_summary.get("min_air_ratio", 1.0)
    air_diff = chal_air - champ_air        # Positive is better

    rejections: List[str] = []

    if auc_diff < min_auc_gain:
        rejections.append(
            f"Insufficient discrimination improvement: ΔAUC={auc_diff:+.4f} < required +{min_auc_gain:.3f}."
        )

    if brier_diff > max_brier_degradation:
        rejections.append(
            f"Probability calibration degraded: ΔBrier={brier_diff:+.4f} > allowable +{max_brier_degradation:.3f}."
        )

    if air_diff < -max_air_degradation:
        rejections.append(
            f"Fairness disparate impact exacerbated: ΔAIR={air_diff:+.4f} (degradation exceeds {max_air_degradation:.2f})."
        )

    if challenger.approval_state != "APPROVED_BY_RISK_COMMITTEE":
        rejections.append(
            f"Challenger approval state is '{challenger.approval_state}', not 'APPROVED_BY_RISK_COMMITTEE'."
        )

    is_promotable = len(rejections) == 0

    if is_promotable:
        rationale = (
            f"PROMOTION_APPROVED: Challenger '{challenger.model_version}' superior to Champion '{champion.model_version}'. "
            f"Gains: ΔAUC={auc_diff:+.4f}, ΔKS={ks_diff:+.2f}%, with stable calibration (ΔBrier={brier_diff:+.4f}) "
            f"and preserved fairness (ΔAIR={air_diff:+.4f})."
        )
    else:
        rationale = (
            f"PROMOTION_DENIED: Challenger '{challenger.model_version}' failed {len(rejections)} governance gate(s). "
            f"Champion '{champion.model_version}' retained."
        )

    return ChampionChallengerComparison(
        champion_version=champion.model_version,
        challenger_version=challenger.model_version,
        auc_diff=auc_diff,
        ks_diff=ks_diff,
        brier_diff=brier_diff,
        fairness_air_diff=air_diff,
        is_promotable=is_promotable,
        rejection_reasons=rejections,
        selection_rationale=rationale,
    )


# -----------------------------------------------------------------------------
# 4. PRIVACY-SAFE SCORING AUDIT MANIFEST
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ScoringAuditManifest:
    """
    Individual transaction scoring audit record.
    Provides complete institutional trace without persisting raw statement PII.
    """
    request_id: str
    model_version: str
    feature_schema_version: str
    timestamp_utc: str
    data_quality_state: str
    provenance_coverage_ratio: float
    credit_score: int
    calibrated_pd: float
    risk_tier: str
    decision: str  # "APPROVE", "MANUAL_REVIEW", "REJECT"
    policy_version: str
    top_explanations: List[Dict[str, Any]]
    governance_warnings: List[str]
    model_artifact_hash: str
    anonymized_borrower_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "model_version": self.model_version,
            "feature_schema_version": self.feature_schema_version,
            "timestamp_utc": self.timestamp_utc,
            "data_quality_state": self.data_quality_state,
            "provenance_coverage_ratio": round(self.provenance_coverage_ratio, 4),
            "credit_score": self.credit_score,
            "calibrated_pd": round(self.calibrated_pd, 4),
            "risk_tier": self.risk_tier,
            "decision": self.decision,
            "policy_version": self.policy_version,
            "top_explanations": self.top_explanations,
            "governance_warnings": self.governance_warnings,
            "model_artifact_hash": self.model_artifact_hash,
            "anonymized_borrower_hash": self.anonymized_borrower_hash,
        }


def generate_scoring_audit_manifest(
    model_version: str,
    feature_schema_version: str,
    data_quality_state: str,
    provenance_coverage: float,
    credit_score: int,
    calibrated_pd: float,
    risk_tier: str,
    decision: str,
    policy_version: str,
    top_explanations: List[Dict[str, Any]],
    governance_warnings: List[str],
    model_artifact_hash: str,
    borrower_seed_str: Optional[str] = None,
) -> ScoringAuditManifest:
    """
    Creates a cryptographically traceable scoring audit record with synthetic anonymized identifier.
    Zero raw statement text, PAN, or account numbers are stored.
    """
    req_id = str(uuid.uuid4())
    salt = "CREDITBRIDGE_ANON_SALT_V2"
    seed = borrower_seed_str or req_id
    anon_id = hashlib.sha256(f"{seed}:{salt}".encode("utf-8")).hexdigest()[:16]

    return ScoringAuditManifest(
        request_id=req_id,
        model_version=model_version,
        feature_schema_version=feature_schema_version,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        data_quality_state=data_quality_state,
        provenance_coverage_ratio=provenance_coverage,
        credit_score=credit_score,
        calibrated_pd=calibrated_pd,
        risk_tier=risk_tier,
        decision=decision,
        policy_version=policy_version,
        top_explanations=top_explanations[:5],  # Top 5 reason codes
        governance_warnings=governance_warnings,
        model_artifact_hash=model_artifact_hash,
        anonymized_borrower_hash=f"anon_{anon_id}",
    )
