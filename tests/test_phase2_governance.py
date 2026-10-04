"""
CreditBridge - Alternative Credit Scoring Engine
Phase 2: Comprehensive Test Suite for Fairness, Security, Provenance, Drift, and Model Governance
Path: tests/test_phase2_governance.py
"""

import numpy as np
import pandas as pd
import pytest

from src.data_quality_state_machine import (
    LOCKED_HISTORY_POLICY,
    DataQualityState,
    DataQualityStateMachine,
)
from src.drift_monitor import (
    DriftSeverity,
    calculate_categorical_psi,
    calculate_numeric_psi,
    get_drift_severity,
    simulate_monthly_production_batches,
)
from src.fairness_engine import (
    PROTECTED_ATTRIBUTE_METADATA,
    ComprehensiveFairnessAuditReport,
    compare_mitigation_strategies,
    compute_expected_calibration_error,
    compute_kamiran_calders_weights,
    optimize_subgroup_thresholds,
    run_comprehensive_fairness_audit,
    wilson_score_interval,
)
from src.model_governance import (
    ModelHealthState,
    ModelLifecycleStatus,
    ModelRegistry,
    ModelRegistryRecord,
    evaluate_challenger_promotion,
    evaluate_model_health,
    generate_scoring_audit_manifest,
)
from src.provenance_tracker import (
    ProvenanceAssessmentSummary,
    build_provenance_manifest,
)
from src.security_hardening import (
    PERSISTENCE_ENVIRONMENT_DISCLOSURE,
    PICKLE_SECURITY_DISCLOSURE,
    ArtifactIntegrityViolation,
    DangerousPayloadViolation,
    PathTraversalViolation,
    UnsupportedFileTypeViolation,
    audit_and_sanitize_filename,
    compute_file_sha256,
    inspect_magic_bytes,
    sanitize_csv_formula_injection,
    sanitize_error_leakage,
    sanitize_malicious_script_tags,
    validate_secure_upload,
    verify_model_artifact_integrity,
)

# =============================================================================
# 1. FAIRNESS AUDIT & STATISTICAL METRICS TESTS
# =============================================================================

class TestFairnessAuditEngine:
    """Verifies group-level fairness metrics, statistical bounds, and disclosures."""

    def test_wilson_score_interval_properties(self):
        """Wilson score interval must lie within [0, 1] and narrow with sample size."""
        low_50, high_50 = wilson_score_interval(25, 50)
        assert 0.0 <= low_50 <= high_50 <= 1.0
        width_50 = high_50 - low_50

        low_500, high_500 = wilson_score_interval(250, 500)
        width_500 = high_500 - low_500
        # Larger sample size must yield narrower confidence interval
        assert width_500 < width_50

    def test_expected_calibration_error(self):
        """Perfect calibration should yield near-zero ECE."""
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_prob = np.array([0.9, 0.8, 0.1, 0.2, 0.85, 0.15])
        ece = compute_expected_calibration_error(y_true, y_prob)
        assert 0.0 <= ece <= 0.30

    def test_subgroup_fairness_audit_execution(self):
        """Audits across age_group, occupation_type, and city_tier."""
        df = pd.DataFrame({
            "age": [20, 24, 30, 45, 60, 22, 35, 55],
            "occupation_type": ["gig_worker", "gig_worker", "salaried", "salaried", "business", "gig_worker", "salaried", "business"],
            "city_tier": ["tier_3", "tier_3", "tier_1", "tier_1", "tier_2", "tier_3", "tier_1", "tier_2"],
            "predicted_prob": [0.45, 0.40, 0.15, 0.10, 0.25, 0.50, 0.12, 0.30],
            "defaulted": [1, 0, 0, 0, 0, 1, 0, 1],
        })

        report = run_comprehensive_fairness_audit(
            df,
            y_true_col="defaulted",
            prob_col="predicted_prob",
            approval_threshold_prob=0.35,
            air_threshold=0.80,
            min_sample_size=5,  # low threshold for synthetic unit test
            model_name="Test Model",
        )

        assert isinstance(report, ComprehensiveFairnessAuditReport)
        assert report.total_evaluated == 8
        assert "occupation_type" in report.subgroup_metrics
        assert "city_tier" in report.subgroup_metrics
        assert "age_group" in report.subgroup_metrics

        # Protected attribute distinction
        assert "age_group" in PROTECTED_ATTRIBUTE_METADATA
        assert PROTECTED_ATTRIBUTE_METADATA["age_group"]["classification"] == "PROTECTED_DEMOGRAPHIC_ATTRIBUTE"

        # Impossibility theorem disclosure must be present
        assert "MATHEMATICAL IMPOSSIBILITY" in report.impossibility_theorem_disclosure

        # Must never automatically label a model fair
        assert report.governance_state in ("FAIRNESS_REVIEW_REQUIRED", "FAIRNESS_MONITORING")


# =============================================================================
# 2. FAIRNESS MITIGATION EXPERIMENTS TESTS
# =============================================================================

class TestFairnessMitigation:
    """Verifies dual mitigation strategies: reweighting and threshold optimization."""

    def test_kamiran_calders_reweighting(self):
        df = pd.DataFrame({
            "occupation_type": ["informal", "informal", "salaried", "salaried"],
            "defaulted": [1, 0, 0, 0],
        })
        weights = compute_kamiran_calders_weights(df, protected_col="occupation_type", target_col="defaulted")
        assert len(weights) == 4
        assert np.isclose(np.sum(weights), 4.0, atol=1e-3)
        assert all(w > 0.0 for w in weights)

    def test_subgroup_threshold_optimization(self):
        df = pd.DataFrame({
            "occupation_type": ["informal"] * 10 + ["salaried"] * 10,
            "predicted_prob": [0.40] * 10 + [0.20] * 10,
            "defaulted": [1, 0] * 5 + [0, 0] * 5,
        })
        thresholds = optimize_subgroup_thresholds(
            df,
            protected_col="occupation_type",
            prob_col="predicted_prob",
            target_col="defaulted",
            objective="equal_opportunity",
            base_threshold=0.35,
        )
        assert "informal" in thresholds
        assert "salaried" in thresholds
        assert 0.10 <= thresholds["informal"] <= 0.80

    def test_mitigation_comparison_report(self):
        df = pd.DataFrame({
            "occupation_type": ["informal"] * 20 + ["salaried"] * 20,
            "predicted_prob": np.linspace(0.1, 0.6, 40),
            "defaulted": [0, 1] * 20,
        })
        base_probs = df["predicted_prob"].to_numpy()
        rew_probs = np.clip(base_probs * 0.95, 0.05, 0.95)

        comp = compare_mitigation_strategies(
            df,
            baseline_probs=base_probs,
            reweighted_probs=rew_probs,
            protected_col="occupation_type",
            target_col="defaulted",
            base_threshold=0.35,
        )
        assert "profiles" in comp
        assert "baseline" in comp["profiles"]
        assert "reweighted" in comp["profiles"]
        assert "threshold_adjusted" in comp["profiles"]
        assert "tradeoff_analysis" in comp


# =============================================================================
# 3. PROVENANCE & EVIDENCE QUALITY TESTS
# =============================================================================

class TestProvenanceTracker:
    """Verifies feature lifecycle provenance states and data quality gates."""

    def test_provenance_state_classification_and_gate(self):
        # Sample feature dictionary with robust evidence
        features = {
            "monthly_income_estimate": 45000.0,
            "cash_flow_volatility": 1200.0,
            "recharge_frequency_per_month": 3.0,
            "avg_recharge_amount": 299.0,
            "discretionary_spend_ratio": 0.22,
            "savings_buffer_ratio": 0.35,
            "inflow_outflow_ratio": 1.4,
            "utility_bill_payment_regularity": 0.95,
            "monthly_upi_transaction_count": 45,
            "monthly_upi_inflow_amount": 35000.0,
            "monthly_upi_outflow_amount": 25000.0,
            "upi_inflow_consistency_score": 0.88,
            "late_night_transaction_ratio": 0.04,
            "age": 28,
            "occupation_type": "salaried",
            "city_tier": "tier_1",
            "total_credit_events": 25,
            "total_debit_events": 60,
            "statement_history_days": 180,
            "zero_balance_days_ratio": 0.0,
            "bounced_cheque_count": 0,
            "loan_repayment_debits": 5000.0,
        }

        summary = build_provenance_manifest(features)
        assert isinstance(summary, ProvenanceAssessmentSummary)
        assert summary.total_features == 22
        assert summary.evidence_coverage > 0.70
        assert summary.provenance_gate == "PASS"

    def test_provenance_block_on_low_evidence(self):
        # Only 2 features present, rest unavailable
        features = {"age": 25, "city_tier": "tier_2"}
        summary = build_provenance_manifest(features)
        assert summary.evidence_coverage < 0.50
        assert summary.provenance_gate == "BLOCK"


# =============================================================================
# 4. SECURITY HARDENING & PENETRATION CONTROLS TESTS
# =============================================================================

class TestSecurityHardening:
    """Verifies upload validation, formula injection, magic bytes, and path traversal."""

    def test_path_traversal_detection(self):
        with pytest.raises(PathTraversalViolation):
            audit_and_sanitize_filename("../etc/passwd.csv")

        with pytest.raises(PathTraversalViolation):
            audit_and_sanitize_filename("..\\..\\windows\\system32.csv")

    def test_null_byte_rejection(self):
        with pytest.raises(DangerousPayloadViolation):
            audit_and_sanitize_filename("valid.csv\x00.exe")

        with pytest.raises(DangerousPayloadViolation):
            validate_secure_upload(b"date,amount\x00malicious", filename="valid.csv")

    def test_unsupported_file_extension(self):
        with pytest.raises(UnsupportedFileTypeViolation):
            audit_and_sanitize_filename("model_weights.pkl")

    def test_dangerous_magic_bytes(self):
        # PE header (MZ)
        with pytest.raises(DangerousPayloadViolation):
            inspect_magic_bytes(b"MZ\x90\x00\x03\x00\x00\x00")

        # ELF header
        with pytest.raises(DangerousPayloadViolation):
            inspect_magic_bytes(b"\x7fELF\x02\x01\x01\x00")

        # Shell shebang
        with pytest.raises(DangerousPayloadViolation):
            inspect_magic_bytes(b"#!/bin/bash\nrm -rf /")

    def test_csv_formula_injection_sanitization(self):
        assert sanitize_csv_formula_injection("=cmd|'/C calc'!A0") == "'=cmd|'/C calc'!A0"
        assert sanitize_csv_formula_injection("+12345") == "'+12345"
        assert sanitize_csv_formula_injection("@SUM(A1:A10)") == "'@SUM(A1:A10)"
        assert sanitize_csv_formula_injection("Normal text") == "Normal text"

    def test_script_tag_sanitization(self):
        malicious = "<script>alert('pwned')</script>"
        sanitized = sanitize_malicious_script_tags(malicious)
        assert "<script>" not in sanitized

    def test_error_message_leakage_redaction(self):
        raw_error = "FileNotFoundError: C:\\Users\\Administrator\\secret\\data.csv with PAN ABCDE1234F and card 1234567812345678"
        sanitized = sanitize_error_leakage(raw_error)
        assert "ABCDE1234F" not in sanitized
        assert "1234567812345678" not in sanitized
        assert "<sanitized_pan>" in sanitized
        assert "<sanitized_account>" in sanitized

    def test_model_artifact_integrity_verification(self, tmp_path):
        test_file = tmp_path / "test_model.pkl"
        test_file.write_bytes(b"fake model content for testing")
        h = compute_file_sha256(test_file)

        # Verification succeeds with correct hash
        verified_h = verify_model_artifact_integrity(test_file, expected_hash=h)
        assert verified_h == h

        # Verification fails if tampered
        with pytest.raises(ArtifactIntegrityViolation):
            verify_model_artifact_integrity(test_file, expected_hash="0000000000000000000000000000000000000000000000000000000000000000")

    def test_security_disclosures_present(self):
        assert "SECURITY LIMITATION OF PICKLE" in PICKLE_SECURITY_DISCLOSURE
        assert "EXECUTION ENVIRONMENT PERSISTENCE POLICY" in PERSISTENCE_ENVIRONMENT_DISCLOSURE


# =============================================================================
# 5. DATA QUALITY STATE MACHINE & UNIFIED HISTORY POLICY TESTS
# =============================================================================

class TestDataQualityStateMachine:
    """Verifies deterministic PASS / WARN / BLOCK states and locked history policy."""

    def test_locked_history_policy_values(self):
        assert LOCKED_HISTORY_POLICY.min_days_block_threshold == 30
        assert LOCKED_HISTORY_POLICY.marginal_days_threshold == 90
        assert LOCKED_HISTORY_POLICY.min_txns_block_threshold == 15

    def test_insufficient_history_triggers_block(self):
        sm = DataQualityStateMachine()
        assessment = sm.evaluate(
            transaction_count=10,  # <15
            history_days=20,       # <30
            duplicate_rate=0.0,
            unknown_category_share=0.0,
            feature_coverage=0.85,
            missingness_rate=0.0,
            imputation_ratio=0.0,
            invalid_transaction_rate=0.0,
        )
        assert assessment.overall_status == DataQualityState.BLOCK
        assert assessment.is_scoreable is False
        assert any("history" in b.lower() or "days" in b.lower() for b in assessment.blocking_reasons)

    def test_marginal_history_triggers_warn(self):
        sm = DataQualityStateMachine()
        assessment = sm.evaluate(
            transaction_count=35,
            history_days=45,       # 30-89 days -> WARN
            duplicate_rate=0.0,
            unknown_category_share=0.0,
            feature_coverage=0.85,
            missingness_rate=0.0,
            imputation_ratio=0.0,
            invalid_transaction_rate=0.0,
        )
        assert assessment.overall_status == DataQualityState.WARN
        assert assessment.is_scoreable is True
        assert assessment.requires_manual_review is True

    def test_clean_statement_triggers_pass(self):
        sm = DataQualityStateMachine()
        assessment = sm.evaluate(
            transaction_count=85,
            history_days=180,      # >=90 days -> PASS
            duplicate_rate=0.0,
            unknown_category_share=0.02,
            feature_coverage=0.90,
            missingness_rate=0.02,
            imputation_ratio=0.02,
            invalid_transaction_rate=0.0,
        )
        assert assessment.overall_status == DataQualityState.PASS
        assert assessment.is_scoreable is True
        assert assessment.requires_manual_review is False


# =============================================================================
# 6. DRIFT MONITORING TESTS
# =============================================================================

class TestDriftMonitoring:
    """Verifies numeric and categorical PSI calculations and batch simulation."""

    def test_numeric_psi_stable_vs_drifted(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(700, 50, size=1000)
        # Identical distribution
        cur_same = rng.normal(700, 50, size=1000)
        psi_same = calculate_numeric_psi(ref, cur_same)
        assert psi_same < 0.10
        assert get_drift_severity(psi_same) == DriftSeverity.STABLE

        # Drastically shifted distribution
        cur_drift = rng.normal(550, 80, size=1000)
        psi_drift = calculate_numeric_psi(ref, cur_drift)
        assert psi_drift > 0.25
        assert get_drift_severity(psi_drift) == DriftSeverity.CRITICAL

    def test_categorical_psi(self):
        ref = pd.Series(["salaried"] * 50 + ["informal"] * 50)
        cur_same = pd.Series(["salaried"] * 50 + ["informal"] * 50)
        psi = calculate_categorical_psi(ref, cur_same)
        assert psi < 0.05

        cur_drift = pd.Series(["salaried"] * 10 + ["informal"] * 90)
        psi_drift = calculate_categorical_psi(ref, cur_drift)
        assert psi_drift > 0.10

    def test_monthly_batch_simulation(self):
        base_df = pd.DataFrame({
            "monthly_income_estimate": [45000.0] * 50,
            "savings_buffer_ratio": [0.35] * 50,
            "cash_flow_volatility": [1200.0] * 50,
            "discretionary_spend_ratio": [0.20] * 50,
            "defaulted": [0] * 40 + [1] * 10,
        })
        batches = simulate_monthly_production_batches(base_df, num_months=6, batch_size=30)
        assert len(batches) == 6
        assert batches[0]["production_month"].iloc[0] == "Month_01"
        assert batches[5]["production_month"].iloc[0] == "Month_06"


# =============================================================================
# 7. MODEL HEALTH, REGISTRY & CHAMPION/CHALLENGER TESTS
# =============================================================================

class TestModelGovernance:
    """Verifies model health states, single-champion registry, and audit manifests."""

    def test_model_health_evaluation(self):
        # Clean health
        h_clean = evaluate_model_health(
            dq_state=DataQualityState.PASS,
            drift_severity=DriftSeverity.STABLE,
            fairness_state="FAIRNESS_MONITORING",
            roc_auc=0.68,
            ks_stat=24.0,
            min_air_ratio=0.88,
        )
        assert h_clean.overall_health == ModelHealthState.HEALTHY
        assert h_clean.can_serve_inference is True

        # Fail-closed on critical drift or DQ block
        h_block = evaluate_model_health(
            dq_state=DataQualityState.BLOCK,
            drift_severity=DriftSeverity.STABLE,
            fairness_state="FAIRNESS_MONITORING",
        )
        assert h_block.overall_health == ModelHealthState.BLOCK
        assert h_block.can_serve_inference is False

    def test_model_registry_single_champion_invariant(self, tmp_path):
        reg_file = tmp_path / "model_registry.json"
        reg = ModelRegistry(registry_file=reg_file)

        # Register model 1 as CHAMPION
        m1 = ModelRegistryRecord(
            model_version="v1.0.0-lr-baseline",
            experiment_id="exp_001",
            dataset_version="v1",
            feature_schema_version="v1",
            metrics={"roc_auc": 0.624, "ks_statistic": 21.08},
            fairness_summary={"min_air_ratio": 0.82},
            calibration_summary={"brier_score": 0.145},
            artifact_hash_sha256="abc123hash",
            lifecycle_status=ModelLifecycleStatus.CHAMPION,
            approval_state="APPROVED_BY_RISK_COMMITTEE",
            created_timestamp="2026-10-04T00:00:00Z",
        )
        reg.register_model(m1)
        champ1 = reg.get_champion()
        assert champ1 is not None
        assert champ1.model_version == "v1.0.0-lr-baseline"

        # Register model 2 as CHAMPION -> model 1 must automatically demote to RETIRED
        m2 = ModelRegistryRecord(
            model_version="v2.0.0-xgb-champion",
            experiment_id="exp_002",
            dataset_version="v1",
            feature_schema_version="v1",
            metrics={"roc_auc": 0.685, "ks_statistic": 28.5},
            fairness_summary={"min_air_ratio": 0.85},
            calibration_summary={"brier_score": 0.138},
            artifact_hash_sha256="def456hash",
            lifecycle_status=ModelLifecycleStatus.CHAMPION,
            approval_state="APPROVED_BY_RISK_COMMITTEE",
            created_timestamp="2026-10-04T01:00:00Z",
        )
        reg.register_model(m2)
        champ2 = reg.get_champion()
        assert champ2 is not None
        assert champ2.model_version == "v2.0.0-xgb-champion"
        m1_updated = reg.get_model("v1.0.0-lr-baseline")
        assert m1_updated is not None
        assert m1_updated.lifecycle_status == ModelLifecycleStatus.RETIRED

    def test_champion_challenger_promotion_logic(self):
        champion = ModelRegistryRecord(
            model_version="v1.0.0",
            experiment_id="exp_1",
            dataset_version="v1",
            feature_schema_version="v1",
            metrics={"roc_auc": 0.620, "ks_statistic": 20.0, "brier_score": 0.150},
            fairness_summary={"min_air_ratio": 0.85},
            calibration_summary={},
            artifact_hash_sha256="h1",
            lifecycle_status=ModelLifecycleStatus.CHAMPION,
            approval_state="APPROVED_BY_RISK_COMMITTEE",
            created_timestamp="2026-10-04T00:00:00Z",
        )

        # Inferior challenger (lower AUC) -> must be rejected
        bad_challenger = ModelRegistryRecord(
            model_version="v1.1.0",
            experiment_id="exp_2",
            dataset_version="v1",
            feature_schema_version="v1",
            metrics={"roc_auc": 0.615, "ks_statistic": 19.0, "brier_score": 0.155},
            fairness_summary={"min_air_ratio": 0.85},
            calibration_summary={},
            artifact_hash_sha256="h2",
            lifecycle_status=ModelLifecycleStatus.CHALLENGER,
            approval_state="APPROVED_BY_RISK_COMMITTEE",
            created_timestamp="2026-10-04T01:00:00Z",
        )
        res = evaluate_challenger_promotion(champion, bad_challenger)
        assert res.is_promotable is False
        assert any("Insufficient discrimination" in r for r in res.rejection_reasons)

    def test_privacy_safe_scoring_audit_manifest(self):
        manifest = generate_scoring_audit_manifest(
            model_version="v1.0.0-champion",
            feature_schema_version="v2.1",
            data_quality_state="PASS",
            provenance_coverage=0.92,
            credit_score=765,
            calibrated_pd=0.065,
            risk_tier="Low Risk",
            decision="APPROVE",
            policy_version="POL_2026_Q4",
            top_explanations=[{"feature": "savings_buffer_ratio", "impact": "+45 pts"}],
            governance_warnings=[],
            model_artifact_hash="abcde12345hash",
            borrower_seed_str="cust_987654",
        )
        d = manifest.to_dict()
        assert d["credit_score"] == 765
        assert d["decision"] == "APPROVE"
        # Synthetic hashed ID used, customer account number never logged
        assert "cust_987654" not in str(d)
        assert d["anonymized_borrower_hash"].startswith("anon_")
