"""
CreditBridge - Phase 1 Scientific Reconstruction Test Suite
Path: tests/test_phase1_scientific_reconstruction.py

Tests for:
1. Temporal dataset contract and anti-leakage defenses (deliberate future injection rejection).
2. Preprocessing leakage prevention (pipeline fits strictly on train split).
3. Reproducibility with identical random seeds.
4. Artifact hash stability (SHA-256 verification against artifact_hash.txt).
5. Rigorous metric generation correctness (ROC-AUC, PR-AUC, KS, Gini, Brier).
6. CreditBridge Risk Score bounds ([300, 900]) and monotonicity.
7. Calibration output validity (Platt/isotonic, bounds in [0, 1]).
8. Champion selection rules and NO-CHAMPION deterministic fallback.
9. Economic decisioning engine (Expected Loss = PD * LGD * EAD, policy simulation).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.economic_decisioning import (
    DecisionPolicyConfig,
    compute_expected_loss,
    make_underwriting_decision,
    simulate_policy_tradeoffs,
)
from src.evaluation_engine import (
    ProbabilityCalibrator,
    compute_full_metrics,
    evaluate_deterministic_champion,
)
from src.experiment_engine import (
    compute_file_sha256,
)
from src.feature_engineering import FeaturePipeline
from src.scoring_utils import (
    MAX_SCORE,
    MIN_SCORE,
    probability_to_credit_score,
    score_to_tier,
)
from src.temporal_contract import (
    TemporalLeakageError,
    TemporalWindowContract,
    assert_zero_temporal_leakage,
    split_temporal_dataset,
)
from src.temporal_data_generator import (
    generate_full_temporal_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# =============================================================================
# 1. TEMPORAL DATASET CONTRACT & LEAKAGE DEFENSE TESTS
# =============================================================================

class TestTemporalLeakageDefense:
    """Verifies that future transactions are prevented from entering feature sets."""

    def test_clean_observation_window_passes(self):
        """Transactions entirely before cutoff must pass without error."""
        contract = TemporalWindowContract(
            observation_months=12,
            prediction_months=3,
        )
        cutoff = "2026-06-01"
        clean_tx = pd.DataFrame({
            "date": ["2025-06-02", "2025-09-15", "2026-01-10", "2026-05-31"],
            "amount": [100.0, 250.0, 50.0, 400.0],
        })

        # contract validation returns True
        valid, msg = contract.validate_transaction_dates(clean_tx["date"], cutoff)
        assert valid is True
        assert msg is None

        # assert_zero_temporal_leakage returns clean df
        res = assert_zero_temporal_leakage(clean_tx, cutoff_date=cutoff)
        assert len(res) == 4

    def test_future_transaction_injection_triggers_leakage_error(self):
        """Deliberately injecting a transaction occurring after cutoff must raise TemporalLeakageError."""
        cutoff = "2026-06-01"
        leaked_tx = pd.DataFrame({
            "date": ["2025-08-01", "2026-01-15", "2026-05-20", "2026-06-15"],  # June 15 is after June 1
            "amount": [100.0, 200.0, 300.0, 500.0],
        })

        contract = TemporalWindowContract()
        valid, msg = contract.validate_transaction_dates(leaked_tx["date"], cutoff)
        assert valid is False
        assert msg is not None
        assert "TEMPORAL LEAKAGE DETECTED" in msg

        with pytest.raises(TemporalLeakageError) as exc_info:
            assert_zero_temporal_leakage(leaked_tx, cutoff_date=cutoff)

        assert "TEMPORAL_LEAKAGE_REJECTED" in str(exc_info.value)

    def test_split_temporal_dataset_partitions_correctly(self):
        """Verifies that temporal splitting produces disjoint train, val, and oot sets."""
        df = pd.DataFrame({
            "borrower_id": [f"B_{i}" for i in range(9)],
            "cohort_split": ["train", "train", "train", "train", "val", "val", "oot", "oot", "oot"],
            "val": range(9),
        })

        train_df, val_df, oot_df = split_temporal_dataset(df)
        assert len(train_df) == 4
        assert len(val_df) == 2
        assert len(oot_df) == 3

        # Must raise if split is empty
        invalid_df = df[df["cohort_split"] != "oot"]
        with pytest.raises(ValueError, match="Invalid temporal split sizes"):
            split_temporal_dataset(invalid_df)



# =============================================================================
# 2. PREPROCESSING LEAKAGE PREVENTION TESTS
# =============================================================================

class TestPreprocessingLeakage:
    """Verifies all preprocessors fit exclusively on training data."""

    def test_feature_pipeline_fit_only_on_train(self):
        """Checks that FeaturePipeline stores statistics computed only on training data."""
        df = generate_full_temporal_dataset(seed=42, train_count=100, val_count=50, oot_count=50)
        train_df, val_df, _ = split_temporal_dataset(df)

        pipeline = FeaturePipeline()
        pipeline.fit(train_df)

        # Store fitted statistics
        assert pipeline.numeric_imputer.statistics_ is not None
        train_imputed_stats = np.array(pipeline.numeric_imputer.statistics_).copy()

        # Transform validation split (which may contain different distributions or missingness)
        val_transformed = pipeline.transform(val_df)

        # The pipeline's internal fitted state must remain unchanged
        np.testing.assert_array_equal(pipeline.numeric_imputer.statistics_, train_imputed_stats)
        assert len(val_transformed) == len(val_df)
        assert not bool(val_transformed.isna().to_numpy().any())



# =============================================================================
# 3. REPRODUCIBILITY TESTS
# =============================================================================

class TestReproducibility:
    """Verifies that synthetic data generation with identical seeds produces bitwise identical data."""

    def test_synthetic_data_generator_identical_seeds(self):
        """Generating synthetic data twice with seed=123 must yield exact identical DataFrames."""
        df1 = generate_full_temporal_dataset(seed=123, train_count=100, val_count=50, oot_count=50)
        df2 = generate_full_temporal_dataset(seed=123, train_count=100, val_count=50, oot_count=50)

        pd.testing.assert_frame_equal(df1, df2)

    def test_synthetic_data_generator_different_seeds(self):
        """Generating synthetic data with different seeds must produce different data."""
        df1 = generate_full_temporal_dataset(seed=123, train_count=100, val_count=50, oot_count=50)
        df2 = generate_full_temporal_dataset(seed=456, train_count=100, val_count=50, oot_count=50)

        with pytest.raises(AssertionError):
            pd.testing.assert_frame_equal(df1, df2)



# =============================================================================
# 4. ARTIFACT HASH STABILITY TESTS
# =============================================================================

class TestArtifactHashStability:
    """Verifies that experiment artifacts match their SHA-256 signatures."""

    def test_latest_experiment_hashes_valid(self):
        """Reads artifact_hash.txt for the latest experiment and verifies each file's SHA-256."""
        latest_pointer = PROJECT_ROOT / "experiments" / "latest_champion.json"
        if not latest_pointer.exists():
            pytest.skip("latest_champion.json not yet generated.")

        with open(latest_pointer, "r", encoding="utf-8") as f:
            data = json.load(f)

        exp_dir = Path(data["experiment_dir"])
        hash_file = exp_dir / "artifact_hash.txt"
        assert hash_file.exists(), f"artifact_hash.txt missing in {exp_dir}"

        with open(hash_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) >= 8, f"Expected at least 8 hashed artifacts, found {len(lines)}"

        for line in lines:
            line = line.strip()
            if not line:
                continue
            expected_hash, fname = line.split("  ", 1)
            target_path = exp_dir / fname
            assert target_path.exists(), f"Artifact file {fname} not found in {exp_dir}"
            actual_hash = compute_file_sha256(target_path)
            assert actual_hash == expected_hash, (
                f"Hash mismatch for {fname}: expected {expected_hash}, got {actual_hash}"
            )


# =============================================================================
# 5. METRIC COMPUTATION CORRECTNESS TESTS
# =============================================================================

class TestMetricComputation:
    """Verifies standard classification and credit risk metrics."""

    def test_perfect_predictions_metrics(self):
        """Checks metrics on a perfectly separable prediction."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_prob = np.array([0.05, 0.10, 0.15, 0.20, 0.80, 0.85, 0.90, 0.95])

        metrics = compute_full_metrics(y_true, y_prob)

        assert metrics["roc_auc"] == pytest.approx(1.0, abs=1e-4)
        assert metrics["pr_auc"] == pytest.approx(1.0, abs=1e-4)
        assert metrics["ks_statistic"] == pytest.approx(100.0, abs=1e-2)
        assert metrics["gini"] == pytest.approx(1.0, abs=1e-4)
        assert metrics["brier_score"] < 0.05

    def test_imbalanced_pr_auc_mandatory(self):
        """Confirms PR-AUC reflects baseline prevalence under imbalanced classes."""
        y_true = np.array([0] * 90 + [1] * 10)  # 10% positive prevalence
        y_prob_random = np.full(100, 0.10)  # Random uninformative predictions

        metrics = compute_full_metrics(y_true, y_prob_random)
        # For an uninformative model, PR-AUC is approximately equal to prevalence (~0.10)
        assert metrics["pr_auc"] == pytest.approx(0.10, abs=0.05)


# =============================================================================
# 6. SCORE BOUNDS & MONOTONICITY TESTS
# =============================================================================

class TestCreditBridgeRiskScore:
    """Verifies presentation-layer score behavior (300-900 scale)."""

    def test_score_strictly_within_bounds(self):
        """Scores must be strictly clamped between 300 and 900."""
        probs = [0.0, 1e-6, 0.01, 0.14, 0.50, 0.90, 1.0 - 1e-6, 1.0]
        scores = [probability_to_credit_score(p) for p in probs]

        for s in scores:
            assert isinstance(s, int)
            assert MIN_SCORE <= s <= MAX_SCORE

    def test_score_monotonic_decrease_with_pd(self):
        """Higher default probability must strictly result in lower or equal credit score."""
        probs = np.linspace(0.01, 0.99, 50)
        scores = [probability_to_credit_score(p) for p in probs]

        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Monotonicity violated at p={probs[i]}: score={scores[i]} vs next score={scores[i+1]}"
            )

    def test_score_to_tier_mapping(self):
        """Checks risk tier classification thresholds."""
        assert score_to_tier(800) == "Low Risk"
        assert score_to_tier(750) == "Low Risk"
        assert score_to_tier(700) == "Moderate Risk"
        assert score_to_tier(650) == "Moderate Risk"
        assert score_to_tier(600) == "High Risk — Manual Review"
        assert score_to_tier(550) == "High Risk — Manual Review"
        assert score_to_tier(500) == "Very High Risk"
        assert score_to_tier(300) == "Very High Risk"


# =============================================================================
# 7. CALIBRATION OUTPUT VALIDITY TESTS
# =============================================================================

class TestCalibrationValidity:
    """Verifies Platt and Isotonic calibration engines."""

    def test_calibrator_bounds_and_brier(self):
        """Calibrated probabilities must strictly stay in [0, 1]."""
        y_val = np.array([0, 0, 0, 1, 0, 1, 1, 1])
        raw_prob_val = np.array([0.2, 0.3, 0.1, 0.7, 0.4, 0.8, 0.9, 0.85])

        calibrator = ProbabilityCalibrator(method="sigmoid")
        calibrator.fit(raw_prob_val, y_val)

        test_raw = np.array([0.05, 0.5, 0.95])
        calibrated = calibrator.predict(test_raw)

        assert np.all(calibrated >= 0.0)
        assert np.all(calibrated <= 1.0)
        # Relative ordering must be preserved
        assert calibrated[0] < calibrated[1] < calibrated[2]


# =============================================================================
# 8. CHAMPION SELECTION RULES TESTS
# =============================================================================

class TestChampionSelection:
    """Verifies multi-gate deterministic champion gating."""

    def test_champion_selected_when_all_gates_pass(self):
        """Candidate with strong metrics passing all gates is designated champion."""
        candidates = [
            {
                "model_name": "Logistic Regression",
                "oot_metrics": {
                    "roc_auc": 0.85,
                    "ks_statistic": 45.0,
                    "brier_score": 0.08,
                    "pr_auc": 0.65,
                },
                "val_metrics": {"roc_auc": 0.86},
                "train_metrics": {"roc_auc": 0.87},
                "fairness_report": {"overall_parity_pass": True},
            }
        ]

        result = evaluate_deterministic_champion(
            candidates=candidates,
            min_oot_auc=0.60,
            min_oot_ks=18.0,
        )

        assert result.decision == "CHAMPION_SELECTED"
        assert result.champion_name == "Logistic Regression"

    def test_no_champion_returned_when_discrimination_fails(self):
        """When candidate AUC is below minimum threshold, return NO-CHAMPION."""
        candidates = [
            {
                "model_name": "Weak Model",
                "oot_metrics": {
                    "roc_auc": 0.54,  # Below min threshold 0.60
                    "ks_statistic": 10.0,  # Below min threshold 18.0
                    "brier_score": 0.25,
                    "pr_auc": 0.12,
                },
                "val_metrics": {"roc_auc": 0.55},
                "train_metrics": {"roc_auc": 0.55},
                "fairness_report": {"overall_parity_pass": True},
            }
        ]

        result = evaluate_deterministic_champion(
            candidates=candidates,
            min_oot_auc=0.60,
            min_oot_ks=18.0,
        )

        assert result.decision == "NO-CHAMPION"
        assert result.champion_name is None
        assert any("discrimination" in r.lower() for r in result.failure_reasons)



# =============================================================================
# 9. ECONOMIC DECISIONING TESTS
# =============================================================================

class TestEconomicDecisioning:
    """Verifies Expected Loss formula and policy simulation."""

    def test_expected_loss_formula(self):
        """Expected Loss must equal PD * LGD * EAD."""
        pd_val = 0.10
        lgd = 0.60
        ead = 25000.0

        el = compute_expected_loss(pd_val, lgd=lgd, ead=ead)
        expected = round(pd_val * lgd * ead, 2)
        assert el == expected

    def test_make_underwriting_decision_distribution(self):
        """Verifies policy splits into APPROVE, REVIEW, DECLINE correctly."""
        config = DecisionPolicyConfig(
            auto_approve_max_pd=0.05,
            manual_review_max_pd=0.20,
            min_score_cutoff=650,
            default_lgd=0.60,
            default_ead=25000.0,
        )

        d1 = make_underwriting_decision(pd_val=0.02, score=750, config=config)
        assert d1["decision"] == "APPROVE"

        d2 = make_underwriting_decision(pd_val=0.12, score=620, config=config)
        assert d2["decision"] == "REVIEW"

        d3 = make_underwriting_decision(pd_val=0.35, score=450, config=config)
        assert d3["decision"] == "DECLINE"

    def test_policy_tradeoffs_computation(self):
        """Verifies aggregate portfolio tradeoff metrics."""
        pds = np.array([0.03, 0.08, 0.25])
        trues = np.array([0, 0, 1])
        config = DecisionPolicyConfig(
            auto_approve_max_pd=0.05,
            manual_review_max_pd=0.20,
            default_lgd=0.60,
            default_ead=20000.0,
        )

        tradeoffs = simulate_policy_tradeoffs(
            predicted_pds=pds,
            true_defaults=trues,
            config=config,
        )

        assert "approval_rate" in tradeoffs
        assert "approved_expected_loss" in tradeoffs
        assert "approved_cohort_bad_rate" in tradeoffs
        assert tradeoffs["total_applicants"] == 3
        assert tradeoffs["approval_count"] == 1
        assert tradeoffs["review_count"] == 1
        assert tradeoffs["decline_count"] == 1

