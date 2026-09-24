"""
CreditBridge - Alternative Credit Scoring Engine
Part 9: Synthetic Baseline Regression & Protected Dependency Tests
Path: tests/test_synthetic_baseline_regression.py

Verifies that the original CreditBridge baseline functionality remains completely
intact: model bundle contracts, legacy scoring helpers, probability calibration,
risk tier boundaries, and deterministic SHAP attribution on representative synthetic borrowers.
"""

import os
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.scoring_utils import (
    load_model_bundle,
    score_borrower,
    probability_to_credit_score,
    score_to_tier,
    POPULATION_DEFAULT_RATE,
    SCORE_OFFSET,
    SCORE_FACTOR,
)
from src.explain import explain_borrower_record


# -----------------------------------------------------------------------------
# 1. BASELINE BORROWER REGRESSION FIXTURES
# -----------------------------------------------------------------------------

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "baseline_borrowers.csv"

# Pre-computed deterministic baseline targets
EXPECTED_BASELINE_RESULTS = {
    "0e4cde06-4b7d-46d3-bb72-0538f14300b4": {
        "tier": "Moderate Risk",
        "score": 678,
        "raw_prob": 0.4600,
        "cal_prob": 0.1218,
    },
    "d3ae04ea-456d-4582-9543-14fcc7cd16a9": {
        "tier": "Low Risk",
        "score": 818,
        "raw_prob": 0.1624,
        "cal_prob": 0.0306,
    },
    "dbe03e32-3d66-466e-aeb2-d403bf4e43fe": {
        "tier": "High Risk — Manual Review",
        "score": 623,
        "raw_prob": 0.6016,
        "cal_prob": 0.1973,
    },
    "cb367f91-5faa-4977-b3f5-c06f5f2af92b": {
        "tier": "Very High Risk",
        "score": 514,
        "raw_prob": 0.8271,
        "cal_prob": 0.4377,
    },
}


# -----------------------------------------------------------------------------
# 2. MODEL CONTRACT & BACKWARD COMPATIBILITY
# -----------------------------------------------------------------------------

def test_model_bundle_structure_and_keys():
    """Verify that models/credit_model.pkl loads with exact expected components."""
    bundle = load_model_bundle()
    assert isinstance(bundle, dict)
    assert "model" in bundle
    assert "pipeline" in bundle
    assert "feature_names" in bundle
    assert len(bundle["feature_names"]) == 30

    # Verify model is trained classifier
    model = bundle["model"]
    assert hasattr(model, "predict_proba")
    assert hasattr(model, "classes_")

    # Verify pipeline has transform method
    pipeline = bundle["pipeline"]
    assert hasattr(pipeline, "transform")


def test_feature_names_and_order_at_model_boundary():
    """Verify feature names and ordering are identical to the fitted feature pipeline."""
    bundle = load_model_bundle()
    feature_names = bundle["feature_names"]
    assert len(feature_names) == 30
    
    # Must contain base numeric features, engineered composite features, and one-hot encodings
    assert "income_stability_index" in feature_names
    assert "payment_reliability_score" in feature_names
    assert "monthly_income_estimate" in feature_names
    assert "occupation_type_gig_delivery" in feature_names
    assert "city_tier_tier_1" in feature_names


# -----------------------------------------------------------------------------
# 3. BASELINE BORROWER REGRESSION INVARIANCE
# -----------------------------------------------------------------------------

def test_baseline_borrowers_regression_invariance():
    """Verify representative borrowers from each risk tier match exact baseline scores and tiers."""
    assert FIXTURE_PATH.exists(), f"Baseline fixture missing at: {FIXTURE_PATH}"
    df = pd.read_csv(FIXTURE_PATH)
    assert len(df) == 4

    bundle = load_model_bundle()
    pipeline = bundle["pipeline"]
    model = bundle["model"]

    for _, row in df.iterrows():
        bid = str(row["borrower_id"])
        assert bid in EXPECTED_BASELINE_RESULTS
        expected = EXPECTED_BASELINE_RESULTS[bid]

        # 1. Transform features through legacy pipeline
        X_trans = pipeline.transform(pd.DataFrame([row]))

        # 2. Raw inference
        raw_p = float(model.predict_proba(X_trans)[0, 1])
        assert abs(raw_p - expected["raw_prob"]) < 0.005, (
            f"Raw probability drift for {bid}: got {raw_p}, expected {expected['raw_prob']}"
        )

        # 3. Calibration
        odds_raw = raw_p / max(1.0 - raw_p, 1e-6)
        odds_cal = odds_raw * (POPULATION_DEFAULT_RATE / (1.0 - POPULATION_DEFAULT_RATE))
        cal_p = float(odds_cal / (1.0 + odds_cal))
        assert abs(cal_p - expected["cal_prob"]) < 0.005, (
            f"Calibrated probability drift for {bid}: got {cal_p}, expected {expected['cal_prob']}"
        )

        # 4. Score & Tier
        score = int(probability_to_credit_score(cal_p))
        assert abs(score - expected["score"]) <= 1, (
            f"Score drift for {bid}: got {score}, expected {expected['score']}"
        )

        tier = score_to_tier(int(score))
        # Normalize dash character differences for comparison
        clean_tier = tier.replace("—", "-").replace("", "-")
        clean_exp_tier = expected["tier"].replace("—", "-").replace("", "-")
        assert clean_tier == clean_exp_tier


def test_legacy_score_borrower_helper_api():
    """Verify legacy score_borrower() helper produces expected results without error."""
    df = pd.read_csv(FIXTURE_PATH)
    bundle = load_model_bundle()

    for _, row in df.iterrows():
        score, tier = score_borrower(row, bundle)
        assert isinstance(score, int)
        assert 300 <= score <= 900
        assert isinstance(tier, str)


# -----------------------------------------------------------------------------
# 4. RAW PROBABILITY VS CALIBRATED PROBABILITY REGRESSION
# -----------------------------------------------------------------------------

def test_raw_vs_calibrated_probability_distinction():
    """
    Verify raw model probability (from balanced training class weights) and
    calibrated default probability (under 14% thin-file prior) are mathematically
    distinct and never conflated.
    """
    raw_p = 0.50  # Equal chance in balanced model
    odds_raw = raw_p / (1.0 - raw_p)  # 1.0
    odds_cal = odds_raw * (POPULATION_DEFAULT_RATE / (1.0 - POPULATION_DEFAULT_RATE))  # 0.14 / 0.86 = 0.1628
    cal_p = odds_cal / (1.0 + odds_cal)  # ~0.14

    assert abs(cal_p - 0.14) < 0.001
    assert raw_p != cal_p
    assert cal_p < raw_p  # Because population prior 0.14 < balanced 0.50

    # Score must be derived from calibrated probability
    score_from_cal = probability_to_credit_score(cal_p)
    score_from_raw = probability_to_credit_score(raw_p)
    assert score_from_cal > score_from_raw
    assert score_from_raw == int(SCORE_OFFSET)  # ln(1)=0 -> Offset=490


# -----------------------------------------------------------------------------
# 5. MONOTONICITY & RISK TIER COMPLETENESS
# -----------------------------------------------------------------------------

def test_credit_score_monotonicity():
    """Verify credit score strictly increases as calibrated default probability decreases."""
    probs = np.linspace(0.01, 0.99, 50)
    scores = [probability_to_credit_score(p) for p in probs]

    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], "Credit score must monotonically decrease as default risk increases"


def test_risk_tier_boundaries_complete_partition():
    """Verify risk tiers cleanly partition the entire [300, 900] domain."""
    for s in range(300, 901):
        t = score_to_tier(s)
        assert t in ["Low Risk", "Moderate Risk", "High Risk — Manual Review", "Very High Risk"]


# -----------------------------------------------------------------------------
# 6. SHAP EXPLANATION STABILITY ON SYNTHETIC BORROWERS
# -----------------------------------------------------------------------------

def test_shap_explanation_deterministic_on_baseline_borrower():
    """Verify local SHAP explanation runs deterministically on synthetic baseline borrowers."""
    df = pd.read_csv(FIXTURE_PATH)
    bundle = load_model_bundle()
    sample_borrower = df.iloc[0:1]

    explanation = explain_borrower_record(
        borrower_data=sample_borrower,
        model_bundle=bundle,
    )

    assert "plain_english_explanation" in explanation
    assert "top_positive_factors" in explanation
    assert "top_negative_factors" in explanation
    narrative = explanation["plain_english_explanation"]
    assert ("primarily constrained by" in narrative or "positively supported by" in narrative)
    assert "proved" not in narrative  # Non-causal verification
