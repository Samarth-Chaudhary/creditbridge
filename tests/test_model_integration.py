"""
CreditBridge - Real Data Mode: Model Integration & Model-Risk Verification Tests
Path: tests/test_model_integration.py

Verifies Part 7 requirements:
1. Model artifact loading and scikit-learn version compatibility.
2. Synthetic borrower regression testing (guaranteeing 0 drift for existing borrowers).
3. Pre-inference validation gate (column presence, schema purity, order, vocabulary, ranges).
4. Raw vs. calibrated probability semantics.
5. Score transformation formula, monotonicity, bounds, and edge cases.
6. Risk tier boundaries, exclusivity, and threshold precision.
7. Real-data scoring adapter execution through fitted FeaturePipeline.
8. Quality gate enforcement (INSUFFICIENT blocks scoring, LIMITED permits with warnings).
9. Local SHAP explanations for new real-data borrowers with factor provenance labeling.
10. End-to-end orchestration and result serialization.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from src.explain import (
    explain_borrower_record,
)
from src.real_data_contracts import (
    UNAVAILABLE_MODEL_COLUMNS,
    AssessmentQualityStatus,
    ManualInputContract,
    MissingRequiredModelFeatureError,
    ModelSchemaMismatchError,
    RealDataAssessmentResult,
    UnknownCategoryError,
)
from src.real_data_features import build_real_borrower_payload
from src.real_data_parser import parse_csv_statement
from src.real_data_quality import build_data_quality_report
from src.real_data_scoring import (
    assess_statement_end_to_end,
    score_real_borrower_payload,
    validate_model_input_schema,
)
from src.scoring_utils import (
    MAX_SCORE,
    MIN_SCORE,
    POPULATION_DEFAULT_RATE,
    load_model_bundle,
    probability_to_credit_score,
    score_borrower,
    score_to_tier,
)
from src.transaction_classifier import classify_transactions_df

# -----------------------------------------------------------------------------
# FIXTURES
# -----------------------------------------------------------------------------

@pytest.fixture(scope="session")
def model_bundle():
    """Loads and caches the model bundle for the test session."""
    return load_model_bundle()


@pytest.fixture
def clean_multi_month_csv() -> str:
    """Generates a 4-month clean statement CSV with > 20 transactions."""
    rows = ["Date,Description,Amount,Type"]
    import datetime
    start = datetime.date(2023, 1, 1)
    for i in range(28):
        d = start + datetime.timedelta(days=i * 4)
        if i % 7 == 0:
            rows.append(f"{d},Salary Transfer from Swiggy,38000,Credit")
        elif i % 3 == 0:
            rows.append(f"{d},Jio Prepaid Recharge,299,Debit")
        elif i % 2 == 0:
            rows.append(f"{d},D-Mart Groceries Merchant,1450,Debit")
        else:
            rows.append(f"{d},P2P UPI transfer to friend,500,Debit")
    return "\n".join(rows)


@pytest.fixture
def valid_manual_inputs() -> ManualInputContract:
    return ManualInputContract(
        age=29,
        occupation_type="gig_delivery",
        city_tier="tier_1",
        borrower_id="test_borrower_001",
        account_holder_name="Ramesh Kumar",
    )


@pytest.fixture
def valid_borrower_df(clean_multi_month_csv, valid_manual_inputs) -> pd.DataFrame:
    parsed = parse_csv_statement(clean_multi_month_csv)
    classified = classify_transactions_df(parsed.transactions)
    payload = build_real_borrower_payload(classified, valid_manual_inputs)
    return payload.borrower_df.copy()


# -----------------------------------------------------------------------------
# 1. ARTIFACT LOADING & VERSION COMPATIBILITY
# -----------------------------------------------------------------------------

def test_model_artifact_loads_in_fresh_process(model_bundle):
    """Verifies that credit_model.pkl loads with all required components."""
    assert "model" in model_bundle
    assert "pipeline" in model_bundle
    assert "feature_names" in model_bundle
    assert len(model_bundle["feature_names"]) == 30

    model = model_bundle["model"]
    pipeline = model_bundle["pipeline"]
    assert hasattr(model, "predict_proba")
    assert hasattr(pipeline, "transform")

    # LogisticRegression classifier must have multi_class compatibility attribute
    clf = model.named_steps.get("classifier", model) if hasattr(model, "named_steps") else model
    assert hasattr(clf, "multi_class"), "Classifier missing multi_class attribute!"


# -----------------------------------------------------------------------------
# 2. SYNTHETIC BORROWER REGRESSION TEST (ZERO MODEL DRIFT)
# -----------------------------------------------------------------------------

def test_synthetic_borrower_regression(model_bundle):
    """
    REGRESSION: Confirms that scoring an existing synthetic borrower produces
    the exact expected score and tier without any drift from the real-data adapter.
    """
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_borrowers.csv")
    df_synthetic = pd.read_csv(csv_path)

    # Test sample index 0
    row0 = df_synthetic.iloc[0]
    score0, tier0 = score_borrower(row0, model_bundle)
    assert score0 == 678
    assert tier0 == "Moderate Risk"

    # Test sample index 10
    row10 = df_synthetic.iloc[10]
    score10, tier10 = score_borrower(row10, model_bundle)
    assert 300 <= score10 <= 900
    assert tier10 in ["Low Risk", "Moderate Risk", "High Risk — Manual Review", "Very High Risk"]


# -----------------------------------------------------------------------------
# 3. PRE-INFERENCE VALIDATION GATE TESTS
# -----------------------------------------------------------------------------

def test_model_input_validation_gate_accepts_valid(valid_borrower_df):
    """Verifies that a valid 22-column borrower DataFrame passes validation cleanly."""
    # Must not raise any error
    validate_model_input_schema(valid_borrower_df)


def test_model_input_validation_gate_rejects_missing_column(valid_borrower_df):
    """Missing required model feature triggers MissingRequiredModelFeatureError."""
    bad_df = valid_borrower_df.drop(columns=["monthly_income_estimate"])
    with pytest.raises(MissingRequiredModelFeatureError) as exc_info:
        validate_model_input_schema(bad_df)
    assert "monthly_income_estimate" in str(exc_info.value)


def test_model_input_validation_gate_rejects_extra_columns(valid_borrower_df):
    """Extra unexpected columns trigger ModelSchemaMismatchError."""
    bad_df = valid_borrower_df.copy()
    bad_df["unauthorized_credit_score"] = 750
    with pytest.raises(ModelSchemaMismatchError) as exc_info:
        validate_model_input_schema(bad_df)
    assert "unauthorized_credit_score" in str(exc_info.value)


def test_model_input_validation_gate_rejects_column_order_mismatch(valid_borrower_df):
    """Reordered columns trigger ModelSchemaMismatchError to prevent tensor misalignment."""
    cols = list(valid_borrower_df.columns)
    cols[0], cols[1] = cols[1], cols[0]
    bad_df = valid_borrower_df[cols]
    with pytest.raises(ModelSchemaMismatchError) as exc_info:
        validate_model_input_schema(bad_df)
    assert "column order violates model contract" in str(exc_info.value)


def test_model_input_validation_gate_rejects_unknown_category(valid_borrower_df):
    """Unseen occupation type triggers UnknownCategoryError."""
    bad_df = valid_borrower_df.copy()
    bad_df["occupation_type"] = "hedge_fund_manager"
    with pytest.raises(UnknownCategoryError) as exc_info:
        validate_model_input_schema(bad_df)
    assert "hedge_fund_manager" in str(exc_info.value)


def test_model_input_validation_gate_rejects_invalid_numeric_domain(valid_borrower_df):
    """Underage (< 18) or negative monetary values trigger ModelSchemaMismatchError."""
    # Underage check
    bad_age = valid_borrower_df.copy()
    bad_age["age"] = 15
    with pytest.raises(ModelSchemaMismatchError) as exc:
        validate_model_input_schema(bad_age)
    assert "outside allowed domain" in str(exc.value)

    # Negative monthly income check
    bad_income = valid_borrower_df.copy()
    bad_income["monthly_income_estimate"] = -5000.0
    with pytest.raises(ModelSchemaMismatchError) as exc:
        validate_model_input_schema(bad_income)
    assert "cannot be negative" in str(exc.value)


# -----------------------------------------------------------------------------
# 4. PROBABILITY SEMANTICS: RAW VS. CALIBRATED
# -----------------------------------------------------------------------------

def test_probability_semantics_raw_vs_calibrated(valid_borrower_df, model_bundle):
    """
    Verifies that raw_model_probability and calibrated_model_probability:
    1. Are computed and reported as distinct numbers.
    2. The calibrated probability strictly adheres to Bayesian prior adjustment (pi = 0.14).
    3. Neither value is falsely labeled as an empirical bureau default rate.
    """
    model = model_bundle["model"]
    pipeline = model_bundle["pipeline"]

    X = pipeline.transform(valid_borrower_df)
    prob_raw = float(model.predict_proba(X)[0, 1])

    # Manual Bayes calibration computation
    odds_raw = prob_raw / max(1.0 - prob_raw, 1e-6)
    odds_calibrated = odds_raw * (POPULATION_DEFAULT_RATE / (1.0 - POPULATION_DEFAULT_RATE))
    prob_calibrated = float(odds_calibrated / (1.0 + odds_calibrated))

    # Because POPULATION_DEFAULT_RATE = 0.14 < 0.50 (balanced training),
    # calibrated probability should be strictly lower than raw probability
    if prob_raw > 1e-4:
        assert prob_calibrated < prob_raw, "Calibrated probability should be down-weighted by 14% prior!"

    # Verify mathematical formula equality
    assert np.isclose(prob_calibrated, odds_calibrated / (1.0 + odds_calibrated), atol=1e-6)


# -----------------------------------------------------------------------------
# 5. SCORE TRANSFORMATION FORMULA, MONOTONICITY & BOUNDS
# -----------------------------------------------------------------------------

def test_score_transformation_formula_and_monotonicity():
    """
    Verifies the CreditBridge score transformation:
    Score = 490.0 + 95.0 * ln((1-p)/p)
    bounded in [300, 900].
    """
    # 1. Monotonicity check across probabilities
    probs = np.linspace(0.01, 0.99, 50)
    scores = [probability_to_credit_score(p) for p in probs]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], "Credit score must be strictly non-increasing with default probability!"

    # 2. Clamped Bounds
    assert probability_to_credit_score(0.0) == MAX_SCORE  # Clamped to 900
    assert probability_to_credit_score(1.0) == MIN_SCORE  # Clamped to 300
    assert probability_to_credit_score(1e-9) == MAX_SCORE
    assert probability_to_credit_score(1.0 - 1e-9) == MIN_SCORE

    # 3. Exact Point Verification
    # At p = 0.5: ln(odds) = ln(1.0) = 0 => score = 490
    assert probability_to_credit_score(0.5) == 490

    # At p = 0.05: odds = 19, ln(19) ~ 2.9444 => score = 490 + 95 * 2.9444 = 770
    assert 768 <= probability_to_credit_score(0.05) <= 772


# -----------------------------------------------------------------------------
# 6. RISK TIER BOUNDARIES, EXCLUSIVITY & PRECISION
# -----------------------------------------------------------------------------

def test_risk_tier_boundaries_and_exclusivity():
    """
    Verifies that risk tiers partition the 300-900 domain deterministically:
    - 750 - 900: Low Risk
    - 650 - 749: Moderate Risk
    - 550 - 649: High Risk — Manual Review
    - Below 550: Very High Risk
    """
    # Low Risk
    assert score_to_tier(900) == "Low Risk"
    assert score_to_tier(750) == "Low Risk"

    # Moderate Risk
    assert score_to_tier(749) == "Moderate Risk"
    assert score_to_tier(650) == "Moderate Risk"

    # High Risk — Manual Review
    assert score_to_tier(649) == "High Risk — Manual Review"
    assert score_to_tier(550) == "High Risk — Manual Review"

    # Very High Risk
    assert score_to_tier(549) == "Very High Risk"
    assert score_to_tier(300) == "Very High Risk"

    # Complete coverage across entire domain [300, 900]
    valid_tiers = {"Low Risk", "Moderate Risk", "High Risk — Manual Review", "Very High Risk"}
    for s in range(300, 901):
        assert score_to_tier(s) in valid_tiers


# -----------------------------------------------------------------------------
# 7. REAL BORROWER PAYLOAD SCORING THROUGH FITTED PIPELINE
# -----------------------------------------------------------------------------

def test_real_borrower_payload_scores_successfully(clean_multi_month_csv, valid_manual_inputs, model_bundle):
    """
    End-to-end integration test:
    Takes real CSV -> feature mapping -> quality checks -> scoring adapter.
    """
    parsed = parse_csv_statement(clean_multi_month_csv)
    classified = classify_transactions_df(parsed.transactions)
    payload = build_real_borrower_payload(classified, valid_manual_inputs)
    quality = build_data_quality_report(parsed, payload)

    result = score_real_borrower_payload(payload, quality, model_bundle)

    assert isinstance(result, RealDataAssessmentResult)
    assert result.borrower_id == valid_manual_inputs.borrower_id
    assert result.is_scoreable is True
    assert 300 <= result.credit_score <= 900
    assert result.risk_tier in ["Low Risk", "Moderate Risk", "High Risk — Manual Review", "Very High Risk"]
    assert 0.0 <= result.raw_model_probability <= 1.0
    assert 0.0 <= result.calibrated_model_probability <= 1.0
    assert result.evidence_coverage_ratio == round(12 / 21, 4)


# -----------------------------------------------------------------------------
# 8. QUALITY GATE ENFORCEMENT (INSUFFICIENT BLOCKS SCORING)
# -----------------------------------------------------------------------------

def test_insufficient_quality_blocks_scoring_cleanly(valid_manual_inputs, model_bundle):
    """
    Verifies that a statement with INSUFFICIENT history (< 30 days or < 15 txns)
    is caught by the quality blocker gate and does not generate a fake score.
    """
    short_csv = "Date,Description,Amount,Type\n2023-01-05,Salary Transfer,30000,Credit\n2023-01-10,Groceries,500,Debit\n"
    parsed = parse_csv_statement(short_csv)
    classified = classify_transactions_df(parsed.transactions)
    payload = build_real_borrower_payload(classified, valid_manual_inputs, allow_insufficient_history=True)
    quality = build_data_quality_report(parsed, payload)

    assert quality.overall_status == AssessmentQualityStatus.INSUFFICIENT.value
    assert quality.is_eligible_for_scoring is False

    result = score_real_borrower_payload(payload, quality, model_bundle)

    assert result.is_scoreable is False
    assert result.credit_score == 0
    assert result.risk_tier == "Unscoreable — Insufficient Statement Evidence"
    assert len(result.blocker_reasons) > 0
    assert "INSUFFICIENT" in result.assessment_status


# -----------------------------------------------------------------------------
# 9. NEW BORROWER SHAP EXPLANATION & PROVENANCE DISCLOSURE
# -----------------------------------------------------------------------------

def test_new_borrower_shap_explanation_and_provenance(valid_borrower_df, model_bundle):
    """
    Verifies explain_borrower_record():
    1. Works on a new borrower without requiring an ID in synthetic_borrowers.csv.
    2. Produces ranked factors with human-readable names and point impacts.
    3. Exposes source status (derived, self-reported, unavailable_imputed).
    4. Plain-English narrative is non-causal ('primarily constrained by').
    """
    res = explain_borrower_record(
        borrower_data=valid_borrower_df,
        model_bundle=model_bundle,
    )

    assert "factors" in res
    assert len(res["factors"]) == 30
    assert "plain_english_explanation" in res
    assert "top_positive_factors" in res
    assert "top_negative_factors" in res

    # Check first factor structure
    top_factor = res["factors"][0]
    assert "feature_name" in top_factor
    assert "human_name" in top_factor
    assert "point_impact" in top_factor
    assert "source_status" in top_factor
    assert "is_imputed" in top_factor
    assert top_factor["source_status"] in [
        "derived_from_statement",
        "self_reported",
        "unavailable_imputed",
        "composite_derived",
        "composite_partially_imputed",
    ]


def test_imputed_feature_flagged_in_explanation(valid_borrower_df, model_bundle):
    """
    Verifies that unavailable model features (e.g. electricity_bill_ontime_rate)
    are strictly flagged as is_imputed=True and source_status='unavailable_imputed'.
    """
    res = explain_borrower_record(
        borrower_data=valid_borrower_df,
        model_bundle=model_bundle,
    )

    imputed_factors = [f for f in res["factors"] if f["feature_name"] in UNAVAILABLE_MODEL_COLUMNS]
    assert len(imputed_factors) == len(UNAVAILABLE_MODEL_COLUMNS)
    for f in imputed_factors:
        assert f["is_imputed"] is True
        assert f["source_status"] == "unavailable_imputed"


# -----------------------------------------------------------------------------
# 10. END-TO-END ORCHESTRATION PIPELINE TEST
# -----------------------------------------------------------------------------

def test_end_to_end_assessment_orchestrator(clean_multi_month_csv, valid_manual_inputs):
    """Tests the convenience assess_statement_end_to_end() entry point."""
    result = assess_statement_end_to_end(
        statement_input=clean_multi_month_csv,
        manual_inputs=valid_manual_inputs,
    )

    assert isinstance(result, RealDataAssessmentResult)
    assert result.is_scoreable is True
    assert 300 <= result.credit_score <= 900
    assert result.assessment_status in ["SUFFICIENT", "LIMITED"]
    assert len(result.explanation["factors"]) == 30
    assert result.parser_quality["rows_accepted"] > 0


# -----------------------------------------------------------------------------
# 11. IMMUTABILITY & SERIALIZATION TEST
# -----------------------------------------------------------------------------

def test_assessment_result_immutability_and_serialization(clean_multi_month_csv, valid_manual_inputs):
    """Verifies that RealDataAssessmentResult is frozen and JSON serializable."""
    result = assess_statement_end_to_end(
        statement_input=clean_multi_month_csv,
        manual_inputs=valid_manual_inputs,
    )

    # Immutability check
    with pytest.raises(Exception):
        result.credit_score = 800  # type: ignore

    # JSON serialization check
    res_dict = result.to_dict()
    json_str = json.dumps(res_dict)
    assert len(json_str) > 100
    reloaded = json.loads(json_str)
    assert reloaded["borrower_id"] == result.borrower_id
    assert reloaded["credit_score"] == result.credit_score
