"""
CreditBridge - Alternative Credit Scoring Engine
Real Data Mode: Model Integration & Scoring Bridge
Path: src/real_data_scoring.py

Connects real statement payloads from Parts 4-6 into the existing fitted model
infrastructure (FeaturePipeline, LogisticRegression classifier, CIBIL-style log-odds
transformation, risk tier mapping, and SHAP explainability) without altering the
trained model or feature-engineering pipeline.
"""

import os
import sys
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd

# Ensure project root is accessible
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.real_data_contracts import (
    RAW_BORROWER_COLUMNS,
    ALLOWED_OCCUPATIONS,
    ALLOWED_CITY_TIERS,
    UNAVAILABLE_MODEL_COLUMNS,
    AssessmentQualityStatus,
    RealBorrowerPayload,
    RealDataQualityReport,
    RealDataAssessmentResult,
    ManualInputContract,
    MissingRequiredModelFeatureError,
    ModelSchemaMismatchError,
    UnknownCategoryError,
    PreprocessingFailureError,
    PredictionFailureError,
    ScoreTransformationFailureError,
    ExplanationFailureError,
)
from src.scoring_utils import (
    POPULATION_DEFAULT_RATE,
    probability_to_credit_score,
    score_to_tier,
    load_model_bundle,
)
from src.explain import explain_borrower_record
from src.real_data_parser import parse_csv_statement
from src.transaction_classifier import classify_transactions_df
from src.real_data_features import build_real_borrower_payload
from src.real_data_quality import build_data_quality_report


# -----------------------------------------------------------------------------
# 1. EXPLICIT PRE-INFERENCE VALIDATION GATE
# -----------------------------------------------------------------------------

def validate_model_input_schema(borrower_df: pd.DataFrame) -> None:
    """
    Enforces a strict validation gate immediately before feature transformation
    and model inference.

    Validation Gate Pipeline:
    required columns -> column ordering -> dtypes -> nullability -> allowed categories -> numeric ranges.

    Raises:
    -------
    MissingRequiredModelFeatureError:
        If any of the 22 required raw columns are missing.
    ModelSchemaMismatchError:
        If extra unexpected columns exist or column ordering differs from contract.
    UnknownCategoryError:
        If categorical inputs violate the trained model vocabulary.
    """
    if not isinstance(borrower_df, pd.DataFrame):
        raise ModelSchemaMismatchError(f"Expected pd.DataFrame input, received: {type(borrower_df)}")

    if len(borrower_df) != 1:
        raise ModelSchemaMismatchError(f"Validation gate accepts single-borrower DataFrame. Received {len(borrower_df)} rows.")

    df_cols = list(borrower_df.columns)

    # 1. Required Columns Check
    missing_cols = [c for c in RAW_BORROWER_COLUMNS if c not in df_cols]
    if missing_cols:
        raise MissingRequiredModelFeatureError(
            f"Input DataFrame is missing required model features: {missing_cols}. "
            f"Expected all 22 columns: {RAW_BORROWER_COLUMNS}"
        )

    # 2. Schema Purity Check (No Unexpected Columns)
    extra_cols = [c for c in df_cols if c not in RAW_BORROWER_COLUMNS]
    if extra_cols:
        raise ModelSchemaMismatchError(
            f"Input DataFrame contains unauthorized unexpected columns: {extra_cols}. "
            "Model input schema must strictly match the 22 contract columns."
        )

    # 3. Column Order Check
    if df_cols != RAW_BORROWER_COLUMNS:
        raise ModelSchemaMismatchError(
            "Input DataFrame column order violates model contract. "
            f"Expected: {RAW_BORROWER_COLUMNS}, Received: {df_cols}"
        )

    row = borrower_df.iloc[0]

    # 4. Mandatory Identity & Categorical Validation
    borrower_id_val = str(row["borrower_id"])
    if not borrower_id_val or pd.isna(row["borrower_id"]):
        raise ModelSchemaMismatchError("borrower_id must not be empty or null.")

    occ_val = str(row["occupation_type"])
    if occ_val not in ALLOWED_OCCUPATIONS:
        raise UnknownCategoryError(
            f"Invalid occupation_type '{occ_val}'. Allowed vocabulary: {ALLOWED_OCCUPATIONS}"
        )

    city_val = str(row["city_tier"])
    if city_val not in ALLOWED_CITY_TIERS:
        raise UnknownCategoryError(
            f"Invalid city_tier '{city_val}'. Allowed vocabulary: {ALLOWED_CITY_TIERS}"
        )

    # 5. Demographic & Numeric Range Verification
    try:
        age_val = float(row["age"])
        if not (18 <= age_val <= 70):
            raise ModelSchemaMismatchError(f"Age {age_val} is outside allowed domain [18, 70].")
    except (ValueError, TypeError) as e:
        raise ModelSchemaMismatchError(f"Invalid age value '{row['age']}': {e}")

    # 6. Non-Negative Monetary & Count Signals
    non_negative_checks = [
        "monthly_income_estimate",
        "recharge_frequency_per_month",
        "avg_recharge_amount",
        "recharge_amount_volatility",
        "monthly_upi_transaction_count",
        "monthly_upi_inflow_avg",
        "monthly_upi_outflow_avg",
        "upi_inflow_volatility_coefficient",
        "p2p_vs_merchant_txn_ratio",
    ]
    for col in non_negative_checks:
        val = row[col]
        if not pd.isna(val) and float(val) < 0.0:
            raise ModelSchemaMismatchError(f"Feature '{col}' cannot be negative. Received: {val}")


# -----------------------------------------------------------------------------
# 2. REAL DATA SCORING BRIDGE ADAPTER
# -----------------------------------------------------------------------------

def score_real_borrower_payload(
    payload: RealBorrowerPayload,
    quality_report: RealDataQualityReport,
    model_bundle: Optional[Dict[str, Any]] = None,
    save_waterfall_path: Optional[str] = None,
) -> RealDataAssessmentResult:
    """
    Authoritative adapter and scoring orchestrator for Real Data Mode.

    Safely feeds real-data features into the existing fitted model infrastructure:
    RealBorrowerPayload -> Schema Validation Gate -> FeaturePipeline ->
    Fitted Model -> Calibration -> CIBIL Score Transformation -> Risk Tier ->
    SHAP Attribution -> RealDataAssessmentResult.

    Parameters:
    -----------
    payload : RealBorrowerPayload
        Structured container emitted by feature mapping in Part 5.
    quality_report : RealDataQualityReport
        Consolidated quality and distribution report emitted by Part 6.
    model_bundle : Optional[dict]
        Pre-loaded model artifact bundle to avoid reloading on repeated requests.
    save_waterfall_path : Optional[str]
        Optional file path to persist individual borrower SHAP waterfall plot.

    Returns:
    --------
    RealDataAssessmentResult:
        Frozen, immutable structured assessment artifact.
    """
    borrower_df = payload.borrower_df
    borrower_id_val = str(borrower_df["borrower_id"].iloc[0])

    # 1. Check Data Quality Blocker Gate
    # Statements with INSUFFICIENT quality or explicit blockers cannot be scored
    if quality_report.overall_status == AssessmentQualityStatus.INSUFFICIENT.value or not quality_report.is_eligible_for_scoring:
        # Build empty explanation reference for blocked assessment
        empty_explanation = {
            "borrower_id": borrower_id_val,
            "raw_model_probability": 0.0,
            "calibrated_model_probability": 0.0,
            "credit_score": 0,
            "risk_tier": "Unscoreable — Insufficient Statement Evidence",
            "plain_english_explanation": (
                "Scoring blocked by data quality gate. "
                + "; ".join(quality_report.blocker_reasons)
            ),
            "factors": [],
            "top_negative_factors": [],
            "top_positive_factors": [],
        }

        feature_status_counts = {
            "total_model_features": quality_report.total_model_features,
            "derived_count": quality_report.derived_count,
            "self_reported_count": quality_report.self_reported_count,
            "unavailable_count": quality_report.unavailable_count,
            "imputed_count": quality_report.imputed_count,
        }

        parser_quality = {
            "rows_received": quality_report.rows_received,
            "rows_accepted": quality_report.rows_accepted,
            "rows_rejected": quality_report.rows_rejected,
            "parse_success_rate": round(
                (quality_report.rows_accepted / max(quality_report.rows_received, 1)) * 100.0, 2
            ),
        }

        return RealDataAssessmentResult(
            borrower_id=borrower_id_val,
            raw_model_probability=0.0,
            calibrated_model_probability=0.0,
            credit_score=0,
            risk_tier="Unscoreable — Insufficient Statement Evidence",
            assessment_status=quality_report.overall_status,
            is_scoreable=False,
            evidence_coverage_ratio=quality_report.evidence_coverage_ratio,
            history_months=quality_report.history_months,
            feature_status_counts=feature_status_counts,
            distribution_flags=[],
            parser_quality=parser_quality,
            explanation=empty_explanation,
            quality_warnings=quality_report.quality_warnings,
            blocker_reasons=quality_report.blocker_reasons,
        )

    # 2. Enforce Strict Pre-Inference Validation Gate
    validate_model_input_schema(borrower_df)

    # 3. Load Model Infrastructure
    if model_bundle is None:
        model_bundle = load_model_bundle()

    model = model_bundle.get("model")
    pipeline = model_bundle.get("pipeline")

    if model is None or pipeline is None:
        raise PreprocessingFailureError("Model bundle does not contain 'model' and 'pipeline' objects.")

    # 4. Feature Transformation Through Fitted Pipeline
    try:
        X_transformed = pipeline.transform(borrower_df)
    except Exception as e:
        raise PreprocessingFailureError(f"Failed to transform borrower features through FeaturePipeline: {e}")

    # Verify transformed dimensions (30 features expected: 21 numeric/composite + 6 occupation one-hot + 3 city one-hot)
    expected_dim = len(pipeline.feature_columns_)
    if X_transformed.shape != (1, expected_dim):
        raise PreprocessingFailureError(
            f"Transformed feature matrix shape mismatch: expected (1, {expected_dim}), got {X_transformed.shape}"
        )

    if X_transformed.isna().any().any():
        raise PreprocessingFailureError("Transformed feature matrix contains unexpected null values after imputation.")

    # 5. Model Inference (Raw Default Probability)
    try:
        prob_raw = float(model.predict_proba(X_transformed)[0, 1])
    except Exception as e:
        raise PredictionFailureError(f"Fitted model inference failed: {e}")

    if not (0.0 <= prob_raw <= 1.0):
        raise PredictionFailureError(f"Model returned invalid probability outside [0, 1]: {prob_raw}")

    # 6. Bayesian Prior Calibration (Thin-File Prior Adjustment)
    # Recovers empirical 14% thin-file baseline default odds from balanced training weights
    odds_raw = prob_raw / max(1.0 - prob_raw, 1e-6)
    odds_calibrated = odds_raw * (POPULATION_DEFAULT_RATE / (1.0 - POPULATION_DEFAULT_RATE))
    prob_calibrated = float(odds_calibrated / (1.0 + odds_calibrated))

    # 7. Credit Score Transformation & Risk Tier Mapping
    try:
        score = int(probability_to_credit_score(prob_calibrated))
        tier = str(score_to_tier(score))
    except Exception as e:
        raise ScoreTransformationFailureError(f"Failed to transform probability to CreditBridge score: {e}")

    # 8. Local SHAP Attribution & Explainability
    try:
        explanation = explain_borrower_record(
            borrower_data=borrower_df,
            model_bundle=model_bundle,
            provenance_records=payload.provenance_records,
            save_waterfall_path=save_waterfall_path,
        )
    except Exception as e:
        raise ExplanationFailureError(f"Failed to compute SHAP attributions for borrower: {e}")

    # 9. Distribution Flags Aggregation
    distribution_flags = [
        d.to_dict()
        for d in quality_report.field_diagnostics
        if d.distribution_status in ("outside_observed_range", "near_boundary", "unseen_in_training")
    ]

    # 10. Assemble Metadata Containers
    feature_status_counts = {
        "total_model_features": quality_report.total_model_features,
        "derived_count": quality_report.derived_count,
        "self_reported_count": quality_report.self_reported_count,
        "unavailable_count": quality_report.unavailable_count,
        "imputed_count": quality_report.imputed_count,
    }

    parser_quality = {
        "rows_received": quality_report.rows_received,
        "rows_accepted": quality_report.rows_accepted,
        "rows_rejected": quality_report.rows_rejected,
        "parse_success_rate": round(
            (quality_report.rows_accepted / max(quality_report.rows_received, 1)) * 100.0, 2
        ),
    }

    # 11. Compile Warnings (Quality warnings + any active distribution warnings)
    combined_warnings = list(quality_report.quality_warnings)
    if quality_report.overall_status == AssessmentQualityStatus.LIMITED.value:
        combined_warnings.insert(
            0,
            "Notice: Assessment evaluated under LIMITED quality status. Scoring permitted for manual underwriting review."
        )

    return RealDataAssessmentResult(
        borrower_id=borrower_id_val,
        raw_model_probability=prob_raw,
        calibrated_model_probability=prob_calibrated,
        credit_score=score,
        risk_tier=tier,
        assessment_status=quality_report.overall_status,
        is_scoreable=True,
        evidence_coverage_ratio=quality_report.evidence_coverage_ratio,
        history_months=quality_report.history_months,
        feature_status_counts=feature_status_counts,
        distribution_flags=distribution_flags,
        parser_quality=parser_quality,
        explanation=explanation,
        quality_warnings=combined_warnings,
        blocker_reasons=quality_report.blocker_reasons,
    )


# -----------------------------------------------------------------------------
# 3. END-TO-END ORCHESTRATION PIPELINE
# -----------------------------------------------------------------------------

def assess_statement_end_to_end(
    statement_input: Union[str, bytes],
    manual_inputs: ManualInputContract,
    model_bundle: Optional[Dict[str, Any]] = None,
    save_waterfall_path: Optional[str] = None,
) -> RealDataAssessmentResult:
    """
    Convenience end-to-end pipeline:
    CSV Statement -> Parsing -> Transaction Classification ->
    Feature Mapping & Provenance -> Data Quality & Distribution Diagnostics ->
    Model Inference, Calibration, Scoring, and SHAP Explainability.

    Parameters:
    -----------
    statement_input : str or bytes
        Path to statement CSV file, raw CSV string, or bytes content.
    manual_inputs : ManualInputContract
        Self-reported user parameters (age, occupation_type, city_tier).
    model_bundle : Optional[dict]
        Cached model artifact bundle.
    save_waterfall_path : Optional[str]
        Optional output path for SHAP waterfall figure.

    Returns:
    --------
    RealDataAssessmentResult:
        Consolidated, immutable evaluation artifact.
    """
    # 1. Parse Statement CSV
    parsed = parse_csv_statement(statement_input)

    # 2. Classify Transactions
    classified_tx = classify_transactions_df(
        parsed.transactions,
        account_holder_name=manual_inputs.account_holder_name,
    )

    # 3. Feature Mapping & Provenance Manifest
    payload = build_real_borrower_payload(
        statement_or_df=classified_tx,
        manual_inputs=manual_inputs,
        allow_insufficient_history=True,
    )

    # 4. Data Quality, Evidence Coverage & Distribution Diagnostics
    quality_report = build_data_quality_report(
        statement=parsed,
        borrower_payload=payload,
    )

    # 5. Model Inference, Scoring, and Explanation
    return score_real_borrower_payload(
        payload=payload,
        quality_report=quality_report,
        model_bundle=model_bundle,
        save_waterfall_path=save_waterfall_path,
    )
