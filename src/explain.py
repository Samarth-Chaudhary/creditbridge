"""
CreditBridge - Alternative Credit Scoring Engine
Stage 2: Explainability & RBI Fair-Lending Compliance Module
Path: src/explain.py

Leverages SHAP (SHapley Additive exPlanations) to produce:
1. Global Feature Importance (summary beeswarm plot saved to assets/shap_summary_plot.png)
2. Local Individual Explanations with Waterfall plots and dynamic plain-English reasoning
   for adverse-action and fair-lending regulatory defensibility.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Union

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Non-interactive headless backend for clean plot saving
import matplotlib.pyplot as plt
import shap  # type: ignore

# Ensure src can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.real_data_contracts import (
        UNAVAILABLE_MODEL_COLUMNS,
        FactorContribution,
    )
    from src.scoring_utils import (
        POPULATION_DEFAULT_RATE,
        SCORE_FACTOR,
        load_model_bundle,
        probability_to_credit_score,
        score_borrower,
        score_to_tier,
    )
except ImportError:
    from real_data_contracts import (  # type: ignore
        UNAVAILABLE_MODEL_COLUMNS,
        FactorContribution,
    )
    from scoring_utils import (  # type: ignore
        POPULATION_DEFAULT_RATE,
        SCORE_FACTOR,
        load_model_bundle,
        probability_to_credit_score,
        score_borrower,
        score_to_tier,
    )


# Feature human-readable financial descriptors
FEATURE_NAME_MAP = {
    "age": "applicant age",
    "upi_inflow_volatility_coefficient": "UPI inflow volatility",
    "recharge_amount_volatility": "telecom recharge ticket volatility",
    "income_stability_index": "composite income stability index",
    "payment_reliability_score": "utility & gig payment reliability score",
    "electricity_bill_ontime_rate": "utility bill on-time payment track record",
    "electricity_bill_avg_delay_days": "utility bill payment delay duration",
    "days_since_last_recharge_lapse": "tenure without telecom balance lapses",
    "p2p_vs_merchant_txn_ratio": "P2P transfer concentration (informal credit proxy)",
    "monthly_upi_transaction_count": "monthly UPI transaction frequency",
    "monthly_upi_inflow_avg": "monthly digital cash inflow volume",
    "monthly_upi_outflow_avg": "monthly digital expense outflow",
    "monthly_income_estimate": "inferred monthly earning capacity",
    "phone_number_tenure_months": "mobile number tenure (identity stability)",
    "app_account_age_months": "fintech app account vintage",
    "recharge_frequency_per_month": "mobile recharge frequency",
    "avg_recharge_amount": "average mobile recharge ticket size",
    "active_weeks_last_6_months": "gig platform active work consistency",
    "gig_platform_rating": "gig platform service rating",
    "avg_weekly_gig_hours": "weekly platform logged-in hours",
    "earnings_coefficient_of_variation": "gig earnings volatility",
    "occupation_type_daily_wage_labor": "daily wage labor occupation",
    "occupation_type_freelance_digital": "freelance digital occupation",
    "occupation_type_gig_delivery": "gig delivery occupation",
    "occupation_type_gig_rideshare": "gig rideshare occupation",
    "occupation_type_informal_retail": "informal retail occupation",
    "occupation_type_small_trader": "small trader occupation",
    "city_tier_tier_1": "Tier 1 urban location",
    "city_tier_tier_2": "Tier 2 urban location",
    "city_tier_tier_3": "Tier 3 urban location",
}


def get_explainer_and_data(model_bundle: Optional[dict] = None, num_samples: int = 1200):
    """Initializes SHAP explainer and prepares background dataset."""
    if model_bundle is None:
        model_bundle = load_model_bundle()

    model = model_bundle["model"]
    pipeline = model_bundle["pipeline"]

    csv_path = os.path.join(project_root, "data", "synthetic_borrowers.csv")
    raw_df = pd.read_csv(csv_path)

    # Subsample for fast, stable SHAP calculation
    sample_df = raw_df.head(num_samples)
    X_sample = pipeline.transform(sample_df)

    # Instantiate appropriate explainer depending on model class
    # For tree-based models (XGBoost)
    if hasattr(model, "get_booster") or "XGB" in model.__class__.__name__:
        explainer = shap.TreeExplainer(model)
    elif hasattr(model, "named_steps") and "classifier" in model.named_steps:
        # Logistic Regression inside Pipeline
        classifier = model.named_steps["classifier"]
        scaler = model.named_steps["scaler"]
        X_scaled = scaler.transform(X_sample)
        explainer = shap.LinearExplainer(classifier, X_scaled)
    else:
        explainer = shap.Explainer(model, X_sample)

    return explainer, X_sample, raw_df, model_bundle


def generate_global_shap_summary(output_path: Optional[str] = None) -> str:
    """
    Generates and saves global SHAP summary beeswarm plot.
    Saved to assets/shap_summary_plot.png.
    """
    if output_path is None:
        assets_dir = os.path.join(project_root, "assets")
        os.makedirs(assets_dir, exist_ok=True)
        output_path = os.path.join(assets_dir, "shap_summary_plot.png")

    explainer, X_sample, _, _ = get_explainer_and_data()

    print("Computing SHAP values for global summary plot...")
    shap_values = explainer(X_sample)

    plt.figure(figsize=(10, 7), dpi=150)
    shap.summary_plot(
        shap_values,
        X_sample,
        show=False,
        max_display=15
    )
    plt.title("CreditBridge: Global SHAP Feature Attribution (Alt-Credit Drivers)", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"Global SHAP summary plot successfully saved to: {output_path}")
    return output_path


def explain_single_borrower(borrower_id: str, df: Optional[pd.DataFrame] = None, save_waterfall_path: Optional[str] = None) -> dict:
    """
    Generates local SHAP explanation and plain-English adverse-action / approval summary
    for a specific borrower.

    Parameters:
    -----------
    borrower_id : str
        Target borrower UUID.
    df : Optional[pd.DataFrame]
        Dataframe to search for borrower (defaults to synthetic_borrowers.csv).
    save_waterfall_path : Optional[str]
        Optional file path to save individual waterfall plot.

    Returns:
    --------
    dict containing:
      - 'borrower_id': UUID
      - 'credit_score': 300-900 score
      - 'risk_tier': classification
      - 'plain_english_explanation': regulatory-defensible explanation text
      - 'top_negative_factors': list of (feature, point_impact)
      - 'top_positive_factors': list of (feature, point_impact)
      - 'shap_values': numpy array
    """
    if df is None:
        csv_path = os.path.join(project_root, "data", "synthetic_borrowers.csv")
        df = pd.read_csv(csv_path)

    match = df[df["borrower_id"] == borrower_id]
    if match.empty:
        raise ValueError(f"Borrower ID '{borrower_id}' not found in dataset.")

    borrower_row = match.iloc[0]
    model_bundle = load_model_bundle()
    model = model_bundle["model"]
    pipeline = model_bundle["pipeline"]

    # Transform single row
    X_single = pipeline.transform(match)
    prob_default = float(model.predict_proba(X_single)[0, 1])
    score, tier = score_borrower(borrower_row, model_bundle)

    # Compute SHAP values for single applicant
    if hasattr(model, "get_booster") or "XGB" in model.__class__.__name__:
        explainer = shap.TreeExplainer(model)
        shap_explanation = explainer(X_single)
        raw_vals = getattr(shap_explanation, "values", shap_explanation)
    elif hasattr(model, "named_steps") and "classifier" in model.named_steps:
        classifier = model.named_steps["classifier"]
        scaler = model.named_steps["scaler"]
        X_scaled = scaler.transform(X_single)
        explainer = shap.LinearExplainer(classifier, scaler.transform(pipeline.transform(df.head(500))))
        shap_explanation = explainer(X_scaled)
        raw_vals = getattr(shap_explanation, "values", shap_explanation)
    else:
        explainer = shap.Explainer(model, pipeline.transform(df.head(500)))
        shap_explanation = explainer(X_single)
        raw_vals = getattr(shap_explanation, "values", shap_explanation)

    raw_shap_values: np.ndarray = np.asarray(raw_vals[0], dtype=float)

    feature_names = list(X_single.columns)

    # Calculate point impact:
    # In CreditBridge log-odds scaling: Score = Offset + Factor * ln(odds)
    # Higher SHAP value -> higher default probability -> lower credit score.
    # Therefore, points delta = -round(SHAP * SCORE_FACTOR)
    point_deltas = -np.round(raw_shap_values * SCORE_FACTOR).astype(int)

    # Sort drivers by absolute magnitude of credit score impact
    ranked_indices = np.argsort(-np.abs(point_deltas))

    neg_drivers = []  # Dragging score DOWN (point_delta < 0, meaning positive SHAP / default risk)
    pos_drivers = []  # Pushing score UP (point_delta > 0, meaning negative SHAP / creditworthy)

    for idx in ranked_indices:
        feat = feature_names[idx]
        delta = int(point_deltas[idx])
        human_name = FEATURE_NAME_MAP.get(feat, feat.replace("_", " "))

        if delta < 0:
            neg_drivers.append((human_name, delta))
        elif delta > 0:
            pos_drivers.append((human_name, delta))

    # Construct genuine dynamic plain-English explanation
    explanation_parts = []
    if neg_drivers and pos_drivers:
        top_neg = neg_drivers[0]
        top_pos = pos_drivers[0]
        explanation_parts.append(
            f"This applicant's score was primarily lowered by {top_neg[0]} ({top_neg[1]} points) "
            f"and partially offset by strong {top_pos[0]} (+{top_pos[1]} points)."
        )
        if len(neg_drivers) > 1:
            second_neg = neg_drivers[1]
            explanation_parts.append(f"Secondary drag came from {second_neg[0]} ({second_neg[1]} points).")
    elif neg_drivers:
        top_neg = neg_drivers[0]
        explanation_parts.append(
            f"This applicant's score was predominantly constrained by {top_neg[0]} ({top_neg[1]} points)."
        )
        if len(neg_drivers) > 1:
            explanation_parts.append(f"Additional risk indicators included {neg_drivers[1][0]} ({neg_drivers[1][1]} points).")
    elif pos_drivers:
        top_pos = pos_drivers[0]
        explanation_parts.append(
            f"This applicant demonstrated an exceptionally clean credit profile, driven primarily by {top_pos[0]} (+{top_pos[1]} points)."
        )
        if len(pos_drivers) > 1:
            explanation_parts.append(f"Strong support was also provided by {pos_drivers[1][0]} (+{pos_drivers[1][1]} points).")
    else:
        explanation_parts.append("Applicant metrics are aligned with population median baseline.")

    full_explanation = " ".join(explanation_parts)

    # Optionally generate and save single-borrower waterfall plot
    if save_waterfall_path is not None:
        plt.figure(figsize=(9, 5), dpi=140)
        shap.plots.waterfall(shap_explanation[0], show=False, max_display=10)
        plt.title(f"CreditBridge SHAP Waterfall - Borrower {borrower_id[:8]}", fontsize=11)
        plt.tight_layout()
        plt.savefig(save_waterfall_path, bbox_inches="tight")
        plt.close()

    return {
        "borrower_id": borrower_id,
        "default_probability": round(prob_default, 4),
        "credit_score": score,
        "risk_tier": tier,
        "plain_english_explanation": full_explanation,
        "top_negative_factors": neg_drivers[:3],
        "top_positive_factors": pos_drivers[:3],
        "shap_explanation": shap_explanation[0]
    }


def explain_borrower_record(
    borrower_data: Union[pd.Series, pd.DataFrame, Dict[str, Any]],
    model_bundle: Optional[Dict[str, Any]] = None,
    provenance_records: Optional[Dict[str, Any]] = None,
    save_waterfall_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generates local SHAP explanation and transparent factor attributions for an
    arbitrary single borrower record (e.g. from real statement processing) without
    requiring that the borrower exists in synthetic_borrowers.csv.

    Ethical and Regulatory Transparency:
    - Distinguishes model feature importance from factual evidence status
      (derived from statement, self-reported, or unavailable/imputed).
    - Uses non-causal language ('contributed to model output', 'primarily constrained by').
    - Reports both raw_model_probability and calibrated_model_probability unambiguously.

    Parameters:
    -----------
    borrower_data : pd.Series, single-row pd.DataFrame, or dict
        Raw applicant features matching the 22-column legacy model schema.
    model_bundle : Optional[dict]
        Pre-loaded model artifact bundle.
    provenance_records : Optional[dict]
        Mapping from feature name to FeatureProvenanceRecord.
    save_waterfall_path : Optional[str]
        Optional file path to save individual waterfall plot.

    Returns:
    --------
    dict containing:
      - 'borrower_id': UUID / session ID
      - 'raw_model_probability': float
      - 'calibrated_model_probability': float
      - 'credit_score': int (300-900)
      - 'risk_tier': str
      - 'plain_english_explanation': str
      - 'factors': List[dict] (ranked by absolute point impact)
      - 'top_negative_factors': List[dict]
      - 'top_positive_factors': List[dict]
      - 'factor_objects': List[FactorContribution]
    """
    if model_bundle is None:
        model_bundle = load_model_bundle()

    model = model_bundle["model"]
    pipeline = model_bundle["pipeline"]

    # Convert to single-row DataFrame
    if isinstance(borrower_data, dict):
        df_row = pd.DataFrame([borrower_data])
    elif isinstance(borrower_data, pd.Series):
        df_row = borrower_data.to_frame().T
    elif isinstance(borrower_data, pd.DataFrame):
        df_row = borrower_data.copy()
    else:
        raise TypeError("borrower_data must be a dict, pd.Series, or pd.DataFrame")

    borrower_id_val = str(df_row["borrower_id"].iloc[0]) if "borrower_id" in df_row.columns else "borrower_unknown"

    # Transform single row through fitted pipeline
    X_single = pipeline.transform(df_row)
    feature_names = list(X_single.columns)

    # Predict raw default probability
    prob_raw = float(model.predict_proba(X_single)[0, 1])

    # Calibrate probability using population default rate prior
    odds_raw = prob_raw / max(1.0 - prob_raw, 1e-6)
    odds_calibrated = odds_raw * (POPULATION_DEFAULT_RATE / (1.0 - POPULATION_DEFAULT_RATE))
    prob_calibrated = odds_calibrated / (1.0 + odds_calibrated)

    score = int(probability_to_credit_score(prob_calibrated))
    tier = score_to_tier(score)

    # Load reference background data for SHAP computation
    csv_path = os.path.join(project_root, "data", "synthetic_borrowers.csv")
    if os.path.exists(csv_path):
        bg_df = pd.read_csv(csv_path).head(300)
        X_bg = pipeline.transform(bg_df)
    else:
        X_bg = X_single

    # Compute SHAP attributions
    if hasattr(model, "named_steps") and "classifier" in model.named_steps:
        classifier = model.named_steps["classifier"]
        scaler = model.named_steps["scaler"]
        X_scaled_bg = scaler.transform(X_bg)
        X_scaled_single = scaler.transform(X_single)
        explainer = shap.LinearExplainer(classifier, X_scaled_bg)
        shap_explanation = explainer(X_scaled_single)
        raw_vals = getattr(shap_explanation, "values", shap_explanation)
    elif hasattr(model, "get_booster") or "XGB" in model.__class__.__name__:
        explainer = shap.TreeExplainer(model)
        shap_explanation = explainer(X_single)
        raw_vals = getattr(shap_explanation, "values", shap_explanation)
    else:
        explainer = shap.Explainer(model, X_bg)
        shap_explanation = explainer(X_single)
        raw_vals = getattr(shap_explanation, "values", shap_explanation)

    raw_shap_values: np.ndarray = np.asarray(raw_vals[0], dtype=float)

    # Convert SHAP values to credit score points impact:
    # Delta = -round(SHAP * SCORE_FACTOR)
    point_deltas = -np.round(raw_shap_values * SCORE_FACTOR).astype(int)

    # Build factor contributions with provenance resolution
    factor_list: List[FactorContribution] = []
    for idx, feat in enumerate(feature_names):
        delta = int(point_deltas[idx])
        shap_val = float(raw_shap_values[idx])
        feat_val = X_single.iloc[0][feat]
        human_name = str(FEATURE_NAME_MAP.get(feat) or feat.replace("_", " "))

        is_imputed = False
        source_status = "derived_from_statement"

        if provenance_records and feat in provenance_records:
            rec = provenance_records[feat]
            is_imputed = getattr(rec, "is_imputed", False)
            p_state = getattr(rec, "provenance_state", "UNKNOWN")
            if is_imputed:
                source_status = "unavailable_imputed"
            elif p_state == "DERIVED":
                source_status = "derived_from_statement"
            elif p_state == "SELF_REPORTED":
                source_status = "self_reported"
            else:
                source_status = p_state.lower()
        elif feat == "income_stability_index":
            source_status = "composite_derived"
        elif feat == "payment_reliability_score":
            source_status = "composite_partially_imputed"
            is_imputed = True
        elif feat.startswith("occupation_type_") or feat.startswith("city_tier_") or feat == "age":
            source_status = "self_reported"
        elif feat in UNAVAILABLE_MODEL_COLUMNS:
            source_status = "unavailable_imputed"
            is_imputed = True
        else:
            source_status = "derived_from_statement"

        contrib = "positive" if delta > 0 else ("negative" if delta < 0 else "neutral")

        factor_list.append(FactorContribution(
            feature_name=feat,
            human_name=human_name,
            point_impact=delta,
            contribution=contrib,
            shap_value=shap_val,
            feature_value=feat_val,
            source_status=source_status,
            is_imputed=is_imputed,
        ))

    # Rank factors by absolute point impact
    ranked_factors = sorted(factor_list, key=lambda f: abs(f.point_impact), reverse=True)
    pos_factors = [f for f in ranked_factors if f.point_impact > 0]
    neg_factors = [f for f in ranked_factors if f.point_impact < 0]

    # Generate transparent plain-English regulatory narrative
    narrative_parts = []
    if neg_factors and pos_factors:
        top_neg = neg_factors[0]
        top_pos = pos_factors[0]
        narrative_parts.append(
            f"This applicant's model score was primarily constrained by {top_neg.human_name} "
            f"({top_neg.point_impact} points, {top_neg.source_status}) "
            f"and positively supported by {top_pos.human_name} "
            f"(+{top_pos.point_impact} points, {top_pos.source_status})."
        )
        if len(neg_factors) > 1:
            second_neg = neg_factors[1]
            narrative_parts.append(
                f"Secondary downward factor: {second_neg.human_name} "
                f"({second_neg.point_impact} points, {second_neg.source_status})."
            )
    elif neg_factors:
        top_neg = neg_factors[0]
        narrative_parts.append(
            f"This applicant's model score was primarily constrained by {top_neg.human_name} "
            f"({top_neg.point_impact} points, {top_neg.source_status})."
        )
        if len(neg_factors) > 1:
            narrative_parts.append(
                f"Secondary downward factor: {neg_factors[1].human_name} "
                f"({neg_factors[1].point_impact} points, {neg_factors[1].source_status})."
            )
    elif pos_factors:
        top_pos = pos_factors[0]
        narrative_parts.append(
            f"This applicant's model score demonstrated positive strength driven by {top_pos.human_name} "
            f"(+{top_pos.point_impact} points, {top_pos.source_status})."
        )
        if len(pos_factors) > 1:
            narrative_parts.append(
                f"Additional positive factor: {pos_factors[1].human_name} "
                f"(+{pos_factors[1].point_impact} points, {pos_factors[1].source_status})."
            )
    else:
        narrative_parts.append("Applicant metrics are aligned with population baseline.")

    # Disclose if top factors relied on imputation
    top_3_imputed = [f for f in ranked_factors[:3] if f.is_imputed]
    if top_3_imputed:
        names = ", ".join(f.human_name for f in top_3_imputed)
        narrative_parts.append(
            f"Notice: Influential factor(s) ({names}) relied on benchmark population median imputation, "
            "as statement evidence did not contain direct telemetry for these signals."
        )

    full_narrative = " ".join(narrative_parts)

    if save_waterfall_path is not None:
        plt.figure(figsize=(9, 5), dpi=140)
        shap.plots.waterfall(shap_explanation[0], show=False, max_display=10)
        plt.title(f"CreditBridge Model Factor Attribution - Borrower {borrower_id_val[:8]}", fontsize=11)
        plt.tight_layout()
        plt.savefig(save_waterfall_path, bbox_inches="tight")
        plt.close()

    return {
        "borrower_id": borrower_id_val,
        "raw_model_probability": round(prob_raw, 4),
        "calibrated_model_probability": round(prob_calibrated, 4),
        "credit_score": score,
        "risk_tier": tier,
        "plain_english_explanation": full_narrative,
        "factors": [f.to_dict() for f in ranked_factors],
        "top_negative_factors": [f.to_dict() for f in neg_factors[:3]],
        "top_positive_factors": [f.to_dict() for f in pos_factors[:3]],
        "factor_objects": factor_list,
    }


if __name__ == "__main__":
    print("=" * 80)
    print("CREDITBRIDGE - SHAP EXPLAINABILITY PIPELINE TEST")
    print("=" * 80)

    # 1. Global summary plot test
    assets_dir = os.path.join(project_root, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    summary_plot_path = os.path.join(assets_dir, "shap_summary_plot.png")
    generate_global_shap_summary(summary_plot_path)

    # 2. Test 3 different borrowers to verify genuinely dynamic, non-template explanations
    csv_path = os.path.join(project_root, "data", "synthetic_borrowers.csv")
    df = pd.read_csv(csv_path)

    # Pick 3 diverse borrowers: one high risk, one moderate, one low risk
    sample_ids = [
        df.iloc[10]["borrower_id"],
        df.iloc[45]["borrower_id"],
        df.iloc[120]["borrower_id"]
    ]

    print("\n" + "=" * 80)
    print("TESTING INDIVIDUAL BORROWER EXPLANATIONS (3 DISTINCT PROFILES)")
    print("=" * 80)
    for i, b_id in enumerate(sample_ids, start=1):
        res = explain_single_borrower(b_id, df=df)
        print(f"\n[Borrower #{i}: {res['borrower_id']}]")
        print(f"Default Probability : {res['default_probability']:.2%}")
        print(f"Calculated Score    : {res['credit_score']} ({res['risk_tier']})")
        print(f"Plain-English Notice: {res['plain_english_explanation']}")
        print(f"Top Negative Factors: {res['top_negative_factors']}")
        print(f"Top Positive Factors: {res['top_positive_factors']}")
    print("=" * 80)
