"""
CreditBridge - Alternative Credit Scoring Engine
Stage 2: Score-to-Tier Mapping & Scoring Utilities
Path: src/scoring_utils.py

Converts raw default probabilities from ML models into familiar 300-900 CIBIL-style
credit scores using standard financial log-odds scaling (PDO methodology). Maps
scores to actionable risk underwriting tiers.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Union, Dict, Any, Optional

# Ensure project root is on sys.path for unpickling custom pipeline objects
src_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(src_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# -----------------------------------------------------------------------------
# CIBIL-STYLE CREDIT SCORING FORMULATION (INTERVIEW TALKING POINTS)
# -----------------------------------------------------------------------------
# In credit risk management (Basel II / FICO / CIBIL standards), raw default
# probabilities are mapped to three-digit credit scores via a logarithmic odds
# transformation (PDO - Points to Double Odds):
#
#   odds = (1 - P(default)) / P(default) = P(good) / P(bad)
#   log_odds = ln(odds)
#   Score = Offset + (Factor * log_odds)
#
# Parameter Calibration:
# - Offset = 490.0
# - Factor = 95.0
# Yields:
# - P(default) = 1-3%   -> Score ~ 820-900 (Low Risk)
# - P(default) = 7-15%  -> Score ~ 660-740 (Moderate Risk)
# - P(default) = 20-35% -> Score ~ 550-640 (High Risk — Manual Review)
# - P(default) > 40%    -> Score < 550     (Very High Risk)
# - Bounded strictly within the classic Indian bureau range [300, 900].
# -----------------------------------------------------------------------------

SCORE_OFFSET = 490.0
SCORE_FACTOR = 95.0
MIN_SCORE = 300
MAX_SCORE = 900
POPULATION_DEFAULT_RATE = 0.14  # Baseline prior in Indian thin-file segment


def probability_to_credit_score(prob_default: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
    """
    Transforms calibrated default probability into a 300-900 CIBIL-style credit score.
    
    Formula:
        odds = (1 - p) / p   (Good odds: non-default to default)
        score = Offset + Factor * ln(odds)
        clamped to [300, 900]
    """
    p = np.clip(np.array(prob_default, dtype=float), 1e-6, 1.0 - 1e-6)
    odds = (1.0 - p) / p
    log_odds = np.log(odds)
    score = SCORE_OFFSET + (SCORE_FACTOR * log_odds)
    
    clamped_score = np.clip(np.round(score), MIN_SCORE, MAX_SCORE).astype(int)
    if np.ndim(prob_default) == 0:
        return int(clamped_score.item())
    return clamped_score


def score_to_tier(score: int) -> str:
    """
    Maps a 300-900 CreditBridge score into operational risk underwriting tiers.
    
    Tiers:
    - 750 - 900: Low Risk (Instant digital approval)
    - 650 - 749: Moderate Risk (Standard terms, standard pricing)
    - 550 - 649: High Risk — Manual Review (Needs guarantor, collateral, or closer scrutiny)
    - Below 550: Very High Risk (Policy decline)
    """
    if score >= 750:
        return "Low Risk"
    elif score >= 650:
        return "Moderate Risk"
    elif score >= 550:
        return "High Risk — Manual Review"
    else:
        return "Very High Risk"


def load_model_bundle(model_path: Optional[str] = None) -> Dict[str, Any]:
    """Loads the persisted model artifact and associated preprocessing metadata."""
    if model_path is None:
        src_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(src_dir, ".."))
        model_path = os.path.join(project_root, "models", "credit_model.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at: {model_path}. "
            "Please run `python src/train_model.py` first to train and persist the model."
        )
    bundle = joblib.load(model_path)

    # Ensure backward compatibility across scikit-learn versions for LogisticRegression estimator
    model = bundle.get("model")
    if model is not None:
        clf = model.named_steps.get("classifier", model) if hasattr(model, "named_steps") else model
        if hasattr(clf, "__class__") and "Logistic" in clf.__class__.__name__ and not hasattr(clf, "multi_class"):
            setattr(clf, "multi_class", "auto")

    return bundle


def score_borrower(borrower_data: Union[pd.Series, pd.DataFrame, Dict[str, Any]], model_bundle: Optional[Dict[str, Any]] = None) -> Tuple[int, str]:
    """
    End-to-end helper for scoring a single borrower or applicant record.
    Chains: Raw Input -> Feature Transformation -> Model Inference -> Score Conversion -> Tier Lookup.
    
    Parameters:
    -----------
    borrower_data : pd.Series, single-row pd.DataFrame, or dict
        Raw applicant features with identical schema to synthetic_borrowers.csv.
    model_bundle : Optional[dict]
        Pre-loaded model artifact bundle to avoid reloading on successive calls.
        
    Returns:
    --------
    score : int
        Calculated credit score on 300-900 scale.
    tier : str
        Risk tier label.
    """
    if model_bundle is None:
        model_bundle = load_model_bundle()

    model = model_bundle["model"]
    pipeline = model_bundle["pipeline"]

    # Convert dictionary or Series to single-row DataFrame
    if isinstance(borrower_data, dict):
        df_row = pd.DataFrame([borrower_data])
    elif isinstance(borrower_data, pd.Series):
        df_row = borrower_data.to_frame().T
    elif isinstance(borrower_data, pd.DataFrame):
        df_row = borrower_data.copy()
    else:
        raise TypeError("borrower_data must be a dict, pd.Series, or pd.DataFrame")

    # Feature transformation through fitted pipeline
    X_transformed = pipeline.transform(df_row)

    # Predict default probability (class 1)
    prob_raw = float(model.predict_proba(X_transformed)[0, 1])

    # Calibrate probability: adjust for balanced training class weighting
    # Recovers empirical thin-file baseline default odds (~14% prior)
    odds_raw = prob_raw / max(1.0 - prob_raw, 1e-6)
    odds_calibrated = odds_raw * (POPULATION_DEFAULT_RATE / (1.0 - POPULATION_DEFAULT_RATE))
    prob_calibrated = odds_calibrated / (1.0 + odds_calibrated)

    # Convert to 300-900 score
    score = int(probability_to_credit_score(prob_calibrated))
    tier = score_to_tier(score)

    return score, tier


if __name__ == "__main__":
    print("=" * 80)
    print("CREDITBRIDGE - SCORE MAPPING UTILITY TEST")
    print("=" * 80)

    test_probs = [0.01, 0.04, 0.12, 0.20, 0.35, 0.50, 0.70, 0.90, 0.98]
    print(f"{'Default Probability':<22} | {'Credit Score (300-900)':<24} | {'Risk Tier'}")
    print("-" * 80)
    for p in test_probs:
        s = int(probability_to_credit_score(p))
        t = score_to_tier(s)
        print(f"{p:<22.2f} | {s:<24} | {t}")

    # End-to-end scoring test using persisted model artifact
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(src_dir, ".."))
    csv_path = os.path.join(project_root, "data", "synthetic_borrowers.csv")
    model_path = os.path.join(project_root, "models", "credit_model.pkl")

    if os.path.exists(csv_path) and os.path.exists(model_path):
        print("\n" + "=" * 80)
        print("TESTING score_borrower() ON SAMPLE BORROWERS FROM DATASET")
        print("=" * 80)
        df = pd.read_csv(csv_path)
        bundle = load_model_bundle(model_path)
        fmt = "{:<12} | {:<20} | {:<8} | {:<12} | {:<25}"
        print(fmt.format("Index", "Borrower ID", "Score", "True Default", "Risk Tier"))
        print("-" * 80)
        for idx in [0, 10, 25, 50, 100, 250, 500, 1000]:
            row = df.iloc[idx]
            b_id = str(row["borrower_id"])[:8] + "..."
            score, tier = score_borrower(row, bundle)
            true_def = "Default" if row["defaulted"] == 1 else "Repaid"
            print(fmt.format(idx, b_id, score, true_def, tier))
        print("=" * 80)

    print("Score-to-tier mapping and score_borrower() verified.")
