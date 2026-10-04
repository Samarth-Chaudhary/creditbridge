"""
CreditBridge - Alternative Credit Scoring Engine
Stage 2: Model Training, Comparison & Evaluation
Path: src/train_model.py

Trains two contrasting models on alternative credit proxy signals:
1. Logistic Regression (Interpretable baseline required by RBI / BFSI risk committees)
2. XGBoost Classifier (High-capacity non-linear gradient boosted trees)

Evaluates performance using AUC-ROC, Precision, Recall, F1, and the KS-Statistic
(Kolmogorov-Smirnov), the gold-standard metric utilized by Indian NBFC risk teams.
Saves the champion model and preprocessing pipeline to models/credit_model.pkl.
"""

import os
import sys
from typing import Any, Dict, Union, cast

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Ensure src can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)



# -----------------------------------------------------------------------------
# DOMAIN & METHODOLOGICAL RATIONALE (INTERVIEW TALKING POINTS)
# -----------------------------------------------------------------------------
# 1. Class Imbalance Strategy (class_weight='balanced' vs SMOTE):
#    - We deliberately chose cost-sensitive learning (algorithmic class weighting)
#      rather than SMOTE (Synthetic Minority Over-sampling Technique).
#    - Rationale: SMOTE synthesizes artificial data points via Euclidean k-NN
#      interpolation. In thin-file financial proxy datasets with mixed discrete/continuous
#      signals (e.g. integer recharge lapses, binary one-hot occupations, bounded ratios),
#      synthetic interpolation creates chemically invalid financial personas (e.g.,
#      a non-gig worker with partial gig hours, or impossible transaction count fractions).
#    - Algorithmic weighting (scale_pos_weight in XGBoost, class_weight='balanced' in Logistic
#      Regression) adjusts the loss function gradient without altering the empirical
#      data topology or inflating false positive rates.
#
# 2. Metric Selection & The KS-Statistic (Kolmogorov-Smirnov):
#    - While AUC-ROC evaluates global ranking, Indian NBFC and bank credit risk teams
#      almost universally track the KS-statistic for regulatory underwriting approval.
#    - Definition: The KS-statistic measures the maximum vertical divergence between the
#      cumulative distribution function of defaulters ('bads') and non-defaulters ('goods').
#    - In retail/alt-lending, a KS between 35% and 55% represents a sweet spot of strong
#      separation power without dangerous over-fitting.
# -----------------------------------------------------------------------------


def compute_ks_statistic(y_true: Union[np.ndarray, pd.Series, Any], y_prob: Union[np.ndarray, pd.Series, Any]) -> float:
    """
    Calculates the Kolmogorov-Smirnov (KS) statistic.
    Measures the maximum separation between the CDF of defaulters and non-defaulters.
    """
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    prob_bads = y_prob_arr[y_true_arr == 1]
    prob_goods = y_prob_arr[y_true_arr == 0]
    ks_result = ks_2samp(prob_bads, prob_goods)
    stat_val = float(cast(Any, ks_result).statistic)
    return round(stat_val * 100.0, 2)


def evaluate_model(model: Any, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5) -> Dict[str, Any]:
    """Computes credit risk evaluation metrics including AUC, F1, and KS-statistic."""
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    auc = float(roc_auc_score(y, y_prob))
    prec = float(precision_score(y, y_pred, zero_division=cast(Any, 0.0)))
    rec = float(recall_score(y, y_pred, zero_division=cast(Any, 0.0)))
    f1 = float(f1_score(y, y_pred, zero_division=cast(Any, 0.0)))
    ks = compute_ks_statistic(y.to_numpy(), y_prob)
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred, target_names=["Non-Default", "Default"], zero_division=cast(Any, 0.0))

    return {
        "auc": round(auc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "ks_stat": ks,
        "confusion_matrix": cm,
        "classification_report": report,
        "probabilities": y_prob
    }


def train_and_evaluate() -> Dict[str, Any]:
    csv_path = os.path.join(project_root, "data", "temporal_synthetic_borrowers.csv")

    # Load or generate temporal dataset
    if os.path.exists(csv_path):
        raw_df = pd.read_csv(csv_path)
    else:
        print("Generating new temporal defensible synthetic dataset...")
        from src.temporal_data_generator import generate_full_temporal_dataset
        raw_df = generate_full_temporal_dataset(seed=42)
        raw_df.to_csv(csv_path, index=False)
        print(f"Persisted temporal dataset to: {csv_path}")

    print(f"Loading data from: {csv_path} (shape: {raw_df.shape})")

    from src.model_suite import run_full_training_suite
    results = run_full_training_suite(
        dataset_df=raw_df,
        random_seed=42,
        experiment_id_prefix="exp_phase1",
        persist_champion_artifact=True,
    )

    all_m = results["all_metrics"]
    lr_oot = all_m["logistic_regression"]
    xgb_oot = all_m["xgboost"]
    base_oot = all_m["base_rate_baseline"]

    print("\n" + "=" * 92)
    print(f"{'CREDIT RISK EVALUATION RESULTS (OUT-OF-TIME OOT SET)':^92}")
    print("=" * 92)
    hdr = f"{'Metric':<25} | {'Base-Rate Baseline':<20} | {'Logistic Regression':<20} | {'XGBoost Classifier':<20}"
    print(hdr)
    print("-" * 92)
    print(f"{'ROC-AUC':<25} | {base_oot['roc_auc']:<20.4f} | {lr_oot['roc_auc']:<20.4f} | {xgb_oot['roc_auc']:<20.4f}")
    print(f"{'PR-AUC':<25} | {base_oot['pr_auc']:<20.4f} | {lr_oot['pr_auc']:<20.4f} | {xgb_oot['pr_auc']:<20.4f}")
    print(f"{'KS-Statistic (%)':<25} | {base_oot['ks_statistic']:<20.2f} | {lr_oot['ks_statistic']:<20.2f} | {xgb_oot['ks_statistic']:<20.2f}")
    print(f"{'Gini':<25} | {base_oot['gini']:<20.4f} | {lr_oot['gini']:<20.4f} | {xgb_oot['gini']:<20.4f}")
    print(f"{'Brier Score':<25} | {base_oot['brier_score']:<20.4f} | {lr_oot['brier_score']:<20.4f} | {xgb_oot['brier_score']:<20.4f}")
    print(f"{'Precision':<25} | {base_oot['precision']:<20.4f} | {lr_oot['precision']:<20.4f} | {xgb_oot['precision']:<20.4f}")
    print(f"{'Recall':<25} | {base_oot['recall']:<20.4f} | {lr_oot['recall']:<20.4f} | {xgb_oot['recall']:<20.4f}")
    print(f"{'F1 Score':<25} | {base_oot['f1']:<20.4f} | {lr_oot['f1']:<20.4f} | {xgb_oot['f1']:<20.4f}")
    print("=" * 92)

    champ_sel = results["champion_selection"]
    print(f"\nChampion Selection Decision: {champ_sel['decision']}")
    print(f"Selected Champion: {results['champion_name']}")
    print(f"Rationale: {champ_sel['rationale']}")
    print(f"Experiment Directory: {results['experiment_dir']}")

    return results


if __name__ == "__main__":
    train_and_evaluate()
