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
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Union, cast

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from scipy.stats import ks_2samp

import xgboost as xgb  # type: ignore
from xgboost import XGBClassifier  # type: ignore

# Ensure src can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.feature_engineering import FeaturePipeline, prepare_features
except ImportError:
    from feature_engineering import FeaturePipeline, prepare_features  # type: ignore


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


def train_and_evaluate() -> None:
    csv_path = os.path.join(project_root, "data", "synthetic_borrowers.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}. Please run data/generate_synthetic_data.py first.")

    print(f"Loading data from: {csv_path}")
    raw_df = pd.read_csv(csv_path)

    # 1. Feature Engineering with clean Train/Val/Test Isolation
    # Split raw records 70% Train, 15% Validation, 15% Test (stratified on defaulted)
    target_col = "defaulted"
    y_raw = raw_df[target_col]

    # First split: 85% train_val, 15% test
    split1 = train_test_split(
        raw_df,
        test_size=0.15,
        stratify=y_raw,
        random_state=42
    )
    df_train_val = cast(pd.DataFrame, split1[0])
    df_test = cast(pd.DataFrame, split1[1])

    # Second split: from train_val, 70/85 (~82.35%) train, 15/85 (~17.65%) validation
    val_fraction = 0.15 / 0.85
    split2 = train_test_split(
        df_train_val,
        test_size=val_fraction,
        stratify=df_train_val[target_col],
        random_state=42
    )
    df_train = cast(pd.DataFrame, split2[0])
    df_val = cast(pd.DataFrame, split2[1])

    print(f"\n[Dataset Split Summary - 70/15/15 Stratified]")
    print(f"  Training samples:   {len(df_train):>5} (Defaults: {df_train[target_col].sum()})")
    print(f"  Validation samples: {len(df_val):>5} (Defaults: {df_val[target_col].sum()})")
    print(f"  Test samples:       {len(df_test):>5} (Defaults: {df_test[target_col].sum()})")

    # Fit FeaturePipeline strictly on Training split to prevent data leakage
    pipeline = FeaturePipeline()
    pipeline.fit(df_train)

    X_train = pipeline.transform(df_train)
    y_train = cast(pd.Series, df_train[target_col].copy())

    X_val = pipeline.transform(df_val)
    y_val = cast(pd.Series, df_val[target_col].copy())

    X_test = pipeline.transform(df_test)
    y_test = cast(pd.Series, df_test[target_col].copy())

    print(f"Transformed feature count: {X_train.shape[1]}")

    # -------------------------------------------------------------------------
    # 2. MODEL 1: Logistic Regression (Interpretable Baseline)
    # -------------------------------------------------------------------------
    print("\nTraining Model 1: Regularized Logistic Regression (cost-sensitive baseline)...")
    # Standardize features for linear model stability and coefficient interpretability
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    lr_model = LogisticRegression(
        C=0.5,
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )
    lr_model.fit(X_train_scaled, y_train)

    # Wrap in a scikit-learn pipeline for simple downstream inference
    lr_full_pipeline = Pipeline([
        ("scaler", scaler),
        ("classifier", lr_model)
    ])

    lr_metrics = evaluate_model(lr_full_pipeline, X_test, y_test)

    # -------------------------------------------------------------------------
    # 3. MODEL 2: XGBoost Classifier (Performance Model)
    # -------------------------------------------------------------------------
    # Calculate scale_pos_weight for imbalance
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos = n_neg / float(n_pos)

    xgb_model = XGBClassifier(
        n_estimators=180,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        random_state=42,
        eval_metric="logloss"
    )
    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    xgb_metrics = evaluate_model(xgb_model, X_test, y_test)

    # -------------------------------------------------------------------------
    # 4. SIDE-BY-SIDE EVALUATION COMPARISON
    # -------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print(f"{'CREDIT RISK EVALUATION RESULTS (TEST SET)':^88}")
    print("=" * 88)
    hdr = f"{'Metric':<25} | {'Logistic Regression':<28} | {'XGBoost Classifier':<28}"
    print(hdr)
    print("-" * 88)
    print(f"{'AUC-ROC':<25} | {lr_metrics['auc']:<28.4f} | {xgb_metrics['auc']:<28.4f}")
    print(f"{'KS-Statistic (%)':<25} | {lr_metrics['ks_stat']:<28.2f} | {xgb_metrics['ks_stat']:<28.2f}")
    print(f"{'Precision':<25} | {lr_metrics['precision']:<28.4f} | {xgb_metrics['precision']:<28.4f}")
    print(f"{'Recall':<25} | {lr_metrics['recall']:<28.4f} | {xgb_metrics['recall']:<28.4f}")
    print(f"{'F1 Score':<25} | {lr_metrics['f1']:<28.4f} | {xgb_metrics['f1']:<28.4f}")
    print("=" * 88)

    print("\n--- Confusion Matrix (Logistic Regression) ---")
    print(lr_metrics["confusion_matrix"])

    print("\n--- Confusion Matrix (XGBoost Classifier) ---")
    print(xgb_metrics["confusion_matrix"])

    print("\n--- Classification Report (XGBoost) ---")
    print(xgb_metrics["classification_report"])

    # -------------------------------------------------------------------------
    # 5. MODEL SELECTION & ARTIFACT PERSISTENCE
    # -------------------------------------------------------------------------
    # Champion selection based on AUC-ROC and KS-statistic
    if xgb_metrics["auc"] >= lr_metrics["auc"]:
        champion_name = "XGBoost Classifier"
        champion_model = xgb_model
        champion_metrics = xgb_metrics
    else:
        champion_name = "Logistic Regression"
        champion_model = lr_full_pipeline
        champion_metrics = lr_metrics

    print(f"\nChampion Model Selected: {champion_name} (AUC: {champion_metrics['auc']:.4f}, KS: {champion_metrics['ks_stat']:.2f}%)")

    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_save_path = os.path.join(models_dir, "credit_model.pkl")

    # Bundle model, fitted feature pipeline, feature column names, and test metrics
    bundle = {
        "model": champion_model,
        "model_name": champion_name,
        "pipeline": pipeline,
        "feature_names": list(X_train.columns),
        "test_metrics": champion_metrics,
        "all_metrics": {
            "logistic_regression": lr_metrics,
            "xgboost": xgb_metrics
        }
    }

    joblib.dump(bundle, model_save_path)
    print(f"Persisted champion model bundle to: {model_save_path}")


if __name__ == "__main__":
    train_and_evaluate()
