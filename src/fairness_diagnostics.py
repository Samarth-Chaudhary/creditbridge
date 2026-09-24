"""
CreditBridge - Alternative Credit Scoring Engine
Real Data Mode: Fairness & Bias Governance Engine
Path: src/fairness_diagnostics.py

Evaluates model score distributions, approval proxy rates, Adverse Impact Ratios (AIR),
and empirical default rates across sensitive demographic and socioeconomic subgroups
(age bands, occupation types, and city tiers).

Implements explicit Responsible AI governance, non-causal attribution boundaries,
and small-sample statistical validity disclosures.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.scoring_utils import (
    load_model_bundle,
    probability_to_credit_score,
    score_to_tier,
    POPULATION_DEFAULT_RATE,
)

# -----------------------------------------------------------------------------
# 1. RESPONSIBLE AI DISCLOSURES & GOVERNANCE STATEMENTS
# -----------------------------------------------------------------------------

GOVERNANCE_DISCLOSURES: Dict[str, str] = {
    "fairness_diagnostic_limitation": (
        "Synthetic fairness analysis is an exploratory diagnostics tool on simulated "
        "distributions. It demonstrates technical evaluation capability but does NOT "
        "prove or certify the absence of discrimination or disparate impact in real-world lending."
    ),
    "shap_non_causality_boundary": (
        "SHAP values reflect mathematical feature attributions within the trained "
        "model manifold. They measure associative model sensitivity, NOT real-world causality, "
        "borrower intent, or character."
    ),
    "illustrative_score_boundary": (
        "The CreditBridge Model Score (300-900) is an illustrative prototype output. "
        "It is NOT a regulated credit bureau score (e.g., CIBIL, Experian, Equifax, CRIF High Mark) "
        "and must never be represented as such."
    ),
    "purpose_limitation": (
        "CreditBridge V2 is a research prototype. It is explicitly prohibited from being "
        "used for automated credit approvals/rejections, loan pricing, collections prioritization, "
        "employment screening, or generalized consumer profiling."
    ),
    "small_sample_interpretation_rule": (
        "Subgroup metrics with sample size N < 30 suffer from high statistical variance. "
        "Disparate impact metrics on small cohorts must not be treated as definitive indicators."
    ),
}


# -----------------------------------------------------------------------------
# 2. SUBGROUP FAIRNESS CONTRACTS
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SubgroupMetric:
    """Detailed performance, score distribution, and fairness metrics for a cohort slice."""
    dimension: str
    subgroup: str
    sample_size: int
    sample_share: float
    is_small_sample: bool
    mean_score: float
    median_score: float
    std_score: float
    min_score: int
    max_score: int
    low_risk_pct: float
    moderate_risk_pct: float
    high_risk_pct: float
    very_high_risk_pct: float
    approval_proxy_rate: float
    adverse_impact_ratio: float
    empirical_default_rate: Optional[float] = None
    subgroup_auc: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "subgroup": self.subgroup,
            "sample_size": self.sample_size,
            "sample_share": round(self.sample_share, 4),
            "is_small_sample": self.is_small_sample,
            "mean_score": round(self.mean_score, 2),
            "median_score": round(self.median_score, 2),
            "std_score": round(self.std_score, 2),
            "min_score": self.min_score,
            "max_score": self.max_score,
            "low_risk_pct": round(self.low_risk_pct, 4),
            "moderate_risk_pct": round(self.moderate_risk_pct, 4),
            "high_risk_pct": round(self.high_risk_pct, 4),
            "very_high_risk_pct": round(self.very_high_risk_pct, 4),
            "approval_proxy_rate": round(self.approval_proxy_rate, 4),
            "adverse_impact_ratio": round(self.adverse_impact_ratio, 4),
            "empirical_default_rate": (
                round(self.empirical_default_rate, 4)
                if self.empirical_default_rate is not None
                else None
            ),
            "subgroup_auc": (
                round(self.subgroup_auc, 4)
                if self.subgroup_auc is not None
                else None
            ),
        }


@dataclass(frozen=True)
class FairnessEvaluationReport:
    """Comprehensive subgroup fairness and bias audit report across all tested dimensions."""
    model_name: str
    total_evaluated: int
    overall_approval_proxy_rate: float
    overall_mean_score: float
    overall_default_rate: Optional[float]
    subgroup_metrics: Dict[str, List[SubgroupMetric]]
    air_flags: List[str]
    small_sample_warnings: List[str]
    governance_disclosures: Dict[str, str] = field(default_factory=lambda: GOVERNANCE_DISCLOSURES)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "total_evaluated": self.total_evaluated,
            "overall_approval_proxy_rate": round(self.overall_approval_proxy_rate, 4),
            "overall_mean_score": round(self.overall_mean_score, 2),
            "overall_default_rate": (
                round(self.overall_default_rate, 4)
                if self.overall_default_rate is not None
                else None
            ),
            "subgroup_metrics": {
                dim: [m.to_dict() for m in metrics]
                for dim, metrics in self.subgroup_metrics.items()
            },
            "air_flags": self.air_flags,
            "small_sample_warnings": self.small_sample_warnings,
            "governance_disclosures": self.governance_disclosures,
        }


# -----------------------------------------------------------------------------
# 3. FAIRNESS DIAGNOSTIC ENGINE
# -----------------------------------------------------------------------------

def evaluate_subgroup_fairness(
    data_or_path: Optional[Union[pd.DataFrame, str, Path]] = None,
    model_bundle: Optional[Dict[str, Any]] = None,
    small_sample_threshold: int = 30,
) -> FairnessEvaluationReport:
    """
    Evaluates model score distributions, approval proxy rates, and Adverse Impact
    Ratios (AIR) across age bands, occupation types, and city tiers.
    
    Parameters:
    -----------
    data_or_path : Optional[DataFrame, str, Path]
        Dataset to evaluate. Defaults to data/synthetic_borrowers.csv.
    model_bundle : Optional[Dict[str, Any]]
        Persisted model artifact bundle. Defaults to models/credit_model.pkl.
    small_sample_threshold : int
        Minimum cohort size to consider metrics statistically reliable (default 30).
        
    Returns:
    --------
    FairnessEvaluationReport:
        Structured audit report containing subgroup metrics, AIR flags, and
        responsible AI disclosures.
    """
    # 1. Resolve Data
    if data_or_path is None:
        project_root = Path(__file__).resolve().parent.parent
        csv_path = project_root / "data" / "synthetic_borrowers.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Synthetic evaluation dataset not found at: {csv_path}")
        df = pd.read_csv(csv_path)
    elif isinstance(data_or_path, (str, Path)):
        df = pd.read_csv(data_or_path)
    else:
        df = data_or_path.copy()

    # 2. Resolve Model
    bundle = model_bundle if model_bundle is not None else load_model_bundle()
    pipeline = bundle["pipeline"]
    model = bundle["model"]

    # Transform features through FeaturePipeline
    X_input = df.drop(columns=["borrower_id", "defaulted"], errors="ignore")
    X_transformed = pipeline.transform(X_input)
    
    # Generate probabilities and scores
    probs_raw = model.predict_proba(X_transformed)[:, 1]
    
    # Apply Bayesian calibration to reflect empirical population default rate
    odds_raw = probs_raw / np.clip(1.0 - probs_raw, 1e-6, 1.0)
    odds_calibrated = odds_raw * (POPULATION_DEFAULT_RATE / (1.0 - POPULATION_DEFAULT_RATE))
    probs_calibrated = odds_calibrated / (1.0 + odds_calibrated)

    scores = np.array([probability_to_credit_score(p) for p in probs_calibrated])
    tiers = np.array([score_to_tier(s) for s in scores])

    eval_df = df.copy()
    eval_df["predicted_prob"] = probs_calibrated
    eval_df["credit_score"] = scores
    eval_df["risk_tier"] = tiers
    # Approval proxy: Low Risk (750+) or Moderate Risk (650-749)
    eval_df["is_approved_proxy"] = eval_df["risk_tier"].isin(["Low Risk", "Moderate Risk"]).astype(int)

    # 3. Create Age Bins if age column present
    if "age" in eval_df.columns:
        binned = pd.cut(
            eval_df["age"],
            bins=[17, 25, 35, 50, 100],
            labels=["18-25", "26-35", "36-50", "51+"],
            right=True
        )
        if isinstance(binned, tuple):
            eval_df["age_group"] = pd.Series(binned[0]).astype(str)
        else:
            eval_df["age_group"] = pd.Series(binned).astype(str)

    total_n = len(eval_df)
    overall_approval = float(eval_df["is_approved_proxy"].mean())
    overall_mean_score = float(eval_df["credit_score"].mean())
    overall_default_rate = float(eval_df["defaulted"].mean()) if "defaulted" in eval_df.columns else None

    # Dimensions to evaluate
    dimensions = []
    for dim in ["age_group", "occupation_type", "city_tier"]:
        if dim in eval_df.columns:
            dimensions.append(dim)

    subgroup_metrics: Dict[str, List[SubgroupMetric]] = {}
    air_flags: List[str] = []
    small_sample_warnings: List[str] = []

    for dim in dimensions:
        dim_metrics: List[SubgroupMetric] = []
        unique_groups = sorted(eval_df[dim].dropna().unique().tolist())

        # Determine benchmark group for Adverse Impact Ratio (highest approval rate with adequate sample)
        group_rates = {}
        for g in unique_groups:
            sub = eval_df[eval_df[dim] == g]
            if len(sub) >= small_sample_threshold:
                group_rates[g] = float(sub["is_approved_proxy"].mean())
        
        if not group_rates:
            # Fallback if all groups are small
            for g in unique_groups:
                sub = eval_df[eval_df[dim] == g]
                group_rates[g] = float(sub["is_approved_proxy"].mean())

        benchmark_rate = max(group_rates.values()) if group_rates else 1.0
        benchmark_rate = max(benchmark_rate, 1e-6)  # avoid div by zero

        for g in unique_groups:
            sub = eval_df[eval_df[dim] == g]
            n_sub = len(sub)
            is_small = (n_sub < small_sample_threshold)
            
            if is_small:
                small_sample_warnings.append(
                    f"Subgroup '{dim}:{g}' has small sample size N={n_sub} (< {small_sample_threshold}); "
                    "metrics have high statistical variance."
                )

            sub_scores = np.asarray(sub["credit_score"])
            sub_tiers = sub["risk_tier"]
            sub_approval = float(sub["is_approved_proxy"].mean())
            air = sub_approval / benchmark_rate

            if air < 0.80 and not is_small:
                air_flags.append(
                    f"Adverse Impact Warning: Subgroup '{dim}:{g}' has AIR = {air:.2f} (< 0.80 benchmark threshold)."
                )

            sub_default = float(sub["defaulted"].mean()) if "defaulted" in sub.columns else None
            
            # Subgroup AUC if both classes present
            sub_auc = None
            if "defaulted" in sub.columns and len(np.unique(sub["defaulted"])) == 2 and n_sub >= 10:
                try:
                    sub_auc = float(roc_auc_score(sub["defaulted"], sub["predicted_prob"]))
                except Exception:
                    sub_auc = None

            metric = SubgroupMetric(
                dimension=dim,
                subgroup=str(g),
                sample_size=n_sub,
                sample_share=n_sub / total_n,
                is_small_sample=is_small,
                mean_score=float(np.mean(sub_scores)),
                median_score=float(np.median(sub_scores)),
                std_score=float(np.std(sub_scores)) if n_sub > 1 else 0.0,
                min_score=int(np.min(sub_scores)),
                max_score=int(np.max(sub_scores)),
                low_risk_pct=float((sub_tiers == "Low Risk").mean()),
                moderate_risk_pct=float((sub_tiers == "Moderate Risk").mean()),
                high_risk_pct=float((sub_tiers == "High Risk — Manual Review").mean()),
                very_high_risk_pct=float((sub_tiers == "Very High Risk").mean()),
                approval_proxy_rate=sub_approval,
                adverse_impact_ratio=air,
                empirical_default_rate=sub_default,
                subgroup_auc=sub_auc,
            )
            dim_metrics.append(metric)

        subgroup_metrics[dim] = dim_metrics

    return FairnessEvaluationReport(
        model_name="Champion_LogisticRegression_Pipeline",
        total_evaluated=total_n,
        overall_approval_proxy_rate=overall_approval,
        overall_mean_score=overall_mean_score,
        overall_default_rate=overall_default_rate,
        subgroup_metrics=subgroup_metrics,
        air_flags=air_flags,
        small_sample_warnings=small_sample_warnings,
        governance_disclosures=GOVERNANCE_DISCLOSURES,
    )
