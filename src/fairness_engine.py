"""
CreditBridge - Alternative Credit Scoring Engine
Phase 2: Comprehensive Subgroup Fairness Audit, Governance & Mitigation Engine
Path: src/fairness_engine.py

Provides:
1. Subgroup fairness auditing across age bands, occupation types, and city tiers.
2. Group-level metrics: count, approval rate, default rate, ROC-AUC, PR-AUC, TPR, FPR, FNR,
   precision, Brier score, ECE, AIR (Adverse Impact Ratio), and Wilson score/bootstrap CIs.
3. Formal documentation of fairness definitions (Demographic Parity, Equal Opportunity,
   Equalized Odds, Predictive Parity, Calibration) and the Impossibility Theorem.
4. Two distinct fairness mitigation algorithms:
   - Sample Reweighting (Kamiran & Calders)
   - Subgroup Threshold Optimization (Equal Opportunity / Demographic Parity)
5. Before/after mitigation tradeoff comparison (Discrimination, Calibration, Expected Loss, AIR).
6. Deterministic FAIRNESS_REVIEW_REQUIRED state assignment.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc as calc_auc,
)
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

# -----------------------------------------------------------------------------
# 1. FAIRNESS DEFINITIONS & IMPOSSIBILITY THEOREM DOCUMENTATION
# -----------------------------------------------------------------------------

FAIRNESS_CRITERIA_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "demographic_parity": {
        "name": "Demographic Parity (Statistical Parity)",
        "formula": "P(Ŷ = 1 | A = a) = P(Ŷ = 1 | A = b)",
        "interpretation": (
            "Requires that the decision rate (e.g. loan approval) is identical across "
            "all demographic subgroups regardless of underlying true default base rates."
        ),
        "tradeoff": (
            "If true creditworthiness or historical default rates differ across groups, "
            "demographic parity forces higher default rates or risk concentration in lower-base-rate groups."
        ),
    },
    "equal_opportunity": {
        "name": "Equal Opportunity (True Positive Rate Parity)",
        "formula": "P(Ŷ = 1 | Y = 1, A = a) = P(Ŷ = 1 | Y = 1, A = b)",
        "interpretation": (
            "Requires that borrowers who will repay their loans have an equal probability "
            "of being approved, regardless of protected group membership."
        ),
        "tradeoff": (
            "Focuses solely on the qualified population (Y=1 repayment). May allow different false positive rates."
        ),
    },
    "equalized_odds": {
        "name": "Equalized Odds (Separation)",
        "formula": "P(Ŷ = 1 | Y = y, A = a) = P(Ŷ = 1 | Y = y, A = b) for all y in {0, 1}",
        "interpretation": (
            "Requires equal True Positive Rates AND equal False Positive Rates across all subgroups."
        ),
        "tradeoff": (
            "Strongest statistical independence constraint; mathematically incompatible with well-calibrated "
            "probabilities when group base rates differ."
        ),
    },
    "predictive_parity": {
        "name": "Predictive Parity (Sufficiency / Equal Precision)",
        "formula": "P(Y = 1 | Ŷ = 1, A = a) = P(Y = 1 | Ŷ = 1, A = b)",
        "interpretation": (
            "Requires that individuals accepted by the model have the same repayment rate across groups."
        ),
        "tradeoff": (
            "Ensures fairness from the lender's risk perspective, but can lead to different acceptance rates."
        ),
    },
    "calibration_within_groups": {
        "name": "Calibration by Group",
        "formula": "P(Y = 1 | R = r, A = a) = P(Y = 1 | R = r, A = b) = r",
        "interpretation": (
            "A predicted default probability of 15% means a 15% true empirical default rate for every subgroup."
        ),
        "tradeoff": (
            "Crucial for risk pricing and Basel capital adequacy. However, under differing group base rates, "
            "calibration is mutually exclusive with Equalized Odds (Kleinberg et al., 2016; Chouldechova, 2017)."
        ),
    },
}

IMPOSSIBILITY_THEOREM_DISCLOSURE: str = (
    "MATHEMATICAL IMPOSSIBILITY OF SIMULTANEOUS FAIRNESS: "
    "Under Kleinberg et al. (2016) and Chouldechova (2017), when base default rates differ across subgroups, "
    "it is mathematically impossible for an underwriting model to simultaneously satisfy: "
    "(1) Demographic Parity, (2) Equalized Odds / Equal Opportunity, and (3) Calibration by Group. "
    "CreditBridge explicitly documents this tradeoff. Credit underwriting prioritizes calibration "
    "and risk-reflective pricing while auditing for and actively mitigating severe Adverse Impact Ratios."
)

PROTECTED_ATTRIBUTE_METADATA: Dict[str, Dict[str, str]] = {
    "age_group": {
        "classification": "PROTECTED_DEMOGRAPHIC_ATTRIBUTE",
        "rationale": "Age is a legally protected class under fair lending regulations (e.g. ECOA / Reg B).",
        "legal_status": "Prohibited from direct adverse differential pricing; audited for disparate impact.",
    },
    "occupation_type": {
        "classification": "SOCIOECONOMIC_PROXY_ATTRIBUTE",
        "rationale": "Occupation type reflects employment stability and informal gig-economy dynamics.",
        "legal_status": "Permitted if predictive of cash flow durability, but monitored for disparate impact proxying.",
    },
    "city_tier": {
        "classification": "GEOGRAPHIC_PROXY_ATTRIBUTE",
        "rationale": "City tier reflects cost of living, banking penetration, and infrastructure access.",
        "legal_status": "Monitored for geographic discrimination (redlining proxies).",
    },
}


# -----------------------------------------------------------------------------
# 2. STATISTICAL UTILITIES: WILSON SCORE INTERVALS & ECE
# -----------------------------------------------------------------------------

def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Computes exact Wilson score confidence interval for a binomial proportion."""
    if total <= 0:
        return (0.0, 0.0)
    z = 1.95996  # 95% confidence standard normal quantile
    p_hat = successes / total
    denominator = 1.0 + (z**2) / total
    centre = p_hat + (z**2) / (2.0 * total)
    margin = z * math.sqrt((p_hat * (1.0 - p_hat) + (z**2) / (4.0 * total)) / total)
    lower = max(0.0, (centre - margin) / denominator)
    upper = min(1.0, (centre + margin) / denominator)
    return (round(lower, 4), round(upper, 4))


def compute_expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Computes Expected Calibration Error (ECE) with equal-frequency or equal-width bins."""
    if len(y_true) == 0:
        return 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)

    for i in range(n_bins):
        bin_mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:
            bin_mask = bin_mask | (y_prob == bin_edges[i + 1])
        bin_count = np.sum(bin_mask)
        if bin_count > 0:
            bin_acc = np.mean(y_true[bin_mask])
            bin_conf = np.mean(y_prob[bin_mask])
            ece += (bin_count / total_samples) * abs(bin_acc - bin_conf)

    return float(round(ece, 4))


# -----------------------------------------------------------------------------
# 3. SUBGROUP AUDIT DATA STRUCTURES
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SubgroupFairnessMetric:
    """Rigorous group-level evaluation metric record with statistical bounds."""
    dimension: str
    subgroup: str
    count: int
    sample_share: float
    is_small_sample: bool
    approval_rate: float
    approval_ci: Tuple[float, float]
    observed_default_rate: Optional[float]
    default_ci: Optional[Tuple[float, float]]
    roc_auc: Optional[float]
    pr_auc: Optional[float]
    tpr: Optional[float]
    fpr: Optional[float]
    fnr: Optional[float]
    precision: Optional[float]
    brier_score: Optional[float]
    ece: Optional[float]
    air_ratio: float
    warning_messages: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "subgroup": self.subgroup,
            "count": self.count,
            "sample_share": round(self.sample_share, 4),
            "is_small_sample": self.is_small_sample,
            "approval_rate": round(self.approval_rate, 4),
            "approval_ci": self.approval_ci,
            "observed_default_rate": round(self.observed_default_rate, 4) if self.observed_default_rate is not None else None,
            "default_ci": self.default_ci,
            "roc_auc": round(self.roc_auc, 4) if self.roc_auc is not None else None,
            "pr_auc": round(self.pr_auc, 4) if self.pr_auc is not None else None,
            "tpr": round(self.tpr, 4) if self.tpr is not None else None,
            "fpr": round(self.fpr, 4) if self.fpr is not None else None,
            "fnr": round(self.fnr, 4) if self.fnr is not None else None,
            "precision": round(self.precision, 4) if self.precision is not None else None,
            "brier_score": round(self.brier_score, 4) if self.brier_score is not None else None,
            "ece": round(self.ece, 4) if self.ece is not None else None,
            "air_ratio": round(self.air_ratio, 4),
            "warning_messages": self.warning_messages,
        }


@dataclass(frozen=True)
class ComprehensiveFairnessAuditReport:
    """Comprehensive fairness audit report containing group evaluations, tradeoffs, and governance status."""
    model_name: str
    total_evaluated: int
    overall_approval_rate: float
    overall_default_rate: Optional[float]
    overall_roc_auc: Optional[float]
    overall_pr_auc: Optional[float]
    overall_brier_score: Optional[float]
    subgroup_metrics: Dict[str, List[SubgroupFairnessMetric]]
    air_breaches: List[str]
    tpr_disparities: List[str]
    governance_state: str  # FAIRNESS_REVIEW_REQUIRED or FAIRNESS_MONITORING
    governance_rationale: str
    definitions: Dict[str, Dict[str, str]] = field(default_factory=lambda: FAIRNESS_CRITERIA_DEFINITIONS)
    impossibility_theorem_disclosure: str = IMPOSSIBILITY_THEOREM_DISCLOSURE
    protected_attribute_metadata: Dict[str, Dict[str, str]] = field(default_factory=lambda: PROTECTED_ATTRIBUTE_METADATA)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "total_evaluated": self.total_evaluated,
            "overall_approval_rate": round(self.overall_approval_rate, 4),
            "overall_default_rate": round(self.overall_default_rate, 4) if self.overall_default_rate is not None else None,
            "overall_roc_auc": round(self.overall_roc_auc, 4) if self.overall_roc_auc is not None else None,
            "overall_pr_auc": round(self.overall_pr_auc, 4) if self.overall_pr_auc is not None else None,
            "overall_brier_score": round(self.overall_brier_score, 4) if self.overall_brier_score is not None else None,
            "subgroup_metrics": {
                dim: [m.to_dict() for m in metrics]
                for dim, metrics in self.subgroup_metrics.items()
            },
            "air_breaches": self.air_breaches,
            "tpr_disparities": self.tpr_disparities,
            "governance_state": self.governance_state,
            "governance_rationale": self.governance_rationale,
            "definitions": self.definitions,
            "impossibility_theorem_disclosure": self.impossibility_theorem_disclosure,
            "protected_attribute_metadata": self.protected_attribute_metadata,
        }


# -----------------------------------------------------------------------------
# 4. AUDIT EVALUATOR
# -----------------------------------------------------------------------------

def run_comprehensive_fairness_audit(
    df: pd.DataFrame,
    y_true_col: Optional[str] = "defaulted",
    prob_col: str = "predicted_prob",
    approval_col: Optional[str] = None,
    approval_threshold_prob: float = 0.35,  # By default approve if PD <= 0.35
    air_threshold: float = 0.80,           # EEOC 4/5ths rule
    tpr_parity_threshold: float = 0.15,
    min_sample_size: int = 30,
    model_name: str = "CreditBridge Baseline LR",
) -> ComprehensiveFairnessAuditReport:
    """
    Executes deep group-level fairness audit across age_group, occupation_type, and city_tier.
    """
    eval_df = df.copy()

    # Create age groups if age present
    if "age" in eval_df.columns and "age_group" not in eval_df.columns:
        binned = pd.cut(
            eval_df["age"],
            bins=[17, 25, 35, 50, 100],
            labels=["18-25", "26-35", "36-50", "51+"],
            right=True
        )
        eval_df["age_group"] = pd.Series(binned).astype(str)

    # Establish approval indicator
    if approval_col and approval_col in eval_df.columns:
        eval_df["_is_approved"] = eval_df[approval_col].astype(int)
    else:
        # Default policy: approve if PD <= approval_threshold_prob
        eval_df["_is_approved"] = (eval_df[prob_col] <= approval_threshold_prob).astype(int)

    total_evaluated = len(eval_df)
    overall_approval_rate = float(eval_df["_is_approved"].mean())

    has_truth = y_true_col is not None and y_true_col in eval_df.columns
    y_true_all = eval_df[y_true_col].to_numpy().astype(int) if has_truth else None
    y_prob_all = eval_df[prob_col].to_numpy().astype(float)

    overall_default_rate = float(np.mean(y_true_all)) if y_true_all is not None else None
    overall_roc_auc = float(roc_auc_score(y_true_all, y_prob_all)) if (y_true_all is not None and len(np.unique(y_true_all)) > 1) else None

    overall_pr_auc: Optional[float] = None
    if y_true_all is not None and len(np.unique(y_true_all)) > 1:
        p_prec, p_rec, _ = precision_recall_curve(y_true_all, y_prob_all)
        overall_pr_auc = float(calc_auc(p_rec, p_prec))

    overall_brier = float(brier_score_loss(y_true_all, y_prob_all)) if y_true_all is not None else None

    # Audit across protected/proxy dimensions
    dimensions = [d for d in ["age_group", "occupation_type", "city_tier"] if d in eval_df.columns]
    subgroup_metrics_dict: Dict[str, List[SubgroupFairnessMetric]] = {}
    air_breaches: List[str] = []
    tpr_disparities: List[str] = []

    for dim in dimensions:
        dim_metrics: List[SubgroupFairnessMetric] = []
        groups = sorted(eval_df[dim].dropna().unique().tolist())

        # Compute approval rates across groups in dimension to find reference rate for AIR
        group_approval_rates: Dict[str, float] = {}
        for g in groups:
            gdf = eval_df[eval_df[dim] == g]
            group_approval_rates[g] = float(gdf["_is_approved"].mean()) if len(gdf) > 0 else 0.0

        max_app_rate = max(group_approval_rates.values()) if group_approval_rates else 1.0
        ref_app_rate = max(max_app_rate, 1e-4)

        dim_tprs: Dict[str, float] = {}

        for g in groups:
            gdf = eval_df[eval_df[dim] == g]
            n_g = len(gdf)
            sample_share = n_g / total_evaluated
            is_small = n_g < min_sample_size

            app_rate = group_approval_rates[g]
            app_count = int(gdf["_is_approved"].sum())
            app_ci = wilson_score_interval(app_count, n_g)
            air = app_rate / ref_app_rate

            warnings: List[str] = []
            if is_small:
                warnings.append(f"Small sample size (N={n_g} < {min_sample_size}); statistical variance is high.")

            if air < air_threshold:
                breach_msg = f"{dim} '{g}' has AIR={air:.2f} (< {air_threshold:.2f} reference rate)."
                warnings.append(breach_msg)
                air_breaches.append(breach_msg)

            obs_default: Optional[float] = None
            default_ci: Optional[Tuple[float, float]] = None
            roc_auc_g: Optional[float] = None
            pr_auc_g: Optional[float] = None
            tpr_g: Optional[float] = None
            fpr_g: Optional[float] = None
            fnr_g: Optional[float] = None
            precision_g: Optional[float] = None
            brier_g: Optional[float] = None
            ece_g: Optional[float] = None

            if has_truth and y_true_col is not None:
                y_g = np.asarray(gdf[y_true_col], dtype=int)
                p_g = np.asarray(gdf[prob_col], dtype=float)
                def_count = int(np.sum(y_g))
                obs_default = float(np.mean(y_g))
                default_ci = wilson_score_interval(def_count, n_g)
                brier_g = float(brier_score_loss(y_g, p_g))
                ece_g = compute_expected_calibration_error(y_g, p_g)

                if len(np.unique(y_g)) > 1 and n_g >= 10:
                    roc_auc_g = float(roc_auc_score(y_g, p_g))
                    prec_arr, rec_arr, _ = precision_recall_curve(y_g, p_g)
                    pr_auc_g = float(calc_auc(rec_arr, prec_arr))

                # Binary classification metrics on predicted default (predict default if PD > threshold)
                y_pred_default = (p_g > approval_threshold_prob).astype(int)
                cm = confusion_matrix(y_g, y_pred_default, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()

                tpr_val = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                fpr_val = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
                fnr_val = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0
                prec_val = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

                tpr_g, fpr_g, fnr_g, precision_g = tpr_val, fpr_val, fnr_val, prec_val
                dim_tprs[g] = tpr_val

            dim_metrics.append(
                SubgroupFairnessMetric(
                    dimension=dim,
                    subgroup=g,
                    count=n_g,
                    sample_share=sample_share,
                    is_small_sample=is_small,
                    approval_rate=app_rate,
                    approval_ci=app_ci,
                    observed_default_rate=obs_default,
                    default_ci=default_ci,
                    roc_auc=roc_auc_g,
                    pr_auc=pr_auc_g,
                    tpr=tpr_g,
                    fpr=fpr_g,
                    fnr=fnr_g,
                    precision=precision_g,
                    brier_score=brier_g,
                    ece=ece_g,
                    air_ratio=air,
                    warning_messages=warnings,
                )
            )

        # Evaluate TPR disparity within dimension
        if len(dim_tprs) >= 2:
            max_tpr = max(dim_tprs.values())
            min_tpr = min(dim_tprs.values())
            diff = max_tpr - min_tpr
            if diff > tpr_parity_threshold:
                tpr_msg = f"{dim} has TPR parity spread of {diff:.2f} (max={max_tpr:.2f}, min={min_tpr:.2f} > {tpr_parity_threshold:.2f})."
                tpr_disparities.append(tpr_msg)

        subgroup_metrics_dict[dim] = dim_metrics

    # Governance State Logic: Never automatically label the model 'FAIR'
    if len(air_breaches) > 0 or len(tpr_disparities) > 0:
        governance_state = "FAIRNESS_REVIEW_REQUIRED"
        governance_rationale = (
            f"FAIRNESS_REVIEW_REQUIRED: Model triggered {len(air_breaches)} Adverse Impact Ratio breaches (<0.80) "
            f"and {len(tpr_disparities)} TPR parity disparities. Human model-risk committee review is mandatory before deployment."
        )
    else:
        governance_state = "FAIRNESS_MONITORING"
        governance_rationale = (
            "FAIRNESS_MONITORING: No hard AIR or TPR disparity thresholds breached on this evaluation dataset. "
            "Continuous monitoring is active. Note: Absence of statistical disparity does NOT constitute a legal certification of fairness."
        )

    return ComprehensiveFairnessAuditReport(
        model_name=model_name,
        total_evaluated=total_evaluated,
        overall_approval_rate=overall_approval_rate,
        overall_default_rate=overall_default_rate,
        overall_roc_auc=overall_roc_auc,
        overall_pr_auc=overall_pr_auc,
        overall_brier_score=overall_brier,
        subgroup_metrics=subgroup_metrics_dict,
        air_breaches=air_breaches,
        tpr_disparities=tpr_disparities,
        governance_state=governance_state,
        governance_rationale=governance_rationale,
    )


# -----------------------------------------------------------------------------
# 5. MITIGATION EXPERIMENT 1: SAMPLE REWEIGHTING (KAMIRAN & CALDERS)
# -----------------------------------------------------------------------------

def compute_kamiran_calders_weights(
    df: pd.DataFrame,
    protected_col: str,
    target_col: str = "defaulted"
) -> np.ndarray:
    """
    Computes statistical reweighting multipliers (Kamiran & Calders, 2012)
    to balance joint distribution of sensitive attribute S and outcome Y.

    W(S=s, Y=y) = [ P(S=s) * P(Y=y) ] / P(S=s, Y=y)
    """
    total = len(df)
    weights = np.ones(total, dtype=float)

    p_y = df[target_col].value_counts(normalize=True).to_dict()
    p_s = df[protected_col].value_counts(normalize=True).to_dict()

    joint_counts = df.groupby([protected_col, target_col]).size().to_dict()

    for idx, (_, row) in enumerate(df.iterrows()):
        s_val = row[protected_col]
        y_val = row[target_col]
        n_sy = joint_counts.get((s_val, y_val), 0)
        p_sy = n_sy / total if n_sy > 0 else 1e-5

        expected_p = p_s.get(s_val, 1e-5) * p_y.get(y_val, 1e-5)
        weight = expected_p / p_sy
        weights[idx] = max(min(weight, 10.0), 0.1)  # Clip weights to prevent variance explosion

    # Normalize weights so sum equals N
    weights = weights * (total / np.sum(weights))
    return weights


# -----------------------------------------------------------------------------
# 6. MITIGATION EXPERIMENT 2: SUBGROUP THRESHOLD ADJUSTMENT
# -----------------------------------------------------------------------------

def optimize_subgroup_thresholds(
    df: pd.DataFrame,
    protected_col: str,
    prob_col: str = "predicted_prob",
    target_col: str = "defaulted",
    objective: str = "equal_opportunity",  # 'equal_opportunity' or 'demographic_parity'
    base_threshold: float = 0.35,
) -> Dict[str, float]:
    """
    Solves for group-specific approval thresholds to equalize True Positive Rate
    or Demographic Parity across groups without degrading probability calibration.
    """
    groups = df[protected_col].dropna().unique().tolist()
    thresholds: Dict[str, float] = {}

    if objective == "demographic_parity":
        # Target: match the overall approval rate
        target_approval_rate = float((df[prob_col] <= base_threshold).mean())
        for g in groups:
            gdf = df[df[protected_col] == g]
            probs = np.asarray(gdf[prob_col], dtype=float)
            # Threshold is the percentile corresponding to target_approval_rate
            thresh = float(np.percentile(probs, target_approval_rate * 100))
            thresholds[str(g)] = round(max(min(thresh, 0.80), 0.10), 4)

    elif objective == "equal_opportunity":
        # Target: match the overall True Positive Rate (identifying repayers Y=0)
        repayers = df[df[target_col] == 0]
        target_tpr = float((repayers[prob_col] <= base_threshold).mean())
        for g in groups:
            g_repayers = df[(df[protected_col] == g) & (df[target_col] == 0)]
            if len(g_repayers) > 0:
                probs = np.asarray(g_repayers[prob_col], dtype=float)
                thresh = float(np.percentile(probs, target_tpr * 100))
                thresholds[str(g)] = round(max(min(thresh, 0.80), 0.10), 4)
            else:
                thresholds[str(g)] = base_threshold
    else:
        for g in groups:
            thresholds[str(g)] = base_threshold

    return thresholds


# -----------------------------------------------------------------------------
# 7. BEFORE / AFTER MITIGATION COMPARISON REPORT
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelMitigationProfile:
    """Performance, calibration, fairness, and economic summary for an experiment."""
    profile_name: str
    mitigation_strategy: str
    roc_auc: float
    pr_auc: float
    brier_score: float
    ece: float
    overall_approval_rate: float
    expected_loss_inr: float
    min_air_ratio: float
    max_tpr_disparity: float
    governance_state: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "mitigation_strategy": self.mitigation_strategy,
            "roc_auc": round(self.roc_auc, 4),
            "pr_auc": round(self.pr_auc, 4),
            "brier_score": round(self.brier_score, 4),
            "ece": round(self.ece, 4),
            "overall_approval_rate": round(self.overall_approval_rate, 4),
            "expected_loss_inr": round(self.expected_loss_inr, 2),
            "min_air_ratio": round(self.min_air_ratio, 4),
            "max_tpr_disparity": round(self.max_tpr_disparity, 4),
            "governance_state": self.governance_state,
        }


def compare_mitigation_strategies(
    test_df: pd.DataFrame,
    baseline_probs: np.ndarray,
    reweighted_probs: np.ndarray,
    protected_col: str = "occupation_type",
    target_col: str = "defaulted",
    base_threshold: float = 0.35,
) -> Dict[str, Any]:
    """
    Executes and compares Baseline vs Reweighted vs Threshold Adjusted models.
    Reports discrimination, calibration, approval rate, expected loss, and fairness tradeoffs.
    Never deletes difficult groups or changes denominators silently.
    """
    eval_df = test_df.copy()
    y_true = eval_df[target_col].to_numpy().astype(int)

    # 1. Baseline Profile
    eval_df["base_prob"] = baseline_probs
    base_audit = run_comprehensive_fairness_audit(
        eval_df,
        y_true_col=target_col,
        prob_col="base_prob",
        approval_threshold_prob=base_threshold,
        model_name="Baseline (Unmitigated)",
    )
    base_dim_metrics = base_audit.subgroup_metrics.get(protected_col, [])
    base_min_air = min([m.air_ratio for m in base_dim_metrics]) if base_dim_metrics else 1.0
    base_tprs = [m.tpr for m in base_dim_metrics if m.tpr is not None]
    base_tpr_diff = (max(base_tprs) - min(base_tprs)) if len(base_tprs) >= 2 else 0.0

    # Economic impact under baseline
    base_approved = eval_df["base_prob"] <= base_threshold
    base_el = float(np.sum(eval_df.loc[base_approved, "base_prob"] * 0.65 * 50000.0))  # LGD=65%, EAD=50k

    p_prec, p_rec, _ = precision_recall_curve(y_true, baseline_probs)
    base_profile = ModelMitigationProfile(
        profile_name="Baseline (Unmitigated)",
        mitigation_strategy="None",
        roc_auc=float(roc_auc_score(y_true, baseline_probs)),
        pr_auc=float(calc_auc(p_rec, p_prec)),
        brier_score=float(brier_score_loss(y_true, baseline_probs)),
        ece=compute_expected_calibration_error(y_true, baseline_probs),
        overall_approval_rate=float(np.mean(base_approved)),
        expected_loss_inr=base_el,
        min_air_ratio=base_min_air,
        max_tpr_disparity=base_tpr_diff,
        governance_state=base_audit.governance_state,
    )

    # 2. Reweighted Model Profile
    eval_df["reweighted_prob"] = reweighted_probs
    reweight_audit = run_comprehensive_fairness_audit(
        eval_df,
        y_true_col=target_col,
        prob_col="reweighted_prob",
        approval_threshold_prob=base_threshold,
        model_name="Mitigation 1: Sample Reweighting",
    )
    rew_dim_metrics = reweight_audit.subgroup_metrics.get(protected_col, [])
    rew_min_air = min([m.air_ratio for m in rew_dim_metrics]) if rew_dim_metrics else 1.0
    rew_tprs = [m.tpr for m in rew_dim_metrics if m.tpr is not None]
    rew_tpr_diff = (max(rew_tprs) - min(rew_tprs)) if len(rew_tprs) >= 2 else 0.0

    rew_approved = eval_df["reweighted_prob"] <= base_threshold
    rew_el = float(np.sum(eval_df.loc[rew_approved, "reweighted_prob"] * 0.65 * 50000.0))
    p_prec_rew, p_rec_rew, _ = precision_recall_curve(y_true, reweighted_probs)

    reweight_profile = ModelMitigationProfile(
        profile_name="Mitigation 1: Sample Reweighting",
        mitigation_strategy="Kamiran-Calders In-Processing Reweighting",
        roc_auc=float(roc_auc_score(y_true, reweighted_probs)),
        pr_auc=float(calc_auc(p_rec_rew, p_prec_rew)),
        brier_score=float(brier_score_loss(y_true, reweighted_probs)),
        ece=compute_expected_calibration_error(y_true, reweighted_probs),
        overall_approval_rate=float(np.mean(rew_approved)),
        expected_loss_inr=rew_el,
        min_air_ratio=rew_min_air,
        max_tpr_disparity=rew_tpr_diff,
        governance_state=reweight_audit.governance_state,
    )

    # 3. Subgroup Threshold Adjusted Profile
    optimized_thresholds = optimize_subgroup_thresholds(
        eval_df,
        protected_col=protected_col,
        prob_col="base_prob",
        target_col=target_col,
        objective="equal_opportunity",
        base_threshold=base_threshold,
    )

    thresh_approved = np.zeros(len(eval_df), dtype=int)
    for idx, (_, row) in enumerate(eval_df.iterrows()):
        grp = str(row[protected_col])
        t = optimized_thresholds.get(grp, base_threshold)
        if row["base_prob"] <= t:
            thresh_approved[idx] = 1

    eval_df["thresh_approved"] = thresh_approved
    thresh_audit = run_comprehensive_fairness_audit(
        eval_df,
        y_true_col=target_col,
        prob_col="base_prob",
        approval_col="thresh_approved",
        model_name="Mitigation 2: Threshold Optimization",
    )
    thresh_dim_metrics = thresh_audit.subgroup_metrics.get(protected_col, [])
    thresh_min_air = min([m.air_ratio for m in thresh_dim_metrics]) if thresh_dim_metrics else 1.0
    thresh_tprs = [m.tpr for m in thresh_dim_metrics if m.tpr is not None]
    thresh_tpr_diff = (max(thresh_tprs) - min(thresh_tprs)) if len(thresh_tprs) >= 2 else 0.0
    thresh_el = float(np.sum(eval_df.loc[eval_df["thresh_approved"] == 1, "base_prob"] * 0.65 * 50000.0))

    thresh_profile = ModelMitigationProfile(
        profile_name="Mitigation 2: Subgroup Threshold Adjustment",
        mitigation_strategy="Post-Processing Equal Opportunity Optimization",
        roc_auc=float(roc_auc_score(y_true, baseline_probs)),  # Model discrimination unchanged
        pr_auc=float(calc_auc(p_rec, p_prec)),
        brier_score=float(brier_score_loss(y_true, baseline_probs)),  # Calibration unchanged
        ece=compute_expected_calibration_error(y_true, baseline_probs),
        overall_approval_rate=float(np.mean(thresh_approved)),
        expected_loss_inr=thresh_el,
        min_air_ratio=thresh_min_air,
        max_tpr_disparity=thresh_tpr_diff,
        governance_state=thresh_audit.governance_state,
    )

    return {
        "protected_attribute": protected_col,
        "base_threshold": base_threshold,
        "optimized_thresholds": optimized_thresholds,
        "profiles": {
            "baseline": base_profile.to_dict(),
            "reweighted": reweight_profile.to_dict(),
            "threshold_adjusted": thresh_profile.to_dict(),
        },
        "tradeoff_analysis": {
            "auc_delta_reweight": round(reweight_profile.roc_auc - base_profile.roc_auc, 4),
            "air_improvement_reweight": round(reweight_profile.min_air_ratio - base_profile.min_air_ratio, 4),
            "air_improvement_threshold": round(thresh_profile.min_air_ratio - base_profile.min_air_ratio, 4),
            "tpr_gap_reduction_threshold": round(base_profile.max_tpr_disparity - thresh_profile.max_tpr_disparity, 4),
            "expected_loss_delta_threshold_inr": round(thresh_profile.expected_loss_inr - base_profile.expected_loss_inr, 2),
        },
    }
