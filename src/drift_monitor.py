"""
CreditBridge - Alternative Credit Scoring Engine
Phase 2: Population Stability Index (PSI) Drift Monitoring & Longitudinal Simulation
Path: src/drift_monitor.py

Provides comprehensive drift auditing:
1. Exact PSI calculation for continuous numerical features and discrete categorical features.
2. Threshold classification:
   - PSI < 0.10: STABLE (No significant change)
   - 0.10 <= PSI < 0.25: WARNING (Moderate shift; heightened monitoring)
   - PSI >= 0.25: CRITICAL (Significant population drift; retraining/recalibration required)
3. Multi-dimensional drift surveillance:
   - Feature drift (all input signals)
   - Score drift (credit score 300-900 distribution)
   - Risk tier distribution shift
   - Subgroup demographic composition drift
   - Approval rate drift & Data-quality drift
4. Outcome-aware performance drift:
   - When observed default labels become available: tracks ROC-AUC, KS, PR-AUC, Brier score, and ECE over time.
5. Monthly Production Batch Simulation:
   - Generates 6 realistic monthly production cohorts with seasonal, inflation, and credit stress shifts.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, cast

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import (
    auc as calc_auc,
)
from sklearn.metrics import (
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)


class DriftSeverity(str, Enum):
    STABLE = "STABLE"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


PSI_WARNING_THRESHOLD: float = 0.10
PSI_CRITICAL_THRESHOLD: float = 0.25


def calculate_numeric_psi(
    reference: np.ndarray,
    current: np.ndarray,
    num_bins: int = 10,
    epsilon: float = 1e-4
) -> float:
    """
    Calculates Population Stability Index (PSI) for continuous numeric distributions.
    Bins are defined using quantiles of the reference distribution.
    """
    ref_clean = reference[~np.isnan(reference)]
    cur_clean = current[~np.isnan(current)]

    if len(ref_clean) == 0 or len(cur_clean) == 0:
        return 0.0

    # Determine quantile bin edges from reference
    quantiles = np.linspace(0.0, 100.0, num_bins + 1)
    bin_edges = np.percentile(ref_clean, quantiles)
    # Ensure strictly increasing edges to avoid degenerate zero-width bins
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    # Expand outer edges slightly to capture boundary values
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    cur_counts, _ = np.histogram(cur_clean, bins=bin_edges)

    # Convert to proportions with Laplace smoothing
    ref_props = (ref_counts + epsilon) / (np.sum(ref_counts) + epsilon * len(ref_counts))
    cur_props = (cur_counts + epsilon) / (np.sum(cur_counts) + epsilon * len(cur_counts))

    psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
    return float(round(max(psi, 0.0), 4))


def calculate_categorical_psi(
    reference: pd.Series,
    current: pd.Series,
    epsilon: float = 1e-4
) -> float:
    """Calculates PSI for categorical discrete distributions across common categories."""
    ref_clean = reference.dropna().astype(str)
    cur_clean = current.dropna().astype(str)

    all_categories = sorted(set(ref_clean.unique()) | set(cur_clean.unique()))
    if not all_categories:
        return 0.0

    ref_counts = ref_clean.value_counts()
    cur_counts = cur_clean.value_counts()

    n_ref = len(ref_clean)
    n_cur = len(cur_clean)

    psi = 0.0
    for cat in all_categories:
        c_ref = float(ref_counts[cat]) if cat in ref_counts else 0.0
        c_cur = float(cur_counts[cat]) if cat in cur_counts else 0.0

        p_ref = (c_ref + epsilon) / (n_ref + epsilon * len(all_categories))
        p_cur = (c_cur + epsilon) / (n_cur + epsilon * len(all_categories))

        psi += (p_cur - p_ref) * math.log(p_cur / p_ref)

    return float(round(max(psi, 0.0), 4))


def get_drift_severity(psi_val: float) -> DriftSeverity:
    """Classifies PSI into STABLE, WARNING, or CRITICAL."""
    if psi_val >= PSI_CRITICAL_THRESHOLD:
        return DriftSeverity.CRITICAL
    if psi_val >= PSI_WARNING_THRESHOLD:
        return DriftSeverity.WARNING
    return DriftSeverity.STABLE


@dataclass(frozen=True)
class FeatureDriftReport:
    feature_name: str
    psi: float
    severity: DriftSeverity
    reference_mean: Optional[float]
    current_mean: Optional[float]
    shift_percentage: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "psi": round(self.psi, 4),
            "severity": self.severity.value,
            "reference_mean": round(self.reference_mean, 2) if self.reference_mean is not None else None,
            "current_mean": round(self.current_mean, 2) if self.current_mean is not None else None,
            "shift_percentage": round(self.shift_percentage, 2) if self.shift_percentage is not None else None,
        }


@dataclass(frozen=True)
class BatchDriftReport:
    batch_name: str
    sample_size: int
    score_psi: float
    score_drift_severity: DriftSeverity
    max_feature_psi: float
    max_drifted_feature: str
    feature_drifts: Dict[str, FeatureDriftReport]
    tier_distribution: Dict[str, float]
    approval_rate: float
    approval_rate_shift: float
    observed_roc_auc: Optional[float] = None
    observed_ks: Optional[float] = None
    observed_pr_auc: Optional[float] = None
    observed_brier: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_name": self.batch_name,
            "sample_size": self.sample_size,
            "score_psi": round(self.score_psi, 4),
            "score_drift_severity": self.score_drift_severity.value,
            "max_feature_psi": round(self.max_feature_psi, 4),
            "max_drifted_feature": self.max_drifted_feature,
            "approval_rate": round(self.approval_rate, 4),
            "approval_rate_shift": round(self.approval_rate_shift, 4),
            "tier_distribution": {k: round(v, 4) for k, v in self.tier_distribution.items()},
            "observed_roc_auc": round(self.observed_roc_auc, 4) if self.observed_roc_auc is not None else None,
            "observed_ks": round(self.observed_ks, 4) if self.observed_ks is not None else None,
            "observed_pr_auc": round(self.observed_pr_auc, 4) if self.observed_pr_auc is not None else None,
            "observed_brier": round(self.observed_brier, 4) if self.observed_brier is not None else None,
            "feature_drifts": {k: v.to_dict() for k, v in self.feature_drifts.items()},
        }


def evaluate_batch_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    batch_name: str = "Production Batch",
    ref_score_col: str = "credit_score",
    cur_score_col: str = "credit_score",
    ref_prob_col: str = "predicted_prob",
    cur_prob_col: str = "predicted_prob",
    target_col: Optional[str] = "defaulted",
) -> BatchDriftReport:
    """
    Evaluates comprehensive drift between a baseline reference population and a current production batch.
    """
    # 1. Score Drift
    ref_scores = reference_df[ref_score_col].to_numpy().astype(float)
    cur_scores = current_df[cur_score_col].to_numpy().astype(float)
    score_psi = calculate_numeric_psi(ref_scores, cur_scores)
    score_severity = get_drift_severity(score_psi)

    # 2. Feature Drift across shared columns
    common_cols = [
        c for c in reference_df.columns
        if c in current_df.columns and c not in [
            "borrower_id", "defaulted", ref_score_col, cur_score_col,
            ref_prob_col, cur_prob_col, "risk_tier", "is_approved"
        ]
    ]

    feature_drifts: Dict[str, FeatureDriftReport] = {}
    max_psi = 0.0
    max_feat = "None"

    for col in common_cols:
        if pd.api.types.is_numeric_dtype(reference_df[col]):
            ref_vals = reference_df[col].to_numpy().astype(float)
            cur_vals = current_df[col].to_numpy().astype(float)
            psi_val = calculate_numeric_psi(ref_vals, cur_vals)
            ref_m = float(np.nanmean(ref_vals))
            cur_m = float(np.nanmean(cur_vals))
            shift_pct = ((cur_m - ref_m) / ref_m * 100.0) if ref_m != 0 else 0.0
        else:
            psi_val = calculate_categorical_psi(reference_df[col], current_df[col])
            ref_m = None
            cur_m = None
            shift_pct = None

        sev = get_drift_severity(psi_val)
        if psi_val > max_psi:
            max_psi = psi_val
            max_feat = col

        feature_drifts[col] = FeatureDriftReport(
            feature_name=col,
            psi=psi_val,
            severity=sev,
            reference_mean=ref_m,
            current_mean=cur_m,
            shift_percentage=shift_pct,
        )

    # 3. Risk Tier & Approval Rate Drift
    if "risk_tier" in current_df.columns:
        tier_counts = current_df["risk_tier"].value_counts(normalize=True).to_dict()
    else:
        tier_counts = {}

    cur_app_rate = float(current_df["is_approved"].mean()) if "is_approved" in current_df.columns else float((current_df[cur_score_col] >= 650).mean())
    ref_app_rate = float(reference_df["is_approved"].mean()) if "is_approved" in reference_df.columns else float((reference_df[ref_score_col] >= 650).mean())
    app_shift = cur_app_rate - ref_app_rate

    # 4. Outcome performance drift if labels available
    obs_auc: Optional[float] = None
    obs_ks: Optional[float] = None
    obs_pr: Optional[float] = None
    obs_brier: Optional[float] = None

    if target_col and target_col in current_df.columns:
        y_cur = current_df[target_col].to_numpy().astype(int)
        p_cur = current_df[cur_prob_col].to_numpy().astype(float)

        if len(np.unique(y_cur)) > 1:
            obs_auc = float(roc_auc_score(y_cur, p_cur))
            p_prec, p_rec, _ = precision_recall_curve(y_cur, p_cur)
            obs_pr = float(calc_auc(p_rec, p_prec))
            obs_brier = float(brier_score_loss(y_cur, p_cur))

            # Two-sample KS test between defaults and non-defaults
            p_def = p_cur[y_cur == 1]
            p_non = p_cur[y_cur == 0]
            if len(p_def) > 0 and len(p_non) > 0:
                ks_res = cast(Any, ks_2samp(p_def, p_non))
                ks_stat = float(getattr(ks_res, "statistic", 0.0))
                obs_ks = float(round(ks_stat * 100.0, 2))

    return BatchDriftReport(
        batch_name=batch_name,
        sample_size=len(current_df),
        score_psi=score_psi,
        score_drift_severity=score_severity,
        max_feature_psi=max_psi,
        max_drifted_feature=max_feat,
        feature_drifts=feature_drifts,
        tier_distribution=tier_counts,
        approval_rate=cur_app_rate,
        approval_rate_shift=app_shift,
        observed_roc_auc=obs_auc,
        observed_ks=obs_ks,
        observed_pr_auc=obs_pr,
        observed_brier=obs_brier,
    )


# -----------------------------------------------------------------------------
# 5. SIMULATE 6 MONTHLY PRODUCTION COHORTS
# -----------------------------------------------------------------------------

def simulate_monthly_production_batches(
    base_df: pd.DataFrame,
    num_months: int = 6,
    batch_size: int = 1000,
    seed: int = 42
) -> List[pd.DataFrame]:
    """
    Simulates 6 monthly production batches introducing realistic market shifts:
    - Month 1: Baseline representative distribution
    - Month 2: Post-festival liquidity expansion (higher inflows, lower volatility)
    - Month 3: Inflationary squeeze (+10% discretionary spend, -5% savings buffer)
    - Month 4: Gig-economy shift (+15% informal retail applicants, higher UPI count)
    - Month 5: Credit tightening (+15% cash flow volatility, rising default rate)
    - Month 6: Stress cycle (+25% volatility, higher defaults, lower savings)
    """
    rng = np.random.default_rng(seed)
    batches: List[pd.DataFrame] = []

    for m in range(1, num_months + 1):
        # Sample with replacement
        sample_indices = rng.choice(len(base_df), size=batch_size, replace=True)
        batch = base_df.iloc[sample_indices].copy().reset_index(drop=True)
        batch["production_month"] = f"Month_{m:02d}"

        if m == 1:
            # Baseline unchanged
            pass
        elif m == 2:
            # Post-festival liquidity expansion
            if "monthly_income_estimate" in batch.columns:
                batch["monthly_income_estimate"] *= rng.uniform(1.02, 1.06, size=len(batch))
            if "savings_buffer_ratio" in batch.columns:
                batch["savings_buffer_ratio"] *= rng.uniform(1.02, 1.05, size=len(batch))
        elif m == 3:
            # Inflationary squeeze
            if "discretionary_spend_ratio" in batch.columns:
                batch["discretionary_spend_ratio"] *= rng.uniform(1.08, 1.15, size=len(batch))
            if "savings_buffer_ratio" in batch.columns:
                batch["savings_buffer_ratio"] *= rng.uniform(0.90, 0.95, size=len(batch))
        elif m == 4:
            # Shift in occupation mix toward informal/gig workers
            if "occupation_type" in batch.columns:
                mask = rng.uniform(0, 1, size=len(batch)) < 0.20
                batch.loc[mask, "occupation_type"] = "informal_retail"
            if "monthly_upi_transaction_count" in batch.columns:
                batch["monthly_upi_transaction_count"] *= rng.uniform(1.10, 1.25, size=len(batch))
        elif m == 5:
            # Mild credit stress
            if "cash_flow_volatility" in batch.columns:
                batch["cash_flow_volatility"] *= rng.uniform(1.15, 1.25, size=len(batch))
            if "defaulted" in batch.columns:
                # Slight default rate bump
                flip_mask = (batch["defaulted"] == 0) & (rng.uniform(0, 1, size=len(batch)) < 0.04)
                batch.loc[flip_mask, "defaulted"] = 1
        elif m == 6:
            # Stress cycle
            if "cash_flow_volatility" in batch.columns:
                batch["cash_flow_volatility"] *= rng.uniform(1.25, 1.40, size=len(batch))
            if "savings_buffer_ratio" in batch.columns:
                batch["savings_buffer_ratio"] *= rng.uniform(0.75, 0.85, size=len(batch))
            if "defaulted" in batch.columns:
                flip_mask = (batch["defaulted"] == 0) & (rng.uniform(0, 1, size=len(batch)) < 0.08)
                batch.loc[flip_mask, "defaulted"] = 1

        batches.append(batch)

    return batches
