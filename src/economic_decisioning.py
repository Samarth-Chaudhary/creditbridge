"""
CreditBridge - Economic Decisioning & Expected Loss Foundation
Path: src/economic_decisioning.py

Implements configurable credit risk economic decisioning:
- Expected Loss (EL) = PD * LGD * EAD
- Configurable Underwriting Policy (APPROVE / REVIEW / DECLINE)
- Portfolio Policy Simulation:
    * Approval rate, review rate, decline rate
    * Portfolio expected loss vs approved expected loss
    * Approved cohort bad rate vs population base rate
    * Tradeoff frontiers across varying threshold configurations
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DecisionPolicyConfig:
    """Configurable credit policy decision parameters."""
    auto_approve_max_pd: float = 0.08      # Borrowers with PD <= 8% auto-approved
    manual_review_max_pd: float = 0.20     # Borrowers with 8% < PD <= 20% sent to manual underwriter
    min_score_cutoff: int = 650            # Minimum CreditBridge Risk Score for auto-approval
    default_lgd: float = 0.65              # Loss Given Default (unsecured thin-file lending benchmark ~65%)
    default_ead: float = 25000.0           # Exposure at Default (representative thin-file loan limit in INR)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_expected_loss(pd_val: float, lgd: float, ead: float) -> float:
    """
    Computes Expected Loss (EL) = PD * LGD * EAD.
    All inputs must be non-negative; PD and LGD are bounded in [0, 1].
    """
    pd_clamped = float(np.clip(pd_val, 0.0, 1.0))
    lgd_clamped = float(np.clip(lgd, 0.0, 1.0))
    ead_clamped = max(float(ead), 0.0)
    return round(pd_clamped * lgd_clamped * ead_clamped, 2)


def make_underwriting_decision(
    pd_val: float,
    score: int,
    config: Optional[DecisionPolicyConfig] = None,
    loan_amount: Optional[float] = None,
    custom_lgd: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Evaluates an individual borrower application against the active policy contract.
    Returns machine-readable decision: APPROVE, REVIEW, or DECLINE, with expected loss.
    """
    cfg = config or DecisionPolicyConfig()
    ead = loan_amount if loan_amount is not None else cfg.default_ead
    lgd = custom_lgd if custom_lgd is not None else cfg.default_lgd
    el = compute_expected_loss(pd_val, lgd, ead)

    if pd_val <= cfg.auto_approve_max_pd and score >= cfg.min_score_cutoff:
        action = "APPROVE"
        rationale = (
            f"Automated Approval: Calibrated PD ({pd_val:.2%}) is within policy limit "
            f"({cfg.auto_approve_max_pd:.2%}) and Risk Score ({score}) meets cutoff ({cfg.min_score_cutoff})."
        )
    elif pd_val <= cfg.manual_review_max_pd:
        action = "REVIEW"
        rationale = (
            f"Manual Review Required: Calibrated PD ({pd_val:.2%}) exceeds automated threshold "
            f"but remains below decline ceiling ({cfg.manual_review_max_pd:.2%}). Underwriter scrutiny required."
        )
    else:
        action = "DECLINE"
        rationale = (
            f"Policy Decline: Calibrated PD ({pd_val:.2%}) exceeds maximum risk tolerance "
            f"({cfg.manual_review_max_pd:.2%}) or Score ({score}) is below underwriting threshold."
        )

    return {
        "decision": action,
        "calibrated_pd": round(pd_val, 4),
        "creditbridge_score": score,
        "exposure_at_default": ead,
        "loss_given_default": lgd,
        "expected_loss_inr": el,
        "rationale": rationale,
        "policy_config": cfg.to_dict(),
    }


def simulate_policy_tradeoffs(
    predicted_pds: Union[np.ndarray, pd.Series],
    true_defaults: Union[np.ndarray, pd.Series],
    config: Optional[DecisionPolicyConfig] = None,
    ead_values: Optional[Union[np.ndarray, pd.Series]] = None,
    lgd: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Simulates portfolio policy performance across an entire applicant cohort.
    Computes approval rates, expected loss, bad-rate reduction, and review overhead.
    """
    cfg = config or DecisionPolicyConfig()
    pds = np.asarray(predicted_pds).astype(float)
    y_true = np.asarray(true_defaults).astype(int)
    n = len(pds)

    if n == 0:
        raise ValueError("Cannot simulate policy on empty applicant cohort.")

    lgd_val = lgd if lgd is not None else cfg.default_lgd
    if ead_values is None:
        eads = np.full(n, cfg.default_ead)
    else:
        eads = np.asarray(ead_values).astype(float)

    # Classify each applicant according to configured thresholds
    approved_mask = pds <= cfg.auto_approve_max_pd
    review_mask = (pds > cfg.auto_approve_max_pd) & (pds <= cfg.manual_review_max_pd)
    declined_mask = pds > cfg.manual_review_max_pd

    app_count = int(approved_mask.sum())
    rev_count = int(review_mask.sum())
    dec_count = int(declined_mask.sum())

    # Individual expected loss
    individual_el = pds * lgd_val * eads
    total_portfolio_el = float(individual_el.sum())
    approved_cohort_el = float(individual_el[approved_mask].sum()) if app_count > 0 else 0.0

    # Bad rates
    population_bad_rate = float(y_true.mean())
    approved_bad_rate = float(y_true[approved_mask].mean()) if app_count > 0 else 0.0
    declined_bad_rate = float(y_true[declined_mask].mean()) if dec_count > 0 else 0.0
    reviewed_bad_rate = float(y_true[review_mask].mean()) if rev_count > 0 else 0.0

    bad_rate_reduction = (
        (population_bad_rate - approved_bad_rate) / max(population_bad_rate, 1e-6)
    ) * 100.0

    return {
        "total_applicants": n,
        "approval_count": app_count,
        "approval_rate": round(app_count / n, 4),
        "review_count": rev_count,
        "review_rate": round(rev_count / n, 4),
        "decline_count": dec_count,
        "decline_rate": round(dec_count / n, 4),
        "total_portfolio_expected_loss": round(total_portfolio_el, 2),
        "approved_expected_loss": round(approved_cohort_el, 2),
        "population_base_bad_rate": round(population_bad_rate, 4),
        "approved_cohort_bad_rate": round(approved_bad_rate, 4),
        "declined_cohort_bad_rate": round(declined_bad_rate, 4),
        "reviewed_cohort_bad_rate": round(reviewed_bad_rate, 4),
        "bad_rate_reduction_pct": round(bad_rate_reduction, 2),
        "policy_config": cfg.to_dict(),
    }
