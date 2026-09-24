"""
CreditBridge - Alternative Credit Scoring Engine
Real Data Mode: Data Quality, Evidence Coverage & Distribution Diagnostics
Path: src/real_data_quality.py

Computes deterministic data-quality metrics, evidence coverage ratios, classification exposure,
and distribution diagnostics relative to the synthetic baseline distribution.
Produces the authoritative RealDataQualityReport for downstream Part 7 model scoring handoff.
"""

import os
import json
from datetime import date
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.real_data_contracts import (
    RAW_BORROWER_COLUMNS,
    UNAVAILABLE_MODEL_COLUMNS,
    DistributionStatus,
    AssessmentQualityStatus,
    SufficiencyTier,
    ProvenanceState,
    NormalizedCategory,
    FieldQualityDiagnostic,
    RealDataQualityReport,
    ParsedStatement,
    RealBorrowerPayload,
)


_CACHED_REFERENCE_DISTRIBUTIONS: Optional[Dict[str, Any]] = None


# -----------------------------------------------------------------------------
# 1. REFERENCE DISTRIBUTION GENERATION & CACHING
# -----------------------------------------------------------------------------

def generate_reference_distributions(
    csv_path: Optional[str] = None,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Computes summary percentiles and category frequencies from the synthetic training baseline.
    Does NOT use the target variable ('defaulted').
    Persists to JSON for fast, deterministic loading without reparsing the 8,000-row CSV every run.
    """
    if csv_path is None:
        src_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(src_dir, ".."))
        csv_path = os.path.join(project_root, "data", "synthetic_borrowers.csv")

    if output_path is None:
        src_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(src_dir, ".."))
        output_path = os.path.join(project_root, "data", "reference_distributions.json")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Synthetic training data not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    # Exclude non-feature columns
    feature_cols = [c for c in RAW_BORROWER_COLUMNS if c != "borrower_id"]

    ref_data: Dict[str, Any] = {
        "metadata": {
            "source_dataset": "data/synthetic_borrowers.csv",
            "total_training_records": len(df),
            "generated_timestamp": date.today().isoformat(),
        },
        "numeric_features": {},
        "categorical_features": {},
    }

    categorical_cols = ["occupation_type", "city_tier"]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    # Compute numeric distribution percentiles
    for col in numeric_cols:
        if col in df.columns:
            s = df[col].dropna().astype(float)
            if len(s) > 0:
                p_min = float(s.min())
                p5 = float(np.percentile(s, 5))
                p25 = float(np.percentile(s, 25))
                p50 = float(np.percentile(s, 50))
                p75 = float(np.percentile(s, 75))
                p95 = float(np.percentile(s, 95))
                p_max = float(s.max())
                mean = float(s.mean())
                std = float(s.std(ddof=1)) if len(s) > 1 else 0.0

                ref_data["numeric_features"][col] = {
                    "count": int(len(s)),
                    "min": round(p_min, 4),
                    "p5": round(p5, 4),
                    "p25": round(p25, 4),
                    "median": round(p50, 4),
                    "p75": round(p75, 4),
                    "p95": round(p95, 4),
                    "max": round(p_max, 4),
                    "mean": round(mean, 4),
                    "std": round(std, 4),
                }

    # Compute categorical distributions
    for col in categorical_cols:
        if col in df.columns:
            s_cat = df[col].dropna().astype(str)
            val_counts = s_cat.value_counts(normalize=True).to_dict()
            ref_data["categorical_features"][col] = {
                "allowed_categories": sorted(list(val_counts.keys())),
                "frequencies": {k: round(float(v), 4) for k, v in val_counts.items()},
            }

    # Ensure output directory exists and save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ref_data, f, indent=2)

    return ref_data


def load_reference_distributions(
    json_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Loads and caches the synthetic reference distribution artifact.
    Generates it if not yet present on disk.
    """
    global _CACHED_REFERENCE_DISTRIBUTIONS
    if _CACHED_REFERENCE_DISTRIBUTIONS is not None:
        return _CACHED_REFERENCE_DISTRIBUTIONS

    if json_path is None:
        src_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(src_dir, ".."))
        json_path = os.path.join(project_root, "data", "reference_distributions.json")

    if not os.path.exists(json_path):
        ref = generate_reference_distributions(output_path=json_path)
    else:
        with open(json_path, "r", encoding="utf-8") as f:
            ref = json.load(f)

    _CACHED_REFERENCE_DISTRIBUTIONS = ref
    return ref


# -----------------------------------------------------------------------------
# 2. FIELD-LEVEL DISTRIBUTION EVALUATION
# -----------------------------------------------------------------------------

def evaluate_field_distribution(
    feature_name: str,
    value: Any,
    provenance_state: str,
    ref_data: Dict[str, Any],
) -> FieldQualityDiagnostic:
    """
    Evaluates an individual feature against the synthetic baseline distribution.
    
    Status Rules:
    - NOT_ASSESSABLE: value is NaN, null, or feature is unavailable.
    - OUTSIDE_OBSERVED_RANGE: strictly < min or > max observed in synthetic baseline.
    - NEAR_BOUNDARY: inside [min, p5) or (p95, max].
    - WITHIN_RANGE: comfortably within [p5, p95].
    - KNOWN_IN_TRAINING: categorical value exists in training categories.
    - UNSEEN_IN_TRAINING: categorical value never seen in training categories.
    """
    numeric_refs = ref_data.get("numeric_features", {})
    categorical_refs = ref_data.get("categorical_features", {})

    # 1. Check if unassessable or NaN
    if value is None or (isinstance(value, (int, float)) and np.isnan(value)) or provenance_state == ProvenanceState.UNAVAILABLE.value:
        return FieldQualityDiagnostic(
            feature_name=feature_name,
            value=None,
            provenance_state=provenance_state,
            distribution_status=DistributionStatus.NOT_ASSESSABLE.value,
            warning_message="Feature is unavailable or imputed; empirical distribution comparison not assessable.",
        )

    # 2. Categorical evaluation
    if feature_name in categorical_refs:
        val_str = str(value).strip()
        allowed = categorical_refs[feature_name].get("allowed_categories", [])
        if val_str in allowed:
            return FieldQualityDiagnostic(
                feature_name=feature_name,
                value=val_str,
                provenance_state=provenance_state,
                distribution_status=DistributionStatus.KNOWN_IN_TRAINING.value,
            )
        else:
            return FieldQualityDiagnostic(
                feature_name=feature_name,
                value=val_str,
                provenance_state=provenance_state,
                distribution_status=DistributionStatus.UNSEEN_IN_TRAINING.value,
                warning_message=f"Category '{val_str}' was unseen in synthetic development baseline. Allowed: {allowed}",
            )

    # 3. Numeric evaluation
    if feature_name in numeric_refs:
        try:
            num_val = float(value)
        except (ValueError, TypeError):
            return FieldQualityDiagnostic(
                feature_name=feature_name,
                value=value,
                provenance_state=provenance_state,
                distribution_status=DistributionStatus.NOT_ASSESSABLE.value,
                warning_message=f"Non-numeric value '{value}' encountered for numeric feature '{feature_name}'.",
            )

        stat = numeric_refs[feature_name]
        p_min = stat["min"]
        p5 = stat["p5"]
        p95 = stat["p95"]
        p_max = stat["max"]

        if num_val < p_min or num_val > p_max:
            status = DistributionStatus.OUTSIDE_OBSERVED_RANGE.value
            warn = (
                f"Value {num_val:g} is outside observed synthetic development range "
                f"[{p_min:g}, {p_max:g}]."
            )
        elif num_val < p5 or num_val > p95:
            status = DistributionStatus.NEAR_BOUNDARY.value
            warn = (
                f"Value {num_val:g} is near distribution boundary (outside 5th-95th percentile: "
                f"[{p5:g}, {p95:g}])."
            )
        else:
            status = DistributionStatus.WITHIN_RANGE.value
            warn = None

        return FieldQualityDiagnostic(
            feature_name=feature_name,
            value=num_val,
            provenance_state=provenance_state,
            distribution_status=status,
            training_min=p_min,
            training_p5=p5,
            training_p95=p95,
            training_max=p_max,
            warning_message=warn,
        )

    # Fallback for features without reference stats (e.g. borrower_id)
    return FieldQualityDiagnostic(
        feature_name=feature_name,
        value=value,
        provenance_state=provenance_state,
        distribution_status=DistributionStatus.NOT_ASSESSABLE.value,
    )


# -----------------------------------------------------------------------------
# 3. AUTHORITATIVE QUALITY REPORT BUILDER
# -----------------------------------------------------------------------------

def build_data_quality_report(
    statement: ParsedStatement,
    borrower_payload: RealBorrowerPayload,
    ref_distributions: Optional[Dict[str, Any]] = None,
) -> RealDataQualityReport:
    """
    Constructs the definitive, comprehensive RealDataQualityReport by synthesizing
    parser diagnostics, transaction intelligence, history sufficiency, evidence coverage,
    and distribution diagnostics.
    
    Parameters:
    -----------
    statement : ParsedStatement
        Output from Part 4 CSV parsing.
    borrower_payload : RealBorrowerPayload
        Output from Part 5 feature mapping.
    ref_distributions : Optional[Dict[str, Any]]
        Pre-loaded reference distribution. Defaults to load_reference_distributions().
        
    Returns:
    --------
    RealDataQualityReport:
        Immutable dataclass instance detailing complete data health.
    """
    if ref_distributions is None:
        ref_distributions = load_reference_distributions()

    # 1. Parse & Transaction Quality Metrics
    rows_rec = int(statement.rows_received)
    rows_acc = int(statement.rows_parsed)
    rows_rej = int(statement.rows_rejected)
    dup_count = int(statement.duplicate_rows)
    dup_rate = round(float(dup_count / max(rows_rec, 1)), 4)

    # Parse diagnostic specifics from rejected rows
    invalid_date_count = 0
    missing_amount_count = 0
    for diag in statement.diagnostics:
        reason_str = str(diag.get("reason", "")).lower()
        if "date" in reason_str:
            invalid_date_count += 1
        if "amount" in reason_str:
            missing_amount_count += 1

    # Transaction level metrics
    tx_df = statement.transactions
    if not tx_df.empty and len(tx_df) > 0:
        unknown_type_count = int((tx_df["transaction_type"] == "unknown").sum())
        unknown_cat_count = int((tx_df["normalized_category"] == NormalizedCategory.UNKNOWN.value).sum())
        cat_coverage = round(float((rows_acc - unknown_cat_count) / max(rows_acc, 1)), 4)
        unknown_cat_share = round(float(unknown_cat_count / max(rows_acc, 1)), 4)
    else:
        unknown_type_count = 0
        unknown_cat_count = 0
        cat_coverage = 0.0
        unknown_cat_share = 0.0

    is_sparse_class = unknown_cat_share > 0.40

    # 2. History Sufficiency Metrics
    hist = borrower_payload.history_report
    hist_start = hist.start_date
    hist_end = hist.end_date
    hist_days = int(hist.total_calendar_days)
    hist_months = float(hist.approx_months)
    hist_tier = str(hist.sufficiency_tier)
    tx_count = int(hist.total_usable_transactions)

    # 3. Feature Evidence Composition
    # Denominator is strictly 21 model-facing features (excludes borrower_id)
    model_features = [c for c in RAW_BORROWER_COLUMNS if c != "borrower_id"]
    total_model_features = len(model_features)  # 21

    prov_records = borrower_payload.provenance_records

    obs_count = 0
    der_count = 0
    self_rep_count = 0
    unavail_count = 0
    imp_count = 0

    for col in model_features:
        rec = prov_records.get(col)
        if rec:
            state = rec.provenance_state
            if state == ProvenanceState.OBSERVED.value:
                obs_count += 1
            elif state == ProvenanceState.DERIVED.value:
                der_count += 1
            elif state == ProvenanceState.SELF_REPORTED.value:
                self_rep_count += 1
            elif state == ProvenanceState.UNAVAILABLE.value:
                unavail_count += 1

            if rec.is_imputed:
                imp_count += 1

    evidence_coverage_ratio = round(float((obs_count + der_count + self_rep_count) / total_model_features), 4)
    derived_coverage_ratio = round(float(der_count / total_model_features), 4)
    self_rep_coverage_ratio = round(float(self_rep_count / total_model_features), 4)
    imputed_ratio = round(float(imp_count / total_model_features), 4)

    # 4. Field-Level Distribution Diagnostics
    field_diags: List[FieldQualityDiagnostic] = []
    outside_range_cnt = 0
    near_bound_cnt = 0
    within_range_cnt = 0

    borrower_row = borrower_payload.borrower_df.iloc[0]

    for col in model_features:
        val = borrower_row[col]
        rec = prov_records.get(col)
        p_state = rec.provenance_state if rec else ProvenanceState.UNAVAILABLE.value

        diag = evaluate_field_distribution(
            feature_name=col,
            value=val,
            provenance_state=p_state,
            ref_data=ref_distributions,
        )
        field_diags.append(diag)

        if diag.distribution_status == DistributionStatus.OUTSIDE_OBSERVED_RANGE.value:
            outside_range_cnt += 1
        elif diag.distribution_status == DistributionStatus.NEAR_BOUNDARY.value:
            near_bound_cnt += 1
        elif diag.distribution_status in (
            DistributionStatus.WITHIN_RANGE.value,
            DistributionStatus.KNOWN_IN_TRAINING.value,
        ):
            within_range_cnt += 1

    # 5. Assessment Summary & Failure Mode Analysis
    quality_warnings: List[str] = []
    blocker_reasons: List[str] = []

    # Check hard blockers
    if rows_acc == 0:
        blocker_reasons.append("Statement contains zero valid transactions.")
    if not hist.is_scoreable:
        blocker_reasons.append(hist.warning_message or "Statement history does not meet minimum underwriting sufficiency.")
    if evidence_coverage_ratio < 0.35:
        blocker_reasons.append(f"Critically low evidence coverage ({evidence_coverage_ratio:.1%}). Below 35% minimum threshold.")

    # Check quality warnings
    if hist_tier == SufficiencyTier.MARGINAL.value:
        quality_warnings.append(
            f"History is limited ({hist_days} days). Short-window monthly metrics capped at manual review tier."
        )
    if is_sparse_class:
        quality_warnings.append(
            f"High unknown transaction share ({unknown_cat_share:.1%}). Cashflow and merchant profiling may be degraded."
        )
    if outside_range_cnt > 0:
        quality_warnings.append(
            f"{outside_range_cnt} feature(s) fall outside the synthetic training distribution range."
        )
    if dup_rate > 0.10:
        quality_warnings.append(
            f"Elevated duplicate transaction rate detected ({dup_rate:.1%})."
        )
    if unavail_count > 0:
        quality_warnings.append(
            f"{unavail_count} model features are unavailable on statements and will be imputed using population medians."
        )

    # Determine Overall Evidence Quality Status
    if len(blocker_reasons) > 0:
        overall_status = AssessmentQualityStatus.INSUFFICIENT.value
        is_eligible = False
    elif hist_tier == SufficiencyTier.MARGINAL.value or is_sparse_class or outside_range_cnt > 4 or evidence_coverage_ratio < 0.50:
        overall_status = AssessmentQualityStatus.LIMITED.value
        is_eligible = True
    else:
        overall_status = AssessmentQualityStatus.SUFFICIENT.value
        is_eligible = True

    return RealDataQualityReport(
        rows_received=rows_rec,
        rows_accepted=rows_acc,
        rows_rejected=rows_rej,
        duplicate_count=dup_count,
        duplicate_rate=dup_rate,
        invalid_date_count=invalid_date_count,
        missing_amount_count=missing_amount_count,
        unknown_transaction_type_count=unknown_type_count,
        unknown_category_count=unknown_cat_count,
        history_start_date=hist_start,
        history_end_date=hist_end,
        history_days=hist_days,
        history_months=hist_months,
        transaction_count=tx_count,
        history_tier=hist_tier,
        category_coverage=cat_coverage,
        unknown_category_share=unknown_cat_share,
        is_classification_sparse=is_sparse_class,
        total_model_features=total_model_features,
        observed_count=obs_count,
        derived_count=der_count,
        self_reported_count=self_rep_count,
        unavailable_count=unavail_count,
        imputed_count=imp_count,
        evidence_coverage_ratio=evidence_coverage_ratio,
        derived_coverage_ratio=derived_coverage_ratio,
        self_reported_coverage_ratio=self_rep_coverage_ratio,
        imputed_ratio=imputed_ratio,
        field_diagnostics=field_diags,
        outside_range_count=outside_range_cnt,
        near_boundary_count=near_bound_cnt,
        within_range_count=within_range_cnt,
        overall_status=overall_status,
        is_eligible_for_scoring=is_eligible,
        quality_warnings=quality_warnings,
        blocker_reasons=blocker_reasons,
    )
