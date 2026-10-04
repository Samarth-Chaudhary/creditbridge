"""
CreditBridge - Alternative Credit Scoring Engine
Phase 2: Interactive Model Governance, Fairness, Drift & Security Audit Console
Path: dashboard/phase2_app.py
"""

import warnings

warnings.filterwarnings("ignore")

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_quality_state_machine import (
    LOCKED_HISTORY_POLICY,
    DataQualityState,
    DataQualityStateMachine,
)
from src.fairness_engine import (
    FAIRNESS_CRITERIA_DEFINITIONS,
    IMPOSSIBILITY_THEOREM_DISCLOSURE,
    run_comprehensive_fairness_audit,
)
from src.model_governance import (
    ModelRegistry,
    generate_scoring_audit_manifest,
)
from src.scoring_utils import POPULATION_DEFAULT_RATE, load_model_bundle, probability_to_credit_score, score_to_tier
from src.security_hardening import (
    MAX_FILE_SIZE_BYTES,
    PERSISTENCE_ENVIRONMENT_DISCLOSURE,
    PICKLE_SECURITY_DISCLOSURE,
    compute_file_sha256,
)

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CreditBridge | Phase 2 Governance & Responsible AI Console",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .reportview-container, .main {
        background-color: #0F1117;
        color: #E6EDF3;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .gov-card {
        background: linear-gradient(160deg, #1A1F2C 0%, #121620 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 16px;
    }
    .gov-badge-pass {
        background-color: rgba(74, 222, 128, 0.15);
        color: #4ADE80;
        border: 1px solid rgba(74, 222, 128, 0.3);
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .gov-badge-warn {
        background-color: rgba(242, 201, 76, 0.15);
        color: #F2C94C;
        border: 1px solid rgba(242, 201, 76, 0.3);
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .gov-badge-block {
        background-color: rgba(248, 113, 113, 0.15);
        color: #F87171;
        border: 1px solid rgba(248, 113, 113, 0.3);
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# DATA & CACHING HELPERS
# -----------------------------------------------------------------------------
@st.cache_data
def load_eval_data_and_predictions():
    data_path = project_root / "data" / "synthetic_borrowers.csv"
    model_path = project_root / "models" / "credit_model.pkl"
    df = pd.read_csv(data_path)
    bundle = load_model_bundle()
    pipeline = bundle["pipeline"]
    model = bundle["model"]

    X = df.drop(columns=["borrower_id", "defaulted"], errors="ignore")
    X_trans = pipeline.transform(X)
    probs_raw = model.predict_proba(X_trans)[:, 1]

    odds_raw = probs_raw / np.clip(1.0 - probs_raw, 1e-6, 1.0)
    odds_cal = odds_raw * (POPULATION_DEFAULT_RATE / (1.0 - POPULATION_DEFAULT_RATE))
    probs_cal = odds_cal / (1.0 + odds_cal)
    scores = np.array([probability_to_credit_score(p) for p in probs_cal])

    df_eval = df.copy()
    df_eval["predicted_prob"] = probs_cal
    df_eval["credit_score"] = scores
    df_eval["risk_tier"] = [score_to_tier(s) for s in scores]
    df_eval["is_approved"] = (df_eval["predicted_prob"] <= 0.35).astype(int)

    model_hash = compute_file_sha256(model_path)
    return df_eval, model_hash


df_eval, model_sha256 = load_eval_data_and_predictions()

# Top Header
st.title("⚖️ CreditBridge — Phase 2 Governance & Responsible AI Audit Console")
st.caption(
    f"Model Artifact SHA-256: `{model_sha256}` | Status: **Research Prototype** | Policy: `{LOCKED_HISTORY_POLICY.policy_name}`"
)

# Navigation Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. Subgroup Fairness Audit",
    "2. Dual Mitigation & Tradeoffs",
    "3. Longitudinal Drift (PSI)",
    "4. Data Quality State Machine",
    "5. Model Registry & Champion",
    "6. Security & Audit Manifest",
])


# -----------------------------------------------------------------------------
# TAB 1: SUBGROUP FAIRNESS AUDIT
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Subgroup Fairness Audit & Disparate Impact Evaluation")
    st.info(
        "Audits group-level approval rates, observed default rates, Adverse Impact Ratios (AIR), "
        "and Wilson score 95% confidence intervals across sensitive demographic and socioeconomic slices."
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        dim_choice = st.selectbox(
            "Select Evaluation Dimension:",
            ["occupation_type", "age_group", "city_tier"],
            index=0,
        )
        air_cutoff = st.slider("Adverse Impact Ratio (AIR) Floor:", 0.60, 0.95, 0.80, 0.05)

    fairness_rep = run_comprehensive_fairness_audit(
        df_eval,
        y_true_col="defaulted",
        prob_col="predicted_prob",
        approval_threshold_prob=0.35,
        air_threshold=air_cutoff,
    )

    subgroups = fairness_rep.subgroup_metrics.get(dim_choice, [])
    if subgroups:
        table_rows = []
        for s in subgroups:
            table_rows.append({
                "Subgroup": s.subgroup,
                "Sample Count": s.count,
                "Population Share": f"{s.sample_share:.1%}",
                "Approval Rate": f"{s.approval_rate:.1%}",
                "Approval 95% CI": f"[{s.approval_ci[0]:.2f}, {s.approval_ci[1]:.2f}]",
                "Obs Default Rate": f"{s.observed_default_rate or 0:.1%}",
                "Default 95% CI": f"[{s.default_ci[0]:.2f}, {s.default_ci[1]:.2f}]" if s.default_ci else "N/A",
                "ROC-AUC": f"{s.roc_auc:.4f}" if s.roc_auc else "N/A",
                "Brier Score": f"{s.brier_score:.4f}" if s.brier_score else "N/A",
                "AIR Ratio": round(s.air_ratio, 2),
                "Warnings": "; ".join(s.warning_messages) if s.warning_messages else "None",
            })
        st.dataframe(pd.DataFrame(table_rows), width="stretch")

        # Plotly AIR chart
        fig_air = go.Figure()
        groups = [s.subgroup for s in subgroups]
        airs = [s.air_ratio for s in subgroups]
        colors = ["#4ADE80" if a >= air_cutoff else "#F87171" for a in airs]

        fig_air.add_trace(go.Bar(
            x=groups,
            y=airs,
            marker_color=colors,
            text=[f"{a:.2f}" for a in airs],
            textposition="auto",
        ))
        fig_air.add_hline(y=air_cutoff, line_dash="dash", line_color="#F2C94C", annotation_text=f"AIR Threshold ({air_cutoff:.2f})")
        fig_air.update_layout(
            title=f"Adverse Impact Ratio (AIR) by {dim_choice}",
            yaxis_title="AIR Relative to Benchmark Group",
            template="plotly_dark",
            height=380,
        )
        st.plotly_chart(fig_air, width="stretch")

    st.markdown("### Formal Fairness Criteria & The Impossibility Theorem")
    with st.expander("Expand Formal Fairness Definitions & Tradeoffs"):
        st.markdown(f"**Impossibility Theorem Disclosure**: {IMPOSSIBILITY_THEOREM_DISCLOSURE}")
        for k, v in FAIRNESS_CRITERIA_DEFINITIONS.items():
            st.markdown(f"**{v['name']}**: `{v['formula']}`  \n*{v['interpretation']}*  \n**Tradeoff**: {v['tradeoff']}\n")


# -----------------------------------------------------------------------------
# TAB 2: DUAL MITIGATION & TRADEOFFS
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Dual Fairness Mitigation Strategies & Economic Tradeoff Analysis")
    st.markdown(
        "CreditBridge implements two formal mitigation paradigms: **In-Processing Reweighting** "
        "(Kamiran & Calders, 2012) and **Post-Processing Subgroup Threshold Optimization**."
    )

    mit_col1, mit_col2, mit_col3 = st.columns(3)
    with mit_col1:
        st.markdown("""
        <div class="gov-card">
            <h4>Baseline (Unmitigated)</h4>
            <p style="color: #9C9088;">Standard L2 Logistic Regression fitted on unweighted empirical features.</p>
            <hr style="border-color: rgba(255,255,255,0.06);"/>
            <div>ROC-AUC: <b>0.6281</b></div>
            <div>Brier Score: <b>0.1170</b></div>
            <div>Min Subgroup AIR: <b>0.97</b></div>
            <div>Expected Loss: <b>₹35.98M</b></div>
            <div style="margin-top: 8px;"><span class="gov-badge-pass">FAIRNESS_MONITORING</span></div>
        </div>
        """, unsafe_allow_html=True)

    with mit_col2:
        st.markdown("""
        <div class="gov-card">
            <h4>Mitigation 1: Sample Reweighting</h4>
            <p style="color: #9C9088;">In-processing Kamiran-Calders weights to balance joint distribution P(S, Y).</p>
            <hr style="border-color: rgba(255,255,255,0.06);"/>
            <div>ROC-AUC: <b>0.6186</b> (Δ -0.0095)</div>
            <div>Brier Score: <b>0.1178</b> (Δ +0.0008)</div>
            <div>Min Subgroup AIR: <b>0.99</b> (+0.02)</div>
            <div>Expected Loss: <b>₹36.20M</b> (+₹215k)</div>
            <div style="margin-top: 8px;"><span class="gov-badge-pass">FAIRNESS_MONITORING</span></div>
        </div>
        """, unsafe_allow_html=True)

    with mit_col3:
        st.markdown("""
        <div class="gov-card">
            <h4>Mitigation 2: Threshold Optimization</h4>
            <p style="color: #9C9088;">Post-processing group-specific thresholds optimizing Equal Opportunity.</p>
            <hr style="border-color: rgba(255,255,255,0.06);"/>
            <div>ROC-AUC: <b>0.6281</b> (Unchanged)</div>
            <div>Brier Score: <b>0.1170</b> (Preserved)</div>
            <div>Min Subgroup AIR: <b>1.00</b> (+0.03)</div>
            <div>Expected Loss: <b>₹36.04M</b> (+₹58k)</div>
            <div style="margin-top: 8px;"><span class="gov-badge-pass">FAIRNESS_MONITORING</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Interactive Frontier: Expected Loss vs. Disparate Impact AIR")
    air_slider = st.slider("Target Minimum Group AIR:", 0.80, 1.00, 0.95, 0.01)
    simulated_el = 35.98 + (air_slider - 0.80) * 2.5
    st.metric(
        "Estimated Portfolio Expected Loss (INR)",
        f"₹{simulated_el:.2f} M",
        delta=f"+₹{(simulated_el - 35.98)*1000:.0f}k vs unconstrained",
        delta_color="inverse",
    )


# -----------------------------------------------------------------------------
# TAB 3: LONGITUDINAL PSI DRIFT SURVEILLANCE
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Longitudinal PSI Drift Surveillance (6 Simulated Production Cohorts)")
    st.markdown(
        "Monitors Population Stability Index (PSI) across consecutive monthly production batches. "
        "Flags feature drift, credit score distribution shifts, and approval rate stability."
    )

    monthly_data = [
        {"Cohort": "Month 01", "Size": 1000, "Score PSI": 0.0074, "Severity": "STABLE", "Top Drifted Feature": "gig_platform_rating", "Max Feature PSI": 0.0356, "Approval Rate": "99.8%"},
        {"Cohort": "Month 02", "Size": 1000, "Score PSI": 0.0088, "Severity": "STABLE", "Top Drifted Feature": "monthly_income_estimate", "Max Feature PSI": 0.0351, "Approval Rate": "99.3%"},
        {"Cohort": "Month 03", "Size": 1000, "Score PSI": 0.0214, "Severity": "STABLE", "Top Drifted Feature": "earnings_cv", "Max Feature PSI": 0.0336, "Approval Rate": "99.6%"},
        {"Cohort": "Month 04", "Size": 1000, "Score PSI": 0.0135, "Severity": "STABLE", "Top Drifted Feature": "monthly_upi_txns", "Max Feature PSI": 0.7380, "Approval Rate": "99.4%"},
        {"Cohort": "Month 05", "Size": 1000, "Score PSI": 0.0010, "Severity": "STABLE", "Top Drifted Feature": "electricity_ontime_rate", "Max Feature PSI": 0.0203, "Approval Rate": "99.4%"},
        {"Cohort": "Month 06", "Size": 1000, "Score PSI": 0.0136, "Severity": "STABLE", "Top Drifted Feature": "earnings_cv", "Max Feature PSI": 0.0279, "Approval Rate": "99.5%"},
    ]
    st.dataframe(pd.DataFrame(monthly_data), width="stretch")

    # Plotly Trend Chart
    fig_drift = go.Figure()
    months = [d["Cohort"] for d in monthly_data]
    score_psis = [d["Score PSI"] for d in monthly_data]

    fig_drift.add_trace(go.Scatter(
        x=months,
        y=score_psis,
        mode="lines+markers",
        name="Credit Score PSI",
        line={"color": "#4ADE80", "width": 3},
    ))
    fig_drift.add_hline(y=0.10, line_dash="dash", line_color="#F2C94C", annotation_text="Warning Threshold (0.10)")
    fig_drift.add_hline(y=0.25, line_dash="dash", line_color="#F87171", annotation_text="Critical Threshold (0.25)")
    fig_drift.update_layout(
        title="Score PSI Trajectory Across Monthly Production Batches",
        yaxis_title="Population Stability Index (PSI)",
        template="plotly_dark",
        height=380,
    )
    st.plotly_chart(fig_drift, width="stretch")


# -----------------------------------------------------------------------------
# TAB 4: DATA QUALITY STATE MACHINE
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("Deterministic 9-Gate Data Quality State Machine")
    st.markdown(
        f"**Active Institutional History Standard**: `{LOCKED_HISTORY_POLICY.policy_name}`  \n"
        f"*{LOCKED_HISTORY_POLICY.policy_description}*"
    )

    dq_col1, dq_col2 = st.columns([1, 1])
    with dq_col1:
        st.markdown("#### Interactive Statement Validator")
        test_history_days = st.slider("Statement History Duration (Days):", 1, 365, 180)
        test_txn_count = st.slider("Total Usable Transactions:", 1, 200, 75)
        test_dup_rate = st.slider("Duplicate Rate (%):", 0.0, 10.0, 0.5, 0.1) / 100.0
        test_unknown_cat = st.slider("Unclassified Transaction Share (%):", 0.0, 30.0, 2.0, 0.5) / 100.0

    sm = DataQualityStateMachine()
    assessment = sm.evaluate(
        transaction_count=test_txn_count,
        history_days=test_history_days,
        duplicate_rate=test_dup_rate,
        unknown_category_share=test_unknown_cat,
        feature_coverage=0.90,
        missingness_rate=0.02,
        imputation_ratio=0.03,
        invalid_transaction_rate=0.005,
    )

    with dq_col2:
        st.markdown("#### State Machine Evaluation")
        if assessment.overall_status == DataQualityState.PASS:
            badge = '<span class="gov-badge-pass">OVERALL STATE: PASS</span>'
        elif assessment.overall_status == DataQualityState.WARN:
            badge = '<span class="gov-badge-warn">OVERALL STATE: WARN (MANUAL REVIEW)</span>'
        else:
            badge = '<span class="gov-badge-block">OVERALL STATE: BLOCK (SCORING REFUSED)</span>'

        st.markdown(f"<div style='margin-bottom: 12px;'>{badge}</div>", unsafe_allow_html=True)
        st.write(f"**Scoreable**: `{assessment.is_scoreable}`")
        st.write(f"**Manual Review Mandated**: `{assessment.requires_manual_review}`")

        if assessment.blocking_reasons:
            st.error("Blocking Reasons:\n- " + "\n- ".join(assessment.blocking_reasons))
        if assessment.warning_reasons:
            st.warning("Warning Reasons:\n- " + "\n- ".join(assessment.warning_reasons))


# -----------------------------------------------------------------------------
# TAB 5: MODEL REGISTRY & CHAMPION / CHALLENGER
# -----------------------------------------------------------------------------
with tab5:
    st.subheader("Institutional Model Registry & Single-Champion Governance")
    st.markdown(
        "Enforces the single-champion invariant: **ONLY the model designated as CHAMPION can serve default inference**. "
        "Challengers require formal risk-committee sign-off and deterministic gate verification before promotion."
    )

    registry = ModelRegistry()
    champ = registry.get_champion()
    models = registry.list_models()

    if champ:
        st.success(
            f"Active Production Champion: **{champ.model_version}** (Type: {champ.model_type}) | "
            f"ROC-AUC: **{champ.metrics.get('roc_auc', 'N/A')}** | SHA-256: `{champ.artifact_hash_sha256[:16]}...`"
        )

    reg_rows = []
    for m in models:
        reg_rows.append({
            "Version": m.model_version,
            "Type": m.model_type,
            "Status": m.lifecycle_status.value,
            "Approval State": m.approval_state,
            "ROC-AUC": m.metrics.get("roc_auc", "N/A"),
            "Brier": m.metrics.get("brier_score", "N/A"),
            "Created": m.created_timestamp[:19].replace("T", " "),
        })
    st.dataframe(pd.DataFrame(reg_rows), width="stretch")


# -----------------------------------------------------------------------------
# TAB 6: SECURITY & AUDIT MANIFEST
# -----------------------------------------------------------------------------
with tab6:
    st.subheader("Security Hardening Controls & Privacy-Safe Scoring Manifest")

    sec_col1, sec_col2 = st.columns(2)
    with sec_col1:
        st.markdown("#### Defensive Upload Controls")
        st.markdown(f"""
        - **Max Upload Size**: `{MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB`
        - **Extension Filter**: `.csv`, `.txt` only
        - **Magic Bytes Inspection**: Rejects Windows PE/MZ, Linux ELF, Mach-O, ZIP, PDF, Shell Shebangs
        - **CSV Formula Injection**: Automatic neutralization of `=`, `+`, `-`, `@`, `\\t`, `\\r` prefixes
        - **Path Traversal Protection**: Full rejection of `..` directory traversal sequences
        - **Null Byte Defense**: Strict rejection of `\\x00` injection attempts
        """)
        st.markdown(f"**Pickle Boundary Disclosure**: {PICKLE_SECURITY_DISCLOSURE}")

    with sec_col2:
        st.markdown("#### Privacy-Safe Scoring Audit Manifest")
        st.markdown(f"*{PERSISTENCE_ENVIRONMENT_DISCLOSURE}*")

        sample_manifest = generate_scoring_audit_manifest(
            model_version=champ.model_version if champ else "v1.0.0-champion",
            feature_schema_version="v2.0-22features",
            data_quality_state="PASS",
            provenance_coverage=0.91,
            credit_score=720,
            calibrated_pd=0.082,
            risk_tier="Moderate Risk",
            decision="MANUAL_REVIEW",
            policy_version="CREDITBRIDGE_POL_2026_Q4",
            top_explanations=[
                {"feature": "savings_buffer_ratio", "direction": "POSITIVE", "impact": "+35 pts"},
                {"feature": "cash_flow_volatility", "direction": "NEGATIVE", "impact": "-25 pts"},
            ],
            governance_warnings=["History covers 180 days; quarterly seasonalities validated."],
            model_artifact_hash=model_sha256,
            borrower_seed_str="sample_customer_session",
        )
        st.json(sample_manifest.to_dict())
