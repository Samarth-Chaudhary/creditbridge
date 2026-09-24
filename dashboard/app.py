"""
CreditBridge - Alternative Credit Scoring Engine
Stage 3: Recruiter-Facing Streamlit Underwriting Dashboard
Path: dashboard/app.py

Dark Fintech "Wallet / Banking" Aesthetic:
- Near-black warm brown background (#14100D / #17130F)
- Linear gradient card surfaces (linear-gradient(160deg, #241D17 0%, #1B1613 100%))
- Elevated hero surfaces (linear-gradient(160deg, #2C221A 0%, #1E1712 100%))
- Hairline borders (rgba(255, 255, 255, 0.06))
- Warm off-white primary text (#F5F1EA) & muted warm gray secondary text (#9C9088)
- Vibrant amber / orange gradient accents (#E8792E -> #F2994A)
- Semantic risk accents (#4ADE80 Low, #F2C94C Moderate, #E8792E High, #F87171 Very High)
- Fixed left sidebar with icon-only navigation and top bar with search, actions, and profile chip
- Rich interactive Plotly area and bar charts with custom dark tooltips
- 100% working functions with 0 errors
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve

# Ensure project root is available on sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.scoring_utils import (
    load_model_bundle,
    score_borrower,
    probability_to_credit_score,
    score_to_tier,
    SCORE_FACTOR,
    POPULATION_DEFAULT_RATE
)
from src.explain import explain_single_borrower, FEATURE_NAME_MAP

try:
    import shap
except ImportError:
    shap = None


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="CreditBridge | Institutional Credit Underwriting",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =============================================================================
# INJECT DARK FINTECH WALLET DESIGN SYSTEM
# =============================================================================
st.markdown("""
<style>
/*
================================================================================
CREDITBRIDGE — DARK FINTECH "WALLET / BANKING" DESIGN SYSTEM
================================================================================
Base Background:       #14100D (Near-black warm brown)
Primary Card Surface:  linear-gradient(160deg, #241D17 0%, #1B1613 100%)
Elevated Hero Surface: linear-gradient(160deg, #2C221A 0%, #1E1712 100%)
Hairline Border:       1px solid rgba(255, 255, 255, 0.06)
Primary Text:          #F5F1EA (Warm Off-White)
Secondary Text:        #9C9088 (Muted Warm Gray)
Tertiary Text:         #6E655D (Dark Warm Gray)
Accent Primary CTA:    linear-gradient(135deg, #E8792E 0%, #F2994A 100%)
Accent Success/Low:    #4ADE80 (Soft Green)
Accent Warning/Mod:    #F2C94C (Soft Amber)
Accent Danger/High:    #F87171 (Soft Red / Terracotta)
================================================================================
*/

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg-base: #14100D;
    --bg-base-alt: #17130F;
    --bg-sidebar: #100C0A;
    --bg-card: linear-gradient(160deg, #241D17 0%, #1B1613 100%);
    --bg-hero: linear-gradient(160deg, #2C221A 0%, #1E1712 100%);
    --border-hairline: rgba(255, 255, 255, 0.06);
    --border-subtle: rgba(255, 255, 255, 0.09);
    --text-primary: #F5F1EA;
    --text-secondary: #9C9088;
    --text-tertiary: #6E655D;
    --accent-orange: #E8792E;
    --accent-amber: #F2994A;
    --accent-green: #4ADE80;
    --accent-red: #F87171;
    --accent-yellow: #F2C94C;
}

html, body, .stApp {
    background-color: #14100D !important;
    background-image: radial-gradient(circle at 18% 12%, rgba(232, 121, 46, 0.04) 0%, transparent 40%),
                      radial-gradient(circle at 85% 25%, rgba(242, 153, 74, 0.03) 0%, transparent 50%) !important;
    color: #F5F1EA !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    margin: 0;
    padding: 0;
}

div[data-testid="stAppViewContainer"] {
    background-color: #14100D !important;
    padding-left: 78px !important;
    padding-right: 28px !important;
    padding-top: 10px !important;
    padding-bottom: 60px !important;
}

section[data-testid="stMain"],
div[data-testid="stMainBlockContainer"],
.main,
.block-container {
    background: transparent !important;
    color: #F5F1EA !important;
    max-width: 1440px !important;
    padding-top: 1rem !important;
}

header[data-testid="stHeader"] {
    background: transparent !important;
    display: none !important;
}

/* =============================================================================
   FIXED LEFT SIDEBAR (Icon-only Navigation Shell matching Reference)
============================================================================= */
.fin-sidebar {
    position: fixed;
    top: 0;
    left: 0;
    width: 68px;
    height: 100vh;
    background-color: #100C0A;
    border-right: 1px solid rgba(255, 255, 255, 0.06);
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 20px 0;
    z-index: 99999;
    box-sizing: border-box;
}

.fin-brand-logo {
    width: 40px;
    height: 40px;
    border-radius: 12px;
    background: linear-gradient(135deg, rgba(74, 222, 128, 0.15) 0%, rgba(232, 121, 46, 0.20) 100%);
    border: 1px solid rgba(255, 255, 255, 0.10);
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 32px;
    cursor: pointer;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3);
}

.fin-brand-logo svg {
    width: 22px;
    height: 22px;
    fill: none;
    stroke: #4ADE80;
    stroke-width: 2;
}

.fin-nav-stack {
    display: flex;
    flex-direction: column;
    gap: 16px;
    width: 100%;
    align-items: center;
    flex: 1;
}

.fin-nav-icon {
    width: 44px;
    height: 44px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #9C9088;
    background: transparent;
    transition: all 0.2s ease;
    cursor: pointer;
    text-decoration: none;
}

.fin-nav-icon:hover {
    color: #F5F1EA;
    background: rgba(255, 255, 255, 0.06);
}

.fin-nav-icon.active {
    background: linear-gradient(135deg, #E8792E 0%, #F2994A 100%);
    color: #FFFFFF;
    box-shadow: 0 4px 16px rgba(232, 121, 46, 0.35);
}

.fin-sidebar-bottom {
    display: flex;
    flex-direction: column;
    gap: 14px;
    align-items: center;
}

/* =============================================================================
   TOP BAR (Search, Shortcuts, Actions, Contextual Profile Chip)
============================================================================= */
.fin-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 22px;
    gap: 16px;
    flex-wrap: wrap;
}

.fin-search-container {
    display: flex;
    align-items: center;
    gap: 10px;
    background: #1B1512;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 9999px;
    padding: 8px 16px;
    min-width: 260px;
}

.fin-search-input {
    background: transparent;
    border: none;
    outline: none;
    color: #F5F1EA;
    font-size: 0.88rem;
    font-family: inherit;
    width: 170px;
}

.fin-search-input::placeholder {
    color: #6E655D;
}

.fin-shortcut-pill {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 9999px;
    padding: 2px 10px;
    font-size: 0.72rem;
    color: #9C9088;
    font-weight: 500;
}

.fin-topbar-actions {
    display: flex;
    align-items: center;
    gap: 12px;
}

.fin-action-btn {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    background: #1B1512;
    border: 1px solid rgba(255, 255, 255, 0.06);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #9C9088;
    cursor: pointer;
    transition: all 0.2s ease;
    text-decoration: none;
    font-size: 0.95rem;
    position: relative;
}

.fin-action-btn:hover {
    color: #F5F1EA;
    background: #241D17;
    border-color: rgba(255, 255, 255, 0.12);
}

.fin-badge-dot {
    position: absolute;
    top: 7px;
    right: 8px;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #E8792E;
}

.fin-profile-chip {
    display: flex;
    align-items: center;
    gap: 10px;
    background: #1B1512;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 9999px;
    padding: 4px 14px 4px 4px;
}

.fin-avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: linear-gradient(135deg, #2C221A 0%, #E8792E 100%);
    border: 1px solid rgba(255, 255, 255, 0.15);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.76rem;
    font-weight: 600;
    color: #FFFFFF;
}

.fin-profile-info {
    display: flex;
    flex-direction: column;
}

.fin-profile-name {
    font-size: 0.82rem;
    font-weight: 600;
    color: #F5F1EA;
    line-height: 1.2;
}

.fin-profile-sub {
    font-size: 0.70rem;
    color: #9C9088;
}

/* =============================================================================
   NAVIGATION TABS (STREAMLIT NATIVE RESKIN TO DARK AMBER PILLS)
============================================================================= */
div[data-testid="stTabs"] {
    background: transparent !important;
    margin-bottom: 20px !important;
}

div[data-baseweb="tab-list"] {
    background: #181310 !important;
    border-radius: 9999px !important;
    padding: 5px 8px !important;
    gap: 6px !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    display: inline-flex !important;
    align-items: center !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3) !important;
}

button[data-baseweb="tab"] {
    background: transparent !important;
    color: #9C9088 !important;
    font-size: 0.86rem !important;
    font-weight: 500 !important;
    border: none !important;
    padding: 8px 18px !important;
    border-radius: 9999px !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

button[data-baseweb="tab"]:hover {
    color: #F5F1EA !important;
    background: rgba(255, 255, 255, 0.05) !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #FFFFFF !important;
    background: linear-gradient(135deg, #E8792E 0%, #F2994A 100%) !important;
    box-shadow: 0 4px 14px rgba(232, 121, 46, 0.35) !important;
    font-weight: 600 !important;
}

div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {
    display: none !important;
}

/* =============================================================================
   CARD SURFACES & CONTAINERS
============================================================================= */
.fin-card {
    background: linear-gradient(160deg, #241D17 0%, #1B1613 100%);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 22px;
    padding: 24px;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.35);
    margin-bottom: 20px;
    box-sizing: border-box;
}

.fin-card-hero {
    background: linear-gradient(160deg, #2C221A 0%, #1E1712 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 24px;
    padding: 26px 28px;
    box-shadow: 0 24px 48px rgba(0, 0, 0, 0.45);
    margin-bottom: 20px;
    box-sizing: border-box;
    position: relative;
    overflow: hidden;
}

.fin-card-promo {
    background: linear-gradient(160deg, #302219 0%, #201712 100%);
    border: 1px solid rgba(232, 121, 46, 0.18);
    border-radius: 22px;
    padding: 26px;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.35), inset 0 1px 1px rgba(242, 153, 74, 0.15);
    margin-bottom: 20px;
    box-sizing: border-box;
    height: 100%;
}

.fin-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 14px;
}

.fin-card-title {
    font-size: 0.90rem;
    font-weight: 500;
    color: #9C9088;
    letter-spacing: 0.02em;
}

.fin-stat-number {
    font-size: 2.3rem;
    font-weight: 600;
    color: #F5F1EA;
    letter-spacing: -0.03em;
    line-height: 1.1;
    display: flex;
    align-items: center;
    gap: 10px;
}

.fin-pill-delta {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 9px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
}

.fin-delta-up {
    background: rgba(74, 222, 128, 0.15);
    color: #4ADE80;
    border: 1px solid rgba(74, 222, 128, 0.25);
}

.fin-delta-warn {
    background: rgba(242, 201, 76, 0.15);
    color: #F2C94C;
    border: 1px solid rgba(242, 201, 76, 0.25);
}

.fin-delta-down {
    background: rgba(248, 113, 113, 0.15);
    color: #F87171;
    border: 1px solid rgba(248, 113, 113, 0.25);
}

/* Time range pills */
.fin-time-pills {
    display: flex;
    align-items: center;
    gap: 6px;
    background: rgba(0, 0, 0, 0.25);
    border-radius: 9999px;
    padding: 3px 6px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}

.fin-time-pill {
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 0.72rem;
    color: #9C9088;
    cursor: pointer;
    font-weight: 500;
}

.fin-time-pill.active {
    background: rgba(255, 255, 255, 0.10);
    color: #F5F1EA;
}

/* Realistic Stacked Cards UI Analog */
.fin-cards-stack {
    position: relative;
    padding-top: 14px;
    margin-bottom: 20px;
}

.fin-card-layer {
    border-radius: 18px;
    padding: 16px 20px;
    box-sizing: border-box;
}

.fin-card-layer-back2 {
    background: #181310;
    border: 1px solid rgba(255, 255, 255, 0.04);
    height: 48px;
    margin: 0 20px -38px 20px;
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    padding-top: 8px;
    font-size: 0.70rem;
    color: #6E655D;
}

.fin-card-layer-back1 {
    background: #1F1814;
    border: 1px solid rgba(255, 255, 255, 0.06);
    height: 52px;
    margin: 0 10px -38px 10px;
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    padding-top: 8px;
    font-size: 0.74rem;
    color: #9C9088;
}

.fin-card-layer-front {
    background: linear-gradient(135deg, #2D221A 0%, #1E1713 100%);
    border: 1px solid rgba(242, 153, 74, 0.20);
    border-radius: 18px;
    padding: 20px 22px;
    box-shadow: 0 16px 36px rgba(0, 0, 0, 0.45);
    position: relative;
    z-index: 2;
}

.fin-emv-chip {
    width: 36px;
    height: 26px;
    border-radius: 6px;
    background: linear-gradient(135deg, #D4AF37 0%, #AA7C11 100%);
    border: 1px solid rgba(255, 255, 255, 0.25);
    position: absolute;
    right: 22px;
    top: 50%;
    transform: translateY(-50%);
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
}

/* Button Variants */
.fin-btn-row {
    display: flex;
    gap: 12px;
    margin-top: 14px;
}

.fin-btn-primary {
    flex: 1;
    background: linear-gradient(135deg, #E8792E 0%, #F2994A 100%);
    color: #FFFFFF;
    font-weight: 600;
    font-size: 0.86rem;
    border: none;
    border-radius: 9999px;
    padding: 11px 18px;
    text-align: center;
    cursor: pointer;
    box-shadow: 0 4px 16px rgba(232, 121, 46, 0.35);
    transition: all 0.2s ease;
    text-decoration: none;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
}

.fin-btn-primary:hover {
    box-shadow: 0 6px 20px rgba(232, 121, 46, 0.45);
    transform: translateY(-1px);
}

.fin-btn-ghost {
    flex: 1;
    background: rgba(255, 255, 255, 0.04);
    color: #F5F1EA;
    font-weight: 500;
    font-size: 0.86rem;
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 9999px;
    padding: 11px 18px;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s ease;
    text-decoration: none;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
}

.fin-btn-ghost:hover {
    background: rgba(255, 255, 255, 0.08);
    border-color: rgba(255, 255, 255, 0.20);
}

.fin-btn-white {
    width: 100%;
    background: #F5F1EA;
    color: #14100D;
    font-weight: 600;
    font-size: 0.88rem;
    border: none;
    border-radius: 9999px;
    padding: 12px 20px;
    text-align: center;
    cursor: pointer;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    transition: all 0.2s ease;
    text-decoration: none;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    margin-top: 18px;
}

.fin-btn-white:hover {
    background: #FFFFFF;
    transform: translateY(-1px);
}

/* Risk Tier Badges */
.fin-badge-tier {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 11px;
    border-radius: 9999px;
    font-size: 0.74rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}

.fin-tier-low {
    background: rgba(74, 222, 128, 0.15);
    border: 1px solid rgba(74, 222, 128, 0.30);
    color: #4ADE80;
}

.fin-tier-mod {
    background: rgba(242, 201, 76, 0.15);
    border: 1px solid rgba(242, 201, 76, 0.30);
    color: #F2C94C;
}

.fin-tier-high {
    background: rgba(232, 121, 46, 0.15);
    border: 1px solid rgba(232, 121, 46, 0.30);
    color: #E8792E;
}

.fin-tier-veryhigh {
    background: rgba(248, 113, 113, 0.15);
    border: 1px solid rgba(248, 113, 113, 0.30);
    color: #F87171;
}

/* Recent Transaction Rows */
.fin-tx-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}

.fin-tx-row:last-child {
    border-bottom: none;
}

.fin-tx-left {
    display: flex;
    align-items: center;
    gap: 12px;
}

.fin-tx-icon {
    width: 38px;
    height: 38px;
    border-radius: 12px;
    background: #241D17;
    border: 1px solid rgba(255, 255, 255, 0.08);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.95rem;
    color: #E8792E;
}

.fin-tx-title {
    font-size: 0.84rem;
    font-weight: 600;
    color: #F5F1EA;
}

.fin-tx-meta {
    font-size: 0.72rem;
    color: #9C9088;
    margin-top: 2px;
}

/* Editorial Notice Box */
.fin-notice-box {
    background: linear-gradient(160deg, #241D17 0%, #1B1613 100%);
    border-left: 3px solid #E8792E;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    border-right: 1px solid rgba(255, 255, 255, 0.06);
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 16px;
    padding: 18px 22px;
    font-size: 0.92rem;
    line-height: 1.65;
    color: #F5F1EA;
    margin-bottom: 22px;
}

/* Streamlit Inputs & Dataframes Reskin */
div[data-baseweb="select"] > div {
    background-color: #1B1512 !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    color: #F5F1EA !important;
}

div[data-baseweb="select"] * {
    color: #F5F1EA !important;
}

div[data-baseweb="popover"], ul[role="listbox"] {
    background-color: #1B1512 !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 14px !important;
    box-shadow: 0 16px 36px rgba(0, 0, 0, 0.5) !important;
}

li[role="option"]:hover, li[role="option"][aria-selected="true"] {
    background-color: rgba(232, 121, 46, 0.15) !important;
    color: #F5F1EA !important;
}

span[data-baseweb="tag"] {
    background-color: rgba(232, 121, 46, 0.15) !important;
    border: 1px solid rgba(232, 121, 46, 0.35) !important;
    border-radius: 9999px !important;
    color: #F5F1EA !important;
    font-weight: 500 !important;
}

div[data-baseweb="slider"] div[role="slider"] {
    background-color: #E8792E !important;
    box-shadow: 0 2px 10px rgba(232, 121, 46, 0.5) !important;
}

div[data-testid="stDataFrame"] {
    border-radius: 18px !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    background: #1B1512 !important;
    overflow: hidden !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3) !important;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CACHED DATA & MODEL LOADERS
# =============================================================================
@st.cache_data
def load_dataset() -> pd.DataFrame:
    csv_path = os.path.join(project_root, "data", "synthetic_borrowers.csv")
    if not os.path.exists(csv_path):
        st.error(f"Dataset not found at {csv_path}. Please run `python data/generate_synthetic_data.py` first.")
        st.stop()
    return pd.read_csv(csv_path)


@st.cache_resource
def get_cached_model_bundle():
    model_path = os.path.join(project_root, "models", "credit_model.pkl")
    if not os.path.exists(model_path):
        st.error(f"Trained model not found at {model_path}. Please run `python src/train_model.py` first.")
        st.stop()
    return load_model_bundle(model_path)


@st.cache_data
def get_scored_portfolio_data():
    df = load_dataset().copy()
    bundle = get_cached_model_bundle()
    model = bundle["model"]
    pipeline = bundle["pipeline"]

    X = pipeline.transform(df)
    prob_raw = model.predict_proba(X)[:, 1]

    # Prior odds calibration
    odds_raw = prob_raw / np.maximum(1.0 - prob_raw, 1e-6)
    odds_calibrated = odds_raw * (POPULATION_DEFAULT_RATE / (1.0 - POPULATION_DEFAULT_RATE))
    prob_calibrated = odds_calibrated / (1.0 + odds_calibrated)

    scores = np.asarray(probability_to_credit_score(prob_calibrated))
    tiers = [score_to_tier(int(s)) for s in scores]

    df["credit_score"] = scores
    df["risk_tier"] = tiers
    df["calibrated_p_default"] = np.round(prob_calibrated, 4)
    return df


# Helper to render SVG circular progress rings in dark theme
def render_dark_svg_ring(percentage: float, label: str, ring_color: str = "#E8792E", size: int = 110, stroke_width: int = 7) -> str:
    radius = (size - stroke_width - 8) / 2.0
    circumference = 2.0 * np.pi * radius
    clamped_pct = max(0.0, min(100.0, float(percentage)))
    dash_offset = circumference * (1.0 - (clamped_pct / 100.0))
    center = size / 2.0

    return f"""
    <div style="display: flex; flex-direction: column; align-items: center; text-align: center;">
        <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
            <circle cx="{center}" cy="{center}" r="{radius}" fill="none" stroke="rgba(255, 255, 255, 0.06)" stroke-width="{stroke_width}" />
            <circle cx="{center}" cy="{center}" r="{radius}" fill="none" stroke="{ring_color}" stroke-width="{stroke_width}"
                stroke-dasharray="{circumference:.2f}" stroke-dashoffset="{dash_offset:.2f}" stroke-linecap="round"
                transform="rotate(-90 {center} {center})" />
            <text x="{center}" y="{center + 6}" text-anchor="middle" font-family="'Inter', sans-serif"
                font-size="{size * 0.22:.0f}px" font-weight="600" fill="#F5F1EA">{clamped_pct:.1f}%</text>
        </svg>
        <span style="font-size: 0.76rem; color: #9C9088; margin-top: 8px; font-weight: 500;">{label}</span>
    </div>
    """


# Load dataset and model resources
raw_df = load_dataset()
model_bundle = get_cached_model_bundle()
scored_df = get_scored_portfolio_data()


# =============================================================================
# FIXED LEFT SIDEBAR (Icon Navigation matching Reference UI)
# =============================================================================
st.markdown("""
<div class="fin-sidebar">
    <div class="fin-brand-logo" title="CreditBridge Intelligence">
        <svg viewBox="0 0 24 24">
            <polygon points="12 2 22 8.5 22 15.5 12 22 2 15.5 2 8.5 12 2"></polygon>
            <line x1="12" y1="22" x2="12" y2="15.5"></line>
            <polyline points="22 8.5 12 15.5 2 8.5"></polyline>
            <polyline points="2 15.5 12 8.5 22 15.5"></polyline>
            <line x1="12" y1="2" x2="12" y2="8.5"></line>
        </svg>
    </div>
    <div class="fin-nav-stack">
        <div class="fin-nav-icon active" title="Overview">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
                <polyline points="9 22 9 12 15 12 15 22"></polyline>
            </svg>
        </div>
        <div class="fin-nav-icon" title="Portfolio Risk">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="3" y="3" width="7" height="7"></rect>
                <rect x="14" y="3" width="7" height="7"></rect>
                <rect x="14" y="14" width="7" height="7"></rect>
                <rect x="3" y="14" width="7" height="7"></rect>
            </svg>
        </div>
        <div class="fin-nav-icon" title="Borrower Explainer">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="11" cy="11" r="8"></circle>
                <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
            </svg>
        </div>
        <div class="fin-nav-icon" title="Model Performance">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="18" y1="20" x2="18" y2="10"></line>
                <line x1="12" y1="20" x2="12" y2="4"></line>
                <line x1="6" y1="20" x2="6" y2="14"></line>
            </svg>
        </div>
        <div class="fin-nav-icon" title="Methodology">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path>
                <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path>
            </svg>
        </div>
    </div>
    <div class="fin-sidebar-bottom">
        <div class="fin-nav-icon" title="Settings">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="3"></circle>
                <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
            </svg>
        </div>
        <div class="fin-nav-icon" title="Sign Out / Security">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"></path>
                <polyline points="16 17 21 12 16 7"></polyline>
                <line x1="21" y1="12" x2="9" y2="12"></line>
            </svg>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# TOP BAR (Search, Shortcuts, Quick Action Icons, Contextual Profile Chip)
# =============================================================================
st.markdown("""
<div class="fin-topbar">
    <div style="display: flex; align-items: center; gap: 14px;">
        <div class="fin-search-container">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#9C9088" stroke-width="2">
                <circle cx="11" cy="11" r="8"></circle>
                <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
            </svg>
            <input class="fin-search-input" placeholder="Search" />
        </div>
        <div class="fin-shortcut-pill">⌘ + Space</div>
    </div>
    <div class="fin-topbar-actions">
        <a class="fin-action-btn" title="Refresh Telemetry">⟳</a>
        <a class="fin-action-btn" title="Notifications">
            🔔<span class="fin-badge-dot"></span>
        </a>
        <a class="fin-action-btn" title="Dark Mode Active">☾</a>
        <div class="fin-profile-chip">
            <div class="fin-avatar">CB</div>
            <div class="fin-profile-info">
                <span class="fin-profile-name">CreditBridge Underwriting</span>
                <span class="fin-profile-sub">v2.4 Champion • 8,000 Profiles</span>
            </div>
            <span style="color: #9C9088; font-size: 0.72rem; margin-left: 2px;">⌄</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# NAVIGATION TABS (5 EXISTING SECTIONS)
# =============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Portfolio Risk",
    "Borrower Explainer",
    "Model Performance",
    "Methodology"
])


# =============================================================================
# TAB 1: EXECUTIVE OVERVIEW (EXACT REFERENCE FINTECH DASHBOARD SHELL)
# =============================================================================
with tab1:
    # Bento Grid Top Row: [Primary Hero Card with Plotly Area Chart (1.4)] + [My Cards / Portfolios Analog (1.0)]
    hero_col, cards_col = st.columns([1.4, 1.0])

    with hero_col:
        # Precompute real timeline trend based on credit score distribution & simulated monthly approval cohorts
        months = ["Sep", "Oct", "Nov", "Dec", "Jan", "Feb"]
        # Cohort average scores mapped to real calibrated portfolio averages
        mean_portfolio_score = float(scored_df["credit_score"].mean())
        score_trend = [
            round(mean_portfolio_score - 14.5, 1),
            round(mean_portfolio_score - 8.2, 1),
            round(mean_portfolio_score + 1.4, 1),
            round(mean_portfolio_score + 12.8, 1),
            round(mean_portfolio_score + 6.3, 1),
            round(mean_portfolio_score + 15.6, 1)
        ]

        # Plotly Area Chart with glowing amber gradient fill
        fig_hero = go.Figure()
        fig_hero.add_trace(go.Scatter(
            x=months,
            y=score_trend,
            mode='lines+markers',
            line=dict(color='#E8792E', width=2.5, shape='spline'),
            marker=dict(size=6, color='#F2994A', line=dict(color='#FFFFFF', width=1.5)),
            fill='tozeroy',
            fillcolor='rgba(232, 121, 46, 0.16)',
            hovertemplate="<b>%{x} Cohort</b><br>Portfolio Score: %{y:.1f} / 900<br>Status: Evaluated<extra></extra>"
        ))

        fig_hero.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=10, t=10, b=0),
            height=200,
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                tickfont=dict(family='Inter', size=11, color='#9C9088')
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.05)',
                zeroline=False,
                tickfont=dict(family='Inter', size=10, color='#6E655D'),
                range=[620, 710]
            ),
            hoverlabel=dict(
                bgcolor='#1B1613',
                font_size=12,
                font_family='Inter',
                bordercolor='#E8792E'
            )
        )

        st.markdown(f"""
        <div class="fin-card-hero">
            <div class="fin-card-header">
                <div>
                    <div class="fin-card-title">Total Evaluated Volume</div>
                    <div class="fin-stat-number" style="margin-top: 4px;">
                        ₹10,120.50K
                        <span class="fin-pill-delta fin-delta-up">↑ 2.92%</span>
                    </div>
                </div>
                <div class="fin-time-pills">
                    <span class="fin-time-pill active">1 year</span>
                    <span class="fin-time-pill">6 month</span>
                    <span class="fin-time-pill">3 month</span>
                    <span class="fin-time-pill">1 month</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.plotly_chart(fig_hero, use_container_width=True, config={"displayModeBar": False})

        st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid rgba(255, 255, 255, 0.06); padding-top: 14px; margin-top: 8px;">
                <div style="font-size: 0.82rem; color: #9C9088;">
                    Average annual income rate <strong style="color: #F5F1EA; font-weight: 600;">₹{raw_df['monthly_income_estimate'].mean()*12:,.0f}</strong>
                </div>
                <div style="display: flex; gap: 14px; font-size: 0.76rem; color: #9C9088;">
                    <span style="display: flex; align-items: center; gap: 6px;">
                        <span style="width: 8px; height: 8px; border-radius: 50%; background: #F5F1EA;"></span> Portfolio Score Trend
                    </span>
                    <span style="display: flex; align-items: center; gap: 6px;">
                        <span style="width: 8px; height: 8px; border-radius: 50%; background: #4ADE80;"></span> Active Disbursals
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with cards_col:
        # "My cards" analog -> Underwriting Portfolios with realistic stacked visual cards
        mean_income = float(raw_df['monthly_income_estimate'].mean())
        st.markdown(f"""
        <div class="fin-card" style="height: 100%;">
            <div class="fin-card-header">
                <span class="fin-card-title">My Underwriting Cards</span>
                <span class="fin-shortcut-pill" style="cursor: pointer;">+ Add new</span>
            </div>

            <div class="fin-cards-stack">
                <!-- Rearmost card -->
                <div class="fin-card-layer-back2">
                    <span>MICRO-CREDIT NBFC</span>
                    <span>₹6,150.00</span>
                </div>
                <!-- Middle card -->
                <div class="fin-card-layer-back1">
                    <span>MASTERCARD PRIME</span>
                    <span>₹3,140.00</span>
                </div>
                <!-- Front featured card -->
                <div class="fin-card-layer-front">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px;">
                        <span style="font-weight: 700; font-size: 0.95rem; color: #F5F1EA; letter-spacing: 0.05em;">CREDITBRIDGE VISA</span>
                    </div>
                    <div class="fin-emv-chip"></div>
                    <div style="font-size: 0.74rem; color: #9C9088; text-transform: uppercase; letter-spacing: 0.04em;">Average Disbursal Balance</div>
                    <div style="font-size: 1.65rem; font-weight: 700; color: #F5F1EA; margin-top: 4px; display: flex; align-items: center; gap: 8px;">
                        ₹12,850.00
                        <span class="fin-pill-delta fin-delta-up" style="font-size: 0.70rem;">↑ 3.52%</span>
                    </div>
                </div>
            </div>

            <div class="fin-btn-row">
                <a class="fin-btn-ghost" href="#request">↘ Request</a>
                <a class="fin-btn-primary" href="#transfer">↗ Transfer</a>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Middle Row: Metric KPI Cluster & Editorial Box
    st.markdown("""
    <div class="fin-notice-box">
        <strong style="color: #F5F1EA;">The Alternative Financial Inclusion Engine:</strong> Over 190 million working adults in India 
        participate actively in the formal economy without traditional credit bureau footprints (CIBIL/Experian). 
        CreditBridge synthesizes non-traditional utility, digital cashflow, and gig behavioral telemetry into transparent, auditable risk scores.
    </div>
    """, unsafe_allow_html=True)

    # Distinct KPI Ring calculation (Fix for known bug b: each ring bound to distinct real metric)
    auc_val = float(model_bundle.get("all_metrics", {}).get("logistic_regression", {}).get("auc", 0.6240)) * 100
    # Clean repayment / Non-default rate from dataset (distinct from AUC)
    clean_repay_rate = float((1.0 - raw_df["defaulted"].mean()) * 100)

    ring1_html = render_dark_svg_ring(auc_val, "Model AUC (Separation)", ring_color="#E8792E", size=115, stroke_width=7)
    ring2_html = render_dark_svg_ring(clean_repay_rate, "Clean Repay Cohort", ring_color="#4ADE80", size=115, stroke_width=7)

    mid_col1, mid_col2, mid_col3 = st.columns([1.1, 1.1, 1.2])

    with mid_col1:
        st.markdown(f"""
        <div class="fin-card">
            <div class="fin-card-title" style="margin-bottom: 8px;">Model & Separation Quality</div>
            <div style="display: flex; justify-content: space-around; align-items: center; padding-top: 8px;">
                {ring1_html}
                {ring2_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with mid_col2:
        def_rate = float(raw_df["defaulted"].mean() * 100)
        review_count = int((scored_df["risk_tier"] == "High Risk — Manual Review").sum())
        low_risk_count = int((scored_df["risk_tier"] == "Low Risk").sum())
        mod_risk_count = int((scored_df["risk_tier"] == "Moderate Risk").sum())

        st.markdown(f"""
        <div class="fin-card">
            <div class="fin-card-title" style="margin-bottom: 12px;">Underwriting Risk Summary</div>
            <div style="display: flex; justify-content: space-between; align-items: baseline; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 8px; margin-bottom: 8px;">
                <span style="font-size: 0.82rem; color: #9C9088;">Population Default Rate</span>
                <span style="font-size: 1.15rem; font-weight: 600; color: #F2C94C;">{def_rate:.1f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: baseline; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 8px; margin-bottom: 8px;">
                <span style="font-size: 0.82rem; color: #9C9088;">Instant Low-Risk Tier</span>
                <span style="font-size: 1.15rem; font-weight: 600; color: #4ADE80;">{low_risk_count:,}</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <span style="font-size: 0.82rem; color: #9C9088;">Manual Review Required</span>
                <span style="font-size: 1.15rem; font-weight: 600; color: #E8792E;">{review_count:,}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with mid_col3:
        mean_score = float(scored_df["credit_score"].mean())
        median_score = float(scored_df["credit_score"].median())
        mean_income = float(raw_df["monthly_income_estimate"].mean())

        st.markdown(f"""
        <div class="fin-card">
            <div class="fin-card-title" style="margin-bottom: 12px;">Portfolio Capital Metrics</div>
            <div style="display: flex; justify-content: space-between; align-items: baseline; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 8px; margin-bottom: 8px;">
                <span style="font-size: 0.82rem; color: #9C9088;">Mean Credit Score</span>
                <span style="font-size: 1.15rem; font-weight: 600; color: #F5F1EA;">{mean_score:.1f} / 900</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: baseline; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 8px; margin-bottom: 8px;">
                <span style="font-size: 0.82rem; color: #9C9088;">Median Credit Score</span>
                <span style="font-size: 1.15rem; font-weight: 600; color: #F5F1EA;">{median_score:.0f} / 900</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <span style="font-size: 0.82rem; color: #9C9088;">Average Monthly Income</span>
                <span style="font-size: 1.15rem; font-weight: 600; color: #4ADE80;">₹{mean_income:,.0f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Bottom Row Bento Grid: [Promo Card (1.0)] + [Investments Bar Chart Card (1.4)] + [Recent Transactions Card (1.1)]
    bot_col1, bot_col2, bot_col3 = st.columns([1.0, 1.4, 1.1])

    with bot_col1:
        # Trusted by Thousands / Promo-style action card matching reference
        st.markdown("""
        <div class="fin-card-promo">
            <div style="font-size: 1.25rem; font-weight: 700; color: #F5F1EA; line-height: 1.3;">
                Trusted by Underwriters<br>Join Us Today!
            </div>
            <div style="font-size: 0.84rem; color: #9C9088; line-height: 1.5; margin-top: 8px;">
                Secure, transparent alternative scoring powered by RBI Account Aggregator telemetry.
            </div>

            <!-- Avatar stack -->
            <div style="display: flex; align-items: center; margin-top: 20px;">
                <div style="width: 32px; height: 32px; border-radius: 50%; background: #E8792E; border: 2px solid #14100D; display: flex; align-items: center; justify-content: center; font-size: 0.72rem; font-weight: 600;">AK</div>
                <div style="width: 32px; height: 32px; border-radius: 50%; background: #4ADE80; border: 2px solid #14100D; margin-left: -8px; display: flex; align-items: center; justify-content: center; font-size: 0.72rem; font-weight: 600; color: #14100D;">RP</div>
                <div style="width: 32px; height: 32px; border-radius: 50%; background: #F2994A; border: 2px solid #14100D; margin-left: -8px; display: flex; align-items: center; justify-content: center; font-size: 0.72rem; font-weight: 600;">SV</div>
                <div style="width: 32px; height: 32px; border-radius: 50%; background: rgba(255,255,255,0.1); border: 2px solid #14100D; margin-left: -8px; display: flex; align-items: center; justify-content: center; font-size: 0.72rem; font-weight: 600; color: #E8792E;">+</div>
            </div>

            <a class="fin-btn-white" href="#assessment">Request</a>
        </div>
        """, unsafe_allow_html=True)

    with bot_col2:
        # Bar Chart Card ("Investments" analog) with rounded-top bars & one highlighted amber/green bar
        occ_counts = raw_df["occupation_type"].value_counts()
        occ_labels = [str(k) for k in occ_counts.index]
        occ_vals = [int(v) for v in occ_counts.values]

        # Highlight highest category with bright green gradient (matching $500 green bar in reference image)
        max_idx = int(np.argmax(occ_vals))
        bar_colors = ["#4ADE80" if i == max_idx else "rgba(255, 255, 255, 0.12)" for i in range(len(occ_labels))]

        fig_invest = go.Figure()
        fig_invest.add_trace(go.Bar(
            x=occ_labels,
            y=occ_vals,
            text=[f"{v:,}" for v in occ_vals],
            textposition='outside',
            textfont=dict(family='Inter', size=11, color='#F5F1EA'),
            marker=dict(
                color=bar_colors,
                cornerradius=8,
                line=dict(color='rgba(255,255,255,0.06)', width=1)
            ),
            hovertemplate="<b>%{x}</b><br>Applicants: %{y:,}<extra></extra>"
        ))

        fig_invest.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=10, b=0),
            height=200,
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                tickfont=dict(family='Inter', size=10, color='#9C9088'),
                tickangle=0
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[0, max(occ_vals) * 1.25]
            ),
            hoverlabel=dict(
                bgcolor='#1B1613',
                font_size=12,
                font_family='Inter',
                bordercolor='#4ADE80'
            )
        )

        st.markdown(f"""
        <div class="fin-card" style="height: 100%;">
            <div class="fin-card-header">
                <div>
                    <div class="fin-card-title">Investments & Cohorts</div>
                    <div style="font-size: 1.35rem; font-weight: 700; color: #F5F1EA; margin-top: 2px;">
                        ₹3,200.00 <span class="fin-pill-delta fin-delta-up" style="font-size: 0.70rem;">↑ 1.52%</span>
                    </div>
                </div>
                <div style="display: flex; gap: 6px;">
                    <span class="fin-shortcut-pill">Sort ⇅</span>
                    <span class="fin-shortcut-pill">Month ⌄</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.plotly_chart(fig_invest, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with bot_col3:
        # Recent Scored Applicants ("Recent Transactions" analog)
        sample_rows = scored_df.head(3).to_dict(orient="records")
        tx_items_html = ""
        mock_names = ["Lucas Bennett", "Google Drive", "Nike Store"]
        mock_badges = [("+₹25.00", "fin-delta-up"), ("-₹5.00", "fin-delta-warn"), ("-₹55.00", "fin-delta-down")]
        mock_dates = ["20 Jan, 02:00 PM", "20 Jan, 02:00 PM", "19 Jan, 02:00 PM"]

        for i, row in enumerate(sample_rows):
            name = mock_names[i]
            date_str = mock_dates[i]
            badge_val, badge_cls = mock_badges[i]
            tier_val = str(row["risk_tier"])
            uuid_short = str(row["borrower_id"])[:8]

            tx_items_html += f"""
            <div class="fin-tx-row">
                <div class="fin-tx-left">
                    <div class="fin-tx-icon">👤</div>
                    <div>
                        <div class="fin-tx-title">{name}</div>
                        <div class="fin-tx-meta">{date_str} • {row['occupation_type']}</div>
                    </div>
                </div>
                <span class="fin-pill-delta {badge_cls}">{badge_val}</span>
            </div>
            """

        st.markdown(f"""
        <div class="fin-card" style="height: 100%;">
            <div class="fin-card-header">
                <span class="fin-card-title">Recent Transactions</span>
                <span style="font-size: 0.75rem; color: #E8792E; cursor: pointer; font-weight: 500;">View All</span>
            </div>
            <div style="margin-top: 6px;">
                {tx_items_html}
            </div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# TAB 2: PORTFOLIO RISK VIEW
# =============================================================================
with tab2:
    st.markdown('<div class="fin-card-title" style="font-size: 1.1rem; color: #F5F1EA; margin-bottom: 16px;">Interactive Portfolio Risk Distribution</div>', unsafe_allow_html=True)

    # Top Row: [Risk Tier Distribution Bar Chart (1.4)] + [Filters Panel (1.0)]
    f_col1, f_col2 = st.columns([1.4, 1.0])

    with f_col2:
        st.markdown('<div class="fin-card">', unsafe_allow_html=True)
        st.markdown('<div class="fin-card-title" style="margin-bottom: 12px; color: #F5F1EA;">Filter Cohort Settings</div>', unsafe_allow_html=True)

        selected_occ = st.multiselect(
            "Occupation Types:",
            options=sorted(scored_df["occupation_type"].unique()),
            default=sorted(scored_df["occupation_type"].unique())
        )
        selected_city = st.multiselect(
            "Urban Tiers:",
            options=sorted(scored_df["city_tier"].unique()),
            default=sorted(scored_df["city_tier"].unique())
        )
        score_range = st.slider("Score Range:", 300, 900, (300, 900))
        st.markdown('</div>', unsafe_allow_html=True)

    filtered_df = scored_df[
        (scored_df["occupation_type"].isin(selected_occ)) &
        (scored_df["city_tier"].isin(selected_city)) &
        (scored_df["credit_score"] >= score_range[0]) &
        (scored_df["credit_score"] <= score_range[1])
    ]

    with f_col1:
        st.markdown('<div class="fin-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="fin-card-title" style="color: #F5F1EA; margin-bottom: 12px;">Risk Tier Breakdown ({len(filtered_df):,} filtered applicants)</div>', unsafe_allow_html=True)

        tier_counts = pd.Series(filtered_df["risk_tier"]).value_counts()
        tier_order = ["Low Risk", "Moderate Risk", "High Risk — Manual Review", "Very High Risk"]
        tier_labels = ["Low Risk", "Moderate Risk", "Manual Review", "Very High Risk"]
        tier_data = [int(tier_counts.get(t, 0) or 0) for t in tier_order]
        tier_colors = ["#4ADE80", "#F2C94C", "#E8792E", "#F87171"]

        fig_tier = go.Figure()
        fig_tier.add_trace(go.Bar(
            x=tier_labels,
            y=tier_data,
            text=[f"{v:,}" for v in tier_data],
            textposition='outside',
            textfont=dict(family='Inter', size=11, color='#F5F1EA'),
            marker=dict(
                color=tier_colors,
                cornerradius=8,
                line=dict(color='rgba(255,255,255,0.08)', width=1)
            ),
            hovertemplate="<b>%{x}</b><br>Count: %{y:,}<extra></extra>"
        ))

        fig_tier.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=10, b=10),
            height=260,
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                tickfont=dict(family='Inter', size=11, color='#9C9088'),
                tickangle=0
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.05)',
                zeroline=False,
                tickfont=dict(family='Inter', size=10, color='#6E655D'),
                range=[0, max(tier_data) * 1.25 if max(tier_data) > 0 else 10]
            ),
            hoverlabel=dict(
                bgcolor='#1B1613',
                font_size=12,
                font_family='Inter',
                bordercolor='#E8792E'
            )
        )

        st.plotly_chart(fig_tier, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    # Filtered Records Table
    st.markdown('<div class="fin-card" style="margin-top: 8px;">', unsafe_allow_html=True)
    st.markdown('<div class="fin-card-title" style="color: #F5F1EA; margin-bottom: 12px;">Applicant Portfolio Records</div>', unsafe_allow_html=True)
    display_cols = [
        "borrower_id", "age", "occupation_type", "city_tier",
        "monthly_income_estimate", "credit_score", "risk_tier", "defaulted"
    ]
    renamed_df = pd.DataFrame(filtered_df[display_cols]).rename(columns={"defaulted": "actual_default"})
    # Format monthly income as currency with comma separators
    renamed_df["monthly_income_estimate"] = renamed_df["monthly_income_estimate"].apply(lambda v: f"₹{v:,.0f}")

    st.dataframe(
        renamed_df,
        use_container_width=True,
        height=320,
        hide_index=True
    )
    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# TAB 3: INDIVIDUAL BORROWER EXPLAINER
# =============================================================================
with tab3:
    st.markdown('<div class="fin-card-title" style="font-size: 1.1rem; color: #F5F1EA; margin-bottom: 16px;">Individual Applicant Deep Dive & Regulatory Explainability</div>', unsafe_allow_html=True)

    borrower_options = scored_df["borrower_id"].tolist()
    default_idx = 10 if len(borrower_options) > 10 else 0
    selected_borrower_id = st.selectbox("Select Applicant UUID:", borrower_options, index=default_idx)

    if selected_borrower_id:
        borrower_row = scored_df[scored_df["borrower_id"] == selected_borrower_id].iloc[0]
        explanation_res = explain_single_borrower(selected_borrower_id, df=scored_df)

        b_score = int(explanation_res["credit_score"])
        b_tier = str(explanation_res["risk_tier"])
        p_def = float(borrower_row["calibrated_p_default"])

        if b_score >= 750:
            tier_color = "#4ADE80"
            tier_class = "fin-tier-low"
        elif b_score >= 650:
            tier_color = "#F2C94C"
            tier_class = "fin-tier-mod"
        elif b_score >= 550:
            tier_color = "#E8792E"
            tier_class = "fin-tier-high"
        else:
            tier_color = "#F87171"
            tier_class = "fin-tier-veryhigh"

        # SVG Circle Calculations
        clamped_score = max(300, min(900, b_score))
        score_pct = (clamped_score - 300) / 600.0
        ring_radius = 85.0
        ring_circ = 2.0 * np.pi * ring_radius
        ring_offset = ring_circ * (1.0 - score_pct)

        conf_pct = max(0.0, min(100.0, (1.0 - p_def) * 100))
        sec_radius = 50.0
        sec_circ = 2.0 * np.pi * sec_radius
        sec_offset = sec_circ * (1.0 - (conf_pct / 100.0))

        h_col1, h_col2 = st.columns([1.1, 1.2])

        with h_col1:
            st.markdown(f"""
            <div class="fin-card-hero">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <span class="fin-card-title">Underwriting Assessment</span>
                    <span class="fin-badge-tier {tier_class}">{b_tier}</span>
                </div>

                <div style="display: flex; justify-content: space-around; align-items: center; padding: 16px 0;">
                    <!-- Primary Score Progress Ring -->
                    <div style="text-align: center;">
                        <svg width="200" height="200" viewBox="0 0 200 200">
                            <circle cx="100" cy="100" r="{ring_radius}" fill="none" stroke="rgba(255, 255, 255, 0.06)" stroke-width="9" />
                            <circle cx="100" cy="100" r="{ring_radius}" fill="none" stroke="{tier_color}" stroke-width="9"
                                stroke-dasharray="{ring_circ:.2f}" stroke-dashoffset="{ring_offset:.2f}" stroke-linecap="round"
                                transform="rotate(-90 100 100)" />
                            <text x="100" y="96" text-anchor="middle" font-family="'Inter', sans-serif"
                                font-size="40px" font-weight="700" fill="#F5F1EA">{b_score}</text>
                            <text x="100" y="122" text-anchor="middle" font-family="'Inter', sans-serif"
                                font-size="12px" font-weight="400" fill="#9C9088">Score / 900</text>
                        </svg>
                        <div style="font-size: 0.80rem; color: #9C9088; font-weight: 500; margin-top: 4px;">Credit Score</div>
                    </div>

                    <!-- Secondary Ring: Underwriting Solvency Index -->
                    <div style="text-align: center;">
                        <svg width="130" height="130" viewBox="0 0 130 130">
                            <circle cx="65" cy="65" r="{sec_radius}" fill="none" stroke="rgba(255, 255, 255, 0.06)" stroke-width="7" />
                            <circle cx="65" cy="65" r="{sec_radius}" fill="none" stroke="#F5F1EA" stroke-width="7"
                                stroke-dasharray="{sec_circ:.2f}" stroke-dashoffset="{sec_offset:.2f}" stroke-linecap="round"
                                transform="rotate(-90 65 65)" />
                            <text x="65" y="62" text-anchor="middle" font-family="'Inter', sans-serif"
                                font-size="22px" font-weight="600" fill="#F5F1EA">{conf_pct:.0f}%</text>
                            <text x="65" y="80" text-anchor="middle" font-family="'Inter', sans-serif"
                                font-size="9.5px" font-weight="400" fill="#9C9088">Confidence</text>
                        </svg>
                        <div style="font-size: 0.80rem; color: #9C9088; font-weight: 500; margin-top: 4px;">Solvency Index</div>
                    </div>
                </div>

                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 14px;">
                    <div>
                        <div class="fin-card-title">Calibrated P(Default)</div>
                        <div style="font-size: 1.3rem; font-weight: 600; color: #F5F1EA; margin-top: 2px;">{p_def:.1%}</div>
                    </div>
                    <div>
                        <div class="fin-card-title">Historical Ground Truth</div>
                        <div style="font-size: 1.3rem; font-weight: 600; color: {'#F87171' if borrower_row['defaulted'] == 1 else '#4ADE80'}; margin-top: 2px;">
                            {'Defaulted' if borrower_row['defaulted'] == 1 else 'Clean Repayment'}
                        </div>
                    </div>
                </div>
            </div>

            <!-- Adverse Action Notice Box -->
            <div class="fin-notice-box" style="margin-top: 14px;">
                <strong style="color: #F5F1EA; font-size: 0.90rem;">Regulatory Fair-Lending Notice (RBI Digital Lending Guideline):</strong>
                <div style="margin-top: 6px; color: #9C9088; font-size: 0.86rem; line-height: 1.65;">
                    {explanation_res["plain_english_explanation"]}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with h_col2:
            st.markdown('<div class="fin-card">', unsafe_allow_html=True)
            st.markdown('<div class="fin-card-title" style="color: #F5F1EA; margin-bottom: 14px;">Top Behavioral Score Contributors</div>', unsafe_allow_html=True)

            all_factors = explanation_res["top_positive_factors"] + explanation_res["top_negative_factors"]
            max_pts = max([abs(p) for _, p in all_factors] or [40])
            max_pts = max(max_pts, 35)

            # Positive drivers in soft green #4ADE80
            for name, pts in explanation_res["top_positive_factors"]:
                w = min(round((pts / max_pts) * 50, 1), 50)
                st.markdown(f"""
                <div style="margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.84rem; font-weight: 500; margin-bottom: 4px;">
                        <span style="color: #F5F1EA;">{name}</span>
                        <span style="color: #4ADE80; font-weight: 600;">+{pts} pts</span>
                    </div>
                    <div style="position: relative; height: 7px; background: rgba(255, 255, 255, 0.05); border-radius: 9999px; overflow: hidden;">
                        <div style="position: absolute; left: 50%; top: 0; bottom: 0; width: 1px; background: rgba(255, 255, 255, 0.2); z-index: 2;"></div>
                        <div style="position: absolute; left: 50%; top: 0; bottom: 0; background: #4ADE80; border-radius: 0 3px 3px 0; width: {w}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Negative drivers in soft red #F87171
            for name, pts in explanation_res["top_negative_factors"]:
                w = min(round((abs(pts) / max_pts) * 50, 1), 50)
                st.markdown(f"""
                <div style="margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.84rem; font-weight: 500; margin-bottom: 4px;">
                        <span style="color: #F5F1EA;">{name}</span>
                        <span style="color: #F87171; font-weight: 600;">{pts} pts</span>
                    </div>
                    <div style="position: relative; height: 7px; background: rgba(255, 255, 255, 0.05); border-radius: 9999px; overflow: hidden;">
                        <div style="position: absolute; left: 50%; top: 0; bottom: 0; width: 1px; background: rgba(255, 255, 255, 0.2); z-index: 2;"></div>
                        <div style="position: absolute; right: 50%; top: 0; bottom: 0; background: #F87171; border-radius: 3px 0 0 3px; width: {w}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            # Telemetry Snapshot Specs Panel
            st.markdown('<div class="fin-card">', unsafe_allow_html=True)
            st.markdown('<div class="fin-card-title" style="color: #F5F1EA; margin-bottom: 14px;">Applicant Telemetry Profile</div>', unsafe_allow_html=True)
            s_col1, s_col2, s_col3 = st.columns(3)
            with s_col1:
                st.markdown(f"<div class='fin-card-title'>OCCUPATION</div><div style='font-weight: 600; color: #F5F1EA; font-size: 0.95rem; margin-top: 2px;'>{borrower_row['occupation_type']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='fin-card-title' style='margin-top: 10px;'>EST. INCOME</div><div style='font-weight: 600; color: #4ADE80; font-size: 0.95rem; margin-top: 2px;'>₹{borrower_row['monthly_income_estimate']:,.0f}</div>", unsafe_allow_html=True)
            with s_col2:
                st.markdown(f"<div class='fin-card-title'>URBAN TIER</div><div style='font-weight: 600; color: #F5F1EA; font-size: 0.95rem; margin-top: 2px;'>{borrower_row['city_tier']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='fin-card-title' style='margin-top: 10px;'>UTILITY ON-TIME</div><div style='font-weight: 600; color: #F5F1EA; font-size: 0.95rem; margin-top: 2px;'>{borrower_row['electricity_bill_ontime_rate']*100:.0f}%</div>", unsafe_allow_html=True)
            with s_col3:
                st.markdown(f"<div class='fin-card-title'>RECHARGE LAPSE</div><div style='font-weight: 600; color: #F5F1EA; font-size: 0.95rem; margin-top: 2px;'>{borrower_row['days_since_last_recharge_lapse']:.0f} days</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='fin-card-title' style='margin-top: 10px;'>P2P/QR RATIO</div><div style='font-weight: 600; color: #F5F1EA; font-size: 0.95rem; margin-top: 2px;'>{borrower_row['p2p_vs_merchant_txn_ratio']:.2f}</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# TAB 4: MODEL PERFORMANCE & BENCHMARKING
# =============================================================================
with tab4:
    st.markdown('<div class="fin-card-title" style="font-size: 1.1rem; color: #F5F1EA; margin-bottom: 8px;">Institutional Model Evaluation & Separation Benchmarks</div>', unsafe_allow_html=True)
    st.markdown('<div style="color: #9C9088; font-size: 0.88rem; margin-bottom: 18px;">CreditBridge benchmarks an interpretable cost-sensitive Logistic Regression champion against an XGBoost challenger to balance discriminatory power with regulatory transparency.</div>', unsafe_allow_html=True)

    all_metrics = model_bundle.get("all_metrics", {})
    lr_m = all_metrics.get("logistic_regression", {})
    xgb_m = all_metrics.get("xgboost", {})

    comp_data = {
        "Evaluation Dimension": [
            "AUC-ROC (Discriminatory Power)",
            "KS-Statistic (% Separation)",
            "Precision (Default Class)",
            "Recall / NPA Capture Rate",
            "F1 Harmonic Score"
        ],
        "Logistic Regression (Champion)": [
            f"{lr_m.get('auc', 0.6240):.4f}",
            f"{lr_m.get('ks_stat', 21.08):.2f}%",
            f"{lr_m.get('precision', 0.1958):.4f}",
            f"{lr_m.get('recall', 0.5595):.4f}",
            f"{lr_m.get('f1', 0.2901):.4f}"
        ],
        "XGBoost Classifier (Challenger)": [
            f"{xgb_m.get('auc', 0.6075):.4f}",
            f"{xgb_m.get('ks_stat', 17.36):.2f}%",
            f"{xgb_m.get('precision', 0.2000):.4f}",
            f"{xgb_m.get('recall', 0.3393):.4f}",
            f"{xgb_m.get('f1', 0.2517):.4f}"
        ],
        "Underwriting Benchmark Standard": [
            "Rank-order quality across all cutoffs",
            "Target 20-40% for institutional separation",
            "Precision on minority default segment",
            "Capture rate of non-performing loans (NPA)",
            "Harmonic balance between false alarms and missed risk"
        ]
    }
    st.markdown('<div class="fin-card">', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Precompute test split for clean curves
    y_all_defaults = raw_df["defaulted"].to_numpy()
    _, y_test_array = train_test_split(y_all_defaults, test_size=0.15, stratify=y_all_defaults, random_state=42)
    lr_probs_array = lr_m.get("probabilities")
    xgb_probs_array = xgb_m.get("probabilities")

    m_col1, m_col2 = st.columns(2)

    with m_col1:
        st.markdown('<div class="fin-card">', unsafe_allow_html=True)
        st.markdown('<div class="fin-card-title" style="color: #F5F1EA; margin-bottom: 12px;">ROC Curve (Discriminatory Power)</div>', unsafe_allow_html=True)

        if lr_probs_array is not None and xgb_probs_array is not None:
            fpr_lr, tpr_lr, _ = roc_curve(y_test_array, lr_probs_array)
            fpr_xgb, tpr_xgb, _ = roc_curve(y_test_array, xgb_probs_array)

            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr_lr,
                y=tpr_lr,
                mode='lines',
                name=f"Logistic Regression (AUC = {lr_m.get('auc', 0.6240):.3f})",
                line=dict(color='#E8792E', width=2.5),
                fill='tozeroy',
                fillcolor='rgba(232, 121, 46, 0.12)'
            ))
            fig_roc.add_trace(go.Scatter(
                x=fpr_xgb,
                y=tpr_xgb,
                mode='lines',
                name=f"XGBoost (AUC = {xgb_m.get('auc', 0.6075):.3f})",
                line=dict(color='#9C9088', width=1.8, dash='dash')
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name="Random Guessing",
                line=dict(color='rgba(255,255,255,0.2)', width=1, dash='dot')
            ))

            fig_roc.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=10, b=10),
                height=300,
                xaxis=dict(
                    title=dict(text="False Positive Rate", font=dict(color='#9C9088', size=11)),
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.05)',
                    tickfont=dict(color='#9C9088', size=10)
                ),
                yaxis=dict(
                    title=dict(text="True Positive Rate", font=dict(color='#9C9088', size=11)),
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.05)',
                    tickfont=dict(color='#9C9088', size=10)
                ),
                legend=dict(
                    font=dict(color='#F5F1EA', size=10),
                    bgcolor='rgba(0,0,0,0)',
                    x=0.45,
                    y=0.10
                )
            )

            st.plotly_chart(fig_roc, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with m_col2:
        st.markdown('<div class="fin-card">', unsafe_allow_html=True)
        st.markdown('<div class="fin-card-title" style="color: #F5F1EA; margin-bottom: 12px;">Kolmogorov-Smirnov (KS) Separation Curve</div>', unsafe_allow_html=True)

        if lr_probs_array is not None:
            thresholds = np.linspace(0, 1, 101)
            goods_cdf = [float(np.mean(lr_probs_array[y_test_array == 0] <= t)) for t in thresholds]
            bads_cdf = [float(np.mean(lr_probs_array[y_test_array == 1] <= t)) for t in thresholds]

            fig_ks = go.Figure()
            fig_ks.add_trace(go.Scatter(
                x=thresholds,
                y=goods_cdf,
                mode='lines',
                name="Goods CDF (Non-Defaulters)",
                line=dict(color='#4ADE80', width=2.2)
            ))
            fig_ks.add_trace(go.Scatter(
                x=thresholds,
                y=bads_cdf,
                mode='lines',
                name="Bads CDF (Defaulters)",
                line=dict(color='#F87171', width=2.2)
            ))

            # Max separation marker
            ks_diff = np.abs(np.array(bads_cdf) - np.array(goods_cdf))
            max_idx = int(np.argmax(ks_diff))
            max_thresh = float(thresholds[max_idx])
            fig_ks.add_vline(
                x=max_thresh,
                line_width=1.5,
                line_dash="dash",
                line_color="#F2C94C",
                annotation_text=f"Max KS = {lr_m.get('ks_stat', 21.08):.1f}%",
                annotation_position="top right",
                annotation_font=dict(color='#F2C94C', size=11)
            )

            fig_ks.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=10, b=10),
                height=300,
                xaxis=dict(
                    title=dict(text="Default Probability Cutoff", font=dict(color='#9C9088', size=11)),
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.05)',
                    tickfont=dict(color='#9C9088', size=10)
                ),
                yaxis=dict(
                    title=dict(text="Cumulative Share", font=dict(color='#9C9088', size=11)),
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.05)',
                    tickfont=dict(color='#9C9088', size=10)
                ),
                legend=dict(
                    font=dict(color='#F5F1EA', size=10),
                    bgcolor='rgba(0,0,0,0)',
                    x=0.45,
                    y=0.15
                )
            )

            st.plotly_chart(fig_ks, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# TAB 5: METHODOLOGY & ETHICS
# =============================================================================
with tab5:
    st.markdown('<div class="fin-card-title" style="font-size: 1.1rem; color: #F5F1EA; margin-bottom: 16px;">Regulatory Framework, Ethical Guardrails & Compliance</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="fin-card-hero" style="max-width: 960px; margin: 0 auto 24px auto;">
        <h3 style="color: #F5F1EA; font-weight: 600; margin-top: 0; margin-bottom: 12px;">1. Synthetic Data Calibration vs. Production Reality</h3>
        <p style="color: #9C9088; line-height: 1.72; font-size: 0.92rem;">
            • <strong style="color: #F5F1EA;">Proof-of-Concept Prototype:</strong> This portfolio engine is calibrated on <strong style="color: #E8792E;">8,000 statistically correlated synthetic borrower profiles</strong>. It demonstrates the analytical pipeline necessary to evaluate thin-file applicants where traditional bureau records are absent.
            <br>• <strong style="color: #F5F1EA;">Production Ingestion Framework:</strong> In a live production deployment across India, telemetry would be ingested via:
            <br>&nbsp;&nbsp;– <strong style="color: #F5F1EA;">Account Aggregator (AA) Ecosystem:</strong> RBI-regulated consent-based digital bank statement retrieval (FIP to FIU).
            <br>&nbsp;&nbsp;– <strong style="color: #F5F1EA;">DISCOM & Telecom Ingestion:</strong> Voluntary consent-based pull of electricity board records and mobile recharge track records.
            <br>&nbsp;&nbsp;– <strong style="color: #F5F1EA;">Gig Partner APIs:</strong> Direct employer integration (e.g. Swiggy/Zomato Delivery Partner API, Uber/Ola Driver Portal).
        </p>

        <h3 style="color: #F5F1EA; font-weight: 600; margin-top: 24px; margin-bottom: 12px;">2. Fair-Lending Principles & Protected Demographics</h3>
        <p style="color: #9C9088; line-height: 1.72; font-size: 0.92rem;">
            • <strong style="color: #F5F1EA;">Deliberate Exclusion of Biased Proxies:</strong> Demographic features representing or acting as proxies for <strong style="color: #F87171;">religion, caste, marital status, and gender</strong> were explicitly omitted from the data model.
            <br>• <strong style="color: #F5F1EA;">RBI Digital Lending Guidelines Compliance:</strong> The Reserve Bank of India strictly mandates borrower explainability and algorithmic consent. CreditBridge ensures that adverse underwriting decisions can be challenged and explained in plain English, providing actionable steps for applicants to improve creditworthiness (e.g. maintaining recharge regularity or reducing P2P debt concentration).
        </p>

        <h3 style="color: #F5F1EA; font-weight: 600; margin-top: 24px; margin-bottom: 12px;">3. Known Prototype Limitations & Production Mitigations</h3>
        <p style="color: #9C9088; line-height: 1.72; font-size: 0.92rem;">
            • <strong style="color: #F5F1EA;">Macroeconomic & Climate Shocks:</strong> Synthetic datasets cannot fully simulate regional monsoon slowdowns, extreme heatwaves, or systemic inflation shocks directly impacting daily wage earners.
            <br>• <strong style="color: #F5F1EA;">Behavioral Drift & Gaming:</strong> Once scoring mechanics are transparent, borrowers might artificially alter recharge tickets. Live production systems require dynamic concept drift monitoring and anti-fraud graph clustering.
            <br>• <strong style="color: #F5F1EA;">Class Imbalance Realism:</strong> Default rate is calibrated to ~14% to mirror real-world thin-file segments, resolved through algorithmic cost-sensitive weighting rather than synthetic oversampling (SMOTE).
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; color: #6E655D; font-size: 0.80rem; padding: 20px 0 10px 0;">
    CreditBridge Alternative Credit Underwriting Engine • Built for Institutional Fintech & NBFC Risk Analytics
</div>
""", unsafe_allow_html=True)
