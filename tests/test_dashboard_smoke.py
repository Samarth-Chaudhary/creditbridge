"""
CreditBridge - Alternative Credit Scoring Engine
Part 9: Read-Only Dashboard Smoke & Contract Compatibility Tests
Path: tests/test_dashboard_smoke.py

Verifies that all backend dependencies and contracts imported by dashboard/app.py
remain intact, and confirms that dashboard/app.py remains 100% syntactically valid
and protected under the Dashboard Firewall.
"""

import ast
from pathlib import Path

import pandas as pd


def test_dashboard_file_integrity_and_firewall():
    """Verify that dashboard/app.py exists, is non-empty, and protected under the Dashboard Firewall."""
    project_root = Path(__file__).resolve().parent.parent
    dashboard_path = project_root / "dashboard" / "app.py"

    assert dashboard_path.exists(), "dashboard/app.py is missing!"
    assert dashboard_path.stat().st_size > 0, "dashboard/app.py is empty!"
    assert dashboard_path.stat().st_size == 53136, (
        f"FIREWALL VIOLATION: dashboard/app.py size mismatch! "
        f"Expected 53136 bytes, got {dashboard_path.stat().st_size} bytes."
    )


def test_dashboard_syntax_validity():
    """Verify that dashboard/app.py is syntactically valid Python code."""
    project_root = Path(__file__).resolve().parent.parent
    dashboard_path = project_root / "dashboard" / "app.py"

    with open(dashboard_path, "r", encoding="utf-8") as f:
        source_code = f.read()

    # Must parse without SyntaxError
    tree = ast.parse(source_code, filename="dashboard/app.py")
    assert tree is not None


def test_dashboard_backend_import_contracts():
    """Verify all backend symbols and contracts imported by dashboard/app.py exist and are callable."""
    # Test scoring_utils imports
    from src.scoring_utils import (
        POPULATION_DEFAULT_RATE,
        SCORE_FACTOR,
        load_model_bundle,
        probability_to_credit_score,
        score_borrower,
        score_to_tier,
    )
    assert callable(load_model_bundle)
    assert callable(score_borrower)
    assert callable(probability_to_credit_score)
    assert callable(score_to_tier)
    assert isinstance(SCORE_FACTOR, float)
    assert isinstance(POPULATION_DEFAULT_RATE, float)

    # Test explain imports
    from src.explain import FEATURE_NAME_MAP, explain_single_borrower
    assert callable(explain_single_borrower)
    assert isinstance(FEATURE_NAME_MAP, dict)
    assert len(FEATURE_NAME_MAP) > 0


def test_dashboard_data_assets_availability():
    """Verify all data and model assets expected by the dashboard are present and readable."""
    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / "models" / "credit_model.pkl"
    data_path = project_root / "data" / "synthetic_borrowers.csv"

    assert model_path.exists()
    assert model_path.stat().st_size == 23277

    assert data_path.exists()
    df = pd.read_csv(data_path, nrows=5)
    assert len(df) == 5
    assert "borrower_id" in df.columns
    assert "defaulted" in df.columns
