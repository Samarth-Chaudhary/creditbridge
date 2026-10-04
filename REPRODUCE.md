# Reproduction Runbook & Clean-Environment Replication Guide

**Project**: CreditBridge Alternative Credit Underwriting Engine  
**Hardware Requirements**: Standard x86_64 CPU (4 cores, 8 GB RAM, 2 GB free disk space). No GPU required.  
**Tested Operating Systems**: Windows 11, Ubuntu 22.04 LTS, macOS Sonoma (Darwin ARM64/x86_64)  
**Supported Python Runtimes**: Python 3.10, Python 3.11, Python 3.12  
**Total Pipeline Execution Time**: ~25 to 45 seconds  

---

## 1. Clean Environment Setup

### 1.1 Clone Repository & Initialize Clean Virtual Environment
```bash
# Clone the repository
git clone https://github.com/Samarth-Chaudhary/creditbridge.git
cd creditbridge

# Create an isolated Python 3.11 virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On macOS / Linux:
source .venv/bin/activate
```

### 1.2 Install Verified Dependencies
```bash
# Upgrade pip and install exact pinned dependencies from lockfile
python -m pip install --upgrade pip
pip install -r requirements.lock
```

---

## 2. Step-by-Step Pipeline Reproduction

### Step 1: Run the Automated Test Pyramid (157 Tests)
Verify that the codebase passes all unit, integration, adversarial, data contract, fairness, security, and dashboard smoke tests:
```bash
pytest
```
*Expected Output*: `157 passed in ~7s` with 0 failures and 0 warnings.

### Step 2: Validate Static Type Consistency & Linter Cleanliness
```bash
# Static type checking via Pyright
npx pyright

# Code style and import order validation via Ruff
ruff check .
```
*Expected Output*: `0 errors, 0 warnings, 0 informations` across all files.

### Step 3: Run Model Governance & Phase 2 Audit Pipeline
Execute the master pipeline runner to generate fairness audits, drift surveillance, and registry states:
```bash
python src/phase2_governance_runner.py
```
*Expected Output*:
```
Phase 2 Master Pipeline executed successfully.
Report written to: PHASE2_GOVERNANCE_REPORT.md
```

### Step 4: Verify Model Artifact Cryptographic Integrity
Verify that the frozen champion model artifact has not been modified or tampered with:
```bash
python -c "
import hashlib
from pathlib import Path

expected_hash = 'bfbabaa4e80accc24495fe9f1ef4c0d1d2bb95b8bf9b835b0744ef6c0729615c'
actual_hash = hashlib.sha256(Path('models/credit_model.pkl').read_bytes()).hexdigest()
assert actual_hash == expected_hash, f'Checksum mismatch: {actual_hash}'
print('Artifact integrity confirmed: SHA-256 matches frozen champion digest.')
"
```

### Step 5: Launch the Interactive Risk & Underwriting Dashboard
```bash
streamlit run dashboard/app.py
```
Navigate to `http://localhost:8501` to inspect all 9 operational tabs:
1. Executive Overview
2. Borrower Assessment
3. Model Performance & Decile Lift
4. Fairness & Bias Mitigation
5. Data Quality & Feature Provenance
6. Drift & Production Surveillance
7. Policy Simulator
8. Model Registry & Audit Trail
9. Methodology, Ethics & Honest Limitations

---

## 3. Quantitative Metric Tolerance Table

When executing tests and training from source, generated metrics must fall within the following documented tolerances:

| Metric | Target Documented Value | Permissible Tolerance Range | Verification Command |
| :--- | :---: | :---: | :--- |
| **Baseline ROC-AUC** | `0.6240` | $[0.6200, 0.6280]$ | `pytest tests/test_synthetic_baseline_regression.py` |
| **Baseline KS Statistic** | `21.08%` | $[20.00\%, 22.50\%]$ | `pytest tests/test_synthetic_baseline_regression.py` |
| **Brier Score (Calibrated)**| `0.1170` | $[0.1100, 0.1250]$ | `pytest tests/test_phase2_governance.py` |
| **Min Subgroup AIR** | `0.97` | $[0.90, 1.05]$ | `pytest tests/test_phase2_governance.py` |
| **Score PSI (Month 01)** | `0.0074` | $[0.0000, 0.0500]$ | `pytest tests/test_phase2_governance.py` |
| **Dashboard File Size** | `52,846 bytes` | Exact Byte Match | `pytest tests/test_dashboard_smoke.py` |
