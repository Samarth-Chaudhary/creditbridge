# Model Governance & MLOps Lifecycle Framework

**Framework Standard**: SR 11-7 (Supervisory Guidance on Model Risk Management) & RBI Guidelines on Digital Lending  
**Authority**: CreditBridge Model Governance Committee (MGC) & Model Risk Management (MRM)  
**Current Governance State**: `FAIRNESS_MONITORING`  
**Operational Health**: `HEALTHY`  
**Active Production Champion**: `v1.0.0-lr-baseline`  

---

## 1. Governance Principles & Invariants

CreditBridge enforces institutional model governance standards designed to prevent silent algorithmic failure, disparate impact, and uncontrolled model sprawl:

1. **Single Champion Invariant**: At any point in time, exactly ONE champion model is registered and authorized to serve default probability inference. Challengers may run in shadow/canary mode but cannot score production portfolios.
2. **Deterministic Artifact Lineage**: Every model deployed to the registry must be cryptographically signed with a SHA-256 digest, linked to an immutable training dataset, random seed, and Git commit hash.
3. **Multi-Gate Promotion Firewall**: No challenger model may be promoted to champion without satisfying quantitative hurdles across discrimination, calibration, fairness, and operational complexity.
4. **Active Drift Surveillance**: Production cohorts are continuously monitored via Population Stability Index (PSI); models are downgraded if demographic or behavioral drift exceeds critical thresholds.

---

## 2. End-to-End Model Lifecycle

```
┌─────────────────┐
│ 1. DEVELOPMENT  │ ─► Synthetic Simulation / Feature Engineering / Cross-Validation
└────────┬────────┘
         ▼
┌─────────────────┐
│ 2. VALIDATION   │ ─► Out-of-Time (OOT) Split / Brier Calibration / Subgroup Fairness Audit
└────────┬────────┘
         ▼
┌─────────────────┐
│ 3. REGISTRATION │ ─► Cryptographic Fingerprinting / Packaging / Registry Shadow Mode
└────────┬────────┘
         ▼
┌─────────────────┐
│ 4. DEPLOYMENT   │ ─► Single Champion Activation / Real-time Audit Manifest Generation
└────────┬────────┘
         ▼
┌─────────────────┐
│ 5. SURVEILLANCE │ ─► Monthly PSI Drift Monitoring / Health State Machine (HEALTHY/MONITOR/BLOCK)
└────────┬────────┘
         ▼
┌─────────────────┐
│ 6. RETIREMENT   │ ─► Deprecation / Champion Decommissioning / Archived Historical Audit Trail
└─────────────────┘
```

---

## 3. Champion / Challenger Promotion Gates

To replace an active champion, a candidate challenger must be formally evaluated against the historical benchmark and pass **ALL FIVE** statutory gates:

| Gate ID | Gate Name | Required Acceptance Criterion | Rationale |
| :---: | :--- | :--- | :--- |
| **G1** | **Discrimination Lift** | $\Delta \text{OOT AUC} \ge +0.0100$ and $\Delta \text{OOT KS} \ge +1.00\%$ | Demonstrates meaningful risk ranking improvement on unseen temporal cohorts. |
| **G2** | **Calibration Accuracy**| $\text{Brier Score} \le \text{Champion Brier}$ and $\text{ECE} \le 0.0300$ | Prevents mispriced loan portfolios or distorted Expected Loss ($EL$). |
| **G3** | **Fairness Parity** | Minimum Subgroup $\text{AIR} \ge 0.80$ and $\Delta \text{AIR} \ge -0.02$ | Enforces EEOC Four-Fifths rule and ensures new model does not exacerbate disparate impact. |
| **G4** | **Monotonicity & Simplicity**| Monotonic score behavior; linear explainability or constrained trees | Prevents non-monotonic score anomalies under adverse cashflow shocks. |
| **G5** | **Integrity & Security** | Exact SHA-256 fingerprint verified; 0 deserialization or magic byte risks | Guarantees pipeline and model supply chain security. |

### Case Study: Evaluation of Challenger `v1.1.0-lr-reweighted`
- **Result**: `PROMOTION_DENIED`
- **Audit Findings**:
  - G1 Failed: OOT AUC decreased by -0.0094 (from 0.6281 to 0.6187).
  - G2 Passed: Brier score remained within tolerance (0.1180 vs 0.1170).
  - G3 Passed: Subgroup AIR improved from 0.97 to 0.99.
  - **Verdict**: Per the Single Champion Invariant, champion `v1.0.0-lr-baseline` was retained because risk discrimination degraded beyond acceptable bounds.

---

## 4. Operational Health State Machine & Surveillance

The production model's operational health is monitored across four operational states:

```
        ┌─────────────┐
        │   HEALTHY   │ ◄─── PSI < 0.10, Approval Shift < 2%, 0 Tampering
        └──────┬──────┘
               │
      0.10 <= PSI < 0.25
               │
               ▼
        ┌─────────────┐
        │   MONITOR   │ ◄─── Elevated feature drift (e.g. festive UPI surge)
        └──────┬──────┘
               │
         PSI >= 0.25 or AIR < 0.80
               │
               ▼
        ┌─────────────┐
        │   REVIEW    │ ◄─── Human underwriter threshold tightened
        └──────┬──────┘
               │
     Tampering / Critical Security Event
               │
               ▼
        ┌─────────────┐
        │    BLOCK    │ ◄─── Automated scoring suspended; fallback to manual review
        └─────────────┘
```

---

## 5. Model Incident Response & Emergency Rollback Runbook

### Incident Triggers
1. **Critical Drift Event**: Monthly Score $\text{PSI} \ge 0.25$ or continuous 3-month upward drift.
2. **Fairness Breach**: Subgroup Adverse Impact Ratio drops below 0.80 on live production cohorts.
3. **Artifact Integrity Violation**: SHA-256 fingerprint mismatch indicating corrupted or tampered file weights.

### Rapid Rollback Procedure
1. **Step 1: Shift Traffic to Safe Mode**  
   Execute emergency CLI command or toggle dashboard switch:
   ```bash
   python -m src.model_governance --action suspend --reason "PSI_BREACH_DETECTED"
   ```
2. **Step 2: Point Registry to Verified Fallback Champion**  
   The governance registry reverts the active pointer to the previous immutable release:
   ```bash
   python -m src.model_governance --action rollback --target "v1.0.0-lr-baseline"
   ```
3. **Step 3: Post-Mortem & Committee Review**  
   Model Risk Management convenes to inspect feature distributions, verify whether macroeconomic shifts occurred, and determine whether model retraining or policy threshold adjustment is required.
