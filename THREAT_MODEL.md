# Threat Model: CreditBridge Alternative Underwriting System

**Security Standard**: STRIDE Threat Modeling Framework & OWASP Top 10 for ML/LLM  
**System Scope**: Document Ingestion, Data Quality Gating, Model Serialization, Scoring API, and Audit Infrastructure  
**Author**: Model-Risk & AppSec Review Team  
**Status**: Active Security Policy  

---

## 1. System Architecture & Attack Surface Decomposition

CreditBridge processes semi-structured financial documents (CSV exports of bank/UPI statements) and executes serialized machine learning models. The system presents several distinct external attack surfaces:

```
[Untrusted Client / User]
       │
       │ HTTP Upload (CSV Bank / UPI Statement)
       ▼
┌────────────────────────────────────────────────────────┐
│ 1. INGESTION & DATA SANITIZATION SURFACE               │
│ - File Size / Row Limits                               │
│ - Magic Byte Header Inspection                         │
│ - Path Traversal & File Name Sanitization              │
│ - CSV Formula Injection Neutralization                 │
│ - PII Detection & Sanitization                         │
└────────────────────────┬───────────────────────────────┘
                         │ Canonical Transactions
                         ▼
┌────────────────────────────────────────────────────────┐
│ 2. MODEL PIPELINE & ARTIFACT INGESTION SURFACE         │
│ - Model Deserialization Security (CWE-502)             │
│ - Cryptographic SHA-256 Hash Verification              │
│ - Single Champion Registry Enforcement                 │
└────────────────────────┬───────────────────────────────┘
                         │ Inferred Probabilities & Decisions
                         ▼
┌────────────────────────────────────────────────────────┐
│ 3. DECISIONING, EXPLAINABILITY & AUDIT SURFACE         │
│ - SHAP Attribution / Model Extraction Resistance      │
│ - Privacy-Safe Audit Manifest Generation               │
│ - Structured Logging (No Sensitive Data Leakage)       │
└────────────────────────────────────────────────────────┘
```

---

## 2. Threat Analysis & Mitigations (STRIDE Matrix)

| Threat Category | Specific Threat Description | Affected Component | Severity | Implemented Technical Mitigation | Verification Test |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **Spoofing** | Adversary uploads synthetic transactions imitating a recognized employer (e.g. `ZOMATO SALARY`) to elevate estimated income. | `transaction_classifier.py` | HIGH | Multi-vector verification requiring recurring cadence, volume thresholds, and cross-channel UPI reconciliation. Marked as `SELF_REPORTED` or `MARGINAL` if variance is anomalous. | `tests/test_real_data_features.py` |
| **Tampering** | Malicious replacement or tampering of serialized model pickle weights (`models/credit_model.pkl`) to force auto-approvals. | Model Registry & Inference Pipeline | CRITICAL | Immutable SHA-256 fingerprint verification (`bfbabaa4...`) enforced before deserialization. Automated halt if digest does not match signed registry manifest. | `tests/test_model_integration.py` |
| **Repudiation** | Borrower or underwriter disputes an underwriting decision or claims model bias. | `audit_log.json` / Manifest | MEDIUM | Cryptographically verifiable audit manifest recording timestamp, request UUID, model version, exact input feature vector, and local SHAP attributions. | `tests/test_phase2_governance.py` |
| **Information Disclosure** | Leakage of Personally Identifiable Information (PII) such as Aadhaar numbers, PAN cards, or phone numbers in logs or error stack traces. | `privacy_security.py` / Logger | HIGH | Regex-based PII redaction scanner masking phone numbers (`XXXXXX1234`), PAN formats (`XXXXX1234X`), and Aadhaar strings prior to persistence or display. | `tests/test_privacy_security.py` |
| **Denial of Service** | Resource exhaustion attack via massive multi-gigabyte statement uploads or decompression bombs. | Ingestion Gateway | HIGH | Strict 10 MB maximum file size cap, 50,000 transaction row limit, and chunked non-blocking streaming parsing. | `tests/test_csv_parser.py` |
| **Elevation of Privilege** | Arbitrary code execution via unsafe Python pickle/joblib deserialization (`__reduce__`). | `scoring_utils.py` | CRITICAL | Python `pickle` is explicitly classified as NOT a security boundary. Deserialization is permitted ONLY from verified internal paths matching the exact hardcoded SHA-256 checksum. | `tests/test_privacy_security.py` |

---

## 3. Deep-Dive Security Controls

### 3.1 Malicious Uploads & Path Traversal (CWE-22 / CWE-434)
- **Control**: Filenames supplied by client `Content-Disposition` headers are never used directly in filesystem operations.
- **Implementation**: The parser uses `Path(filename).name` with strict regex filtering (`^[a-zA-Z0-9_\-\.]+$`), stripping null bytes (`\x00`), `../`, and directory separators.
- **Binary Magic Byte Inspection**: Inspects the first 16 bytes of every uploaded file. Executable binary headers (`MZ` for PE/EXE, `\x7fELF` for Linux binaries, PKzip archives) are immediately rejected with `FileTypeError`.

### 3.2 CSV Formula / Command Injection (CWE-1236)
- **Control**: Underwriters frequently export audit logs into spreadsheet tools (Microsoft Excel, LibreOffice Calc). If a transaction narration begins with execution characters (`=`, `+`, `-`, `@`, `\t`, `\r`), the spreadsheet could execute arbitrary system commands via DDE.
- **Implementation**: All text fields are scanned; any formula-triggering prefix is prepended with a single quote (`'`) to force plain-text literal rendering.

### 3.3 Pickle Deserialization Risk & CWE-502 Mitigation
- **Risk**: Python's `pickle` and `joblib` formats execute arbitrary Python bytecode during unpickling. If an attacker replaces `models/credit_model.pkl` with a malicious payload, loading the model executes code with the web server's privileges.
- **Defense**:
  1. Immutable storage permissions on the `models/` directory in production.
  2. Mandatory pre-deserialization SHA-256 checksum validation:
     ```python
     computed_hash = hashlib.sha256(open("models/credit_model.pkl", "rb").read()).hexdigest()
     if computed_hash != EXPECTED_CHAMPION_HASH:
         raise SecurityIntegrityViolation("Model checksum mismatch! Tampering detected.")
     ```

### 3.4 Dependency Compromise & Supply Chain Hardening
- **Control**: All direct and transitive third-party dependencies are pinned with exact versions and integrity hashes in `requirements.lock`.
- **Policy**: Routine automated vulnerability scanning via GitHub Actions CI (`pip-audit` / `safety`) to detect known CVEs in upstream libraries.

### 3.5 Logging Leakage & PII Redaction
- **Control**: Production logs must never contain raw transaction narratives or customer identifiers.
- **Implementation**: The logging pipeline sanitizes customer phone numbers (`\d{10}`), Indian Permanent Account Numbers (PAN: `[A-Z]{5}[0-9]{4}[A-Z]`), and bank account numbers, replacing them with cryptographic hashes or masked substrings.

---

## 4. Security Incident Response & Rollback Runbook

If model tampering, high PSI drift (>0.25), or abnormal error rates occur:
1. **Immediate State Transition**: The governance state machine switches from `HEALTHY` to `BLOCK`.
2. **Automated Traffic Halting**: The inference API rejects incoming scoring requests with HTTP 503 and transfers applicants to human underwriter queue.
3. **Artifact Integrity Audit**: Automatic comparison of all filesystem models against the signed Git commit hashes.
4. **Rollback**: Instantaneous atomic symlink pointer reversion to the previous immutable model release.
