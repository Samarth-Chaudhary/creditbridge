"""
CreditBridge - Alternative Credit Scoring Engine
Unit & Security Regression Tests: Privacy, Security & Responsible AI
Path: tests/test_privacy_security.py

Tests defensive upload validation, magic byte inspection, path traversal prevention,
zero disk persistence of personal statements, error message sanitization,
network isolation, audit manifest generation, and subgroup fairness governance.
"""

import io
import os
import socket
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.real_data_contracts import (
    FileTypeError,
    OversizedFileError,
    PathTraversalError,
    SecurityViolationError,
    InvalidDataError,
    ManualInputContract,
)
from src.privacy_security import (
    validate_upload_security,
    assert_no_credential_fields,
    sanitize_error_message,
    create_audit_manifest,
    verify_network_isolation,
    MAX_FILE_SIZE_BYTES,
    MAX_CSV_ROWS,
)
from src.real_data_parser import parse_csv_statement
from src.real_data_scoring import assess_statement_end_to_end
from src.fairness_diagnostics import evaluate_subgroup_fairness, FairnessEvaluationReport


# -----------------------------------------------------------------------------
# 1. DEFENSIVE UPLOAD & FILE-TYPE VALIDATION TESTS
# -----------------------------------------------------------------------------

def test_unsupported_file_extension_rejected():
    """Verify upload validation rejects unsupported file extensions."""
    # Test .exe, .sh, .bin, .pdf
    for bad_ext in ["payload.exe", "script.sh", "firmware.bin", "statement.pdf", "data.json"]:
        with pytest.raises(FileTypeError) as excinfo:
            validate_upload_security(b"sample content", filename=bad_ext)
        assert "Unsupported file extension" in str(excinfo.value)


def test_oversized_file_rejected():
    """Verify files exceeding 10 MB are rejected before parsing."""
    oversized_bytes = b"Date,Amount\n2023-01-01,100\n" + (b"0" * (MAX_FILE_SIZE_BYTES + 512))
    with pytest.raises(OversizedFileError) as excinfo:
        validate_upload_security(oversized_bytes, filename="huge.csv")
    assert "exceeds maximum permitted size" in str(excinfo.value)


def test_executable_and_binary_magic_bytes_rejected():
    """Verify binary/executable payloads are detected via magic signatures and rejected."""
    # Windows PE executable (MZ header)
    mz_payload = b"MZ\x90\x00\x03\x00\x00\x00"
    with pytest.raises(SecurityViolationError) as excinfo:
        validate_upload_security(mz_payload, filename="statement.csv")
    assert "Windows Executable" in str(excinfo.value)

    # Linux ELF binary
    elf_payload = b"\x7fELF\x02\x01\x01\x00"
    with pytest.raises(SecurityViolationError) as excinfo:
        validate_upload_security(elf_payload, filename="statement.csv")
    assert "Linux ELF" in str(excinfo.value)

    # Zip/Office/Jar archive
    zip_payload = b"PK\x03\x04\x14\x00\x00\x00"
    with pytest.raises(SecurityViolationError) as excinfo:
        validate_upload_security(zip_payload, filename="statement.csv")
    assert "Zip / Office" in str(excinfo.value)


def test_path_traversal_prevention():
    """Verify directory traversal sequences in upload filenames are blocked."""
    traversal_filenames = [
        "../etc/passwd",
        "..\\..\\windows\\system32\\cmd.exe",
        "nested/../../secret.csv",
        "dir/../test.csv",
    ]
    for bad_path in traversal_filenames:
        with pytest.raises(PathTraversalError) as excinfo:
            validate_upload_security(b"Date,Amount\n2023-01-01,100", filename=bad_path)
        assert "Path traversal sequence detected" in str(excinfo.value)


def test_null_byte_injection_blocked():
    """Verify null-byte injection in filenames is blocked."""
    with pytest.raises(PathTraversalError) as excinfo:
        validate_upload_security(b"Date,Amount\n2023-01-01,100", filename="statement.csv\x00.exe")
    assert "Null byte injection" in str(excinfo.value)


def test_max_csv_rows_dos_safeguard():
    """Verify statements with > 50,000 rows are rejected to prevent memory exhaustion."""
    # Construct CSV exceeding MAX_CSV_ROWS (simulate 50,001 rows)
    lines = ["Date,Amount,Counterparty"] + ["2023-01-01,100,Merchant"] * (MAX_CSV_ROWS + 1)
    huge_csv_text = "\n".join(lines)
    
    with pytest.raises(InvalidDataError) as excinfo:
        parse_csv_statement(huge_csv_text)
    assert f"exceeds maximum allowed limit of {MAX_CSV_ROWS:,} rows" in str(excinfo.value)


# -----------------------------------------------------------------------------
# 2. DATA PERSISTENCE & LEAK AUDIT TESTS
# -----------------------------------------------------------------------------

def test_raw_statement_is_never_written_to_disk(tmp_path):
    """
    Verify that executing the full Real Data evaluation pipeline leaves
    ZERO raw statement traces or customer tokens on disk.
    """
    secret_marker = "SENSITIVE_CUSTOMER_TOKEN_9988776655"
    csv_content = (
        f"Date,Amount,Counterparty\n"
        f"2023-01-01,2500,{secret_marker}\n"
        f"2023-01-15,3000,Swiggy\n"
        f"2023-02-01,1500,Airtel\n"
    )

    inputs = ManualInputContract(age=30, occupation_type="gig_delivery", city_tier="tier_1")
    result = assess_statement_end_to_end(csv_content, manual_inputs=inputs)
    assert result is not None

    # Audit the workspace directory: no file should contain secret_marker
    project_root = Path(__file__).resolve().parent.parent
    scanned_files = 0
    for root, dirs, files in os.walk(project_root):
        # Exclude git, pycache, and test file itself
        if any(ignored in root for ignored in [".git", "__pycache__", ".pytest_cache"]):
            continue
        for file in files:
            if file.endswith(".pyc") or file == "test_privacy_security.py":
                continue
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    assert secret_marker not in content, (
                        f"CRITICAL PRIVACY VIOLATION: Raw customer data found written on disk at {file_path}!"
                    )
                scanned_files += 1
            except Exception:
                pass
    assert scanned_files > 10, "Sanity check: verified disk scan checked multiple files."


def test_error_message_sanitization():
    """Verify that exception messages do not leak PII, card/account numbers, PANs, or local paths."""
    raw_error_message = (
        "Failed at C:\\Users\\Administrator\\secret_data\\statement.csv: "
        "Customer PAN ABCDE1234F with Account 1234567890123456 and email user@creditbridge.com "
        "and phone 9876543210 encountered parsing error."
    )
    sanitized = sanitize_error_message(raw_error_message)

    assert "C:\\Users\\Administrator" not in sanitized
    assert "<sanitized_path>" in sanitized
    assert "1234567890123456" not in sanitized
    assert "<sanitized_num>" in sanitized
    assert "user@creditbridge.com" not in sanitized
    assert "<sanitized_email>" in sanitized
    assert "ABCDE1234F" not in sanitized
    assert "<sanitized_pan>" in sanitized
    assert "9876543210" not in sanitized
    assert "<sanitized_phone>" in sanitized


def test_forbidden_credential_fields_rejected():
    """Verify that any payload containing credentials, PINs, or passwords raises a SecurityViolationError."""
    forbidden_payloads = [
        {"password": "secret_password"},
        {"user_pin": "1234"},
        {"upi_pin": "9999"},
        {"otp": "554433"},
        {"cvv": "123"},
        {"api_key": "live_key_xyz"},
    ]
    for payload in forbidden_payloads:
        with pytest.raises(SecurityViolationError) as excinfo:
            assert_no_credential_fields(payload)
        assert "Forbidden credential/security field detected" in str(excinfo.value)


# -----------------------------------------------------------------------------
# 3. NETWORK ISOLATION & AUDITABILITY TESTS
# -----------------------------------------------------------------------------

def test_network_isolation_and_no_outbound_calls(monkeypatch):
    """Verify the entire Real Data Mode pipeline executes with 0 outbound network calls."""
    network_called = False

    def guard_socket_connect(*args, **kwargs):
        nonlocal network_called
        network_called = True
        raise ConnectionRefusedError("PROHIBITED: Outbound network call in offline Real Data Mode!")

    # Intercept socket connect
    monkeypatch.setattr(socket.socket, "connect", guard_socket_connect)

    # Run assessment
    csv_data = "Date,Amount,Counterparty\n2023-01-01,1200,Swiggy\n2023-01-05,800,Zomato\n"
    inputs = ManualInputContract(age=25, occupation_type="gig_delivery", city_tier="tier_1")
    result = assess_statement_end_to_end(csv_data, manual_inputs=inputs)

    assert not network_called, "Outbound network connection was attempted during Real Data scoring!"
    assert result is not None

    # Check isolation status
    isolation = verify_network_isolation()
    assert isolation["is_offline"] is True
    assert isolation["external_api_calls_enabled"] is False
    assert isolation["operating_mode"] == "AIR_GAPPED_LOCAL"


def test_audit_manifest_captures_metadata_without_raw_statement():
    """Verify AuditManifest records session telemetry without retaining raw statements or PII."""
    csv_data = "Date,Amount,Counterparty\n2023-01-01,2000,Swiggy\n2023-01-10,1500,Zomato\n"
    parsed = parse_csv_statement(csv_data)
    inputs = ManualInputContract(age=32, occupation_type="gig_rideshare", city_tier="tier_2")
    result = assess_statement_end_to_end(csv_data, manual_inputs=inputs)

    manifest = create_audit_manifest(parsed_statement=parsed, assessment_result=result)
    manifest_dict = manifest.to_dict()

    assert manifest.raw_statement_retained is False
    assert manifest.contains_pii is False
    assert manifest_dict["total_rows_parsed"] == 2
    assert manifest_dict["usable_transactions"] == 2
    assert manifest_dict["parser_version"] == "2.0.0"
    assert manifest_dict["model_version"] == "v1_synthetic_lr_pipeline"
    assert "Swiggy" not in str(manifest_dict)
    assert "Zomato" not in str(manifest_dict)


# -----------------------------------------------------------------------------
# 4. SUBGROUP FAIRNESS & RESPONSIBLE AI GOVERNANCE TESTS
# -----------------------------------------------------------------------------

def test_fairness_diagnostics_subgroup_metrics():
    """Verify fairness diagnostics engine runs on synthetic baseline and produces audit report."""
    report = evaluate_subgroup_fairness()

    assert isinstance(report, FairnessEvaluationReport)
    assert report.total_evaluated == 8000
    assert 300 <= report.overall_mean_score <= 900
    assert 0.0 <= report.overall_approval_proxy_rate <= 1.0

    # Check required dimensions
    dimensions = ["age_group", "occupation_type", "city_tier"]
    for dim in dimensions:
        assert dim in report.subgroup_metrics
        metrics = report.subgroup_metrics[dim]
        assert len(metrics) > 0

        for m in metrics:
            assert m.sample_size > 0
            assert 300 <= m.mean_score <= 900
            assert 0.0 <= m.approval_proxy_rate <= 1.0
            assert 0.0 <= m.adverse_impact_ratio <= 1.5
            if m.empirical_default_rate is not None:
                assert 0.0 <= m.empirical_default_rate <= 1.0

    # Verify Responsible AI governance disclosures are intact
    disclosures = report.governance_disclosures
    assert "fairness_diagnostic_limitation" in disclosures
    assert "does NOT prove or certify the absence of discrimination" in disclosures["fairness_diagnostic_limitation"]
    assert "shap_non_causality_boundary" in disclosures
    assert "associative model sensitivity, NOT real-world causality" in disclosures["shap_non_causality_boundary"]
    assert "illustrative_score_boundary" in disclosures
    assert "NOT a regulated credit bureau score" in disclosures["illustrative_score_boundary"]
    assert "purpose_limitation" in disclosures
    assert "prohibited from being used for automated credit approvals" in disclosures["purpose_limitation"]
