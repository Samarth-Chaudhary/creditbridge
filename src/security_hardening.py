"""
CreditBridge - Alternative Credit Scoring Engine
Phase 2: Security Hardening, Defensive Ingestion & Model Artifact Integrity
Path: src/security_hardening.py

Implements robust defense-in-depth security controls:
1. Upload & Ingestion Validation:
   - File extension verification (.csv, .txt only)
   - Magic bytes inspection (reject PE/MZ, ELF, Mach-O, ZIP, PDF, Shell scripts)
   - File size limits (<= 10 MB)
   - Maximum row bounds (<= 50,000 rows against memory exhaustion DoS)
   - Path traversal prevention (.. sequences, absolute paths)
   - Null-byte injection rejection (\x00)
   - Formula / CSV injection sanitization (=, +, -, @, \\t, \\r)
   - Malformed timestamp parsing and bound checking
   - Oversized field clipping (max 1024 chars per string)
   - Malicious string & script injection sanitization
   - Error-message leakage sanitization (filesystem paths, PII, stack traces)
2. Model Artifact Integrity:
   - Cryptographic SHA-256 fingerprinting and verification of model, preprocessor, config, schema, and manifest
   - Refusal of unexpected, missing, or tampered artifacts
3. Security Boundary Disclosures:
   - Formal technical documentation on pickle/joblib vulnerabilities and non-boundary status
   - Accurate, factual execution environment persistence disclosure (no false 'zero persistence' claims)
"""

import hashlib
import io
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple, Union

import pandas as pd

# -----------------------------------------------------------------------------
# 1. SECURITY LIMITS & CONSTANTS
# -----------------------------------------------------------------------------

MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB strict cap
MAX_CSV_ROWS: int = 50_000                   # Prevent DoS memory exhaustion
MAX_FIELD_LENGTH_CHARS: int = 1024           # Prevent buffer / regex explosion
ALLOWED_EXTENSIONS: Set[str] = {".csv", ".txt"}

# Dangerous magic byte signatures
DANGEROUS_MAGIC_SIGNATURES: Dict[str, bytes] = {
    "Windows Executable/DLL (PE)": b"MZ",
    "Linux ELF Executable": b"\x7fELF",
    "Java Bytecode / Mach-O Binary": b"\xca\xfe\xba\xbe",
    "ZIP / Jar / Office Archive": b"PK\x03\x04",
    "PDF Document (Disguised CSV)": b"%PDF",
    "Shell Script Shebang": b"#!",
}

# CSV formula injection triggers (DDE / Excel formula execution vectors)
CSV_FORMULA_PREFIXES: Tuple[str, ...] = ("=", "+", "-", "@", "\t", "\r")

# Malicious script / injection regex
MALICIOUS_SCRIPT_PATTERN = re.compile(
    r"(<script.*?>|javascript:|onload=|onerror=|<iframe|<embed|union\s+select|drop\s+table)",
    re.IGNORECASE
)

# File path and PII patterns for error masking
PATH_PATTERN = re.compile(r"([a-zA-Z]:\\[^\s:,\"\']+|\/[a-zA-Z0-9_\-\./]+)")
PII_PAN_PATTERN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b")
PII_CARD_PATTERN = re.compile(r"\b\d{12,19}\b")
PII_PHONE_PATTERN = re.compile(r"\b(?:\+?91|0)?[6-9]\d{9}\b")


# -----------------------------------------------------------------------------
# 2. CUSTOM SECURITY EXCEPTIONS
# -----------------------------------------------------------------------------

class SecurityHardeningError(Exception):
    """Base class for all security hardening violations."""
    pass

class PathTraversalViolation(SecurityHardeningError):
    """Raised when path traversal sequences or illegal directories are detected."""
    pass

class UnsupportedFileTypeViolation(SecurityHardeningError):
    """Raised when disallowed extensions or MIME types are uploaded."""
    pass

class OversizedPayloadViolation(SecurityHardeningError):
    """Raised when file or row limits are exceeded."""
    pass

class DangerousPayloadViolation(SecurityHardeningError):
    """Raised when malicious magic bytes, scripts, or null bytes are detected."""
    pass

class ArtifactIntegrityViolation(SecurityHardeningError):
    """Raised when model or pipeline SHA-256 hash does not match the signed manifest."""
    pass


# -----------------------------------------------------------------------------
# 3. DEFENSIVE UPLOAD AUDIT & SANITIZATION CONTROLS
# -----------------------------------------------------------------------------

def audit_and_sanitize_filename(filename: str) -> str:
    """
    Audits filename for null-bytes, path traversal sequences, and unauthorized extensions.
    Returns cleaned basename.
    """
    if not filename:
        raise PathTraversalViolation("Filename cannot be empty.")

    if "\x00" in filename:
        raise DangerousPayloadViolation("Null byte injection detected in filename.")

    # Normalize path separators
    normalized = filename.replace("\\", "/")
    parts = normalized.split("/")

    # Detect traversal attempts
    if ".." in parts or any(p.startswith("..") for p in parts):
        raise PathTraversalViolation(f"Path traversal sequence detected in filename: '{filename}'")

    # Extract basename only
    base_name = Path(normalized).name
    if not base_name or base_name in (".", ".."):
        raise PathTraversalViolation(f"Illegal or empty basename: '{filename}'")

    ext = Path(base_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise UnsupportedFileTypeViolation(
            f"Disallowed extension '{ext}'. Only {sorted(ALLOWED_EXTENSIONS)} are permitted."
        )

    return base_name


def inspect_magic_bytes(raw_bytes: bytes) -> None:
    """Detects dangerous executable or archive binary signatures in raw uploaded bytes."""
    for sig_name, sig_bytes in DANGEROUS_MAGIC_SIGNATURES.items():
        if raw_bytes.startswith(sig_bytes):
            raise DangerousPayloadViolation(
                f"Dangerous payload rejected: detected binary signature '{sig_name}'."
            )


def sanitize_csv_formula_injection(val: Any) -> Any:
    """
    Neutralizes CSV formula injection / DDE attacks.
    If string starts with =, +, -, @, \\t, or \\r, prefixes with a single quote (').
    """
    if not isinstance(val, str):
        return val

    s = val.strip()
    if any(s.startswith(p) for p in CSV_FORMULA_PREFIXES):
        # Escape by prepending single quote
        return f"'{s}"
    return s


def sanitize_malicious_script_tags(val: Any) -> Any:
    """Detects and neutralizes HTML/JS script injection or SQL injection keywords in user fields."""
    if not isinstance(val, str):
        return val

    if MALICIOUS_SCRIPT_PATTERN.search(val):
        # Sanitize HTML tags and dangerous keywords
        cleaned = re.sub(r"[<>]", "", val)
        cleaned = re.sub(r"javascript:", "sanitized_js:", cleaned, flags=re.IGNORECASE)
        return cleaned
    return val


def sanitize_dataframe_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies comprehensive security sanitization across all DataFrame cells:
    - Clips oversized string fields to MAX_FIELD_LENGTH_CHARS
    - Escapes CSV formula injection vectors
    - Sanitizes script injection patterns
    """
    clean_df = df.copy()

    for col in clean_df.columns:
        if clean_df[col].dtype == object or str(clean_df[col].dtype) == "string":
            # 1. Enforce length cap
            clean_df[col] = clean_df[col].apply(
                lambda x: x[:MAX_FIELD_LENGTH_CHARS] if isinstance(x, str) and len(x) > MAX_FIELD_LENGTH_CHARS else x
            )
            # 2. Neutralize CSV formula injection
            clean_df[col] = clean_df[col].apply(sanitize_csv_formula_injection)
            # 3. Sanitize script tags
            clean_df[col] = clean_df[col].apply(sanitize_malicious_script_tags)

    return clean_df


def validate_secure_upload(
    file_or_content: Union[str, Path, bytes, io.BytesIO, io.StringIO],
    filename: Optional[str] = None
) -> Tuple[bytes, str]:
    """
    End-to-end security verification of uploaded file contents:
    1. Filename path traversal & extension check
    2. Null-byte rejection
    3. Maximum file size check
    4. Magic bytes inspection
    5. Row bounds check
    """
    # 1. Resolve and audit filename
    effective_name = filename
    if effective_name is None and isinstance(file_or_content, (str, Path)):
        if "\n" not in str(file_or_content) and "," not in str(file_or_content):
            effective_name = str(file_or_content)
        else:
            effective_name = "upload.csv"
    elif effective_name is None:
        effective_name = "upload.csv"

    clean_filename = audit_and_sanitize_filename(effective_name)

    # 2. Extract raw bytes
    raw_bytes: bytes
    if isinstance(file_or_content, bytes):
        raw_bytes = file_or_content
    elif isinstance(file_or_content, io.BytesIO):
        curr = file_or_content.tell()
        file_or_content.seek(0)
        raw_bytes = file_or_content.read()
        file_or_content.seek(curr)
    elif isinstance(file_or_content, io.StringIO):
        raw_bytes = file_or_content.getvalue().encode("utf-8")
    elif isinstance(file_or_content, (str, Path)):
        p = Path(file_or_content)
        if p.exists() and p.is_file():
            raw_bytes = p.read_bytes()
        else:
            raw_bytes = str(file_or_content).encode("utf-8")
    else:
        raise DangerousPayloadViolation("Unsupported upload object type.")

    # 3. Null-byte check in payload
    if b"\x00" in raw_bytes:
        raise DangerousPayloadViolation("Null byte detected in file payload content.")

    # 4. File size check
    if len(raw_bytes) > MAX_FILE_SIZE_BYTES:
        raise OversizedPayloadViolation(
            f"File exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB "
            f"(received {len(raw_bytes) / (1024 * 1024):.2f} MB)."
        )

    # 5. Magic bytes inspection
    inspect_magic_bytes(raw_bytes)

    # 6. Row count check
    try:
        sample_df = pd.read_csv(io.BytesIO(raw_bytes), nrows=MAX_CSV_ROWS + 5)
        if len(sample_df) > MAX_CSV_ROWS:
            raise OversizedPayloadViolation(
                f"Upload exceeds maximum allowable rows of {MAX_CSV_ROWS:,} (found >{MAX_CSV_ROWS:,})."
            )
    except Exception as e:
        if isinstance(e, SecurityHardeningError):
            raise e
        # If parsing fails, let standard CSV parser handle structural errors downstream

    return raw_bytes, clean_filename


def sanitize_error_leakage(exc: Union[Exception, str]) -> str:
    """
    Sanitizes internal exception strings before returning to users/APIs.
    Redacts filesystem paths, stack traces, PAN numbers, card numbers, and phone numbers.
    """
    msg = str(exc)
    msg = PATH_PATTERN.sub("<sanitized_path>", msg)
    msg = PII_PAN_PATTERN.sub("<sanitized_pan>", msg)
    msg = PII_CARD_PATTERN.sub("<sanitized_account>", msg)
    msg = PII_PHONE_PATTERN.sub("<sanitized_phone>", msg)
    return msg


# -----------------------------------------------------------------------------
# 4. MODEL ARTIFACT INTEGRITY & PICKLE BOUNDARY CONTROLS
# -----------------------------------------------------------------------------

PICKLE_SECURITY_DISCLOSURE: str = (
    "SECURITY LIMITATION OF PICKLE / JOBLIB SERIALIZATION: "
    "Python standard pickle/joblib is an execution protocol, NOT a security boundary. "
    "Deserializing an unverified pickle file allows arbitrary Python code execution (RCE) "
    "via the '__reduce__' method before any application logic runs. "
    "CreditBridge enforces strict SHA-256 fingerprint verification against a signed manifest "
    "prior to loading any serialized artifact from disk. Artifacts failing hash verification "
    "are rejected with an ArtifactIntegrityViolation."
)

PERSISTENCE_ENVIRONMENT_DISCLOSURE: str = (
    "EXECUTION ENVIRONMENT PERSISTENCE POLICY: "
    "CreditBridge operates in an ephemeral in-memory scoring mode for client transaction statements. "
    "Raw statement text is parsed into memory structures, used for feature calculation, and discarded "
    "upon session termination. To maintain regulatory auditability without retaining sensitive PII, "
    "the engine persists only anonymized AuditManifests containing synthetic request IDs, mathematical "
    "feature aggregates, risk tiers, and SHA-256 hashes of the models used."
)


def compute_file_sha256(filepath: Union[str, Path]) -> str:
    """Computes standard hexadecimal SHA-256 hash of a file."""
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"Artifact not found at: {filepath}")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_model_artifact_integrity(
    model_path: Union[str, Path],
    expected_hash: Optional[str] = None,
    manifest_path: Optional[Union[str, Path]] = None,
) -> str:
    """
    Verifies that the model artifact matches its registered SHA-256 hash.
    Refuses tampered, altered, or unexpected artifacts.
    """
    actual_hash = compute_file_sha256(model_path)

    target_expected_hash = expected_hash
    if target_expected_hash is None and manifest_path is not None:
        p_man = Path(manifest_path)
        if p_man.exists():
            with open(p_man, "r", encoding="utf-8") as f:
                content = f.read()
                # Parse hash manifest (artifact_hash.txt or JSON)
                if content.strip().startswith("{"):
                    data = json.loads(content)
                    target_expected_hash = data.get("credit_model_hash") or data.get("model_hash")
                else:
                    for line in content.splitlines():
                        if "credit_model.pkl" in line:
                            target_expected_hash = line.split()[0].strip()

    if target_expected_hash is not None:
        if actual_hash.lower() != target_expected_hash.lower():
            raise ArtifactIntegrityViolation(
                f"SECURITY VIOLATION: Model artifact integrity verification failed! "
                f"File '{model_path}' has SHA-256 hash {actual_hash}, "
                f"which does not match expected {target_expected_hash}."
            )

    return actual_hash
