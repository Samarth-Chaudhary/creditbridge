"""
CreditBridge - Alternative Credit Scoring Engine
Real Data Mode: Deterministic Transaction Classifier
Path: src/transaction_classifier.py

Implements transparent, deterministic transaction classification rules according to Part 5 specification.
Stores normalized category, confidence score, rule name, and classification evidence.
Never pretends keyword/rule matching is machine learning.
Prefers UNKNOWN over unjustified classification.
"""

import re
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import pandas as pd

from src.real_data_contracts import (
    NormalizedCategory,
    TransactionType,
)


@dataclass(frozen=True)
class ClassificationResult:
    """Structured result of transaction classification."""
    category: str
    confidence: float
    rule_name: str
    evidence: str


# -----------------------------------------------------------------------------
# DETERMINISTIC REGEX RULES & ENTITY LISTS
# -----------------------------------------------------------------------------

REVERSAL_PATTERN = re.compile(
    r"\b(REV|REVERSAL|FAILED|BOUNCE|RETURN|RET|DECLINED|CANCEL|CHARGEBACK)\b",
    re.IGNORECASE,
)

REFUND_PATTERN = re.compile(
    r"\b(REFUND|RFND|CASHBACK|REIMBURSEMENT|SETTLE-REFUND)\b",
    re.IGNORECASE,
)

INTERNAL_TRANSFER_PATTERN = re.compile(
    r"\b(SELF|OWN\s*ACC?|OWN\s*A/C|TO\s*SELF|SWEEP|AUTOSWEEP|AUTO-SWEEP|MOD\s*TRF|FD\s*CLOSURE|FDR\s*REDEMPTION)\b",
    re.IGNORECASE,
)

LOAN_ENTITIES = [
    "KREDITBEE", "MONEYVIEW", "NAVI", "BAJAJ", "KISHT", "MUTHOOT", "MANAPPURAM",
    "LENDINGKART", "EARLYSALARY", "FIBE", "CASHE", "PAYME", "CREDITVIDYA",
    "LOAN", "DISBURS", "DISBURSEMENT", "FINANCE", "NBFC", "DHANI", "CRED"
]

SALARY_ENTITIES = [
    "SALARY", "PAYROLL", "MONTHLY SAL", "WAGES", "STIPEND", "CMS/", "ECS-CR",
    "SAL/", "SAL TRF", "CORPORATE PAYOUT", "SALARY CREDIT"
]

GIG_ENTITIES = [
    "ZOMATO", "SWIGGY", "UBER", "OLA", "RAPIDO", "BLINKIT", "ZEPTO", "DUNZO",
    "URBAN COMPANY", "SHADOWFAX", "PORTER", "AMAZON FLEX", "FLIPKART LOGISTICS",
    "DELHIVERY", "EKART", "LOADSHARE", "BIGBASKET BBNOPL"
]

BUSINESS_ENTITIES = [
    "BHARATPE", "PAYTM QR", "RAZORPAY", "PINELABS", "STRIPE", "CASHFREE",
    "CCAVENUE", "INSTAMOJO", "POS CR", "MERCHANT SETTLE", "QR SETTLEMENT",
    "BILLING SETTLE", "DAILY SETTLEMENT", "EDC SETTLEMENT"
]

TELCO_ENTITIES = [
    "JIO", "AIRTEL", "VI ", "VODAFONE", "IDEA", "BSNL", "MTNL", "RECHARGE",
    "PREPAID", "MOBIKWIK RECHARGE", "FREECHARGE", "AIRTEL PAYMENTS BANK RECH"
]

UTILITY_ENTITIES = [
    "BESCOM", "MSEDCL", "TATA POWER", "ADANI ELEC", "TORRENT POWER", "PSPCL",
    "BSES", "UPPCL", "WBSEDCL", "DHBVN", "UHBVN", "IGL", "MAHANAGAR GAS", "GAIL",
    "GUJARAT GAS", "BWSSB", "DJB", "BILLDESK", "BBPS", "ELECTRICITY", "WATER BOARD",
    "PIPED GAS", "MUNICIPAL", "PROPERTY TAX", "MAHAVITARAN"
]

COMMERCIAL_MERCHANT_ENTITIES = [
    "AMAZON", "FLIPKART", "MYNTRA", "AJIO", "NYKAA", "DMART", "RELIANCE RETAIL",
    "MORE RETAIL", "BIG BAZAAR", "SPENCERS", "APOLLO PHARMACY", "MEDPLUS",
    "NETMEDS", "PHARMEASY", "TATA 1MG", "MAKEMYTRIP", "IRCTC", "BOOKMYSHOW",
    "PVR", "INOX", "DECATHLON", "IKEA", "LENSKART", "DOMINOS", "MCDONALD",
    "KFC", "PIZZA HUT", "STARBUCKS", "CCD"
]


def classify_transaction(
    narration: Optional[str],
    counterparty: Optional[str] = None,
    amount: float = 0.0,
    txn_type: str = TransactionType.UNKNOWN.value,
    account_holder_name: Optional[str] = None,
) -> ClassificationResult:
    """
    Deterministically classifies a single transaction with explicit rule auditability.
    
    Priority Hierarchy:
    1. Reversals
    2. Refunds / Cashbacks
    3. Self-transfers / Sweeps
    4. Loans / Lending Disbursals
    5. Telecom Recharges
    6. Utility Payments
    7. Gig Income Payouts
    8. Salary / Payroll Credits
    9. Business QR / Settlement Inflows
    10. Commercial Merchant Debits
    11. Person-like / P2P Transfers
    12. Unknown (safe conservative fallback)
    """
    text = f"{counterparty or ''} {narration or ''}".upper().strip()

    if not text:
        return ClassificationResult(
            category=NormalizedCategory.UNKNOWN.value,
            confidence=0.0,
            rule_name="RULE_EMPTY_NARRATION",
            evidence="No text evidence present in narration or counterparty",
        )

    # 1. Reversals
    if txn_type == TransactionType.REVERSAL.value or REVERSAL_PATTERN.search(text):
        matched = REVERSAL_PATTERN.search(text)
        token = matched.group(0) if matched else "REVERSAL_TYPE"
        return ClassificationResult(
            category=NormalizedCategory.REFUND.value,
            confidence=0.98,
            rule_name="RULE_REVERSAL_DETECTED",
            evidence=f"Reversal keyword or type detected: '{token}'",
        )

    # 2. Refunds / Reimbursements
    if txn_type == TransactionType.REFUND.value or REFUND_PATTERN.search(text):
        matched = REFUND_PATTERN.search(text)
        token = matched.group(0) if matched else "REFUND_TYPE"
        return ClassificationResult(
            category=NormalizedCategory.REFUND.value,
            confidence=0.95,
            rule_name="RULE_REFUND_DETECTED",
            evidence=f"Refund or cashback keyword detected: '{token}'",
        )

    matched_int = INTERNAL_TRANSFER_PATTERN.search(text)
    if matched_int:
        token = matched_int.group(0)
        return ClassificationResult(
            category=NormalizedCategory.TRANSFER.value,
            confidence=0.95,
            rule_name="RULE_INTERNAL_TRANSFER_KEYWORD",
            evidence=f"Internal transfer / sweep keyword detected: '{token}'",
        )

    if account_holder_name:
        clean_holder = account_holder_name.upper().strip()
        if clean_holder and len(clean_holder) >= 3 and clean_holder in text:
            return ClassificationResult(
                category=NormalizedCategory.TRANSFER.value,
                confidence=0.92,
                rule_name="RULE_SELF_TRANSFER_NAME_MATCH",
                evidence=f"Counterparty contains account holder name: '{clean_holder}'",
            )

    # 4. Loan Disbursal or Repayment
    for loan_kw in LOAN_ENTITIES:
        if loan_kw in text:
            return ClassificationResult(
                category=NormalizedCategory.LOAN_RELATED.value,
                confidence=0.90,
                rule_name="RULE_LOAN_ENTITY_MATCH",
                evidence=f"Matched lending/loan entity keyword: '{loan_kw}'",
            )

    # 5. Telecom Recharge
    for telco in TELCO_ENTITIES:
        if telco in text:
            return ClassificationResult(
                category=NormalizedCategory.TELECOM_RECHARGE.value,
                confidence=0.95,
                rule_name="RULE_TELCO_ENTITY_MATCH",
                evidence=f"Matched telecom operator/recharge keyword: '{telco}'",
            )

    # 6. Utility Bill Payment
    for util in UTILITY_ENTITIES:
        if util in text:
            return ClassificationResult(
                category=NormalizedCategory.UTILITY.value,
                confidence=0.95,
                rule_name="RULE_UTILITY_ENTITY_MATCH",
                evidence=f"Matched electricity/water/gas/billdesk provider: '{util}'",
            )

    # 7. Gig Worker Payouts
    for gig in GIG_ENTITIES:
        if gig in text:
            if txn_type == TransactionType.CREDIT.value:
                return ClassificationResult(
                    category=NormalizedCategory.GIG_INCOME_LIKE.value,
                    confidence=0.92,
                    rule_name="RULE_GIG_INFLOW_MATCH",
                    evidence=f"Matched gig platform credit payout: '{gig}'",
                )
            else:
                return ClassificationResult(
                    category=NormalizedCategory.MERCHANT_SPEND.value,
                    confidence=0.90,
                    rule_name="RULE_GIG_MERCHANT_DEBIT",
                    evidence=f"Matched gig platform customer order debit: '{gig}'",
                )

    # 8. Salary Credits
    for sal in SALARY_ENTITIES:
        if sal in text:
            if txn_type == TransactionType.CREDIT.value:
                return ClassificationResult(
                    category=NormalizedCategory.SALARY_LIKE.value,
                    confidence=0.92,
                    rule_name="RULE_SALARY_CREDIT_MATCH",
                    evidence=f"Matched corporate payroll / salary credit marker: '{sal}'",
                )

    # 9. Business Merchant QR / Settlement Inflows
    for biz in BUSINESS_ENTITIES:
        if biz in text:
            return ClassificationResult(
                category=NormalizedCategory.BUSINESS_INFLOW.value,
                confidence=0.90,
                rule_name="RULE_BUSINESS_QR_MATCH",
                evidence=f"Matched merchant acquiring / settlement gateway: '{biz}'",
            )

    # 10. Commercial Merchant Debits
    for merch in COMMERCIAL_MERCHANT_ENTITIES:
        if merch in text:
            return ClassificationResult(
                category=NormalizedCategory.MERCHANT_SPEND.value,
                confidence=0.88,
                rule_name="RULE_COMMERCIAL_MERCHANT_MATCH",
                evidence=f"Matched consumer merchant: '{merch}'",
            )

    # 11. Generic Merchant or P2P patterns
    if re.search(r"\b(MERCHANT|STORE|SHOP|RETAIL|BAZAAR|MART|PHARMACY)\b", text):
        return ClassificationResult(
            category=NormalizedCategory.MERCHANT_SPEND.value,
            confidence=0.80,
            rule_name="RULE_GENERIC_MERCHANT_KEYWORD",
            evidence="Matched generic retail merchant vocabulary",
        )

    if re.search(r"\b(P2P|TRANSFER|TRF|IMPS|NEFT|RTGS)\b", text) or "@" in text:
        return ClassificationResult(
            category=NormalizedCategory.PERSON_LIKE.value,
            confidence=0.70,
            rule_name="RULE_P2P_TRANSFER_PATTERN",
            evidence="Matched peer-to-peer VPA or bank transfer pattern",
        )

    # 12. Unknown Fallback
    return ClassificationResult(
        category=NormalizedCategory.UNKNOWN.value,
        confidence=0.0,
        rule_name="RULE_UNKNOWN_NO_MATCH",
        evidence="No reliable deterministic rule matched this counterparty or narration",
    )


def classify_transactions_df(
    df: pd.DataFrame,
    account_holder_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Applies deterministic classification over an entire transaction DataFrame,
    adding normalized_category, classification_confidence, rule_name, and rule_evidence
    without mutating existing raw columns.
    """
    res_df = df.copy()
    categories = []
    confidences = []
    rule_names = []
    evidences = []

    for _, row in res_df.iterrows():
        cparty = row.get("counterparty")
        narr = row.get("raw_category")
        raw_amt = row.get("amount")
        amt = float(raw_amt) if raw_amt is not None and not pd.isna(raw_amt) else 0.0
        ttype = str(row.get("transaction_type") or TransactionType.UNKNOWN.value)

        res = classify_transaction(
            narration=narr,
            counterparty=cparty,
            amount=amt,
            txn_type=ttype,
            account_holder_name=account_holder_name,
        )
        categories.append(res.category)
        confidences.append(res.confidence)
        rule_names.append(res.rule_name)
        evidences.append(res.evidence)

    res_df["normalized_category"] = categories
    res_df["classification_confidence"] = confidences
    res_df["classification_rule"] = rule_names
    res_df["classification_evidence"] = evidences
    return res_df
