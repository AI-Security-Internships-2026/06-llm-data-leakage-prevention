"""
detector.py — PII Leakage Detection Probe
CNIT/PNTLab Pisa — AI Security Internship 2026
Student: Muhammad Hashim Mughal | Week: 04

"""

import re

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

_cnic_recognizer = PatternRecognizer(
    supported_entity="PK_CNIC",
    patterns=[
        Pattern(name="pk_cnic", regex=r"\b\d{5}-\d{7}-\d\b", score=0.85)
    ],
    context=["cnic", "national identity", "identity card", "nadra"],
)
analyzer.registry.add_recognizer(_cnic_recognizer)

_ssn_recognizer = PatternRecognizer(
    supported_entity="US_SSN",
    patterns=[
        Pattern(
            name="us_ssn_with_context",
            regex=r"\b\d{3}-\d{2}-\d{4}\b",
            score=0.9,
        )
    ],
    context=["ssn", "social security", "social security number", "taxpayer"],
)
analyzer.registry.add_recognizer(_ssn_recognizer)

_iban_recognizer = PatternRecognizer(
    supported_entity="IBAN_CODE",
    patterns=[
        Pattern(
            name="iban_context_boosted",
            regex=r"\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b",
            score=0.75,
        )
    ],
    context=[
        "iban", "bank account", "account number", "wire", "transfer",
        "beneficiary", "swift", "payment", "remittance", "credit",
    ],
)
analyzer.registry.add_recognizer(_iban_recognizer)

_HIGH_RISK_TYPES = {
    "CREDIT_CARD", "IBAN_CODE", "MEDICAL_LICENSE",
    "US_SSN", "UK_NHS", "PK_CNIC",
}

_NOISE_TYPES = {"DATE_TIME", "NRP"}

_SUPPORTED_LANGUAGES = {"en"}

SUPPORTED_ENTITIES = [
    "EMAIL_ADDRESS", "CREDIT_CARD", "PHONE_NUMBER", "PERSON",
    "US_SSN", "IBAN_CODE", "PK_CNIC", "LOCATION",
    "MEDICAL_LICENSE", "UK_NHS",
]

_SPACED_CARD_RE  = re.compile(r"\b(\d{4})[ ](\d{4})[ ](\d{4})[ ](\d{4})\b")
_HYPHEN_CARD_RE  = re.compile(r"\b(\d{4})-(\d{4})-(\d{4})-(\d{4})\b")
_DOT_CARD_RE     = re.compile(r"\b(\d{4})\.(\d{4})\.(\d{4})\.(\d{4})\b")

_DOT_PHONE_RE    = re.compile(r"\b(\d{3})\.(\d{3})\.(\d{4})\b")

_OBFUSC_EMAIL_RE = re.compile(
    r"([\w.+\-]+?)"
    r"\s*(?:\[at\]|\(at\)|\bAT\b)\s*"
    r"([\w\-]+)"
    r"(?:\s*(?:\[dot\]|\bDOT\b)\s*|\.)"
    r"(\w{2,6})",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    """
    Pre-process text to expose PII that is obscured by formatting or obfuscation
    before passing it to Presidio.

    Transformations applied (in order):
      1. Spaced credit card  → raw digits   4111 1111 1111 1111 → 4111111111111111
      2. Hyphen credit card  → raw digits   4111-1111-1111-1111 → 4111111111111111
      3. Dot credit card     → raw digits   4111.1111.1111.1111 → 4111111111111111
      4. Dot phone           → hyphen phone 800.555.0199        → 800-555-0199
      5. Obfuscated email    → standard     alice [at] example [dot] com
                                            alice(at)example.com
                                            alice AT example DOT com
                                                         → alice@example.com
    """
    text = _SPACED_CARD_RE.sub(r"\1\2\3\4", text)
    text = _HYPHEN_CARD_RE.sub(r"\1\2\3\4", text)
    text = _DOT_CARD_RE.sub(r"\1\2\3\4", text)

    text = _DOT_PHONE_RE.sub(r"\1-\2-\3", text)

    text = _OBFUSC_EMAIL_RE.sub(
        lambda m: f"{m.group(1)}@{m.group(2)}.{m.group(3)}", text
    )

    return text


def _compute_risk(results) -> str:
    signal = [r for r in results if r.entity_type not in _NOISE_TYPES]

    if not signal:
        return "CLEAN"
    types = {r.entity_type for r in signal}
    if types & _HIGH_RISK_TYPES:
        return "HIGH"
    if len(signal) >= 3 and any(r.score >= 0.7 for r in signal):
        return "HIGH"
    if any(r.score >= 0.7 for r in signal):
        return "MEDIUM"
    return "LOW"


def detect_pii(text: str, language: str = "en") -> dict:
    """
    Detect PII in *text* and return a structured result.

    Returns
    -------
    dict with keys:
      text        – original input text
      entities    – list of detected entities (type, start, end, score, length)
      risk_level  – CLEAN | LOW | MEDIUM | HIGH
      sanitized   – text with all PII replaced by <REDACTED>
    """
    if not text or not text.strip():
        return {"text": text, "entities": [], "risk_level": "CLEAN", "sanitized": text}

    if language not in _SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language '{language}'. Supported: {sorted(_SUPPORTED_LANGUAGES)}"
        )

    normalized = normalize_text(text)
    results = analyzer.analyze(text=normalized, language=language)

    entities = [
        {
            "type": r.entity_type,
            "start": r.start,
            "end": r.end,
            "score": round(r.score, 3),
            "length": r.end - r.start,
        }
        for r in results
    ]

    sanitized_text = normalized
    if results:
        anonymized = anonymizer.anonymize(
            text=normalized,
            analyzer_results=results,
            operators={"DEFAULT": OperatorConfig("replace", {"new_value": "<REDACTED>"})},
        )
        sanitized_text = anonymized.text

    return {
        "text": text,
        "entities": entities,
        "risk_level": _compute_risk(results),
        "sanitized": sanitized_text,
    }