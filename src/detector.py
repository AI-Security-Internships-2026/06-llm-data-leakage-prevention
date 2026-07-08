"""
detector.py — PII Leakage Detection Probe
CNIT/PNTLab Pisa — AI Security Internship 2026
Student: Muhammad Hashim Mughal | Week: 03
Changes (Week 03):
  - Added custom PatternRecognizer for Pakistani CNIC (PK_CNIC)
  - Removed text_slice from entity output (was echoing raw PII back to caller)
  - Fixed risk logic: count-based HIGH now also requires a score threshold
Changes (Week 03 patch):
  - Added custom US_SSN recognizer with stronger context to beat PHONE_NUMBER overlap
  - Excluded DATE_TIME and NRP from risk scoring (too noisy, not personal identifiers)
"""

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

_HIGH_RISK_TYPES = {
    "CREDIT_CARD", "IBAN_CODE", "MEDICAL_LICENSE",
    "US_SSN", "UK_NHS", "PK_CNIC",
}

_NOISE_TYPES = {"DATE_TIME", "NRP"}

_SUPPORTED_LANGUAGES = {"en"}


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
    if not text or not text.strip():
        return {"text": text, "entities": [], "risk_level": "CLEAN", "sanitized": text}

    if language not in _SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language '{language}'. Supported: {sorted(_SUPPORTED_LANGUAGES)}"
        )

    results = analyzer.analyze(text=text, language=language)

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

    sanitized_text = text
    if results:
        anonymized = anonymizer.anonymize(
            text=text,
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