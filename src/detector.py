"""
detector.py — PII Leakage Detection Probe
CNIT/PNTLab Pisa — AI Security Internship 2026
Author: Muhammad Hashim Mughal | Week: 02
"""

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

_HIGH_RISK_TYPES = {"CREDIT_CARD", "IBAN_CODE", "MEDICAL_LICENSE", "US_SSN", "UK_NHS"}

def _compute_risk(results) -> str:
    if not results:
        return "CLEAN"
    types = {r.entity_type for r in results}
    if types & _HIGH_RISK_TYPES or len(results) >= 3:
        return "HIGH"
    if any(r.score >= 0.7 for r in results):
        return "MEDIUM"
    return "LOW"

def detect_pii(text: str, language: str = "en") -> dict:
    if not text or not text.strip():
        return {"text": text, "entities": [], "risk_level": "CLEAN", "sanitized": text}

    results = analyzer.analyze(text=text, language=language)

    entities = [
        {
            "type": r.entity_type,
            "start": r.start,
            "end": r.end,
            "score": round(r.score, 3),
            "text_slice": text[r.start:r.end],
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