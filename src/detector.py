"""
detector.py — PII Leakage Detection Probe
CNIT/PNTLab Pisa — AI Security Internship 2026
Student: Muhammad Hashim Mughal | Week: 06

Week 06 — Commit 1
-------------------
Suppress IN_PAN and US_DRIVER_LICENSE from the default entity list.
Presidio's built-in recognisers for these two types produce a high
false-positive rate on generic alphanumeric strings found in real
email bodies (confirmed in Week 05 Enron synthetic eval: 245 IN_PAN
hits, 87 US_DRIVER_LICENSE hits on clean text).
Fix: pass the explicit SUPPORTED_ENTITIES allow-list to
analyzer.analyze() so Presidio only runs the recognisers we trust.
"""

import re

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# ── Custom recognizers ────────────────────────────────────────────────────────

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

# Week 05: threshold 0.75 → 0.65 and significantly expanded context list
# Addresses supervisor feedback: IBAN recall was 0.825 (weakest entity)
_iban_recognizer = PatternRecognizer(
    supported_entity="IBAN_CODE",
    patterns=[
        Pattern(
            name="iban_context_boosted",
            regex=r"\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b",
            score=0.65,
        )
    ],
    context=[
        "iban", "bank account", "account number", "wire", "transfer",
        "beneficiary", "swift", "payment", "remittance", "credit",
        "account", "sort code", "routing", "recipient", "payee",
        "credit transfer", "bic", "sepa", "direct debit", "deposit",
        "bank", "financial", "transaction", "funds", "balance",
    ],
)
analyzer.registry.add_recognizer(_iban_recognizer)

# ── Risk classification sets ──────────────────────────────────────────────────

_HIGH_RISK_TYPES = {
    "CREDIT_CARD", "IBAN_CODE", "MEDICAL_LICENSE",
    "US_SSN", "UK_NHS", "PK_CNIC",
}

_NOISE_TYPES = {"DATE_TIME", "NRP"}

_SUPPORTED_LANGUAGES = {"en"}

# Week 06 Commit 1: explicit entity allow-list passed to analyzer.analyze().
# IN_PAN and US_DRIVER_LICENSE are intentionally excluded — their Presidio
# built-in recognisers fire too often on generic text (ref codes, model IDs,
# employee numbers). Presidio will only run recognisers for types listed here.
SUPPORTED_ENTITIES = [
    "EMAIL_ADDRESS", "CREDIT_CARD", "PHONE_NUMBER", "PERSON",
    "US_SSN", "IBAN_CODE", "PK_CNIC", "LOCATION",
    "MEDICAL_LICENSE", "UK_NHS",
]

# ── Normalization regexes ─────────────────────────────────────────────────────

_SPACED_CARD_RE = re.compile(r"\b(\d{4})[ ](\d{4})[ ](\d{4})[ ](\d{4})\b")
_HYPHEN_CARD_RE = re.compile(r"\b(\d{4})-(\d{4})-(\d{4})-(\d{4})\b")
_DOT_CARD_RE    = re.compile(r"\b(\d{4})\.(\d{4})\.(\d{4})\.(\d{4})\b")

_DOT_PHONE_RE   = re.compile(r"\b(\d{3})\.(\d{3})\.(\d{4})\b")

# Week 05 fix: also handle +44 20 7946 0958 → +442079460958 for Presidio
_UK_PHONE_RE    = re.compile(r"(\+44)\s+(\d{2})\s+(\d{4})\s+(\d{4})\b")

# Week 05 fix: improved to handle alice AT example.com (real dot, no DOT keyword)
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
    Pre-process text to expose PII obscured by formatting or obfuscation
    before passing it to Presidio.

    Transformations applied (in order):
      1. Spaced credit card  → raw digits   4111 1111 1111 1111 → 4111111111111111
      2. Hyphen credit card  → raw digits   4111-1111-1111-1111 → 4111111111111111
      3. Dot credit card     → raw digits   4111.1111.1111.1111 → 4111111111111111
      4. Dot phone           → hyphen phone 800.555.0199        → 800-555-0199
      5. UK spaced phone     → compact      +44 20 7946 0958    → +442079460958
      6. Obfuscated email    → standard     alice [at] example [dot] com
                                            alice(at)example.com
                                            alice AT example DOT com
                                            alice AT example.com
                                                         → alice@example.com
    """
    text = _SPACED_CARD_RE.sub(r"\1\2\3\4", text)
    text = _HYPHEN_CARD_RE.sub(r"\1\2\3\4", text)
    text = _DOT_CARD_RE.sub(r"\1\2\3\4", text)

    text = _DOT_PHONE_RE.sub(r"\1-\2-\3", text)
    text = _UK_PHONE_RE.sub(r"\1\2\3\4", text)

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


def detect_pii(text: str, language: str = "en", use_stage2: bool = False) -> dict:
    """
    Detect PII in *text* and return a structured result.

    Parameters
    ----------
    text       : input string to analyse
    language   : ISO 639-1 language code (only "en" supported)
    use_stage2 : if True, pass MEDIUM/LOW outputs to the LLM-as-judge
                 Stage 2 layer (llm_judge.py) for a second opinion.
                 HIGH outputs skip Stage 2 — already caught by Stage 1.

    Returns
    -------
    dict with keys:
      text           – original input text
      entities       – list of detected entities (type, start, end, score, length)
      risk_level     – CLEAN | LOW | MEDIUM | HIGH
      sanitized      – text with all PII replaced by <REDACTED>
      stage2_used    – bool (only present when use_stage2=True)
      stage2_flagged – bool (only present when use_stage2=True)
    """
    if not text or not text.strip():
        return {"text": text, "entities": [], "risk_level": "CLEAN", "sanitized": text}

    if language not in _SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language '{language}'. Supported: {sorted(_SUPPORTED_LANGUAGES)}"
        )

    normalized = normalize_text(text)
    # Week 06 Commit 1: pass entities= so Presidio only runs allowed recognisers.
    # This is the single line that suppresses IN_PAN and US_DRIVER_LICENSE.
    results = analyzer.analyze(
        text=normalized,
        language=language,
        entities=SUPPORTED_ENTITIES,
    )

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

    risk = _compute_risk(results)
    output = {
        "text": text,
        "entities": entities,
        "risk_level": risk,
        "sanitized": sanitized_text,
    }

    # Stage 2: LLM-as-judge — only for MEDIUM/LOW outputs from Stage 1
    if use_stage2:
        stage2_flagged = False
        if risk in ("MEDIUM", "LOW", "CLEAN"):
            from llm_judge import judge_text  # lazy import — keeps Stage 1 fast
            judge = judge_text(text)
            if judge.is_pii:
                risk = "HIGH"
                output["risk_level"] = "HIGH"
                stage2_flagged = True
        output["stage2_used"] = True
        output["stage2_flagged"] = stage2_flagged

    return output