"""
kv_attack.risk_signal
======================
Issue #7 — Structured privacy-risk signal for cache policy decisions.

Converts the raw output of src/detector.detect_pii() into a rich
RiskSignal dataclass that the cache policy engine (cache_policy.py)
can act on deterministically.

New fields vs existing detector output
---------------------------------------
  entity_types        – deduplicated list of detected entity type strings
  pii_flag            – True if any non-noise PII entity was detected
  detector_confidence – max Presidio score across all detected entities (0–1)
  semantic_sensitivity – domain tag: NONE | CONTACT | MEDICAL | FINANCIAL | ENTERPRISE
  risk_score          – continuous score in [0.0, 1.0]
  risk_level          – LOW | MEDIUM | HIGH  (CLEAN is normalised to LOW=0.0)
  reason              – human-readable explanation of the decision

Usage
-----
    from kv_attack.risk_signal import analyze_prompt, build_risk_signal
    signal = analyze_prompt("Patient Mary Johnson has diabetes.")
    print(signal.risk_level)          # HIGH
    print(signal.semantic_sensitivity) # MEDICAL
    print(signal.risk_score)           # 0.93
"""
from __future__ import annotations

import sys
import os
from dataclasses import dataclass
from typing import List


SENSITIVITY_NONE       = "NONE"
SENSITIVITY_CONTACT    = "CONTACT"
SENSITIVITY_MEDICAL    = "MEDICAL"
SENSITIVITY_FINANCIAL  = "FINANCIAL"
SENSITIVITY_ENTERPRISE = "ENTERPRISE"

_ENTITY_DOMAIN: dict[str, str] = {
    "PERSON":          SENSITIVITY_CONTACT,
    "EMAIL_ADDRESS":   SENSITIVITY_CONTACT,
    "PHONE_NUMBER":    SENSITIVITY_CONTACT,
    "LOCATION":        SENSITIVITY_CONTACT,
    "CREDIT_CARD":     SENSITIVITY_FINANCIAL,
    "IBAN_CODE":       SENSITIVITY_FINANCIAL,
    "US_SSN":          SENSITIVITY_FINANCIAL,
    "US_BANK_NUMBER":  SENSITIVITY_FINANCIAL,
    "PK_CNIC":         SENSITIVITY_FINANCIAL,
    "MEDICAL_LICENSE": SENSITIVITY_MEDICAL,
    "UK_NHS":          SENSITIVITY_MEDICAL,
}

_DOMAIN_PRIORITY = [
    SENSITIVITY_NONE,
    SENSITIVITY_CONTACT,
    SENSITIVITY_ENTERPRISE,
    SENSITIVITY_MEDICAL,
    SENSITIVITY_FINANCIAL,
]

_NOISE_TYPES = {"DATE_TIME", "NRP"}



@dataclass
class RiskSignal:
    """
    Structured privacy-risk signal produced by analyze_prompt().

    All fields are read-only after construction. Serialise with to_dict().
    """
    entity_types:          List[str]
    pii_flag:              bool
    detector_confidence:   float
    semantic_sensitivity:  str
    risk_score:            float
    risk_level:            str
    reason:                str

    def to_dict(self) -> dict:
        return {
            "entity_types":         self.entity_types,
            "pii_flag":             self.pii_flag,
            "detector_confidence":  round(self.detector_confidence, 4),
            "semantic_sensitivity": self.semantic_sensitivity,
            "risk_score":           round(self.risk_score, 4),
            "risk_level":           self.risk_level,
            "reason":               self.reason,
        }



def _infer_semantic_sensitivity(entity_types: list[str]) -> str:
    """Return the highest-priority semantic domain for the detected entity types."""
    if not entity_types:
        return SENSITIVITY_NONE

    domains_found: set[str] = set()
    for et in entity_types:
        domain = _ENTITY_DOMAIN.get(et)
        if domain:
            domains_found.add(domain)

    if "PERSON" in entity_types and "LOCATION" in entity_types:
        domains_found.add(SENSITIVITY_ENTERPRISE)

    if not domains_found:
        return SENSITIVITY_CONTACT

    best = SENSITIVITY_NONE
    for domain in _DOMAIN_PRIORITY:
        if domain in domains_found:
            best = domain
    return best


def _compute_risk_score(
    entity_types:       list[str],
    detector_confidence: float,
    risk_level:         str,
) -> float:
    """
    Map (entity_types, max_confidence, risk_level) to a continuous [0, 1] score.

    Band anchors (base + confidence × width):
      CLEAN  → 0.00            (no PII)
      LOW    → 0.10 – 0.25
      MEDIUM → 0.35 – 0.55
      HIGH   → 0.65 – 1.00
    """
    level = risk_level.upper()
    if level in ("CLEAN", "LOW") and not entity_types:
        return 0.0
    if level == "CLEAN":
        return round(0.10 * detector_confidence, 4)
    elif level == "LOW":
        base, width = 0.10, 0.15
    elif level == "MEDIUM":
        base, width = 0.35, 0.20
    else:
        base, width = 0.65, 0.35

    return round(min(base + width * detector_confidence, 1.0), 4)


def _build_reason(
    entity_types:         list[str],
    semantic_sensitivity: str,
    risk_level:           str,
    detector_confidence:  float,
) -> str:
    """Build a human-readable explanation for the risk decision."""
    if not entity_types:
        return "No PII entities detected; prompt classified as clean."

    shown = entity_types[:5]
    suffix = f" (+{len(entity_types)-5} more)" if len(entity_types) > 5 else ""
    entity_str = ", ".join(shown) + suffix
    conf_pct   = round(detector_confidence * 100, 1)

    level_word = {
        "CLEAN":  "clean",
        "LOW":    "low-risk",
        "MEDIUM": "medium-risk",
        "HIGH":   "high-risk",
    }.get(risk_level.upper(), risk_level.lower())

    return (
        f"Detected {semantic_sensitivity} PII [{entity_str}] "
        f"at {conf_pct}% confidence; classified as {level_word}."
    )



def build_risk_signal(detect_output: dict) -> RiskSignal:
    """
    Build a RiskSignal from the output of detector.detect_pii().

    Parameters
    ----------
    detect_output : dict
        Return value of src/detector.detect_pii().  Must contain keys:
        'entities' (list of dicts with 'type' and 'score') and 'risk_level' (str).

    Returns
    -------
    RiskSignal
    """
    entities   = detect_output.get("entities", [])
    risk_level = detect_output.get("risk_level", "CLEAN")

    signal_entities = [e for e in entities if e.get("type") not in _NOISE_TYPES]

    raw_types           = [e["type"] for e in signal_entities]
    seen: set[str]      = set()
    entity_types        = [t for t in raw_types if not (t in seen or seen.add(t))]

    pii_flag             = len(entity_types) > 0
    detector_confidence  = max((e["score"] for e in signal_entities), default=0.0)
    semantic_sensitivity = _infer_semantic_sensitivity(entity_types)

    normalised_level = risk_level if risk_level.upper() != "CLEAN" else "LOW"
    risk_score       = _compute_risk_score(entity_types, detector_confidence, risk_level)

    if not pii_flag:
        normalised_level = "LOW"
        risk_score       = 0.0

    reason = _build_reason(entity_types, semantic_sensitivity, normalised_level, detector_confidence)

    return RiskSignal(
        entity_types         = entity_types,
        pii_flag             = pii_flag,
        detector_confidence  = detector_confidence,
        semantic_sensitivity = semantic_sensitivity,
        risk_score           = risk_score,
        risk_level           = normalised_level,
        reason               = reason,
    )


def analyze_prompt(text: str, use_stage2: bool = False) -> RiskSignal:
    """
    Convenience wrapper: run detector.detect_pii() and return a RiskSignal.

    Parameters
    ----------
    text        : prompt text to analyse
    use_stage2  : if True, pass MEDIUM/LOW through LLM-as-judge (slower)

    Returns
    -------
    RiskSignal
    """
    _src = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _src not in sys.path:
        sys.path.insert(0, _src)

    from detector import detect_pii
    raw = detect_pii(text, use_stage2=use_stage2)
    return build_risk_signal(raw)