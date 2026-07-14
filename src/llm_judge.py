"""
llm_judge.py — Stage 2: LLM-as-Judge PII Classifier
CNIT/PNTLab Pisa — AI Security Internship 2026
Student: Muhammad Hashim Mughal | Week: 05

"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# Lazy model holder — loaded once on first call
_pipeline = None
_MODEL_ID  = "facebook/bart-large-mnli"

_LABEL_PII    = "contains personal information"
_LABEL_CLEAN  = "does not contain personal information"
_CANDIDATE_LABELS = [_LABEL_PII, _LABEL_CLEAN]

# Confidence threshold: if the model scores "contains PII" above this, flag it
_CONFIDENCE_THRESHOLD = 0.70

# ── Fallback keyword heuristic ────────────────────────────────────────────────
# Used when the HuggingFace model cannot be loaded.
# Covers patterns Stage 1 is known to miss: inference-based leakage phrasing,
# unusual obfuscation, or borderline cases.
_FALLBACK_PATTERNS = [
    re.compile(r"\bat\b.*\bdot\b", re.IGNORECASE),          # alice at example dot com
    re.compile(r"\b\d{3}\s\d{3}\s\d{4}\b"),                 # 800 555 0199 (spaces)
    re.compile(r"(?:my|his|her|their)\s+(?:email|phone|number|address|ssn|card)\b", re.IGNORECASE),
    re.compile(r"(?:call|reach|contact)\s+(?:me|him|her|them)\s+(?:at|on)\b", re.IGNORECASE),
    re.compile(r"\b(?:social\s+security|taxpayer\s+id|national\s+id)\b", re.IGNORECASE),
    re.compile(r"\b(?:credit|debit)\s+card\b", re.IGNORECASE),
    re.compile(r"\b(?:iban|swift|bic|sort\s+code)\b", re.IGNORECASE),
    re.compile(r"\bpassport\s+(?:number|no\.?)\b", re.IGNORECASE),
    re.compile(r"\bdate\s+of\s+birth\b", re.IGNORECASE),
    re.compile(r"\b(?:home|work|mobile|cell)\s+(?:number|phone|address)\b", re.IGNORECASE),
]


@dataclass
class JudgeResult:
    is_pii: bool
    confidence: float
    reasoning: str          # "zero-shot" | "fallback-keyword" | "fallback-no-match"
    model: Optional[str]    # model ID or None for fallback


def _load_pipeline():
    """Load the zero-shot classification pipeline (once, lazily)."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    try:
        from transformers import pipeline as hf_pipeline
        _pipeline = hf_pipeline(
            "zero-shot-classification",
            model=_MODEL_ID,
            device=-1,          # CPU; set to 0 for GPU
        )
        return _pipeline
    except Exception:
        return None


def _fallback_judge(text: str) -> JudgeResult:
    """Keyword/regex heuristic used when the HuggingFace model is unavailable."""
    for pattern in _FALLBACK_PATTERNS:
        if pattern.search(text):
            return JudgeResult(
                is_pii=True,
                confidence=0.75,
                reasoning="fallback-keyword",
                model=None,
            )
    return JudgeResult(
        is_pii=False,
        confidence=0.60,
        reasoning="fallback-no-match",
        model=None,
    )


def judge_text(text: str) -> JudgeResult:
    """
    Run the Stage 2 LLM-as-judge on *text*.

    Returns a JudgeResult indicating whether the text likely contains PII
    that Stage 1 (Presidio) missed.

    Parameters
    ----------
    text : the original (pre-normalization) text from the caller

    Returns
    -------
    JudgeResult with fields: is_pii, confidence, reasoning, model
    """
    if not text or not text.strip():
        return JudgeResult(is_pii=False, confidence=1.0, reasoning="empty", model=None)

    pipe = _load_pipeline()

    if pipe is None:
        # Model unavailable — use keyword fallback
        return _fallback_judge(text)

    try:
        result = pipe(text, candidate_labels=_CANDIDATE_LABELS, truncation=True)
        # result["labels"][0] is the top-scored label
        scores = dict(zip(result["labels"], result["scores"]))
        pii_score = scores.get(_LABEL_PII, 0.0)
        is_pii = pii_score >= _CONFIDENCE_THRESHOLD
        return JudgeResult(
            is_pii=is_pii,
            confidence=round(pii_score, 4),
            reasoning="zero-shot",
            model=_MODEL_ID,
        )
    except Exception:
        # Model errored mid-call — graceful fallback
        return _fallback_judge(text)


def judge_batch(texts: list[str]) -> list[JudgeResult]:
    """
    Run judge_text on a list of texts.
    Uses HuggingFace batch inference when the model is available.
    """
    if not texts:
        return []

    pipe = _load_pipeline()

    if pipe is None:
        return [_fallback_judge(t) for t in texts]

    try:
        raw_results = pipe(texts, candidate_labels=_CANDIDATE_LABELS, truncation=True)
        out = []
        for result in raw_results:
            scores = dict(zip(result["labels"], result["scores"]))
            pii_score = scores.get(_LABEL_PII, 0.0)
            out.append(JudgeResult(
                is_pii=pii_score >= _CONFIDENCE_THRESHOLD,
                confidence=round(pii_score, 4),
                reasoning="zero-shot",
                model=_MODEL_ID,
            ))
        return out
    except Exception:
        return [_fallback_judge(t) for t in texts]