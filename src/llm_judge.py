"""
llm_judge.py — Stage 2: LLM-as-Judge PII Classifier
CNIT/PNTLab Pisa — AI Security Internship 2026
Student: Muhammad Hashim Mughal | Week: 05

Changes in Week 05 (patch)
--------------------------
- Added 2 missing fallback patterns to catch E23 and E25:
    bank_account_re : "bank account number", "account number"
    cnic_passport_re: "cnic", "passport details", "passport number"
- These two cases were FN in the eval_suite --v2 run
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

_pipeline = None
_MODEL_ID  = "facebook/bart-large-mnli"

_LABEL_PII    = "contains personal information"
_LABEL_CLEAN  = "does not contain personal information"
_CANDIDATE_LABELS = [_LABEL_PII, _LABEL_CLEAN]

_CONFIDENCE_THRESHOLD = 0.70

_FALLBACK_PATTERNS = [
    re.compile(r"\bat\b.*\bdot\b", re.IGNORECASE),
    re.compile(r"\b\d{3}\s\d{3}\s\d{4}\b"),
    re.compile(r"(?:my|his|her|their)\s+(?:email|phone|number|address|ssn|card)\b", re.IGNORECASE),
    re.compile(r"(?:call|reach|contact)\s+(?:me|him|her|them)\s+(?:at|on)\b", re.IGNORECASE),
    re.compile(r"\b(?:social\s+security|taxpayer\s+id|national\s+id)\b", re.IGNORECASE),
    re.compile(r"\b(?:credit|debit)\s+card\b", re.IGNORECASE),
    re.compile(r"\b(?:iban|swift|bic|sort\s+code)\b", re.IGNORECASE),
    re.compile(r"\bpassport\s+(?:number|no\.?|details)\b", re.IGNORECASE),
    re.compile(r"\bdate\s+of\s+birth\b", re.IGNORECASE),
    re.compile(r"\b(?:home|work|mobile|cell)\s+(?:number|phone|address)\b", re.IGNORECASE),
    # Week 05 patch: catch E23 "bank account number on the invoice"
    re.compile(r"\b(?:bank\s+account|account\s+number)\b", re.IGNORECASE),
    # Week 05 patch: catch E25 "CNIC and passport details"
    re.compile(r"\bcnic\b", re.IGNORECASE),
]


@dataclass
class JudgeResult:
    is_pii: bool
    confidence: float
    reasoning: str
    model: Optional[str]


def _load_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    try:
        from transformers import pipeline as hf_pipeline
        _pipeline = hf_pipeline(
            "zero-shot-classification",
            model=_MODEL_ID,
            device=-1,
        )
        return _pipeline
    except Exception:
        return None


def _fallback_judge(text: str) -> JudgeResult:
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
    if not text or not text.strip():
        return JudgeResult(is_pii=False, confidence=1.0, reasoning="empty", model=None)

    pipe = _load_pipeline()

    if pipe is None:
        return _fallback_judge(text)

    try:
        result = pipe(text, candidate_labels=_CANDIDATE_LABELS, truncation=True)
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
        return _fallback_judge(text)


def judge_batch(texts: list[str]) -> list[JudgeResult]:
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