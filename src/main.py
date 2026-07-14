"""
main.py — LLM Data Leakage Prevention: FastAPI Entry Point
CNIT/PNTLab Pisa — AI Security Internship 2026
Student: Muhammad Hashim Mughal | Week: 05

"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from detector import detect_pii, SUPPORTED_ENTITIES, _HIGH_RISK_TYPES, _SUPPORTED_LANGUAGES

app = FastAPI(
    title="LLM Data Leakage Prevention API",
    description=(
        "Detects and sanitises PII in LLM output using Microsoft Presidio "
        "with custom recognisers, adversarial normalisation, and an optional "
        "LLM-as-judge Stage 2 layer (facebook/bart-large-mnli)."
    ),
    version="0.5.0",
)


# ── Request / Response models ─────────────────────────────────────────────────

class DetectRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language: Optional[str] = Field("en")


class DetectV2Request(BaseModel):
    text: str = Field(..., min_length=1)
    language: Optional[str] = Field("en")


class BatchDetectRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)
    language: Optional[str] = Field("en")


# ── Health / Info ─────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def health_check():
    return {"status": "ok", "version": "0.5.0"}


@app.get("/info", tags=["Health"])
def info():
    """Return Stage 1 detector capabilities."""
    return {
        "version": "0.5.0",
        "supported_entities": SUPPORTED_ENTITIES,
        "high_risk_entities": sorted(_HIGH_RISK_TYPES),
        "supported_languages": sorted(_SUPPORTED_LANGUAGES),
        "normalizations": [
            "spaced_credit_card",
            "hyphen_credit_card",
            "dot_credit_card",
            "dot_phone",
            "uk_spaced_phone",
            "email_obfuscation",
        ],
    }


@app.get("/info/v2", tags=["Health"])
def info_v2():
    """Return Stage 1 + Stage 2 (LLM-as-judge) pipeline capabilities."""
    return {
        "version": "0.5.0",
        "pipeline": "two-stage",
        "stage1": {
            "engine": "Microsoft Presidio",
            "supported_entities": SUPPORTED_ENTITIES,
            "normalizations": [
                "spaced_credit_card", "hyphen_credit_card", "dot_credit_card",
                "dot_phone", "uk_spaced_phone", "email_obfuscation",
            ],
        },
        "stage2": {
            "engine": "HuggingFace zero-shot classification",
            "model": "facebook/bart-large-mnli",
            "fallback": "keyword-heuristic (if model unavailable)",
            "triggered_on": ["MEDIUM", "LOW", "CLEAN"],
            "skipped_on": ["HIGH"],
            "confidence_threshold": 0.70,
        },
    }


# ── Stage 1 only: /detect ─────────────────────────────────────────────────────

@app.post("/detect", tags=["Detection"])
def detect_endpoint(request: DetectRequest):
    """Detect and sanitise PII using Stage 1 (Presidio) only."""
    try:
        result = detect_pii(text=request.text, language=request.language)
        result["entity_count"] = len(result["entities"])
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/batch", tags=["Detection"])
def detect_batch_endpoint(request: BatchDetectRequest):
    """
    Detect PII in up to 50 texts (Stage 1 only).
    Each item is processed independently — one failure does not abort the batch.
    """
    if len(request.texts) > 50:
        raise HTTPException(status_code=400, detail="Batch size limit is 50.")

    results = []
    error_count = 0

    for text in request.texts:
        try:
            r = detect_pii(text=text, language=request.language)
            r["entity_count"] = len(r["entities"])
            r["error"] = None
            results.append(r)
        except Exception as e:
            error_count += 1
            results.append({
                "text": text,
                "entities": [],
                "risk_level": "ERROR",
                "sanitized": text,
                "entity_count": 0,
                "error": str(e),
            })

    return {
        "results": results,
        "total": len(results),
        "errors": error_count,
    }


# ── Stage 1 + Stage 2: /detect/v2 ────────────────────────────────────────────

@app.post("/detect/v2", tags=["Detection v2"])
def detect_v2_endpoint(request: DetectV2Request):
    """
    Two-stage detection:
      Stage 1 — Presidio + custom recognisers (always runs, fast)
      Stage 2 — LLM-as-judge (runs only when Stage 1 returns MEDIUM/LOW/CLEAN)

    Additional response fields vs /detect:
      stage2_used    — bool: whether Stage 2 was invoked
      stage2_flagged — bool: whether Stage 2 escalated the risk to HIGH
    """
    try:
        result = detect_pii(
            text=request.text,
            language=request.language,
            use_stage2=True,
        )
        result["entity_count"] = len(result["entities"])
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))