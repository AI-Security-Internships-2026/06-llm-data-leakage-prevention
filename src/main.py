"""
main.py — LLM Data Leakage Prevention: FastAPI Entry Point
CNIT/PNTLab Pisa — AI Security Internship 2026
Student: Muhammad Hashim Mughal | Week: 04

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
        "with custom recognisers and adversarial normalisation."
    ),
    version="0.4.0",
)


class DetectRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language: Optional[str] = Field("en")


class BatchDetectRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)
    language: Optional[str] = Field("en")


@app.get("/", tags=["Health"])
def health_check():
    return {"status": "ok", "version": "0.4.0"}


@app.get("/info", tags=["Health"])
def info():
    """Return detector capabilities: entities, risk types, languages, normalisations."""
    return {
        "version": "0.4.0",
        "supported_entities": SUPPORTED_ENTITIES,
        "high_risk_entities": sorted(_HIGH_RISK_TYPES),
        "supported_languages": sorted(_SUPPORTED_LANGUAGES),
        "normalizations": [
            "spaced_credit_card",
            "hyphen_credit_card",
            "dot_credit_card",
            "dot_phone",
            "email_obfuscation",
        ],
    }


@app.post("/detect", tags=["Detection"])
def detect_endpoint(request: DetectRequest):
    """Detect and sanitise PII in a single text string."""
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
    Detect PII in up to 50 texts.
    Each item is processed independently — one failure does not abort the batch.
    Failed items return risk_level='ERROR' with an error message.
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