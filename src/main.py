"""
main.py — LLM Data Leakage Prevention: FastAPI Entry Point
CNIT/PNTLab Pisa — AI Security Internship 2026
Student : Muhammad Hashim Mughal | Week: 02
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from detector import detect_pii

app = FastAPI(
    title="LLM Data Leakage Prevention API",
    description="Detects and sanitises PII in LLM output using Microsoft Presidio.",
    version="0.2.0",
)

class DetectRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language: Optional[str] = Field("en")

class BatchDetectRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)
    language: Optional[str] = Field("en")

@app.get("/", tags=["Health"])
def health_check():
    return {"status": "ok", "version": "0.2.0"}

@app.post("/detect", tags=["Detection"])
def detect_endpoint(request: DetectRequest):
    try:
        result = detect_pii(text=request.text, language=request.language)
        result["entity_count"] = len(result["entities"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/batch", tags=["Detection"])
def detect_batch_endpoint(request: BatchDetectRequest):
    if len(request.texts) > 50:
        raise HTTPException(status_code=400, detail="Batch size limit is 50.")
    try:
        results = []
        for text in request.texts:
            r = detect_pii(text=text, language=request.language)
            r["entity_count"] = len(r["entities"])
            results.append(r)
        return {"results": results, "total": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))