"""
test_pipeline_v2.py — Integration tests for the two-stage v2 pipeline
CNIT/PNTLab Pisa — AI Security Internship 2026
Student: Muhammad Hashim Mughal | Week: 05

"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from main import app
from llm_judge import JudgeResult

client = TestClient(app)


# ── /info/v2 ─────────────────────────────────────────────────────────────────

class TestInfoV2:
    def test_info_v2_status_200(self):
        r = client.get("/info/v2")
        assert r.status_code == 200

    def test_info_v2_has_pipeline_key(self):
        r = client.get("/info/v2")
        assert r.json()["pipeline"] == "two-stage"

    def test_info_v2_has_stage1_and_stage2(self):
        data = client.get("/info/v2").json()
        assert "stage1" in data
        assert "stage2" in data

    def test_info_v2_stage2_model_listed(self):
        data = client.get("/info/v2").json()
        assert "bart-large-mnli" in data["stage2"]["model"]


# ── /detect/v2 response schema ────────────────────────────────────────────────

class TestDetectV2Schema:
    def test_returns_200(self):
        r = client.post("/detect/v2", json={"text": "hello world"})
        assert r.status_code == 200

    def test_has_standard_keys(self):
        r = client.post("/detect/v2", json={"text": "hello world"})
        data = r.json()
        for key in ("text", "entities", "risk_level", "sanitized", "entity_count"):
            assert key in data, f"Missing key: {key}"

    def test_has_stage2_keys(self):
        r = client.post("/detect/v2", json={"text": "hello world"})
        data = r.json()
        assert "stage2_used" in data
        assert "stage2_flagged" in data

    def test_stage2_used_is_bool(self):
        r = client.post("/detect/v2", json={"text": "hello world"})
        assert isinstance(r.json()["stage2_used"], bool)

    def test_stage2_flagged_is_bool(self):
        r = client.post("/detect/v2", json={"text": "hello world"})
        assert isinstance(r.json()["stage2_flagged"], bool)

    def test_empty_text_returns_422(self):
        r = client.post("/detect/v2", json={"text": ""})
        assert r.status_code == 422


# ── Stage 2 skip for HIGH risk ────────────────────────────────────────────────

class TestStage2SkippedForHighRisk:
    def test_credit_card_stage2_not_flagged(self):
        r = client.post("/detect/v2", json={"text": "Card: 4111111111111111"})
        data = r.json()
        assert data["risk_level"] == "HIGH"
        assert data["stage2_flagged"] is False

    def test_ssn_stage2_not_flagged(self):
        r = client.post("/detect/v2", json={"text": "SSN: 078-05-1120"})
        data = r.json()
        assert data["risk_level"] == "HIGH"
        assert data["stage2_flagged"] is False

    def test_iban_stage2_not_flagged(self):
        r = client.post(
            "/detect/v2",
            json={"text": "Wire to IBAN GB29NWBK60161331926819"}
        )
        data = r.json()
        assert data["risk_level"] == "HIGH"
        assert data["stage2_flagged"] is False


# ── Stage 2 invoked for CLEAN / LOW ──────────────────────────────────────────

class TestStage2InvokedForLowRisk:
    def test_clean_text_stage2_used(self):
        r = client.post(
            "/detect/v2",
            json={"text": "The pipeline runs every 6 hours."}
        )
        assert r.json()["stage2_used"] is True

    def test_clean_text_stage2_not_flagged_by_default(self):
        """Clean technical text: fallback heuristic should NOT flag it."""
        r = client.post(
            "/detect/v2",
            json={"text": "Gradient descent converged after 150 epochs."}
        )
        data = r.json()
        assert data["stage2_used"] is True
        assert data["stage2_flagged"] is False


# ── Stage 2 escalation ────────────────────────────────────────────────────────

class TestStage2Escalation:
    def test_inference_pii_escalated_by_stage2(self):
        """
        'Please use the iban I provided earlier for the transfer'
        → Stage 1: CLEAN (no literal IBAN present)
        → Stage 2 fallback keyword 'iban' → flags as PII → escalated to HIGH
        """
        r = client.post(
            "/detect/v2",
            json={"text": "Please use the iban I provided earlier for the transfer"}
        )
        data = r.json()
        assert data["stage2_used"] is True
        assert data["stage2_flagged"] is True
        assert data["risk_level"] == "HIGH"

    def test_fake_judge_escalates_risk(self):
        """Mock Stage 2 always returns is_pii=True — risk must become HIGH."""
        def always_pii(text):
            return JudgeResult(is_pii=True, confidence=0.99,
                               reasoning="zero-shot", model="fake")

        with patch("llm_judge.judge_text", side_effect=always_pii):
            r = client.post(
                "/detect/v2",
                json={"text": "The pipeline runs every 6 hours."}
            )
        data = r.json()
        assert data["risk_level"] == "HIGH"
        assert data["stage2_flagged"] is True

    def test_fake_judge_clean_keeps_risk(self):
        """Mock Stage 2 always returns is_pii=False — risk stays CLEAN."""
        def never_pii(text):
            return JudgeResult(is_pii=False, confidence=0.05,
                               reasoning="zero-shot", model="fake")

        with patch("llm_judge.judge_text", side_effect=never_pii):
            r = client.post(
                "/detect/v2",
                json={"text": "The pipeline runs every 6 hours."}
            )
        data = r.json()
        assert data["risk_level"] == "CLEAN"
        assert data["stage2_flagged"] is False


# ── v1 endpoint still works alongside v2 ─────────────────────────────────────

class TestV1StillWorks:
    def test_v1_detect_still_200(self):
        r = client.post("/detect", json={"text": "hello world"})
        assert r.status_code == 200

    def test_v1_has_no_stage2_keys(self):
        r = client.post("/detect", json={"text": "hello world"})
        data = r.json()
        assert "stage2_used" not in data
        assert "stage2_flagged" not in data

    def test_batch_still_works(self):
        r = client.post("/detect/batch", json={"texts": ["hello", "card: 4111111111111111"]})
        assert r.status_code == 200
        assert r.json()["total"] == 2