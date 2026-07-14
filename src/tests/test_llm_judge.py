"""
test_llm_judge.py — Unit tests for the Stage 2 LLM-as-judge
CNIT/PNTLab Pisa — AI Security Internship 2026
Student: Muhammad Hashim Mughal | Week: 05

Strategy
--------
The HuggingFace model (bart-large-mnli) may not be available in all
test environments (no internet, first-run download, CI without model cache).
Tests are therefore split into two groups:

  1. Always-run tests — patch _load_pipeline to return None so the fallback
     keyword heuristic is exercised. These run in all environments.

  2. Model tests — marked xfail(strict=False) — run the real model if present,
     skip gracefully if it cannot be loaded. These pass in environments with
     the model cached and are treated as informational otherwise.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import patch

import llm_judge as lj
from llm_judge import judge_text, judge_batch, JudgeResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def _no_model():
    """Patch _load_pipeline to simulate 'model unavailable'."""
    return patch("llm_judge._load_pipeline", return_value=None)


# ── Return type and shape ─────────────────────────────────────────────────────

class TestReturnShape:
    def test_judge_text_returns_dataclass(self):
        with _no_model():
            result = judge_text("hello world")
        assert isinstance(result, JudgeResult)

    def test_all_fields_present(self):
        with _no_model():
            r = judge_text("hello world")
        assert hasattr(r, "is_pii")
        assert hasattr(r, "confidence")
        assert hasattr(r, "reasoning")
        assert hasattr(r, "model")

    def test_is_pii_is_bool(self):
        with _no_model():
            r = judge_text("my email is alice@example.com")
        assert isinstance(r.is_pii, bool)

    def test_confidence_in_range(self):
        with _no_model():
            r = judge_text("my email is alice@example.com")
        assert 0.0 <= r.confidence <= 1.0

    def test_empty_string_not_pii(self):
        with _no_model():
            r = judge_text("")
        assert r.is_pii is False

    def test_whitespace_not_pii(self):
        with _no_model():
            r = judge_text("   ")
        assert r.is_pii is False


# ── Fallback keyword heuristic (model=None) ───────────────────────────────────

class TestFallbackHeuristic:
    def test_at_dot_obfuscation_flagged(self):
        with _no_model():
            r = judge_text("send it to alice at example dot com")
        assert r.is_pii is True
        assert r.reasoning == "fallback-keyword"

    def test_iban_keyword_flagged(self):
        with _no_model():
            r = judge_text("Please use the iban number provided above")
        assert r.is_pii is True

    def test_ssn_keyword_flagged(self):
        with _no_model():
            r = judge_text("Social security number on file")
        assert r.is_pii is True

    def test_credit_card_phrase_flagged(self):
        with _no_model():
            r = judge_text("Please charge the credit card on file")
        assert r.is_pii is True

    def test_dob_phrase_flagged(self):
        with _no_model():
            r = judge_text("Enter your date of birth below")
        assert r.is_pii is True

    def test_contact_at_flagged(self):
        with _no_model():
            r = judge_text("Contact me at the number above")
        assert r.is_pii is True

    def test_clean_technical_text_not_flagged(self):
        with _no_model():
            r = judge_text("The REST endpoint returns HTTP 200 on success.")
        assert r.is_pii is False
        assert r.reasoning == "fallback-no-match"

    def test_clean_science_text_not_flagged(self):
        with _no_model():
            r = judge_text("The sample was heated to 250 degrees for 30 minutes.")
        assert r.is_pii is False

    def test_clean_code_not_flagged(self):
        with _no_model():
            r = judge_text("def add(x, y):\n    return x + y")
        assert r.is_pii is False

    def test_model_is_none_in_fallback(self):
        with _no_model():
            r = judge_text("contact me at my home phone")
        assert r.model is None


# ── judge_batch ───────────────────────────────────────────────────────────────

class TestJudgeBatch:
    def test_empty_list_returns_empty(self):
        with _no_model():
            results = judge_batch([])
        assert results == []

    def test_batch_length_matches_input(self):
        texts = ["hello", "alice@example.com", "contact me at home phone"]
        with _no_model():
            results = judge_batch(texts)
        assert len(results) == len(texts)

    def test_batch_returns_list_of_judge_results(self):
        with _no_model():
            results = judge_batch(["hello", "social security number"])
        for r in results:
            assert isinstance(r, JudgeResult)

    def test_clean_text_in_batch_not_flagged(self):
        with _no_model():
            results = judge_batch(["The pipeline runs every 6 hours."])
        assert results[0].is_pii is False


# ── Integration with detect_pii (Stage 2 flag) ───────────────────────────────

class TestPipelineIntegration:
    def test_use_stage2_adds_keys(self):
        with _no_model():
            # Patch at detector level too so it uses our no-model version
            with patch("detector.judge_text", side_effect=lj.judge_text):
                from detector import detect_pii
                r = detect_pii("hello world", use_stage2=True)
        assert "stage2_used" in r
        assert "stage2_flagged" in r

    def test_stage2_skipped_for_high_risk(self):
        """HIGH risk from Stage 1 should NOT call Stage 2."""
        call_log = []

        def fake_judge(text):
            call_log.append(text)
            return JudgeResult(is_pii=True, confidence=0.99,
                               reasoning="zero-shot", model="fake")

        with patch("detector.judge_text", side_effect=fake_judge):
            from detector import detect_pii
            r = detect_pii("Card: 4111111111111111", use_stage2=True)

        # HIGH risk — Stage 2 must not have been called
        assert len(call_log) == 0
        assert r["risk_level"] == "HIGH"
        assert r["stage2_flagged"] is False

    def test_stage2_called_for_clean(self):
        """CLEAN output from Stage 1 SHOULD trigger Stage 2."""
        call_log = []

        def fake_judge(text):
            call_log.append(text)
            return JudgeResult(is_pii=False, confidence=0.40,
                               reasoning="zero-shot", model="fake")

        with patch("detector.judge_text", side_effect=fake_judge):
            from detector import detect_pii
            detect_pii("The pipeline runs every 6 hours.", use_stage2=True)

        assert len(call_log) == 1

    def test_stage2_escalates_risk(self):
        """If Stage 2 flags PII, risk_level must become HIGH."""
        def fake_judge(text):
            return JudgeResult(is_pii=True, confidence=0.92,
                               reasoning="zero-shot", model="fake")

        with patch("detector.judge_text", side_effect=fake_judge):
            from detector import detect_pii
            r = detect_pii("contact me at home phone", use_stage2=True)

        assert r["risk_level"] == "HIGH"
        assert r["stage2_flagged"] is True


# ── Optional: model-backed tests (xfail if model not cached) ─────────────────

class TestWithRealModel:
    @pytest.mark.xfail(strict=False, reason="Requires bart-large-mnli model download")
    def test_real_model_flags_email(self):
        r = judge_text("Please reach out to alice@example.com for the report.")
        assert r.is_pii is True
        assert r.reasoning == "zero-shot"

    @pytest.mark.xfail(strict=False, reason="Requires bart-large-mnli model download")
    def test_real_model_clean_text(self):
        r = judge_text("Gradient descent converged after 150 epochs.")
        assert r.is_pii is False
        assert r.reasoning == "zero-shot"