"""
test_detector.py — Unit tests for the PII leakage detection probe
CNIT/PNTLab Pisa — AI Security Internship 2026
Student: Muhammad Hashim Mughal | Week: 03

Run:
    cd 06-llm-data-leakage-prevention
    pytest src/tests/test_detector.py -v
"""

import sys
import os

# Allow running from any working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from detector import detect_pii


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def detected_types(result: dict) -> set:
    return {e["type"] for e in result["entities"]}


# ---------------------------------------------------------------------------
# 1. Edge cases — empty / whitespace / structure
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_string_is_clean(self):
        r = detect_pii("")
        assert r["risk_level"] == "CLEAN"
        assert r["entities"] == []

    def test_whitespace_only_is_clean(self):
        r = detect_pii("     ")
        assert r["risk_level"] == "CLEAN"

    def test_return_keys_always_present(self):
        r = detect_pii("Hello world")
        assert {"text", "entities", "risk_level", "sanitized"} <= r.keys()

    def test_risk_level_is_valid_enum(self):
        for text in ["", "hello", "email@test.com", "4111111111111111"]:
            r = detect_pii(text)
            assert r["risk_level"] in {"CLEAN", "LOW", "MEDIUM", "HIGH"}

    def test_no_pii_in_entity_output(self):
        """text_slice must NOT appear in entity dicts — we removed it in Week 03."""
        r = detect_pii("Pay with card 4111111111111111")
        for entity in r["entities"]:
            assert "text_slice" not in entity, (
                "text_slice was found in entity output — this re-leaks PII. Remove it."
            )

    def test_unsupported_language_raises(self):
        with pytest.raises(ValueError, match="Unsupported language"):
            detect_pii("Bonjour", language="fr")


# ---------------------------------------------------------------------------
# 2. Known-leaking inputs — detection correctness
# ---------------------------------------------------------------------------

class TestLeakingInputs:
    def test_email_detected(self):
        r = detect_pii("Please reach out to alice@example.com for details.")
        assert "EMAIL_ADDRESS" in detected_types(r)

    def test_credit_card_detected(self):
        r = detect_pii("Payment processed with card 4111111111111111.")
        assert "CREDIT_CARD" in detected_types(r)

    def test_us_ssn_detected(self):
        r = detect_pii("Her social security number is 078-05-1120.")
        assert "US_SSN" in detected_types(r)

    def test_phone_number_detected(self):
        r = detect_pii("Call us at +1-800-555-0199 any time.")
        assert "PHONE_NUMBER" in detected_types(r)

    def test_iban_detected(self):
        r = detect_pii("Transfer funds to GB29NWBK60161331926819 by Friday.")
        assert "IBAN_CODE" in detected_types(r)

    def test_pk_cnic_detected(self):
        r = detect_pii("National ID: 35202-1234567-8")
        assert "PK_CNIC" in detected_types(r), (
            "Pakistani CNIC not detected — check that the custom recognizer is registered."
        )

    def test_pii_embedded_in_technical_text(self):
        r = detect_pii(
            "ERROR [2026-06-20]: user_email=ops@company.org exceeded quota limit."
        )
        assert "EMAIL_ADDRESS" in detected_types(r)


# ---------------------------------------------------------------------------
# 3. Known-non-leaking inputs — no false positives
# ---------------------------------------------------------------------------

class TestNonLeakingInputs:
    def test_technical_documentation(self):
        r = detect_pii("The REST API uses JSON over HTTPS with OAuth2 bearer tokens.")
        assert r["risk_level"] == "CLEAN", f"False positive: {r['entities']}"

    def test_code_snippet(self):
        r = detect_pii("def add(x, y):\n    return x + y")
        assert r["risk_level"] == "CLEAN", f"False positive: {r['entities']}"

    def test_medical_terminology_without_personal_info(self):
        r = detect_pii("Hypertension is managed with ACE inhibitors and beta-blockers.")
        assert r["risk_level"] == "CLEAN", f"False positive: {r['entities']}"

    def test_generic_business_text(self):
        r = detect_pii("Quarterly revenue increased 12% year-over-year.")
        assert r["risk_level"] == "CLEAN", f"False positive: {r['entities']}"

    def test_scientific_text(self):
        r = detect_pii("The sample was heated to 250°C for 30 minutes in a sealed vessel.")
        assert r["risk_level"] == "CLEAN", f"False positive: {r['entities']}"


# ---------------------------------------------------------------------------
# 4. Risk level correctness
# ---------------------------------------------------------------------------

class TestRiskLevels:
    def test_credit_card_is_high_risk(self):
        r = detect_pii("Card: 4111111111111111")
        assert r["risk_level"] == "HIGH"

    def test_ssn_is_high_risk(self):
        r = detect_pii("SSN: 078-05-1120")
        assert r["risk_level"] == "HIGH"

    def test_iban_is_high_risk(self):
        r = detect_pii("IBAN: GB29NWBK60161331926819")
        assert r["risk_level"] == "HIGH"

    def test_pk_cnic_is_high_risk(self):
        r = detect_pii("CNIC: 35202-1234567-8")
        assert r["risk_level"] == "HIGH"

    def test_clean_text_is_clean(self):
        r = detect_pii("No personal data here at all.")
        assert r["risk_level"] == "CLEAN"

    def test_multiple_low_entities_without_high_score_not_auto_high(self):
        """3 low-confidence entities must not blindly become HIGH (Week 03 fix)."""
        # This is tested implicitly by the precision/recall suite; here we just
        # verify the risk function doesn't escalate 3 low-confidence generic names.
        r = detect_pii("Alice, Bob, and Carol attended the meeting.")
        # Risk may be LOW or MEDIUM depending on confidence, but must NOT be
        # HIGH if no entity score is >= 0.7 and no high-risk type is present.
        if r["risk_level"] == "HIGH":
            scores = [e["score"] for e in r["entities"]]
            types = {e["type"] for e in r["entities"]}
            high_risk_types = {"CREDIT_CARD", "IBAN_CODE", "MEDICAL_LICENSE", "US_SSN", "UK_NHS", "PK_CNIC"}
            assert (
                any(s >= 0.7 for s in scores) or bool(types & high_risk_types)
            ), f"Got HIGH risk with no justification — scores={scores}, types={types}"


# ---------------------------------------------------------------------------
# 5. Sanitization correctness
# ---------------------------------------------------------------------------

class TestSanitization:
    def test_email_redacted_in_sanitized(self):
        r = detect_pii("Reach me at bob@example.com please.")
        assert "bob@example.com" not in r["sanitized"]
        assert "<REDACTED>" in r["sanitized"]

    def test_credit_card_redacted(self):
        r = detect_pii("My card number is 4111111111111111.")
        assert "4111111111111111" not in r["sanitized"]

    def test_clean_text_sanitized_unchanged(self):
        text = "No personal information in this sentence."
        r = detect_pii(text)
        assert r["sanitized"] == text

    def test_pk_cnic_redacted(self):
        r = detect_pii("ID: 35202-1234567-8 is my CNIC.")
        assert "35202-1234567-8" not in r["sanitized"]
        assert "<REDACTED>" in r["sanitized"]