"""
test_detector.py — Unit tests for the PII leakage detection probe
CNIT/PNTLab Pisa — AI Security Internship 2026
Student: Muhammad Hashim Mughal | Week: 04
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from detector import detect_pii


def detected_types(result: dict) -> set:
    return {e["type"] for e in result["entities"]}


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
        r = detect_pii("Pay with card 4111111111111111")
        for entity in r["entities"]:
            assert "text_slice" not in entity

    def test_unsupported_language_raises(self):
        with pytest.raises(ValueError, match="Unsupported language"):
            detect_pii("Bonjour", language="fr")


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
        assert "PK_CNIC" in detected_types(r)

    def test_pii_embedded_in_technical_text(self):
        r = detect_pii("ERROR [2026-06-20]: user_email=ops@company.org exceeded quota limit.")
        assert "EMAIL_ADDRESS" in detected_types(r)


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
        r = detect_pii("Alice, Bob, and Carol attended the meeting.")
        if r["risk_level"] == "HIGH":
            scores = [e["score"] for e in r["entities"]]
            types = {e["type"] for e in r["entities"]}
            high_risk_types = {"CREDIT_CARD", "IBAN_CODE", "MEDICAL_LICENSE", "US_SSN", "UK_NHS", "PK_CNIC"}
            assert (
                any(s >= 0.7 for s in scores) or bool(types & high_risk_types)
            ), f"Got HIGH risk with no justification — scores={scores}, types={types}"


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


class TestAdversarialCreditCard:
    def test_card_no_delimiter_detected(self):
        r = detect_pii("Card: 4111111111111111")
        assert "CREDIT_CARD" in detected_types(r)

    def test_card_spaced_groups_detected(self):
        r = detect_pii("Card number: 4111 1111 1111 1111")
        assert "CREDIT_CARD" in detected_types(r)

    def test_card_hyphen_groups_detected(self):
        r = detect_pii("Declined card: 4111-1111-1111-1111")
        assert "CREDIT_CARD" in detected_types(r)

    @pytest.mark.xfail(reason="Dot-delimited card is a known false negative")
    def test_card_dot_groups_known_gap(self):
        r = detect_pii("Export row: 4111.1111.1111.1111")
        assert "CREDIT_CARD" in detected_types(r)

    def test_card_in_json_payload_detected(self):
        r = detect_pii('{"user": "alice", "card": "4111111111111111", "exp": "12/28"}')
        assert "CREDIT_CARD" in detected_types(r)

    def test_card_risk_is_not_clean(self):
        r = detect_pii("Billing card 4111 1111 1111 1111 on file.")
        assert r["risk_level"] != "CLEAN"


class TestAdversarialEmail:
    def test_standard_email_detected(self):
        r = detect_pii("Contact: alice@example.com")
        assert "EMAIL_ADDRESS" in detected_types(r)

    @pytest.mark.xfail(reason="[at] obfuscation is a known false negative")
    def test_email_at_bracket_obfuscated(self):
        r = detect_pii("Reach me at alice [at] example [dot] com")
        assert "EMAIL_ADDRESS" in detected_types(r)

    @pytest.mark.xfail(reason="(at) obfuscation is a known false negative")
    def test_email_at_paren_obfuscated(self):
        r = detect_pii("Email: alice(at)example.com")
        assert "EMAIL_ADDRESS" in detected_types(r)

    @pytest.mark.xfail(reason="ALL CAPS AT obfuscation is a known false negative")
    def test_email_caps_at_obfuscated(self):
        r = detect_pii("Send to alice AT example DOT com")
        assert "EMAIL_ADDRESS" in detected_types(r)

    def test_email_subdomain_detected(self):
        r = detect_pii("Forward to ops@mail.company.org")
        assert "EMAIL_ADDRESS" in detected_types(r)

    def test_email_plus_addressing_detected(self):
        r = detect_pii("Notifications go to alice+alerts@example.com")
        assert "EMAIL_ADDRESS" in detected_types(r)


class TestAdversarialEmbedded:
    def test_email_in_json_string(self):
        r = detect_pii('{"email": "alice@example.com", "role": "admin"}')
        assert "EMAIL_ADDRESS" in detected_types(r)

    def test_credit_card_in_json(self):
        r = detect_pii('{"payment": {"card": "4111111111111111", "exp": "12/28"}}')
        assert "CREDIT_CARD" in detected_types(r)

    def test_ssn_in_json(self):
        r = detect_pii('{"applicant": {"ssn": "078-05-1120", "name": "John"}}')
        assert "US_SSN" in detected_types(r)

    def test_email_in_python_fstring(self):
        r = detect_pii('send_email(to="alice@example.com", subject="Welcome")')
        assert "EMAIL_ADDRESS" in detected_types(r)

    def test_email_in_sql_query(self):
        r = detect_pii("SELECT * FROM users WHERE email = 'alice@example.com';")
        assert "EMAIL_ADDRESS" in detected_types(r)

    def test_phone_in_log_line(self):
        r = detect_pii("INFO [2026-07-01 10:22:31] sms_sent to=+1-800-555-0199 status=delivered")
        assert "PHONE_NUMBER" in detected_types(r)

    def test_cnic_in_xml(self):
        r = detect_pii("<citizen><cnic>35202-1234567-8</cnic><name>Test</name></citizen>")
        assert "PK_CNIC" in detected_types(r)

    def test_iban_in_markdown_table(self):
        r = detect_pii("| Beneficiary | GB29NWBK60161331926819 | GBP | Active |")
        assert "IBAN_CODE" in detected_types(r)


class TestAdversarialPhoneVariants:
    def test_phone_us_standard(self):
        r = detect_pii("Call us at +1-800-555-0199.")
        assert "PHONE_NUMBER" in detected_types(r)

    def test_phone_parenthesis_format(self):
        r = detect_pii("Reach us at (800) 555-0199 anytime.")
        assert "PHONE_NUMBER" in detected_types(r)

    def test_phone_dot_format(self):
        r = detect_pii("Support line: 800.555.0199")
        assert "PHONE_NUMBER" in detected_types(r)

    def test_phone_no_country_code(self):
        r = detect_pii("Call 800-555-0199 for assistance.")
        assert "PHONE_NUMBER" in detected_types(r)

    def test_phone_spaces_format(self):
        r = detect_pii("Contact number: (800) 555 0199")
        assert "PHONE_NUMBER" in detected_types(r)

    @pytest.mark.xfail(reason="International format with country code is a known gap")
    def test_phone_international_format(self):
        r = detect_pii("Dial +44 20 7946 0958 for the London office.")
        assert "PHONE_NUMBER" in detected_types(r)


class TestAdversarialMarkdownTables:
    def test_email_in_markdown_table(self):
        r = detect_pii("| Name | Email | Role |\n| Alice | alice@example.com | Admin |")
        assert "EMAIL_ADDRESS" in detected_types(r)

    def test_phone_in_markdown_table(self):
        r = detect_pii("| Department | Contact |\n| Support | +1-800-555-0199 |")
        assert "PHONE_NUMBER" in detected_types(r)

    def test_ssn_in_markdown_table(self):
        r = detect_pii("| Employee | SSN |\n| John Doe | 078-05-1120 |")
        assert "US_SSN" in detected_types(r)

    def test_cnic_in_markdown_table(self):
        r = detect_pii("| Applicant | CNIC |\n| Test User | 35202-1234567-8 |")
        assert "PK_CNIC" in detected_types(r)

    def test_credit_card_in_markdown_table(self):
        r = detect_pii("| Customer | Card |\n| Alice | 4111111111111111 |")
        assert "CREDIT_CARD" in detected_types(r)

    def test_multiple_pii_in_markdown_table(self):
        r = detect_pii("| Name | Email | Phone |\n| Alice | alice@example.com | +1-800-555-0199 |")
        types = detected_types(r)
        assert "EMAIL_ADDRESS" in types
        assert "PHONE_NUMBER" in types