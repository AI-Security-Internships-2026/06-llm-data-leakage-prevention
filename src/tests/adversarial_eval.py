"""
adversarial_eval.py — Adversarial Evaluation Suite
CNIT/PNTLab Pisa — AI Security Internship 2026
Student: Muhammad Hashim Mughal | Week: 04

"""

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from detector import detect_pii

ADVERSARIAL_CASES = [
    {
        "id": "ADV-CC-01",
        "category": "credit_card_format",
        "text": "Card: 4111111111111111",
        "expected_label": "LEAKING",
        "description": "No-delimiter card — baseline",
    },
    {
        "id": "ADV-CC-02",
        "category": "credit_card_format",
        "text": "Card number: 4111 1111 1111 1111",
        "expected_label": "LEAKING",
        "description": "Space-delimited card",
    },
    {
        "id": "ADV-CC-03",
        "category": "credit_card_format",
        "text": "Declined card: 4111-1111-1111-1111",
        "expected_label": "LEAKING",
        "description": "Hyphen-delimited card",
    },
    {
        "id": "ADV-CC-04",
        "category": "credit_card_format",
        "text": "Export row: 4111.1111.1111.1111",
        "expected_label": "LEAKING",
        "description": "Dot-delimited card — handled by normalize_text()",
    },
    {
        "id": "ADV-CC-05",
        "category": "credit_card_format",
        "text": '{"payment": {"card": "4111111111111111"}}',
        "expected_label": "LEAKING",
        "description": "Card in nested JSON",
    },
    {
        "id": "ADV-CC-06",
        "category": "credit_card_format",
        "text": '{"payment": {"card": "4111.1111.1111.1111", "exp": "12/28"}}',
        "expected_label": "LEAKING",
        "description": "Dot-delimited card in JSON payload",
    },
    {
        "id": "ADV-EM-01",
        "category": "email_obfuscation",
        "text": "Contact: alice@example.com",
        "expected_label": "LEAKING",
        "description": "Standard email — baseline",
    },
    {
        "id": "ADV-EM-02",
        "category": "email_obfuscation",
        "text": "Reach me at alice [at] example [dot] com",
        "expected_label": "LEAKING",
        "description": "[at]/[dot] obfuscation — handled by normalize_text()",
    },
    {
        "id": "ADV-EM-03",
        "category": "email_obfuscation",
        "text": "Email: alice(at)example.com",
        "expected_label": "LEAKING",
        "description": "(at) obfuscation — handled by normalize_text()",
    },
    {
        "id": "ADV-EM-04",
        "category": "email_obfuscation",
        "text": "Send to alice AT example DOT com",
        "expected_label": "LEAKING",
        "description": "CAPS AT/DOT obfuscation — handled by normalize_text()",
    },
    {
        "id": "ADV-EM-05",
        "category": "email_obfuscation",
        "text": "Forward to ops@mail.company.org",
        "expected_label": "LEAKING",
        "description": "Multi-label subdomain email",
    },
    {
        "id": "ADV-EM-06",
        "category": "email_obfuscation",
        "text": "Notifications to alice+alerts@example.io",
        "expected_label": "LEAKING",
        "description": "Plus-addressing email",
    },
    {
        "id": "ADV-EM-07",
        "category": "email_obfuscation",
        "text": "Bounce-To: noreply+bounce@subdomain.company.co.uk",
        "expected_label": "LEAKING",
        "description": "Multi-level TLD (.co.uk) with plus addressing",
    },
    {
        "id": "ADV-PH-01",
        "category": "phone_format",
        "text": "Call us at +1-800-555-0199.",
        "expected_label": "LEAKING",
        "description": "E.164 with hyphens",
    },
    {
        "id": "ADV-PH-02",
        "category": "phone_format",
        "text": "Phone: (800) 555-0199",
        "expected_label": "LEAKING",
        "description": "Parenthesis format",
    },
    {
        "id": "ADV-PH-03",
        "category": "phone_format",
        "text": "Dot format phone: 800.555.0199",
        "expected_label": "LEAKING",
        "description": "Dot-delimited US phone — handled by normalize_text()",
    },
    {
        "id": "ADV-IB-01",
        "category": "iban",
        "text": "Wire the payment to IBAN GB29NWBK60161331926819.",
        "expected_label": "LEAKING",
        "description": "IBAN with keyword prefix",
    },
    {
        "id": "ADV-IB-02",
        "category": "iban",
        "text": "Bank account: DE89370400440532013000",
        "expected_label": "LEAKING",
        "description": "IBAN with bank account context — no IBAN keyword",
    },
    {
        "id": "ADV-IB-03",
        "category": "iban",
        "text": "| Beneficiary | FR7614508711002120144503422 | EUR |",
        "expected_label": "LEAKING",
        "description": "IBAN in markdown table — no prose context",
    },
    {
        "id": "ADV-EMB-01",
        "category": "embedded",
        "text": '{"email": "alice@example.com", "role": "admin"}',
        "expected_label": "LEAKING",
        "description": "Email in JSON object",
    },
    {
        "id": "ADV-EMB-02",
        "category": "embedded",
        "text": "SELECT * FROM users WHERE email = 'alice@example.com';",
        "expected_label": "LEAKING",
        "description": "Email in SQL query",
    },
    {
        "id": "ADV-EMB-03",
        "category": "embedded",
        "text": "<citizen><cnic>35202-1234567-8</cnic></citizen>",
        "expected_label": "LEAKING",
        "description": "CNIC in XML element",
    },
    {
        "id": "ADV-EMB-04",
        "category": "embedded",
        "text": "alice,35,alice@example.com,+1-800-555-0199,New York",
        "expected_label": "LEAKING",
        "description": "Multi-PII in CSV row — email + phone",
    },
    {
        "id": "ADV-CLN-01",
        "category": "clean",
        "text": "The API returns HTTP 200 on success and 422 on validation failure.",
        "expected_label": "CLEAN",
        "description": "Technical documentation",
    },
    {
        "id": "ADV-CLN-02",
        "category": "clean",
        "text": "No personal data is stored or processed by this endpoint.",
        "expected_label": "CLEAN",
        "description": "Privacy statement without actual PII",
    },
]


def run_adversarial_eval(output_path: str = "experiments/results/adversarial_eval.json"):
    results = []
    passed = 0
    failed = 0

    for case in ADVERSARIAL_CASES:
        detection = detect_pii(case["text"])
        predicted = "CLEAN" if detection["risk_level"] == "CLEAN" else "LEAKING"
        correct = predicted == case["expected_label"]

        if correct:
            passed += 1
        else:
            failed += 1

        results.append({
            **case,
            "predicted": predicted,
            "risk_level": detection["risk_level"],
            "entities_detected": [e["type"] for e in detection["entities"]],
            "correct": correct,
        })

    total = len(results)
    accuracy = round(passed / total, 4)

    output = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "total": total,
        "passed": passed,
        "failed": failed,
        "accuracy": accuracy,
        "results": results,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Adversarial eval: {passed}/{total} passed ({accuracy:.1%})")
    if failed:
        print("\nFailed cases:")
        for r in results:
            if not r["correct"]:
                print(f"  [{r['id']}] {r['description']}")
                print(f"    Expected={r['expected_label']}  Got={r['predicted']}  Entities={r['entities_detected']}")

    print(f"\nResults saved to {output_path}")
    return output


if __name__ == "__main__":
    run_adversarial_eval()