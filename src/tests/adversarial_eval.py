"""
adversarial_eval.py — Adversarial Robustness Evaluation
CNIT/PNTLab Pisa — AI Security Internship 2026
Student: Muhammad Hashim Mughal | Week: 04
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.detector import detect_pii

ADVERSARIAL_CASES = [
    {"id": "ADV-CC-01", "text": "Card: 4111111111111111",              "expected": "LEAKING", "category": "credit_card", "variant": "plain"},
    {"id": "ADV-CC-02", "text": "Card number: 4111 1111 1111 1111",    "expected": "LEAKING", "category": "credit_card", "variant": "spaced"},
    {"id": "ADV-CC-03", "text": "Declined card: 4111-1111-1111-1111",  "expected": "LEAKING", "category": "credit_card", "variant": "hyphen"},
    {"id": "ADV-CC-04", "text": "Export row: 4111.1111.1111.1111",     "expected": "LEAKING", "category": "credit_card", "variant": "dot"},
    {"id": "ADV-CC-05", "text": '{"card": "4111111111111111"}',        "expected": "LEAKING", "category": "credit_card", "variant": "json"},
    {"id": "ADV-EM-01", "text": "Contact: alice@example.com",           "expected": "LEAKING", "category": "email", "variant": "standard"},
    {"id": "ADV-EM-02", "text": "Reach me at alice [at] example [dot] com", "expected": "LEAKING", "category": "email", "variant": "bracket_at"},
    {"id": "ADV-EM-03", "text": "Email: alice(at)example.com",          "expected": "LEAKING", "category": "email", "variant": "paren_at"},
    {"id": "ADV-EM-04", "text": "Send to alice AT example DOT com",     "expected": "LEAKING", "category": "email", "variant": "caps_at"},
    {"id": "ADV-EM-05", "text": "Forward to ops@mail.company.org",      "expected": "LEAKING", "category": "email", "variant": "subdomain"},
    {"id": "ADV-PH-01", "text": "Call us at +1-800-555-0199.",          "expected": "LEAKING", "category": "phone", "variant": "e164"},
    {"id": "ADV-PH-02", "text": "Reach us at (800) 555-0199.",          "expected": "LEAKING", "category": "phone", "variant": "parenthesis"},
    {"id": "ADV-PH-03", "text": "Support line: 800.555.0199",           "expected": "LEAKING", "category": "phone", "variant": "dot"},
    {"id": "ADV-PH-04", "text": "Call 800-555-0199 for assistance.",    "expected": "LEAKING", "category": "phone", "variant": "plain"},
    {"id": "ADV-EB-01", "text": '{"email": "alice@example.com"}',       "expected": "LEAKING", "category": "embedded", "variant": "email_json"},
    {"id": "ADV-EB-02", "text": "SELECT * FROM users WHERE email = 'alice@example.com';", "expected": "LEAKING", "category": "embedded", "variant": "email_sql"},
    {"id": "ADV-EB-03", "text": '{"applicant": {"ssn": "078-05-1120"}}', "expected": "LEAKING", "category": "embedded", "variant": "ssn_json"},
    {"id": "ADV-EB-04", "text": "| Name | Email |\n| Alice | alice@example.com |", "expected": "LEAKING", "category": "embedded", "variant": "email_markdown"},
    {"id": "ADV-CL-01", "text": "No personal information here.",        "expected": "CLEAN",   "category": "clean", "variant": "prose"},
    {"id": "ADV-CL-02", "text": "def add(x, y): return x + y",         "expected": "CLEAN",   "category": "clean", "variant": "code"},
    {"id": "ADV-CL-03", "text": "Revenue grew 12% year-over-year.",     "expected": "CLEAN",   "category": "clean", "variant": "business"},
]


def run_adversarial_eval() -> dict:
    print(f"Running adversarial evaluation on {len(ADVERSARIAL_CASES)} cases...\n")

    results = []
    tp = fp = tn = fn = 0

    for case in ADVERSARIAL_CASES:
        t0 = time.perf_counter()
        result = detect_pii(case["text"])
        latency_ms = round((time.perf_counter() - t0) * 1000, 3)

        predicted = "CLEAN" if result["risk_level"] == "CLEAN" else "LEAKING"
        correct = predicted == case["expected"]

        if case["expected"] == "LEAKING" and predicted == "LEAKING":
            tp += 1
        elif case["expected"] == "CLEAN" and predicted == "LEAKING":
            fp += 1
        elif case["expected"] == "CLEAN" and predicted == "CLEAN":
            tn += 1
        else:
            fn += 1

        status = "PASS" if correct else "FAIL"
        print(f"  [{status}] {case['id']} ({case['category']}/{case['variant']})")

        results.append({
            "id": case["id"],
            "category": case["category"],
            "variant": case["variant"],
            "text": case["text"],
            "expected": case["expected"],
            "predicted": predicted,
            "risk_level": result["risk_level"],
            "correct": correct,
            "latency_ms": latency_ms,
        })

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (tp + tn) / len(ADVERSARIAL_CASES)

    print(f"\n  TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  Accuracy  : {accuracy:.4f}")

    false_negatives = [r for r in results if r["expected"] == "LEAKING" and not r["correct"]]
    if false_negatives:
        print(f"\n  False negatives ({len(false_negatives)}):")
        for fn_case in false_negatives:
            print(f"    {fn_case['id']} — {fn_case['variant']}: {fn_case['text'][:60]}")

    return {
        "meta": {
            "run_at": datetime.now(timezone.utc).isoformat(),
            "total_cases": len(ADVERSARIAL_CASES),
            "detector": "PIIDetector (Presidio + custom recognisers + Week 04 normalizer)",
        },
        "summary": {
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "accuracy":  round(accuracy, 4),
        },
        "results": results,
    }


if __name__ == "__main__":
    output = run_adversarial_eval()

    os.makedirs("experiments/results", exist_ok=True)
    out_path = "experiments/results/adversarial_eval.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {out_path}")