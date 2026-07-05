"""
redteam_eval.py — Red Team Evaluation: POST /detect endpoint
CNIT/PNTLab Pisa — AI Security Internship 2026
Student : Muhammad Hashim Mughal | Week: 03

Usage (server must be running first):
    uvicorn src.main:app --reload
    python src/redteam_eval.py
"""

import json
import os
import requests

BASE_URL = "http://127.0.0.1:8000"

EVAL_CASES = [
    # ── LEAKING (10) ────────────────────────────────────────────────────────
    {"id": "L01", "text": "Please contact John at john.doe@example.com for more information.", "expected_risk": ["HIGH", "MEDIUM"]},
    {"id": "L02", "text": "Payment was processed using card number 4111111111111111.", "expected_risk": ["HIGH", "MEDIUM"]},
    {"id": "L03", "text": "The applicant's SSN is 078-05-1120.", "expected_risk": ["HIGH", "MEDIUM"]},
    {"id": "L04", "text": "For support call +1-800-555-0199 during business hours.", "expected_risk": ["HIGH", "MEDIUM"]},
    {"id": "L05", "text": "Wire the balance to IBAN GB29NWBK60161331926819 by end of month.", "expected_risk": ["HIGH", "MEDIUM"]},
    {"id": "L06", "text": "Jane Smith's email is jane.smith@corp.com and her direct line is +44 20 7946 0958.", "expected_risk": ["HIGH", "MEDIUM"]},
    {"id": "L07", "text": "ERROR 2026-06-20T14:32Z user=ops@company.org quota_exceeded=true", "expected_risk": ["HIGH", "MEDIUM"]},
    {"id": "L08", "text": "Dr. Sarah Connor prescribed metformin 500mg to patient Robert Miles.", "expected_risk": ["HIGH", "MEDIUM"]},
    {"id": "L09", "text": "National Identity Number: 35202-1234567-8", "expected_risk": ["HIGH", "MEDIUM"]},
    {"id": "L10", "text": '{"user": "alice", "email": "alice@internal.io", "card": "4111111111111111"}', "expected_risk": ["HIGH", "MEDIUM"]},
    # ── CLEAN (10) ──────────────────────────────────────────────────────────
    {"id": "N01", "text": "The REST endpoint accepts JSON over HTTPS and returns a 200 status on success.", "expected_risk": ["CLEAN"]},
    {"id": "N02", "text": "def detect(text: str) -> dict:\n    return analyze(text)", "expected_risk": ["CLEAN"]},
    {"id": "N03", "text": "Hypertension is treated with ACE inhibitors and calcium channel blockers.", "expected_risk": ["CLEAN"]},
    {"id": "N04", "text": "Quarterly revenue grew 12% year-over-year, driven by SaaS subscriptions.", "expected_risk": ["CLEAN"]},
    {"id": "N05", "text": "The sample was heated to 250°C for 30 minutes inside a sealed autoclave.", "expected_risk": ["CLEAN"]},
    {"id": "N06", "text": "The government announced updated energy consumption guidelines this quarter.", "expected_risk": ["CLEAN"]},
    {"id": "N07", "text": "We compare Model A against Model B on the standard benchmark dataset.", "expected_risk": ["CLEAN"]},
    {"id": "N08", "text": "No personal information is collected or stored by this service.", "expected_risk": ["CLEAN"]},
    {"id": "N09", "text": "The pipeline runs every 6 hours and writes output to the data warehouse.", "expected_risk": ["CLEAN"]},
    {"id": "N10", "text": "All configuration values are loaded from environment variables at startup.", "expected_risk": ["CLEAN"]},
]


def run_redteam_eval():
    results = []
    tp = tn = fp = fn = 0

    for case in EVAL_CASES:
        resp = requests.post(f"{BASE_URL}/detect", json={"text": case["text"]})
        resp.raise_for_status()
        data = resp.json()

        risk = data["risk_level"]
        is_leaking = "CLEAN" not in case["expected_risk"]
        predicted_leaking = risk != "CLEAN"

        if is_leaking and predicted_leaking:
            label = "TP"
            tp += 1
        elif not is_leaking and not predicted_leaking:
            label = "TN"
            tn += 1
        elif not is_leaking and predicted_leaking:
            label = "FP"
            fp += 1
        else:
            label = "FN"
            fn += 1

        results.append({
            "id": case["id"],
            "text": case["text"],
            "expected_risk": case["expected_risk"],
            "actual_risk": risk,
            "entities_detected": data["entities"],
            "label": label,
        })

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = (tp + tn) / len(results)

    # 5 examples of what the detector missed (FN cases)
    missed_examples = [r for r in results if r["label"] == "FN"][:5]

    output = {
        "metrics": {
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "accuracy":  round(accuracy, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "total": len(results),
        },
        "missed_examples": missed_examples,
        "all_results": results,
    }

    os.makedirs("experiments/results", exist_ok=True)
    out_path = "experiments/results/redteam_eval.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Precision : {precision:.3f}")
    print(f"Recall    : {recall:.3f}")
    print(f"F1        : {f1:.3f}")
    print(f"Accuracy  : {accuracy:.3f}")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_redteam_eval()