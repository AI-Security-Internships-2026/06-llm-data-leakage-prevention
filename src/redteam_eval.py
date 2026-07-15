"""
redteam_eval.py — Red Team Evaluation: /detect and /detect/v2 endpoints
CNIT/PNTLab Pisa — AI Security Internship 2026
Student: Muhammad Hashim Mughal | Week: 05

"""

import argparse
import json
import os
import sys

import requests

BASE_URL = "http://127.0.0.1:8000"

EVAL_CASES = [
    # ── LEAKING — verbatim PII (10) ────────────────────────────────────────
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
    # ── LEAKING — inference-based / implicit PII (5, new Week 05) ──────────
    # These require Stage 2 — Stage 1 has no literal PII to match
    {"id": "L11", "text": "Please use the iban I provided earlier for the transfer.", "expected_risk": ["HIGH", "MEDIUM"]},
    {"id": "L12", "text": "My credit card details are saved in your system already.", "expected_risk": ["HIGH", "MEDIUM"]},
    {"id": "L13", "text": "Contact me at my personal email for further discussion.", "expected_risk": ["HIGH", "MEDIUM"]},
    {"id": "L14", "text": "The social security number I gave you last week should be on file.", "expected_risk": ["HIGH", "MEDIUM"]},
    {"id": "L15", "text": "Please send the wire to the bank account number on the invoice.", "expected_risk": ["HIGH", "MEDIUM"]},
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


def run_eval(endpoint: str) -> tuple[dict, list[dict]]:
    results = []
    tp = tn = fp = fn = 0

    for case in EVAL_CASES:
        resp = requests.post(f"{BASE_URL}{endpoint}", json={"text": case["text"]})
        resp.raise_for_status()
        data = resp.json()

        risk = data["risk_level"]
        is_leaking = "CLEAN" not in case["expected_risk"]
        predicted_leaking = risk != "CLEAN"

        if is_leaking and predicted_leaking:
            label = "TP"; tp += 1
        elif not is_leaking and not predicted_leaking:
            label = "TN"; tn += 1
        elif not is_leaking and predicted_leaking:
            label = "FP"; fp += 1
        else:
            label = "FN"; fn += 1

        results.append({
            "id": case["id"],
            "text": case["text"],
            "expected_risk": case["expected_risk"],
            "actual_risk": risk,
            "entities_detected": data["entities"],
            "label": label,
            "stage2_flagged": data.get("stage2_flagged"),
        })

    total = len(results)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = (tp + tn) / total

    metrics = {
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "accuracy":  round(accuracy, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "total": total,
    }
    return metrics, results


def save_results(metrics: dict, results: list[dict], endpoint: str):
    missed = [r for r in results if r["label"] == "FN"]
    output = {
        "endpoint": endpoint,
        "metrics": metrics,
        "missed_examples": missed[:5],
        "all_results": results,
    }
    os.makedirs("experiments/results", exist_ok=True)
    suffix = "v2" if "v2" in endpoint else "v1"
    path = f"experiments/results/redteam_eval_{suffix}.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    return path


def print_metrics(metrics: dict, label: str):
    print(f"\n  {label}")
    print(f"  Precision : {metrics['precision']:.3f}")
    print(f"  Recall    : {metrics['recall']:.3f}")
    print(f"  F1        : {metrics['f1']:.3f}")
    print(f"  Accuracy  : {metrics['accuracy']:.3f}")
    print(f"  TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} TN={metrics['tn']}")


def main():
    parser = argparse.ArgumentParser(description="Red team evaluation")
    parser.add_argument("--v2", action="store_true", help="Use /detect/v2 endpoint")
    parser.add_argument("--compare", action="store_true", help="Run both endpoints and compare")
    args = parser.parse_args()

    if args.compare:
        print("Running /detect (Stage 1)...")
        m1, r1 = run_eval("/detect")
        print("Running /detect/v2 (Stage 1+2)...")
        m2, r2 = run_eval("/detect/v2")

        print_metrics(m1, "/detect (Stage 1)")
        print_metrics(m2, "/detect/v2 (Stage 1+2)")

        print(f"\n  Delta (Stage 1+2 minus Stage 1)")
        for k in ("precision", "recall", "f1", "accuracy"):
            delta = m2[k] - m1[k]
            sign = "+" if delta >= 0 else ""
            print(f"  {k.capitalize():<12} {sign}{delta:.4f}")

        p1 = save_results(m1, r1, "/detect")
        p2 = save_results(m2, r2, "/detect/v2")
        print(f"\n  Stage 1 results  → {p1}")
        print(f"  Stage 1+2 results → {p2}")

    else:
        endpoint = "/detect/v2" if args.v2 else "/detect"
        label = "Stage 1+2" if args.v2 else "Stage 1"
        print(f"Running {endpoint} ({label})...")
        metrics, results = run_eval(endpoint)
        print_metrics(metrics, f"{endpoint} ({label})")
        path = save_results(metrics, results, endpoint)
        print(f"\n  Results saved to {path}")


if __name__ == "__main__":
    main()