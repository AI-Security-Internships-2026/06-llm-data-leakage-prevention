"""
eval_suite.py — Hand-labelled Precision / Recall Evaluation Suite
CNIT/PNTLab Pisa — AI Security Internship 2026
Student: Muhammad Hashim Mughal | Week: 05

"""

import sys
import os
import json
import argparse
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from detector import detect_pii


EVAL_CASES = [
    # ── LEAKING — clear verbatim PII (Week 03) ───────────────────────────
    {
        "id": "E01",
        "description": "Email address in plain prose",
        "text": "Please contact john.doe@example.com for further information.",
        "label": "LEAKING",
        "expected": {"EMAIL_ADDRESS"},
    },
    {
        "id": "E02",
        "description": "Visa credit card number",
        "text": "Payment processed with card number 4111111111111111.",
        "label": "LEAKING",
        "expected": {"CREDIT_CARD"},
    },
    {
        "id": "E03",
        "description": "US Social Security Number",
        "text": "The applicant's SSN is 078-05-1120.",
        "label": "LEAKING",
        "expected": {"US_SSN"},
    },
    {
        "id": "E04",
        "description": "International phone number",
        "text": "For support please call +1-800-555-0199 during business hours.",
        "label": "LEAKING",
        "expected": {"PHONE_NUMBER"},
    },
    {
        "id": "E05",
        "description": "IBAN bank account number",
        "text": "Wire the balance to IBAN GB29NWBK60161331926819 by end of month.",
        "label": "LEAKING",
        "expected": {"IBAN_CODE"},
    },
    {
        "id": "E06",
        "description": "Multiple PII types (name + email + phone)",
        "text": "Jane Smith's email is jane.smith@corp.com and her direct line is +44 20 7946 0958.",
        "label": "LEAKING",
        "expected": {"PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"},
    },
    {
        "id": "E07",
        "description": "PII embedded inside error log line",
        "text": "ERROR 2026-06-20T14:32Z user=ops@company.org quota_exceeded=true",
        "label": "LEAKING",
        "expected": {"EMAIL_ADDRESS"},
    },
    {
        "id": "E08",
        "description": "Medical context with patient name",
        "text": "Dr. Sarah Connor prescribed metformin 500mg to patient Robert Miles.",
        "label": "LEAKING",
        "expected": {"PERSON"},
    },
    {
        "id": "E09",
        "description": "Pakistani CNIC number",
        "text": "National Identity Number: 35202-1234567-8",
        "label": "LEAKING",
        "expected": {"PK_CNIC"},
    },
    {
        "id": "E10",
        "description": "PII in JSON-like payload",
        "text": '{"user": "alice", "email": "alice@internal.io", "card": "4111111111111111"}',
        "label": "LEAKING",
        "expected": {"EMAIL_ADDRESS", "CREDIT_CARD"},
    },
    # ── CLEAN (Week 03) ──────────────────────────────────────────────────
    {
        "id": "E11",
        "description": "Technical API documentation",
        "text": "The REST endpoint accepts JSON over HTTPS and returns a 200 status on success.",
        "label": "CLEAN",
        "expected": set(),
    },
    {
        "id": "E12",
        "description": "Python code snippet",
        "text": "def detect(text: str) -> dict:\n    return analyze(text)",
        "label": "CLEAN",
        "expected": set(),
    },
    {
        "id": "E13",
        "description": "Medical terminology without personal info",
        "text": "Hypertension is treated with ACE inhibitors and calcium channel blockers.",
        "label": "CLEAN",
        "expected": set(),
    },
    {
        "id": "E14",
        "description": "Generic quarterly report sentence",
        "text": "Quarterly revenue grew 12% year-over-year, driven by SaaS subscriptions.",
        "label": "CLEAN",
        "expected": set(),
    },
    {
        "id": "E15",
        "description": "Scientific experiment description",
        "text": "The sample was heated to 250°C for 30 minutes inside a sealed autoclave.",
        "label": "CLEAN",
        "expected": set(),
    },
    {
        "id": "E16",
        "description": "Neutral news-style sentence",
        "text": "The government announced updated energy consumption guidelines this quarter.",
        "label": "CLEAN",
        "expected": set(),
    },
    {
        "id": "E17",
        "description": "Placeholder variable names (not real people)",
        "text": "We compare Model A against Model B on the standard benchmark dataset.",
        "label": "CLEAN",
        "expected": set(),
    },
    {
        "id": "E18",
        "description": "Policy statement with no personal data",
        "text": "No personal information is collected or stored by this service.",
        "label": "CLEAN",
        "expected": set(),
    },
    # ── LEAKING — inference-based PII (Week 05) ───────────────────────────
    # Stage 1 returns CLEAN on these — no literal PII to match.
    # Stage 2 (LLM-as-judge) catches them via keyword heuristic or zero-shot.
    {
        "id": "E19",
        "description": "Implicit IBAN reference (no literal IBAN present)",
        "text": "Please use the iban I provided earlier for the transfer.",
        "label": "LEAKING",
        "expected": set(),
        "expected_stage2": True,
    },
    {
        "id": "E20",
        "description": "Implicit credit card reference",
        "text": "My credit card details are already saved in your system.",
        "label": "LEAKING",
        "expected": set(),
        "expected_stage2": True,
    },
    {
        "id": "E21",
        "description": "Implicit email reference",
        "text": "Contact me at my personal email for further discussion.",
        "label": "LEAKING",
        "expected": set(),
        "expected_stage2": True,
    },
    {
        "id": "E22",
        "description": "Implicit SSN reference",
        "text": "The social security number I gave you last week is on file.",
        "label": "LEAKING",
        "expected": set(),
        "expected_stage2": True,
    },
    {
        "id": "E23",
        "description": "Implicit bank account reference",
        "text": "Please send the wire to the bank account number on the invoice.",
        "label": "LEAKING",
        "expected": set(),
        "expected_stage2": True,
    },
    {
        "id": "E24",
        "description": "Implicit phone reference",
        "text": "You can reach me at my home phone anytime after 6pm.",
        "label": "LEAKING",
        "expected": set(),
        "expected_stage2": True,
    },
    {
        "id": "E25",
        "description": "Implicit CNIC/passport reference",
        "text": "My CNIC and passport details are attached to the form I submitted.",
        "label": "LEAKING",
        "expected": set(),
        "expected_stage2": True,
    },
]

# Stage 1 gate — lower because inference cases (E19-E25) require Stage 2
# Stage 1 is only expected to catch verbatim PII (E01-E10)
PRECISION_GATE    = 0.90
RECALL_GATE       = 0.55
F1_GATE           = 0.70

# Stage 2 gate — full pipeline should catch all 25 cases
STAGE2_PRECISION_GATE = 0.90
STAGE2_RECALL_GATE    = 0.95
STAGE2_F1_GATE        = 0.95


def run_evaluation(cases: list = None, use_stage2: bool = False,
                   verbose: bool = True) -> dict:
    if cases is None:
        cases = EVAL_CASES

    details = []

    for case in cases:
        result = detect_pii(case["text"], use_stage2=use_stage2)
        detected = {e["type"] for e in result["entities"]}
        expected = case.get("expected", set())

        is_leaking = case["label"] == "LEAKING"
        predicted_leaking = result["risk_level"] != "CLEAN"

        if is_leaking and predicted_leaking:
            label = "TP"
        elif not is_leaking and not predicted_leaking:
            label = "TN"
        elif not is_leaking and predicted_leaking:
            label = "FP"
        else:
            label = "FN"

        details.append({
            "id": case["id"],
            "description": case.get("description", ""),
            "label": label,
            "risk_level": result["risk_level"],
            "expected": sorted(expected),
            "detected": sorted(detected),
            "missed": sorted(expected - detected),
            "extra": sorted(detected - expected),
            "expected_stage2": case.get("expected_stage2", False),
            "stage2_flagged": result.get("stage2_flagged"),
        })

    tp = sum(1 for d in details if d["label"] == "TP")
    tn = sum(1 for d in details if d["label"] == "TN")
    fp = sum(1 for d in details if d["label"] == "FP")
    fn = sum(1 for d in details if d["label"] == "FN")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = (tp + tn) / len(details) if details else 0.0

    if verbose:
        _print_report(details, tp, tn, fp, fn, precision, recall, f1, accuracy,
                      use_stage2=use_stage2)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "details": details,
    }


def _print_report(details, tp, tn, fp, fn, precision, recall, f1, accuracy,
                  use_stage2: bool = False):
    W = 88
    mode = "Stage 1 + Stage 2 (LLM-as-judge)" if use_stage2 else "Stage 1 (Presidio)"
    p_gate = STAGE2_PRECISION_GATE if use_stage2 else PRECISION_GATE
    r_gate = STAGE2_RECALL_GATE    if use_stage2 else RECALL_GATE
    f_gate = STAGE2_F1_GATE        if use_stage2 else F1_GATE

    print("=" * W)
    print("  PII Leakage Detection — Precision / Recall Evaluation")
    print(f"  CNIT/PNTLab Pisa · AI Security Internship 2026  |  Pipeline: {mode}")
    print("=" * W)

    header = f"{'':2} {'ID':<5} {'Class':<5} {'Risk':<8} {'Missed entity types':<26} {'Extra (FP types)':<20} {'S2'}"
    print(f"\n{header}")
    print("-" * W)

    for d in details:
        ok     = "✓" if d["label"] in ("TP", "TN") else "✗"
        missed = ", ".join(d["missed"]) if d["missed"] else "—"
        extra  = ", ".join(d["extra"])  if d["extra"]  else "—"
        s2     = "★" if d.get("stage2_flagged") else " "
        print(f"{ok} {d['id']:<5} {d['label']:<5} {d['risk_level']:<8} {missed:<26} {extra:<20} {s2}")

    if use_stage2:
        print("  ★ = escalated by Stage 2 (LLM-as-judge)")

    print("\n" + "=" * W)
    print("  Confusion matrix")
    print("=" * W)
    print(f"  {'':18} Predicted LEAKING   Predicted CLEAN")
    print(f"  {'Actually LEAKING':<20} TP = {tp:<15}  FN = {fn}")
    print(f"  {'Actually CLEAN':<20} FP = {fp:<15}  TN = {tn}")

    print("\n" + "=" * W)
    print("  Metrics (text-level binary classification)")
    print("=" * W)
    print(f"  Precision  {precision:6.3f}   TP/(TP+FP)  = {tp}/{tp+fp}  (gate ≥ {p_gate})")
    print(f"  Recall     {recall:6.3f}   TP/(TP+FN)  = {tp}/{tp+fn}  (gate ≥ {r_gate})")
    print(f"  F1 score   {f1:6.3f}                           (gate ≥ {f_gate})")
    print(f"  Accuracy   {accuracy:6.3f}   (TP+TN)/N   = {tp+tn}/{len(details)}")
    print()

    all_missed = []
    for d in details:
        all_missed.extend(d["missed"])
    if all_missed:
        missed_counts = Counter(all_missed)
        print("  Entity types most commonly missed:")
        for etype, count in missed_counts.most_common():
            print(f"    {count}×  {etype}")
        print()

    if use_stage2:
        s2_catches = [d for d in details if d.get("stage2_flagged")]
        print(f"  Stage 2 escalations: {len(s2_catches)}")
        for d in s2_catches:
            print(f"    {d['id']}  {d['description'][:55]}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Hand-labelled eval suite")
    parser.add_argument("--v2", action="store_true",
                        help="Use two-stage pipeline (Stage 1 + LLM-as-judge)")
    parser.add_argument("--output-dir", type=str, default="experiments/results")
    args = parser.parse_args()

    metrics = run_evaluation(use_stage2=args.v2)

    if args.v2:
        passed = (
            metrics["precision"] >= STAGE2_PRECISION_GATE and
            metrics["recall"]    >= STAGE2_RECALL_GATE    and
            metrics["f1"]        >= STAGE2_F1_GATE
        )
    else:
        passed = (
            metrics["precision"] >= PRECISION_GATE and
            metrics["recall"]    >= RECALL_GATE    and
            metrics["f1"]        >= F1_GATE
        )

    print(f"  Gate {'PASSED ✓' if passed else 'FAILED ✗'}")

    os.makedirs(args.output_dir, exist_ok=True)
    suffix = "v2" if args.v2 else "v1"
    out_path = os.path.join(args.output_dir, f"eval_suite_{suffix}.json")
    with open(out_path, "w") as f:
        json.dump({
            "metrics": {
                "precision": metrics["precision"],
                "recall":    metrics["recall"],
                "f1":        metrics["f1"],
                "accuracy":  metrics["accuracy"],
                "tp": metrics["tp"], "fp": metrics["fp"],
                "fn": metrics["fn"], "tn": metrics["tn"],
            },
            "details": metrics["details"],
        }, f, indent=2)
    print(f"  Results → {out_path}\n")

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()