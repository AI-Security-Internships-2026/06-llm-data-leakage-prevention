"""
eval_suite.py — Precision / Recall Evaluation for PII Leakage Detection
CNIT/PNTLab Pisa — AI Security Internship 2026
Student: Muhammad Hashim Mughal | Week: 03

Usage:
    cd 06-llm-data-leakage-prevention
    python src/tests/eval_suite.py

What it measures
----------------
Text-level binary classification:
  - Positive  = text contains PII  (leaking)
  - Negative  = text is PII-free   (non-leaking)

  TP  detector flags a leaking text
  TN  detector passes a clean text
  FP  detector flags a clean text  (false alarm)
  FN  detector misses a leaking text

Derived metrics:
  Precision  = TP / (TP + FP)   — "when we cry wolf, are we right?"
  Recall     = TP / (TP + FN)   — "do we catch all the leaks?"
  F1         = harmonic mean of precision and recall
  Accuracy   = (TP + TN) / total

Entity-type coverage (informational):
  For each leaking case, shows which expected entity types were detected
  and which were missed — useful for identifying gaps in Presidio coverage.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from detector import detect_pii


# ---------------------------------------------------------------------------
# Labeled dataset
# ---------------------------------------------------------------------------
# Each entry:
#   id          — unique case ID (L = leaking, N = non-leaking)
#   description — human-readable label for the report
#   text        — the input string
#   expected    — set of PII entity types expected (empty set = no PII)

EVAL_CASES = [
    # ── LEAKING ─────────────────────────────────────────────────────────────
    {
        "id": "L01",
        "description": "Email address in plain prose",
        "text": "Please contact John at john.doe@example.com for more information.",
        "expected": {"EMAIL_ADDRESS"},
    },
    {
        "id": "L02",
        "description": "Visa credit card number",
        "text": "Payment was processed using card number 4111111111111111.",
        "expected": {"CREDIT_CARD"},
    },
    {
        "id": "L03",
        "description": "US Social Security Number",
        "text": "The applicant's SSN is 078-05-1120.",
        "expected": {"US_SSN"},
    },
    {
        "id": "L04",
        "description": "International phone number",
        "text": "For support call +1-800-555-0199 during business hours.",
        "expected": {"PHONE_NUMBER"},
    },
    {
        "id": "L05",
        "description": "IBAN bank account number",
        "text": "Wire the balance to IBAN GB29NWBK60161331926819 by end of month.",
        "expected": {"IBAN_CODE"},
    },
    {
        "id": "L06",
        "description": "Multiple PII types (name + email + phone)",
        "text": "Jane Smith's email is jane.smith@corp.com and her direct line is +44 20 7946 0958.",
        "expected": {"PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"},
    },
    {
        "id": "L07",
        "description": "PII embedded inside error log line",
        "text": "ERROR 2026-06-20T14:32Z user=ops@company.org quota_exceeded=true",
        "expected": {"EMAIL_ADDRESS"},
    },
    {
        "id": "L08",
        "description": "Medical context with patient name",
        "text": "Dr. Sarah Connor prescribed metformin 500mg to patient Robert Miles.",
        "expected": {"PERSON"},
    },
    {
        "id": "L09",
        "description": "Pakistani CNIC number",
        "text": "National Identity Number: 35202-1234567-8",
        "expected": {"PK_CNIC"},
    },
    {
        "id": "L10",
        "description": "PII in JSON-like payload",
        "text": '{"user": "alice", "email": "alice@internal.io", "card": "4111111111111111"}',
        "expected": {"EMAIL_ADDRESS", "CREDIT_CARD"},
    },
    # ── NON-LEAKING ─────────────────────────────────────────────────────────
    {
        "id": "N01",
        "description": "Technical API documentation",
        "text": "The REST endpoint accepts JSON over HTTPS and returns a 200 status on success.",
        "expected": set(),
    },
    {
        "id": "N02",
        "description": "Python code snippet",
        "text": "def detect(text: str) -> dict:\n    return analyze(text)",
        "expected": set(),
    },
    {
        "id": "N03",
        "description": "Medical terminology without personal info",
        "text": "Hypertension is treated with ACE inhibitors and calcium channel blockers.",
        "expected": set(),
    },
    {
        "id": "N04",
        "description": "Generic quarterly report sentence",
        "text": "Quarterly revenue grew 12% year-over-year, driven by SaaS subscriptions.",
        "expected": set(),
    },
    {
        "id": "N05",
        "description": "Scientific experiment description",
        "text": "The sample was heated to 250°C for 30 minutes inside a sealed autoclave.",
        "expected": set(),
    },
    {
        "id": "N06",
        "description": "Neutral news-style sentence",
        "text": "The government announced updated energy consumption guidelines this quarter.",
        "expected": set(),
    },
    {
        "id": "N07",
        "description": "Placeholder variable names (not real people)",
        "text": "We compare Model A against Model B on the standard benchmark dataset.",
        "expected": set(),
    },
    {
        "id": "N08",
        "description": "Policy statement with no personal data",
        "text": "No personal information is collected or stored by this service.",
        "expected": set(),
    },
    {
        "id": "N09",
        "description": "Pipeline/infrastructure description with no personal data",
        "text": "The pipeline runs every 6 hours and writes output to the data warehouse.",
        "expected": set(),
    },
    {
        "id": "N10",
        "description": "Configuration note with no personal data",
        "text": "All configuration values are loaded from environment variables at startup.",
        "expected": set(),
    },
]


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------

def run_evaluation(cases: list = None, verbose: bool = True) -> dict:
    """
    Run detect_pii on every labeled case and return precision/recall metrics.

    Parameters
    ----------
    cases   : list of case dicts; defaults to the module-level EVAL_CASES
    verbose : print the per-case table and summary to stdout

    Returns
    -------
    dict with keys: precision, recall, f1, accuracy, tp, fp, fn, tn, details
    """
    if cases is None:
        cases = EVAL_CASES

    details = []

    for case in cases:
        result = detect_pii(case["text"])
        detected = {e["type"] for e in result["entities"]}
        expected = case["expected"]

        is_leaking = bool(expected)
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
            "description": case["description"],
            "label": label,
            "risk_level": result["risk_level"],
            "expected": sorted(expected),
            "detected": sorted(detected),
            "missed": sorted(expected - detected),
            "extra": sorted(detected - expected),
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
        _print_report(details, tp, tn, fp, fn, precision, recall, f1, accuracy)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "details": details,
    }


def _print_report(details, tp, tn, fp, fn, precision, recall, f1, accuracy):
    W = 80
    print("=" * W)
    print("  PII Leakage Detection — Precision / Recall Evaluation")
    print(f"  CNIT/PNTLab Pisa · AI Security Internship 2026")
    print("=" * W)

    header = f"{'':2} {'ID':<5} {'Class':<5} {'Risk':<8} {'Missed entity types':<26} {'Extra (FP types)'}"
    print(f"\n{header}")
    print("-" * W)

    for d in details:
        ok = "✓" if d["label"] in ("TP", "TN") else "✗"
        missed = ", ".join(d["missed"]) if d["missed"] else "—"
        extra  = ", ".join(d["extra"])  if d["extra"]  else "—"
        print(f"{ok} {d['id']:<5} {d['label']:<5} {d['risk_level']:<8} {missed:<26} {extra}")

    print("\n" + "=" * W)
    print("  Confusion matrix")
    print("=" * W)
    print(f"  {'':18} Predicted LEAKING   Predicted CLEAN")
    print(f"  {'Actually LEAKING':<20} TP = {tp:<15}  FN = {fn}")
    print(f"  {'Actually CLEAN':<20} FP = {fp:<15}  TN = {tn}")

    print("\n" + "=" * W)
    print("  Metrics (text-level binary classification)")
    print("=" * W)
    print(f"  Precision  {precision:6.3f}   TP/(TP+FP)  = {tp}/{tp+fp}")
    print(f"  Recall     {recall:6.3f}   TP/(TP+FN)  = {tp}/{tp+fn}")
    print(f"  F1 score   {f1:6.3f}")
    print(f"  Accuracy   {accuracy:6.3f}   (TP+TN)/N   = {tp+tn}/{len(details)}")
    print()

    # Missed entity analysis
    all_missed = []
    for d in details:
        all_missed.extend(d["missed"])
    if all_missed:
        from collections import Counter
        missed_counts = Counter(all_missed)
        print("  Entity types most commonly missed:")
        for etype, count in missed_counts.most_common():
            print(f"    {count}×  {etype}")
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    metrics = run_evaluation()

    # Exit non-zero if recall drops below 0.80 — a hard floor for a
    # leakage-prevention tool (missing leaks is worse than false alarms).
    if metrics["recall"] < 0.80:
        print("⚠  Recall below 0.80 — too many leaking inputs are being missed.")
        sys.exit(1)