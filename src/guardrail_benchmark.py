"""
guardrail_benchmark.py — External Guardrail Framework Comparison
CNIT/PNTLab Pisa — AI Security Internship 2026
Student: Muhammad Hashim Mughal | Issue: #6

Security task
-------------
Text-level binary PII / secret leakage detection:
given a text string, decide LEAKING (1) or CLEAN (0).

Implementations compared
------------------------
A. Our detector  — src/detector.py  (Stage 1: Presidio + custom recognisers)
B. scrubadub     — scrubadub 2.0.1  + scrubadub-spacy 2.0.0 (SpacyEntityDetector
                   on en_core_web_lg) + EmailDetector, PhoneDetector,
                   CreditCardDetector, SSNDetector
C. detect-secrets — detect-secrets 1.4.0 (Protect AI)
                   Plugins: HexHighEntropyString, Base64HighEntropyString,
                   ArtifactoryDetector, AWSKeyDetector, BasicAuthDetector,
                   CloudantDetector, GitHubTokenDetector, MailchimpDetector,
                   PrivateKeyDetector, SlackDetector, StripeDetector,
                   TwilioKeyDetector, KeywordDetector
                   Note: detect-secrets targets credentials/secrets, not
                   personal PII. Low recall on PII cases is expected and
                   reported honestly per issue #6 constraints.

Evaluation dataset
------------------
30 hand-labelled cases from src/tests/eval_suite.py (E01–E30).
Same cases, same order, no per-framework tuning.

Reproduction
------------
    cd 06-llm-data-leakage-prevention
    python src/guardrail_benchmark.py

Output
------
    experiments/results/guardrail_comparison.json
"""

from __future__ import annotations

import sys
import os
import json
import time
import platform
import importlib.metadata
from statistics import median
from typing import Callable

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# ── eval dataset (same 30 cases as src/tests/eval_suite.py E01–E30) ──────────
EVAL_CASES = [
    # ── LEAKING — verbatim PII (Week 03) ─────────────────────────────────
    {"id": "E01", "text": "Please contact john.doe@example.com for further information.",
     "label": "LEAKING", "description": "Email address in plain prose"},
    {"id": "E02", "text": "Payment processed with card number 4111111111111111.",
     "label": "LEAKING", "description": "Visa credit card number"},
    {"id": "E03", "text": "The applicant's SSN is 078-05-1120.",
     "label": "LEAKING", "description": "US Social Security Number"},
    {"id": "E04", "text": "For support please call +1-800-555-0199 during business hours.",
     "label": "LEAKING", "description": "International phone number"},
    {"id": "E05", "text": "Wire the balance to IBAN GB29NWBK60161331926819 by end of month.",
     "label": "LEAKING", "description": "IBAN bank account number"},
    {"id": "E06", "text": "Jane Smith's email is jane.smith@corp.com and her direct line is +44 20 7946 0958.",
     "label": "LEAKING", "description": "Multiple PII types (name + email + phone)"},
    {"id": "E07", "text": "ERROR 2026-06-20T14:32Z user=ops@company.org quota_exceeded=true",
     "label": "LEAKING", "description": "PII embedded inside error log line"},
    {"id": "E08", "text": "Dr. Sarah Connor prescribed metformin 500mg to patient Robert Miles.",
     "label": "LEAKING", "description": "Medical context with patient name"},
    {"id": "E09", "text": "National Identity Number: 35202-1234567-8",
     "label": "LEAKING", "description": "Pakistani CNIC number"},
    {"id": "E10", "text": '{"user": "alice", "email": "alice@internal.io", "card": "4111111111111111"}',
     "label": "LEAKING", "description": "PII in JSON-like payload"},
    # ── CLEAN (Week 03) ───────────────────────────────────────────────────
    {"id": "E11", "text": "The REST endpoint accepts JSON over HTTPS and returns a 200 status on success.",
     "label": "CLEAN", "description": "Technical API documentation"},
    {"id": "E12", "text": "def detect(text: str) -> dict:\n    return analyze(text)",
     "label": "CLEAN", "description": "Python code snippet"},
    {"id": "E13", "text": "Hypertension is treated with ACE inhibitors and calcium channel blockers.",
     "label": "CLEAN", "description": "Medical terminology without personal info"},
    {"id": "E14", "text": "Quarterly revenue grew 12% year-over-year, driven by SaaS subscriptions.",
     "label": "CLEAN", "description": "Generic quarterly report sentence"},
    {"id": "E15", "text": "The sample was heated to 250\u00b0C for 30 minutes inside a sealed autoclave.",
     "label": "CLEAN", "description": "Scientific experiment description"},
    {"id": "E16", "text": "The government announced updated energy consumption guidelines this quarter.",
     "label": "CLEAN", "description": "Neutral news-style sentence"},
    {"id": "E17", "text": "We compare Model A against Model B on the standard benchmark dataset.",
     "label": "CLEAN", "description": "Placeholder variable names (not real people)"},
    {"id": "E18", "text": "No personal information is collected or stored by this service.",
     "label": "CLEAN", "description": "Policy statement with no personal data"},
    # ── LEAKING — inference-based / implicit PII (Week 05) ────────────────
    {"id": "E19", "text": "Please use the iban I provided earlier for the transfer.",
     "label": "LEAKING", "description": "Implicit IBAN reference (no literal IBAN)"},
    {"id": "E20", "text": "My credit card details are already saved in your system.",
     "label": "LEAKING", "description": "Implicit credit card reference"},
    {"id": "E21", "text": "Contact me at my personal email for further discussion.",
     "label": "LEAKING", "description": "Implicit email reference"},
    {"id": "E22", "text": "The social security number I gave you last week is on file.",
     "label": "LEAKING", "description": "Implicit SSN reference"},
    {"id": "E23", "text": "Please send the wire to the bank account number on the invoice.",
     "label": "LEAKING", "description": "Implicit bank account reference"},
    {"id": "E24", "text": "You can reach me at my home phone anytime after 6pm.",
     "label": "LEAKING", "description": "Implicit phone reference"},
    {"id": "E25", "text": "My CNIC and passport details are attached to the form I submitted.",
     "label": "LEAKING", "description": "Implicit CNIC/passport reference"},
    # ── CLEAN — FP regression cases (Week 06) ────────────────────────────
    {"id": "E26", "text": "The experiment reference code ABCDE1234F has been logged in the system.",
     "label": "CLEAN", "description": "Alphanumeric ref code (previously FP: IN_PAN)"},
    {"id": "E27", "text": "Your order number is D12345678. Expected delivery in 3-5 business days.",
     "label": "CLEAN", "description": "Order number with letter prefix (previously FP: US_DRIVER_LICENSE)"},
    {"id": "E28", "text": "Your order ID is ORD-2024-001. Please keep this for your records.",
     "label": "CLEAN", "description": "Long numeric order ID (previously FP: US_BANK_NUMBER)"},
    {"id": "E29", "text": "Track your parcel using tracking number 9876543210123.",
     "label": "CLEAN", "description": "Shipment tracking number (previously FP: US_BANK_NUMBER)"},
    {"id": "E30", "text": "Please quote ticket ID XKCD5678A when contacting support.",
     "label": "CLEAN", "description": "Support ticket ID (previously FP: IN_PAN)"},
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _metrics(results: list[dict]) -> dict:
    """Compute binary classification metrics from per-case results."""
    tp = sum(1 for r in results if r["predicted"] == "LEAKING" and r["ground_truth"] == "LEAKING")
    fp = sum(1 for r in results if r["predicted"] == "LEAKING" and r["ground_truth"] == "CLEAN")
    tn = sum(1 for r in results if r["predicted"] == "CLEAN"   and r["ground_truth"] == "CLEAN")
    fn = sum(1 for r in results if r["predicted"] == "CLEAN"   and r["ground_truth"] == "LEAKING")

    precision  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall     = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1         = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)
    fpr        = fp / (fp + tn) if (fp + tn) > 0 else 0.0   # false-positive rate
    fnr        = fn / (fn + tp) if (fn + tp) > 0 else 0.0   # false-negative rate
    accuracy   = (tp + tn) / len(results) if results else 0.0

    latencies  = [r["latency_ms"] for r in results]
    latencies.sort()
    p50 = round(median(latencies), 3)
    p95_idx = max(0, int(len(latencies) * 0.95) - 1)
    p95 = round(latencies[p95_idx], 3)
    total_s = sum(latencies) / 1000
    throughput = round(len(results) / total_s, 2) if total_s > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "accuracy":  round(accuracy,  4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "throughput_samples_per_sec": throughput,
    }


def _run(name: str, fn: Callable[[str], bool], cases: list[dict]) -> dict:
    """Run one detector over all cases and return structured results."""
    print(f"\n[{name}] running {len(cases)} cases...")
    per_case = []
    failures = 0
    for c in cases:
        t0 = time.perf_counter()
        try:
            flagged = fn(c["text"])
            error = None
        except Exception as e:
            flagged = False
            error = str(e)
            failures += 1
        elapsed_ms = (time.perf_counter() - t0) * 1000

        predicted = "LEAKING" if flagged else "CLEAN"
        correct   = predicted == c["label"]
        print(f"  {c['id']} [{c['label']:7s}] → {predicted:7s} {'✓' if correct else '✗'}  ({elapsed_ms:.1f}ms)")

        per_case.append({
            "case_id":      c["id"],
            "description":  c["description"],
            "ground_truth": c["label"],
            "predicted":    predicted,
            "correct":      correct,
            "latency_ms":   round(elapsed_ms, 3),
            "error":        error,
        })

    m = _metrics(per_case)
    print(f"  → P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  "
          f"TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}  "
          f"failures={failures}")
    return {"metrics": m, "execution_failures": failures, "per_case": per_case}


# ── Implementation A: Our detector (Stage 1) ─────────────────────────────────

def _build_our_detector():
    from detector import detect_pii
    def _detect(text: str) -> bool:
        result = detect_pii(text, use_stage2=False)
        return result["risk_level"] != "CLEAN"
    return _detect


# ── Implementation B: scrubadub ───────────────────────────────────────────────

def _build_scrubadub():
    import scrubadub
    import scrubadub_spacy

    # TextBlobNameDetector excluded — requires NLTK punkt_tab corpus not available here.
    # SpacyEntityDetector (en_core_web_lg) covers PERSON/ORG/LOC instead.
    scrubber = scrubadub.Scrubber(detector_list=[
        scrubadub.detectors.EmailDetector,
        scrubadub.detectors.PhoneDetector,
        scrubadub.detectors.CreditCardDetector,
        scrubadub.detectors.CredentialDetector,
    ])
    scrubber.add_detector(scrubadub_spacy.detectors.SpacyEntityDetector(
        model="en_core_web_lg",
        named_entities=["PERSON", "ORG", "GPE", "LOC"],
    ))

    def _detect(text: str) -> bool:
        cleaned = scrubber.clean(text)
        return "{{" in cleaned   # scrubadub wraps redacted tokens in {{ }}

    return _detect


# ── Implementation C: detect-secrets (Protect AI) ────────────────────────────

def _build_detect_secrets():
    from detect_secrets import SecretsCollection
    from detect_secrets.settings import transient_settings

    config = {
        "plugins_used": [
            {"name": "HexHighEntropyString",    "limit": 3.0},
            {"name": "Base64HighEntropyString", "limit": 4.5},
            {"name": "ArtifactoryDetector"},
            {"name": "AWSKeyDetector"},
            {"name": "BasicAuthDetector"},
            {"name": "CloudantDetector"},
            {"name": "GitHubTokenDetector"},
            {"name": "MailchimpDetector"},
            {"name": "PrivateKeyDetector"},
            {"name": "SlackDetector"},
            {"name": "StripeDetector"},
            {"name": "TwilioKeyDetector"},
            {"name": "KeywordDetector", "keyword_exclude": ""},
        ],
        "filters_used": [],
    }

    def _detect(text: str) -> bool:
        with transient_settings(config):
            secrets = SecretsCollection()
            secrets.scan_file.__func__   # confirm method exists
            import tempfile, os
            # detect-secrets works on files; write text to a temp file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            ) as f:
                f.write(text)
                tmp = f.name
            try:
                secrets.scan_file(tmp)
            finally:
                os.unlink(tmp)
            return len(secrets) > 0

    return _detect


# ── version helpers ───────────────────────────────────────────────────────────

def _pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "unknown"


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    out_path = os.path.join(
        os.path.dirname(__file__), "..",
        "experiments", "results", "guardrail_comparison.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print("=" * 60)
    print("Guardrail Benchmark — Issue #6")
    print(f"Cases: {len(EVAL_CASES)}  |  "
          f"LEAKING: {sum(1 for c in EVAL_CASES if c['label']=='LEAKING')}  "
          f"CLEAN: {sum(1 for c in EVAL_CASES if c['label']=='CLEAN')}")
    print("=" * 60)

    # ── build detectors ───────────────────────────────────────────────────
    print("\nLoading detectors...")

    our_fn = _build_our_detector()
    print("  [A] our detector       OK")

    scrubadub_fn = _build_scrubadub()
    print("  [B] scrubadub          OK")

    ds_fn = _build_detect_secrets()
    print("  [C] detect-secrets     OK")

    # ── run ───────────────────────────────────────────────────────────────
    results_our  = _run("A: Our detector (Stage 1)",    our_fn,       EVAL_CASES)
    results_scr  = _run("B: scrubadub 2.0.1",           scrubadub_fn, EVAL_CASES)
    results_ds   = _run("C: detect-secrets 1.4.0",      ds_fn,        EVAL_CASES)

    # ── assemble output ───────────────────────────────────────────────────
    output = {
        "meta": {
            "benchmark": "Issue #6 — PII and secret leakage detection comparison",
            "student": "Muhammad Hashim Mughal",
            "run_date": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ).isoformat(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "eval_cases_total": len(EVAL_CASES),
            "eval_cases_leaking": sum(1 for c in EVAL_CASES if c["label"] == "LEAKING"),
            "eval_cases_clean":   sum(1 for c in EVAL_CASES if c["label"] == "CLEAN"),
            "dataset_source": "src/tests/eval_suite.py cases E01–E30 (same as weekly eval suite)",
            "task": "Binary text-level PII/secret leakage detection (LEAKING vs CLEAN)",
        },
        "implementations": {
            "our_detector": {
                "name": "Our Detector — Stage 1",
                "framework": "Microsoft Presidio (repo implementation)",
                "component": "src/detector.py — detect_pii(use_stage2=False)",
                "version": {
                    "presidio_analyzer":  _pkg_version("presidio-analyzer"),
                    "presidio_anonymizer": _pkg_version("presidio-anonymizer"),
                    "spacy":              _pkg_version("spacy"),
                    "spacy_model":        "en_core_web_lg-3.7.1",
                },
                "configuration": {
                    "entities": [
                        "EMAIL_ADDRESS", "CREDIT_CARD", "PHONE_NUMBER", "PERSON",
                        "US_SSN", "IBAN_CODE", "PK_CNIC", "LOCATION",
                        "MEDICAL_LICENSE", "UK_NHS", "US_BANK_NUMBER",
                    ],
                    "custom_recognizers": ["PK_CNIC", "US_SSN (context-boosted)", "IBAN_CODE (context-boosted)"],
                    "excluded_entities": ["IN_PAN", "US_DRIVER_LICENSE"],
                    "score_gates": {"US_BANK_NUMBER": 0.80},
                    "normalize_text": True,
                    "stage2_llm_judge": False,
                },
                "threshold": "risk_level != CLEAN → LEAKING",
                "no_external_api": True,
                **results_our,
            },
            "scrubadub": {
                "name": "scrubadub + scrubadub-spacy",
                "framework": "scrubadub (independent open-source)",
                "component": "scrubadub.Scrubber + SpacyEntityDetector (en_core_web_lg)",
                "version": {
                    "scrubadub":       _pkg_version("scrubadub"),
                    "scrubadub_spacy": _pkg_version("scrubadub-spacy"),
                    "spacy":           _pkg_version("spacy"),
                    "spacy_model":     "en_core_web_lg-3.7.1",
                },
                "configuration": {
                    "detectors": [
                        "EmailDetector",
                        "PhoneDetector",
                        "CreditCardDetector",
                        "CredentialDetector",
                        "SpacyEntityDetector(PERSON, ORG, GPE, LOC) — replaces TextBlobNameDetector (NLTK punkt_tab unavailable)",
                    ],
                    "decision_rule": "any {{redacted}} token in output → LEAKING",
                },
                "threshold": "presence of scrubadub redaction markers",
                "no_external_api": True,
                **results_scr,
            },
            "detect_secrets": {
                "name": "detect-secrets (Protect AI)",
                "framework": "Protect AI — detect-secrets",
                "component": "SecretsCollection with all built-in plugins",
                "version": {
                    "detect_secrets": _pkg_version("detect-secrets"),
                },
                "configuration": {
                    "plugins": [
                        "HexHighEntropyString (limit=3.0, param: limit)",
                        "Base64HighEntropyString (limit=4.5, param: limit)",
                        "ArtifactoryDetector", "AWSKeyDetector",
                        "BasicAuthDetector", "CloudantDetector",
                        "GitHubTokenDetector", "MailchimpDetector",
                        "PrivateKeyDetector", "SlackDetector",
                        "StripeDetector", "TwilioKeyDetector",
                        "KeywordDetector",
                    ],
                    "note": (
                        "detect-secrets is designed for credential/secret scanning "
                        "(API keys, tokens, passwords), not general PII. "
                        "Low recall on PII cases (names, phone numbers, IBAN, CNIC) "
                        "is expected and reported honestly per issue #6 constraints. "
                        "It is included as the Protect AI representative, which is "
                        "listed in the issue requirements."
                    ),
                },
                "threshold": "len(SecretsCollection) > 0 → LEAKING",
                "no_external_api": True,
                **results_ds,
            },
        },
        "summary_comparison": {},
        "reproduction": {
            "commands": [
                "cd 06-llm-data-leakage-prevention",
                "pip install scrubadub scrubadub-spacy detect-secrets==1.4.0",
                "python src/guardrail_benchmark.py",
            ],
            "output_file": "experiments/results/guardrail_comparison.json",
            "note": (
                "llm-guard (0.3.10) and guardrails-ai (0.10.2) were attempted "
                "but could not be installed on Windows/Python 3.13: llm-guard "
                "pins sentencepiece==0.2.0 which requires a C++ build toolchain "
                "unavailable in this environment; guardrails-ai requires Rust/Cargo "
                "via the litellm dependency. Both are marked NOT_COMPARABLE below."
            ),
            "not_comparable": {
                "llm_guard": {
                    "reason": "sentencepiece==0.2.0 build failure on Windows/Python 3.13 (no MSVC toolchain)",
                    "attempted_version": "0.3.10",
                    "error": "FileNotFoundError: [WinError 2] sentencepiece wheel build failed",
                },
                "guardrails_ai": {
                    "reason": "litellm dependency requires Rust/Cargo on Windows; Cargo not on PATH",
                    "attempted_version": "0.10.2",
                    "error": "metadata-generation-failed: litellm requires Rust toolchain",
                },
                "nvidia_nemo_guardrails": {
                    "reason": "Not comparable — NeMo Guardrails is a conversational flow/policy "
                              "framework, not a PII detector. No equivalent PII scanner component.",
                },
                "meta_llama_firewall": {
                    "reason": "Not comparable — LlamaFirewall targets prompt injection and "
                              "agent safety, not text-level PII/secret scanning.",
                },
            },
        },
    }

    # ── summary table ─────────────────────────────────────────────────────
    for key, impl in output["implementations"].items():
        m = impl["metrics"]
        output["summary_comparison"][key] = {
            "precision": m["precision"],
            "recall":    m["recall"],
            "f1":        m["f1"],
            "accuracy":  m["accuracy"],
            "tp": m["tp"], "fp": m["fp"], "tn": m["tn"], "fn": m["fn"],
            "false_positive_rate": m["false_positive_rate"],
            "false_negative_rate": m["false_negative_rate"],
            "latency_p50_ms": m["latency_p50_ms"],
            "latency_p95_ms": m["latency_p95_ms"],
            "throughput_samples_per_sec": m["throughput_samples_per_sec"],
            "execution_failures": impl["execution_failures"],
        }

    # ── print summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    header = f"{'Impl':<30} {'P':>6} {'R':>6} {'F1':>6} {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4} {'p50ms':>7} {'p95ms':>7}"
    print(header)
    print("-" * len(header))
    labels = {
        "our_detector":  "A: Our detector (Stage 1)",
        "scrubadub":     "B: scrubadub 2.0.1",
        "detect_secrets":"C: detect-secrets 1.4.0",
    }
    for key, label in labels.items():
        s = output["summary_comparison"][key]
        print(
            f"{label:<30} "
            f"{s['precision']:>6.3f} {s['recall']:>6.3f} {s['f1']:>6.3f} "
            f"{s['tp']:>4} {s['fp']:>4} {s['tn']:>4} {s['fn']:>4} "
            f"{s['latency_p50_ms']:>7.1f} {s['latency_p95_ms']:>7.1f}"
        )

    # ── write JSON ────────────────────────────────────────────────────────
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults written to: {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()