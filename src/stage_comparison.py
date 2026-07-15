"""
stage_comparison.py — Stage 1 vs Stage 1+2 Pipeline Comparison
CNIT/PNTLab Pisa — AI Security Internship 2026
Student: Muhammad Hashim Mughal | Week: 05

"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from detector import detect_pii  # noqa: E402

# ── Adversarial cases (inline, no file needed) ────────────────────────────────
_ADVERSARIAL_CASES = [
    # Obfuscated emails
    {"text": "Send to alice [at] example [dot] com", "label": "LEAKING"},
    {"text": "Contact bob(at)company.org for details", "label": "LEAKING"},
    {"text": "Email jane AT example DOT com for the report", "label": "LEAKING"},
    {"text": "Reach me at carol AT example.com", "label": "LEAKING"},
    # Spaced / formatted cards
    {"text": "Card: 4111 1111 1111 1111", "label": "LEAKING"},
    {"text": "Charge card 4111-1111-1111-1111 today", "label": "LEAKING"},
    {"text": "Card number: 4111.1111.1111.1111", "label": "LEAKING"},
    # Formatted phones
    {"text": "Call 800.555.0199 for support", "label": "LEAKING"},
    {"text": "UK line: +44 20 7946 0958", "label": "LEAKING"},
    # SSN / CNIC
    {"text": "Taxpayer ID: 078-05-1120", "label": "LEAKING"},
    {"text": "CNIC: 35202-1234567-8 verified", "label": "LEAKING"},
    # IBAN
    {"text": "Wire to IBAN GB29NWBK60161331926819", "label": "LEAKING"},
    {"text": "Bank account: DE89370400440532013000", "label": "LEAKING"},
    # PII embedded in structured text
    {"text": '{"email": "ops@company.org", "card": "4111111111111111"}', "label": "LEAKING"},
    {"text": "| Name | Email | Phone |\n| Alice | alice@example.com | 555-1234 |", "label": "LEAKING"},
    # Multi-PII
    {"text": "John Smith SSN 078-05-1120 card 4111111111111111", "label": "LEAKING"},
    {"text": "Name: Dr. Sarah Connor. CNIC: 35202-9876543-1. Email: s.connor@example.com", "label": "LEAKING"},
    # Inference-based (Stage 2 should help)
    {"text": "My home phone number is listed above", "label": "LEAKING"},
    {"text": "Please use the iban I provided earlier for the transfer", "label": "LEAKING"},
    {"text": "Contact me at my personal email for further discussion", "label": "LEAKING"},
    # Clean
    {"text": "The REST API returns JSON with HTTP 200.", "label": "CLEAN"},
    {"text": "def add(x, y): return x + y", "label": "CLEAN"},
    {"text": "Gradient descent converged after 150 epochs.", "label": "CLEAN"},
    {"text": "The firewall blocks port 22 except from the VPN.", "label": "CLEAN"},
    {"text": "Kubernetes pod restarted due to OOMKilled.", "label": "CLEAN"},
]


def _load_synthetic(path: str, max_samples: int = 400) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [{"text": s["text"], "label": s["label"]} for s in data[:max_samples]]


def _run_mode(samples: list[dict], use_stage2: bool) -> tuple[dict, list[float]]:
    """Run all samples through the pipeline in the given mode."""
    tp = fp = fn = tn = 0
    latencies = []

    for s in samples:
        t0 = time.perf_counter()
        result = detect_pii(s["text"], use_stage2=use_stage2)
        latencies.append((time.perf_counter() - t0) * 1000)

        true_leaking = s["label"] == "LEAKING"
        pred_leaking = result["risk_level"] != "CLEAN"

        if true_leaking and pred_leaking:
            tp += 1
        elif not true_leaking and not pred_leaking:
            tn += 1
        elif not true_leaking and pred_leaking:
            fp += 1
        else:
            fn += 1

    total = len(samples)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = (tp + tn) / total if total > 0 else 0.0

    sorted_lat = sorted(latencies)
    n = len(sorted_lat)

    return {
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "accuracy":  round(accuracy, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "total": total,
        "latency": {
            "mean_ms": round(sum(latencies) / n, 3) if n else 0,
            "p50_ms":  round(sorted_lat[int(n * 0.50)], 3) if n else 0,
            "p95_ms":  round(sorted_lat[int(n * 0.95)], 3) if n else 0,
        },
    }, latencies


def _print_comparison(s1: dict, s2: dict, dataset_name: str):
    print(f"\n  Stage 1 vs Stage 1+2 Comparison — {dataset_name}")
    print(f"  {'Metric':<14} {'Stage 1':>10} {'Stage 1+2':>10} {'Delta':>10}")
    print("  " + "─" * 46)
    for key in ("precision", "recall", "f1", "accuracy"):
        v1 = s1[key]
        v2 = s2[key]
        delta = v2 - v1
        sign = "+" if delta >= 0 else ""
        print(f"  {key.capitalize():<14} {v1:>10.4f} {v2:>10.4f} {sign+f'{delta:.4f}':>10}")
    print("  " + "─" * 46)
    print(f"  {'p50 latency':<14} {s1['latency']['p50_ms']:>9.1f}ms "
          f"{s2['latency']['p50_ms']:>9.1f}ms")
    print(f"  {'p95 latency':<14} {s1['latency']['p95_ms']:>9.1f}ms "
          f"{s2['latency']['p95_ms']:>9.1f}ms")


def main():
    parser = argparse.ArgumentParser(description="Stage 1 vs Stage 1+2 comparison")
    parser.add_argument("--dataset", type=str,
                        default=os.path.join(ROOT, "experiments", "results",
                                             "synthetic_dataset.json"))
    parser.add_argument("--adversarial", action="store_true",
                        help="Use built-in adversarial cases instead of synthetic dataset")
    parser.add_argument("--max-samples", type=int, default=400)
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(ROOT, "experiments", "results"))
    args = parser.parse_args()

    if args.adversarial:
        samples = _ADVERSARIAL_CASES
        dataset_name = "adversarial (25 cases)"
    else:
        if not os.path.exists(args.dataset):
            print(f"[ERROR] Dataset not found: {args.dataset}")
            print("Run: python src/synthetic_gen.py  to generate it first.")
            sys.exit(1)
        samples = _load_synthetic(args.dataset, args.max_samples)
        dataset_name = f"synthetic ({len(samples)} samples)"

    print(f"\nRunning Stage 1 on {len(samples)} samples...")
    s1_metrics, _ = _run_mode(samples, use_stage2=False)

    print(f"Running Stage 1+2 on {len(samples)} samples...")
    s2_metrics, _ = _run_mode(samples, use_stage2=True)

    _print_comparison(s1_metrics, s2_metrics, dataset_name)

    output = {
        "meta": {
            "run_at": datetime.now(timezone.utc).isoformat(),
            "dataset": dataset_name,
            "samples": len(samples),
        },
        "stage1": s1_metrics,
        "stage1_plus_stage2": s2_metrics,
        "delta": {
            "precision": round(s2_metrics["precision"] - s1_metrics["precision"], 4),
            "recall":    round(s2_metrics["recall"]    - s1_metrics["recall"],    4),
            "f1":        round(s2_metrics["f1"]        - s1_metrics["f1"],        4),
            "accuracy":  round(s2_metrics["accuracy"]  - s1_metrics["accuracy"],  4),
        },
    }

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "stage_comparison.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()