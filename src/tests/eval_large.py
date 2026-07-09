"""
eval_large.py — Large-Scale Evaluation Suite
CNIT/PNTLab Pisa — AI Security Internship 2026
Student: Muhammad Hashim Mughal | Week: 04

"""

import argparse
import json
import os
import sys
import time
from typing import Any


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.detector import PIIDetector  # noqa: E402

DETECTOR = PIIDetector()

DEFAULT_DATASET = os.path.join(ROOT, "experiments", "results", "synthetic_dataset.json")



def load_dataset(path: str) -> list[dict[str, Any]]:
    """Load the synthetic dataset JSON produced by synthetic_gen.py."""
    if not os.path.exists(path):
        print(f"[ERROR] Dataset not found: {path}")
        print("  Run: python src/synthetic_gen.py")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples from {path}")
    leaking = sum(1 for s in data if s["label"] == "LEAKING")
    clean   = sum(1 for s in data if s["label"] == "CLEAN")
    print(f"  Leaking : {leaking}")
    print(f"  Clean   : {clean}")
    return data


def run_detector(text: str) -> tuple[str, float]:
    """
    Run the PII detector on a single text.

    Returns
    -------
    predicted_label : "LEAKING" or "CLEAN"
    latency_ms      : wall-clock time in milliseconds
    """
    t0 = time.perf_counter()
    result = DETECTOR.analyze(text)
    latency_ms = (time.perf_counter() - t0) * 1000

    predicted = "CLEAN" if result["risk_level"] == "CLEAN" else "LEAKING"
    return predicted, latency_ms


def evaluate(dataset: list[dict[str, Any]]) -> dict[str, Any]:
    """Run full evaluation pipeline. Returns results dict."""
    results: dict[str, Any] = {
        "total": len(dataset),
        "binary_classification": {},
        "per_entity_metrics": {},
        "latency": {},
        "samples": [],
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="Large-scale PII detector evaluation")
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help="Path to synthetic_dataset.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(ROOT, "experiments", "results"),
        help="Directory to write evaluation results",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    results = evaluate(dataset)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "synthetic_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()