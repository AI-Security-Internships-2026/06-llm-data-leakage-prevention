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
    t0 = time.perf_counter()
    result = DETECTOR.analyze(text)
    latency_ms = (time.perf_counter() - t0) * 1000

    predicted = "CLEAN" if result["risk_level"] == "CLEAN" else "LEAKING"
    return predicted, latency_ms



def binary_classification(dataset: list[dict[str, Any]]) -> tuple[dict, list[dict]]:
  
    TP = FP = TN = FN = 0
    sample_results = []
    latencies = []

    print(f"\nRunning binary classification on {len(dataset)} samples...")

    for i, sample in enumerate(dataset):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{len(dataset)}")

        true_label = sample["label"]           # "LEAKING" or "CLEAN"
        pred_label, latency_ms = run_detector(sample["text"])
        latencies.append(latency_ms)

        if true_label == "LEAKING" and pred_label == "LEAKING":
            TP += 1
        elif true_label == "CLEAN" and pred_label == "LEAKING":
            FP += 1
        elif true_label == "CLEAN" and pred_label == "CLEAN":
            TN += 1
        else:
            FN += 1

        sample_results.append({
            "id": sample["id"],
            "true_label": true_label,
            "pred_label": pred_label,
            "entity_types": sample.get("entity_types", []),
            "latency_ms": round(latency_ms, 3),
            "correct": true_label == pred_label,
        })

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = (TP + TN) / len(dataset) if dataset else 0.0

    metrics = {
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "accuracy":  round(accuracy, 4),
    }

    return metrics, sample_results, latencies


def print_confusion_matrix(metrics: dict) -> None:
    TP, FP = metrics["TP"], metrics["FP"]
    FN, TN = metrics["FN"], metrics["TN"]
    print("\n  Confusion Matrix")
    print("  ─────────────────────────────────")
    print("                Pred LEAKING  Pred CLEAN")
    print(f"  True LEAKING     {TP:>6}        {FN:>6}")
    print(f"  True CLEAN       {FP:>6}        {TN:>6}")
    print("  ─────────────────────────────────")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")




def evaluate(dataset: list[dict[str, Any]]) -> dict[str, Any]:
    """Run full evaluation pipeline. Returns results dict."""
    metrics, sample_results, latencies = binary_classification(dataset)
    print_confusion_matrix(metrics)

    results: dict[str, Any] = {
        "total": len(dataset),
        "binary_classification": metrics,
        "per_entity_metrics": {},   
        "latency": {},            
        "samples": sample_results,
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