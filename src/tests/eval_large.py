"""
eval_large.py — Large-Scale Evaluation Suite
CNIT/PNTLab Pisa — AI Security Internship 2026
Student: Muhammad Hashim Mughal | Week: 04

Evaluates the PII detector on the 1,200-sample synthetic dataset produced
by src/synthetic_gen.py. Measures text-level binary classification
(LEAKING vs CLEAN), per-entity-type precision/recall, and p50/p95 latency.

Usage
-----
    python src/tests/eval_large.py
    python src/tests/eval_large.py --dataset experiments/results/synthetic_dataset.json
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Any

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.detector import detect_pii  # noqa: E402


DEFAULT_DATASET = os.path.join(ROOT, "experiments", "results", "synthetic_dataset.json")


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> list[dict[str, Any]]:
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


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def run_detector(text: str) -> tuple[str, float]:
    t0 = time.perf_counter()
    result = detect_pii(text)
    latency_ms = (time.perf_counter() - t0) * 1000
    predicted = "CLEAN" if result["risk_level"] == "CLEAN" else "LEAKING"
    return predicted, latency_ms


# ---------------------------------------------------------------------------
# Binary classification
# ---------------------------------------------------------------------------

def binary_classification(dataset: list[dict[str, Any]]) -> tuple[dict, list[dict], list[float]]:
    TP = FP = TN = FN = 0
    sample_results = []
    latencies = []

    print(f"\nRunning binary classification on {len(dataset)} samples...")
    for i, sample in enumerate(dataset):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{len(dataset)}")

        true_label = sample["label"]
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


# ---------------------------------------------------------------------------
# Per-entity-type recall breakdown
# ---------------------------------------------------------------------------

def per_entity_metrics(sample_results: list[dict]) -> dict[str, dict]:
    entity_total: dict[str, int]    = defaultdict(int)
    entity_detected: dict[str, int] = defaultdict(int)

    for s in sample_results:
        for etype in s.get("entity_types", []):
            entity_total[etype] += 1
            if s["pred_label"] == "LEAKING":
                entity_detected[etype] += 1

    metrics: dict[str, dict] = {}
    for etype, total in sorted(entity_total.items()):
        detected = entity_detected.get(etype, 0)
        missed   = total - detected
        recall   = round(detected / total, 4) if total > 0 else 0.0
        metrics[etype] = {
            "total": total, "detected": detected,
            "missed": missed, "recall": recall,
        }

    print("\n  Per-Entity-Type Recall")
    print("  ─────────────────────────────────────────────────────")
    print(f"  {'Entity Type':<20} {'Total':>6} {'Detected':>9} {'Missed':>7} {'Recall':>8}")
    print("  ─────────────────────────────────────────────────────")
    for etype, m in metrics.items():
        print(f"  {etype:<20} {m['total']:>6} {m['detected']:>9} {m['missed']:>7} {m['recall']:>8.4f}")
    print("  ─────────────────────────────────────────────────────")
    return metrics


# ---------------------------------------------------------------------------
# Latency benchmarking
# ---------------------------------------------------------------------------

def latency_stats(latencies: list[float]) -> dict:
    if not latencies:
        return {}
    sorted_lat = sorted(latencies)
    n = len(sorted_lat)

    def percentile(p: float) -> float:
        idx = min(int(n * p / 100), n - 1)
        return round(sorted_lat[idx], 3)

    stats = {
        "n_samples": n,
        "mean_ms":   round(sum(latencies) / n, 3),
        "min_ms":    round(sorted_lat[0], 3),
        "max_ms":    round(sorted_lat[-1], 3),
        "p50_ms":    percentile(50),
        "p95_ms":    percentile(95),
    }

    print("\n  Latency Benchmarks")
    print("  ─────────────────────────────────")
    print(f"  Samples : {stats['n_samples']}")
    print(f"  Mean    : {stats['mean_ms']:.3f} ms")
    print(f"  Min     : {stats['min_ms']:.3f} ms")
    print(f"  Max     : {stats['max_ms']:.3f} ms")
    print(f"  p50     : {stats['p50_ms']:.3f} ms")
    print(f"  p95     : {stats['p95_ms']:.3f} ms")
    print("  ─────────────────────────────────")
    return stats


# ---------------------------------------------------------------------------
# Confusion matrix printer
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# JSON results output with summary
# ---------------------------------------------------------------------------

def build_results(
    dataset: list[dict],
    metrics: dict,
    sample_results: list[dict],
    entity_metrics: dict,
    lat_stats: dict,
    dataset_path: str,
) -> dict[str, Any]:
    """Build the full results dict saved to synthetic_eval.json."""
    false_negatives = [s for s in sample_results if
                       s["true_label"] == "LEAKING" and s["pred_label"] == "CLEAN"]
    false_positives = [s for s in sample_results if
                       s["true_label"] == "CLEAN" and s["pred_label"] == "LEAKING"]

    return {
        "meta": {
            "run_at": datetime.utcnow().isoformat() + "Z",
            "dataset": dataset_path,
            "total_samples": len(dataset),
            "detector": "PIIDetector (Presidio + custom recognisers)",
        },
        "summary": {
            "precision": metrics["precision"],
            "recall":    metrics["recall"],
            "f1":        metrics["f1"],
            "accuracy":  metrics["accuracy"],
            "p50_latency_ms": lat_stats.get("p50_ms"),
            "p95_latency_ms": lat_stats.get("p95_ms"),
        },
        "binary_classification": metrics,
        "per_entity_metrics": entity_metrics,
        "latency": lat_stats,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "samples": sample_results,
    }


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def evaluate(dataset: list[dict[str, Any]], dataset_path: str) -> dict[str, Any]:
    metrics, sample_results, latencies = binary_classification(dataset)
    print_confusion_matrix(metrics)
    entity_metrics = per_entity_metrics(sample_results)
    lat_stats      = latency_stats(latencies)
    results        = build_results(dataset, metrics, sample_results,
                                   entity_metrics, lat_stats, dataset_path)
    return results


def main():
    parser = argparse.ArgumentParser(description="Large-scale PII detector evaluation")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(ROOT, "experiments", "results"))
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    results = evaluate(dataset, args.dataset)

    os.makedirs(args.output_dir, exist_ok=True)

    # Full results (all samples)
    out_path = os.path.join(args.output_dir, "synthetic_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Summary only (lightweight)
    summary_path = os.path.join(args.output_dir, "synthetic_eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta":    results["meta"],
            "summary": results["summary"],
            "binary_classification": results["binary_classification"],
            "per_entity_metrics":    results["per_entity_metrics"],
            "latency":               results["latency"],
        }, f, indent=2)

    print(f"\nFull results saved to : {out_path}")
    print(f"Summary saved to      : {summary_path}")


if __name__ == "__main__":
    main()