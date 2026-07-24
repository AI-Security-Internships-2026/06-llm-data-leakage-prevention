"""
hf_pii_eval.py — HuggingFace english_pii_43k Evaluation
CNIT/PNTLab Pisa — AI Security Internship 2026
Student: Muhammad Hashim Mughal | Week: 06

"""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from detector import detect_pii  # noqa: E402

# ── Label mapping: HF dataset label → Presidio entity type ───────────────────
# Labels NOT in this map are "UNSUPPORTED" by the current detector.

LABEL_MAP: dict[str, str] = {
    # Identity
    "FIRSTNAME":          "PERSON",
    "LASTNAME":           "PERSON",
    "MIDDLENAME":         "PERSON",
    "PREFIX":             "PERSON",
    "USERNAME":           "PERSON",

    # Contact
    "EMAIL":              "EMAIL_ADDRESS",
    "PHONENUMBER":        "PHONE_NUMBER",

    # Financial
    "CREDITCARDNUMBER":   "CREDIT_CARD",
    "MASKEDNUMBER":       "CREDIT_CARD",
    "IBAN":               "IBAN_CODE",

    # Location
    "STREET":             "LOCATION",
    "CITY":               "LOCATION",
    "STATE":              "LOCATION",
    "COUNTY":             "LOCATION",
    "ZIPCODE":            "LOCATION",
    "BUILDINGNUMBER":     "LOCATION",

    # Identity documents
    "SSN":                "US_SSN",

    # Not mapped (detector has no recogniser for these):
    # AGE, DOB, DATE, TIME, GENDER, HEIGHT, EYECOLOR, SEX,
    # PASSWORD, PIN, CREDITCARDCVV, CREDITCARDISSUER,
    # IPV4, IPV6, IP, MAC, URL, USERAGENT,
    # PHONEIMEI, VEHICLEVIN, VEHICLEVRM,
    # BITCOINADDRESS, LITECOINADDRESS, ETHEREUMADDRESS,
    # ACCOUNTNUMBER, ACCOUNTNAME,
    # CURRENCYCODE, CURRENCYNAME, CURRENCYSYMBOL, CURRENCY, AMOUNT,
    # JOBTITLE, JOBTYPE, JOBAREA, ORDINALDIRECTION,
    # NEARBYGPSCOORDINATE, SECONDARYADDRESS, COMPANYNAME,
}

# Entity types the detector supports (used for coverage reporting)
DETECTOR_ENTITIES = {
    "EMAIL_ADDRESS", "CREDIT_CARD", "PHONE_NUMBER", "PERSON",
    "US_SSN", "IBAN_CODE", "PK_CNIC", "LOCATION",
    "MEDICAL_LICENSE", "UK_NHS", "US_BANK_NUMBER",
}


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_jsonl(path: str, n: int, seed: int = 42) -> list[dict]:
    """Load up to *n* English records from the JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("language", "en") != "en":
                continue
            records.append(obj)

    rng = random.Random(seed)
    if len(records) > n:
        records = rng.sample(records, n)
    return records


# ── Span overlap matching ─────────────────────────────────────────────────────

def _iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    """Character-level Intersection-over-Union of two spans."""
    inter_start = max(a_start, b_start)
    inter_end   = min(a_end,   b_end)
    inter = max(0, inter_end - inter_start)
    if inter == 0:
        return 0.0
    union = (a_end - a_start) + (b_end - b_start) - inter
    return inter / union if union > 0 else 0.0


def match_spans(
    pred_spans: list[dict],
    gold_spans: list[dict],
    iou_threshold: float = 0.5,
) -> tuple[int, int, int]:
    """
    Greedy span matching.

    pred_spans : [{"start": int, "end": int, "presidio_type": str}, …]
    gold_spans : [{"start": int, "end": int, "label": str, "presidio_type": str}, …]

    Returns (TP, FP, FN) counts across all entity types combined.
    """
    matched_gold = set()
    matched_pred = set()

    for pi, p in enumerate(pred_spans):
        for gi, g in enumerate(gold_spans):
            if gi in matched_gold:
                continue
            # Only count as TP if entity types align OR gold label is unsupported
            if p["presidio_type"] != g["presidio_type"] and g["presidio_type"] != "UNSUPPORTED":
                continue
            if _iou(p["start"], p["end"], g["start"], g["end"]) >= iou_threshold:
                matched_gold.add(gi)
                matched_pred.add(pi)
                break

    tp = len(matched_gold)
    fp = len(pred_spans) - len(matched_pred)
    fn = len(gold_spans) - len(matched_gold)
    return tp, fp, fn


# ── Per-entity-type tracking ──────────────────────────────────────────────────

def _entity_match(
    pred_spans: list[dict],
    gold_spans: list[dict],
    iou_threshold: float = 0.5,
) -> dict[str, dict[str, int]]:
    """
    Returns per-entity-type {TP, FP, FN} dict.
    Unsupported gold labels are grouped under "UNSUPPORTED".
    """
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})

    matched_gold = set()
    matched_pred = set()

    for pi, p in enumerate(pred_spans):
        best_gi = None
        best_iou = 0.0
        for gi, g in enumerate(gold_spans):
            if gi in matched_gold:
                continue
            if p["presidio_type"] != g["presidio_type"] and g["presidio_type"] != "UNSUPPORTED":
                continue
            iou = _iou(p["start"], p["end"], g["start"], g["end"])
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_gi = gi

        if best_gi is not None:
            matched_gold.add(best_gi)
            matched_pred.add(pi)
            etype = gold_spans[best_gi]["presidio_type"]
            counts[etype]["TP"] += 1
        else:
            counts[p["presidio_type"]]["FP"] += 1

    for gi, g in enumerate(gold_spans):
        if gi not in matched_gold:
            counts[g["presidio_type"]]["FN"] += 1

    return counts


# ── Main evaluation loop ──────────────────────────────────────────────────────

def evaluate(records: list[dict]) -> tuple[list[dict], list[float]]:
    results = []
    latencies = []
    per_entity: dict[str, dict[str, int]] = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})
    unsupported_labels: dict[str, int] = defaultdict(int)

    for i, rec in enumerate(records):
        text = rec.get("source_text", "")
        if not text:
            continue

        # Build gold spans
        gold_spans = []
        for mask in rec.get("privacy_mask", []):
            raw_label = mask.get("label", "")
            presidio_type = LABEL_MAP.get(raw_label, "UNSUPPORTED")
            if presidio_type == "UNSUPPORTED":
                unsupported_labels[raw_label] += 1
            gold_spans.append({
                "start":         mask["start"],
                "end":           mask["end"],
                "value":         mask.get("value", ""),
                "raw_label":     raw_label,
                "presidio_type": presidio_type,
            })

        # Run detector
        t0 = time.perf_counter()
        detection = detect_pii(text)
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)

        # Build predicted spans
        pred_spans = [
            {
                "start":         e["start"],
                "end":           e["end"],
                "presidio_type": e["type"],
                "score":         e.get("score", 0),
            }
            for e in detection.get("entities", [])
        ]

        # Per-sample matching
        tp, fp, fn = match_spans(pred_spans, gold_spans)
        entity_counts = _entity_match(pred_spans, gold_spans)

        for etype, c in entity_counts.items():
            per_entity[etype]["TP"] += c["TP"]
            per_entity[etype]["FP"] += c["FP"]
            per_entity[etype]["FN"] += c["FN"]

        results.append({
            "record_id":       rec.get("id", i),
            "text_snippet":    text[:80] + ("…" if len(text) > 80 else ""),
            "gold_labels":     [g["raw_label"] for g in gold_spans],
            "pred_types":      [p["presidio_type"] for p in pred_spans],
            "risk_level":      detection["risk_level"],
            "tp": tp, "fp": fp, "fn": fn,
            "latency_ms":      round(latency_ms, 3),
        })

    return results, latencies, dict(per_entity), dict(unsupported_labels)


# ── Aggregation & metrics ─────────────────────────────────────────────────────

def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {
        "TP": tp, "FP": fp, "FN": fn,
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
    }


def aggregate(
    results: list[dict],
    latencies: list[float],
    per_entity: dict,
    unsupported_labels: dict,
) -> dict:
    total_tp = sum(r["tp"] for r in results)
    total_fp = sum(r["fp"] for r in results)
    total_fn = sum(r["fn"] for r in results)

    n = len(latencies)
    sorted_lat = sorted(latencies)
    p50 = sorted_lat[int(n * 0.50)] if n else 0
    p95 = sorted_lat[int(n * 0.95)] if n else 0

    entity_metrics = {
        etype: _prf(c["TP"], c["FP"], c["FN"])
        for etype, c in sorted(per_entity.items())
    }

    # Separate supported vs unsupported for clarity
    supported_metrics = {k: v for k, v in entity_metrics.items()
                         if k in DETECTOR_ENTITIES}
    unsupported_metrics = {k: v for k, v in entity_metrics.items()
                           if k not in DETECTOR_ENTITIES}

    return {
        "total_samples": len(results),
        "overall": _prf(total_tp, total_fp, total_fn),
        "latency": {
            "mean_ms": round(sum(latencies) / n, 3) if n else 0,
            "p50_ms":  round(p50, 3),
            "p95_ms":  round(p95, 3),
        },
        "supported_entity_metrics":   supported_metrics,
        "unsupported_entity_metrics": unsupported_metrics,
        "unsupported_label_counts": dict(
            sorted(unsupported_labels.items(), key=lambda x: -x[1])
        ),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Presidio detector on english_pii_43k JSONL dataset"
    )
    parser.add_argument(
        "--jsonl", required=True,
        help="Path to english_pii_43k.jsonl (or any subset JSONL)"
    )
    parser.add_argument("--samples",    type=int, default=1000,
                        help="Number of records to evaluate (default: 1000)")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(ROOT, "experiments", "results"))
    args = parser.parse_args()

    if not os.path.isfile(args.jsonl):
        print(f"[ERROR] JSONL file not found: {args.jsonl}")
        sys.exit(1)

    print(f"Loading up to {args.samples} English records from: {args.jsonl}")
    records = load_jsonl(args.jsonl, args.samples, args.seed)
    print(f"Loaded {len(records)} records.")
    print("Running detection…\n")

    results, latencies, per_entity, unsupported_labels = evaluate(records)
    summary = aggregate(results, latencies, per_entity, unsupported_labels)

    # ── Console output ────────────────────────────────────────────────────────
    ov = summary["overall"]
    print(f"  Samples evaluated : {summary['total_samples']}")
    print(f"  Overall TP/FP/FN  : {ov['TP']} / {ov['FP']} / {ov['FN']}")
    print(f"  Overall precision : {ov['precision']:.3f}")
    print(f"  Overall recall    : {ov['recall']:.3f}")
    print(f"  Overall F1        : {ov['f1']:.3f}")
    print(f"  p50 latency       : {summary['latency']['p50_ms']} ms")
    print(f"  p95 latency       : {summary['latency']['p95_ms']} ms")

    print("\n  Per-entity metrics (SUPPORTED by detector):")
    print(f"  {'Entity type':<22} {'P':>6} {'R':>6} {'F1':>6}  TP / FP / FN")
    print("  " + "-" * 62)
    for etype, m in summary["supported_entity_metrics"].items():
        if m["TP"] + m["FP"] + m["FN"] == 0:
            continue
        print(f"  {etype:<22} {m['precision']:>6.3f} {m['recall']:>6.3f} "
              f"{m['f1']:>6.3f}  {m['TP']} / {m['FP']} / {m['FN']}")

    print("\n  UNSUPPORTED labels (no Presidio recogniser — all become FN):")
    for label, count in list(summary["unsupported_label_counts"].items())[:15]:
        print(f"    {count:>4}  {label}")
    if len(summary["unsupported_label_counts"]) > 15:
        remaining = len(summary["unsupported_label_counts"]) - 15
        print(f"    … and {remaining} more (see full results JSON)")

    # ── Save outputs ──────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    full_output = {
        "meta": {
            "run_at":            datetime.now(timezone.utc).isoformat(),
            "dataset":           args.jsonl,
            "samples_requested": args.samples,
            "samples_processed": len(results),
            "seed":              args.seed,
            "detector":          "PIIDetector Stage 1 (Presidio + custom recognisers)",
            "iou_threshold":     0.5,
        },
        "summary": summary,
        "results": results,
    }

    full_path    = os.path.join(args.output_dir, "hf_pii_eval.json")
    summary_path = os.path.join(args.output_dir, "hf_pii_eval_summary.json")

    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"meta": full_output["meta"], "summary": summary},
                  f, indent=2, ensure_ascii=False)

    print(f"\nFull results  → {full_path}")
    print(f"Summary       → {summary_path}")


if __name__ == "__main__":
    main()