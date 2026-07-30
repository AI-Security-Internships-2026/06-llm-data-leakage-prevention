"""
enron_eval.py — Enron Email Corpus Evaluation
CNIT/PNTLab Pisa — AI Security Internship 2026
Student: Muhammad Hashim Mughal | Week: 05

"""

import argparse
import email
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

# ── Email parsing ─────────────────────────────────────────────────────────────

def _strip_quoted_lines(body: str) -> str:
    """Remove quoted reply lines (starting with '>') to avoid double-counting."""
    lines = [ln for ln in body.splitlines() if not ln.strip().startswith(">")]
    return "\n".join(lines)


def parse_eml_file(filepath: str) -> dict:
    """Parse a single raw RFC-2822 email file."""
    with open(filepath, "r", errors="replace") as f:
        msg = email.message_from_file(f)

    body_parts = []
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    body_parts.append(payload.decode("utf-8", errors="replace"))
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            body_parts.append(payload.decode("utf-8", errors="replace"))

    body = _strip_quoted_lines("\n".join(body_parts))

    return {
        "from":    msg.get("From", ""),
        "to":      msg.get("To", ""),
        "subject": msg.get("Subject", ""),
        "body":    body,
    }


def load_from_maildir(maildir: str, n: int, seed: int = 42) -> list[dict]:
    """Collect up to *n* email files from a maildir directory tree."""
    all_files = []
    for root, dirs, files in os.walk(maildir):
        # Skip hidden directories (e.g. .gitkeep parent)
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fname in files:
            if not fname.startswith("."):
                all_files.append(os.path.join(root, fname))

    rng = random.Random(seed)
    selected = rng.sample(all_files, min(n, len(all_files)))
    emails = []
    for path in selected:
        try:
            emails.append(parse_eml_file(path))
        except Exception:
            pass
    return emails


def load_from_csv(csv_path: str, n: int, seed: int = 42) -> list[dict]:
    """Load emails from a Kaggle-style CSV (columns: file, message)."""
    import pandas as pd
    df = pd.read_csv(csv_path, nrows=None)
    df = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)

    emails = []
    for _, row in df.iterrows():
        raw = row.get("message", "")
        try:
            msg = email.message_from_string(str(raw))
            body_parts = []
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        payload = part.get_payload(decode=False)
                        if payload:
                            body_parts.append(str(payload))
            else:
                body_parts.append(str(msg.get_payload(decode=False) or ""))
            body = _strip_quoted_lines("\n".join(body_parts))
            emails.append({
                "from":    msg.get("From", ""),
                "to":      msg.get("To", ""),
                "subject": msg.get("Subject", ""),
                "body":    body,
            })
        except Exception:
            pass
    return emails


def load_synthetic_emails(n: int, seed: int = 42) -> list[dict]:
    """
    Generate synthetic email-style texts using the existing synthetic_gen module.
    Used when the Enron corpus is not available locally.
    """
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from synthetic_gen import generate_dataset

    samples = generate_dataset(n_samples=n, seed=seed)
    emails = []
    for s in samples:
        emails.append({
            "from":    "sender@example.com",
            "to":      "recipient@example.com",
            "subject": "Automated test",
            "body":    s.text,
            "_label":  s.label,
            "_entity_types": s.entity_types,
        })
    return emails


# ── Detection runner ──────────────────────────────────────────────────────────

def build_email_text(em: dict, max_body_chars: int | None = None) -> str:
    """
    Assemble the text that will be passed to detect_pii.

    Zone-aware truncation strategy (Week 07 latency investigation):
    - Headers (From / To / Subject) are always included in full — they are
      guaranteed PII locations (email addresses, names) and are short (<200 chars).
    - The body is truncated to *max_body_chars* if that parameter is set.
      None means no truncation (original behaviour).

    This prevents the p95 latency spike caused by newsletter/digest emails
    (e.g. "Enron Mentions", 6 056 ms) while preserving recall on header PII
    that never appears past a 2 000-character cut-off anyway.
    """
    header = (
        f"From: {em.get('from', '')}\n"
        f"To: {em.get('to', '')}\n"
        f"Subject: {em.get('subject', '')}\n\n"
    )
    body = em.get("body", "")
    if max_body_chars is not None:
        body = body[:max_body_chars]
    return (header + body).strip()


def run_detection_on_emails(
    emails: list[dict],
    max_body_chars: int | None = None,
) -> tuple[list[dict], list[float]]:
    """Run detect_pii on each email and collect results + latencies.

    Parameters
    ----------
    emails        : list of parsed email dicts (from/to/subject/body)
    max_body_chars: if set, body text is truncated to this many characters
                   before analysis. Headers are always included in full.
                   None = no truncation (original behaviour).
    """
    results = []
    latencies = []

    for i, em in enumerate(emails):
        full_text = build_email_text(em, max_body_chars)

        if not full_text:
            continue

        t0 = time.perf_counter()
        detection = detect_pii(full_text)
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)

        results.append({
            "email_index": i,
            "from": em.get("from", ""),
            "subject": em.get("subject", ""),
            "risk_level": detection["risk_level"],
            "entity_types": [e["type"] for e in detection["entities"]],
            "entity_count": len(detection["entities"]),
            "latency_ms": round(latency_ms, 3),
            "text_chars": len(full_text),          # Week 07: track input length
            # Ground truth (only available in synthetic mode)
            "true_label": em.get("_label"),
            "true_entity_types": em.get("_entity_types"),
        })

    return results, latencies


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate(results: list[dict], latencies: list[float]) -> dict:
    risk_dist: dict[str, int] = defaultdict(int)
    entity_dist: dict[str, int] = defaultdict(int)

    for r in results:
        risk_dist[r["risk_level"]] += 1
        for etype in r["entity_types"]:
            entity_dist[etype] += 1

    total = len(results)
    n_leaking = total - risk_dist.get("CLEAN", 0)

    sorted_lat = sorted(latencies)
    n = len(sorted_lat)
    p50 = sorted_lat[int(n * 0.50)] if n else 0
    p95 = sorted_lat[int(n * 0.95)] if n else 0

    # Synthetic-mode metrics (when ground truth is available)
    synthetic_metrics = None
    if results and results[0].get("true_label") is not None:
        tp = fn = tn = fp = 0
        for r in results:
            is_leaking_true = r["true_label"] == "LEAKING"
            is_leaking_pred = r["risk_level"] != "CLEAN"
            if is_leaking_true and is_leaking_pred:
                tp += 1
            elif is_leaking_true and not is_leaking_pred:
                fn += 1
            elif not is_leaking_true and is_leaking_pred:
                fp += 1
            else:
                tn += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        accuracy  = (tp + tn) / total if total > 0 else 0.0
        synthetic_metrics = {
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "accuracy":  round(accuracy, 4),
        }

    return {
        "total_emails": total,
        "leaking": n_leaking,
        "clean": risk_dist.get("CLEAN", 0),
        "leaking_ratio": round(n_leaking / total, 3) if total else 0,
        "risk_distribution": dict(risk_dist),
        "entity_type_distribution": dict(
            sorted(entity_dist.items(), key=lambda x: -x[1])
        ),
        "latency": {
            "mean_ms": round(sum(latencies) / n, 3) if n else 0,
            "p50_ms":  round(p50, 3),
            "p95_ms":  round(p95, 3),
        },
        "synthetic_metrics": synthetic_metrics,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Enron email corpus evaluation")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--maildir",   type=str, help="Path to Enron maildir root")
    source.add_argument("--csv",       type=str, help="Path to Kaggle emails.csv")
    source.add_argument("--synthetic", action="store_true",
                        help="Use synthetic emails (no Enron corpus needed)")
    parser.add_argument("--samples",    type=int, default=500)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(ROOT, "experiments", "results"))
    # Week 07: latency investigation options
    parser.add_argument(
        "--max-body-chars", type=int, default=None,
        metavar="N",
        help=(
            "Truncate email body to at most N characters before detection. "
            "Headers (From/To/Subject) are always scanned in full. "
            "Default: no truncation."
        ),
    )
    parser.add_argument(
        "--benchmark-truncation", action="store_true",
        help=(
            "Run the same email set under five truncation strategies "
            "(full, 500, 1000, 2000, 4000 body chars) and print a "
            "side-by-side latency + entity-count comparison table. "
            "No JSON output is written in this mode."
        ),
    )
    args = parser.parse_args()

    # Load emails
    if args.maildir:
        if not os.path.isdir(args.maildir):
            print(f"[ERROR] maildir not found: {args.maildir}")
            sys.exit(1)
        print(f"Loading up to {args.samples} emails from maildir: {args.maildir}")
        emails = load_from_maildir(args.maildir, args.samples, args.seed)
    elif args.csv:
        if not os.path.isfile(args.csv):
            print(f"[ERROR] CSV not found: {args.csv}")
            sys.exit(1)
        print(f"Loading up to {args.samples} emails from CSV: {args.csv}")
        emails = load_from_csv(args.csv, args.samples, args.seed)
    else:
        print(f"No corpus specified — running on {args.samples} synthetic emails.")
        emails = load_synthetic_emails(args.samples, args.seed)

    print(f"Loaded {len(emails)} emails.")

    # ── Benchmark mode ────────────────────────────────────────────────────────
    if args.benchmark_truncation:
        strategies = [
            ("Full (no limit)", None),
            ("4 000 chars",     4000),
            ("2 000 chars",     2000),
            ("1 000 chars",     1000),
            ("500 chars",        500),
        ]
        print("\nRunning truncation benchmark — each strategy on the same email set...\n")
        print(f"{'Strategy':<20} {'p50 ms':>8} {'p95 ms':>8} {'p99 ms':>8} "
              f"{'max ms':>8} {'mean ms':>8} {'avg entities':>13}")
        print("-" * 80)
        for label, max_chars in strategies:
            run_results, run_latencies = run_detection_on_emails(emails, max_chars)
            sl = sorted(run_latencies)
            n  = len(sl)
            p50 = sl[int(n * 0.50)]
            p95 = sl[int(n * 0.95)]
            p99 = sl[int(n * 0.99)]
            mx  = sl[-1]
            mean = sum(run_latencies) / n
            avg_ent = sum(r["entity_count"] for r in run_results) / len(run_results)
            print(f"{label:<20} {p50:>8.1f} {p95:>8.1f} {p99:>8.1f} "
                  f"{mx:>8.1f} {mean:>8.1f} {avg_ent:>13.1f}")
        print()
        return

    # ── Normal run ────────────────────────────────────────────────────────────
    print("Running detection...")

    results, latencies = run_detection_on_emails(emails, args.max_body_chars)
    summary = aggregate(results, latencies)

    print(f"\n  Total emails  : {summary['total_emails']}")
    print(f"  Leaking       : {summary['leaking']} ({summary['leaking_ratio']:.1%})")
    print(f"  Clean         : {summary['clean']}")
    print(f"  p50 latency   : {summary['latency']['p50_ms']} ms")
    print(f"  p95 latency   : {summary['latency']['p95_ms']} ms")
    print("\n  Entity type distribution:")
    for etype, count in summary["entity_type_distribution"].items():
        print(f"    {count:>4}  {etype}")
    if summary["synthetic_metrics"]:
        m = summary["synthetic_metrics"]
        print(f"\n  Synthetic P/R/F1: {m['precision']:.3f} / {m['recall']:.3f} / {m['f1']:.3f}")

    os.makedirs(args.output_dir, exist_ok=True)

    full_output = {
        "meta": {
            "run_at": datetime.now(timezone.utc).isoformat(),
            "source": args.maildir or args.csv or "synthetic",
            "samples_requested": args.samples,
            "samples_processed": len(results),
            "seed": args.seed,
            "detector": "PIIDetector Stage 1 (Presidio + custom recognisers)",
            "max_body_chars": args.max_body_chars,   # Week 07: record truncation setting
        },
        "summary": summary,
        "results": results,
    }

    full_path = os.path.join(args.output_dir, "enron_eval.json")
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False)

    summary_path = os.path.join(args.output_dir, "enron_eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"meta": full_output["meta"], "summary": summary}, f, indent=2)

    print(f"\nFull results  → {full_path}")
    print(f"Summary       → {summary_path}")


if __name__ == "__main__":
    main()