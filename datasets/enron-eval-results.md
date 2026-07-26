# Enron Email Corpus Evaluation Results

**Evaluator:** `src/enron_eval.py`
**Detector:** Stage 1 — Presidio + custom recognisers (Week 06 build)

---

## Week 06 — Real Corpus Run _(current)_

**Mode:** Real corpus — `datasets/enron/emails.csv` (Kaggle CSV)
**Run date:** 2026-07-23
**Samples:** 500 (seed=42)

### Summary

| Metric | Value |
|---|---|
| Total emails processed | 500 |
| Flagged as leaking | 500 (100%) |
| Clean | 0 (0%) |
| Risk: HIGH | 447 (89.4%) |
| Risk: MEDIUM | 53 (10.6%) |
| Mean latency | 158.6 ms |
| p50 latency | 80.7 ms |
| p95 latency | 549.1 ms |

### Entity Type Distribution

| Entity Type | Count |
|---|---|
| EMAIL_ADDRESS | 4,238 |
| PERSON | 4,093 |
| LOCATION | 1,230 |
| PHONE_NUMBER | 274 |
| UK_NHS | 13 |
| US_SSN | 7 |
| IBAN_CODE | 5 |

### Notes

- **100% leaking ratio is expected and correct.** Every real Enron email
  contains at least one name or email address in headers or signatures.
  Real corporate email is genuinely PII-dense; this is not a detector error.
- No ground-truth per-email labels exist for the real corpus, so precision
  and recall cannot be computed. Entity distribution is reported instead.
- **Latency increase vs Week 05:** p50 jumped from ~20 ms to 80.7 ms and
  p95 reached 549 ms. Real emails are significantly longer than synthetic
  texts (headers + quoted reply chains + signatures), which scales Presidio's
  spaCy NLP pipeline linearly. Truncating input to the first 2,000 characters
  before analysis would reduce latency but risks missing PII in email bodies —
  trade-off to investigate in Week 07.
- `CREDIT_CARD`, `US_BANK_NUMBER`, `PK_CNIC`, and `IBAN_CODE` are rare or
  absent in real Enron email, consistent with the corpus being general
  corporate communication rather than financial transaction records.
- `UK_NHS` detections (13) are likely false positives — NHS-format number
  patterns coincidentally matching Enron reference codes. Flagged for
  investigation.

---

## Week 05 — Synthetic Fallback Run _(superseded)_

**Mode:** Synthetic email fallback (`--synthetic`, 500 samples, seed=42)
**Run date:** 2026-07-15
**Detector:** Stage 1 only — Presidio + custom recognisers (Week 05 build)

> ⚠️ The real Enron maildir corpus was not available locally during Week 05.
> The synthetic fallback was used as a placeholder. These results are superseded
> by the Week 06 real corpus run above.

### Summary

| Metric | Value |
|---|---|
| Total emails processed | 500 |
| Flagged as leaking | 275 (55%) |
| Clean | 225 (45%) |
| p50 latency | ~20 ms |
| p95 latency | ~28 ms |

### Entity Type Distribution

| Entity Type | Count |
|---|---|
| EMAIL_ADDRESS | most frequent |
| PERSON | high |
| CREDIT_CARD | moderate |
| US_SSN | moderate |
| PK_CNIC | moderate |
| IBAN_CODE | present |
| PHONE_NUMBER | present |
| LOCATION | present |

### Notes

- The synthetic fallback's 55% leaking split was an artifact of the generator's
  default `leaking_ratio=0.55`, not a property of real email.
- Entity counts were not recorded precisely for this run; qualitative labels
  only. See Week 06 run for exact figures.

---

## Week-on-Week Comparison

| Metric | Week 05 (synthetic) | Week 06 (real corpus) |
|---|---|---|
| Source | synthetic fallback | `datasets/enron/emails.csv` |
| Samples | 500 | 500 |
| Leaking ratio | 55% | 100% |
| p50 latency | ~20 ms | 80.7 ms |
| p95 latency | ~28 ms | 549.1 ms |
| Precision / Recall | N/A (no labels) | N/A (no labels) |

---

## How to Re-run

```bash
# Real corpus — Kaggle CSV (recommended)
python src/enron_eval.py --csv datasets/enron/emails.csv --samples 500

# Real corpus — maildir format
python src/enron_eval.py --maildir datasets/enron/maildir --samples 500

# Synthetic fallback (no corpus needed)
python src/enron_eval.py --synthetic --samples 500
```

---

## Output Files

| File | Description |
|---|---|
| `experiments/results/enron_eval.json` | Full per-email results (Week 06 real run) |
| `experiments/results/enron_eval_summary.json` | Aggregated metrics (Week 06 real run) |