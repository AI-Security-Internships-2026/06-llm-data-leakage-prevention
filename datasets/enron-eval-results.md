# Enron Email Corpus Evaluation Results — Week 05

**Evaluator:** `src/enron_eval.py`
**Mode:** Synthetic email fallback (500 samples, seed=42)
**Detector:** Stage 1 only — Presidio + custom recognisers (Week 05 build)
**Run date:** 2026-07-15

---

## Summary

| Metric | Value |
|---|---|
| Total emails processed | 500 |
| Flagged as leaking | 275 (55%) |
| Clean | 225 (45%) |
| p50 latency | ~20 ms |
| p95 latency | ~28 ms |

---

## Entity Type Distribution

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

---

## Notes

- Enron maildir corpus was not available locally during Week 05 evaluation.
  The synthetic fallback mode (`--synthetic`) was used instead, which generates
  email-style texts using `synthetic_gen.py` templates.
- The synthetic fallback produces a balanced leaking/clean split (55%/45%)
  matching the dataset generator's default `leaking_ratio=0.55`.
- Full results are saved to `experiments/results/enron_eval.json`.
- Real Enron corpus evaluation (500 emails from the maildir) is planned
  for Week 06 once the corpus is downloaded.

---

## How to Re-run

```bash
# Synthetic fallback (no corpus needed)
python src/enron_eval.py --synthetic --samples 500

# Real Enron maildir (once downloaded)
python src/enron_eval.py --maildir datasets/enron/maildir --samples 500

# Kaggle CSV format
python src/enron_eval.py --csv datasets/enron/emails.csv --samples 500
```

---

## Output Files

| File | Description |
|---|---|
| `experiments/results/enron_eval.json` | Full per-email results |
| `experiments/results/enron_eval_summary.json` | Aggregated metrics |