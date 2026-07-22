# Datasets

This directory documents the datasets used in the LLM Data Leakage Prevention project.

---

## 1. Synthetic PII Dataset

**File:** `experiments/results/synthetic_dataset.json`
**Generator:** `src/synthetic_gen.py`
**Size:** 1,200 samples (660 LEAKING / 540 CLEAN)
**Seed:** 42

### Entity type distribution

| Entity Type | Samples |
|---|---|
| PERSON | ~170 |
| EMAIL_ADDRESS | ~122 |
| US_SSN | ~102 |
| PK_CNIC | ~95 |
| CREDIT_CARD | ~87 |
| PHONE_NUMBER | ~73 |
| LOCATION | ~69 |
| IBAN_CODE | ~57 |

### Week 05 changes

- IBAN pool expanded from 15 → 30 IBANs (all major SEPA/SWIFT country prefixes)
- 4 new IBAN-focused multi-PII templates added
- 5 new CLEAN templates added (GDPR, BGP, PCI-DSS, load balancing, differential privacy)
- IBAN recall improved from 0.825 → 1.000 on the synthetic benchmark

### Regenerate

```bash
python src/synthetic_gen.py --samples 1200 --seed 42
```

---

## 2. Enron Email Corpus

**Reference:** `datasets/enron-emails.md`
**Results:** `datasets/enron-eval-results.md`
**Evaluator:** `src/enron_eval.py`

The Enron email corpus contains ~500,000 emails from Enron employees, widely used
in NLP and privacy research (ProPILE paper, Lukas et al. 2023).

### Download options

**Option A — CMU maildir format (official)**
```
https://www.cs.cmu.edu/~./enron/enron_mail_20150507.tar.gz
```
Extract to `datasets/enron/maildir/` then run:
```bash
python src/enron_eval.py --maildir datasets/enron/maildir --samples 500
```

**Option B — Kaggle CSV**
```
https://www.kaggle.com/datasets/wcukierski/enron-email-dataset
```
Download `emails.csv` to `datasets/enron/` then run:
```bash
python src/enron_eval.py --csv datasets/enron/emails.csv --samples 500
```

**Option C — Synthetic fallback (no corpus needed)**
```bash
python src/enron_eval.py --synthetic --samples 500
```

### Week 05 status

Synthetic fallback was used in Week 05 (corpus download pending).
Real corpus evaluation is planned for Week 06.

---

## 3. Adversarial Test Cases

**Inline in:** `src/tests/adversarial_eval.py` and `src/stage_comparison.py`
**Size:** 25 cases (20 LEAKING / 5 CLEAN)

Covers: obfuscated emails, formatted credit cards, dot phones, UK international
phones, SSN/CNIC, IBANs embedded in prose, multi-PII, and inference-based leakage.

Run:
```bash
python src/tests/adversarial_eval.py
```