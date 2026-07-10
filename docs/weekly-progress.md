# Weekly Progress Log: LLM Data Leakage Prevention: Detection and Mitigation

**Student:** Muhammad Hashim Mughal
**GitHub username:** Hashim-Mughal

---

## How to Use This File

Add a new section every Friday before opening your weekly Pull Request.
Be honest — problems and blockers are normal and help your supervisor support you.

---

## Week 1

**Branch:** `hashim-mughal-week-01`
**PR link:** https://github.com/AI-Security-Internships-2026/06-llm-data-leakage-prevention/pull/1

### Completed this week
- [x] Read README and proposal
- [x] Set up local environment (Python venv, dependencies)
- [x] Ran `src/main.py` successfully
- [x] Wrote personal introduction (below)
- [x] Identified 5 related tools/datasets in `docs/literature-review.md`

### Personal Introduction
I'm Muhammad Hashim Mughal, a Software Engineering student at NUST SEECS (Batch 2024–2028). I have a full-stack background across Next.js, NestJS, React, MySQL, and MongoDB, and have built and deployed a production SaaS platform (ExamsCave) end-to-end. I also have applied ML experience from a Stanford/DeepLearning.AI Machine Learning Specialization, a YOLOv8 license plate detection project, and a student score prediction model. I'm looking forward to going deeper into the security side of LLM systems, especially how rule-based detection (Presidio/spaCy) compares to LLM-as-judge approaches for catching leakage.

### Problems / Blockers
Minor: `source .venv/bin/activate` doesn't work on Windows PowerShell — had to use `.venv\Scripts\Activate.ps1` instead. No other issues; clean install of all dependencies, `src/main.py` ran without errors on first try.

### Next week plan
- Read the 5 tools identified this week in depth
- Implement first leakage detection probe using Presidio
- Add 10 academic paper summaries to `docs/literature-review.md`

---

## Week 2

**Branch:** `hashim-week-02`
**PR link:** https://github.com/AI-Security-Internships-2026/06-llm-data-leakage-prevention/pull/2

### Completed this week
- [x] Created branch `hashim-week-02` from `dev`
- [x] Rebased onto latest `dev` after Week 1 merge
- [x] Implemented PII leakage detection probe using Microsoft Presidio (`src/detector.py`)
- [x] Built FastAPI endpoint exposing the detector (`src/main.py`)
- [x] Verified detection works on sample texts — detected EMAIL_ADDRESS, PERSON, CREDIT_CARD, US_BANK_NUMBER entities with correct confidence scores and sanitized output
- [x] Added 10 academic paper summaries to `docs/literature-review.md` (target met)
- [x] Updated `docs/literature-review.md` reference table — replaced overlapping Tool 5 with NVIDIA NeMo Guardrails for better attack/defense coverage

### Implementation Notes
Built a two-file prototype:
- `src/detector.py` — core detection logic using Presidio Analyzer + Anonymizer. Detects PII entity types (PERSON, EMAIL_ADDRESS, CREDIT_CARD, PHONE_NUMBER, etc.), computes a risk level (HIGH/MEDIUM/LOW/CLEAN), and returns a sanitized version with all PII replaced by `<REDACTED>`.
- `src/main.py` — FastAPI app with three endpoints: `GET /` health check, `POST /detect` for single text, `POST /detect/batch` for up to 50 texts.

Tested manually via Swagger UI at `http://127.0.0.1:8000/docs`. Sample input containing name, email, and credit card number correctly returned 4 entities with `risk_level: HIGH` and fully sanitized output.

### Problems / Blockers
- `en_core_web_lg` (400MB spaCy model) required multiple download attempts due to connection drops — eventually succeeded on retry.
- PyTorch not installed — harmless warning on startup, Presidio does not require it.

### Next week plan
- Add custom PII recognizers (e.g. Pakistani CNIC format)
- Write unit tests for `detector.py`
- Begin architecture proposal in `docs/proposal.md`


## Week 3

**Branch:** `hashim-week-03`
**PR link:** _[Add link after opening PR]_

### Completed this week

- [x] Fixed `text_slice` bug in `src/detector.py` — entity output was
      echoing raw PII back to the caller; replaced with `length` field only
- [x] Fixed risk logic — 3 low-confidence entities no longer blindly
      escalate to HIGH without a score threshold
- [x] Added custom `PatternRecognizer` for Pakistani CNIC (`PK_CNIC`,
      format `XXXXX-XXXXXXX-X`) with context keywords
- [x] Added `PK_CNIC` to `_HIGH_RISK_TYPES`
- [x] Added language validation — unsupported language now raises `ValueError`
      instead of silently passing to Presidio
- [x] Created `src/tests/__init__.py`
- [x] Created `src/tests/test_detector.py` — 27 pytest unit tests across
      5 categories: edge cases, leaking inputs, non-leaking inputs,
      risk levels, sanitisation
- [x] Created `src/tests/eval_suite.py` — 18-case labelled evaluation
      suite (10 leaking, 8 clean) measuring text-level precision, recall,
      F1, and accuracy; exits non-zero if recall < 0.80
- [x] Pinned all versions in `requirements.txt`; added `pytest` and `httpx`
- [x] Completed `docs/proposal.md` — architecture design document

### Implementation Notes

The eval suite is the key new piece this week (supervisor's PR feedback).
It runs the detector against 18 ground-truth labelled inputs and reports
a confusion matrix plus precision/recall/F1. The 0.80 recall floor in
the exit code means CI can gate on this automatically once a pipeline
is set up.

The `text_slice` removal was a correctness fix — a leakage-prevention
tool that echoes the raw PII value in its own response payload defeats
its own purpose.

### Problems / Blockers

None this week. All tests pass locally.

### Next week plan

- Generate synthetic evaluation dataset using Faker (1000+ samples)
  to get statistically meaningful precision/recall numbers
- Document Enron corpus setup in `datasets/enron-emails.md`
- Begin investigating false negatives in the current eval suite —
  especially inference-based and association-based leakage
  (Paper 5 and Paper 8 from literature review)

---

## Week 4

**Branch:** `hashim-week-04`
**PR link:** _[Add link after opening PR]_

### Completed this week

- [x] Implemented `src/synthetic_gen.py` — Faker-based synthetic PII dataset
      generator producing 1,200 labelled samples across 8 entity types
      (EMAIL_ADDRESS, CREDIT_CARD, PHONE_NUMBER, PERSON, US_SSN, IBAN_CODE,
      PK_CNIC, LOCATION) with CLI arguments `--samples`, `--seed`,
      `--leaking-ratio`, `--output-dir`
- [x] Generated `experiments/results/synthetic_dataset.json` (1,200 samples,
      660 leaking / 540 clean) and `synthetic_dataset_stats.json`
- [x] Documented synthetic dataset in `datasets/synthetic-pii.md` — schema,
      generation parameters, entity distribution, limitations
- [x] Documented Enron email corpus in `datasets/enron-emails.md` — download
      instructions, preprocessing steps, privacy notes, citation
- [x] Pinned `faker==40.28.1` in `requirements.txt`
- [x] Implemented `src/tests/eval_large.py` — large-scale evaluation suite
      with text-level binary classification, per-entity-type recall breakdown,
      p50/p95 latency benchmarking, confusion matrix printer, and JSON output
- [x] Added adversarial test classes to `src/tests/test_detector.py`:
      spaced/hyphen credit cards, obfuscated emails, PII in JSON/SQL/code,
      phone number variants, PII in markdown tables (48 tests: 44 passed,
      4 xfailed)
- [x] Patched `src/detector.py` with pre-processing `normalize_text()`
      pipeline: strips spaces and hyphens from 16-digit card numbers,
      converts dot-formatted phones to hyphen format before Presidio analysis
- [x] Implemented `src/tests/adversarial_eval.py` — 21-case adversarial
      evaluation script covering credit card formats, email obfuscation,
      phone variants, and PII embedded in structured text

### Evaluation Results (1,200-sample synthetic dataset)

| Metric | Value |
|---|---|
| Precision | 0.9341 |
| Recall | 0.9667 |
| F1 | 0.9501 |
| Accuracy | 0.9442 |
| p50 latency | 14 ms |
| p95 latency | 19 ms |

### Per-Entity Recall

| Entity Type | Recall |
|---|---|
| EMAIL_ADDRESS | 1.0000 |
| PERSON | 1.0000 |
| US_SSN | 1.0000 |
| PK_CNIC | 1.0000 |
| CREDIT_CARD | 0.9655 |
| PHONE_NUMBER | 0.9589 |
| LOCATION | 0.9130 |
| IBAN_CODE | 0.8246 |

### Implementation Notes

The biggest deliverable this week was scaling the evaluation from 18 hand-labelled
cases (Week 3) to 1,200 synthetic samples, giving statistically meaningful P/R/F1
numbers. The per-entity breakdown revealed IBAN_CODE as the weakest entity at 0.82
recall — caused by a limited pool of 5 IBAN patterns in the generator rather than
a detector gap. This is documented as a known limitation in `datasets/synthetic-pii.md`.

The `normalize_text()` pre-processing step in `detector.py` directly addresses
adversarial false negatives found during testing: spaced card numbers
(`4111 1111 1111 1111`) and hyphen-delimited cards (`4111-1111-1111-1111`) are
now stripped to raw digits before Presidio analysis. Dot-formatted phones
(`800.555.0199`) are normalized to `800-555-0199`.

Known remaining false negatives (marked `xfail` in the test suite):
- Obfuscated emails: `[at]`, `(at)`, `AT` variants
- Dot-delimited credit cards: `4111.1111.1111.1111`
- UK international phone format

These require semantic understanding beyond regex normalization and are planned
for Stage 2 (LLM-as-judge) in Week 5–6.

### Problems / Blockers

- `datetime.utcnow()` deprecation warning in Python 3.13 — fixed by switching
  to `datetime.now(timezone.utc)` in `adversarial_eval.py`

### Next week plan

- Implement Stage 2: LLM-as-judge using a HuggingFace model to catch
  inference-based and obfuscated PII that Presidio misses
- Run evaluation comparing Stage 1 alone vs Stage 1 + Stage 2 on recall
- Begin Enron corpus evaluation on a 500-email sample