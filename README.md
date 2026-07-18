# LLM Data Leakage Prevention: Detection and Mitigation

> **CNIT/PNTLab Pisa · TECIP · Scuola Superiore Sant'Anna — AI Security Internship 2026**
> **Student:** Muhammad Hashim Mughal | NUST SEECS, Batch 2024–2028

---

## Research Problem

Research and implement mechanisms that detect when an LLM is about to leak
sensitive data (PII, secrets, internal documents) and automatically sanitise
or refuse the response.

---

## Current Status — Week 5

| Component | Status |
|---|---|
| Stage 1: Presidio + custom recognisers | ✅ Complete |
| Stage 2: LLM-as-judge (bart-large-mnli) | ✅ Complete |
| FastAPI `/detect` and `/detect/v2` | ✅ Complete |
| Synthetic dataset (1,200 samples) | ✅ Complete |
| Enron corpus evaluation | 🔄 Synthetic fallback done; real corpus Week 6 |
| Final report | 🔄 Week 8 |

### Week 5 Key Results

| Eval | Stage 1 | Stage 1 + Stage 2 |
|---|---|---|
| Synthetic F1 (1,200 samples) | 0.9512 | — |
| IBAN recall | **1.000** (was 0.825) | — |
| Adversarial recall (25 cases) | 0.850 | **1.000** |
| Hand-labelled suite (25 cases) | F1 0.741 | **F1 1.000** |
| Pytest tests | **114 passing** | — |

---

## Architecture

```
LLM output
    │
    ▼
normalize_text()          ← strips formatting/obfuscation
    │
    ▼
Stage 1: Presidio         ← NER + regex + custom recognisers
    │                        PK_CNIC, US_SSN, IBAN_CODE
    │
    ├── HIGH risk ──────► sanitize + return (Stage 2 skipped)
    │
    └── MEDIUM/LOW/CLEAN ► Stage 2: LLM-as-judge
                               facebook/bart-large-mnli
                               (keyword fallback if model offline)
                                   │
                                   ├── PII flagged ► escalate to HIGH
                                   └── clean      ► return as-is
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/AI-Security-Internships-2026/06-llm-data-leakage-prevention.git
cd 06-llm-data-leakage-prevention

# 2. Install
pip install -r requirements.txt

# 3. Run Stage 1 API
uvicorn src.main:app --reload

# 4. Test Stage 1
curl -X POST http://127.0.0.1:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "Card: 4111111111111111"}'

# 5. Test Stage 1 + Stage 2
curl -X POST http://127.0.0.1:8000/detect/v2 \
  -H "Content-Type: application/json" \
  -d '{"text": "Please use the iban I provided earlier"}'
```

---

## Running Tests

```bash
# Full test suite (114 tests)
pytest src/tests/ -v

# Stage 1 unit tests only
pytest src/tests/test_detector.py -v

# Stage 2 / LLM-as-judge tests
pytest src/tests/test_llm_judge.py -v

# v2 pipeline integration tests
pytest src/tests/test_pipeline_v2.py -v
```

---

## Evaluation Scripts

```bash
# Hand-labelled eval (25 cases)
python src/tests/eval_suite.py          # Stage 1
python src/tests/eval_suite.py --v2    # Stage 1 + Stage 2

# Large-scale synthetic eval (1,200 samples)
python src/tests/eval_large.py

# Adversarial eval (25 cases)
python src/tests/adversarial_eval.py

# Stage 1 vs Stage 1+2 comparison
python src/stage_comparison.py --adversarial

# Enron eval (synthetic fallback)
python src/enron_eval.py --synthetic --samples 500
```

---

## Repository Structure

```
src/
├── detector.py          Stage 1: Presidio + custom recognisers
├── llm_judge.py         Stage 2: LLM-as-judge (bart-large-mnli)
├── main.py              FastAPI: /detect, /detect/v2, /detect/batch
├── synthetic_gen.py     1,200-sample labelled dataset generator
├── enron_eval.py        Enron corpus evaluator
├── stage_comparison.py  Stage 1 vs Stage 1+2 comparison
├── redteam_eval.py      Red team HTTP evaluation
└── tests/
    ├── conftest.py           Shared pytest fixtures
    ├── test_detector.py      67 Stage 1 unit tests
    ├── test_llm_judge.py     24 Stage 2 unit tests
    ├── test_pipeline_v2.py   21 v2 integration tests
    ├── eval_suite.py         25-case hand-labelled eval
    ├── eval_large.py         1,200-sample large-scale eval
    └── adversarial_eval.py   25-case adversarial eval
docs/
├── proposal.md          Architecture design document
├── literature-review.md 10 paper summaries
├── weekly-progress.md   Weeks 1–5 progress log
└── final-report.md      (Week 8)
datasets/
├── README.md            Dataset documentation
├── enron-emails.md      Enron corpus setup guide
└── enron-eval-results.md Enron eval results (Week 5)
experiments/results/
├── synthetic_dataset.json        1,200 labelled samples
├── synthetic_eval.json           Large-scale eval results
├── adversarial_eval.json         Adversarial eval results
├── stage_comparison.json         Stage 1 vs Stage 1+2
├── enron_eval.json               Enron eval results
└── eval_suite_v2.json            Hand-labelled suite results
```

---

## Deliverables

| Deliverable | Due | Status |
|---|---|---|
| Literature review | Week 2 | ✅ 10 papers |
| Architecture design | Week 3 | ✅ |
| Working prototype | Week 6 | ✅ Two-stage pipeline |
| Evaluation results | Week 7 | 🔄 In progress |
| Final report | Week 8 | 🔄 Pending |

---

## Supervisor Note

This repository is managed by **CNIT/PNTLab Pisa, TECIP, Scuola Superiore Sant'Anna**.
Please contact your supervisor before making architectural changes.
All code must be original or properly attributed.
Do **not** commit API keys, passwords, or large datasets — see `.gitignore`.