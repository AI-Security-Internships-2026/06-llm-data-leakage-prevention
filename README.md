# LLM Data Leakage Prevention: Detection and Mitigation

> **CNIT/PNTLab Pisa · TECIP · Scuola Superiore Sant'Anna — AI Security Internship 2026**

---

## Research Problem

Research and implement mechanisms that detect when an LLM is about to leak sensitive data (PII, secrets, internal documents) and automatically sanitise or refuse the response.

---

## Objectives

1. Conduct a systematic literature review on the topic.
2. Design and implement a proof-of-concept prototype.
3. Evaluate the prototype on real or benchmark datasets.
4. Document findings in a final technical report.
5. Present results to the research group.

---

## Expected Deliverables

| Deliverable | Due |
|---|---|
| Literature review (`docs/literature-review.md`) | Week 2 |
| Architecture design document (`docs/proposal.md`) | Week 3 |
| Working prototype (`src/`) | Week 6 |
| Evaluation results (`experiments/results/`) | Week 7 |
| Final report (`docs/final-report.md`) | Sep 8 — see "Roadmap to September 8" below; superseded from the original Week 8 date |
---

## Recommended Technology Stack

```
Python, Presidio, HuggingFace, FastAPI, Regex, spaCy
```

See `requirements.txt` for pinned dependencies.

---

## Weekly Workflow

```
Monday     – Review weekly tasks in tasks/week-XX.md
Tue–Thu    – Implementation / experiments
Friday     – Document progress in docs/weekly-progress.md
Friday     – Open weekly Pull Request from your branch → dev
```

---

## Branching Policy

| Branch | Purpose |
|---|---|
| `main` | Stable, supervisor-reviewed code only |
| `dev` | Integration branch — merge weekly PRs here |
| `<your-name>-week-XX` | Your working branch for each week |

**Students must never push directly to `main`.**

---

## Pull Request Policy

- One PR per week, targeting the `dev` branch.
- PR title format: `[Week XX] Brief description`
- PR description must reference the weekly task file and summarise what was done.
- A supervisor or co-student must review before merging.

---

## Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/AI-Security-Internships-2026/06-llm-data-leakage-prevention.git
cd 06-llm-data-leakage-prevention

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create your weekly branch
git checkout dev
git pull origin dev
git checkout -b your-name-week-01

# 5. Run the starter script
python src/main.py
```

---

## Roadmap to September 8, 2026

**Current state:** real Enron-corpus and HF-PII evaluation, benchmarked against scrubadub and detect-secrets. A Phase 2 research assignment already exists (issue #9): the KV-cache timing side-channel, flagged for "journal-track rigor" — this is the project's real novel-contribution track.

**Novel contribution target:** the KV-cache timing side-channel in issue #9 — showing that PII can leak through inference *timing* even when the text-level detector catches everything, would be a genuinely publishable-quality result.

| Date | Milestone |
|---|---|
| Aug 2 | Finalize the week-6 guardrail comparison and false-positive fixes already in progress |
| Aug 9 | Phase 2 start (issue #9): reproduce the KV-cache timing side-channel PII leakage |
| Aug 16 | Build a detection/mitigation approach for the timing side-channel |
| Aug 23 | Benchmark mitigation overhead vs. security improvement |
| Aug 30 | Full journal-track write-up of the timing side-channel finding |
| Sep 6 | Paper draft |
| **Sep 8** | **Final submission** |

---

## Supervisor Note

This repository is managed by **CNIT/PNTLab Pisa, TECIP, Scuola Superiore Sant'Anna**.
Please contact your supervisor before making architectural changes.
All code must be original or properly attributed.
Do **not** commit API keys, passwords, or large datasets — see `.gitignore`.
