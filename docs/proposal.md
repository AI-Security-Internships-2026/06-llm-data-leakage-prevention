# Research Proposal: LLM Data Leakage Prevention: Detection and Mitigation

**Student:** Muhammad Hashim Mughal
**Supervisor:** _[Fill in supervisor name]_
**Start date:** 2026-06-
**Expected end date:** 2026-08-

---

## 1. Background

Large Language Models are increasingly deployed in enterprise and research
settings where they process or have access to sensitive data — including
personally identifiable information (PII), financial records, and internal
documents. A growing body of research (Carlini et al., 2021; Nasr et al.,
2023) has demonstrated that LLMs memorise and can reproduce verbatim
fragments of their training data, and that even production-hardened,
RLHF-aligned models remain vulnerable. At the same time, Lukas et al.
(2023) show that leakage is not limited to verbatim reproduction — models
can also leak PII through inference and association. These risks motivate
the need for robust inference-time detection and sanitisation mechanisms
that operate on model outputs before they reach the user.

This project is carried out within the AI Security research agenda of
CNIT/PNTLab Pisa (TECIP, Scuola Superiore Sant'Anna).

---

## 2. Problem Statement

Current LLM deployments lack a reliable, low-latency output-side filter
that can detect and neutralise PII leakage across multiple leakage
modalities (verbatim, inference-based, and association-based). Existing
tools either operate only at the input prompt level, focus exclusively on
training-time mitigations such as deduplication, or rely on heavyweight
LLM-as-judge approaches that introduce unacceptable latency for real-time
applications. There is no publicly available, extensible prototype that
combines rule-based PII detection with a lightweight risk scoring layer
and exposes it as a production-ready API endpoint. This project addresses
that gap by building, evaluating, and documenting such a system.

---

## 3. Research Questions

1. **RQ1** — How accurately does a rule-based detector (Microsoft Presidio +
   custom recognisers) detect PII in LLM outputs, measured by
   precision, recall, and F1 on a labelled evaluation set?

2. **RQ2** — Which PII entity types and leakage contexts (embedded in logs,
   JSON payloads, medical prose, multilingual text) cause the most
   false negatives, and can a second detection layer (LLM-as-judge)
   close that gap?

3. **RQ3** — What is the latency and accuracy tradeoff between the
   rule-based layer alone versus the two-layer pipeline, and at what
   throughput does the two-layer approach become impractical for
   real-time use?

---

## 4. Proposed Methodology

### 4.1 Data Collection / Dataset

| Dataset | Source | Licence | Use |
|---|---|---|---|
| Enron Email Corpus | CMU / Kaggle | Research use | Real-world PII leakage evaluation |
| Synthetic PII texts | Generated via Faker (Python) | MIT | Controlled precision/recall tests |
| garak leakreplay probes | NVIDIA / GitHub | Apache 2.0 | Training-data memorisation tests |

Datasets are not committed to the repository. Each will be documented
in `datasets/<name>.md` following the project dataset policy.

### 4.2 Approach

The system is a two-stage inference-time pipeline:

```
LLM output text
│
▼
┌─────────────────────────────┐
│  Stage 1: Rule-based layer  │  ← Presidio Analyzer + custom recognisers
│  (Presidio + regex/NER)     │    (PK_CNIC, domain-specific patterns)
└────────────┬────────────────┘
             │  risk_level, entities
             ▼
┌─────────────────────────────┐
│  Risk gate                  │  HIGH → sanitise and return
│                             │  MEDIUM/LOW → pass to Stage 2
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Stage 2: LLM-as-judge      │  ← Planned Week 5–6
│  (lightweight model prompt) │
└────────────┬────────────────┘
             │
             ▼
      Sanitised output
      + risk metadata
```

Stage 1 is fully implemented as of Week 2–3. Stage 2 is planned for
Weeks 5–6 once the evaluation benchmark is established.

### 4.3 Evaluation Metrics

| Metric | Definition | Target |
|---|---|---|
| Precision | TP / (TP + FP) | ≥ 0.90 |
| Recall | TP / (TP + FN) | ≥ 0.85 |
| F1 score | Harmonic mean of P and R | ≥ 0.87 |
| p95 latency | 95th-percentile response time | ≤ 200 ms |

Recall is weighted more heavily than precision — missing a leak is
a worse outcome than a false alarm in a security context.

### 4.4 Tooling

| Tool | Role |
|---|---|
| Microsoft Presidio | Core PII detection and anonymisation |
| spaCy `en_core_web_lg` | NER backbone for Presidio |
| FastAPI + Uvicorn | API layer |
| pytest + httpx | Unit and integration testing |
| Faker | Synthetic PII dataset generation |
| garak | Adversarial leakage probing |

---

## 5. Expected Outcome

A working, documented, and evaluated prototype API that accepts raw LLM
output text and returns a risk classification, a list of detected PII
entities (without echoing the raw values), and a fully sanitised version
of the text. The prototype will be accompanied by a labelled evaluation
dataset, a precision/recall report, and a final technical report
documenting findings and limitations.

---

## 6. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Presidio misses inference-based leakage | High | Add LLM-as-judge layer in Week 5–6; document gap in final report |
| Enron corpus licence restrictions | Low | Fall back to fully synthetic dataset generated with Faker |
| Stage 2 latency too high for real-time | Medium | Benchmark early; use async endpoints; consider caching |
| Scope too broad | Medium | RQ1 is the hard deliverable; RQ2–3 are extensions if time allows |
| No API authentication in prototype | Known | Documented limitation; out of scope for research prototype |

---

_Last updated: 2026-07-05_