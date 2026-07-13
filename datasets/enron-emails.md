# Dataset: Enron Email Corpus

## Overview

- **Source:** Carnegie Mellon University (CMU) / FERC / Kaggle mirror
- **Original URL:** https://www.cs.cmu.edu/~enron/
- **Kaggle mirror:** https://www.kaggle.com/datasets/wcukierski/enron-email-dataset
- **Licence:** Public domain (released by FERC during investigation); verify before redistribution
- **Version / date downloaded:** 2001–2002 emails; dataset made public 2004
- **Size:** ~500,000 emails; ~1.3 GB raw (tar.gz); Kaggle CSV ~423 MB
- **Format:** Raw `.maildir` directory structure or single CSV (`emails.csv`) on Kaggle

---

## Purpose

The Enron corpus is the standard real-world benchmark for PII leakage evaluation
in the NLP/privacy literature (used in ProPILE, Lukas et al. 2023, and others
from the literature review). It contains real email traffic from Enron employees
including names, email addresses, phone numbers, and financial references —
making it ideal for evaluating our detector on realistic, non-synthetic text.

Specifically used for:

- **RQ1** — Evaluating Presidio precision/recall on real-world email text
- **RQ2** — Identifying which PII entity types the detector misses in natural prose
- Comparison baseline against the synthetic dataset results

---

## Download Instructions

### Option A — Kaggle CLI (recommended)

```bash
pip install kaggle
kaggle datasets download -d wcukierski/enron-email-dataset
unzip enron-email-dataset.zip -d datasets/enron/
```

Requires a Kaggle account and `~/.kaggle/kaggle.json` API token.

### Option B — CMU direct

```bash
wget https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz
tar -xzf enron_mail_20150507.tar.gz -d datasets/enron/
```

### Option C — HuggingFace Datasets

```python
from datasets import load_dataset
ds = load_dataset("enron_spam", split="train")
```

---

## File Structure (maildir format)

```
enron_mail_20150507/
└── maildir/
    ├── allen-p/          # per-employee mailbox
    │   ├── inbox/
    │   ├── sent/
    │   └── ...
    ├── arnold-j/
    └── ...               # 150 employees total
```

Each file is a raw RFC 2822 email with headers (From, To, Subject, Date) and body.

---

## Preprocessing Steps

1. Parse raw emails using Python `email` standard library
2. Extract `From`, `To`, `Subject`, `Body` fields
3. Strip quoted reply chains (lines starting with `>`) to avoid duplicate PII counting
4. Filter to English-language emails only (corpus is ~99% English)
5. Remove attachments and non-text MIME parts
6. Sample a representative subset (recommended: 5,000 emails) for evaluation

```python
import email
import os

def parse_email(filepath: str) -> dict:
    with open(filepath, "r", errors="replace") as f:
        msg = email.message_from_file(f)
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                body += part.get_payload(decode=True).decode("utf-8", errors="replace")
    else:
        body = msg.get_payload(decode=True).decode("utf-8", errors="replace")
    return {
        "from": msg["From"],
        "to": msg["To"],
        "subject": msg["Subject"],
        "body": body,
    }
```

---

## Train / Val / Test Split

Not applicable — this corpus is used for **evaluation only**, not model training.

Recommended evaluation subset: **5,000 randomly sampled emails** (seed=42)
stratified across at least 10 employee mailboxes to avoid per-employee bias.

---

## Known PII Types Present

| Entity Type | Prevalence | Notes |
|---|---|---|
| `EMAIL_ADDRESS` | Very high | In headers and body text |
| `PERSON` | High | Names in signatures, greetings, CC lines |
| `PHONE_NUMBER` | Medium | In signatures and contact blocks |
| `LOCATION` | Medium | Office addresses in signatures |
| `CREDIT_CARD` | Low | Rare; some financial emails |
| `US_SSN` | Very low | Rare; appears in HR-related emails |

---

## Licence and Privacy Notes

- The corpus was made public by FERC (Federal Energy Regulatory Commission)
  as part of its investigation into Enron's collapse.
- It contains real PII of real individuals. Handle with care:
  - Do **not** commit raw email text to the repository
  - Do **not** publish individual email content in reports
  - Use only for research evaluation; anonymise results at the individual level
- For publication, cite the CMU release and note the FERC source.

---

## Citation

```
Klimt, B., & Yang, Y. (2004).
The Enron Corpus: A New Dataset for Email Classification Research.
In European Conference on Machine Learning (ECML), pp. 217–226.
```

---

## Notes

- Raw dataset is **not committed** to this repository — see `datasets/README.md` policy.
- Expected location after download: `datasets/enron/` (gitignored).
- The `.gitignore` already excludes `*.tar.gz` and large data directories.
- Evaluation scripts using this corpus will be added in Week 5.