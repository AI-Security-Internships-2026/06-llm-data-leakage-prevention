# Dataset: HuggingFace english_pii_43k

## Overview

- **Source:** HuggingFace — [ai4privacy/pii-masking-43k](https://huggingface.co/datasets/ai4privacy/pii-masking-43k)
- **Licence:** Apache 2.0
- **Size:** ~43,000 samples across multiple languages; English subset ≈ 17,000 records
- **Format:** JSONL — one JSON object per line
- **File to place at:** `datasets/english_pii_43k.jsonl`

---

## Purpose

This dataset provides **ground-truth character-level span annotations** for a wide
range of PII entity types, making it ideal for computing per-entity precision /
recall / F1 against the Presidio-based detector.

Specifically used for:

- **RQ1** — Evaluating detector precision/recall with exact span matching (IoU ≥ 0.5)
- **RQ2** — Identifying which PII entity types Presidio misses entirely
- **Coverage gap analysis** — The dataset contains ~30 label types; our detector
  supports ~11, so unsupported labels (PASSWORD, IPV4, MAC, VEHICLEVIN, etc.) are
  tracked as false negatives and surfaced explicitly in the eval output

---

## Record Structure

Each JSONL line contains:

| Field | Description |
|---|---|
| `source_text` | Raw text containing real PII |
| `target_text` | Anonymised version with `[LABEL]` placeholders |
| `privacy_mask` | List of `{value, start, end, label}` ground-truth spans |
| `span_labels` | BIO-tagged character spans |
| `mbert_text_tokens` | mBERT tokenisation |
| `mbert_bio_labels` | BIO labels aligned to mBERT tokens |
| `id` | Integer record ID |
| `language` | ISO language code (`"en"` for English) |
| `set` | Train/dev/test split |

---

## Entity Type Coverage

The dataset contains ~30 distinct PII label types. Mapping to Presidio:

| HF Label | Presidio Entity | Supported |
|---|---|---|
| FIRSTNAME, LASTNAME, MIDDLENAME, PREFIX, USERNAME | PERSON | ✅ |
| EMAIL | EMAIL_ADDRESS | ✅ |
| PHONENUMBER | PHONE_NUMBER | ✅ |
| CREDITCARDNUMBER, MASKEDNUMBER | CREDIT_CARD | ✅ |
| IBAN | IBAN_CODE | ✅ |
| SSN | US_SSN | ✅ |
| STREET, CITY, STATE, COUNTY, ZIPCODE, BUILDINGNUMBER | LOCATION | ✅ |
| PASSWORD, PIN, CREDITCARDCVV | — | ❌ |
| IPV4, IPV6, IP, MAC, URL, USERAGENT | — | ❌ |
| PHONEIMEI, VEHICLEVIN, VEHICLEVRM | — | ❌ |
| BITCOINADDRESS, LITECOINADDRESS, ETHEREUMADDRESS | — | ❌ |
| AGE, DOB, DATE, TIME, GENDER, HEIGHT, EYECOLOR, SEX | — | ❌ |
| ACCOUNTNUMBER, ACCOUNTNAME | — | ❌ |
| CURRENCYCODE, CURRENCYNAME, CURRENCYSYMBOL, CURRENCY, AMOUNT | — | ❌ |
| JOBTITLE, JOBTYPE, JOBAREA, ORDINALDIRECTION | — | ❌ |
| NEARBYGPSCOORDINATE, SECONDARYADDRESS, COMPANYNAME | — | ❌ |

---

## Download

```bash
# Option A — HuggingFace CLI
pip install huggingface_hub
huggingface-cli download ai4privacy/pii-masking-43k \
  --repo-type dataset \
  --local-dir datasets/pii-masking-43k

# Option B — Python
from datasets import load_dataset
ds = load_dataset("ai4privacy/pii-masking-43k", split="train")
ds.filter(lambda x: x["language"] == "en").to_json("datasets/english_pii_43k.jsonl")
```

---

## Evaluation

```bash
# Quick run (1,000 samples)
python src/hf_pii_eval.py --jsonl datasets/english_pii_43k.jsonl

# Full English subset (≈17k samples)
python src/hf_pii_eval.py --jsonl datasets/english_pii_43k.jsonl --samples 17000

# Custom output directory
python src/hf_pii_eval.py \
  --jsonl datasets/english_pii_43k.jsonl \
  --samples 1000 \
  --seed 42 \
  --output-dir experiments/results
```

Outputs written to `experiments/results/`:
- `hf_pii_eval.json` — full per-sample results
- `hf_pii_eval_summary.json` — aggregated metrics only

---

## Key Differences vs Existing Datasets

| Property | Synthetic PII | Enron Emails | **english_pii_43k** |
|---|---|---|---|
| Source | Faker (generated) | Real email corpus | Crowdsourced / synthetic templates |
| Size | 1,200 samples | ~500k emails | ~43k records |
| Ground truth | Binary LEAKING/CLEAN | None | Character-level spans + labels |
| Entity types | 8 | — | ~30 |
| Eval metric | Precision/Recall/F1 (binary) | Risk distribution | Span-level P/R/F1 per entity |
| Novel coverage | — | Real-world prose | Crypto addresses, VINs, MACs, IPs, passwords |