## Issue 9 Table 2 — Per-Category PII Detection Support

| Category | Presidio supported? | Entities detected | Key limitation |
|---|---|---|---|
| NON_SENSITIVE | True | none | No PII present; Presidio correctly returns LOW. |
| CONTACT | True | PERSON, EMAIL_ADDRESS, PHONE_NUMBER | Well-formatted contact PII; Presidio recall ~0. |
| MEDICAL | partial | PERSON | Medical conditions (hypothyroidism, arthritis) are NOT a Presidio entity type. |
| FINANCIAL | True | PERSON, CREDIT_CARD, IBAN_CODE, US_BANK_NUMBER | Credit card, IBAN and bank numbers are well-supported; Presidio recall ~0. |
| ENTERPRISE_CONFIDENTIAL | partial | PERSON | Project/product code names (TITAN, AURORA) are NOT PII entities in Presidio. |

## FNR Sweep Summary (P0 Presidio/regex)

| FNR injected | Sensitive shared | Residual leakage risk |
|---|---|---|
| 0% | 0% | LOW |
| 5% | 5% | LOW |
| 10% | 10% | MEDIUM |
| 20% | 20% | HIGH |
| 30% | 30% | HIGH |

_5 sweep points: 0%, 5%, 10%, 20%, 30% FNR. Linear relationship confirmed._
