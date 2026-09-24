# Issue #7 Table 2: Risk Level → Detector Condition → Cache Action

| Risk Level | Detector Condition | Cache Action (D4) | Cross-tenant Reuse | Example Entities |
|------------|-------------------|-------------------|-------------------|------------------|
| LOW / CLEAN | No PII detected, risk_score < 0.25 | SHARE (no salt) | ✅ Permitted | — |
| MEDIUM | PII detected, confidence < 0.7, not high-risk type | RESTRICT (tenant salt) | ❌ Cross-tenant blocked | PERSON, EMAIL, PHONE |
| HIGH | High-risk PII detected (CREDIT_CARD, SSN, IBAN, MEDICAL_LICENSE) or ≥3 entities conf≥0.7 | ISOLATE (unique UUID salt) | ❌ Blocked (even intra-tenant) | CREDIT_CARD, IBAN_CODE, US_SSN, MEDICAL_LICENSE |
