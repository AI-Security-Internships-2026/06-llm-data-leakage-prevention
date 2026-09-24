## Issue 5 Table 1 — Oracle Quality + Attack Success by Load Level

| Load | Tenants | KS p-value | Oracle feasible | ASR | 95% CI | Exact | Median Q | P95 Q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L0 | 0 | 1.98e-29 | ✓ | 0.900 | (0.744, 0.965) | 27/30 | 79 | 151 |
| L1 | 2 | 1.98e-29 | ✓ | 1.000 | (0.886, 1.000) | 30/30 | 80 | 136 |
| L2 | 5 | 1.98e-29 | ✓ | 0.900 | (0.744, 0.965) | 27/30 | 96 | 137 |
| L3 | 10 | 1.98e-29 | ✓ | 1.000 | (0.886, 1.000) | 30/30 | 88 | 134 |
| L4 | 20 | 1.98e-29 | ✓ | 0.967 | (0.833, 0.994) | 29/30 | 98 | 138 |

_Source: load-test-2026-09-22-ddd807bd.json (real vLLM, n=30/level, seed=42)_
_Note: KS p=1.98e-29 (min float) at all levels — distributions maximally separated. ROC-AUC not computed in real run; see KS test as separability proxy._
