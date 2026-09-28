## Issue 6 Table — Backend Comparison (Real Empirical Data)

| Backend | Version | N | AUC | Timing gap (ms) | Exact recovery | ASR | 95% CI | Median Q | P95 Q | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| vLLM | 0.27.1 | 30 | 0.9999 | 621.1 | 29/30 | 0.967 | (0.833, 0.994) | 79.5 | 132.1 | Attack fully works |
| SGLang | 0.5.20 | 30 | 0.9999 | 557.9 | 2/30 | 0.067 | (0.019, 0.213) | 59.5 | 113.6 | Oracle strong; attack fails — RadixAttention block boundaries differ from vLLM APC |

**Key finding:** Both backends expose a strong KS timing oracle (AUC=0.9999, KS p<2e-29). The two-stage attack succeeds on vLLM (96.7% ASR) but fails on SGLang (6.7% ASR). Cause: SGLang RadixAttention uses different prefix-block boundary alignment than vLLM APC — Stage-1 name detection detects the wrong boundary position, so Stage-2 condition enumeration targets the wrong cache region. Backend and model effects are not conflated — same model (DeepSeek-R1-Distill-Llama-8B), same seed (42), same victims.

_Source: backend-cmp-2026-09-25-59c23b91.json (real vLLM 0.27.1 + SGLang 0.5.20, n=30/backend, seed=42)_
