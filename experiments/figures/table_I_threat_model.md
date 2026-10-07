## Table I — Threat Model

| Property | Value |
|---|---|
| Attack type | KV-cache timing side-channel |
| Attacker access | Black-box API (shared LLM endpoint) |
| Attacker knowledge | System prompt prefix known (Scenario S2) |
| Target | Private PII in victim system prompt |
| Observable | Time-to-first-token (TTFT) per API call |
| Candidate space | 100 names × 20 conditions = 2,000 combinations |
| Model | deepseek-ai/DeepSeek-R1-Distill-Llama-8B |
| Backend | vLLM 0.27.1 (APC on, block_size=16) |
| Seeds | 42, 43, 44 (3 independent seeds) |
| Oracle gap (D0) | 697 ms (AUC=1.000, TPR@1%FPR=1.000) |
