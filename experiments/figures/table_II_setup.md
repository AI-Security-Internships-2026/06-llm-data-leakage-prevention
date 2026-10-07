## Table II — Experimental Setup

| Parameter | Value |
|---|---|
| Hardware | NVIDIA GB10, 122 GB unified memory, CUDA 13.0 |
| vLLM version | 0.27.1 |
| SGLang version | 0.5.20 (Issue #6 only) |
| Model | deepseek-ai/DeepSeek-R1-Distill-Llama-8B |
| Block size | 16 tokens |
| GPU memory utilization | 0.40 |
| Seeds (standard profile) | 42, 43, 44 |
| n_victims A2 (per defense×seed) | 6 |
| n_victims A1 (per defense×seed) | 3 |
| n_victims A0 (per defense×seed) | 2 |
| Calibration probes | 50 per seed |
| Trial isolation | epoch (contamination check passed) |
| CI method | Wilson score (95%) |
| Total trials | 453 / 453 (0 errors) |
| Risk model | Presidio NER (en_core_web_lg) |
