# Module Classification — Implementation Traceability

**Issue #1 deliverable**: Internal Table — every source module classified
as `paper`, `baseline`, `legacy`, or `experimental`.

> Generated: 2026-09-11 | Branch: hashim-week-12+13

---

## Component | Canonical module | Classification | Notes

| Component | Canonical module | Status | Notes |
|---|---|---|---|
| **Entry point** | `experiments/run_kv_attack.py` | **paper** | Single reproduction entry point. Reads `configs/paper_attack.yaml`. Delegates to `week13_harness` for vLLM runs. |
| **Paper config** | `configs/paper_attack.yaml` | **paper** | All paper-critical parameters. |
| **Smoke-test config** | `configs/smoke_test.yaml` | **paper** | 3-victim mock run for CI / reproducibility validation. |
| **Two-stage victim seeder** | `src/kv_attack/two_stage_victim_seeder.py` | **paper** | Week 13 redesigned template. Splits name / condition into non-overlapping KV block regions (128 + 64 blocks). Required for Stage 1 to work. |
| **Two-stage reconstructor** | `src/kv_attack/two_stage_reconstructor.py` | **paper** | Week 13 canonical attack algorithm. Implements 12.62× BLQ improvement. |
| **Week 13 harness** | `src/kv_attack/week13_harness.py` | **paper** | End-to-end vLLM harness: seeding → calibration → reconstruction → metrics. Used by `run_kv_attack.py` for vLLM backend. |
| **Backend abstraction** | `src/kv_attack/backends/base.py` | **paper** | Abstract `BackendClient` / `BackendInfo`. Used by all backends. |
| **vLLM backend** | `src/kv_attack/backends/vllm_backend.py` | **paper** | Production backend for paper experiments (NVIDIA GB10). |
| **Mock backend** | `src/kv_attack/backends/mock_backend.py` | **paper** | Deterministic in-process oracle. Used for CI / offline smoke tests. |
| **Two-stage mock backend** | `src/kv_attack/backends/two_stage_mock_backend.py` | **paper** | Two-stage-aware mock. Understands name/condition sentinels for realistic smoke tests. |
| **Mock tokenizer** | `src/kv_attack/mock_tokenizer.py` | **paper** | Lightweight tokenizer shim (no HuggingFace weights). Used in offline runs. |
| **Cache eviction** | `src/kv_attack/cache_eviction.py` | **paper** | Filler-request cache flush used between victims. |
| **KV attack `__init__`** | `src/kv_attack/__init__.py` | **paper** | Global constants: `MODEL_ID`, `BLOCK_SIZE`, timing params, vocabulary. |
| **Mitigation: Prefix Wall** | `src/kv_attack/mitigations/prefix_wall.py` | **paper** | M1 / M2 mitigations evaluated in the paper. |
| **Pareto runner** | `src/kv_attack/pareto_runner.py` | **paper** | M3 Presidio-gated selective isolation. Generates Pareto frontier (security/latency). |
| **Mitigation eval** | `src/kv_attack/mitigation_eval.py` | **paper** | Runs attack under M1/M2/M3 mitigations. |
| **Bits analysis** | `src/kv_attack/bits_analysis.py` | **paper** | Information-theoretic BLQ derivation. Analytical values; not presented as empirical. |
| **Adaptive reconstructor** | `src/kv_attack/adaptive_reconstructor.py` | **baseline** | Week 12 single-stage adaptive search. Superseded by `two_stage_reconstructor.py` for the paper. Retained for ablation comparison. |
| **Multi-backend harness** | `src/kv_attack/multi_backend_harness.py` | **baseline** | Week 12 harness. Used baseline linear scan across vLLM, TGI, mock. Superseded by `week13_harness.py`. |
| **Harness (week 10)** | `src/kv_attack/harness.py` | **legacy** | Week 10 baseline linear scan. Hardcodes `week10-baseline` run ID. Not used in paper. Kept for BLQ comparison baseline. |
| **Reconstructor (v1)** | `src/kv_attack/reconstructor.py` | **legacy** | Week 10 linear reconstructor. Used by legacy `harness.py`. Superseded by adaptive and two-stage variants. |
| **Attacker (calibration)** | `src/kv_attack/attacker.py` | **legacy** | Week 10 `calibrate_threshold` function. Superseded by `calibrate_two_stage()` in `two_stage_reconstructor.py`. |
| **Victim seeder (v1)** | `src/kv_attack/victim_seeder.py` | **legacy** | Week 10/12 seeder. Name+condition in same block — prevents Stage 1 from working. **Do not use for paper pipeline.** Retained for baseline replication. |
| **TGI backend** | `src/kv_attack/backends/tgi_backend.py` | **experimental** | HuggingFace TGI backend. Not used in paper (TGI APC disabled by default). |
| **SGLang backend** | `src/kv_attack/backends/sglang_backend.py` | **paper** | Issue #6 cross-backend validation. Live on SGLang 0.5.19. Cache-timing oracle confirmed (AUC=0.9999); two-stage exact-recovery algorithm needs backend-specific block-alignment retuning (0% ASR vs 68% on vLLM) — documented as expected negative result, not a bug. |
| **Week 13 no-chunked variant** | `experiments/results/kv_week13_nochunked.json` | **experimental** | Ablation with chunked_prefill=false. |
| **Detector** | `src/detector.py` | **baseline** | Presidio + BART NER detector for PII in prompts / responses. Used in guardrail experiments. |
| **Guardrail benchmark** | `src/guardrail_benchmark.py` | **baseline** | Benchmark of detector + guardrail pipeline. |
| **HF PII eval** | `src/hf_pii_eval.py` | **baseline** | Hugging Face PII dataset evaluation. |
| **Enron eval** | `src/enron_eval.py` | **baseline** | Enron email dataset PII exfiltration evaluation. |
| **Stage comparison** | `src/stage_comparison.py` | **baseline** | Compares Stage-1-only vs Stage-1+Stage-2 detection (guardrail context). |
| **Synthetic gen** | `src/synthetic_gen.py` | **baseline** | Generates synthetic PII dataset. |
| **LLM judge** | `src/llm_judge.py` | **baseline** | LLM-as-judge for qualitative eval. |
| **Red-team eval** | `src/redteam_eval.py` | **baseline** | Adversarial red-team prompt evaluation. |
| **Main pipeline** | `src/main.py` | **legacy** | Early monolithic entry point. Superseded by `run_kv_attack.py`. |

---

## Superseded Paths

These modules are **not** part of the paper pipeline. They are retained for
reproducibility of earlier weekly results only.

- `src/kv_attack/harness.py` — Week 10 linear scan
- `src/kv_attack/reconstructor.py` — Week 10 linear reconstructor
- `src/kv_attack/attacker.py` — Week 10 calibration (hardcoded thresholds)
- `src/kv_attack/victim_seeder.py` — Week 10/12 seeder (flawed block layout)
- `src/kv_attack/adaptive_reconstructor.py` — Week 12 single-stage adaptive
- `src/kv_attack/multi_backend_harness.py` — Week 12 harness
- `src/main.py` — original monolithic entry point

---

## Canonical Paper Pipeline

```
python experiments/run_kv_attack.py --config configs/paper_attack.yaml
```

Internal flow:
```
run_kv_attack.py
  └─► week13_harness.py          (vLLM) / _run_mock_attack() (mock)
        ├─► two_stage_victim_seeder.py   [victim_seeding]
        ├─► two_stage_reconstructor.calibrate_two_stage()   [calibration]
        └─► For each victim:
              ├─► evict_cache_two_stage()                    [cache flush]
              ├─► reconstruct_victim_two_stage()             [stage1 + stage2]
              │     ├─► Stage 1: scan 100 names (T1 gate)
              │     └─► Stage 2: scan 20 conditions (T2 gate)
              └─► aggregate_two_stage()                      [metrics]
```
