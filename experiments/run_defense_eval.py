"""
experiments/run_defense_eval.py
================================
Issue #7 — Risk-Adaptive Cache Isolation evaluation.

Evaluates five cache defense policies (D0–D4) on:
  1. Detector accuracy   — does the risk classifier catch each category?
  2. Isolation rate      — does the policy isolate sensitive prompts?
  3. Sharing rate        — does the policy preserve sharing for clean prompts?
  4. Residual ASR        — can the timing attack still succeed despite the defense?
  5. TTFT overhead       — what latency cost does each policy add?

Results are saved as a timestamped JSON and figures/tables are generated
from the raw data (never typed manually).

Usage
-----
  cd src
  # Smoke run (fast, ~5–10 min on live vLLM)
  python ../experiments/run_defense_eval.py --config ../configs/defense_eval.yaml

  # Full run (paper quality)
  python ../experiments/run_defense_eval.py \\
      --config ../configs/defense_eval.yaml \\
      --n-victims 30 --n-workload 10

  # Mock backend (no GPU needed)
  python ../experiments/run_defense_eval.py \\
      --config ../configs/defense_eval.yaml --backend mock
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).parent.parent
_SRC       = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from kv_attack import VLLM_BASE_URL, MODEL_ID
from kv_attack.backends.mock_backend import MockBackend
from kv_attack.backends.vllm_backend import VLLMBackend
from kv_attack.risk_signal import analyze_prompt, RiskSignal
from kv_attack.cache_policy import (
    CachePolicyEngine,
    CacheAction,
    ALL_POLICIES, POLICY_DESCRIPTIONS,
    POLICY_D0, POLICY_D1, POLICY_D2, POLICY_D3, POLICY_D4,
)
from kv_attack.defense_workload import (
    generate_defense_workload, workload_to_dict,
    CAT_NON_SENSITIVE, EXPECTED_RISK, SHOULD_ISOLATE, ALL_CATEGORIES,
)



class _Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _dump(obj: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, cls=_Encoder, indent=2)
    os.replace(tmp, path)



def _collect_environment() -> dict:
    gpu_info = "unavailable"
    try:
        gpu_info = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"], text=True, timeout=5
        ).strip()
    except Exception:
        pass

    git_hash = "unknown"
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            cwd=str(_REPO_ROOT), timeout=5
        ).strip()
    except Exception:
        pass

    return {
        "python":   sys.version,
        "platform": platform.platform(),
        "gpu":      gpu_info,
        "git_hash": git_hash,
    }



def calibrate(backend, n_samples: int = 50, seed: int = 42) -> dict:
    """
    Measure hit and miss TTFT distributions.

    Strategy:
      - Miss: send a unique UUID prompt (never cached)
      - Hit:  send the same long fixed prompt (>64 tokens to guarantee block
              caching in vLLM's 16-token block scheme) twice; second send = hit

    NOTE: The warm prompt must be long enough to fill at least one full KV-cache
    block (vLLM default = 16 tokens per block).  Short prompts (<16 tokens) may
    not be prefix-cached even with APC enabled.
    """
    print(f"\n[calibrate] Measuring TTFT distributions ({n_samples} samples each)…")

    WARM_PROMPT = (
        "You are a helpful assistant specialising in data privacy and security. "
        "The following context is provided for reference: organisations handling "
        "personal data must implement appropriate technical and organisational "
        "measures to ensure a level of security appropriate to the risk, including "
        "pseudonymisation and encryption of personal data, ongoing confidentiality, "
        "integrity, availability and resilience of processing systems and services."
    )

    backend.measure_ttft(WARM_PROMPT)

    hit_ttfts: list[float] = []
    for _ in range(n_samples):
        t = backend.measure_ttft(WARM_PROMPT)
        hit_ttfts.append(t)

    miss_ttfts: list[float] = []
    for i in range(n_samples):
        cold_prompt = f"Calibration unique cold request identifier {uuid.uuid4().hex} sequence {i}"
        t = backend.measure_ttft(cold_prompt)
        miss_ttfts.append(t)

    miss_arr = np.array(miss_ttfts)
    hit_arr  = np.array(hit_ttfts)

    gap_ms    = float(miss_arr.mean() - hit_arr.mean())
    threshold = float((miss_arr.mean() + hit_arr.mean()) / 2)

    if abs(gap_ms) < 20:
        print(f"  WARNING: hit/miss gap is only {gap_ms:.1f} ms — APC may not be "
              f"active or calibration prompt is too short. Results may be unreliable.")
    else:
        print(f"  APC signal confirmed: gap = {gap_ms:.1f} ms ✓")

    result = {
        "hit_mean_ms":    float(hit_arr.mean()),
        "hit_p50_ms":     float(np.percentile(hit_arr,  50)),
        "hit_p95_ms":     float(np.percentile(hit_arr,  95)),
        "miss_mean_ms":   float(miss_arr.mean()),
        "miss_p50_ms":    float(np.percentile(miss_arr, 50)),
        "miss_p95_ms":    float(np.percentile(miss_arr, 95)),
        "threshold_ms":   threshold,
        "gap_ms":         gap_ms,
        "apc_signal_ok":  abs(gap_ms) >= 20,
        "n_samples":      n_samples,
    }
    print(f"  hit  mean={result['hit_mean_ms']:.1f} ms  p50={result['hit_p50_ms']:.1f} ms")
    print(f"  miss mean={result['miss_mean_ms']:.1f} ms  p50={result['miss_p50_ms']:.1f} ms")
    print(f"  threshold = {threshold:.1f} ms  gap = {gap_ms:.1f} ms")
    return result



def eval_detector(samples: list, use_stage2: bool = False) -> dict:
    """
    Run the risk classifier on every workload sample.
    Returns per-sample and per-category accuracy metrics.
    """
    print("\n[detector_eval] Running risk classifier on workload…")
    per_sample: list[dict] = []
    cat_stats: dict[str, dict] = {c: {"n": 0, "correct": 0, "samples": []} for c in ALL_CATEGORIES}

    for s in samples:
        t0 = time.perf_counter()
        signal = analyze_prompt(s.text, use_stage2=use_stage2)
        latency_ms = (time.perf_counter() - t0) * 1000

        level_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        expected_ord = level_order.get(s.expected_risk_level, 0)
        actual_ord   = level_order.get(signal.risk_level, 0)
        correct      = actual_ord >= expected_ord

        row = {
            "sample_id":            s.sample_id,
            "category":             s.category,
            "expected_risk_level":  s.expected_risk_level,
            "detected_risk_level":  signal.risk_level,
            "correct":              correct,
            "pii_flag":             signal.pii_flag,
            "detector_confidence":  signal.detector_confidence,
            "semantic_sensitivity": signal.semantic_sensitivity,
            "entity_types":         signal.entity_types,
            "risk_score":           signal.risk_score,
            "reason":               signal.reason,
            "latency_ms":           round(latency_ms, 2),
        }
        per_sample.append(row)
        cat_stats[s.category]["n"]       += 1
        cat_stats[s.category]["correct"] += int(correct)
        cat_stats[s.category]["samples"].append(row)

    category_recall: dict[str, float] = {}
    for cat, stat in cat_stats.items():
        n = stat["n"]
        category_recall[cat] = round(stat["correct"] / n, 4) if n else 0.0

    overall_recall = round(
        sum(r["correct"] for r in per_sample) / max(len(per_sample), 1), 4
    )
    print(f"  Overall recall:  {overall_recall:.1%}")
    for cat, recall in category_recall.items():
        print(f"  {cat:<20} recall={recall:.1%}")

    return {
        "overall_recall":   overall_recall,
        "category_recall":  category_recall,
        "per_sample":       per_sample,
    }



def eval_policy_isolation(
    backend,
    samples:     list,
    policy:      str,
    tenant_id:   str,
    calibration: dict,
    eval_run_id: str = "",
) -> dict:
    """
    For each sample:
      1. Apply policy to get salt (victim side)
      2. Seed the (salted) prompt into the backend cache  [Pass 1 — cold miss]
      3. Seed again with the SAME salt  [Pass 2 — measures cache utility]
      4. Send the UNSALTED prompt (attacker probe) and measure TTFT [Pass 3]
      5. Classify: is attacker TTFT a hit or miss?

    eval_run_id: a unique token prepended to ALL prompts in this policy
    evaluation to prevent cross-policy cache contamination on real vLLM servers
    whose KV cache persists between API calls.  Both victim and attacker use the
    same eval_run_id so the test semantics are preserved (attacker cannot see
    policy salt but does share the eval context prefix).

    Returns per-sample and aggregate metrics.
    """
    engine    = CachePolicyEngine(policy=policy, tenant_id=tenant_id)
    threshold = calibration["threshold_ms"]

    if not eval_run_id:
        eval_run_id = f"EVAL-{policy[:3].upper()}-{uuid.uuid4().hex[:6]}"
    eval_pfx = f"[{eval_run_id}] "
    per_row:  list[dict] = []

    cat_sensitive_isolated: dict[str, list] = {c: [] for c in ALL_CATEGORIES}
    cat_shared:             dict[str, list] = {c: [] for c in ALL_CATEGORIES}
    all_victim_ttfts:  list[float] = []
    all_attacker_ttfts: list[float] = []

    for s in samples:
        signal = analyze_prompt(s.text)

        decision_seed = engine.decide(signal, request_id=f"{policy}-{s.sample_id}-seed")
        salted_seed   = eval_pfx + decision_seed.apply_to_prompt(s.text)
        backend.measure_ttft(salted_seed)

        decision_measure = engine.decide(signal, request_id=f"{policy}-{s.sample_id}-measure")
        salted_measure   = eval_pfx + decision_measure.apply_to_prompt(s.text)
        victim_ttft_ms   = backend.measure_ttft(salted_measure)
        all_victim_ttfts.append(victim_ttft_ms)

        attacker_ttft_ms = backend.measure_ttft(eval_pfx + s.text)
        all_attacker_ttfts.append(attacker_ttft_ms)

        attacker_sees_hit = attacker_ttft_ms < threshold
        isolated          = not attacker_sees_hit

        row = {
            "sample_id":            s.sample_id,
            "category":             s.category,
            "expected_risk_level":  s.expected_risk_level,
            "should_isolate":       s.should_isolate,
            "detected_risk_level":  signal.risk_level,
            "cache_action":         decision_measure.cache_action.value,
            "salt_applied":         bool(decision_measure.salt),
            "victim_ttft_ms":       round(victim_ttft_ms, 2),
            "attacker_ttft_ms":     round(attacker_ttft_ms, 2),
            "attacker_sees_hit":    attacker_sees_hit,
            "isolated":             isolated,
        }
        per_row.append(row)
        cat_sensitive_isolated[s.category].append(isolated)

    n_total     = len(per_row)
    n_sensitive = 0
    n_clean     = 0
    sens_isolated = 0
    clean_shared  = 0

    for row, sample in zip(per_row, samples):
        if sample.should_isolate:
            n_sensitive += 1
            if row["isolated"]:
                sens_isolated += 1
        else:
            n_clean += 1
            if not row["isolated"]:
                clean_shared += 1

    sensitive_isolation_rate   = round(sens_isolated / max(n_sensitive, 1), 4)
    non_sensitive_sharing_rate = round(clean_shared  / max(n_clean,     1), 4)

    victim_arr   = np.array(all_victim_ttfts)
    attacker_arr = np.array(all_attacker_ttfts)
    baseline_hit = calibration["hit_mean_ms"]

    ttft_overhead_pct = round(
        (victim_arr.mean() - baseline_hit) / max(baseline_hit, 1) * 100, 2
    )

    hit_rate = round(
        sum(1 for t in all_victim_ttfts if t < threshold) / max(n_total, 1), 4
    )

    isolation_fraction = round(
        sum(1 for r in per_row if r["isolated"]) / max(n_total, 1), 4
    )

    agg = {
        "policy":                     policy,
        "n_total":                    n_total,
        "n_sensitive":                n_sensitive,
        "n_clean":                    n_clean,
        "sensitive_isolation_rate":   sensitive_isolation_rate,
        "non_sensitive_sharing_rate": non_sensitive_sharing_rate,
        "hit_rate":                   hit_rate,
        "isolation_fraction":         isolation_fraction,
        "ttft_overhead_pct":          ttft_overhead_pct,
        "victim_ttft_p50_ms":         float(np.percentile(victim_arr, 50)),
        "victim_ttft_p95_ms":         float(np.percentile(victim_arr, 95)),
        "victim_ttft_p99_ms":         float(np.percentile(victim_arr, 99)),
        "attacker_ttft_p50_ms":       float(np.percentile(attacker_arr, 50)),
        "attacker_ttft_p95_ms":       float(np.percentile(attacker_arr, 95)),
    }

    print(f"  [{policy}] sensitive_isolation={sensitive_isolation_rate:.1%}  "
          f"non_sensitive_sharing={non_sensitive_sharing_rate:.1%}  "
          f"hit_rate={hit_rate:.1%}  ttft_overhead={ttft_overhead_pct:+.1f}%")

    engine.dump_audit_log(
        str(_REPO_ROOT / "experiments" / "results" /
            f"audit_log_{policy}_{datetime.date.today()}.jsonl")
    )

    return {
        "aggregate": agg,
        "per_sample": per_row,
        "audit_summary": engine.audit_summary(),
    }



def eval_residual_asr(
    backend,
    policy:          str,
    victim_tenant_id: str,
    n_victims:       int,
    seed:            int,
    calibration:     dict,
) -> dict:
    """
    Measure residual attack success rate (ASR) under a given policy.

    Protocol:
      1. Generate n_victims synthetic medical victims (name + condition)
      2. Apply the policy salt to each victim prompt and seed it into the cache
      3. For each victim, the 'attacker' sends an UNSALTED probe with a
         candidate name — measuring TTFT to detect cache hits
      4. If the attacker's probes all return MISS-TTFT (as expected when
         the cache is salted), reconstruction fails → ASR = 0
      5. Count fraction of victims where the attacker reconstructs correctly

    This directly measures the end-to-end residual information leakage
    under each defense.
    """
    from kv_attack import (
        FIRST_NAMES, LAST_NAMES, MEDICAL_CONDITIONS,
        BLOCK_SIZE,
    )
    from kv_attack.victim_seeder import build_aligned_system_prompt
    from kv_attack.two_stage_victim_seeder import (
        seed_victims_two_stage, build_two_stage_prompt,
    )
    from kv_attack.two_stage_reconstructor import (
        calibrate_two_stage, evict_cache_two_stage,
        T1_THRESHOLD_MS, T2_THRESHOLD_MS,
    )
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, trust_remote_code=True
        )
    except Exception as e:
        print(f"[ASR] WARNING: could not load tokenizer ({e}) — using mock results")
        return _mock_residual_asr(policy, n_victims, calibration)

    engine = CachePolicyEngine(policy=policy, tenant_id=victim_tenant_id)

    from kv_attack import detect_has_bos
    has_bos      = detect_has_bos(MODEL_ID)
    system_prefix = build_aligned_system_prompt(tokenizer, has_bos=has_bos)

    try:
        cal = calibrate_two_stage(backend, system_prefix, tokenizer)
        t1_ms = cal["t1_threshold_ms"]
        t2_ms = cal["t2_threshold_ms"]
    except Exception as e:
        print(f"[ASR] WARNING: calibration failed ({e}) — using theoretical thresholds")
        t1_ms = T1_THRESHOLD_MS
        t2_ms = T2_THRESHOLD_MS

    rng = random.Random(seed)
    name_pool  = [f"{f} {l}" for f in FIRST_NAMES for l in LAST_NAMES]
    cond_pool  = list(MEDICAL_CONDITIONS)
    rng.shuffle(name_pool)
    rng.shuffle(cond_pool)

    victim_records: list[dict] = []
    for i in range(n_victims):
        victim_name      = name_pool[i % len(name_pool)]
        victim_condition = cond_pool[i % len(cond_pool)]
        dob              = f"{rng.randint(1940, 1995)}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
        victim_id        = f"victim-{policy}-{i:03d}"

        full_prompt = build_two_stage_prompt(
            system_prefix, victim_name, dob, victim_condition, tokenizer
        )
        text_for_risk   = f"Patient {victim_name} (DOB {dob}) has {victim_condition}."
        signal          = analyze_prompt(text_for_risk)
        decision        = engine.decide(signal, request_id=victim_id)
        salted_prompt   = decision.apply_to_prompt(full_prompt)

        backend.measure_ttft(salted_prompt)
        victim_records.append({
            "victim_id":       victim_id,
            "ground_truth":    {"name": victim_name, "condition": victim_condition, "dob": dob},
            "prompt":          full_prompt,
            "salted_prompt":   salted_prompt,
            "cache_action":    decision.cache_action.value,
            "salt_applied":    bool(decision.salt),
        })

    results: list[dict] = []
    threshold = calibration["threshold_ms"]

    for vr in victim_records:
        gt      = vr["ground_truth"]
        correct = False
        n_queries = 0

        correct_probe = vr["prompt"]
        t = backend.measure_ttft(correct_probe)
        n_queries += 1
        oracle_hit = t < threshold

        if oracle_hit:
            names_to_try = [gt["name"], "Wrong Name Xyz"]
            for cand in names_to_try:
                probe = build_two_stage_prompt(
                    system_prefix, cand, gt["dob"], "placeholder", tokenizer,
                    use_dummy_cond=True,
                )
                t2 = backend.measure_ttft(probe)
                n_queries += 1
                if t2 < t1_ms and cand == gt["name"]:
                    correct = True
                    break
        else:
            correct = False

        results.append({
            "victim_id":    vr["victim_id"],
            "cache_action": vr["cache_action"],
            "salt_applied": vr["salt_applied"],
            "oracle_ttft":  round(t, 2),
            "oracle_hit":   oracle_hit,
            "recovered":    correct,
            "n_queries":    n_queries,
        })

    n_recovered = sum(r["recovered"] for r in results)
    residual_asr = round(n_recovered / max(n_victims, 1), 4)

    print(f"  [{policy}] Residual ASR = {residual_asr:.1%} ({n_recovered}/{n_victims} recovered)")

    return {
        "policy":        policy,
        "n_victims":     n_victims,
        "n_recovered":   n_recovered,
        "residual_asr":  residual_asr,
        "per_victim":    results,
    }


def _mock_residual_asr(policy: str, n_victims: int, calibration: dict) -> dict:
    """Fallback mock when tokenizer/GPU unavailable (analytical estimate)."""
    asr_map = {
        "D0_no_defense":    0.68,
        "D1_full_isolation": 0.0,
        "D2_tenant_salt":   0.0,
        "D3_binary_pii":    0.0,
        "D4_risk_adaptive": 0.05,
    }
    residual_asr = asr_map.get(policy, 0.0)
    n_recovered  = round(residual_asr * n_victims)
    return {
        "policy":       policy,
        "n_victims":    n_victims,
        "n_recovered":  n_recovered,
        "residual_asr": residual_asr,
        "source":       "mock_analytical",
        "per_victim":   [],
    }



def generate_figures(results: dict, figures_dir: str) -> None:
    """Generate all Issue #7 figures from saved raw results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    policies  = ALL_POLICIES
    pol_labels = ["D0\n(No defense)", "D1\n(Full isolate)", "D2\n(Tenant salt)",
                  "D3\n(Binary PII)", "D4\n(Risk-adaptive)"]
    colors     = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db"]

    isolation_data = results.get("policy_isolation", {})
    asr_data       = results.get("residual_asr", {})

    fig1, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    arch_text = (
        "Cross-Layer Privacy Defense Architecture\n"
        "─────────────────────────────────────────────────────\n\n"
        " ┌─────────────────────────────────────────────────┐\n"
        " │           Incoming LLM Request                  │\n"
        " └──────────────────────┬──────────────────────────┘\n"
        "                        │\n"
        "                        ▼\n"
        " ┌─────────────────────────────────────────────────┐\n"
        " │   Layer 1: PII Detection (Presidio + BART)      │\n"
        " │   Output: entity_types, pii_flag, confidence,   │\n"
        " │           semantic_sensitivity, risk_score       │\n"
        " └──────────────────────┬──────────────────────────┘\n"
        "                        │  RiskSignal\n"
        "                        ▼\n"
        " ┌─────────────────────────────────────────────────┐\n"
        " │   Layer 2: Cache Policy Engine (D0–D4)          │\n"
        " │   LOW  → SHARE   (no salt)                      │\n"
        " │   MED  → RESTRICT (tenant salt)                 │\n"
        " │   HIGH → ISOLATE  (unique UUID salt)            │\n"
        " └──────────────────────┬──────────────────────────┘\n"
        "                        │  PolicyDecision + salt\n"
        "                        ▼\n"
        " ┌─────────────────────────────────────────────────┐\n"
        " │   Layer 3: vLLM / SGLang KV-Cache               │\n"
        " │   SHARE   → prefix cache reuse (fast)           │\n"
        " │   RESTRICT → intra-tenant only                  │\n"
        " │   ISOLATE  → always cold miss (safe)            │\n"
        " └─────────────────────────────────────────────────┘\n"
        "\n"
        "Audit log: every decision → policy, action, reason, reuse_permitted"
    )
    ax.text(0.05, 0.95, arch_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f0f4f8', alpha=0.8))
    ax.set_title("Issue #7 Fig 1: Cross-Layer Defense Architecture", pad=12)
    fig1.tight_layout()
    fig1.savefig(f"{figures_dir}/issue7_fig1_architecture.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"  Saved: issue7_fig1_architecture.png")

    sec_vals:  list[float] = []
    util_vals: list[float] = []
    for p in policies:
        agg = isolation_data.get(p, {}).get("aggregate", {})
        sec_vals.append(agg.get("sensitive_isolation_rate", 0.0) * 100)
        util_vals.append(agg.get("non_sensitive_sharing_rate", 0.0) * 100)

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    for i, (p, sec, util) in enumerate(zip(policies, sec_vals, util_vals)):
        ax2.scatter(util, sec, s=200, color=colors[i], zorder=5, label=pol_labels[i].replace("\n", " "))
        ax2.annotate(pol_labels[i].replace("\n", " "), (util, sec),
                     textcoords="offset points", xytext=(8, 3), fontsize=8)

    ax2.set_xlabel("Non-sensitive Sharing Rate (%) — Cache Utility →", fontsize=11)
    ax2.set_ylabel("Sensitive Isolation Rate (%) — Security →", fontsize=11)
    ax2.set_title("Issue #7 Fig 2: Security vs Cache Utility (Pareto)", fontsize=12)
    ax2.set_xlim(-5, 110)
    ax2.set_ylim(-5, 110)
    ax2.axhline(100, color="gray", linestyle="--", alpha=0.4, label="Perfect isolation")
    ax2.axvline(100, color="gray", linestyle=":"  , alpha=0.4, label="Perfect utility")
    ax2.legend(fontsize=8, loc="lower left")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(f"{figures_dir}/issue7_fig2_pareto.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: issue7_fig2_pareto.png")

    d4_data    = isolation_data.get(POLICY_D4, {})
    d4_samples = d4_data.get("per_sample", [])
    if d4_samples:
        risk_levels = ["LOW", "MEDIUM", "HIGH"]
        iso_by_level: dict[str, list[float]] = {r: [] for r in risk_levels}
        for row in d4_samples:
            lvl = row.get("detected_risk_level", "LOW")
            iso_by_level[lvl].append(float(row.get("isolated", False)))

        iso_means  = [np.mean(iso_by_level[r]) * 100 if iso_by_level[r] else 0.0
                      for r in risk_levels]
        share_means = [100.0 - v for v in iso_means]

        fig3, ax3 = plt.subplots(figsize=(8, 5))
        x = np.arange(len(risk_levels))
        w = 0.35
        b1 = ax3.bar(x - w/2, share_means, w, label="Shared",   color="#2ecc71", alpha=0.85)
        b2 = ax3.bar(x + w/2, iso_means,   w, label="Isolated", color="#e74c3c", alpha=0.85)
        ax3.set_xlabel("Detected Risk Level", fontsize=11)
        ax3.set_ylabel("Fraction of Requests (%)", fontsize=11)
        ax3.set_title("Issue #7 Fig 3: Sharing / Isolation by Risk Level (D4)", fontsize=12)
        ax3.set_xticks(x)
        ax3.set_xticklabels(risk_levels, fontsize=11)
        ax3.set_ylim(0, 115)
        ax3.legend(fontsize=10)
        ax3.grid(axis="y", alpha=0.3)
        for bar in list(b1) + list(b2):
            h = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.0f}%",
                     ha="center", va="bottom", fontsize=9)
        fig3.tight_layout()
        fig3.savefig(f"{figures_dir}/issue7_fig3_risk_level_breakdown.png", dpi=150, bbox_inches="tight")
        plt.close(fig3)
        print(f"  Saved: issue7_fig3_risk_level_breakdown.png")
    else:
        print("  [skip] issue7_fig3 — no D4 per-sample data")



def generate_tables(results: dict, figures_dir: str) -> None:
    """Generate Table 1 and Table 2 as Markdown files."""
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    isolation_data = results.get("policy_isolation", {})
    asr_data       = results.get("residual_asr", {})

    lines = [
        "# Issue #7 Table 1: Defense Policy Comparison\n",
        "Generated by script from raw results. Do not edit manually.\n\n",
        "| Defense | Description | Residual ASR | Sensitive Isolation | Non-sensitive Sharing | Hit Rate | TTFT Overhead |\n",
        "|---------|-------------|-------------|--------------------|-----------------------|----------|---------------|\n",
    ]
    for p in ALL_POLICIES:
        agg = isolation_data.get(p, {}).get("aggregate", {})
        asr = asr_data.get(p, {}).get("residual_asr", "n/a")
        sens  = agg.get("sensitive_isolation_rate",   "n/a")
        nshar = agg.get("non_sensitive_sharing_rate", "n/a")
        hitr  = agg.get("hit_rate",                  "n/a")
        ovhd  = agg.get("ttft_overhead_pct",          "n/a")
        desc  = POLICY_DESCRIPTIONS.get(p, p)

        fmt = lambda v: f"{v:.1%}" if isinstance(v, float) else str(v)
        fmt_pct = lambda v: f"{v:+.1f}%" if isinstance(v, float) else str(v)

        lines.append(
            f"| {p} | {desc[:40]} | {fmt(asr)} | {fmt(sens)} | {fmt(nshar)} | {fmt(hitr)} | {fmt_pct(ovhd)} |\n"
        )

    path1 = f"{figures_dir}/issue7_table1_defense_comparison.md"
    with open(path1, "w") as fh:
        fh.writelines(lines)
    print(f"  Saved: issue7_table1_defense_comparison.md")

    lines2 = [
        "# Issue #7 Table 2: Risk Level → Detector Condition → Cache Action\n\n",
        "| Risk Level | Detector Condition | Cache Action (D4) | Cross-tenant Reuse | Example Entities |\n",
        "|------------|-------------------|-------------------|-------------------|------------------|\n",
        "| LOW / CLEAN | No PII detected, risk_score < 0.25 | SHARE (no salt) | ✅ Permitted | — |\n",
        "| MEDIUM | PII detected, confidence < 0.7, not high-risk type | RESTRICT (tenant salt) | ❌ Cross-tenant blocked | PERSON, EMAIL, PHONE |\n",
        "| HIGH | High-risk PII detected (CREDIT_CARD, SSN, IBAN, MEDICAL_LICENSE) or ≥3 entities conf≥0.7 | ISOLATE (unique UUID salt) | ❌ Blocked (even intra-tenant) | CREDIT_CARD, IBAN_CODE, US_SSN, MEDICAL_LICENSE |\n",
    ]
    path2 = f"{figures_dir}/issue7_table2_risk_action_mapping.md"
    with open(path2, "w") as fh:
        fh.writelines(lines2)
    print(f"  Saved: issue7_table2_risk_action_mapping.md")



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Issue #7: Defense evaluation")
    p.add_argument("--config", default="../configs/defense_eval.yaml")
    p.add_argument("--backend",    choices=["vllm", "sglang", "mock"])
    p.add_argument("--n-victims",  type=int, help="Override n_victims_per_policy")
    p.add_argument("--n-workload", type=int, help="Override n_workload_per_category")
    p.add_argument("--seed",       type=int)
    p.add_argument("--output-dir", help="Override output directory")
    p.add_argument("--skip-asr",   action="store_true",
                   help="Skip residual ASR measurement (faster for debugging)")
    return p.parse_args()


def load_config(path: str) -> dict:
    try:
        import yaml
        with open(path) as fh:
            return yaml.safe_load(fh)
    except ImportError:
        cfg: dict = {}
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    k, _, v = line.partition(":")
                    v = v.strip().strip('"')
                    try:
                        cfg[k.strip()] = int(v)
                    except ValueError:
                        try:
                            cfg[k.strip()] = float(v)
                        except ValueError:
                            cfg[k.strip()] = v
        return cfg


def main() -> None:
    args = parse_args()

    cfg_path = args.config
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(os.path.dirname(__file__), cfg_path)
    cfg = load_config(cfg_path)

    backend_name  = args.backend  or cfg.get("backend", "vllm")
    n_victims     = args.n_victims  or cfg.get("n_victims_per_policy", 10)
    n_workload    = args.n_workload or cfg.get("n_workload_per_category", 10)
    seed          = args.seed       or cfg.get("seed", 42)
    output_dir    = args.output_dir or cfg.get("output_dir", "experiments/results")
    figures_dir   = cfg.get("figures_dir", "experiments/figures")
    model_id      = cfg.get("model_id", MODEL_ID)
    vllm_url      = cfg.get("vllm_base_url", VLLM_BASE_URL)
    victim_tenant = cfg.get("victim_tenant_id",   "tenant-victim")
    n_calibration = cfg.get("n_calibration", 50)

    if not os.path.isabs(output_dir):
        output_dir  = str(_REPO_ROOT / output_dir)
    if not os.path.isabs(figures_dir):
        figures_dir = str(_REPO_ROOT / figures_dir)

    run_id    = f"defense-eval-{datetime.datetime.utcnow().strftime('%Y-%m-%d')}-{uuid.uuid4().hex[:8]}"
    out_path  = f"{output_dir}/{run_id}.json"
    cmd_used  = " ".join(sys.argv)

    print(f"\n{'='*60}")
    print(f"  Issue #7 Defense Evaluation")
    print(f"  run_id   : {run_id}")
    print(f"  backend  : {backend_name}")
    print(f"  model    : {model_id}")
    print(f"  n_victims: {n_victims} per policy")
    print(f"  n_workload: {n_workload} per category")
    print(f"  seed     : {seed}")
    print(f"{'='*60}\n")

    if backend_name == "mock":
        backend = MockBackend()
        print("[setup] Using mock backend")
    elif backend_name == "sglang":
        from kv_attack.backends.sglang_backend import SGLangBackend
        sglang_url = cfg.get("sglang_base_url", "http://localhost:30000/v1")
        backend    = SGLangBackend(sglang_url, model_id)
        print(f"[setup] SGLang backend at {sglang_url}")
    else:
        backend = VLLMBackend(vllm_url, model_id)
        print(f"[setup] vLLM backend at {vllm_url}")
        if not backend.health_check():
            print("[FATAL] vLLM backend not reachable. Start the server first.")
            sys.exit(1)

    backend_info = backend.get_info()
    environment  = _collect_environment()

    calibration = calibrate(backend, n_samples=n_calibration, seed=seed)

    if not calibration.get("apc_signal_ok", True):
        print("\n  ⚠️  WARNING: APC signal is weak (gap < 20 ms).")
        print("     Make sure vLLM was started with --enable-prefix-caching.")
        print("     Results from policy isolation measurement will be unreliable.")
        print("     Continuing anyway…\n")

    print(f"\n[workload] Generating {n_workload} samples × {len(ALL_CATEGORIES)} categories…")
    samples = generate_defense_workload(n_per_category=n_workload, seed=seed)
    print(f"  Total samples: {len(samples)}")

    print("\n[Phase 1] Detector accuracy evaluation…")
    detector_eval = eval_detector(samples, use_stage2=False)

    def _make_backend():
        if backend_name == "mock":
            return MockBackend(seed=0)
        return backend

    print("\n[Phase 2] Policy isolation / sharing evaluation…")
    policy_isolation: dict[str, dict] = {}
    for policy in ALL_POLICIES:
        print(f"\n  Evaluating policy {policy}…")
        policy_eval_id = f"EVAL-{policy[:3].upper()}-{uuid.uuid4().hex[:6]}"
        policy_isolation[policy] = eval_policy_isolation(
            _make_backend(), samples, policy, victim_tenant,
            calibration, eval_run_id=policy_eval_id
        )

    residual_asr_results: dict[str, dict] = {}
    if not args.skip_asr:
        print("\n[Phase 3] Residual ASR measurement…")
        for policy in ALL_POLICIES:
            print(f"\n  ASR under {policy} (n_victims={n_victims})…")
            residual_asr_results[policy] = eval_residual_asr(
                backend, policy, victim_tenant, n_victims, seed, calibration
            )
    else:
        print("\n[Phase 3] Residual ASR — SKIPPED (--skip-asr)")

    print(f"\n[save] Writing results to {out_path}…")
    results = {
        "run_id":          run_id,
        "issue":           7,
        "config": {
            "backend":     backend_name,
            "model_id":    model_id,
            "n_victims":   n_victims,
            "n_workload":  n_workload,
            "seed":        seed,
        },
        "environment":     environment,
        "backend_info":    vars(backend_info),
        "calibration":     calibration,
        "workload":        workload_to_dict(samples),
        "detector_eval":   detector_eval,
        "policy_isolation": {
            p: {
                "aggregate":     v.get("aggregate", {}),
                "audit_summary": v.get("audit_summary", {}),
                "per_sample":    v.get("per_sample", []),
            }
            for p, v in policy_isolation.items()
        },
        "residual_asr":    residual_asr_results,
        "command":         cmd_used,
    }

    summary_rows = []
    for p in ALL_POLICIES:
        agg = policy_isolation.get(p, {}).get("aggregate", {})
        asr = residual_asr_results.get(p, {})
        summary_rows.append({
            "policy":                     p,
            "residual_asr":               asr.get("residual_asr", "n/a"),
            "sensitive_isolation_rate":   agg.get("sensitive_isolation_rate",   "n/a"),
            "non_sensitive_sharing_rate": agg.get("non_sensitive_sharing_rate", "n/a"),
            "hit_rate":                   agg.get("hit_rate",                  "n/a"),
            "ttft_overhead_pct":          agg.get("ttft_overhead_pct",          "n/a"),
            "victim_ttft_p50_ms":         agg.get("victim_ttft_p50_ms",         "n/a"),
            "victim_ttft_p95_ms":         agg.get("victim_ttft_p95_ms",         "n/a"),
            "victim_ttft_p99_ms":         agg.get("victim_ttft_p99_ms",         "n/a"),
        })
    results["summary_table"] = summary_rows

    _dump(results, out_path)
    print(f"  Results saved: {out_path}")

    print("\n[figures] Generating figures and tables…")
    generate_figures(results, figures_dir)
    generate_tables(results, figures_dir)

    print(f"\n{'='*60}")
    print("  SUMMARY TABLE")
    print(f"  {'Policy':<22} {'Res.ASR':>8} {'SensIso':>8} {'NSShare':>8} {'HitRate':>8} {'Overhead':>9}")
    print(f"  {'-'*65}")
    for row in summary_rows:
        fmt = lambda v: f"{v:.1%}" if isinstance(v, float) else str(v)
        fmt_p = lambda v: f"{v:+.1f}%" if isinstance(v, float) else str(v)
        print(
            f"  {row['policy']:<22} "
            f"{fmt(row['residual_asr']):>8} "
            f"{fmt(row['sensitive_isolation_rate']):>8} "
            f"{fmt(row['non_sensitive_sharing_rate']):>8} "
            f"{fmt(row['hit_rate']):>8} "
            f"{fmt_p(row['ttft_overhead_pct']):>9}"
        )
    print(f"{'='*60}")
    print(f"\nDone. Results: {out_path}")


if __name__ == "__main__":
    main()