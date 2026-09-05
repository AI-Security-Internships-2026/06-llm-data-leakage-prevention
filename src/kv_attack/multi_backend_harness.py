"""
kv_attack.multi_backend_harness
================================
Week 12 — Multi-backend attack harness (v2, bugfix 2026-08-29).

Usage
-----
cd src
python -m kv_attack.multi_backend_harness \
    --backends vllm \
    --n-victims 5 \
    --model-id deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --output ../experiments/results/kv_week12_vllm_5vic.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import time
import uuid
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from kv_attack import MODEL_ID, detect_has_bos
from kv_attack.backends import get_backend, BackendClient
from kv_attack.backends.base import BackendInfo
from kv_attack.victim_seeder import build_private_block, build_aligned_system_prompt
from kv_attack.adaptive_reconstructor import (
    calibrate_threshold_backend,
    reconstruct_victim_adaptive,
    aggregate_information_theory,
    evict_cache_full,
    AdaptiveReconstructionResult,
    EVICT_N_REQUESTS,
)


# ── Victim seeding ────────────────────────────────────────────────────────────

def _seed_victims(
    backend       : BackendClient,
    tokenizer     : AutoTokenizer,
    system_prefix : str,
    n_victims     : int,
    seed          : int,
) -> list[dict]:
    import random
    from datetime import date, timedelta
    from kv_attack import FIRST_NAMES, LAST_NAMES, MEDICAL_CONDITIONS
    from kv_attack.victim_seeder import count_private_blocks

    rng = random.Random(seed)

    def _random_dob(r):
        start = date(1935, 1, 1)
        delta = (date(2006, 12, 31) - start).days
        return (start + timedelta(days=r.randint(0, delta))).isoformat()

    records = []
    for i in range(n_victims):
        name      = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        dob       = _random_dob(rng)
        condition = rng.choice(MEDICAL_CONDITIONS)
        private   = build_private_block(name, dob, condition)
        prompt    = system_prefix + " " + private
        n_priv    = count_private_blocks(tokenizer, system_prefix, name, dob, condition)

        try:
            backend.measure_ttft(prompt)     # seeds the KV cache
        except Exception as exc:
            print(f"[seed] victim {i} seed failed: {exc}")
            continue

        records.append({
            "victim_id"        : i,
            "prompt"           : prompt,
            "ground_truth"     : {"name": name, "dob": dob, "condition": condition},
            "n_private_blocks" : n_priv,
        })
        print(f"[seed] Victim {i}: name='{name}'  condition='{condition}'  "
              f"n_priv_blocks={n_priv}")

    return records


# ── Single-backend runner ─────────────────────────────────────────────────────

def _run_single_backend(
    backend_name  : str,
    backend       : BackendClient,
    tokenizer     : AutoTokenizer,
    n_victims     : int,
    seed          : int,
    n_calibration : int,
    model_id      : str,
) -> dict:
    print(f"\n{'=' * 65}")
    print(f"[harness] Backend: {backend_name.upper()}  model: {model_id}")
    print(f"{'=' * 65}")

    if not backend.health_check():
        return {"status": "UNREACHABLE",
                "error": f"Backend '{backend_name}' health check failed."}

    info: BackendInfo = backend.get_info()
    print(f"[harness] framework={info.framework} v{info.framework_ver}  "
          f"APC metric detected={info.apc_enabled}")

    system_prefix, n_prefix_tokens = build_aligned_system_prompt(
        tokenizer, has_bos=detect_has_bos(model_id)
    )
    print(f"[harness] system_prefix tokens={n_prefix_tokens}")

    # Seed victims
    print(f"\n[harness] Seeding {n_victims} victims ...")
    victim_records = _seed_victims(backend, tokenizer, system_prefix,
                                   n_victims=n_victims, seed=seed)
    if not victim_records:
        return {"status": "SEED_FAILED", "error": "No victims seeded."}

    # Miss prompt factory
    def miss_prompt_factory() -> str:
        uid = f"MISS{uuid.uuid4().hex[:16].upper()}"
        return system_prefix + " " + build_private_block(
            uid, "1900-01-01", "FAKE_CONDITION_XYZ"
        )

    # Calibration (uses victim 0's cached prompt as the known hit)
    print(f"\n[harness] Calibrating ({n_calibration} samples each) ...")
    try:
        calibration = calibrate_threshold_backend(
            backend             = backend,
            known_cached_prompt = victim_records[0]["prompt"],
            miss_prompt_factory = miss_prompt_factory,
            n_samples           = n_calibration,
        )
    except RuntimeError as exc:
        return {"status": "CALIBRATION_FAILED", "error": str(exc)}

    threshold_ms = calibration["threshold_ms"]
    print(f"[harness] threshold={threshold_ms:.1f} ms  "
          f"delta={calibration['delta_ms']:.1f} ms  "
          f"p={calibration['ks_p_value']:.2e}")

    # Attack each victim
    print(f"\n[harness] Attacking {len(victim_records)} victims ...")
    results: list[AdaptiveReconstructionResult] = []
    t_start = time.perf_counter()

    for i, record in enumerate(victim_records):
        gt = record["ground_truth"]
        print(f"\n[harness] ── Victim {i + 1}/{len(victim_records)} ──")
        print(f"[harness]    GT: name='{gt['name']}'  condition='{gt['condition']}'")

        # Full cache eviction — cycles 1,500,000 tokens through the 719,008-token cache
        print(f"[harness]    Evicting cache (victim-structured, {EVICT_N_REQUESTS} prompts) ...")
        evict_calls = evict_cache_full(backend, system_prefix)
        print(f"[harness]    Eviction done ({evict_calls} calls). Re-seeding victim ...")

        # Re-seed this victim after eviction
        try:
            backend.measure_ttft(record["prompt"])
        except Exception as exc:
            print(f"[harness]    WARNING: reseed failed: {exc}")
            continue

        # candidate_seed must incorporate the run seed so that two independent
        # runs (different models, same victim_id) produce different shuffle orders.
        # Using seed * 1000 + i avoids collisions for any reasonable victim count.
        result = reconstruct_victim_adaptive(
            backend        = backend,
            tokenizer      = tokenizer,
            system_prefix  = system_prefix,
            threshold_ms   = threshold_ms,
            victim_record  = record,
            known_dob      = True,
            candidate_seed = seed * 1000 + i,
        )
        results.append(result)

        status = "✓ EXACT" if result.exact_match else \
                 f"TRR={result.token_recovery_rate:.2f}"
        blq = result.information_theory.get("bits_leaked_per_query", 0)
        print(f"[harness]    → {status}  calls={result.total_api_calls}  "
              f"ARPT={result.arpt}  BLQ={blq:.5f} bits/query")

    total_time = time.perf_counter() - t_start

    if not results:
        return {"status": "ATTACK_FAILED", "error": "No results produced."}

    it_summary = aggregate_information_theory(results)
    aggregate  = _build_aggregate(results, total_time, calibration)

    print(f"\n[harness] {backend_name.upper()} done: "
          f"SR={aggregate['success_rate']:.4f}  "
          f"mean_calls={aggregate['mean_total_api_calls']:.0f}  "
          f"mean_BLQ={it_summary['mean_blq']:.5f}")

    return {
        "status"             : "OK",
        "backend_info"       : {
            "backend_name"  : info.backend_name,
            "framework"     : info.framework,
            "framework_ver" : info.framework_ver,
            "model_id"      : model_id,
            "base_url"      : info.base_url,
            "apc_enabled"   : info.apc_enabled,
        },
        "calibration"        : calibration,
        "results"            : [_result_to_dict(r) for r in results],
        "aggregate"          : aggregate,
        "information_theory" : it_summary,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _result_to_dict(r: AdaptiveReconstructionResult) -> dict:
    return {
        "victim_id"          : r.victim_id,
        "ground_truth"       : r.ground_truth,
        "recovered"          : r.recovered,
        "token_recovery_rate": round(r.token_recovery_rate, 4),
        "exact_match"        : r.exact_match,
        "confirmed_hit"      : r.confirmed_hit,
        "total_api_calls"    : r.total_api_calls,
        "arpt"               : r.arpt,
        "n_private_blocks"   : r.n_private_blocks,
        "information_theory" : r.information_theory,
        "top_scan_results"   : r.scan_results,
        "algorithm"          : r.algorithm,
    }


def _build_aggregate(
    results    : list[AdaptiveReconstructionResult],
    total_time : float,
    calibration: dict,
) -> dict:
    trr   = [r.token_recovery_rate for r in results]
    sr    = [int(r.exact_match)     for r in results]
    arpt  = [r.arpt                 for r in results]
    calls = [r.total_api_calls      for r in results]

    rng   = np.random.default_rng(42)
    boots = [float(np.mean(rng.choice(arpt, size=len(arpt), replace=True)))
             for _ in range(10_000)]
    arpt_ci = (round(float(np.percentile(boots, 2.5)), 2),
               round(float(np.percentile(boots, 97.5)), 2))

    return {
        "algorithm"            : "linear_early_exit",
        "n_victims"            : len(results),
        "mean_trr"             : round(float(np.mean(trr)), 4),
        "success_rate"         : round(float(np.mean(sr)),  4),
        "confirmed_hit_rate"   : round(float(np.mean([int(r.confirmed_hit) for r in results])), 4),
        "mean_arpt"            : round(float(np.mean(arpt)), 2),
        "arpt_ci_95"           : list(arpt_ci),
        "mean_total_api_calls" : round(float(np.mean(calls)), 1),
        "mean_timing_gap_ms"   : round(calibration["delta_ms"], 2),
        "ks_p_value"           : calibration["ks_p_value"],
        "total_attack_time_s"  : round(total_time, 1),
        "target_trr"           : 0.85,
        "target_sr"            : 0.80,
        "trr_target_met"       : float(np.mean(trr)) >= 0.85,
        "sr_target_met"        : float(np.mean(sr))  >= 0.80,
    }


def _cross_backend_comparison(backend_results: dict) -> dict:
    gaps  = {}; srs = {}; blqs = {}; vuln = {}
    for bname, bdata in backend_results.items():
        if bdata.get("status") != "OK":
            gaps[bname] = srs[bname] = blqs[bname] = None
            vuln[bname] = False
            continue
        cal = bdata.get("calibration", {})
        agg = bdata.get("aggregate",   {})
        it  = bdata.get("information_theory", {})
        gap = cal.get("delta_ms", 0.0)
        sr  = agg.get("success_rate", 0.0)
        gaps[bname]  = round(gap, 2)
        srs[bname]   = round(sr, 4)
        blqs[bname]  = round(it.get("mean_blq", 0.0), 6)
        vuln[bname]  = (gap > 10.0 and sr >= 0.80)

    return {
        "timing_gaps_ms" : gaps,
        "success_rates"  : srs,
        "mean_blq"       : blqs,
        "vulnerable"     : vuln,
        "generalises"    : all(v for v in vuln.values()),
    }


# ── Master harness ────────────────────────────────────────────────────────────

def run_multi_backend(
    backend_names : list[str],
    n_victims     : int  = 5,
    output_path   : str  = "experiments/results/kv_week12_multibackend.json",
    seed          : int  = 42,
    n_calibration : int  = 200,
    vllm_url      : str  = "http://localhost:8001/v1",
    tgi_url       : str  = "http://localhost:8002",
    model_id      : str  = MODEL_ID,
) -> dict:
    run_id    = f"week12-multibackend-{datetime.date.today().isoformat()}"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print("\n" + "=" * 65)
    print(f"[harness] KV-Cache Timing Attack — Week 12 (v2 bugfix)")
    print(f"[harness] Run ID   : {run_id}")
    print(f"[harness] Model    : {model_id}")
    print(f"[harness] Backends : {backend_names}")
    print(f"[harness] Victims  : {n_victims}  |  Seed: {seed}")
    print(f"[harness] Eviction : {EVICT_N_REQUESTS} victim-structured prompts × 178 blocks = "
          f"{EVICT_N_REQUESTS * 178:,} blocks (1.95× cache, same subtree as victim data)")
    print("=" * 65)

    backend_instances: dict[str, BackendClient] = {}
    for name in backend_names:
        if name == "vllm":
            backend_instances[name] = get_backend(
                "vllm", base_url=vllm_url, model_id=model_id)
        elif name == "tgi":
            backend_instances[name] = get_backend(
                "tgi", base_url=tgi_url, model_id=model_id)
        elif name == "mock":
            backend_instances[name] = get_backend("mock", seed=seed)
        else:
            print(f"[harness] WARNING: unknown backend '{name}' — skipped.")

    backend_results: dict[str, dict] = {}
    for bname, binstance in backend_instances.items():
        backend_results[bname] = _run_single_backend(
            backend_name  = bname,
            backend       = binstance,
            tokenizer     = tokenizer,
            n_victims     = n_victims,
            seed          = seed,
            n_calibration = n_calibration,
            model_id      = model_id,
        )

    comparison = _cross_backend_comparison(backend_results)

    all_it = {
        bname: bdata.get("information_theory", {})
        for bname, bdata in backend_results.items()
        if bdata.get("status") == "OK"
    }

    output = {
        "run_id"                       : run_id,
        "model"                        : model_id,
        "hardware"                     : "NVIDIA GB10 (119.7 GB unified memory, Blackwell)",
        "scenario"                     : "S2",
        "algorithm"                    : "linear_early_exit",
        "vllm_version"                 : "0.27.1",
        "n_victims_per_backend"        : n_victims,
        "n_calibration_samples"        : n_calibration,
        "eviction_blocks_per_victim"   : EVICT_N_REQUESTS * 178,
        "backends"                     : backend_results,
        "cross_backend_comparison"     : comparison,
        "information_theory_per_backend": all_it,
        "bugfix_notes": {
            "v1_bug1": "Stage 1 representative-condition probe only hits when victim condition = MEDICAL_CONDITIONS[0]. Fixed: full linear scan.",
            "v1_bug2": f"Random eviction prompts lived in a different APC subtree from victim data — vLLM LRU only evicts within the same subtree. Fixed: victim-structured eviction prompts (system_prefix + EVICT<uuid>), {EVICT_N_REQUESTS} × 178 blocks = {EVICT_N_REQUESTS * 178:,} victim-subtree blocks (1.95× cache capacity).",
        },
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n[harness] Results → {out_path}")

    print("\n" + "=" * 65)
    print("[harness] WEEK 12 SUMMARY")
    print("=" * 65)
    for bname, bdata in backend_results.items():
        if bdata.get("status") == "OK":
            agg = bdata["aggregate"]
            it  = bdata["information_theory"]
            print(f"  {bname.upper()}: "
                  f"SR={agg['success_rate']:.4f}  "
                  f"TRR={agg['mean_trr']:.4f}  "
                  f"calls={agg['mean_total_api_calls']:.0f}  "
                  f"BLQ={it['mean_blq']:.5f} bits/q  "
                  f"Δ={agg['mean_timing_gap_ms']:.0f} ms")
        else:
            print(f"  {bname.upper()}: {bdata['status']} — {bdata.get('error','')}")
    print(f"  Generalises: {comparison['generalises']}")
    print("=" * 65)

    return output


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="KV-cache timing attack — Week 12 multi-backend harness"
    )
    p.add_argument("--backends",      nargs="+", default=["vllm"],
                   choices=["vllm", "tgi", "mock"])
    p.add_argument("--n-victims",     type=int, default=5)
    p.add_argument("--output",        type=str,
                   default="../experiments/results/kv_week12_multibackend.json")
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--n-calibration", type=int, default=200)
    p.add_argument("--vllm-url",      type=str, default="http://localhost:8001/v1")
    p.add_argument("--tgi-url",       type=str, default="http://localhost:8002")
    p.add_argument("--model-id",      type=str, default=MODEL_ID)
    args = p.parse_args()

    run_multi_backend(
        backend_names = args.backends,
        n_victims     = args.n_victims,
        output_path   = args.output,
        seed          = args.seed,
        n_calibration = args.n_calibration,
        vllm_url      = args.vllm_url,
        tgi_url       = args.tgi_url,
        model_id      = args.model_id,
    )


if __name__ == "__main__":
    main()