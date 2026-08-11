

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .attacker import (
    CacheHitOracle,
    CalibrationResult,
    MockVLLMClient,
    calibrate_threshold,
    DEFAULT_N_REPEATS,
    DEFAULT_N_SAMPLES,
)
from .cache_eviction import evict_cache
from .reconstructor import (
    VictimReconstructionResult,
    compute_aggregate,
    reconstruct_victim_s2,
)
from .victim_seeder import (
    VictimRecord,
    generate_victim_records,
    seed_single_victim,
    seed_victims,
)

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────

DEFAULT_MODEL       = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_OUTPUT      = "experiments/results/kv_attack_results.json"
DEFAULT_N_VICTIMS   = 50
DEFAULT_N_EVICT     = 20
DEFAULT_SEED        = 42


# ── Serialisation helpers ──────────────────────────────────────────────────────

def _calibration_to_dict(cal: CalibrationResult) -> dict:
    return dataclasses.asdict(cal)


def _field_result_to_dict(fr) -> dict:
    return dataclasses.asdict(fr)


def _victim_result_to_dict(vr: VictimReconstructionResult) -> dict:
    return {
        "victim_id":           vr.victim_id,
        "ground_truth_fields": vr.ground_truth_fields,
        "field_results":       [_field_result_to_dict(f) for f in vr.field_results],
        "token_recovery_rate": round(vr.token_recovery_rate, 4),
        "exact_match":         vr.exact_match,
        "total_queries":       vr.total_queries,
        "arpt":                round(vr.arpt, 2),
        "wall_time_s":         round(vr.wall_time_s, 2),
    }


def _aggregate_to_dict(agg) -> dict:
    return {
        "n_victims":           agg.n_victims,
        "mean_trr":            round(agg.mean_trr, 4),
        "success_rate":        round(agg.success_rate, 4),
        "mean_arpt":           round(agg.mean_arpt, 2),
        "arpt_ci_95":          [round(x, 2) for x in agg.arpt_ci_95],
        "mean_timing_gap_ms":  round(agg.mean_timing_gap_ms, 2),
        "mean_wall_time_s":    round(agg.mean_wall_time_s, 2),
    }


# ── Main orchestration ─────────────────────────────────────────────────────────

def run_attack(
    client,
    model:      str,
    n_victims:  int  = DEFAULT_N_VICTIMS,
    n_evict:    int  = DEFAULT_N_EVICT,
    n_repeats:  int  = DEFAULT_N_REPEATS,
    n_cal:      int  = DEFAULT_N_SAMPLES,
    seed:       int  = DEFAULT_SEED,
    cache_salt: bool = False,
) -> tuple[CalibrationResult, list[VictimReconstructionResult]]:
    """
    Full attack pipeline:
      1. Seed victim KV-cache entries
      2. Calibrate hit/miss threshold (KS-test)
      3. For each victim: evict → reconstruct → collect results

    Parameters
    ----------
    client      : OpenAI-compatible client (real or MockVLLMClient)
    model       : vLLM model ID
    n_victims   : number of synthetic victims to attack
    n_evict     : eviction requests between victims
    n_repeats   : TTFT measurement repetitions per probe
    n_cal       : calibration samples per distribution
    seed        : RNG seed (passed to victim_seeder)
    cache_salt  : informational flag — not enforced here; log a warning if True

    Returns
    -------
    (CalibrationResult, list[VictimReconstructionResult])
    """
    if cache_salt:
        logger.warning(
            "cache_salt=True detected — this run targets a protected deployment. "
            "Timing gap will likely not be significant. "
            "This flag is informational only; the harness does not enforce it."
        )

    # ── Phase A: Generate victim records (pure Faker, no vLLM calls yet) ───────
    logger.info("=" * 60)
    logger.info("PHASE A — Generating synthetic victim ground-truth records")
    logger.info("=" * 60)
    all_records: list[VictimRecord] = generate_victim_records(
        n_victims=n_victims,
        seed=seed,
    )
    if not all_records:
        raise RuntimeError("No victim records generated.")

    # ── Phase B: Seed canary → calibrate threshold ────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE B — Calibrating hit/miss TTFT threshold")
    logger.info("=" * 60)
    # Seed victim 0 as the calibration canary (not counted in results).
    canary = all_records[0]
    ok = seed_single_victim(client, model, canary)
    if not ok:
        raise RuntimeError("Canary seeding failed — check vLLM is running.")

    calibration_result = calibrate_threshold(
        client=client,
        cached_prompt=canary.prompt,
        model=model,
        n_samples=n_cal,
    )

    oracle = CacheHitOracle(
        client=client,
        model=model,
        threshold_ms=calibration_result.threshold_ms,
        n_repeats=n_repeats,
    )
    logger.info(
        "Threshold: %.1f ms | gap: %.1f ms | KS p=%.2e",
        calibration_result.threshold_ms,
        calibration_result.timing_gap_ms,
        calibration_result.ks_p_value,
    )

    # ── Phase C: Per-victim evict → seed → reconstruct ─────────────────────────
    # Each cycle:
    #   1. Evict: send n_evict unique noise requests → LRU evicts previous victim
    #   2. Seed:  plant THIS victim's prompt in the now-clean cache
    #   3. Reconstruct: probe candidate tokens, measure TTFT, recover fields
    logger.info("=" * 60)
    logger.info("PHASE C — Per-victim: evict → seed → reconstruct")
    logger.info("=" * 60)
    all_results: list[VictimReconstructionResult] = []

    victims_to_attack = all_records[1:]   # skip canary

    for idx, victim in enumerate(victims_to_attack):
        logger.info(
            "─── Victim %d/%d (id=%d) ───────────────────────────",
            idx + 1, len(victims_to_attack), victim.victim_id,
        )

        # 1. Evict previous cache entries
        evict_cache(client=client, model=model, n_requests=n_evict)

        # 2. Seed this victim fresh
        if not seed_single_victim(client, model, victim):
            logger.warning("Skipping victim %d (seeding failed).", victim.victim_id)
            continue

        # 3. Reconstruct
        result = reconstruct_victim_s2(
            oracle=oracle,
            victim_id=victim.victim_id,
            ground_truth=victim.private_fields,
        )
        all_results.append(result)

    return calibration_result, all_results


# ── Results writer ─────────────────────────────────────────────────────────────

def write_results(
    output_path:   Path,
    framework:     str,
    model:         str,
    apc_enabled:   bool,
    cache_salt:    bool,
    scenario:      str,
    n_victims:     int,
    calibration:   CalibrationResult,
    results:       list[VictimReconstructionResult],
    mode:          str,
) -> None:
    """Write attack results to the canonical JSON schema (Proposal §4.6)."""
    aggregate = compute_aggregate(results)

    output = {
        "run_id":       f"week10-{mode}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "framework":    framework,
        "model":        model,
        "apc_enabled":  apc_enabled,
        "cache_salt":   cache_salt,
        "scenario":     scenario,
        "n_victims":    n_victims,
        "calibration":  _calibration_to_dict(calibration),
        "results":      [_victim_result_to_dict(r) for r in results],
        "aggregate":    _aggregate_to_dict(aggregate),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("Results written → %s", output_path)
    _print_summary(aggregate, calibration)


def _print_summary(agg, cal: CalibrationResult) -> None:
    """Pretty-print a summary table to stdout."""
    print("\n" + "=" * 60)
    print("  Week 10 Attack Results — Summary")
    print("=" * 60)
    print(f"  Victims attacked      : {agg.n_victims}")
    print(f"  Token Recovery Rate   : {agg.mean_trr:.4f}  (target ≥ 0.85)")
    print(f"  Exact Match Rate (SR) : {agg.success_rate:.4f}  (target ≥ 0.80)")
    print(f"  Mean ARPT             : {agg.mean_arpt:.1f} queries/field")
    print(f"  ARPT 95% CI           : [{agg.arpt_ci_95[0]:.1f}, {agg.arpt_ci_95[1]:.1f}]")
    print(f"  Mean Timing Gap       : {agg.mean_timing_gap_ms:.1f} ms")
    print(f"  Mean Wall Time        : {agg.mean_wall_time_s:.1f} s/victim")
    print("-" * 60)
    print(f"  Calibration gap       : {cal.timing_gap_ms:.1f} ms")
    print(f"  KS p-value            : {cal.ks_p_value:.2e}  (threshold 1e-8)")
    print(f"  Calibration passed    : {cal.passed}")
    print("=" * 60 + "\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="KV-Cache Timing Side-Channel Attack Harness (Week 10)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mode", choices=["real", "mock"], default="mock",
        help="'real' uses a live vLLM instance; 'mock' simulates timing locally (no GPU needed).",
    )
    p.add_argument("--host",       default="localhost",        help="vLLM API host (real mode only).")
    p.add_argument("--port",       type=int, default=8001,     help="vLLM API port (real mode only).")
    p.add_argument("--model",      default=DEFAULT_MODEL,      help="vLLM model ID.")
    p.add_argument("--n-victims",  type=int, default=DEFAULT_N_VICTIMS, help="Number of synthetic victims.")
    p.add_argument("--n-evict",    type=int, default=DEFAULT_N_EVICT,   help="Eviction requests between victims.")
    p.add_argument("--n-repeats",  type=int, default=DEFAULT_N_REPEATS, help="TTFT repetitions per probe.")
    p.add_argument("--n-cal",      type=int, default=DEFAULT_N_SAMPLES, help="Calibration samples per distribution.")
    p.add_argument("--seed",       type=int, default=DEFAULT_SEED,      help="RNG seed.")
    p.add_argument("--cache-salt", action="store_true",        help="Flag: target deployment has cache_salt (protected).")
    p.add_argument("--output",     default=DEFAULT_OUTPUT,     help="Path to write results JSON.")
    p.add_argument("--verbose",    action="store_true",        help="Enable DEBUG logging.")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Build client ───────────────────────────────────────────────────────────
    if args.mode == "real":
        try:
            from openai import OpenAI
        except ImportError:
            logger.error("openai package not installed. Run: pip install openai>=1.0")
            return 1

        base_url = f"http://{args.host}:{args.port}/v1"
        client   = OpenAI(base_url=base_url, api_key="EMPTY")
        framework = "vllm"
        logger.info("Real mode: connecting to vLLM at %s", base_url)

    else:
        client    = MockVLLMClient(seed=args.seed)
        framework = "mock"
        logger.info("Mock mode: simulating vLLM timing (no GPU required).")

    # ── Run attack ─────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        calibration, results = run_attack(
            client=client,
            model=args.model,
            n_victims=args.n_victims,
            n_evict=args.n_evict,
            n_repeats=args.n_repeats,
            n_cal=args.n_cal,
            seed=args.seed,
            cache_salt=args.cache_salt,
        )
    except AssertionError as exc:
        logger.error("Calibration failed: %s", exc)
        return 2
    except RuntimeError as exc:
        logger.error("Attack failed: %s", exc)
        return 3

    total_time = time.perf_counter() - t0
    logger.info("Total runtime: %.1f s", total_time)

    # ── Write results ──────────────────────────────────────────────────────────
    write_results(
        output_path=Path(args.output),
        framework=framework,
        model=args.model,
        apc_enabled=True,
        cache_salt=args.cache_salt,
        scenario="S2",
        n_victims=len(results),
        calibration=calibration,
        results=results,
        mode=args.mode,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
