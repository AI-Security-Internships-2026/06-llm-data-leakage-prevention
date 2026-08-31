"""
kv_attack.bits_analysis
========================
Week 12 — Formal information-theoretic analysis of the KV-cache timing attack.

This module produces the analysis section of the arXiv draft (Section 4).
It can be run standalone against the Week 10 results JSON or the Week 12
multi-backend results JSON, or imported by other scripts.

Key quantities computed
-----------------------

Prior entropy H₀
    H₀ = log₂(|V|)  where V = name × condition vocabulary
    With |names| = 100, |conditions| = 20 → |V| = 2 000 → H₀ ≈ 10.97 bits

Posterior entropy H₁ (after a confirmed hit)
    H₁ = 0  (attacker has identified the exact (name, condition) pair)

Bits leaked per query BLQ
    BLQ = (H₀ - H₁) / Q = H₀ / Q  where Q = total API calls

Stage-level entropy reduction
    After Stage 1 (name confirmed, Q₁ queries):
        H₁ = log₂(|conditions|) = log₂(20) ≈ 4.32 bits  (condition still unknown)
        Bits reduced by Stage 1 = H₀ - H₁ = 10.97 - 4.32 = 6.65 bits
        BLQ₁ = 6.65 / Q₁

    After Stage 2 (condition confirmed, Q₂ additional queries):
        H₂ = 0
        Bits reduced by Stage 2 = H₁ = 4.32 bits
        BLQ₂ = 4.32 / Q₂

Query budget bounds
    Adaptive algorithm:
        Q_min = 1  (victim is the first candidate probed)
        Q_expected ≈ N_names/2 + N_conditions/2  (uniform random victim)
        Q_max = N_names + N_conditions = 120
    Linear scan (Week 10):
        Q_min = 1
        Q_expected = (N_names × N_conditions) / 2 = 1 000
        Q_max = N_names × N_conditions = 2 000

    Adaptive improvement at Q_expected: 1 000 / 70 ≈ 14.3×

Comparison table (for arXiv draft Table 2)
-------------------------------------------
| Algorithm      | Q_expected | BLQ_expected | BLQ_improvement |
|----------------|-----------|-------------|-----------------|
| Linear scan    | 1 000     | 0.011 b/q   | 1×              |
| Adaptive (W12) | ~70       | 0.157 b/q   | ~14×            |

Usage
-----
# Analyse Week 10 baseline results
python -m kv_attack.bits_analysis \\
    --input ../experiments/results/kv_attack_results.json \\
    --output ../experiments/results/kv_week12_bits_analysis.json

# Analyse Week 12 multi-backend results
python -m kv_attack.bits_analysis \\
    --input ../experiments/results/kv_week12_multibackend.json \\
    --output ../experiments/results/kv_week12_bits_analysis.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


# ── Vocabulary constants ──────────────────────────────────────────────────────

N_NAMES      = 100    # 10 first × 10 last
N_CONDITIONS = 20
VOCAB_SIZE   = N_NAMES * N_CONDITIONS   # 2 000

H_PRIOR      = math.log2(VOCAB_SIZE)                  # H₀ ≈ 10.965
H_NAME_KNOWN = math.log2(N_CONDITIONS)                 # log₂(20) ≈ 4.322
H_BOTH_KNOWN = 0.0                                     # H₂ = 0


# ── Analytical BLQ for the two algorithms ────────────────────────────────────

def analytical_linear_scan() -> dict:
    """
    Closed-form analysis of the Week 10 linear scan.

    Assumes the vocabulary is shuffled uniformly at random before scanning.
    The hit position is uniform in [1, |V|], so E[Q] = (|V| + 1) / 2.
    """
    q_min      = 1
    q_expected = (VOCAB_SIZE + 1) / 2   # ≈ 1000.5
    q_max      = VOCAB_SIZE             # 2 000

    blq_min      = H_PRIOR / q_min
    blq_expected = H_PRIOR / q_expected
    blq_max_case = H_PRIOR / q_max

    return {
        "algorithm"       : "linear_scan",
        "vocab_size"      : VOCAB_SIZE,
        "prior_entropy_bits": round(H_PRIOR, 4),
        "q_min"           : q_min,
        "q_expected"      : round(q_expected, 1),
        "q_max"           : q_max,
        "blq_best_case"   : round(blq_min,      6),
        "blq_expected"    : round(blq_expected,  6),
        "blq_worst_case"  : round(blq_max_case,  6),
        "note": (
            "Linear scan shuffles candidates. "
            "E[Q] = (|V|+1)/2 ≈ 1000 for |V|=2000. "
            "Week 10 empirical: Q̄ = 762 (5 victims) / 1303 (50 victims), "
            "consistent with uniform-shuffle expectation."
        ),
    }


def analytical_adaptive_two_stage() -> dict:
    """
    Closed-form analysis of the Week 12 adaptive two-stage algorithm.

    Stage 1: scan N_NAMES (100) candidates, each with one representative
             condition. Hit position is uniform in [1, N_NAMES].
             E[Q₁] = (N_NAMES + 1) / 2 ≈ 50.5  (without early exit)
             E[Q₁] ≈ 1  (with early exit on first hit, since Δ >> σ)

             In our regime (Δ=488 ms >> σ=4.7 ms), the first probed victim
             name always returns a decisive hit. The early-exit heuristic
             means Stage 1 terminates at probe k where k is the position of
             the victim in the shuffled name list. E[k] = (N_NAMES+1)/2 ≈ 50.5
             BUT with early exit the scan stops at the FIRST hit, not after
             scanning all names. So E[Q₁_with_early_exit] = 50.5 probes on
             average (the victim is equally likely to be at any position).
             Plus ≈ 12.5 reseed calls (every 4 probes) → E[Q₁] ≈ 63.

    Stage 2: scan N_CONDITIONS (20) for the surviving name.
             Hit is at position uniform in [1, N_CONDITIONS].
             E[Q₂] = (N_CONDITIONS + 1) / 2 ≈ 10.5
             Plus ≈ 2.6 reseed calls → E[Q₂] ≈ 13.

    Confirmation: N_REPEATS_CONFIRM = 3 calls.

    E[Q_total] ≈ 63 + 13 + 3 = 79 calls.
    Q_max = N_NAMES + N_CONDITIONS + N_REPEATS_CONFIRM = 123.

    BLQ_expected = H₀ / E[Q] ≈ 10.97 / 79 ≈ 0.139 bits/query.
    """
    # Stage 1 parameters
    reseed_every = 4
    q1_scan_expected  = (N_NAMES + 1) / 2            # avg hit position in Stage 1
    q1_reseed_expected= q1_scan_expected / reseed_every  # reseed calls
    q1_expected       = q1_scan_expected + q1_reseed_expected
    q1_max            = N_NAMES + N_NAMES // reseed_every

    bits_after_stage1 = H_NAME_KNOWN   # log₂(20) — name known, condition unknown

    # Stage 2 parameters
    q2_scan_expected  = (N_CONDITIONS + 1) / 2
    q2_reseed_expected= q2_scan_expected / reseed_every
    q2_expected       = q2_scan_expected + q2_reseed_expected
    q2_max            = N_CONDITIONS + N_CONDITIONS // reseed_every

    confirm_calls  = 3   # N_REPEATS_CONFIRM
    q_total_expected = q1_expected + q2_expected + confirm_calls
    q_total_max      = q1_max + q2_max + confirm_calls

    blq_stage1_expected   = (H_PRIOR - bits_after_stage1) / q1_expected
    blq_stage2_expected   = bits_after_stage1 / q2_expected
    blq_overall_expected  = H_PRIOR / q_total_expected
    blq_overall_best_case = H_PRIOR / (1 + 1 + confirm_calls)   # first probe in each stage hits

    return {
        "algorithm"            : "adaptive_two_stage",
        "vocab_size"           : VOCAB_SIZE,
        "prior_entropy_bits"   : round(H_PRIOR, 4),

        "stage1": {
            "probes_names"          : N_NAMES,
            "bits_reduced"          : round(H_PRIOR - bits_after_stage1, 4),
            "q_scan_expected"       : round(q1_scan_expected, 1),
            "q_reseed_expected"     : round(q1_reseed_expected, 1),
            "q_total_expected"      : round(q1_expected, 1),
            "q_max"                 : q1_max,
            "blq_stage1_expected"   : round(blq_stage1_expected, 6),
            "bits_remaining_after"  : round(bits_after_stage1, 4),
        },
        "stage2": {
            "probes_conditions"     : N_CONDITIONS,
            "bits_reduced"          : round(bits_after_stage1, 4),
            "q_scan_expected"       : round(q2_scan_expected, 1),
            "q_reseed_expected"     : round(q2_reseed_expected, 1),
            "q_total_expected"      : round(q2_expected, 1),
            "q_max"                 : q2_max,
            "blq_stage2_expected"   : round(blq_stage2_expected, 6),
            "bits_remaining_after"  : 0.0,
        },
        "confirmation_calls"       : confirm_calls,

        "q_total_expected"         : round(q_total_expected, 1),
        "q_total_max"              : q_total_max,
        "q_total_min"              : 1 + 1 + confirm_calls,

        "blq_best_case"            : round(blq_overall_best_case, 6),
        "blq_expected"             : round(blq_overall_expected, 6),
        "blq_worst_case"           : round(H_PRIOR / q_total_max, 6),
    }


def improvement_table(
    linear   : dict,
    adaptive : dict,
) -> dict:
    """
    Build the comparison table for arXiv draft Table 2.
    """
    q_lin = linear["q_expected"]
    q_adp = adaptive["q_total_expected"]
    q_improvement = round(q_lin / q_adp, 2)

    blq_lin = linear["blq_expected"]
    blq_adp = adaptive["blq_expected"]
    blq_improvement = round(blq_adp / blq_lin, 2)

    return {
        "table_caption": (
            "Table 2: Query efficiency of linear scan (Week 10) vs "
            "adaptive two-stage (Week 12). "
            "Vocabulary V = 2000 (100 names × 20 conditions), H₀ = 10.97 bits."
        ),
        "rows": [
            {
                "algorithm"      : "Linear scan (Week 10)",
                "q_expected"     : q_lin,
                "q_max"          : linear["q_max"],
                "blq_expected"   : blq_lin,
                "blq_best_case"  : linear["blq_best_case"],
                "blq_worst_case" : linear["blq_worst_case"],
            },
            {
                "algorithm"      : "Adaptive two-stage (Week 12)",
                "q_expected"     : q_adp,
                "q_max"          : adaptive["q_total_max"],
                "blq_expected"   : blq_adp,
                "blq_best_case"  : adaptive["blq_best_case"],
                "blq_worst_case" : adaptive["blq_worst_case"],
            },
        ],
        "improvement": {
            "query_reduction_factor" : q_improvement,
            "blq_improvement_factor" : blq_improvement,
            "note"                   : (
                f"Adaptive algorithm requires {q_improvement}× fewer queries "
                f"on average, yielding a {blq_improvement}× improvement in "
                f"bits leaked per query."
            ),
        },
    }


# ── Empirical BLQ from results JSON ──────────────────────────────────────────

def empirical_blq_from_results(results_path: str) -> dict:
    """
    Compute empirical BLQ from a results JSON file.

    Handles both Week 10 format (results[].total_api_calls)
    and Week 12 format (backends.*.results[].total_api_calls).
    """
    data = json.loads(Path(results_path).read_text())

    def _extract_calls(d: dict) -> list[int]:
        """Flatten all total_api_calls from any known results schema."""
        calls = []
        # Week 10 flat format
        if "results" in d and isinstance(d["results"], list):
            for r in d["results"]:
                if "total_api_calls" in r:
                    calls.append(r["total_api_calls"])
        # Week 12 multi-backend format
        if "backends" in d:
            for bdata in d["backends"].values():
                if isinstance(bdata, dict) and "results" in bdata:
                    for r in bdata["results"]:
                        if "total_api_calls" in r:
                            calls.append(r["total_api_calls"])
        return calls

    calls = _extract_calls(data)
    if not calls:
        return {"error": "No total_api_calls found in results file."}

    calls_arr = np.array(calls, dtype=float)
    blq_arr   = H_PRIOR / calls_arr

    # CDF at standard budget points
    budget_points = [50, 100, 120, 200, 500, 762, 1000, 1303, 2000]
    cdf = {
        str(q): round(float(np.mean(calls_arr <= q)), 4)
        for q in budget_points
    }

    return {
        "source_file"       : results_path,
        "n_victims"         : len(calls),
        "prior_entropy_bits": round(H_PRIOR, 4),
        "empirical_calls": {
            "mean"  : round(float(calls_arr.mean()),  2),
            "median": round(float(np.median(calls_arr)), 2),
            "std"   : round(float(calls_arr.std()),   2),
            "min"   : int(calls_arr.min()),
            "max"   : int(calls_arr.max()),
            "p25"   : round(float(np.percentile(calls_arr, 25)), 1),
            "p75"   : round(float(np.percentile(calls_arr, 75)), 1),
        },
        "empirical_blq": {
            "mean"  : round(float(blq_arr.mean()),   6),
            "median": round(float(np.median(blq_arr)), 6),
            "std"   : round(float(blq_arr.std()),    6),
            "min"   : round(float(blq_arr.min()),    6),
            "max"   : round(float(blq_arr.max()),    6),
        },
        "query_budget_cdf": cdf,
    }


# ── Main analysis runner ──────────────────────────────────────────────────────

def run_bits_analysis(
    input_path  : str,
    output_path : str,
) -> dict:
    """
    Full information-theoretic analysis.

    Combines:
      1. Analytical closed-form for both algorithms.
      2. Empirical BLQ from the provided results JSON.
      3. Improvement table for arXiv draft.
    """
    linear   = analytical_linear_scan()
    adaptive = analytical_adaptive_two_stage()
    table    = improvement_table(linear, adaptive)
    empirical= empirical_blq_from_results(input_path)

    output = {
        "analysis_title"         : "KV-Cache Timing Attack: Information-Theoretic Analysis (Week 12)",
        "analytical_linear_scan" : linear,
        "analytical_adaptive"    : adaptive,
        "improvement_table"      : table,
        "empirical"              : empirical,
        "arxiv_section4_summary" : {
            "H0_bits"               : round(H_PRIOR, 4),
            "H_after_stage1_bits"   : round(H_NAME_KNOWN, 4),
            "H_after_stage2_bits"   : 0.0,
            "linear_q_expected"     : linear["q_expected"],
            "adaptive_q_expected"   : adaptive["q_total_expected"],
            "linear_blq_expected"   : linear["blq_expected"],
            "adaptive_blq_expected" : adaptive["blq_expected"],
            "query_reduction"       : table["improvement"]["query_reduction_factor"],
            "blq_improvement"       : table["improvement"]["blq_improvement_factor"],
            "empirical_mean_calls"  : empirical.get("empirical_calls", {}).get("mean"),
            "empirical_mean_blq"    : empirical.get("empirical_blq", {}).get("mean"),
            "narrative": (
                f"The prior uncertainty about a victim's identity is "
                f"H₀ = log₂({VOCAB_SIZE}) ≈ {H_PRIOR:.2f} bits "
                f"({N_NAMES} names × {N_CONDITIONS} conditions). "
                f"The linear scan (Week 10) recovers this in an expected "
                f"{linear['q_expected']:.0f} queries, yielding "
                f"BLQ ≈ {linear['blq_expected']:.4f} bits/query. "
                f"The adaptive two-stage algorithm (Week 12) achieves the same "
                f"recovery in an expected {adaptive['q_total_expected']:.0f} queries "
                f"(worst-case {adaptive['q_total_max']}), yielding "
                f"BLQ ≈ {adaptive['blq_expected']:.4f} bits/query — a "
                f"{table['improvement']['blq_improvement_factor']}× improvement. "
                f"Stage 1 (name elimination) reduces uncertainty by "
                f"{H_PRIOR - H_NAME_KNOWN:.2f} bits at a cost of "
                f"~{adaptive['stage1']['q_total_expected']:.0f} queries; "
                f"Stage 2 (condition scan) accounts for the remaining "
                f"{H_NAME_KNOWN:.2f} bits in ~{adaptive['stage2']['q_total_expected']:.0f} queries."
            ),
        },
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"[bits_analysis] Results written → {out_path}")

    print("\n" + "=" * 65)
    print("[bits_analysis] INFORMATION-THEORETIC SUMMARY")
    print("=" * 65)
    print(f"  Prior entropy H₀        : {H_PRIOR:.4f} bits")
    print(f"  After Stage 1 H₁        : {H_NAME_KNOWN:.4f} bits (condition unknown)")
    print(f"  After Stage 2 H₂        : 0.0000 bits (fully recovered)")
    print(f"  Linear scan E[Q]        : {linear['q_expected']:.0f} queries")
    print(f"  Linear scan BLQ         : {linear['blq_expected']:.6f} bits/query")
    print(f"  Adaptive E[Q]           : {adaptive['q_total_expected']:.0f} queries")
    print(f"  Adaptive BLQ            : {adaptive['blq_expected']:.6f} bits/query")
    print(f"  Query reduction         : {table['improvement']['query_reduction_factor']}×")
    print(f"  BLQ improvement         : {table['improvement']['blq_improvement_factor']}×")
    if empirical.get("empirical_calls"):
        print(f"  Empirical mean Q        : {empirical['empirical_calls']['mean']:.1f} queries")
        print(f"  Empirical mean BLQ      : {empirical['empirical_blq']['mean']:.6f} bits/query")
    print("=" * 65)

    return output


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Week 12 information-theoretic analysis of KV-cache timing attack"
    )
    p.add_argument(
        "--input",  type=str,
        default="../experiments/results/kv_attack_results.json",
        help="Path to Week 10 or Week 12 results JSON",
    )
    p.add_argument(
        "--output", type=str,
        default="../experiments/results/kv_week12_bits_analysis.json",
    )
    args = p.parse_args()
    run_bits_analysis(args.input, args.output)


if __name__ == "__main__":
    main()