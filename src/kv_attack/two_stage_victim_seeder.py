"""
kv_attack.two_stage_victim_seeder
==================================
Week 13 — Redesigned prompt template for the true two-stage adaptive attack.

PROBLEM WITH WEEK 12 TEMPLATE
-------------------------------
Week 10/12 template:
    [BOS | system(271 tok) | {name}. {condition}. filler×14 | END]

Both name AND condition live in the SAME first private KV block (block N).
This prevents Stage 1 (name elimination) from working: probing
    system + name_candidate + FIXED_CONDITION
only produces a cache hit when the victim's condition ALSO matches
FIXED_CONDITION (1/20 probability). For 19/20 victims Stage 1 fails.

WEEK 13 FIX — Split into two independent block regions
--------------------------------------------------------
New template:
    Block N          … N+127  (128 blocks = 2048 tokens): name-only content
    Block N+128      … N+191  ( 64 blocks = 1024 tokens): condition + filler

Stage 1 probes system + name_block(cand_name) + dummy_condition_block
    → If cand_name == victim_name:
          blocks N … N+127 hit  → TTFT drops proportionally
          (128 blocks out of 192 → TTFT_S1_HIT ≈ miss - 128/192 × delta)
      Else: block N misses → hash-chain breaks → all 192 blocks miss

Stage 2 probes system + name_block(confirmed_name) + cond_block(cand_condition)
    → If cand_condition == victim_condition:
          all 192 blocks hit → full TTFT_HIT
      Else: blocks N+128… miss → TTFT stays at TTFT_S1_HIT level

THRESHOLD DERIVATION (from Week 10/12 empirical values)
---------------------------------------------------------
  hit_ttft  =  90.2 ms  (192/192 blocks hit)
  miss_ttft = 613.6 ms  (  0/192 blocks hit)
  delta     = 523.4 ms

  TTFT for right-name, wrong-condition  (128/192 blocks hit):
      T_S1_HIT = miss_ttft - (128/192) × delta = 613.6 - 349.6 = 264.0 ms

  Stage 1 threshold T1 = midpoint(miss_ttft, T_S1_HIT)
      T1 = (613.6 + 264.0) / 2 = 438.8 ms
      → anything below T1 → name confirmed

  Stage 2 threshold T2 = midpoint(T_S1_HIT, hit_ttft)
      T2 = (264.0 + 90.2) / 2 = 177.1 ms
      → anything below T2 → condition confirmed

EXPECTED QUERY COUNTS
----------------------
  Stage 1 (100 names): E[Q_s1] = (100+1)/2 = 50.5 scans
                      + 12.6 reseeds = 63.1 total
  Stage 2 ( 20 conditions): E[Q_s2] = (20+1)/2 = 10.5 scans
                           +  2.6 reseeds = 13.1 total
  Confirmation: 3 calls
  Total E[Q] = 79.2  (vs 1000.5 for linear scan)  → 12.63× improvement
"""

from __future__ import annotations

import random
from datetime import date, timedelta

from openai import OpenAI
from transformers import AutoTokenizer

from kv_attack import (
    MODEL_ID, BLOCK_SIZE,
    MEDICAL_CONDITIONS, FIRST_NAMES, LAST_NAMES,
)
from kv_attack.victim_seeder import (
    _SYSTEM_PREFIX_RAW,
    _PAD_WORD,
    _random_dob,
    build_aligned_system_prompt,
)


# ── Name-block filler (makes name occupy exactly NAME_BLOCKS complete blocks) ─
#
# NAME_BLOCKS × BLOCK_SIZE = 128 × 16 = 2048 tokens for the name region.
# After the 4-6 token name itself, we need ~2042 tokens of padding.
# This filler is IDENTICAL for all name candidates — only the first few tokens
# (the name) differ, which is exactly what we need for Stage 1 sensitivity.
#
NAME_BLOCKS = 128    # 128 × 16 = 2048 tokens for name block
COND_BLOCKS = 64     # 64  × 16 = 1024 tokens for condition block

_NAME_FILLER = (
    "Patient name field. Identity confirmed by attending physician. "
    "Biometric verification completed and logged in the hospital identity "
    "management system in accordance with Joint Commission standards for "
    "patient identification and safety protocol compliance documentation. "
    "Name verified against government-issued photo identification at time "
    "of admission. All aliases and previous names recorded in administrative "
    "system. Legal name used for all billing and insurance correspondence. "
    "Next-of-kin records updated and cross-referenced with emergency contact "
    "database. Patient consented to use of legal name in all clinical records "
    "and communications with treating team per hospital privacy policy. "
)

_CONDITION_FILLER = (
    "Diagnosis confirmed by attending physician following comprehensive "
    "clinical assessment. ICD-10 code assigned and recorded in the electronic "
    "health record. All relevant diagnostic criteria documented with supporting "
    "laboratory values, imaging findings, and specialist consultation notes. "
    "Treatment plan reviewed and updated at each clinical encounter. Medication "
    "reconciliation completed. Patient education provided. Follow-up scheduled. "
    "Care coordination with primary care physician initiated per hospital "
    "discharge planning protocol and multidisciplinary team recommendations. "
)

# Dummy fixed condition used in Stage 1 probes (value doesn't matter —
# only the name block needs to match for Stage 1 to detect a cache hit).
_DUMMY_CONDITION = "diabetes"

# Analytical TTFT thresholds (derived from Week 10/12 empirical measurements)
T1_THRESHOLD_MS = 438.8   # Stage 1: name confirmed if TTFT < T1
T2_THRESHOLD_MS = 177.1   # Stage 2: condition confirmed if TTFT < T2

# Intermediate TTFT expected when name matches but condition does not
T_S1_HIT_MS = 264.0


# ── Template builders ─────────────────────────────────────────────────────────

def build_name_block(name: str, tokenizer: AutoTokenizer) -> str:
    """
    Build the name block string: name + filler, padded to exactly
    NAME_BLOCKS × BLOCK_SIZE tokens.

    The padding ensures that block N+128 starts at a clean boundary,
    making Stage 2 (condition scan) independent of the name length.
    """
    base = f"{name}. " + _NAME_FILLER

    # Iteratively pad until we reach exactly NAME_BLOCKS full blocks
    # (or as close as possible within 1 token due to BPE rounding).
    for _ in range(BLOCK_SIZE * 4):
        toks = tokenizer.encode(base, add_special_tokens=False)
        n_complete = len(toks) // BLOCK_SIZE
        if n_complete >= NAME_BLOCKS:
            break
        base = base + _PAD_WORD

    return base


def build_condition_block(condition: str, tokenizer: AutoTokenizer) -> str:
    """
    Build the condition block string: condition + filler, padded to exactly
    COND_BLOCKS × BLOCK_SIZE tokens.
    """
    base = f"{condition}. " + _CONDITION_FILLER

    for _ in range(BLOCK_SIZE * 4):
        toks = tokenizer.encode(base, add_special_tokens=False)
        n_complete = len(toks) // BLOCK_SIZE
        if n_complete >= COND_BLOCKS:
            break
        base = base + _PAD_WORD

    return base


def build_two_stage_prompt(
    system_prefix   : str,
    name            : str,
    dob             : str,
    condition       : str,
    tokenizer       : AutoTokenizer,
    *,
    use_dummy_cond  : bool = False,
) -> str:
    """
    Assemble the full two-stage prompt.

    If use_dummy_cond=True, substitutes _DUMMY_CONDITION for the condition
    block — used during Stage 1 name probing.
    """
    name_block = build_name_block(name, tokenizer)
    cond_str   = _DUMMY_CONDITION if use_dummy_cond else condition
    cond_block = build_condition_block(cond_str, tokenizer)
    return f"{system_prefix} {name_block} {cond_block} END OF PATIENT RECORD."


def count_two_stage_blocks(
    system_prefix : str,
    name          : str,
    dob           : str,
    condition     : str,
    tokenizer     : AutoTokenizer,
    has_bos       : bool = True,
) -> dict:
    """
    Return block counts for the two-stage template layout.
    Useful for smoke-testing the template structure.
    """
    bos_off    = 1 if has_bos else 0
    prefix_tok = tokenizer.encode(system_prefix, add_special_tokens=False)
    name_tok   = tokenizer.encode(build_name_block(name, tokenizer),
                                  add_special_tokens=False)
    cond_tok   = tokenizer.encode(build_condition_block(condition, tokenizer),
                                  add_special_tokens=False)

    n_prefix_blocks = (bos_off + len(prefix_tok)) // BLOCK_SIZE
    n_name_blocks   = len(name_tok) // BLOCK_SIZE
    n_cond_blocks   = len(cond_tok) // BLOCK_SIZE
    total_tokens    = bos_off + len(prefix_tok) + len(name_tok) + len(cond_tok)

    return {
        "prefix_blocks" : n_prefix_blocks,
        "name_blocks"   : n_name_blocks,
        "cond_blocks"   : n_cond_blocks,
        "total_private_blocks": n_name_blocks + n_cond_blocks,
        "total_tokens"  : total_tokens,
        "within_limit"  : total_tokens <= 4096,
        "t1_threshold_ms": T1_THRESHOLD_MS,
        "t2_threshold_ms": T2_THRESHOLD_MS,
    }


# ── Victim seeder ─────────────────────────────────────────────────────────────

def seed_victims_two_stage(
    client        : OpenAI,
    tokenizer     : AutoTokenizer,
    system_prefix : str,
    n_victims     : int = 50,
    seed          : int = 42,
) -> list[dict]:
    """
    Seed n_victims using the two-stage separated template and return records.

    Each record:
        victim_id          : int
        prompt             : full prompt string
        stage1_probe       : name block only (dummy condition) — for Stage 1 reseed
        ground_truth       : {name, dob, condition}
        n_name_blocks      : int
        n_cond_blocks      : int
        n_total_blocks     : int
    """
    rng     = random.Random(seed)
    records = []

    print(f"[seeder_v2] Seeding {n_victims} victims (two-stage template)...")

    for i in range(n_victims):
        name      = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        dob       = _random_dob(rng)
        condition = rng.choice(MEDICAL_CONDITIONS)

        full_prompt   = build_two_stage_prompt(
            system_prefix, name, dob, condition, tokenizer
        )
        stage1_probe  = build_two_stage_prompt(
            system_prefix, name, dob, condition, tokenizer, use_dummy_cond=True
        )

        counts = count_two_stage_blocks(system_prefix, name, dob, condition, tokenizer)

        try:
            client.completions.create(
                model      = MODEL_ID,
                prompt     = full_prompt,
                max_tokens = 1,
                temperature= 0.0,
            )
        except Exception as exc:
            print(f"[seeder_v2] WARNING: victim {i} seed failed: {exc}")
            continue

        records.append({
            "victim_id"         : i,
            "prompt"            : full_prompt,
            "stage1_probe"      : stage1_probe,
            "ground_truth"      : {"name": name, "dob": dob, "condition": condition},
            "n_name_blocks"     : counts["name_blocks"],
            "n_cond_blocks"     : counts["cond_blocks"],
            "n_total_blocks"    : counts["total_private_blocks"],
            "total_tokens"      : counts["total_tokens"],
        })

        if (i + 1) % 10 == 0:
            print(f"[seeder_v2]   {i+1}/{n_victims} seeded "
                  f"(name_blocks={counts['name_blocks']}, "
                  f"cond_blocks={counts['cond_blocks']}, "
                  f"total_tokens={counts['total_tokens']})")

    print(f"[seeder_v2] Done — {len(records)} victims seeded.")
    return records


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== two_stage_victim_seeder smoke test ===\n")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    prefix, n_tok = build_aligned_system_prompt(tok)
    print(f"System prefix: {n_tok} tokens, (1+{n_tok}) % {BLOCK_SIZE} = {(1+n_tok)%BLOCK_SIZE}")

    test_cases = [
        ("Mary Smith",    "2001-07-12", "hypothyroidism"),
        ("John Williams", "1987-12-23", "atrial fibrillation"),
    ]
    for name, dob, cond in test_cases:
        counts = count_two_stage_blocks(prefix, name, dob, cond, tok)
        print(f"\n  {name} / {cond}")
        for k, v in counts.items():
            print(f"    {k}: {v}")

    print("\n--- Template structure check ---")
    print(f"  NAME_BLOCKS target : {NAME_BLOCKS}  (need >= 64 for timing sensitivity)")
    print(f"  COND_BLOCKS target : {COND_BLOCKS}")
    print(f"  T1 threshold       : {T1_THRESHOLD_MS} ms (Stage 1, name confirmed)")
    print(f"  T2 threshold       : {T2_THRESHOLD_MS} ms (Stage 2, condition confirmed)")
    print(f"  Expected T_S1_HIT  : {T_S1_HIT_MS} ms (right name, wrong condition)")
    print(f"  E[Q] adaptive      : ~79 queries  (vs ~1000 linear)")
