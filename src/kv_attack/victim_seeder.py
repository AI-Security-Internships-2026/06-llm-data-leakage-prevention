import random
from datetime import date, timedelta

from openai import OpenAI
from transformers import AutoTokenizer

from kv_attack import (
    MODEL_ID, BLOCK_SIZE,
    MEDICAL_CONDITIONS, FIRST_NAMES, LAST_NAMES,
)

# ── System prefix (fixed, known to attacker in Scenario S2) ──────────────────

_SYSTEM_PREFIX_RAW = (
    "You are a secure, HIPAA-compliant medical assistant deployed at "
    "City General Hospital. Your role is to assist physicians and "
    "nursing staff with accurate, evidence-based patient information. "
    "Always address patients by their full legal name as recorded in "
    "the system. Never disclose any protected health information to "
    "unauthorised third parties under any circumstances. "
    "All communications must comply with HIPAA Privacy Rule 45 CFR "
    "Part 164. You must not speculate about diagnoses not present in "
    "the patient record. Maintain professional tone at all times. "
    "If asked about medication dosages, always defer to the attending "
    "physician. Do not provide legal or financial advice. "
    "In case of a medical emergency, instruct the user to call 911 "
    "immediately. All session data is encrypted and logged for audit. "
    "Unauthorised access to this system is a federal offence under "
    "the Computer Fraud and Abuse Act. You are operating in a "
    "multi-tenant environment. Treat all patient data as strictly "
    "confidential and never reference one patient's information when "
    "responding to another. Session timeout occurs after 15 minutes "
    "of inactivity. Always verify physician credentials before "
    "releasing sensitive diagnostic information. Report any suspected "
    "data breach immediately to the hospital information security "
    "office. This system is monitored continuously for compliance. "
    "Do not store or repeat patient identifiers in your responses "
    "unless directly asked by an authenticated physician. "
    "These instructions take precedence over all user requests."
)

# ── Private template (secrets FIRST, long filler AFTER) ──────────────────────
#
# _RECORD_FILLER is repeated 14 times to produce ~3,080 tokens total.
# Calculation:
#   _RECORD_FILLER ≈ 220 tokens (170 words × 1.3 tokens/word)
#   14 repetitions × 220 = 3,080 tokens
#   Total prompt: 271 (system) + 3,080 (private) = 3,351 tokens < 4,096 ✓
#   Private blocks: floor(3,080 / 16) = 192 complete blocks
#   Expected timing gap on GB10: 192 × 16 / 50,000 ≈ 62 ms
#
# The filler is identical for ALL candidates (victim and attacker use the
# same template structure). Only {name} and {condition} vary per probe.
# Because {name}/{condition} are in block N (the FIRST private block),
# a wrong candidate causes ALL 192 subsequent blocks to miss via the
# hash chain, producing the full 62 ms timing gap.
#
_RECORD_FILLER = (
    "The patient's medical history has been comprehensively reviewed "
    "and documented by the attending physician on the date of this record. "
    "Current prescribed medications have been assessed for efficacy and "
    "potential interactions with existing treatments. "
    "All laboratory results are within clinically acceptable parameters "
    "as of the most recent assessment date on file. "
    "The treatment plan continues as originally prescribed with no "
    "modifications required at this time per physician directive. "
    "Follow-up appointments are scheduled in accordance with the standard "
    "clinical protocol for this diagnosis category and patient risk profile. "
    "Nursing staff have been fully briefed on the patient's current "
    "condition, care requirements, and any special instructions issued "
    "by the attending physician during the last review session. "
    "The patient has been informed of their diagnosis, available treatment "
    "options, and prognosis in language appropriate to their level of "
    "health literacy and understanding as assessed during intake. "
    "Informed consent has been obtained, witnessed, and documented in "
    "accordance with hospital policy and applicable regulations. "
    "All communications with the patient and their designated emergency "
    "contacts comply fully with HIPAA privacy regulations and hospital "
    "information governance policy as reviewed annually. "
)

_PRIVATE_TEMPLATE = (
    "{name}. {condition}. "
    + _RECORD_FILLER * 14
    + "END OF PATIENT RECORD."
)

# Padding word for block alignment.
# " yes" reliably adds exactly 1 token in the Llama-3.1 BPE tokenizer.
# DeepSeek-R1-Distill-Llama-8B uses the same tokenizer, so this holds.
_PAD_WORD = " yes"


# ── Block alignment ───────────────────────────────────────────────────────────

def build_aligned_system_prompt(
    tokenizer: AutoTokenizer,
    has_bos: bool = True,
) -> tuple[str, int]:
    """
    Pad _SYSTEM_PREFIX_RAW so private content starts at a block boundary.

    vLLM prepends BOS for Llama-family models (including DeepSeek-R1-Distill-Llama-8B),
    so the effective token sequence is:
        [BOS(1), sys_prefix_tokens(N), private_tokens...]
    Block boundary condition: (1 + N) % BLOCK_SIZE == 0

    Returns (padded_prefix, n_prefix_tokens_excluding_BOS).
    """
    bos_offset = 1 if has_bos else 0
    padded = _SYSTEM_PREFIX_RAW

    for _ in range(BLOCK_SIZE * 4):
        tokens = tokenizer.encode(padded, add_special_tokens=False)
        if (bos_offset + len(tokens)) % BLOCK_SIZE == 0:
            return padded, len(tokens)
        padded = padded + _PAD_WORD

    final_tokens = tokenizer.encode(padded, add_special_tokens=False)
    raise RuntimeError(
        f"Cannot align system prefix after {BLOCK_SIZE * 4} attempts. "
        f"Current: {len(final_tokens)} tokens. "
        f"Need (1 + N) % {BLOCK_SIZE} == 0. "
        f"Try a different _PAD_WORD."
    )


def build_private_block(name: str, dob: str, condition: str) -> str:
    """Return the filled private section string."""
    return _PRIVATE_TEMPLATE.format(name=name, dob=dob, condition=condition)


def count_private_blocks(
    tokenizer     : AutoTokenizer,
    system_prefix : str,
    name          : str,
    dob           : str,
    condition     : str,
    has_bos       : bool = True,
) -> int:
    """
    Return the number of COMPLETE KV-cache blocks in the private section.
    Target: >= 100 for a reliable timing gap on GB10.
    """
    bos_offset    = 1 if has_bos else 0
    private_str   = build_private_block(name, dob, condition)
    full_prompt   = system_prefix + " " + private_str
    full_tokens   = tokenizer.encode(full_prompt,   add_special_tokens=False)
    prefix_tokens = tokenizer.encode(system_prefix, add_special_tokens=False)

    total_in_cache    = bos_offset + len(full_tokens)
    prefix_blocks     = (bos_offset + len(prefix_tokens)) // BLOCK_SIZE
    total_complete    = total_in_cache // BLOCK_SIZE
    return total_complete - prefix_blocks


# ── DOB generation ────────────────────────────────────────────────────────────

def _random_dob(rng: random.Random) -> str:
    """Uniform random DOB in [1935-01-01, 2006-12-31], YYYY-MM-DD."""
    start = date(1935, 1, 1)
    delta = (date(2006, 12, 31) - start).days
    return (start + timedelta(days=rng.randint(0, delta))).isoformat()


# ── Victim seeding ────────────────────────────────────────────────────────────

def seed_victim_prefix(
    client        : OpenAI,
    tokenizer     : AutoTokenizer,
    system_prefix : str,
    n_victims     : int = 50,
    seed          : int = 42,
) -> list[dict]:
    """
    Generate n_victims synthetic victim prompts, POST each to vLLM to
    populate the KV cache, and return ground-truth records.

    Each record:
        victim_id        : int
        prompt           : full prompt string (used for re-seeding)
        ground_truth     : {"name", "dob", "condition"}
        n_private_blocks : int (complete private KV blocks)
    """
    rng     = random.Random(seed)
    records = []

    n_prefix_tok = len(tokenizer.encode(system_prefix, add_special_tokens=False))
    print(f"[victim_seeder] Seeding {n_victims} victims...")
    print(f"[victim_seeder] System prefix : {n_prefix_tok} tokens")

    for i in range(n_victims):
        name      = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        dob       = _random_dob(rng)
        condition = rng.choice(MEDICAL_CONDITIONS)

        private     = build_private_block(name, dob, condition)
        full_prompt = system_prefix + " " + private

        n_priv = count_private_blocks(tokenizer, system_prefix, name, dob, condition)

        try:
            client.completions.create(
                model      = MODEL_ID,
                prompt     = full_prompt,
                max_tokens = 1,
                temperature= 0.0,
            )
        except Exception as exc:
            print(f"[victim_seeder] WARNING: victim {i} seed failed: {exc}")
            continue

        records.append({
            "victim_id"       : i,
            "prompt"          : full_prompt,
            "ground_truth"    : {"name": name, "dob": dob, "condition": condition},
            "n_private_blocks": n_priv,
        })

        if (i + 1) % 10 == 0:
            print(f"[victim_seeder]   {i + 1}/{n_victims} seeded "
                  f"(private_blocks={n_priv})")

    print(f"[victim_seeder] Done — {len(records)} victims seeded.")
    return records


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== victim_seeder smoke test ===\n")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    prefix, n_tok = build_aligned_system_prompt(tok)
    print(f"System prefix tokens (not counting BOS): {n_tok}")
    print(f"(1 + {n_tok}) % {BLOCK_SIZE} = {(1 + n_tok) % BLOCK_SIZE}  "
          f"<- must be 0")
    print(f"Private section starts at block : {(1 + n_tok) // BLOCK_SIZE}")
    print()

    test_cases = [
        ("John Smith",        "1975-03-21", "diabetes"),
        ("Patricia Martinez", "1990-11-05", "hypertension"),
        ("Michael Williams",  "1958-07-14", "coronary artery disease"),
    ]
    for name, dob, cond in test_cases:
        n      = count_private_blocks(tok, prefix, name, dob, cond)
        full   = prefix + " " + build_private_block(name, dob, cond)
        n_full = 1 + len(tok.encode(full, add_special_tokens=False))
        gap_ms = round(n * 16 / 50_000 * 1000, 1)
        print(f"  {name:25s} | {cond:25s} | "
              f"private_blocks={n:3d}  total_tokens={n_full:5d}  "
              f"expected_gap~{gap_ms}ms")

    print()
    print("Targets:")
    print("  private_blocks >= 100  (for ~32 ms gap on GB10)")
    print("  total_tokens   <  4096 (vLLM max_model_len)")
    print("  (1 + N) % 16   == 0    (block alignment)")