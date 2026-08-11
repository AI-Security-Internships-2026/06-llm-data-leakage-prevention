

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from faker import Faker

logger = logging.getLogger(__name__)

# ── Domain vocabulary ──────────────────────────────────────────────────────────
# Shared with reconstructor.py — defines the closed-vocabulary assumption.
# Victims are generated from these lists so the attacker vocabulary has 100%
# coverage, matching the realistic scenario where both victim and attacker
# operate in the same application domain (US medical chatbot).

FIRST_NAMES: list[str] = [
    "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael",
    "Linda", "William", "Barbara", "David", "Elizabeth", "Richard", "Susan",
    "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen", "Christopher",
    "Lisa", "Daniel", "Nancy", "Matthew", "Betty", "Anthony", "Margaret",
    "Mark", "Sandra", "Donald", "Ashley", "Steven", "Emily", "Paul", "Kimberly",
    "Andrew", "Donna", "Joshua", "Michelle", "Kenneth", "Carol", "Kevin",
    "Amanda", "Brian", "Melissa", "George", "Deborah", "Timothy", "Stephanie",
]

LAST_NAMES: list[str] = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
    "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green",
    "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
    "Carter", "Roberts",
]

MEDICAL_CONDITIONS: list[str] = [
    "diabetes",
    "hypertension",
    "asthma",
    "chronic kidney disease",
    "heart failure",
    "COPD",
    "depression",
    "anxiety disorder",
    "hypothyroidism",
    "atrial fibrillation",
    "osteoarthritis",
    "rheumatoid arthritis",
    "epilepsy",
    "Parkinson disease",
    "Alzheimer disease",
    "schizophrenia",
    "bipolar disorder",
    "obesity",
    "anemia",
    "liver cirrhosis",
]

# System prompt template — the "known" part of Scenario S2.
# The attacker knows this structure (e.g. by querying the API with an empty
# user turn and inspecting the shared public system prompt).
SYSTEM_TEMPLATE: str = (
    "You are a medical assistant. Answer questions about patient symptoms. "
    "Patient name: {name}. DOB: {dob}. Condition: {condition}."
)

# Known prefix shared by ALL victims — this is the template preamble the
# attacker knows verbatim. Used as the starting point for reconstruction.
KNOWN_PREFIX: str = (
    "You are a medical assistant. Answer questions about patient symptoms. "
    "Patient name: "
)


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class VictimRecord:
    """Ground-truth record for one synthetic victim."""
    victim_id:      int
    prompt:         str          # full prompt sent to vLLM
    first_name:     str
    last_name:      str
    name:           str          # "{first_name} {last_name}"
    dob:            str          # ISO-8601 "YYYY-MM-DD"
    condition:      str
    seeded:         bool = True  # False if the vLLM call failed

    @property
    def private_fields(self) -> dict[str, str]:
        return {
            "first_name": self.first_name,
            "last_name":  self.last_name,
            "dob":        self.dob,
            "condition":  self.condition,
        }

    @property
    def private_tokens(self) -> list[str]:
        """Ordered list of private atomic values the attacker must recover."""
        return [self.first_name, self.last_name, self.condition]


# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_victim_prompt(
    first_name: str,
    last_name:  str,
    dob:        str,
    condition:  str,
) -> str:
    """Render SYSTEM_TEMPLATE with concrete private field values."""
    return SYSTEM_TEMPLATE.format(
        name=f"{first_name} {last_name}",
        dob=dob,
        condition=condition,
    )


# ── Seeder ─────────────────────────────────────────────────────────────────────

def generate_victim_records(
    n_victims: int = 50,
    seed:      int = 42,
) -> list[VictimRecord]:
    """
    Generate *n_victims* synthetic VictimRecord objects using Faker.
    Does NOT send any requests to vLLM — pure record generation.
    Use this when you want to control per-victim seeding timing yourself
    (e.g. the per-victim evict→seed→attack harness loop).

    Parameters / Returns: same as seed_victims() minus client/model.
    """
    faker = Faker("en_US")
    Faker.seed(seed)
    random.seed(seed)

    records: list[VictimRecord] = []
    for i in range(n_victims):
        # Names drawn from the closed attacker vocabulary — guarantees 100%
        # recall coverage (realistic: both victim and attacker use same domain)
        first_name = random.choice(FIRST_NAMES)
        last_name  = random.choice(LAST_NAMES)
        dob_str    = faker.date_of_birth(minimum_age=20, maximum_age=91).isoformat()
        condition  = random.choice(MEDICAL_CONDITIONS)
        prompt     = build_victim_prompt(first_name, last_name, dob_str, condition)

        records.append(VictimRecord(
            victim_id=i + 1,
            prompt=prompt,
            first_name=first_name,
            last_name=last_name,
            name=f"{first_name} {last_name}",
            dob=dob_str,
            condition=condition,
            seeded=False,   # not yet sent to vLLM
        ))

    logger.info("Generated %d victim records (not yet seeded to vLLM).", n_victims)
    return records


def seed_single_victim(client, model: str, record: VictimRecord) -> bool:
    """Send one victim's prompt to vLLM (max_tokens=1) to populate KV cache."""
    try:
        client.completions.create(
            model=model,
            prompt=record.prompt,
            max_tokens=1,
            temperature=0.0,
            stream=False,
        )
        record.seeded = True
        return True
    except Exception as exc:
        logger.warning("Seeding victim %d failed: %s", record.victim_id, exc)
        return False


def seed_victims(
    client,
    model:     str,
    n_victims: int = 50,
    seed:      int = 42,
) -> list[VictimRecord]:
    """
    Generate *n_victims* synthetic medical prompts, send each to vLLM with
    max_tokens=1 (prefill only; output discarded), and return ground-truth
    VictimRecord objects for later evaluation.

    Parameters
    ----------
    client    : OpenAI-compatible client (real vLLM or MockVLLMClient)
    model     : model ID string passed to vLLM
    n_victims : number of synthetic victims to seed (default 50)
    seed      : RNG seed for reproducibility (default 42)

    Returns
    -------
    list[VictimRecord] — only successfully seeded victims are included.

    Notes
    -----
    - Faker `en_US` provides ~300+ distinct first/last name combinations.
    - DOB sampled uniformly over 1935-2006 (ages 20-91 at 2026).
    - Condition sampled uniformly from MEDICAL_CONDITIONS (20 diagnoses).
    - max_tokens=1 and temperature=0.0 minimise compute cost; we only need
      the prefill phase to populate the KV cache blocks.
    """
    faker = Faker("en_US")
    Faker.seed(seed)
    random.seed(seed)

    records: list[VictimRecord] = []

    logger.info("Seeding %d victim prefixes into vLLM KV cache ...", n_victims)

    for i in range(n_victims):
        first_name = random.choice(FIRST_NAMES)
        last_name  = random.choice(LAST_NAMES)
        dob: date  = faker.date_of_birth(minimum_age=20, maximum_age=91)
        dob_str    = dob.isoformat()               # "YYYY-MM-DD"
        condition  = random.choice(MEDICAL_CONDITIONS)

        prompt = build_victim_prompt(first_name, last_name, dob_str, condition)

        seeded = False
        try:
            # max_tokens=1: we only need the prefill to populate KV blocks.
            # stream=False: we do not measure TTFT here (victim is not the attacker).
            client.completions.create(
                model=model,
                prompt=prompt,
                max_tokens=1,
                temperature=0.0,
                stream=False,
            )
            seeded = True
        except Exception as exc:
            logger.warning("Victim %d seeding failed: %s", i + 1, exc)

        record = VictimRecord(
            victim_id=i + 1,
            prompt=prompt,
            first_name=first_name,
            last_name=last_name,
            name=f"{first_name} {last_name}",
            dob=dob_str,
            condition=condition,
            seeded=seeded,
        )
        records.append(record)

        if seeded:
            logger.debug(
                "  [%02d] Seeded: %-12s %-12s | %s | %s",
                i + 1, first_name, last_name, dob_str, condition,
            )

    n_ok = sum(r.seeded for r in records)
    logger.info("Seeded %d/%d victims successfully.", n_ok, n_victims)
    return records
