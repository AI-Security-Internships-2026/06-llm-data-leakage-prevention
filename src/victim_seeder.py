"""
victim_seeder.py — Synthetic Victim Prefix Seeder
==========================================================
Generates synthetic medical-chatbot prompts and seeds the vLLM KV cache,
simulating a legitimate co-tenant populating shared GPU memory.

Attack scenario: S2 (Proposal §3.4)
  Known  : SYSTEM_TEMPLATE structure
  Unknown: patient name, DOB, medical condition (the private fields)

Week 10 — Phase 1, Issue #9
AI Security Internship 2026 — ONT Lab / CNIT-PNTLab Pisa
Muhammad Hashim Mughal
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from faker import Faker

logger = logging.getLogger(__name__)

# ── Domain constants ───────────────────────────────────────────────────────────

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
        first_name = faker.first_name()
        last_name  = faker.last_name()
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