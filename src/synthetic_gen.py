"""
synthetic_gen.py — Synthetic PII Dataset Generator
CNIT/PNTLab Pisa — AI Security Internship 2026
Student: Muhammad Hashim Mughal | Week: 04

"""

import argparse
import json
import os
import random
from dataclasses import dataclass, field, asdict
from typing import Callable

from faker import Faker

fake = Faker("en_US")
Faker.seed(42)


@dataclass
class EntityTemplate:
    entity_type: str
    generator: Callable[[], str]
    templates: list[str]
    is_high_risk: bool = False


_CNIC_DIGITS = lambda: (
    f"{random.randint(10000,99999)}-"
    f"{random.randint(1000000,9999999)}-"
    f"{random.randint(0,9)}"
)

_IBAN_POOL = [
    "GB29NWBK60161331926819",
    "DE89370400440532013000",
    "FR7614508711002120144503422",
    "NL91ABNA0417164300",
    "ES9121000418450200051332",
    "IT60X0542811101000000123456",
    "PL61109010140000071219812874",
    "SE4550000000058398257466",
    "CH9300762011623852957",
    "AT611904300234573201",
    "BE68539007547034",
    "DK5000400440116243",
    "NO9386011117947",
    "PT50000201231234567890154",
    "IE29AIBK93115212345678",
]

ENTITY_TEMPLATES: list[EntityTemplate] = [
    EntityTemplate(
        entity_type="EMAIL_ADDRESS",
        generator=fake.email,
        templates=[
            "Please contact {pii} for further information.",
            "Send your application to {pii} by Friday.",
            "The account email on file is {pii}.",
            "ERROR: user={pii} failed authentication.",
            "CC: {pii} on all future correspondence.",
            '{"email": "{pii}", "role": "admin"}',
            "Reply-To: {pii}",
            "Forwarded message from {pii} regarding the quarterly review.",
        ],
        is_high_risk=False,
    ),
    EntityTemplate(
        entity_type="CREDIT_CARD",
        generator=fake.credit_card_number,
        templates=[
            "Payment processed with card number {pii}.",
            "Card on file: {pii}.",
            "Transaction declined for card {pii}.",
            'Billing details: {{"card": "{pii}", "exp": "12/28"}}',
            "Please charge {pii} for the renewal.",
            "Refund issued to card ending in {pii}.",
        ],
        is_high_risk=True,
    ),
    EntityTemplate(
        entity_type="PHONE_NUMBER",
        generator=fake.phone_number,
        templates=[
            "Call us at {pii} for support.",
            "The customer's phone number is {pii}.",
            "Contact: {pii}",
            "Reach the on-call engineer at {pii}.",
            "SMS verification sent to {pii}.",
            "Emergency contact: {pii}.",
        ],
        is_high_risk=False,
    ),
    EntityTemplate(
        entity_type="PERSON",
        generator=fake.name,
        templates=[
            "{pii} submitted the expense report.",
            "Patient: {pii}, DOB unknown.",
            "The contract was signed by {pii}.",
            "Dear {pii}, please review the attached document.",
            "Manager: {pii}",
            "{pii} was granted admin access on Monday.",
            "CC: {pii}",
        ],
        is_high_risk=False,
    ),
    EntityTemplate(
        entity_type="US_SSN",
        generator=fake.ssn,
        templates=[
            "The applicant's SSN is {pii}.",
            "Social security number on file: {pii}.",
            "Taxpayer ID: {pii}",
            "SSN: {pii} — please verify before processing.",
            "W2 form submitted for SSN {pii}.",
        ],
        is_high_risk=True,
    ),
    EntityTemplate(
        entity_type="IBAN_CODE",
        generator=lambda: random.choice(_IBAN_POOL),
        templates=[
            "Wire the payment to IBAN {pii}.",
            "Bank account: {pii}",
            "Transfer {pii} by end of day.",
            "Beneficiary IBAN: {pii}",
            "Please credit IBAN {pii} for the refund.",
        ],
        is_high_risk=True,
    ),
    EntityTemplate(
        entity_type="PK_CNIC",
        generator=_CNIC_DIGITS,
        templates=[
            "National Identity Number: {pii}",
            "CNIC: {pii} — verify with NADRA.",
            "The applicant's CNIC is {pii}.",
            "Identity card number {pii} confirmed.",
            "Please attach a copy of your CNIC ({pii}).",
        ],
        is_high_risk=True,
    ),
    EntityTemplate(
        entity_type="LOCATION",
        generator=fake.address,
        templates=[
            "Delivery address: {pii}",
            "The registered office is at {pii}.",
            "Shipping to: {pii}",
        ],
        is_high_risk=False,
    ),
]

_ENTITY_MAP = {t.entity_type: t for t in ENTITY_TEMPLATES}

CLEAN_TEMPLATES = [
    "The REST endpoint accepts JSON over HTTPS and returns a 200 status on success.",
    "def detect(text: str) -> dict:\n    return analyze(text)",
    "Hypertension is treated with ACE inhibitors and calcium channel blockers.",
    "Quarterly revenue grew 12% year-over-year, driven by SaaS subscriptions.",
    "The sample was heated to 250°C for 30 minutes inside a sealed autoclave.",
    "The government announced updated energy consumption guidelines this quarter.",
    "We compare Model A against Model B on the standard benchmark dataset.",
    "No personal information is collected or stored by this service.",
    "The pipeline runs every 6 hours and writes output to the data warehouse.",
    "All configuration values are loaded from environment variables at startup.",
    "SELECT * FROM logs WHERE level = 'ERROR' AND timestamp > NOW() - INTERVAL 1 HOUR;",
    "Nginx returned a 502 Bad Gateway — check upstream health.",
    "The model achieved 94.3% accuracy on the held-out test set.",
    "Memory usage peaked at 7.2 GB during the batch inference job.",
    "Updated the CI pipeline to run linting before unit tests.",
    "Kubernetes pod restarted 3 times due to OOMKilled errors.",
    "The license agreement prohibits redistribution without written consent.",
    "chmod 644 /etc/nginx/nginx.conf && systemctl reload nginx",
    "Gradient descent converged after 150 epochs with a final loss of 0.042.",
    "The experiment used a 80/10/10 train/validation/test split.",
    "Backpropagation computes gradients using the chain rule of calculus.",
    "The database index reduced query latency from 320ms to 18ms.",
    "Version 2.4.1 patches a null pointer dereference in the parser module.",
    "The team uses trunk-based development with short-lived feature branches.",
    "Prometheus scrapes metrics every 15 seconds from all service endpoints.",
    "The paper proposes a novel attention mechanism for long-document summarisation.",
    "Federated learning distributes model training across multiple edge devices.",
    "The autoclave cycle maintains 121°C at 15 PSI for 30 minutes.",
    "git rebase -i HEAD~5 squashes the last five commits into one.",
    "The firewall blocks all inbound traffic on port 22 except from the VPN range.",
]

MULTI_PII_TEMPLATES = [
    lambda: (
        f"{fake.name()}'s email is {fake.email()} and her direct line is {fake.phone_number()}.",
        ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"],
    ),
    lambda: (
        f'{{"user": "{fake.user_name()}", "email": "{fake.email()}", "card": "{fake.credit_card_number()}"}}',
        ["EMAIL_ADDRESS", "CREDIT_CARD"],
    ),
    lambda: (
        f"The applicant {fake.name()} has SSN {fake.ssn()} on file.",
        ["PERSON", "US_SSN"],
    ),
    lambda: (
        f"ERROR [user={fake.email()}]: quota_exceeded=true at endpoint /api/v1/export",
        ["EMAIL_ADDRESS"],
    ),
    lambda: (
        f"Dr. {fake.name()} prescribed metformin 500mg to patient {fake.name()}.",
        ["PERSON"],
    ),
    lambda: (
        f"Applicant: {fake.name()}, CNIC: {_CNIC_DIGITS()}",
        ["PERSON", "PK_CNIC"],
    ),
    lambda: (
        f"Please wire funds to IBAN {random.choice(_IBAN_POOL)} and cc {fake.email()} on the confirmation.",
        ["IBAN_CODE", "EMAIL_ADDRESS"],
    ),
    lambda: (
        f"US SSN: {fake.ssn()}  |  PK CNIC: {_CNIC_DIGITS()}  |  Submitted by {fake.name()}.",
        ["US_SSN", "PK_CNIC", "PERSON"],
    ),
    lambda: (
        f"Customer called from {fake.phone_number()} to dispute charge on card {fake.credit_card_number()}.",
        ["PHONE_NUMBER", "CREDIT_CARD"],
    ),
    lambda: (
        f"Delivery for {fake.name()} to be sent to {fake.address()}.",
        ["PERSON", "LOCATION"],
    ),
]


@dataclass
class Sample:
    id: str
    text: str
    label: str
    entity_types: list[str] = field(default_factory=list)
    context: str = ""


def _generate_single_pii_sample(idx: int, rng: random.Random) -> Sample:
    template_obj: EntityTemplate = rng.choice(ENTITY_TEMPLATES)
    pii_value = template_obj.generator()
    template_str = rng.choice(template_obj.templates)
    text = template_str.replace("{pii}", str(pii_value))
    return Sample(
        id=f"L{idx:04d}",
        text=text,
        label="LEAKING",
        entity_types=[template_obj.entity_type],
        context="mixed",
    )


def _generate_multi_pii_sample(idx: int, rng: random.Random) -> Sample:
    fn = rng.choice(MULTI_PII_TEMPLATES)
    text, entity_types = fn()
    return Sample(
        id=f"LM{idx:04d}",
        text=text,
        label="LEAKING",
        entity_types=entity_types,
        context="multi",
    )


def _generate_clean_sample(idx: int, rng: random.Random) -> Sample:
    text = rng.choice(CLEAN_TEMPLATES)
    return Sample(
        id=f"N{idx:04d}",
        text=text,
        label="CLEAN",
        entity_types=[],
        context="clean",
    )


def generate_dataset(
    n_samples: int = 1200,
    leaking_ratio: float = 0.55,
    multi_pii_ratio: float = 0.20,
    seed: int = 42,
) -> list[Sample]:
    rng = random.Random(seed)
    Faker.seed(seed)

    n_leaking = int(n_samples * leaking_ratio)
    n_clean = n_samples - n_leaking
    n_multi = int(n_leaking * multi_pii_ratio)
    n_single = n_leaking - n_multi

    samples: list[Sample] = []

    for i in range(n_single):
        samples.append(_generate_single_pii_sample(i + 1, rng))

    for i in range(n_multi):
        samples.append(_generate_multi_pii_sample(i + 1, rng))

    for i in range(n_clean):
        samples.append(_generate_clean_sample(i + 1, rng))

    rng.shuffle(samples)
    return samples


def compute_stats(samples: list[Sample]) -> dict:
    leaking = [s for s in samples if s.label == "LEAKING"]
    clean   = [s for s in samples if s.label == "CLEAN"]

    entity_freq: dict[str, int] = {}
    for s in leaking:
        for e in s.entity_types:
            entity_freq[e] = entity_freq.get(e, 0) + 1

    return {
        "total": len(samples),
        "leaking": len(leaking),
        "clean": len(clean),
        "leaking_ratio": round(len(leaking) / len(samples), 3),
        "entity_type_distribution": dict(
            sorted(entity_freq.items(), key=lambda x: -x[1])
        ),
        "avg_text_length_chars": round(
            sum(len(s.text) for s in samples) / len(samples), 1
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Synthetic PII dataset generator")
    parser.add_argument("--samples", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--leaking-ratio", type=float, default=0.55)
    parser.add_argument("--multi-pii-ratio", type=float, default=0.20)
    parser.add_argument("--output-dir", type=str, default="experiments/results")
    args = parser.parse_args()

    samples = generate_dataset(
        n_samples=args.samples,
        leaking_ratio=args.leaking_ratio,
        multi_pii_ratio=args.multi_pii_ratio,
        seed=args.seed,
    )

    stats = compute_stats(samples)

    print(f"Total: {stats['total']}  |  Leaking: {stats['leaking']}  |  Clean: {stats['clean']}")
    for etype, count in stats["entity_type_distribution"].items():
        print(f"  {count:>4}  {etype}")

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "synthetic_dataset.json"), "w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in samples], f, indent=2, ensure_ascii=False)

    with open(os.path.join(args.output_dir, "synthetic_dataset_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()