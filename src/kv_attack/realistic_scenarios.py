"""
kv_attack.realistic_scenarios
==============================
Issue #4 — Three realistic synthetic prompt scenarios + three attacker-
knowledge levels.

Scenarios
---------
T0  Controlled (large two-stage regions)  — existing paper layout, kept as baseline
T1  Medical      — patient name + diagnosis in a realistic clinical note template
T2  Financial    — customer name + sensitive financial field in a bank statement template
T3  Enterprise   — employee/project name + confidential field in an HR / project brief

Attacker knowledge levels
--------------------------
K1  Exact template, small dictionary    (10×10 names, 20 conditions)
K2  Exact template, larger dictionary   (30×30 names, 50 conditions)
K3  Partial template knowledge          (attacker knows rough structure but not exact wording)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable



_FIRST_NAMES_MED = [
    "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
    "William", "Barbara", "David", "Susan", "Richard", "Jessica", "Joseph", "Sarah",
    "Thomas", "Karen", "Charles", "Lisa", "Christopher", "Nancy", "Daniel", "Betty",
    "Matthew", "Margaret", "Anthony", "Sandra", "Mark", "Ashley",
]
_LAST_NAMES_MED = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Wilson", "Martinez", "Anderson", "Taylor", "Thomas", "Hernandez", "Moore",
    "Martin", "Jackson", "Thompson", "White", "Lopez", "Lee", "Gonzalez", "Harris",
    "Clark", "Lewis", "Robinson", "Walker", "Perez", "Hall", "Young",
]
_CONDITIONS_MED_20 = [
    "diabetes", "hypertension", "asthma", "arthritis", "depression",
    "anxiety", "COPD", "obesity", "hypothyroidism", "hyperlipidemia",
    "coronary artery disease", "chronic kidney disease", "heart failure",
    "atrial fibrillation", "osteoporosis", "Parkinson's disease",
    "multiple sclerosis", "epilepsy", "migraine", "sleep apnea",
]
_CONDITIONS_MED_50 = _CONDITIONS_MED_20 + [
    "type 2 diabetes", "rheumatoid arthritis", "celiac disease", "Crohn's disease",
    "lupus", "fibromyalgia", "psoriasis", "gout", "anemia", "chronic pain",
    "bipolar disorder", "PTSD", "schizophrenia", "autism spectrum disorder",
    "ADHD", "dementia", "Alzheimer's disease", "Huntington's disease",
    "ALS", "muscular dystrophy", "cystic fibrosis", "sickle cell disease",
    "hemophilia", "thalassemia", "polycystic ovary syndrome",
    "endometriosis", "uterine fibroids", "prostate cancer", "breast cancer",
    "colorectal cancer",
]

_CUSTOMER_FIRST = [
    "Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Kate", "Leo", "Mia", "Noah", "Olivia", "Paul",
    "Quinn", "Rachel", "Sam", "Tara", "Uma", "Victor", "Wendy", "Xander",
    "Yara", "Zoe", "Aaron", "Beth", "Carl", "Diana",
]
_CUSTOMER_LAST = [
    "Adams", "Baker", "Campbell", "Dixon", "Evans", "Foster", "Green",
    "Hughes", "Ingram", "Jenkins", "King", "Lambert", "Morgan", "Nash",
    "Owen", "Parker", "Quinn", "Reed", "Scott", "Turner", "Underwood",
    "Vaughan", "Walsh", "Xavier", "Young", "Zhang", "Abbott", "Burns",
    "Collins", "Dean",
]
_FINANCIAL_FIELDS_20 = [
    "credit score 480", "credit score 720", "bankruptcy filed 2023",
    "net worth $2.1M", "net worth $45K", "monthly income $3,200",
    "monthly income $18,500", "overdue balance $12,400", "overdue balance $800",
    "investment portfolio $500K", "investment portfolio $8K",
    "tax lien filed 2022", "wire fraud investigation open",
    "salary garnishment active", "offshore account declared",
    "insider trading flagged", "loan default 2021",
    "401k balance $1.2M", "401k balance $4,300",
    "child support arrears $34,000",
]
_FINANCIAL_FIELDS_50 = _FINANCIAL_FIELDS_20 + [
    "credit score 550", "credit score 810", "foreclosure 2020",
    "pension value $320K", "pension value $12K", "alimony payments active",
    "student loan default", "medical debt $89K", "gambling losses $200K",
    "crypto holdings $1.4M", "crypto holdings $300", "IRS audit open",
    "FBAR violation pending", "mortgage arrears $24K", "eviction filed",
    "wage theft claim", "SBA loan default", "PPP fraud investigation",
    "securities fraud charge", "money laundering flagged",
    "account frozen by court", "inheritance dispute active",
    "divorce settlement $1.8M", "trust fund $4.2M", "annuity $800/mo",
    "disability claim active", "workers comp settlement $120K",
    "class action participant", "whistleblower claim filed",
    "restitution order $450K",
]

_EMP_FIRST = [
    "Alexander", "Bella", "Connor", "Daisy", "Ethan", "Fiona", "George",
    "Hannah", "Ivan", "Julia", "Kevin", "Laura", "Marcus", "Nina",
    "Oscar", "Petra", "Quentin", "Rosa", "Stefan", "Theresa",
    "Ulrich", "Valeria", "Walter", "Xenia", "Yorick", "Zelda",
    "Adrien", "Brigitte", "Claude", "Delphine",
]
_EMP_LAST = [
    "Müller", "Schmidt", "Schneider", "Fischer", "Weber", "Meyer", "Wagner",
    "Becker", "Schulz", "Hoffmann", "Schäfer", "Koch", "Bauer", "Richter",
    "Klein", "Wolf", "Schröder", "Neumann", "Schwarz", "Zimmermann",
    "Braun", "Krüger", "Hofmann", "Hartmann", "Lange", "Schmitt", "Werner",
    "Schmitz", "Krause", "Meier",
]
_ENTERPRISE_FIELDS_20 = [
    "termination for cause", "PIP active", "salary $42,000",
    "salary $210,000", "bonus $0 (frozen)", "bonus $85,000",
    "whistleblower complaint filed", "sexual harassment claim open",
    "NDA breach investigation", "IP theft allegation",
    "equity 0.8% ($1.2M vested)", "equity forfeited",
    "project AURORA — classified", "project HYDRA — classified",
    "security clearance revoked", "background check failed",
    "immigration status: H-1B expiring", "non-compete violation flagged",
    "redundancy package offered", "demotion pending approval",
]
_ENTERPRISE_FIELDS_50 = _ENTERPRISE_FIELDS_20 + [
    "salary $95,000", "salary $350,000", "bonus $12,000",
    "bonus $500,000", "stock options 50,000", "stock options 0 (unvested)",
    "relocation allowance $30K", "severance 6 months",
    "performance rating: exceeds", "performance rating: unsatisfactory",
    "promotion recommended", "demotion effective Q1",
    "project TITAN — secret", "project ZEUS — top secret",
    "drug test failed", "alcohol policy violation",
    "expense fraud $4,200", "expense fraud $89,000",
    "customer data breach — involved", "insider threat flagged",
    "reference check: negative", "conflict of interest: undisclosed",
    "dual employment discovered", "moonlighting policy breach",
    "social media policy violation", "code of conduct breach",
    "medical leave 6 months", "FMLA exhausted",
    "workplace injury claim", "ergonomic adjustment required",
]



@dataclass
class ScenarioConfig:
    """Definition of one realistic prompt scenario."""
    name          : str
    domain        : str
    first_names   : list[str]
    last_names    : list[str]
    secret_fields : list[str]
    build_prompt  : Callable
    field_label   : str
    description   : str


def _build_medical_prompt(system_prefix: str, name: str, dob: str, secret: str, tok) -> str:
    return (
        f"{system_prefix}\n\n"
        f"PATIENT CLINICAL NOTE\n"
        f"=====================\n"
        f"Patient: {name}\n"
        f"Date of Birth: {dob}\n"
        f"Primary Diagnosis: {secret}\n"
        f"Treating Physician: Dr. A. Patel\n"
        f"Note: Patient presents with documented history. See attached records.\n"
        f"Please summarise the treatment options available for this patient.\n"
    )


def _build_financial_prompt(system_prefix: str, name: str, dob: str, secret: str, tok) -> str:
    return (
        f"{system_prefix}\n\n"
        f"CONFIDENTIAL CUSTOMER RECORD\n"
        f"=============================\n"
        f"Customer: {name}\n"
        f"Account opened: {dob}\n"
        f"Sensitive field: {secret}\n"
        f"Branch: Downtown Financial Centre\n"
        f"Relationship manager: J. Harrison\n"
        f"Please provide appropriate next steps for this account.\n"
    )


def _build_enterprise_prompt(system_prefix: str, name: str, dob: str, secret: str, tok) -> str:
    return (
        f"{system_prefix}\n\n"
        f"CONFIDENTIAL HR/PROJECT RECORD\n"
        f"================================\n"
        f"Employee/Project: {name}\n"
        f"Record date: {dob}\n"
        f"Confidential field: {secret}\n"
        f"Department: Human Resources (Restricted)\n"
        f"HR Business Partner: M. Chen\n"
        f"Please advise on the appropriate HR process for this matter.\n"
    )



@dataclass
class KnowledgeLevel:
    name          : str
    description   : str
    first_names   : list[str]
    last_names    : list[str]
    secret_fields : list[str]
    partial_template: bool = False


K1_MEDICAL = KnowledgeLevel(
    name="K1_exact_small",
    description="Exact template, small dictionary (10×10 names, 20 conditions)",
    first_names   = _FIRST_NAMES_MED[:10],
    last_names    = _LAST_NAMES_MED[:10],
    secret_fields = _CONDITIONS_MED_20,
    partial_template=False,
)

K2_MEDICAL = KnowledgeLevel(
    name="K2_exact_large",
    description="Exact template, larger dictionary (30×30 names, 50 conditions)",
    first_names   = _FIRST_NAMES_MED,
    last_names    = _LAST_NAMES_MED,
    secret_fields = _CONDITIONS_MED_50,
    partial_template=False,
)

K3_MEDICAL = KnowledgeLevel(
    name="K3_partial_template",
    description="Partial template knowledge — attacker uses approximate prompt wording",
    first_names   = _FIRST_NAMES_MED[:10],
    last_names    = _LAST_NAMES_MED[:10],
    secret_fields = _CONDITIONS_MED_20,
    partial_template=True,
)

K1_FINANCIAL = KnowledgeLevel(
    name="K1_exact_small",
    description="Exact template, small dictionary",
    first_names   = _CUSTOMER_FIRST[:10],
    last_names    = _CUSTOMER_LAST[:10],
    secret_fields = _FINANCIAL_FIELDS_20,
    partial_template=False,
)
K2_FINANCIAL = KnowledgeLevel(
    name="K2_exact_large",
    description="Exact template, larger dictionary",
    first_names   = _CUSTOMER_FIRST,
    last_names    = _CUSTOMER_LAST,
    secret_fields = _FINANCIAL_FIELDS_50,
    partial_template=False,
)
K3_FINANCIAL = KnowledgeLevel(
    name="K3_partial_template",
    description="Partial template knowledge",
    first_names   = _CUSTOMER_FIRST[:10],
    last_names    = _CUSTOMER_LAST[:10],
    secret_fields = _FINANCIAL_FIELDS_20,
    partial_template=True,
)

K1_ENTERPRISE = KnowledgeLevel(
    name="K1_exact_small",
    description="Exact template, small dictionary",
    first_names   = _EMP_FIRST[:10],
    last_names    = _EMP_LAST[:10],
    secret_fields = _ENTERPRISE_FIELDS_20,
    partial_template=False,
)
K2_ENTERPRISE = KnowledgeLevel(
    name="K2_exact_large",
    description="Exact template, larger dictionary",
    first_names   = _EMP_FIRST,
    last_names    = _EMP_LAST,
    secret_fields = _ENTERPRISE_FIELDS_50,
    partial_template=False,
)
K3_ENTERPRISE = KnowledgeLevel(
    name="K3_partial_template",
    description="Partial template knowledge",
    first_names   = _EMP_FIRST[:10],
    last_names    = _EMP_LAST[:10],
    secret_fields = _ENTERPRISE_FIELDS_20,
    partial_template=True,
)


SCENARIOS: dict[str, ScenarioConfig] = {
    "T1_medical": ScenarioConfig(
        name="T1_medical", domain="medical",
        first_names=_FIRST_NAMES_MED, last_names=_LAST_NAMES_MED,
        secret_fields=_CONDITIONS_MED_20,
        build_prompt=_build_medical_prompt,
        field_label="Primary Diagnosis",
        description="Realistic clinical note: patient name + medical condition",
    ),
    "T2_financial": ScenarioConfig(
        name="T2_financial", domain="financial",
        first_names=_CUSTOMER_FIRST, last_names=_CUSTOMER_LAST,
        secret_fields=_FINANCIAL_FIELDS_20,
        build_prompt=_build_financial_prompt,
        field_label="Sensitive financial field",
        description="Bank/financial record: customer name + sensitive financial field",
    ),
    "T3_enterprise": ScenarioConfig(
        name="T3_enterprise", domain="enterprise",
        first_names=_EMP_FIRST, last_names=_EMP_LAST,
        secret_fields=_ENTERPRISE_FIELDS_20,
        build_prompt=_build_enterprise_prompt,
        field_label="Confidential HR/project field",
        description="HR/project record: employee name + confidential field",
    ),
}

KNOWLEDGE_LEVELS: dict[str, dict[str, KnowledgeLevel]] = {
    "T1_medical"    : {"K1": K1_MEDICAL,    "K2": K2_MEDICAL,    "K3": K3_MEDICAL},
    "T2_financial"  : {"K1": K1_FINANCIAL,  "K2": K2_FINANCIAL,  "K3": K3_FINANCIAL},
    "T3_enterprise" : {"K1": K1_ENTERPRISE, "K2": K2_ENTERPRISE, "K3": K3_ENTERPRISE},
}



def generate_realistic_victims(
    scenario : ScenarioConfig,
    knowledge: KnowledgeLevel,
    n_victims: int,
    seed     : int,
) -> list[dict]:
    """Generate victims for a realistic scenario + knowledge level combo."""
    from datetime import date, timedelta
    rng = random.Random(seed)

    victims = []
    for i in range(n_victims):
        fn     = rng.choice(scenario.first_names[:10])
        ln     = rng.choice(scenario.last_names[:10])
        secret = rng.choice(scenario.secret_fields[:20])
        start  = date(1955, 1, 1)
        dob    = (start + timedelta(days=rng.randint(0, 20000))).isoformat()
        name   = f"{fn} {ln}"
        victims.append({
            "victim_id"      : i,
            "scenario"       : scenario.name,
            "knowledge_level": knowledge.name,
            "name"           : name,
            "dob"            : dob,
            "secret"         : secret,
            "field_label"    : scenario.field_label,
            "attacker_names" : [f"{f} {l}"
                                for f in knowledge.first_names
                                for l in knowledge.last_names],
            "attacker_secrets": knowledge.secret_fields,
            "partial_template": knowledge.partial_template,
        })
    return victims


def candidate_entropy_bits(victim: dict) -> float:
    """H0 = log2(|names| × |secrets|) for this victim's attacker knowledge."""
    import math
    n = len(victim["attacker_names"]) * len(victim["attacker_secrets"])
    return math.log2(max(n, 1))
