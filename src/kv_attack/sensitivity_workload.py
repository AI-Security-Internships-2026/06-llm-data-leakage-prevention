"""
kv_attack.sensitivity_workload
================================
Issue #9 — Versioned ground-truth sensitivity workload for PII-detector
error-propagation experiments.

Extends Issue-7 defense_workload.py with:
  - Workload versioning (v1.0) for result traceability
  - Supported vs unsupported PII category flags
    (Presidio CAN detect FINANCIAL/CONTACT; CANNOT detect MEDICAL_CONDITION)
  - Field-level ground-truth labels (which specific PII fields are present)
  - Controlled FN/FP injection for threshold sweep experiments
  - Category-level leakage tracking

Sensitivity categories
----------------------
  NON_SENSITIVE  – no PII; baseline clean prompts
  CONTACT        – PERSON, EMAIL, PHONE  [SUPPORTED by Presidio]
  FINANCIAL      – CREDIT_CARD, IBAN, SSN  [SUPPORTED by Presidio]
  MEDICAL        – patient name + medical condition
                   Name field [SUPPORTED], condition [UNSUPPORTED]
  ENTERPRISE     – PERSON + LOCATION in confidential context
                   [PARTIALLY SUPPORTED — detected as MEDIUM not HIGH]

Supported vs unsupported
------------------------
A PII category is SUPPORTED when Presidio's entity recognisers can detect
it at HIGH confidence (score >= 0.70) and classify it as HIGH risk.
MEDICAL_CONDITION is explicitly unsupported because Presidio has no
MEDICAL_CONDITION entity type — it detects the patient's NAME but not
the diagnosis, so medical prompts are classified as MEDIUM (not HIGH).

Controlled error injection
--------------------------
  inject_fn(samples, fnr, rng)
      Flip `fnr` fraction of truly-sensitive samples to "predicted LOW"
      (simulating detector miss — false negative).

  inject_fp(samples, fpr, rng)
      Flip `fpr` fraction of truly-clean samples to "predicted HIGH"
      (simulating false alarm — false positive).

Usage
-----
    from kv_attack.sensitivity_workload import (
        build_ground_truth_workload, inject_fn, inject_fp, WORKLOAD_VERSION,
    )
    samples = build_ground_truth_workload(n_per_category=10, seed=42)
    fn_samples = inject_fn(samples, fnr=0.20, rng=random.Random(42))
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import List, Optional

WORKLOAD_VERSION = "1.0"


CAT_NON_SENSITIVE = "NON_SENSITIVE"
CAT_CONTACT       = "CONTACT"
CAT_MEDICAL       = "MEDICAL"
CAT_FINANCIAL     = "FINANCIAL"
CAT_ENTERPRISE    = "ENTERPRISE"

ALL_CATEGORIES = [
    CAT_NON_SENSITIVE, CAT_CONTACT, CAT_MEDICAL, CAT_FINANCIAL, CAT_ENTERPRISE,
]

GT_RISK: dict[str, str] = {
    CAT_NON_SENSITIVE: "LOW",
    CAT_CONTACT:       "MEDIUM",
    CAT_MEDICAL:       "HIGH",
    CAT_FINANCIAL:     "HIGH",
    CAT_ENTERPRISE:    "HIGH",
}

PII_SUPPORTED: dict[str, bool] = {
    CAT_NON_SENSITIVE: True,
    CAT_CONTACT:       True,
    CAT_FINANCIAL:     True,
    CAT_MEDICAL:       False,
    CAT_ENTERPRISE:    False,
}

GT_FIELDS: dict[str, List[str]] = {
    CAT_NON_SENSITIVE: [],
    CAT_CONTACT:       ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"],
    CAT_FINANCIAL:     ["CREDIT_CARD", "IBAN_CODE", "US_SSN"],
    CAT_MEDICAL:       ["PERSON", "MEDICAL_CONDITION"],
    CAT_ENTERPRISE:    ["PERSON", "LOCATION", "CONFIDENTIAL_CONTEXT"],
}

SHOULD_ISOLATE: dict[str, bool] = {
    CAT_NON_SENSITIVE: False,
    CAT_CONTACT:       False,
    CAT_FINANCIAL:     True,
    CAT_MEDICAL:       True,
    CAT_ENTERPRISE:    True,
}



@dataclass
class GTSample:
    """One ground-truth labeled sample."""
    sample_id:              str
    category:               str
    gt_risk_level:          str
    pii_supported:          bool
    should_isolate:         bool
    gt_fields:              List[str]
    text:                   str
    predicted_risk_level:   Optional[str] = None
    error_type:             Optional[str] = None
    controlled_error:       bool = False

    def to_dict(self) -> dict:
        return {
            "sample_id":            self.sample_id,
            "category":             self.category,
            "gt_risk_level":        self.gt_risk_level,
            "pii_supported":        self.pii_supported,
            "should_isolate":       self.should_isolate,
            "gt_fields":            self.gt_fields,
            "text":                 self.text,
            "predicted_risk_level": self.predicted_risk_level,
            "error_type":           self.error_type,
            "controlled_error":     self.controlled_error,
        }



_NON_SENSITIVE = [
    "Explain the difference between supervised and unsupervised learning.",
    "What are the advantages of renewable energy sources?",
    "Describe the structure of DNA and its role in inheritance.",
    "How does the TCP/IP protocol stack work?",
    "What are the main causes of inflation in modern economies?",
    "Explain quantum entanglement and its relevance to computing.",
    "Write a Python function that computes the Fibonacci sequence.",
    "Summarise the key events of the French Revolution.",
    "How do solar panels convert sunlight into electricity?",
    "List the main programming paradigms with one example each.",
    "What is the difference between RAM and ROM?",
    "Explain how vaccines work at the cellular level.",
    "What is the greenhouse effect and why does it matter?",
    "Describe the water cycle and its ecosystem importance.",
    "How does a relational database differ from a document store?",
]

_CONTACT = [
    "Please contact John Smith at john.smith@acme.com or call +1-555-234-5678.",
    "The project lead Emily Chen can be reached at e.chen@techinc.net or +44 20 7946 0234.",
    "CC the legal team at legal@lawfirm.co.uk; primary contact: David Wilson, +1 415 555 0187.",
    "Forward queries to billing@globalcorp.io; call Robert Davis at 212-555-0198.",
    "New hire Michael Anderson starts Monday — email m.anderson@newco.com.",
    "Sarah Brown is reachable at s.brown@finance.com or +1-312-555-0167.",
    "For accounts, email accounts@bigco.com or call James Taylor on (646) 555-0122.",
    "Patricia Moore: p.moore@healthnet.org or +1 303 555 0189.",
    "Alice Johnson: alice.j@helpdesk.org or (800) 555-0101.",
    "Send invoice to billing@startup.io; contact: Kevin Lee, +1 628 555 0144.",
]

_MEDICAL = [
    "Patient Mary Johnson, DOB 1975-04-22, presents with Type 2 diabetes mellitus requiring insulin.",
    "James Williams (DOB 1962-11-08): severe hypertension; prescribed lisinopril 10 mg daily.",
    "Linda Martinez, born 1988-07-14, penicillin allergy; community-acquired pneumonia diagnosed.",
    "Barbara Davis (DOB 1950-03-31) admitted for acute myocardial infarction; cath scheduled.",
    "William Thompson, DOB 1944-09-19: stage 3 chronic kidney disease. Nephrologist referral issued.",
    "Susan Rodriguez, 1971-06-05, post-op follow-up after laparoscopic cholecystectomy; no issues.",
    "Charles Lee (DOB 1985-02-28): anxiety disorder and major depressive episode; starting sertraline.",
    "Dorothy Harris, born 1958-12-17, new-onset atrial fibrillation; anticoagulation initiated.",
    "Kevin Clark (DOB 2001-08-09): asthma exacerbation; salbutamol nebuliser administered in ED.",
    "Margaret Lewis (DOB 1967-10-24): rheumatoid arthritis flare; methotrexate adjusted to 15 mg/week.",
]

_FINANCIAL = [
    "Transfer EUR 12,500 from GB29NWBK60161331926819 to vendor; card 4532015112830366.",
    "Wire USD 8,000 to IBAN DE89370400440532013000; SSN of holder: 078-05-1120.",
    "Card 5500005555555559 (exp 09/27, CVV 456) authorised for subscription renewal at $299/mo.",
    "Transfer from IBAN FR7630006000011234567890189 to US54BB8901023456789012.",
    "Invoice settled via card 4111111111111111; billing ref BR-2026-0094; amount £3,750.",
    "Reimbursement: account GB82WEST12345698765432; SSN 123-45-6789; amount $4,200.",
    "Mortgage to account ES9121000418450200051332; loan ID LN-9982; instalment EUR 1,456.",
    "Client details: US00ZZ0000098765432101; routing 021000089; balance trigger $50,000.",
    "Refund to card 371449635398431 (Amex); transaction TX-88234; net $1,840.",
    "Payroll: IBAN IT60X0542811101000000123456; monthly EUR 5,200; tax ref IT-TAX-2026-0042.",
]

_ENTERPRISE = [
    "Project Falcon (CONFIDENTIAL): engineer Robert Chen in Austin targeting Q4 revenue $8.2M.",
    "Internal: Jennifer Park (Chicago) transitions to Project Sigma; budget $1.5M restricted.",
    "RESTRICTED: David Kim's team in Seattle negotiating $22M acquisition of Starteco Ltd.",
    "HR CONFIDENTIAL: Patricia Nguyen (London) flagged for PIP; review with Michael Torres Friday.",
    "Board EYES ONLY: Linda Shah (CEO, New York) approved headcount reduction of 120 roles Q1.",
    "Project Orion CLASSIFIED: Christopher Evans (Berlin) holds master credentials for DataVault.",
    "Restricted: Nancy Yamamoto (Tokyo) sole signatory for APAC-EAST-2026; limit ¥85,000,000.",
    "Deal memo CONFIDENTIAL: Thomas Müller (Frankfurt) leading bid on Vega Corp; EUR 340M.",
    "INTERNAL ONLY: Sandra Kowalski (Warsaw) has admin rights to zero-trust policy engine.",
    "Exec summary CONFIDENTIAL: Brian Okafor (Lagos) approves offshore entity dissolution; $2.8M.",
]

_BANKS: dict[str, List[str]] = {
    CAT_NON_SENSITIVE: _NON_SENSITIVE,
    CAT_CONTACT:       _CONTACT,
    CAT_MEDICAL:       _MEDICAL,
    CAT_FINANCIAL:     _FINANCIAL,
    CAT_ENTERPRISE:    _ENTERPRISE,
}



def build_ground_truth_workload(
    n_per_category: int = 10,
    seed:           int = 42,
    version:        str = WORKLOAD_VERSION,
) -> List[GTSample]:
    """
    Build a reproducible versioned ground-truth workload.

    Parameters
    ----------
    n_per_category : samples per category (5 categories → total = 5 × n)
    seed           : random seed for reproducibility
    version        : workload version tag (stored in metadata)

    Returns
    -------
    List[GTSample] — shuffled, with unique texts (no cross-category duplicates)
    """
    rng     = random.Random(seed)
    samples: List[GTSample] = []

    for cat in ALL_CATEGORIES:
        bank = _BANKS[cat]
        if n_per_category <= len(bank):
            texts = [bank[i] for i in rng.sample(range(len(bank)), n_per_category)]
        else:
            texts = []
            for i in range(n_per_category):
                base = bank[i % len(bank)]
                texts.append(f"{base}  [sid:{cat[:3].lower()}-{i:03d}]")

        for idx, text in enumerate(texts):
            samples.append(GTSample(
                sample_id            = f"{cat.lower()[:4]}-{idx:03d}",
                category             = cat,
                gt_risk_level        = GT_RISK[cat],
                pii_supported        = PII_SUPPORTED[cat],
                should_isolate       = SHOULD_ISOLATE[cat],
                gt_fields            = list(GT_FIELDS[cat]),
                text                 = text,
                predicted_risk_level = None,
                error_type           = None,
                controlled_error     = False,
            ))

    rng.shuffle(samples)
    return samples



def inject_fn(
    samples:    List[GTSample],
    fnr:        float,
    rng:        random.Random,
) -> List[GTSample]:
    """
    Inject controlled false negatives: flip `fnr` fraction of truly-sensitive
    samples (gt_risk_level HIGH or MEDIUM) to predicted_risk_level = "LOW".

    Parameters
    ----------
    samples : ground-truth workload (not mutated — returns a deep copy)
    fnr     : false-negative rate in [0.0, 1.0]
    rng     : random instance for reproducibility
    """
    result  = copy.deepcopy(samples)
    targets = [s for s in result if s.gt_risk_level in ("HIGH", "MEDIUM")]
    n_flip  = max(0, round(fnr * len(targets)))
    chosen  = rng.sample(targets, min(n_flip, len(targets)))
    chosen_ids = {s.sample_id for s in chosen}
    for s in result:
        if s.sample_id in chosen_ids:
            s.predicted_risk_level = "LOW"
            s.error_type           = "FN"
            s.controlled_error     = True
    return result


def inject_fp(
    samples:    List[GTSample],
    fpr_rate:   float,
    rng:        random.Random,
) -> List[GTSample]:
    """
    Inject controlled false positives: flip `fpr_rate` fraction of truly-clean
    samples (gt_risk_level LOW) to predicted_risk_level = "HIGH".

    Parameters
    ----------
    samples  : ground-truth workload (not mutated — returns a deep copy)
    fpr_rate : false-positive rate in [0.0, 1.0]
    rng      : random instance for reproducibility
    """
    result  = copy.deepcopy(samples)
    targets = [s for s in result if s.gt_risk_level == "LOW"]
    n_flip  = max(0, round(fpr_rate * len(targets)))
    chosen  = rng.sample(targets, min(n_flip, len(targets)))
    chosen_ids = {s.sample_id for s in chosen}
    for s in result:
        if s.sample_id in chosen_ids:
            s.predicted_risk_level = "HIGH"
            s.error_type           = "FP"
            s.controlled_error     = True
    return result


def workload_metadata(
    samples: List[GTSample],
    version: str = WORKLOAD_VERSION,
) -> dict:
    """Serialise workload metadata for result files."""
    by_cat: dict[str, dict] = {}
    for s in samples:
        c = s.category
        if c not in by_cat:
            by_cat[c] = {"n": 0, "n_fn": 0, "n_fp": 0, "supported": PII_SUPPORTED[c]}
        by_cat[c]["n"] += 1
        if s.error_type == "FN": by_cat[c]["n_fn"] += 1
        if s.error_type == "FP": by_cat[c]["n_fp"] += 1
    return {
        "version":      version,
        "n_total":      len(samples),
        "n_controlled_errors": sum(1 for s in samples if s.controlled_error),
        "by_category":  by_cat,
        "samples":      [s.to_dict() for s in samples],
    }