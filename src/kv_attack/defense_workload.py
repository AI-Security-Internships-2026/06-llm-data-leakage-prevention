"""
kv_attack.defense_workload
===========================
Issue #7 — Labeled synthetic workload for defense evaluation.

Generates a reproducible set of prompts across five sensitivity categories:
  NON_SENSITIVE  – general knowledge / task prompts, no personal data
  CONTACT        – prompts containing names, emails, phone numbers
  MEDICAL        – prompts containing patient names + medical conditions
  FINANCIAL      – prompts containing IBAN, credit card, SSN, bank account
  ENTERPRISE     – prompts containing employee/project + confidential data

Each sample carries a ground-truth sensitivity label that the defense
evaluation uses to measure:
  - Detector recall per category (did the detector catch it?)
  - Policy isolation rate per category (did the policy isolate it?)
  - Residual leakage per category (did the attack succeed despite defense?)

Usage
-----
    from kv_attack.defense_workload import generate_defense_workload
    samples = generate_defense_workload(n_per_category=10, seed=42)
    for s in samples:
        print(s.category, s.expected_risk_level, s.text[:60])
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List


CAT_NON_SENSITIVE = "NON_SENSITIVE"
CAT_CONTACT       = "CONTACT"
CAT_MEDICAL       = "MEDICAL"
CAT_FINANCIAL     = "FINANCIAL"
CAT_ENTERPRISE    = "ENTERPRISE"

ALL_CATEGORIES = [
    CAT_NON_SENSITIVE,
    CAT_CONTACT,
    CAT_MEDICAL,
    CAT_FINANCIAL,
    CAT_ENTERPRISE,
]

EXPECTED_RISK: dict[str, str] = {
    CAT_NON_SENSITIVE: "LOW",
    CAT_CONTACT:       "MEDIUM",
    CAT_MEDICAL:       "HIGH",
    CAT_FINANCIAL:     "HIGH",
    CAT_ENTERPRISE:    "HIGH",
}

SHOULD_ISOLATE: dict[str, bool] = {
    CAT_NON_SENSITIVE: False,
    CAT_CONTACT:       False,
    CAT_MEDICAL:       True,
    CAT_FINANCIAL:     True,
    CAT_ENTERPRISE:    True,
}



@dataclass
class WorkloadSample:
    """One labeled synthetic prompt."""
    sample_id:            str
    category:             str
    expected_risk_level:  str
    should_isolate:       bool
    text:                 str
    ground_truth_entities: List[str]

    def to_dict(self) -> dict:
        return {
            "sample_id":             self.sample_id,
            "category":              self.category,
            "expected_risk_level":   self.expected_risk_level,
            "should_isolate":        self.should_isolate,
            "text":                  self.text,
            "ground_truth_entities": self.ground_truth_entities,
        }



_NON_SENSITIVE_TEMPLATES = [
    "Explain the concept of transformer attention mechanisms in machine learning.",
    "What are the main differences between supervised and unsupervised learning?",
    "Summarise the key events of the French Revolution in three paragraphs.",
    "Write a Python function that computes the Fibonacci sequence up to N.",
    "What is the capital city of Argentina, and what is it known for?",
    "Describe the water cycle and its importance for Earth's ecosystem.",
    "How does the TCP/IP protocol stack work in computer networking?",
    "What are the advantages of using renewable energy sources?",
    "Explain the difference between RAM and ROM in computer architecture.",
    "Write a brief introduction to the history of the Roman Empire.",
    "What is quantum entanglement and why does it matter for computing?",
    "List the main programming paradigms and give one example of each.",
    "How do solar panels convert sunlight into electrical energy?",
    "Describe the structure of DNA and its role in genetic inheritance.",
    "What are the main causes of inflation in modern economies?",
]

_CONTACT_TEMPLATES = [
    "Please contact John Smith at john.smith@acme.com or call +1-555-234-5678 to schedule a meeting.",
    "Our support team member Alice Johnson can be reached at alice.j@helpdesk.org or (800) 555-0101.",
    "Forward this invoice to billing@globalcorp.io; for questions, call Robert Davis at 212-555-0198.",
    "The project lead is Emily Chen — email her at e.chen@techinc.net or ring +44 20 7946 0234.",
    "Delivery confirmation will be sent to maria.garcia AT courier DOT com; phone: 0800-555-0134.",
    "CC the legal team at legal@lawfirm.co.uk; primary contact: David Wilson, +1 415 555 0187.",
    "You can reach the account manager, Sarah Brown, at s.brown@finance.com or +1-312-555-0167.",
    "For billing enquiries, email accounts@bigco.com or call James Taylor on (646) 555-0122.",
    "Notify Patricia Moore at p.moore@healthnet.org or +1 303 555 0189 before proceeding.",
    "The new hire, Michael Anderson, starts Monday — his email is m.anderson@newco.com.",
]

_MEDICAL_TEMPLATES = [
    "Patient Mary Johnson, DOB 1975-04-22, presents with a confirmed diagnosis of Type 2 diabetes mellitus requiring insulin therapy.",
    "Record update for James Williams (DOB 1962-11-08): severe hypertension; prescribed lisinopril 10 mg daily.",
    "Linda Martinez, born 1988-07-14, has a documented penicillin allergy; new diagnosis of community-acquired pneumonia.",
    "Barbara Davis (DOB 1950-03-31) admitted for acute myocardial infarction; cardiac catheterisation scheduled for 08:00.",
    "Patient: William Thompson, DOB 1944-09-19. Condition: stage 3 chronic kidney disease. Nephrologist referral issued.",
    "Susan Rodriguez, 1971-06-05, post-operative follow-up after laparoscopic cholecystectomy; no complications noted.",
    "Charles Lee (DOB 1985-02-28): anxiety disorder and major depressive episode; starting sertraline 50 mg.",
    "Dorothy Harris, born 1958-12-17, presents with new-onset atrial fibrillation; anticoagulation therapy initiated.",
    "Patient Kevin Clark (DOB 2001-08-09): asthma exacerbation; salbutamol nebuliser administered in ED.",
    "Margaret Lewis (DOB 1967-10-24): rheumatoid arthritis flare; methotrexate dosage adjusted to 15 mg/week.",
]

_FINANCIAL_TEMPLATES = [
    "Transfer EUR 12,500 from account GB29NWBK60161331926819 to vendor ref V-4421; authorised by card 4532015112830366.",
    "Wire USD 8,000 to IBAN DE89370400440532013000; SSN of account holder: 078-05-1120; confirm receipt.",
    "Customer credit card 5500005555555559 (exp 09/27, CVV 456) authorised for subscription renewal at $299/mo.",
    "Bank transfer pending: source IBAN FR7630006000011234567890189; destination account US54BB8901023456789012.",
    "Invoice settled via card number 4111111111111111; billing reference BR-2026-0094; amount: £3,750.",
    "Employee reimbursement: account GB82WEST12345698765432; SSN 123-45-6789; amount: $4,200 approved.",
    "Mortgage disbursement to account ES9121000418450200051332; loan ID LN-9982; monthly instalment EUR 1,456.",
    "Client bank details: account number US00ZZ0000098765432101; routing 021000089; balance trigger: $50,000.",
    "Refund authorised to card 371449635398431 (Amex); transaction ID TX-88234; net: $1,840.",
    "Payroll batch: IBAN IT60X0542811101000000123456; monthly salary EUR 5,200; tax ref IT-TAX-2026-0042.",
]

_ENTERPRISE_TEMPLATES = [
    "Project Falcon (CONFIDENTIAL): lead engineer Robert Chen in Austin is targeting Q4 revenue of $8.2M; do not distribute.",
    "Internal memo — Employee Jennifer Park (Chicago office) is transitioning to Project Sigma; budget: $1.5M restricted.",
    "Strategic plan RESTRICTED: David Kim's team in Seattle is negotiating a $22M acquisition of Starteco Ltd.",
    "HR CONFIDENTIAL: Patricia Nguyen (London) flagged for PIP; manager review with Michael Torres scheduled this Friday.",
    "Board briefing EYES ONLY: Linda Shah (CEO, New York) approved headcount reduction of 120 roles in Q1 2027.",
    "Project Orion CLASSIFIED: Christopher Evans (Berlin) holds master credentials for DataVault production environment.",
    "Restricted: Nancy Yamamoto (Tokyo) is the sole signatory for account APAC-EAST-2026; limit: ¥85,000,000.",
    "Confidential deal memo: Thomas Müller (Frankfurt) leading bid on Vega Corp — valuation estimate EUR 340M.",
    "INTERNAL ONLY: Sandra Kowalski (Warsaw) has admin rights to the zero-trust policy engine; rotation overdue.",
    "Exec summary CONFIDENTIAL: Brian Okafor (Lagos) approves offshore entity dissolution; legal cost: $2.8M.",
]

_GROUND_TRUTH_ENTITIES: dict[str, List[str]] = {
    CAT_NON_SENSITIVE: [],
    CAT_CONTACT:       ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"],
    CAT_MEDICAL:       ["PERSON", "MEDICAL_LICENSE"],
    CAT_FINANCIAL:     ["CREDIT_CARD", "IBAN_CODE", "US_SSN"],
    CAT_ENTERPRISE:    ["PERSON", "LOCATION"],
}



def generate_defense_workload(
    n_per_category: int = 10,
    seed:           int = 42,
) -> List[WorkloadSample]:
    """
    Generate a reproducible labeled synthetic workload.

    Parameters
    ----------
    n_per_category : number of samples per sensitivity category
    seed           : random seed for reproducibility

    Returns
    -------
    List[WorkloadSample] — total length = n_per_category × len(ALL_CATEGORIES)
    """
    rng = random.Random(seed)
    samples: List[WorkloadSample] = []

    template_banks: dict[str, List[str]] = {
        CAT_NON_SENSITIVE: _NON_SENSITIVE_TEMPLATES,
        CAT_CONTACT:       _CONTACT_TEMPLATES,
        CAT_MEDICAL:       _MEDICAL_TEMPLATES,
        CAT_FINANCIAL:     _FINANCIAL_TEMPLATES,
        CAT_ENTERPRISE:    _ENTERPRISE_TEMPLATES,
    }

    for cat in ALL_CATEGORIES:
        bank = template_banks[cat]

        if n_per_category <= len(bank):
            indices = rng.sample(range(len(bank)), n_per_category)
            chosen  = [bank[i] for i in indices]
            chosen_texts = chosen
        else:
            chosen_texts = []
            for i in range(n_per_category):
                base = bank[i % len(bank)]
                chosen_texts.append(f"{base}  [sid:{cat[:3].lower()}-{i:03d}]")

        for idx, text in enumerate(chosen_texts):
            sample_id = f"{cat.lower()[:4]}-{idx:03d}"
            samples.append(
                WorkloadSample(
                    sample_id             = sample_id,
                    category              = cat,
                    expected_risk_level   = EXPECTED_RISK[cat],
                    should_isolate        = SHOULD_ISOLATE[cat],
                    text                  = text,
                    ground_truth_entities = list(_GROUND_TRUTH_ENTITIES[cat]),
                )
            )

    rng.shuffle(samples)
    return samples


def workload_to_dict(samples: List[WorkloadSample]) -> dict:
    """Serialise a workload to a JSON-safe dict."""
    by_category: dict[str, int] = {}
    for s in samples:
        by_category[s.category] = by_category.get(s.category, 0) + 1
    return {
        "total_samples":    len(samples),
        "by_category":      by_category,
        "samples":          [s.to_dict() for s in samples],
    }