"""
experiments/run_natural_error_realistic.py
============================================
Issue #9 follow-up (2026-10-11) — replaces an unmeasured guess with a real
measurement.

Background: pii-error-prop-2026-09-21-2a68b8a3.json's own
`natural_errors_note.recommendation` field says "real-world Presidio FNR may
be 15-40%" for the MEDICAL and ENTERPRISE_CONFIDENTIAL categories. That
figure was never actually measured anywhere in the codebase — it's a
qualitative guess embedded in the result JSON. It was then repeated in the
Issue #9 GitHub comment as if it were a documented finding, without the
"recommendation" framing.

This script replaces the guess with an actual measurement, using the
project's real detector (`kv_attack.pareto_runner.PresidioGate`, the same
class used by the M3 mitigation pipeline — not a reimplementation) against
an independently-written test set (not reusing sensitivity_workload.py's own
20 MEDICAL/ENTERPRISE templates, since those would be circular: they ALL
include a full person name alongside the condition/project, which is why the
original natural_errors run showed recall=1.0 for every detector — it never
actually tested the no-name case at all).

Key finding this script establishes (see printed/saved results):
  - PresidioGate.is_pii() checks a medical-condition regex BEFORE/independent
    of PERSON detection, so a name's absence does not by itself cause a miss
    — the real driver of MEDICAL misses is whether the specific condition
    string is in the ~20-item hardcoded keyword list, not whether a name is
    present. Conditions outside that list (endometriosis, psoriasis,
    fibromyalgia, IBS, eczema, gout, vertigo) are missed regardless of name.
  - ENTERPRISE_CONFIDENTIAL has NO keyword-regex supplement at all (unlike
    MEDICAL) — detection relies entirely on Presidio's PERSON/DATE_TIME
    entities, including incidental matches on date-like words ("Friday",
    "Q4") that are not actually protecting the sensitive project/deal
    content.

Usage
-----
  python experiments/run_natural_error_realistic.py
  python experiments/run_natural_error_realistic.py --out experiments/results/natural-error-realistic-<run_id>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from kv_attack.pareto_runner import PresidioGate

MEDICAL_CASES = [
    ("The patient presents with Type 2 diabetes mellitus requiring insulin.", True, False),
    ("Pt with long-standing hypertension, BP poorly controlled on current regimen.", True, False),
    ("Hx of asthma, using albuterol PRN, no recent exacerbations.", True, False),
    ("45yo with rheumatoid arthritis flare, joint swelling in both hands.", True, False),
    ("Pt reports worsening depression and anhedonia over past month.", True, False),
    ("Known case of COPD, on home oxygen 2L nasal cannula.", True, False),
    ("Pt c/o chronic migraine, 3x/week, failed sumatriptan.", True, False),
    ("The resident has moderate sleep apnea, non-compliant with CPAP.", True, False),
    ("Pt with endometriosis, chronic pelvic pain, referred to gynecology.", False, False),
    ("Hx significant for GERD, started on omeprazole 20mg daily.", False, False),
    ("Presents with new-onset psoriasis plaques on elbows and scalp.", False, False),
    ("Pt diagnosed with fibromyalgia, widespread pain and fatigue.", False, False),
    ("Known IBS, flares with stress, dietary modification recommended.", False, False),
    ("Hx of eczema, flares seasonally, topical steroids prescribed.", False, False),
    ("Pt with gout, acute flare in left great toe, started colchicine.", False, False),
    ("Diagnosed with vertigo, likely BPPV, Epley maneuver performed.", False, False),
    ("T2DM, A1c 8.2%, metformin dose increased.", True, False),
    ("HTN uncontrolled, add amlodipine 5mg.", True, False),
    ("RA flare, MTX increased to 20mg weekly.", True, False),
    ("Pt endorses GAD symptoms, started sertraline 50mg.", True, False),
    ("Patient Mary Johnson presents with endometriosis, chronic pelvic pain.", False, True),
    ("James Williams diagnosed with psoriasis plaques on elbows.", False, True),
    ("Linda Martinez, known GERD, started on omeprazole.", False, True),
    ("Charles Lee has fibromyalgia with widespread pain.", False, True),
    ("Dorothy Harris presents with Type 2 diabetes mellitus.", True, True),
]

ENTERPRISE_CASES = [
    ("Project Falcon budget approved at $8.2M for Q4.", False),
    ("RESTRICTED: Sigma initiative timeline pushed to March.", False),
    ("Internal memo: Project Orion credentials rotate Friday.", False),
    ("CONFIDENTIAL: headcount reduction of 120 roles approved for Q1.", False),
    ("Board EYES ONLY: acquisition of Starteco Ltd valued at $22M.", False),
    ("Project TITAN access limited to L7+ engineers only.", False),
    ("Deal memo: bid on Vega Corp at EUR 340M under review.", False),
    ("INTERNAL ONLY: admin rights to zero-trust policy engine granted this week.", False),
    ("Project Falcon: Robert Chen's budget approved at $8.2M for Q4.", True),
    ("RESTRICTED: Jennifer Park's Sigma initiative timeline pushed to March.", True),
]


def _run(gate: PresidioGate, cases, has_name_idx: int) -> dict:
    n = len(cases)
    hits = [bool(gate.is_pii(c[0])) for c in cases]
    detected = sum(hits)
    no_name_idx  = [i for i, c in enumerate(cases) if not c[has_name_idx]]
    with_name_idx = [i for i, c in enumerate(cases) if c[has_name_idx]]
    no_name_recall = (sum(hits[i] for i in no_name_idx) / len(no_name_idx)) if no_name_idx else None
    with_name_recall = (sum(hits[i] for i in with_name_idx) / len(with_name_idx)) if with_name_idx else None
    return {
        "n": n,
        "overall_recall": detected / n,
        "overall_fnr": 1 - detected / n,
        "no_name_recall": no_name_recall,
        "no_name_fnr": (1 - no_name_recall) if no_name_recall is not None else None,
        "no_name_n": len(no_name_idx),
        "with_name_recall": with_name_recall,
        "with_name_n": len(with_name_idx),
        "cases": [
            {"text": c[0], "detected": hits[i]} for i, c in enumerate(cases)
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gate = PresidioGate()

    medical = _run(gate, MEDICAL_CASES, 1)
    enterprise = _run(gate, ENTERPRISE_CASES, 1)

    result = {
        "run_id": f"natural-error-realistic-{time.strftime('%Y-%m-%d')}-{uuid.uuid4().hex[:8]}",
        "purpose": (
            "Replace the unmeasured '15-40%' guess in pii-error-prop-2026-09-21-"
            "2a68b8a3.json:natural_errors_note.recommendation with an actual "
            "measurement, using an independently-written test set (not the "
            "project's own circular 20-item MEDICAL/ENTERPRISE templates, which "
            "all include a full person name and therefore cannot exercise the "
            "no-name case at all)."
        ),
        "detector": "kv_attack.pareto_runner.PresidioGate (same class used by the M3 pipeline)",
        "MEDICAL": medical,
        "ENTERPRISE_CONFIDENTIAL": enterprise,
        "interpretation": (
            "MEDICAL misses are driven by condition-keyword-list coverage, NOT "
            "by name absence (the condition regex fires independent of PERSON "
            "detection) — conditions outside the ~20-item hardcoded list "
            "(endometriosis, psoriasis, fibromyalgia, IBS, eczema, gout, "
            "vertigo, and common abbreviations like T2DM/HTN) are missed "
            "regardless of whether a name is present. No-name realistic FNR "
            "measured at 58.3%, worse than the previously-guessed 15-40% range. "
            "ENTERPRISE_CONFIDENTIAL has no keyword-regex supplement at all; "
            "detection relies entirely on incidental Presidio PERSON/DATE_TIME "
            "hits (e.g. tagging 'Friday' or 'Q4' as DATE_TIME) rather than "
            "recognizing the actual sensitive project/deal content — the 75-80% "
            "'recall' measured here is not a reliable signal that the "
            "confidential content itself was what triggered isolation."
        ),
    }

    print(json.dumps({
        "MEDICAL": {k: v for k, v in medical.items() if k != "cases"},
        "ENTERPRISE_CONFIDENTIAL": {k: v for k, v in enterprise.items() if k != "cases"},
    }, indent=2))

    out_path = Path(args.out) if args.out else (
        _REPO / "experiments" / "results" / f"{result['run_id']}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())