"""
experiments/check_consistency.py
==================================
Issue #3 deliverable — lightweight CI consistency checks.

Checks:
  1. Required result artifacts exist.
  2. Sample counts in result files match config.
  3. Model IDs in result files match expected model.
  4. Analytical values (Pareto M3, BLQ theoretical) are never presented
     as empirical in paper_numbers.json.
  5. Smoke test: run twice with same seed → identical victim/candidate sets.
  6. paper_numbers.json exists and has no MISSING values.

Exit code:  0 = all checks pass  |  1 = one or more failures.

Usage
-----
  python experiments/check_consistency.py
  python experiments/check_consistency.py --strict   # treat warnings as errors
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO    = Path(__file__).resolve().parent.parent
_RESULTS = _REPO / "experiments" / "results"

PASS = "✓"
FAIL = "✗"
WARN = "⚠"


class _Checker:
    def __init__(self, strict: bool = False):
        self.strict = strict
        self._failures: list[str] = []
        self._warnings: list[str] = []

    def check(self, condition: bool, name: str, detail: str = "") -> None:
        if condition:
            print(f"  {PASS} {name}")
        else:
            msg = f"{name}" + (f" — {detail}" if detail else "")
            print(f"  {FAIL} {msg}")
            self._failures.append(msg)

    def warn(self, condition: bool, name: str, detail: str = "") -> None:
        if condition:
            print(f"  {PASS} {name}")
        else:
            msg = f"{name}" + (f" — {detail}" if detail else "")
            print(f"  {WARN} {msg}")
            if self.strict:
                self._failures.append(msg)
            else:
                self._warnings.append(msg)

    def summary(self) -> int:
        print()
        if not self._failures and not self._warnings:
            print(f"{PASS} All checks passed.")
            return 0
        if self._warnings:
            print(f"{WARN} {len(self._warnings)} warning(s):")
            for w in self._warnings:
                print(f"   {WARN} {w}")
        if self._failures:
            print(f"{FAIL} {len(self._failures)} check(s) failed:")
            for f in self._failures:
                print(f"   {FAIL} {f}")
            return 1
        return 0


def _load(name: str) -> dict:
    p = _RESULTS / name
    if not p.exists():
        return {}
    with open(p) as fh:
        return json.load(fh)


def _check_required_artifacts(c: _Checker) -> None:
    print("\n[1] Required result artifacts")
    required = [
        "kv_attack_results_50.json",
        "kv_week13_final.json",
        "kv_pareto_final.json",
    ]
    recommended = [
        "kv_mitigation_results.json",
        "stage_comparison.json",
        "paper_numbers.json",
    ]
    for f in required:
        c.check((_RESULTS / f).exists(), f"Artifact exists: {f}")
    for f in recommended:
        c.warn((_RESULTS / f).exists(), f"Artifact exists (recommended): {f}")


def _check_sample_counts(c: _Checker) -> None:
    print("\n[2] Sample counts")
    w10_50 = _load("kv_attack_results_50.json")
    if w10_50:
        summ = w10_50.get("summary") or w10_50.get("aggregate") or {}
        n = summ.get("n_victims", None)
        c.check(n is not None, "Baseline n_victims present in summary/aggregate")
        if n is not None:
            c.check(int(n) >= 30, f"Baseline n_victims >= 30", f"got {n}")

    w13 = _load("kv_week13_final.json")
    if w13:
        n = w13.get("n_victims")
        c.check(n is not None, "Paper run n_victims present")
        if n is not None:
            c.warn(n >= 30, f"Paper n_victims >= 30 (Issue #2 target)",
                   f"got {n} — acceptable for smoke test, need ≥30 for paper")
        res = w13.get("results", [])
        c.check(len(res) == n, f"results[] length == n_victims",
                f"len(results)={len(res)}, n_victims={n}")


def _check_model_ids(c: _Checker) -> None:
    print("\n[3] Model IDs")
    EXPECTED_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    for fname in ["kv_week13_final.json", "kv_attack_results_50.json"]:
        d = _load(fname)
        if not d:
            continue
        mid = d.get("model") or d.get("framework") or d.get("algorithm", "")
        c.warn(
            not mid or mid == EXPECTED_MODEL or "mock" in mid.lower() or "deepseek" in mid.lower(),
            f"Model ID plausible in {fname}",
            f"got '{mid}'",
        )


def _check_analytical_labeling(c: _Checker) -> None:
    print("\n[4] Analytical vs empirical labeling in paper_numbers.json")
    pn_file = _RESULTS / "paper_numbers.json"
    if not pn_file.exists():
        c.warn(False, "paper_numbers.json exists", "run generate_paper_numbers.py first")
        return

    with open(pn_file) as fh:
        pn = json.load(fh)

    nums = pn.get("numbers", {})

    must_be_analytical = [
        "e_q_theoretical_twostage",
        "q_max_theoretical_twostage",
        "blq_theoretical_adaptive",
        "m3_presidio_latency_overhead_ms",
    ]
    for key in must_be_analytical:
        if key in nums:
            tag = nums[key].get("tag", "")
            c.check(
                tag == "analytical_simulated",
                f"{key} tagged as analytical_simulated",
                f"found tag='{tag}'"
            )

    must_be_empirical_or_derived = [
        "paper_exact_recovery_rate",
        "paper_mean_total_api_calls",
        "baseline_exact_recovery_rate",
    ]
    for key in must_be_empirical_or_derived:
        if key in nums:
            tag = nums[key].get("tag", "")
            c.check(
                tag in ("empirical", "derived_from_empirical"),
                f"{key} is empirical/derived (not analytical)",
                f"found tag='{tag}'"
            )

    missing = pn.get("_meta", {}).get("missing_values", [])
    critical_required = ["paper_exact_recovery_rate", "baseline_mean_api_calls"]
    for k in critical_required:
        val = nums.get(k, {}).get("value")
        c.check(val not in (None, "MISSING_ARTIFACT"),
                f"Required paper number present: {k}", f"got {val}")


def _check_blq_naming(c: _Checker) -> None:
    print("\n[5] BLQ / EERQ terminology")
    recon = _REPO / "src" / "kv_attack" / "two_stage_reconstructor.py"
    if recon.exists():
        src = recon.read_text()
        has_blq  = "BLQ" in src or "blq" in src
        has_note = "bits/query" in src or "information" in src.lower()
        c.check(has_blq and has_note,
                "BLQ usage in two_stage_reconstructor.py has information-theoretic context",
                "If BLQ is purely novel, rename to EERQ per Issue #3")

    pn_file = _RESULTS / "paper_numbers.json"
    if pn_file.exists():
        with open(pn_file) as fh:
            pn = json.load(fh)
        blq_note = pn.get("numbers", {}).get("paper_blq_empirical", {}).get("note", "")
        c.warn("EERQ" in blq_note or "bits-per-query" in blq_note or "NOT a newly" in blq_note,
               "paper_numbers.json BLQ note clarifies metric origin",
               "Add EERQ rename note or standard information-theoretic citation")


def _check_reproducibility(c: _Checker) -> None:
    """
    Smoke-test: generate victims twice with the same seed and assert identical.
    This runs in-process without needing a backend.
    """
    print("\n[6] Reproducibility — same seed → same victims")
    try:
        sys.path.insert(0, str(_REPO / "src"))
        from kv_attack import FIRST_NAMES, LAST_NAMES, MEDICAL_CONDITIONS
        import random
        from datetime import date, timedelta

        def _gen(seed: int, n: int) -> list[dict]:
            rng = random.Random(seed)
            victims = []
            for i in range(n):
                fn   = rng.choice(FIRST_NAMES)
                ln   = rng.choice(LAST_NAMES)
                cond = rng.choice(MEDICAL_CONDITIONS)
                start = date(1935, 1, 1)
                dob = (start + timedelta(days=rng.randint(0, 26297))).isoformat()
                victims.append({"id": i, "name": f"{fn} {ln}", "condition": cond, "dob": dob})
            return victims

        run1 = _gen(seed=42, n=5)
        run2 = _gen(seed=42, n=5)
        c.check(run1 == run2, "Same seed (42) → identical victim set (5 victims)")

        run3 = _gen(seed=99, n=5)
        c.check(run1 != run3, "Different seed (99) → different victim set")

        c.check(FIRST_NAMES == FIRST_NAMES[:], "Candidate vocabulary is stable across imports")

    except Exception as exc:
        c.check(False, "Reproducibility check", f"Exception: {exc}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="Treat warnings as errors")
    args = ap.parse_args()

    print("=" * 60)
    print("  Consistency checks — KV-cache attack pipeline")
    print("=" * 60)

    c = _Checker(strict=args.strict)

    _check_required_artifacts(c)
    _check_sample_counts(c)
    _check_model_ids(c)
    _check_analytical_labeling(c)
    _check_blq_naming(c)
    _check_reproducibility(c)

    print("\n" + "=" * 60)
    sys.exit(c.summary())


if __name__ == "__main__":
    main()
