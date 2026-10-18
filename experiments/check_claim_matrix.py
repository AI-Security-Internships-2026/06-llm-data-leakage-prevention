"""
experiments/check_claim_matrix.py
==================================
Issue #3 follow-up — pin every numeric claim in docs/paper/claim_evidence_matrix.md
to experiments/results/paper_numbers.json so the matrix can never silently drift
from the real result files again.

Background: as of 2026-09-22 the matrix had drifted. It stated
success_rate=1.0, mean_total_api_calls=79.8, and blq_improvement_factor=11.52x,
which were the hardcoded *fallback defaults* inside generate_paper_numbers.py's
source (used only when a source file is missing) — not the live computed
values. paper_numbers.json itself was correct the whole time (0 assertion
failures) and nothing ever checked the matrix against it. Fixed 2026-10-11.

This script is intentionally NOT a markdown parser (too fragile). It hardcodes
the expected value for each numeric claim and asserts it against the live
paper_numbers.json/result files on every run. When you change a number in the
matrix, update the matching assertion here in the same commit — if they ever
disagree again, this script is the thing that should catch it, not a human
re-reading a 180-line markdown table.

Usage
-----
  python experiments/check_claim_matrix.py
  python experiments/check_claim_matrix.py --strict   # exit 1 on any failure
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_REPO    = Path(__file__).resolve().parent.parent
_RESULTS = _REPO / "experiments" / "results"

H0 = 10.9658


def _load(name: str) -> dict:
    p = _RESULTS / name
    if not p.exists():
        print(f"[FAIL] missing required artifact: {p}")
        return {}
    with open(p) as fh:
        return json.load(fh)


def close(a, b, tol=1e-3):
    return abs(a - b) <= tol * max(1.0, abs(b))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    failures: list[str] = []

    pn   = _load("paper_numbers.json")
    w13  = _load("kv_week13_final.json")
    w50  = _load("kv_attack_results_50.json")
    pareto_src = _RESULTS / "pareto_runner_src_check"

    def check(name: str, got, want, tol=1e-3, numeric=True):
        ok = close(got, want, tol) if numeric else (got == want)
        print(f"  {'OK ' if ok else 'FAIL'}  {name}: matrix says {want}, "
              f"live data says {got}")
        if not ok:
            failures.append(name)

    print("=== Claim-evidence matrix vs live data ===\n")

    if w13 and "aggregate" in w13:
        agg = w13["aggregate"]
        check("paper_exact_recovery_rate (matrix: 0.68)",
              agg.get("success_rate"), 0.68)
        check("paper_mean_total_api_calls (matrix: 66.98)",
              agg.get("mean_total_api_calls"), 66.98, tol=0.01)
        check("paper_n_victims (matrix: 50)",
              agg.get("n_victims"), 50, numeric=False)
        check("sr_target_met is False (matrix states this is a negative result)",
              agg.get("sr_target_met", "MISSING"), False, numeric=False)
    if w13 and "calibration" in w13:
        calib = w13["calibration"]
        check("T1 threshold actually used (matrix: 1011.86 ms, not the 438.8 analytical estimate)",
              calib.get("t1_threshold_ms"), 1011.8573, tol=0.01)
        check("T2 threshold actually used (matrix: 314.37 ms, not the 177.1 analytical estimate)",
              calib.get("t2_threshold_ms"), 314.3744, tol=0.01)
        check("hit_mean_ms (matrix: 190.46, not 87.6)", calib.get("hit_mean_ms"), 190.4627, tol=0.01)
        check("miss_mean_ms (matrix: 1589.69, not 576.1)", calib.get("miss_mean_ms"), 1589.6878, tol=0.01)
    else:
        failures.append("kv_week13_final.json:aggregate missing")

    if w13 and w50 and "aggregate" in w13 and "aggregate" in w50:
        mean_two_stage = w13["aggregate"]["mean_total_api_calls"]
        mean_baseline   = w50["aggregate"]["mean_total_api_calls"]
        eerq_two_stage  = H0 / mean_two_stage
        eerq_baseline   = H0 / mean_baseline
        improvement     = eerq_two_stage / eerq_baseline
        check("EERQ empirical two-stage (matrix: 0.1637)", eerq_two_stage, 0.1637, tol=0.01)
        check("EERQ improvement vs n=50 baseline (matrix: 19.4x)", improvement, 19.4, tol=0.02)
        check("baseline mean_total_api_calls (matrix: 1302.5)", mean_baseline, 1302.5, tol=0.01)
    else:
        failures.append("cannot compute EERQ improvement — a source file is missing")

    if w13 and w50:
        m13 = w13.get("model", "")
        m50 = w50.get("model", "")
        same_model = (m13 == m50)
        print(f"  {'INFO'}  testbed match: w13 model={m13!r} vs w50 model={m50!r} "
              f"(matrix explicitly documents these as DIFFERENT — same_model={same_model})")
        if same_model:
            print("  [NOTE] Models now match — if this was intentionally fixed, "
                  "update the 'NOT_SAME_TESTBED' caveat in the matrix's Results "
                  "section, it is now stale in the other direction.")

    pareto_path = _REPO / "src" / "kv_attack" / "pareto_runner.py"
    if pareto_path.exists():
        src = pareto_path.read_text()
        has_driver_license = '"US_DRIVER_LICENSE"' in src and "PII_ENTITIES" in src
        check("PII_ENTITIES includes US_DRIVER_LICENSE (matrix: True)",
              has_driver_license, True, numeric=False)
    else:
        failures.append("src/kv_attack/pareto_runner.py missing")
    
    rerun_path = _RESULTS / "kv_week13_final_rerun.json"
    rerun_probes = _RESULTS / "kv_week13_final_rerun.json.raw_probes.jsonl"
    if rerun_path.exists() and rerun_probes.exists():
        rerun = json.load(open(rerun_path))
        n_probe_lines = sum(1 for _ in open(rerun_probes))
        check("rerun raw_probes line count (matrix: 3107)", n_probe_lines, 3107, numeric=False)
        check("rerun success_rate (matrix: 0.98)", rerun["aggregate"]["success_rate"], 0.98, tol=0.01)
        check("canonical success_rate still 0.68 (matrix explicitly keeps this as primary, not the rerun's 0.98)",
              w13["aggregate"]["success_rate"] if w13 else None, 0.68, tol=0.01)
    else:
        failures.append("kv_week13_final_rerun.json or its raw_probes.jsonl missing — "
                         "Issue #2 raw-probe-evidence claim in the matrix is not backed by a committed file")

    nat_err_candidates = sorted(_RESULTS.glob("natural-error-realistic-*.json"))
    if nat_err_candidates:
        nat = json.load(open(nat_err_candidates[-1]))
        check("MEDICAL no-name FNR (matrix: 0.50)",
              nat["MEDICAL"]["no_name_fnr"], 0.50, tol=0.01)
        check("ENTERPRISE no-name FNR (matrix: 0.25)",
              nat["ENTERPRISE_CONFIDENTIAL"]["no_name_fnr"], 0.25, tol=0.01)
    else:
        failures.append("no natural-error-realistic-*.json found — Issue #9 "
                         "finding in the matrix is not backed by a committed result file")

    print()
    if failures:
        print(f"[FAIL] {len(failures)} claim(s) no longer match the live data:")
        for f in failures:
            print(f"  ✗ {f}")
        print("\nUpdate docs/paper/claim_evidence_matrix.md AND the matching "
              "assertion in this file together.")
        if args.strict:
            return 1
    else:
        print("[OK] All checked claims match the live result files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())