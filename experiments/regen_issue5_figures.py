"""
experiments/regen_issue5_figures.py
======================================
Regenerate Issue #5 figures from an EXISTING load-test results JSON
(no need to rerun the sweep). Removes stale PNGs from earlier runs.

Usage
-----
  python experiments/regen_issue5_figures.py experiments/results/load-test-2026-09-15-68b50862.json
"""
import sys, json
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "experiments"))

from run_load_test_v2 import _generate_figures

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python regen_issue5_figures.py <path_to_load_test_json>")
        sys.exit(1)
    with open(sys.argv[1]) as fh:
        payload = json.load(fh)
    _generate_figures(payload, _REPO / "experiments" / "figures")
    print("Done.")
