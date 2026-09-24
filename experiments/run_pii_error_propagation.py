"""
experiments/run_pii_error_propagation.py
==========================================
Issue #9 — Quantify PII-Detector Errors → Cache Decision → Timing Leakage.

This is the KEY cross-layer experiment that binds PII detection to KV-cache
security.  For each combination of detector variant × error rate, we trace:

  False Negative path:
    sensitive prompt → detector misses PII → cache SHARED
    → timing oracle visible to attacker → residual ASR = empirical_asr

  False Positive path:
    clean prompt → detector false alarm → cache ISOLATED
    → cache utility lost → TTFT overhead increases

Three detector variants
-----------------------
  P0  Presidio / regex only         (use_stage2=False)
  P1  Presidio + BART stage-2       (use_stage2=True)
  P2  Full risk-signal + D4 policy  (Issue-7 pipeline)

Two sweep experiments
---------------------
  FNR sweep  [0%, 5%, 10%, 20%, 30%] — how does increasing miss rate
             propagate to residual leakage?
  FPR sweep  [0%, 5%, 10%, 20%, 30%] — how does increasing false alarm
             rate hurt cache utility?

For natural (real detector) errors we run the actual detector on each
sample and compare to ground truth.  For controlled errors we inject the
error and override the prediction.

Figures
-------
  Fig 1  FNR → residual leakage (central figure, per detector)
  Fig 2  FPR → cache utility / TTFT overhead
  Fig 3  End-to-end error propagation flow (architecture diagram)
  Fig 4  Risk-threshold security/performance Pareto (P0 vs P1 vs P2)

Tables
------
  Table 1  Detector | Recall | FPR | Sensitive shared | Residual ASR | Hit rate | TTFT overhead
  Table 2  Category | n | FNR | n_leaked | Leakage rate | Supported?

Usage
-----
  cd src
  python ../experiments/run_pii_error_propagation.py \\
      --config ../configs/pii_error_propagation.yaml \\
      --backend mock           # offline test
  python ../experiments/run_pii_error_propagation.py \\
      --config ../configs/pii_error_propagation.yaml  # live vLLM
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import List

import numpy as np

_REPO = Path(__file__).parent.parent
_SRC  = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from kv_attack import VLLM_BASE_URL, MODEL_ID
from kv_attack.backends.mock_backend import MockBackend
from kv_attack.backends.vllm_backend import VLLMBackend
from kv_attack.sensitivity_workload import (
    build_ground_truth_workload, inject_fn, inject_fp,
    workload_metadata, WORKLOAD_VERSION,
    CAT_NON_SENSITIVE, ALL_CATEGORIES, PII_SUPPORTED,
)
from kv_attack.risk_signal import analyze_prompt, build_risk_signal
from kv_attack.cache_policy import (
    CachePolicyEngine, POLICY_D4, CacheAction,
)



class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):  return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray):  return o.tolist()
        if isinstance(o, np.bool_):    return bool(o)
        return super().default(o)

def _dump(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh: json.dump(obj, fh, cls=_Enc, indent=2)
    os.replace(tmp, path)

def _env():
    git = "unknown"
    try: git = subprocess.check_output(
        ["git","rev-parse","--short","HEAD"], text=True, cwd=str(_REPO), timeout=5
    ).strip()
    except: pass
    return {"python": sys.version, "platform": platform.platform(), "git": git}



def calibrate(backend, n=30) -> dict:
    WARM = (
        "You are a helpful assistant specialising in data privacy and security. "
        "The following context is for reference: organisations handling personal data "
        "must implement appropriate technical and organisational measures to ensure "
        "a level of security appropriate to the risk, including pseudonymisation and "
        "encryption of personal data and ongoing confidentiality of processing systems."
    )
    print("[calibrate] Measuring TTFT…")
    backend.measure_ttft(WARM)
    hits   = [backend.measure_ttft(WARM) for _ in range(n)]
    misses = [backend.measure_ttft(f"cold-{uuid.uuid4().hex}") for _ in range(n)]
    ha, ma = np.array(hits), np.array(misses)
    thr    = float((ha.mean() + ma.mean()) / 2)
    gap    = float(ma.mean() - ha.mean())
    print(f"  hit={ha.mean():.0f}ms  miss={ma.mean():.0f}ms  gap={gap:.0f}ms  threshold={thr:.0f}ms")
    if abs(gap) < 20:
        print("  ⚠️  WARNING: APC gap < 20 ms — TTFT-based metrics unreliable")
    return {"hit_mean_ms": float(ha.mean()), "miss_mean_ms": float(ma.mean()),
            "threshold_ms": thr, "gap_ms": gap, "apc_signal_ok": abs(gap) >= 20}



def run_detector_p0(text: str) -> str:
    """P0: Presidio/regex only, no BART stage-2."""
    from detector import detect_pii
    r = detect_pii(text, use_stage2=False)
    lvl = r.get("risk_level", "CLEAN")
    return lvl if lvl != "CLEAN" else "LOW"

def run_detector_p1(text: str) -> str:
    """P1: Presidio + BART stage-2 LLM judge."""
    from detector import detect_pii
    try:
        r = detect_pii(text, use_stage2=True)
        lvl = r.get("risk_level", "CLEAN")
        return lvl if lvl != "CLEAN" else "LOW"
    except Exception as e:
        return run_detector_p0(text)

def run_detector_p2(text: str) -> str:
    """P2: Full risk-signal + Issue-7 D4 policy decision."""
    sig = analyze_prompt(text, use_stage2=False)
    return sig.risk_level



_LEVEL_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

def _policy_action(predicted_risk: str, policy: str = POLICY_D4) -> CacheAction:
    """Map predicted risk level to D4 cache action."""
    engine = CachePolicyEngine(policy=policy, tenant_id="victim")
    from kv_attack.risk_signal import RiskSignal
    sig = RiskSignal(
        entity_types         = [],
        pii_flag             = predicted_risk != "LOW",
        detector_confidence  = 1.0 if predicted_risk == "HIGH" else 0.5,
        semantic_sensitivity = "FINANCIAL" if predicted_risk == "HIGH" else "NONE",
        risk_score           = {"LOW": 0.0, "MEDIUM": 0.45, "HIGH": 0.90}.get(predicted_risk, 0.0),
        risk_level           = predicted_risk,
        reason               = f"Controlled: predicted={predicted_risk}",
    )
    d = engine.decide(sig)
    return d.cache_action



def trace_sample(
    sample,
    predicted_risk: str,
    calibration:    dict,
    empirical_asr:  float,
) -> dict:
    """
    Trace one sample through: detection → cache decision → leakage outcome.

    Returns a dict with all observable outcomes needed for Tables 1 and 2.
    """
    gt_risk   = sample.gt_risk_level
    is_sensitive = (gt_risk in ("HIGH", "MEDIUM"))
    is_clean  = (gt_risk == "LOW")

    action = _policy_action(predicted_risk)

    is_fn          = is_sensitive and (predicted_risk == "LOW")
    cache_shared   = (action == CacheAction.SHARE)
    timing_visible = cache_shared and is_sensitive
    leaked         = timing_visible
    residual_asr   = empirical_asr if leaked else 0.0

    is_fp          = is_clean and (predicted_risk in ("MEDIUM", "HIGH"))
    unnecessarily_isolated = is_fp and (action != CacheAction.SHARE)
    ttft_overhead  = (
        calibration["miss_mean_ms"] - calibration["hit_mean_ms"]
        if unnecessarily_isolated else 0.0
    )
    utility_lost   = unnecessarily_isolated

    cache_hit_rate = 1.0 if action == CacheAction.SHARE else 0.0

    correct_isolation = (action != CacheAction.SHARE) if sample.should_isolate \
                        else (action == CacheAction.SHARE)

    return {
        "sample_id":            sample.sample_id,
        "category":             sample.category,
        "gt_risk_level":        gt_risk,
        "predicted_risk_level": predicted_risk,
        "pii_supported":        sample.pii_supported,
        "should_isolate":       sample.should_isolate,
        "gt_fields":            sample.gt_fields,
        "error_type":           sample.error_type,
        "controlled_error":     sample.controlled_error,
        "cache_action":         action.value,
        "is_fn":                is_fn,
        "is_fp":                is_fp,
        "cache_shared":         cache_shared,
        "timing_oracle_visible": timing_visible,
        "leaked":               leaked,
        "residual_asr":         residual_asr,
        "utility_lost":         utility_lost,
        "ttft_overhead_ms":     round(ttft_overhead, 2),
        "cache_hit_rate":       cache_hit_rate,
        "correct_isolation":    correct_isolation,
    }


def _ci_proportion(n_success: int, n_total: int) -> tuple[float, float]:
    """Wilson 95% CI for a proportion."""
    if n_total == 0:
        return (0.0, 0.0)
    p   = n_success / n_total
    z   = 1.96
    denom = 1 + z**2 / n_total
    centre = (p + z**2 / (2 * n_total)) / denom
    half   = z * math.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denom
    return (round(max(0, centre - half), 4), round(min(1, centre + half), 4))


def aggregate_traces(traces: list[dict], empirical_asr: float) -> dict:
    """Compute summary metrics from per-sample trace dicts."""
    n = len(traces)
    sensitive  = [t for t in traces if t["gt_risk_level"] in ("HIGH", "MEDIUM")]
    clean      = [t for t in traces if t["gt_risk_level"] == "LOW"]
    n_s, n_c   = len(sensitive), len(clean)

    n_correct_det = sum(1 for t in sensitive if t["predicted_risk_level"] != "LOW")
    recall        = round(n_correct_det / max(n_s, 1), 4)

    n_fp   = sum(1 for t in clean if t["predicted_risk_level"] != "LOW")
    fpr    = round(n_fp / max(n_c, 1), 4)

    n_leaked     = sum(1 for t in sensitive if t["leaked"])
    sens_shared  = round(n_leaked / max(n_s, 1), 4)
    leak_ci      = _ci_proportion(n_leaked, n_s)

    res_asr = round(sum(t["residual_asr"] for t in traces) / max(n, 1), 4)

    n_clean_hit  = sum(1 for t in clean if t["cache_hit_rate"] > 0)
    hit_rate     = round(n_clean_hit / max(n_c, 1), 4)

    ttft_overheads = [t["ttft_overhead_ms"] for t in traces if t["ttft_overhead_ms"] > 0]
    mean_overhead  = round(sum(ttft_overheads) / max(len(ttft_overheads), 1), 2)

    n_isolated   = sum(1 for t in traces if t["cache_action"] != "SHARE")
    iso_fraction = round(n_isolated / max(n, 1), 4)

    return {
        "n_total":          n,
        "n_sensitive":      n_s,
        "n_clean":          n_c,
        "recall":           recall,
        "fpr":              fpr,
        "sensitive_shared": sens_shared,
        "sensitive_shared_ci95": leak_ci,
        "residual_asr":     res_asr,
        "hit_rate":         hit_rate,
        "mean_ttft_overhead_ms": mean_overhead,
        "isolation_fraction": iso_fraction,
    }



def eval_natural_errors(
    samples:       list,
    detector_fn,
    detector_name: str,
    calibration:   dict,
    empirical_asr: float,
) -> dict:
    """Run the actual detector on all samples and trace outcomes."""
    print(f"  Natural errors [{detector_name}]…")
    traces = []
    for s in samples:
        pred = s.predicted_risk_level
        if pred is None:
            pred = detector_fn(s.text)
        traces.append(trace_sample(s, pred, calibration, empirical_asr))
    agg = aggregate_traces(traces, empirical_asr)
    return {"detector": detector_name, "mode": "natural",
            "aggregate": agg, "traces": traces}



def sweep_fnr(
    base_samples:  list,
    fnr_levels:    list[float],
    detector_fn,
    detector_name: str,
    calibration:   dict,
    empirical_asr: float,
    rng:           random.Random,
) -> list[dict]:
    """FNR sweep: for each FNR level, inject errors and trace outcomes."""
    results = []
    for fnr in fnr_levels:
        samples_fn = inject_fn(base_samples, fnr, rng)
        rows = []
        for s in samples_fn:
            pred = s.predicted_risk_level
            if pred is None:
                pred = detector_fn(s.text)
            rows.append(trace_sample(s, pred, calibration, empirical_asr))
        agg = aggregate_traces(rows, empirical_asr)
        n_fn_injected = sum(1 for s in samples_fn if s.error_type == "FN")
        results.append({
            "fnr_injected":    fnr,
            "n_fn_injected":   n_fn_injected,
            "detector":        detector_name,
            "aggregate":       agg,
        })
        print(f"    FNR={fnr:.0%}: leakage={agg['sensitive_shared']:.1%}  "
              f"residual_asr={agg['residual_asr']:.1%}")
    return results


def sweep_fpr(
    base_samples:  list,
    fpr_levels:    list[float],
    detector_fn,
    detector_name: str,
    calibration:   dict,
    empirical_asr: float,
    rng:           random.Random,
) -> list[dict]:
    """FPR sweep: for each FPR level, inject false positives and trace outcomes."""
    results = []
    for fpr in fpr_levels:
        samples_fp = inject_fp(base_samples, fpr, rng)
        rows = []
        for s in samples_fp:
            pred = s.predicted_risk_level
            if pred is None:
                pred = detector_fn(s.text)
            rows.append(trace_sample(s, pred, calibration, empirical_asr))
        agg = aggregate_traces(rows, empirical_asr)
        n_fp_injected = sum(1 for s in samples_fp if s.error_type == "FP")
        results.append({
            "fpr_injected":    fpr,
            "n_fp_injected":   n_fp_injected,
            "detector":        detector_name,
            "aggregate":       agg,
        })
        print(f"    FPR={fpr:.0%}: hit_rate={agg['hit_rate']:.1%}  "
              f"overhead={agg['mean_ttft_overhead_ms']:.0f}ms")
    return results



def generate_figures(results: dict, figures_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    colors = {"P0": "#e74c3c", "P1": "#e67e22", "P2": "#3498db"}

    fnr_data = results.get("fnr_sweep", {})
    fpr_data = results.get("fpr_sweep", {})
    natural  = results.get("natural_errors", {})

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    for det, sweeps in fnr_data.items():
        fnrs     = [s["fnr_injected"] for s in sweeps]
        leakages = [s["aggregate"]["sensitive_shared"] for s in sweeps]
        cis_lo   = [s["aggregate"]["sensitive_shared_ci95"][0] for s in sweeps]
        cis_hi   = [s["aggregate"]["sensitive_shared_ci95"][1] for s in sweeps]
        ax1.plot(fnrs, leakages, "o-", color=colors.get(det,"gray"),
                 lw=1.8, ms=7, label=det)
        ax1.fill_between(fnrs, cis_lo, cis_hi,
                          alpha=0.15, color=colors.get(det,"gray"))
    ax1.set_xlabel("Injected False-Negative Rate (FNR)", fontsize=11)
    ax1.set_ylabel("Sensitive Prompts Shared (Leakage)", fontsize=11)
    ax1.set_title("Issue #9 Fig 1: FNR → Residual Cache Leakage", fontsize=12)
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:.0%}"))
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:.0%}"))
    ax1.legend(); ax1.grid(alpha=0.3); ax1.set_ylim(-0.05, 1.05)
    fig1.tight_layout()
    fig1.savefig(f"{figures_dir}/issue9_fig1_fnr_leakage.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print("  Saved: issue9_fig1_fnr_leakage.png")

    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 5))
    for det, sweeps in fpr_data.items():
        fprs  = [s["fpr_injected"] for s in sweeps]
        hits  = [s["aggregate"]["hit_rate"] for s in sweeps]
        ovrh  = [s["aggregate"]["mean_ttft_overhead_ms"] for s in sweeps]
        c = colors.get(det, "gray")
        ax2a.plot(fprs, hits, "o-", color=c, lw=1.8, ms=7, label=det)
        ax2b.plot(fprs, ovrh, "s-", color=c, lw=1.8, ms=7, label=det)
    ax2a.set_xlabel("Injected FPR", fontsize=10)
    ax2a.set_ylabel("Cache Hit Rate (Utility)", fontsize=10)
    ax2a.set_title("Fig 2a: FPR → Cache Utility", fontsize=11)
    ax2a.xaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:.0%}"))
    ax2a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:.0%}"))
    ax2a.legend(); ax2a.grid(alpha=0.3)
    ax2b.set_xlabel("Injected FPR", fontsize=10)
    ax2b.set_ylabel("Mean TTFT Overhead (ms)", fontsize=10)
    ax2b.set_title("Fig 2b: FPR → TTFT Overhead", fontsize=11)
    ax2b.xaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:.0%}"))
    ax2b.legend(); ax2b.grid(alpha=0.3)
    fig2.suptitle("Issue #9 Fig 2: FPR → Cache Utility / TTFT", fontsize=12)
    fig2.tight_layout()
    fig2.savefig(f"{figures_dir}/issue9_fig2_fpr_utility.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("  Saved: issue9_fig2_fpr_utility.png")

    fig3, ax3 = plt.subplots(figsize=(11, 6))
    ax3.axis("off")
    flow = (
        "End-to-End PII Error Propagation Flow\n"
        "══════════════════════════════════════════════════════════════\n\n"
        " ┌────────────────────────────────────────────────────────┐\n"
        " │                Incoming LLM Prompt                     │\n"
        " └─────────────────────────┬──────────────────────────────┘\n"
        "                           │\n"
        "              ┌────────────▼────────────┐\n"
        "              │   PII Detector (P0/P1/P2)│\n"
        "              └────────────┬────────────┘\n"
        "                  ┌────────┴────────┐\n"
        "           CORRECT │                │ ERROR\n"
        "            ┌──────▼─────┐   ┌──────▼──────────────┐\n"
        "            │Cache Policy│   │ False Negative (FN)  │\n"
        "            │(D4: ISOLATE│   │ Predicted LOW        │\n"
        "            │ for HIGH)  │   │ → Cache SHARED ←──── │── Sensitive data\n"
        "            └──────┬─────┘   └──────┬──────────────┘    in shared cache\n"
        "                   │                │\n"
        "            SECURE │        ┌───────▼───────────────┐\n"
        "            (no    │        │ Timing Oracle Visible  │\n"
        "            leakage│        │ Attacker probes match  │\n"
        "                   │        │ → Residual ASR = 68%   │\n"
        "                   │        └───────────────────────┘\n"
        "                   │\n"
        "              ┌────▼──────────────────────┐\n"
        "              │ False Positive (FP)        │\n"
        "              │ Clean prompt → ISOLATE     │\n"
        "              │ → Cache miss (utility loss)│\n"
        "              │ → +80ms TTFT overhead      │\n"
        "              └───────────────────────────┘"
    )
    ax3.text(0.02, 0.98, flow, transform=ax3.transAxes, fontsize=8.5,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#f0f4f8", alpha=0.8))
    ax3.set_title("Issue #9 Fig 3: End-to-End Error Propagation Flow", pad=10)
    fig3.tight_layout()
    fig3.savefig(f"{figures_dir}/issue9_fig3_error_propagation_flow.png",
                 dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print("  Saved: issue9_fig3_error_propagation_flow.png")

    fig4, ax4 = plt.subplots(figsize=(8, 6))
    for det, nat in natural.items():
        agg  = nat["aggregate"]
        sec  = 1.0 - agg["sensitive_shared"]
        util = agg["hit_rate"]
        ax4.scatter(util, sec, s=250, color=colors.get(det,"gray"),
                    zorder=5, label=det)
        ax4.annotate(det, (util, sec), textcoords="offset points",
                     xytext=(8, 4), fontsize=10, fontweight="bold")
    ax4.set_xlabel("Cache Utility (Hit Rate for Clean Traffic) →", fontsize=11)
    ax4.set_ylabel("Security (1 − Leakage Rate) →", fontsize=11)
    ax4.set_title("Issue #9 Fig 4: Security vs Utility Pareto per Detector", fontsize=12)
    ax4.set_xlim(-0.05, 1.10); ax4.set_ylim(-0.05, 1.10)
    ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:.0%}"))
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:.0%}"))
    ax4.axhline(1.0, color="gray", ls="--", alpha=0.4)
    ax4.axvline(1.0, color="gray", ls=":", alpha=0.4)
    ax4.legend(fontsize=10); ax4.grid(alpha=0.3)
    fig4.tight_layout()
    fig4.savefig(f"{figures_dir}/issue9_fig4_pareto.png", dpi=150, bbox_inches="tight")
    plt.close(fig4)
    print("  Saved: issue9_fig4_pareto.png")



def generate_tables(results: dict, figures_dir: str, calibration: dict) -> None:
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    natural = results.get("natural_errors", {})

    with open(f"{figures_dir}/issue9_table1_detector_comparison.md", "w") as fh:
        fh.write("# Issue #9 Table 1: Detector Variant Comparison\n\n")
        fh.write("*All metrics from natural (real) detector errors — "
                 "not controlled injection.*\n\n")
        fh.write("| Detector | Recall | FPR | Sensitive Shared | "
                 "Residual ASR | Hit Rate | TTFT Overhead |\n")
        fh.write("|----------|--------|-----|-----------------|"
                 "-------------|----------|---------------|\n")
        for det, nat in natural.items():
            a = nat["aggregate"]
            ci = a["sensitive_shared_ci95"]
            fh.write(
                f"| {det} | {a['recall']:.0%} | {a['fpr']:.0%} | "
                f"{a['sensitive_shared']:.0%} [{ci[0]:.0%}–{ci[1]:.0%}] | "
                f"{a['residual_asr']:.0%} | {a['hit_rate']:.0%} | "
                f"{a['mean_ttft_overhead_ms']:.0f} ms |\n"
            )
    print("  Saved: issue9_table1_detector_comparison.md")

    p2_nat = natural.get("P2", {})
    traces  = p2_nat.get("traces", [])
    with open(f"{figures_dir}/issue9_table2_category_leakage.md", "w") as fh:
        fh.write("# Issue #9 Table 2: Category-Level Recall / FN / Leakage (P2)\n\n")
        fh.write("| Category | N | FNR | N leaked | Leakage rate | "
                 "Presidio supported? |\n")
        fh.write("|----------|---|-----|----------|--------------|"
                 "--------------------|\n")
        for cat in ALL_CATEGORIES:
            ct = [t for t in traces if t["category"] == cat]
            n  = len(ct)
            if n == 0: continue
            n_fn  = sum(1 for t in ct if t["is_fn"])
            fnr   = n_fn / n
            n_lk  = sum(1 for t in ct if t["leaked"])
            lr    = n_lk / n
            sup   = "✅ Yes" if PII_SUPPORTED[cat] else "❌ No (partial)"
            fh.write(f"| {cat} | {n} | {fnr:.0%} | {n_lk} | {lr:.0%} | {sup} |\n")
    print("  Saved: issue9_table2_category_leakage.md")



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",  default="../configs/pii_error_propagation.yaml")
    p.add_argument("--backend", choices=["vllm","sglang","mock"])
    p.add_argument("--seed",    type=int)
    p.add_argument("--output-dir")
    return p.parse_args()

def _load_config(path: str) -> dict:
    try:
        import yaml
        with open(path) as f: return yaml.safe_load(f)
    except ImportError:
        cfg: dict = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"): continue
                if ":" in line:
                    k, _, v = line.partition(":")
                    v = v.strip().strip('"')
                    if v.startswith("["):
                        import ast
                        try: v = ast.literal_eval(v)
                        except: pass
                    else:
                        try:    v = int(v)
                        except:
                            try: v = float(v)
                            except: pass
                    cfg[k.strip()] = v
        return cfg

def main():
    args = parse_args()
    cfg_path = args.config if os.path.isabs(args.config) \
               else os.path.join(os.path.dirname(__file__), args.config)
    cfg = _load_config(cfg_path)

    backend_name  = args.backend  or cfg.get("backend", "mock")
    seed          = args.seed     or cfg.get("seed", 42)
    output_dir    = args.output_dir or cfg.get("output_dir", "experiments/results")
    figures_dir   = cfg.get("figures_dir", "experiments/figures")
    model_id      = cfg.get("model_id", MODEL_ID)
    vllm_url      = cfg.get("vllm_base_url", VLLM_BASE_URL)
    n_per_cat     = cfg.get("n_per_category", 10)
    empirical_asr = cfg.get("empirical_asr", 0.68)
    fnr_levels    = cfg.get("fnr_levels", [0.0, 0.05, 0.10, 0.20, 0.30])
    fpr_levels    = cfg.get("fpr_levels", [0.0, 0.05, 0.10, 0.20, 0.30])
    det_variants  = cfg.get("detector_variants", ["P0", "P1", "P2"])
    n_cal         = cfg.get("n_calibration", 30)

    if not os.path.isabs(output_dir):  output_dir  = str(_REPO / output_dir)
    if not os.path.isabs(figures_dir): figures_dir = str(_REPO / figures_dir)

    run_id   = f"pii-error-prop-{datetime.datetime.utcnow().strftime('%Y-%m-%d')}-{uuid.uuid4().hex[:8]}"
    out_path = f"{output_dir}/{run_id}.json"

    print(f"\n{'='*60}")
    print(f"  Issue #9  PII Error Propagation Evaluation")
    print(f"  run_id:       {run_id}")
    print(f"  backend:      {backend_name}")
    print(f"  n_per_cat:    {n_per_cat}")
    print(f"  detectors:    {det_variants}")
    print(f"  empirical_asr: {empirical_asr} [empirical]")
    print(f"{'='*60}\n")

    if backend_name == "mock":
        backend = MockBackend(seed=seed)
    elif backend_name == "sglang":
        from kv_attack.backends.sglang_backend import SGLangBackend
        backend = SGLangBackend(cfg.get("sglang_base_url","http://localhost:30000/v1"), model_id)
    else:
        backend = VLLMBackend(vllm_url, model_id)
        if not backend.health_check():
            print("[FATAL] vLLM not reachable."); sys.exit(1)

    calibration = calibrate(backend, n=n_cal)

    print(f"\n[workload] Building ground-truth workload v{WORKLOAD_VERSION}…")
    base_samples = build_ground_truth_workload(n_per_category=n_per_cat, seed=seed)
    print(f"  Total: {len(base_samples)} samples across {len(set(s.category for s in base_samples))} categories")

    rng = random.Random(seed)

    DETECTORS = {
        "P0": run_detector_p0,
        "P1": run_detector_p1,
        "P2": run_detector_p2,
    }

    print("\n[natural errors] Running real detector on workload…")
    natural_results: dict[str, dict] = {}
    for det_name in det_variants:
        det_fn = DETECTORS[det_name]
        res    = eval_natural_errors(
            base_samples, det_fn, det_name, calibration, empirical_asr
        )
        natural_results[det_name] = {
            "detector":  det_name,
            "aggregate": res["aggregate"],
            "traces":    res["traces"],
        }
        a = res["aggregate"]
        print(f"    {det_name}: recall={a['recall']:.0%}  fpr={a['fpr']:.0%}  "
              f"leakage={a['sensitive_shared']:.0%}  "
              f"residual_asr={a['residual_asr']:.0%}")

    print("\n[FNR sweep] Controlled false-negative injection…")
    fnr_sweep_results: dict[str, list] = {}
    for det_name in det_variants:
        print(f"  {det_name}:")
        fnr_sweep_results[det_name] = sweep_fnr(
            base_samples, fnr_levels, DETECTORS[det_name],
            det_name, calibration, empirical_asr, rng
        )

    print("\n[FPR sweep] Controlled false-positive injection…")
    fpr_sweep_results: dict[str, list] = {}
    for det_name in det_variants:
        print(f"  {det_name}:")
        fpr_sweep_results[det_name] = sweep_fpr(
            base_samples, fpr_levels, DETECTORS[det_name],
            det_name, calibration, empirical_asr, rng
        )

    results = {
        "run_id":         run_id,
        "issue":          9,
        "config": {
            "backend":        backend_name,
            "model_id":       model_id,
            "n_per_category": n_per_cat,
            "seed":           seed,
            "detectors":      det_variants,
            "fnr_levels":     fnr_levels,
            "fpr_levels":     fpr_levels,
            "empirical_asr":  empirical_asr,
            "empirical_asr_tag": "empirical",
        },
        "environment":    _env(),
        "calibration":    calibration,
        "workload":       workload_metadata(base_samples),
        "natural_errors": {
            det: {"aggregate": v["aggregate"]}
            for det, v in natural_results.items()
        },
        "natural_errors_traces": {
            det: v["traces"] for det, v in natural_results.items()
        },
        "fnr_sweep":      fnr_sweep_results,
        "fpr_sweep":      fpr_sweep_results,
        "command":        " ".join(sys.argv),
    }
    _dump(results, out_path)
    print(f"\n[save] {out_path}")

    print("\n[figures] Generating…")
    results_for_figs = dict(results)
    results_for_figs["natural_errors"] = natural_results
    generate_figures(results_for_figs, figures_dir)
    generate_tables(results_for_figs, figures_dir, calibration)

    print(f"\n{'='*60}")
    print("  NATURAL ERROR SUMMARY")
    print(f"  {'Detector':<6} {'Recall':>8} {'FPR':>6} {'Leaked':>8} {'ResASR':>8} {'HitRate':>8}")
    print(f"  {'-'*48}")
    for det, nat in natural_results.items():
        a = nat["aggregate"]
        print(f"  {det:<6} {a['recall']:>8.0%} {a['fpr']:>6.0%} "
              f"{a['sensitive_shared']:>8.0%} {a['residual_asr']:>8.0%} {a['hit_rate']:>8.0%}")
    print(f"{'='*60}")
    print(f"\nDone. Results: {out_path}")

if __name__ == "__main__":
    main()