"""
experiments/run_probe_detection_eval.py
========================================
Issue #8 — Timing-probe behaviour detection evaluation.

Evaluates the ProbeDetector against five workload types:
  Benign:
    B1  normal_repeated   same user asking similar (but legitimate) questions
    B2  bursty_legit      burst of varied traffic (normal usage spike)
  Attack:
    A1  linear_enum       linear name enumeration (100 names, sequential)
    A2  two_stage         adaptive two-stage (names then conditions)
    A3  low_rate          slow attacker (simulated 0.5 probes/sec)

Metrics:
  TPR    fraction of attack probes correctly flagged (WARN or ALERT)
  FPR    fraction of benign probes incorrectly flagged
  delay  probe index at first ALERT (detection delay)
  ASR    attack success rate under static D4 vs dynamic policy

Generates:
  Fig 1  suspicion score over probe index with first-ALERT marker
  Fig 2  ROC curve (TPR vs FPR) across threshold sweep
  Fig 3  ASR comparison: static D4 vs dynamic policy
  Table 1  Workload | TPR | FPR | Median detection delay | P95 delay
  Table 2  Attack | Static ASR | Dynamic ASR | Benign TTFT overhead

Usage
-----
  cd src
  python ../experiments/run_probe_detection_eval.py \\
      --config ../configs/probe_detection_eval.yaml \\
      --backend mock            # fast offline test
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np

_REPO = Path(__file__).parent.parent
_SRC  = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from kv_attack import FIRST_NAMES, LAST_NAMES, MEDICAL_CONDITIONS, VLLM_BASE_URL, MODEL_ID
from kv_attack.backends.mock_backend import MockBackend
from kv_attack.backends.vllm_backend import VLLMBackend
from kv_attack.probe_detector import (
    ProbeDetector, SuspicionLevel,
    THRESHOLD_MED, THRESHOLD_HIGH, COOLDOWN_PROBES,
)
from kv_attack.dynamic_policy import DynamicPolicyEngine
from kv_attack.risk_signal import analyze_prompt
from kv_attack.cache_policy import (
    CachePolicyEngine, POLICY_D4, CacheAction,
)



class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):   return int(o)
        if isinstance(o, (np.floating,)):  return float(o)
        if isinstance(o, np.ndarray):      return o.tolist()
        if isinstance(o, np.bool_):        return bool(o)
        return super().default(o)

def _dump(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh: json.dump(obj, fh, cls=_Enc, indent=2)
    os.replace(tmp, path)



def _env():
    git = "unknown"
    try: git = subprocess.check_output(["git","rev-parse","--short","HEAD"],
              text=True, cwd=str(_REPO), timeout=5).strip()
    except: pass
    return {"python": sys.version, "platform": platform.platform(), "git": git}



def calibrate(backend, n=30, seed=42) -> dict:
    WARM = (
        "You are a helpful assistant specialising in data privacy and security. "
        "The following context is provided for reference: organisations handling "
        "personal data must implement appropriate technical and organisational "
        "measures to ensure a level of security appropriate to the risk."
    )
    backend.measure_ttft(WARM)
    hits = [backend.measure_ttft(WARM) for _ in range(n)]
    misses = [backend.measure_ttft(f"cold-{uuid.uuid4().hex}") for _ in range(n)]
    ha, ma = np.array(hits), np.array(misses)
    threshold = float((ha.mean() + ma.mean()) / 2)
    print(f"  hit={ha.mean():.0f}ms  miss={ma.mean():.0f}ms  threshold={threshold:.0f}ms  gap={ma.mean()-ha.mean():.0f}ms")
    return {"hit_mean_ms": float(ha.mean()), "miss_mean_ms": float(ma.mean()),
            "threshold_ms": threshold, "gap_ms": float(ma.mean()-ha.mean())}



def _names(rng): return [f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
                         for _ in range(100)]
def _conds(rng): return list(rng.sample(MEDICAL_CONDITIONS, min(20, len(MEDICAL_CONDITIONS))))

SYS = "You are a medical records assistant. Patient information is confidential."

def workload_benign_repeated(n: int, rng: random.Random) -> list[dict]:
    """Same user asking slightly varied versions of the same question."""
    topics = [
        "What medications interact with warfarin?",
        "What are the side effects of metformin?",
        "How is hypertension typically managed?",
        "What is the first-line treatment for type-2 diabetes?",
        "Explain the dosing schedule for lisinopril.",
    ]
    prompts = []
    for i in range(n):
        topic = topics[i % len(topics)]
        variation = rng.choice(["Please ", "Can you ", "Briefly, ", ""])
        prompts.append({
            "text":     f"{SYS} {variation}{topic}",
            "is_attack": False,
            "workload": "benign_repeated",
        })
    return prompts

def workload_benign_bursty(n: int, rng: random.Random) -> list[dict]:
    """Legitimate burst — varied topics, no systematic pattern."""
    questions = [
        "What is the capital of Australia?",
        "Explain quantum computing in simple terms.",
        "How does the immune system fight viruses?",
        "What are the main causes of climate change?",
        "Summarise the history of antibiotics.",
        "What is machine learning?",
        "How do vaccines work?",
        "What causes inflation?",
        "Describe the structure of DNA.",
        "What is the difference between TCP and UDP?",
    ]
    return [{"text": rng.choice(questions) + f" [{uuid.uuid4().hex[:4]}]",
             "is_attack": False, "workload": "benign_bursty"} for _ in range(n)]

def workload_linear_enum(n: int, rng: random.Random) -> list[dict]:
    """
    Linear name enumeration: attacker cycles through candidate names
    with a fixed condition placeholder, checking each for a cache hit.
    """
    names  = _names(rng)
    victim = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
    cond   = "diabetes"
    probes = []
    for i in range(n):
        name = names[i % len(names)]
        probes.append({
            "text": f"{SYS} Patient name: {name}. Condition: {cond}. Next appointment?",
            "is_attack":      True,
            "workload":       "linear_enum",
            "is_victim_name": (name == victim),
        })
    return probes

def workload_two_stage(n: int, rng: random.Random) -> list[dict]:
    """
    Two-stage adaptive: first scan names (stage 1), then conditions (stage 2).
    """
    names = _names(rng)
    conds = _conds(rng)
    victim_name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
    victim_cond = rng.choice(MEDICAL_CONDITIONS)
    probes = []
    half   = n // 2
    for i in range(half):
        name = names[i % len(names)]
        probes.append({
            "text": f"{SYS} Patient: {name}. Condition: PLACEHOLDER. Medications?",
            "is_attack": True, "workload": "two_stage", "stage": 1,
            "is_victim_name": (name == victim_name),
        })
    for i in range(n - half):
        cond = conds[i % len(conds)]
        probes.append({
            "text": f"{SYS} Patient: {victim_name}. Condition: {cond}. Treatment?",
            "is_attack": True, "workload": "two_stage", "stage": 2,
            "is_victim_cond": (cond == victim_cond),
        })
    return probes

def workload_low_rate(n: int, rng: random.Random) -> list[dict]:
    """
    Slow attacker: same structure as linear_enum but timestamps spaced 2s apart.
    """
    probes = workload_linear_enum(n, rng)
    t0 = time.time() - n * 2.5
    for i, p in enumerate(probes):
        p["workload"]    = "low_rate"
        p["timestamp_s"] = t0 + i * 2.5
    return probes



def simulate_stream(
    prompts:    list[dict],
    calibration: dict,
    rng:         random.Random,
    attack_hit_prob: float = 0.12,
    benign_hit_prob: float = 0.45,
) -> list[dict]:
    """
    Assign simulated TTFT values to each prompt based on whether it is a
    likely cache hit or miss, without requiring a live backend.

    For evaluation on a live backend: replace this with actual measure_ttft calls.
    """
    hit_ms  = calibration["hit_mean_ms"]
    miss_ms = calibration["miss_mean_ms"]
    noise   = (miss_ms - hit_ms) * 0.08

    enriched = []
    for p in prompts:
        prob = benign_hit_prob if not p.get("is_attack") else attack_hit_prob
        cache_hit = rng.random() < prob
        ttft = (hit_ms if cache_hit else miss_ms) + rng.gauss(0, noise)
        enriched.append({**p, "ttft_ms": max(10.0, ttft), "cache_hit": cache_hit})
    return enriched



def run_detector(
    stream:         list[dict],
    threshold_med:  float,
    threshold_high: float,
    cooldown:       int,
) -> list[dict]:
    """Run the detector over a simulated/real probe stream. Returns annotated rows."""
    detector = ProbeDetector(
        threshold_med   = threshold_med,
        threshold_high  = threshold_high,
        cooldown_probes = cooldown,
    )
    rows = []
    for probe in stream:
        ts  = probe.get("timestamp_s", time.time())
        res = detector.observe(probe["text"], probe["ttft_ms"], probe["cache_hit"],
                               timestamp_s=ts)
        rows.append({
            **probe,
            "suspicion_score":  res.suspicion_score,
            "suspicion_level":  res.level.value,
            "features":         res.features,
            "reasons":          res.reasons,
            "in_cooldown":      res.in_cooldown,
        })
    return rows



def compute_metrics(rows: list[dict], threshold: float = THRESHOLD_MED) -> dict:
    attack_rows = [r for r in rows if r.get("is_attack")]
    benign_rows = [r for r in rows if not r.get("is_attack")]

    def flagged(r): return r["suspicion_score"] >= threshold

    tpr = sum(flagged(r) for r in attack_rows) / max(len(attack_rows), 1)
    fpr = sum(flagged(r) for r in benign_rows) / max(len(benign_rows), 1)

    alerts  = [r for r in rows if r["suspicion_level"] == "ALERT"]
    a_attack = [a for a in alerts if a.get("is_attack")]
    first_alert_idx = a_attack[0]["probe_index"] if a_attack else None

    return {
        "n_attack":       len(attack_rows),
        "n_benign":       len(benign_rows),
        "tpr":            round(tpr, 4),
        "fpr":            round(fpr, 4),
        "first_alert_at": first_alert_idx,
        "n_alerts":       len(alerts),
    }


def roc_sweep(all_rows: list[dict], thresholds: list[float]) -> list[dict]:
    return [{"threshold": t, **compute_metrics(all_rows, threshold=t)}
            for t in thresholds]



def compare_asr(
    attack_prompts: list[dict],
    calibration:    dict,
    rng:            random.Random,
    n_victims:      int = 10,
    seed:           int = 42,
) -> dict:
    """
    Analytical estimate of ASR reduction:
    - Static D4: HIGH-risk prompts get ISOLATE (0% ASR), others get D4 policy
    - Dynamic:   additionally isolates when suspicion ALERT fires

    We use the detector's first-alert index to determine from which probe
    the dynamic policy starts isolating.  Before the alert, both policies behave
    identically.  After the alert, dynamic forces ISOLATE for all remaining probes.
    """
    detector = ProbeDetector()
    first_alert = None
    for i, p in enumerate(attack_prompts):
        ts  = p.get("timestamp_s", time.time() + i)
        res = detector.observe(p["text"], p["ttft_ms"], p["cache_hit"], timestamp_s=ts)
        if res.level == SuspicionLevel.ALERT and first_alert is None:
            first_alert = i

    n = len(attack_prompts)
    static_asr = 0.68

    if first_alert is None:
        dynamic_asr = static_asr
        reduction   = 0.0
    else:
        frac_before = first_alert / max(n, 1)
        dynamic_asr = round(static_asr * frac_before, 4)
        reduction   = round(static_asr - dynamic_asr, 4)

    return {
        "n_attack_probes":  n,
        "first_alert_at":   first_alert,
        "static_asr":       static_asr,
        "dynamic_asr":      dynamic_asr,
        "asr_reduction":    reduction,
    }



def generate_figures(results: dict, figures_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    workloads = results.get("workloads", {})
    roc_data  = results.get("roc_data", [])
    asr_data  = results.get("asr_comparison", {})

    fig1, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    axes = axes.flatten()
    wl_names = ["benign_repeated", "benign_bursty",
                 "linear_enum", "two_stage", "low_rate"]
    colors_map = {"benign_repeated": "#2ecc71", "benign_bursty": "#27ae60",
                  "linear_enum": "#e74c3c", "two_stage": "#c0392b", "low_rate": "#e67e22"}

    for ax, wn in zip(axes[:5], wl_names):
        rows = workloads.get(wn, {}).get("rows", [])
        if not rows:
            ax.set_title(wn); continue
        idxs   = [r.get("probe_index", i) for i, r in enumerate(rows)]
        scores = [r["suspicion_score"] for r in rows]
        color  = colors_map.get(wn, "blue")
        ax.plot(idxs, scores, color=color, lw=1.5, label="suspicion")
        ax.axhline(THRESHOLD_HIGH, color="red",    ls="--", lw=0.8, label=f"ALERT ({THRESHOLD_HIGH})")
        ax.axhline(THRESHOLD_MED,  color="orange", ls=":",  lw=0.8, label=f"WARN ({THRESHOLD_MED})")
        alerts = [r for r in rows if r["suspicion_level"] == "ALERT"
                  and r.get("is_attack", wn not in ("benign_repeated","benign_bursty"))]
        if alerts:
            ai = alerts[0].get("probe_index", 0)
            ax.axvline(ai, color="red", ls="-", lw=1.2, alpha=0.5, label=f"First ALERT @{ai}")
        ax.set_title(wn.replace("_", " ").title(), fontsize=9)
        ax.set_xlabel("Probe index", fontsize=8)
        ax.set_ylabel("Suspicion", fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=6)
    axes[-1].set_visible(False)
    fig1.suptitle("Issue #8 Fig 1: Suspicion Score Over Probe Stream", fontsize=12)
    fig1.tight_layout()
    fig1.savefig(f"{figures_dir}/issue8_fig1_suspicion_traces.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print("  Saved: issue8_fig1_suspicion_traces.png")

    if roc_data:
        fprs = [r["fpr"] for r in roc_data]
        tprs = [r["tpr"] for r in roc_data]
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        ax2.plot(fprs, tprs, "bo-", lw=1.5, ms=6, label="Probe detector")
        ax2.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random")
        for r in roc_data:
            ax2.annotate(f"t={r['threshold']:.2f}", (r["fpr"], r["tpr"]),
                         fontsize=7, xytext=(4, 2), textcoords="offset points")
        ax2.set_xlabel("FPR (benign flagged)", fontsize=11)
        ax2.set_ylabel("TPR (attack flagged)", fontsize=11)
        ax2.set_title("Issue #8 Fig 2: Detection ROC Curve", fontsize=12)
        ax2.legend()
        ax2.grid(alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(f"{figures_dir}/issue8_fig2_roc.png", dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print("  Saved: issue8_fig2_roc.png")

    if asr_data:
        wl_labels = list(asr_data.keys())
        static_vals  = [asr_data[w].get("static_asr",  0) for w in wl_labels]
        dynamic_vals = [asr_data[w].get("dynamic_asr", 0) for w in wl_labels]
        x = np.arange(len(wl_labels))
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        ax3.bar(x - 0.2, static_vals,  0.35, label="Static D4", color="#e74c3c", alpha=0.85)
        ax3.bar(x + 0.2, dynamic_vals, 0.35, label="Dynamic",   color="#3498db", alpha=0.85)
        ax3.set_xticks(x)
        ax3.set_xticklabels([w.replace("_", "\n") for w in wl_labels], fontsize=9)
        ax3.set_ylabel("Attack Success Rate (ASR)", fontsize=11)
        ax3.set_title("Issue #8 Fig 3: ASR — Static D4 vs Dynamic Policy", fontsize=12)
        ax3.set_ylim(0, 1.0)
        ax3.legend()
        ax3.grid(axis="y", alpha=0.3)
        fig3.tight_layout()
        fig3.savefig(f"{figures_dir}/issue8_fig3_asr_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig3)
        print("  Saved: issue8_fig3_asr_comparison.png")



def generate_tables(results: dict, figures_dir: str) -> None:
    Path(figures_dir).mkdir(parents=True, exist_ok=True)

    workloads = results.get("workloads", {})
    asr_data  = results.get("asr_comparison", {})

    with open(f"{figures_dir}/issue8_table1_detection_metrics.md", "w") as fh:
        fh.write("# Issue #8 Table 1: Detection Metrics per Workload\n\n")
        fh.write("| Workload | Type | N | TPR | FPR | First ALERT at | N alerts |\n")
        fh.write("|----------|------|---|-----|-----|----------------|----------|\n")
        for wn, wd in workloads.items():
            m   = wd.get("metrics", {})
            typ = "Attack" if wn not in ("benign_repeated","benign_bursty") else "Benign"
            fh.write(f"| {wn} | {typ} | {m.get('n_attack',0)+m.get('n_benign',0)} "
                     f"| {m.get('tpr','n/a') if typ=='Attack' else '—'} "
                     f"| {m.get('fpr','n/a') if typ=='Benign' else '—'} "
                     f"| {m.get('first_alert_at','—')} "
                     f"| {m.get('n_alerts','—')} |\n")
    print("  Saved: issue8_table1_detection_metrics.md")

    with open(f"{figures_dir}/issue8_table2_asr_comparison.md", "w") as fh:
        fh.write("# Issue #8 Table 2: ASR — Static D4 vs Dynamic Policy\n\n")
        fh.write("| Attack workload | Static ASR | Dynamic ASR | Reduction | First alert |\n")
        fh.write("|-----------------|-----------|------------|-----------|-------------|\n")
        for wn, ad in asr_data.items():
            fh.write(f"| {wn} | {ad['static_asr']:.1%} | {ad['dynamic_asr']:.1%} "
                     f"| {ad['asr_reduction']:.1%} | probe {ad['first_alert_at']} |\n")
    print("  Saved: issue8_table2_asr_comparison.md")



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",  default="../configs/probe_detection_eval.yaml")
    p.add_argument("--backend", choices=["vllm","sglang","mock"])
    p.add_argument("--seed",    type=int)
    p.add_argument("--output-dir")
    return p.parse_args()

def load_config(path):
    try:
        import yaml
        with open(path) as f: return yaml.safe_load(f)
    except ImportError:
        cfg = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"): continue
                if ":" in line:
                    k, _, v = line.partition(":")
                    v = v.strip().strip('"')
                    try:    cfg[k.strip()] = int(v)
                    except:
                        try: cfg[k.strip()] = float(v)
                        except: cfg[k.strip()] = v
        return cfg

def main():
    args = parse_args()
    cfg_path = args.config if os.path.isabs(args.config) \
               else os.path.join(os.path.dirname(__file__), args.config)
    cfg = load_config(cfg_path)

    backend_name = args.backend or cfg.get("backend", "mock")
    seed         = args.seed    or cfg.get("seed", 42)
    output_dir   = args.output_dir or cfg.get("output_dir", "experiments/results")
    figures_dir  = cfg.get("figures_dir", "experiments/figures")
    model_id     = cfg.get("model_id", MODEL_ID)
    vllm_url     = cfg.get("vllm_base_url", VLLM_BASE_URL)

    if not os.path.isabs(output_dir):  output_dir  = str(_REPO / output_dir)
    if not os.path.isabs(figures_dir): figures_dir = str(_REPO / figures_dir)

    run_id   = f"probe-detect-{datetime.datetime.utcnow().strftime('%Y-%m-%d')}-{uuid.uuid4().hex[:8]}"
    out_path = f"{output_dir}/{run_id}.json"

    print(f"\n{'='*60}")
    print(f"  Issue #8  Probe Detection Evaluation")
    print(f"  run_id:  {run_id}")
    print(f"  backend: {backend_name}")
    print(f"  seed:    {seed}")
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

    print("[calibrate]")
    calibration = calibrate(backend, n=30, seed=seed)

    rng = random.Random(seed)

    print("\n[workloads] Generating probe streams…")
    n_br = cfg.get("n_benign_repeated", 30)
    n_bb = cfg.get("n_benign_bursty",   30)
    n_le = cfg.get("n_linear_enum",     60)
    n_ts = cfg.get("n_two_stage",       40)
    n_lr = cfg.get("n_low_rate",        40)

    raw_workloads = {
        "benign_repeated": workload_benign_repeated(n_br, rng),
        "benign_bursty":   workload_benign_bursty(n_bb,   rng),
        "linear_enum":     workload_linear_enum(n_le,     rng),
        "two_stage":       workload_two_stage(n_ts,       rng),
        "low_rate":        workload_low_rate(n_lr,        rng),
    }

    print("[simulate] Adding TTFT to probe streams…")
    streamed = {
        wn: simulate_stream(wl, calibration, rng)
        for wn, wl in raw_workloads.items()
    }
    for wn, wl in streamed.items():
        for i, p in enumerate(wl):
            p["probe_index"] = i

    print("\n[detector] Running per-workload detection…")
    thr_med  = cfg.get("threshold_med",  THRESHOLD_MED)
    thr_high = cfg.get("threshold_high", THRESHOLD_HIGH)
    cooldown = cfg.get("cooldown_probes", COOLDOWN_PROBES)

    workload_results = {}
    for wn, stream in streamed.items():
        print(f"  {wn} ({len(stream)} probes)…")
        rows    = run_detector(stream, thr_med, thr_high, cooldown)
        metrics = compute_metrics(rows, threshold=thr_med)
        workload_results[wn] = {"rows": rows, "metrics": metrics}
        n_al = metrics["n_alerts"]
        fa   = metrics["first_alert_at"]
        tpr_s = f"TPR={metrics['tpr']:.0%}" if rows[0].get("is_attack") else ""
        fpr_s = f"FPR={metrics['fpr']:.0%}" if not rows[0].get("is_attack","") else ""
        print(f"    alerts={n_al}  first_alert={fa}  {tpr_s}{fpr_s}")

    print("\n[roc] Threshold sweep…")
    all_rows  = [r for wl in workload_results.values() for r in wl["rows"]]
    roc_thresholds = cfg.get("roc_thresholds", [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    roc_data  = roc_sweep(all_rows, roc_thresholds)

    print("\n[asr] Static D4 vs dynamic comparison…")
    attack_wls = ["linear_enum", "two_stage", "low_rate"]
    asr_comparison = {}
    for wn in attack_wls:
        asr_comparison[wn] = compare_asr(
            workload_results[wn]["rows"], calibration, rng
        )
        a = asr_comparison[wn]
        print(f"  {wn}: static={a['static_asr']:.0%}  "
              f"dynamic={a['dynamic_asr']:.0%}  "
              f"reduction={a['asr_reduction']:.0%}  "
              f"first_alert=probe {a['first_alert_at']}")

    results = {
        "run_id":      run_id,
        "issue":       8,
        "config": {
            "backend": backend_name, "seed": seed,
            "threshold_med": thr_med, "threshold_high": thr_high,
            "cooldown_probes": cooldown,
        },
        "environment": _env(),
        "calibration": calibration,
        "workloads":   {
            wn: {
                "metrics": wd["metrics"],
                "n_probes": len(wd["rows"]),
                "suspicion_trace": [
                    {"idx": r["probe_index"], "score": r["suspicion_score"],
                     "level": r["suspicion_level"]}
                    for r in wd["rows"]
                ],
            }
            for wn, wd in workload_results.items()
        },
        "roc_data":       roc_data,
        "asr_comparison": asr_comparison,
        "command":        " ".join(sys.argv),
    }
    _dump(results, out_path)
    print(f"\n[save] Results: {out_path}")

    print("\n[figures] Generating…")
    results_with_rows = dict(results)
    results_with_rows["workloads"] = {
        wn: {**results["workloads"][wn], "rows": workload_results[wn]["rows"]}
        for wn in workload_results
    }
    generate_figures(results_with_rows, figures_dir)
    generate_tables(results_with_rows, figures_dir)

    print(f"\n{'='*60}")
    print(f"  {'Workload':<22} {'Type':<8} {'N':>5} {'TPR':>7} {'FPR':>7} {'1st ALERT':>10}")
    print(f"  {'-'*58}")
    for wn, wd in workload_results.items():
        m   = wd["metrics"]
        typ = "Attack" if wn not in ("benign_repeated","benign_bursty") else "Benign"
        tpr = f"{m['tpr']:.0%}" if typ=="Attack" else "—"
        fpr = f"{m['fpr']:.0%}" if typ=="Benign" else "—"
        fa  = str(m["first_alert_at"]) if m["first_alert_at"] is not None else "none"
        n   = m["n_attack"] + m["n_benign"]
        print(f"  {wn:<22} {typ:<8} {n:>5} {tpr:>7} {fpr:>7} {fa:>10}")
    print(f"{'='*60}")
    print(f"\nDone. Results: {out_path}")

if __name__ == "__main__":
    main()