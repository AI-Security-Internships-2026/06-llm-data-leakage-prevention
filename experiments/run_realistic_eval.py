"""
experiments/run_realistic_eval.py
===================================
Issue #4 — Evaluate adaptive attack on realistic prompt layouts and
attacker knowledge levels.

Sweeps:
  - Scenarios: T1_medical, T2_financial, T3_enterprise
  - Knowledge levels: K1 (exact/small), K2 (exact/large), K3 (partial template)
  - Algorithms: linear, two_stage_adaptive

Usage
-----
  # Full sweep (requires vLLM server):
  python experiments/run_realistic_eval.py --config configs/realistic_attack.yaml

  # Mock-backend smoke test (no GPU):
  python experiments/run_realistic_eval.py --config configs/realistic_attack.yaml --backend mock

  # Single scenario:
  python experiments/run_realistic_eval.py --config configs/realistic_attack.yaml \\
      --scenario T1_medical --knowledge K1
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
import time
import uuid
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from kv_attack.realistic_scenarios import (
    SCENARIOS, KNOWLEDGE_LEVELS,
    generate_realistic_victims, candidate_entropy_bits,
)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",    default="configs/realistic_attack.yaml")
    p.add_argument("--backend",   choices=["vllm", "sglang", "mock"], default="vllm")
    p.add_argument("--base-url",  default="http://localhost:8001/v1")
    p.add_argument("--model-id",  default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    p.add_argument("--n-victims", type=int, default=30)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--scenario",  choices=list(SCENARIOS.keys()) + ["all"], default="all")
    p.add_argument("--knowledge", choices=["K1", "K2", "K3", "all"], default="all")
    p.add_argument("--output-dir", default="experiments/results")
    return p.parse_args()


def _mock_attack(victims: list[dict], t1_ms: float, t2_ms: float,
                 knowledge_level: str, algorithm: str, seed: int) -> list[dict]:
    """
    Simulate the attack using a deterministic mock oracle.
    Returns results in the standard schema.
    """
    import random, numpy as np
    rng = random.Random(seed)

    hit_ms, s1_ms, miss_ms, noise = 184.4, 519.4, 1336.2, 15.0

    results = []
    for v in victims:
        names   = v["attacker_names"]
        secrets = v["attacker_secrets"]
        is_partial = v.get("partial_template", False)

        partial_penalty = 3 if is_partial else 1

        stage1_calls = 0
        found_name   = None

        for cand_name in names:
            ttft_samples = []
            for _ in range(3 if algorithm != "linear" else 1):
                if cand_name == v["name"] and not is_partial:
                    ttft_samples.append(rng.gauss(s1_ms, noise))
                else:
                    ttft_samples.append(rng.gauss(miss_ms, noise * partial_penalty))
            mean_ttft = float(np.mean(ttft_samples))
            stage1_calls += len(ttft_samples)
            if mean_ttft < t1_ms:
                found_name = cand_name
                break

        stage2_calls = 0
        found_secret = None
        if found_name and algorithm == "two_stage_adaptive":
            for cand_sec in secrets:
                ttft = rng.gauss(
                    hit_ms if cand_sec == v["secret"] else s1_ms,
                    noise * partial_penalty
                )
                stage2_calls += 1
                if ttft < t2_ms:
                    found_secret = cand_sec
                    break
        elif found_name:
            for cand_sec in secrets:
                combined = f"{found_name}|{cand_sec}"
                ttft = rng.gauss(
                    hit_ms if (found_name == v["name"] and cand_sec == v["secret"]) else miss_ms,
                    noise * partial_penalty
                )
                stage1_calls += 1
                if ttft < t1_ms:
                    found_secret = cand_sec
                    break

        exact = (found_name == v["name"]) and (found_secret == v["secret"])
        total = stage1_calls + stage2_calls
        h0    = candidate_entropy_bits(v)

        results.append({
            "victim_id"         : v["victim_id"],
            "scenario"          : v["scenario"],
            "knowledge_level"   : v["knowledge_level"],
            "algorithm"         : algorithm,
            "exact_match"       : exact,
            "name_recovered"    : found_name == v["name"],
            "secret_recovered"  : found_secret == v["secret"],
            "total_api_calls"   : total,
            "stage1_api_calls"  : stage1_calls,
            "stage2_api_calls"  : stage2_calls,
            "candidate_space"   : len(names) * len(secrets),
            "entropy_bits"      : round(h0, 4),
            "partial_template"  : is_partial,
            "t1_threshold_ms"   : t1_ms,
            "t2_threshold_ms"   : t2_ms,
        })

    return results


def _real_attack(
    backend, victims: list[dict], t1_ms: float, t2_ms: float,
    knowledge_level: str, algorithm: str, seed: int
) -> list[dict]:
    """
    Run the attack against a REAL backend (vLLM/SGLang) using actual TTFT probes.
    Mirrors _mock_attack's logic but calls backend.measure_ttft() for real timing.
    """
    import numpy as np

    results = []
    for v in victims:
        names   = v["attacker_names"]
        secrets = v["attacker_secrets"]

        stage1_calls = 0
        found_name   = None
        n_repeats    = 3 if algorithm != "linear" else 1

        for cand_name in names:
            ttft_samples = []
            for _ in range(n_repeats):
                prompt = v["scenario_prompt_fn"](cand_name, v.get("dummy_secret", secrets[0]))
                ttft_samples.append(backend.measure_ttft(prompt))
            stage1_calls += len(ttft_samples)
            mean_ttft = float(np.mean(ttft_samples))
            if mean_ttft < t1_ms:
                found_name = cand_name
                break

        stage2_calls = 0
        found_secret = None
        if found_name and algorithm == "two_stage_adaptive":
            for cand_sec in secrets:
                prompt = v["scenario_prompt_fn"](found_name, cand_sec)
                ttft = backend.measure_ttft(prompt)
                stage2_calls += 1
                if ttft < t2_ms:
                    found_secret = cand_sec
                    break
        elif found_name:
            for cand_sec in secrets:
                prompt = v["scenario_prompt_fn"](found_name, cand_sec)
                ttft = backend.measure_ttft(prompt)
                stage1_calls += 1
                if ttft < t1_ms:
                    found_secret = cand_sec
                    break

        exact = (found_name == v["name"]) and (found_secret == v["secret"])
        total = stage1_calls + stage2_calls
        h0    = candidate_entropy_bits(v)

        results.append({
            "victim_id"         : v["victim_id"],
            "scenario"          : v["scenario"],
            "knowledge_level"   : v["knowledge_level"],
            "algorithm"         : algorithm,
            "exact_match"       : exact,
            "name_recovered"    : found_name == v["name"],
            "secret_recovered"  : found_secret == v["secret"],
            "total_api_calls"   : total,
            "stage1_api_calls"  : stage1_calls,
            "stage2_api_calls"  : stage2_calls,
            "candidate_space"   : len(names) * len(secrets),
            "entropy_bits"      : round(h0, 4),
            "partial_template"  : v.get("partial_template", False),
            "t1_threshold_ms"   : t1_ms,
            "t2_threshold_ms"   : t2_ms,
        })

    return results


def _aggregate(results: list[dict], algorithm: str) -> dict:
    import numpy as np
    n     = len(results)
    exact = [r["exact_match"] for r in results]
    calls = [r["total_api_calls"] for r in results]
    h0s   = [r["entropy_bits"] for r in results]
    return {
        "algorithm"             : algorithm,
        "n_victims"             : n,
        "exact_recovery_count"  : int(sum(exact)),
        "asr"                   : float(sum(exact)) / n,
        "mean_total_api_calls"  : round(float(np.mean(calls)), 2),
        "median_total_api_calls": float(np.median(calls)),
        "p95_total_api_calls"   : float(np.percentile(calls, 95)),
        "mean_entropy_bits"     : round(float(np.mean(h0s)), 4),
        "mean_blq"              : round(float(np.mean(h0s)) / max(float(np.mean(calls)), 1), 6),
    }


def _incremental_save(args, all_conditions: list, t1_ms: float, t2_ms: float) -> Path:
    """Write current progress to disk immediately after every condition finishes."""
    run_id   = getattr(args, "_run_id", None)
    if run_id is None:
        run_id = f"realistic-eval-{datetime.date.today()}-{uuid.uuid4().hex[:8]}"
        args._run_id = run_id
    out_path = Path(args.output_dir) / f"{run_id}.json"

    payload = {
        "run_id"      : run_id,
        "backend"     : args.backend,
        "model_id"    : args.model_id,
        "n_victims"   : args.n_victims,
        "seed"        : args.seed,
        "t1_ms"       : t1_ms,
        "t2_ms"       : t2_ms,
        "n_conditions_done": len(all_conditions),
        "conditions"  : all_conditions,
        "timestamp"   : datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    tmp_path = out_path.with_suffix(".tmp.json")
    with open(tmp_path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    tmp_path.replace(out_path)

    print(f"    [Incremental save: {len(all_conditions)} conditions → {out_path}]")
    return out_path


def _calibrate_scenario_thresholds(backend, scenario, sample_dob: str, n_samples: int = 15) -> tuple[float, float]:
    """
    Calibrate T1/T2 thresholds for THIS scenario's prompt template by measuring
    a real miss distribution (wrong name+secret) and a real hit distribution
    (correct name, but scanning through the block regions).

    Different prompt templates (medical/financial/enterprise) have very
    different lengths and structures than the controlled two-stage prompt,
    so thresholds must be re-measured per scenario rather than reused.
    """
    import numpy as np, random
    rng = random.Random(0)

    fake_name   = "Zephyr Quicksilver"
    fake_secret = "nonexistent-control-value-xyz"

    miss_samples = []
    for _ in range(n_samples):
        prompt = scenario.build_prompt("", fake_name, sample_dob, fake_secret, None)
        miss_samples.append(backend.measure_ttft(prompt))

    true_name   = scenario.first_names[0] + " " + scenario.last_names[0]
    true_secret = scenario.secret_fields[0]
    seed_prompt = scenario.build_prompt("", true_name, sample_dob, true_secret, None)
    backend.measure_ttft(seed_prompt)

    hit_samples = []
    for _ in range(n_samples):
        prompt = scenario.build_prompt("", true_name, sample_dob, true_secret, None)
        hit_samples.append(backend.measure_ttft(prompt))

    miss_p50 = float(np.percentile(miss_samples, 50))
    hit_p50  = float(np.percentile(hit_samples, 50))

    t1_ms = (miss_p50 + hit_p50) / 2
    t2_ms = t1_ms * 0.5

    print(f"    [calibrate] {scenario.name}: miss_p50={miss_p50:.0f}ms  "
          f"hit_p50={hit_p50:.0f}ms  → T1={t1_ms:.0f}ms  T2={t2_ms:.0f}ms")

    return t1_ms, t2_ms


def main():
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    scenarios  = list(SCENARIOS.keys()) if args.scenario == "all" else [args.scenario]
    know_levels = ["K1", "K2", "K3"]   if args.knowledge == "all" else [args.knowledge]
    algorithms  = ["linear", "two_stage_adaptive"]

    T1_MS_FALLBACK = 1017.3
    T2_MS_FALLBACK = 377.2

    backend = None
    if args.backend == "vllm":
        from kv_attack.backends.vllm_backend import VLLMBackend
        backend = VLLMBackend(base_url=args.base_url, model_id=args.model_id)
        print(f"[realistic_eval] Using REAL vLLM backend @ {args.base_url}")
    elif args.backend == "sglang":
        from kv_attack.backends.sglang_backend import make_sglang_backend
        backend = make_sglang_backend(base_url=args.base_url, model_id=args.model_id)
    else:
        print("[realistic_eval] Using MOCK backend (simulated — no real API calls)")

    all_conditions = []
    scenario_thresholds: dict[str, tuple[float, float]] = {}

    for sc_name in scenarios:
        scenario = SCENARIOS[sc_name]

        if backend is not None and sc_name not in scenario_thresholds:
            print(f"\n  [calibrate] Measuring TTFT for {sc_name} template...")
            sample_dob = "1980-01-01"
            t1_ms, t2_ms = _calibrate_scenario_thresholds(backend, scenario, sample_dob)
            scenario_thresholds[sc_name] = (t1_ms, t2_ms)
        T1_MS, T2_MS = scenario_thresholds.get(sc_name, (T1_MS_FALLBACK, T2_MS_FALLBACK))

        for kl_name in know_levels:
            kl = KNOWLEDGE_LEVELS[sc_name][kl_name]
            victims = generate_realistic_victims(scenario, kl, args.n_victims, args.seed)

            for v in victims:
                v["scenario_prompt_fn"] = (
                    lambda cand_name, cand_secret, _sc=scenario, _dob=v["dob"]:
                        _sc.build_prompt("", cand_name, _dob, cand_secret, None)
                )

            print(f"\n{'='*60}")
            print(f"  Scenario: {sc_name} | Knowledge: {kl_name}")
            print(f"  Candidates: {len(victims[0]['attacker_names'])} names × "
                  f"{len(victims[0]['attacker_secrets'])} secrets = "
                  f"{len(victims[0]['attacker_names'])*len(victims[0]['attacker_secrets'])}")
            print(f"  H0: {candidate_entropy_bits(victims[0]):.2f} bits")

            for algo in algorithms:
                print(f"  Running {algo}...")
                t0 = time.time()

                if backend is not None:
                    print(f"    Seeding {len(victims)} victims into cache...")
                    for v in victims:
                        seed_prompt = v["scenario_prompt_fn"](v["name"], v["secret"])
                        backend.measure_ttft(seed_prompt)
                    results = _real_attack(backend, victims, T1_MS, T2_MS, kl_name, algo, args.seed)
                else:
                    results = _mock_attack(victims, T1_MS, T2_MS, kl_name, algo, args.seed)

                elapsed = time.time() - t0
                agg     = _aggregate(results, algo)
                agg["wall_clock_s"] = round(elapsed, 2)

                print(f"    ASR={agg['asr']:.2f}  "
                      f"median_q={agg['median_total_api_calls']:.0f}  "
                      f"BLQ={agg['mean_blq']:.4f}  "
                      f"elapsed={elapsed:.1f}s")

                clean_results = [
                    {k: v for k, v in r.items()} for r in results
                ]

                all_conditions.append({
                    "scenario"     : sc_name,
                    "knowledge"    : kl_name,
                    "algorithm"    : algo,
                    "aggregate"    : agg,
                    "results"      : clean_results,
                })

                _incremental_save(args, all_conditions, T1_MS, T2_MS)


    out_path = _incremental_save(args, all_conditions, T1_MS, T2_MS)

    with open(out_path) as fh:
        payload = json.load(fh)

    print(f"\n[realistic_eval] Final results saved → {out_path}")

    _generate_issue4_figures(payload, Path(args.output_dir).parent / "figures")



def _generate_issue4_figures(data: dict, fig_dir: Path) -> None:
    """Generate Figures 1-3 and Tables 1-2 for Issue #4."""
    import csv, io
    fig_dir.mkdir(parents=True, exist_ok=True)
    conditions = data["conditions"]

    headers = ["Template", "Prior (knowledge)", "Candidate space",
               "Attack", "Exact recovery", "Median queries", "Time (s)"]
    rows = []
    for c in conditions:
        agg = c["aggregate"]
        rows.append({
            "Template"          : c["scenario"],
            "Prior (knowledge)" : c["knowledge"],
            "Candidate space"   : f"{int(2**agg['mean_entropy_bits'])} "
                                  f"({agg['mean_entropy_bits']:.1f} bits)",
            "Attack"            : c["algorithm"],
            "Exact recovery"    : f"{agg['asr']*100:.1f}% ({agg['exact_recovery_count']}/{agg['n_victims']})",
            "Median queries"    : f"{agg['median_total_api_calls']:.0f}",
            "Time (s)"          : f"{agg.get('wall_clock_s', 'n/a')}",
        })

    md  = "| " + " | ".join(headers) + " |\n"
    md += "| " + " | ".join(["---"]*len(headers)) + " |\n"
    for r in rows:
        md += "| " + " | ".join(str(r[h]) for h in headers) + " |\n"
    (fig_dir / "issue4_table1_attack_comparison.md").write_text(md)

    buf = io.StringIO()
    w   = csv.DictWriter(buf, fieldnames=headers); w.writeheader(); w.writerows(rows)
    (fig_dir / "issue4_table1_attack_comparison.csv").write_text(buf.getvalue())

    fail_rows = []
    for c in conditions:
        res     = c["results"]
        partial = c["knowledge"] == "K3"
        n_name_fail = sum(1 for r in res if not r.get("name_recovered", False))
        n_sec_fail  = sum(1 for r in res
                          if r.get("name_recovered", False) and not r.get("secret_recovered", False))
        n_both_fail = sum(1 for r in res if not r["exact_match"])
        fail_rows.append({
            "Scenario"      : c["scenario"],
            "Knowledge"     : c["knowledge"],
            "Algorithm"     : c["algorithm"],
            "Name failures" : n_name_fail,
            "Secret failures": n_sec_fail,
            "Total failures": n_both_fail,
            "Cause"         : "Partial template noise" if partial else "Candidate not in dict",
        })
    h2  = ["Scenario", "Knowledge", "Algorithm", "Name failures",
           "Secret failures", "Total failures", "Cause"]
    md2 = "| " + " | ".join(h2) + " |\n"
    md2+= "| " + " | ".join(["---"]*len(h2)) + " |\n"
    for r in fail_rows:
        md2 += "| " + " | ".join(str(r[h]) for h in h2) + " |\n"
    (fig_dir / "issue4_table2_failure_categories.md").write_text(md2)

    fig_data = {
        "fig1_median_queries_vs_entropy": [
            {"scenario": c["scenario"], "knowledge": c["knowledge"],
             "algorithm": c["algorithm"],
             "entropy_bits": c["aggregate"]["mean_entropy_bits"],
             "median_queries": c["aggregate"]["median_total_api_calls"]}
            for c in conditions
        ],
        "fig2_asr_by_scenario": [
            {"scenario": c["scenario"], "knowledge": c["knowledge"],
             "algorithm": c["algorithm"], "asr": c["aggregate"]["asr"]}
            for c in conditions
        ],
    }
    (fig_dir / "issue4_figures_data.json").write_text(json.dumps(fig_data, indent=2))

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = {"linear": "#e55", "two_stage_adaptive": "#3a3"}
        markers = {"K1": "o", "K2": "s", "K3": "^"}
        for c in conditions:
            agg = c["aggregate"]
            ax.scatter(agg["mean_entropy_bits"], agg["median_total_api_calls"],
                       color=colors.get(c["algorithm"], "#888"),
                       marker=markers.get(c["knowledge"], "o"),
                       s=80, alpha=0.8,
                       label=f"{c['algorithm']}|{c['knowledge']}")
        ax.set_xlabel("Candidate space entropy H0 (bits)")
        ax.set_ylabel("Median API queries")
        ax.set_title("Fig 1 (Issue #4) — Median Queries vs Entropy")
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / "issue4_fig1_queries_vs_entropy.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        x_labels = [f"{c['scenario']}\n{c['knowledge']}" for c in conditions
                    if c["algorithm"] == "two_stage_adaptive"]
        asrs     = [c["aggregate"]["asr"] for c in conditions
                    if c["algorithm"] == "two_stage_adaptive"]
        asrs_lin = [c["aggregate"]["asr"] for c in conditions
                    if c["algorithm"] == "linear"]
        xs = list(range(len(x_labels)))
        ax.bar([x-0.2 for x in xs], asrs_lin, 0.35, label="Linear",         color="#e55", alpha=0.8)
        ax.bar([x+0.2 for x in xs], asrs,     0.35, label="Two-stage adapt.", color="#3a3", alpha=0.8)
        ax.set_xticks(xs); ax.set_xticklabels(x_labels, fontsize=7)
        ax.set_ylim(0, 1.1); ax.set_ylabel("ASR (exact recovery rate)")
        ax.set_title("Fig 2 (Issue #4) — ASR: Controlled vs Realistic Templates")
        ax.legend(); fig.tight_layout()
        fig.savefig(fig_dir / "issue4_fig2_asr_by_scenario.png", dpi=150)
        plt.close(fig)

        print(f"[realistic_eval] Figures saved → {fig_dir}")
    except ImportError:
        print("[realistic_eval] matplotlib not installed — JSON data saved; plots skipped.")


if __name__ == "__main__":
    main()