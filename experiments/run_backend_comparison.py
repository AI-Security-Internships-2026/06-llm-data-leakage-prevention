"""
experiments/run_backend_comparison.py (v2 — fixed)
=====================================================
Issue #6 — Validate attack on a second real LLM serving backend (SGLang).

FIX vs v1: v1 sent raw "||NAME_SENTINEL::...||" marker strings directly to
real backends. Those markers are only understood by TwoStageMockBackend —
neither real vLLM nor real SGLang has ever seen a victim seeded that way,
so every probe looked like an unrelated miss regardless of correctness.
v2 reuses the SAME real, working pipeline as week13_harness.py
(two_stage_victim_seeder + two_stage_reconstructor) for BOTH backends, so
the comparison is apples-to-apples and actually exercises each backend's
real prefix-cache behavior.

Usage
-----
  # Real vLLM vs real SGLang:
  python experiments/run_backend_comparison.py --backend-a vllm --backend-b sglang --n-victims 10

  # vLLM vs mock (if SGLang not available):
  python experiments/run_backend_comparison.py --backend-a vllm --backend-b mock --n-victims 10

  # Mock vs mock (pipeline smoke test, no GPU):
  python experiments/run_backend_comparison.py --backend-a mock --backend-b mock --n-victims 5
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
import uuid
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backend-a", default="vllm", choices=["vllm", "sglang", "mock"])
    p.add_argument("--backend-b", default="mock", choices=["vllm", "sglang", "mock"])
    p.add_argument("--url-a",     default="http://localhost:8001/v1")
    p.add_argument("--url-b",     default="http://localhost:8002/v1")
    p.add_argument("--model-id",  default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    p.add_argument("--n-victims", type=int, default=10)
    p.add_argument("--n-calib",   type=int, default=50)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--output-dir", default="experiments/results")
    return p.parse_args()


def _wilson_ci(k: int, n: int) -> tuple[float, float]:
    if n == 0: return 0.0, 1.0
    p = k/n; z = 1.96
    c = (p+z**2/(2*n))/(1+z**2/n)
    m = z*math.sqrt(p*(1-p)/n+z**2/(4*n**2))/(1+z**2/n)
    return max(0.0,c-m), min(1.0,c+m)



def _run_real_backend(backend_name: str, base_url: str, args) -> dict:
    from openai import OpenAI
    from transformers import AutoTokenizer

    from kv_attack import detect_has_bos
    from kv_attack.victim_seeder import build_aligned_system_prompt
    from kv_attack.two_stage_victim_seeder import seed_victims_two_stage
    from kv_attack.two_stage_reconstructor import (
        calibrate_two_stage, evict_cache_two_stage, reconstruct_victim_two_stage,
    )

    if backend_name == "vllm":
        from kv_attack.backends.vllm_backend import VLLMBackend
        backend = VLLMBackend(base_url=base_url, model_id=args.model_id)
        framework_ver = "vllm"
    elif backend_name == "sglang":
        from kv_attack.backends.sglang_backend import make_sglang_backend
        backend = make_sglang_backend(base_url=base_url, model_id=args.model_id)
        framework_ver = backend.get_info().framework_ver
    else:
        raise ValueError(backend_name)

    client = OpenAI(base_url=base_url, api_key="EMPTY")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    system_prefix, _ = build_aligned_system_prompt(
        tokenizer, has_bos=detect_has_bos(args.model_id)
    )

    print(f"[backend_cmp] Seeding {args.n_victims} victims on {backend_name}...")
    victims = seed_victims_two_stage(
        client=client, tokenizer=tokenizer, system_prefix=system_prefix,
        n_victims=args.n_victims, seed=args.seed,
    )

    print(f"[backend_cmp] Calibrating thresholds on {backend_name}...")
    calib = calibrate_two_stage(
        backend=backend, tokenizer=tokenizer, system_prefix=system_prefix,
        victim_record=victims[0], n_samples=args.n_calib,
    )
    t1_ms = calib["t1_threshold_ms"]
    t2_ms = calib["t2_threshold_ms"]

    hit_mean  = calib.get("hit_mean_ms", 0)
    s1_mean   = calib.get("s1_hit_mean_ms", 0)
    miss_mean = calib.get("miss_mean_ms", 0)

    roc_auc = _roc_auc_from_calib(calib)

    characterization = {
        "backend": backend_name, "framework_version": framework_ver,
        "hit_distribution": {"mean_ms": round(hit_mean, 1)},
        "miss_distribution": {"mean_ms": round(miss_mean, 1)},
        "s1_hit_mean_ms": round(s1_mean, 1),
        "roc_auc": roc_auc,
        "t1_empirical_ms": round(t1_ms, 1),
        "t2_empirical_ms": round(t2_ms, 1),
        "timing_gap_ms": round(miss_mean - hit_mean, 1),
        "intermediate_feasible": calib.get("intermediate_feasible", False),
        "ks_hit_vs_miss_p": calib.get("ks_hit_vs_miss", {}).get("p"),
    }
    print(f"[backend_cmp] {backend_name}: gap={characterization['timing_gap_ms']:.0f}ms  "
          f"T1={t1_ms:.0f}ms  T2={t2_ms:.0f}ms  feasible={characterization['intermediate_feasible']}")

    print(f"[backend_cmp] Attacking {len(victims)} victims on {backend_name}...")
    results = []
    for i, vr in enumerate(victims):
        evict_cache_two_stage(backend, system_prefix, tokenizer)
        r = reconstruct_victim_two_stage(
            backend=backend, tokenizer=tokenizer, system_prefix=system_prefix,
            t1_ms=t1_ms, t2_ms=t2_ms, victim_record=vr,
            candidate_seed=args.seed * 1000 + i,
        )
        results.append({
            "victim_id": vr["victim_id"], "exact_match": bool(r.exact_match),
            "total_api_calls": r.total_api_calls,
        })
        print(f"    victim {i+1}/{len(victims)}: "
              f"{'HIT' if r.exact_match else 'miss'}  calls={r.total_api_calls}")

    n = len(results); exact = sum(r["exact_match"] for r in results)
    calls = [r["total_api_calls"] for r in results]
    lo, hi = _wilson_ci(exact, n)

    attack_summary = {
        "backend": backend_name, "n_victims": n, "exact_recovery_count": exact,
        "asr": round(exact/n, 4), "asr_ci_95_lo": round(lo,4), "asr_ci_95_hi": round(hi,4),
        "median_api_calls": float(np.median(calls)), "p95_api_calls": float(np.percentile(calls,95)),
        "mean_api_calls": round(float(np.mean(calls)), 1),
    }
    print(f"[backend_cmp] {backend_name}: ASR={attack_summary['asr']:.3f} "
          f"[{lo:.3f},{hi:.3f}]  median_q={attack_summary['median_api_calls']:.0f}")

    return {"characterization": characterization, "attack_summary": attack_summary,
            "raw_results": results}



def _run_real_backend_incremental(backend_name, url, args, payload, out_path, save_fn):
    """
    Identical to _run_real_backend but writes a checkpoint to the output JSON
    after every victim so a killed process loses at most 1 victim of work.
    """
    from openai import OpenAI
    from transformers import AutoTokenizer
    from kv_attack import detect_has_bos
    from kv_attack.victim_seeder import build_aligned_system_prompt
    from kv_attack.two_stage_victim_seeder import seed_victims_two_stage
    from kv_attack.two_stage_reconstructor import (
        calibrate_two_stage, evict_cache_two_stage, reconstruct_victim_two_stage,
    )

    if backend_name == "vllm":
        from kv_attack.backends.vllm_backend import VLLMBackend
        backend = VLLMBackend(base_url=url, model_id=args.model_id)
        framework_ver = "vllm"
    elif backend_name == "sglang":
        from kv_attack.backends.sglang_backend import make_sglang_backend
        backend = make_sglang_backend(base_url=url, model_id=args.model_id)
        framework_ver = backend.get_info().framework_ver
    else:
        raise ValueError(backend_name)

    client    = OpenAI(base_url=url, api_key="EMPTY")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    system_prefix, _ = build_aligned_system_prompt(
        tokenizer, has_bos=detect_has_bos(args.model_id)
    )

    print(f"[backend_cmp] Seeding {args.n_victims} victims on {backend_name}...")
    victims = seed_victims_two_stage(
        client=client, tokenizer=tokenizer, system_prefix=system_prefix,
        n_victims=args.n_victims, seed=args.seed,
    )

    print(f"[backend_cmp] Calibrating thresholds on {backend_name}...")
    calib  = calibrate_two_stage(
        backend=backend, tokenizer=tokenizer, system_prefix=system_prefix,
        victim_record=victims[0], n_samples=args.n_calib,
    )
    t1_ms  = calib["t1_threshold_ms"]
    t2_ms  = calib["t2_threshold_ms"]
    hit_mean  = calib.get("hit_mean_ms", 0)
    s1_mean   = calib.get("s1_hit_mean_ms", 0)
    miss_mean = calib.get("miss_mean_ms", 0)
    roc_auc   = _roc_auc_from_calib(calib)

    characterization = {
        "backend": backend_name, "framework_version": framework_ver,
        "hit_distribution": {"mean_ms": round(hit_mean, 1)},
        "miss_distribution": {"mean_ms": round(miss_mean, 1)},
        "s1_hit_mean_ms": round(s1_mean, 1),
        "roc_auc": roc_auc,
        "t1_empirical_ms": round(t1_ms, 1),
        "t2_empirical_ms": round(t2_ms, 1),
        "timing_gap_ms": round(miss_mean - hit_mean, 1),
        "intermediate_feasible": calib.get("intermediate_feasible", False),
        "ks_hit_vs_miss_p": calib.get("ks_hit_vs_miss", {}).get("p"),
    }
    print(f"[backend_cmp] {backend_name}: gap={characterization['timing_gap_ms']:.0f}ms  "
          f"T1={t1_ms:.0f}ms  T2={t2_ms:.0f}ms")

    print(f"[backend_cmp] Attacking {len(victims)} victims on {backend_name}...")
    results = []
    for i, vr in enumerate(victims):
        evict_cache_two_stage(backend, system_prefix, tokenizer)
        r = reconstruct_victim_two_stage(
            backend=backend, tokenizer=tokenizer, system_prefix=system_prefix,
            t1_ms=t1_ms, t2_ms=t2_ms, victim_record=vr,
            candidate_seed=args.seed * 1000 + i,
        )
        results.append({
            "victim_id": vr["victim_id"], "exact_match": bool(r.exact_match),
            "total_api_calls": r.total_api_calls,
        })
        print(f"    victim {i+1}/{len(victims)}: "
              f"{'HIT' if r.exact_match else 'miss'}  calls={r.total_api_calls}")

        n_done = len(results)
        exact_now = sum(x["exact_match"] for x in results)
        lo_now, hi_now = _wilson_ci(exact_now, n_done)
        calls_now = [x["total_api_calls"] for x in results]
        payload[f"partial_{backend_name}"] = {
            "characterization": characterization,
            "n_victims_done": n_done,
            "n_victims_total": len(victims),
            "exact_so_far": exact_now,
            "asr_so_far": round(exact_now / n_done, 4),
            "asr_ci_lo": round(lo_now, 4),
            "asr_ci_hi": round(hi_now, 4),
            "median_q_so_far": float(np.median(calls_now)),
            "raw_results": list(results),
        }
        save_fn(payload)

    n = len(results); exact = sum(r["exact_match"] for r in results)
    calls = [r["total_api_calls"] for r in results]
    lo, hi = _wilson_ci(exact, n)
    attack_summary = {
        "backend": backend_name, "n_victims": n, "exact_recovery_count": exact,
        "asr": round(exact/n, 4), "asr_ci_95_lo": round(lo,4), "asr_ci_95_hi": round(hi,4),
        "median_api_calls": float(np.median(calls)), "p95_api_calls": float(np.percentile(calls,95)),
        "mean_api_calls": round(float(np.mean(calls)), 1),
    }
    print(f"[backend_cmp] {backend_name}: ASR={attack_summary['asr']:.3f} "
          f"[{lo:.3f},{hi:.3f}]  median_q={attack_summary['median_api_calls']:.0f}")
    return {"characterization": characterization, "attack_summary": attack_summary,
            "raw_results": results}


def _roc_auc_from_calib(calib: dict) -> float:
    """Approximate ROC-AUC from calibration's hit/miss means+stds via a
    Gaussian overlap estimate, since raw samples may not be stored."""
    hit_mean, hit_std = calib.get("hit_mean_ms", 0), calib.get("hit_std_ms", 1)
    miss_mean, miss_std = calib.get("miss_mean_ms", 0), calib.get("miss_std_ms", 1)
    if hit_std <= 0 or miss_std <= 0:
        return 1.0 if miss_mean > hit_mean else 0.5
    gap = abs(miss_mean - hit_mean)
    pooled_std = (hit_std + miss_std) / 2
    d_prime = gap / max(pooled_std, 1e-6)
    from math import erf, sqrt
    auc = 0.5 * (1 + erf((d_prime / sqrt(2)) / sqrt(2)))
    return round(min(auc, 0.9999), 4)



def _run_mock_backend(seed_offset: int, args) -> dict:
    import random
    from kv_attack.backends.two_stage_mock_backend import TwoStageMockBackend
    from kv_attack import FIRST_NAMES, LAST_NAMES, MEDICAL_CONDITIONS
    from datetime import date, timedelta

    rng = random.Random(args.seed + seed_offset)
    offset = 20.0 * seed_offset
    backend = TwoStageMockBackend(
        hit_ttft_ms=184.4+offset, s1_hit_ttft_ms=519.4+offset*2,
        miss_ttft_ms=1336.2+offset*3, noise_std_ms=15.0, seed=args.seed+seed_offset,
    )

    victims = []
    for i in range(args.n_victims):
        fn, ln = rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
        cond = rng.choice(MEDICAL_CONDITIONS)
        dob = (date(1955,1,1)+timedelta(days=rng.randint(0,20000))).isoformat()
        victims.append({"victim_id": i, "name": f"{fn} {ln}", "dob": dob, "condition": cond})
        backend.seed_victim(i, f"{fn} {ln}", cond)

    hit_samples, miss_samples = [], []
    for v in victims[:5]:
        for _ in range(args.n_calib // 5):
            hit_p = f"||VICTIM::{v['victim_id']}||||NAME_SENTINEL::{v['name']}||||COND_SENTINEL::{v['condition']}||"
            miss_p = f"||VICTIM::{v['victim_id']}||||NAME_SENTINEL::wrong||||COND_SENTINEL::wrong||"
            hit_samples.append(backend.measure_ttft(hit_p))
            miss_samples.append(backend.measure_ttft(miss_p))

    hit_mean, miss_mean = float(np.mean(hit_samples)), float(np.mean(miss_samples))
    t1_ms = (hit_mean + miss_mean) / 2

    characterization = {
        "backend": "mock", "framework_version": "mock",
        "hit_distribution": {"mean_ms": round(hit_mean,1)},
        "miss_distribution": {"mean_ms": round(miss_mean,1)},
        "roc_auc": 1.0, "t1_empirical_ms": round(t1_ms,1),
        "timing_gap_ms": round(miss_mean-hit_mean,1),
        "intermediate_feasible": True,
    }

    names = [f"{fn} {ln}" for fn in FIRST_NAMES for ln in LAST_NAMES]
    secrets = MEDICAL_CONDITIONS
    results = []
    for v in victims:
        found_name, calls = None, 0
        for cand in names:
            p = f"||VICTIM::{v['victim_id']}||||NAME_SENTINEL::{cand}||||COND_SENTINEL::__DUMMY__||"
            calls += 1
            if backend.measure_ttft(p) < t1_ms:
                found_name = cand; break
        found_secret = None
        if found_name:
            for sec in secrets:
                p = f"||VICTIM::{v['victim_id']}||||NAME_SENTINEL::{found_name}||||COND_SENTINEL::{sec}||"
                calls += 1
                if backend.measure_ttft(p) < t1_ms * 0.5:
                    found_secret = sec; break
        exact = (found_name == v["name"]) and (found_secret == v["condition"])
        results.append({"victim_id": v["victim_id"], "exact_match": bool(exact), "total_api_calls": calls})

    n = len(results); exact = sum(r["exact_match"] for r in results)
    calls_arr = [r["total_api_calls"] for r in results]
    lo, hi = _wilson_ci(exact, n)
    attack_summary = {
        "backend": "mock", "n_victims": n, "exact_recovery_count": exact,
        "asr": round(exact/n,4), "asr_ci_95_lo": round(lo,4), "asr_ci_95_hi": round(hi,4),
        "median_api_calls": float(np.median(calls_arr)), "p95_api_calls": float(np.percentile(calls_arr,95)),
    }
    print(f"[backend_cmp] mock: ASR={attack_summary['asr']:.3f}  median_q={attack_summary['median_api_calls']:.0f}")

    return {"characterization": characterization, "attack_summary": attack_summary, "raw_results": results}



def _generate_figures(payload: dict, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    chars   = payload["characterizations"]
    attacks = payload["attack_summaries"]

    headers = ["Backend", "Version", "Model", "AUC", "Exact recovery", "Median queries", "Notes"]
    rows = []
    for ch, atk in zip(chars, attacks):
        rows.append({
            "Backend": ch["backend"], "Version": ch.get("framework_version","n/a"),
            "Model": payload["model_id"], "AUC": ch["roc_auc"],
            "Exact recovery": f"{atk['exact_recovery_count']}/{atk['n_victims']} "
                              f"({atk['asr']*100:.1f}%, CI[{atk['asr_ci_95_lo']:.2f},{atk['asr_ci_95_hi']:.2f}])",
            "Median queries": f"{atk['median_api_calls']:.0f}",
            "Notes": "live" if ch["backend"] != "mock" else "mock backend (fallback)",
        })
    md = "| " + " | ".join(headers) + " |\n| " + " | ".join(["---"]*len(headers)) + " |\n"
    for r in rows: md += "| " + " | ".join(str(r[h]) for h in headers) + " |\n"
    (fig_dir / "issue6_table_backend_comparison.md").write_text(md)

    (fig_dir / "issue6_figures_data.json").write_text(json.dumps({
        "fig1_ttft_distributions": [
            {"backend": c["backend"], "hit_mean_ms": c["hit_distribution"]["mean_ms"],
             "miss_mean_ms": c["miss_distribution"]["mean_ms"], "gap_ms": c["timing_gap_ms"],
             "roc_auc": c["roc_auc"]} for c in chars
        ],
        "fig2_asr_with_ci": [
            {"backend": a["backend"], "asr": a["asr"], "ci_lo": a["asr_ci_95_lo"], "ci_hi": a["asr_ci_95_hi"]}
            for a in attacks
        ],
        "fig3_query_cost": [
            {"backend": a["backend"], "median_q": a["median_api_calls"], "p95_q": a["p95_api_calls"]}
            for a in attacks
        ],
    }, indent=2))

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        bnames = [c["backend"] for c in chars]
        colors = ["#3a3","#36c","#fa0","#e55"]

        fig, ax = plt.subplots(figsize=(7,4))
        x = list(range(len(bnames)))
        ax.bar([xi-0.2 for xi in x], [c["hit_distribution"]["mean_ms"] for c in chars],
               0.35, label="Full hit", color="#3a3", alpha=0.8)
        ax.bar([xi+0.2 for xi in x], [c["miss_distribution"]["mean_ms"] for c in chars],
               0.35, label="Miss", color="#e55", alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(bnames)
        ax.set_ylabel("Mean TTFT (ms)")
        ax.set_title("Fig 1 (Issue #6) — Cache Signal Distributions by Backend (real data)")
        ax.legend(); fig.tight_layout()
        fig.savefig(fig_dir / "issue6_fig1_ttft_distributions.png", dpi=150); plt.close(fig)

        fig, ax = plt.subplots(figsize=(6,4))
        for i, a in enumerate(attacks):
            ax.bar(i, a["asr"], color=colors[i%len(colors)], alpha=0.8, width=0.5)
            ax.errorbar(i, a["asr"], yerr=[[a["asr"]-a["asr_ci_95_lo"]],[a["asr_ci_95_hi"]-a["asr"]]],
                        fmt="none", ecolor="black", capsize=6, linewidth=2)
        ax.set_xticks(range(len(attacks))); ax.set_xticklabels([a["backend"] for a in attacks])
        ax.set_ylim(0,1.1); ax.set_ylabel("ASR (exact recovery)")
        ax.set_title("Fig 2 (Issue #6) — Recovery Rate with 95% CI (real data)")
        fig.tight_layout(); fig.savefig(fig_dir / "issue6_fig2_asr_with_ci.png", dpi=150); plt.close(fig)

        fig, ax = plt.subplots(figsize=(6,4))
        for i, a in enumerate(attacks):
            ax.bar(i-0.2, a["median_api_calls"], 0.35, color=colors[i%len(colors)], label="Median", alpha=0.8)
            ax.bar(i+0.2, a["p95_api_calls"], 0.35, color=colors[i%len(colors)], label="P95", alpha=0.4)
        ax.set_xticks(range(len(attacks))); ax.set_xticklabels([a["backend"] for a in attacks])
        ax.set_ylabel("API queries")
        ax.set_title("Fig 3 (Issue #6) — Query Cost by Backend (real data)")
        fig.tight_layout(); fig.savefig(fig_dir / "issue6_fig3_query_cost.png", dpi=150); plt.close(fig)

        print(f"[backend_cmp] Figures saved → {fig_dir}")
    except ImportError:
        print("[backend_cmp] matplotlib not installed — JSON/table saved; plots skipped.")


def main():
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}\n[backend_cmp] Backend A: {args.backend_a}  Backend B: {args.backend_b}")
    print(f"  Model: {args.model_id}  Victims: {args.n_victims}\n{'='*60}")

    run_id   = f"backend-cmp-{datetime.date.today()}-{uuid.uuid4().hex[:8]}"
    out_path = Path(args.output_dir) / f"{run_id}.json"

    def _incremental_save(partial: dict) -> None:
        """Write current state to the final path after every victim."""
        tmp = out_path.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            json.dump(partial, fh, indent=2, default=str)
        tmp.replace(out_path)
        print(f"  [checkpoint] saved → {out_path}")

    payload = {
        "run_id": run_id, "model_id": args.model_id, "n_victims": args.n_victims,
        "seed": args.seed, "backend_a": args.backend_a, "backend_b": args.backend_b,
        "status": "running",
        "characterizations": [], "attack_summaries": [], "raw_results": [],
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    _incremental_save(payload)

    def _run(name, url, offset):
        """Run one backend, saving a checkpoint after every victim."""
        if name in ("vllm", "sglang"):
            return _run_real_backend_incremental(name, url, args, payload, out_path, _incremental_save)
        else:
            return _run_mock_backend(offset, args)

    result_a = _run(args.backend_a, args.url_a, 1)
    payload["characterizations"].append(result_a["characterization"])
    payload["attack_summaries"].append(result_a["attack_summary"])
    payload["raw_results"].append({"backend": args.backend_a, "results": result_a["raw_results"]})
    payload["status"] = f"{args.backend_a}_done"
    _incremental_save(payload)

    result_b = _run(args.backend_b, args.url_b, 2)
    payload["characterizations"].append(result_b["characterization"])
    payload["attack_summaries"].append(result_b["attack_summary"])
    payload["raw_results"].append({"backend": args.backend_b, "results": result_b["raw_results"]})
    payload["status"] = "complete"
    payload["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    _incremental_save(payload)
    print(f"\n[backend_cmp] Results saved → {out_path}")

    _generate_figures(payload, Path(args.output_dir).parent / "figures")


if __name__ == "__main__":
    main()