"""
experiments/run_load_test.py (v2 — fixed)
============================================
Issue #5 — Evaluate attack under multi-tenant load, cache pressure,
and network noise.

FIX vs v1: v1 sent raw "||NAME_SENTINEL::...||" marker strings directly to a
real vLLM server. Those markers are only understood by TwoStageMockBackend —
real vLLM has never seen a victim seeded in that format, so every probe
against real vLLM looked like an unrelated miss regardless of correctness.
v2 reuses the SAME real, working pipeline as week13_harness.py
(two_stage_victim_seeder + two_stage_reconstructor) for the vLLM path, and
keeps the fast simulated path for --backend mock.

Load levels:
  L0 = 0  background tenants (zero-load baseline)
  L1 = 2  background tenants
  L2 = 5  background tenants
  L3 = 10 background tenants
  L4 = 20 background tenants (highest stable — adapt to hardware)

Usage
-----
  # Real vLLM sweep (slow — reuses the real attack per load level):
  python experiments/run_load_test.py --backend vllm --n-victims 10

  # Mock sweep (no GPU — fast pipeline validation):
  python experiments/run_load_test.py --backend mock --n-victims 5

  # Single load level:
  python experiments/run_load_test.py --backend vllm --load-level L2 --n-victims 10
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import random
import sys
import time
import uuid
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from kv_attack.load_generator import LoadGenerator, LOAD_LEVELS


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backend",    choices=["vllm", "mock"], default="vllm")
    p.add_argument("--base-url",   default="http://localhost:8001/v1")
    p.add_argument("--model-id",   default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    p.add_argument("--n-victims",  type=int, default=10,
                   help="Victims PER load level (real backend is slow — keep modest)")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--load-level", choices=list(LOAD_LEVELS.keys()) + ["all"], default="all")
    p.add_argument("--n-calib",    type=int, default=50)
    p.add_argument("--output-dir", default="experiments/results")
    return p.parse_args()


def _wilson_ci(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    z = 1.96
    c = (p + z**2/(2*n)) / (1 + z**2/n)
    m = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / (1 + z**2/n)
    return max(0.0, c-m), min(1.0, c+m)


def _incremental_save(args, run_id: str, all_levels: list[dict]) -> Path:
    out_path = Path(args.output_dir) / f"{run_id}.json"
    payload = {
        "run_id"        : run_id,
        "backend"       : args.backend,
        "model_id"      : args.model_id,
        "n_victims_per_level": args.n_victims,
        "seed"          : args.seed,
        "n_levels_done" : len(all_levels),
        "levels"        : all_levels,
        "timestamp"     : datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    class _Enc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, np.bool_): return bool(o)
            return super().default(o)

    tmp_path = out_path.with_suffix(".tmp.json")
    with open(tmp_path, "w") as fh:
        json.dump(payload, fh, indent=2, cls=_Enc)
    tmp_path.replace(out_path)
    print(f"  [Incremental save: {len(all_levels)} load levels → {out_path}]")
    return out_path



def _run_real_load_level(level_name: str, n_tenants: int, args) -> dict:
    """
    Run the REAL two-stage attack under background load, using the exact
    same seeding/calibration/reconstruction functions proven to work in
    week13_harness.py (68% SR on 50 victims, real DeepSeek server).
    """
    from openai import OpenAI
    from transformers import AutoTokenizer

    from kv_attack import detect_has_bos
    from kv_attack.backends.vllm_backend import VLLMBackend
    from kv_attack.victim_seeder import build_aligned_system_prompt
    from kv_attack.two_stage_victim_seeder import seed_victims_two_stage
    from kv_attack.two_stage_reconstructor import (
        calibrate_two_stage, evict_cache_two_stage, reconstruct_victim_two_stage,
    )

    print(f"\n{'─'*60}")
    print(f"[load_test] Load level {level_name} ({n_tenants} background tenants)")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    backend   = VLLMBackend(base_url=args.base_url, model_id=args.model_id)
    client    = OpenAI(base_url=args.base_url, api_key="EMPTY")

    system_prefix, _ = build_aligned_system_prompt(
        tokenizer, has_bos=detect_has_bos(args.model_id)
    )

    loader = LoadGenerator(
        base_url=args.base_url, model_id=args.model_id,
        n_tenants=n_tenants, seed=args.seed,
    )

    with loader:
        level_seed = args.seed + hash(level_name) % 10000
        victims = seed_victims_two_stage(
            client=client, tokenizer=tokenizer, system_prefix=system_prefix,
            n_victims=args.n_victims, seed=level_seed,
        )

        print(f"[load_test] Calibrating thresholds under {level_name} load...")
        calib = calibrate_two_stage(
            backend=backend, tokenizer=tokenizer, system_prefix=system_prefix,
            victim_record=victims[0], n_samples=args.n_calib,
        )
        t1_ms = calib["t1_threshold_ms"]
        t2_ms = calib["t2_threshold_ms"]

        oracle = {
            "hit_p50_ms"     : round(calib.get("hit_mean_ms", 0), 1),
            "s1_hit_p50_ms"  : round(calib.get("s1_hit_mean_ms", 0), 1),
            "miss_p50_ms"    : round(calib.get("miss_mean_ms", 0), 1),
            "t1_threshold_ms": round(t1_ms, 1),
            "t2_threshold_ms": round(t2_ms, 1),
            "ks_hit_vs_miss_p"   : calib.get("ks_hit_vs_miss", {}).get("p"),
            "ks_s1hit_vs_miss_p" : calib.get("ks_s1hit_vs_miss", {}).get("p"),
            "ks_hit_vs_s1hit_p"  : calib.get("ks_hit_vs_s1hit", {}).get("p"),
            "intermediate_feasible": calib.get("intermediate_feasible", False),
        }
        print(f"[load_test] {level_name}: T1={t1_ms:.0f}ms T2={t2_ms:.0f}ms "
              f"feasible={oracle['intermediate_feasible']}")

        results = []
        for i, vr in enumerate(victims):
            print(f"[load_test] {level_name}: attacking victim {i+1}/{len(victims)}...")
            evict_cache_two_stage(backend, system_prefix, tokenizer)
            r = reconstruct_victim_two_stage(
                backend=backend, tokenizer=tokenizer, system_prefix=system_prefix,
                t1_ms=t1_ms, t2_ms=t2_ms, victim_record=vr,
                candidate_seed=level_seed * 1000 + i,
            )
            results.append({
                "victim_id"        : vr["victim_id"],
                "exact_match"      : bool(r.exact_match),
                "total_api_calls"  : r.total_api_calls,
                "stage1_api_calls" : r.stage1_api_calls,
                "stage2_api_calls" : r.stage2_api_calls,
            })

        n     = len(results)
        exact = sum(r["exact_match"] for r in results)
        calls = [r["total_api_calls"] for r in results]
        lo, hi = _wilson_ci(exact, n)

        attack_summary = {
            "asr"                : round(exact / n, 4) if n else 0.0,
            "exact_recovery_count": exact,
            "n_victims"          : n,
            "asr_ci_95_lo"       : round(lo, 4),
            "asr_ci_95_hi"       : round(hi, 4),
            "median_api_calls"   : float(np.median(calls)) if calls else 0,
            "p95_api_calls"      : float(np.percentile(calls, 95)) if calls else 0,
        }
        print(f"[load_test] {level_name}: ASR={attack_summary['asr']:.3f} "
              f"[{lo:.3f},{hi:.3f}]  median_q={attack_summary['median_api_calls']:.0f}")

    loader_stats = loader._aggregate_stats()

    return {
        "level"       : level_name,
        "n_tenants"   : n_tenants,
        "oracle"      : oracle,
        "attack"      : attack_summary,
        "loader_stats": loader_stats,
        "raw_results" : results,
    }



def _run_mock_load_level(level_name: str, n_tenants: int, args) -> dict:
    from kv_attack import FIRST_NAMES, LAST_NAMES, MEDICAL_CONDITIONS
    from datetime import date, timedelta

    rng = random.Random(args.seed + hash(level_name) % 10000)
    noise_factor = 1.0 + (n_tenants / 10.0)

    print(f"\n{'─'*55}")
    print(f"[load_test] (MOCK) Load level {level_name} ({n_tenants} tenants)")

    hit_mean, miss_mean = 184.4, 1336.2
    hit_samples  = [max(rng.gauss(hit_mean,  15.0*noise_factor), 1.0) for _ in range(args.n_calib)]
    miss_samples = [max(rng.gauss(miss_mean, 15.0*noise_factor), 1.0) for _ in range(args.n_calib)]

    pos = [-x for x in hit_samples]; neg = [-x for x in miss_samples]
    all_thr = sorted(set(pos+neg), reverse=True)
    tprs, fprs = [0.0], [0.0]
    for thr in all_thr:
        tp = sum(1 for s in pos if s >= thr); fp = sum(1 for s in neg if s >= thr)
        tprs.append(tp/max(len(pos),1)); fprs.append(fp/max(len(neg),1))
    tprs.append(1.0); fprs.append(1.0)
    auc = sum((fprs[i+1]-fprs[i])*(tprs[i+1]+tprs[i])/2 for i in range(len(fprs)-1))

    def _tpr_at(target):
        for i,f in enumerate(fprs):
            if f>=target: return tprs[i]
        return 1.0

    t1_ms = (float(np.percentile(hit_samples,50)) + float(np.percentile(miss_samples,50))) / 2

    oracle = {
        "hit_p50_ms": round(float(np.percentile(hit_samples,50)),1),
        "hit_p95_ms": round(float(np.percentile(hit_samples,95)),1),
        "hit_p99_ms": round(float(np.percentile(hit_samples,99)),1),
        "miss_p50_ms": round(float(np.percentile(miss_samples,50)),1),
        "miss_p95_ms": round(float(np.percentile(miss_samples,95)),1),
        "miss_p99_ms": round(float(np.percentile(miss_samples,99)),1),
        "roc_auc": round(auc,4),
        "tpr_at_1pct_fpr": round(_tpr_at(0.01),4),
        "tpr_at_5pct_fpr": round(_tpr_at(0.05),4),
    }

    names = [f"{fn} {ln}" for fn in FIRST_NAMES for ln in LAST_NAMES]
    secrets = MEDICAL_CONDITIONS
    victims = []
    for i in range(args.n_victims):
        start = date(1955,1,1)
        dob = (start + timedelta(days=rng.randint(0,20000))).isoformat()
        victims.append({"victim_id": i, "name": rng.choice(names),
                        "condition": rng.choice(secrets), "dob": dob})

    results = []
    for v in victims:
        stage1_calls, found_name = 0, None
        for cand in names:
            ttft = rng.gauss(hit_mean if cand==v["name"] else miss_mean, 15.0*noise_factor)
            stage1_calls += 1
            if ttft < t1_ms:
                found_name = cand; break
        found_secret, stage2_calls = None, 0
        if found_name:
            for sec in secrets:
                ttft = rng.gauss(hit_mean if sec==v["condition"] else miss_mean, 15.0*noise_factor)
                stage2_calls += 1
                if ttft < t1_ms:
                    found_secret = sec; break
        exact = (found_name==v["name"]) and (found_secret==v["condition"])
        results.append({"victim_id": v["victim_id"], "exact_match": bool(exact),
                        "total_api_calls": stage1_calls+stage2_calls,
                        "stage1_api_calls": stage1_calls, "stage2_api_calls": stage2_calls})

    n = len(results); exact = sum(r["exact_match"] for r in results)
    calls = [r["total_api_calls"] for r in results]
    lo, hi = _wilson_ci(exact, n)

    attack_summary = {
        "asr": round(exact/n,4) if n else 0.0, "exact_recovery_count": exact,
        "n_victims": n, "asr_ci_95_lo": round(lo,4), "asr_ci_95_hi": round(hi,4),
        "median_api_calls": float(np.median(calls)) if calls else 0,
        "p95_api_calls": float(np.percentile(calls,95)) if calls else 0,
    }
    print(f"[load_test] (MOCK) {level_name}: ASR={attack_summary['asr']:.3f}  "
          f"AUC={auc:.4f}  median_q={attack_summary['median_api_calls']:.0f}")

    return {"level": level_name, "n_tenants": n_tenants, "oracle": oracle,
            "attack": attack_summary, "loader_stats": {"n_tenants": 0, "total_requests": 0},
            "raw_results": results}


def _generate_figures(payload: dict, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    levels = payload["levels"]

    h1 = ["Load","Tenants","AUC","ASR","Median queries","Feasible"]
    rows1 = []
    for lv in levels:
        rows1.append({
            "Load": lv["level"], "Tenants": lv["n_tenants"],
            "AUC": lv["oracle"].get("roc_auc", "n/a"),
            "ASR": f"{lv['attack']['asr']:.3f}",
            "Median queries": f"{lv['attack']['median_api_calls']:.0f}",
            "Feasible": lv["oracle"].get("intermediate_feasible", "n/a"),
        })
    md1 = "| " + " | ".join(h1) + " |\n| " + " | ".join(["---"]*len(h1)) + " |\n"
    for r in rows1: md1 += "| " + " | ".join(str(r[h]) for h in h1) + " |\n"
    (fig_dir / "issue5_table1_load_oracle_attack.md").write_text(md1)

    (fig_dir / "issue5_figures_data.json").write_text(json.dumps({
        "fig1_asr_vs_load": [{"level": l["level"], "n_tenants": l["n_tenants"],
                              "asr": l["attack"]["asr"]} for l in levels],
    }, indent=2))
    print(f"[load_test] Table/data saved → {fig_dir}")


def main():
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    levels_to_run = list(LOAD_LEVELS.keys()) if args.load_level == "all" else [args.load_level]
    run_id = f"load-test-{datetime.date.today()}-{uuid.uuid4().hex[:8]}"

    all_levels = []
    for level_name in levels_to_run:
        n_tenants = LOAD_LEVELS[level_name]
        if args.backend == "vllm":
            result = _run_real_load_level(level_name, n_tenants, args)
        else:
            result = _run_mock_load_level(level_name, n_tenants, args)
        all_levels.append(result)
        _incremental_save(args, run_id, all_levels)

    out_path = Path(args.output_dir) / f"{run_id}.json"
    with open(out_path) as fh:
        payload = json.load(fh)
    print(f"\n[load_test] Final results → {out_path}")
    _generate_figures(payload, Path(args.output_dir).parent / "figures")


if __name__ == "__main__":
    main()