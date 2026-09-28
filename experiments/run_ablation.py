"""
experiments/run_ablation.py
Issue #10 — Ablation study: contribution of each D5 component

Tests 5 ablation conditions:
  AB0  — No detector, no behavior detection  (= D0 baseline)
  AB1  — Content risk only (Presidio NER, no confidence gate)
  AB2  — Content risk + confidence threshold (= D4 risk-adaptive)
  AB3  — Behavior detection only (probe_detector, no content risk)
  FULL — All components: content risk + confidence + behavior (= D5)

Usage:
  python experiments/run_ablation.py \\
      --backend vllm \\
      --base-url http://localhost:8001/v1 \\
      --n-victims 20 \\
      --seed 42 \\
      --output-dir experiments/results
"""

import argparse, datetime, json, uuid, os, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

ABLATION_CONDITIONS = {
    "AB0_none": {
        "description": "No content detector, no behavior detection (D0 baseline)",
        "use_content_risk":   False,
        "use_confidence":     False,
        "use_behavior":       False,
        "default_action":     "SHARE",
    },
    "AB1_content_only": {
        "description": "Content risk detector only (Presidio NER, no confidence gate)",
        "use_content_risk":   True,
        "use_confidence":     False,
        "use_behavior":       False,
        "confidence_threshold": None,
    },
    "AB2_content_confidence": {
        "description": "Content risk + confidence threshold (= D4 risk-adaptive)",
        "use_content_risk":   True,
        "use_confidence":     True,
        "use_behavior":       False,
        "confidence_threshold": 0.7,
    },
    "AB3_behavior_only": {
        "description": "Behavior/probe detection only (no content risk gate)",
        "use_content_risk":   False,
        "use_confidence":     False,
        "use_behavior":       True,
        "threshold_med":      0.40,
        "threshold_high":     0.65,
    },
    "FULL_all_components": {
        "description": "All components: content risk + confidence + behavior (= D5)",
        "use_content_risk":   True,
        "use_confidence":     True,
        "use_behavior":       True,
        "confidence_threshold": 0.7,
        "threshold_med":      0.40,
        "threshold_high":     0.65,
    },
}


def _wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denom
    margin = z * (p*(1-p)/n + z**2/(4*n**2))**0.5 / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _incremental_save(path, payload):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    tmp.replace(path)


def _run_ablation_condition(cond_name, cond_cfg, backend_client,
                             tokenizer, system_prefix, victims,
                             t1_ms, t2_ms, args, payload, out_path):
    """Run one ablation condition and return per-victim results."""
    from kv_attack.risk_signal import analyze_prompt
    from kv_attack.probe_detector import ProbeDetector, SuspicionLevel
    from kv_attack.two_stage_reconstructor import (
        evict_cache_two_stage, reconstruct_victim_two_stage,
    )

    use_content  = cond_cfg["use_content_risk"]
    use_confid   = cond_cfg["use_confidence"]
    use_behavior = cond_cfg["use_behavior"]
    conf_thresh  = cond_cfg.get("confidence_threshold", 0.7)
    thr_med      = cond_cfg.get("threshold_med", 0.40)
    thr_high     = cond_cfg.get("threshold_high", 0.65)

    detector = ProbeDetector(
        threshold_med=thr_med, threshold_high=thr_high
    ) if use_behavior else None

    results = []
    cache_decisions = {"SHARE": 0, "RESTRICT": 0, "ISOLATE": 0}

    for i, vr in enumerate(victims):
        prompt_text = vr.get("system_prompt", vr.get("prompt", ""))

        if use_content:
            sig = analyze_prompt(prompt_text)
            risk_level = sig.get("risk_level", "LOW")
            confidence = sig.get("confidence", 1.0)
            if use_confid and confidence < conf_thresh:
                risk_level = "LOW"
        else:
            risk_level = "LOW"

        susp_level = SuspicionLevel.LOW
        if use_behavior and detector:
            susp_level = detector.current_suspicion_level()

        if risk_level == "HIGH" or susp_level == SuspicionLevel.HIGH:
            action = "ISOLATE"
        elif risk_level == "MEDIUM" or susp_level == SuspicionLevel.MED:
            action = "RESTRICT"
        else:
            action = cond_cfg.get("default_action", "SHARE")

        cache_decisions[action] += 1

        if action == "ISOLATE":
            results.append({
                "victim_id": vr["victim_id"], "exact_match": False,
                "total_api_calls": 0, "blocked_by": action,
            })
        else:
            evict_cache_two_stage(backend_client, system_prefix, tokenizer)
            r = reconstruct_victim_two_stage(
                backend=backend_client, tokenizer=tokenizer,
                system_prefix=system_prefix,
                t1_ms=t1_ms, t2_ms=t2_ms,
                victim_record=vr,
                candidate_seed=args.seed * 1000 + i,
            )
            if use_behavior and detector:
                detector.record_probe(prompt_text, i)
            results.append({
                "victim_id": vr["victim_id"],
                "exact_match": bool(r.exact_match),
                "total_api_calls": r.total_api_calls,
                "blocked_by": None,
            })

        n = len(results); exact = sum(r["exact_match"] for r in results)
        lo, hi = _wilson_ci(exact, n)
        payload[f"partial_{cond_name}"] = {
            "n_done": n, "n_total": len(victims),
            "exact": exact, "asr_so_far": round(exact/n, 4),
            "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
            "cache_decisions": dict(cache_decisions),
            "raw_results": list(results),
        }
        _incremental_save(out_path, payload)

        if (i+1) % 5 == 0 or (i+1) == len(victims):
            print(f"    [{cond_name}] {i+1}/{len(victims)}: "
                  f"ASR={exact/n:.3f}  action={action}")

    n = len(results); exact = sum(r["exact_match"] for r in results)
    calls = [r["total_api_calls"] for r in results if r["total_api_calls"] > 0]
    lo, hi = _wilson_ci(exact, n)
    isolated = cache_decisions.get("ISOLATE", 0)
    return {
        "condition": cond_name,
        "description": cond_cfg["description"],
        "n_victims": n, "exact": exact,
        "asr": round(exact/n, 4),
        "asr_ci_lo": round(lo, 4), "asr_ci_hi": round(hi, 4),
        "isolation_fraction": round(isolated/n, 4) if n else 0,
        "cache_decisions": cache_decisions,
        "median_api_calls": float(np.median(calls)) if calls else 0,
        "p95_api_calls": float(np.percentile(calls, 95)) if calls else 0,
        "raw_results": results,
    }


def main():
    p = argparse.ArgumentParser(description="Issue #10: Ablation study")
    p.add_argument("--backend",   default="vllm",
                   choices=["vllm","sglang","mock"])
    p.add_argument("--base-url",  default="http://localhost:8001/v1")
    p.add_argument("--model-id",  default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    p.add_argument("--n-victims", type=int, default=20)
    p.add_argument("--n-calib",   type=int, default=50)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--output-dir",default="experiments/results")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    run_id   = f"ablation-{datetime.date.today()}-{uuid.uuid4().hex[:8]}"
    out_path = Path(args.output_dir) / f"{run_id}.json"

    print(f"\n{'='*60}")
    print(f"[ablation] Issue #10 Ablation Study")
    print(f"  Backend: {args.backend}  Victims: {args.n_victims}  Seed: {args.seed}")
    print(f"  Conditions: {list(ABLATION_CONDITIONS.keys())}")
    print(f"  Output: {out_path}")
    print(f"{'='*60}\n")

    payload = {
        "run_id": run_id, "issue": 10, "study": "ablation",
        "backend": args.backend, "model_id": args.model_id,
        "n_victims": args.n_victims, "seed": args.seed,
        "conditions_config": ABLATION_CONDITIONS,
        "results": [], "status": "running",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    _incremental_save(out_path, payload)

    from openai import OpenAI
    from transformers import AutoTokenizer
    from kv_attack import detect_has_bos
    from kv_attack.victim_seeder import build_aligned_system_prompt
    from kv_attack.two_stage_victim_seeder import seed_victims_two_stage
    from kv_attack.two_stage_reconstructor import calibrate_two_stage

    client    = OpenAI(base_url=args.base_url, api_key="EMPTY")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    system_prefix, _ = build_aligned_system_prompt(
        tokenizer, has_bos=detect_has_bos(args.model_id)
    )

    if args.backend == "vllm":
        from kv_attack.backends.vllm_backend import VLLMBackend
        backend_client = VLLMBackend(base_url=args.base_url,
                                      model_id=args.model_id)
    elif args.backend == "sglang":
        from kv_attack.backends.sglang_backend import make_sglang_backend
        backend_client = make_sglang_backend(base_url=args.base_url,
                                              model_id=args.model_id)
    else:
        from kv_attack.backends.two_stage_mock_backend import TwoStageMockBackend
        backend_client = TwoStageMockBackend(seed=args.seed)

    print(f"[ablation] Seeding {args.n_victims} victims (seed={args.seed})...")
    victims = seed_victims_two_stage(
        client=client, tokenizer=tokenizer,
        system_prefix=system_prefix,
        n_victims=args.n_victims, seed=args.seed,
    )

    print(f"[ablation] Calibrating thresholds ({args.n_calib} probes)...")
    calib = calibrate_two_stage(
        backend=backend_client, tokenizer=tokenizer,
        system_prefix=system_prefix,
        victim_record=victims[0], n_samples=args.n_calib,
    )
    t1_ms = calib["t1_threshold_ms"]
    t2_ms = calib["t2_threshold_ms"]
    print(f"[ablation] T1={t1_ms:.1f}ms  T2={t2_ms:.1f}ms")

    payload["calibration"] = {
        "t1_ms": t1_ms, "t2_ms": t2_ms,
        "hit_mean_ms": calib.get("hit_mean_ms"),
        "miss_mean_ms": calib.get("miss_mean_ms"),
    }
    _incremental_save(out_path, payload)

    all_results = []
    for cond_name, cond_cfg in ABLATION_CONDITIONS.items():
        print(f"\n[ablation] Running {cond_name}: {cond_cfg['description']}")
        result = _run_ablation_condition(
            cond_name, cond_cfg, backend_client,
            tokenizer, system_prefix, victims,
            t1_ms, t2_ms, args, payload, out_path,
        )
        all_results.append(result)
        payload["results"] = all_results
        payload["status"]  = f"{cond_name}_done"
        _incremental_save(out_path, payload)
        print(f"  → ASR={result['asr']:.3f} [{result['asr_ci_lo']:.3f},{result['asr_ci_hi']:.3f}]"
              f"  isolation={result['isolation_fraction']:.3f}")

    payload["status"] = "complete"
    payload["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    _incremental_save(out_path, payload)

    print(f"\n{'='*60}")
    print(f"[ablation] COMPLETE — {out_path}")
    print(f"{'='*60}")
    print(f"{'Condition':<30} {'ASR':>6} {'95%CI':>18} {'Isolation':>10} {'Median Q':>10}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['condition']:<30} {r['asr']:>6.3f} "
              f"({r['asr_ci_lo']:.3f},{r['asr_ci_hi']:.3f})  "
              f"{r['isolation_fraction']:>10.3f}  {r['median_api_calls']:>10.1f}")


if __name__ == "__main__":
    main()
