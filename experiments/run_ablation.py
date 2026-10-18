"""
experiments/run_ablation.py
Issue #10 - Ablation study: contribution of each D5 component

Conditions:
  AB0  - No detector, no behavior detection  (= D0 baseline)
  AB1  - Content risk only (Presidio NER, no confidence gate)
  AB2  - Content risk + confidence threshold (= D4 risk-adaptive)
  AB3  - Behavior detection only (probe_detector, no content risk)
  FULL - Content risk + confidence + behavior (= D5)

Design (state this in the paper):
  * Content gate: static per-victim decision from analyze_prompt().
      HIGH -> ISOLATE (attack blocked), MEDIUM -> RESTRICT (no effect on attack),
      LOW  -> SHARE.
  * Behavior gate: a FRESH ProbeDetector per victim (one attacker session).
    Every attacker probe (each backend call, not each repeated sample) is fed to
    detector.observe(). If the detector raises ALERT, the attack is aborted and
    probes_before_block is recorded.

Usage:
  python experiments/run_ablation.py --backend vllm \
      --base-url http://localhost:8001/v1 --n-victims 20 --seed 42
"""

import argparse, datetime, json, uuid, os, sys, subprocess, platform
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kv_attack.probe_detector import ProbeDetector, SuspicionLevel

ABLATION_CONDITIONS = {
    "AB0_none": {
        "description": "No content detector, no behavior detection (D0 baseline)",
        "use_content_risk": False, "use_confidence": False, "use_behavior": False,
        "default_action": "SHARE",
    },
    "AB1_content_only": {
        "description": "Content risk detector only (Presidio NER, no confidence gate)",
        "use_content_risk": True, "use_confidence": False, "use_behavior": False,
        "confidence_threshold": None,
    },
    "AB2_content_confidence": {
        "description": "Content risk + confidence threshold (= D4 risk-adaptive)",
        "use_content_risk": True, "use_confidence": True, "use_behavior": False,
        "confidence_threshold": 0.7,
    },
    "AB3_behavior_only": {
        "description": "Behavior/probe detection only (no content risk gate)",
        "use_content_risk": False, "use_confidence": False, "use_behavior": True,
        "threshold_med": 0.40, "threshold_high": 0.65,
    },
    "FULL_all_components": {
        "description": "All components: content risk + confidence + behavior (= D5)",
        "use_content_risk": True, "use_confidence": True, "use_behavior": True,
        "confidence_threshold": 0.7, "threshold_med": 0.40, "threshold_high": 0.65,
    },
}


class AttackBlocked(Exception):
    """Raised when the behavior detector fires ALERT mid-attack."""


class DetectorTap:
    """Backend proxy: forwards every call, feeds attacker probes to a
    ProbeDetector, aborts the attack on ALERT."""

    def __init__(self, backend, detector, hit_thresh_ms, victim_prompt,
                 sim_interval_s=None):
        self._b = backend
        self.det = detector
        self.hit_thresh = hit_thresh_ms
        self.victim_prompt = victim_prompt
        self.sim_interval_s = sim_interval_s
        self.n_probes = 0

    def __getattr__(self, name):
        return getattr(self._b, name)

    def _observe(self, prompt, ttft_ms):
        if prompt == self.victim_prompt:
            return
        self.n_probes += 1
        ts = self.n_probes * self.sim_interval_s if self.sim_interval_s else None
        res = self.det.observe(prompt, float(ttft_ms),
                               cache_hit=bool(ttft_ms < self.hit_thresh),
                               timestamp_s=ts)
        if res.level == SuspicionLevel.ALERT:
            raise AttackBlocked(f"ALERT after {self.n_probes} probes")

    @staticmethod
    def _prompt_arg(a, k):
        return a[0] if a else k.get("prompt", "")

    def measure_ttft(self, *a, **k):
        t = self._b.measure_ttft(*a, **k)
        self._observe(self._prompt_arg(a, k), t)
        return t

    def measure_mean_ttft(self, *a, **k):
        t = self._b.measure_mean_ttft(*a, **k)
        self._observe(self._prompt_arg(a, k), t)
        return t

    def is_cache_hit(self, *a, **k):
        out = self._b.is_cache_hit(*a, **k)
        if isinstance(out, tuple) and len(out) == 2:
            self._observe(self._prompt_arg(a, k), out[1])
        return out


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


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return None


def _provenance():
    prov = {
        "command": " ".join(sys.argv),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
    }
    try:
        prov["gpu"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        prov["gpu"] = None
    return prov


def _run_ablation_condition(cond_name, cond_cfg, backend_client, tokenizer,
                            system_prefix, victims, t1_ms, t2_ms, args,
                            payload, out_path, sim_interval_s):
    from kv_attack.risk_signal import analyze_prompt
    from kv_attack.two_stage_reconstructor import (
        evict_cache_two_stage, reconstruct_victim_two_stage,
    )

    use_content  = cond_cfg["use_content_risk"]
    use_confid   = cond_cfg["use_confidence"]
    use_behavior = cond_cfg["use_behavior"]
    conf_thresh  = cond_cfg.get("confidence_threshold") or 0.7
    thr_med      = cond_cfg.get("threshold_med", 0.40)
    thr_high     = cond_cfg.get("threshold_high", 0.65)

    results = []
    cache_decisions = {"SHARE": 0, "RESTRICT": 0, "ISOLATE": 0}

    for i, vr in enumerate(victims):
        prompt_text = vr.get("system_prompt", vr.get("prompt", ""))

        content_info = {}
        if use_content:
            sig = analyze_prompt(prompt_text)
            risk_level = sig.risk_level
            content_info = {
                "raw_risk_level": sig.risk_level,
                "detector_confidence": round(sig.detector_confidence, 4),
                "entity_types": sig.entity_types,
            }
            if use_confid and sig.detector_confidence < conf_thresh:
                risk_level = "LOW"
        else:
            risk_level = "LOW"

        if risk_level == "HIGH":
            action = "ISOLATE"
        elif risk_level == "MEDIUM":
            action = "RESTRICT"
        else:
            action = cond_cfg.get("default_action", "SHARE")

        if action == "ISOLATE":
            results.append({
                "victim_id": vr["victim_id"], "exact_match": False,
                "total_api_calls": 0, "blocked_by": "CONTENT",
                "risk_level": risk_level,
            })
        else:
            evict_cache_two_stage(backend_client, system_prefix, tokenizer)
            be, tap = backend_client, None
            if use_behavior:
                det = ProbeDetector(client_id=f"attacker-{i}",
                                    threshold_med=thr_med,
                                    threshold_high=thr_high)
                tap = DetectorTap(backend_client, det, t1_ms, vr["prompt"],
                                  sim_interval_s=sim_interval_s)
                be = tap
            try:
                r = reconstruct_victim_two_stage(
                    backend=be, tokenizer=tokenizer, system_prefix=system_prefix,
                    t1_ms=t1_ms, t2_ms=t2_ms, victim_record=vr,
                    candidate_seed=args.seed * 1000 + i,
                )
                results.append({
                    "victim_id": vr["victim_id"],
                    "exact_match": bool(r.exact_match),
                    "total_api_calls": r.total_api_calls,
                    "blocked_by": None, "risk_level": risk_level,
                    "probes_seen": tap.n_probes if tap else None,
                    "max_suspicion": (max((x.suspicion_score for x in tap.det.results),
                                          default=0.0) if tap else None),
                })
            except AttackBlocked:
                action = "ISOLATE"
                results.append({
                    "victim_id": vr["victim_id"], "exact_match": False,
                    "total_api_calls": tap.n_probes, "blocked_by": "BEHAVIOR",
                    "probes_before_block": tap.n_probes, "risk_level": risk_level,
                    "last_detector_state": tap.det.results[-1].to_dict(),
                })

        results[-1].update(content_info)
        cache_decisions[action] += 1

        n = len(results); exact = sum(r["exact_match"] for r in results)
        lo, hi = _wilson_ci(exact, n)
        payload[f"partial_{cond_name}"] = {
            "n_done": n, "n_total": len(victims), "exact": exact,
            "asr_so_far": round(exact/n, 4),
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
    blocked_content  = sum(r["blocked_by"] == "CONTENT"  for r in results)
    blocked_behavior = sum(r["blocked_by"] == "BEHAVIOR" for r in results)
    pbb = [r["probes_before_block"] for r in results if "probes_before_block" in r]
    return {
        "condition": cond_name, "description": cond_cfg["description"],
        "n_victims": n, "exact": exact, "asr": round(exact/n, 4),
        "asr_ci_lo": round(lo, 4), "asr_ci_hi": round(hi, 4),
        "isolation_fraction": round(cache_decisions["ISOLATE"]/n, 4) if n else 0,
        "blocked_by_content": blocked_content,
        "blocked_by_behavior": blocked_behavior,
        "median_probes_before_block": float(np.median(pbb)) if pbb else None,
        "cache_decisions": cache_decisions,
        "median_api_calls": float(np.median(calls)) if calls else 0,
        "p95_api_calls": float(np.percentile(calls, 95)) if calls else 0,
        "raw_results": results,
    }


def main():
    p = argparse.ArgumentParser(description="Issue #10: Ablation study")
    p.add_argument("--backend", default="vllm", choices=["vllm", "sglang", "mock"])
    p.add_argument("--base-url", default="http://localhost:8001/v1")
    p.add_argument("--model-id", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    p.add_argument("--n-victims", type=int, default=20)
    p.add_argument("--n-calib", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="experiments/results")
    p.add_argument("--sim-interval-s", type=float, default=None,
                   help="Simulated seconds between probes for the behavior "
                        "detector's frequency feature. Defaults to 2.0 on the "
                        "mock backend (instant responses), off on real backends.")
    args = p.parse_args()

    sim_interval_s = args.sim_interval_s
    if sim_interval_s is None and args.backend == "mock":
        sim_interval_s = 2.0

    os.makedirs(args.output_dir, exist_ok=True)
    run_id   = f"ablation-{datetime.date.today()}-{uuid.uuid4().hex[:8]}"
    out_path = Path(args.output_dir) / f"{run_id}.json"

    print(f"\n{'='*60}")
    print(f"[ablation] Issue #10 Ablation Study")
    print(f"  Backend: {args.backend}  Victims: {args.n_victims}  Seed: {args.seed}")
    print(f"  Conditions: {list(ABLATION_CONDITIONS.keys())}")
    print(f"  Simulated clock: {sim_interval_s}")
    print(f"  Output: {out_path}")
    print(f"{'='*60}\n")

    payload = {
        "run_id": run_id, "issue": 10, "study": "ablation",
        "backend": args.backend, "model_id": args.model_id,
        "base_url": args.base_url, "n_victims": args.n_victims,
        "n_calib": args.n_calib, "seed": args.seed,
        "simulated_clock_interval_s": sim_interval_s,
        "provenance": _provenance(),
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
        tokenizer, has_bos=detect_has_bos(args.model_id))

    if args.backend == "vllm":
        from kv_attack.backends.vllm_backend import VLLMBackend
        backend_client = VLLMBackend(base_url=args.base_url, model_id=args.model_id)
    elif args.backend == "sglang":
        from kv_attack.backends.sglang_backend import make_sglang_backend
        backend_client = make_sglang_backend(base_url=args.base_url,
                                             model_id=args.model_id)
    else:
        from kv_attack.backends.two_stage_mock_backend import TwoStageMockBackend
        backend_client = TwoStageMockBackend(seed=args.seed)

    print(f"[ablation] Seeding {args.n_victims} victims (seed={args.seed})...")
    victims = seed_victims_two_stage(
        client=client, tokenizer=tokenizer, system_prefix=system_prefix,
        n_victims=args.n_victims, seed=args.seed)
    print(f"[ablation] victim record keys: {sorted(victims[0].keys())}")
    if "system_prompt" not in victims[0] and "prompt" not in victims[0]:
        raise RuntimeError("Victim records have neither 'system_prompt' nor "
                           "'prompt'; content detector would see empty text.")
    if "prompt" not in victims[0]:
        raise RuntimeError("Victim records lack 'prompt'; DetectorTap needs it.")

    print(f"[ablation] Calibrating thresholds ({args.n_calib} probes)...")
    calib = calibrate_two_stage(
        backend=backend_client, tokenizer=tokenizer, system_prefix=system_prefix,
        victim_record=victims[0], n_samples=args.n_calib)
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
            cond_name, cond_cfg, backend_client, tokenizer, system_prefix,
            victims, t1_ms, t2_ms, args, payload, out_path, sim_interval_s)
        all_results.append(result)
        payload["results"] = all_results
        payload["status"]  = f"{cond_name}_done"
        _incremental_save(out_path, payload)
        print(f"  -> ASR={result['asr']:.3f} "
              f"[{result['asr_ci_lo']:.3f},{result['asr_ci_hi']:.3f}]  "
              f"isolation={result['isolation_fraction']:.3f}  "
              f"content_blocks={result['blocked_by_content']}  "
              f"behavior_blocks={result['blocked_by_behavior']}")

    payload["status"] = "complete"
    payload["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    _incremental_save(out_path, payload)

    print(f"\n{'='*60}\n[ablation] COMPLETE - {out_path}\n{'='*60}")
    print(f"{'Condition':<26} {'ASR':>6} {'95% CI':>16} {'Isol':>6} "
          f"{'CBlk':>5} {'BBlk':>5} {'MedQ':>7}")
    print("-" * 78)
    for r in all_results:
        print(f"{r['condition']:<26} {r['asr']:>6.3f} "
              f"({r['asr_ci_lo']:.3f},{r['asr_ci_hi']:.3f}) "
              f"{r['isolation_fraction']:>6.3f} {r['blocked_by_content']:>5} "
              f"{r['blocked_by_behavior']:>5} {r['median_api_calls']:>7.1f}")


if __name__ == "__main__":
    main()