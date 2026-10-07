from __future__ import annotations

import argparse
import contextlib
import copy
import datetime
import hashlib
import io
import json
import os
import platform
import random
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from kv_attack import (FIRST_NAMES, LAST_NAMES, MEDICAL_CONDITIONS,
                       RESEED_EVERY, detect_has_bos)
from kv_attack.defended_serving import (DEFENSES, RISK_FNS, DefendedServer,
                                        VictimReseedPrompt)

LEVELS_DEFAULT = {"L0": 0, "L1": 2, "L2": 8, "L3": 16}
H_NAME, H_COND = float(np.log2(100)), float(np.log2(20))


def shash(s: str) -> int:
    return int(hashlib.blake2b(s.encode(), digest_size=4).hexdigest(), 16)


def _git(*a):
    try:
        return subprocess.check_output(["git", *a], cwd=ROOT, stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return None


def hardware_info() -> dict:
    gpu = None
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        pass
    return {"python": platform.python_version(), "platform": platform.platform(),
            "hostname": platform.node(), "cpu_count": os.cpu_count(), "gpu": gpu}


def _quiet():
    return contextlib.redirect_stdout(io.StringIO())


class JsonlWriter:
    def __init__(self, path: Path):
        self.path, self._lock = path, threading.Lock()

    def write(self, obj: dict) -> None:
        line = json.dumps(obj, default=str)
        with self._lock, open(self.path, "a") as f:
            f.write(line + "\n")
            f.flush()

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for ln in self.path.read_text().splitlines():
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
        return out


class Ctx:
    """Everything a trial needs; built once per run."""

    def __init__(self, args, cfg):
        self.kind = "simulated" if args.backend == "sim" else "real"
        self.backend_name = args.backend
        self.cfg, self.args = cfg, args
        self.model_id = args.model_id or cfg["model_id"]
        self.base_url = args.base_url
        self.shared_risk_cache: dict = {}
        self.reset_mode, self.epoch = "evict", None
        if args.backend == "sim":
            from kv_attack.sim_backend import (SimPrefixCacheBackend, SimTokenizer,
                                               install_fast_block_builders)
            install_fast_block_builders()
            self.tok = SimTokenizer()
            self.raw = SimPrefixCacheBackend(self.tok, seed=0)
            self.has_bos = True
            risk = "regex"
        else:
            from transformers import AutoTokenizer
            from kv_attack.sim_backend import install_memoized_block_builders
            install_memoized_block_builders()
            self.tok = AutoTokenizer.from_pretrained(self.model_id)
            self.has_bos = detect_has_bos(self.model_id)
            if args.backend == "vllm":
                from kv_attack.backends.vllm_backend import VLLMBackend
                self.raw = VLLMBackend(base_url=args.base_url, model_id=self.model_id)
            else:
                from kv_attack.backends.sglang_backend import make_sglang_backend
                self.raw = make_sglang_backend(base_url=args.base_url, model_id=self.model_id)
            risk = "presidio"
        if args.risk_model != "auto":
            risk = args.risk_model
        self.risk_name, self.risk_fn = risk, RISK_FNS[risk]
        from kv_attack.victim_seeder import build_aligned_system_prompt
        self.sys, _ = build_aligned_system_prompt(self.tok, has_bos=self.has_bos)

    def spec(self, name):
        import dataclasses
        s = DEFENSES[name]
        bd = self.cfg["behavior_detector"]
        return dataclasses.replace(
            s, conf_threshold=self.cfg["confidence_gate"], thr_med=bd["threshold_med"],
            thr_high=bd["threshold_high"], cooldown_probes=bd["cooldown_probes"])

    def aligned_prefix(self, epoch: str) -> str:
        """System prefix carrying a per-trial nonce, padded so private content still
        starts on a block boundary (same rule as victim_seeder.build_aligned_system_prompt)."""
        from kv_attack import victim_seeder as vs
        from kv_attack import BLOCK_SIZE
        padded = f"Session {epoch}. " + vs._SYSTEM_PREFIX_RAW
        bos = 1 if self.has_bos else 0
        for _ in range(BLOCK_SIZE * 4):
            if (bos + len(self.tok.encode(padded, add_special_tokens=False))) % BLOCK_SIZE == 0:
                return padded
            padded += vs._PAD_WORD
        raise RuntimeError("cannot align epoch prefix")

    def contamination_check(self, calib) -> tuple[bool, dict]:
        """Do blocks cached under epoch A ever hit under epoch B?  (must not)"""
        from kv_attack.two_stage_victim_seeder import build_two_stage_prompt
        a, b = self.aligned_prefix("chkA"), self.aligned_prefix("chkB")
        pa = build_two_stage_prompt(a, "Mary Smith", "1990-01-01", "asthma", self.tok)
        pb = build_two_stage_prompt(b, "Mary Smith", "1990-01-01", "asthma", self.tok)
        pm = build_two_stage_prompt(a, f"Zq{uuid.uuid4().hex[:10]}", "1990-01-01", "asthma", self.tok)
        self.raw.measure_ttft(pa)
        hit, cross, miss = (self.raw.measure_ttft(pa), self.raw.measure_ttft(pb),
                            self.raw.measure_ttft(pm))
        ok = cross > hit + 0.8 * (miss - hit)
        return ok, {"hit_ms": hit, "cross_epoch_ms": cross, "miss_ms": miss}

    def begin(self, key: str) -> "Ctx":
        """Return a per-trial view with a fresh cache namespace (or flush the cache)."""
        t = copy.copy(self)
        if self.reset_mode == "epoch":
            t.epoch = f"{shash(key):08x}"
            t.sys = self.aligned_prefix(t.epoch)
        else:
            t.epoch = None
            self.reset_cache()
        return t

    def canon(self, prompt: str) -> str:
        """Text the CONTENT detector sees: harness nonce removed."""
        return prompt.replace(f"Session {self.epoch}. ", "", 1) if self.epoch else prompt

    def reset_cache(self):
        """Out-of-band between-trial cache reset (not attributed to any tenant)."""
        if self.kind == "simulated":
            self.raw.flush()
        else:
            from kv_attack.two_stage_reconstructor import evict_cache_two_stage
            with _quiet():
                evict_cache_two_stage(self.raw, self.sys, self.tok)

    def set_load(self, level_name: str):
        if self.kind == "simulated":
            self.raw.set_load(LEVELS_DEFAULT_IDX.get(level_name, 0))

    def info(self) -> dict:
        b = self.raw.get_info()
        return {"backend_name": b.backend_name, "framework": b.framework,
                "framework_ver": b.framework_ver, "model_id": self.model_id,
                "base_url": self.base_url, "apc_enabled": b.apc_enabled, "extra": b.extra}


LEVELS_DEFAULT_IDX = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}


def make_victims(ctx: Ctx, seed: int, n: int) -> list[dict]:
    from kv_attack.two_stage_victim_seeder import (_random_dob, build_two_stage_prompt,
                                                   count_two_stage_blocks)
    rng = random.Random(seed)
    out = []
    for i in range(n):
        name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        dob, cond = _random_dob(rng), rng.choice(MEDICAL_CONDITIONS)
        prompt = build_two_stage_prompt(ctx.sys, name, dob, cond, ctx.tok)
        cnt = count_two_stage_blocks(ctx.sys, name, dob, cond, ctx.tok, has_bos=ctx.has_bos)
        out.append({"victim_id": i, "prompt": VictimReseedPrompt(prompt),
                    "ground_truth": {"name": name, "dob": dob, "condition": cond},
                    "n_name_blocks": cnt["name_blocks"], "n_cond_blocks": cnt["cond_blocks"],
                    "n_total_blocks": cnt["total_private_blocks"]})
    return out


def rebind_victim(ctx: Ctx, v: dict) -> dict:
    """Same victim, prompt rebuilt under the trial's system prefix."""
    if ctx.epoch is None:
        return v
    from kv_attack.two_stage_victim_seeder import build_two_stage_prompt
    g = v["ground_truth"]
    out = dict(v)
    out["prompt"] = VictimReseedPrompt(
        build_two_stage_prompt(ctx.sys, g["name"], g["dob"], g["condition"], ctx.tok))
    return out


def new_server(ctx: Ctx, defense, fnr, key, calib):
    srv = DefendedServer(ctx.raw, ctx.spec(defense), risk_fn=lambda p: ctx.risk_fn(p),
                         fnr=fnr, seed=shash(key) & 0xFFFF,
                         hit_threshold_ms=calib["t1_threshold_ms"], canon_fn=ctx.canon)
    srv._risk_cache = ctx.shared_risk_cache
    return srv


def calibrate(ctx: Ctx, seed: int, victims: list[dict]) -> dict:
    from kv_attack.two_stage_reconstructor import calibrate_two_stage
    ctx.reset_cache()
    if ctx.kind == "simulated":
        ctx.raw.reseed(shash(f"calib{seed}"))
    with _quiet():
        c = calibrate_two_stage(ctx.raw, ctx.tok, ctx.sys, victims[0],
                                n_samples=ctx.cfg["n_calibration"])
    return c


def preflight(calib: dict, info: dict, force: bool) -> list[str]:
    """Oracle-validity gate.  The 2026-09-29 defense-eval run (APC off, gap 1.2 ms,
    D0 ASR 0/30) would have been rejected here."""
    problems = []
    gap = calib["miss_mean_ms"] - calib["hit_mean_ms"]
    if not info.get("apc_enabled", True):
        problems.append("backend reports apc_enabled=False")
    if gap < 100:
        problems.append(f"hit/miss gap {gap:.1f} ms < 100 ms (no usable timing oracle)")
    _feasible = calib.get("intermediate_feasible", False)
    if isinstance(_feasible, str):
        _feasible = _feasible.strip().lower() == "true"
    if not _feasible:
        problems.append("intermediate S1-hit level not separable (two-stage infeasible)")
    return [] if force else problems


def _linear(view, ctx, victim, calib, cand_seed, early_exit):
    from kv_attack.adaptive_reconstructor import _build_candidate_list
    from kv_attack.two_stage_victim_seeder import build_two_stage_prompt
    gt = victim["ground_truth"]
    best_t, best = float("inf"), None
    for i, (n, c) in enumerate(_build_candidate_list(seed=cand_seed)):
        if i > 0 and i % RESEED_EVERY == 0:
            view.measure_ttft(victim["prompt"])
        t = view.measure_ttft(build_two_stage_prompt(ctx.sys, n, gt["dob"], c, ctx.tok))
        if t < best_t:
            best_t, best = t, (n, c)
        if early_exit and t < calib["t2_threshold_ms"]:
            break
    return {"name": best[0], "condition": best[1]}


def run_attack(name, view, ctx, victim, calib, cand_seed):
    if name == "A2":
        from kv_attack.two_stage_reconstructor import reconstruct_victim_two_stage
        with _quiet():
            r = reconstruct_victim_two_stage(
                backend=view, tokenizer=ctx.tok, system_prefix=ctx.sys,
                t1_ms=calib["t1_threshold_ms"], t2_ms=calib["t2_threshold_ms"],
                victim_record=victim, candidate_seed=cand_seed)
        return r.recovered, r.total_api_calls
    rec = _linear(view, ctx, victim, calib, cand_seed, early_exit=(name == "A1"))
    return rec, None


def canon(cfg, defense: str) -> str:
    return cfg["aliases"].get(defense, defense)


def build_cells(cfg, profile) -> list[dict]:
    cells: dict[tuple, dict] = {}

    def add(attack, defense, fnr=0.0, load="L0", group="core"):
        d = canon(cfg, defense)
        if not DEFENSES[d].use_content:
            fnr = 0.0
        k = (attack, d, round(float(fnr), 4), load)
        c = cells.setdefault(k, {"attack": attack, "defense": d, "fnr": k[2], "load": load,
                                 "groups": []})
        if group not in c["groups"]:
            c["groups"].append(group)

    for blk in cfg["core"]:
        for d in blk["defenses"]:
            add(blk["attack"], d, group="core")
    fs = cfg["fn_sweep"]
    for d in fs["defenses"]:
        for f in [0.0] + list(fs["fnr"]):
            add(fs["attack"], d, fnr=f, group="fn_sweep")
    ab = cfg["ablation"]
    for f in ab["fnr"]:
        for d in ab["conditions"]:
            add(ab["attack"], d, fnr=f, group="ablation")
    ld = cfg["load"]
    for lvl in ld["levels"]:
        for d in ld["defenses"]:
            add(ld["attack"], d, load=lvl, group="load")
    return list(cells.values())


def trial_key(cell, seed, vidx):
    return (f"{cell['attack']}|{cell['defense']}|fnr={cell['fnr']}|"
            f"{cell['load']}|seed={seed}|v={vidx}")


def run_security_trial(ctx: Ctx, cell, seed, vidx, victim, calib) -> dict:
    key = trial_key(cell, seed, vidx)
    rec = {"trial_key": key, **{k: cell[k] for k in ("attack", "defense", "fnr", "load")},
           "groups": cell["groups"], "seed": seed, "victim_idx": vidx,
           "ground_truth": victim["ground_truth"], "error": None}
    loader = None
    try:
        ctx = ctx.begin(key)
        victim = rebind_victim(ctx, victim)
        if ctx.kind == "simulated":
            ctx.raw.reseed(shash(key))
            ctx.set_load(cell["load"])
        elif cell["load"] != "L0":
            from kv_attack.load_generator import LoadGenerator
            loader = LoadGenerator(base_url=ctx.base_url, model_id=ctx.model_id,
                                   n_tenants=ctx.cfg["load"]["levels"][cell["load"]],
                                   seed=shash(key) & 0xFFFF)
            loader.start()
        srv = new_server(ctx, cell["defense"], cell["fnr"], key, calib)
        srv.client("tenant-victim", "victim").measure_ttft(str(victim["prompt"]))
        view = srv.client("tenant-attacker", "attacker")
        t0 = time.perf_counter()
        recovered, reported_calls = run_attack(
            cell["attack"], view, ctx, victim, calib, seed * 1000 + victim["victim_id"])
        wall = time.perf_counter() - t0
        audit = srv.drain_audit()
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        return rec
    finally:
        if loader is not None:
            rec["loader_stats"] = loader.stop()
    gt = victim["ground_truth"]
    att = [a for a in audit if a["role"] == "attacker"]
    vic = [a for a in audit if a["role"] == "victim"]
    acts = {}
    for a in att:
        acts[a["action"]] = acts.get(a["action"], 0) + 1
    first_ns = next((i for i, a in enumerate(att) if a["action"] != "SHARE"), None)
    name_ok, cond_ok = recovered["name"] == gt["name"], recovered["condition"] == gt["condition"]
    rec.update({
        "recovered": recovered, "name_ok": name_ok, "cond_ok": cond_ok,
        "exact_match": bool(name_ok and cond_ok),
        "leaked_bits": round(H_NAME * name_ok + H_COND * cond_ok, 4),
        "n_attacker_requests": len(att),
        "total_api_calls": reported_calls if reported_calls is not None else len(att) + len(vic) - 1,
        "victim_action": vic[0]["action"] if vic else None,
        "victim_risk": vic[0]["risk"] if vic else None,
        "attacker_actions": acts, "first_nonshare_attacker_req": first_ns,
        "n_alerts": sum(a["sus_level"] == "ALERT" for a in att),
        "max_suspicion": max([a["sus"] for a in att], default=0.0),
        "decide_ms_total": round(sum(a["decide_ms"] for a in audit), 3),
        "attack_time_model_s": round(sum(a["ttft_ms"] for a in att) / 1e3, 3),
        "wall_s": round(wall, 3), "risk_model": ctx.risk_name})
    return rec


def run_oracle(ctx: Ctx, defense, seed, calib, writer: JsonlWriter, per_class: int):
    """Measure what an attacker's TTFT oracle looks like UNDER each defense.
    classes: full_hit (exact replay of victim prompt), s1_hit (right name, unique wrong
    condition), miss (unique wrong name).  One class per victim => no self-contamination."""
    from kv_attack.two_stage_victim_seeder import _random_dob, build_two_stage_prompt
    key0 = f"oracle|{defense}|{seed}"
    ctx = ctx.begin(key0)
    if ctx.kind == "simulated":
        ctx.raw.reseed(shash(key0))
    srv = new_server(ctx, defense, 0.0, key0, calib)
    rng = random.Random(shash(key0))
    names = [f"{a} {b}" for a in FIRST_NAMES for b in LAST_NAMES]
    rng.shuffle(names)
    vc, ac = srv.client("tenant-victim", "victim"), srv.client("tenant-attacker", "attacker")
    classes = ["full_hit", "s1_hit", "miss"]
    for i in range(per_class * 3):
        cls, name = classes[i % 3], names[i]
        cond, dob = rng.choice(MEDICAL_CONDITIONS), _random_dob(rng)
        vc.measure_ttft(build_two_stage_prompt(ctx.sys, name, dob, cond, ctx.tok))
        if cls == "full_hit":
            probe = build_two_stage_prompt(ctx.sys, name, dob, cond, ctx.tok)
        elif cls == "s1_hit":
            probe = build_two_stage_prompt(ctx.sys, name, dob, f"zz{uuid.uuid4().hex[:10]}", ctx.tok)
        else:
            probe = build_two_stage_prompt(ctx.sys, f"Zq{uuid.uuid4().hex[:10]}", dob, cond, ctx.tok)
        ttft, r = srv.handle_full("tenant-attacker", "attacker", probe)
        writer.write({"defense": defense, "seed": seed, "class": cls, "ttft_ms": ttft,
                      "action": r["action"], "risk": r["risk"]})
    srv.drain_audit()


def scrape_prefix_metrics(base_url):
    """Best-effort vLLM /metrics prefix-cache counters (raw lines kept)."""
    if not base_url:
        return None
    try:
        import urllib.request
        url = base_url.rsplit("/v1", 1)[0] + "/metrics"
        txt = urllib.request.urlopen(url, timeout=5).read().decode()
        return [ln for ln in txt.splitlines() if "prefix_cache" in ln and not ln.startswith("#")]
    except Exception:
        return None


class GpuSampler:
    def __init__(self):
        self.samples, self._stop, self._t = [], threading.Event(), None

    def start(self):
        def loop():
            while not self._stop.is_set():
                try:
                    o = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                         "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL,
                        text=True).strip().split("\n")[0].split(",")
                    self.samples.append((time.time(), float(o[0]), float(o[1])))
                except Exception:
                    return
                self._stop.wait(1.0)
        self._t = threading.Thread(target=loop, daemon=True)
        self._t.start()

    def stop(self):
        self._stop.set()
        if self._t:
            self._t.join(timeout=3)
        if not self.samples:
            return None
        u = [s[1] for s in self.samples]
        m = [s[2] for s in self.samples]
        return {"n": len(u), "util_mean": float(np.mean(u)), "util_max": float(np.max(u)),
                "mem_mib_max": float(np.max(m))}


def benign_workload(ctx: Ctx, seed: int, n: int, pcfg: dict):
    rng = random.Random(seed)
    words = [f"w{rng.randrange(60000)}" for _ in range(60000)]
    docs = []
    for d in range(pcfg["n_documents"]):
        drng = random.Random(shash(f"doc{seed}-{d}"))
        docs.append(" ".join(drng.choice(words) for _ in range(pcfg["document_blocks"] * 16 + 8)))
    reqs = []
    for i in range(n):
        tenant = f"tenant-b{rng.randrange(pcfg['n_benign_tenants'])}"
        doc = docs[min(int(rng.expovariate(0.6)), len(docs) - 1)]
        sens = rng.random() < pcfg["p_sensitive"]
        if sens:
            fn, ln = rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
            q = (f"Draft a reply to {fn} {ln} (email {fn.lower()}.{ln.lower()}@example.org, "
                 f"phone 555-{rng.randrange(100,999)}-{rng.randrange(1000,9999)}) about their "
                 f"{rng.choice(MEDICAL_CONDITIONS)} follow-up visit.")
        else:
            q = f"Summarize the key obligations in section {rng.randrange(1, 40)} of the document."
        reqs.append({"i": i, "tenant": tenant, "sensitive": sens,
                     "prompt": f"{ctx.sys} {doc} {q}"})
    return reqs


def run_perf(ctx: Ctx, defense, level, seed, calib, n_req, w_req: JsonlWriter,
             w_run: JsonlWriter):
    pcfg = ctx.cfg["perf"]
    key = f"perf|{defense}|{level}|{seed}"
    ctx = ctx.begin(key)
    loader = None
    if ctx.kind == "simulated":
        ctx.raw.reseed(shash(key))
        ctx.set_load(level)
    elif level != "L0":
        from kv_attack.load_generator import LoadGenerator
        loader = LoadGenerator(base_url=ctx.base_url, model_id=ctx.model_id,
                               n_tenants=ctx.cfg["load"]["levels"][level], seed=seed)
        loader.start()
    srv = new_server(ctx, defense, 0.0, key, calib)
    reqs = benign_workload(ctx, seed, n_req, pcfg)
    gpu = GpuSampler() if ctx.kind == "real" else None
    m0 = scrape_prefix_metrics(ctx.base_url) if ctx.kind == "real" else None
    h0, b0 = (ctx.raw.tot_hit_blocks, ctx.raw.tot_blocks) if ctx.kind == "simulated" else (0, 0)
    if gpu:
        gpu.start()
    out = [None] * n_req

    def one(r):
        ttft, rec = srv.handle_full(r["tenant"], "benign", r["prompt"])
        out[r["i"]] = (ttft, rec)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=pcfg["concurrency"]) as ex:
        list(ex.map(one, reqs))
    wall = time.perf_counter() - t0
    gsum = gpu.stop() if gpu else None
    m1 = scrape_prefix_metrics(ctx.base_url) if ctx.kind == "real" else None
    if loader is not None:
        loader.stop()
    ttfts = np.array([o[0] for o in out])
    for r, (ttft, rec) in zip(reqs, out):
        w_req.write({"defense": defense, "level": level, "seed": seed, "tenant": r["tenant"],
                     "sensitive": r["sensitive"], "ttft_ms": ttft, "action": rec["action"],
                     "risk": rec["risk"], "decide_ms": rec["decide_ms"], "i": r["i"]})
    model_thr = n_req / max(1e-9, (ttfts.sum() / 1e3) / pcfg["concurrency"])
    run = {"defense": defense, "level": level, "seed": seed, "n": n_req,
           "concurrency": pcfg["concurrency"], "wall_s": round(wall, 3),
           "throughput_rps_wall": round(n_req / wall, 3),
           "throughput_rps_model": round(model_thr, 3),
           "throughput_kind": "wall" if ctx.kind == "real" else "simulated_model",
           "gpu": gsum, "vllm_prefix_metrics_before": m0, "vllm_prefix_metrics_after": m1}
    if ctx.kind == "simulated":
        run["cache_hit_rate"] = (ctx.raw.tot_hit_blocks - h0) / max(1, ctx.raw.tot_blocks - b0)
        run["cache_hit_rate_source"] = "sim_ground_truth_blocks"
    w_run.write(run)


def dry_run(cfg, prof, req_s=0.7, reset_mode='auto'):
    cells = build_cells(cfg, prof)
    seeds, nv = prof["seeds"], prof["n_victims"]
    est_reqs = {"A2": 90, "A1": 1000, "A0": 2000}
    evict_s = 0.0 if reset_mode in ('auto', 'epoch') else 500 * req_s
    tot = 0.0
    by = {}
    for c in cells:
        n = len(seeds) * nv[c["attack"]]
        t = n * (est_reqs[c["attack"]] * req_s + evict_s)
        tot += t
        by[c["attack"]] = by.get(c["attack"], 0) + t
    npf = len(cfg["perf"]["defenses"]) * len(cfg["perf"]["load_levels"]) * len(seeds)
    perf = npf * (prof["n_perf_requests"] * req_s / cfg["perf"]["concurrency"] + evict_s)
    orc = len(cfg["perf"]["defenses"]) * len(seeds) * (
        cfg["n_oracle_victims"] * 3 * 2 * req_s + evict_s)
    print(f"cells={len(cells)}  trials={sum(len(seeds)*nv[c['attack']] for c in cells)}")
    for a, t in by.items():
        print(f"  {a}: {t/3600:6.1f} h")
    print(f"  perf ({npf} runs): {perf/3600:6.1f} h   oracle: {orc/3600:5.1f} h")
    print(f"  TOTAL ~ {(tot+perf+orc)/3600:.1f} h at {req_s}s/request "
          f"(reset_mode={reset_mode}; 'evict' adds 500 filler requests per trial)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["vllm", "sglang", "sim"], required=True)
    ap.add_argument("--base-url", default="http://localhost:8001/v1")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--config", default=str(ROOT / "configs" / "final_benchmark.yaml"))
    ap.add_argument("--profile", default="standard")
    ap.add_argument("--out-root", default=str(ROOT / "experiments" / "results" / "final"))
    ap.add_argument("--resume", default=None, help="existing run dir to continue")
    ap.add_argument("--groups", default="oracle,security,perf")
    ap.add_argument("--risk-model", default="auto", choices=["auto", "presidio", "regex"])
    ap.add_argument("--server-flags", default="", help="exact server launch command/flags")
    ap.add_argument("--reset-mode", choices=["auto", "epoch", "evict"], default="auto",
                    help="between-trial isolation: 'evict' = 500-request flush (original protocol); "
                         "'epoch' = per-trial prefix nonce, validated by a contamination test")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="ignore preflight failure (recorded)")
    ap.add_argument("--limit-trials", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    prof = cfg["profiles"][args.profile]
    if args.dry_run:
        return dry_run(cfg, prof, reset_mode=args.reset_mode)
    groups = set(args.groups.split(","))

    ctx = Ctx(args, cfg)
    if args.resume:
        run_dir = Path(args.resume)
    else:
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
        run_dir = Path(args.out_root) / f"{ctx.backend_name}-{ts}-{uuid.uuid4().hex[:8]}"
        run_dir.mkdir(parents=True)
    (run_dir / "processed").mkdir(exist_ok=True)
    W = {n: JsonlWriter(run_dir / f"{n}.jsonl")
         for n in ("calibration", "trials", "oracle", "perf", "perf_runs")}

    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update({
        "run_id": run_dir.name, "backend_kind": ctx.kind, "backend": ctx.backend_name,
        "issue": 10, "profile": args.profile, "profile_values": prof,
        "config_path": os.path.relpath(args.config, ROOT), "config": cfg,
        "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        "command": " ".join([sys.executable] + sys.argv), "server_flags": args.server_flags,
        "risk_model": ctx.risk_name, "backend_info": ctx.info(),
        "git_commit": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain")),
        "hardware": hardware_info(), "status": "running",
        "started_utc": manifest.get("started_utc",
                                    datetime.datetime.now(datetime.timezone.utc).isoformat())})
    if ctx.kind == "simulated":
        manifest["WARNING"] = "SIMULATED backend: pipeline validation only, NOT evidence"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"[bench] run dir: {run_dir}\n[bench] kind={ctx.kind} profile={args.profile} "
          f"risk_model={ctx.risk_name}")

    seeds, nv = prof["seeds"], prof["n_victims"]
    max_v = max(nv.values())
    ctx.reset_mode = "evict"
    victims = {s: make_victims(ctx, s, max_v) for s in seeds}
    calibs = {}
    done_cal = {r["seed"]: r for r in W["calibration"].read()}
    for s in seeds:
        if s in done_cal:
            calibs[s] = done_cal[s]["calibration"]
            continue
        print(f"[bench] calibrating seed {s} ...")
        c = calibrate(ctx, s, victims[s])
        bad = preflight(c, ctx.info(), args.force)
        W["calibration"].write({"seed": s, "calibration": c, "preflight_problems": bad,
                                "forced": args.force})
        if bad:
            manifest["status"] = "ABORTED_PREFLIGHT"
            manifest["preflight_problems"] = bad
            manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
            sys.exit("[bench] PREFLIGHT FAILED (timing oracle not usable): " + "; ".join(bad) +
                     "\n        Nothing was measured. Fix the server (APC on? block size?) and re-run.")
        calibs[s] = c

    if args.reset_mode in ("auto", "epoch"):
        ok, diag = ctx.contamination_check(calibs[seeds[0]])
        manifest["contamination_check"] = {"passed": ok, **diag}
        if ok:
            ctx.reset_mode = "epoch"
        elif args.reset_mode == "epoch":
            sys.exit(f"[bench] epoch isolation FAILED contamination test: {diag}")
    manifest["reset_mode"] = ctx.reset_mode
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"[bench] trial isolation: {ctx.reset_mode}")

    if "oracle" in groups:
        done = {(r["defense"], r["seed"]) for r in W["oracle"].read()}
        for d in cfg["perf"]["defenses"]:
            for s in seeds:
                if (d, s) in done:
                    continue
                print(f"[bench] oracle {d} seed={s}")
                run_oracle(ctx, d, s, calibs[s], W["oracle"], cfg["n_oracle_victims"])

    if "security" in groups:
        cells = build_cells(cfg, prof)
        todo = []
        have = {r["trial_key"] for r in W["trials"].read() if not r.get("error")}
        for c in cells:
            for s in seeds:
                for v in range(nv[c["attack"]]):
                    if trial_key(c, s, v) not in have:
                        todo.append((c, s, v))
        random.Random(cfg["order_seed"]).shuffle(todo)
        if args.limit_trials:
            todo = todo[:args.limit_trials]
        print(f"[bench] {len(todo)} security trials to run ({len(have)} already done)")
        t_start = time.time()
        for n, (c, s, v) in enumerate(todo, 1):
            rec = run_security_trial(ctx, c, s, v, victims[s][v], calibs[s])
            W["trials"].write(rec)
            if n % 10 == 0 or n == len(todo):
                el = time.time() - t_start
                print(f"[bench]   {n}/{len(todo)}  elapsed {el/60:.1f} min  "
                      f"eta {(el/n)*(len(todo)-n)/60:.1f} min  last={rec['trial_key']} "
                      f"exact={rec.get('exact_match')} err={rec['error']}")

    if "perf" in groups:
        done = {(r["defense"], r["level"], r["seed"]) for r in W["perf_runs"].read()}
        p = cfg["perf"]
        for s in seeds:
            for lvl in p["load_levels"]:
                for d in p["defenses"]:
                    if (d, lvl, s) in done:
                        continue
                    print(f"[bench] perf {d} {lvl} seed={s}")
                    run_perf(ctx, d, lvl, s, calibs[s], prof["n_perf_requests"],
                             W["perf"], W["perf_runs"])

    manifest["status"] = "complete"
    manifest["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    try:
        from analysis.metrics import compute_and_save
        compute_and_save(run_dir)
    except Exception as e:
        print(f"[bench] WARNING: processing failed ({e}); run analysis.metrics later")
    print(f"[bench] DONE {run_dir}")


if __name__ == "__main__":
    main()