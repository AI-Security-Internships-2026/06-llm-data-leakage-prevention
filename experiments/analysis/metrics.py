"""
experiments/analysis/metrics.py  --  Issue #10
Recompute ALL headline metrics from the raw JSONL logs of one run directory.
reproduce_paper.py calls compute() itself; it never trusts stored aggregates.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import stats as S


def _read(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for ln in p.read_text().splitlines():
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            pass
    return out


def load_run(run_dir) -> dict:
    d = Path(run_dir)
    return {"dir": d, "manifest": json.loads((d / "manifest.json").read_text()),
            "calibration": _read(d / "calibration.jsonl"), "trials": _read(d / "trials.jsonl"),
            "oracle": _read(d / "oracle.jsonl"), "perf": _read(d / "perf.jsonl"),
            "perf_runs": _read(d / "perf_runs.jsonl")}


def cell_id(t) -> str:
    return f"{t['attack']}|{t['defense']}|fnr={t['fnr']}|{t['load']}"


def security_cells(trials: list[dict]) -> dict:
    ok = [t for t in trials if not t.get("error")]
    by = defaultdict(list)
    for t in ok:
        by[cell_id(t)].append(t)
    out = {}
    for cid, ts in by.items():
        k, n = sum(t["exact_match"] for t in ts), len(ts)
        per_seed = defaultdict(list)
        for t in ts:
            per_seed[t["seed"]].append(float(t["exact_match"]))
        calls = [t["total_api_calls"] for t in ts]
        att_reqs = sum(t["n_attacker_requests"] for t in ts)
        att_ns = sum(sum(v for a, v in t["attacker_actions"].items() if a != "SHARE") for t in ts)
        fns = [t["first_nonshare_attacker_req"] for t in ts if t["first_nonshare_attacker_req"] is not None]
        lo_w, hi_w = S.wilson_ci(k, n)
        cb = S.cluster_bootstrap_ci(per_seed, n_boot=2000, seed=1)
        out[cid] = {
            "attack": ts[0]["attack"], "defense": ts[0]["defense"], "fnr": ts[0]["fnr"],
            "load": ts[0]["load"], "n": n, "k_exact": k, "asr": k / n, "wilson95": [lo_w, hi_w],
            "seed_cluster_bootstrap95": [cb[1], cb[2]], "n_seeds": len(per_seed),
            "asr_by_seed": {str(s): float(np.mean(v)) for s, v in per_seed.items()},
            "queries": {**S.quantiles(calls, (50, 95)), "mean": float(np.mean(calls))},
            "queries_successful_median": (float(np.median([t["total_api_calls"] for t in ts if t["exact_match"]]))
                                          if k else None),
            "leaked_bits_mean": float(np.mean([t["leaked_bits"] for t in ts])),
            "name_recovery": float(np.mean([t["name_ok"] for t in ts])),
            "attack_time_model_s_median": float(np.median([t["attack_time_model_s"] for t in ts])),
            "victim_isolated_frac": float(np.mean([t["victim_action"] not in (None, "SHARE") for t in ts])),
            "attacker_request_isolated_frac": (att_ns / att_reqs) if att_reqs else None,
            "behavior_alert_frac_trials": float(np.mean([t["n_alerts"] > 0 for t in ts])),
            "first_nonshare_attacker_req_median": (float(np.median(fns)) if fns else None),
            "groups": ts[0]["groups"],
        }
    return out


def paired_vs_baseline(trials, cells) -> dict:
    """Each non-D0 cell vs D0 for the SAME attack (fnr=0, L0): paired by (seed, victim)."""
    ok = [t for t in trials if not t.get("error")]
    idx = {(cell_id(t), t["seed"], t["victim_idx"]): t["exact_match"] for t in ok}
    res, pv = {}, {}
    for cid, c in cells.items():
        base = f"{c['attack']}|D0_no_defense|fnr=0.0|{c['load']}"
        if cid == base or base not in cells:
            continue
        b, a = cells[base], c
        d, lo, hi = S.risk_difference_ci(b["k_exact"], b["n"], a["k_exact"], a["n"])
        odds, p = S.fisher_exact(b["k_exact"], b["n"], a["k_exact"], a["n"])
        b01 = sum(1 for (cc, s, v), m in idx.items() if cc == base and m and idx.get((cid, s, v)) is False)
        b10 = sum(1 for (cc, s, v), m in idx.items() if cc == base and (not m) and idx.get((cid, s, v)) is True)
        pm = (float(__import__("scipy.stats", fromlist=["binomtest"]).binomtest(b01, b01 + b10, 0.5).pvalue)
              if b01 + b10 else 1.0)
        res[cid] = {"baseline": base, "risk_reduction": d, "risk_reduction_ci95": [lo, hi],
                    "cohens_h": S.cohens_h(b["asr"], a["asr"]), "fisher_odds": odds,
                    "fisher_p": p, "mcnemar_discordant": [b01, b10], "mcnemar_p": pm,
                    "relative_leakage": (a["asr"] / b["asr"]) if b["asr"] > 0 else None}
        pv[cid] = p
    adj = S.holm(pv)
    for cid in res:
        res[cid]["fisher_p_holm"] = adj[cid]
    return res


def oracle_quality(rows) -> dict:
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["defense"]][r["class"]].append((r["seed"], r["ttft_ms"]))
    out = {}
    for d, cl in by.items():
        hit = np.array([t for _, t in cl["full_hit"]]); s1 = np.array([t for _, t in cl["s1_hit"]])
        miss = np.array([t for _, t in cl["miss"]])
        per_seed_auc = {}
        for s in sorted({s for s, _ in cl["full_hit"]}):
            h = [t for ss, t in cl["full_hit"] if ss == s]; m = [t for ss, t in cl["miss"] if ss == s]
            per_seed_auc[str(s)] = S.roc_auc(h, m, higher_is_positive=False)
        out[d] = {"n": {k: len(v) for k, v in cl.items()},
                  "median_ms": {"full_hit": float(np.median(hit)), "s1_hit": float(np.median(s1)),
                                "miss": float(np.median(miss))},
                  "auc_hit_vs_miss": S.roc_auc(hit, miss, higher_is_positive=False),
                  "auc_by_seed": per_seed_auc,
                  "auc_s1_vs_miss": S.roc_auc(s1, miss, higher_is_positive=False),
                  "tpr_at_1pct_fpr": S.tpr_at_fpr(hit, miss, 0.01),
                  "gap_ms": float(np.median(miss) - np.median(hit)),
                  "raw": {"full_hit": hit.tolist(), "s1_hit": s1.tolist(), "miss": miss.tolist()}}
        ab = S.bootstrap_ci(np.arange(len(hit)), stat=lambda idx: S.roc_auc(
            hit[np.asarray(idx, int) % len(hit)], miss[np.asarray(idx, int) % len(miss)], False),
            n_boot=500, seed=2)
        out[d]["auc_bootstrap95"] = [ab[1], ab[2]]
    return out


def perf_metrics(rows, runs) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[(r["defense"], r["level"])].append(r)
    rb = defaultdict(list)
    for r in runs:
        rb[(r["defense"], r["level"])].append(r)
    out = {}
    for (d, lvl), rs in by.items():
        t = np.array([r["ttft_ms"] for r in rs])
        per_seed = defaultdict(list)
        for r in rs:
            per_seed[r["seed"]].append(r["ttft_ms"])
        p50 = S.cluster_bootstrap_ci(per_seed, stat=np.median, n_boot=1000, seed=3)
        sens = [r for r in rs if r["sensitive"]]; clean = [r for r in rs if not r["sensitive"]]
        runs_ = rb.get((d, lvl), [])
        thr = [x["throughput_rps_wall" if x["throughput_kind"] == "wall" else "throughput_rps_model"]
               for x in runs_]
        chr_ = [x["cache_hit_rate"] for x in runs_ if x.get("cache_hit_rate") is not None]
        out[f"{d}|{lvl}"] = {
            "defense": d, "level": lvl, "n": len(rs), **S.quantiles(t, (50, 95, 99)),
            "ttft_mean": float(t.mean()), "p50_ci95": [p50[1], p50[2]],
            "isolation_fraction": float(np.mean([r["action"] != "SHARE" for r in rs])),
            "sensitive_share_rate": (float(np.mean([r["action"] == "SHARE" for r in sens])) if sens else None),
            "clean_isolation_rate": (float(np.mean([r["action"] != "SHARE" for r in clean])) if clean else None),
            "decide_ms_mean": float(np.mean([r["decide_ms"] for r in rs])),
            "decide_ms_p95": float(np.percentile([r["decide_ms"] for r in rs], 95)),
            "throughput_rps_mean": (float(np.mean(thr)) if thr else None),
            "throughput_kind": (runs_[0]["throughput_kind"] if runs_ else None),
            "cache_hit_rate_mean": (float(np.mean(chr_)) if chr_ else None),
            "gpu": [x["gpu"] for x in runs_ if x.get("gpu")] or None,
            "_ttft": t.tolist()}
    for key, m in out.items():
        base = out.get(f"D0_no_defense|{m['level']}")
        if base and m["defense"] != "D0_no_defense":
            bt, mt = np.array(base["_ttft"]), np.array(m["_ttft"])
            m["overhead_vs_D0"] = {
                "p50_pct": 100 * (m["p50"] / base["p50"] - 1), "p95_pct": 100 * (m["p95"] / base["p95"] - 1),
                "p99_pct": 100 * (m["p99"] / base["p99"] - 1),
                "mannwhitney_p": S.mannwhitney_p(mt, bt), "cliffs_delta": S.cliffs_delta(mt, bt),
                "throughput_pct": (100 * (m["throughput_rps_mean"] / base["throughput_rps_mean"] - 1)
                                   if m["throughput_rps_mean"] and base["throughput_rps_mean"] else None)}
    return out


def compute(run_dir) -> dict:
    r = load_run(run_dir)
    cells = security_cells(r["trials"])
    errs = [t for t in r["trials"] if t.get("error")]
    perf = perf_metrics(r["perf"], r["perf_runs"])
    res = {"run_id": r["manifest"]["run_id"], "backend_kind": r["manifest"]["backend_kind"],
           "status": r["manifest"].get("status"), "n_trials": len(r["trials"]),
           "n_trial_errors": len(errs), "error_examples": [e["error"] for e in errs[:3]],
           "security": cells, "paired_vs_D0": paired_vs_baseline(r["trials"], cells),
           "oracle": oracle_quality(r["oracle"]), "perf": perf}
    for v in res["perf"].values():
        v.pop("_ttft", None)
    return res


def compute_and_save(run_dir) -> Path:
    res = compute(run_dir)
    out = Path(run_dir) / "processed" / "metrics.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float))
    return out
