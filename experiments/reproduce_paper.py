"""
experiments/reproduce_paper.py  --  Issue #10 single reproduction command

    python experiments/reproduce_paper.py                       # newest complete REAL run per backend
    python experiments/reproduce_paper.py --run-dir <dir> [--run-dir <dir2>]
    python experiments/reproduce_paper.py --run-dir <sim dir> --allow-simulated   # pipeline check only

What it does
  1. loads the raw logs of the run(s) (trials / oracle / perf JSONL),
  2. RECOMPUTES every metric with experiments/analysis (stored aggregates are ignored),
  3. writes Figures 1-8 and Tables I-VI (.png/.md/.csv) + provenance.json + ACCEPTANCE.md,
  4. exits non-zero if any acceptance criterion of Issue #10 is not met.

No paper number is typed in this file.  Every value comes from a raw artifact or from
constants imported from the code base.  Simulated data are refused unless
--allow-simulated, and then every output is stamped SIMULATED.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from analysis import metrics as M

CORE = ["D0_no_defense", "D1_full_isolation", "D2_tenant_salt", "D3_binary_pii",
        "D4_risk_adaptive", "D5_risk_behavior"]
SHORT = {"D0_no_defense": "D0", "D1_full_isolation": "D1", "D2_tenant_salt": "D2",
         "D3_binary_pii": "D3", "D4_risk_adaptive": "D4", "D5_risk_behavior": "D5",
         "AB2_content_confidence": "AB2", "AB3_behavior_only": "AB3",
         "AB0_none": "AB0", "AB1_content_only": "AB1", "FULL_all_components": "FULL"}
COL = {"D0": "#d62728", "D1": "#7f7f7f", "D2": "#8c564b", "D3": "#ff7f0e", "D4": "#2ca02c",
       "D5": "#1f77b4", "AB2": "#9467bd", "AB3": "#17becf", "AB0": "#d62728",
       "AB1": "#2ca02c", "FULL": "#1f77b4"}


class Ctx:
    def __init__(self, runs, out, simulated):
        self.runs, self.out, self.sim = runs, out, simulated
        self.primary = runs[0]
        self.m = self.primary["metrics"]
        self.cfg = self.primary["manifest"]["config"]
        self.alias = self.cfg["aliases"]
        self.made: list[str] = []

    def cell(self, attack, defense, fnr=0.0, load="L0", run=None):
        from kv_attack.defended_serving import DEFENSES
        d = self.alias.get(defense, defense)
        if not DEFENSES[d].use_content:
            fnr = 0.0
        m = (run or self.primary)["metrics"]
        return m["security"].get(f"{attack}|{d}|fnr={float(fnr)}|{load}")

    def perf(self, defense, level="L0"):
        return self.m["perf"].get(f"{self.alias.get(defense, defense)}|{level}")


def ci(c):
    return f"[{c['wilson95'][0]:.2f},{c['wilson95'][1]:.2f}]"


def f(x, nd=2, none="n/a"):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return none
    return f"{x:.{nd}f}"


def stamp(c: Ctx, fig):
    if c.sim:
        fig.text(0.5, 0.5, "SIMULATED - NOT EVIDENCE", fontsize=34, color="red", alpha=0.18,
                 ha="center", va="center", rotation=25, weight="bold")


def save_fig(c: Ctx, fig, name):
    stamp(c, fig)
    fig.tight_layout()
    p = c.out / f"{name}.png"
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    c.made.append(p.name)
    print(f"  ok {p.name}")


def save_table(c: Ctx, name, title, header, rows, note=""):
    banner = "> **SIMULATED DATA - pipeline validation only, not evidence.**\n\n" if c.sim else ""
    md = f"## {title}\n\n{banner}| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n"
    for r in rows:
        md += "| " + " | ".join(str(x) for x in r) + " |\n"
    if note:
        md += f"\n{note}\n"
    (c.out / f"{name}.md").write_text(md)
    with open(c.out / f"{name}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    c.made += [f"{name}.md", f"{name}.csv"]
    print(f"  ok {name}.md/.csv")


def yerr(cells):
    a = np.array([x["asr"] for x in cells])
    lo = np.clip(a - np.array([x["wilson95"][0] for x in cells]), 0, None)
    hi = np.clip(np.array([x["wilson95"][1] for x in cells]) - a, 0, None)
    return a, [lo, hi]


def fig1(c):
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.axis("off")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.6)

    def box(x, y, w, h, t, fc):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec="k", lw=1.2))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8.5)

    def arr(x1, y1, x2, y2):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", lw=1.3))
    box(0.2, 3.3, 2.0, 0.9, "Victim tenant\n(prompt with PII)", "#cfe8ff")
    box(0.2, 1.85, 2.0, 0.9, "Benign tenants\n(shared documents)", "#e6e6e6")
    box(0.2, 0.4, 2.0, 0.9, "Attacker tenant\n(black-box TTFT probes)", "#ffd6d6")
    box(3.2, 0.9, 3.4, 3.0, "DEFENDED SERVING LAYER\n\nL1 content risk (Presidio)\n+ confidence gate\n"
        "L2 probe-behaviour detector\nL3 cache policy:\nSHARE / RESTRICT (tenant salt)\n/ ISOLATE (unique salt)", "#fff2cc")
    box(7.6, 1.6, 3.0, 1.6, "Shared inference backend\nAutomatic Prefix Cache\n(vLLM / SGLang)", "#d5f5d5")
    for y0, y1 in ((3.75, 3.4), (2.3, 2.4), (0.85, 1.4)):
        arr(2.2, y0, 3.2, y1)
    arr(6.6, 2.4, 7.6, 2.4)
    ax.annotate("TTFT = timing oracle", (2.3, 0.55), (7.7, 1.65),
                arrowprops=dict(arrowstyle="->", ls="--", color="red", lw=1.3,
                                connectionstyle="arc3,rad=-0.25"), color="red", fontsize=8.5)
    ax.set_title("Fig 1  Threat model and cross-layer architecture (conceptual)")
    save_fig(c, fig, "fig1_threat_model_architecture")


def fig2(c):
    o = c.m["oracle"]
    show = [d for d in ("D0_no_defense", "D2_tenant_salt", "D5_risk_behavior") if d in o]
    if not show:
        return print("  -- fig2: no oracle data")
    fig, axes = plt.subplots(1, len(show), figsize=(4.6 * len(show), 3.8), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, d in zip(axes, show):
        for cl, col, lab in (("full_hit", "#2ca02c", "full hit"), ("s1_hit", "#ff7f0e", "name-only hit"),
                             ("miss", "#d62728", "miss")):
            v = np.sort(o[d]["raw"][cl])
            ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=col, label=lab)
        ax.set_title(f"{SHORT[d]}  AUC(hit vs miss)={o[d]['auc_hit_vs_miss']:.3f}", fontsize=10)
        ax.set_xlabel("attacker-observed TTFT (ms)")
    axes[0].set_ylabel("ECDF")
    axes[0].legend(fontsize=8)
    fig.suptitle("Fig 2  Multi-level TTFT oracle as seen by the attacker (raw probes)")
    save_fig(c, fig, "fig2_ttft_oracle")


def fig3(c):
    rows = [(lab, c.cell(a, "D0_no_defense")) for a, lab in
            (("A0", "A0 enumeration"), ("A1", "A1 adaptive"), ("A2", "A2 two-stage"))]
    rows = [(l, x) for l, x in rows if x]
    if not rows:
        return print("  -- fig3: no D0 attack cells")
    tr = [t for t in c.primary["trials"] if not t.get("error") and t["defense"] == "D0_no_defense"
          and t["fnr"] == 0 and t["load"] == "L0"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    rng = np.random.default_rng(0)
    for i, (lab, cl) in enumerate(rows):
        q = [t["total_api_calls"] for t in tr if t["attack"] == cl["attack"]]
        ax.scatter(i + rng.uniform(-.12, .12, len(q)), q, s=14, alpha=.55, color="#1f77b4")
        ax.plot([i - .3, i + .3], [cl["queries"]["p50"]] * 2, "k-", lw=2)
        ax.plot([i - .3, i + .3], [cl["queries"]["p95"]] * 2, "k--", lw=1.4)
        ax.text(i, max(q) * 1.15, f"ASR {cl['asr']:.2f} {ci(cl)}\nn={cl['n']}", ha="center", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylim(top=ax.get_ylim()[1] * 2.2)
    ax.set_ylabel("queries per victim (log)")
    ax.set_title("Fig 3  Attack query efficiency, no defense, same testbed\nsolid = median, dashed = p95")
    save_fig(c, fig, "fig3_attack_efficiency")


def fig4(c):
    lv = list(c.cfg["load"]["levels"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for d in c.cfg["load"]["defenses"]:
        cells = [c.cell("A2", d, load=l) for l in lv]
        if not all(cells):
            continue
        a, e = yerr(cells)
        axes[0].errorbar(range(len(lv)), a, yerr=e, marker="o", capsize=4, color=COL[SHORT[d]], label=SHORT[d])
        axes[1].plot(range(len(lv)), [x["queries"]["p50"] for x in cells], "o-", color=COL[SHORT[d]], label=SHORT[d])
    for d in ("D0_no_defense", "D5_risk_behavior"):
        pf = [c.perf(d, l) for l in lv]
        if all(pf):
            axes[2].plot(range(len(lv)), [p["p50"] for p in pf], "o-", color=COL[SHORT[d]], label=f"{SHORT[d]} p50")
            axes[2].plot(range(len(lv)), [p["p95"] for p in pf], "s--", color=COL[SHORT[d]], alpha=.6, label=f"{SHORT[d]} p95")
    for ax, t in zip(axes, ("A2 attack success rate (95% Wilson)", "A2 median queries", "benign TTFT (ms)")):
        ax.set_xticks(range(len(lv)))
        ax.set_xticklabels(lv)
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("background load level")
        ax.legend(fontsize=7)
    axes[0].set_ylim(-.05, 1.05)
    fig.suptitle("Fig 4  Robustness vs multi-tenant load")
    save_fig(c, fig, "fig4_load_robustness")


def fig5(c):
    if len(c.runs) < 2:
        fig, ax = plt.subplots(figsize=(6.5, 2.6))
        ax.axis("off")
        ax.text(.5, .5, "Cross-backend comparison needs >= 2 backend runs\n(only one run supplied)",
                ha="center", va="center", fontsize=11)
        ax.set_title("Fig 5  Cross-backend (NOT AVAILABLE)")
        return save_fig(c, fig, "fig5_cross_backend")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    names = [f"{r['manifest']['backend']} {r['manifest']['backend_info'].get('framework_ver', '')}" for r in c.runs]
    for i, r in enumerate(c.runs):
        m = r["metrics"]
        o = m["oracle"].get("D0_no_defense")
        if o:
            axes[0].bar(i, o["gap_ms"], color="#1f77b4")
            axes[0].text(i, o["gap_ms"], f"AUC {o['auc_hit_vs_miss']:.3f}", ha="center", fontsize=8)
        for j, d in enumerate(("D0_no_defense", "D5_risk_behavior")):
            cc = c.cell("A2", d, run=r)
            if cc:
                axes[1].bar(i * 3 + j, cc["asr"], color=COL[SHORT[d]])
                axes[1].errorbar(i * 3 + j, cc["asr"], yerr=[[cc["asr"] - cc["wilson95"][0]], [cc["wilson95"][1] - cc["asr"]]],
                                 color="k", capsize=4)
            p = m["perf"].get(f"{d}|L0")
            if p:
                axes[2].bar(i * 3 + j, p["p50"], color=COL[SHORT[d]])
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, fontsize=8)
    axes[0].set_title("oracle gap: miss - hit (ms)", fontsize=10)
    for ax, t in ((axes[1], "A2 ASR  (red D0, blue D5)"), (axes[2], "benign TTFT p50 (red D0, blue D5)")):
        ax.set_xticks([i * 3 + .5 for i in range(len(names))])
        ax.set_xticklabels(names, fontsize=8)
        ax.set_title(t, fontsize=10)
    fig.suptitle("Fig 5  Cross-backend validation (same protocol, separate server launches)")
    save_fig(c, fig, "fig5_cross_backend")


def fig6(c):
    fs = c.cfg["fn_sweep"]
    xs = [0.0] + list(fs["fnr"])
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ok = False
    for d in fs["defenses"]:
        cells = [c.cell("A2", d, fnr=x) for x in xs]
        if not all(cells):
            continue
        ok = True
        a, e = yerr(cells)
        ax.errorbar(xs, a, yerr=e, marker="o", capsize=4, color=COL[SHORT[d]], label=SHORT[d])
    if not ok:
        return print("  -- fig6: no fn_sweep data")
    ax.plot([0, 1], [0, 1], "k:", alpha=.4, label="y = injected FNR")
    ax.set_xlabel("injected detector false-negative rate")
    ax.set_ylabel("A2 attack success rate")
    ax.set_title("Fig 6  Detector false negatives -> leakage (measured end-to-end)")
    ax.legend()
    save_fig(c, fig, "fig6_fn_to_leakage")


def pareto_points(c):
    pts, base = [], c.perf("D0_no_defense")
    for d in CORE + ["AB2_content_confidence", "AB3_behavior_only"]:
        cl, pf = c.cell("A2", d), c.perf(d)
        if cl and pf and base:
            pts.append({"d": d, "asr": cl["asr"], "ci": cl["wilson95"], "ovh": 100 * (pf["p50"] / base["p50"] - 1)})
    return pts


def fig7(c):
    pts = pareto_points(c)
    if not pts:
        return print("  -- fig7: missing security/perf pairs")
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for p in pts:
        ax.errorbar(p["ovh"], p["asr"],
                    yerr=[[max(0, p["asr"] - p["ci"][0])], [max(0, p["ci"][1] - p["asr"])]],
                    fmt="o", ms=9, color=COL[SHORT[p["d"]]], capsize=3)
        ax.annotate(SHORT[p["d"]], (p["ovh"], p["asr"]), xytext=(6, 6), textcoords="offset points")
    dom = lambda q, p: q["ovh"] <= p["ovh"] and q["asr"] <= p["asr"] and (q["ovh"] < p["ovh"] or q["asr"] < p["asr"])
    front = sorted([p for p in pts if not any(dom(q, p) for q in pts if q is not p)], key=lambda p: p["ovh"])
    ax.step([p["ovh"] for p in front], [p["asr"] for p in front], where="post", color="k", lw=1, alpha=.5)
    ax.set_xlabel("benign TTFT p50 overhead vs D0 (%)  [measured]")
    ax.set_ylabel("residual leakage: A2 ASR  [measured]")
    ax.set_title("Fig 7  Security-performance Pareto (all points empirical, same testbed)\nbottom-left is better")
    save_fig(c, fig, "fig7_security_performance_pareto")


def fig8(c):
    ab = c.cfg["ablation"]
    fig, axes = plt.subplots(1, len(ab["fnr"]), figsize=(6 * len(ab["fnr"]), 4), sharey=True)
    for ax, fnr in zip(np.atleast_1d(axes), ab["fnr"]):
        for i, cond in enumerate(ab["conditions"]):
            cl = c.cell("A2", cond, fnr=fnr)
            if not cl:
                continue
            ax.bar(i, cl["asr"], color=COL[SHORT[cond]], alpha=.85)
            ax.errorbar(i, cl["asr"],
                        yerr=[[max(0, cl["asr"] - cl["wilson95"][0])], [max(0, cl["wilson95"][1] - cl["asr"])]],
                        color="k", capsize=4)
        ax.set_xticks(range(len(ab["conditions"])))
        ax.set_xticklabels([SHORT[x] for x in ab["conditions"]])
        ax.set_title(f"injected detector FNR = {fnr}")
        ax.set_ylabel("A2 ASR")
    fig.suptitle("Fig 8  Ablation: AB0 none, AB1 content, AB2 +confidence, AB3 behaviour only, FULL")
    save_fig(c, fig, "fig8_ablation")


def table1(c):
    from kv_attack import BLOCK_SIZE, FIRST_NAMES, LAST_NAMES, MEDICAL_CONDITIONS
    nn, nc = len(FIRST_NAMES) * len(LAST_NAMES), len(MEDICAL_CONDITIONS)
    rows = [("Attack class", "KV-cache (automatic prefix cache) timing side-channel"),
            ("Attacker capability", "black-box completion API as an ordinary tenant; observes TTFT only"),
            ("Attacker knowledge", "system-prompt prefix, template layout, candidate dictionaries"),
            ("Candidate space", f"{nn} names x {nc} conditions = {nn * nc}"),
            ("Attacker calibration", "thresholds learned once per seed on an UNDEFENDED replica (strong attacker)"),
            ("Defender knowledge", "sees every request; content detector + probe-behaviour detector"),
            ("Trust boundary", "victim and attacker are different tenants of one shared backend"),
            ("Cache block size", f"{BLOCK_SIZE} tokens"),
            ("Target", "victim record (name, condition) placed in the shared prompt prefix"),
            ("Out of scope", "co-located hardware side channels; attacker-controlled system prompt")]
    save_table(c, "table_I_threat_model", "Table I - Threat model", ["Property", "Value"], rows)


def table2(c):
    rows = []
    for r in c.runs:
        m = r["manifest"]
        t1 = [x["calibration"]["t1_threshold_ms"] for x in r["calibration"]]
        t2 = [x["calibration"]["t2_threshold_ms"] for x in r["calibration"]]
        rows.append((m["backend"], f"{m['backend_info']['framework']} {m['backend_info']['framework_ver']}",
                     m["backend_info"]["model_id"], m["hardware"].get("gpu"), m.get("server_flags") or "(not recorded)",
                     m.get("reset_mode"), m["risk_model"], m["profile"], m["profile_values"]["seeds"],
                     m["profile_values"]["n_victims"], f"{min(t1):.0f}-{max(t1):.0f}", f"{min(t2):.0f}-{max(t2):.0f}",
                     (m.get("git_commit") or "?")[:8], "DIRTY" if m.get("git_dirty") else "clean", m["run_id"]))
    save_table(c, "table_II_experimental_setup", "Table II - Experimental setup",
               ["backend", "framework", "model", "GPU", "server flags", "trial isolation", "content detector",
                "profile", "seeds", "victims/seed", "T1 ms", "T2 ms", "git", "tree", "run id"], rows,
               note="CIs: 95% Wilson (proportions), seed-clustered bootstrap (medians). Tests: Fisher exact and exact "
                    "McNemar (paired by seed+victim), Holm-corrected over the defense family; effect sizes Cohen's h "
                    "and Cliff's delta.")


def table3(c):
    rows = []
    for a in ("A0", "A1", "A2"):
        cl = c.cell(a, "D0_no_defense")
        if cl:
            rows.append((a, cl["n"], cl["n_seeds"], f"{cl['k_exact']}/{cl['n']}", f(cl["asr"]), ci(cl),
                         f"[{cl['seed_cluster_bootstrap95'][0]:.2f},{cl['seed_cluster_bootstrap95'][1]:.2f}]",
                         f(cl["queries"]["p50"], 0), f(cl["queries"]["p95"], 0),
                         f(cl["queries_successful_median"], 0), f(cl["leaked_bits_mean"]),
                         f(cl["attack_time_model_s_median"], 1)))
    save_table(c, "table_III_attack_comparison", "Table III - Attack comparison (no defense, same testbed)",
               ["attack", "n", "seeds", "exact", "ASR", "Wilson95", "seed-boot95", "median Q", "p95 Q",
                "median Q (succ.)", "leaked bits", "attack time (model s)"], rows,
               note="A0 exhaustive enumeration (no early exit); A1 Week-12 linear scan with early exit; A2 corrected "
                    "two-stage. All three run on the identical two-stage template and victim set. Leaked bits = "
                    "log2(100) if name correct + log2(20) if condition correct.")


def table4(c):
    rows = []
    for d in CORE:
        cl, pf, o = c.cell("A2", d), c.perf(d), c.m["oracle"].get(d)
        pv = c.m["paired_vs_D0"].get(f"A2|{d}|fnr=0.0|L0")
        ov = (pf or {}).get("overhead_vs_D0")
        rows.append((SHORT[d],
                     f"{cl['k_exact']}/{cl['n']} ({f(cl['asr'])}) {ci(cl)}" if cl else "n/a",
                     (f"{f(pv['risk_reduction'])} [{f(pv['risk_reduction_ci95'][0])},{f(pv['risk_reduction_ci95'][1])}] "
                      f"h={f(pv['cohens_h'])} p_holm={pv['fisher_p_holm']:.3g}") if pv else "-",
                     f(o["auc_hit_vs_miss"], 3) if o else "n/a", f(o["tpr_at_1pct_fpr"]) if o else "n/a",
                     f(pf["p50"], 0) if pf else "n/a", f(pf["p95"], 0) if pf else "n/a", f(pf["p99"], 0) if pf else "n/a",
                     (f"{ov['p50_pct']:+.1f}% / {ov['p95_pct']:+.1f}%" if ov else "0"),
                     f(pf["throughput_rps_mean"]) if pf else "n/a",
                     f(pf["cache_hit_rate_mean"]) if pf else "n/a",
                     f(pf["isolation_fraction"]) if pf else "n/a", f(pf["decide_ms_mean"], 3) if pf else "n/a"))
    save_table(c, "table_IV_defense_comparison", "Table IV - Defense comparison (attack A2)",
               ["defense", "ASR k/n (ASR) Wilson95", "risk reduction vs D0 [CI] h p_holm", "oracle AUC",
                "TPR@1%FPR", "TTFT p50", "p95", "p99", "overhead vs D0 (p50/p95)", "throughput", "cache hit",
                "isolation frac", "decision ms"], rows,
               note="Security columns are MEASURED end-to-end against a live attack through the defended serving "
                    "layer (nothing derived). Performance columns: benign workload, load L0. Oracle columns: "
                    "attacker-side probe phase.")


def table5(c):
    fs, rows = c.cfg["fn_sweep"], []
    for d in fs["defenses"]:
        pf = c.perf(d)
        for x in [0.0] + list(fs["fnr"]):
            cl = c.cell("A2", d, fnr=x)
            if cl:
                rows.append((SHORT[d], x, f"{cl['k_exact']}/{cl['n']}", f(cl["asr"]), ci(cl), f(cl["leaked_bits_mean"]),
                             f(cl["victim_isolated_frac"]),
                             f(pf["sensitive_share_rate"]) if pf else "n/a",
                             f(pf["clean_isolation_rate"]) if pf else "n/a"))
    save_table(c, "table_V_detector_error_propagation", "Table V - Detector-error propagation",
               ["defense", "injected FNR", "exact", "A2 ASR", "Wilson95", "leaked bits", "victim isolated",
                "natural sensitive-share rate (FN exposure)", "natural clean-isolation rate (FP cost)"], rows,
               note="Injected FNR downgrades each detected request to LOW independently (all requests, including "
                    "attacker probes). Natural columns use the benign workload's ground-truth labels.")


def table6(c):
    ab, rows, base = c.cfg["ablation"], [], c.perf("D0_no_defense")
    for fnr in ab["fnr"]:
        for cond in ab["conditions"]:
            cl, pf = c.cell("A2", cond, fnr=fnr), c.perf(cond)
            ov = f"{100 * (pf['p50'] / base['p50'] - 1):+.1f}%" if pf and base else "n/a"
            rows.append((fnr, SHORT[cond], f"{cl['k_exact']}/{cl['n']}" if cl else "n/a", f(cl["asr"]) if cl else "n/a",
                         ci(cl) if cl else "n/a", f(cl["victim_isolated_frac"]) if cl else "n/a",
                         f(cl["attacker_request_isolated_frac"]) if cl else "n/a",
                         f(cl["queries"]["p50"], 0) if cl else "n/a", f(pf["isolation_fraction"]) if pf else "n/a",
                         f(pf["clean_isolation_rate"]) if pf else "n/a", ov))
    save_table(c, "table_VI_ablation", "Table VI - Ablation",
               ["injected FNR", "condition", "exact", "A2 ASR", "Wilson95", "victim isolated",
                "attacker requests isolated", "median Q", "benign isolation", "benign clean isolation (FP)",
                "TTFT p50 overhead vs D0"], rows,
               note="AB0 = D0, AB1 = D4, FULL = D5 by construction (trials shared, not re-run). AB2 adds the "
                    "confidence gate; AB3 uses the behaviour detector only.")


def acceptance(c: Ctx, figs_ok: bool):
    cfg, m, mf = c.cfg, c.m, c.primary["manifest"]
    prof = mf["profile_values"]
    checks = []

    def chk(name, ok, detail=""):
        checks.append((name, "PASS" if ok is True else ("PARTIAL" if ok == "partial" else "FAIL"), detail))
    chk("Evidence is from a REAL backend (not simulated)", not c.sim, mf["backend_kind"])
    exp = {(b["attack"], d): prof["n_victims"][b["attack"]] * len(prof["seeds"]) for b in cfg["core"] for d in b["defenses"]}
    miss = [f"{a}/{SHORT[d]}: {(c.cell(a, d) or {}).get('n', 0)}/{n}" for (a, d), n in exp.items()
            if (c.cell(a, d) or {}).get("n", 0) < n]
    chk("Core matrix evaluated on same testbed (all cells complete, no trial errors)",
        not miss and m["n_trial_errors"] == 0,
        f"incomplete: {miss}; trial errors: {m['n_trial_errors']}" if miss or m["n_trial_errors"] else "all cells complete")
    chk("D5 compared with D0-D4 (attack A2)", all(c.cell("A2", d) for d in CORE))
    chk("AB0-AB3 + FULL completed",
        all(c.cell("A2", x, fnr=fn) for x in cfg["ablation"]["conditions"] for fn in cfg["ablation"]["fnr"]))
    chk("Security + performance measured together",
        all(c.perf(d, l) for d in cfg["perf"]["defenses"] for l in cfg["perf"]["load_levels"]) and bool(m["security"]))
    chk("95% CIs and effect sizes reported",
        bool(m["paired_vs_D0"]) and all("wilson95" in v for v in m["security"].values()))
    cal_bad = [r["seed"] for r in c.primary["calibration"] if r["preflight_problems"] or r.get("forced")]
    auc0 = (m["oracle"].get("D0_no_defense") or {}).get("auc_hit_vs_miss", 0)
    chk("Timing oracle valid under D0 (preflight passed, AUC >= 0.99)", not cal_bad and auc0 >= 0.99,
        f"AUC(D0)={auc0:.3f}; bad seeds={cal_bad}")
    chk("Enough replication (>=3 seeds, >=18 A2 trials per core cell)",
        len(prof["seeds"]) >= 3 and all((c.cell("A2", d) or {"n": 0})["n"] >= 18 for d in CORE),
        f"seeds={len(prof['seeds'])}")
    chk("Cross-backend comparison (>= 2 real backends)", True if len(c.runs) >= 2 else "partial",
        f"{len(c.runs)} run(s): {[r['manifest']['backend'] for r in c.runs]}")
    chk("Reproducible provenance (clean git tree, command, server flags recorded)",
        (not mf.get("git_dirty")) and bool(mf.get("command")) and bool(mf.get("server_flags")),
        f"dirty={mf.get('git_dirty')} flags={'yes' if mf.get('server_flags') else 'NO'}")
    chk("Figures 1-8 and Tables I-VI generated", figs_ok, f"{len(c.made)} files")
    md = ("# Issue #10 acceptance report\n\n" + ("**SIMULATED RUN - not evidence.**\n\n" if c.sim else "")
          + "| criterion | status | detail |\n|---|---|---|\n" + "".join(f"| {a} | **{b}** | {d} |\n" for a, b, d in checks))
    (c.out / "ACCEPTANCE.md").write_text(md)
    print("\n" + md)
    return all(b == "PASS" for _, b, _ in checks)


def find_real_runs(root: Path):
    best = {}
    for p in sorted(root.glob("*/manifest.json")):
        mf = json.loads(p.read_text())
        if mf.get("backend_kind") == "real" and mf.get("status") == "complete":
            best[mf["backend"]] = p.parent
    return list(best.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", action="append", help="run directory (repeat for several backends)")
    ap.add_argument("--results-root", default=str(ROOT / "experiments" / "results" / "final"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow-simulated", action="store_true")
    a = ap.parse_args()

    dirs = [Path(x) for x in a.run_dir] if a.run_dir else find_real_runs(Path(a.results_root))
    if not dirs:
        sys.exit("No complete REAL run found under " + a.results_root + "\nRun experiments/run_final_benchmark.py "
                 "against vLLM/SGLang first (see docs/paper/ISSUE10_REPRODUCE.md).")
    runs = []
    for d in dirs:
        r = M.load_run(d)
        r["metrics"] = M.compute(d)
        runs.append(r)
    sim = any(r["manifest"]["backend_kind"] != "real" for r in runs)
    if sim and not a.allow_simulated:
        sys.exit("Refusing SIMULATED run(s) as paper evidence. Use --allow-simulated for a pipeline check.")
    out = Path(a.out) if a.out else ROOT / "experiments" / "figures" / ("final_SIMULATED" if sim else "final")
    out.mkdir(parents=True, exist_ok=True)
    c = Ctx(runs, out, sim)
    print(f"Runs: {[str(d) for d in dirs]}\nOutput: {out}\n\nFigures / tables")
    for fn in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, table1, table2, table3, table4, table5, table6):
        fn(c)
    prefixes = ("fig1", "fig2", "fig3", "fig4", "fig5", "fig6", "fig7", "fig8", "table_I_", "table_II_",
                "table_III_", "table_IV_", "table_V_", "table_VI_")
    figs_ok = all(any(x.startswith(p) for x in c.made) for p in prefixes)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
    prov = {"runs": [{"dir": str(d), "manifest_sha256": sha(d / "manifest.json"), "trials_sha256": sha(d / "trials.jsonl"),
                      "oracle_sha256": sha(d / "oracle.jsonl"), "perf_sha256": sha(d / "perf.jsonl")} for d in dirs],
            "simulated": sim, "command": " ".join(sys.argv),
            "reproduce_git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                            text=True).stdout.strip() or None}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2))
    (out / "metrics_recomputed.json").write_text(json.dumps(runs[0]["metrics"], indent=1, default=float))
    ok = acceptance(c, figs_ok)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
