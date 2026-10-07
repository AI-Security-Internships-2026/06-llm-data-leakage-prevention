import glob, json, sys
from pathlib import Path
R = Path(__file__).resolve().parent / "results"
OUT = Path(__file__).resolve().parent.parent / "docs" / "paper" / "ISSUE10_AUDIT_FINDINGS.json"
F = []
def add(sev, code, msg, **ev): F.append({"severity": sev, "code": code, "message": msg, "evidence": ev})
def load(p):
    try: return json.load(open(p))
    except Exception as e: return e

for p in sorted(R.glob("*.json")):
    d = load(p)
    if isinstance(d, Exception):
        add("HIGH", "CORRUPT_JSON", f"{p.name} is not valid JSON", error=str(d)[:120])

w13, pn = load(R/"kv_week13_final.json"), load(R/"paper_numbers.json")
matrix = (Path(__file__).resolve().parent.parent/"docs/paper/claim_evidence_matrix.md").read_text()
agg = w13["aggregate"]
add("INFO", "A2_CANONICAL", "kv_week13_final.json aggregate",
    n=agg["n_victims"], success_rate=agg["success_rate"], mean_calls=agg["mean_total_api_calls"],
    t1=w13["calibration"]["t1_threshold_ms"], sr_target_met=agg["sr_target_met"])
if "100% exact recovery" in matrix and agg["success_rate"] < 1:
    add("HIGH", "CLAIM_MISMATCH_ASR", "claim_evidence_matrix asserts 100% exact recovery / 79.8 mean queries / 11.52x "
        "but the canonical artifact says otherwise", artifact_success_rate=agg["success_rate"],
        artifact_mean_calls=agg["mean_total_api_calls"], artifact_improvement=agg["blq_improvement_factor"])
t1_pn = pn["numbers"]["t1_threshold_ms"]["value"]; t1_art = w13["calibration"]["t1_threshold_ms"]
if abs(t1_pn - t1_art) > 1:
    add("MED", "THRESHOLD_MISMATCH", "paper_numbers.json reports analytical T1/T2 while the attack used calibrated ones",
        paper_numbers_t1=t1_pn, used_t1=t1_art)

b50 = load(R/"kv_attack_results_50.json")
if b50["model"] != w13["model"]:
    add("HIGH", "NOT_SAME_TESTBED", "A0 baseline and A2 ran on different models/templates; the '>10x' comparison crosses testbeds",
        a0_model=b50["model"], a2_model=w13["model"])

for p in sorted(R.glob("defense-eval-*.json")):
    d = load(p)
    if isinstance(d, Exception): continue
    cal, bi = d.get("calibration", {}), d.get("backend_info", {})
    if cal.get("apc_signal_ok") is False or (bi.get("apc_enabled") is False and cal.get("gap_ms", 999) < 50):
        add("HIGH", "ORACLE_NOT_USABLE", f"{p.name}: timing oracle unusable (APC off / gap {cal.get('gap_ms')} ms)",
            n_victims=d["config"]["n_victims"], d0_residual=d["residual_asr"].get("D0_no_defense", {}).get("residual_asr"))
    meta = d.get("residual_asr", {}).get("_meta")
    if meta and "Derived estimate" in meta.get("caveat", ""):
        add("HIGH", "DERIVED_NOT_MEASURED", f"{p.name}: residual ASR is a formula, not a measurement", formula=meta["computation"],
            n_victims=d["config"]["n_victims"])
    if not any(k.startswith("D5") for k in d.get("policy_isolation", {})):
        add("HIGH", "NO_D5", f"{p.name}: contains no D5 results", policies=list(d.get("policy_isolation", {})))

ab = []
for p in sorted(R.glob("ablation-*.json")):
    d = load(p)
    if isinstance(d, Exception): continue
    ab.append(d)
    if d["n_victims"] < 30: add("MED", "ABLATION_UNDERPOWERED", f"{p.name}: n={d['n_victims']}, 1 seed, no perf metrics", ci_upper_for_0_of_5=0.4345)
if len(ab) >= 2:
    a3 = {x["run_id"]: [r for r in x["results"] if r["condition"].startswith("AB3")][0]["asr"] for x in ab}
    if len(set(a3.values())) > 1:
        add("HIGH", "AB3_UNSTABLE", "AB3 (behaviour-only) ASR differs between runs (depends on probe timing assumption)", asr_by_run=a3)

for pref in ("a1-attack", "d5-eval"):
    if not glob.glob(str(R/f"{pref}-*.json")):
        add("HIGH", "MISSING_ARTIFACT", f"no {pref}-*.json: attack A1 / defense D5 never evaluated under that name")

lt = load(R/"load-test-2026-09-22-ddd807bd.json"); bc = load(R/"backend-cmp-2026-09-25-59c23b91.json")
add("INFO", "A2_ASR_ACROSS_RUNS", "A2 ASR differs between earlier issues (different thresholds/seeds/n)",
    week13_final=agg["success_rate"], load_test_L0=lt["levels"][0]["attack"]["asr"],
    backend_cmp=[(s.get("backend"), s.get("asr")) for s in bc.get("attack_summaries", [])])

OUT.write_text(json.dumps(F, indent=2))
for f in F: print(f"[{f['severity']:4}] {f['code']:22} {f['message']}")
print(f"\n{sum(f['severity']=='HIGH' for f in F)} HIGH, {sum(f['severity']=='MED' for f in F)} MED  -> {OUT}")
