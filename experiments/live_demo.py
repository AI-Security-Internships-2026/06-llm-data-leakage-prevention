"""
experiments/live_demo.py   (v2: adds --fnr and prints detector alerts)
Small LIVE demo: run the same KV-cache timing attack against every defense (D0 to D5)
on a running server and print each result as it happens.

--fnr F  makes the content (PII) detector miss a fraction F of victims (fnr=1.0: it
         misses all of them).  This is the repo's own false-negative injection
         (DefendedServer fnr) used in the final benchmark; nothing else changes.

Usage (from the repo root, vLLM already running on port 8001):
    python experiments/live_demo.py --backend vllm --n-victims 3
    python experiments/live_demo.py --backend vllm --n-victims 3 --fnr 1.0 --defenses D0_no_defense,D4_risk_adaptive,D5_risk_behavior
Pipeline check without a GPU (output is tagged SIMULATED, NOT a result):
    python experiments/live_demo.py --backend sim --n-victims 2
"""
import argparse, datetime, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
import yaml
import run_final_benchmark as rfb

ALL = ["D0_no_defense", "D1_full_isolation", "D2_tenant_salt",
       "D3_binary_pii", "D4_risk_adaptive", "D5_risk_behavior"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["vllm", "sglang", "sim"], required=True)
    ap.add_argument("--base-url", default="http://localhost:8001/v1")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--config", default=str(ROOT / "configs" / "final_benchmark.yaml"))
    ap.add_argument("--risk-model", default="auto", choices=["auto", "presidio", "regex"])
    ap.add_argument("--reset-mode", default="auto", choices=["auto", "epoch", "evict"])
    ap.add_argument("--attack", default="A2", choices=["A0", "A1", "A2"])
    ap.add_argument("--n-victims", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fnr", type=float, default=0.0,
                    help="probability the PII/content detector misses a victim (1.0 = always misses)")
    ap.add_argument("--defenses", default=",".join(ALL))
    ap.add_argument("--out-dir", default=str(ROOT / "experiments" / "results" / "live_demo"),
                    help="folder for the saved results (a new file per run, never overwritten)")
    ap.add_argument("--no-save", action="store_true", help="do not write any result file")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    defenses = args.defenses.split(",")
    ctx = rfb.Ctx(args, cfg)
    tag = "SIMULATED (pipeline check only, NOT a result)" if ctx.kind == "simulated" else "LIVE"
    print("=" * 72)
    print(f"  KV-cache timing attack ({args.attack}) vs defenses   [{tag}]")
    print(f"  backend: {args.backend}   model: {ctx.model_id}")
    print(f"  victims: {args.n_victims}   seed: {args.seed}   risk model: {ctx.risk_name}")
    if args.fnr > 0:
        print(f"  PII DETECTOR MISS RATE (fnr): {args.fnr:.0%}  <- detector deliberately blinded")
    print("=" * 72)

    victims = rfb.make_victims(ctx, args.seed, args.n_victims)
    out_file = None
    if not args.no_save:
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
        out_file = Path(args.out_dir) / f"live-{args.backend}-{args.attack}-fnr{args.fnr}-{stamp}.jsonl"
        print(f"  results will be saved to: {out_file}")

    print("\nCalibrating the timing signal ...")
    ctx.reset_mode = "evict"
    calib = rfb.calibrate(ctx, args.seed, victims)
    gap = calib["miss_mean_ms"] - calib["hit_mean_ms"]
    print(f"  cache hit  ~ {calib['hit_mean_ms']:.0f} ms    cache miss ~ {calib['miss_mean_ms']:.0f} ms    gap ~ {gap:.0f} ms")
    bad = rfb.preflight(calib, ctx.info(), False)
    if bad:
        sys.exit("Timing signal not usable: " + "; ".join(bad) +
                 "\nCheck that vLLM was started with --enable-prefix-caching --block-size 16")
    if args.reset_mode in ("auto", "epoch"):
        ok, _ = ctx.contamination_check(calib)
        if ok:
            ctx.reset_mode = "epoch"
    print(f"  trial isolation: {ctx.reset_mode}\n")

    totals = {d: [0, 0] for d in defenses}
    for v in victims:
        g = v["ground_truth"]
        print("#" * 72)
        print(f"  VICTIM {v['victim_id'] + 1} of {len(victims)}   secret: {g['name']}  |  {g['condition']}")
        print("#" * 72)
        sys.stdout.flush()
        for d in defenses:
            cell = {"attack": args.attack, "defense": d, "fnr": args.fnr, "load": "L0", "groups": ["live"]}
            t0 = time.time()
            rec = rfb.run_security_trial(ctx, cell, args.seed, v["victim_id"], v, calib)
            if out_file is not None:
                with open(out_file, "a") as fh:
                    fh.write(json.dumps({**rec, "run_kind": ctx.kind, "backend": args.backend,
                                         "saved_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()},
                                        default=str) + "\n")
            if rec.get("error"):
                print(f"  {d:<20} ERROR {rec['error']}")
                continue
            ok = bool(rec["exact_match"])
            totals[d][0] += ok
            totals[d][1] += 1
            rn = rec["recovered"]
            blocked = {k: n for k, n in rec["attacker_actions"].items() if k != "SHARE"}
            print(f"  {d:<20} {'LEAKED ' if ok else 'BLOCKED'}  "
                  f"guess: {rn.get('name')} / {rn.get('condition')}   "
                  f"queries={rec['total_api_calls']}  time={time.time()-t0:.0f}s"
                  + (f"   defense actions={blocked}" if blocked else ""))
            print(f"  {'':<20} server saw victim as risk={rec['victim_risk']} -> {rec['victim_action']}"
                  + (f"   ATTACK DETECTED: {rec['n_alerts']} alerts, first non-SHARE reply at attacker request #{rec['first_nonshare_attacker_req']}, max suspicion {rec['max_suspicion']:.2f}"
                     if rec["n_alerts"] else ""))
            sys.stdout.flush()
        done = "   ".join(f"{d.split('_')[0]} {t[0]}/{t[1]}" for d, t in totals.items())
        print(f"\n  leaked so far ->  {done}\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()