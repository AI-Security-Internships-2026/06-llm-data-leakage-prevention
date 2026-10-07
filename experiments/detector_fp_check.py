import argparse, datetime, json, os, platform, random, subprocess, sys, uuid
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from kv_attack.probe_detector import ProbeDetector, SuspicionLevel

QUESTIONS = [
    "How do I reset my router password", "Write a haiku about autumn",
    "Explain the difference between TCP and UDP", "Give me a recipe for lentil soup",
    "What is the capital of Australia", "Summarize the causes of World War I",
    "Draft an email asking for a deadline extension", "How does a bloom filter work",
    "Suggest names for a bakery", "Translate good morning into French",
    "What are the side effects of caffeine", "Explain gradient descent simply",
    "Plan a three day trip to Lisbon", "Why is the sky blue",
    "Write a SQL query to find duplicate emails", "How do vaccines train the immune system",
    "Compare Python and Rust for systems work", "Give tips for a job interview",
    "What causes inflation", "Explain how a transformer attends to tokens",
    "Recommend a workout for beginners", "How to center a div in CSS",
    "Describe the plot of Hamlet", "What is the boiling point of water at altitude",
    "Write a limerick about a cat", "How do I fix a leaking tap",
    "Explain public key cryptography", "List healthy breakfast ideas",
]
NAMES = ["Alice", "Bob", "Carol", "Dave", "Erin", "Frank", "Grace", "Heidi"]
TOPICS = ["billing", "shipping", "login", "refund", "outage", "upgrade"]
CONDS = ["diabetes", "asthma", "migraine", "hypertension", "anxiety", "arthritis"]
LONG = ("You are a support assistant for Acme Corp. Follow the policy handbook carefully, "
        "answer politely, cite the relevant section, never reveal internal notes, and keep "
        "replies under one hundred and fifty words. ") * 3


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    m = z * (p*(1-p)/n + z*z/(4*n*n))**0.5 / d
    return max(0.0, c-m), min(1.0, c+m)


SCENARIOS = {
    "benign_diverse_chat": {
        "label": "BENIGN", "p_hit": 0.6,
        "prompt": lambda r, i: r.choice(QUESTIONS),
        "interval": lambda r: r.uniform(8, 90),
    },
    "benign_templated_app": {
        "label": "BENIGN", "p_hit": 0.6,
        "prompt": lambda r, i: f"Summarize support ticket {r.randint(1000,9999)} for customer {r.choice(NAMES)} about {r.choice(TOPICS)}",
        "interval": lambda r: r.uniform(3, 20),
    },
    "benign_batch_job": {
        "label": "BENIGN", "p_hit": 0.8,
        "prompt": lambda r, i: f"Summarize support ticket {r.randint(1000,9999)} for customer {r.choice(NAMES)} about {r.choice(TOPICS)}",
        "interval": lambda r: r.uniform(0.2, 1.0),
    },
    "benign_long_template_slow": {
        "label": "BENIGN", "p_hit": 0.9,
        "prompt": lambda r, i: LONG + f"Summarize ticket {r.randint(1000,9999)} for {r.choice(NAMES)}",
        "interval": lambda r: r.uniform(20, 60),
    },
    "attacker_fast": {
        "label": "ATTACK", "p_hit": 0.1,
        "prompt": lambda r, i: f"The patient {NAMES[i % len(NAMES)]} has {CONDS[(i // len(NAMES)) % len(CONDS)]}",
        "interval": lambda r: r.uniform(0.1, 0.5),
    },
    "attacker_paced": {
        "label": "ATTACK", "p_hit": 0.1,
        "prompt": lambda r, i: f"The patient {NAMES[i % len(NAMES)]} has {CONDS[(i // len(NAMES)) % len(CONDS)]}",
        "interval": lambda r: r.uniform(25, 40),
    },
    "attacker_paced_longprompt": {
        "label": "ATTACK", "p_hit": 0.1,
        "prompt": lambda r, i: LONG + f"The patient {NAMES[i % len(NAMES)]} has {CONDS[(i // len(NAMES)) % len(CONDS)]}",
        "interval": lambda r: r.uniform(25, 40),
    },
    "attacker_paced_shuffled": {
        "label": "ATTACK", "p_hit": 0.1,
        "prompt": lambda r, i: (f"{r.choice(QUESTIONS)}. " + f"The patient {r.choice(NAMES)} has {r.choice(CONDS)}"),
        "interval": lambda r: r.uniform(25, 40),
    },
}


def run_session(cfg, rng, probes, thr_med, thr_high):
    det = ProbeDetector(client_id="sim", threshold_med=thr_med, threshold_high=thr_high)
    t = 0.0
    for i in range(probes):
        t += cfg["interval"](rng)
        hit = rng.random() < cfg["p_hit"]
        ttft = rng.gauss(90, 10) if hit else rng.gauss(680, 60)
        res = det.observe(cfg["prompt"](rng, i), ttft, hit, timestamp_s=t)
        if res.level == SuspicionLevel.ALERT:
            return i + 1, res.to_dict()
    return None, det.results[-1].to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sessions", type=int, default=200)
    ap.add_argument("--probes", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--thr-med", type=float, default=0.40)
    ap.add_argument("--thr-high", type=float, default=0.65)
    ap.add_argument("--output-dir", default="experiments/results")
    a = ap.parse_args()

    os.makedirs(a.output_dir, exist_ok=True)
    run_id = f"detector-fp-{datetime.date.today()}-{uuid.uuid4().hex[:8]}"
    rng = random.Random(a.seed)
    rows, out = [], {}
    for name, cfg in SCENARIOS.items():
        alerts, sess = [], []
        for _ in range(a.n_sessions):
            first, last = run_session(cfg, rng, a.probes, a.thr_med, a.thr_high)
            alerts.append(first)
            sess.append(last)
        fired = [x for x in alerts if x is not None]
        lo, hi = wilson(len(fired), a.n_sessions)
        feat_mean = {k: round(float(np.mean([s["features"][k] for s in sess])), 3)
                     for k in sess[0]["features"]}
        row = {"scenario": name, "label": cfg["label"], "n_sessions": a.n_sessions,
               "alert_rate": round(len(fired)/a.n_sessions, 4),
               "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
               "median_probes_to_alert": float(np.median(fired)) if fired else None,
               "mean_final_features": feat_mean}
        rows.append(row)

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                         stderr=subprocess.DEVNULL).strip()
    except Exception:
        commit = None
    payload = {"run_id": run_id, "study": "detector_fp_check", "synthetic": True,
               "seed": a.seed, "n_sessions": a.n_sessions, "probes_per_session": a.probes,
               "thr_med": a.thr_med, "thr_high": a.thr_high,
               "command": " ".join(sys.argv), "git_commit": commit,
               "platform": platform.platform(), "results": rows}
    path = Path(a.output_dir) / f"{run_id}.json"
    path.write_text(json.dumps(payload, indent=2))

    print(f"\n{'Scenario':<26}{'Label':<8}{'AlertRate':>10}{'95% CI':>18}{'MedProbes':>11}")
    print("-" * 73)
    for r in rows:
        mp = "-" if r["median_probes_to_alert"] is None else f"{r['median_probes_to_alert']:.0f}"
        print(f"{r['scenario']:<26}{r['label']:<8}{r['alert_rate']:>10.3f}"
              f"  ({r['ci_lo']:.3f},{r['ci_hi']:.3f}){mp:>11}")
    print("\nMean final features per scenario:")
    for r in rows:
        print(f"  {r['scenario']:<26}{r['mean_final_features']}")
    print(f"\nSaved: {path}\n(SYNTHETIC sanity check - not a paper result)")


if __name__ == "__main__":
    main()