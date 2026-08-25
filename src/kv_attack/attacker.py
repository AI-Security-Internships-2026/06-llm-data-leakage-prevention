
import time
from typing import Callable

import numpy as np
import scipy.stats
from openai import OpenAI

from kv_attack import MODEL_ID, N_REPEATS_FAST, KS_ALPHA


# ── TTFT measurement ─────────────────────────────────────────────────────────

def measure_ttft(client: OpenAI, prompt: str) -> float:
    """
    Measure Time-to-First-Token (TTFT) in milliseconds via streaming API.
    Timer stops at the first generated token (= end of prefill + 1 decode).
    Standard TTFT definition used in Papers 1-5.
    """
    t0 = time.perf_counter()
    stream = client.completions.create(
        model      = MODEL_ID,
        prompt     = prompt,
        max_tokens = 1,
        temperature= 0.0,
        stream     = True,
    )
    for _ in stream:
        break
    return (time.perf_counter() - t0) * 1_000.0   # ms


def measure_ttft_repeated(client: OpenAI, prompt: str, n: int) -> np.ndarray:
    """Return an ndarray of n TTFT measurements (ms) for the same prompt."""
    return np.array([measure_ttft(client, prompt) for _ in range(n)])


# ── Threshold calibration ─────────────────────────────────────────────────────

def calibrate_threshold(
    client               : OpenAI,
    known_cached_prompt  : str,
    miss_prompt_factory  : Callable[[], str],
    n_samples            : int = 200,
) -> dict:
    """
    Offline calibration. Measures TTFT distributions for:
      - HITS  : known_cached_prompt (all N samples use the same cached prompt)
      - MISSES: miss_prompt_factory() called once per sample (unique each time)

    Parameters
    ----------
    known_cached_prompt : str
        Full prompt whose KV blocks are already in cache. All N hit samples
        use this same prompt — repeated hits are fast and consistent.
    miss_prompt_factory : callable() -> str
        Returns a unique full prompt on every call. The prompt must be the
        same total length as known_cached_prompt and must NOT be in cache.
        Using a unique UUID in the first private block guarantees a cold miss
        on every call and cascades all 193 private blocks into misses.

    Returns dict with threshold, statistics, and recommended N_REPEATS.
    """
    print(f"[calibrate] Measuring {n_samples} HIT samples ...")
    hit_ttfts = measure_ttft_repeated(client, known_cached_prompt, n=n_samples)

    print(f"[calibrate] Measuring {n_samples} MISS samples "
          f"(unique prompt per sample — guaranteed cold miss) ...")
    miss_ttfts = np.array([
        measure_ttft(client, miss_prompt_factory())
        for _ in range(n_samples)
    ])

    # ── KS test ──────────────────────────────────────────────────────────────
    ks_stat, p_val = scipy.stats.ks_2samp(hit_ttfts, miss_ttfts)

    hit_mean  = float(hit_ttfts.mean())
    hit_std   = float(hit_ttfts.std())
    miss_mean = float(miss_ttfts.mean())
    miss_std  = float(miss_ttfts.std())
    delta_ms  = miss_mean - hit_mean

    print(f"[calibrate] HIT  mean={hit_mean:.2f} ms  std={hit_std:.2f} ms")
    print(f"[calibrate] MISS mean={miss_mean:.2f} ms  std={miss_std:.2f} ms")
    print(f"[calibrate] Delta = {delta_ms:.2f} ms  "
          f"({'✓ misses slower' if delta_ms > 0 else '✗ INVERTED'})")
    print(f"[calibrate] KS stat={ks_stat:.4f}  p={p_val:.3e}")

    if delta_ms <= 0:
        raise RuntimeError(
            f"Timing gap INVERTED (delta={delta_ms:.2f} ms). "
            f"Hits are slower than misses. "
            f"Check that miss_prompt_factory returns same-length prompts."
        )

    if p_val >= KS_ALPHA:
        raise RuntimeError(
            f"Timing gap NOT statistically significant "
            f"(p={p_val:.2e} >= alpha={KS_ALPHA:.0e}). "
            f"Delta={delta_ms:.2f} ms, hit_std={hit_std:.2f} ms, "
            f"miss_std={miss_std:.2f} ms. "
            f"If miss_std is large, the miss prompt is being cached — "
            f"verify miss_prompt_factory returns a unique prompt each call. "
            f"If delta is small, extend _PRIVATE_TEMPLATE in victim_seeder.py."
        )
    print(f"[calibrate] ✓ Gap confirmed significant at p < {KS_ALPHA:.0e}")

    # ── Youden-J threshold (O(n log n)) ──────────────────────────────────────
    all_vals   = np.concatenate([hit_ttfts,          miss_ttfts       ])
    all_labels = np.concatenate([np.ones(n_samples), np.zeros(n_samples)])

    order         = np.argsort(all_vals)
    sorted_vals   = all_vals[order]
    sorted_labels = all_labels[order]

    cum_hits   = np.cumsum(sorted_labels)
    cum_miss   = np.cumsum(1 - sorted_labels)
    total_hits = int(cum_hits[-1])
    total_miss = int(cum_miss[-1])

    sens     = cum_hits[:-1] / total_hits
    spec     = 1.0 - cum_miss[:-1] / total_miss
    j_scores = sens + spec - 1.0
    best_idx  = int(np.argmax(j_scores))
    threshold = float((sorted_vals[best_idx] + sorted_vals[best_idx + 1]) / 2.0)

    # ── Recommended N_REPEATS ─────────────────────────────────────────────────
    n_candidates  = 2000
    pooled_sigma  = float(np.sqrt((hit_std**2 + miss_std**2) / 2))
    recommended_n = None
    for n_try in range(1, 201):
        se_diff          = pooled_sigma * np.sqrt(2.0 / n_try)
        p_miss_above_hit = float(scipy.stats.norm.cdf(-delta_ms / se_diff))
        p_all_correct    = (1.0 - p_miss_above_hit) ** n_candidates
        if p_all_correct >= 0.99:
            recommended_n = n_try
            break

    if recommended_n is None:
        recommended_n = 200
        print(f"[calibrate] WARNING: recommended N_REPEATS > 200. "
              f"Gap too small. Extend _PRIVATE_TEMPLATE.")
    else:
        print(f"[calibrate] Recommended N_REPEATS for 99% SR: {recommended_n} "
              f"(currently N_FAST={N_REPEATS_FAST})")

    return {
        "threshold_ms"      : threshold,
        "hit_mean_ms"       : hit_mean,
        "hit_std_ms"        : hit_std,
        "miss_mean_ms"      : miss_mean,
        "miss_std_ms"       : miss_std,
        "delta_ms"          : delta_ms,
        "ks_stat"           : float(ks_stat),
        "ks_p_value"        : float(p_val),
        "youden_j"          : float(j_scores[best_idx]),
        "recommended_n_rpts": recommended_n,
    }


# ── Per-probe classification ──────────────────────────────────────────────────

def measure_mean_ttft(
    client: OpenAI,
    prompt: str,
    n: int = N_REPEATS_FAST,
) -> float:
    """Return mean TTFT (ms) over n measurements."""
    return float(measure_ttft_repeated(client, prompt, n).mean())


def is_cache_hit(
    client       : OpenAI,
    prompt       : str,
    threshold_ms : float,
    n            : int = N_REPEATS_FAST,
) -> tuple[bool, float]:
    """
    Classify prompt as cache hit (low TTFT) or miss (high TTFT).
    Returns (is_hit, mean_ttft_ms).
    """
    mean_ttft = measure_mean_ttft(client, prompt, n=n)
    return mean_ttft < threshold_ms, mean_ttft
