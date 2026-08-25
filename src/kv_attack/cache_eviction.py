
import random
import time
import uuid
import urllib.parse
import urllib.request

from openai import OpenAI

from kv_attack import VLLM_BASE_URL, MODEL_ID, EVICT_REQUESTS, EVICT_TOKENS

# Filler vocabulary: common English words with NO medical/legal/financial terms.
# Using neutral words guarantees eviction prompts never accidentally collide
# with any victim or attacker prefix in the cache.
_FILLER_WORDS = [
    "the", "of", "and", "in", "to", "a", "is", "that", "for", "on",
    "are", "with", "as", "at", "be", "this", "from", "or", "by", "an",
    "but", "not", "they", "which", "one", "had", "all", "were", "their",
    "there", "been", "has", "more", "when", "who", "will", "each",
    "about", "up", "out", "then", "them", "these", "so", "some", "would",
    "into", "than", "time", "only", "could", "new", "its", "two", "also",
    "after", "use", "many", "first", "way", "water", "long", "little",
]


def _make_eviction_prompt(token_count: int) -> str:
    """
    Build a unique eviction prompt.
    UUID prefix guarantees it never matches any cached victim or attacker prefix.
    Word count is approximate — tokenisation may vary by ±5 tokens.
    """
    unique_id = uuid.uuid4().hex[:12]
    words     = random.choices(_FILLER_WORDS, k=token_count)
    return f"[EVICT-{unique_id}] " + " ".join(words)


def _get_cache_hit_rate(base_url: str = VLLM_BASE_URL) -> float | None:
    """
    Query the vLLM Prometheus endpoint for the current KV cache hit rate (%).
    Returns None if the endpoint is unavailable or the metric is absent.

    ── URL construction (FIX #10) ───────────────────────────────────────────
    Uses urllib.parse so any trailing slashes, query strings, or port numbers
    in VLLM_BASE_URL are handled correctly.
    """
    try:
        parsed   = urllib.parse.urlparse(base_url)
        # Replace path with /metrics regardless of what path was in base_url
        metrics_url = urllib.parse.urlunparse(
            parsed._replace(path="/metrics", query="", fragment="")
        )
        with urllib.request.urlopen(metrics_url, timeout=3) as resp:
            for line in resp.read().decode().splitlines():
                # vLLM 0.27.x Prometheus metric name
                if line.startswith("vllm:gpu_prefix_cache_hit_rate_perc"):
                    parts = line.split()
                    return float(parts[-1])
    except Exception:
        pass
    return None


def evict_cache(
    client     : OpenAI,
    base_url   : str = VLLM_BASE_URL,
    n_requests : int = EVICT_REQUESTS,
    token_length: int = EVICT_TOKENS,
    confirm    : bool = True,
) -> dict:
    """
    Flood the KV cache with unique unrelated prompts to trigger LRU eviction
    of ALL existing victim and attacker KV blocks.

    After eviction, the cache is effectively empty. The caller must then
    re-seed the target victim before launching reconstruction.

    Parameters
    ----------
    confirm : bool
        If True, poll Prometheus after eviction and verify hit rate ≈ 0%.
        Log a warning if rate is unexpectedly high (eviction may have failed).

    Returns a dict with eviction statistics for inclusion in results JSON.
    """
    hit_rate_before = _get_cache_hit_rate(base_url) if confirm else None
    t0 = time.perf_counter()

    for i in range(n_requests):
        prompt = _make_eviction_prompt(token_length)
        try:
            client.completions.create(
                model      = MODEL_ID,
                prompt     = prompt,
                max_tokens = 1,
                temperature= 0.0,
            )
        except Exception as exc:
            print(f"[eviction] WARNING: request {i} failed: {exc}")

    elapsed_s = time.perf_counter() - t0

    hit_rate_after    = _get_cache_hit_rate(base_url) if confirm else None
    eviction_confirmed = (
        hit_rate_after is not None and hit_rate_after < 5.0
    )

    if confirm:
        if hit_rate_after is not None:
            status = "✓" if eviction_confirmed else "⚠"
            print(f"[eviction] {status} Cache hit rate after eviction: "
                  f"{hit_rate_after:.1f}%  "
                  f"(was {hit_rate_before:.1f}%)")
            if not eviction_confirmed:
                print("[eviction] WARNING: hit rate still high — "
                      "increase EVICT_REQUESTS or EVICT_TOKENS.")
        else:
            print(f"[eviction] Done ({n_requests} requests, "
                  f"{elapsed_s:.1f} s). "
                  f"Prometheus not available — cannot confirm eviction.")

    return {
        "n_eviction_requests" : n_requests,
        "token_length"        : token_length,
        "blocks_estimated"    : n_requests * (token_length // 16),
        "elapsed_s"           : round(elapsed_s, 2),
        "hit_rate_before_pct" : hit_rate_before,
        "hit_rate_after_pct"  : hit_rate_after,
        "eviction_confirmed"  : eviction_confirmed,
    }
