

from __future__ import annotations

import logging
import random
import time
import uuid
from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.stats

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_HOST       = "localhost"
DEFAULT_PORT       = 8001
DEFAULT_N_REPEATS = 30        # probe repetitions to reduce TTFT variance
DEFAULT_N_SAMPLES  = 100       # calibration samples per distribution
KS_ALPHA           = 1e-8      # Bonferroni-corrected significance (Paper 14)
MIN_GAP_MS = 1.5      # reject calibration if gap < 10 ms (Proposal §3.5)


# ── TTFT measurement ───────────────────────────────────────────────────────────

def measure_ttft(client, prompt: str, model: str) -> float:
    """
    Measure wall-clock Time-To-First-Token (ms) for *prompt* via the
    streaming completions API.

    The timer starts immediately before the API call and stops on receipt
    of the first streamed chunk (= first token generated). This matches
    the attacker's observable signal: only API latency, no server internals.

    Parameters
    ----------
    client : OpenAI-compatible client
    prompt : the full prompt string to send
    model  : vLLM model ID

    Returns
    -------
    float — TTFT in milliseconds
    """
    t0 = time.perf_counter()
    stream = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=1,      # only first token matters; stop immediately
        temperature=0.0,
        stream=True,
    )
    # Consume exactly one chunk (= first token)
    for _ in stream:
        break
    elapsed_ms = (time.perf_counter() - t0) * 1_000
    return elapsed_ms


def measure_ttft_repeated(
    client,
    prompt:    str,
    model:     str,
    n_repeats: int = DEFAULT_N_REPEATS,
) -> tuple[float, float]:
    """
    Measure TTFT *n_repeats* times and return (mean_ms, std_ms).

    Repeated measurement is essential: a single TTFT observation has high
    variance due to OS scheduling, network jitter, and GPU load. Paper 11
    uses 10 repetitions; we follow that protocol.
    """
    samples = [measure_ttft(client, prompt, model) for _ in range(n_repeats)]
    return float(np.mean(samples)), float(np.std(samples))


# ── Threshold calibration ──────────────────────────────────────────────────────

@dataclass
class CalibrationResult:
    """Result of the offline hit/miss threshold calibration."""
    threshold_ms:   float    # decision boundary (ms)
    hit_mean_ms:    float
    hit_std_ms:     float
    miss_mean_ms:   float
    miss_std_ms:    float
    timing_gap_ms:  float    # miss_mean - hit_mean
    ks_statistic:   float
    ks_p_value:     float
    passed:         bool     # True if gap >= MIN_GAP_MS and p < KS_ALPHA


def calibrate_threshold(
    client,
    cached_prompt:  str,          # a prompt we know is in the cache (hit ground truth)
    model:          str,
    n_samples:      int = DEFAULT_N_SAMPLES,
) -> CalibrationResult:
    """
    Offline calibration phase: measure TTFT distributions for known hits
    and misses, run a two-sample KS-test, and set the midpoint threshold.

    Protocol (Paper 14 / Gu et al. 2025):
      1. Send *cached_prompt* n_samples times → hit distribution
      2. Send n_samples unique random prompts → miss distribution
      3. KS-test at α=1e-8 confirms separability
      4. Threshold = midpoint of the two means

    Parameters
    ----------
    client        : OpenAI-compatible client
    cached_prompt : a prompt already in the vLLM KV cache (hit oracle)
    model         : vLLM model ID
    n_samples     : samples per distribution (default 100)

    Returns
    -------
    CalibrationResult — includes threshold, gap stats, and KS-test result.
    Raises AssertionError if the timing gap is not statistically significant.
    """
    logger.info("Calibrating hit/miss threshold (%d samples each) ...", n_samples)

    # Hit distribution: send the known-cached prompt repeatedly
    hit_ttfts: list[float] = []
    for i in range(n_samples):
        ttft = measure_ttft(client, cached_prompt, model)
        hit_ttfts.append(ttft)
        if (i + 1) % 20 == 0:
            logger.debug("  Hit sample %d/%d: %.1f ms", i + 1, n_samples, ttft)

    # Miss distribution: send unique random prompts (guaranteed cache misses)
    miss_ttfts: list[float] = []
    for i in range(n_samples):
        noise = f"[CAL-MISS-{uuid.uuid4().hex[:8]}] " + " ".join(
            random.choices(
                ["alpha", "beta", "gamma", "delta", "epsilon", "zeta",
                 "theta", "iota", "kappa", "lambda", "mu", "nu", "xi"],
                k=80,
            )
        )
        ttft = measure_ttft(client, noise, model)
        miss_ttfts.append(ttft)
        if (i + 1) % 20 == 0:
            logger.debug("  Miss sample %d/%d: %.1f ms", i + 1, n_samples, ttft)

    # Two-sample KS-test
    ks_stat, ks_p = scipy.stats.ks_2samp(hit_ttfts, miss_ttfts)

    hit_mean  = float(np.mean(hit_ttfts))
    hit_std   = float(np.std(hit_ttfts))
    miss_mean = float(np.mean(miss_ttfts))
    miss_std  = float(np.std(miss_ttfts))
    gap_ms    = miss_mean - hit_mean
    threshold = (hit_mean + miss_mean) / 2.0

    passed = (ks_p < KS_ALPHA) and (gap_ms >= MIN_GAP_MS)

    result = CalibrationResult(
        threshold_ms  = threshold,
        hit_mean_ms   = hit_mean,
        hit_std_ms    = hit_std,
        miss_mean_ms  = miss_mean,
        miss_std_ms   = miss_std,
        timing_gap_ms = gap_ms,
        ks_statistic  = float(ks_stat),
        ks_p_value    = float(ks_p),
        passed        = passed,
    )

    logger.info(
        "Calibration: hit=%.1f±%.1f ms | miss=%.1f±%.1f ms | gap=%.1f ms | "
        "KS p=%.2e | passed=%s",
        hit_mean, hit_std, miss_mean, miss_std, gap_ms, ks_p, passed,
    )

    if not passed:
        raise AssertionError(
            f"Timing gap not significant or too small: gap={gap_ms:.1f} ms, "
            f"KS p={ks_p:.2e}. "
            "Check that vLLM APC is enabled (--enable-prefix-caching) and "
            "cache_salt is NOT set."
        )

    return result


# ── Cache-hit oracle ───────────────────────────────────────────────────────────

class CacheHitOracle:
    """
    Stateful wrapper around the timing measurements that classifies
    individual probes as cache hits or misses.

    The oracle is calibrated once offline (calibrate_threshold) and then
    used repeatedly by the reconstructor for every candidate probe.
    """

    def __init__(
        self,
        client,
        model:         str,
        threshold_ms:  float,
        n_repeats:     int = DEFAULT_N_REPEATS,
    ) -> None:
        self.client       = client
        self.model        = model
        self.threshold_ms = threshold_ms
        self.n_repeats    = n_repeats
        self._total_calls = 0

    @property
    def total_api_calls(self) -> int:
        return self._total_calls

    def query(self, prompt: str) -> tuple[bool, float, float]:
        """
        Send *prompt* to vLLM, measure TTFT n_repeats times, and classify.

        Returns
        -------
        (is_hit, mean_ttft_ms, std_ttft_ms)
        """
        ttfts = [measure_ttft(self.client, prompt, self.model)
                 for _ in range(self.n_repeats)]
        self._total_calls += self.n_repeats

        mean_ms = float(np.mean(ttfts))
        std_ms  = float(np.std(ttfts))
        is_hit  = mean_ms < self.threshold_ms

        return is_hit, mean_ms, std_ms

    def reset_call_counter(self) -> None:
        self._total_calls = 0


# ── Mock client (no GPU required) ─────────────────────────────────────────────

class _MockCompletion:
    """Minimal stand-in for openai.types.Completion."""
    class _Choice:
        text = ""
        finish_reason = "length"
    choices = [_Choice()]


class _MockStream:
    """Minimal stand-in for a streaming completions response."""
    def __iter__(self):
        yield _MockCompletion()


class MockVLLMClient:
    """
    Simulates a vLLM OpenAI-compatible completions endpoint without a GPU.

    Behaviour
    ---------
    - Maintains an LRU cache of "cached" prompt strings with a fixed
      max_capacity (default 3). This simulates limited GPU memory: when
      capacity is exceeded (e.g. after 20 eviction requests each filling
      ~200 tokens), the oldest victim entries are evicted — matching
      vLLM's LRU eviction policy.
    - When queried with a prompt that *starts with* a cached victim prompt,
      returns a simulated hit TTFT: N(hit_mean_ms, hit_std_ms).
    - Otherwise returns a miss TTFT: N(miss_mean_ms, miss_std_ms).

    Attack semantics:
      The harness seeds ONE victim at a time (per-victim seed → attack →
      evict cycle). The cache holds that victim's prompt plus calibration
      entries. Eviction requests (20 × UUID-prefixed noise) push old
      victim entries past the LRU limit, clearing the way for the next
      victim — faithfully mimicking real vLLM GPU memory pressure.

    Default timing parameters reflect RTX 4090 measurements (Proposal §5):
      Hit  : mean=15 ms, std=3 ms
      Miss : mean=75 ms, std=8 ms
      Gap  : ~60 ms (safely above the 10 ms MIN_GAP_MS threshold)
    """

    def __init__(
        self,
        hit_mean_ms:  float = 15.0,
        hit_std_ms:   float = 3.0,
        miss_mean_ms: float = 75.0,
        miss_std_ms:  float = 8.0,
        max_capacity: int   = 3,    # max victim-sized entries before LRU evicts
        seed:         int   = 0,
    ) -> None:
        # OrderedDict used as an LRU cache (most-recent at end)
        from collections import OrderedDict
        self._cache:        "OrderedDict[str, None]" = OrderedDict()
        self._max_capacity  = max_capacity
        self._hit_mean_ms   = hit_mean_ms
        self._hit_std_ms    = hit_std_ms
        self._miss_mean_ms  = miss_mean_ms
        self._miss_std_ms   = miss_std_ms
        self._rng           = np.random.default_rng(seed)
        self.completions    = self  # mimic openai.Client().completions

    # ── Internal cache management ──────────────────────────────────────────────

    def seed_prompt(self, prompt: str) -> None:
        """
        Add *prompt* to the simulated KV cache (LRU, most-recent at end).
        When capacity is exceeded, evict the oldest entry — mimicking
        vLLM's LRU eviction of low-ref-count KV blocks.
        """
        if prompt in self._cache:
            self._cache.move_to_end(prompt)
        else:
            self._cache[prompt] = None
            if len(self._cache) > self._max_capacity:
                self._cache.popitem(last=False)   # evict oldest

    def evict_all(self) -> None:
        """Clear the entire simulated KV cache."""
        self._cache.clear()

    def _is_cached(self, prompt: str) -> bool:
        """
        True if the probe *prompt* shares a meaningful prefix with any
        currently-cached victim prompt.

        Mimics vLLM block-level hash matching: a cache hit occurs when the
        probe's token sequence matches a cached sequence for at least one
        full 16-token block beyond the shared system preamble (~64 chars).
        We use character-level prefix comparison as an approximation.
        """
        BLOCK_CHAR_PROXY = 64   # ~16 tokens × ~4 chars/token
        for cached in self._cache:
            # Only consider prompts that are genuine victim prompts (longer
            # than a noise eviction request). Noise prompts start with [EVICT-
            if cached.startswith("[EVICT-") or cached.startswith("[CAL-MISS-"):
                continue
            min_len = min(len(cached), len(prompt))
            if (
                min_len > BLOCK_CHAR_PROXY
                and prompt[:min_len] == cached[:min_len]
            ):
                return True
        return False

    # ── completions.create interface ───────────────────────────────────────────

    def create(
        self,
        model:       str,
        prompt:      str,
        max_tokens:  int   = 1,
        temperature: float = 0.0,
        stream:      bool  = False,
        **kwargs,
    ):
        """
        Simulate a vLLM completions call.
        - stream=False (seeding / eviction): add prompt to LRU cache.
        - stream=True  (attacker probe): simulate TTFT based on hit/miss.
        """
        if not stream:
            self.seed_prompt(prompt)
            return _MockCompletion()

        is_hit = self._is_cached(prompt)
        if is_hit:
            delay_ms = self._rng.normal(self._hit_mean_ms, self._hit_std_ms)
        else:
            delay_ms = self._rng.normal(self._miss_mean_ms, self._miss_std_ms)

        delay_ms = max(1.0, delay_ms)
        time.sleep(delay_ms / 1_000)
        return _MockStream()
