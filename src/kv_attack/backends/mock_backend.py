"""
kv_attack.backends.mock_backend
================================
Fully in-process, deterministic mock backend.

Purpose
-------
- Unit tests and offline CI — no GPU or network required.
- Reproducing exact experimental conditions with a fixed RNG seed.
- Rapid prototyping of new attack / mitigation variants.

Behaviour
---------
The mock caches prompt hashes in a Python set. On the first ``_send_prompt``
call for a prompt, the mock inserts it into the "cache" and returns a sample
from the miss distribution. On subsequent calls with the same prompt, it
returns a sample from the hit distribution.

Both distributions are Gaussian, parameterised to match the Week 10 empirical
results (hit_mean=87.6 ms, miss_mean=576.1 ms, noise_std ≈ 4 ms) by default.

APC can be disabled by setting ``apc_enabled=False``, in which case ALL
prompts return miss-distribution samples (simulating --no-enable-prefix-caching).

Multi-tenant isolation
----------------------
Pass ``tenant_isolation=True`` to simulate a server where each tenant's
cached blocks are invisible to other tenants (CacheSolidarity / PrefixWall
semantics). In this mode the cache key is ``(prompt_hash, tenant_id)`` so
an attacker's probe of a victim's prompt never returns a hit.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from kv_attack.backends.base import BackendClient, BackendInfo


class MockBackend(BackendClient):
    """
    Deterministic in-process timing oracle simulator.

    Parameters
    ----------
    hit_ttft_ms     : float  Mean TTFT for a cache hit  (default: 87.6 ms)
    miss_ttft_ms    : float  Mean TTFT for a cache miss (default: 576.1 ms)
    noise_std_ms    : float  Gaussian noise std for both distributions (4.7 ms)
    apc_enabled     : bool   If False, all calls return miss distribution
    tenant_isolation: bool   If True, cross-tenant hits return miss distribution
    seed            : int    RNG seed for reproducibility
    """

    def __init__(
        self,
        hit_ttft_ms     : float = 87.6,
        miss_ttft_ms    : float = 576.1,
        noise_std_ms    : float = 4.7,
        apc_enabled     : bool  = True,
        tenant_isolation: bool  = False,
        seed            : int   = 0,
    ):
        self.hit_ttft_ms      = hit_ttft_ms
        self.miss_ttft_ms     = miss_ttft_ms
        self.noise_std_ms     = noise_std_ms
        self.apc_enabled      = apc_enabled
        self.tenant_isolation = tenant_isolation
        self._rng             = np.random.default_rng(seed)
        # cache: set of (prompt_hash,) or (prompt_hash, tenant_id)
        self._cache: set[tuple] = set()
        # request counters for stats
        self.n_hits   = 0
        self.n_misses = 0

    # ── Abstract interface ─────────────────────────────────────────────────────

    def health_check(self) -> bool:
        return True   # always healthy

    def get_info(self) -> BackendInfo:
        return BackendInfo(
            backend_name  = "mock",
            framework     = "mock",
            framework_ver = "1.0",
            model_id      = "mock-llama-3.1-8b",
            base_url      = "mock://localhost",
            apc_enabled   = self.apc_enabled,
            extra         = {
                "hit_ttft_ms"      : self.hit_ttft_ms,
                "miss_ttft_ms"     : self.miss_ttft_ms,
                "noise_std_ms"     : self.noise_std_ms,
                "tenant_isolation" : self.tenant_isolation,
            },
        )

    def _send_prompt(self, prompt: str, tenant_id: int = 0) -> float:
        """
        Return a simulated TTFT (ms) for the given prompt.

        Cache logic:
          - apc_enabled=False  → always miss
          - apc_enabled=True, tenant_isolation=False → shared cache (vulnerable)
          - apc_enabled=True, tenant_isolation=True  → per-tenant cache
        """
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        cache_key   = (prompt_hash,) if not self.tenant_isolation \
                      else (prompt_hash, tenant_id)

        if self.apc_enabled and cache_key in self._cache:
            self.n_hits += 1
            return float(
                self._rng.normal(self.hit_ttft_ms, self.noise_std_ms)
            )
        else:
            self.n_misses += 1
            # Insert into cache (simulate vLLM caching the computed KV blocks)
            if self.apc_enabled:
                self._cache.add(cache_key)
            return float(
                self._rng.normal(self.miss_ttft_ms, self.noise_std_ms)
            )

    # ── Extra helpers for testing ──────────────────────────────────────────────

    def seed_prompt(self, prompt: str, tenant_id: int = 0) -> None:
        """Manually insert a prompt into the mock cache (simulates victim seeding)."""
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        cache_key   = (prompt_hash,) if not self.tenant_isolation \
                      else (prompt_hash, tenant_id)
        self._cache.add(cache_key)

    def reset_cache(self) -> None:
        """Evict all blocks (simulates cache_eviction.evict_cache)."""
        self._cache.clear()
        self.n_hits   = 0
        self.n_misses = 0

    def stats(self) -> dict:
        return {
            "n_hits"   : self.n_hits,
            "n_misses" : self.n_misses,
            "hit_rate" : self.n_hits / max(1, self.n_hits + self.n_misses),
        }