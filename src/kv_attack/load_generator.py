"""
kv_attack.load_generator
=========================
Issue #5 — Background tenant simulator for multi-tenant load testing.

Spawns N background threads that continuously send inference requests to
the vLLM server, simulating co-tenant activity. This increases cache
pressure and introduces timing noise into TTFT measurements.

Load levels (adapt exact thread counts to hardware):
  L0 — 0 background tenants  (zero-load baseline)
  L1 — ~2 background tenants
  L2 — ~5 background tenants
  L3 — ~10 background tenants
  L4 — highest stable load (determined empirically)

Usage
-----
    from kv_attack.load_generator import LoadGenerator

    gen = LoadGenerator(base_url="http://localhost:8001/v1",
                        model_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                        n_tenants=5, seed=0)
    gen.start()          # background threads start
    # ... run your attack ...
    stats = gen.stop()   # returns throughput stats
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field


_BG_PROMPTS = [
    "What is the capital of France?",
    "Explain the water cycle in two sentences.",
    "List three benefits of exercise.",
    "What is machine learning?",
    "How does photosynthesis work?",
    "Name five programming languages.",
    "What causes thunder?",
    "Describe the water treatment process.",
    "What is the speed of light?",
    "How many planets are in the solar system?",
    "What is a neural network?",
    "Explain supply and demand briefly.",
    "What is the Pythagorean theorem?",
    "What is DNA?",
    "Name three renewable energy sources.",
    "What is inflation?",
    "How does GPS work?",
    "What is blockchain?",
    "Explain what a vaccine does.",
    "What is the purpose of a firewall?",
]


@dataclass
class TenantStats:
    """Statistics collected from one background tenant thread."""
    tenant_id       : int
    n_requests      : int   = 0
    n_errors        : int   = 0
    total_elapsed_s : float = 0.0
    start_time      : float = field(default_factory=time.time)

    @property
    def throughput_rps(self) -> float:
        elapsed = time.time() - self.start_time
        return self.n_requests / max(elapsed, 1e-6)


class LoadGenerator:
    """
    Spawns background tenant threads that continuously send inference
    requests to the vLLM server.

    Parameters
    ----------
    base_url   : vLLM OpenAI-compatible endpoint
    model_id   : model to query
    n_tenants  : number of concurrent background threads
    seed       : RNG seed for prompt selection
    max_tokens : tokens per background request (keep small for speed)
    rps_cap    : max requests/sec per tenant (None = unlimited)
    """

    def __init__(
        self,
        base_url   : str,
        model_id   : str,
        n_tenants  : int  = 0,
        seed       : int  = 0,
        max_tokens : int  = 8,
        rps_cap    : float | None = None,
    ):
        self.base_url   = base_url
        self.model_id   = model_id
        self.n_tenants  = n_tenants
        self.seed       = seed
        self.max_tokens = max_tokens
        self.rps_cap    = rps_cap

        self._stop_event = threading.Event()
        self._threads   : list[threading.Thread] = []
        self._stats     : list[TenantStats]      = []
        self._lock       = threading.Lock()


    def start(self) -> None:
        """Start all background tenant threads."""
        if self.n_tenants == 0:
            return
        self._stop_event.clear()
        for i in range(self.n_tenants):
            stats = TenantStats(tenant_id=i)
            self._stats.append(stats)
            t = threading.Thread(
                target=self._tenant_loop,
                args=(i, stats),
                daemon=True,
                name=f"bg-tenant-{i}",
            )
            self._threads.append(t)
            t.start()

    def stop(self) -> dict:
        """Stop all threads and return aggregate statistics."""
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=5.0)
        self._threads.clear()
        return self._aggregate_stats()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()


    def _tenant_loop(self, tenant_id: int, stats: TenantStats) -> None:
        """Continuously send background inference requests until stopped."""
        try:
            from openai import OpenAI
            client = OpenAI(base_url=self.base_url, api_key="EMPTY")
        except ImportError:
            return

        rng = random.Random(self.seed + tenant_id * 1000)
        min_interval = 1.0 / self.rps_cap if self.rps_cap else 0.0

        while not self._stop_event.is_set():
            t0     = time.perf_counter()
            prompt = rng.choice(_BG_PROMPTS)
            try:
                client.completions.create(
                    model      = self.model_id,
                    prompt     = prompt,
                    max_tokens = self.max_tokens,
                    temperature= 0.0,
                    stream     = False,
                )
                with self._lock:
                    stats.n_requests      += 1
                    stats.total_elapsed_s += time.perf_counter() - t0
            except Exception:
                with self._lock:
                    stats.n_errors += 1

            elapsed = time.perf_counter() - t0
            if min_interval > elapsed:
                time.sleep(min_interval - elapsed)

    def _aggregate_stats(self) -> dict:
        total_req  = sum(s.n_requests for s in self._stats)
        total_err  = sum(s.n_errors   for s in self._stats)
        total_time = sum(s.total_elapsed_s for s in self._stats)
        return {
            "n_tenants"       : self.n_tenants,
            "total_requests"  : total_req,
            "total_errors"    : total_err,
            "mean_latency_s"  : total_time / max(total_req, 1),
            "aggregate_rps"   : sum(s.throughput_rps for s in self._stats),
            "per_tenant"      : [
                {
                    "tenant_id"    : s.tenant_id,
                    "n_requests"   : s.n_requests,
                    "n_errors"     : s.n_errors,
                    "throughput_rps": s.throughput_rps,
                }
                for s in self._stats
            ],
        }



LOAD_LEVELS = {
    "L0": 0,
    "L1": 2,
    "L2": 5,
    "L3": 10,
    "L4": 20,
}
