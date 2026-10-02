

from __future__ import annotations

import re

import numpy as np

from kv_attack.backends.base import BackendClient, BackendInfo


# ── Sentinel patterns (must match two_stage_victim_seeder) ────────────────────
_NAME_RE   = re.compile(r"\|\|NAME_SENTINEL::([^|]+)\|\|")
_COND_RE   = re.compile(r"\|\|COND_SENTINEL::([^|]+)\|\|")
_VICTIM_RE = re.compile(r"\|\|VICTIM::(\d+)\|\|")


class TwoStageMockBackend(BackendClient):
    """
    In-process two-stage-aware timing oracle.

    Parameters
    ----------
    hit_ttft_ms    : TTFT for full cache hit  (default: 87.6 ms)
    s1_hit_ttft_ms : TTFT for Stage-1 hit only  (default: 264.0 ms)
    miss_ttft_ms   : TTFT for full miss  (default: 576.1 ms)
    noise_std_ms   : Gaussian noise (default: 4.7 ms)
    apc_enabled    : If False, all probes return miss  (mitigation simulation)
    seed           : RNG seed
    """

    def __init__(
        self,
        hit_ttft_ms    : float = 87.6,
        s1_hit_ttft_ms : float = 264.0,
        miss_ttft_ms   : float = 576.1,
        noise_std_ms   : float = 4.7,
        apc_enabled    : bool  = True,
        seed           : int   = 0,
    ):
        self.hit_ttft_ms    = hit_ttft_ms
        self.s1_hit_ttft_ms = s1_hit_ttft_ms
        self.miss_ttft_ms   = miss_ttft_ms
        self.noise_std_ms   = noise_std_ms
        self.apc_enabled    = apc_enabled
        self._rng           = np.random.default_rng(seed)
        # victim_id → {"name": str, "condition": str}
        self._victims: dict[int, dict[str, str]] = {}

    # ── Victim seeding ────────────────────────────────────────────────────────

    def seed_victim(self, victim_id: int, name: str, condition: str) -> None:
        """Register a victim so their blocks appear in the mock cache."""
        self._victims[victim_id] = {"name": name, "condition": condition}

    def clear_victim(self, victim_id: int) -> None:
        """Evict a victim's blocks from the mock cache."""
        self._victims.pop(victim_id, None)

    # ── BackendClient interface ───────────────────────────────────────────────

    def health_check(self) -> bool:
        return True

    def get_info(self) -> BackendInfo:
        return BackendInfo(
            backend_name  = "two_stage_mock",
            framework     = "mock",
            framework_ver = "2.0",
            model_id      = "mock-deepseek-r1-distill-llama-8b",
            base_url      = "mock://localhost",
            apc_enabled   = self.apc_enabled,
            extra         = {
                "hit_ttft_ms"    : self.hit_ttft_ms,
                "s1_hit_ttft_ms" : self.s1_hit_ttft_ms,
                "miss_ttft_ms"   : self.miss_ttft_ms,
                "noise_std_ms"   : self.noise_std_ms,
                "n_victims"      : len(self._victims),
            },
        )

    def _send_prompt(self, prompt: str, tenant_id: int = 0) -> float:
        """
        Classify the probe and return a sampled TTFT.

        Classification priority:
          1. APC disabled → always miss
          2. Sentinel extraction → check victim registry
          3. No sentinels → treat as miss (calibration / filler prompts)
        """
        if not self.apc_enabled:
            return self._sample(self.miss_ttft_ms)

        name_m   = _NAME_RE.search(prompt)
        cond_m   = _COND_RE.search(prompt)
        victim_m = _VICTIM_RE.search(prompt)

        if name_m is None:
            # calibration or eviction filler — always miss
            return self._sample(self.miss_ttft_ms)

        probe_name = name_m.group(1).strip()

        # Find the active victim for this probe
        if victim_m is not None:
            vid = int(victim_m.group(1))
            victim = self._victims.get(vid)
        else:
            # Fall back to checking all cached victims
            victim = next(iter(self._victims.values()), None) if len(self._victims) == 1 else None

        if victim is None:
            return self._sample(self.miss_ttft_ms)

        name_hit = (probe_name == victim["name"])

        if not name_hit:
            return self._sample(self.miss_ttft_ms)

        # Name matched — Stage 1 hit. Now check condition.
        probe_cond = cond_m.group(1).strip() if cond_m else None
        if probe_cond is None or probe_cond == "__DUMMY__":
            return self._sample(self.s1_hit_ttft_ms)

        cond_hit = (probe_cond == victim["condition"])
        if cond_hit:
            return self._sample(self.hit_ttft_ms)
        else:
            return self._sample(self.s1_hit_ttft_ms)

    def _sample(self, mean_ms: float) -> float:
        val = float(self._rng.normal(mean_ms, self.noise_std_ms))
        return max(val, 1.0)
