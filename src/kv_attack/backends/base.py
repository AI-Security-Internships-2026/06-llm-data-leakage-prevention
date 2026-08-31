"""
kv_attack.backends.base
=======================
Abstract base class and shared data structures for inference backends.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class BackendInfo:
    """
    Static information about a backend / running server instance.

    Populated by BackendClient.get_info() and embedded in every results JSON
    so the reader knows exactly what was running during the experiment.
    """
    backend_name   : str           # "vllm" | "tgi" | "mock"
    framework      : str           # e.g. "vllm", "text-generation-inference"
    framework_ver  : str           # e.g. "0.27.1", "2.1.4"
    model_id       : str           # HuggingFace model string
    base_url       : str           # server endpoint
    apc_enabled    : bool          # Automatic Prefix Caching enabled?
    extra          : dict          # backend-specific key-value pairs


class BackendClient(abc.ABC):
    """
    Protocol that every backend adapter must satisfy.

    All timing methods return milliseconds (float).
    """

    # ── Abstract interface ─────────────────────────────────────────────────────

    @abc.abstractmethod
    def health_check(self) -> bool:
        """Return True if the server is reachable and responsive."""

    @abc.abstractmethod
    def get_info(self) -> BackendInfo:
        """Return static metadata about this backend instance."""

    @abc.abstractmethod
    def _send_prompt(self, prompt: str) -> float:
        """
        Send *prompt* to the server and return wall-clock TTFT in milliseconds.

        This is the hot path — it must be as lean as possible.
        Subclasses implement the transport-specific request here.
        """

    # ── Concrete timing helpers ────────────────────────────────────────────────

    def measure_ttft(self, prompt: str) -> float:
        """Single TTFT measurement (ms). Delegates to _send_prompt."""
        return self._send_prompt(prompt)

    def measure_ttft_repeated(self, prompt: str, n: int) -> np.ndarray:
        """Return n TTFT measurements (ms) for the same prompt."""
        return np.array([self._send_prompt(prompt) for _ in range(n)])

    def measure_mean_ttft(self, prompt: str, n: int = 1) -> float:
        """Return mean TTFT (ms) over n measurements."""
        return float(self.measure_ttft_repeated(prompt, n).mean())

    def is_cache_hit(
        self,
        prompt       : str,
        threshold_ms : float,
        n            : int = 1,
    ) -> tuple[bool, float]:
        """
        Classify prompt as cache HIT or MISS.

        Returns (is_hit, mean_ttft_ms).
        """
        mean_ttft = self.measure_mean_ttft(prompt, n)
        return mean_ttft < threshold_ms, mean_ttft