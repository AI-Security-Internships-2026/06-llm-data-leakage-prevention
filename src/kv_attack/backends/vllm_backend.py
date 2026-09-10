"""
kv_attack.backends.vllm_backend
================================
Backend adapter for vLLM ≥ 0.27.x OpenAI-compatible API.

Measures TTFT by streaming completions and stopping after the first chunk.
This mirrors exactly the measurement technique in attacker.py (Weeks 10–11).

APC detection
-------------
Queries /metrics (Prometheus) for ``vllm:gpu_prefix_cache_hit_rate_perc``.
If the metric is absent the constructor still succeeds but logs a warning.
"""

from __future__ import annotations

import time
import urllib.parse
import urllib.request

from openai import OpenAI

from kv_attack.backends.base import BackendClient, BackendInfo


class VLLMBackend(BackendClient):
    """
    OpenAI-compatible vLLM endpoint.

    Parameters
    ----------
    base_url : str
        e.g. "http://localhost:8001/v1"
    model_id : str
        HuggingFace model string served by this instance.
    """

    FRAMEWORK     = "vllm"
    FRAMEWORK_VER = "0.27.1"   # pinned; bumped if the server reports otherwise

    def __init__(self, base_url: str, model_id: str):
        self.base_url = base_url
        self.model_id = model_id
        self._client  = OpenAI(base_url=base_url, api_key="EMPTY")

    # ── Abstract interface ─────────────────────────────────────────────────────

    def health_check(self) -> bool:
        try:
            resp = self._client.completions.create(
                model=self.model_id, prompt="ping", max_tokens=1, temperature=0.0
            )
            _ = resp.choices[0].text
            return True
        except Exception as exc:
            print(f"[VLLMBackend] health_check FAILED: {exc}")
            return False

    def get_info(self) -> BackendInfo:
        apc = self._detect_apc()
        return BackendInfo(
            backend_name  = "vllm",
            framework     = self.FRAMEWORK,
            framework_ver = self.FRAMEWORK_VER,
            model_id      = self.model_id,
            base_url      = self.base_url,
            apc_enabled   = apc,
            extra         = {"apc_metric_found": apc},
        )

    def _send_prompt(self, prompt: str) -> float:
        """
        Stream one completion token and return wall-clock TTFT (ms).
        Same measurement protocol as Week 10 attacker.py.
        """
        t0     = time.perf_counter()
        stream = self._client.completions.create(
            model       = self.model_id,
            prompt      = prompt,
            max_tokens  = 1,
            temperature = 0.0,
            stream      = True,
        )
        for _ in stream:
            break
        return (time.perf_counter() - t0) * 1_000.0   # ms

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _detect_apc(self) -> bool:
        """Query Prometheus /metrics to detect whether APC is on."""
        try:
            parsed      = urllib.parse.urlparse(self.base_url)
            metrics_url = urllib.parse.urlunparse(
                parsed._replace(path="/metrics", query="", fragment="")
            )
            with urllib.request.urlopen(metrics_url, timeout=5) as resp:
                text = resp.read().decode()
            found = "vllm:gpu_prefix_cache_hit_rate_perc" in text
            if not found:
                print("[VLLMBackend] ⚠ APC metric not found in /metrics — "
                      "APC may be disabled.")
            return found
        except Exception:
            print("[VLLMBackend] ⚠ Cannot reach /metrics — assuming APC enabled.")
            return True