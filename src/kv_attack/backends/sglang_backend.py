from __future__ import annotations

import json
import time
import urllib.request

from openai import OpenAI

from kv_attack.backends.base import BackendClient, BackendInfo


class SGLangBackend(BackendClient):
    """
    SGLang OpenAI-compatible endpoint.

    Parameters
    ----------
    base_url : str
        e.g. "http://localhost:8002/v1"
    model_id : str
        HuggingFace model string served by this instance.
    """

    FRAMEWORK = "sglang"

    def __init__(self, base_url: str, model_id: str):
        self.base_url = base_url
        self.model_id = model_id
        # Explicit timeout — without this, a stalled/never-closing stream
        # can hang the client indefinitely with no error (observed on
        # SGLang 0.5.19 under certain repeated-identical-prompt conditions).
        self._client  = OpenAI(base_url=base_url, api_key="EMPTY", timeout=30.0, max_retries=1)
        self._version = self._detect_version()

    # ── BackendClient interface ───────────────────────────────────────────────

    def health_check(self) -> bool:
        """Return True if SGLang server is reachable and healthy."""
        try:
            url = self.base_url.rstrip("/v1").rstrip("/") + "/health"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def get_info(self) -> BackendInfo:
        """Query /get_server_info for SGLang-specific metadata."""
        extra: dict = {}
        try:
            base = self.base_url.rstrip("/v1").rstrip("/")
            req  = urllib.request.Request(f"{base}/get_server_info")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                extra["sglang_info"] = data
                extra["prefix_cache"] = data.get("prefix_caching", "unknown")
                extra["cache_hit_rate"] = data.get("cache_hit_rate", None)
        except Exception as exc:
            extra["get_server_info_error"] = str(exc)

        return BackendInfo(
            backend_name  = "sglang",
            framework     = self.FRAMEWORK,
            framework_ver = self._version,
            model_id      = self.model_id,
            base_url      = self.base_url,
            apc_enabled   = True,   # SGLang RadixAttention is always on
            extra         = extra,
        )

    def _send_prompt(self, prompt: str, tenant_id: int = 0) -> float:
        """
        Send prompt and return TTFT in ms using a NON-streaming request.

        NOTE: an earlier version used stream=True and read only the first
        chunk. On SGLang 0.5.19 this was observed to hang indefinitely
        (no error, no timeout) under certain repeated-identical-prompt
        conditions — likely an edge case in how SGLang's radix-tree cache
        interacts with streamed responses for exact-duplicate prompts.
        Non-streaming + an explicit client-level timeout (set in __init__)
        avoids that hang entirely: TTFT is measured as wall-clock time to
        the single returned token, which for max_tokens=1 is equivalent to
        prefill time + one decode step (same latency profile that the
        cache-hit/cache-miss signal depends on).
        """
        t0 = time.perf_counter()
        try:
            resp = self._client.completions.create(
                model       = self.model_id,
                prompt      = prompt,
                max_tokens  = 1,
                temperature = 0.0,
                stream      = False,
            )
        except Exception as exc:
            # Timeout or transport error — return a large sentinel value so
            # calibration/attack code treats this as an unambiguous "miss"
            # rather than crashing the whole run.
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[sglang_backend] WARNING: request failed after "
                  f"{elapsed_ms:.0f}ms ({exc!r}); treating as miss (9999ms)")
            return 9999.0
        return (time.perf_counter() - t0) * 1000.0

    # ── Internal ──────────────────────────────────────────────────────────────

    def _detect_version(self) -> str:
        """Try to detect SGLang version from /get_server_info."""
        try:
            base = self.base_url.rstrip("/v1").rstrip("/")
            req  = urllib.request.Request(f"{base}/get_server_info")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                return data.get("version", "unknown")
        except Exception:
            return "unknown"


def make_sglang_backend(base_url: str, model_id: str) -> SGLangBackend:
    """Factory with health check — raises SystemExit if server unreachable."""
    backend = SGLangBackend(base_url=base_url, model_id=model_id)
    if not backend.health_check():
        raise SystemExit(
            f"[sglang_backend] SGLang not reachable at {base_url}\n"
            f"Start it with:\n"
            f"  python -m sglang.launch_server \\\n"
            f"      --model-path {model_id} \\\n"
            f"      --port 8002 \\\n"
            f"      --mem-fraction-static 0.85"
        )
    info = backend.get_info()
    print(f"[sglang_backend] Connected: {info.framework_ver} / {info.model_id}")
    return backend