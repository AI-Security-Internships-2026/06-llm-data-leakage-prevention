"""
kv_attack.backends.tgi_backend
================================
Backend adapter for HuggingFace Text Generation Inference (TGI) ≥ 2.x.

TGI API differences vs vLLM
----------------------------
1. Endpoint  : POST /generate   (not /v1/completions)
2. Streaming : POST /generate_stream  (Server-Sent Events, not OpenAI chunks)
3. APC       : TGI calls this "prefix caching" — enabled with
               ``--prefix-caching true`` (TGI ≥ 2.0).
4. Metrics   : GET /metrics  (Prometheus, same port as the API)
               metric name: ``tgi_request_mean_time_per_token_duration_seconds``

Attack surface
--------------
TGI's prefix cache uses the same hash-chained block scheme as vLLM (both
derived from paged-attention literature). The timing oracle is structurally
identical: a cached prefix saves prefill computation → lower TTFT; a miss
forces full prefill → higher TTFT. The absolute delta will differ because
TGI's Rust backend has different decode overheads, but the signal exists.

Implementation notes
--------------------
- TTFT is measured via /generate_stream — timer stops at first SSE ``data:``
  line containing a non-empty token.
- ``/generate`` (non-streaming) reports ``details.prefill_ms`` which is more
  precise but only available when ``decoder_input_details=true`` and TGI ≥ 2.1.
  We use /generate_stream for consistency with the vLLM adapter (same wall-clock
  TTFT definition used in Week 10).
- Token budget: ``max_new_tokens=1``, ``temperature=0.0`` (greedy),
  ``do_sample=false``.

Launch command (reference)
--------------------------
    docker run --gpus all --shm-size 1g \\
        -p 8002:80 \\
        ghcr.io/huggingface/text-generation-inference:2.1.4 \\
        --model-id meta-llama/Llama-3.1-8B-Instruct \\
        --prefix-caching true \\
        --max-total-tokens 4096 \\
        --dtype bfloat16
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from http.client import HTTPResponse
from typing import Iterator

from kv_attack.backends.base import BackendClient, BackendInfo


class TGIBackend(BackendClient):
    """
    HuggingFace TGI ≥ 2.x backend.

    Parameters
    ----------
    base_url : str
        Root URL of the TGI server, e.g. "http://localhost:8002".
    model_id : str
        HuggingFace model string (used for metadata only — TGI is single-model).
    """

    FRAMEWORK     = "text-generation-inference"
    FRAMEWORK_VER_FALLBACK = "2.x"

    def __init__(self, base_url: str, model_id: str):
        # Normalize: strip trailing slash
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self._framework_ver: str | None = None   # lazily fetched from /info

    # ── Abstract interface ─────────────────────────────────────────────────────

    def health_check(self) -> bool:
        """GET /health — TGI returns 200 when ready."""
        try:
            url = f"{self.base_url}/health"
            with urllib.request.urlopen(url, timeout=5) as resp:
                return resp.status == 200
        except Exception as exc:
            print(f"[TGIBackend] health_check FAILED: {exc}")
            return False

    def get_info(self) -> BackendInfo:
        ver, apc = self._fetch_server_info()
        return BackendInfo(
            backend_name  = "tgi",
            framework     = self.FRAMEWORK,
            framework_ver = ver,
            model_id      = self.model_id,
            base_url      = self.base_url,
            apc_enabled   = apc,
            extra         = {
                "generate_endpoint": f"{self.base_url}/generate",
                "stream_endpoint"  : f"{self.base_url}/generate_stream",
            },
        )

    def _send_prompt(self, prompt: str) -> float:
        """
        Stream one token from /generate_stream and return wall-clock TTFT (ms).

        SSE wire format (TGI ≥ 2.x):
            data: {"token": {"id": 123, "text": " the", ...}, ...}

        We stop timing at the first data line that contains a non-empty token.
        """
        url     = f"{self.base_url}/generate_stream"
        payload = json.dumps({
            "inputs"     : prompt,
            "parameters" : {
                "max_new_tokens": 1,
                "temperature"   : 0.0,
                "do_sample"     : False,
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data    = payload,
            headers = {
                "Content-Type" : "application/json",
                "Accept"       : "text/event-stream",
            },
            method  = "POST",
        )

        t0: float = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                for line in self._iter_sse_lines(resp):
                    if line.startswith("data:"):
                        payload_str = line[5:].strip()
                        if payload_str and payload_str != "[DONE]":
                            try:
                                event = json.loads(payload_str)
                                token_text = (
                                    event.get("token", {}).get("text", "")
                                )
                                if token_text:
                                    return (time.perf_counter() - t0) * 1_000.0
                            except json.JSONDecodeError:
                                pass
        except Exception as exc:
            # Network errors are treated as very slow (miss-like) responses
            # to avoid silently skipping probes.
            elapsed = (time.perf_counter() - t0) * 1_000.0
            print(f"[TGIBackend] _send_prompt error after {elapsed:.0f} ms: {exc}")
            return elapsed

        # Fallback if stream ends without a token event (shouldn't happen)
        return (time.perf_counter() - t0) * 1_000.0

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _iter_sse_lines(resp: HTTPResponse) -> Iterator[str]:
        """
        Iterate UTF-8 lines from an SSE HTTP response.
        Yields lines without trailing newlines.
        """
        buf = b""
        while True:
            chunk = resp.read(512)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                yield line.decode("utf-8", errors="replace")

    def _fetch_server_info(self) -> tuple[str, bool]:
        """
        GET /info — TGI 2.x returns a JSON object with version and config.
        Returns (framework_version, prefix_caching_enabled).
        """
        try:
            url = f"{self.base_url}/info"
            with urllib.request.urlopen(url, timeout=5) as resp:
                info = json.loads(resp.read())
            ver = info.get("version", self.FRAMEWORK_VER_FALLBACK)
            apc = bool(info.get("prefix_caching", False))
            return ver, apc
        except Exception:
            return self.FRAMEWORK_VER_FALLBACK, True   # assume APC on if unknown