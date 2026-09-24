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
    FRAMEWORK_VER = "0.27.1"

    def __init__(self, base_url: str, model_id: str):
        self.base_url = base_url
        self.model_id = model_id
        self._client  = OpenAI(base_url=base_url, api_key="EMPTY", timeout=30.0, max_retries=1)


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
        Return wall-clock TTFT (ms) using a NON-streaming request, with a
        hard thread-based timeout as a safety net.

        HISTORY: streaming (stream=True, read first chunk) is the
        theoretically correct way to isolate pure prefill/TTFT from
        decode+packaging time, and was used successfully in the original
        50-victim/68%-SR baseline run. However, after this environment's
        `openai` SDK version was churned by an unrelated SGLang install
        (2.6.1 -> 1.99.9 -> 3.15.0), streaming requests started returning
        server-side 200 OK immediately (confirmed in vLLM server logs) but
        the client-side stream iterator never yielded a chunk — a SDK/
        server SSE-format incompatibility introduced by that version churn,
        not a genuine server hang. Non-streaming avoids this entirely (same
        approach already proven reliable against the SGLang backend) at the
        cost of including one decode step's latency in the TTFT measurement
        (negligible for max_tokens=1). The thread+queue hard timeout is
        kept regardless, as a safety net against any future stalls.
        """
        import threading, queue

        def _do_request(result_q: queue.Queue) -> None:
            try:
                t0 = time.perf_counter()
                self._client.completions.create(
                    model       = self.model_id,
                    prompt      = prompt,
                    max_tokens  = 1,
                    temperature = 0.0,
                    stream      = False,
                )
                result_q.put(("ok", (time.perf_counter() - t0) * 1_000.0))
            except Exception as exc:
                result_q.put(("error", exc))

        result_q: queue.Queue = queue.Queue(maxsize=1)
        t = threading.Thread(target=_do_request, args=(result_q,), daemon=True)
        t.start()

        try:
            status, value = result_q.get(timeout=30.0)
        except queue.Empty:
            print(f"[VLLMBackend] WARNING: request HARD-TIMED-OUT after 30s; "
                  f"treating as miss (9999ms). Orphaned thread abandoned "
                  f"(daemon=True, will not block run).")
            return 9999.0

        if status == "error":
            print(f"[VLLMBackend] WARNING: request failed ({value!r}); "
                  f"treating as miss (9999ms)")
            return 9999.0
        return value


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