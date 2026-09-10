"""
kv_attack.backends
==================
Pluggable backend abstraction for the Week 12 multi-backend evaluation.

Each backend wraps a running inference server and exposes a single method:
    measure_ttft(prompt: str) -> float   # milliseconds

The harness picks the right backend by name; the attack and reconstruction
code never imports from a concrete adapter directly.

Supported backends
------------------
  vllm   — vLLM ≥ 0.27.x OpenAI-compatible endpoint (APC enabled)
  tgi    — HuggingFace Text Generation Inference ≥ 2.x  /generate endpoint
  mock   — deterministic in-process simulator used for unit tests /
            offline CI runs (no GPU required)

Usage
-----
    from kv_attack.backends import get_backend
    backend = get_backend("vllm", base_url="http://localhost:8001")
    ttft_ms = backend.measure_ttft(prompt)
"""

from kv_attack.backends.base import BackendClient, BackendInfo
from kv_attack.backends.vllm_backend import VLLMBackend
from kv_attack.backends.tgi_backend import TGIBackend
from kv_attack.backends.mock_backend import MockBackend

_REGISTRY: dict[str, type[BackendClient]] = {
    "vllm": VLLMBackend,
    "tgi":  TGIBackend,
    "mock": MockBackend,
}


def get_backend(name: str, **kwargs) -> BackendClient:
    """
    Instantiate and return a backend by name.

    Parameters
    ----------
    name : str
        One of "vllm", "tgi", "mock".
    **kwargs
        Passed directly to the backend constructor.
        vllm  → base_url (str), model_id (str)
        tgi   → base_url (str), model_id (str)
        mock  → hit_ttft_ms (float), miss_ttft_ms (float), noise_std_ms (float)
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown backend '{name}'. "
            f"Available: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[name](**kwargs)


def list_backends() -> list[str]:
    return sorted(_REGISTRY.keys())


__all__ = [
    "BackendClient", "BackendInfo",
    "VLLMBackend", "TGIBackend", "MockBackend",
    "get_backend", "list_backends",
]