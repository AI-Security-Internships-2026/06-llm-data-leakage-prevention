from __future__ import annotations

import threading
import zlib
from collections import OrderedDict

import numpy as np

from kv_attack.backends.base import BackendClient, BackendInfo

BLOCK_SIZE = 16


class SimTokenizer:
    """Deterministic stand-in for a HF tokenizer (no hash() randomisation)."""

    bos_token_id = 128000

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        toks = [zlib.crc32(w.encode()) for w in text.split()]
        return ([self.bos_token_id] + toks) if add_special_tokens else toks


class SimPrefixCacheBackend(BackendClient):
    def __init__(
        self,
        tokenizer: SimTokenizer | None = None,
        *,
        capacity_blocks: int = 45_697,
        base_ms: float = 85.0,
        per_block_ms: float = 3.05,
        noise_ms: float = 6.0,
        apc_enabled: bool = True,
        seed: int = 0,
    ):
        self.tok = tokenizer or SimTokenizer()
        self.capacity = capacity_blocks
        self.base_ms = base_ms
        self.per_block_ms = per_block_ms
        self.noise_ms = noise_ms
        self.apc_enabled = apc_enabled
        self._rng = np.random.default_rng(seed)
        self._cache: OrderedDict[int, None] = OrderedDict()
        self._lock = threading.Lock()
        self._load_level = 0
        self.n_requests = 0
        self.tot_hit_blocks = 0
        self.tot_blocks = 0
        self.last_hit_blocks = 0
        self.last_total_blocks = 0

    def set_load(self, level: int) -> None:
        """Background-load proxy: adds queueing delay and jitter (level 0..n)."""
        self._load_level = max(0, int(level))

    def reseed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def flush(self) -> None:
        with self._lock:
            self._cache.clear()

    def health_check(self) -> bool:
        return True

    def get_info(self) -> BackendInfo:
        return BackendInfo(
            backend_name="sim", framework="sim-apc", framework_ver="1.0",
            model_id="sim-model", base_url="sim://local",
            apc_enabled=self.apc_enabled,
            extra={"capacity_blocks": self.capacity, "base_ms": self.base_ms,
                   "per_block_ms": self.per_block_ms, "noise_ms": self.noise_ms},
        )

    def _send_prompt(self, prompt: str) -> float:
        toks = [self.tok.bos_token_id] + self.tok.encode(prompt)
        n_blocks = len(toks) // BLOCK_SIZE
        with self._lock:
            hit = 0
            h = 0
            keys = []
            for b in range(n_blocks):
                h = hash((h, tuple(toks[b * BLOCK_SIZE:(b + 1) * BLOCK_SIZE])))
                keys.append(h)
            if self.apc_enabled:
                for k in keys:
                    if k in self._cache:
                        hit += 1
                        self._cache.move_to_end(k)
                    else:
                        break
                for k in keys:
                    self._cache[k] = None
                    self._cache.move_to_end(k)
                while len(self._cache) > self.capacity:
                    self._cache.popitem(last=False)
            miss = n_blocks - hit
            tail = len(toks) - n_blocks * BLOCK_SIZE
            self.last_hit_blocks, self.last_total_blocks = hit, n_blocks
            self.n_requests += 1
            self.tot_hit_blocks += hit
            self.tot_blocks += n_blocks
            lvl = self._load_level
            mean = (self.base_ms + self.per_block_ms * (miss + tail / BLOCK_SIZE)
                    + 9.0 * lvl)
            noise = float(self._rng.normal(0.0, self.noise_ms * (1 + 0.5 * lvl)))
        return max(1.0, mean + noise)


def install_fast_block_builders() -> None:
    """Replace the O(n^2) pad loops in two_stage_victim_seeder with an
    equivalent arithmetic version (same output length semantics) -- sim only.
    Memoised so the 100 names / 20 conditions are built once."""
    import functools
    from kv_attack import two_stage_victim_seeder as tv

    def _pad(base: str, unit: str, blocks: int, tokenizer) -> str:
        need = blocks * BLOCK_SIZE
        t0 = len(tokenizer.encode(base, add_special_tokens=False))
        if t0 >= need:
            return base
        per = max(1, len(tokenizer.encode(base + " " + unit, add_special_tokens=False)) - t0)
        k = -(-(need - t0) // per)
        out = base + (" " + unit) * k
        while len(tokenizer.encode(out, add_special_tokens=False)) < need:
            out += " " + unit
        return out

    @functools.lru_cache(maxsize=4096)
    def _name_block(name: str, tok_id: int) -> str:
        base = f"{name}. " + (f" {name}." * 40) + " " + tv._NAME_FILLER
        return _pad(base, name, tv.NAME_BLOCKS, _TOK[tok_id])

    @functools.lru_cache(maxsize=4096)
    def _cond_block(cond: str, tok_id: int) -> str:
        base = f"{cond}. " + (f" {cond}." * 20) + " " + tv._CONDITION_FILLER
        return _pad(base, cond, tv.COND_BLOCKS, _TOK[tok_id])

    _TOK: dict[int, object] = {}

    def build_name_block(name, tokenizer):
        _TOK[id(tokenizer)] = tokenizer
        return _name_block(name, id(tokenizer))

    def build_condition_block(condition, tokenizer):
        _TOK[id(tokenizer)] = tokenizer
        return _cond_block(condition, id(tokenizer))

    tv.build_name_block = build_name_block
    tv.build_condition_block = build_condition_block


def install_memoized_block_builders() -> None:
    """REAL-backend counterpart: wrap the ORIGINAL block builders in an LRU
    cache.  Output is byte-identical to the originals (so block alignment and
    all earlier artifacts stay comparable); it only avoids rebuilding the same
    100 names / 20 conditions thousands of times."""
    import functools
    from kv_attack import two_stage_victim_seeder as tv
    if getattr(tv, "_issue10_memoized", False):
        return
    orig_name, orig_cond = tv.build_name_block, tv.build_condition_block
    reg: dict[int, object] = {}

    @functools.lru_cache(maxsize=8192)
    def _n(name, tid):
        return orig_name(name, reg[tid])

    @functools.lru_cache(maxsize=8192)
    def _c(cond, tid):
        return orig_cond(cond, reg[tid])

    def build_name_block(name, tokenizer):
        reg[id(tokenizer)] = tokenizer
        return _n(name, id(tokenizer))

    def build_condition_block(condition, tokenizer):
        reg[id(tokenizer)] = tokenizer
        return _c(condition, id(tokenizer))

    tv.build_name_block, tv.build_condition_block = build_name_block, build_condition_block
    tv._issue10_memoized = True
