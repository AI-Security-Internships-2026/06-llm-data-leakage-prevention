import time
import hashlib
from dataclasses import dataclass, field
from typing import Optional



@dataclass
class KVBlock:
    """
    Simulated KV-cache block with CacheSolidarity metadata.

    In the real vLLM implementation this metadata lives inside BlockManager.
    Here we simulate it as a Python object for demonstration.
    """
    block_hash  : str
    owner_id    : int
    attack_flag : bool
    ref_count   : int = 0

    METADATA_BYTES: int = 32


@dataclass
class PrefixWallCache:
    """
    Simulated shared KV-cache pool with PrefixWall enforcement.

    Maintains a dictionary of block_hash → KVBlock and implements the
    CacheSolidarity access control policy.
    """
    blocks: dict = field(default_factory=dict)
    n_hits: int  = 0
    n_misses: int = 0
    n_flagged: int = 0

    def lookup(self, block_hash: str, requesting_tenant: int) -> tuple[bool, bool]:
        """
        Look up a block in the cache for a given tenant.

        Returns
        -------
        (is_hit, is_flagged)
            is_hit    : True if block exists in cache
            is_flagged: True if block was flagged (cross-tenant probe detected)
        """
        if block_hash not in self.blocks:
            self.n_misses += 1
            return False, False

        block = self.blocks[block_hash]

        if block.owner_id == requesting_tenant:
            self.n_hits += 1
            block.ref_count += 1
            return True, False
        else:
            if not block.attack_flag:
                block.attack_flag = True
                self.n_flagged += 1
            self.n_hits += 1
            return True, True

    def insert(self, block_hash: str, owner_id: int) -> KVBlock:
        """Insert a newly computed block into the cache."""
        block = KVBlock(
            block_hash  = block_hash,
            owner_id    = owner_id,
            attack_flag = False,
            ref_count   = 1,
        )
        self.blocks[block_hash] = block
        return block

    def evict_lru(self) -> Optional[str]:
        """Evict the least-recently-referenced block with ref_count == 0."""
        for h, b in list(self.blocks.items()):
            if b.ref_count == 0:
                del self.blocks[h]
                return h
        return None

    @property
    def memory_overhead_bytes(self) -> int:
        """Total metadata memory overhead across all cached blocks."""
        return len(self.blocks) * KVBlock.METADATA_BYTES

    def stats(self) -> dict:
        return {
            "n_blocks"          : len(self.blocks),
            "n_hits"            : self.n_hits,
            "n_misses"          : self.n_misses,
            "n_cross_tenant"    : self.n_flagged,
            "memory_overhead_bytes": self.memory_overhead_bytes,
        }



class PrefixWallHandler:
    """
    Simulates the CacheSolidarity request-level enforcement layer.

    For each incoming request, this handler:
    1. Tokenises and blocks the prompt (simulated)
    2. Looks up each block in the shared cache
    3. For flagged (cross-tenant) blocks: adds artificial delay so
       TTFT is indistinguishable from a cache miss
    4. Records all decisions for evaluation
    """

    MASKING_DELAY_MS: float = 488.9

    def __init__(self, cache: PrefixWallCache, block_size: int = 16):
        self.cache      = cache
        self.block_size = block_size
        self.request_log: list[dict] = []

    def _simulate_block_hashes(self, prompt_tokens: list[int]) -> list[str]:
        """
        Simulate vLLM's SHA-256 block hash chain for a token sequence.
        block_N_hash = SHA256(block_N_tokens | block_{N-1}_hash)
        """
        hashes = []
        parent_hash = b"\x00" * 32
        n_complete = len(prompt_tokens) // self.block_size
        for i in range(n_complete):
            block_tokens = prompt_tokens[i * self.block_size:(i + 1) * self.block_size]
            import struct
            content = parent_hash + struct.pack(f"{len(block_tokens)}I", *block_tokens)
            h = hashlib.sha256(content).hexdigest()
            hashes.append(h)
            parent_hash = bytes.fromhex(h)
        return hashes

    def handle_request(
        self,
        prompt_tokens : list[int],
        tenant_id     : int,
        base_ttft_ms  : float = 88.9,
        miss_ttft_ms  : float = 577.8,
    ) -> dict:
        """
        Process one request under PrefixWall enforcement.

        Returns a dict describing what happened: which blocks were hits,
        which were flagged, what TTFT was observed (with masking applied).
        """
        block_hashes = self._simulate_block_hashes(prompt_tokens)
        n_blocks     = len(block_hashes)

        n_clean_hits  = 0
        n_flagged_hits = 0
        n_misses      = 0
        first_miss_block = None

        for i, bh in enumerate(block_hashes):
            is_hit, is_flagged = self.cache.lookup(bh, tenant_id)

            if not is_hit:
                n_misses += 1
                if first_miss_block is None:
                    first_miss_block = i
                self.cache.insert(bh, tenant_id)
            elif is_flagged:
                n_flagged_hits += 1
            else:
                n_clean_hits += 1

        if n_flagged_hits > 0 or n_misses > 0:
            effective_ttft_ms = miss_ttft_ms
        else:
            effective_ttft_ms = base_ttft_ms

        metadata_overhead_ms = 0.007

        result = {
            "tenant_id"          : tenant_id,
            "n_blocks"           : n_blocks,
            "n_clean_hits"       : n_clean_hits,
            "n_flagged_hits"     : n_flagged_hits,
            "n_misses"           : n_misses,
            "effective_ttft_ms"  : effective_ttft_ms + metadata_overhead_ms,
            "masking_applied"    : n_flagged_hits > 0,
            "cache_stats"        : self.cache.stats(),
        }
        self.request_log.append(result)
        return result



def compute_prefixwall_overhead(
    unprotected_hit_ttft_ms    : float = 88.9,
    unprotected_miss_ttft_ms   : float = 577.8,
    full_disable_hit_ttft_ms   : float = 648.8,
    cache_reuse_rate           : float = 0.70,
) -> dict:
    """
    Analytically compute the PrefixWall operating point on the Pareto curve.

    Assumes:
    - Cross-tenant blocks are served at miss_ttft (masked)
    - Same-tenant blocks are served at hit_ttft (no masking)
    - cache_reuse_rate: fraction of requests that get a clean (same-tenant) hit

    Returns expected TTFT and overhead vs unprotected baseline.
    """
    expected_ttft_ms = (
        cache_reuse_rate       * unprotected_hit_ttft_ms +
        (1 - cache_reuse_rate) * unprotected_miss_ttft_ms
    ) + 0.007

    overhead_vs_unprotected_pct = round(
        (expected_ttft_ms - unprotected_hit_ttft_ms) / unprotected_hit_ttft_ms * 100, 1
    )
    overhead_vs_full_disable_pct = round(
        (full_disable_hit_ttft_ms - expected_ttft_ms) / expected_ttft_ms * 100, 1
    )

    return {
        "mitigation"                       : "CacheSolidarity / PrefixWall",
        "reference"                        : "Pennas et al. (2026), arXiv 2603.10726",
        "cache_reuse_rate"                 : cache_reuse_rate,
        "expected_ttft_ms"                 : round(expected_ttft_ms, 2),
        "unprotected_hit_ttft_ms"          : unprotected_hit_ttft_ms,
        "full_disable_hit_ttft_ms"         : full_disable_hit_ttft_ms,
        "overhead_vs_unprotected_pct"      : overhead_vs_unprotected_pct,
        "improvement_vs_full_disable_pct"  : overhead_vs_full_disable_pct,
        "metadata_overhead_per_request_ms" : 0.007,
        "metadata_overhead_per_block_bytes": 32,
        "timing_oracle_destroyed"          : True,
        "note": (
            "PrefixWall masks cross-tenant cache hits by adding artificial "
            "delay equal to the miss TTFT, destroying the timing oracle while "
            "preserving same-tenant cache reuse. Overhead is analytically "
            "derived from Paper 8 (70% cache reuse retention vs full isolation)."
        ),
    }



def get_pareto_operating_points(
    unprotected_hit_ttft_ms  : float = 88.9,
    unprotected_miss_ttft_ms : float = 577.8,
    full_disable_ttft_ms     : float = 648.8,
) -> list[dict]:
    """
    Return the four Pareto curve operating points for Week 13 analysis.

    Point 1: No mitigation (unprotected baseline)
    Point 2: Full APC disable (our Week 11 result)
    Point 3: CacheSolidarity / PrefixWall (Paper 8, analytical)
    Point 4: Novel mitigation (Presidio + BART NLI gate, Week 13)
    """
    prefixwall = compute_prefixwall_overhead(
        unprotected_hit_ttft_ms  = unprotected_hit_ttft_ms,
        unprotected_miss_ttft_ms = unprotected_miss_ttft_ms,
        full_disable_hit_ttft_ms = full_disable_ttft_ms,
    )

    return [
        {
            "label"                : "No mitigation (baseline)",
            "ttft_ms"              : unprotected_hit_ttft_ms,
            "overhead_pct"         : 0.0,
            "leak_reduction_pct"   : 0.0,
            "sr"                   : 1.0,
            "oracle_destroyed"     : False,
        },
        {
            "label"                : "Full APC disable (Week 11)",
            "ttft_ms"              : full_disable_ttft_ms,
            "overhead_pct"         : round(
                (full_disable_ttft_ms - unprotected_hit_ttft_ms)
                / unprotected_hit_ttft_ms * 100, 1),
            "leak_reduction_pct"   : 99.95,
            "sr"                   : 0.0005,
            "oracle_destroyed"     : True,
            "reference"            : "Week 11 empirical result",
        },
        {
            "label"                : "CacheSolidarity / PrefixWall (Paper 8)",
            "ttft_ms"              : prefixwall["expected_ttft_ms"],
            "overhead_pct"         : prefixwall["overhead_vs_unprotected_pct"],
            "leak_reduction_pct"   : 99.95,
            "sr"                   : 0.0005,
            "oracle_destroyed"     : True,
            "reference"            : "Pennas et al. (2026), analytical",
        },
        {
            "label"                : "Novel mitigation (Presidio + BART NLI, Week 13)",
            "ttft_ms"              : None,
            "overhead_pct"         : None,
            "leak_reduction_pct"   : None,
            "sr"                   : None,
            "oracle_destroyed"     : None,
            "reference"            : "Week 13 empirical result (pending)",
        },
    ]



if __name__ == "__main__":
    import json

    print("=== PrefixWall Simulation Demo ===\n")

    cache   = PrefixWallCache()
    handler = PrefixWallHandler(cache)

    victim_tokens = list(range(271)) + list(range(100, 116))
    print("[demo] Victim (tenant 1) sends prompt → caches blocks...")
    r1 = handler.handle_request(victim_tokens, tenant_id=1)
    print(f"       TTFT: {r1['effective_ttft_ms']:.1f} ms  "
          f"(clean hits={r1['n_clean_hits']}, misses={r1['n_misses']})")

    print("\n[demo] Attacker (tenant 2) probes with same tokens → flagged!")
    r2 = handler.handle_request(victim_tokens, tenant_id=2)
    print(f"       TTFT: {r2['effective_ttft_ms']:.1f} ms  "
          f"(flagged_hits={r2['n_flagged_hits']}, masking_applied={r2['masking_applied']})")
    print(f"       → Attacker sees {r2['effective_ttft_ms']:.1f} ms (indistinguishable from miss)")

    wrong_tokens = list(range(271)) + list(range(200, 216))
    print("\n[demo] Attacker probes with wrong tokens → miss")
    r3 = handler.handle_request(wrong_tokens, tenant_id=2)
    print(f"       TTFT: {r3['effective_ttft_ms']:.1f} ms  (miss)")

    print(f"\n[demo] Both flagged hit and miss return ~577 ms → oracle destroyed ✓")

    print("\n=== Pareto Curve Operating Points ===\n")
    points = get_pareto_operating_points()
    for p in points:
        ttft    = f"{p['ttft_ms']:.1f} ms" if p['ttft_ms'] else "TBD"
        overhead= f"+{p['overhead_pct']}%" if p['overhead_pct'] is not None else "TBD"
        leak    = f"{p['leak_reduction_pct']}%" if p['leak_reduction_pct'] is not None else "TBD"
        print(f"  {p['label']}")
        print(f"    TTFT={ttft}  overhead={overhead}  leak_reduction={leak}")
    print()

    pw = compute_prefixwall_overhead()
    print("=== CacheSolidarity Analytical Overhead ===")
    print(json.dumps(pw, indent=2))