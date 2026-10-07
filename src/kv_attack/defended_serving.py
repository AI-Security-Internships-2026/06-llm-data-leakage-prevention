from __future__ import annotations

import dataclasses
import hashlib
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from kv_attack.backends.base import BackendClient
from kv_attack.cache_policy import (
    POLICY_D0, POLICY_D1, POLICY_D2, POLICY_D3, POLICY_D4, CacheAction,
    CachePolicyEngine,
)
from kv_attack.dynamic_policy import DynamicPolicyEngine
from kv_attack.probe_detector import (
    ProbeDetector, SuspicionLevel, SuspicionResult,
)
from kv_attack.risk_signal import RiskSignal, build_risk_signal


class VictimReseedPrompt(str):
    """Marker type: attack code re-sends ``victim_record['prompt']`` to keep the
    victim's blocks warm.  That is the VICTIM's own traffic, not the attacker's,
    and must be routed through the victim's tenant.  (A plain string compare
    would misroute an attacker's *correct guess*, which has identical text.)"""


@dataclass(frozen=True)
class DefenseSpec:
    name: str
    base_policy: str
    use_content: bool
    use_confidence: bool = False
    conf_threshold: float = 0.7
    use_behavior: bool = False
    thr_med: float = 0.40
    thr_high: float = 0.65
    cooldown_probes: int = 10
    description: str = ""


def _spec(name, pol, content, conf=False, beh=False, desc=""):
    return DefenseSpec(name, pol, content, conf, 0.7, beh, description=desc)


DEFENSES: dict[str, DefenseSpec] = {s.name: s for s in [
    _spec("D0_no_defense", POLICY_D0, False, desc="shared cache"),
    _spec("D1_full_isolation", POLICY_D1, False, desc="unique salt per request"),
    _spec("D2_tenant_salt", POLICY_D2, False, desc="per-tenant salt"),
    _spec("D3_binary_pii", POLICY_D3, True, desc="isolate if any PII"),
    _spec("D4_risk_adaptive", POLICY_D4, True, desc="LOW share / MED restrict / HIGH isolate"),
    _spec("D5_risk_behavior", POLICY_D4, True, True, True,
          desc="D4 + confidence gate + probe-behaviour detector (proposed)"),
    _spec("AB0_none", POLICY_D0, False, desc="no component"),
    _spec("AB1_content_only", POLICY_D4, True, desc="content risk only (= D4)"),
    _spec("AB2_content_confidence", POLICY_D4, True, True, desc="content + confidence gate"),
    _spec("AB3_behavior_only", POLICY_D0, False, False, True, desc="behaviour only"),
    _spec("FULL_all_components", POLICY_D4, True, True, True, desc="all components (= D5)"),
]}
CORE_DEFENSES = ["D0_no_defense", "D1_full_isolation", "D2_tenant_salt",
                 "D3_binary_pii", "D4_risk_adaptive", "D5_risk_behavior"]
ABLATION = ["AB0_none", "AB1_content_only", "AB2_content_confidence",
            "AB3_behavior_only", "FULL_all_components"]

_LOW = build_risk_signal({"entities": [], "risk_level": "CLEAN"})
_OK = SuspicionResult(probe_index=-1, suspicion_score=0.0, level=SuspicionLevel.OK,
                      features={}, reasons=[], in_cooldown=False, cooldown_remaining=0)


def presidio_risk(text: str) -> RiskSignal:
    """The paper's content detector (Presidio NER + custom recognisers)."""
    from kv_attack.risk_signal import analyze_prompt
    return analyze_prompt(text)


def regex_risk(text: str) -> RiskSignal:
    """Cheap deterministic stand-in used ONLY with the simulated backend."""
    import re
    from kv_attack import FIRST_NAMES, LAST_NAMES, MEDICAL_CONDITIONS
    global _RX
    try:
        _RX
    except NameError:
        _RX = (re.compile(r"\b(?:%s) (?:%s)\b" % ("|".join(FIRST_NAMES), "|".join(LAST_NAMES))),
               re.compile("|".join(re.escape(c) for c in MEDICAL_CONDITIONS)),
               re.compile(r"[\w.]+@[\w.]+\.\w+"), re.compile(r"\b\d{3}[- ]\d{3}[- ]\d{4}\b"))
    name, cond, mail, phone = _RX
    ents = []
    if name.search(text):
        ents.append({"type": "PERSON", "score": 0.85})
    if cond.search(text):
        ents.append({"type": "MEDICAL_LICENSE", "score": 0.75})
    if mail.search(text):
        ents.append({"type": "EMAIL_ADDRESS", "score": 0.95})
    if phone.search(text):
        ents.append({"type": "PHONE_NUMBER", "score": 0.80})
    types = {e["type"] for e in ents}
    if "PERSON" in types and "MEDICAL_LICENSE" in types:
        lvl = "HIGH"
    elif types:
        lvl = "MEDIUM"
    else:
        lvl = "CLEAN"
    return build_risk_signal({"entities": ents, "risk_level": lvl})


RISK_FNS: dict[str, Callable[[str], RiskSignal]] = {
    "presidio": presidio_risk, "regex": regex_risk}


class DefendedServer:
    def __init__(self, backend: BackendClient, spec: DefenseSpec, *,
                 risk_fn: Callable[[str], RiskSignal] = presidio_risk,
                 fnr: float = 0.0, seed: int = 0,
                 hit_threshold_ms: float = 300.0,
                 victim_tenant: str = "tenant-victim",
                 attacker_think_s: float = 0.0, other_think_s: float = 5.0,
                 canon_fn: Optional[Callable[[str], str]] = None):
        self.backend = backend
        self.canon_fn = canon_fn or (lambda p: p)
        self.spec = spec
        self.risk_fn = risk_fn
        self.fnr = float(fnr)
        self.hit_threshold_ms = hit_threshold_ms
        self.victim_tenant = victim_tenant
        self.think = {"attacker": attacker_think_s}
        self.other_think_s = other_think_s
        self._rng = np.random.default_rng([seed, 7919])
        self._lock = threading.RLock()
        self._risk_cache: dict[bytes, RiskSignal] = {}
        self._pol: dict[str, object] = {}
        self._vclock: dict[str, float] = {}
        self._last_sus: dict[str, SuspicionResult] = {}
        self.audit: list[dict] = []

    def client(self, tenant: str, role: str) -> "TenantClient":
        return TenantClient(self, tenant, role)

    def set_hit_threshold(self, ms: float) -> None:
        self.hit_threshold_ms = float(ms)

    def drain_audit(self) -> list[dict]:
        with self._lock:
            out, self.audit = self.audit, []
        return out

    def _engine(self, tenant: str):
        eng = self._pol.get(tenant)
        if eng is None:
            s = self.spec
            if s.use_behavior:
                eng = DynamicPolicyEngine(client_id=tenant, victim_tenant_id=tenant)
                eng.detector = ProbeDetector(
                    client_id=tenant, threshold_med=s.thr_med,
                    threshold_high=s.thr_high, cooldown_probes=s.cooldown_probes)
            else:
                eng = CachePolicyEngine(policy=s.base_policy, tenant_id=tenant)
            self._pol[tenant] = eng
        return eng

    def _risk(self, prompt: str) -> RiskSignal:
        text = self.canon_fn(prompt)
        key = hashlib.blake2b(text.encode(), digest_size=16).digest()
        sig = self._risk_cache.get(key)
        if sig is None:
            sig = self.risk_fn(text)
            self._risk_cache[key] = sig
        if self.fnr > 0 and sig.pii_flag and self._rng.random() < self.fnr:
            sig = dataclasses.replace(sig, risk_level="LOW", pii_flag=False,
                                      risk_score=0.0, entity_types=[])
        if (self.spec.use_confidence and sig.pii_flag
                and sig.detector_confidence < self.spec.conf_threshold):
            sig = dataclasses.replace(sig, risk_level="LOW", pii_flag=False, risk_score=0.0)
        return sig

    def handle(self, tenant: str, role: str, prompt: str, *, observe: bool = True) -> float:
        return self.handle_full(tenant, role, prompt, observe=observe)[0]

    def handle_full(self, tenant: str, role: str, prompt: str, *, observe: bool = True):
        """Serve one request; return (ttft_ms, audit_record)."""
        s = self.spec
        t0 = time.perf_counter()
        with self._lock:
            sig = self._risk(prompt) if s.use_content else _LOW
            eng = self._engine(tenant)
            if s.use_behavior:
                sus = self._last_sus.get(tenant, _OK)
                dec = eng.decide(sig, sus)
            else:
                dec = eng.decide(sig)
            eff = dec.apply_to_prompt(prompt)
            decide_ms = (time.perf_counter() - t0) * 1e3
        ttft = self.backend.measure_ttft(eff)
        sus_score, sus_level = 0.0, "OK"
        with self._lock:
            vt = self._vclock.get(tenant, 0.0) + ttft / 1e3 + self.think.get(role, self.other_think_s)
            self._vclock[tenant] = vt
            if s.use_behavior and observe:
                res = eng.detector.observe(prompt, float(ttft),
                                           cache_hit=bool(ttft < self.hit_threshold_ms),
                                           timestamp_s=vt)
                self._last_sus[tenant] = res
                sus_score, sus_level = res.suspicion_score, res.level.value
            rec = {
                "tenant": tenant, "role": role, "action": dec.cache_action.value,
                "risk": sig.risk_level, "ttft_ms": round(float(ttft), 3),
                "decide_ms": round(decide_ms, 4), "sus": sus_score, "sus_level": sus_level,
                "observed": bool(observe)}
            self.audit.append(rec)
        return float(ttft), rec


class TenantClient(BackendClient):
    """BackendClient facade for one tenant.  Calls carrying a
    ``VictimReseedPrompt`` are routed as the victim's own traffic."""

    def __init__(self, server: DefendedServer, tenant: str, role: str,
                 observe: bool = True):
        self.server, self.tenant, self.role, self.observe = server, tenant, role, observe

    def unmonitored(self) -> "TenantClient":
        """Same tenant, but its traffic is not fed to the behaviour detector
        (used only for the attacker's cache-eviction flood; documented)."""
        return TenantClient(self.server, self.tenant, self.role, observe=False)

    def health_check(self) -> bool:
        return self.server.backend.health_check()

    def get_info(self):
        return self.server.backend.get_info()

    def _send_prompt(self, prompt: str) -> float:
        if isinstance(prompt, VictimReseedPrompt):
            return self.server.handle(self.server.victim_tenant, "victim", str(prompt))
        return self.server.handle(self.tenant, self.role, prompt, observe=self.observe)
