"""
kv_attack.dynamic_policy
=========================
Issue #8 — Dynamic cache policy that combines the Issue-7 privacy-risk
signal with the Issue-8 suspicion score.

Combined policy matrix
----------------------
                     Suspicion LOW    Suspicion MED    Suspicion HIGH
  Privacy LOW        SHARE            RESTRICT          ISOLATE
  Privacy MEDIUM     RESTRICT         RESTRICT          ISOLATE
  Privacy HIGH       ISOLATE          ISOLATE           ISOLATE

When suspicion is HIGH the policy escalates to ISOLATE regardless of
privacy risk — an attacker cannot benefit from a shared or restricted
cache entry once detection fires.

Cooldown / recovery
-------------------
After a HIGH suspicion alert the client enters a cooldown: all their
requests are ISOLATE for COOLDOWN_PROBES further requests even if
subsequent probe scores drop below the alert threshold.

Audit
-----
Every decision is logged with: timestamp, privacy_risk, suspicion_score,
suspicion_level, resulting cache_action, reason, and whether the decision
was triggered by cooldown or live score.

Usage
-----
    from kv_attack.dynamic_policy import DynamicPolicyEngine
    from kv_attack.risk_signal    import analyze_prompt
    from kv_attack.probe_detector import ProbeDetector, SuspicionLevel

    engine   = DynamicPolicyEngine(client_id="client-A", victim_tenant_id="tA")
    detector = engine.detector

    for prompt, ttft_ms, cache_hit in stream:
        signal   = analyze_prompt(prompt)
        sus_res  = detector.observe(prompt, ttft_ms, cache_hit)
        decision = engine.decide(signal, sus_res, request_id="req-001")
        effective_prompt = decision.apply_to_prompt(prompt)
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List

from kv_attack.cache_policy import CacheAction, PolicyDecision
from kv_attack.probe_detector import ProbeDetector, SuspicionLevel, SuspicionResult
from kv_attack.risk_signal import RiskSignal



@dataclass
class DynamicDecision:
    """Combined privacy + suspicion cache decision."""
    request_id:          str
    client_id:           str
    privacy_risk_level:  str
    suspicion_level:     str
    suspicion_score:     float
    cache_action:        CacheAction
    salt:                str
    reason:              str
    triggered_by:        str
    timestamp_s:         float
    cross_tenant_reuse_permitted: bool

    def apply_to_prompt(self, prompt: str) -> str:
        if self.cache_action == CacheAction.SHARE or not self.salt:
            return prompt
        return f"{self.salt} {prompt}"

    def to_dict(self) -> dict:
        return {
            "request_id":                 self.request_id,
            "client_id":                  self.client_id,
            "privacy_risk_level":         self.privacy_risk_level,
            "suspicion_level":            self.suspicion_level,
            "suspicion_score":            round(self.suspicion_score, 4),
            "cache_action":               self.cache_action.value,
            "salt":                       self.salt,
            "reason":                     self.reason,
            "triggered_by":               self.triggered_by,
            "timestamp_s":                round(self.timestamp_s, 3),
            "cross_tenant_reuse_permitted": self.cross_tenant_reuse_permitted,
        }



class DynamicPolicyEngine:
    """
    Combines Issue-7 risk signal with Issue-8 suspicion score into a
    dynamic cache isolation policy.

    Parameters
    ----------
    client_id       : identifier for this client / tenant
    victim_tenant_id: used for the RESTRICT (tenant-salt) action
    """

    def __init__(
        self,
        client_id:        str = "client",
        victim_tenant_id: str = "default",
    ):
        self.client_id        = client_id
        self.victim_tenant_id = victim_tenant_id
        self.detector         = ProbeDetector(client_id=client_id)
        self._audit:          List[dict] = []


    def decide(
        self,
        risk_signal:    RiskSignal,
        suspicion_result: SuspicionResult,
        request_id:     str = "",
    ) -> DynamicDecision:
        """
        Apply the combined policy matrix and return a DynamicDecision.

        The caller must have already called detector.observe() for this probe
        to get the suspicion_result; this method only reads from it.
        """
        request_id = request_id or f"req-{uuid.uuid4().hex[:8]}"

        priv  = risk_signal.risk_level.upper()
        sus   = suspicion_result.level
        score = suspicion_result.suspicion_score
        in_cd = suspicion_result.in_cooldown

        if in_cd:
            action      = CacheAction.ISOLATE
            salt        = f"[ISO:{uuid.uuid4().hex}]"
            triggered   = "cooldown"
            cross       = False
            reason      = (
                f"In cooldown after prior ALERT — strict isolation applied. "
                f"Cooldown remaining: {suspicion_result.cooldown_remaining} probes."
            )
        elif sus == SuspicionLevel.ALERT:
            action      = CacheAction.ISOLATE
            salt        = f"[ISO:{uuid.uuid4().hex}]"
            triggered   = "suspicion"
            cross       = False
            reason      = (
                f"HIGH suspicion ({score:.3f}) — probe behaviour matches "
                f"timing attack pattern. Strict isolation applied. "
                f"Reasons: {'; '.join(suspicion_result.reasons)}"
            )
        elif priv == "HIGH":
            action      = CacheAction.ISOLATE
            salt        = f"[ISO:{uuid.uuid4().hex}]"
            triggered   = "privacy"
            cross       = False
            reason      = (
                f"HIGH privacy risk ({risk_signal.semantic_sensitivity}) — "
                f"strict isolation regardless of suspicion. {risk_signal.reason}"
            )
        elif sus == SuspicionLevel.WARN or priv == "MEDIUM":
            action      = CacheAction.RESTRICT
            salt        = self._tenant_salt()
            triggered   = "suspicion" if sus == SuspicionLevel.WARN else "privacy"
            cross       = False
            reason      = (
                f"MEDIUM risk/suspicion — tenant-restricted sharing. "
                f"Privacy={priv}, Suspicion={sus.value} ({score:.3f})."
            )
        else:
            action      = CacheAction.SHARE
            salt        = ""
            triggered   = "normal"
            cross       = True
            reason      = (
                f"LOW privacy ({priv}) and LOW suspicion ({score:.3f}) — "
                f"normal cache sharing permitted."
            )

        decision = DynamicDecision(
            request_id                 = request_id,
            client_id                  = self.client_id,
            privacy_risk_level         = priv,
            suspicion_level            = sus.value,
            suspicion_score            = score,
            cache_action               = action,
            salt                       = salt,
            reason                     = reason,
            triggered_by               = triggered,
            timestamp_s                = time.time(),
            cross_tenant_reuse_permitted = cross,
        )
        self._audit.append(decision.to_dict())
        return decision


    def _tenant_salt(self) -> str:
        h = hashlib.sha256(self.victim_tenant_id.encode()).hexdigest()[:8]
        return f"[T:{h}]"


    @property
    def audit_log(self) -> List[dict]:
        return list(self._audit)

    def dump_audit_log(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            for entry in self._audit:
                fh.write(json.dumps(entry) + "\n")

    def summary(self) -> dict:
        det_sum = self.detector.summary()
        actions = {}
        for d in self._audit:
            a = d["cache_action"]
            actions[a] = actions.get(a, 0) + 1
        return {
            "client_id":     self.client_id,
            "n_decisions":   len(self._audit),
            "action_counts": actions,
            "detector":      det_sum,
        }