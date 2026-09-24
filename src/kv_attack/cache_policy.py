"""
kv_attack.cache_policy
=======================
Issue #7 — Risk-adaptive KV-cache isolation policy engine.

Implements five named defense policies (D0–D4) that translate a RiskSignal
into a concrete cache action.  Every decision is audit-logged.

Policy definitions
------------------
  D0  No defense (shared cache, unprotected baseline)
  D1  Full isolation — disable prefix caching for every request
  D2  Tenant / user salt — add a per-tenant salt so tenants cannot share
      cache with each other, but the same tenant's requests can still hit
  D3  Binary PII isolation — if detector flagged any PII: isolate;
      otherwise: share normally
  D4  Risk-adaptive (novel contribution) —
        risk LOW   → normal sharing (same as D0)
        risk MEDIUM → restricted sharing (tenant salt, D2-style)
        risk HIGH  → strict per-request isolation (unique salt, D1-style)

Cache actions
-------------
  SHARE    – no salt; prompt sent to backend as-is; cross-tenant reuse allowed
  RESTRICT – tenant salt prepended; intra-tenant reuse allowed; cross-tenant blocked
  ISOLATE  – unique per-request UUID salt prepended; no reuse at all

Salt mechanics
--------------
The salt is a short ASCII string prepended to the prompt before it is sent
to the LLM backend.  Because KV-cache prefix matching is performed on the
tokenised hash chain, any difference in the very first token(s) breaks the
entire shared-prefix chain, guaranteeing a full cache miss for any request
that does not know the exact salt.

An attacker who does not know the salt will always receive MISS-TTFT when
probing a salted victim's cache blocks, making the timing oracle useless.

Audit log
---------
Every call to CachePolicyEngine.decide() appends one entry to an in-memory
list accessible via engine.audit_log.  Call engine.dump_audit_log(path) to
persist as newline-delimited JSON.

Usage
-----
    from kv_attack.cache_policy import CachePolicyEngine, POLICY_D4
    from kv_attack.risk_signal import analyze_prompt

    engine = CachePolicyEngine(policy=POLICY_D4, tenant_id="tenant-A")
    signal = analyze_prompt("Patient Alice Johnson has hypertension.")
    decision = engine.decide(signal, request_id="req-001")

    # Apply the salt (if any) to the prompt before sending to the backend
    effective_prompt = decision.apply_to_prompt(original_prompt)
    backend.measure_ttft(effective_prompt)

    # Check whether this decision permits cross-tenant cache reuse
    print(decision.cross_tenant_reuse_permitted)   # False
    print(decision.cache_action)                   # ISOLATE
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional

from kv_attack.risk_signal import RiskSignal



class CacheAction(str, Enum):
    SHARE    = "SHARE"
    RESTRICT = "RESTRICT"
    ISOLATE  = "ISOLATE"



POLICY_D0 = "D0_no_defense"
POLICY_D1 = "D1_full_isolation"
POLICY_D2 = "D2_tenant_salt"
POLICY_D3 = "D3_binary_pii"
POLICY_D4 = "D4_risk_adaptive"

ALL_POLICIES = [POLICY_D0, POLICY_D1, POLICY_D2, POLICY_D3, POLICY_D4]

POLICY_DESCRIPTIONS = {
    POLICY_D0: "No defense — full shared cache (unprotected baseline)",
    POLICY_D1: "Full isolation — every request gets a unique salt (cache disabled)",
    POLICY_D2: "Tenant salt — same tenant shares cache; cross-tenant blocked",
    POLICY_D3: "Binary PII — if any PII detected: isolate; otherwise: share",
    POLICY_D4: "Risk-adaptive — LOW→share, MEDIUM→tenant-restrict, HIGH→isolate",
}


@dataclass
class PolicyDecision:
    """
    The output of CachePolicyEngine.decide().

    Attributes
    ----------
    policy              : which policy produced this decision
    cache_action        : SHARE | RESTRICT | ISOLATE
    salt                : the string prepended to the prompt (empty for SHARE)
    reason              : human-readable explanation
    cross_tenant_reuse_permitted : True only when cache_action == SHARE
    risk_level          : the risk level that drove this decision
    risk_score          : continuous risk score (0–1)
    semantic_sensitivity: domain tag from RiskSignal
    entity_types        : detected entity types
    timestamp_s         : Unix timestamp of the decision
    request_id          : caller-provided request identifier (optional)
    tenant_id           : the tenant for which this decision was made
    """
    policy:                       str
    cache_action:                 CacheAction
    salt:                         str
    reason:                       str
    cross_tenant_reuse_permitted: bool
    risk_level:                   str
    risk_score:                   float
    semantic_sensitivity:         str
    entity_types:                 List[str]
    timestamp_s:                  float
    request_id:                   str
    tenant_id:                    str

    def apply_to_prompt(self, prompt: str) -> str:
        """
        Prepend the policy salt to *prompt*.

        If cache_action == SHARE, the prompt is returned unchanged.
        Otherwise, the salt string is prepended verbatim, which changes
        the tokenised prefix and breaks the KV-cache hash chain.
        """
        if self.cache_action == CacheAction.SHARE or not self.salt:
            return prompt
        return f"{self.salt} {prompt}"

    def to_dict(self) -> dict:
        return {
            "policy":                       self.policy,
            "cache_action":                 self.cache_action.value,
            "salt":                         self.salt,
            "reason":                       self.reason,
            "cross_tenant_reuse_permitted": self.cross_tenant_reuse_permitted,
            "risk_level":                   self.risk_level,
            "risk_score":                   round(self.risk_score, 4),
            "semantic_sensitivity":         self.semantic_sensitivity,
            "entity_types":                 self.entity_types,
            "timestamp_s":                  round(self.timestamp_s, 3),
            "request_id":                   self.request_id,
            "tenant_id":                    self.tenant_id,
        }



class CachePolicyEngine:
    """
    Translates a RiskSignal into a PolicyDecision for a given policy.

    Parameters
    ----------
    policy    : one of POLICY_D0 … POLICY_D4
    tenant_id : identifier for the requesting tenant/user; used for D2/D4 RESTRICT
    """

    def __init__(self, policy: str, tenant_id: str = "default"):
        if policy not in ALL_POLICIES:
            raise ValueError(
                f"Unknown policy '{policy}'. Must be one of: {ALL_POLICIES}"
            )
        self.policy    = policy
        self.tenant_id = tenant_id
        self._audit_log: List[dict] = []


    def decide(
        self,
        signal:     RiskSignal,
        request_id: str = "",
    ) -> PolicyDecision:
        """
        Apply the policy to *signal* and return a PolicyDecision.

        The decision is appended to self.audit_log immediately.

        Parameters
        ----------
        signal     : RiskSignal from analyze_prompt() or build_risk_signal()
        request_id : optional caller-supplied request identifier for tracing
        """
        request_id = request_id or f"req-{uuid.uuid4().hex[:8]}"

        if self.policy == POLICY_D0:
            decision = self._d0(signal, request_id)
        elif self.policy == POLICY_D1:
            decision = self._d1(signal, request_id)
        elif self.policy == POLICY_D2:
            decision = self._d2(signal, request_id)
        elif self.policy == POLICY_D3:
            decision = self._d3(signal, request_id)
        else:
            decision = self._d4(signal, request_id)

        self._audit_log.append(decision.to_dict())
        return decision


    def _d0(self, signal: RiskSignal, request_id: str) -> PolicyDecision:
        """D0: no defense — always share."""
        return PolicyDecision(
            policy                       = POLICY_D0,
            cache_action                 = CacheAction.SHARE,
            salt                         = "",
            reason                       = "D0: no defense — shared cache for all requests.",
            cross_tenant_reuse_permitted = True,
            risk_level                   = signal.risk_level,
            risk_score                   = signal.risk_score,
            semantic_sensitivity         = signal.semantic_sensitivity,
            entity_types                 = signal.entity_types,
            timestamp_s                  = time.time(),
            request_id                   = request_id,
            tenant_id                    = self.tenant_id,
        )

    def _d1(self, signal: RiskSignal, request_id: str) -> PolicyDecision:
        """D1: full isolation — unique UUID per request."""
        salt = f"[ISO:{uuid.uuid4().hex}]"
        return PolicyDecision(
            policy                       = POLICY_D1,
            cache_action                 = CacheAction.ISOLATE,
            salt                         = salt,
            reason                       = "D1: full isolation — unique salt applied to every request.",
            cross_tenant_reuse_permitted = False,
            risk_level                   = signal.risk_level,
            risk_score                   = signal.risk_score,
            semantic_sensitivity         = signal.semantic_sensitivity,
            entity_types                 = signal.entity_types,
            timestamp_s                  = time.time(),
            request_id                   = request_id,
            tenant_id                    = self.tenant_id,
        )

    def _d2(self, signal: RiskSignal, request_id: str) -> PolicyDecision:
        """D2: tenant salt — all requests from the same tenant share a consistent salt."""
        salt = self._tenant_salt()
        return PolicyDecision(
            policy                       = POLICY_D2,
            cache_action                 = CacheAction.RESTRICT,
            salt                         = salt,
            reason                       = (
                f"D2: tenant salt applied — cache isolated to tenant '{self.tenant_id}'; "
                f"intra-tenant reuse permitted."
            ),
            cross_tenant_reuse_permitted = False,
            risk_level                   = signal.risk_level,
            risk_score                   = signal.risk_score,
            semantic_sensitivity         = signal.semantic_sensitivity,
            entity_types                 = signal.entity_types,
            timestamp_s                  = time.time(),
            request_id                   = request_id,
            tenant_id                    = self.tenant_id,
        )

    def _d3(self, signal: RiskSignal, request_id: str) -> PolicyDecision:
        """D3: binary PII isolation — if pii_flag: isolate, else: share."""
        if signal.pii_flag:
            salt         = f"[ISO:{uuid.uuid4().hex}]"
            action       = CacheAction.ISOLATE
            reason       = (
                f"D3: PII detected ({signal.risk_level}) — strict isolation applied. "
                f"Entities: {signal.entity_types}."
            )
            cross_tenant = False
        else:
            salt         = ""
            action       = CacheAction.SHARE
            reason       = "D3: no PII detected — normal cache sharing."
            cross_tenant = True

        return PolicyDecision(
            policy                       = POLICY_D3,
            cache_action                 = action,
            salt                         = salt,
            reason                       = reason,
            cross_tenant_reuse_permitted = cross_tenant,
            risk_level                   = signal.risk_level,
            risk_score                   = signal.risk_score,
            semantic_sensitivity         = signal.semantic_sensitivity,
            entity_types                 = signal.entity_types,
            timestamp_s                  = time.time(),
            request_id                   = request_id,
            tenant_id                    = self.tenant_id,
        )

    def _d4(self, signal: RiskSignal, request_id: str) -> PolicyDecision:
        """D4: risk-adaptive — LOW→share, MEDIUM→restrict, HIGH→isolate."""
        level = signal.risk_level.upper()

        if level == "HIGH":
            salt         = f"[ISO:{uuid.uuid4().hex}]"
            action       = CacheAction.ISOLATE
            cross_tenant = False
            reason       = (
                f"D4 HIGH risk ({signal.semantic_sensitivity}, "
                f"conf={signal.detector_confidence:.2f}): strict per-request isolation. "
                f"Entities: {signal.entity_types}. {signal.reason}"
            )
        elif level == "MEDIUM":
            salt         = self._tenant_salt()
            action       = CacheAction.RESTRICT
            cross_tenant = False
            reason       = (
                f"D4 MEDIUM risk ({signal.semantic_sensitivity}, "
                f"conf={signal.detector_confidence:.2f}): tenant-restricted sharing. "
                f"Entities: {signal.entity_types}. {signal.reason}"
            )
        else:
            salt         = ""
            action       = CacheAction.SHARE
            cross_tenant = True
            reason       = (
                f"D4 LOW risk (score={signal.risk_score:.3f}): "
                f"normal cache sharing permitted. {signal.reason}"
            )

        return PolicyDecision(
            policy                       = POLICY_D4,
            cache_action                 = action,
            salt                         = salt,
            reason                       = reason,
            cross_tenant_reuse_permitted = cross_tenant,
            risk_level                   = signal.risk_level,
            risk_score                   = signal.risk_score,
            semantic_sensitivity         = signal.semantic_sensitivity,
            entity_types                 = signal.entity_types,
            timestamp_s                  = time.time(),
            request_id                   = request_id,
            tenant_id                    = self.tenant_id,
        )


    def _tenant_salt(self) -> str:
        """Return a deterministic, short salt derived from the tenant ID."""
        h = hashlib.sha256(self.tenant_id.encode()).hexdigest()[:8]
        return f"[T:{h}]"


    @property
    def audit_log(self) -> List[dict]:
        """Read-only view of all decisions logged in this session."""
        return list(self._audit_log)

    def dump_audit_log(self, path: str) -> None:
        """Write audit log as newline-delimited JSON to *path*."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            for entry in self._audit_log:
                fh.write(json.dumps(entry) + "\n")
        print(f"[CachePolicyEngine] Audit log written: {path} ({len(self._audit_log)} entries)")

    def reset_audit_log(self) -> None:
        """Clear the in-memory audit log (e.g. between policies in an eval loop)."""
        self._audit_log = []

    def audit_summary(self) -> dict:
        """Return counts of cache actions in the current log."""
        counts: dict[str, int] = {a.value: 0 for a in CacheAction}
        for entry in self._audit_log:
            action = entry.get("cache_action", "SHARE")
            counts[action] = counts.get(action, 0) + 1
        total = len(self._audit_log)
        return {
            "total_decisions":       total,
            "share_count":           counts.get("SHARE", 0),
            "restrict_count":        counts.get("RESTRICT", 0),
            "isolate_count":         counts.get("ISOLATE", 0),
            "isolation_fraction":    round(
                (counts.get("RESTRICT", 0) + counts.get("ISOLATE", 0)) / max(total, 1), 4
            ),
        }