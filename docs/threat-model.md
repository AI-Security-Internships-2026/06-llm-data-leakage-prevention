# Threat Model: Cross-Tenant KV-Cache Timing Side-Channel

**Project:** LLM Data Leakage Prevention — Detection and Mitigation
**Scope:** Phase 2 (Issue #9) — reproduction, detection/mitigation, and overhead benchmarking
**Status:** Draft

---

## 1. System Model

We consider a **multi-tenant LLM inference service** built on a modern serving
engine that performs cross-request prefix caching. Concretely, we target
**SGLang with RadixAttention**, whose default configuration reuses previously
computed key–value (KV) cache blocks across *all* requests handled by the same
engine instance, regardless of which client, session, or application submitted
them.

Multiple independent front-end applications (distinct processes, distinct
authentication, distinct network origins/ports, no shared application code)
issue completion requests to a **single shared backend engine**. The engine
maintains one global radix tree of cached prefixes. A request whose prompt
prefix already exists in the tree skips recomputation for the matching tokens,
producing (i) a measurably lower time-to-first-token (TTFT) and, where the API
surfaces it, (ii) an explicit `cached_tokens` count in the response metadata.

This prefix cache is a **shared resource across mutually distrusting tenants**.
The application-layer isolation (separate logins, separate services) sits
*above* this shared resource and therefore does not constrain it.

## 2. Assets

- **Primary asset:** short, structured secrets that a victim tenant places in a
  prompt — in our reproduction, a numeric PIN standing in for PII, credentials,
  OTPs, or account identifiers.
- **Generalizable asset:** any secret-bearing prompt prefix (system prompts,
  retrieved RAG passages, API keys) whose tokens align with cache-block
  boundaries.

## 3. Adversary Model

**Capabilities.** The adversary is an *ordinary tenant* of the same backend.
They:
- hold only their own valid credentials on their own front-end application;
- can submit arbitrary prompts and observe per-request TTFT (and any
  `cached_tokens` field the endpoint returns);
- cannot read the victim's traffic, memory, logs, or KV cache directly;
- cannot execute code on the server or the victim's client.

**Knowledge.** The adversary knows the prompt template the victim's application
uses (a realistic assumption — templates are static, often public, and
recoverable). They know the secret's format (e.g. a 4-digit PIN).

**Goal.** Reconstruct the victim's secret by using cache-hit observations as an
oracle: a correctly guessed prefix collides with the victim's cached prefix and
yields a hit; an incorrect guess does not. This reduces recovery from an
exponential search over the whole secret to a **prefix-by-prefix linear search**
(≈ `base × length` probes instead of `base^length`).

**Out of scope.** We do not consider a malicious server operator, side channels
below the serving engine (hardware cache, speculative execution), or attacks
requiring co-located code execution.

## 4. Attack Preconditions (Explicit)

The attack is **configuration-dependent**, and stating its preconditions is part
of the contribution. It requires *all* of:

1. **Cross-request prefix caching enabled** on the shared engine (SGLang
   RadixAttention default).
2. **No per-tenant cache partitioning / keying** — cache lookups are not scoped
   by tenant, session, or API key.
3. **Secret persistence in cache** between the victim's submission and the
   adversary's probes (the entry has not been LRU-evicted under load).
4. **Tokenization alignment** — the guessing unit (e.g. a digit) maps to a cache
   block boundary so hits are observable per position.
5. **An observable hit signal** — TTFT difference above measurement noise, or an
   explicit `cached_tokens` field.

Where hosted commercial APIs (OpenAI, DeepSeek, etc.) scope prompt caching per
organization/account, precondition (2) fails for an external adversary; this is
*why* the reproduction targets a self-hosted SGLang deployment rather than a
hosted API, and that scoping choice is itself a reportable result.

## 5. Relationship to Prior Work — Positioning Against PromptPeek

The **mechanism** we reproduce — multi-tenant KV-cache sharing turned into a
timing/hit oracle that leaks prompt content across tenants — is **established
prior art**, most directly in **PromptPeek** [CITE], and adjacent to the broader
line of work on prompt-cache timing side channels and prompt-cache auditing
[CITE — e.g. timing-side-channel input-stealing and cache-auditing papers,
2024–2025]. We do **not** claim the attack primitive as novel, and we cite these
works as the origin of the threat.

Our reproduction contributes **methodological confirmation and a controlled
testbed**, not a new attack:

| Dimension | PromptPeek (prior art) | This work |
|---|---|---|
| Attack primitive (cross-tenant KV-cache hit oracle) | Introduced | **Reproduced**, not claimed as novel |
| Demonstration setting | Per prior paper | Two fully isolated front-ends over one SGLang backend, showing app-layer isolation is irrelevant to the leak |
| Contribution focus | Demonstrate the leak exists | **Detection + mitigation of the leak, with a security-vs-overhead benchmark** |
| Deliverable | Attack | Defense evaluated against the reproduced attack |

**Framing rule for the paper:** the reproduction is the *motivation / threat-model
section*, establishing that the leak is real and that existing isolation does not
stop it. The **novel core** is the detection/mitigation and its quantified
overhead trade-off (Sections below). This keeps the reproduction from being
mistaken for the contribution — the failure mode that would draw a
"known-attack" desk reject at a Transactions venue.

> **Citation TODO:** insert exact PromptPeek bibliographic entry (authors, venue,
> year, DOI) plus 2–3 adjacent cache-side-channel references into
> `docs/literature-review.md` and cross-reference here. Verify the PromptPeek
> claim set against the published text so the "reproduced vs. novel" boundary is
> stated accurately.

## 6. Contribution Claims (What Is Actually Novel)

C1. A **detection** method that flags cache-probing behavior from a tenant
(e.g. structured, monotone prefix-extension query sequences and/or anomalous
per-tenant cache-hit-rate distributions) — a signal PromptPeek does not provide.

C2. A **mitigation** for SGLang-class engines: per-tenant cache-key salting /
secret-aware cache bypass / TTFT normalization, that provably removes the hit
oracle for the adversary while preserving legitimate intra-tenant reuse.

C3. A **security-vs-performance benchmark** quantifying the throughput/latency
cost of each mitigation against the measured drop in attack success rate — the
trade-off curve is the empirical contribution.

## 7. Validity and Limitations to Report

- Report attack success **with its conditions**: e.g. "30/30 PIN recovery, no
  concurrent load, warm cache, client-side TTFT, N=30." A bare "100%" invites
  skepticism.
- Include a **realistic-load** run (concurrent background traffic + cache
  eviction) to show whether the leak — and the defense — hold outside the quiet
  lab.
- State the **noise model** and probes-per-secret, and whether the oracle used
  was TTFT timing or the explicit `cached_tokens` field (the latter is a direct
  oracle, a cleaner and stronger result).
