

from __future__ import annotations

import logging
import random
import uuid

logger = logging.getLogger(__name__)

# ── Eviction vocabulary ────────────────────────────────────────────────────────
# Deliberately non-medical, non-financial: reduces risk of accidental cache
# overlap with victim prompts.

EVICTION_VOCABULARY: list[str] = [
    # Sciences
    "photosynthesis", "mitochondria", "thermodynamics", "electromagnet",
    "gravitational", "subatomic", "chromosome", "riboflavin", "isotope",
    "exothermic", "polynomial", "differential", "eigenvalue", "stochastic",
    # Geography
    "archipelago", "tectonic", "stratosphere", "permafrost", "longitude",
    "equatorial", "meridional", "topographic", "hydrological", "peninsular",
    # Technology
    "semiconductor", "bandwidth", "cryptography", "transistor", "oscillator",
    "microprocessor", "infrared", "spectrometer", "capacitance", "impedance",
    # Misc long words (maximise token count per word)
    "extraordinarily", "incomprehensible", "disproportionate", "circumnavigation",
    "electromagnetic", "metamorphosis", "disestablishment", "unconstitutional",
    "anthropomorphic", "counterproductive", "interdisciplinary", "transcontinental",
    "biogeochemical", "thermodynamical", "phenomenological", "epistemological",
]


def evict_cache(
    client,
    model:         str,
    n_requests:    int = 20,
    token_length:  int = 200,
) -> None:
    """
    Send *n_requests* unique noise prompts to vLLM to trigger LRU eviction
    of victim KV-cache blocks.

    Each noise prompt:
    - Contains a unique UUID prefix → guaranteed cache miss → fills new blocks
    - Is ~200 tokens long → displaces approximately as many victim blocks as
      a typical victim prompt (Proposal §3.5: victim prompts are ~180-220 tokens)

    Parameters
    ----------
    client       : OpenAI-compatible client (real vLLM or MockVLLMClient)
    model        : vLLM model ID
    n_requests   : number of eviction requests (default 20; Paper 11 used 15)
    token_length : approximate word count per eviction prompt (default 200)
    """
    logger.debug("Evicting KV cache (%d requests × ~%d tokens) ...", n_requests, token_length)

    for i in range(n_requests):
        noise_words = random.choices(EVICTION_VOCABULARY, k=token_length)
        noise_prompt = (
            f"[EVICT-{uuid.uuid4().hex}] "
            + " ".join(noise_words)
        )
        try:
            client.completions.create(
                model=model,
                prompt=noise_prompt,
                max_tokens=1,
                temperature=0.0,
                stream=False,
            )
        except Exception as exc:
            logger.warning("Eviction request %d/%d failed: %s", i + 1, n_requests, exc)

    logger.debug("Cache eviction complete.")
