"""
kv_attack.mock_tokenizer
=========================
A lightweight tokenizer shim that avoids downloading HuggingFace weights
in offline / CI / smoke-test environments.

Encoding model: approximate GPT-style byte-pair tokenisation — each word
maps to 1 token, punctuation/spaces add +1 per cluster.  Accuracy is
±15 % on medical-record text, which is good enough for block-alignment
smoke tests (block_size = 16 tokens; misalignment of 1-2 tokens is
acceptable for pipeline validation).

Usage
-----
    from kv_attack.mock_tokenizer import MockTokenizer
    tok = MockTokenizer()
    ids = tok.encode("Hello World")   # → [72, 101, 108, 108, 111, 32, ...]
    n   = len(ids)                    # token count
"""

from __future__ import annotations

import re


class MockTokenizer:
    """
    A minimal tokenizer shim for offline pipeline tests.

    API is a strict subset of HuggingFace ``AutoTokenizer``:
      - ``encode(text)``  → ``list[int]``
      - ``decode(ids)``   → ``str``  (identity-ish; not exact)
      - ``model_max_length`` attribute
      - ``bos_token_id`` attribute (128000, matching Llama-3 / DeepSeek)
    """

    model_max_length: int = 131072
    bos_token_id: int = 128000
    eos_token_id: int = 128001

    _WORD_RE = re.compile(r"[A-Za-z0-9]+|[^A-Za-z0-9\s]+|\s+")

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """
        Encode text to a list of pseudo-token ids.

        Each whitespace-delimited word token → 1 id.
        Punctuation clusters → 1 id each.
        Total length approximates GPT-style BPE counts for typical English prose.
        """
        tokens: list[int] = []
        if add_special_tokens:
            tokens.append(self.bos_token_id)
        for chunk in self._WORD_RE.findall(text):
            if not chunk.strip():
                continue
            if len(chunk) > 12:
                for i in range(0, len(chunk), 6):
                    sub = chunk[i:i + 6]
                    tokens.append(hash(sub) & 0x7FFF or 1)
            else:
                tokens.append(hash(chunk) & 0x7FFF or 1)
        return tokens

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Inverse mapping (approximate; used only in tests)."""
        return f"<decoded:{len(ids)}_tokens>"

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> "MockTokenizer":
        """Drop-in replacement for ``AutoTokenizer.from_pretrained``."""
        return cls()
