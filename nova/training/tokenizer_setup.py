"""BPE tokenizer setup for NOVA 2.4B pretraining.

Uses GPT-2 tokenizer with special reasoning tokens.
Final vocab_size = 50261.
"""

from __future__ import annotations

import torch
from transformers import AutoTokenizer

SPECIAL_TOKENS = ["<think>", "</think>", "<answer>", "</answer>"]
VOCAB_SIZE = 50261

_cached_tokenizer: AutoTokenizer | None = None


def get_tokenizer() -> AutoTokenizer:
    """Return the NOVA tokenizer, creating and caching on first call.

    GPT-2 base vocab (50257) + 4 special tokens = 50261.
    """
    global _cached_tokenizer
    if _cached_tokenizer is not None:
        return _cached_tokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    assert len(tokenizer) == VOCAB_SIZE, (
        f"Expected vocab_size={VOCAB_SIZE}, got {len(tokenizer)}"
    )

    _cached_tokenizer = tokenizer
    return tokenizer


def encode(text: str, max_length: int | None = None) -> list[int]:
    """Encode text to token ids."""
    tokenizer = get_tokenizer()
    kwargs = {"add_special_tokens": False}
    if max_length is not None:
        kwargs["max_length"] = max_length
        kwargs["truncation"] = True
    return tokenizer.encode(text, **kwargs)


def decode(token_ids: list[int] | torch.Tensor) -> str:
    """Decode token ids back to text."""
    tokenizer = get_tokenizer()
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def encode_batch(texts: list[str], max_length: int | None = None) -> dict:
    """Encode a batch of texts, returning padded tensors."""
    tokenizer = get_tokenizer()
    kwargs = {"padding": True, "return_tensors": "pt", "add_special_tokens": False}
    if max_length is not None:
        kwargs["max_length"] = max_length
        kwargs["truncation"] = True
    return tokenizer(texts, **kwargs)
