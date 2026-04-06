"""Streaming data pipeline for NOVA 2.4B pretraining.

Streams FineWeb-Edu (sample-10BT) + StarCoder (python) with 80/20
text/code mixing. Uses BPE tokenizer for proper subword handling.
Sequence packing with EOS separators for throughput.
"""

from __future__ import annotations

import os
import random
from typing import Iterator

import torch

from .tokenizer_setup import get_tokenizer, VOCAB_SIZE


class PretrainDataLoader:
    """Streaming data loader with text/code interleaving and sequence packing.

    Yields {"input_ids": tensor[B, seq_len], "labels": tensor[B, seq_len]}.
    Labels are shifted input_ids (causal LM objective).
    """

    def __init__(
        self,
        seq_len: int = 2048,
        batch_size: int = 8,
        buffer_size: int = 10_000,
        text_ratio: float = 0.8,
        seed: int = 42,
        split: str = "train",
        val_fraction: float = 0.001,
    ):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.text_ratio = text_ratio
        self.split = split
        self.val_fraction = val_fraction
        self.rng = random.Random(seed)

        self.tokenizer = get_tokenizer()
        self.eos_id = self.tokenizer.eos_token_id

        self.text_tokens = 0
        self.code_tokens = 0

    def _stream_text(self) -> Iterator[list[int]]:
        from datasets import load_dataset

        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        )
        for ex in ds:
            text = ex.get("text", "")
            if not text or len(text) < 100:
                continue
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if tokens:
                yield tokens

    def _stream_code(self) -> Iterator[list[int]]:
        from datasets import load_dataset

        hf_token = os.environ.get("HF_TOKEN")
        try:
            ds = load_dataset(
                "bigcode/starcoderdata",
                data_dir="python",
                split="train",
                streaming=True,
                token=hf_token,
            )
        except Exception:
            try:
                ds = load_dataset(
                    "bigcode/the-stack-dedup",
                    data_dir="data/python",
                    split="train",
                    streaming=True,
                    token=hf_token,
                )
            except Exception:
                print("  WARNING: Code dataset unavailable, text-only mode")
                return

        for ex in ds:
            content = ex.get("content", ex.get("text", ""))
            if not content or len(content) < 100:
                continue
            tokens = self.tokenizer.encode(content, add_special_tokens=False)
            if tokens:
                yield tokens

    def _interleaved_stream(self) -> Iterator[tuple[list[int], str]]:
        """Interleave text and code streams at the configured ratio."""
        text_gen = self._stream_text()
        code_gen = self._stream_code()
        code_exhausted = False

        while True:
            use_code = (
                not code_exhausted
                and self.rng.random() >= self.text_ratio
            )

            if use_code:
                try:
                    tokens = next(code_gen)
                    yield tokens, "code"
                    continue
                except StopIteration:
                    code_exhausted = True

            try:
                tokens = next(text_gen)
                yield tokens, "text"
            except StopIteration:
                return

    def _packed_sequences(self) -> Iterator[list[int]]:
        """Pack documents into fixed-length sequences with EOS separators."""
        token_buf: list[int] = []
        pack_len = self.seq_len + 1  # +1 for label shift

        for tokens, source in self._interleaved_stream():
            token_buf.extend(tokens)
            token_buf.append(self.eos_id)

            if source == "text":
                self.text_tokens += len(tokens)
            else:
                self.code_tokens += len(tokens)

            while len(token_buf) >= pack_len:
                yield token_buf[:pack_len]
                token_buf = token_buf[pack_len:]

    def stream(self) -> Iterator[dict[str, torch.Tensor]]:
        """Yield batches of {"input_ids": ..., "labels": ...}."""
        buffer: list[list[int]] = []

        for seq in self._packed_sequences():
            buffer.append(seq)

            if len(buffer) >= self.buffer_size:
                self.rng.shuffle(buffer)

            if len(buffer) >= self.batch_size:
                batch_seqs = [buffer.pop() for _ in range(self.batch_size)]
                stacked = torch.tensor(batch_seqs, dtype=torch.long)
                yield {
                    "input_ids": stacked[:, :-1],
                    "labels": stacked[:, 1:],
                }

    def get_ratio(self) -> tuple[float, float]:
        total = self.text_tokens + self.code_tokens
        if total == 0:
            return 0.0, 0.0
        return self.text_tokens / total, self.code_tokens / total

    @property
    def vocab_size(self) -> int:
        return VOCAB_SIZE
