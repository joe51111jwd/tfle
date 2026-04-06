"""Streaming data pipeline for NOVA 2.4B pretraining.

Streams FineWeb-Edu (sample-10BT) + StarCoder (python) with 80/20
text/code mixing. Uses BPE tokenizer for proper subword handling.
Sequence packing with EOS separators for throughput.

Also provides FilteredPretrainDataLoader for Stage 2 perplexity-based
filtering (yields only hard examples above a PPL threshold).
"""

from __future__ import annotations

import math
import os
import random
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class FilteredPretrainDataLoader(PretrainDataLoader):
    """PretrainDataLoader with perplexity-based filtering for Stage 2.

    Extends PretrainDataLoader to only yield documents that the model finds
    hard (above a perplexity threshold). The threshold is set by scoring a
    sample of the corpus and taking the median PPL.

    Usage:
        loader = FilteredPretrainDataLoader(model, device, seq_len=2048, ...)
        loader.initial_score(max_docs=10_000)
        for batch in loader.stream():
            ...
        # Periodically re-score as model improves:
        loader.rescore(max_docs=10_000)
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        seq_len: int = 2048,
        batch_size: int = 8,
        buffer_size: int = 10_000,
        text_ratio: float = 0.8,
        seed: int = 42,
        split: str = "train",
        ppl_percentile: float = 50.0,
        score_batch_size: int = 4,
    ):
        super().__init__(
            seq_len=seq_len,
            batch_size=batch_size,
            buffer_size=buffer_size,
            text_ratio=text_ratio,
            seed=seed,
            split=split,
        )
        self.model = model
        self.device = device
        self.ppl_percentile = ppl_percentile
        self.score_batch_size = score_batch_size
        self._ppl_threshold: float | None = None
        self._scored = False
        self._docs_accepted = 0
        self._docs_rejected = 0

    @torch.no_grad()
    def _compute_doc_ppl(self, token_ids: list[int]) -> float:
        """Compute per-token perplexity for a single tokenized document."""
        if len(token_ids) < 2:
            return 0.0

        ids = token_ids[: self.seq_len + 1]
        input_ids = torch.tensor(
            [ids[:-1]], dtype=torch.long, device=self.device
        )
        labels = torch.tensor(
            [ids[1:]], dtype=torch.long, device=self.device
        )

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = self.model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                labels.reshape(-1),
                reduction="mean",
            )

        return math.exp(min(loss.item(), 20.0))

    def initial_score(self, max_docs: int = 10_000) -> float:
        """Run model over a corpus sample before Stage 2 to set PPL threshold.

        Streams documents from the parent's interleaved stream, scores each,
        and sets the threshold at ppl_percentile of the distribution.

        Returns:
            The computed PPL threshold.
        """
        self.model.eval()
        scores: list[float] = []

        text_gen = self._stream_text()
        code_gen = self._stream_code()
        code_exhausted = False

        for i in range(max_docs):
            use_code = (
                not code_exhausted
                and self.rng.random() >= self.text_ratio
            )
            doc_tokens = None

            if use_code:
                try:
                    doc_tokens = next(code_gen)
                except StopIteration:
                    code_exhausted = True

            if doc_tokens is None:
                try:
                    doc_tokens = next(text_gen)
                except StopIteration:
                    break

            ppl = self._compute_doc_ppl(doc_tokens)
            if ppl > 0:
                scores.append(ppl)

        if not scores:
            self._ppl_threshold = 0.0
            self._scored = True
            self.model.train()
            return 0.0

        scores.sort()
        idx = int(len(scores) * self.ppl_percentile / 100.0)
        idx = min(idx, len(scores) - 1)
        self._ppl_threshold = scores[idx]
        self._scored = True

        self.model.train()
        return self._ppl_threshold

    def rescore(self, max_docs: int = 10_000) -> float:
        """Re-score the corpus and update threshold as model improves.

        Should be called periodically (e.g., every 50K steps).

        Returns:
            The new PPL threshold.
        """
        self._docs_accepted = 0
        self._docs_rejected = 0
        return self.initial_score(max_docs=max_docs)

    def _interleaved_stream(self) -> Iterator[tuple[list[int], str]]:
        """Override parent to filter out easy documents."""
        for tokens, source in super()._interleaved_stream():
            if self._ppl_threshold is not None:
                ppl = self._compute_doc_ppl(tokens)
                if ppl < self._ppl_threshold:
                    self._docs_rejected += 1
                    continue

            self._docs_accepted += 1
            yield tokens, source

    @property
    def ppl_threshold(self) -> float | None:
        return self._ppl_threshold

    @property
    def filter_stats(self) -> dict:
        total = self._docs_accepted + self._docs_rejected
        return {
            "accepted": self._docs_accepted,
            "rejected": self._docs_rejected,
            "total": total,
            "accept_rate": self._docs_accepted / max(total, 1),
            "threshold": self._ppl_threshold,
        }
