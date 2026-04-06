"""3-stage pretraining support for NOVA 2.4B.

Stage 1: Synthetic curriculum from pre-generated JSONL files
Stage 2: Hard-only real data via perplexity filtering
Stage 3: Brief unfiltered pass at lower LR

Classes:
  SyntheticCurriculumLoader - Loads and tokenizes JSONL curriculum files
  PerplexityFilter          - Scores/filters documents by model perplexity
  STEStabilityMonitor       - Detects STE training instability
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer_setup import get_tokenizer, VOCAB_SIZE


# ── Stage configs ────────────────────────────────────────

STAGE_CONFIGS = {
    1: {
        "lr": 3e-4,
        "min_lr": 3e-5,
        "warmup_steps": 200,
        "total_steps": 10_000,
        "weight_decay": 0.01,
        "seq_len": 512,
        "batch_size": 8,
        "gradient_accumulation": 1,
        "description": "Synthetic curriculum warmup",
    },
    2: {
        "lr": 3e-4,
        "min_lr": 3e-5,
        "warmup_steps": 2000,
        "total_steps": 200_000,
        "weight_decay": 0.1,
        "seq_len": 2048,
        "batch_size": 8,
        "gradient_accumulation": 8,
        "ppl_rescore_every": 50_000,
        "description": "Hard-only real data with perplexity filtering",
    },
    3: {
        "lr": 1e-4,
        "min_lr": 1e-5,
        "warmup_steps": 500,
        "total_steps": 50_000,
        "weight_decay": 0.1,
        "seq_len": 2048,
        "batch_size": 8,
        "gradient_accumulation": 4,
        "description": "Unfiltered data cooldown",
    },
}


def get_stage_config(stage: int) -> dict:
    """Return the default config for a given pretraining stage."""
    if stage not in STAGE_CONFIGS:
        raise ValueError(f"Invalid stage {stage}, must be 1, 2, or 3")
    return dict(STAGE_CONFIGS[stage])


# ── SyntheticCurriculumLoader ────────────────────────────

class SyntheticCurriculumLoader:
    """Loads pre-generated JSONL curriculum files, tokenizes with BPE,
    and packs into fixed-length sequences.

    JSONL format: each line is {"text": "..."} or {"content": "..."}.
    Files are loaded in sorted order from the given directory.
    """

    def __init__(
        self,
        data_dir: str | Path,
        seq_len: int = 512,
        batch_size: int = 8,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.shuffle = shuffle

        import random
        self.rng = random.Random(seed)

        self.tokenizer = get_tokenizer()
        self.eos_id = self.tokenizer.eos_token_id

        self._files = sorted(self.data_dir.glob("*.jsonl"))
        if not self._files:
            raise FileNotFoundError(
                f"No .jsonl files found in {self.data_dir}"
            )

    def _read_documents(self) -> Iterator[str]:
        """Yield raw text from all JSONL files."""
        for path in self._files:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = obj.get("text") or obj.get("content", "")
                    if text:
                        yield text

    def _tokenized_stream(self) -> Iterator[list[int]]:
        """Tokenize documents into token-id lists."""
        for text in self._read_documents():
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if tokens:
                yield tokens

    def _packed_sequences(self) -> Iterator[list[int]]:
        """Pack tokenized documents into fixed-length sequences."""
        token_buf: list[int] = []
        pack_len = self.seq_len + 1  # +1 for label shift

        for tokens in self._tokenized_stream():
            token_buf.extend(tokens)
            token_buf.append(self.eos_id)

            while len(token_buf) >= pack_len:
                yield token_buf[:pack_len]
                token_buf = token_buf[pack_len:]

    def stream(self) -> Iterator[dict[str, torch.Tensor]]:
        """Yield batches of {"input_ids": ..., "labels": ...}."""
        buffer: list[list[int]] = []

        for seq in self._packed_sequences():
            buffer.append(seq)

            if self.shuffle and len(buffer) >= 1000:
                self.rng.shuffle(buffer)

            if len(buffer) >= self.batch_size:
                batch_seqs = [buffer.pop() for _ in range(self.batch_size)]
                stacked = torch.tensor(batch_seqs, dtype=torch.long)
                yield {
                    "input_ids": stacked[:, :-1],
                    "labels": stacked[:, 1:],
                }

    def get_ratio(self) -> tuple[float, float]:
        """Compatibility with PretrainDataLoader interface."""
        return 1.0, 0.0


# ── PerplexityFilter ─────────────────────────────────────

class PerplexityFilter:
    """Scores documents by model perplexity and filters for hard examples.

    Hard examples = documents with perplexity above the corpus median.
    The filter is re-scored periodically as the model improves.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        seq_len: int = 2048,
        score_batch_size: int = 4,
        percentile: float = 50.0,
    ):
        self.model = model
        self.device = device
        self.seq_len = seq_len
        self.score_batch_size = score_batch_size
        self.percentile = percentile

        self.tokenizer = get_tokenizer()
        self.eos_id = self.tokenizer.eos_token_id
        self._ppl_threshold: float | None = None
        self._corpus_scores: list[tuple[int, float]] = []

    @torch.no_grad()
    def _compute_ppl(self, token_ids: list[int]) -> float:
        """Compute per-token perplexity for a single document."""
        if len(token_ids) < 2:
            return 0.0

        ids = token_ids[: self.seq_len + 1]
        input_ids = torch.tensor([ids[:-1]], dtype=torch.long, device=self.device)
        labels = torch.tensor([ids[1:]], dtype=torch.long, device=self.device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = self.model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                labels.reshape(-1),
                reduction="mean",
            )

        return math.exp(min(loss.item(), 20.0))  # cap to avoid overflow

    def score_corpus(
        self,
        documents: Iterator[list[int]],
        max_docs: int = 10_000,
    ) -> float:
        """Score a corpus of documents and set the PPL threshold.

        Args:
            documents: Iterator of tokenized documents (list[int] each).
            max_docs: Max documents to score for efficiency.

        Returns:
            The computed PPL threshold (median by default).
        """
        self.model.eval()
        scores: list[float] = []

        for i, doc_tokens in enumerate(documents):
            if i >= max_docs:
                break
            ppl = self._compute_ppl(doc_tokens)
            if ppl > 0:
                scores.append(ppl)
                self._corpus_scores.append((i, ppl))

        if not scores:
            self._ppl_threshold = 0.0
            return 0.0

        scores.sort()
        idx = int(len(scores) * self.percentile / 100.0)
        idx = min(idx, len(scores) - 1)
        self._ppl_threshold = scores[idx]

        self.model.train()
        return self._ppl_threshold

    def is_hard(self, token_ids: list[int]) -> bool:
        """Check if a document is above the PPL threshold (hard example)."""
        if self._ppl_threshold is None:
            return True  # no threshold set yet, accept everything
        ppl = self._compute_ppl(token_ids)
        return ppl >= self._ppl_threshold

    def filter_dataloader(
        self,
        documents: Iterator[list[int]],
    ) -> Iterator[list[int]]:
        """Yield only hard documents (above PPL threshold)."""
        for doc_tokens in documents:
            if self.is_hard(doc_tokens):
                yield doc_tokens

    def rescore_and_update(
        self,
        documents: Iterator[list[int]],
        max_docs: int = 10_000,
    ) -> float:
        """Re-score corpus and update the threshold as the model improves.

        Returns the new threshold.
        """
        self._corpus_scores.clear()
        return self.score_corpus(documents, max_docs=max_docs)

    @property
    def threshold(self) -> float | None:
        return self._ppl_threshold


# ── STEStabilityMonitor ──────────────────────────────────

@dataclass
class StabilityStatus:
    """Result of a stability check."""
    state: str  # "stable", "warning", "critical"
    reason: str = ""
    train_loss: float = 0.0
    val_loss: float = 0.0
    grad_norm: float = 0.0
    loss_gap: float = 0.0


class STEStabilityMonitor:
    """Monitors STE training for instability patterns.

    Tracks:
      - Train/val loss divergence (val >> train)
      - Gradient norm spikes (> 5x running average)
      - Consecutive loss increases

    On "critical": auto-reduces LR by 50%.
    """

    def __init__(
        self,
        window_size: int = 100,
        divergence_threshold: float = 0.5,
        grad_spike_factor: float = 5.0,
        max_consecutive_increases: int = 10,
        lr_reduction_factor: float = 0.5,
    ):
        self.window_size = window_size
        self.divergence_threshold = divergence_threshold
        self.grad_spike_factor = grad_spike_factor
        self.max_consecutive_increases = max_consecutive_increases
        self.lr_reduction_factor = lr_reduction_factor

        self._train_losses: deque[float] = deque(maxlen=window_size)
        self._val_losses: deque[float] = deque(maxlen=window_size)
        self._grad_norms: deque[float] = deque(maxlen=window_size)
        self._consecutive_increases: int = 0
        self._prev_loss: float | None = None
        self._lr_reductions: int = 0

    def record_train_loss(self, loss: float):
        """Record a training loss value."""
        self._train_losses.append(loss)

        if self._prev_loss is not None:
            if loss > self._prev_loss:
                self._consecutive_increases += 1
            else:
                self._consecutive_increases = 0
        self._prev_loss = loss

    def record_val_loss(self, loss: float):
        """Record a validation loss value."""
        self._val_losses.append(loss)

    def record_grad_norm(self, norm: float):
        """Record a gradient norm value."""
        self._grad_norms.append(norm)

    def check(self) -> StabilityStatus:
        """Assess current training stability.

        Returns StabilityStatus with state in {"stable", "warning", "critical"}.
        """
        if len(self._train_losses) < 10:
            return StabilityStatus(state="stable", reason="insufficient data")

        avg_train = sum(self._train_losses) / len(self._train_losses)
        avg_val = (
            sum(self._val_losses) / len(self._val_losses)
            if self._val_losses
            else avg_train
        )
        avg_grad = (
            sum(self._grad_norms) / len(self._grad_norms)
            if self._grad_norms
            else 0.0
        )
        latest_grad = self._grad_norms[-1] if self._grad_norms else 0.0
        loss_gap = avg_val - avg_train

        # Critical checks
        if self._consecutive_increases >= self.max_consecutive_increases:
            return StabilityStatus(
                state="critical",
                reason=f"{self._consecutive_increases} consecutive loss increases",
                train_loss=avg_train,
                val_loss=avg_val,
                grad_norm=latest_grad,
                loss_gap=loss_gap,
            )

        if avg_grad > 0 and latest_grad > avg_grad * self.grad_spike_factor:
            return StabilityStatus(
                state="critical",
                reason=f"grad norm spike: {latest_grad:.2f} vs avg {avg_grad:.2f}",
                train_loss=avg_train,
                val_loss=avg_val,
                grad_norm=latest_grad,
                loss_gap=loss_gap,
            )

        # Warning checks
        if loss_gap > self.divergence_threshold:
            return StabilityStatus(
                state="warning",
                reason=f"train/val divergence: gap={loss_gap:.4f}",
                train_loss=avg_train,
                val_loss=avg_val,
                grad_norm=latest_grad,
                loss_gap=loss_gap,
            )

        half_consec = self.max_consecutive_increases // 2
        if self._consecutive_increases >= half_consec:
            return StabilityStatus(
                state="warning",
                reason=f"{self._consecutive_increases} consecutive loss increases",
                train_loss=avg_train,
                val_loss=avg_val,
                grad_norm=latest_grad,
                loss_gap=loss_gap,
            )

        return StabilityStatus(
            state="stable",
            reason="all metrics nominal",
            train_loss=avg_train,
            val_loss=avg_val,
            grad_norm=latest_grad,
            loss_gap=loss_gap,
        )

    def maybe_reduce_lr(
        self,
        optimizer: torch.optim.Optimizer,
        status: StabilityStatus,
    ) -> bool:
        """If status is critical, reduce LR by 50%. Returns True if reduced."""
        if status.state != "critical":
            return False

        for pg in optimizer.param_groups:
            pg["lr"] *= self.lr_reduction_factor

        self._lr_reductions += 1
        self._consecutive_increases = 0
        return True

    @property
    def lr_reductions(self) -> int:
        return self._lr_reductions

    def reset(self):
        """Reset all tracked state."""
        self._train_losses.clear()
        self._val_losses.clear()
        self._grad_norms.clear()
        self._consecutive_increases = 0
        self._prev_loss = None
