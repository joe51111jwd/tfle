"""Memory-mapped dataset + sequence packing for cached distillation.

Provides three primary building blocks for the Phase-1 distillation path:

1. ``CachedDistillationDataset`` — random-access dataset over the memory-mapped
   token stream and the top-K teacher logits produced by
   ``phase1_cache_teacher.py``. Each sample is one ``seq_len`` slice containing
   the input tokens, the int16 indices of the top-K teacher logits, and the
   bfloat16 values packed as uint16.

2. ``PackedSequenceDataset`` — wraps a collection of variable-length documents
   and packs them into fixed-length ``seq_len`` sequences for FlashAttention.
   Produces ``cu_seqlens`` (block-diagonal attention boundaries) and position
   ids that reset at each document boundary. Packing efficiency target is
   ``>= 95%`` (enforced as an informational warning, not a hard error).

3. ``make_dataloader`` — thin wrapper around ``torch.utils.data.DataLoader``
   that wires in the standard NOVA defaults used by the training scripts.

The bfloat16 -> uint16 view trick is intentional: numpy does not have a native
bfloat16 dtype, so the cache file stores bfloat16 bytes under a uint16 view
and we reinterpret them via ``torch.Tensor.view(torch.bfloat16)`` at load
time. This avoids any dequantization cost on the hot path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

DEFAULT_K: int = 128
DEFAULT_SEQ_LEN: int = 2048
PACKING_EFFICIENCY_WARN: float = 0.95


# ── Cached teacher-logit dataset ────────────────────────────────────


class CachedDistillationDataset(Dataset):
    """Random-access dataset over cached teacher logits.

    Args:
        token_path: Path to the mem-mapped int32 token stream.
        indices_path: Path to the mem-mapped int16 top-K indices.
        values_path: Path to the mem-mapped uint16 top-K values (bf16 bytes).
        seq_len: Fixed sequence length each sample slices out.
        K: Number of cached top logits per token.

    Each sample returns ``(tokens, indices, values)`` where:
        tokens  - LongTensor of shape ``[seq_len]``
        indices - ShortTensor of shape ``[seq_len, K]`` (int16 vocab ids)
        values  - BFloat16 tensor of shape ``[seq_len, K]`` (logits)
    """

    def __init__(
        self,
        token_path: str | Path,
        indices_path: str | Path,
        values_path: str | Path,
        seq_len: int = DEFAULT_SEQ_LEN,
        K: int = DEFAULT_K,
    ) -> None:
        token_path = Path(token_path)
        indices_path = Path(indices_path)
        values_path = Path(values_path)

        for p in (token_path, indices_path, values_path):
            if not p.exists():
                raise FileNotFoundError(f"Cache file not found: {p}")

        self.token_path = token_path
        self.indices_path = indices_path
        self.values_path = values_path
        self.seq_len = seq_len
        self.K = K

        self.tokens = np.memmap(token_path, dtype=np.int32, mode="r")
        num_tokens = self.tokens.shape[0]

        raw_indices = np.memmap(indices_path, dtype=np.int16, mode="r")
        raw_values = np.memmap(values_path, dtype=np.uint16, mode="r")

        if raw_indices.shape[0] % K != 0:
            raise ValueError(
                f"indices file length {raw_indices.shape[0]} is not divisible by K={K}"
            )
        if raw_values.shape[0] % K != 0:
            raise ValueError(
                f"values file length {raw_values.shape[0]} is not divisible by K={K}"
            )

        self.indices = raw_indices.reshape(-1, K)
        self.values = raw_values.reshape(-1, K)

        if self.indices.shape[0] != num_tokens:
            raise ValueError(
                f"tokens ({num_tokens}) and indices rows ({self.indices.shape[0]}) mismatch"
            )
        if self.values.shape[0] != num_tokens:
            raise ValueError(
                f"tokens ({num_tokens}) and values rows ({self.values.shape[0]}) mismatch"
            )

        self.num_tokens = num_tokens
        self.num_sequences = num_tokens // seq_len
        if self.num_sequences == 0:
            raise ValueError(
                f"Cache has {num_tokens} tokens, need at least seq_len={seq_len}"
            )

        logger.info(
            f"CachedDistillationDataset: {self.num_sequences:,} sequences "
            f"({self.num_tokens:,} tokens, K={K}, seq_len={seq_len})"
        )

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self.num_sequences:
            raise IndexError(f"index {idx} out of range [0, {self.num_sequences})")

        start = idx * self.seq_len
        end = start + self.seq_len

        tokens_np = np.asarray(self.tokens[start:end], dtype=np.int32).copy()
        indices_np = np.asarray(self.indices[start:end], dtype=np.int16).copy()
        values_np = np.asarray(self.values[start:end], dtype=np.uint16).copy()

        tokens = torch.from_numpy(tokens_np).long()
        indices = torch.from_numpy(indices_np).short()
        values_raw = torch.from_numpy(values_np)
        values = values_raw.view(torch.bfloat16)

        return tokens, indices, values


# ── Document packing ────────────────────────────────────────────────


@dataclass
class PackedBatch:
    """Container returned by ``PackedSequenceDataset``.

    Fields:
        input_ids:     LongTensor ``[seq_len]`` of packed tokens.
        position_ids:  LongTensor ``[seq_len]`` that resets at each doc start.
        cu_seqlens:    Int32 tensor of cumulative sequence lengths for
                       FlashAttention's variable-length interface. Shape is
                       ``[num_docs + 1]`` and ends with ``seq_len``.
        doc_lengths:   Int32 tensor ``[num_docs]`` of per-document lengths
                       (useful for masking / statistics).
    """

    input_ids: torch.Tensor
    position_ids: torch.Tensor
    cu_seqlens: torch.Tensor
    doc_lengths: torch.Tensor


def pack_documents(
    docs: Sequence[Sequence[int]],
    seq_len: int,
    pad_token_id: int,
) -> list[PackedBatch]:
    """Pack variable-length documents into fixed-length sequences.

    Greedy first-fit: walks documents in order, appending each to the current
    pack if it fits whole. Documents longer than ``seq_len`` are split into
    ``seq_len``-sized chunks. Trailing space in each pack is filled with
    ``pad_token_id`` and recorded as a zero-length stub doc (so the attention
    mask can ignore it via ``cu_seqlens``).

    Args:
        docs: Iterable of token-id lists (already tokenized).
        seq_len: Target pack length.
        pad_token_id: Token id used to fill trailing padding.

    Returns:
        List of ``PackedBatch`` instances, one per packed sequence.
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")

    packs: list[PackedBatch] = []
    cur_tokens: list[int] = []
    cur_positions: list[int] = []
    cur_doc_lengths: list[int] = []

    def _flush() -> None:
        if not cur_tokens:
            return
        pad_needed = seq_len - len(cur_tokens)
        if pad_needed > 0:
            cur_tokens.extend([pad_token_id] * pad_needed)
            cur_positions.extend(range(pad_needed))
        cu = [0]
        for length in cur_doc_lengths:
            cu.append(cu[-1] + length)
        if pad_needed > 0:
            cu.append(cu[-1] + pad_needed)

        packs.append(
            PackedBatch(
                input_ids=torch.tensor(cur_tokens, dtype=torch.long),
                position_ids=torch.tensor(cur_positions, dtype=torch.long),
                cu_seqlens=torch.tensor(cu, dtype=torch.int32),
                doc_lengths=torch.tensor(
                    cur_doc_lengths + ([pad_needed] if pad_needed > 0 else []),
                    dtype=torch.int32,
                ),
            )
        )
        cur_tokens.clear()
        cur_positions.clear()
        cur_doc_lengths.clear()

    for doc in docs:
        doc_list = list(doc)
        if not doc_list:
            continue

        offset = 0
        while offset < len(doc_list):
            remaining = len(doc_list) - offset
            space = seq_len - len(cur_tokens)
            if space == 0:
                _flush()
                space = seq_len
            take = min(remaining, space)
            chunk = doc_list[offset : offset + take]
            cur_tokens.extend(chunk)
            cur_positions.extend(range(take))
            cur_doc_lengths.append(take)
            offset += take
            if len(cur_tokens) == seq_len:
                _flush()

    _flush()
    return packs


class PackedSequenceDataset(Dataset):
    """Pre-packs a corpus of documents into fixed-length sequences.

    Packing is done eagerly in ``__init__`` so every epoch sees the same
    deterministic packs. This matches how distillation caches are prebuilt
    and avoids re-packing work during training.

    Args:
        docs:          Iterable of token-id lists.
        seq_len:       Target sequence length (default 2048).
        pad_token_id:  Padding token id.
    """

    def __init__(
        self,
        docs: Iterable[Sequence[int]],
        seq_len: int = DEFAULT_SEQ_LEN,
        pad_token_id: int = 0,
    ) -> None:
        doc_list = [list(d) for d in docs]
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id
        self.packs = pack_documents(doc_list, seq_len, pad_token_id)

        total_real = sum(len(d) for d in doc_list)
        total_slots = max(len(self.packs) * seq_len, 1)
        self.efficiency = total_real / total_slots

        logger.info(
            f"PackedSequenceDataset: {len(self.packs):,} packs "
            f"({len(doc_list):,} docs, efficiency={self.efficiency:.1%})"
        )
        if self.efficiency < PACKING_EFFICIENCY_WARN:
            logger.warning(
                f"Packing efficiency {self.efficiency:.1%} below "
                f"target {PACKING_EFFICIENCY_WARN:.0%}"
            )

    def __len__(self) -> int:
        return len(self.packs)

    def __getitem__(self, idx: int) -> PackedBatch:
        return self.packs[idx]


# ── DataLoader wrapper ──────────────────────────────────────────────


def _default_collate(batch: list[tuple]) -> tuple:
    """Stack cached-dataset samples into batched tensors."""
    tokens = torch.stack([b[0] for b in batch], dim=0)
    indices = torch.stack([b[1] for b in batch], dim=0)
    values = torch.stack([b[2] for b in batch], dim=0)
    return tokens, indices, values


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    shuffle: bool = True,
    drop_last: bool = True,
    persistent_workers: bool = True,
    collate_fn=None,
) -> DataLoader:
    """Return a DataLoader with NOVA's standard training defaults.

    Uses a simple stacking collate for ``CachedDistillationDataset``; pass
    ``collate_fn`` explicitly for other dataset shapes (e.g. packed batches).
    """
    if collate_fn is None and isinstance(dataset, CachedDistillationDataset):
        collate_fn = _default_collate

    kwargs: dict = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "shuffle": shuffle,
        "drop_last": drop_last,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = persistent_workers
    if collate_fn is not None:
        kwargs["collate_fn"] = collate_fn

    return DataLoader(dataset, **kwargs)
