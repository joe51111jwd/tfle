#!/usr/bin/env python3
"""
Phase 1: Cache top-K teacher logits for distillation
=====================================================
Runs a frozen teacher (Qwen-2.5-1.5B by default) over the pretokenized corpus
and caches the top-K logits for every token on NVMe. The training loop then
streams these cached distributions via ``CachedDistillationDataset`` instead
of re-running the teacher each epoch.

Design notes:
  - Teacher runs in fp16 (NOT quantized). On H100/A100 the fp16 matmul path
    is faster than 4-bit for a 1.5B model since throughput is memory-bound.
  - Work is sharded across ranks with a simple contiguous split — the token
    stream is embarrassingly parallel so no cross-rank communication is
    needed during the forward pass.
  - Cache files are mem-mapped:
        tokens_mixed.npy      int32     [N]          input token ids
        cached_indices.npy    int16     [N, K]       top-K vocab ids
        cached_values.npy     uint16    [N, K]       top-K logits (bf16 bytes)
    N ~ 1.342B, K = 128, so size ~= N * K * (2 + 2) = ~687 GB.
  - Throughput target: ~200K+ tok/s aggregate across 8xH100 ~= 2h for 1.342B.
  - Checkpointing: each rank writes to its own shard slice, tracked by a
    ``progress_rank{r}.json`` file. On resume, workers fast-forward past the
    tokens already written.

Usage:
  torchrun --nproc_per_node=8 nova/scripts/phase1_cache_teacher.py \
      --teacher_model "Qwen/Qwen2.5-1.5B" \
      --token_path /data/nova/tokens_mixed.pt \
      --output_dir /data/cached_logits/ \
      --K 128 \
      --batch_size 32 \
      --seq_len 2048
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TOKEN_FILENAME = "tokens_mixed.npy"
INDICES_FILENAME = "cached_indices.npy"
VALUES_FILENAME = "cached_values.npy"
PROGRESS_FILENAME = "progress_rank{rank}.json"
META_FILENAME = "cache_meta.json"


# ── Distributed helpers ─────────────────────────────────────────────


def is_ddp() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def get_rank() -> int:
    return int(os.environ.get("RANK", 0))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def is_main() -> bool:
    return get_rank() == 0


def log(msg: str) -> None:
    if is_main():
        print(msg, flush=True)


def setup_distributed() -> torch.device:
    if is_ddp():
        dist.init_process_group(backend="nccl")
        local_rank = get_local_rank()
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


# ── Token loading ───────────────────────────────────────────────────


def load_tokens(token_path: Path) -> np.ndarray:
    """Load the pretokenized corpus as a contiguous int32 array."""
    if not token_path.exists():
        raise FileNotFoundError(f"Token file not found: {token_path}")

    suffix = token_path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(str(token_path), mmap_mode="r")
    elif suffix == ".pt":
        obj = torch.load(str(token_path), map_location="cpu", weights_only=False)
        if isinstance(obj, torch.Tensor):
            arr = obj.numpy()
        elif isinstance(obj, dict) and "input_ids" in obj:
            t = obj["input_ids"]
            arr = t.numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
        else:
            raise ValueError(f"Unsupported .pt contents in {token_path}")
    else:
        arr = np.memmap(token_path, dtype=np.int32, mode="r")

    arr = np.ascontiguousarray(arr.reshape(-1), dtype=np.int32)
    return arr


# ── Teacher loading ─────────────────────────────────────────────────


def load_teacher(model_name: str, device: torch.device) -> tuple:
    """Load the frozen teacher and its tokenizer in fp16."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise RuntimeError(
            "transformers not installed. Run: pip install transformers accelerate"
        )

    log(f"[teacher] loading {model_name} in fp16...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "sdpa",
    )
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    params = sum(p.numel() for p in model.parameters())
    log(f"[teacher] loaded {params / 1e9:.2f}B params on {device}")
    return model, tokenizer


# ── Cache file plumbing ─────────────────────────────────────────────


@dataclass
class CacheLayout:
    output_dir: Path
    num_tokens: int
    K: int
    vocab_size: int

    @property
    def token_path(self) -> Path:
        return self.output_dir / TOKEN_FILENAME

    @property
    def indices_path(self) -> Path:
        return self.output_dir / INDICES_FILENAME

    @property
    def values_path(self) -> Path:
        return self.output_dir / VALUES_FILENAME

    @property
    def meta_path(self) -> Path:
        return self.output_dir / META_FILENAME


def allocate_cache(layout: CacheLayout) -> None:
    """Create mem-mapped output files of the correct sizes (main rank only)."""
    layout.output_dir.mkdir(parents=True, exist_ok=True)

    token_bytes = layout.num_tokens * np.dtype(np.int32).itemsize
    indices_bytes = layout.num_tokens * layout.K * np.dtype(np.int16).itemsize
    values_bytes = layout.num_tokens * layout.K * np.dtype(np.uint16).itemsize

    for path, size in (
        (layout.token_path, token_bytes),
        (layout.indices_path, indices_bytes),
        (layout.values_path, values_bytes),
    ):
        if path.exists() and path.stat().st_size == size:
            continue
        if path.exists():
            path.unlink()
        with open(path, "wb") as f:
            f.seek(size - 1)
            f.write(b"\x00")

    meta = {
        "num_tokens": layout.num_tokens,
        "K": layout.K,
        "vocab_size": layout.vocab_size,
        "token_dtype": "int32",
        "indices_dtype": "int16",
        "values_dtype": "uint16",
        "values_encoding": "bfloat16_as_uint16",
    }
    layout.meta_path.write_text(json.dumps(meta, indent=2))

    total_gb = (token_bytes + indices_bytes + values_bytes) / (1024**3)
    log(f"[cache] allocated {total_gb:.1f} GB under {layout.output_dir}")


def open_cache_writers(layout: CacheLayout) -> tuple:
    """Open the output mem-maps for writing (called by every rank)."""
    tokens_mm = np.memmap(
        layout.token_path, dtype=np.int32, mode="r+", shape=(layout.num_tokens,)
    )
    indices_mm = np.memmap(
        layout.indices_path,
        dtype=np.int16,
        mode="r+",
        shape=(layout.num_tokens, layout.K),
    )
    values_mm = np.memmap(
        layout.values_path,
        dtype=np.uint16,
        mode="r+",
        shape=(layout.num_tokens, layout.K),
    )
    return tokens_mm, indices_mm, values_mm


# ── Shard assignment + resume state ─────────────────────────────────


def compute_shard(num_tokens: int, rank: int, world: int, seq_len: int) -> tuple[int, int]:
    """Split the token stream into contiguous ``seq_len``-aligned shards."""
    num_sequences = num_tokens // seq_len
    seqs_per_rank = num_sequences // world
    extras = num_sequences % world
    start_seq = rank * seqs_per_rank + min(rank, extras)
    count = seqs_per_rank + (1 if rank < extras else 0)
    start = start_seq * seq_len
    end = start + count * seq_len
    return start, end


def load_progress(output_dir: Path, rank: int) -> int:
    path = output_dir / PROGRESS_FILENAME.format(rank=rank)
    if not path.exists():
        return 0
    try:
        data = json.loads(path.read_text())
        return int(data.get("next_token_index", 0))
    except Exception as e:
        logger.warning(f"[rank {rank}] could not read progress file: {e}")
        return 0


def save_progress(output_dir: Path, rank: int, next_token_index: int) -> None:
    path = output_dir / PROGRESS_FILENAME.format(rank=rank)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"next_token_index": int(next_token_index)}))
    tmp.replace(path)


# ── Core forward pass ───────────────────────────────────────────────


@torch.no_grad()
def cache_batch(
    model,
    input_ids: torch.Tensor,
    K: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one forward pass and return top-K (indices, values).

    ``input_ids`` is shape ``[B, seq_len]`` (int64). Returns:
        top_idx   - int16 tensor ``[B, seq_len, K]``
        top_val   - bfloat16 tensor ``[B, seq_len, K]``
    """
    with torch.amp.autocast("cuda", dtype=torch.float16):
        outputs = model(input_ids=input_ids)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

    top_val, top_idx = torch.topk(logits, k=K, dim=-1)
    top_val_bf16 = top_val.to(torch.bfloat16)
    top_idx_i16 = top_idx.to(torch.int16)
    return top_idx_i16, top_val_bf16


def write_slice(
    tokens_mm: np.memmap,
    indices_mm: np.memmap,
    values_mm: np.memmap,
    token_start: int,
    input_ids_cpu: np.ndarray,
    top_idx_cpu: np.ndarray,
    top_val_cpu_uint16: np.ndarray,
) -> None:
    """Write a contiguous token-aligned slice to the cache files."""
    num_flat = input_ids_cpu.shape[0]
    end = token_start + num_flat
    tokens_mm[token_start:end] = input_ids_cpu
    indices_mm[token_start:end] = top_idx_cpu
    values_mm[token_start:end] = top_val_cpu_uint16


# ── Main caching loop ───────────────────────────────────────────────


def run_caching(
    args: argparse.Namespace,
    device: torch.device,
    rank: int,
    world: int,
) -> None:
    token_path = Path(args.token_path)
    output_dir = Path(args.output_dir)

    tokens_np = load_tokens(token_path)
    total_tokens = tokens_np.shape[0]
    usable_tokens = (total_tokens // args.seq_len) * args.seq_len
    if usable_tokens < total_tokens:
        log(
            f"[data] truncating {total_tokens - usable_tokens} trailing tokens "
            f"to align to seq_len={args.seq_len}"
        )
    tokens_np = tokens_np[:usable_tokens]
    log(f"[data] {usable_tokens:,} tokens loaded from {token_path}")

    model, tokenizer = load_teacher(args.teacher_model, device)
    vocab_size = getattr(model.config, "vocab_size", len(tokenizer))

    if vocab_size > 32767:
        raise RuntimeError(
            f"teacher vocab_size={vocab_size} exceeds int16 range (32767). "
            f"The cache format stores top-K indices as int16 to hit the "
            f"512 bytes/token budget. Either use a smaller-vocab teacher, "
            f"or switch the layout to int32 indices (doubles index size)."
        )

    layout = CacheLayout(
        output_dir=output_dir,
        num_tokens=usable_tokens,
        K=args.K,
        vocab_size=vocab_size,
    )

    if is_main():
        allocate_cache(layout)
    if is_ddp():
        dist.barrier()

    tokens_mm, indices_mm, values_mm = open_cache_writers(layout)

    shard_start, shard_end = compute_shard(
        usable_tokens, rank, world, args.seq_len
    )
    shard_tokens = shard_end - shard_start
    log_rank = f"[rank {rank}]"
    print(
        f"{log_rank} shard=[{shard_start:,}, {shard_end:,}) "
        f"({shard_tokens:,} tokens)",
        flush=True,
    )

    resume_from = load_progress(output_dir, rank)
    if resume_from > shard_start and resume_from < shard_end:
        print(
            f"{log_rank} resuming from token {resume_from:,} "
            f"({(resume_from - shard_start) / max(shard_tokens, 1):.1%})",
            flush=True,
        )
        cursor = resume_from
    else:
        cursor = shard_start

    tokens_per_step = args.batch_size * args.seq_len
    total_steps = math.ceil((shard_end - cursor) / tokens_per_step)

    start_time = time.time()
    last_log_time = start_time
    step = 0
    tokens_processed = 0
    last_saved_cursor = cursor

    while cursor < shard_end:
        remaining = shard_end - cursor
        take = min(tokens_per_step, remaining)
        actual_batch = take // args.seq_len
        if actual_batch == 0:
            break

        flat = tokens_np[cursor : cursor + actual_batch * args.seq_len]
        batch_np = flat.reshape(actual_batch, args.seq_len).astype(np.int64)
        batch_cpu_int32 = flat.astype(np.int32)

        input_ids = torch.from_numpy(batch_np).to(device, non_blocking=True)
        top_idx, top_val = cache_batch(model, input_ids, args.K)

        top_idx_cpu = top_idx.reshape(-1, args.K).cpu().numpy().astype(np.int16)
        top_val_bytes = (
            top_val.reshape(-1, args.K).contiguous().view(torch.uint16).cpu().numpy()
        )

        write_slice(
            tokens_mm,
            indices_mm,
            values_mm,
            cursor,
            batch_cpu_int32,
            top_idx_cpu,
            top_val_bytes,
        )

        cursor += actual_batch * args.seq_len
        tokens_processed += actual_batch * args.seq_len
        step += 1

        if step % args.log_every == 0 or cursor >= shard_end:
            now = time.time()
            window = now - last_log_time
            elapsed = now - start_time
            tput = tokens_processed / max(elapsed, 1e-6)
            remaining_tokens = shard_end - cursor
            eta_sec = remaining_tokens / max(tput, 1e-6)
            eta_h = eta_sec / 3600.0
            pct = (cursor - shard_start) / max(shard_tokens, 1)
            print(
                f"{log_rank} step={step}/{total_steps} "
                f"tokens={cursor - shard_start:,}/{shard_tokens:,} "
                f"({pct:.1%}) "
                f"tput={tput / 1e3:.1f}K tok/s "
                f"elapsed={elapsed / 60:.1f}m "
                f"eta={eta_h:.2f}h",
                flush=True,
            )
            last_log_time = now

        if step % args.checkpoint_every == 0 or cursor >= shard_end:
            tokens_mm.flush()
            indices_mm.flush()
            values_mm.flush()
            save_progress(output_dir, rank, cursor)
            last_saved_cursor = cursor

    tokens_mm.flush()
    indices_mm.flush()
    values_mm.flush()
    if last_saved_cursor != cursor:
        save_progress(output_dir, rank, cursor)

    elapsed = time.time() - start_time
    agg_tput = tokens_processed / max(elapsed, 1e-6)
    print(
        f"{log_rank} DONE: {tokens_processed:,} tokens in {elapsed / 60:.1f}m "
        f"({agg_tput / 1e3:.1f}K tok/s)",
        flush=True,
    )

    if is_ddp():
        dist.barrier()

    if is_main():
        log("[cache] all ranks finished.")


# ── CLI ─────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cache top-K teacher logits for distillation.")
    p.add_argument("--teacher_model", type=str, default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--token_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--K", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--log_every", type=int, default=1000)
    p.add_argument("--checkpoint_every", type=int, default=500)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = setup_distributed()
    rank = get_rank()
    world = get_world_size()

    log("=" * 60)
    log("NOVA PHASE-1: TEACHER LOGIT CACHING")
    log("=" * 60)
    log(f"  teacher:    {args.teacher_model}")
    log(f"  tokens:     {args.token_path}")
    log(f"  output:     {args.output_dir}")
    log(f"  K:          {args.K}")
    log(f"  batch_size: {args.batch_size}")
    log(f"  seq_len:    {args.seq_len}")
    log(f"  world:      {world}")

    try:
        run_caching(args, device, rank, world)
    finally:
        if is_ddp():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
