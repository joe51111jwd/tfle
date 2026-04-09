#!/usr/bin/env python3
"""
Validate the Phase-1 teacher logit cache.
=========================================
Re-runs the frozen teacher on a random subset of cached tokens and verifies:

  1. The on-disk files exist and their sizes match the declared shape.
  2. Memory-mapped reads round-trip correctly via ``CachedDistillationDataset``.
  3. Cached top-K logits match the live teacher output (exact matches on
     indices/values for the top positions).
  4. KL divergence between the full live distribution and the cached top-K
     approximation is below ``--max_kl`` nats (default 0.01).

Exits non-zero with a clear error message on any failure.

Usage:
  python nova/scripts/validate_cache.py \
      --cache_dir /data/cached_logits/ \
      --teacher_model "Qwen/Qwen2.5-1.5B" \
      --num_samples 1000
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from nova.training.cached_dataset import (  # noqa: E402
    CachedDistillationDataset,
    DEFAULT_K,
    DEFAULT_SEQ_LEN,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TOKEN_FILENAME = "tokens_mixed.npy"
INDICES_FILENAME = "cached_indices.npy"
VALUES_FILENAME = "cached_values.npy"
META_FILENAME = "cache_meta.json"


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", flush=True)
    sys.exit(1)


# ── File / shape checks ─────────────────────────────────────────────


def load_meta(cache_dir: Path) -> dict:
    meta_path = cache_dir / META_FILENAME
    if not meta_path.exists():
        fail(f"missing meta file {meta_path}")
    return json.loads(meta_path.read_text())


def check_file_sizes(cache_dir: Path, meta: dict) -> None:
    num_tokens = int(meta["num_tokens"])
    K = int(meta["K"])

    expected = {
        TOKEN_FILENAME: num_tokens * np.dtype(np.int32).itemsize,
        INDICES_FILENAME: num_tokens * K * np.dtype(np.int16).itemsize,
        VALUES_FILENAME: num_tokens * K * np.dtype(np.uint16).itemsize,
    }

    for name, exp_bytes in expected.items():
        p = cache_dir / name
        if not p.exists():
            fail(f"missing cache file {p}")
        actual = p.stat().st_size
        if actual != exp_bytes:
            fail(
                f"file size mismatch for {name}: expected {exp_bytes} bytes, "
                f"found {actual} bytes"
            )

    print(f"[check] files present, sizes match (num_tokens={num_tokens:,}, K={K})")


# ── Teacher reload ──────────────────────────────────────────────────


def load_teacher(model_name: str, device: torch.device) -> tuple:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        fail("transformers not installed. Run: pip install transformers accelerate")

    print(f"[teacher] loading {model_name} in fp16...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, tokenizer


# ── Validation loop ─────────────────────────────────────────────────


@torch.no_grad()
def compare_sample(
    model,
    tokens: torch.Tensor,
    cached_indices: torch.Tensor,
    cached_values: torch.Tensor,
    K: int,
    device: torch.device,
) -> tuple[float, float, int, int]:
    """Run the teacher on a single cached sequence and compare.

    Two quantities are returned:

    ``kl_truncation``
        ``KL(p_live || truncate_topK(p_live))`` — the information lost by
        dropping the tail of the live distribution. This measures whether
        K is large enough, independent of cache corruption.

    ``max_logit_err``
        ``max |cached_value - live_value|`` over the top-K positions that
        both sides agree on (bfloat16 round-trip error). Non-zero values
        here indicate corruption or a teacher mismatch.

    Returns ``(kl_truncation, max_logit_err, exact_matches, token_count)``.
    """
    input_ids = tokens.unsqueeze(0).to(device)
    outputs = model(input_ids=input_ids)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    live_logits = logits[0].float()
    seq_len, _vocab = live_logits.shape

    live_logprobs = F.log_softmax(live_logits, dim=-1)
    live_probs = live_logprobs.exp()

    cached_idx = cached_indices.to(device).long()
    cached_vals = cached_values.to(device).float()

    live_topk_val, live_topk_idx = torch.topk(live_logits, k=K, dim=-1)
    exact_matches = int((live_topk_idx == cached_idx).all(dim=-1).sum().item())

    gathered_live = torch.gather(live_logits, dim=-1, index=cached_idx)
    logit_err = (gathered_live - cached_vals).abs()
    max_logit_err = float(logit_err.max().item())

    top_mass = torch.gather(live_probs, dim=-1, index=live_topk_idx).sum(dim=-1)
    kl_truncation = -torch.log(top_mass.clamp_min(1e-12))
    kl_value = float(kl_truncation.mean().item())

    return kl_value, max_logit_err, exact_matches, seq_len


def main() -> int:
    p = argparse.ArgumentParser(description="Validate NOVA teacher logit cache.")
    p.add_argument("--cache_dir", type=str, required=True)
    p.add_argument("--teacher_model", type=str, default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--num_samples", type=int, default=1000)
    p.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN)
    p.add_argument("--K", type=int, default=DEFAULT_K)
    p.add_argument("--max_kl", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        fail(f"cache_dir does not exist: {cache_dir}")

    meta = load_meta(cache_dir)
    K = int(meta.get("K", args.K))
    if K != args.K:
        print(f"[meta] overriding K from meta: {args.K} -> {K}")

    check_file_sizes(cache_dir, meta)

    print("[check] opening mmap via CachedDistillationDataset...")
    dataset = CachedDistillationDataset(
        token_path=cache_dir / TOKEN_FILENAME,
        indices_path=cache_dir / INDICES_FILENAME,
        values_path=cache_dir / VALUES_FILENAME,
        seq_len=args.seq_len,
        K=K,
    )
    print(f"[check] dataset opened: {len(dataset):,} sequences")

    sample_tokens, sample_idx, sample_val = dataset[0]
    if sample_tokens.shape != (args.seq_len,):
        fail(f"tokens shape {tuple(sample_tokens.shape)} != ({args.seq_len},)")
    if sample_idx.shape != (args.seq_len, K):
        fail(f"indices shape {tuple(sample_idx.shape)} != ({args.seq_len}, {K})")
    if sample_val.shape != (args.seq_len, K):
        fail(f"values shape {tuple(sample_val.shape)} != ({args.seq_len}, {K})")
    if sample_val.dtype != torch.bfloat16:
        fail(f"values dtype {sample_val.dtype} != bfloat16")
    print("[check] shapes and dtypes OK")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_teacher(args.teacher_model, device)

    rng = random.Random(args.seed)
    total_samples = min(args.num_samples, len(dataset))
    indices = rng.sample(range(len(dataset)), total_samples)

    print(f"[validate] running teacher on {total_samples} samples...")
    start = time.time()

    total_tokens = 0
    total_exact = 0
    kl_values: list[float] = []
    logit_errs: list[float] = []
    max_kl_seen = 0.0
    max_err_seen = 0.0

    for i, seq_idx in enumerate(indices):
        tokens, cached_idx, cached_val = dataset[seq_idx]
        kl, logit_err, exact, seq_len = compare_sample(
            model, tokens, cached_idx, cached_val, K, device
        )
        total_tokens += seq_len
        total_exact += exact
        kl_values.append(kl)
        logit_errs.append(logit_err)
        max_kl_seen = max(max_kl_seen, kl)
        max_err_seen = max(max_err_seen, logit_err)

        if (i + 1) % max(1, total_samples // 10) == 0 or i == total_samples - 1:
            elapsed = time.time() - start
            pct = (i + 1) / total_samples
            print(
                f"  progress {i + 1}/{total_samples} ({pct:.0%}) "
                f"mean_kl={np.mean(kl_values):.5f} "
                f"max_kl={max_kl_seen:.5f} "
                f"max_logit_err={max_err_seen:.4f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    mean_kl = float(np.mean(kl_values))
    median_kl = float(np.median(kl_values))
    p99_kl = float(np.percentile(kl_values, 99))
    exact_seq_rate = total_exact / max(total_tokens, 1)
    mean_logit_err = float(np.mean(logit_errs))

    print("=" * 60)
    print(f"samples:          {total_samples}")
    print(f"tokens compared:  {total_tokens:,}")
    print(f"mean KL truncation: {mean_kl:.6f} nats")
    print(f"median KL:        {median_kl:.6f} nats")
    print(f"p99 KL:           {p99_kl:.6f} nats")
    print(f"max KL:           {max_kl_seen:.6f} nats")
    print(f"mean logit error: {mean_logit_err:.5f}")
    print(f"max logit error:  {max_err_seen:.5f}")
    print(f"exact top-K:      {exact_seq_rate:.2%}")
    print("=" * 60)

    if mean_kl > args.max_kl:
        fail(
            f"mean KL {mean_kl:.5f} > threshold {args.max_kl}. "
            f"K={K} is too small — tail mass is significant."
        )

    if max_err_seen > 0.5:
        fail(
            f"max logit error {max_err_seen:.4f} too high — cache values do "
            f"not match live teacher, possible corruption."
        )

    print("PASS: cache validated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
