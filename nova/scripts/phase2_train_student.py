#!/usr/bin/env python3
"""
Phase 2: NOVA 18B-token distillation from cached teacher logits.
===============================================================
Eight-GPU DDP training loop with every optimization from the max-efficiency
spec wired in: fused BitLinear kernel, Liger patches, adaptive KL on sparse
top-K cache, WSD LR schedule, sequence-length + batch-size curricula,
gradient checkpointing, Flash Attention 2, torch.compile (max-autotune),
8-bit AdamW (bitsandbytes if present, falls back to fused AdamW), time-based
async checkpointing, and a live training log that the sidecar monitor tails.

Launch:
    torchrun --nproc_per_node=8 nova/scripts/phase2_train_student.py \
        --resume_checkpoint /data/nova/ckpt_latest.pt \
        --cache_dir /data/cached_logits/ \
        --output_dir /data/checkpoints/

Abort / auto-recovery:
    - NaN/Inf in loss            -> LR *= 0.5, reload last checkpoint
    - 3 consecutive eval regress -> LR *= 0.7
    - train-eval gap > 0.3       -> raise dropout 0.1 -> 0.2, LR *= 0.9
    - train-eval gap > 0.5       -> stop, save, report
    - throughput < 50K tok/s     -> log WARN (does not abort)
    - GPU util  < 80%            -> log WARN (does not abort)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from nova.model import Nova2_4B  # noqa: E402
from nova.model.config import NOVA_1B_QWEN, NovaConfig  # noqa: E402
from nova.training.adaptive_kl import (  # noqa: E402
    combined_distillation_loss,
    temperature_anneal,
)
from nova.training.cached_dataset import CachedDistillationDataset  # noqa: E402
from nova.training.checkpoint import CheckpointManager, CheckpointState  # noqa: E402
from nova.training.curriculum import (  # noqa: E402
    BatchSizeCurriculum,
    SequenceLengthCurriculum,
)
from nova.training.fused_bitlinear import FusedBitLinear  # noqa: E402
from nova.training.liger_integration import patch_with_liger  # noqa: E402
from nova.training.wsd_scheduler import WSDScheduler  # noqa: E402


# ── Constants ───────────────────────────────────────────────────────

TOTAL_TOKENS: int = 18_000_000_000
WARMUP_STEPS: int = 2_000
PEAK_LR: float = 3e-4
MIN_LR: float = 3e-6
STABLE_FRACTION: float = 0.80
DECAY_FRACTION: float = 0.20

PER_DEVICE_BATCH_SIZE: int = 16
WORLD_EFFECTIVE_BATCH: int = 128
MAX_SEQ_LEN: int = 2048
SEQ_WARMUP_FRACTION: float = 0.15
BS_WARMUP_FRACTION: float = 0.05
BS_START: int = 4
BS_END: int = 16

TOPK: int = 128
AKL_ALPHA: float = 0.5
CE_ALPHA: float = 0.3
DISTILL_ALPHA: float = 0.7
TEMP_START: float = 2.0
TEMP_END: float = 1.0

WEIGHT_DECAY: float = 0.1
BETA1: float = 0.9
BETA2: float = 0.95
GRAD_CLIP: float = 1.0

EVAL_EVERY_TOKENS: int = 250_000_000
CKPT_EVERY_TOKENS: int = 500_000_000
ASYNC_CKPT_EVERY_MIN: float = 15.0
EVAL_SPLIT_FRACTION: float = 0.05
EVAL_MAX_BATCHES: int = 64

DROPOUT_DEFAULT: float = 0.1
DROPOUT_ESCALATED: float = 0.2

THROUGHPUT_WARN_TOKPS: int = 50_000
GPU_UTIL_WARN_PCT: int = 80
GAP_WARN: float = 0.30
GAP_STOP: float = 0.50
EVAL_REGRESS_TRIGGER: int = 3

COST_PER_HOUR_USD: float = 15.60
BUDGET_CAP_USD: float = 1000.0

SEP = "=" * 60

logger = logging.getLogger("nova.phase2")


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


def setup_distributed() -> torch.device:
    if is_ddp():
        dist.init_process_group(backend="nccl")
        local_rank = get_local_rank()
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def all_reduce_mean(value: float, device: torch.device) -> float:
    if not is_ddp():
        return value
    t = torch.tensor([value], device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / get_world_size())


def setup_logging(output_dir: Path, rank: int) -> Path:
    log_path = output_dir / "training.log"
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        handlers: list[logging.Handler] = [
            logging.FileHandler(log_path, mode="a"),
            logging.StreamHandler(sys.stdout),
        ]
    else:
        handlers = [logging.StreamHandler(sys.stdout)]

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for h in handlers:
        h.setFormatter(fmt)
        root.addHandler(h)
    return log_path


def gpu_util_pct() -> Optional[int]:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        utils = [int(x.strip()) for x in r.stdout.strip().split("\n") if x.strip()]
        if not utils:
            return None
        return sum(utils) // len(utils)
    except Exception:
        return None


# ── Model plumbing ──────────────────────────────────────────────────


class EmbeddingDropout:
    """Mutable dropout-after-embed hook. ``p`` can be raised mid-training."""

    def __init__(self, p: float):
        self.p = p
        self.training = True

    def __call__(self, module: nn.Module, inputs: tuple, output: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0:
            return output
        return F.dropout(output, p=self.p, training=True)


def build_student(device: torch.device) -> Nova2_4B:
    config = NovaConfig(**asdict(NOVA_1B_QWEN))
    config.gradient_checkpointing = True
    model = Nova2_4B(config).to(device=device, dtype=torch.bfloat16)
    params = model.count_parameters()
    if is_main():
        logger.info(
            f"Student: {params['total_M']:.1f}M params "
            f"(vocab={config.vocab_size}, layers={config.n_layers})"
        )
    return model


def replace_bitlinear_with_fused(model: nn.Module) -> int:
    """Swap every `BitLinear` for `FusedBitLinear` in-place via state_dict copy."""
    from nova.model.bitlinear import BitLinear

    swaps = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, BitLinear):
            continue
        parent_path, attr = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model
        if parent_path:
            for part in parent_path.split("."):
                parent = getattr(parent, part)
        new = FusedBitLinear(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
        )
        new.load_state_dict(module.state_dict())
        new = new.to(device=module.weight.device, dtype=module.weight.dtype)
        setattr(parent, attr, new)
        swaps += 1

    if is_main():
        logger.info(f"Replaced {swaps} BitLinear -> FusedBitLinear")
    return swaps


def attach_embedding_dropout(model: nn.Module, p: float) -> EmbeddingDropout:
    hook = EmbeddingDropout(p)
    raw = model.module if hasattr(model, "module") else model
    raw.embed_tokens.register_forward_hook(hook)
    return hook


def build_model(device: torch.device, compile_model: bool) -> tuple[nn.Module, EmbeddingDropout]:
    """Build student -> fused bitlinear -> liger -> dropout hook -> compile -> DDP."""
    model = build_student(device)
    replace_bitlinear_with_fused(model)
    model, _ = patch_with_liger(model)
    model = model.to(device=device, dtype=torch.bfloat16)
    model.train()

    embed_drop = attach_embedding_dropout(model, DROPOUT_DEFAULT)

    if compile_model:
        try:
            model = torch.compile(model, mode="max-autotune")  # type: ignore
            if is_main():
                logger.info("torch.compile enabled (mode=max-autotune)")
        except Exception as exc:
            if is_main():
                logger.warning(f"torch.compile failed: {exc}")

    if is_ddp():
        model = DDP(
            model,
            device_ids=[get_local_rank()],
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )

    return model, embed_drop


def build_optimizer(
    model: nn.Module, lr: float, weight_decay: float, beta1: float, beta2: float,
) -> torch.optim.Optimizer:
    """8-bit AdamW via bitsandbytes if available, otherwise fused AdamW."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() < 2 or "norm" in n or "bias" in n:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    try:
        import bitsandbytes as bnb  # type: ignore

        opt = bnb.optim.AdamW8bit(groups, lr=lr, betas=(beta1, beta2))
        if is_main():
            logger.info("Optimizer: bitsandbytes AdamW8bit")
        return opt
    except Exception as exc:
        if is_main():
            logger.warning(f"bitsandbytes unavailable ({exc}); using fused AdamW")

    return torch.optim.AdamW(
        groups, lr=lr, betas=(beta1, beta2),
        fused=torch.cuda.is_available(),
    )


# ── Dataset / loaders ───────────────────────────────────────────────


def _collate(batch: list[tuple]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = torch.stack([b[0] for b in batch], dim=0)
    indices = torch.stack([b[1] for b in batch], dim=0)
    values = torch.stack([b[2] for b in batch], dim=0)
    return tokens, indices, values


def build_datasets(
    cache_dir: Path, seq_len: int, eval_fraction: float, seed: int,
) -> tuple[Subset, Subset]:
    """Load the cached dataset and split into contiguous-tail eval + shuffled train."""
    meta_path = cache_dir / "cache_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"cache_meta.json missing under {cache_dir}")
    meta = json.loads(meta_path.read_text())
    K = int(meta.get("K", TOPK))

    base = CachedDistillationDataset(
        token_path=cache_dir / "tokens_mixed.npy",
        indices_path=cache_dir / "cached_indices.npy",
        values_path=cache_dir / "cached_values.npy",
        seq_len=seq_len,
        K=K,
    )

    n = len(base)
    eval_n = max(1, int(n * eval_fraction))
    train_n = n - eval_n
    rng = np.random.default_rng(seed)
    train_indices = list(range(train_n))
    rng.shuffle(train_indices)
    eval_indices = list(range(train_n, n))

    if is_main():
        logger.info(
            f"Cache loaded: {n:,} seqs | train={train_n:,} eval={eval_n:,} K={K}"
        )
    return Subset(base, train_indices), Subset(base, eval_indices)


def build_train_loader(
    dataset: Subset, batch_size: int, world: int, rank: int, seed: int,
) -> DataLoader:
    sampler: DistributedSampler | None = None
    if world > 1:
        sampler = DistributedSampler(
            dataset, num_replicas=world, rank=rank, shuffle=True,
            seed=seed, drop_last=True,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        drop_last=True,
        persistent_workers=True,
        collate_fn=_collate,
    )


def build_eval_loader(dataset: Subset, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
        collate_fn=_collate,
    )


def slice_to_seq_len(
    tokens: torch.Tensor, indices: torch.Tensor, values: torch.Tensor, seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cur = tokens.size(1)
    if seq_len >= cur:
        return tokens, indices, values
    return tokens[:, :seq_len], indices[:, :seq_len, :], values[:, :seq_len, :]


def prepare_batch(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    seq_len: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move batch to device, slice to curriculum seq_len/bs, shift for next-token."""
    tokens, indices, values = batch
    tokens = tokens.to(device, non_blocking=True)
    indices = indices.to(device, non_blocking=True).long()
    values = values.to(device, non_blocking=True).to(torch.float32)
    tokens, indices, values = slice_to_seq_len(tokens, indices, values, seq_len)

    if batch_size < tokens.size(0):
        tokens = tokens[:batch_size]
        indices = indices[:batch_size]
        values = values[:batch_size]

    input_ids = tokens[:, :-1].contiguous()
    labels = tokens[:, 1:].contiguous()
    t_idx = indices[:, 1:, :].contiguous()
    t_val = values[:, 1:, :].contiguous()
    return input_ids, labels, t_idx, t_val


# ── Eval ────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    temperature: float,
    embed_dropout: EmbeddingDropout,
    seq_len: int,
) -> float:
    """Return average eval loss over at most EVAL_MAX_BATCHES batches."""
    model.eval()
    embed_dropout.training = False

    total = 0.0
    n = 0
    for batch in loader:
        input_ids, labels, t_idx, t_val = prepare_batch(
            batch, device, seq_len, PER_DEVICE_BATCH_SIZE,
        )
        logits = model(input_ids)
        loss, _ = combined_distillation_loss(
            logits, t_idx, t_val, labels,
            temperature=temperature,
            akl_alpha=AKL_ALPHA, ce_alpha=CE_ALPHA, distill_alpha=DISTILL_ALPHA,
        )
        total += float(loss.detach().item())
        n += 1
        if n >= EVAL_MAX_BATCHES:
            break

    model.train()
    embed_dropout.training = True
    return total / max(n, 1)


# ── Async checkpoint thread ─────────────────────────────────────────


class AsyncCheckpointer(threading.Thread):
    """Background thread that drains a queue of state dicts to disk.

    The main loop builds a CPU-side state dict (cheap relative to the save)
    and hands it off; the thread never touches CUDA so training cannot stall.
    """

    def __init__(self, save_dir: Path, prefix: str = "nova_1b_phase2"):
        super().__init__(daemon=True)
        self.save_dir = save_dir
        self.prefix = prefix
        self._cond = threading.Condition()
        self._pending: Optional[dict] = None
        self._stop = False

    def enqueue(self, payload: dict) -> None:
        with self._cond:
            self._pending = payload
            self._cond.notify()

    def stop(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify()
        self.join(timeout=60)

    def run(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and not self._stop:
                    self._cond.wait()
                if self._stop and self._pending is None:
                    return
                payload = self._pending
                self._pending = None
            try:
                step = payload.get("step", 0)
                path = self.save_dir / f"{self.prefix}_async_step{step}.pt"
                torch.save(payload, str(path))
                logger.info(f"Async checkpoint saved: {path.name}")
            except Exception as exc:
                logger.error(f"Async checkpoint failed: {exc}")


def cpu_state_dict(module: nn.Module) -> dict:
    raw = module.module if hasattr(module, "module") else module
    return {k: v.detach().cpu() for k, v in raw.state_dict().items()}


# ── Training state ──────────────────────────────────────────────────


@dataclass
class TrainState:
    step: int = 0
    tokens_seen: int = 0
    best_eval: float = float("inf")
    eval_history: list[float] = field(default_factory=list)
    eval_regress_count: int = 0
    dropout_escalated: bool = False
    last_ckpt_tokens: int = 0
    last_eval_tokens: int = 0
    last_async_ckpt_time: float = 0.0

    def state_dict(self) -> dict:
        return asdict(self)

    def load_state_dict(self, d: dict) -> None:
        for k, v in d.items():
            if hasattr(self, k):
                setattr(self, k, v)


@dataclass
class TrainCtx:
    """Everything the training loop needs, wired together by ``build_training_context``."""

    args: argparse.Namespace
    device: torch.device
    world: int
    rank: int
    total_steps: int
    model: nn.Module
    embed_drop: EmbeddingDropout
    optimizer: torch.optim.Optimizer
    scheduler: WSDScheduler
    seq_curr: SequenceLengthCurriculum
    bs_curr: BatchSizeCurriculum
    train_loader: DataLoader
    eval_loader: DataLoader
    ckpt_mgr: CheckpointManager
    async_ckpt: Optional[AsyncCheckpointer]
    train_state: TrainState
    stop_flag: dict[str, bool]
    loss_window: list[float] = field(default_factory=list)
    tokens_window: int = 0
    last_log_time: float = 0.0
    t0: float = 0.0


# ── Setup + resume ──────────────────────────────────────────────────


def compute_total_steps(total_tokens: int, effective_batch: int, seq_len: int) -> int:
    return max(1, total_tokens // (effective_batch * seq_len))


def log_config(args: argparse.Namespace, total_steps: int, log_path: Path, world: int) -> None:
    if not is_main():
        return
    logger.info(SEP)
    logger.info("NOVA PHASE-2: 18B-TOKEN DISTILLATION")
    logger.info(SEP)
    logger.info(f"  world size        : {world}")
    logger.info(f"  cache_dir         : {args.cache_dir}")
    logger.info(f"  output_dir        : {args.output_dir}")
    logger.info(f"  resume_checkpoint : {args.resume_checkpoint}")
    logger.info(f"  total tokens      : {TOTAL_TOKENS:,}")
    logger.info(f"  per-device bs     : {PER_DEVICE_BATCH_SIZE}")
    logger.info(f"  effective bs      : {WORLD_EFFECTIVE_BATCH}")
    logger.info(f"  max seq_len       : {MAX_SEQ_LEN}")
    logger.info(f"  peak LR           : {PEAK_LR}")
    logger.info(f"  total steps       : {total_steps:,}")
    logger.info(f"  log file          : {log_path}")


def resume_from_checkpoint(
    ctx: TrainCtx, resume_path: str,
) -> bool:
    """Load model/optimizer/scheduler/train_state from an explicit checkpoint path."""
    if not resume_path or not Path(resume_path).exists():
        return False

    if is_main():
        logger.info(f"Resume: loading {resume_path}")
    ckpt = torch.load(resume_path, map_location=ctx.device, weights_only=False)

    raw = ctx.model.module if hasattr(ctx.model, "module") else ctx.model
    if hasattr(raw, "_orig_mod"):
        raw._orig_mod.load_state_dict(ckpt["model_state_dict"])
    else:
        raw.load_state_dict(ckpt["model_state_dict"])
    if "optimizer_state_dict" in ckpt:
        ctx.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "scheduler_state_dict" in ckpt:
        ctx.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if "train_state" in ckpt:
        ctx.train_state.load_state_dict(ckpt["train_state"])
    elif "step" in ckpt:
        ctx.train_state.step = int(ckpt["step"])
        ctx.train_state.tokens_seen = int(ckpt.get("tokens_seen", 0))
    if "seq_curr" in ckpt:
        ctx.seq_curr.load_state_dict(ckpt["seq_curr"])
    if "bs_curr" in ckpt:
        ctx.bs_curr.load_state_dict(ckpt["bs_curr"])
    return True


def build_training_context(args: argparse.Namespace) -> TrainCtx:
    """Build model, optimizer, scheduler, loaders, checkpointer; resume if possible."""
    device = setup_distributed()
    rank = get_rank()
    world = get_world_size()

    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    log_path = setup_logging(output_dir, rank)

    total_steps = compute_total_steps(TOTAL_TOKENS, WORLD_EFFECTIVE_BATCH, MAX_SEQ_LEN)
    log_config(args, total_steps, log_path, world)

    model, embed_drop = build_model(device, compile_model=args.compile)
    optimizer = build_optimizer(model, PEAK_LR, WEIGHT_DECAY, BETA1, BETA2)
    scheduler = WSDScheduler(
        optimizer,
        total_steps=total_steps,
        peak_lr=PEAK_LR,
        min_lr=MIN_LR,
        warmup_steps=WARMUP_STEPS,
        stable_fraction=STABLE_FRACTION,
        decay_fraction=DECAY_FRACTION,
    )

    seq_curr = SequenceLengthCurriculum(
        total_steps=total_steps,
        start=128, end=MAX_SEQ_LEN,
        warmup_fraction=SEQ_WARMUP_FRACTION,
        stochastic_seq_len=True,
        seed=1234 + rank,
    )
    bs_curr = BatchSizeCurriculum(
        total_steps=total_steps,
        start=BS_START, end=BS_END,
        warmup_fraction=BS_WARMUP_FRACTION,
    )

    ckpt_mgr = CheckpointManager(
        save_dir=output_dir,
        save_every_minutes=ASYNC_CKPT_EVERY_MIN,
        keep=3,
        prefix="nova_1b_phase2",
    )
    train_state = TrainState()

    train_set, eval_set = build_datasets(
        cache_dir, MAX_SEQ_LEN, EVAL_SPLIT_FRACTION, seed=42,
    )
    train_loader = build_train_loader(
        train_set, batch_size=PER_DEVICE_BATCH_SIZE,
        world=world, rank=rank, seed=42,
    )
    eval_loader = build_eval_loader(eval_set, batch_size=PER_DEVICE_BATCH_SIZE)

    async_ckpt = AsyncCheckpointer(output_dir) if is_main() else None
    if async_ckpt is not None:
        async_ckpt.start()

    ctx = TrainCtx(
        args=args,
        device=device,
        world=world,
        rank=rank,
        total_steps=total_steps,
        model=model,
        embed_drop=embed_drop,
        optimizer=optimizer,
        scheduler=scheduler,
        seq_curr=seq_curr,
        bs_curr=bs_curr,
        train_loader=train_loader,
        eval_loader=eval_loader,
        ckpt_mgr=ckpt_mgr,
        async_ckpt=async_ckpt,
        train_state=train_state,
        stop_flag={"stop": False},
    )

    # Resume: explicit path first, fall back to latest in output_dir.
    resumed = resume_from_checkpoint(ctx, args.resume_checkpoint)
    if not resumed:
        state = ckpt_mgr.resume(model, optimizer, scheduler, None, device=device)
        if state is not None:
            train_state.step = state.step
            train_state.tokens_seen = state.tokens_seen
            train_state.best_eval = state.best_val_loss
            resumed = True

    if train_state.dropout_escalated:
        embed_drop.p = DROPOUT_ESCALATED

    if is_main():
        if resumed:
            logger.info(
                f"Resumed: step={train_state.step:,} "
                f"tokens={train_state.tokens_seen:,} dropout={embed_drop.p}"
            )
        else:
            logger.info("Starting from step 0")

    ckpt_mgr.register_model(
        model=model, optimizer=optimizer, scheduler=scheduler, scaler=None,
        step=train_state.step, tokens=train_state.tokens_seen,
    )

    ctx.t0 = time.time()
    ctx.last_log_time = ctx.t0
    train_state.last_async_ckpt_time = ctx.t0
    return ctx


# ── Abort / recovery ────────────────────────────────────────────────


def handle_nan(ctx: TrainCtx, loss: torch.Tensor) -> bool:
    """Recover from non-finite loss. Returns True if recovery succeeded."""
    logger.error(
        f"Non-finite loss at step {ctx.train_state.step}: {loss.item()} "
        "- reloading last checkpoint, LR *= 0.5"
    )
    ctx.optimizer.zero_grad(set_to_none=True)
    for pg in ctx.optimizer.param_groups:
        pg["lr"] *= 0.5
    ctx.scheduler.peak_lr *= 0.5
    recovered = ctx.ckpt_mgr.resume(
        ctx.model, ctx.optimizer, ctx.scheduler, None, device=ctx.device,
    )
    if recovered is None:
        logger.critical("No checkpoint to recover from - STOP")
        ctx.stop_flag["stop"] = True
        return False
    ctx.train_state.step = recovered.step
    ctx.train_state.tokens_seen = recovered.tokens_seen
    return True


def scale_lr(ctx: TrainCtx, factor: float, reason: str) -> None:
    logger.warning(f"{reason} - LR *= {factor}")
    for pg in ctx.optimizer.param_groups:
        pg["lr"] *= factor
    ctx.scheduler.peak_lr *= factor


def check_gap_and_regression(ctx: TrainCtx, train_loss: float, eval_loss: float) -> None:
    """Apply gap/regression-based corrections after an eval step."""
    gap = train_loss - eval_loss
    ts = ctx.train_state

    ts.eval_history.append(eval_loss)
    if len(ts.eval_history) >= 2 and eval_loss > ts.eval_history[-2]:
        ts.eval_regress_count += 1
    else:
        ts.eval_regress_count = 0

    if eval_loss < ts.best_eval:
        ts.best_eval = eval_loss

    if ts.eval_regress_count >= EVAL_REGRESS_TRIGGER:
        scale_lr(ctx, 0.7, f"eval regressed {ts.eval_regress_count}x")
        ts.eval_regress_count = 0

    if gap > GAP_STOP:
        logger.critical(f"train-eval gap {gap:.2f} > {GAP_STOP} - STOP")
        ctx.stop_flag["stop"] = True
        return

    if gap > GAP_WARN and not ts.dropout_escalated:
        logger.warning(
            f"train-eval gap {gap:.2f} > {GAP_WARN} - raising dropout "
            f"to {DROPOUT_ESCALATED}"
        )
        ctx.embed_drop.p = DROPOUT_ESCALATED
        scale_lr(ctx, 0.9, "gap warn")
        ts.dropout_escalated = True


# ── Eval + checkpoint triggers ──────────────────────────────────────


def should_eval(ts: TrainState) -> bool:
    return ts.tokens_seen - ts.last_eval_tokens >= EVAL_EVERY_TOKENS


def should_sync_ckpt(ts: TrainState) -> bool:
    return is_main() and (ts.tokens_seen - ts.last_ckpt_tokens >= CKPT_EVERY_TOKENS)


def should_async_ckpt(ts: TrainState, now: float) -> bool:
    return (
        is_main()
        and now - ts.last_async_ckpt_time >= ASYNC_CKPT_EVERY_MIN * 60.0
    )


def run_eval_step(ctx: TrainCtx, temperature: float, cur_seq: int) -> None:
    """Run periodic eval, log the [EVAL NM] line, apply corrections."""
    ts = ctx.train_state
    ts.last_eval_tokens = ts.tokens_seen

    eval_loss = evaluate(
        ctx.model, ctx.eval_loader, ctx.device,
        temperature=temperature,
        embed_dropout=ctx.embed_drop,
        seq_len=cur_seq,
    )
    eval_loss = all_reduce_mean(eval_loss, ctx.device)

    train_loss_avg = (
        sum(ctx.loss_window) / len(ctx.loss_window)
        if ctx.loss_window else float("nan")
    )
    gap = train_loss_avg - eval_loss

    now = time.time()
    elapsed = max(now - ctx.last_log_time, 1e-6)
    tok_s = ctx.tokens_window / elapsed

    ctx.loss_window.clear()
    ctx.tokens_window = 0
    ctx.last_log_time = now

    m_tokens = ts.tokens_seen // 1_000_000
    logger.info(
        f"[EVAL {m_tokens}M] train_loss={train_loss_avg:.2f} "
        f"eval_loss={eval_loss:.2f} gap={gap:.2f} "
        f"temp={temperature:.1f} seq_len={cur_seq} "
        f"tok/s={int(tok_s / 1000)}K"
    )

    if tok_s < THROUGHPUT_WARN_TOKPS:
        logger.warning(
            f"throughput {int(tok_s / 1000)}K tok/s below "
            f"{THROUGHPUT_WARN_TOKPS // 1000}K - debug"
        )
    util = gpu_util_pct()
    if util is not None and util < GPU_UTIL_WARN_PCT:
        logger.warning(f"GPU util {util}% below {GPU_UTIL_WARN_PCT}% - debug")

    check_gap_and_regression(ctx, train_loss_avg, eval_loss)


def save_sync_checkpoint(ctx: TrainCtx) -> None:
    ts = ctx.train_state
    ts.last_ckpt_tokens = ts.tokens_seen
    state = CheckpointState(
        step=ts.step,
        tokens_seen=ts.tokens_seen,
        best_val_loss=ts.best_eval,
        config={
            "total_tokens": TOTAL_TOKENS,
            "total_steps": ctx.total_steps,
            "seq_curr": ctx.seq_curr.state_dict(),
            "bs_curr": ctx.bs_curr.state_dict(),
            "train_state": ts.state_dict(),
        },
    )
    ctx.ckpt_mgr.save(ctx.model, ctx.optimizer, ctx.scheduler, None, state)


def enqueue_async_checkpoint(ctx: TrainCtx) -> None:
    ts = ctx.train_state
    ts.last_async_ckpt_time = time.time()
    if ctx.async_ckpt is None:
        return
    payload = {
        "model_state_dict": cpu_state_dict(ctx.model),
        "optimizer_state_dict": ctx.optimizer.state_dict(),
        "scheduler_state_dict": ctx.scheduler.state_dict(),
        "step": ts.step,
        "tokens_seen": ts.tokens_seen,
        "best_val_loss": ts.best_eval,
        "train_state": ts.state_dict(),
        "seq_curr": ctx.seq_curr.state_dict(),
        "bs_curr": ctx.bs_curr.state_dict(),
    }
    ctx.async_ckpt.enqueue(payload)


# ── Core train step ─────────────────────────────────────────────────


def train_step(
    ctx: TrainCtx,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> bool:
    """One forward/backward/optimizer step. Returns False on NaN that couldn't recover."""
    ts = ctx.train_state

    cur_seq = ctx.seq_curr.step()
    cur_bs = ctx.bs_curr.step()
    temperature = temperature_anneal(
        ts.step, ctx.total_steps, TEMP_START, TEMP_END,
    )

    input_ids, labels, t_idx, t_val = prepare_batch(
        batch, ctx.device, cur_seq, cur_bs,
    )

    logits = ctx.model(input_ids)
    loss, _ = combined_distillation_loss(
        logits, t_idx, t_val, labels,
        temperature=temperature,
        akl_alpha=AKL_ALPHA, ce_alpha=CE_ALPHA, distill_alpha=DISTILL_ALPHA,
    )

    if not torch.isfinite(loss):
        return handle_nan(ctx, loss)

    ctx.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(ctx.model.parameters(), GRAD_CLIP)
    ctx.optimizer.step()
    ctx.scheduler.step()

    step_tokens = input_ids.numel() * ctx.world
    ts.tokens_seen += step_tokens
    ts.step += 1
    ctx.tokens_window += step_tokens
    ctx.loss_window.append(float(loss.detach().item()))

    ctx.ckpt_mgr._step = ts.step
    ctx.ckpt_mgr._tokens = ts.tokens_seen

    if should_eval(ts):
        run_eval_step(ctx, temperature, cur_seq)
    if should_sync_ckpt(ts):
        save_sync_checkpoint(ctx)
    if should_async_ckpt(ts, time.time()):
        enqueue_async_checkpoint(ctx)

    del logits, loss
    return True


# ── Main loop + shutdown ────────────────────────────────────────────


def training_loop(ctx: TrainCtx) -> None:
    ts = ctx.train_state
    if is_main():
        logger.info(SEP)
        logger.info("TRAINING")
        logger.info(SEP)

    while ts.step < ctx.total_steps and not ctx.stop_flag["stop"]:
        sampler = getattr(ctx.train_loader, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(ts.step // max(len(ctx.train_loader), 1))

        for batch in ctx.train_loader:
            if ts.step >= ctx.total_steps or ctx.stop_flag["stop"]:
                break
            if not train_step(ctx, batch):
                break


def final_save_and_report(ctx: TrainCtx) -> None:
    if not is_main():
        return
    ts = ctx.train_state
    state = CheckpointState(
        step=ts.step,
        tokens_seen=ts.tokens_seen,
        best_val_loss=ts.best_eval,
        config={
            "total_tokens": TOTAL_TOKENS,
            "total_steps": ctx.total_steps,
            "train_state": ts.state_dict(),
            "seq_curr": ctx.seq_curr.state_dict(),
            "bs_curr": ctx.bs_curr.state_dict(),
        },
    )
    ctx.ckpt_mgr.save(ctx.model, ctx.optimizer, ctx.scheduler, None, state)

    elapsed_h = (time.time() - ctx.t0) / 3600.0
    cost = COST_PER_HOUR_USD * elapsed_h
    logger.info(SEP)
    logger.info(
        f"DONE: step={ts.step:,} tokens={ts.tokens_seen / 1e9:.2f}B "
        f"best_eval={ts.best_eval:.4f} elapsed={elapsed_h:.2f}h cost=${cost:.2f}"
    )
    logger.info(SEP)


def shutdown(ctx: TrainCtx) -> None:
    if ctx.async_ckpt is not None:
        ctx.async_ckpt.stop()
    if is_ddp():
        dist.destroy_process_group()


def install_stop_signal(ctx: TrainCtx) -> None:
    def _usr1(signum, frame):  # noqa: ARG001
        logger.warning("SIGUSR1 received - graceful stop at next step boundary")
        ctx.stop_flag["stop"] = True

    signal.signal(signal.SIGUSR1, _usr1)


def train(args: argparse.Namespace) -> None:
    ctx = build_training_context(args)
    install_stop_signal(ctx)
    try:
        training_loop(ctx)
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt - force_save + shutdown")
        if is_main():
            ctx.ckpt_mgr.force_save()
    finally:
        final_save_and_report(ctx)
        shutdown(ctx)


# ── CLI ─────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NOVA Phase-2 18B-token distillation training loop."
    )
    p.add_argument("--resume_checkpoint", type=str, default="")
    p.add_argument("--cache_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--compile", action="store_true", default=True)
    p.add_argument("--no_compile", dest="compile", action="store_false")
    return p.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
