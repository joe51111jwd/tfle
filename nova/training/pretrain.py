#!/usr/bin/env python3
"""NOVA 2.4B STE pretraining loop.

Supports:
  - DDP (torchrun --nproc_per_node=N pretrain.py) for 8xA100
  - DataParallel (python pretrain.py) for 4x5090
  - Auto-detects which mode based on env vars
  - Mixed precision bfloat16
  - Gradient checkpointing (toggle)
  - Streaming FineWeb-Edu + StarCoder data
  - Time-based checkpointing for spot instance safety

Launch:
  torchrun --nproc_per_node=8 nova/training/pretrain.py   # DDP
  python nova/training/pretrain.py                         # DataParallel
"""

from __future__ import annotations

import os
import sys
import json
import math
import time
import subprocess
from pathlib import Path
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nova.model import Nova2_4B, NOVA_2_4B
from nova.model.config import NovaConfig
from nova.training.data_loader import PretrainDataLoader
from nova.training.checkpoint import CheckpointManager, CheckpointState
from nova.training.tokenizer_setup import VOCAB_SIZE

# ── Defaults ──────────────────────────────────────────────

CKPT_DIR = Path(os.environ.get("NOVA_CKPT_DIR", str(PROJECT_ROOT / "checkpoints" / "nova_2_4b")))
LOG_DIR = Path(os.environ.get("NOVA_LOG_DIR", str(PROJECT_ROOT / "logs")))

DEFAULTS = {
    "lr": 3e-4,
    "min_lr": 3e-5,
    "warmup_steps": 2000,
    "total_steps": 200_000,
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,
    "seq_len": 2048,
    "batch_size": 8,
    "buffer_size": 10_000,
    "eval_every": 1000,
    "log_every": 100,
    "save_every_minutes": 30.0,
    "gradient_checkpointing": True,
    "eval_batches": 50,
}

SEP = "=" * 60


# ── Helpers ───────────────────────────────────────────────

def is_ddp() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def get_rank() -> int:
    return int(os.environ.get("RANK", 0))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def is_main_process() -> bool:
    return get_rank() == 0


def log(msg: str):
    if is_main_process():
        print(msg, flush=True)


def gpu_stats() -> str:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        lines = r.stdout.strip().split("\n")
        parts = [l.split(", ") for l in lines]
        utils = [int(p[0]) for p in parts]
        mems = [f"{int(p[1])}/{int(p[2])}" for p in parts]
        return f"GPU: {utils}% | VRAM: {mems}MB"
    except Exception:
        return ""


def get_lr(step: int, warmup: int, total: int, lr: float, min_lr: float) -> float:
    """Warmup + cosine decay schedule."""
    if step < warmup:
        return lr * step / max(warmup, 1)
    if step >= total:
        return min_lr
    progress = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer: torch.optim.Optimizer, lr: float):
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def get_raw_model(model: nn.Module) -> nn.Module:
    if hasattr(model, "module"):
        return model.module
    return model


# ── Setup ─────────────────────────────────────────────────

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = get_local_rank()
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}")


def setup_dp() -> torch.device:
    return torch.device("cuda:0")


def build_model(device: torch.device, gradient_checkpointing: bool) -> nn.Module:
    config = NovaConfig(
        vocab_size=VOCAB_SIZE,
        gradient_checkpointing=gradient_checkpointing,
    )
    model = Nova2_4B(config).to(device)
    params = model.count_parameters()
    log(f"  Model: {params['total_M']:.1f}M params ({params['total_B']:.2f}B)")
    log(f"  Pattern: {model.layer_pattern}")
    return model


def wrap_model(model: nn.Module, device: torch.device) -> nn.Module:
    n_gpus = torch.cuda.device_count()

    if is_ddp():
        model = DDP(model, device_ids=[get_local_rank()], find_unused_parameters=False)
        log(f"  DDP on {get_world_size()} GPUs")
    elif n_gpus > 1:
        model = nn.DataParallel(model)
        log(f"  DataParallel on {n_gpus} GPUs")
    else:
        log(f"  Single GPU")

    return model


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.AdamW:
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2 or "norm" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": cfg["weight_decay"]},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        param_groups,
        lr=cfg["lr"],
        betas=(cfg["beta1"], cfg["beta2"]),
        fused=torch.cuda.is_available(),
    )


# ── Eval ──────────────────────────────────────────────────

def evaluate(
    model: nn.Module,
    device: torch.device,
    cfg: dict,
) -> float:
    """Run evaluation on a fresh validation stream. Returns avg loss."""
    model.eval()
    val_loader = PretrainDataLoader(
        seq_len=cfg["seq_len"],
        batch_size=cfg["batch_size"],
        buffer_size=2000,
        split="val",
        seed=9999,
    )

    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader.stream():
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits.reshape(-1, VOCAB_SIZE),
                    labels.reshape(-1),
                    ignore_index=-100,
                )

            total_loss += loss.item()
            n_batches += 1
            if n_batches >= cfg["eval_batches"]:
                break

    model.train()
    return total_loss / max(n_batches, 1)


# ── Training Loop ─────────────────────────────────────────

def train(cfg: dict | None = None):
    cfg = {**DEFAULTS, **(cfg or {})}

    # Setup device and distributed
    if is_ddp():
        device = setup_ddp()
    else:
        device = setup_dp()

    log(f"\n{SEP}")
    log(f"NOVA 2.4B STE PRETRAINING")
    log(f"{SEP}\n")
    log(f"  Config: {json.dumps(cfg, indent=2)}")

    # Build model
    model = build_model(device, cfg["gradient_checkpointing"])
    model = wrap_model(model, device)

    # Optimizer (no scheduler -- we do manual LR)
    optimizer = build_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda")

    # Dummy scheduler for checkpoint compatibility
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

    # Checkpoint manager
    ckpt_mgr = CheckpointManager(
        save_dir=CKPT_DIR,
        save_every_minutes=cfg["save_every_minutes"],
    )

    # Resume if checkpoint exists
    start_step = 0
    tokens_seen = 0
    best_val_loss = float("inf")

    state = ckpt_mgr.resume(model, optimizer, scheduler, scaler, device=device)
    if state is not None:
        start_step = state.step
        tokens_seen = state.tokens_seen
        best_val_loss = state.best_val_loss
        log(f"  Resumed at step {start_step}, {tokens_seen:,} tokens, best_val={best_val_loss:.4f}")

    # Data loader
    loader = PretrainDataLoader(
        seq_len=cfg["seq_len"],
        batch_size=cfg["batch_size"],
        buffer_size=cfg["buffer_size"],
        seed=42 + get_rank(),
    )

    # Logging
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Training
    model.train()
    running_loss = 0.0
    t0 = time.time()
    step = start_step

    log(f"\n{SEP}")
    log(f"Training from step {start_step} to {cfg['total_steps']}")
    log(f"{SEP}\n")

    for batch in loader.stream():
        if step >= cfg["total_steps"]:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        batch_tokens = input_ids.numel()

        # LR schedule
        current_lr = get_lr(step, cfg["warmup_steps"], cfg["total_steps"],
                            cfg["lr"], cfg["min_lr"])
        set_lr(optimizer, current_lr)

        # Forward + backward
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                labels.reshape(-1),
                ignore_index=-100,
            )

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        step += 1
        tokens_seen += batch_tokens * get_world_size()
        running_loss += loss.item()

        # Logging
        if step % cfg["log_every"] == 0:
            avg_loss = running_loss / cfg["log_every"]
            elapsed = time.time() - t0
            tps = tokens_seen / max(elapsed, 1)
            gs = gpu_stats() if step % (cfg["log_every"] * 5) == 0 else ""

            log(f"  Step {step:6d}/{cfg['total_steps']} | "
                f"Loss {loss.item():.4f} | Avg {avg_loss:.4f} | "
                f"LR {current_lr:.2e} | "
                f"{tps:,.0f} tok/s | "
                f"{tokens_seen/1e9:.2f}B tokens {gs}")
            running_loss = 0.0

        # Eval
        if step % cfg["eval_every"] == 0 and is_main_process():
            val_loss = evaluate(model, device, cfg)
            improved = val_loss < best_val_loss
            best_val_loss = min(best_val_loss, val_loss)
            log(f"  EVAL step {step}: val_loss={val_loss:.4f} "
                f"best={best_val_loss:.4f} {'(new best)' if improved else ''}")

        # Time-based checkpoint
        if ckpt_mgr.should_save() and is_main_process():
            ckpt_state = CheckpointState(
                step=step,
                tokens_seen=tokens_seen,
                best_val_loss=best_val_loss,
                config=cfg,
            )
            ckpt_mgr.save(model, optimizer, scheduler, scaler, ckpt_state)

    # Final checkpoint
    if is_main_process():
        ckpt_state = CheckpointState(
            step=step,
            tokens_seen=tokens_seen,
            best_val_loss=best_val_loss,
            config=cfg,
        )
        final_path = ckpt_mgr.save(model, optimizer, scheduler, scaler, ckpt_state)

        elapsed = time.time() - t0
        text_r, code_r = loader.get_ratio()
        summary = {
            "steps": step,
            "tokens_seen": tokens_seen,
            "best_val_loss": best_val_loss,
            "time_hours": elapsed / 3600,
            "text_ratio": text_r,
            "code_ratio": code_r,
            "final_checkpoint": str(final_path),
        }
        with open(LOG_DIR / "pretrain_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        log(f"\n{SEP}")
        log(f"DONE: {step} steps | {tokens_seen/1e9:.2f}B tokens | "
            f"best_val={best_val_loss:.4f} | {elapsed/3600:.1f}h")
        log(f"Text/Code: {text_r:.0%}/{code_r:.0%}")
        log(f"Checkpoint: {final_path}")
        log(f"{SEP}")

    # Cleanup DDP
    if is_ddp():
        dist.destroy_process_group()


if __name__ == "__main__":
    train()
