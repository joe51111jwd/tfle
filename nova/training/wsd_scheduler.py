"""Warmup-Stable-Decay (WSD) learning-rate scheduler for NOVA.

Three phases:
    1. Linear warmup from 0 to peak_lr over warmup_steps.
    2. Stable at peak_lr for stable_fraction of the post-warmup budget.
    3. Cosine decay from peak_lr to min_lr over the remaining decay_fraction.

The big advantage over plain cosine is that you can insert mid-training
evaluations or dataset swaps during the stable phase without the LR schedule
forcing a premature decay. The decay phase is short and aggressive, so the
model spends most of its tokens at the peak LR.

Compatible with PyTorch optimizer.step() — call scheduler.step() after each
optimizer step. Supports state_dict / load_state_dict for checkpoint resume.
"""
from __future__ import annotations

import math

import torch


class WSDScheduler:
    """Warmup + stable + cosine-decay LR schedule.

    Args:
        optimizer: PyTorch optimizer to control.
        total_steps: total number of training steps.
        peak_lr: peak learning rate (top of warmup, held during stable).
        min_lr: final learning rate at the end of decay.
        warmup_steps: linear warmup length in steps.
        stable_fraction: fraction of (total - warmup) to spend at peak.
        decay_fraction: fraction of (total - warmup) to spend in cosine decay.
            Must satisfy stable_fraction + decay_fraction <= 1.0.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        peak_lr: float = 3e-4,
        min_lr: float = 3e-6,
        warmup_steps: int = 2000,
        stable_fraction: float = 0.80,
        decay_fraction: float = 0.20,
    ):
        if total_steps <= 0:
            raise ValueError(f"total_steps must be > 0, got {total_steps}")
        if warmup_steps < 0 or warmup_steps >= total_steps:
            raise ValueError(
                f"warmup_steps must be in [0, total_steps), got {warmup_steps}"
            )
        if peak_lr <= 0 or min_lr < 0 or min_lr > peak_lr:
            raise ValueError(
                f"expected 0 < min_lr <= peak_lr, got min={min_lr} peak={peak_lr}"
            )
        if stable_fraction < 0 or decay_fraction < 0:
            raise ValueError("stable_fraction and decay_fraction must be >= 0")
        if stable_fraction + decay_fraction > 1.0 + 1e-9:
            raise ValueError(
                f"stable_fraction + decay_fraction must be <= 1.0, got "
                f"{stable_fraction + decay_fraction}"
            )

        self.optimizer = optimizer
        self.total_steps = total_steps
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.stable_fraction = stable_fraction
        self.decay_fraction = decay_fraction

        post_warmup = total_steps - warmup_steps
        self.stable_steps = int(post_warmup * stable_fraction)
        self.decay_steps = max(1, post_warmup - self.stable_steps)
        self.decay_start = warmup_steps + self.stable_steps

        self.last_step = 0
        self._set_lr(self.get_lr(0))

    def get_lr(self, step: int) -> float:
        """Compute LR for a given global step (does not mutate state)."""
        if step < 0:
            return 0.0
        if step < self.warmup_steps:
            return self.peak_lr * (step + 1) / max(self.warmup_steps, 1)
        if step < self.decay_start:
            return self.peak_lr
        if step >= self.total_steps:
            return self.min_lr

        progress = (step - self.decay_start) / max(self.decay_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.peak_lr - self.min_lr) * cosine

    def step(self, step: int | None = None) -> float:
        """Advance one step and apply the new LR to the optimizer."""
        if step is None:
            self.last_step += 1
        else:
            self.last_step = step
        lr = self.get_lr(self.last_step)
        self._set_lr(lr)
        return lr

    def _set_lr(self, lr: float) -> None:
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def get_last_lr(self) -> list[float]:
        return [pg["lr"] for pg in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        return {
            "last_step": self.last_step,
            "total_steps": self.total_steps,
            "peak_lr": self.peak_lr,
            "min_lr": self.min_lr,
            "warmup_steps": self.warmup_steps,
            "stable_fraction": self.stable_fraction,
            "decay_fraction": self.decay_fraction,
            "stable_steps": self.stable_steps,
            "decay_steps": self.decay_steps,
            "decay_start": self.decay_start,
        }

    def load_state_dict(self, state: dict) -> None:
        self.last_step = state["last_step"]
        self.total_steps = state["total_steps"]
        self.peak_lr = state["peak_lr"]
        self.min_lr = state["min_lr"]
        self.warmup_steps = state["warmup_steps"]
        self.stable_fraction = state["stable_fraction"]
        self.decay_fraction = state["decay_fraction"]
        self.stable_steps = state["stable_steps"]
        self.decay_steps = state["decay_steps"]
        self.decay_start = state["decay_start"]
        self._set_lr(self.get_lr(self.last_step))
