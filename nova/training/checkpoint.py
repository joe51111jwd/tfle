"""Checkpoint management for NOVA pretraining.

Handles save/resume with time-based checkpointing for spot instance safety.
Saves model, optimizer, scheduler, scaler, step, and tokens_seen.
"""

from __future__ import annotations

import os
import time
import subprocess
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class CheckpointState:
    step: int
    tokens_seen: int
    best_val_loss: float
    config: dict


class CheckpointManager:
    """Time-based checkpoint manager for spot instance safety.

    Saves every `save_every_minutes` and keeps the N most recent checkpoints.
    """

    def __init__(
        self,
        save_dir: str | Path,
        save_every_minutes: float = 30.0,
        keep: int = 3,
        prefix: str = "nova_2_4b",
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_every_minutes = save_every_minutes
        self.keep = keep
        self.prefix = prefix
        self._last_save_time = time.time()

    def should_save(self) -> bool:
        elapsed = (time.time() - self._last_save_time) / 60.0
        return elapsed >= self.save_every_minutes

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        scaler: torch.amp.GradScaler,
        state: CheckpointState,
    ) -> Path:
        """Save a full checkpoint. Returns the path."""
        raw_model = model.module if hasattr(model, "module") else model

        ckpt_path = self.save_dir / f"{self.prefix}_step{state.step}.pt"
        torch.save(
            {
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "step": state.step,
                "tokens_seen": state.tokens_seen,
                "best_val_loss": state.best_val_loss,
                "config": state.config,
            },
            str(ckpt_path),
        )

        self._last_save_time = time.time()
        self.cleanup_old()
        print(f"  Checkpoint saved: {ckpt_path.name} (step {state.step})")
        return ckpt_path

    def resume(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        scaler: torch.amp.GradScaler | None = None,
        device: torch.device | str = "cpu",
    ) -> CheckpointState | None:
        """Find and load the latest checkpoint. Returns state or None."""
        latest = self._find_latest()
        if latest is None:
            print("  No checkpoint found, starting fresh")
            return None

        print(f"  Resuming from {latest.name}")
        ckpt = torch.load(str(latest), map_location=device, weights_only=False)

        raw_model = model.module if hasattr(model, "module") else model
        raw_model.load_state_dict(ckpt["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler is not None and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])

        return CheckpointState(
            step=ckpt.get("step", 0),
            tokens_seen=ckpt.get("tokens_seen", 0),
            best_val_loss=ckpt.get("best_val_loss", float("inf")),
            config=ckpt.get("config", {}),
        )

    def _find_latest(self) -> Path | None:
        """Find the checkpoint with the highest step number."""
        ckpts = sorted(
            self.save_dir.glob(f"{self.prefix}_step*.pt"),
            key=lambda p: self._extract_step(p),
        )
        return ckpts[-1] if ckpts else None

    def _extract_step(self, path: Path) -> int:
        stem = path.stem
        try:
            return int(stem.split("_step")[-1])
        except (ValueError, IndexError):
            return 0

    def cleanup_old(self):
        """Remove old checkpoints, keeping the most recent `keep` files."""
        ckpts = sorted(
            self.save_dir.glob(f"{self.prefix}_step*.pt"),
            key=lambda p: self._extract_step(p),
        )
        if len(ckpts) <= self.keep:
            return

        for old in ckpts[: len(ckpts) - self.keep]:
            old.unlink()
            print(f"  Removed old checkpoint: {old.name}")

    def push_git_lfs(self, message: str = "checkpoint update"):
        """Push checkpoints to git-lfs (optional, best-effort)."""
        try:
            cwd = str(self.save_dir)
            subprocess.run(["git", "add", "*.pt"], cwd=cwd, capture_output=True, timeout=30)
            subprocess.run(
                ["git", "commit", "-m", message],
                cwd=cwd, capture_output=True, timeout=30,
            )
            subprocess.run(["git", "push"], cwd=cwd, capture_output=True, timeout=120)
            print(f"  Git-lfs push complete")
        except Exception as e:
            print(f"  Git-lfs push failed (non-fatal): {e}")
