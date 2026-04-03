"""Shared utilities for TFLE proving ground experiments."""

from __future__ import annotations

import json
import math
import os
import time

import torch


def setup_device() -> torch.device:
    """Auto-detect best available device, checking GPU memory."""
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        best_gpu = 0
        best_free = 0
        for i in range(n_gpus):
            free, total = torch.cuda.mem_get_info(i)
            free_gb = free / (1024 ** 3)
            total_gb = total / (1024 ** 3)
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} — "
                  f"{free_gb:.1f}/{total_gb:.1f} GB free")
            if free_gb > best_free:
                best_free = free_gb
                best_gpu = i
        if best_free < 1.0:
            print("  All GPUs have <1 GB free, falling back to CPU.")
            return torch.device("cpu")
        print(f"  Using GPU {best_gpu} ({best_free:.1f} GB free)")
        return torch.device(f"cuda:{best_gpu}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("  Using MPS (Apple Silicon)")
        return torch.device("mps")
    else:
        print("  Using CPU")
        return torch.device("cpu")


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from cross-entropy loss."""
    return math.exp(min(loss, 20.0))  # cap to avoid overflow


def save_results(results: dict | list, path: str):
    """Save results as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {path}")


def plot_loss_curves(
    tfle_log: list[dict],
    ste_log: list[dict],
    save_path: str,
    title: str = "Stage 1: Character-Level LM",
):
    """Plot TFLE vs STE loss curves on same axes. Save as PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot.")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss curves
    ax = axes[0]
    if tfle_log:
        ax.plot(
            [r["step"] for r in tfle_log],
            [r["val_loss"] for r in tfle_log],
            label="TFLE", color="blue",
        )
    if ste_log:
        ax.plot(
            [r["step"] for r in ste_log],
            [r["val_loss"] for r in ste_log],
            label="STE", color="orange",
        )
    ax.set_xlabel("Step")
    ax.set_ylabel("Validation Loss")
    ax.set_title(f"{title} — Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Perplexity curves
    ax = axes[1]
    if tfle_log:
        ax.plot(
            [r["step"] for r in tfle_log],
            [r["val_perplexity"] for r in tfle_log],
            label="TFLE", color="blue",
        )
    if ste_log:
        ax.plot(
            [r["step"] for r in ste_log],
            [r["val_perplexity"] for r in ste_log],
            label="STE", color="orange",
        )
    ax.set_xlabel("Step")
    ax.set_ylabel("Validation Perplexity")
    ax.set_title(f"{title} — Perplexity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # TFLE-specific: acceptance rate
    ax = axes[2]
    if tfle_log and "acceptance_rate" in tfle_log[0]:
        ax.plot(
            [r["step"] for r in tfle_log],
            [r["acceptance_rate"] for r in tfle_log],
            label="Acceptance Rate", color="green",
        )
        ax.set_xlabel("Step")
        ax.set_ylabel("Acceptance Rate")
        ax.set_title(f"{title} — TFLE Acceptance Rate")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Plot saved to {save_path}")


class Timer:
    def __init__(self):
        self.start_time = time.time()

    def elapsed(self) -> float:
        return time.time() - self.start_time

    def elapsed_str(self) -> str:
        s = self.elapsed()
        if s < 60:
            return f"{s:.1f}s"
        elif s < 3600:
            return f"{s / 60:.1f}m"
        return f"{s / 3600:.1f}h"
