"""Backpropagation + STE baseline for comparison with TFLE."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def ste_ternary(x: torch.Tensor) -> torch.Tensor:
    """Straight-Through Estimator for ternary quantization.

    Forward: quantize to {-1, 0, +1}
    Backward: pass gradient through as if quantization didn't happen
    """
    # Threshold-based ternary quantization
    threshold = 0.05 * x.abs().mean()
    out = torch.zeros_like(x)
    out[x > threshold] = 1.0
    out[x < -threshold] = -1.0
    # STE: gradient flows through unchanged
    return x + (out - x).detach()


class STETernaryLinear(nn.Module):
    """Linear layer with STE ternary quantization."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_ternary = ste_ternary(self.weight)
        return F.linear(x, w_ternary)


class STEBaselineModel(nn.Module):
    """Ternary MLP trained with backprop + STE for comparison."""

    def __init__(self, layer_sizes: list[int]):
        super().__init__()
        self.layers = nn.ModuleList()
        for in_f, out_f in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.layers.append(STETernaryLinear(in_f, out_f))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


@dataclass
class BaselineResult:
    train_accuracies: list[tuple[int, float]] = field(default_factory=list)
    val_accuracies: list[tuple[int, float]] = field(default_factory=list)
    final_accuracy: float = 0.0
    total_steps: int = 0
    training_time_seconds: float = 0.0
    memory_usage: dict = field(default_factory=dict)


def train_ste_baseline(
    layer_sizes: list[int],
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    total_steps: int = 100_000,
    lr: float = 0.001,
    eval_interval: int = 500,
    verbose: bool = True,
) -> tuple[STEBaselineModel, BaselineResult]:
    """Train a ternary model with backprop + STE."""
    model = STEBaselineModel(layer_sizes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    result = BaselineResult()

    # Memory estimate
    n_params = sum(p.numel() for p in model.parameters())
    result.memory_usage = {
        "weight_bytes": n_params * 4,  # FP32 weights
        "gradient_bytes": n_params * 4,  # FP32 gradients
        "optimizer_bytes": n_params * 8,  # Adam: 2 states per param
        "total_bytes": n_params * 16,
        "total_mb": n_params * 16 / (1024 * 1024),
    }

    start_time = time.time()
    step = 0
    pbar = tqdm(total=total_steps, disable=not verbose, desc="STE Baseline")

    while step < total_steps:
        for batch_x, batch_y in train_loader:
            if step >= total_steps:
                break

            if batch_x.dim() > 2:
                batch_x = batch_x.view(batch_x.size(0), -1)

            model.train()
            logits = model(batch_x)
            loss = F.cross_entropy(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % eval_interval == 0:
                model.eval()
                with torch.no_grad():
                    if val_loader is not None:
                        total_correct = 0
                        total_samples = 0
                        for vx, vy in val_loader:
                            if vx.dim() > 2:
                                vx = vx.view(vx.size(0), -1)
                            preds = model(vx).argmax(dim=-1)
                            total_correct += (preds == vy).sum().item()
                            total_samples += vy.size(0)
                        acc = total_correct / max(total_samples, 1)
                    else:
                        preds = logits.argmax(dim=-1)
                        acc = (preds == batch_y).float().mean().item()

                result.val_accuracies.append((step, acc))
                if verbose:
                    pbar.set_postfix({"acc": f"{acc:.4f}", "loss": f"{loss.item():.4f}"})

            step += 1
            pbar.update(1)

    pbar.close()

    result.total_steps = step
    result.training_time_seconds = time.time() - start_time
    if result.val_accuracies:
        result.final_accuracy = result.val_accuracies[-1][1]

    return model, result
