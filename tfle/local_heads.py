"""Gradient-Free Local Classification Heads — Ternary, TFLE-trained.

Each hidden layer gets a tiny ternary head: layer_output @ head_weights → logits.
Trained by TFLE (same trace system, same accept/reject). No backprop anywhere.

This provides the supervised signal CDLL was missing: I(T; Y).
Combined fitness: α * local_accuracy + (1-α) * compression.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import TFLEConfig


class TernaryLocalHead:
    """Ternary classification head attached to one hidden layer.

    Weights are {-1, 0, +1}. Trained by TFLE — propose flips, evaluate
    accuracy, accept/reject with annealing. Same algorithm as main layers.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        config: TFLEConfig,
        device: torch.device,
    ):
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.device = device
        self.config = config

        # Ternary weights: (hidden_dim, num_classes)
        probs = torch.tensor([0.33, 0.34, 0.33])
        indices = torch.multinomial(probs.expand(hidden_dim * num_classes, -1), 1).squeeze()
        self.weights = (indices - 1).reshape(hidden_dim, num_classes).to(torch.int8).to(device)

        # Traces for TFLE training
        self.pos_traces = torch.zeros(hidden_dim, num_classes, dtype=torch.float16, device=device)
        self.neg_traces = torch.zeros(hidden_dim, num_classes, dtype=torch.float16, device=device)

        self.step_count = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, hidden_dim) → logits: (batch, num_classes)"""
        return x @ self.weights.to(dtype=x.dtype, device=x.device)

    def accuracy(self, x: torch.Tensor, labels: torch.Tensor) -> float:
        """Classification accuracy on this batch."""
        with torch.no_grad():
            logits = self.forward(x)
            preds = logits.argmax(dim=-1)
            return (preds == labels).float().mean().item()

    def train_step(
        self,
        layer_output: torch.Tensor,
        labels: torch.Tensor,
        temperature: float,
        flip_rate: float = 0.03,
    ) -> dict:
        """One TFLE training step on the head weights.

        Evaluate accuracy before/after flip, accept/reject with annealing.
        """
        self.step_count += 1

        # Current accuracy
        acc_before = self.accuracy(layer_output, labels)

        # Select candidates from traces
        n_weights = self.weights.numel()
        n_flip = max(1, int(n_weights * flip_rate))
        n_explore = max(1, int(n_weights * 0.003))

        scores = (self.neg_traces.float() - 0.8 * self.pos_traces.float()).flatten()
        _, guided = torch.topk(scores, min(n_flip, n_weights))
        explore = torch.randint(0, n_weights, (n_explore,), device=self.device)
        candidates = torch.cat([guided, explore]).unique()

        # Propose flips (vectorized)
        proposed = self.weights.flatten().clone().long()
        offsets = torch.randint(1, 3, (candidates.numel(),), device=self.device)
        proposed[candidates] = ((proposed[candidates] + 1 + offsets) % 3 - 1)
        proposed = proposed.reshape(self.weights.shape).to(torch.int8)

        # Evaluate
        old_weights = self.weights
        self.weights = proposed
        acc_after = self.accuracy(layer_output, labels)
        self.weights = old_weights

        delta = acc_after - acc_before

        # Accept/reject
        import math
        accepted = False
        if delta > 0:
            accepted = True
        elif temperature > 0:
            try:
                prob = math.exp(delta / max(temperature, 1e-10))
                accepted = torch.rand(1).item() < prob
            except OverflowError:
                pass

        if accepted:
            self.weights = proposed

        # Update traces
        self.pos_traces *= 0.95
        self.neg_traces *= 0.95
        rows = candidates // self.num_classes
        cols = candidates % self.num_classes
        if accepted and delta > 0:
            self.pos_traces[rows, cols] += delta
        elif not accepted:
            self.neg_traces[rows, cols] += abs(delta) if delta != 0 else 0.01

        return {
            "accepted": accepted,
            "acc_before": acc_before,
            "acc_after": acc_after if accepted else acc_before,
            "delta": delta,
        }
