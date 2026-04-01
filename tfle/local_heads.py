"""Mono-Forward Local Classifier Heads.

Tiny classifier attached to each layer that trains with standard backprop.
Provides a local fitness signal without needing full-model forward passes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TFLEConfig


class LocalClassifierHead(nn.Module):
    """Tiny classifier head attached to a single layer.

    Architecture: Linear -> ReLU -> Linear (num_classes)
    Trains with backprop (~50K params). Provides local cross-entropy fitness.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        config: TFLEConfig,
        device: torch.device,
    ):
        super().__init__()
        hidden = config.local_head_hidden
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        ).to(device)
        self.optimizer = torch.optim.Adam(
            self.classifier.parameters(), lr=config.local_head_lr
        )
        self.device = device

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        return self.classifier(activations.detach())

    def compute_fitness(
        self, activations: torch.Tensor, labels: torch.Tensor
    ) -> float:
        """Compute fitness as negative cross-entropy loss (higher = better).

        Args:
            activations: (B, features) layer output.
            labels: (B,) integer class labels.

        Returns:
            Negative cross-entropy loss (scalar).
        """
        self.classifier.eval()
        with torch.no_grad():
            logits = self.forward(activations)
            loss = F.cross_entropy(logits, labels)
        return -loss.item()

    def compute_accuracy(
        self, activations: torch.Tensor, labels: torch.Tensor
    ) -> float:
        """Compute classification accuracy for monitoring.

        Args:
            activations: (B, features) layer output.
            labels: (B,) integer class labels.

        Returns:
            Accuracy in [0, 1].
        """
        self.classifier.eval()
        with torch.no_grad():
            logits = self.forward(activations)
            preds = logits.argmax(dim=-1)
            return (preds == labels).float().mean().item()

    def update(
        self, activations: torch.Tensor, labels: torch.Tensor
    ) -> float:
        """Train the classifier head for one step.

        Args:
            activations: (B, features) layer output (detached).
            labels: (B,) integer class labels.

        Returns:
            Training loss (scalar).
        """
        self.classifier.train()
        logits = self.forward(activations)
        loss = F.cross_entropy(logits, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
