"""Convolutional TFLE model for CIFAR-10."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .config import TFLEConfig
from .conv_layers import TFLEConvLayer
from .corruption import corrupt_data
from .layers import TFLELayer


class TFLEConvModel:
    """Ternary CNN trained with TFLE for image classification."""

    def __init__(self, config: TFLEConfig, n_classes: int = 10):
        self.config = config
        self.n_classes = n_classes

        # Conv layers: [3, 32, 32] -> [32, 16, 16] -> [64, 8, 8] -> [128, 4, 4]
        self.conv1 = TFLEConvLayer(3, 32, 3, config, layer_idx=0, padding=1)
        self.conv2 = TFLEConvLayer(32, 64, 3, config, layer_idx=1, padding=1)
        self.conv3 = TFLEConvLayer(64, 128, 3, config, layer_idx=2, padding=1)

        # FC layers: 128*4*4 = 2048 -> 256 -> n_classes
        self.fc1 = TFLELayer(2048, 256, config, layer_idx=3)
        self.fc2 = TFLELayer(256, n_classes, config, layer_idx=4)

        self.conv_layers = [self.conv1, self.conv2, self.conv3]
        self.fc_layers = [self.fc1, self.fc2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, 3, 32, 32] -> [batch, n_classes]"""
        x = F.relu(self.conv1.forward(x))
        x = F.max_pool2d(x, 2)  # -> [B, 32, 16, 16]
        x = F.relu(self.conv2.forward(x))
        x = F.max_pool2d(x, 2)  # -> [B, 64, 8, 8]
        x = F.relu(self.conv3.forward(x))
        x = F.max_pool2d(x, 2)  # -> [B, 128, 4, 4]
        x = x.flatten(start_dim=1)  # -> [B, 2048]
        x = F.relu(self.fc1.forward(x))
        x = self.fc2.forward(x)
        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).argmax(dim=-1)

    def train_step(
        self,
        x: torch.Tensor,
        temperature: float,
        labels: Optional[torch.Tensor] = None,
    ) -> list[dict]:
        """Train all layers for one step using contrastive fitness."""
        all_metrics = []

        # Train conv layers
        current_real = x
        current_corrupted = corrupt_data(x, self.config, labels)

        for conv in self.conv_layers:
            metrics = conv.train_step(current_real, current_corrupted, temperature)
            all_metrics.append(metrics)
            current_real = F.max_pool2d(F.relu(conv.forward(current_real)), 2)
            current_corrupted = F.max_pool2d(F.relu(conv.forward(current_corrupted)), 2)

        # Flatten for FC layers
        real_flat = current_real.flatten(start_dim=1)
        corrupt_flat = current_corrupted.flatten(start_dim=1)

        for fc in self.fc_layers:
            metrics = fc.train_step(real_flat, corrupt_flat, temperature)
            all_metrics.append(metrics)
            real_flat = F.relu(fc.forward(real_flat))
            corrupt_flat = F.relu(fc.forward(corrupt_flat))

        return all_metrics

    def evaluate(self, x: torch.Tensor, labels: torch.Tensor) -> dict:
        with torch.no_grad():
            logits = self.forward(x)
            preds = logits.argmax(dim=-1)
            accuracy = (preds == labels).float().mean().item()
            loss = F.cross_entropy(logits, labels).item()
        return {"accuracy": accuracy, "loss": loss}

    def get_total_params(self) -> int:
        total = sum(c.n_weights for c in self.conv_layers)
        total += sum(fc.in_features * fc.out_features for fc in self.fc_layers)
        return total

    def get_memory_usage_bytes(self) -> dict:
        total_params = self.get_total_params()
        weight_bytes = total_params
        trace_bytes = total_params * 2
        if self.config.separate_pos_neg_traces:
            trace_bytes *= 2
        return {
            "weight_bytes": weight_bytes,
            "trace_bytes": trace_bytes,
            "total_bytes": weight_bytes + trace_bytes,
            "total_mb": (weight_bytes + trace_bytes) / (1024 * 1024),
        }
