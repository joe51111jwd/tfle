"""TFLE Model: composes ternary layers into a full network."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .config import TFLEConfig
from .corruption import corrupt_data
from .layers import TFLELayer


class TFLEModel:
    """A multi-layer ternary network trained with TFLE."""

    def __init__(self, config: TFLEConfig):
        self.config = config
        self.layers: list[TFLELayer] = []
        for i, (in_f, out_f) in enumerate(
            zip(config.layer_sizes[:-1], config.layer_sizes[1:])
        ):
            self.layers.append(TFLELayer(in_f, out_f, config, layer_idx=i))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass through all layers."""
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass and return class predictions."""
        logits = self.forward(x)
        return logits.argmax(dim=-1)

    def train_step(
        self,
        x: torch.Tensor,
        temperature: float,
        labels: Optional[torch.Tensor] = None,
    ) -> list[dict]:
        """Train all layers for one step.

        Returns list of per-layer metrics.
        """
        from .config import FitnessType

        use_task_loss = (
            self.config.fitness_type == FitnessType.TASK_LOSS
            and labels is not None
        )

        x_corrupted = corrupt_data(x, self.config, labels)

        all_metrics = []
        current_real = x
        current_corrupted = x_corrupted

        for layer in self.layers:
            # Build task loss closure if using task-aware fitness
            task_loss_fn = None
            if use_task_loss:
                # Capture x and labels for the closure
                _x, _labels = x, labels
                def task_loss_fn(_x=_x, _labels=_labels):
                    with torch.no_grad():
                        logits = self.forward(_x)
                        return F.cross_entropy(logits, _labels).item()

            metrics = layer.train_step(
                current_real, current_corrupted, temperature,
                task_loss_fn=task_loss_fn,
            )
            all_metrics.append(metrics)

            # Forward for next layer input (using current weights, post-update)
            current_real = F.relu(layer.forward(current_real))
            current_corrupted = F.relu(layer.forward(current_corrupted))

        return all_metrics

    def evaluate(self, x: torch.Tensor, labels: torch.Tensor) -> dict:
        """Evaluate model accuracy."""
        with torch.no_grad():
            logits = self.forward(x)
            preds = logits.argmax(dim=-1)
            accuracy = (preds == labels).float().mean().item()
            loss = F.cross_entropy(logits, labels).item()
        return {"accuracy": accuracy, "loss": loss}

    def get_total_params(self) -> int:
        return sum(layer.in_features * layer.out_features for layer in self.layers)

    def get_memory_usage_bytes(self) -> dict:
        """Estimate memory usage."""
        total_params = self.get_total_params()
        weight_bytes = total_params  # int8 = 1 byte per param
        trace_bytes = total_params * 2  # float16 = 2 bytes
        if self.config.separate_pos_neg_traces:
            trace_bytes *= 2
        return {
            "weight_bytes": weight_bytes,
            "trace_bytes": trace_bytes,
            "total_bytes": weight_bytes + trace_bytes,
            "total_mb": (weight_bytes + trace_bytes) / (1024 * 1024),
        }

    def save_checkpoint(self, path: str):
        state = {
            "config": self.config,
            "weights": [layer.weights.clone() for layer in self.layers],
        }
        torch.save(state, path)

    def load_checkpoint(self, path: str):
        state = torch.load(path, weights_only=False)
        for layer, weights in zip(self.layers, state["weights"]):
            layer.weights = weights
