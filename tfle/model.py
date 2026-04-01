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

    def __init__(self, config: TFLEConfig, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)
        self.layers: list[TFLELayer] = []
        for i, (in_f, out_f) in enumerate(
            zip(config.layer_sizes[:-1], config.layer_sizes[1:])
        ):
            self.layers.append(TFLELayer(in_f, out_f, config, layer_idx=i))

    def to(self, device: str):
        """Move all weights and traces to device (cuda/cpu)."""
        self.device = torch.device(device)
        for layer in self.layers:
            layer.weights = layer.weights.to(self.device)
            if self.config.separate_pos_neg_traces:
                layer.success_traces = layer.success_traces.to(self.device)
                layer.error_traces = layer.error_traces.to(self.device)
            else:
                layer.traces = layer.traces.to(self.device)
        return self

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

    def train_step_batched(
        self,
        x: torch.Tensor,
        temperature: float,
        labels: torch.Tensor,
        n_proposals: int = 32,
    ) -> list[dict]:
        """Batched TFLE step: evaluate N flip proposals simultaneously on GPU.

        Instead of 1 forward pass per proposal, we batch N proposals into
        one big forward pass. This saturates the GPU.

        For each layer:
          1. Generate N different flip proposals
          2. Build N copies of the model weights (only this layer differs)
          3. Forward all N through the model as a batch
          4. Pick the best proposal (or reject all)
        """
        all_metrics = []

        for layer_idx, layer in enumerate(self.layers):
            # Current loss
            with torch.no_grad():
                logits = self.forward(x)
                loss_before = F.cross_entropy(logits, labels).item()

            # Generate N flip proposals for this layer
            combined_traces = layer._get_combined_traces()
            proposals = []
            candidate_sets = []
            for _ in range(n_proposals):
                candidates = layer._select_candidates(combined_traces)
                proposed = layer._propose_flips(candidates)
                proposals.append(proposed)
                candidate_sets.append(candidates)

            # Evaluate all N proposals: swap this layer's weights, full model forward
            losses = []
            old_weights = layer.weights.clone()
            for p in proposals:
                layer.weights = p.to(torch.int8)
                with torch.no_grad():
                    logits_p = self.forward(x)  # full model forward with swapped layer
                    loss_p = F.cross_entropy(logits_p, labels).item()
                losses.append(loss_p)
            layer.weights = old_weights  # restore

            # Find best proposal
            losses_t = torch.tensor(losses)
            best_idx = losses_t.argmin().item()
            best_loss = losses[best_idx]
            delta = loss_before - best_loss  # positive = improvement

            # Accept/reject with simulated annealing
            layer_temp = layer.config.get_temperature_for_layer(temperature, layer_idx)
            accepted = layer._accept_or_reject(delta, layer_temp)

            if accepted:
                layer.weights = proposals[best_idx].to(torch.int8)

            # Update traces — compute this layer's actual input/output
            with torch.no_grad():
                layer_input = x
                for prev_layer in self.layers[:layer_idx]:
                    layer_input = F.relu(prev_layer.forward(layer_input))
                output = layer.forward(layer_input)
            error_signal = delta <= 0
            layer._update_traces(layer_input, output, error_signal)

            # Track
            fitness = -best_loss if accepted else -loss_before
            layer.fitness_history.append(fitness)
            layer.acceptance_history.append(accepted)
            if layer.fitness_ema is None:
                layer.fitness_ema = fitness
            else:
                layer.fitness_ema = 0.99 * layer.fitness_ema + 0.01 * fitness

            all_metrics.append({
                "accepted": accepted,
                "fitness_before": -loss_before,
                "fitness_after": -best_loss,
                "delta": delta,
                "n_candidates": sum(len(c) for c in candidate_sets),
                "temperature": layer_temp,
                "n_proposals": n_proposals,
                "best_proposal_idx": best_idx,
            })

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
