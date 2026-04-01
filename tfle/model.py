"""TFLE Model: composes ternary layers into a full network."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from .cdll import CDLLFitness
from .config import FitnessType, TFLEConfig, resolve_device
from .corruption import corrupt_data
from .layers import TFLELayer, generate_k_proposals
from .local_heads import LocalClassifierHead


def batched_task_loss_eval(
    model: TFLEModel,
    layer_idx: int,
    proposals_K: torch.Tensor,
    x: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Evaluate K weight proposals for a single layer using cached prefix/suffix.

    Args:
        model: The TFLE model.
        layer_idx: Which layer the proposals are for.
        proposals_K: Tensor of shape (K, in_features, out_features) with proposed weights.
        x: Input batch (B, input_features).
        labels: Target labels (B,).

    Returns:
        Tensor of shape (K,) with mean cross-entropy loss for each proposal.
    """
    K = proposals_K.shape[0]
    with torch.no_grad():
        # 1. Cache prefix: forward through layers before layer_idx
        h = x
        for i in range(layer_idx):
            h = model.layers[i].forward(h)
            if i < len(model.layers) - 1:
                h = F.relu(h)

        # 2. Batched varying layer
        h_expanded = h.unsqueeze(0).expand(K, -1, -1)
        w_float = proposals_K.float()
        varied = torch.bmm(h_expanded, w_float)
        if layer_idx < len(model.layers) - 1:
            varied = F.relu(varied)

        # 3. Batched suffix: forward through layers after layer_idx
        K_val, B, F_out = varied.shape
        h_flat = varied.reshape(K_val * B, F_out)
        for i in range(layer_idx + 1, len(model.layers)):
            h_flat = model.layers[i].forward(h_flat)
            if i < len(model.layers) - 1:
                h_flat = F.relu(h_flat)

        # 4. All K losses at once
        logits = h_flat.reshape(K_val, B, -1)
        labels_exp = labels.unsqueeze(0).expand(K_val, -1)
        losses = F.cross_entropy(
            logits.reshape(K_val * B, -1),
            labels_exp.reshape(K_val * B),
            reduction='none',
        ).reshape(K_val, B).mean(dim=1)

    return losses


class TFLEModel:
    """A multi-layer ternary network trained with TFLE."""

    def __init__(self, config: TFLEConfig, device: str | None = None):
        self.config = config
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = resolve_device(config)
        self.layers: list[TFLELayer] = []
        for i, (in_f, out_f) in enumerate(
            zip(config.layer_sizes[:-1], config.layer_sizes[1:])
        ):
            self.layers.append(
                TFLELayer(in_f, out_f, config, layer_idx=i, device=self.device)
            )

        # CDLL fitness evaluators (one per layer)
        self.cdll_fitness: list[CDLLFitness] = []
        if config.fitness_type in (FitnessType.CDLL, FitnessType.HYBRID_LOCAL):
            for i, (in_f, out_f) in enumerate(
                zip(config.layer_sizes[:-1], config.layer_sizes[1:])
            ):
                self.cdll_fitness.append(
                    CDLLFitness(in_f, out_f, i, config, self.device)
                )

        # Local classifier heads (one per layer)
        self.local_heads: list[LocalClassifierHead] = []
        if config.fitness_type in (FitnessType.MONO_FORWARD, FitnessType.HYBRID_LOCAL):
            num_classes = config.layer_sizes[-1]  # output dim = num classes
            for _, out_f in zip(config.layer_sizes[:-1], config.layer_sizes[1:]):
                self.local_heads.append(
                    LocalClassifierHead(out_f, num_classes, config, self.device)
                )

    def to(self, device: str | torch.device):
        """Move all weights and traces to device (cuda/mps/cpu)."""
        self.device = torch.device(device) if isinstance(device, str) else device
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
        x = x.to(self.device)
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass and return class predictions."""
        logits = self.forward(x)
        return logits.argmax(dim=-1)

    def _train_step_task_loss_batched(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        temperature: float,
    ) -> list[dict]:
        """Batched K-proposal training for task-loss fitness.

        For each layer (sequentially):
        1. Compute current loss (baseline)
        2. Generate K proposals using generate_k_proposals
        3. Include current weights as proposal 0
        4. Evaluate all K+1 proposals with batched_task_loss_eval
        5. Pick best, accept/reject with Boltzmann
        6. Update traces
        """
        K = self.config.num_parallel_proposals
        all_metrics = []

        # Compute prefix activations for each layer
        with torch.no_grad():
            layer_inputs = [x]
            h = x
            for i in range(len(self.layers) - 1):
                h = self.layers[i].forward(h)
                h = F.relu(h)
                layer_inputs.append(h)

        for layer in self.layers:
            layer_input = layer_inputs[layer.layer_idx]
            layer.step_count += 1

            # Decay cooldowns
            expired = [k for k, v in layer.cooldown_map.items() if v <= 0]
            for k in expired:
                del layer.cooldown_map[k]
            for k in layer.cooldown_map:
                layer.cooldown_map[k] -= 1

            # Select candidates
            combined_traces = layer._get_combined_traces()
            candidates = layer._select_candidates(combined_traces)

            # Generate K proposals + baseline (current weights) as proposal 0
            proposals = generate_k_proposals(
                layer.weights, candidates, K, self.device
            )
            # Prepend current weights as proposal 0
            current_w = layer.weights.to(self.device).unsqueeze(0)
            all_proposals = torch.cat([current_w, proposals], dim=0)  # (K+1, in, out)

            # Batched evaluation
            losses = batched_task_loss_eval(
                self, layer.layer_idx, all_proposals, x, labels
            )

            # Proposal 0 = baseline
            loss_before = losses[0].item()
            fitness_before = -loss_before

            # Best of K proposals (indices 1..K)
            proposal_losses = losses[1:]
            best_k_idx = proposal_losses.argmin().item()
            best_loss = proposal_losses[best_k_idx].item()
            fitness_after = -best_loss
            delta = fitness_after - fitness_before

            # Accept/reject with Boltzmann
            layer_temp = self.config.get_temperature_for_layer(
                temperature, layer.layer_idx
            )
            accepted = layer._accept_or_reject(delta, layer_temp)

            if accepted:
                layer.weights = all_proposals[best_k_idx + 1].to(torch.int8)
            else:
                if self.config.flip_revert_on_reject.value == "cooldown":
                    for idx in candidates.tolist():
                        r = idx // layer.out_features
                        c = idx % layer.out_features
                        layer.cooldown_map[(r, c)] = self.config.cooldown_steps

                if self.config.tabu_list_size > 0:
                    flat_weights = layer.weights.flatten()
                    for idx in candidates.tolist():
                        r = idx // layer.out_features
                        c = idx % layer.out_features
                        layer.tabu_set.append((r, c, int(flat_weights[idx].item())))

            # Update traces using the correct layer input
            output = layer.forward(layer_input)
            error_signal = delta <= 0
            layer._update_traces(layer_input, output, error_signal)

            # Track fitness
            if self.config.fitness_baseline.value == "relative":
                current = fitness_after if accepted else fitness_before
                if layer.fitness_ema is None:
                    layer.fitness_ema = current
                else:
                    layer.fitness_ema = (
                        self.config.fitness_ema_decay * layer.fitness_ema
                        + (1 - self.config.fitness_ema_decay) * current
                    )

            layer.fitness_history.append(fitness_after if accepted else fitness_before)
            layer.acceptance_history.append(accepted)

            all_metrics.append({
                "accepted": accepted,
                "fitness_before": fitness_before,
                "fitness_after": fitness_after,
                "delta": delta,
                "n_candidates": len(candidates),
                "temperature": layer_temp,
            })

        return all_metrics

    def _compute_layer_inputs(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward through all layers, caching each layer's input."""
        inputs = [x]
        h = x
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                h = layer.forward(h)
                if i < len(self.layers) - 1:
                    h = F.relu(h)
                inputs.append(h)
        return inputs  # len = num_layers + 1 (last is final output)

    def _train_step_local(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        temperature: float,
        mode: str,
    ) -> list[dict]:
        """Train with local fitness: CDLL, Mono-Forward, or Hybrid.

        Local fitness means each layer's fitness depends only on its own
        input and output — no full-model forward pass needed. Layers
        could theoretically train in parallel (future optimization).

        Args:
            mode: "cdll", "mono_forward", or "hybrid_local"
        """
        K = max(1, self.config.num_parallel_proposals)
        all_metrics = []

        # Cache layer inputs (refreshed each step)
        layer_inputs = self._compute_layer_inputs(x)

        for layer_idx, layer in enumerate(self.layers):
            layer_in = layer_inputs[layer_idx]
            layer.step_count += 1

            # Decay cooldowns
            expired = [k for k, v in layer.cooldown_map.items() if v <= 0]
            for k in expired:
                del layer.cooldown_map[k]
            for k in layer.cooldown_map:
                layer.cooldown_map[k] -= 1

            # Current output
            with torch.no_grad():
                current_out = layer.forward(layer_in)
                if layer_idx < len(self.layers) - 1:
                    current_out_act = F.relu(current_out)
                else:
                    current_out_act = current_out

            # Current fitness
            fitness_before = self._local_fitness(
                layer_idx, layer_in, current_out_act, labels, mode
            )

            # Select candidates and generate proposals
            combined_traces = layer._get_combined_traces()
            candidates = layer._select_candidates(combined_traces)

            if K > 1:
                proposals = generate_k_proposals(layer.weights, candidates, K, self.device)
            else:
                proposals = layer._propose_flips(candidates).unsqueeze(0)

            # Evaluate each proposal with local fitness
            best_fitness = fitness_before
            best_k = -1
            for k in range(proposals.shape[0]):
                with torch.no_grad():
                    w_float = proposals[k].float().to(self.device)
                    out_k = layer_in @ w_float
                    if layer_idx < len(self.layers) - 1:
                        out_k = F.relu(out_k)

                f_k = self._local_fitness(layer_idx, layer_in, out_k, labels, mode)
                if f_k > best_fitness:
                    best_fitness = f_k
                    best_k = k

            delta = best_fitness - fitness_before

            # Accept/reject
            layer_temp = self.config.get_temperature_for_layer(temperature, layer_idx)
            accepted = layer._accept_or_reject(delta, layer_temp)

            if accepted and best_k >= 0:
                layer.weights = proposals[best_k].to(torch.int8)

            # Update local classifier head if using mono-forward or hybrid
            if mode in ("mono_forward", "hybrid_local") and self.local_heads:
                self.local_heads[layer_idx].update(current_out_act, labels)

            # Update traces
            output = layer.forward(layer_in)
            error_signal = delta <= 0
            layer._update_traces(layer_in, output, error_signal)

            layer.fitness_history.append(best_fitness if accepted else fitness_before)
            layer.acceptance_history.append(accepted)

            all_metrics.append({
                "accepted": accepted,
                "fitness_before": fitness_before,
                "fitness_after": best_fitness if accepted else fitness_before,
                "delta": delta,
                "n_candidates": len(candidates),
                "temperature": layer_temp,
            })

        return all_metrics

    def _local_fitness(
        self,
        layer_idx: int,
        layer_input: torch.Tensor,
        layer_output: torch.Tensor,
        labels: torch.Tensor,
        mode: str,
    ) -> float:
        """Compute local fitness for a layer."""
        if mode == "cdll":
            return self.cdll_fitness[layer_idx].compute(layer_input, layer_output)
        elif mode == "mono_forward":
            return self.local_heads[layer_idx].compute_fitness(layer_output, labels)
        elif mode == "hybrid_local":
            cdll_f = self.cdll_fitness[layer_idx].compute(layer_input, layer_output)
            mono_f = self.local_heads[layer_idx].compute_fitness(layer_output, labels)
            lam = self.config.local_head_lambda
            return mono_f + lam * cdll_f
        return 0.0

    def train_step(
        self,
        x: torch.Tensor,
        temperature: float,
        labels: Optional[torch.Tensor] = None,
    ) -> list[dict]:
        """Train all layers for one step.

        Routes to the appropriate training method based on config.fitness_type.
        Returns list of per-layer metrics.
        """
        x = x.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)

        ft = self.config.fitness_type

        # Local fitness methods (CDLL, Mono-Forward, Hybrid)
        if ft == FitnessType.CDLL and labels is not None:
            return self._train_step_local(x, labels, temperature, "cdll")
        if ft == FitnessType.MONO_FORWARD and labels is not None:
            return self._train_step_local(x, labels, temperature, "mono_forward")
        if ft == FitnessType.HYBRID_LOCAL and labels is not None:
            return self._train_step_local(x, labels, temperature, "hybrid_local")

        # Task-loss with batched K-proposals
        use_task_loss = ft == FitnessType.TASK_LOSS and labels is not None
        if use_task_loss and self.config.num_parallel_proposals > 1:
            return self._train_step_task_loss_batched(x, labels, temperature)

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
        x = x.to(self.device)
        labels = labels.to(self.device)
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
