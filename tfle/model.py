"""TFLE Model: composes ternary layers into a full network."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from .cdll import CDLLFitness
from .config import FitnessType, TFLEConfig, build_device_map, resolve_device
from .corruption import corrupt_data
from .layers import TFLELayer, generate_k_proposals
from .local_heads import TernaryLocalHead


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
                h = F.layer_norm(h, h.shape[-1:])

        # 2. Batched varying layer
        h_expanded = h.unsqueeze(0).expand(K, -1, -1)
        w_float = proposals_K.float()
        varied = torch.bmm(h_expanded, w_float)
        if layer_idx < len(model.layers) - 1:
            varied = F.layer_norm(F.relu(varied), varied.shape[-1:])

        # 3. Batched suffix: forward through layers after layer_idx
        K_val, B, F_out = varied.shape
        h_flat = varied.reshape(K_val * B, F_out)
        for i in range(layer_idx + 1, len(model.layers)):
            h_flat = model.layers[i].forward(h_flat)
            if i < len(model.layers) - 1:
                h_flat = F.layer_norm(F.relu(h_flat), h_flat.shape[-1:])

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

        # Multi-GPU setup
        if config.multi_gpu and torch.cuda.is_available() and torch.cuda.device_count() >= 2:
            self.device_map = build_device_map(config)
            self.multi_gpu = True
            self.device = torch.device(f"cuda:{config.gpu_devices[0]}")
        else:
            self.device_map = None
            self.multi_gpu = False
            if device is not None:
                self.device = torch.device(device)
            else:
                self.device = resolve_device(config)

        def _layer_device(i: int) -> torch.device:
            return self.device_map[i] if self.multi_gpu else self.device

        self.layers: list[TFLELayer] = []
        for i, (in_f, out_f) in enumerate(
            zip(config.layer_sizes[:-1], config.layer_sizes[1:])
        ):
            self.layers.append(
                TFLELayer(in_f, out_f, config, layer_idx=i, device=_layer_device(i))
            )

        # CDLL fitness evaluators (one per layer, on layer's device)
        self.cdll_fitness: list[CDLLFitness] = []
        if config.fitness_type in (FitnessType.CDLL, FitnessType.HYBRID_LOCAL):
            for i, (in_f, out_f) in enumerate(
                zip(config.layer_sizes[:-1], config.layer_sizes[1:])
            ):
                self.cdll_fitness.append(
                    CDLLFitness(in_f, out_f, i, config, _layer_device(i))
                )

        # Local classifier heads (one per layer, on layer's device)
        self.local_heads: list[TernaryLocalHead] = []
        if config.fitness_type in (FitnessType.MONO_FORWARD, FitnessType.HYBRID_LOCAL):
            num_classes = config.layer_sizes[-1]
            for i, (_, out_f) in enumerate(
                zip(config.layer_sizes[:-1], config.layer_sizes[1:])
            ):
                self.local_heads.append(
                    TernaryLocalHead(out_f, num_classes, config, _layer_device(i))
                )

    def to(self, device):
        """Move all weights and traces to device(s).

        Args:
            device: str, torch.device, or dict[int, torch.device] for multi-GPU.
        """
        if isinstance(device, dict):
            # Multi-GPU: per-layer device map
            self.device_map = device
            self.multi_gpu = True
            self.device = next(iter(device.values()))
            for i, layer in enumerate(self.layers):
                dev = device.get(i, self.device)
                layer.weights = layer.weights.to(dev)
                layer.device = dev
                if self.config.separate_pos_neg_traces:
                    layer.success_traces = layer.success_traces.to(dev)
                    layer.error_traces = layer.error_traces.to(dev)
                else:
                    layer.traces = layer.traces.to(dev)
                if i < len(self.cdll_fitness) and self.cdll_fitness[i].decoder is not None:
                    self.cdll_fitness[i].decoder = self.cdll_fitness[i].decoder.to(dev)
                    self.cdll_fitness[i].device = dev
                if i < len(self.local_heads):
                    self.local_heads[i].classifier = self.local_heads[i].classifier.to(dev)
                    self.local_heads[i].device = dev
        else:
            # Single device
            self.device = torch.device(device) if isinstance(device, str) else device
            self.multi_gpu = False
            self.device_map = None
            for layer in self.layers:
                layer.weights = layer.weights.to(self.device)
                layer.device = self.device
                if self.config.separate_pos_neg_traces:
                    layer.success_traces = layer.success_traces.to(self.device)
                    layer.error_traces = layer.error_traces.to(self.device)
                else:
                    layer.traces = layer.traces.to(self.device)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass through all layers (handles cross-device transfers)."""
        x = x.to(self.device)
        for i, layer in enumerate(self.layers):
            if self.multi_gpu:
                x = x.to(layer.device)
            x = layer.forward(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.layer_norm(x, x.shape[-1:])
        if self.multi_gpu:
            x = x.to(self.device)
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
                h = F.layer_norm(h, h.shape[-1:])
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
        """Forward through all layers, caching each layer's input on correct device."""
        if self.multi_gpu:
            inputs = [x.to(self.layers[0].device)]
        else:
            inputs = [x]
        h = inputs[0]
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                if self.multi_gpu:
                    h = h.to(layer.device)
                h = layer.forward(h)
                if i < len(self.layers) - 1:
                    h = F.relu(h)
                    h = F.layer_norm(h, h.shape[-1:])
                # Store on the NEXT layer's device (that's who needs it)
                if self.multi_gpu and i + 1 < len(self.layers):
                    inputs.append(h.to(self.layers[i + 1].device))
                else:
                    inputs.append(h)
        return inputs

    def _train_single_layer_local(
        self,
        layer_idx: int,
        layer: TFLELayer,
        layer_in: torch.Tensor,
        labels: torch.Tensor,
        temperature: float,
        mode: str,
        K: int,
    ) -> dict:
        """Train one layer with local fitness. Self-contained, no cross-layer deps."""
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
                current_out_act = F.layer_norm(F.relu(current_out), current_out.shape[-1:])
            else:
                current_out_act = current_out

        fitness_before = self._local_fitness(layer_idx, layer_in, current_out_act, labels, mode)

        # Select candidates and generate proposals
        combined_traces = layer._get_combined_traces()
        candidates = layer._select_candidates(combined_traces)

        if K > 1:
            proposals = generate_k_proposals(layer.weights, candidates, K, layer.device)
        else:
            proposals = layer._propose_flips(candidates).unsqueeze(0)

        # Evaluate each proposal with local fitness
        best_fitness = fitness_before
        best_k = -1
        for k in range(proposals.shape[0]):
            with torch.no_grad():
                w_float = proposals[k].float().to(layer.device)
                out_k = layer_in @ w_float
                if layer_idx < len(self.layers) - 1:
                    out_k = F.layer_norm(F.relu(out_k), out_k.shape[-1:])
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

        # Update local classifier head
        if mode in ("mono_forward", "hybrid_local") and layer_idx < len(self.local_heads):
            self.local_heads[layer_idx].update(current_out_act, labels)

        # Update traces
        output = layer.forward(layer_in)
        error_signal = delta <= 0
        layer._update_traces(layer_in, output, error_signal)

        layer.fitness_history.append(best_fitness if accepted else fitness_before)
        layer.acceptance_history.append(accepted)

        return {
            "accepted": accepted,
            "fitness_before": fitness_before,
            "fitness_after": best_fitness if accepted else fitness_before,
            "delta": delta,
            "n_candidates": len(candidates),
            "temperature": layer_temp,
        }

    def _train_step_local(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        temperature: float,
        mode: str,
    ) -> list[dict]:
        """Train with local fitness: CDLL, Mono-Forward, or Hybrid.

        Local fitness means each layer's fitness depends only on its own
        input and output. With multi-GPU, layers on different GPUs train
        in parallel via CUDA streams.
        """
        K = max(1, self.config.num_parallel_proposals)

        # Cache layer inputs (sequential — each depends on previous)
        layer_inputs = self._compute_layer_inputs(x)

        # Multi-GPU: layer-parallel via CUDA streams
        if self.multi_gpu:
            all_metrics: list[dict | None] = [None] * len(self.layers)

            # Cache labels on each device
            labels_per_device: dict[torch.device, torch.Tensor] = {}
            for dev in set(self.device_map.values()):
                labels_per_device[dev] = labels.to(dev)

            # Create one stream per device
            streams: dict[torch.device, torch.cuda.Stream] = {}
            for dev in set(self.device_map.values()):
                streams[dev] = torch.cuda.Stream(device=dev)

            # Launch all layers in parallel
            for layer_idx, layer in enumerate(self.layers):
                dev = self.device_map[layer_idx]
                stream = streams[dev]
                with torch.cuda.stream(stream):
                    metrics = self._train_single_layer_local(
                        layer_idx, layer, layer_inputs[layer_idx],
                        labels_per_device[dev], temperature, mode, K,
                    )
                    all_metrics[layer_idx] = metrics

            # Sync all streams
            for stream in streams.values():
                stream.synchronize()

            return all_metrics
        else:
            # Single GPU: sequential
            all_metrics_seq = []
            for layer_idx, layer in enumerate(self.layers):
                metrics = self._train_single_layer_local(
                    layer_idx, layer, layer_inputs[layer_idx],
                    labels, temperature, mode, K,
                )
                all_metrics_seq.append(metrics)
            return all_metrics_seq

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
            return self.cdll_fitness[layer_idx].compute(layer_input, layer_output, labels)
        elif mode == "mono_forward":
            return self.local_heads[layer_idx].compute_fitness(layer_output, labels)
        elif mode == "hybrid_local":
            cdll_f = self.cdll_fitness[layer_idx].compute(layer_input, layer_output, labels)
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
            current_real = layer.forward(current_real)
            current_real = F.layer_norm(F.relu(current_real), current_real.shape[-1:])
            current_corrupted = layer.forward(current_corrupted)
            current_corrupted = F.layer_norm(F.relu(current_corrupted), current_corrupted.shape[-1:])

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
        """Save checkpoint. Always saves weights to CPU for portability."""
        state = {
            "config": self.config,
            "weights": [layer.weights.cpu().clone() for layer in self.layers],
            "device_map": {k: str(v) for k, v in self.device_map.items()} if self.multi_gpu else None,
        }
        torch.save(state, path)

    def load_checkpoint(self, path: str):
        """Load checkpoint. Weights are placed on the correct device."""
        state = torch.load(path, weights_only=False, map_location="cpu")
        for i, (layer, weights) in enumerate(zip(self.layers, state["weights"])):
            dev = self.device_map[i] if self.multi_gpu else self.device
            layer.weights = weights.to(dev)
