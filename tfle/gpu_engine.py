"""SearchParallelEngine — auto-detect GPUs, assign layer pools, coordinate proposals.

Core loop per step:
  1. Scatter eval batch to all GPUs
  2. Each GPU: propose flips -> forward on full batch -> compute fitness -> return delta
  3. Gather deltas, pick best, Boltzmann accept, broadcast
  4. Update traces on layer's device

Scaling:
  GPUs <= num_layers:  1 GPU per layer, parallel across layers
  GPUs > num_layers:   K = GPUs / num_layers proposals per layer

Fitness:
  Wake phase: CDLL (compression-driven, local per layer)
  Sleep phase: micro-critic (representation quality, local per layer)
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import TFLEConfig, resolve_device, build_device_map
from .cdll import CDLLFitness
from .layers import TFLELayer, generate_k_proposals
from .model import TFLEModel
from .swt import SleepWakeScheduler
from .annealing import TemperatureScheduler


class SearchParallelEngine:
    """GPU-parallel TFLE training engine.

    Auto-detects GPUs, distributes layers, scales proposals to fill VRAM.
    Integrates CDLL (wake) and micro-critic (sleep) fitness.
    """

    def __init__(self, model: TFLEModel, config: TFLEConfig):
        self.model = model
        self.config = config
        self.device = model.device

        # GPU inventory
        self.n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.devices = [torch.device(f"cuda:{i}") for i in range(self.n_gpus)] or [self.device]

        # Layer -> GPU assignment
        if model.multi_gpu and model.device_map:
            self.layer_devices = model.device_map
        elif self.n_gpus > 0:
            self.layer_devices = build_device_map(config)
        else:
            self.layer_devices = {i: self.device for i in range(len(model.layers))}

        # Proposals per layer
        n_layers = len(model.layers)
        if config.proposals_per_layer > 0:
            self.K = config.proposals_per_layer
        elif self.n_gpus > n_layers:
            self.K = max(1, self.n_gpus // n_layers) * config.num_parallel_proposals
        else:
            self.K = config.num_parallel_proposals

        # Auto-scale eval batch to fill VRAM
        self.eval_batch_size = config.fitness_eval_batch_size
        if self.n_gpus > 0 and config.vram_target_fraction > 0:
            self.eval_batch_size = self._auto_batch_size()

        # CDLL fitness evaluators
        self.cdll: list[CDLLFitness] = []
        for i, (in_f, out_f) in enumerate(
            zip(config.layer_sizes[:-1], config.layer_sizes[1:])
        ):
            dev = self.layer_devices.get(i, self.device)
            self.cdll.append(CDLLFitness(in_f, out_f, i, config, dev))

        # SWT scheduler
        self.swt = SleepWakeScheduler(model, config, self.device) if config.swt_enabled else None

        # Temperature
        self.scheduler = TemperatureScheduler(config)

        # Stats
        self.step_count = 0
        self.wake_steps_done = 0
        self.sleep_steps_done = 0

    def _auto_batch_size(self) -> int:
        """Estimate eval batch size to fill target fraction of smallest GPU VRAM."""
        min_vram = min(
            torch.cuda.get_device_properties(i).total_memory for i in range(self.n_gpus)
        )
        target_bytes = int(min_vram * self.config.vram_target_fraction)
        # Each eval needs: batch * max_layer_width * 4 bytes (float32) * 2 (input + output)
        max_width = max(self.config.layer_sizes)
        bytes_per_sample = max_width * 4 * 2
        batch = target_bytes // max(bytes_per_sample * self.K, 1)
        return max(64, min(batch, 8192))

    def train_step(self, x: torch.Tensor, labels: torch.Tensor) -> dict:
        """One training step. Handles wake/sleep dispatch.

        Returns dict with per-layer metrics + global stats.
        """
        temperature = self.scheduler.get_temperature()

        if self.swt and not self.swt.is_wake():
            # SLEEP: train on replay buffer with critic fitness
            metrics = self._sleep_step(temperature)
            self.swt.step()
            self.sleep_steps_done += 1

            # End of sleep cycle
            if self.swt.is_wake():
                self.swt.on_sleep_end()

            return {"phase": "sleep", "layers": metrics, "temperature": temperature,
                    "step": self.step_count}

        # WAKE: train on new data with CDLL fitness
        metrics = self._wake_step(x, labels, temperature)
        self.scheduler.step_update(
            sum(m.get("delta", 0) for m in metrics) / max(len(metrics), 1)
        )

        if self.swt:
            # Collect layer activations for critic training
            layer_acts = self._get_layer_activations(x)
            self.swt.on_wake_step(x, labels, layer_acts)
            self.swt.step()

        self.step_count += 1
        self.wake_steps_done += 1

        return {"phase": "wake", "layers": metrics, "temperature": temperature,
                "step": self.step_count}

    def _wake_step(self, x: torch.Tensor, labels: torch.Tensor, temperature: float) -> list[dict]:
        """Wake: CDLL fitness, search-parallel across layers."""
        x = x.to(self.device)
        labels = labels.to(self.device)

        # Cache layer inputs (sequential — each depends on previous)
        layer_inputs = self.model._compute_layer_inputs(x)

        all_metrics: list[dict] = []

        # If multi-GPU, use CUDA streams for layer parallelism
        if self.model.multi_gpu and self.n_gpus >= 2:
            all_metrics = [{}] * len(self.model.layers)
            streams = {dev: torch.cuda.Stream(device=dev) for dev in set(self.layer_devices.values())}

            for layer_idx, layer in enumerate(self.model.layers):
                dev = self.layer_devices[layer_idx]
                stream = streams[dev]
                with torch.cuda.stream(stream):
                    m = self._train_layer_cdll(
                        layer_idx, layer, layer_inputs[layer_idx],
                        labels.to(dev), temperature
                    )
                    all_metrics[layer_idx] = m

            for stream in streams.values():
                stream.synchronize()
        else:
            for layer_idx, layer in enumerate(self.model.layers):
                m = self._train_layer_cdll(
                    layer_idx, layer, layer_inputs[layer_idx],
                    labels, temperature
                )
                all_metrics.append(m)

        return all_metrics

    def _sleep_step(self, temperature: float) -> list[dict]:
        """Sleep: micro-critic fitness on replay buffer."""
        if not self.swt:
            return []

        batch = self.swt.get_sleep_batch(batch_size=self.eval_batch_size)
        if batch is None:
            return []

        x, labels = batch
        x = x.to(self.device)
        labels = labels.to(self.device)
        layer_inputs = self.model._compute_layer_inputs(x)

        all_metrics = []
        for layer_idx, layer in enumerate(self.model.layers):
            m = self._train_layer_critic(
                layer_idx, layer, layer_inputs[layer_idx], temperature
            )
            all_metrics.append(m)

        return all_metrics

    def _train_layer_cdll(
        self, layer_idx: int, layer: TFLELayer,
        layer_in: torch.Tensor, labels: torch.Tensor, temperature: float,
    ) -> dict:
        """Train one layer with CDLL fitness + K proposals."""
        layer.step_count += 1
        dev = layer.device

        # Current output + CDLL fitness
        with torch.no_grad():
            current_out = layer.forward(layer_in)
            if layer_idx < len(self.model.layers) - 1:
                current_out = F.relu(current_out)
        fitness_before = self.cdll[layer_idx].compute(layer_in, current_out)

        # Generate K proposals
        combined_traces = layer._get_combined_traces()
        candidates = layer._select_candidates(combined_traces)
        proposals = generate_k_proposals(layer.weights, candidates, self.K, dev)

        # Batched forward: all K proposals at once via bmm
        with torch.no_grad():
            h_exp = layer_in.unsqueeze(0).expand(self.K, -1, -1)  # (K, B, in)
            w_float = proposals.float().to(dev)                     # (K, in, out)
            outs_k = torch.bmm(h_exp, w_float)                     # (K, B, out)
            if layer_idx < len(self.model.layers) - 1:
                outs_k = F.relu(outs_k)

        # Evaluate CDLL for each proposal
        best_fitness = fitness_before
        best_k = -1
        for k in range(self.K):
            f_k = self.cdll[layer_idx].compute(layer_in, outs_k[k])
            if f_k > best_fitness:
                best_fitness = f_k
                best_k = k

        delta = best_fitness - fitness_before

        # Boltzmann accept/reject
        layer_temp = self.config.get_temperature_for_layer(temperature, layer_idx)
        accepted = layer._accept_or_reject(delta, layer_temp)

        if accepted and best_k >= 0:
            layer.weights = proposals[best_k].to(torch.int8)

        # Update traces
        output = layer.forward(layer_in)
        layer._update_traces(layer_in, output, delta <= 0)
        layer.fitness_history.append(best_fitness if accepted else fitness_before)
        layer.acceptance_history.append(accepted)

        return {
            "accepted": accepted, "fitness_before": fitness_before,
            "fitness_after": best_fitness if accepted else fitness_before,
            "delta": delta, "n_candidates": len(candidates),
            "temperature": layer_temp, "fitness_type": "cdll",
        }

    def _train_layer_critic(
        self, layer_idx: int, layer: TFLELayer,
        layer_in: torch.Tensor, temperature: float,
    ) -> dict:
        """Train one layer with micro-critic fitness (sleep phase)."""
        layer.step_count += 1
        dev = layer.device

        # Current output + critic fitness
        with torch.no_grad():
            current_out = layer.forward(layer_in)
            if layer_idx < len(self.model.layers) - 1:
                current_out = F.relu(current_out)
        fitness_before = self.swt.get_critic_fitness(layer_idx, current_out)

        # Generate K proposals
        combined_traces = layer._get_combined_traces()
        candidates = layer._select_candidates(combined_traces)
        proposals = generate_k_proposals(layer.weights, candidates, self.K, dev)

        # Batched forward: all K at once
        with torch.no_grad():
            h_exp = layer_in.unsqueeze(0).expand(self.K, -1, -1)
            w_float = proposals.float().to(dev)
            outs_k = torch.bmm(h_exp, w_float)
            if layer_idx < len(self.model.layers) - 1:
                outs_k = F.relu(outs_k)

        # Evaluate critic for each proposal
        best_fitness = fitness_before
        best_k = -1
        for k in range(self.K):
            f_k = self.swt.get_critic_fitness(layer_idx, outs_k[k])
            if f_k > best_fitness:
                best_fitness = f_k
                best_k = k

        delta = best_fitness - fitness_before
        layer_temp = self.config.get_temperature_for_layer(temperature, layer_idx)
        accepted = layer._accept_or_reject(delta, layer_temp)

        if accepted and best_k >= 0:
            layer.weights = proposals[best_k].to(torch.int8)

        # Train the critic on current activations
        self.swt.micro_critics[layer_idx].train_step(current_out)

        output = layer.forward(layer_in)
        layer._update_traces(layer_in, output, delta <= 0)
        layer.fitness_history.append(best_fitness if accepted else fitness_before)
        layer.acceptance_history.append(accepted)

        return {
            "accepted": accepted, "fitness_before": fitness_before,
            "fitness_after": best_fitness if accepted else fitness_before,
            "delta": delta, "n_candidates": len(candidates),
            "temperature": layer_temp, "fitness_type": "critic",
        }

    def _get_layer_activations(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Get activations at each layer for critic training."""
        acts = []
        h = x.to(self.device)
        with torch.no_grad():
            for i, layer in enumerate(self.model.layers):
                if self.model.multi_gpu:
                    h = h.to(layer.device)
                h = layer.forward(h)
                if i < len(self.model.layers) - 1:
                    h = F.relu(h)
                acts.append(h.detach())
        return acts

    def get_status(self) -> str:
        """Dashboard status line."""
        n_layers = len(self.model.layers)
        phase = "Wake" if (not self.swt or self.swt.is_wake()) else "Sleep"
        temp = self.scheduler.get_temperature()

        cdll_scores = []
        for i, cdll in enumerate(self.cdll):
            if self.model.layers[i].fitness_history:
                cdll_scores.append(f"{self.model.layers[i].fitness_history[-1]:.3f}")
            else:
                cdll_scores.append("?")

        gpu_info = ""
        if self.n_gpus > 0:
            utils = []
            for i in range(self.n_gpus):
                try:
                    util = torch.cuda.utilization(i)
                    utils.append(f"GPU{i}: {util}%")
                except Exception:
                    utils.append(f"GPU{i}: ?")
            gpu_info = " | ".join(utils)

        return (f"Step {self.step_count} | {phase} | "
                f"CDLL: [{','.join(cdll_scores)}] | Temp: {temp:.3f} | {gpu_info}")
