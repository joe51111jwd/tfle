"""Ternary convolutional layers with TFLE training capability."""

from __future__ import annotations

import math
from collections import deque
from typing import Optional

import torch
import torch.nn.functional as F

from .config import (
    AcceptanceFunction,
    CreditNormalization,
    FlipDirectionBias,
    SelectionMethod,
    TFLEConfig,
    TraceIncrement,
)
from .layers import compute_goodness, random_ternary


def ternary_conv2d(
    x: torch.Tensor,
    weights: torch.Tensor,
    stride: int = 1,
    padding: int = 0,
) -> torch.Tensor:
    """Ternary 2D convolution."""
    return F.conv2d(x, weights.float(), stride=stride, padding=padding)


class TFLEConvLayer:
    """A ternary convolutional layer with TFLE training."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        config: TFLEConfig,
        layer_idx: int = 0,
        stride: int = 1,
        padding: int = 0,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.config = config
        self.layer_idx = layer_idx

        # Ternary conv weights: [out_channels, in_channels, kernel_size, kernel_size]
        n_weights = out_channels * in_channels * kernel_size * kernel_size
        flat = random_ternary(1, n_weights, config).flatten()
        self.weights = flat.reshape(out_channels, in_channels, kernel_size, kernel_size)

        trace_dtype = torch.float16 if config.trace_dtype == "float16" else torch.float32
        shape = (out_channels, in_channels, kernel_size, kernel_size)
        if config.separate_pos_neg_traces:
            self.success_traces = torch.zeros(shape, dtype=trace_dtype)
            self.error_traces = torch.zeros(shape, dtype=trace_dtype)
        else:
            self.traces = torch.zeros(shape, dtype=trace_dtype)

        self.fitness_ema: Optional[float] = None
        self.fitness_history: deque[float] = deque(maxlen=config.fitness_history_window)
        self.acceptance_history: deque[bool] = deque(maxlen=config.fitness_history_window)
        self.cooldown_map: dict[int, int] = {}
        self.tabu_set: deque[tuple[int, int]] = deque(maxlen=config.tabu_list_size)
        self.step_count = 0

    @property
    def n_weights(self) -> int:
        return self.weights.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ternary_conv2d(x, self.weights, self.stride, self.padding)

    def _get_combined_traces(self) -> torch.Tensor:
        cfg = self.config
        if cfg.separate_pos_neg_traces:
            error_trace = self.error_traces.float()
            success_trace = self.success_traces.float()
            combined = error_trace - cfg.protection_threshold * success_trace
        else:
            combined = self.traces.float()
        return combined

    def _select_candidates(self, combined_traces: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        n = self.n_weights
        flat_traces = combined_traces.flatten()

        flip_rate = cfg.get_flip_rate_for_layer(self.layer_idx)
        n_candidates = max(
            cfg.min_candidates_per_step,
            min(int(flip_rate * n), int(cfg.max_candidates_fraction * n)),
        )

        if cfg.selection_method == SelectionMethod.TRACE_WEIGHTED:
            scores = flat_traces.clone()
            if cfg.credit_normalization == CreditNormalization.LAYER_NORM:
                mean = scores.mean()
                std = scores.std() + 1e-8
                scores = (scores - mean) / std

            if cfg.separate_pos_neg_traces:
                success_flat = self.success_traces.float().flatten()
                threshold = torch.quantile(success_flat, cfg.protection_threshold)
                protected = success_flat >= threshold
                scores[protected] = float("-inf")

            n_explore = max(1, int(cfg.exploration_rate * n))
            n_guided = n_candidates - n_explore

            if n_guided > 0 and scores.max() > float("-inf"):
                valid_mask = scores > float("-inf")
                valid_scores = scores.clone()
                valid_scores[~valid_mask] = float("-inf")
                _, guided_idx = torch.topk(valid_scores, min(n_guided, valid_mask.sum().item()))
            else:
                guided_idx = torch.tensor([], dtype=torch.long)

            explore_idx = torch.randint(0, n, (n_explore,))
            candidates = torch.cat([guided_idx, explore_idx]).unique()
        else:
            candidates = torch.randperm(n)[:n_candidates]

        return candidates

    def _propose_flips(self, candidates: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        proposed = self.weights.flatten().clone()

        for idx in candidates.tolist():
            current = proposed[idx].item()
            if cfg.flip_direction_bias == FlipDirectionBias.CREDIT_BIASED:
                flat_err = (
                    self.error_traces.float().flatten()
                    if cfg.separate_pos_neg_traces
                    else self.traces.float().flatten()
                )
                err = flat_err[idx].item()
                if err > 0:
                    if current == 0:
                        new_val = 1 if torch.rand(1).item() > 0.5 else -1
                    elif torch.rand(1).item() < cfg.zero_gravity:
                        new_val = 0
                    else:
                        new_val = -current
                else:
                    options = [v for v in [-1, 0, 1] if v != current]
                    new_val = options[torch.randint(0, len(options), (1,)).item()]
            else:
                options = [v for v in [-1, 0, 1] if v != current]
                new_val = options[torch.randint(0, len(options), (1,)).item()]
            proposed[idx] = new_val

        return proposed.reshape(self.weights.shape)

    def _compute_contrastive_fitness(
        self, x_real: torch.Tensor, x_corrupted: torch.Tensor, weights: torch.Tensor
    ) -> float:
        out_real = F.conv2d(x_real, weights.float(), stride=self.stride, padding=self.padding)
        out_fake = F.conv2d(x_corrupted, weights.float(), stride=self.stride, padding=self.padding)
        # Flatten spatial dims for goodness
        out_real_flat = out_real.flatten(start_dim=1)
        out_fake_flat = out_fake.flatten(start_dim=1)
        goodness_real = compute_goodness(out_real_flat, self.config.goodness_metric).mean()
        goodness_fake = compute_goodness(out_fake_flat, self.config.goodness_metric).mean()
        return (goodness_real - goodness_fake).item()

    def _accept_or_reject(self, delta: float, temperature: float) -> bool:
        if delta > 0:
            return True
        if temperature <= 0:
            return False
        cfg = self.config
        if cfg.acceptance_function == AcceptanceFunction.BOLTZMANN:
            prob = math.exp(min(delta / temperature, 0))
            return torch.rand(1).item() < prob
        elif cfg.acceptance_function == AcceptanceFunction.THRESHOLD:
            return delta > -temperature
        elif cfg.acceptance_function == AcceptanceFunction.METROPOLIS:
            prob = min(1.0, math.exp(min(delta / temperature, 0)))
            return torch.rand(1).item() < prob
        return False

    def _update_traces(self, x: torch.Tensor, output: torch.Tensor, is_error: bool):
        cfg = self.config
        if cfg.separate_pos_neg_traces:
            self.success_traces *= cfg.trace_decay
            self.error_traces *= cfg.trace_decay
        else:
            self.traces *= cfg.trace_decay

        # Activity-based increment for conv weights
        w_abs = self.weights.abs().float()
        # Use weight magnitude as proxy for activity
        increment = w_abs * (output.abs().mean().item() + 1e-8)

        if cfg.trace_increment == TraceIncrement.BINARY:
            increment = (w_abs > 0).float()
        elif cfg.trace_increment == TraceIncrement.SQUARED:
            increment = increment ** 2

        if cfg.separate_pos_neg_traces:
            if is_error:
                self.error_traces += increment.to(self.error_traces.dtype)
            else:
                self.success_traces += increment.to(self.success_traces.dtype)
        else:
            if is_error:
                self.traces += increment.to(self.traces.dtype)
            else:
                self.traces -= 0.5 * increment.to(self.traces.dtype)

    def train_step(
        self,
        x_real: torch.Tensor,
        x_corrupted: torch.Tensor,
        temperature: float,
        is_error: Optional[bool] = None,
    ) -> dict:
        self.step_count += 1

        # Decay cooldowns
        expired = [k for k, v in self.cooldown_map.items() if v <= 0]
        for k in expired:
            del self.cooldown_map[k]
        for k in self.cooldown_map:
            self.cooldown_map[k] -= 1

        fitness_before = self._compute_contrastive_fitness(x_real, x_corrupted, self.weights)
        combined_traces = self._get_combined_traces()
        candidates = self._select_candidates(combined_traces)
        proposed_weights = self._propose_flips(candidates)
        fitness_after = self._compute_contrastive_fitness(x_real, x_corrupted, proposed_weights)
        delta = fitness_after - fitness_before

        layer_temp = self.config.get_temperature_for_layer(temperature, self.layer_idx)
        accepted = self._accept_or_reject(delta, layer_temp)

        if accepted:
            self.weights = proposed_weights.to(torch.int8)

        output = self.forward(x_real)
        error_signal = is_error if is_error is not None else (delta <= 0)
        self._update_traces(x_real, output, error_signal)

        self.fitness_history.append(fitness_after if accepted else fitness_before)
        self.acceptance_history.append(accepted)

        return {
            "accepted": accepted,
            "fitness_before": fitness_before,
            "fitness_after": fitness_after,
            "delta": delta,
            "n_candidates": len(candidates),
            "temperature": layer_temp,
        }

    def get_weight_distribution(self) -> dict[str, float]:
        w = self.weights.flatten().float()
        total = w.numel()
        return {
            "neg1": (w == -1).sum().item() / total,
            "zero": (w == 0).sum().item() / total,
            "pos1": (w == 1).sum().item() / total,
        }

    def get_acceptance_rate(self) -> float:
        if not self.acceptance_history:
            return 0.0
        return sum(self.acceptance_history) / len(self.acceptance_history)
