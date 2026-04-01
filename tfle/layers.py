"""Ternary neural network layers with TFLE training capability."""

from __future__ import annotations

import math
from collections import deque
from typing import Optional

import torch
import torch.nn.functional as F

from .config import (
    AcceptanceFunction,
    CreditNormalization,
    FitnessType,
    FlipDirectionBias,
    GoodnessMetric,
    InitMethod,
    SelectionMethod,
    TFLEConfig,
    TraceIncrement,
)


def random_ternary(rows: int, cols: int, config: TFLEConfig) -> torch.Tensor:
    """Initialize ternary weights based on config."""
    if config.init_seed is not None:
        torch.manual_seed(config.init_seed)

    if config.init_method == InitMethod.BALANCED_RANDOM:
        probs = torch.tensor([
            (1 - config.init_zero_bias) / 2,
            config.init_zero_bias,
            (1 - config.init_zero_bias) / 2,
        ])
        indices = torch.multinomial(probs.expand(rows * cols, -1), 1).squeeze()
        weights = indices.float() - 1.0  # maps 0->-1, 1->0, 2->+1
        return weights.reshape(rows, cols).to(torch.int8)

    elif config.init_method == InitMethod.SPARSE_RANDOM:
        probs = torch.tensor([0.2, 0.6, 0.2])
        indices = torch.multinomial(probs.expand(rows * cols, -1), 1).squeeze()
        weights = indices.float() - 1.0
        return weights.reshape(rows, cols).to(torch.int8)

    elif config.init_method == InitMethod.KAIMING_TERNARY:
        fan_in = rows
        p_nonzero = min(1.0, 1.0 / math.sqrt(fan_in))
        p_zero = 1.0 - p_nonzero
        probs = torch.tensor([p_nonzero / 2, p_zero, p_nonzero / 2])
        indices = torch.multinomial(probs.expand(rows * cols, -1), 1).squeeze()
        weights = indices.float() - 1.0
        return weights.reshape(rows, cols).to(torch.int8)

    raise ValueError(f"Unknown init method: {config.init_method}")


def ternary_matmul(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Efficient ternary matrix multiplication.

    For ternary weights, multiply becomes add/subtract/skip:
    w=+1 -> add input, w=-1 -> subtract input, w=0 -> skip
    """
    if weights.dtype == torch.int8:
        return x @ weights.to(dtype=x.dtype, device=x.device)
    return x @ weights.to(device=x.device)


def compute_goodness(activations: torch.Tensor, metric: GoodnessMetric) -> torch.Tensor:
    """Compute goodness score from activations."""
    if metric == GoodnessMetric.SUM_OF_SQUARES:
        return (activations ** 2).sum(dim=-1)
    elif metric == GoodnessMetric.MEAN_ACTIVATION:
        return activations.abs().mean(dim=-1)
    elif metric == GoodnessMetric.MAX_ACTIVATION:
        return activations.abs().max(dim=-1).values
    elif metric == GoodnessMetric.ENTROPY:
        probs = F.softmax(activations, dim=-1)
        return -(probs * (probs + 1e-8).log()).sum(dim=-1)
    raise ValueError(f"Unknown goodness metric: {metric}")


class TFLELayer:
    """A single ternary layer with TFLE training capability."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: TFLEConfig,
        layer_idx: int = 0,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        self.layer_idx = layer_idx

        self.weights = random_ternary(in_features, out_features, config)

        trace_dtype = torch.float16 if config.trace_dtype == "float16" else torch.float32
        if config.trace_dtype == "int8":
            trace_dtype = torch.float16  # use float16 internally, quantize on save

        if config.separate_pos_neg_traces:
            self.success_traces = torch.zeros(in_features, out_features, dtype=trace_dtype)
            self.error_traces = torch.zeros(in_features, out_features, dtype=trace_dtype)
        else:
            self.traces = torch.zeros(in_features, out_features, dtype=trace_dtype)

        self.error_correlation_history: deque[tuple[torch.Tensor, float]] = deque(
            maxlen=config.error_correlation_window
        )

        self.fitness_ema: Optional[float] = None
        self.fitness_history: deque[float] = deque(maxlen=config.fitness_history_window)
        self.acceptance_history: deque[bool] = deque(maxlen=config.fitness_history_window)

        # Cooldown tracking
        self.cooldown_map: dict[tuple[int, int], int] = {}

        # Tabu list
        self.tabu_set: deque[tuple[int, int, int]] = deque(maxlen=config.tabu_list_size)

        self.step_count = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ternary_matmul(x, self.weights)

    def _get_combined_traces(self) -> torch.Tensor:
        """Get the combined credit signal for candidate selection."""
        cfg = self.config
        if cfg.separate_pos_neg_traces:
            error_trace = self.error_traces.float()
            success_trace = self.success_traces.float()
            # High error trace + low success trace = good candidate
            combined = error_trace - cfg.protection_threshold * success_trace
        else:
            combined = self.traces.float()
        return combined

    def _select_candidates(self, combined_traces: torch.Tensor) -> torch.Tensor:
        """Select weight indices to consider flipping."""
        cfg = self.config
        n_weights = self.in_features * self.out_features
        flat_traces = combined_traces.flatten()

        flip_rate = cfg.get_flip_rate_for_layer(self.layer_idx)
        n_candidates = max(
            cfg.min_candidates_per_step,
            min(int(flip_rate * n_weights), int(cfg.max_candidates_fraction * n_weights)),
        )

        if cfg.selection_method == SelectionMethod.TRACE_WEIGHTED:
            # Blend trace magnitude with error correlation
            scores = flat_traces
            if cfg.credit_normalization == CreditNormalization.LAYER_NORM:
                mean = scores.mean()
                std = scores.std() + 1e-8
                scores = (scores - mean) / std
            elif cfg.credit_normalization == CreditNormalization.RANK:
                ranks = scores.argsort().argsort().float()
                scores = ranks / ranks.max()

            # Protection: zero out scores for well-performing weights
            if cfg.separate_pos_neg_traces:
                success_flat = self.success_traces.float().flatten()
                threshold = torch.quantile(success_flat, cfg.protection_threshold)
                protected = success_flat >= threshold
                scores[protected] = float("-inf")

            # Add exploration: some random candidates
            n_explore = max(1, int(cfg.exploration_rate * n_weights))
            n_guided = n_candidates - n_explore

            if n_guided > 0 and scores.max() > float("-inf"):
                valid_mask = scores > float("-inf")
                valid_scores = scores.clone()
                valid_scores[~valid_mask] = float("-inf")
                _, guided_idx = torch.topk(valid_scores, min(n_guided, valid_mask.sum().item()))
            else:
                guided_idx = torch.tensor([], dtype=torch.long, device=flat_traces.device)

            explore_idx = torch.randint(0, n_weights, (n_explore,), device=flat_traces.device)
            candidates = torch.cat([guided_idx, explore_idx]).unique()

        elif cfg.selection_method == SelectionMethod.UNIFORM_RANDOM:
            candidates = torch.randperm(n_weights, device=flat_traces.device)[:n_candidates]

        elif cfg.selection_method == SelectionMethod.ANTI_CORRELATED:
            scores = flat_traces
            _, candidates = torch.topk(scores, n_candidates)

        else:
            candidates = torch.randperm(n_weights, device=flat_traces.device)[:n_candidates]

        # Filter out weights in cooldown
        if cfg.flip_revert_on_reject.value == "cooldown":
            valid = []
            for idx in candidates.tolist():
                r, c = idx // self.out_features, idx % self.out_features
                if self.cooldown_map.get((r, c), 0) <= 0:
                    valid.append(idx)
            candidates = torch.tensor(valid, dtype=torch.long) if valid else candidates[:1]

        # Filter out tabu entries
        if cfg.tabu_list_size > 0:
            flat_weights = self.weights.flatten()
            valid = []
            for idx in candidates.tolist():
                r, c = idx // self.out_features, idx % self.out_features
                current_val = flat_weights[idx].item()
                if (r, c, int(current_val)) not in self.tabu_set:
                    valid.append(idx)
            if valid:
                candidates = torch.tensor(valid, dtype=torch.long)

        return candidates

    def _propose_flips(self, candidates: torch.Tensor) -> torch.Tensor:
        """Propose new ternary values for candidate weights (vectorized)."""
        proposed = self.weights.flatten().clone()
        n = candidates.numel()
        if n == 0:
            return proposed.reshape(self.in_features, self.out_features)

        current_vals = proposed[candidates]  # (n,)

        # Vectorized random flip: for each candidate, pick one of the two other trit values
        # Map: -1 can go to {0, 1}, 0 can go to {-1, 1}, 1 can go to {-1, 0}
        # Strategy: add a random offset of +1 or +2 (mod 3), then map back to {-1, 0, 1}
        # Trit values -1, 0, 1 map to indices 0, 1, 2
        idx_vals = (current_vals.long() + 1)  # map to 0, 1, 2
        offsets = torch.randint(1, 3, (n,), device=candidates.device)  # +1 or +2
        new_idx = (idx_vals + offsets) % 3
        new_vals = (new_idx - 1).to(proposed.dtype)  # map back to -1, 0, 1

        proposed[candidates] = new_vals
        return proposed.reshape(self.in_features, self.out_features)

    def _compute_contrastive_fitness(
        self, x_real: torch.Tensor, x_corrupted: torch.Tensor, weights: torch.Tensor
    ) -> float:
        """Compute contrastive fitness for given weights."""
        w = weights.to(dtype=x_real.dtype, device=x_real.device)
        out_real = x_real @ w
        out_fake = x_corrupted @ w
        goodness_real = compute_goodness(out_real, self.config.goodness_metric).mean()
        goodness_fake = compute_goodness(out_fake, self.config.goodness_metric).mean()
        return (goodness_real - goodness_fake).item()

    def _accept_or_reject(
        self, delta: float, temperature: float
    ) -> bool:
        """Decide whether to accept proposed flips."""
        cfg = self.config
        if delta > 0:
            return True

        if temperature <= 0:
            return False

        if cfg.acceptance_function == AcceptanceFunction.BOLTZMANN:
            prob = math.exp(min(delta / temperature, 0))
            return torch.rand(1).item() < prob

        elif cfg.acceptance_function == AcceptanceFunction.THRESHOLD:
            return delta > -temperature

        elif cfg.acceptance_function == AcceptanceFunction.METROPOLIS:
            prob = min(1.0, math.exp(min(delta / temperature, 0)))
            return torch.rand(1).item() < prob

        return False

    def _update_traces(
        self, x: torch.Tensor, output: torch.Tensor, is_error: bool
    ):
        """Update temporal credit traces."""
        cfg = self.config

        # Decay all traces
        if cfg.separate_pos_neg_traces:
            self.success_traces *= cfg.trace_decay
            self.error_traces *= cfg.trace_decay
        else:
            self.traces *= cfg.trace_decay

        # Compute activity: which weights were active
        # A weight w[i,j] is "active" if input[i] is nonzero (since w is ternary)
        input_activity = x.abs().mean(dim=0)  # [in_features]

        # Weight activity ~ input magnitude * weight magnitude
        w_abs = self.weights.abs().float()
        if cfg.trace_activation_threshold > 0:
            input_activity = input_activity * (input_activity > cfg.trace_activation_threshold)

        if cfg.trace_increment == TraceIncrement.BINARY:
            increment = (w_abs > 0).float() * (input_activity.unsqueeze(1) > 0).float()
        elif cfg.trace_increment == TraceIncrement.MAGNITUDE:
            increment = w_abs * input_activity.unsqueeze(1)
        elif cfg.trace_increment == TraceIncrement.SQUARED:
            increment = (w_abs * input_activity.unsqueeze(1)) ** 2
        else:
            increment = w_abs * input_activity.unsqueeze(1)

        # Update appropriate trace
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
        task_loss_fn=None,
    ) -> dict:
        """Execute one TFLE training step for this layer.

        Args:
            task_loss_fn: Optional callable() -> float. If provided and config.fitness_type
                is TASK_LOSS, this function evaluates the full model's loss with current
                weights. The layer swaps its weights before/after calling it.

        Returns dict with metrics: accepted, fitness_before, fitness_after, delta, n_candidates
        """
        self.step_count += 1

        # Decay cooldowns
        expired = [k for k, v in self.cooldown_map.items() if v <= 0]
        for k in expired:
            del self.cooldown_map[k]
        for k in self.cooldown_map:
            self.cooldown_map[k] -= 1

        # 1. Current fitness
        use_task_loss = (
            self.config.fitness_type == FitnessType.TASK_LOSS
            and task_loss_fn is not None
        )
        if use_task_loss:
            # Fitness = negative loss (higher is better)
            fitness_before = -task_loss_fn()
        else:
            fitness_before = self._compute_contrastive_fitness(x_real, x_corrupted, self.weights)

        # 2. Select candidates
        combined_traces = self._get_combined_traces()
        candidates = self._select_candidates(combined_traces)

        # 3. Propose flips
        proposed_weights = self._propose_flips(candidates)

        # 4. Evaluate — swap weights temporarily for task_loss evaluation
        if use_task_loss:
            old_weights = self.weights
            self.weights = proposed_weights.to(torch.int8)
            fitness_after = -task_loss_fn()
            self.weights = old_weights  # restore for accept/reject decision
        else:
            fitness_after = self._compute_contrastive_fitness(x_real, x_corrupted, proposed_weights)
        delta = fitness_after - fitness_before

        # 5. Accept or reject
        layer_temp = self.config.get_temperature_for_layer(temperature, self.layer_idx)
        accepted = self._accept_or_reject(delta, layer_temp)

        if accepted:
            self.weights = proposed_weights.to(torch.int8)
        else:
            # Cooldown rejected candidates
            if self.config.flip_revert_on_reject.value == "cooldown":
                for idx in candidates.tolist():
                    r, c = idx // self.out_features, idx % self.out_features
                    self.cooldown_map[(r, c)] = self.config.cooldown_steps

            # Add to tabu list
            if self.config.tabu_list_size > 0:
                flat_weights = self.weights.flatten()
                for idx in candidates.tolist():
                    r, c = idx // self.out_features, idx % self.out_features
                    self.tabu_set.append((r, c, int(flat_weights[idx].item())))

        # 6. Update traces
        output = self.forward(x_real)
        error_signal = is_error if is_error is not None else (delta <= 0)
        self._update_traces(x_real, output, error_signal)

        # Track fitness
        if self.config.fitness_baseline.value == "relative":
            if self.fitness_ema is None:
                self.fitness_ema = fitness_after if accepted else fitness_before
            else:
                current = fitness_after if accepted else fitness_before
                self.fitness_ema = (
                    self.config.fitness_ema_decay * self.fitness_ema
                    + (1 - self.config.fitness_ema_decay) * current
                )

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
        """Return the ratio of -1, 0, +1 weights."""
        w = self.weights.flatten().float()
        total = w.numel()
        return {
            "neg1": (w == -1).sum().item() / total,
            "zero": (w == 0).sum().item() / total,
            "pos1": (w == 1).sum().item() / total,
        }

    def get_trace_statistics(self) -> dict[str, float]:
        """Return statistics about temporal traces."""
        if self.config.separate_pos_neg_traces:
            err = self.error_traces.float()
            suc = self.success_traces.float()
            return {
                "error_trace_mean": err.mean().item(),
                "error_trace_max": err.max().item(),
                "error_trace_std": err.std().item(),
                "success_trace_mean": suc.mean().item(),
                "success_trace_max": suc.max().item(),
                "success_trace_std": suc.std().item(),
            }
        else:
            t = self.traces.float()
            return {
                "trace_mean": t.mean().item(),
                "trace_max": t.max().item(),
                "trace_std": t.std().item(),
            }

    def get_acceptance_rate(self) -> float:
        """Return recent acceptance rate."""
        if not self.acceptance_history:
            return 0.0
        return sum(self.acceptance_history) / len(self.acceptance_history)
