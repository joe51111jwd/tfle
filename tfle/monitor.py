"""Convergence monitoring and diagnostics."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .config import TFLEConfig


@dataclass
class LayerMetrics:
    """Accumulated metrics for a single layer."""

    fitness_values: deque = field(default_factory=lambda: deque(maxlen=500))
    acceptance_rates: deque = field(default_factory=lambda: deque(maxlen=500))
    weight_distributions: list = field(default_factory=list)
    trace_stats: list = field(default_factory=list)
    is_oscillating: bool = False
    effective_flip_rate: float = 0.0


class ConvergenceMonitor:
    """Monitors training convergence and detects issues."""

    def __init__(self, config: TFLEConfig, n_layers: int):
        self.config = config
        self.n_layers = n_layers
        self.layer_metrics: list[LayerMetrics] = [LayerMetrics() for _ in range(n_layers)]
        self.global_fitness_history: deque[float] = deque(
            maxlen=config.fitness_history_window
        )
        self.val_accuracy_history: list[tuple[int, float]] = []
        self.best_val_accuracy = 0.0
        self.steps_since_improvement = 0

    def record_step(
        self,
        layer_idx: int,
        step_metrics: dict,
        weight_dist: dict | None = None,
        trace_stats: dict | None = None,
    ):
        lm = self.layer_metrics[layer_idx]
        fitness = step_metrics.get("fitness_after", step_metrics.get("fitness_before", 0))
        lm.fitness_values.append(fitness)
        lm.acceptance_rates.append(1.0 if step_metrics["accepted"] else 0.0)

        if weight_dist is not None:
            lm.weight_distributions.append(weight_dist)
        if trace_stats is not None:
            lm.trace_stats.append(trace_stats)

    def record_global_fitness(self, fitness: float):
        self.global_fitness_history.append(fitness)

    def record_validation(self, step: int, accuracy: float):
        self.val_accuracy_history.append((step, accuracy))
        if accuracy > self.best_val_accuracy:
            self.best_val_accuracy = accuracy
            self.steps_since_improvement = 0
        else:
            self.steps_since_improvement += 1

    def detect_oscillation(self, layer_idx: int) -> bool:
        """Detect if a layer's fitness is oscillating."""
        lm = self.layer_metrics[layer_idx]
        values = list(lm.fitness_values)
        if len(values) < 20:
            return False

        recent = values[-20:]
        diffs = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
        sign_changes = sum(
            1 for i in range(len(diffs) - 1) if diffs[i] * diffs[i + 1] < 0
        )
        is_oscillating = sign_changes > len(diffs) * 0.6
        lm.is_oscillating = is_oscillating
        return is_oscillating

    def get_convergence_rate(self) -> float:
        """Average fitness improvement per step over recent window."""
        values = list(self.global_fitness_history)
        if len(values) < 10:
            return float("inf")
        half = len(values) // 2
        first_half_mean = sum(values[:half]) / half
        second_half_mean = sum(values[half:]) / (len(values) - half)
        return (second_half_mean - first_half_mean) / max(half, 1)

    def should_early_stop(self) -> bool:
        """Check if training should stop early."""
        if not self.val_accuracy_history:
            return False
        return self.steps_since_improvement >= self.config.early_stopping_patience

    def get_layer_acceptance_rate(self, layer_idx: int) -> float:
        rates = list(self.layer_metrics[layer_idx].acceptance_rates)
        if not rates:
            return 0.0
        return sum(rates) / len(rates)

    def get_summary(self) -> dict:
        """Get a summary of all monitoring metrics."""
        summary = {
            "convergence_rate": self.get_convergence_rate(),
            "best_val_accuracy": self.best_val_accuracy,
            "steps_since_improvement": self.steps_since_improvement,
            "layers": [],
        }
        for i in range(self.n_layers):
            layer_info = {
                "acceptance_rate": self.get_layer_acceptance_rate(i),
                "is_oscillating": self.detect_oscillation(i),
                "recent_fitness": (
                    list(self.layer_metrics[i].fitness_values)[-1]
                    if self.layer_metrics[i].fitness_values
                    else None
                ),
            }
            summary["layers"].append(layer_info)
        return summary
