"""Temperature scheduling for simulated annealing."""

from __future__ import annotations

import math
from collections import deque

from .config import CoolingSchedule, TFLEConfig


class TemperatureScheduler:
    """Manages temperature for simulated annealing."""

    def __init__(self, config: TFLEConfig):
        self.config = config
        self.temperature = config.initial_temperature
        self.step = 0
        self.fitness_history: deque[float] = deque(maxlen=config.plateau_window)
        self.best_fitness = float("-inf")
        self.steps_without_improvement = 0

    def get_temperature(self) -> float:
        return max(self.temperature, self.config.min_temperature)

    def step_update(self, current_fitness: float | None = None):
        """Update temperature after a training step."""
        self.step += 1
        cfg = self.config

        if current_fitness is not None:
            self.fitness_history.append(current_fitness)
            if current_fitness > self.best_fitness:
                self.best_fitness = current_fitness
                self.steps_without_improvement = 0
            else:
                self.steps_without_improvement += 1

        if cfg.cooling_schedule == CoolingSchedule.EXPONENTIAL:
            self.temperature *= cfg.cooling_rate

        elif cfg.cooling_schedule == CoolingSchedule.LINEAR:
            temp_range = cfg.initial_temperature - cfg.min_temperature
            decay_per_step = temp_range / cfg.total_training_steps
            self.temperature = max(
                cfg.min_temperature,
                cfg.initial_temperature - decay_per_step * self.step,
            )

        elif cfg.cooling_schedule == CoolingSchedule.COSINE:
            progress = self.step / cfg.total_training_steps
            self.temperature = cfg.min_temperature + 0.5 * (
                cfg.initial_temperature - cfg.min_temperature
            ) * (1 + math.cos(math.pi * progress))

        elif cfg.cooling_schedule == CoolingSchedule.ADAPTIVE:
            if self.steps_without_improvement > cfg.plateau_window // 2:
                self.temperature *= 1.01  # warm up slightly
            else:
                self.temperature *= cfg.cooling_rate

        # Reheat on plateau
        if cfg.reheat_on_plateau and self.steps_without_improvement >= cfg.plateau_window:
            self.temperature = min(
                self.temperature * cfg.reheat_factor,
                cfg.initial_temperature,
            )
            self.steps_without_improvement = 0

        self.temperature = max(self.temperature, cfg.min_temperature)

    def reset(self):
        self.temperature = self.config.initial_temperature
        self.step = 0
        self.fitness_history.clear()
        self.best_fitness = float("-inf")
        self.steps_without_improvement = 0
