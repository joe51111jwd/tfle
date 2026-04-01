"""Training loop for TFLE."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from torch.utils.data import DataLoader
from tqdm import tqdm

from .annealing import TemperatureScheduler
from .config import TFLEConfig
from .model import TFLEModel
from .monitor import ConvergenceMonitor


@dataclass
class TrainingResult:
    """Results from a training run."""

    train_accuracies: list[tuple[int, float]] = field(default_factory=list)
    val_accuracies: list[tuple[int, float]] = field(default_factory=list)
    layer_metrics_log: list[dict] = field(default_factory=list)
    final_accuracy: float = 0.0
    total_steps: int = 0
    training_time_seconds: float = 0.0
    memory_usage: dict = field(default_factory=dict)
    stopped_early: bool = False


class TFLETrainer:
    """Orchestrates the TFLE training loop."""

    def __init__(
        self,
        model: TFLEModel,
        config: TFLEConfig,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        checkpoint_dir: str | None = None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = TemperatureScheduler(config)
        self.monitor = ConvergenceMonitor(config, len(model.layers))
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train(self, verbose: bool = True) -> TrainingResult:
        result = TrainingResult()
        result.memory_usage = self.model.get_memory_usage_bytes()
        start_time = time.time()

        step = 0
        epoch = 0
        pbar = tqdm(
            total=self.config.total_training_steps, disable=not verbose, desc="TFLE Training",
        )

        while step < self.config.total_training_steps:
            epoch += 1
            for batch_x, batch_y in self.train_loader:
                if step >= self.config.total_training_steps:
                    break

                # Flatten images if needed (e.g. MNIST 28x28 -> 784)
                if batch_x.dim() > 2:
                    batch_x = batch_x.view(batch_x.size(0), -1)

                # Move to model device (GPU if available)
                device = getattr(self.model, 'device', None)
                if device is not None:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)

                temperature = self.scheduler.get_temperature()

                # Use batched mode if available and configured
                n_proposals = getattr(self.config, 'n_proposals', 0)
                if n_proposals > 1 and hasattr(self.model, 'train_step_batched'):
                    layer_metrics = self.model.train_step_batched(
                        batch_x, temperature, batch_y, n_proposals=n_proposals
                    )
                else:
                    layer_metrics = self.model.train_step(batch_x, temperature, batch_y)

                # Record metrics
                for i, lm in enumerate(layer_metrics):
                    self.monitor.record_step(
                        i, lm,
                        weight_dist=(
                            self.model.layers[i].get_weight_distribution()
                            if self.config.log_weight_distribution and step % 100 == 0
                            else None
                        ),
                        trace_stats=(
                            self.model.layers[i].get_trace_statistics()
                            if self.config.log_trace_statistics and step % 100 == 0
                            else None
                        ),
                    )

                avg_fitness = sum(
                    m["fitness_after"] if m["accepted"] else m["fitness_before"]
                    for m in layer_metrics
                ) / len(layer_metrics)
                self.monitor.record_global_fitness(avg_fitness)
                self.scheduler.step_update(avg_fitness)

                # Handle oscillation
                if self.config.oscillation_detection and step % 50 == 0:
                    for i in range(len(self.model.layers)):
                        self.monitor.detect_oscillation(i)

                # Evaluation
                if step % self.config.eval_interval == 0:
                    eval_result = self._evaluate()
                    result.val_accuracies.append((step, eval_result["accuracy"]))
                    self.monitor.record_validation(step, eval_result["accuracy"])

                    if verbose:
                        acc_rates = [
                            self.monitor.get_layer_acceptance_rate(i)
                            for i in range(len(self.model.layers))
                        ]
                        pbar.set_postfix({
                            "acc": f"{eval_result['accuracy']:.4f}",
                            "temp": f"{temperature:.4f}",
                            "accept": f"{sum(acc_rates)/len(acc_rates):.2f}",
                        })

                # Checkpoint
                if (
                    self.checkpoint_dir
                    and self.config.checkpoint_interval > 0
                    and step % self.config.checkpoint_interval == 0
                    and step > 0
                ):
                    self.model.save_checkpoint(
                        str(self.checkpoint_dir / f"checkpoint_step{step}.pt")
                    )

                # Early stopping
                if self.monitor.should_early_stop():
                    result.stopped_early = True
                    if verbose:
                        pbar.write(f"Early stopping at step {step}")
                    break

                step += 1
                pbar.update(1)

            if result.stopped_early:
                break

        pbar.close()

        result.total_steps = step
        result.training_time_seconds = time.time() - start_time
        if result.val_accuracies:
            result.final_accuracy = result.val_accuracies[-1][1]

        return result

    def _evaluate(self) -> dict:
        """Evaluate on validation set or a subset of training data."""
        if self.val_loader is not None:
            all_correct = 0
            all_total = 0
            all_loss = 0.0
            n_batches = 0
            for batch_x, batch_y in self.val_loader:
                if batch_x.dim() > 2:
                    batch_x = batch_x.view(batch_x.size(0), -1)
                device = getattr(self.model, 'device', None)
                if device is not None:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                eval_result = self.model.evaluate(batch_x, batch_y)
                all_correct += eval_result["accuracy"] * batch_x.size(0)
                all_total += batch_x.size(0)
                all_loss += eval_result["loss"]
                n_batches += 1
            return {
                "accuracy": all_correct / max(all_total, 1),
                "loss": all_loss / max(n_batches, 1),
            }

        # Fallback: use first batch of training data
        batch_x, batch_y = next(iter(self.train_loader))
        if batch_x.dim() > 2:
            batch_x = batch_x.view(batch_x.size(0), -1)
        return self.model.evaluate(batch_x, batch_y)
