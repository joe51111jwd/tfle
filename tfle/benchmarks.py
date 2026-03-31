"""Benchmarking utilities for rigorous TFLE evaluation."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader

from .config import TFLEConfig
from .model import TFLEModel
from .training import TFLETrainer


@dataclass
class BenchmarkRun:
    """A single benchmark run with all metadata."""

    name: str
    config: dict = field(default_factory=dict)
    result: dict = field(default_factory=dict)
    hardware_info: dict = field(default_factory=dict)
    timestamp: str = ""


@dataclass
class BenchmarkSuite:
    """Collection of benchmark runs for analysis."""

    runs: list[BenchmarkRun] = field(default_factory=list)

    def add_run(self, run: BenchmarkRun):
        self.runs.append(run)

    def save(self, path: str):
        data = [
            {
                "name": r.name,
                "config": r.config,
                "result": r.result,
                "hardware_info": r.hardware_info,
                "timestamp": r.timestamp,
            }
            for r in self.runs
        ]
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> BenchmarkSuite:
        with open(path) as f:
            data = json.load(f)
        suite = cls()
        for d in data:
            suite.runs.append(BenchmarkRun(**d))
        return suite


def get_hardware_info() -> dict:
    """Gather hardware information for reproducibility."""
    import platform

    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    elif torch.backends.mps.is_available():
        info["accelerator"] = "Apple MPS"
    else:
        info["accelerator"] = "CPU only"
    return info


def benchmark_flip_rates(
    train_loader: DataLoader,
    val_loader: DataLoader,
    layer_sizes: list[int],
    flip_rates: list[float],
    steps_per_run: int = 2000,
    eval_interval: int = 200,
) -> BenchmarkSuite:
    """Benchmark different flip rates to find the optimal range."""
    suite = BenchmarkSuite()
    hw_info = get_hardware_info()

    for rate in flip_rates:
        config = TFLEConfig(
            layer_sizes=layer_sizes,
            flip_rate=rate,
            total_training_steps=steps_per_run,
            eval_interval=eval_interval,
            early_stopping_patience=steps_per_run,
        )
        model = TFLEModel(config)
        trainer = TFLETrainer(model, config, train_loader, val_loader)
        result = trainer.train(verbose=False)

        run = BenchmarkRun(
            name=f"flip_rate_{rate}",
            config={"flip_rate": rate, "layer_sizes": layer_sizes, "steps": steps_per_run},
            result={
                "final_accuracy": result.final_accuracy,
                "training_time": result.training_time_seconds,
                "val_curve": result.val_accuracies,
                "memory_mb": result.memory_usage.get("total_mb", 0),
            },
            hardware_info=hw_info,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        suite.add_run(run)
        acc = result.final_accuracy
        t = result.training_time_seconds
        print(f"  flip_rate={rate:.3f}: acc={acc:.4f} time={t:.1f}s")

    return suite


def benchmark_temperature_schedules(
    train_loader: DataLoader,
    val_loader: DataLoader,
    layer_sizes: list[int],
    initial_temps: list[float],
    cooling_rates: list[float],
    steps_per_run: int = 2000,
    eval_interval: int = 200,
) -> BenchmarkSuite:
    """Benchmark different temperature configurations."""

    suite = BenchmarkSuite()
    hw_info = get_hardware_info()

    for temp in initial_temps:
        for rate in cooling_rates:
            config = TFLEConfig(
                layer_sizes=layer_sizes,
                initial_temperature=temp,
                cooling_rate=rate,
                total_training_steps=steps_per_run,
                eval_interval=eval_interval,
                early_stopping_patience=steps_per_run,
            )
            model = TFLEModel(config)
            trainer = TFLETrainer(model, config, train_loader, val_loader)
            result = trainer.train(verbose=False)

            run = BenchmarkRun(
                name=f"temp_{temp}_rate_{rate}",
                config={"initial_temperature": temp, "cooling_rate": rate},
                result={
                    "final_accuracy": result.final_accuracy,
                    "training_time": result.training_time_seconds,
                    "val_curve": result.val_accuracies,
                },
                hardware_info=hw_info,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            suite.add_run(run)
            print(f"  temp={temp:.1f}, rate={rate:.4f}: acc={result.final_accuracy:.4f}")

    return suite


def benchmark_memory_comparison(
    layer_sizes: list[int],
) -> dict:
    """Compare memory usage between TFLE and backprop+STE."""
    from .baseline import STEBaselineModel

    config = TFLEConfig(layer_sizes=layer_sizes)
    tfle_model = TFLEModel(config)
    ste_model = STEBaselineModel(layer_sizes)

    n_params = tfle_model.get_total_params()
    tfle_mem = tfle_model.get_memory_usage_bytes()

    ste_n_params = sum(p.numel() for p in ste_model.parameters())
    ste_mem = {
        "weight_bytes": ste_n_params * 4,
        "gradient_bytes": ste_n_params * 4,
        "optimizer_bytes": ste_n_params * 8,
        "total_bytes": ste_n_params * 16,
        "total_mb": ste_n_params * 16 / (1024 * 1024),
    }

    return {
        "n_params": n_params,
        "layer_sizes": layer_sizes,
        "tfle": tfle_mem,
        "ste_backprop": ste_mem,
        "memory_ratio": ste_mem["total_mb"] / max(tfle_mem["total_mb"], 0.01),
    }


def run_convergence_analysis(
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TFLEConfig,
    n_seeds: int = 3,
) -> dict:
    """Run multiple seeds and analyze convergence statistics."""
    all_curves = []
    all_final_accs = []
    all_times = []

    for seed in range(n_seeds):
        cfg = TFLEConfig(**{
            **{f.name: getattr(config, f.name) for f in config.__dataclass_fields__.values()},
            "init_seed": seed,
        })
        model = TFLEModel(cfg)
        trainer = TFLETrainer(model, cfg, train_loader, val_loader)
        result = trainer.train(verbose=False)

        all_curves.append(result.val_accuracies)
        all_final_accs.append(result.final_accuracy)
        all_times.append(result.training_time_seconds)
        acc = result.final_accuracy
        t = result.training_time_seconds
        print(f"  Seed {seed}: acc={acc:.4f} time={t:.1f}s")

    import numpy as np

    final_accs = np.array(all_final_accs)
    times = np.array(all_times)

    return {
        "n_seeds": n_seeds,
        "mean_accuracy": float(final_accs.mean()),
        "std_accuracy": float(final_accs.std()),
        "min_accuracy": float(final_accs.min()),
        "max_accuracy": float(final_accs.max()),
        "mean_time": float(times.mean()),
        "std_time": float(times.std()),
        "all_curves": all_curves,
    }
