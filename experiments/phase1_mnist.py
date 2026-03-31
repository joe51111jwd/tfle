"""Phase 1: MNIST proof-of-concept for TFLE vs backprop+STE baseline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Add parent to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from tfle.baseline import train_ste_baseline
from tfle.config import TFLEConfig
from tfle.model import TFLEModel
from tfle.training import TFLETrainer


def get_mnist_loaders(batch_size: int = 64) -> tuple[DataLoader, DataLoader]:
    """Load MNIST train and test sets."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    data_dir = Path(__file__).parent.parent / "data"
    train_dataset = datasets.MNIST(str(data_dir), train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(str(data_dir), train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader


def run_tfle_experiment(
    total_steps: int = 10_000,
    eval_interval: int = 200,
    verbose: bool = True,
) -> dict:
    """Run TFLE training on MNIST."""
    config = TFLEConfig(
        layer_sizes=[784, 512, 256, 10],
        total_training_steps=total_steps,
        eval_interval=eval_interval,
        fitness_eval_batch_size=64,
        initial_temperature=10.0,
        cooling_rate=0.9997,
        flip_rate=0.03,
        trace_decay=0.95,
        early_stopping_patience=total_steps,  # don't early stop for Phase 1
    )

    train_loader, val_loader = get_mnist_loaders(batch_size=config.fitness_eval_batch_size)
    model = TFLEModel(config)
    trainer = TFLETrainer(model, config, train_loader, val_loader)

    if verbose:
        print("=" * 60)
        print("TFLE Training on MNIST")
        print(f"Architecture: {config.layer_sizes}")
        print(f"Total params: {model.get_total_params():,}")
        print(f"Memory usage: {model.get_memory_usage_bytes()['total_mb']:.2f} MB")
        print(f"Steps: {total_steps}, Eval every: {eval_interval}")
        print("=" * 60)

    result = trainer.train(verbose=verbose)

    return {
        "method": "TFLE",
        "final_accuracy": result.final_accuracy,
        "total_steps": result.total_steps,
        "training_time": result.training_time_seconds,
        "memory_mb": result.memory_usage.get("total_mb", 0),
        "val_accuracies": result.val_accuracies,
        "stopped_early": result.stopped_early,
    }


def run_baseline_experiment(
    total_steps: int = 10_000,
    eval_interval: int = 200,
    verbose: bool = True,
) -> dict:
    """Run STE baseline training on MNIST."""
    layer_sizes = [784, 512, 256, 10]
    train_loader, val_loader = get_mnist_loaders(batch_size=64)

    if verbose:
        print("=" * 60)
        print("STE Baseline Training on MNIST")
        print(f"Architecture: {layer_sizes}")
        print("=" * 60)

    model, result = train_ste_baseline(
        layer_sizes=layer_sizes,
        train_loader=train_loader,
        val_loader=val_loader,
        total_steps=total_steps,
        lr=0.001,
        eval_interval=eval_interval,
        verbose=verbose,
    )

    return {
        "method": "STE Baseline",
        "final_accuracy": result.final_accuracy,
        "total_steps": result.total_steps,
        "training_time": result.training_time_seconds,
        "memory_mb": result.memory_usage.get("total_mb", 0),
        "val_accuracies": result.val_accuracies,
    }


def run_comparison(total_steps: int = 5000, eval_interval: int = 200):
    """Run both methods and compare."""
    print("\n" + "=" * 70)
    print("PHASE 1: TFLE vs STE Baseline on MNIST")
    print("=" * 70 + "\n")

    tfle_results = run_tfle_experiment(total_steps, eval_interval)
    print()
    baseline_results = run_baseline_experiment(total_steps, eval_interval)

    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(f"{'Metric':<25} {'TFLE':>15} {'STE Baseline':>15}")
    print("-" * 55)
    t, b = tfle_results, baseline_results
    print(f"{'Final Accuracy':<25} {t['final_accuracy']:>14.4f} {b['final_accuracy']:>14.4f}")
    print(f"{'Training Time (s)':<25} {t['training_time']:>14.1f} {b['training_time']:>14.1f}")
    print(f"{'Memory (MB)':<25} {t['memory_mb']:>14.2f} {b['memory_mb']:>14.2f}")
    print(f"{'Total Steps':<25} {t['total_steps']:>15} {b['total_steps']:>15}")

    results = {"tfle": tfle_results, "baseline": baseline_results}

    # Convert non-serializable items
    for key in results:
        results[key]["val_accuracies"] = [
            {"step": s, "accuracy": a} for s, a in results[key]["val_accuracies"]
        ]

    output_path = Path(__file__).parent.parent / "results_phase1.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    run_comparison(total_steps=5000, eval_interval=200)
