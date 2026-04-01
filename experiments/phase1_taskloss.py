"""Phase 1 with TASK_LOSS fitness — the fix for TFLE convergence.

This uses cross-entropy loss as the fitness signal instead of contrastive goodness.
Each proposed flip is evaluated by: "did it reduce the model's actual loss?"

Target: >85% accuracy on MNIST (vs 10.31% with contrastive fitness).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).parent.parent))

from tfle.config import TFLEConfig, FitnessType, CoolingSchedule
from tfle.model import TFLEModel
from tfle.training import TFLETrainer
from tfle.baseline import train_ste_baseline


def get_mnist_loaders(batch_size: int = 64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    data_dir = Path(__file__).parent.parent / "data"
    train = datasets.MNIST(str(data_dir), train=True, download=True, transform=transform)
    test = datasets.MNIST(str(data_dir), train=False, download=True, transform=transform)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0),
    )


def run_taskloss_experiment(
    total_steps: int = 20_000,
    eval_interval: int = 500,
):
    """Run TFLE with task-loss fitness on MNIST."""
    config = TFLEConfig(
        layer_sizes=[784, 256, 10],  # Smaller for faster convergence proof
        total_training_steps=total_steps,
        eval_interval=eval_interval,
        fitness_eval_batch_size=64,

        # THE FIX: use task loss instead of contrastive fitness
        fitness_type=FitnessType.TASK_LOSS,

        # Annealing: start low, decay slowly
        initial_temperature=0.5,
        min_temperature=0.001,
        cooling_rate=0.9999,
        cooling_schedule=CoolingSchedule.EXPONENTIAL,

        # Flip parameters
        flip_rate=0.02,
        trace_decay=0.95,
        exploration_rate=0.005,

        # Don't early stop — let it run
        early_stopping_patience=total_steps,
    )

    train_loader, val_loader = get_mnist_loaders(batch_size=64)
    model = TFLEModel(config)
    trainer = TFLETrainer(model, config, train_loader, val_loader)

    print("=" * 60)
    print("TFLE + TASK_LOSS FITNESS on MNIST")
    print(f"Architecture: {config.layer_sizes}")
    print(f"Params: {model.get_total_params():,}")
    print(f"Memory: {model.get_memory_usage_bytes()['total_mb']:.2f} MB")
    print(f"Fitness: TASK_LOSS (cross-entropy)")
    print(f"Steps: {total_steps}")
    print("=" * 60)

    result = trainer.train(verbose=True)

    tfle_results = {
        "method": "TFLE (task_loss)",
        "final_accuracy": result.final_accuracy,
        "total_steps": result.total_steps,
        "training_time": result.training_time_seconds,
        "memory_mb": result.memory_usage.get("total_mb", 0),
        "val_accuracies": result.val_accuracies,
    }

    # Run STE baseline for comparison
    print("\n" + "=" * 60)
    print("STE BASELINE")
    print("=" * 60)

    ste_results = train_ste_baseline(
        config.layer_sizes,
        train_loader,
        val_loader,
        total_steps=20_000,
        lr=config.learning_rate_ste,
        verbose=True,
    )

    # Summary
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  TFLE (task_loss): {tfle_results['final_accuracy']:.4f}")
    print(f"  STE baseline:    {ste_results['final_accuracy']:.4f}")
    print(f"  Gap:             {ste_results['final_accuracy'] - tfle_results['final_accuracy']:.4f}")
    print(f"  TFLE time:       {tfle_results['training_time']:.1f}s")
    print(f"  TFLE memory:     {tfle_results['memory_mb']:.2f} MB")

    # Save
    all_results = {"tfle_taskloss": tfle_results, "ste_baseline": ste_results}
    with open("results_taskloss.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to results_taskloss.json")

    return all_results


if __name__ == "__main__":
    run_taskloss_experiment()
