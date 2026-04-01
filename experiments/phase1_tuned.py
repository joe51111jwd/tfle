"""Phase 1 TUNED: Task-loss fitness with proper temperature schedule.

Previous run: 23.54% at 20K steps (temp=0.5, barely decayed, 28% accept rate)
This run: cosine temp schedule (0.08 → 0.001), 100K steps, targeting >85%

Key changes from previous:
  - Temperature: 0.08 start (was 0.5) — much more selective from the start
  - Schedule: cosine decay (was exponential with rate=0.9999 that barely moved)
  - Steps: 100K (was 20K) — 5x more search
  - Reheat on plateau: enabled — escape local optima
  - Eval every 1000 steps (was 500) — less overhead
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
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


def run():
    total_steps = 5_000  # Quick run (~2 min), scale to 100K on server

    config = TFLEConfig(
        layer_sizes=[784, 256, 10],
        total_training_steps=total_steps,
        eval_interval=250,
        fitness_eval_batch_size=64,

        # THE FIX: task-loss fitness
        fitness_type=FitnessType.TASK_LOSS,

        # TUNED temperature: low start, cosine decay
        initial_temperature=0.08,
        min_temperature=0.001,
        cooling_schedule=CoolingSchedule.COSINE,

        # Reheat if stuck
        reheat_on_plateau=True,
        plateau_window=2000,
        reheat_factor=2.0,

        # Flip parameters
        flip_rate=0.02,
        trace_decay=0.95,
        exploration_rate=0.005,
        exploration_decay=True,
        exploration_min=0.001,

        # Don't early stop
        early_stopping_patience=total_steps,
    )

    train_loader, val_loader = get_mnist_loaders(batch_size=64)
    model = TFLEModel(config)
    trainer = TFLETrainer(model, config, train_loader, val_loader)

    print("=" * 60)
    print("TFLE TUNED — Task-Loss Fitness, Cosine Temperature")
    print(f"Architecture: {config.layer_sizes}")
    print(f"Params: {model.get_total_params():,}")
    print(f"Memory: {model.get_memory_usage_bytes()['total_mb']:.2f} MB")
    print(f"Fitness: TASK_LOSS")
    print(f"Temperature: {config.initial_temperature} → {config.min_temperature} (cosine)")
    print(f"Reheat: on plateau (window={config.plateau_window}, factor={config.reheat_factor})")
    print(f"Steps: {total_steps:,}")
    print("=" * 60)

    start = time.time()
    result = trainer.train(verbose=True)
    tfle_time = time.time() - start

    tfle_results = {
        "method": "TFLE (task_loss, tuned)",
        "final_accuracy": result.final_accuracy,
        "total_steps": result.total_steps,
        "training_time": tfle_time,
        "memory_mb": result.memory_usage.get("total_mb", 0),
        "val_accuracies": result.val_accuracies,
    }

    # Find best accuracy
    best_acc = max((acc for _, acc in result.val_accuracies), default=0)
    best_step = 0
    for step, acc in result.val_accuracies:
        if acc == best_acc:
            best_step = step
            break

    print(f"\n{'='*60}")
    print(f"TFLE RESULTS")
    print(f"{'='*60}")
    print(f"  Final accuracy:  {result.final_accuracy:.4f}")
    print(f"  Best accuracy:   {best_acc:.4f} (step {best_step:,})")
    print(f"  Training time:   {tfle_time:.0f}s ({tfle_time/60:.1f}m)")
    print(f"  Memory:          {result.memory_usage.get('total_mb', 0):.2f} MB")

    # Run STE baseline (same architecture, 100K steps)
    print(f"\n{'='*60}")
    print("STE BASELINE")
    print(f"{'='*60}")

    ste_results = train_ste_baseline(
        config.layer_sizes,
        train_loader,
        val_loader,
        total_steps=5_000,
        lr=config.learning_rate_ste,
        verbose=True,
    )

    ste_best = max((acc for _, acc in ste_results.get("val_accuracies", [(0, ste_results["final_accuracy"])])), default=ste_results["final_accuracy"])

    # Summary
    gap = ste_best - best_acc
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"  TFLE (tuned):    {best_acc:.4f}")
    print(f"  STE baseline:    {ste_best:.4f}")
    print(f"  Gap:             {gap:.4f} ({gap*100:.1f}pp)")
    print(f"  TFLE memory:     {result.memory_usage.get('total_mb', 0):.2f} MB")

    if best_acc > 0.85:
        print(f"\n  TARGET HIT: >85% accuracy achieved!")
    elif best_acc > 0.50:
        print(f"\n  STRONG PROGRESS: >50%, scaling should close the gap")
    elif best_acc > 0.25:
        print(f"\n  IMPROVEMENT over previous (23.5%), tuning is working")
    else:
        print(f"\n  NEEDS MORE WORK")

    # Save results
    all_results = {
        "tfle_tuned": tfle_results,
        "ste_baseline": {
            "method": "STE",
            "final_accuracy": ste_results["final_accuracy"],
            "total_steps": ste_results.get("total_steps", total_steps),
            "training_time": ste_results.get("training_time", 0),
        },
        "best_tfle_accuracy": best_acc,
        "best_tfle_step": best_step,
        "gap_to_ste": gap,
    }
    with open("results_tuned.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to results_tuned.json")


if __name__ == "__main__":
    run()
