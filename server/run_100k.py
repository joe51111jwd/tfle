"""TFLE 100K Step Training — Run on 2x RTX 5090 (vast.ai)

Task-loss fitness with tuned cosine temperature schedule.
Runs TFLE + STE baseline, saves results + checkpoints.

Usage on server:
    cd /workspace/tfle
    pip install torch torchvision tqdm
    python server/run_100k.py
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


def get_mnist_loaders(batch_size: int = 128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test = datasets.MNIST("./data", train=False, download=True, transform=transform)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
    )


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} — {torch.cuda.get_device_properties(i).total_mem / 1e9:.1f} GB")

    total_steps = 100_000

    config = TFLEConfig(
        layer_sizes=[784, 256, 10],
        total_training_steps=total_steps,
        eval_interval=500,
        fitness_eval_batch_size=128,

        # Task-loss fitness
        fitness_type=FitnessType.TASK_LOSS,

        # Tuned cosine temperature
        initial_temperature=0.08,
        min_temperature=0.001,
        cooling_schedule=CoolingSchedule.COSINE,

        # Reheat on plateau
        reheat_on_plateau=True,
        plateau_window=2000,
        reheat_factor=2.0,

        # Flip parameters
        flip_rate=0.02,
        trace_decay=0.95,
        exploration_rate=0.005,
        exploration_decay=True,
        exploration_min=0.001,

        # Checkpointing
        checkpoint_interval=10_000,

        # No early stop
        early_stopping_patience=total_steps,
    )

    train_loader, val_loader = get_mnist_loaders(batch_size=128)
    model = TFLEModel(config)
    trainer = TFLETrainer(model, config, train_loader, val_loader, checkpoint_dir="checkpoints")

    print("=" * 60)
    print("TFLE 100K — Task-Loss Fitness, Cosine Temperature")
    print(f"Architecture: {config.layer_sizes}")
    print(f"Params: {model.get_total_params():,}")
    print(f"Memory: {model.get_memory_usage_bytes()['total_mb']:.2f} MB")
    print(f"Fitness: TASK_LOSS (cross-entropy)")
    print(f"Temperature: {config.initial_temperature} → {config.min_temperature} (cosine)")
    print(f"Reheat: plateau_window={config.plateau_window}, factor={config.reheat_factor}")
    print(f"Steps: {total_steps:,}")
    print(f"Batch size: 128")
    print("=" * 60)

    start = time.time()
    result = trainer.train(verbose=True)
    tfle_time = time.time() - start

    best_acc = max((acc for _, acc in result.val_accuracies), default=0)
    best_step = 0
    for step, acc in result.val_accuracies:
        if acc == best_acc:
            best_step = step
            break

    print(f"\n{'='*60}")
    print(f"TFLE RESULTS — 100K STEPS")
    print(f"{'='*60}")
    print(f"  Final accuracy:  {result.final_accuracy:.4f}")
    print(f"  Best accuracy:   {best_acc:.4f} (step {best_step:,})")
    print(f"  Training time:   {tfle_time:.0f}s ({tfle_time/60:.1f}m)")
    print(f"  Memory:          {result.memory_usage.get('total_mb', 0):.2f} MB")
    print(f"  Steps completed: {result.total_steps:,}")

    # Save checkpoint
    model.save_checkpoint("checkpoints/tfle_100k_final.pt")

    # STE baseline — same architecture, same steps
    print(f"\n{'='*60}")
    print("STE BASELINE — 100K STEPS")
    print(f"{'='*60}")

    ste_start = time.time()
    ste_results = train_ste_baseline(
        config.layer_sizes,
        train_loader,
        val_loader,
        total_steps=total_steps,
        lr=config.learning_rate_ste,
        verbose=True,
    )
    ste_time = time.time() - ste_start

    ste_best = ste_results["final_accuracy"]
    if "val_accuracies" in ste_results:
        ste_best = max((acc for _, acc in ste_results["val_accuracies"]), default=ste_best)

    # Final comparison
    gap = ste_best - best_acc
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Method':<25} {'Best Acc':>10} {'Time':>10} {'Memory':>10}")
    print(f"  {'-'*55}")
    print(f"  {'TFLE (task-loss, tuned)':<25} {best_acc:>9.4f} {tfle_time:>9.0f}s {result.memory_usage.get('total_mb', 0):>9.2f} MB")
    print(f"  {'STE (backprop)':<25} {ste_best:>9.4f} {ste_time:>9.0f}s {'~8':>9} MB")
    print(f"  {'Gap':<25} {gap:>9.4f}")
    print(f"{'='*60}")

    if best_acc >= 0.85:
        print("\n  >>> TARGET HIT: 85%+ ACHIEVED <<<")
    elif best_acc >= 0.50:
        print(f"\n  STRONG: {best_acc:.1%} — scale to more steps or bigger model")
    else:
        print(f"\n  PROGRESS: {best_acc:.1%} — needs more tuning")

    # Save everything
    all_results = {
        "experiment": "TFLE 100K task-loss cosine temp",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": device,
        "config": {
            "layer_sizes": config.layer_sizes,
            "total_steps": total_steps,
            "fitness_type": "TASK_LOSS",
            "initial_temperature": config.initial_temperature,
            "min_temperature": config.min_temperature,
            "cooling_schedule": "COSINE",
            "flip_rate": config.flip_rate,
            "batch_size": 128,
        },
        "tfle": {
            "final_accuracy": result.final_accuracy,
            "best_accuracy": best_acc,
            "best_step": best_step,
            "total_steps": result.total_steps,
            "training_time_s": tfle_time,
            "memory_mb": result.memory_usage.get("total_mb", 0),
            "val_history": [(s, a) for s, a in result.val_accuracies],
        },
        "ste_baseline": {
            "final_accuracy": ste_results["final_accuracy"],
            "best_accuracy": ste_best,
            "training_time_s": ste_time,
        },
        "gap": gap,
        "target_hit": best_acc >= 0.85,
    }

    with open("results_100k.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to results_100k.json")
    print(f"Checkpoint saved to checkpoints/tfle_100k_final.pt")


if __name__ == "__main__":
    run()
