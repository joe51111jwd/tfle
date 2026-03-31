"""Phase 2: CIFAR-10 with convolutional TFLE model."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from tfle.annealing import TemperatureScheduler
from tfle.config import TFLEConfig
from tfle.conv_model import TFLEConvModel


def get_cifar10_loaders(batch_size: int = 64) -> tuple[DataLoader, DataLoader]:
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    data_dir = Path(__file__).parent.parent / "data"
    train_dataset = datasets.CIFAR10(
        str(data_dir), train=True, download=True, transform=transform_train,
    )
    test_dataset = datasets.CIFAR10(
        str(data_dir), train=False, download=True, transform=transform_test,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def run_cifar10_experiment(
    total_steps: int = 10_000,
    eval_interval: int = 500,
    verbose: bool = True,
) -> dict:
    config = TFLEConfig(
        total_training_steps=total_steps,
        eval_interval=eval_interval,
        fitness_eval_batch_size=64,
        initial_temperature=15.0,
        cooling_rate=0.9998,
        flip_rate=0.03,
        trace_decay=0.95,
        early_stopping_patience=total_steps,
    )

    train_loader, val_loader = get_cifar10_loaders(batch_size=64)
    model = TFLEConvModel(config, n_classes=10)
    scheduler = TemperatureScheduler(config)

    if verbose:
        print("=" * 60)
        print("TFLE ConvNet on CIFAR-10")
        print(f"Total params: {model.get_total_params():,}")
        print(f"Memory usage: {model.get_memory_usage_bytes()['total_mb']:.2f} MB")
        print(f"Steps: {total_steps}")
        print("=" * 60)

    val_accuracies = []
    start_time = time.time()
    step = 0
    pbar = tqdm(total=total_steps, disable=not verbose, desc="CIFAR-10 TFLE")

    while step < total_steps:
        for batch_x, batch_y in train_loader:
            if step >= total_steps:
                break

            temperature = scheduler.get_temperature()
            metrics = model.train_step(batch_x, temperature, batch_y)

            avg_fitness = sum(
                m["fitness_after"] if m["accepted"] else m["fitness_before"]
                for m in metrics
            ) / len(metrics)
            scheduler.step_update(avg_fitness)

            if step % eval_interval == 0:
                total_correct = 0
                total_samples = 0
                for vx, vy in val_loader:
                    result = model.evaluate(vx, vy)
                    total_correct += result["accuracy"] * vx.size(0)
                    total_samples += vx.size(0)
                acc = total_correct / max(total_samples, 1)
                val_accuracies.append((step, acc))
                if verbose:
                    pbar.set_postfix({"acc": f"{acc:.4f}", "temp": f"{temperature:.3f}"})

            step += 1
            pbar.update(1)

    pbar.close()
    training_time = time.time() - start_time

    return {
        "method": "TFLE ConvNet",
        "final_accuracy": val_accuracies[-1][1] if val_accuracies else 0.0,
        "total_steps": step,
        "training_time": training_time,
        "memory_mb": model.get_memory_usage_bytes()["total_mb"],
        "val_accuracies": [{"step": s, "accuracy": a} for s, a in val_accuracies],
    }


if __name__ == "__main__":
    results = run_cifar10_experiment(total_steps=2000, eval_interval=500)
    print(f"\nFinal accuracy: {results['final_accuracy']:.4f}")
    print(f"Training time: {results['training_time']:.1f}s")

    output_path = Path(__file__).parent.parent / "results_phase2_cifar10.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")
