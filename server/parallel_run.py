"""Run TFLE experiments in parallel across 2 GPUs.

GPU 0: Experiments 1, 3a, 4a, 5
GPU 1: Experiments 2, 3b, 4b, 6
STE baselines run sequentially after.
"""

import json
import sys
import time
import os
import multiprocessing as mp
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).parent.parent))

from tfle.config import TFLEConfig, FitnessType, CoolingSchedule, SelectionMethod
from tfle.model import TFLEModel
from tfle.training import TFLETrainer
from tfle.baseline import train_ste_baseline

RESULTS_DIR = Path("/workspace/tfle/results")
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
CHECKPOINTS_DIR = Path("/workspace/tfle/checkpoints")
CHECKPOINTS_DIR.mkdir(exist_ok=True, parents=True)


def get_mnist(batch_size=128):
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST("./data", train=True, download=True, transform=t)
    test = datasets.MNIST("./data", train=False, download=True, transform=t)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True),
    )


def get_cifar10(batch_size=128):
    t_train = transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    t_test = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train = datasets.CIFAR10("./data", train=True, download=True, transform=t_train)
    test = datasets.CIFAR10("./data", train=False, download=True, transform=t_test)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True),
    )


def run_single(name, config, dataset, gpu_id, save_ckpt=False):
    """Run one experiment on a specific GPU. Called as subprocess."""
    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] START: {name}")

    if dataset == "mnist":
        train_loader, val_loader = get_mnist(128)
    else:
        train_loader, val_loader = get_cifar10(128)

    model = TFLEModel(config, device=device).to(device)
    ckpt_dir = str(CHECKPOINTS_DIR / name) if save_ckpt else None
    trainer = TFLETrainer(model, config, train_loader, val_loader, checkpoint_dir=ckpt_dir)

    start = time.time()
    result = trainer.train(verbose=True)
    elapsed = time.time() - start

    best_acc = max((a for _, a in result.val_accuracies), default=0)
    best_step = next((s for s, a in result.val_accuracies if a == best_acc), 0)

    if save_ckpt:
        model.save_checkpoint(str(CHECKPOINTS_DIR / f"{name}_final.pt"))

    out = {
        "name": name, "gpu": gpu_id, "final_accuracy": result.final_accuracy,
        "best_accuracy": best_acc, "best_step": best_step,
        "total_steps": result.total_steps, "training_time_s": elapsed,
        "memory_mb": result.memory_usage.get("total_mb", 0),
        "val_history": [(s, a) for s, a in result.val_accuracies],
        "config": {
            "layer_sizes": config.layer_sizes,
            "initial_temperature": config.initial_temperature,
            "cooling_schedule": config.cooling_schedule.value,
            "selection_method": config.selection_method.value,
            "flip_rate": config.flip_rate,
        },
    }
    # Save individual result
    with open(RESULTS_DIR / f"{name}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[GPU {gpu_id}] DONE: {name} — best={best_acc:.4f} in {elapsed:.0f}s")
    return out


def worker(args):
    """Multiprocessing worker."""
    return run_single(*args)


def main():
    print("=" * 60)
    print("TFLE PARALLEL RUN — 2x RTX 5090")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print("=" * 60)

    # Download data first (single-threaded to avoid race)
    print("Downloading datasets...")
    get_mnist(64)
    get_cifar10(64)
    print("Data ready.\n")

    # ─── Define all experiments ───────────────────────────────
    experiments = []

    # GPU 0 experiments
    experiments.append(("exp1_temp020", TFLEConfig(
        layer_sizes=[784, 256, 10], total_training_steps=100_000, eval_interval=500,
        fitness_eval_batch_size=128, fitness_type=FitnessType.TASK_LOSS,
        initial_temperature=0.20, min_temperature=0.001,
        cooling_schedule=CoolingSchedule.COSINE,
        reheat_on_plateau=True, plateau_window=3000, reheat_factor=2.5,
        flip_rate=0.02, trace_decay=0.95, exploration_rate=0.005,
        early_stopping_patience=100_000,
    ), "mnist", 0, True))

    # GPU 1 experiments
    experiments.append(("exp2_temp010", TFLEConfig(
        layer_sizes=[784, 256, 10], total_training_steps=100_000, eval_interval=500,
        fitness_eval_batch_size=128, fitness_type=FitnessType.TASK_LOSS,
        initial_temperature=0.10, min_temperature=0.0005,
        cooling_schedule=CoolingSchedule.COSINE,
        reheat_on_plateau=True, plateau_window=3000, reheat_factor=2.5,
        flip_rate=0.02, trace_decay=0.95, exploration_rate=0.005,
        early_stopping_patience=100_000,
    ), "mnist", 1, True))

    # Run first pair in parallel
    print("=" * 60)
    print("PHASE 1: Temperature comparison (parallel on 2 GPUs)")
    print("=" * 60)
    start = time.time()
    with mp.Pool(2) as pool:
        results_1 = pool.map(worker, experiments[:2])
    phase1_time = time.time() - start
    print(f"\nPhase 1 done in {phase1_time:.0f}s")

    # Determine better temp
    r1, r2 = results_1
    better = r1 if r1["best_accuracy"] >= r2["best_accuracy"] else r2
    best_temp = better["config"]["initial_temperature"]
    print(f"Better temp: {best_temp} ({better['name']}: {better['best_accuracy']:.4f})")

    # ─── Phase 2: Ablations (parallel) ────────────────────────
    ablations = [
        ("abl_random_select", TFLEConfig(
            layer_sizes=[784, 256, 10], total_training_steps=50_000, eval_interval=500,
            fitness_eval_batch_size=128, fitness_type=FitnessType.TASK_LOSS,
            initial_temperature=best_temp, min_temperature=0.001,
            cooling_schedule=CoolingSchedule.COSINE,
            selection_method=SelectionMethod.UNIFORM_RANDOM,
            flip_rate=0.02, trace_decay=0.95, exploration_rate=0.005,
            early_stopping_patience=50_000,
        ), "mnist", 0, False),
        ("abl_trace_select", TFLEConfig(
            layer_sizes=[784, 256, 10], total_training_steps=50_000, eval_interval=500,
            fitness_eval_batch_size=128, fitness_type=FitnessType.TASK_LOSS,
            initial_temperature=best_temp, min_temperature=0.001,
            cooling_schedule=CoolingSchedule.COSINE,
            selection_method=SelectionMethod.TRACE_WEIGHTED,
            flip_rate=0.02, trace_decay=0.95, exploration_rate=0.005,
            early_stopping_patience=50_000,
        ), "mnist", 1, False),
    ]

    print(f"\n{'='*60}")
    print("PHASE 2: Selection ablation (parallel)")
    print("=" * 60)
    with mp.Pool(2) as pool:
        results_2 = pool.map(worker, ablations)

    # ─── Phase 3: Long run + CIFAR-10 (parallel) ─────────────
    long_runs = [
        ("exp5_500k_best", TFLEConfig(
            layer_sizes=[784, 256, 10], total_training_steps=500_000, eval_interval=2000,
            fitness_eval_batch_size=128, fitness_type=FitnessType.TASK_LOSS,
            initial_temperature=best_temp, min_temperature=0.001,
            cooling_schedule=CoolingSchedule.COSINE,
            reheat_on_plateau=True, plateau_window=5000, reheat_factor=2.5,
            flip_rate=0.02, trace_decay=0.95, exploration_rate=0.005,
            checkpoint_interval=50_000, early_stopping_patience=500_000,
        ), "mnist", 0, True),
        ("exp6_cifar10", TFLEConfig(
            layer_sizes=[3072, 512, 256, 10], total_training_steps=100_000, eval_interval=1000,
            fitness_eval_batch_size=128, fitness_type=FitnessType.TASK_LOSS,
            initial_temperature=best_temp, min_temperature=0.001,
            cooling_schedule=CoolingSchedule.COSINE,
            reheat_on_plateau=True, plateau_window=3000, reheat_factor=2.5,
            flip_rate=0.015, trace_decay=0.95, exploration_rate=0.005,
            early_stopping_patience=100_000,
        ), "cifar10", 1, True),
    ]

    print(f"\n{'='*60}")
    print("PHASE 3: 500K MNIST + CIFAR-10 (parallel)")
    print("=" * 60)
    with mp.Pool(2) as pool:
        results_3 = pool.map(worker, long_runs)

    # ─── Phase 4: STE baselines ───────────────────────────────
    print(f"\n{'='*60}")
    print("PHASE 4: STE baselines")
    print("=" * 60)
    mnist_train, mnist_val = get_mnist(128)
    cifar_train, cifar_val = get_cifar10(128)

    ste_m = train_ste_baseline([784, 256, 10], mnist_train, mnist_val, total_steps=100_000, verbose=True)
    ste_c = train_ste_baseline([3072, 512, 256, 10], cifar_train, cifar_val, total_steps=100_000, verbose=True)

    # ─── Final summary ────────────────────────────────────────
    total_time = time.time() - start
    all_results = {
        "experiments": results_1 + results_2 + results_3,
        "baselines": {
            "ste_mnist": {"final_accuracy": ste_m["final_accuracy"]},
            "ste_cifar10": {"final_accuracy": ste_c["final_accuracy"]},
        },
        "total_time_s": total_time,
    }

    print(f"\n{'='*60}")
    print(f"ALL DONE — {total_time/3600:.1f} hours")
    print(f"{'='*60}")
    for r in all_results["experiments"]:
        print(f"  {r['name']:<25} best={r['best_accuracy']:.4f}  time={r['training_time_s']:.0f}s  gpu={r['gpu']}")
    print(f"  {'STE MNIST':<25} final={ste_m['final_accuracy']:.4f}")
    print(f"  {'STE CIFAR-10':<25} final={ste_c['final_accuracy']:.4f}")

    with open(RESULTS_DIR / "results_final.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR}/results_final.json")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
