"""
TFLE Mega Run — 6 Hours on 2x RTX 5090
========================================

Experiment Plan:
  1. MNIST 100K steps, task-loss fitness, cosine temp (0.08→0.001)     ~20 min
  2. MNIST 100K steps, task-loss fitness, lower temp (0.02→0.001)      ~20 min
  3. MNIST ablation: random selection vs trace-weighted (50K each)     ~20 min
  4. MNIST ablation: fixed temp vs cosine vs adaptive (50K each)       ~30 min
  5. MNIST 500K steps, best config from above                          ~90 min
  6. CIFAR-10 100K steps, best config                                  ~60 min
  7. STE baselines for all above                                       ~20 min

Total estimated: ~4.5 hrs, leaving buffer for setup + results.
All results saved to /workspace/tfle/results/
"""

import json
import sys
import time
import os
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
START_TIME = time.time()


def elapsed():
    return f"{(time.time() - START_TIME) / 60:.1f}m"


def get_mnist(batch_size=128):
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST("./data", train=True, download=True, transform=t)
    test = datasets.MNIST("./data", train=False, download=True, transform=t)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
    )


def get_cifar10(batch_size=128):
    t_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    t_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train = datasets.CIFAR10("./data", train=True, download=True, transform=t_train)
    test = datasets.CIFAR10("./data", train=False, download=True, transform=t_test)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
    )


def run_experiment(name, config, train_loader, val_loader, save_checkpoint=False):
    """Run a single TFLE experiment and return results."""
    print(f"\n{'='*60}")
    print(f"[{elapsed()}] EXPERIMENT: {name}")
    print(f"{'='*60}")
    print(f"  Architecture: {config.layer_sizes}")
    print(f"  Steps: {config.total_training_steps:,}")
    print(f"  Fitness: {config.fitness_type.value}")
    print(f"  Temp: {config.initial_temperature} → {config.min_temperature} ({config.cooling_schedule.value})")
    print(f"  Selection: {config.selection_method.value}")
    print(f"  Flip rate: {config.flip_rate}")

    model = TFLEModel(config, device=DEVICE).to(DEVICE)
    ckpt_dir = str(CHECKPOINTS_DIR / name.replace(" ", "_")) if save_checkpoint else None
    trainer = TFLETrainer(model, config, train_loader, val_loader, checkpoint_dir=ckpt_dir)

    start = time.time()
    result = trainer.train(verbose=True)
    train_time = time.time() - start

    best_acc = max((acc for _, acc in result.val_accuracies), default=0)
    best_step = 0
    for step, acc in result.val_accuracies:
        if acc == best_acc:
            best_step = step
            break

    if save_checkpoint:
        model.save_checkpoint(str(CHECKPOINTS_DIR / f"{name.replace(' ', '_')}_final.pt"))

    print(f"  [{elapsed()}] Done: best={best_acc:.4f} @ step {best_step:,}, time={train_time:.0f}s")

    return {
        "name": name,
        "final_accuracy": result.final_accuracy,
        "best_accuracy": best_acc,
        "best_step": best_step,
        "total_steps": result.total_steps,
        "training_time_s": train_time,
        "memory_mb": result.memory_usage.get("total_mb", 0),
        "val_history": [(s, a) for s, a in result.val_accuracies],
        "config": {
            "layer_sizes": config.layer_sizes,
            "fitness_type": config.fitness_type.value,
            "initial_temperature": config.initial_temperature,
            "min_temperature": config.min_temperature,
            "cooling_schedule": config.cooling_schedule.value,
            "selection_method": config.selection_method.value,
            "flip_rate": config.flip_rate,
            "exploration_rate": config.exploration_rate,
        },
    }


def run_ste_baseline(name, layer_sizes, train_loader, val_loader, total_steps=50_000):
    """Run STE baseline for comparison."""
    print(f"\n[{elapsed()}] STE BASELINE: {name}")
    start = time.time()
    results = train_ste_baseline(layer_sizes, train_loader, val_loader,
                                  total_steps=total_steps, lr=0.001, verbose=True)
    train_time = time.time() - start
    print(f"  [{elapsed()}] Done: {results['final_accuracy']:.4f}, time={train_time:.0f}s")
    return {
        "name": name,
        "final_accuracy": results["final_accuracy"],
        "training_time_s": train_time,
    }


def save_results(all_results, filename):
    path = RESULTS_DIR / filename
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved: {path}")


def main():
    print("=" * 60)
    print("TFLE MEGA RUN — 6 Hours on 2x RTX 5090")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print("=" * 60)

    mnist_train, mnist_val = get_mnist(batch_size=128)
    all_results = {}

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT 1: MNIST 100K, cosine temp 0.08
    # ═══════════════════════════════════════════════════════════
    config1 = TFLEConfig(
        layer_sizes=[784, 256, 10],
        total_training_steps=100_000,
        eval_interval=500,
        fitness_eval_batch_size=128,
        fitness_type=FitnessType.TASK_LOSS,
        initial_temperature=0.08,
        min_temperature=0.001,
        cooling_schedule=CoolingSchedule.COSINE,
        reheat_on_plateau=True,
        plateau_window=2000,
        reheat_factor=2.0,
        flip_rate=0.02,
        trace_decay=0.95,
        exploration_rate=0.005,
        early_stopping_patience=100_000,
    )
    r1 = run_experiment("mnist_100k_cosine_008", config1, mnist_train, mnist_val, save_checkpoint=True)
    all_results["exp1_cosine_008"] = r1
    save_results(all_results, "results_running.json")

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT 2: MNIST 100K, lower temp 0.02
    # ═══════════════════════════════════════════════════════════
    config2 = TFLEConfig(
        layer_sizes=[784, 256, 10],
        total_training_steps=100_000,
        eval_interval=500,
        fitness_eval_batch_size=128,
        fitness_type=FitnessType.TASK_LOSS,
        initial_temperature=0.02,
        min_temperature=0.0005,
        cooling_schedule=CoolingSchedule.COSINE,
        reheat_on_plateau=True,
        plateau_window=2000,
        reheat_factor=2.0,
        flip_rate=0.02,
        trace_decay=0.95,
        exploration_rate=0.005,
        early_stopping_patience=100_000,
    )
    r2 = run_experiment("mnist_100k_cosine_002", config2, mnist_train, mnist_val, save_checkpoint=True)
    all_results["exp2_cosine_002"] = r2
    save_results(all_results, "results_running.json")

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT 3: Ablation — random vs trace-weighted selection
    # ═══════════════════════════════════════════════════════════
    # Use whichever temp was better from exp1 vs exp2
    better_temp = 0.08 if r1["best_accuracy"] >= r2["best_accuracy"] else 0.02
    print(f"\n[{elapsed()}] Better temp: {better_temp} (exp1={r1['best_accuracy']:.4f}, exp2={r2['best_accuracy']:.4f})")

    config3a = TFLEConfig(
        layer_sizes=[784, 256, 10],
        total_training_steps=50_000,
        eval_interval=500,
        fitness_eval_batch_size=128,
        fitness_type=FitnessType.TASK_LOSS,
        initial_temperature=better_temp,
        min_temperature=0.001,
        cooling_schedule=CoolingSchedule.COSINE,
        selection_method=SelectionMethod.UNIFORM_RANDOM,
        flip_rate=0.02,
        trace_decay=0.95,
        exploration_rate=0.005,
        early_stopping_patience=50_000,
    )
    r3a = run_experiment("ablation_random_selection", config3a, mnist_train, mnist_val)
    all_results["exp3a_random_select"] = r3a

    config3b = TFLEConfig(
        layer_sizes=[784, 256, 10],
        total_training_steps=50_000,
        eval_interval=500,
        fitness_eval_batch_size=128,
        fitness_type=FitnessType.TASK_LOSS,
        initial_temperature=better_temp,
        min_temperature=0.001,
        cooling_schedule=CoolingSchedule.COSINE,
        selection_method=SelectionMethod.TRACE_WEIGHTED,
        flip_rate=0.02,
        trace_decay=0.95,
        exploration_rate=0.005,
        early_stopping_patience=50_000,
    )
    r3b = run_experiment("ablation_trace_weighted", config3b, mnist_train, mnist_val)
    all_results["exp3b_trace_weighted"] = r3b
    save_results(all_results, "results_running.json")

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT 4: Ablation — cooling schedules
    # ═══════════════════════════════════════════════════════════
    for schedule_name, schedule in [("exponential", CoolingSchedule.EXPONENTIAL),
                                     ("adaptive", CoolingSchedule.ADAPTIVE)]:
        cfg = TFLEConfig(
            layer_sizes=[784, 256, 10],
            total_training_steps=50_000,
            eval_interval=500,
            fitness_eval_batch_size=128,
            fitness_type=FitnessType.TASK_LOSS,
            initial_temperature=better_temp,
            min_temperature=0.001,
            cooling_schedule=schedule,
            cooling_rate=0.99997,  # for exponential
            flip_rate=0.02,
            trace_decay=0.95,
            exploration_rate=0.005,
            early_stopping_patience=50_000,
        )
        r = run_experiment(f"ablation_schedule_{schedule_name}", cfg, mnist_train, mnist_val)
        all_results[f"exp4_{schedule_name}"] = r

    save_results(all_results, "results_running.json")

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT 5: MNIST 500K steps, best config
    # ═══════════════════════════════════════════════════════════
    # Find best config from experiments so far
    best_exp = max(all_results.values(), key=lambda x: x.get("best_accuracy", 0))
    print(f"\n[{elapsed()}] Best config so far: {best_exp['name']} ({best_exp['best_accuracy']:.4f})")
    best_cfg = best_exp["config"]

    config5 = TFLEConfig(
        layer_sizes=best_cfg["layer_sizes"],
        total_training_steps=500_000,
        eval_interval=2000,
        fitness_eval_batch_size=128,
        fitness_type=FitnessType.TASK_LOSS,
        initial_temperature=best_cfg["initial_temperature"],
        min_temperature=float(best_cfg.get("min_temperature", 0.001)),
        cooling_schedule=CoolingSchedule(best_cfg["cooling_schedule"]),
        reheat_on_plateau=True,
        plateau_window=5000,
        reheat_factor=2.0,
        flip_rate=float(best_cfg.get("flip_rate", 0.02)),
        trace_decay=0.95,
        exploration_rate=float(best_cfg.get("exploration_rate", 0.005)),
        checkpoint_interval=50_000,
        early_stopping_patience=500_000,
    )
    r5 = run_experiment("mnist_500k_best_config", config5, mnist_train, mnist_val, save_checkpoint=True)
    all_results["exp5_500k_best"] = r5
    save_results(all_results, "results_running.json")

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT 6: CIFAR-10 100K steps
    # ═══════════════════════════════════════════════════════════
    print(f"\n[{elapsed()}] Loading CIFAR-10...")
    cifar_train, cifar_val = get_cifar10(batch_size=128)

    config6 = TFLEConfig(
        layer_sizes=[3072, 512, 256, 10],  # 32x32x3 = 3072
        total_training_steps=100_000,
        eval_interval=1000,
        fitness_eval_batch_size=128,
        fitness_type=FitnessType.TASK_LOSS,
        initial_temperature=best_cfg["initial_temperature"],
        min_temperature=float(best_cfg.get("min_temperature", 0.001)),
        cooling_schedule=CoolingSchedule(best_cfg["cooling_schedule"]),
        reheat_on_plateau=True,
        plateau_window=3000,
        reheat_factor=2.0,
        flip_rate=0.015,
        trace_decay=0.95,
        exploration_rate=0.005,
        early_stopping_patience=100_000,
    )
    r6 = run_experiment("cifar10_100k", config6, cifar_train, cifar_val, save_checkpoint=True)
    all_results["exp6_cifar10"] = r6
    save_results(all_results, "results_running.json")

    # ═══════════════════════════════════════════════════════════
    # BASELINES
    # ═══════════════════════════════════════════════════════════
    print(f"\n[{elapsed()}] Running STE baselines...")

    ste_mnist = run_ste_baseline("STE_MNIST", [784, 256, 10], mnist_train, mnist_val, total_steps=100_000)
    all_results["ste_mnist"] = ste_mnist

    ste_cifar = run_ste_baseline("STE_CIFAR10", [3072, 512, 256, 10], cifar_train, cifar_val, total_steps=100_000)
    all_results["ste_cifar10"] = ste_cifar

    # ═══════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════
    total_time = time.time() - START_TIME

    print(f"\n{'='*70}")
    print(f"MEGA RUN COMPLETE — {total_time/3600:.1f} hours")
    print(f"{'='*70}")
    print(f"\n{'Experiment':<35} {'Best Acc':>10} {'Time':>10}")
    print(f"{'-'*55}")
    for key, val in all_results.items():
        name = val.get("name", key)
        acc = val.get("best_accuracy", val.get("final_accuracy", 0))
        t = val.get("training_time_s", 0)
        print(f"  {name:<33} {acc:>9.4f} {t:>9.0f}s")

    mnist_best = max(
        (v.get("best_accuracy", 0) for k, v in all_results.items() if "mnist" in k and "ste" not in k),
        default=0
    )
    cifar_best = all_results.get("exp6_cifar10", {}).get("best_accuracy", 0)
    ste_m = all_results.get("ste_mnist", {}).get("final_accuracy", 0)
    ste_c = all_results.get("ste_cifar10", {}).get("final_accuracy", 0)

    print(f"\n{'='*55}")
    print(f"  MNIST:   TFLE best = {mnist_best:.4f}, STE = {ste_m:.4f}, gap = {ste_m - mnist_best:.4f}")
    print(f"  CIFAR10: TFLE best = {cifar_best:.4f}, STE = {ste_c:.4f}, gap = {ste_c - cifar_best:.4f}")
    print(f"  Total time: {total_time/3600:.1f} hours")
    print(f"{'='*55}")

    if mnist_best >= 0.85:
        print("\n  >>> MNIST TARGET HIT: 85%+ <<<")
    if cifar_best >= 0.40:
        print(f"  >>> CIFAR-10 LEARNING CONFIRMED <<<")

    save_results(all_results, "results_final.json")
    print(f"\nAll results saved to {RESULTS_DIR}/results_final.json")
    print(f"Checkpoints in {CHECKPOINTS_DIR}/")


if __name__ == "__main__":
    main()
