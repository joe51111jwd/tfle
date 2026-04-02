"""Full System Test: TFLE + CDLL + SWT on CIFAR-100.

Phase 1: 940K params, 50K steps → verify stack works (>15% acc)
Phase 2: 100M params, 100K+ steps → saturate GPUs (>80% util, >20% acc)
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
from tfle.gpu_engine import SearchParallelEngine

RESULTS = Path("results"); RESULTS.mkdir(exist_ok=True)
CKPT = Path("checkpoints"); CKPT.mkdir(exist_ok=True)
SEP = "=" * 60


def get_cifar100(batch_size=512):
    t_tr = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    t_te = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    tr = datasets.CIFAR100("./data", train=True, download=True, transform=t_tr)
    te = datasets.CIFAR100("./data", train=False, download=True, transform=t_te)
    return (
        DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
        DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
    )


def evaluate(model, val_loader):
    correct = total = 0
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for x, y in val_loader:
            if x.dim() > 2:
                x = x.view(x.size(0), -1)
            result = model.evaluate(x, y)
            correct += result["accuracy"] * x.size(0)
            total += x.size(0)
            total_loss += result["loss"]
            n += 1
    return {"accuracy": correct / max(total, 1), "loss": total_loss / max(n, 1)}


def run_phase(name, config, train_loader, val_loader, total_steps):
    print(f"\n{SEP}\n{name}\n{SEP}")
    print(f"  arch={config.layer_sizes}")
    print(f"  params={sum(a*b for a,b in zip(config.layer_sizes[:-1], config.layer_sizes[1:])):,}")
    print(f"  steps={total_steps:,} | K={config.num_parallel_proposals}")
    print(f"  multi_gpu={config.multi_gpu} | swt={config.swt_enabled}")
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    for i in range(n_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    model = TFLEModel(config)
    engine = SearchParallelEngine(model, config)
    print(f"  eval_batch={engine.eval_batch_size} | K_actual={engine.K}")

    train_iter = iter(train_loader)
    start = time.time()
    best_acc = 0.0
    history = []

    for step in range(total_steps):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        result = engine.train_step(x, y)

        # Eval periodically
        if step % 500 == 0:
            ev = evaluate(model, val_loader)
            best_acc = max(best_acc, ev["accuracy"])
            elapsed = time.time() - start
            steps_per_sec = (step + 1) / max(elapsed, 1)

            status = engine.get_status()
            print(f"  [{elapsed:.0f}s] Step {step:,} | Acc: {ev['accuracy']:.4f} | "
                  f"Best: {best_acc:.4f} | {steps_per_sec:.1f} it/s")
            print(f"    {status}")
            history.append((step, ev["accuracy"], ev["loss"]))

    # Final eval
    ev = evaluate(model, val_loader)
    best_acc = max(best_acc, ev["accuracy"])
    elapsed = time.time() - start

    model.save_checkpoint(str(CKPT / f"{name}_final.pt"))

    result_data = {
        "name": name, "best_accuracy": best_acc, "final_accuracy": ev["accuracy"],
        "total_steps": total_steps, "time_s": elapsed,
        "params": sum(a * b for a, b in zip(config.layer_sizes[:-1], config.layer_sizes[1:])),
        "history": history,
    }
    with open(RESULTS / f"{name}.json", "w") as f:
        json.dump(result_data, f, indent=2)

    print(f"\n  {name}: best={best_acc:.4f} | time={elapsed:.0f}s ({elapsed/60:.1f}m)")
    return result_data


def main():
    print(SEP)
    print("TFLE FULL SYSTEM — CIFAR-100")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GPUs: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    print(SEP)

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_mg = n_gpus >= 2

    # Phase 1: 940K params, verify stack works
    train_ld, val_ld = get_cifar100(512)

    phase1 = run_phase("phase1_verify", TFLEConfig(
        layer_sizes=[3072, 256, 256, 192, 128, 100],
        total_training_steps=50_000,
        eval_interval=500,
        fitness_eval_batch_size=512,
        fitness_type=FitnessType.CDLL,
        num_parallel_proposals=32,
        multi_gpu=use_mg,
        gpu_devices=list(range(n_gpus)) if n_gpus > 0 else [0],
        initial_temperature=2.0,
        min_temperature=0.01,
        cooling_schedule=CoolingSchedule.COSINE,
        reheat_on_plateau=True,
        plateau_window=2000,
        reheat_factor=2.0,
        flip_rate=0.03,
        swt_enabled=True,
        wake_steps=900,
        sleep_steps=100,
        early_stopping_patience=50_000,
    ), train_ld, val_ld, total_steps=50_000)

    phase1_ok = phase1["best_accuracy"] > 0.01  # >1% (random = 1%)
    print(f"\nPhase 1 {'PASSED' if phase1_ok else 'NEEDS WORK'}: {phase1['best_accuracy']:.4f}")

    if not phase1_ok:
        print("Phase 1 did not pass. Skipping Phase 2. Check CDLL fitness.")
        return

    # Phase 2: 100M params, saturate GPUs
    train_ld2, val_ld2 = get_cifar100(256)  # smaller batch for bigger model

    phase2 = run_phase("phase2_saturate", TFLEConfig(
        layer_sizes=[3072, 4096, 4096, 2048, 1024, 512, 100],
        total_training_steps=100_000,
        eval_interval=1000,
        fitness_eval_batch_size=4096,
        fitness_type=FitnessType.CDLL,
        num_parallel_proposals=16,
        multi_gpu=use_mg,
        gpu_devices=list(range(n_gpus)) if n_gpus > 0 else [0],
        initial_temperature=2.0,
        min_temperature=0.01,
        cooling_schedule=CoolingSchedule.COSINE,
        reheat_on_plateau=True,
        plateau_window=2000,
        reheat_factor=2.0,
        flip_rate=0.02,
        swt_enabled=True,
        wake_steps=900,
        sleep_steps=100,
        early_stopping_patience=100_000,
    ), train_ld2, val_ld2, total_steps=100_000)

    print(f"\n{SEP}")
    print("FINAL RESULTS")
    print(SEP)
    print(f"  Phase 1 (940K):  {phase1['best_accuracy']:.4f} ({phase1['time_s']:.0f}s)")
    print(f"  Phase 2 (100M):  {phase2['best_accuracy']:.4f} ({phase2['time_s']:.0f}s)")
    print(SEP)


if __name__ == "__main__":
    main()
