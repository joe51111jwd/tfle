"""GPU 1 — CIFAR-10 + MNIST variants. Run alongside gpu0.py."""

import json, sys, time, os
from pathlib import Path
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.insert(0, str(Path(__file__).parent.parent))

from tfle.config import TFLEConfig, FitnessType, CoolingSchedule, SelectionMethod
from tfle.model import TFLEModel
from tfle.training import TFLETrainer
from tfle.baseline import train_ste_baseline

RESULTS = Path("/workspace/tfle/results")
RESULTS.mkdir(exist_ok=True, parents=True)
CKPT = Path("/workspace/tfle/checkpoints")
CKPT.mkdir(exist_ok=True, parents=True)

def get_mnist(bs=512):
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    tr = datasets.MNIST("./data", train=True, download=True, transform=t)
    te = datasets.MNIST("./data", train=False, download=True, transform=t)
    return DataLoader(tr, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True), \
           DataLoader(te, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

def get_cifar10(bs=512):
    t_tr = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
    t_te = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
    tr = datasets.CIFAR10("./data", train=True, download=True, transform=t_tr)
    te = datasets.CIFAR10("./data", train=False, download=True, transform=t_te)
    return DataLoader(tr, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True), \
           DataLoader(te, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

def run(name, config, train_ld, val_ld, save_ckpt=False):
    print(f"\n{'='*60}\n[GPU 1] {name}\n{'='*60}")
    print(f"  arch={config.layer_sizes} steps={config.total_training_steps:,} temp={config.initial_temperature}")
    model = TFLEModel(config, device="cuda").to("cuda")
    ckpt_dir = str(CKPT / name) if save_ckpt else None
    trainer = TFLETrainer(model, config, train_ld, val_ld, checkpoint_dir=ckpt_dir)
    t0 = time.time()
    result = trainer.train(verbose=True)
    elapsed = time.time() - t0
    best = max((a for _, a in result.val_accuracies), default=0)
    best_s = next((s for s, a in result.val_accuracies if a == best), 0)
    if save_ckpt:
        model.save_checkpoint(str(CKPT / f"{name}_final.pt"))
    out = {"name": name, "gpu": 1, "best_accuracy": best, "best_step": best_s,
           "final_accuracy": result.final_accuracy, "time_s": elapsed,
           "val_history": [(s, a) for s, a in result.val_accuracies],
           "config": {"temp": config.initial_temperature, "schedule": config.cooling_schedule.value,
                      "selection": config.selection_method.value, "flip_rate": config.flip_rate,
                      "layers": config.layer_sizes}}
    with open(RESULTS / f"{name}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[GPU 1] {name}: best={best:.4f} @ step {best_s:,} ({elapsed:.0f}s)")
    return out

def main():
    print(f"GPU 1: {torch.cuda.get_device_name(0)}")
    all_results = []

    # Exp 2: MNIST 100K, temp 0.10
    m_tr, m_val = get_mnist(512)
    all_results.append(run("mnist_100k_t010_cosine", TFLEConfig(
        layer_sizes=[784, 512, 256, 10], total_training_steps=100_000, eval_interval=500,
        fitness_eval_batch_size=512, fitness_type=FitnessType.TASK_LOSS,
        initial_temperature=0.10, min_temperature=0.0005, cooling_schedule=CoolingSchedule.COSINE,
        reheat_on_plateau=True, plateau_window=3000, reheat_factor=2.5,
        flip_rate=0.02, trace_decay=0.95, exploration_rate=0.005,
        early_stopping_patience=100_000,
    ), m_tr, m_val, save_ckpt=True))

    # Exp 3b: ablation trace-weighted 50K
    all_results.append(run("abl_trace_weighted", TFLEConfig(
        layer_sizes=[784, 512, 256, 10], total_training_steps=50_000, eval_interval=500,
        fitness_eval_batch_size=512, fitness_type=FitnessType.TASK_LOSS,
        initial_temperature=0.20, min_temperature=0.001, cooling_schedule=CoolingSchedule.COSINE,
        selection_method=SelectionMethod.TRACE_WEIGHTED,
        flip_rate=0.02, trace_decay=0.95, exploration_rate=0.005,
        early_stopping_patience=50_000,
    ), m_tr, m_val))

    # Exp 4: adaptive schedule 50K
    all_results.append(run("abl_adaptive_schedule", TFLEConfig(
        layer_sizes=[784, 512, 256, 10], total_training_steps=50_000, eval_interval=500,
        fitness_eval_batch_size=512, fitness_type=FitnessType.TASK_LOSS,
        initial_temperature=0.20, min_temperature=0.001, cooling_schedule=CoolingSchedule.ADAPTIVE,
        cooling_rate=0.99997,
        flip_rate=0.02, trace_decay=0.95, exploration_rate=0.005,
        early_stopping_patience=50_000,
    ), m_tr, m_val))

    # Exp 6: CIFAR-10 100K
    c_tr, c_val = get_cifar10(512)
    all_results.append(run("cifar10_100k", TFLEConfig(
        layer_sizes=[3072, 512, 256, 10], total_training_steps=100_000, eval_interval=1000,
        fitness_eval_batch_size=512, fitness_type=FitnessType.TASK_LOSS,
        initial_temperature=0.20, min_temperature=0.001, cooling_schedule=CoolingSchedule.COSINE,
        reheat_on_plateau=True, plateau_window=3000, reheat_factor=2.5,
        flip_rate=0.015, trace_decay=0.95, exploration_rate=0.005,
        early_stopping_patience=100_000,
    ), c_tr, c_val, save_ckpt=True))

    # Exp 7: CIFAR-10 500K long
    all_results.append(run("cifar10_500k_long", TFLEConfig(
        layer_sizes=[3072, 512, 256, 10], total_training_steps=500_000, eval_interval=2000,
        fitness_eval_batch_size=512, fitness_type=FitnessType.TASK_LOSS,
        initial_temperature=0.20, min_temperature=0.001, cooling_schedule=CoolingSchedule.COSINE,
        reheat_on_plateau=True, plateau_window=5000, reheat_factor=2.5,
        flip_rate=0.015, trace_decay=0.95, exploration_rate=0.005,
        checkpoint_interval=50_000, early_stopping_patience=500_000,
    ), c_tr, c_val, save_ckpt=True))

    # STE baseline CIFAR-10
    print(f"\n[GPU 1] STE BASELINE CIFAR-10")
    ste = train_ste_baseline([3072, 512, 256, 10], c_tr, c_val, total_steps=100_000, verbose=True)
    with open(RESULTS / "ste_cifar10.json", "w") as f:
        json.dump({"name": "ste_cifar10", "final_accuracy": ste["final_accuracy"]}, f, indent=2)

    print(f"\n{'='*60}\nGPU 1 SUMMARY\n{'='*60}")
    for r in all_results:
        print(f"  {r['name']:<30} best={r['best_accuracy']:.4f}")
    print(f"  {'STE CIFAR-10':<30} final={ste['final_accuracy']:.4f}")
    with open(RESULTS / "gpu1_all.json", "w") as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()
