"""GPU 0 — MNIST experiments. Run alongside gpu1.py."""

import json, sys, time, os
from pathlib import Path
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

def run(name, config, train_ld, val_ld, save_ckpt=False):
    print(f"\n{'='*60}\n[GPU 0] {name}\n{'='*60}")
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
    out = {"name": name, "gpu": 0, "best_accuracy": best, "best_step": best_s,
           "final_accuracy": result.final_accuracy, "time_s": elapsed,
           "val_history": [(s, a) for s, a in result.val_accuracies],
           "config": {"temp": config.initial_temperature, "schedule": config.cooling_schedule.value,
                      "selection": config.selection_method.value, "flip_rate": config.flip_rate,
                      "layers": config.layer_sizes}}
    with open(RESULTS / f"{name}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[GPU 0] {name}: best={best:.4f} @ step {best_s:,} ({elapsed:.0f}s)")
    return out

def main():
    print(f"GPU 0: {torch.cuda.get_device_name(0)}")
    train_ld, val_ld = get_mnist(512)
    all_results = []

    # Exp 1: 100K, temp 0.20, cosine
    all_results.append(run("mnist_100k_t020_cosine", TFLEConfig(
        layer_sizes=[784, 512, 256, 10], total_training_steps=100_000, eval_interval=500,
        fitness_eval_batch_size=512, fitness_type=FitnessType.TASK_LOSS,
        initial_temperature=0.20, min_temperature=0.001, cooling_schedule=CoolingSchedule.COSINE,
        reheat_on_plateau=True, plateau_window=3000, reheat_factor=2.5,
        flip_rate=0.02, trace_decay=0.95, exploration_rate=0.005,
        early_stopping_patience=100_000,
    ), train_ld, val_ld, save_ckpt=True))

    # Exp 3a: ablation random selection 50K
    all_results.append(run("abl_random_select", TFLEConfig(
        layer_sizes=[784, 512, 256, 10], total_training_steps=50_000, eval_interval=500,
        fitness_eval_batch_size=512, fitness_type=FitnessType.TASK_LOSS,
        initial_temperature=0.20, min_temperature=0.001, cooling_schedule=CoolingSchedule.COSINE,
        selection_method=SelectionMethod.UNIFORM_RANDOM,
        flip_rate=0.02, trace_decay=0.95, exploration_rate=0.005,
        early_stopping_patience=50_000,
    ), train_ld, val_ld))

    # Exp 5: 500K long run, best config
    all_results.append(run("mnist_500k_long", TFLEConfig(
        layer_sizes=[784, 512, 256, 10], total_training_steps=500_000, eval_interval=2000,
        fitness_eval_batch_size=512, fitness_type=FitnessType.TASK_LOSS,
        initial_temperature=0.20, min_temperature=0.001, cooling_schedule=CoolingSchedule.COSINE,
        reheat_on_plateau=True, plateau_window=5000, reheat_factor=2.5,
        flip_rate=0.02, trace_decay=0.95, exploration_rate=0.005,
        checkpoint_interval=50_000, early_stopping_patience=500_000,
    ), train_ld, val_ld, save_ckpt=True))

    # If time left, run another 500K
    all_results.append(run("mnist_500k_lower_flip", TFLEConfig(
        layer_sizes=[784, 512, 256, 10], total_training_steps=500_000, eval_interval=2000,
        fitness_eval_batch_size=512, fitness_type=FitnessType.TASK_LOSS,
        initial_temperature=0.15, min_temperature=0.0005, cooling_schedule=CoolingSchedule.COSINE,
        reheat_on_plateau=True, plateau_window=5000, reheat_factor=2.0,
        flip_rate=0.01, trace_decay=0.97, exploration_rate=0.003,
        checkpoint_interval=50_000, early_stopping_patience=500_000,
    ), train_ld, val_ld, save_ckpt=True))

    # STE baseline
    print(f"\n[GPU 0] STE BASELINE")
    ste = train_ste_baseline([784, 512, 256, 10], train_ld, val_ld, total_steps=100_000, verbose=True)
    with open(RESULTS / "ste_mnist.json", "w") as f:
        json.dump({"name": "ste_mnist", "final_accuracy": ste["final_accuracy"]}, f, indent=2)

    # Summary
    print(f"\n{'='*60}\nGPU 0 SUMMARY\n{'='*60}")
    for r in all_results:
        print(f"  {r['name']:<30} best={r['best_accuracy']:.4f}")
    print(f"  {'STE baseline':<30} final={ste['final_accuracy']:.4f}")
    with open(RESULTS / "gpu0_all.json", "w") as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()
