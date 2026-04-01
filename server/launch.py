"""Config-driven TFLE launcher — reads config.yaml, runs experiments on specified GPUs.

Usage:
    python server/launch.py                    # Run all from config.yaml
    python server/launch.py --gpu 0            # Run only GPU 0 experiments
    python server/launch.py --config my.yaml   # Use custom config
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_gpu_worker(gpu_id: int, experiments: list, config: dict):
    """Run a list of experiments sequentially on one GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from tfle.config import TFLEConfig, FitnessType, CoolingSchedule
    from tfle.model import TFLEModel
    from tfle.training import TFLETrainer
    from tfle.baseline import train_ste_baseline

    results_dir = Path(config["server"]["results_dir"])
    ckpt_dir = Path(config["server"]["checkpoint_dir"])
    results_dir.mkdir(exist_ok=True, parents=True)
    ckpt_dir.mkdir(exist_ok=True, parents=True)

    def get_data(dataset, batch_size):
        if dataset == "mnist":
            t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            tr = datasets.MNIST("./data", train=True, download=True, transform=t)
            te = datasets.MNIST("./data", train=False, download=True, transform=t)
        elif dataset == "cifar10":
            t_tr = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
            t_te = transforms.Compose([transforms.ToTensor(),
                transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
            tr = datasets.CIFAR10("./data", train=True, download=True, transform=t_tr)
            te = datasets.CIFAR10("./data", train=False, download=True, transform=t_te)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        return (DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
                DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True))

    all_results = []
    print(f"\n[GPU {gpu_id}] Starting {len(experiments)} experiments on {torch.cuda.get_device_name(0)}")

    for exp in experiments:
        name = exp["name"]
        print(f"\n{'='*60}\n[GPU {gpu_id}] {name}\n{'='*60}")

        dataset = exp.get("dataset", "mnist")
        batch_size = exp.get("batch_size", 512)
        train_ld, val_ld = get_data(dataset, batch_size)

        # STE baseline
        if exp.get("type") == "ste_baseline":
            print(f"  Running STE baseline...")
            t0 = time.time()
            ste = train_ste_baseline(exp["architecture"], train_ld, val_ld,
                                      total_steps=exp.get("steps", 100_000), verbose=True)
            elapsed = time.time() - t0
            result = {"name": name, "gpu": gpu_id, "type": "ste",
                      "final_accuracy": ste["final_accuracy"], "time_s": elapsed}
            all_results.append(result)
            with open(results_dir / f"{name}.json", "w") as f:
                json.dump(result, f, indent=2)
            print(f"[GPU {gpu_id}] {name}: {ste['final_accuracy']:.4f} ({elapsed:.0f}s)")
            continue

        # TFLE experiment
        cooling_map = {"cosine": CoolingSchedule.COSINE, "exponential": CoolingSchedule.EXPONENTIAL,
                       "adaptive": CoolingSchedule.ADAPTIVE, "linear": CoolingSchedule.LINEAR}

        tfle_config = TFLEConfig(
            layer_sizes=exp["architecture"],
            total_training_steps=exp.get("steps", 100_000),
            eval_interval=exp.get("eval_interval", 500),
            fitness_eval_batch_size=batch_size,
            fitness_type=FitnessType.TASK_LOSS,
            initial_temperature=exp.get("temperature", 0.20),
            min_temperature=exp.get("min_temperature", 0.001),
            cooling_schedule=cooling_map.get(exp.get("cooling", "cosine"), CoolingSchedule.COSINE),
            reheat_on_plateau=True,
            plateau_window=exp.get("plateau_window", 3000),
            reheat_factor=exp.get("reheat_factor", 2.5),
            flip_rate=exp.get("flip_rate", 0.02),
            trace_decay=exp.get("trace_decay", 0.95),
            exploration_rate=exp.get("exploration_rate", 0.005),
            n_proposals=exp.get("n_proposals", 0),
            checkpoint_interval=exp.get("checkpoint_interval", 50_000),
            early_stopping_patience=exp.get("steps", 100_000),
        )

        print(f"  arch={tfle_config.layer_sizes} steps={tfle_config.total_training_steps:,}")
        print(f"  temp={tfle_config.initial_temperature} n_proposals={tfle_config.n_proposals}")

        model = TFLEModel(tfle_config, device="cuda").to("cuda")
        ckpt_path = str(ckpt_dir / name) if exp.get("save_checkpoint") else None
        trainer = TFLETrainer(model, tfle_config, train_ld, val_ld, checkpoint_dir=ckpt_path)

        t0 = time.time()
        result = trainer.train(verbose=True)
        elapsed = time.time() - t0

        best_acc = max((a for _, a in result.val_accuracies), default=0)
        best_step = next((s for s, a in result.val_accuracies if a == best_acc), 0)

        if exp.get("save_checkpoint"):
            model.save_checkpoint(str(ckpt_dir / f"{name}_final.pt"))

        out = {
            "name": name, "gpu": gpu_id, "type": "tfle",
            "best_accuracy": best_acc, "best_step": best_step,
            "final_accuracy": result.final_accuracy, "time_s": elapsed,
            "val_history": [(s, a) for s, a in result.val_accuracies],
            "config": {k: str(v) for k, v in exp.items()},
        }
        all_results.append(out)
        with open(results_dir / f"{name}.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"[GPU {gpu_id}] {name}: best={best_acc:.4f} @ step {best_step:,} ({elapsed:.0f}s)")

    # Save all results for this GPU
    with open(results_dir / f"gpu{gpu_id}_all.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[GPU {gpu_id}] ALL DONE — {len(all_results)} experiments")
    for r in all_results:
        acc = r.get("best_accuracy", r.get("final_accuracy", 0))
        print(f"  {r['name']:<30} {acc:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="server/config.yaml")
    parser.add_argument("--gpu", type=int, default=None, help="Run only this GPU's experiments")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    gpus = config["server"]["gpus"]
    print("=" * 60)
    print("TFLE CONFIG-DRIVEN LAUNCHER")
    print(f"Config: {args.config}")
    print(f"GPUs: {gpus}")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    if args.gpu is not None:
        # Single GPU mode
        key = f"gpu_{args.gpu}"
        if key not in config:
            print(f"No experiments for GPU {args.gpu}")
            return
        run_gpu_worker(args.gpu, config[key], config)
    else:
        # Multi-GPU: spawn one process per GPU
        procs = []
        for gpu_id in gpus:
            key = f"gpu_{gpu_id}"
            if key not in config:
                continue
            # Launch as subprocess to avoid CUDA fork issues
            cmd = [sys.executable, __file__, "--config", args.config, "--gpu", str(gpu_id)]
            log = open(f"gpu{gpu_id}.log", "w")
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd="/workspace/tfle")
            procs.append((gpu_id, p, log))
            print(f"Launched GPU {gpu_id} (PID {p.pid})")

        # Wait for all
        for gpu_id, p, log in procs:
            p.wait()
            log.close()
            print(f"GPU {gpu_id} finished (exit code {p.returncode})")

        # Merge results
        results_dir = Path(config["server"]["results_dir"])
        all_files = list(results_dir.glob("*.json"))
        print(f"\n{'='*60}\nALL COMPLETE — {len(all_files)} result files in {results_dir}")


if __name__ == "__main__":
    main()
