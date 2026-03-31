"""Phase 3: Comprehensive benchmarks and analysis."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).parent.parent))

from tfle.analysis import (
    generate_summary_report,
    plot_memory_comparison,
    plot_training_curves,
)
from tfle.benchmarks import (
    benchmark_flip_rates,
    benchmark_memory_comparison,
    benchmark_temperature_schedules,
    get_hardware_info,
    run_convergence_analysis,
)
from tfle.config import TFLEConfig


def get_mnist_loaders(batch_size: int = 64) -> tuple[DataLoader, DataLoader]:
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


def run_full_benchmarks(
    steps: int = 2000,
    eval_interval: int = 200,
    output_dir: str = "results",
):
    """Run all Phase 3 benchmarks."""
    output_path = Path(__file__).parent.parent / output_dir
    output_path.mkdir(exist_ok=True)

    layer_sizes = [784, 512, 256, 10]
    train_loader, val_loader = get_mnist_loaders()

    print("\n" + "=" * 60)
    print("PHASE 3: COMPREHENSIVE BENCHMARKS")
    print("=" * 60)

    hw_info = get_hardware_info()
    print(f"\nHardware: {hw_info.get('platform', 'unknown')}")
    print(f"PyTorch: {hw_info.get('torch_version', 'unknown')}")
    print(f"Accelerator: {hw_info.get('accelerator', hw_info.get('gpu', 'unknown'))}")

    # 1. Flip rate benchmark
    print("\n--- Benchmark 1: Flip Rate Sensitivity ---")
    flip_rates = [0.01, 0.02, 0.03, 0.05, 0.08]
    flip_suite = benchmark_flip_rates(
        train_loader, val_loader, layer_sizes,
        flip_rates, steps_per_run=steps, eval_interval=eval_interval,
    )
    flip_suite.save(str(output_path / "benchmark_flip_rates.json"))

    # 2. Temperature schedule benchmark
    print("\n--- Benchmark 2: Temperature Schedules ---")
    temp_suite = benchmark_temperature_schedules(
        train_loader, val_loader, layer_sizes,
        initial_temps=[5.0, 10.0, 20.0],
        cooling_rates=[0.9995, 0.9998],
        steps_per_run=steps, eval_interval=eval_interval,
    )
    temp_suite.save(str(output_path / "benchmark_temperatures.json"))

    # 3. Memory comparison
    print("\n--- Benchmark 3: Memory Comparison ---")
    for sizes in [[784, 512, 256, 10], [784, 1024, 512, 256, 10]]:
        mem_data = benchmark_memory_comparison(sizes)
        tfle_mb = mem_data['tfle']['total_mb']
        ste_mb = mem_data['ste_backprop']['total_mb']
        ratio = mem_data['memory_ratio']
        print(f"  {sizes}: TFLE={tfle_mb:.2f}MB, STE={ste_mb:.2f}MB, ratio={ratio:.1f}x")
    plot_memory_comparison(mem_data, save_path=str(output_path / "memory_comparison.png"))

    # 4. Convergence analysis with multiple seeds
    print("\n--- Benchmark 4: Convergence Analysis (3 seeds) ---")
    config = TFLEConfig(
        layer_sizes=layer_sizes,
        total_training_steps=steps,
        eval_interval=eval_interval,
        early_stopping_patience=steps,
    )
    convergence = run_convergence_analysis(train_loader, val_loader, config, n_seeds=3)
    mean = convergence['mean_accuracy']
    std = convergence['std_accuracy']
    print(f"  Mean accuracy: {mean:.4f} +/- {std:.4f}")

    # 5. Training curve comparison (TFLE vs baseline)
    print("\n--- Benchmark 5: TFLE vs Baseline Training Curves ---")
    from experiments.phase1_mnist import run_baseline_experiment, run_tfle_experiment

    tfle_results = run_tfle_experiment(
        total_steps=steps, eval_interval=eval_interval, verbose=False,
    )
    baseline_results = run_baseline_experiment(
        total_steps=steps, eval_interval=eval_interval, verbose=False,
    )

    curves = {
        "TFLE": tfle_results["val_accuracies"],
        "Backprop+STE": baseline_results["val_accuracies"],
    }
    plot_training_curves(
        curves, title="TFLE vs Backprop+STE on MNIST",
        save_path=str(output_path / "training_curves.png"),
    )

    # Summary
    all_results = {
        "tfle": tfle_results,
        "baseline": baseline_results,
        "convergence": convergence,
        "hardware": hw_info,
    }

    report = generate_summary_report(all_results)
    print(report)

    with open(output_path / "full_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    with open(output_path / "summary_report.txt", "w") as f:
        f.write(report)

    print(f"\nAll results saved to {output_path}/")
    return all_results


if __name__ == "__main__":
    run_full_benchmarks(steps=2000, eval_interval=200)
