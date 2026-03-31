"""Analysis and visualization tools for TFLE results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(
    results: dict[str, list[tuple[int, float]]],
    title: str = "Training Curves",
    ylabel: str = "Accuracy",
    save_path: str | None = None,
):
    """Plot training curves for multiple methods."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for name, curve in results.items():
        steps = [p[0] if isinstance(p, (list, tuple)) else p["step"] for p in curve]
        values = [p[1] if isinstance(p, (list, tuple)) else p["accuracy"] for p in curve]
        ax.plot(steps, values, label=name, linewidth=2)

    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    plt.close()
    return fig


def plot_memory_comparison(
    memory_data: dict,
    save_path: str | None = None,
):
    """Bar chart comparing memory usage between methods."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    categories = ["Weights", "Traces/Gradients", "Optimizer"]

    tfle = memory_data["tfle"]
    ste = memory_data["ste_backprop"]

    tfle_vals = [
        tfle["weight_bytes"] / (1024**2),
        tfle["trace_bytes"] / (1024**2),
        0,
    ]
    ste_vals = [
        ste["weight_bytes"] / (1024**2),
        ste["gradient_bytes"] / (1024**2),
        ste["optimizer_bytes"] / (1024**2),
    ]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width / 2, tfle_vals, width, label="TFLE", color="#2196F3")
    bars2 = ax.bar(x + width / 2, ste_vals, width, label="Backprop+STE", color="#FF5722")

    ax.set_ylabel("Memory (MB)", fontsize=12)
    ax.set_title(f"Memory Usage Comparison ({memory_data['n_params']:,} params)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    plt.close()
    return fig


def plot_weight_distribution_over_time(
    distributions: list[dict[str, float]],
    title: str = "Weight Distribution Over Training",
    save_path: str | None = None,
):
    """Plot how weight distribution changes during training."""
    if not distributions:
        return None

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    steps = range(len(distributions))
    neg1 = [d["neg1"] for d in distributions]
    zero = [d["zero"] for d in distributions]
    pos1 = [d["pos1"] for d in distributions]

    ax.stackplot(
        steps,
        neg1,
        zero,
        pos1,
        labels=["-1", "0", "+1"],
        colors=["#E53935", "#9E9E9E", "#43A047"],
        alpha=0.8,
    )

    ax.set_xlabel("Training Steps (x100)", fontsize=12)
    ax.set_ylabel("Weight Ratio", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


def plot_acceptance_rate(
    rates: list[float],
    title: str = "Flip Acceptance Rate",
    save_path: str | None = None,
):
    """Plot acceptance rate over time."""
    if not rates:
        return None

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(rates, color="#1976D2", linewidth=1.5)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="50%")
    ax.axhline(y=0.05, color="red", linestyle="--", alpha=0.5, label="5% (stuck)")
    ax.axhline(y=0.8, color="orange", linestyle="--", alpha=0.5, label="80% (too permissive)")

    ax.set_xlabel("Steps", fontsize=12)
    ax.set_ylabel("Acceptance Rate", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


def generate_summary_report(results: dict) -> str:
    """Generate a text summary of benchmark results."""
    lines = [
        "=" * 70,
        "TFLE BENCHMARK SUMMARY REPORT",
        "=" * 70,
        "",
    ]

    if "tfle" in results and "baseline" in results:
        tfle = results["tfle"]
        baseline = results["baseline"]
        lines.extend([
            "COMPARISON: TFLE vs STE Baseline",
            "-" * 40,
            f"  TFLE Final Accuracy:     {tfle.get('final_accuracy', 0):.4f}",
            f"  Baseline Final Accuracy: {baseline.get('final_accuracy', 0):.4f}",
            f"  TFLE Training Time:      {tfle.get('training_time', 0):.1f}s",
            f"  Baseline Training Time:  {baseline.get('training_time', 0):.1f}s",
            f"  TFLE Memory:             {tfle.get('memory_mb', 0):.2f} MB",
            f"  Baseline Memory:         {baseline.get('memory_mb', 0):.2f} MB",
            f"  Memory Savings:          "
            f"{baseline.get('memory_mb', 1) / max(tfle.get('memory_mb', 1), 0.01):.1f}x",
            "",
        ])

    for key, value in results.items():
        if key in ("tfle", "baseline"):
            continue
        if isinstance(value, dict):
            lines.append(f"\n{key.upper()}")
            lines.append("-" * 40)
            for k, v in value.items():
                if isinstance(v, float):
                    lines.append(f"  {k}: {v:.4f}")
                else:
                    lines.append(f"  {k}: {v}")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)
