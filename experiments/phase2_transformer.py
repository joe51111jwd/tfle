"""Phase 2: Small transformer experiment with TFLE training."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from tfle.annealing import TemperatureScheduler
from tfle.config import TFLEConfig
from tfle.transformer import TFLETransformerModel


def generate_simple_sequences(
    n_sequences: int,
    seq_len: int,
    vocab_size: int,
) -> torch.Tensor:
    """Generate simple pattern sequences for testing.

    Patterns: repeating sequences, counting sequences, etc.
    """
    sequences = []
    for _ in range(n_sequences):
        pattern_type = torch.randint(0, 3, (1,)).item()
        if pattern_type == 0:
            # Repeating pattern: [a, b, c, a, b, c, ...]
            pattern_len = torch.randint(2, 5, (1,)).item()
            pattern = torch.randint(0, vocab_size, (pattern_len,))
            seq = pattern.repeat(seq_len // pattern_len + 1)[:seq_len]
        elif pattern_type == 1:
            # Counting: [0, 1, 2, 3, ...] mod vocab_size
            start = torch.randint(0, vocab_size, (1,)).item()
            seq = torch.arange(start, start + seq_len) % vocab_size
        else:
            # Random but with local repetition
            seq = torch.randint(0, vocab_size, (seq_len,))
            # Copy some segments
            for _ in range(seq_len // 4):
                src = torch.randint(0, seq_len - 3, (1,)).item()
                dst = torch.randint(0, seq_len - 3, (1,)).item()
                length = min(3, seq_len - max(src, dst))
                seq[dst : dst + length] = seq[src : src + length].clone()

        sequences.append(seq)

    return torch.stack(sequences)


def run_transformer_experiment(
    vocab_size: int = 256,
    embed_dim: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    ff_dim: int = 256,
    max_seq_len: int = 64,
    total_steps: int = 5000,
    eval_interval: int = 500,
    batch_size: int = 16,
    verbose: bool = True,
) -> dict:
    config = TFLEConfig(
        total_training_steps=total_steps,
        eval_interval=eval_interval,
        initial_temperature=15.0,
        cooling_rate=0.9998,
        flip_rate=0.02,
        trace_decay=0.95,
    )

    model = TFLETransformerModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        ff_dim=ff_dim,
        max_seq_len=max_seq_len,
        config=config,
    )

    scheduler = TemperatureScheduler(config)

    # Generate training and validation data
    train_data = generate_simple_sequences(500, max_seq_len, vocab_size)
    val_data = generate_simple_sequences(100, max_seq_len, vocab_size)

    if verbose:
        print("=" * 60)
        print("TFLE Transformer Experiment")
        print(f"Vocab: {vocab_size}, Embed: {embed_dim}, Heads: {n_heads}, Layers: {n_layers}")
        print(f"Total ternary params: {model.get_total_params():,}")
        print(f"Steps: {total_steps}")
        print("=" * 60)

    val_metrics_log = []
    start_time = time.time()
    pbar = tqdm(total=total_steps, disable=not verbose, desc="Transformer TFLE")

    for step in range(total_steps):
        # Sample batch
        indices = torch.randint(0, len(train_data), (batch_size,))
        batch = train_data[indices]

        temperature = scheduler.get_temperature()
        layer_metrics = model.train_step(batch, temperature)

        avg_fitness = sum(
            m["fitness_after"] if m["accepted"] else m["fitness_before"]
            for m in layer_metrics
        ) / max(len(layer_metrics), 1)
        scheduler.step_update(avg_fitness)

        if step % eval_interval == 0:
            eval_result = model.evaluate(val_data[:batch_size], val_data[:batch_size])
            val_metrics_log.append({"step": step, **eval_result})
            if verbose:
                pbar.set_postfix({
                    "loss": f"{eval_result['loss']:.3f}",
                    "acc": f"{eval_result['accuracy']:.4f}",
                    "temp": f"{temperature:.3f}",
                })

        pbar.update(1)

    pbar.close()
    training_time = time.time() - start_time

    return {
        "method": "TFLE Transformer",
        "total_params": model.get_total_params(),
        "total_steps": total_steps,
        "training_time": training_time,
        "val_metrics": val_metrics_log,
        "final_loss": val_metrics_log[-1]["loss"] if val_metrics_log else float("inf"),
        "final_accuracy": val_metrics_log[-1]["accuracy"] if val_metrics_log else 0.0,
    }


if __name__ == "__main__":
    results = run_transformer_experiment(total_steps=1000, eval_interval=200, verbose=True)
    print(f"\nFinal loss: {results['final_loss']:.4f}")
    print(f"Final accuracy: {results['final_accuracy']:.4f}")

    output_path = Path(__file__).parent.parent / "results_phase2_transformer.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")
