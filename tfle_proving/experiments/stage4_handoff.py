"""Stage 4: STE→TFLE Handoff on Text.

Proves TFLE can maintain and improve an STE-trained text model.
1. Train STE model to good performance (50K steps)
2. Extract ternary weights → load into TFLE model
3. Continue training with gentle TFLE params
4. Measure: does performance stay stable? Improve? Degrade?
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, ROOT)

from tfle.config import (
    TFLEConfig,
    CoolingSchedule,
    FitnessType,
    SelectionMethod,
    InitMethod,
)
from tfle.baseline import ste_ternary
from tfle_proving.data.loader import create_char_dataloaders
from tfle_proving.models.char_lm import CharLM, STECharLM
from tfle_proving.training.tfle_text_trainer import TFLETextTrainer
from tfle_proving.training.ste_text_trainer import STETextTrainer
from tfle_proving.training.utils import (
    setup_device,
    compute_perplexity,
    save_results,
    plot_loss_curves,
    Timer,
)


def make_handoff_config(total_steps: int = 10000) -> TFLEConfig:
    """Gentle TFLE config for handoff — barely change per step."""
    return TFLEConfig(
        fitness_type=FitnessType.TASK_LOSS,
        # Very conservative flipping
        flip_rate=0.003,
        selection_method=SelectionMethod.TRACE_WEIGHTED,
        protection_threshold=0.6,  # Protect more weights
        min_candidates_per_step=5,
        max_candidates_fraction=0.02,
        # More proposals but gentler
        num_parallel_proposals=128,
        # Cool temperature — model is already trained
        initial_temperature=0.5,
        min_temperature=0.05,
        cooling_schedule=CoolingSchedule.COSINE,
        cooling_rate=0.9997,
        reheat_on_plateau=True,
        plateau_window=2000,
        reheat_factor=2.0,
        # Traces
        trace_decay=0.95,
        separate_pos_neg_traces=True,
        # Training loop
        total_training_steps=total_steps,
        eval_interval=200,
        # Light regularizer
        cdll_alpha_start=0.02,
        # Minimal exploration
        exploration_rate=0.002,
        exploration_min=0.0005,
        # Depth scaling
        depth_scaled_flip_rate=True,
        flip_rate_depth_scale=0.85,
        depth_scaled_temperature=True,
        temperature_depth_scale=0.8,
    )


def extract_ternary_weights(ste_model: STECharLM) -> list[torch.Tensor]:
    """Extract quantized ternary weights from an STE model."""
    weights = []
    for layer in ste_model.ternary_layers:
        # Apply STE quantization to get the ternary values
        with torch.no_grad():
            w_ternary = ste_ternary(layer.weight)
            # Convert to int8 {-1, 0, +1}
            w_int8 = w_ternary.round().clamp(-1, 1).to(torch.int8)
            # STETernaryLinear stores weight as (out, in), TFLELayer as (in, out)
            weights.append(w_int8.t().contiguous())
    return weights


def handoff_ste_to_tfle(
    ste_model: STECharLM,
    config: TFLEConfig,
    device: torch.device,
    hidden_sizes: list[int],
) -> CharLM:
    """Create TFLE CharLM and load STE's trained weights + embedding."""
    tfle_model = CharLM(
        vocab_size=ste_model.vocab_size,
        embed_dim=ste_model.embed_dim,
        context_len=ste_model.context_len,
        hidden_sizes=hidden_sizes,
        config=config,
        device=device,
    )

    # Transfer embedding
    with torch.no_grad():
        tfle_model.embedding.load_state_dict(ste_model.embedding.state_dict())

    # Transfer ternary weights
    ternary_weights = extract_ternary_weights(ste_model)
    for i, w in enumerate(ternary_weights):
        tfle_model.layers[i].weights = w.to(device)

    return tfle_model


@torch.no_grad()
def evaluate_model(model, val_loader, device) -> dict:
    """Evaluate any model (CharLM or STECharLM) on val set."""
    total_loss = 0.0
    total_samples = 0
    for inputs, targets in val_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model.forward(inputs) if hasattr(model, 'ternary_forward') else model(inputs)
        loss = F.cross_entropy(logits, targets, reduction="sum")
        total_loss += loss.item()
        total_samples += targets.shape[0]
    avg_loss = total_loss / max(total_samples, 1)
    return {"loss": avg_loss, "perplexity": compute_perplexity(avg_loss)}


def main():
    parser = argparse.ArgumentParser(description="Stage 4: STE→TFLE Handoff")
    parser.add_argument("--ste-steps", type=int, default=50000)
    parser.add_argument("--tfle-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--hidden", type=str, default="512,512,256")
    parser.add_argument("--ste-lr", type=float, default=1e-3)
    parser.add_argument("--K", type=int, default=128)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    args = parser.parse_args()

    hidden_sizes = [int(x) for x in args.hidden.split(",")]
    base_dir = str(Path(__file__).resolve().parents[1])
    results_dir = args.results_dir or os.path.join(base_dir, "results")
    checkpoint_dir = args.checkpoint_dir or os.path.join(base_dir, "checkpoints")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("Detecting device...")
    device = setup_device()

    print("\nLoading data...")
    train_loader, val_loader, _ = create_char_dataloaders(
        context_len=args.context_len,
        batch_size=args.batch_size,
        num_workers=2,
        stride=3,
    )

    # =========================================================
    # PHASE 1: Train STE model to good performance
    # =========================================================
    print(f"\n{'='*60}")
    print(f"PHASE 1: STE Pretraining ({args.ste_steps} steps)")
    print(f"{'='*60}")

    ste_model = STECharLM(
        vocab_size=256,
        embed_dim=args.embed_dim,
        context_len=args.context_len,
        hidden_sizes=hidden_sizes,
    )
    ste_trainer = STETextTrainer(
        ste_model, train_loader, val_loader, device, lr=args.ste_lr
    )
    ste_log = ste_trainer.train(
        total_steps=args.ste_steps,
        eval_every=500,
        results_dir=results_dir,
    )

    # Record STE performance at handoff point
    ste_eval = evaluate_model(ste_model, val_loader, device)
    print(f"\nSTE at handoff: loss={ste_eval['loss']:.3f}, ppl={ste_eval['perplexity']:.1f}")

    # Find best STE performance from the log
    best_ste = min(ste_log, key=lambda x: x["val_loss"])
    print(f"Best STE during training: loss={best_ste['val_loss']:.3f}, "
          f"ppl={best_ste['val_perplexity']:.1f} at step {best_ste['step']}")

    # Save STE checkpoint
    ste_ckpt = os.path.join(checkpoint_dir, "ste_pretrained.pt")
    torch.save(ste_model.state_dict(), ste_ckpt)

    # =========================================================
    # PHASE 2: Handoff — transfer weights to TFLE model
    # =========================================================
    print(f"\n{'='*60}")
    print(f"PHASE 2: STE→TFLE Handoff")
    print(f"{'='*60}")

    handoff_config = make_handoff_config(total_steps=args.tfle_steps)
    tfle_model = handoff_ste_to_tfle(ste_model, handoff_config, device, hidden_sizes)

    # Evaluate immediately after handoff (should match STE)
    post_handoff_eval = evaluate_model(tfle_model, val_loader, device)
    print(f"Immediately after handoff: loss={post_handoff_eval['loss']:.3f}, "
          f"ppl={post_handoff_eval['perplexity']:.1f}")

    degradation_pct = (
        (post_handoff_eval["loss"] - ste_eval["loss"]) / ste_eval["loss"] * 100
    )
    print(f"Handoff degradation: {degradation_pct:+.1f}%")

    # =========================================================
    # PHASE 3: TFLE fine-tuning
    # =========================================================
    print(f"\n{'='*60}")
    print(f"PHASE 3: TFLE Fine-tuning ({args.tfle_steps} steps)")
    print(f"{'='*60}")

    tfle_trainer = TFLETextTrainer(
        model=tfle_model,
        config=handoff_config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        embed_lr=1e-4,  # Lower LR for fine-tuning
    )
    tfle_log = tfle_trainer.train(
        total_steps=args.tfle_steps,
        eval_every=200,
        checkpoint_every=2000,
        checkpoint_dir=checkpoint_dir,
        results_dir=results_dir,
    )

    # =========================================================
    # RESULTS
    # =========================================================
    print(f"\n{'='*60}")
    print("STAGE 4 HANDOFF RESULTS")
    print(f"{'='*60}")

    tfle_final = evaluate_model(tfle_model, val_loader, device)
    best_ste_loss = best_ste["val_loss"]
    best_ste_ppl = best_ste["val_perplexity"]

    # Metrics at key checkpoints
    tfle_1k = next((e for e in tfle_log if e["step"] >= 1000), tfle_log[-1])
    tfle_5k = next((e for e in tfle_log if e["step"] >= 5000), tfle_log[-1])

    # Degradation after 1K TFLE steps
    deg_1k = (tfle_1k["val_loss"] - ste_eval["loss"]) / ste_eval["loss"] * 100
    deg_final = (tfle_final["loss"] - ste_eval["loss"]) / ste_eval["loss"] * 100

    # Determine status
    if deg_1k > 10:
        status_1k = "FAIL (>10% degradation)"
    elif deg_1k > 5:
        status_1k = "MARGINAL (5-10% degradation)"
    else:
        status_1k = "PASS (<5% degradation)"

    if tfle_final["loss"] > ste_eval["loss"] * 1.1:
        status_final = "FAIL (still degraded)"
    elif tfle_final["loss"] <= ste_eval["loss"]:
        status_final = "GOOD (exceeded STE level)"
    else:
        status_final = "PASS (recovered)"

    # Stability check
    if tfle_log:
        losses = [e["val_loss"] for e in tfle_log[-10:]]
        loss_std = torch.tensor(losses).std().item() if len(losses) > 1 else 0
        if loss_std > 1.0:
            stability = "FAIL (oscillating)"
        elif loss_std > 0.3:
            stability = "PASS (stable)"
        else:
            stability = "GOOD (very stable)"
    else:
        stability = "N/A"

    summary = {
        "ste_pretrain_steps": args.ste_steps,
        "ste_best_loss": best_ste_loss,
        "ste_best_ppl": best_ste_ppl,
        "ste_handoff_loss": ste_eval["loss"],
        "ste_handoff_ppl": ste_eval["perplexity"],
        "post_handoff_loss": post_handoff_eval["loss"],
        "post_handoff_ppl": post_handoff_eval["perplexity"],
        "handoff_degradation_pct": degradation_pct,
        "tfle_1k_loss": tfle_1k["val_loss"],
        "tfle_1k_degradation_pct": deg_1k,
        "tfle_1k_status": status_1k,
        "tfle_final_loss": tfle_final["loss"],
        "tfle_final_ppl": tfle_final["perplexity"],
        "tfle_final_degradation_pct": deg_final,
        "tfle_final_status": status_final,
        "stability": stability,
    }

    for k, v in summary.items():
        print(f"  {k}: {v}")

    save_results(summary, os.path.join(results_dir, "stage4_summary.json"))

    # Plot handoff loss curve
    # Combine: STE training log + handoff point + TFLE fine-tuning log
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(12, 5))

        # STE phase
        ste_steps = [e["step"] for e in ste_log]
        ste_losses = [e["val_loss"] for e in ste_log]
        ax.plot(ste_steps, ste_losses, label="STE Pretrain", color="orange")

        # TFLE phase (offset steps)
        offset = args.ste_steps
        tfle_steps = [offset + e["step"] for e in tfle_log]
        tfle_losses = [e["val_loss"] for e in tfle_log]
        ax.plot(tfle_steps, tfle_losses, label="TFLE Fine-tune", color="blue")

        # Handoff marker
        ax.axvline(x=offset, color="red", linestyle="--", alpha=0.7, label="Handoff")

        ax.set_xlabel("Step")
        ax.set_ylabel("Validation Loss")
        ax.set_title("Stage 4: STE→TFLE Handoff")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "stage4_handoff_curve.png"), dpi=150)
        plt.close()
        print(f"Plot saved.")
    except ImportError:
        pass

    print("\nDone.")


if __name__ == "__main__":
    main()
