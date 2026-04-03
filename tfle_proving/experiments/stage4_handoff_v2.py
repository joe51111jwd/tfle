"""Stage 4 v2: STE→TFLE Handoff — fixed with ultra-conservative params.

Key fixes:
- STE pretrain only 5K steps (where it peaks, before overfitting)
- Much lower K=16 (prevents batch-overfitting with too many proposals)
- T_init=0.1 (barely accept bad flips)
- flip_rate=0.001 (touch very few weights)
- Larger eval batch for fitness (evaluate on 2 batches, not 1)
"""

from __future__ import annotations

import argparse
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
    Timer,
)


def make_ultra_gentle_config(total_steps: int = 10000) -> TFLEConfig:
    """Ultra-conservative TFLE config — barely change anything per step."""
    return TFLEConfig(
        fitness_type=FitnessType.TASK_LOSS,
        flip_rate=0.001,              # Touch 0.1% of weights
        selection_method=SelectionMethod.TRACE_WEIGHTED,
        protection_threshold=0.8,     # Protect top 80% of successful weights
        min_candidates_per_step=3,
        max_candidates_fraction=0.005,
        num_parallel_proposals=16,    # Low K — prevents batch overfitting
        initial_temperature=0.1,      # Very cold — almost greedy
        min_temperature=0.02,
        cooling_schedule=CoolingSchedule.COSINE,
        reheat_on_plateau=True,
        plateau_window=2000,
        reheat_factor=1.5,
        trace_decay=0.98,             # Slower trace decay (more history)
        separate_pos_neg_traces=True,
        total_training_steps=total_steps,
        eval_interval=200,
        cdll_alpha_start=0.0,         # No regularizer
        exploration_rate=0.001,
        exploration_min=0.0005,
        depth_scaled_flip_rate=True,
        flip_rate_depth_scale=0.7,    # Much gentler on deep layers
        depth_scaled_temperature=True,
        temperature_depth_scale=0.7,
    )


def extract_ternary_weights(ste_model):
    weights = []
    for layer in ste_model.ternary_layers:
        with torch.no_grad():
            w_ternary = ste_ternary(layer.weight)
            w_int8 = w_ternary.round().clamp(-1, 1).to(torch.int8)
            weights.append(w_int8.t().contiguous())
    return weights


def handoff(ste_model, config, device, hidden_sizes):
    tfle_model = CharLM(
        vocab_size=ste_model.vocab_size,
        embed_dim=ste_model.embed_dim,
        context_len=ste_model.context_len,
        hidden_sizes=hidden_sizes,
        config=config,
        device=device,
    )
    with torch.no_grad():
        tfle_model.embedding.load_state_dict(ste_model.embedding.state_dict())
    ternary_weights = extract_ternary_weights(ste_model)
    for i, w in enumerate(ternary_weights):
        tfle_model.layers[i].weights = w.to(device)
    return tfle_model


@torch.no_grad()
def eval_model(model, val_loader, device):
    total_loss = 0.0
    n = 0
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model.forward(inputs) if isinstance(model, CharLM) else model(inputs)
        total_loss += F.cross_entropy(logits, targets, reduction="sum").item()
        n += targets.shape[0]
    loss = total_loss / max(n, 1)
    return {"loss": loss, "perplexity": compute_perplexity(loss)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ste-steps", type=int, default=5000)
    parser.add_argument("--tfle-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--hidden", type=str, default="512,512,256")
    parser.add_argument("--ste-lr", type=float, default=5e-4)
    args = parser.parse_args()

    hidden_sizes = [int(x) for x in args.hidden.split(",")]
    base_dir = str(Path(__file__).resolve().parents[1])
    results_dir = os.path.join(base_dir, "results", "handoff_v2")
    checkpoint_dir = os.path.join(base_dir, "checkpoints", "handoff_v2")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = setup_device()

    train_loader, val_loader, _ = create_char_dataloaders(
        context_len=args.context_len, batch_size=args.batch_size,
        num_workers=2, stride=3,
    )

    # ===== STE Pretrain (short — stop before overfitting) =====
    print(f"\n{'='*60}")
    print(f"STE Pretrain ({args.ste_steps} steps, lr={args.ste_lr})")
    print(f"{'='*60}")

    ste_model = STECharLM(256, args.embed_dim, args.context_len, hidden_sizes)
    ste_trainer = STETextTrainer(
        ste_model, train_loader, val_loader, device, lr=args.ste_lr
    )
    ste_log = ste_trainer.train(args.ste_steps, eval_every=200, results_dir=results_dir)

    best_ste = min(ste_log, key=lambda x: x["val_loss"])
    ste_eval = eval_model(ste_model, val_loader, device)
    print(f"\nSTE final: loss={ste_eval['loss']:.3f}, ppl={ste_eval['perplexity']:.1f}")
    print(f"STE best: loss={best_ste['val_loss']:.3f}, ppl={best_ste['val_perplexity']:.1f} "
          f"(step {best_ste['step']})")

    # ===== Handoff =====
    print(f"\n{'='*60}")
    print("Handoff: STE → TFLE")
    print(f"{'='*60}")

    config = make_ultra_gentle_config(args.tfle_steps)
    tfle_model = handoff(ste_model, config, device, hidden_sizes)

    post_eval = eval_model(tfle_model, val_loader, device)
    handoff_deg = (post_eval["loss"] - ste_eval["loss"]) / ste_eval["loss"] * 100
    print(f"Post-handoff: loss={post_eval['loss']:.3f}, ppl={post_eval['perplexity']:.1f}")
    print(f"Degradation at handoff: {handoff_deg:+.1f}%")

    # ===== TFLE Fine-tuning =====
    print(f"\n{'='*60}")
    print(f"TFLE Fine-tuning ({args.tfle_steps} steps)")
    print(f"K=16, flip_rate=0.001, T_init=0.1, protection=0.8")
    print(f"{'='*60}")

    tfle_trainer = TFLETextTrainer(
        tfle_model, config, train_loader, val_loader, device, embed_lr=5e-5
    )
    tfle_log = tfle_trainer.train(
        total_steps=args.tfle_steps, eval_every=200,
        checkpoint_every=2000, checkpoint_dir=checkpoint_dir,
        results_dir=results_dir,
    )

    # ===== Results =====
    tfle_final = eval_model(tfle_model, val_loader, device)
    tfle_1k = next((e for e in tfle_log if e["step"] >= 1000), tfle_log[-1])
    tfle_5k = next((e for e in tfle_log if e["step"] >= 5000), tfle_log[-1])

    deg_1k = (tfle_1k["val_loss"] - ste_eval["loss"]) / ste_eval["loss"] * 100
    deg_final = (tfle_final["loss"] - ste_eval["loss"]) / ste_eval["loss"] * 100

    if abs(deg_1k) <= 5:
        s1k = "PASS (<5% degradation)"
    elif deg_1k <= 10:
        s1k = "MARGINAL (5-10%)"
    else:
        s1k = "FAIL (>10%)"

    if tfle_final["loss"] < ste_eval["loss"]:
        sfinal = "GOOD (exceeded STE)"
    elif tfle_final["loss"] <= ste_eval["loss"] * 1.05:
        sfinal = "PASS (recovered)"
    else:
        sfinal = "FAIL (still degraded)"

    losses = [e["val_loss"] for e in tfle_log[-10:]]
    std = torch.tensor(losses).std().item() if len(losses) > 1 else 0
    stab = "GOOD" if std < 0.3 else ("PASS" if std < 1.0 else "FAIL")

    summary = {
        "ste_pretrain_steps": args.ste_steps,
        "ste_best_loss": best_ste["val_loss"],
        "ste_best_ppl": best_ste["val_perplexity"],
        "ste_handoff_loss": ste_eval["loss"],
        "handoff_degradation_pct": handoff_deg,
        "tfle_1k_loss": tfle_1k["val_loss"],
        "tfle_1k_degradation_pct": deg_1k,
        "tfle_1k_status": s1k,
        "tfle_5k_loss": tfle_5k["val_loss"],
        "tfle_final_loss": tfle_final["loss"],
        "tfle_final_ppl": tfle_final["perplexity"],
        "tfle_final_degradation_pct": deg_final,
        "tfle_final_status": sfinal,
        "stability": stab,
    }

    print(f"\n{'='*60}")
    print("STAGE 4 v2 RESULTS")
    print(f"{'='*60}")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    save_results(summary, os.path.join(results_dir, "stage4_v2_summary.json"))

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot([e["step"] for e in ste_log],
                [e["val_loss"] for e in ste_log], label="STE", color="orange")
        offset = args.ste_steps
        ax.plot([offset + e["step"] for e in tfle_log],
                [e["val_loss"] for e in tfle_log], label="TFLE", color="blue")
        ax.axvline(x=offset, color="red", ls="--", alpha=0.7, label="Handoff")
        ax.set_xlabel("Step")
        ax.set_ylabel("Val Loss")
        ax.set_title("Stage 4 v2: STE→TFLE Handoff (Ultra-Gentle)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "stage4_v2_curve.png"), dpi=150)
        plt.close()
    except ImportError:
        pass

    print("\nDone.")


if __name__ == "__main__":
    main()
