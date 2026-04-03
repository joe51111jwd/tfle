"""Stage 1: Character-level text prediction — TFLE vs STE.

Proves TFLE can learn next-token prediction on text.
Runs both TFLE and STE on TinyShakespeare, compares results.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

# Setup imports
ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, ROOT)

from tfle.config import (
    TFLEConfig,
    CoolingSchedule,
    FitnessType,
    SelectionMethod,
    InitMethod,
)
from tfle_proving.data.loader import create_char_dataloaders
from tfle_proving.models.char_lm import CharLM, STECharLM
from tfle_proving.training.tfle_text_trainer import TFLETextTrainer
from tfle_proving.training.ste_text_trainer import STETextTrainer
from tfle_proving.training.utils import (
    setup_device,
    compute_perplexity,
    save_results,
    plot_loss_curves,
)


def make_tfle_config(
    total_steps: int = 20000,
    K: int = 64,
    flip_rate: float = 0.01,
) -> TFLEConfig:
    """TFLE config tuned for character-level text."""
    return TFLEConfig(
        # Fitness
        fitness_type=FitnessType.TASK_LOSS,
        # Candidate selection
        flip_rate=flip_rate,
        selection_method=SelectionMethod.TRACE_WEIGHTED,
        protection_threshold=0.3,
        min_candidates_per_step=10,
        max_candidates_fraction=0.05,
        # Proposals
        num_parallel_proposals=K,
        # Temperature / annealing
        initial_temperature=1.5,
        min_temperature=0.1,
        cooling_schedule=CoolingSchedule.COSINE,
        cooling_rate=0.9997,
        reheat_on_plateau=True,
        plateau_window=3000,
        reheat_factor=2.5,
        # Traces
        trace_decay=0.95,
        separate_pos_neg_traces=True,
        # Training loop
        total_training_steps=total_steps,
        eval_interval=500,
        # Init
        init_method=InitMethod.BALANCED_RANDOM,
        init_zero_bias=0.5,
        # CDLL (light regularizer via cdll_alpha_start, not used as primary)
        cdll_alpha_start=0.05,
        # Exploration
        exploration_rate=0.005,
        exploration_min=0.001,
        # Depth scaling
        depth_scaled_flip_rate=True,
        flip_rate_depth_scale=0.85,
        depth_scaled_temperature=True,
        temperature_depth_scale=0.8,
    )


def run_tfle(
    config: TFLEConfig,
    train_loader,
    val_loader,
    device: torch.device,
    total_steps: int,
    eval_every: int,
    results_dir: str,
    checkpoint_dir: str,
    context_len: int = 128,
    embed_dim: int = 32,
    hidden_sizes: list[int] | None = None,
) -> list[dict]:
    """Run TFLE training and return log."""
    if hidden_sizes is None:
        hidden_sizes = [512, 512, 256]

    model = CharLM(
        vocab_size=256,
        embed_dim=embed_dim,
        context_len=context_len,
        hidden_sizes=hidden_sizes,
        config=config,
        device=device,
    )

    trainer = TFLETextTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        embed_lr=1e-3,
    )

    return trainer.train(
        total_steps=total_steps,
        eval_every=eval_every,
        checkpoint_every=2000,
        checkpoint_dir=checkpoint_dir,
        results_dir=results_dir,
    )


def run_ste(
    train_loader,
    val_loader,
    device: torch.device,
    total_steps: int,
    eval_every: int,
    results_dir: str,
    context_len: int = 128,
    embed_dim: int = 32,
    hidden_sizes: list[int] | None = None,
    lr: float = 1e-3,
) -> list[dict]:
    """Run STE baseline and return log."""
    if hidden_sizes is None:
        hidden_sizes = [512, 512, 256]

    model = STECharLM(
        vocab_size=256,
        embed_dim=embed_dim,
        context_len=context_len,
        hidden_sizes=hidden_sizes,
    )

    trainer = STETextTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=lr,
    )

    return trainer.train(
        total_steps=total_steps,
        eval_every=eval_every,
        results_dir=results_dir,
    )


def compare_results(tfle_log: list[dict], ste_log: list[dict]) -> dict:
    """Compare TFLE vs STE final results."""
    tfle_final = tfle_log[-1] if tfle_log else {}
    ste_final = ste_log[-1] if ste_log else {}

    tfle_ppl = tfle_final.get("val_perplexity", float("inf"))
    ste_ppl = ste_final.get("val_perplexity", float("inf"))
    gap = tfle_ppl / ste_ppl if ste_ppl > 0 else float("inf")

    # Check if loss decreased
    tfle_loss_start = tfle_log[0]["val_loss"] if tfle_log else float("inf")
    tfle_loss_end = tfle_final.get("val_loss", float("inf"))
    loss_decreased = tfle_loss_end < tfle_loss_start * 0.95

    # Determine pass/fail
    if not loss_decreased:
        status = "FAIL"
    elif gap > 5:
        status = "PASS (marginal)"
    elif gap > 2:
        status = "PASS"
    else:
        status = "GOOD"

    summary = {
        "status": status,
        "tfle_final_loss": tfle_final.get("val_loss"),
        "tfle_final_ppl": tfle_ppl,
        "ste_final_loss": ste_final.get("val_loss"),
        "ste_final_ppl": ste_ppl,
        "perplexity_gap": f"{gap:.1f}x",
        "loss_decreased": loss_decreased,
        "tfle_sample": tfle_final.get("sample", "N/A"),
        "ste_sample": ste_final.get("sample", "N/A"),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Char-level LM")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--hidden", type=str, default="512,512,256")
    parser.add_argument("--K", type=int, default=64, help="Proposals per step")
    parser.add_argument("--flip-rate", type=float, default=0.01)
    parser.add_argument("--skip-ste", action="store_true")
    parser.add_argument("--skip-tfle", action="store_true")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    args = parser.parse_args()

    hidden_sizes = [int(x) for x in args.hidden.split(",")]
    base_dir = str(Path(__file__).resolve().parents[1])
    results_dir = args.results_dir or os.path.join(base_dir, "results")
    checkpoint_dir = args.checkpoint_dir or os.path.join(base_dir, "checkpoints")

    # Device
    print("Detecting device...")
    device = setup_device()

    # Data
    print("\nLoading data...")
    train_loader, val_loader, vocab_size = create_char_dataloaders(
        context_len=args.context_len,
        batch_size=args.batch_size,
        num_workers=2,
        stride=3,
    )

    # TFLE
    tfle_log = []
    if not args.skip_tfle:
        config = make_tfle_config(
            total_steps=args.steps, K=args.K, flip_rate=args.flip_rate
        )
        tfle_log = run_tfle(
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            total_steps=args.steps,
            eval_every=args.eval_every,
            results_dir=results_dir,
            checkpoint_dir=checkpoint_dir,
            context_len=args.context_len,
            embed_dim=args.embed_dim,
            hidden_sizes=hidden_sizes,
        )

    # STE baseline
    ste_log = []
    if not args.skip_ste:
        ste_log = run_ste(
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            total_steps=args.steps,
            eval_every=args.eval_every,
            results_dir=results_dir,
            context_len=args.context_len,
            embed_dim=args.embed_dim,
            hidden_sizes=hidden_sizes,
        )

    # Compare
    if tfle_log and ste_log:
        summary = compare_results(tfle_log, ste_log)
        print(f"\n{'='*60}")
        print("STAGE 1 RESULTS")
        print(f"{'='*60}")
        for k, v in summary.items():
            print(f"  {k}: {v}")

        save_results(summary, os.path.join(results_dir, "stage1_summary.json"))
        plot_loss_curves(
            tfle_log, ste_log,
            os.path.join(results_dir, "stage1_loss_curves.png"),
        )
    elif tfle_log:
        print("\n[TFLE only — no STE baseline for comparison]")
        save_results(tfle_log, os.path.join(results_dir, "tfle_log.json"))
    elif ste_log:
        print("\n[STE only — no TFLE for comparison]")

    print("\nDone.")


if __name__ == "__main__":
    main()
