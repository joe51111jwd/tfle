"""Dual-GPU orchestrator for v2 experiments.

GPU 0 — Track A: AttentionLM + phased co-evolution (the main event)
GPU 1 — Track B: Tuned STE baseline on same architecture + Track C long-horizon MLP
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, ROOT)


def track_a(gpu_id: int):
    """Track A: Attention model + phased co-evolution."""
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    sys.path.insert(0, ROOT)

    from tfle.config import TFLEConfig, FitnessType, CoolingSchedule, SelectionMethod
    from tfle_proving_v2.data.loader import create_seq_dataloaders
    from tfle_proving_v2.models.attention_lm import AttentionLM
    from tfle_proving_v2.tfle_v2.improved_trainer import PhasedTrainer

    device = torch.device("cuda:0")
    base = os.path.join(ROOT, "tfle_proving_v2")

    print(f"[GPU {gpu_id}] Track A: Attention + Co-Evolution")
    print(f"[GPU {gpu_id}] {torch.cuda.get_device_name(0)}")

    # Sequence-level data (predict next token at every position)
    train_loader, val_loader, vocab = create_seq_dataloaders(
        context_len=256, batch_size=64, num_workers=2, stride=128,
    )

    # TFLE config for phase 3 (full training)
    config = TFLEConfig(
        fitness_type=FitnessType.TASK_LOSS,
        flip_rate=0.005,
        selection_method=SelectionMethod.TRACE_WEIGHTED,
        protection_threshold=0.3,
        num_parallel_proposals=64,
        initial_temperature=1.0,
        min_temperature=0.05,
        cooling_schedule=CoolingSchedule.COSINE,
        reheat_on_plateau=True,
        plateau_window=3000,
        reheat_factor=2.5,
        trace_decay=0.95,
        separate_pos_neg_traces=True,
        total_training_steps=40000,
        min_candidates_per_step=5,
        max_candidates_fraction=0.03,
        exploration_rate=0.003,
        depth_scaled_flip_rate=True,
        flip_rate_depth_scale=0.85,
    )

    model = AttentionLM(
        vocab_size=256, d_model=128, n_heads=4, n_layers=2, d_ff=512,
        max_len=512, config=config, device=device,
    )
    print(f"[GPU {gpu_id}] Ternary: {model.get_ternary_param_count():,}, "
          f"Float: {model.get_float_param_count():,}")

    trainer = PhasedTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        phase1_steps=5000,
        phase2_steps=10000,
        embed_lr_init=1e-3,
        phase2_flip_rate=0.001,
        phase2_K=32,
        phase2_tolerance=0.05,
        phase3_tolerance_init=0.05,
        phase3_tolerance_final=0.01,
        use_antithetic=False,
        seq_model=True,
    )

    log = trainer.train(
        total_steps=40000,
        eval_every=500,
        checkpoint_every=5000,
        checkpoint_dir=os.path.join(base, "checkpoints", "track_a"),
        results_dir=os.path.join(base, "results", "track_a"),
    )

    # Verify ternary contribution
    print(f"\n[GPU {gpu_id}] Verifying ternary contribution...")
    val_learned = trainer.evaluate()

    saved_info = model.verify_ternary_contribution()
    saved_weights = saved_info["saved_weights"]
    val_random = trainer.evaluate()
    model.restore_weights(saved_weights)

    contrib = {
        "learned_loss": val_learned["loss"],
        "random_loss": val_random["loss"],
        "learned_ppl": val_learned["perplexity"],
        "random_ppl": val_random["perplexity"],
        "tfle_contributed": val_learned["loss"] < val_random["loss"],
        "contribution_pct": (val_random["loss"] - val_learned["loss"]) / val_random["loss"] * 100,
    }
    print(f"[GPU {gpu_id}] Learned ppl={contrib['learned_ppl']:.1f}, "
          f"Random ppl={contrib['random_ppl']:.1f}")
    print(f"[GPU {gpu_id}] TFLE contributed: {contrib['tfle_contributed']} "
          f"({contrib['contribution_pct']:+.1f}%)")

    with open(os.path.join(base, "results", "track_a", "contribution.json"), "w") as f:
        json.dump(contrib, f, indent=2)

    # Text quality metrics
    print(f"\n[GPU {gpu_id}] Text quality analysis...")
    sample = model.generate_text("ROMEO:\n", length=500)

    # Check for real English words
    try:
        words = sample.split()
        # Simple check: words that are all ASCII lowercase/uppercase letters
        english_like = sum(1 for w in words if w.isalpha() and len(w) > 1)
        word_rate = english_like / max(len(words), 1)
    except Exception:
        word_rate = 0.0

    quality = {
        "sample_500": sample,
        "word_count": len(sample.split()),
        "english_word_rate": word_rate,
        "char_entropy": _char_entropy(sample),
    }
    with open(os.path.join(base, "results", "track_a", "text_quality.json"), "w") as f:
        json.dump(quality, f, indent=2, default=str)

    print(f"[GPU {gpu_id}] Word recognition rate: {word_rate:.1%}")
    print(f"[GPU {gpu_id}] Char entropy: {quality['char_entropy']:.2f} bits "
          f"(English ~4.0-4.5, random ~8.0)")
    print(f"\n[GPU {gpu_id}] Track A COMPLETE")


def track_b(gpu_id: int):
    """Track B: Tuned STE on same attention architecture + long-horizon MLP."""
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    sys.path.insert(0, ROOT)

    from tfle_proving_v2.data.loader import create_seq_dataloaders, create_char_dataloaders
    from tfle_proving_v2.models.attention_lm import STEAttentionLM
    from tfle_proving_v2.baselines.tuned_ste import TunedSTETrainer

    device = torch.device("cuda:0")
    base = os.path.join(ROOT, "tfle_proving_v2")

    # === Part 1: Tuned STE on attention model ===
    print(f"\n[GPU {gpu_id}] Track B Part 1: Tuned STE Attention Baseline")

    train_loader, val_loader, _ = create_seq_dataloaders(
        context_len=256, batch_size=64, num_workers=2, stride=128,
    )

    ste_model = STEAttentionLM(
        vocab_size=256, d_model=128, n_heads=4, n_layers=2, d_ff=512, max_len=512,
    )
    n_params = sum(p.numel() for p in ste_model.parameters())
    print(f"[GPU {gpu_id}] STE params: {n_params:,}")

    ste_trainer = TunedSTETrainer(
        ste_model, train_loader, val_loader, device,
        lr=3e-4, warmup_steps=500, weight_decay=0.01, grad_clip=1.0,
        seq_model=True,
    )
    ste_log = ste_trainer.train(
        total_steps=40000, eval_every=500,
        results_dir=os.path.join(base, "results", "track_b_ste"),
    )

    # Text quality
    ste_model.eval()
    sample = ste_model.generate_text("ROMEO:\n", length=500)
    words = sample.split()
    english_like = sum(1 for w in words if w.isalpha() and len(w) > 1)
    word_rate = english_like / max(len(words), 1)

    ste_summary = {
        "final_loss": ste_log[-1]["val_loss"] if ste_log else None,
        "final_ppl": ste_log[-1]["val_ppl"] if ste_log else None,
        "best_loss": min(e["val_loss"] for e in ste_log) if ste_log else None,
        "word_rate": word_rate,
        "char_entropy": _char_entropy(sample),
        "sample": sample[:500],
    }
    with open(os.path.join(base, "results", "track_b_ste", "summary.json"), "w") as f:
        json.dump(ste_summary, f, indent=2, default=str)

    print(f"\n[GPU {gpu_id}] STE final: ppl={ste_summary['final_ppl']:.1f}, "
          f"word_rate={word_rate:.1%}")

    # === Part 2: Long-horizon MLP comparison (Track C) ===
    print(f"\n[GPU {gpu_id}] Track B Part 2: Long-Horizon MLP (100K steps)")

    from tfle.config import TFLEConfig, FitnessType, CoolingSchedule, SelectionMethod
    from tfle_proving.models.char_lm import CharLM, STECharLM
    from tfle_proving.training.ste_text_trainer import STETextTrainer
    from tfle_proving_v2.tfle_v2.improved_trainer import PhasedTrainer

    train_mlp, val_mlp, _ = create_char_dataloaders(
        context_len=128, batch_size=512, num_workers=2, stride=3,
    )

    # Tuned STE MLP — 100K steps
    ste_mlp = STECharLM(256, 32, 128, [512, 512, 256])
    ste_mlp_trainer = TunedSTETrainer(
        ste_mlp, train_mlp, val_mlp, device,
        lr=3e-4, warmup_steps=500, weight_decay=0.01, seq_model=False,
    )
    ste_mlp_log = ste_mlp_trainer.train(
        total_steps=100000, eval_every=1000,
        results_dir=os.path.join(base, "results", "track_c_ste_mlp"),
    )

    print(f"\n[GPU {gpu_id}] Track B COMPLETE")


def _char_entropy(text: str) -> float:
    """Compute character-level entropy in bits."""
    import math
    from collections import Counter
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    entropy = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def main():
    print("Launching v2 experiments on 2x RTX 5090")
    print("GPU 0: Track A (Attention + Co-Evolution)")
    print("GPU 1: Track B (Tuned STE + Long-Horizon)")

    p0 = mp.Process(target=track_a, args=(0,))
    p1 = mp.Process(target=track_b, args=(1,))

    p0.start()
    p1.start()

    p0.join()
    p1.join()

    print("\n\nAll v2 experiments complete!")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
