"""Dual-GPU parallel run: Stage 1 on GPU 0, Stage 4 handoff on GPU 1.

Uses v2 trainer with anti-overfitting fixes:
- Fresh batches per step
- Elite re-evaluation on second batch
- Large K + large batch (5090 has 32 GB)
"""

from __future__ import annotations

import argparse
import os
import sys
import json
import multiprocessing as mp
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, ROOT)


def run_stage1(gpu_id: int):
    """Stage 1: From-scratch char LM with v2 trainer."""
    import torch
    import torch.nn.functional as F
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    sys.path.insert(0, ROOT)
    from tfle.config import TFLEConfig, FitnessType, CoolingSchedule, SelectionMethod
    from tfle_proving.data.loader import create_char_dataloaders
    from tfle_proving.models.char_lm import CharLM
    from tfle_proving.training.tfle_text_trainer_v2 import TFLETextTrainerV2
    from tfle_proving.training.utils import compute_perplexity, save_results

    device = torch.device("cuda:0")
    base = os.path.join(ROOT, "tfle_proving")

    train_loader, val_loader, _ = create_char_dataloaders(
        context_len=128, batch_size=1024, num_workers=4, stride=3
    )

    config = TFLEConfig(
        fitness_type=FitnessType.TASK_LOSS,
        flip_rate=0.008,
        selection_method=SelectionMethod.TRACE_WEIGHTED,
        protection_threshold=0.3,
        min_candidates_per_step=10,
        max_candidates_fraction=0.05,
        num_parallel_proposals=256,
        initial_temperature=1.5,
        min_temperature=0.1,
        cooling_schedule=CoolingSchedule.COSINE,
        reheat_on_plateau=True,
        plateau_window=3000,
        reheat_factor=2.5,
        trace_decay=0.95,
        separate_pos_neg_traces=True,
        total_training_steps=40000,
        eval_interval=500,
        cdll_alpha_start=0.05,
        exploration_rate=0.005,
        exploration_min=0.001,
        depth_scaled_flip_rate=True,
        flip_rate_depth_scale=0.85,
        depth_scaled_temperature=True,
        temperature_depth_scale=0.8,
    )

    model = CharLM(
        vocab_size=256, embed_dim=32, context_len=128,
        hidden_sizes=[512, 512, 256], config=config, device=device,
    )

    trainer = TFLETextTrainerV2(
        model=model, config=config,
        train_loader=train_loader, val_loader=val_loader,
        device=device, embed_lr=1e-3, reeval=True,
    )

    log = trainer.train(
        total_steps=40000, eval_every=500, checkpoint_every=5000,
        checkpoint_dir=os.path.join(base, "checkpoints", "stage1_5090"),
        results_dir=os.path.join(base, "results", "stage1_5090"),
    )

    # Final eval
    final = trainer.evaluate()
    print(f"\n[GPU {gpu_id}] Stage 1 FINAL: val_loss={final['loss']:.3f}, ppl={final['perplexity']:.1f}")
    save_results(
        {"final_loss": final["loss"], "final_ppl": final["perplexity"], "steps": 40000},
        os.path.join(base, "results", "stage1_5090", "stage1_final.json"),
    )


def run_stage4(gpu_id: int):
    """Stage 4: Handoff with v2 anti-overfitting + layer-wise mode."""
    import torch
    import torch.nn.functional as F
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    sys.path.insert(0, ROOT)
    from tfle.config import TFLEConfig, FitnessType, CoolingSchedule, SelectionMethod
    from tfle.baseline import ste_ternary
    from tfle_proving.data.loader import create_char_dataloaders
    from tfle_proving.models.char_lm import CharLM, STECharLM
    from tfle_proving.training.tfle_text_trainer_v2 import TFLETextTrainerV2
    from tfle_proving.training.ste_text_trainer import STETextTrainer
    from tfle_proving.training.utils import compute_perplexity, save_results

    device = torch.device("cuda:0")
    base = os.path.join(ROOT, "tfle_proving")
    results_dir = os.path.join(base, "results", "stage4_5090")
    os.makedirs(results_dir, exist_ok=True)

    train_loader, val_loader, _ = create_char_dataloaders(
        context_len=128, batch_size=1024, num_workers=4, stride=3
    )

    # Phase 1: STE pretrain (5K steps, lower LR for stability)
    print(f"\n[GPU {gpu_id}] Phase 1: STE Pretrain (5K steps)")
    ste = STECharLM(256, 32, 128, [512, 512, 256])
    ste_trainer = STETextTrainer(ste, train_loader, val_loader, device, lr=5e-4)
    ste_log = ste_trainer.train(5000, eval_every=500, results_dir=results_dir)

    ste_eval_loss = 0.0
    n = 0
    ste.eval()
    with torch.no_grad():
        for vx, vy in val_loader:
            vx, vy = vx.to(device), vy.to(device)
            ste_eval_loss += F.cross_entropy(ste(vx), vy, reduction="sum").item()
            n += vy.shape[0]
    ste_loss = ste_eval_loss / n
    print(f"[GPU {gpu_id}] STE baseline: loss={ste_loss:.3f}, ppl={compute_perplexity(ste_loss):.1f}")

    # Phase 2a: All-layer handoff WITH re-evaluation
    print(f"\n[GPU {gpu_id}] Phase 2a: All-layer handoff WITH re-eval")
    config_all = TFLEConfig(
        fitness_type=FitnessType.TASK_LOSS,
        flip_rate=0.001, num_parallel_proposals=16,
        initial_temperature=0.01, min_temperature=0.005,
        cooling_schedule=CoolingSchedule.COSINE,
        protection_threshold=0.8,
        selection_method=SelectionMethod.UNIFORM_RANDOM,
        separate_pos_neg_traces=True, trace_decay=0.99,
        total_training_steps=3000,
        min_candidates_per_step=3, max_candidates_fraction=0.005,
        exploration_rate=0.001,
        depth_scaled_flip_rate=True, flip_rate_depth_scale=0.7,
    )

    tfle_all = CharLM(256, 32, 128, [512, 512, 256], config_all, device)
    tfle_all.embedding.load_state_dict(ste.embedding.state_dict())
    for i, layer in enumerate(ste.ternary_layers):
        with torch.no_grad():
            w = ste_ternary(layer.weight).round().clamp(-1, 1).to(torch.int8).t().contiguous()
            tfle_all.layers[i].weights = w.to(device)

    trainer_all = TFLETextTrainerV2(
        tfle_all, config_all, train_loader, val_loader, device,
        embed_lr=1e-5, reeval=True,
    )
    log_all = trainer_all.train(
        total_steps=3000, eval_every=200,
        checkpoint_dir=os.path.join(base, "checkpoints", "s4_all_reeval"),
        results_dir=os.path.join(results_dir, "all_reeval"),
    )

    # Phase 2b: Output-layer-only handoff WITH re-evaluation
    print(f"\n[GPU {gpu_id}] Phase 2b: Output-layer-only handoff WITH re-eval")
    config_out = TFLEConfig(
        fitness_type=FitnessType.TASK_LOSS,
        flip_rate=0.002, num_parallel_proposals=32,
        initial_temperature=0.05, min_temperature=0.01,
        cooling_schedule=CoolingSchedule.COSINE,
        protection_threshold=0.6,
        selection_method=SelectionMethod.TRACE_WEIGHTED,
        separate_pos_neg_traces=True, trace_decay=0.98,
        total_training_steps=5000,
        min_candidates_per_step=5, max_candidates_fraction=0.01,
        exploration_rate=0.002,
        depth_scaled_flip_rate=False,
    )

    tfle_out = CharLM(256, 32, 128, [512, 512, 256], config_out, device)
    tfle_out.embedding.load_state_dict(ste.embedding.state_dict())
    for i, layer in enumerate(ste.ternary_layers):
        with torch.no_grad():
            w = ste_ternary(layer.weight).round().clamp(-1, 1).to(torch.int8).t().contiguous()
            tfle_out.layers[i].weights = w.to(device)

    last_layer = len(tfle_out.layers) - 1
    trainer_out = TFLETextTrainerV2(
        tfle_out, config_out, train_loader, val_loader, device,
        embed_lr=1e-5, reeval=True, active_layers=[last_layer],
    )
    log_out = trainer_out.train(
        total_steps=5000, eval_every=200,
        checkpoint_dir=os.path.join(base, "checkpoints", "s4_output_reeval"),
        results_dir=os.path.join(results_dir, "output_reeval"),
    )

    # Phase 2c: Layer-wise handoff (one layer at a time, bottom-up)
    print(f"\n[GPU {gpu_id}] Phase 2c: Layer-wise handoff (one at a time)")
    config_lw = TFLEConfig(
        fitness_type=FitnessType.TASK_LOSS,
        flip_rate=0.001, num_parallel_proposals=16,
        initial_temperature=0.02, min_temperature=0.005,
        cooling_schedule=CoolingSchedule.COSINE,
        protection_threshold=0.7,
        selection_method=SelectionMethod.TRACE_WEIGHTED,
        separate_pos_neg_traces=True, trace_decay=0.98,
        total_training_steps=1500,
        min_candidates_per_step=3, max_candidates_fraction=0.005,
        exploration_rate=0.001,
        depth_scaled_flip_rate=False,
    )

    tfle_lw = CharLM(256, 32, 128, [512, 512, 256], config_lw, device)
    tfle_lw.embedding.load_state_dict(ste.embedding.state_dict())
    for i, layer in enumerate(ste.ternary_layers):
        with torch.no_grad():
            w = ste_ternary(layer.weight).round().clamp(-1, 1).to(torch.int8).t().contiguous()
            tfle_lw.layers[i].weights = w.to(device)

    lw_results = []
    for layer_idx in range(len(tfle_lw.layers)):
        print(f"\n  Training layer {layer_idx}/{len(tfle_lw.layers)-1}...")
        config_lw.total_training_steps = 1500
        trainer_lw = TFLETextTrainerV2(
            tfle_lw, config_lw, train_loader, val_loader, device,
            embed_lr=1e-5, reeval=True, active_layers=[layer_idx],
        )
        log_lw = trainer_lw.train(
            total_steps=1500, eval_every=300,
            checkpoint_dir=os.path.join(base, "checkpoints", f"s4_lw_L{layer_idx}"),
            results_dir=os.path.join(results_dir, f"layerwise_L{layer_idx}"),
        )
        # Eval after this layer
        ev = trainer_lw.evaluate()
        deg = (ev["loss"] - ste_loss) / ste_loss * 100
        lw_results.append({
            "layer": layer_idx, "loss": ev["loss"],
            "ppl": ev["perplexity"], "degradation_pct": deg,
        })
        print(f"  Layer {layer_idx} done: loss={ev['loss']:.3f}, deg={deg:+.1f}%")

    # Summary
    print(f"\n{'='*60}")
    print(f"[GPU {gpu_id}] STAGE 4 SUMMARY (5090, with re-eval)")
    print(f"{'='*60}")
    print(f"STE baseline: loss={ste_loss:.3f}")

    all_final = trainer_all.evaluate()
    out_final = trainer_out.evaluate()
    lw_final = trainer_lw.evaluate() if lw_results else {"loss": 0}

    summary = {
        "ste_loss": ste_loss,
        "all_layer_reeval": {
            "final_loss": all_final["loss"],
            "degradation_pct": (all_final["loss"] - ste_loss) / ste_loss * 100,
        },
        "output_only_reeval": {
            "final_loss": out_final["loss"],
            "degradation_pct": (out_final["loss"] - ste_loss) / ste_loss * 100,
        },
        "layer_wise": {
            "final_loss": lw_final["loss"],
            "degradation_pct": (lw_final["loss"] - ste_loss) / ste_loss * 100,
            "per_layer": lw_results,
        },
    }
    for k, v in summary.items():
        print(f"  {k}: {v}")

    save_results(summary, os.path.join(results_dir, "stage4_5090_summary.json"))


def main():
    # Run both experiments in parallel on separate GPUs
    p1 = mp.Process(target=run_stage1, args=(0,))
    p2 = mp.Process(target=run_stage4, args=(1,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()

    print("\n\nBoth GPU experiments complete!")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
