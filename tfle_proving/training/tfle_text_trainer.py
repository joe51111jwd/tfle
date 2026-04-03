"""TFLE training loop for next-token text prediction.

Hybrid training: TFLE flips for ternary layers + Adam for float32 embedding.
Uses batched K-proposal evaluation with cached prefix/suffix for efficiency.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tfle.annealing import TemperatureScheduler
from tfle.config import TFLEConfig, CoolingSchedule, FitnessType
from tfle.layers import generate_k_proposals

from .utils import compute_perplexity, save_results, Timer


class TFLETextTrainer:
    """Trains a CharLM model using TFLE for ternary layers + Adam for embedding."""

    def __init__(
        self,
        model,
        config: TFLEConfig,
        train_loader,
        val_loader,
        device: torch.device,
        embed_lr: float = 1e-3,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Temperature scheduler
        self.scheduler = TemperatureScheduler(config)

        # Adam optimizer for embedding only
        self.embed_optimizer = torch.optim.Adam(
            model.embedding.parameters(), lr=embed_lr
        )

        self.log: list[dict] = []
        self.step = 0

    def train(
        self,
        total_steps: int = 20000,
        eval_every: int = 500,
        checkpoint_every: int = 2000,
        checkpoint_dir: str = "checkpoints",
        results_dir: str = "results",
        prompt: str = "ROMEO:\n",
    ) -> list[dict]:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        timer = Timer()
        data_iter = iter(self.train_loader)

        print(f"\n{'='*60}")
        print(f"TFLE Text Training — {total_steps} steps")
        print(f"Ternary params: {self.model.get_ternary_param_count():,}")
        print(f"Embedding params: {self.model.get_embed_param_count():,}")
        print(f"K={self.config.num_parallel_proposals}, "
              f"flip_rate={self.config.flip_rate}, "
              f"T_init={self.config.initial_temperature}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")

        while self.step < total_steps:
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                inputs, targets = next(data_iter)

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            metrics = self._train_step(inputs, targets)

            # Temperature update
            self.scheduler.step_update(-metrics["loss"])

            # Eval + logging
            if self.step % eval_every == 0:
                val_metrics = self.evaluate()
                entry = {
                    "step": self.step,
                    "train_loss": metrics["loss"],
                    "val_loss": val_metrics["loss"],
                    "train_perplexity": compute_perplexity(metrics["loss"]),
                    "val_perplexity": compute_perplexity(val_metrics["loss"]),
                    "acceptance_rate": metrics["acceptance_rate"],
                    "temperature": metrics["temperature"],
                    "accepted_layers": metrics["accepted_layers"],
                    "elapsed": timer.elapsed_str(),
                }

                # Generate sample at milestones
                if self.step % (eval_every * 4) == 0:
                    sample = self.model.generate_text(prompt, length=200)
                    entry["sample"] = sample

                self.log.append(entry)
                self._print_status(entry)

            # Checkpoint
            if self.step > 0 and self.step % checkpoint_every == 0:
                ckpt_path = os.path.join(
                    checkpoint_dir, f"tfle_char_step{self.step}.pt"
                )
                self.model.save_checkpoint(ckpt_path)
                save_results(self.log, os.path.join(results_dir, "tfle_log.json"))

            self.step += 1

        # Final save
        self.model.save_checkpoint(
            os.path.join(checkpoint_dir, "tfle_char_final.pt")
        )
        save_results(self.log, os.path.join(results_dir, "tfle_log.json"))
        return self.log

    def _train_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> dict:
        """One training step: TFLE flip proposals + embedding backprop."""
        temperature = self.scheduler.get_temperature()
        K = self.config.num_parallel_proposals

        # 1. Pre-compute embedding (no grad for TFLE evaluation)
        with torch.no_grad():
            flat_embedded = self.model.embed(inputs)

        # 2. TFLE: train each ternary layer sequentially with K-proposal search
        accepted_count = 0
        total_delta = 0.0
        for layer_idx in range(len(self.model.layers)):
            layer_metrics = self._tfle_layer_step(
                flat_embedded, targets, layer_idx, K, temperature
            )
            if layer_metrics["accepted"]:
                accepted_count += 1
            total_delta += layer_metrics["delta"]

        # 3. Embedding update via backprop (uses updated ternary weights)
        self.embed_optimizer.zero_grad()
        embedded_grad = self.model.embed(inputs)  # with grad tracking
        logits = self.model.ternary_forward(embedded_grad)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        self.embed_optimizer.step()

        n_layers = len(self.model.layers)
        return {
            "loss": loss.item(),
            "accepted_layers": accepted_count,
            "acceptance_rate": accepted_count / n_layers,
            "avg_delta": total_delta / n_layers,
            "temperature": temperature,
        }

    def _tfle_layer_step(
        self,
        flat_embedded: torch.Tensor,
        targets: torch.Tensor,
        layer_idx: int,
        K: int,
        temperature: float,
    ) -> dict:
        """K-proposal TFLE step for one ternary layer."""
        layer = self.model.layers[layer_idx]
        layer.step_count += 1

        # Decay cooldowns
        expired = [k for k, v in layer.cooldown_map.items() if v <= 0]
        for k in expired:
            del layer.cooldown_map[k]
        for k in layer.cooldown_map:
            layer.cooldown_map[k] -= 1

        # Select candidates via trace-weighted scoring
        combined_traces = layer._get_combined_traces()
        candidates = layer._select_candidates(combined_traces)

        # Generate K proposals + current weights as baseline
        proposals = generate_k_proposals(
            layer.weights, candidates, K, self.device
        )
        current_w = layer.weights.unsqueeze(0).to(self.device)
        all_proposals = torch.cat([current_w, proposals], dim=0)

        # Batched eval: all K+1 proposals at once
        losses = self._batched_eval(flat_embedded, targets, layer_idx, all_proposals)

        loss_before = losses[0].item()
        fitness_before = -loss_before

        best_k = losses[1:].argmin().item()
        fitness_after = -losses[best_k + 1].item()
        delta = fitness_after - fitness_before

        # Accept/reject with Boltzmann
        layer_temp = self.config.get_temperature_for_layer(temperature, layer.layer_idx)
        accepted = layer._accept_or_reject(delta, layer_temp)

        if accepted and delta != 0:
            layer.weights = all_proposals[best_k + 1].to(torch.int8)

        # Update traces
        with torch.no_grad():
            h = flat_embedded
            for i in range(layer_idx):
                h = self.model.layers[i].forward(h)
                h = F.relu(h)
                h = F.layer_norm(h, h.shape[-1:])
            output = layer.forward(h)
        layer._update_traces(h, output, delta <= 0)

        layer.fitness_history.append(fitness_after if accepted else fitness_before)
        layer.acceptance_history.append(accepted)

        return {
            "accepted": accepted,
            "fitness_before": fitness_before,
            "fitness_after": fitness_after,
            "delta": delta,
            "n_candidates": len(candidates),
            "temperature": layer_temp,
        }

    @torch.no_grad()
    def _batched_eval(
        self,
        flat_embedded: torch.Tensor,
        targets: torch.Tensor,
        layer_idx: int,
        proposals: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate proposals for one layer using cached prefix/suffix.

        Args:
            flat_embedded: (B, flat_dim) — embedding output
            targets: (B,) — target char indices
            layer_idx: which layer proposals are for
            proposals: (K+1, in_features, out_features) — weight proposals

        Returns:
            (K+1,) losses for each proposal
        """
        K_plus_1 = proposals.shape[0]

        # Prefix: forward through layers before layer_idx (using current weights)
        h = flat_embedded
        for i in range(layer_idx):
            h = self.model.layers[i].forward(h)
            if i < len(self.model.layers) - 1:
                h = F.relu(h)
                h = F.layer_norm(h, h.shape[-1:])

        # Batched varying layer: (K+1, B, in) @ (K+1, in, out) -> (K+1, B, out)
        h_expanded = h.unsqueeze(0).expand(K_plus_1, -1, -1)
        w_float = proposals.float()
        varied = torch.bmm(h_expanded, w_float)
        if layer_idx < len(self.model.layers) - 1:
            varied = F.relu(varied)
            varied = F.layer_norm(varied, varied.shape[-1:])

        # Suffix: forward through remaining layers
        K_val, B, F_out = varied.shape
        h_flat = varied.reshape(K_val * B, F_out)
        for i in range(layer_idx + 1, len(self.model.layers)):
            h_flat = self.model.layers[i].forward(h_flat)
            if i < len(self.model.layers) - 1:
                h_flat = F.relu(h_flat)
                h_flat = F.layer_norm(h_flat, h_flat.shape[-1:])

        # Compute CE loss per proposal
        logits = h_flat.reshape(K_val, B, -1)
        targets_exp = targets.unsqueeze(0).expand(K_val, -1)
        losses = F.cross_entropy(
            logits.reshape(K_val * B, -1),
            targets_exp.reshape(K_val * B),
            reduction="none",
        ).reshape(K_val, B).mean(dim=1)

        return losses

    @torch.no_grad()
    def evaluate(self) -> dict:
        """Evaluate on validation set."""
        total_loss = 0.0
        total_samples = 0
        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            logits = self.model.forward(inputs)
            loss = F.cross_entropy(logits, targets, reduction="sum")
            total_loss += loss.item()
            total_samples += targets.shape[0]
        avg_loss = total_loss / max(total_samples, 1)
        return {
            "loss": avg_loss,
            "perplexity": compute_perplexity(avg_loss),
            "samples": total_samples,
        }

    def _print_status(self, entry: dict):
        parts = [
            f"step={entry['step']:>6d}",
            f"train_loss={entry['train_loss']:.3f}",
            f"val_loss={entry['val_loss']:.3f}",
            f"val_ppl={entry['val_perplexity']:.1f}",
            f"accept={entry['acceptance_rate']:.2f}",
            f"T={entry['temperature']:.4f}",
            f"[{entry['elapsed']}]",
        ]
        print(" | ".join(parts))
        if entry.get("sample"):
            preview = entry["sample"][:120].replace("\n", "\\n")
            print(f"  Sample: {preview}...")
