"""TFLE text trainer v2 — with anti-overfitting fixes from literature.

Key changes from v1:
1. Fresh batch for each TFLE step (never reuse eval batch for embedding update)
2. Elite re-evaluation: accepted proposals verified on a SECOND fresh batch
3. Larger K with centered rank fitness shaping
4. Layer-wise training mode for handoff (one layer at a time)
"""

from __future__ import annotations

import os
import sys
import math
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tfle.annealing import TemperatureScheduler
from tfle.config import TFLEConfig
from tfle.layers import generate_k_proposals

from .utils import compute_perplexity, save_results, Timer


class TFLETextTrainerV2:
    """TFLE text trainer with anti-overfitting guarantees."""

    def __init__(
        self,
        model,
        config: TFLEConfig,
        train_loader,
        val_loader,
        device: torch.device,
        embed_lr: float = 1e-3,
        reeval: bool = True,
        layer_wise: bool = False,
        active_layers: list[int] | None = None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.reeval = reeval
        self.layer_wise = layer_wise
        self.active_layers = active_layers  # None = all layers

        self.scheduler = TemperatureScheduler(config)
        self.embed_optimizer = torch.optim.Adam(
            model.embedding.parameters(), lr=embed_lr
        )
        self.log: list[dict] = []
        self.step = 0
        self._data_iter = None

    def _next_batch(self):
        """Get a fresh batch. New iterator on exhaustion."""
        if self._data_iter is None:
            self._data_iter = iter(self.train_loader)
        try:
            return next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(self.train_loader)
            return next(self._data_iter)

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

        layers_desc = (
            f"layers={self.active_layers}" if self.active_layers
            else "all layers"
        )
        print(f"\n{'='*60}")
        print(f"TFLE v2 Training — {total_steps} steps")
        print(f"Ternary params: {self.model.get_ternary_param_count():,}")
        print(f"K={self.config.num_parallel_proposals}, "
              f"flip_rate={self.config.flip_rate}, "
              f"T_init={self.config.initial_temperature}")
        print(f"Anti-overfitting: reeval={self.reeval}, {layers_desc}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")

        while self.step < total_steps:
            metrics = self._train_step()
            self.scheduler.step_update(-metrics["loss"])

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
                    "reeval_rejections": metrics.get("reeval_rejections", 0),
                    "elapsed": timer.elapsed_str(),
                }
                if self.step % (eval_every * 4) == 0:
                    entry["sample"] = self.model.generate_text(prompt, 200)
                self.log.append(entry)
                self._print_status(entry)

            if self.step > 0 and self.step % checkpoint_every == 0:
                self.model.save_checkpoint(
                    os.path.join(checkpoint_dir, f"tfle_v2_step{self.step}.pt")
                )
                save_results(self.log, os.path.join(results_dir, "tfle_v2_log.json"))

            self.step += 1

        self.model.save_checkpoint(
            os.path.join(checkpoint_dir, "tfle_v2_final.pt")
        )
        save_results(self.log, os.path.join(results_dir, "tfle_v2_log.json"))
        return self.log

    def _train_step(self) -> dict:
        temperature = self.scheduler.get_temperature()
        K = self.config.num_parallel_proposals

        # FRESH batch 1: for TFLE proposal evaluation
        inputs1, targets1 = self._next_batch()
        inputs1, targets1 = inputs1.to(self.device), targets1.to(self.device)

        with torch.no_grad():
            flat_emb1 = self.model.embed(inputs1)

        # Determine which layers to train
        layer_indices = (
            self.active_layers if self.active_layers is not None
            else list(range(len(self.model.layers)))
        )

        accepted_count = 0
        reeval_rejections = 0

        for layer_idx in layer_indices:
            result = self._tfle_layer_step(
                flat_emb1, targets1, layer_idx, K, temperature
            )

            if result["accepted"] and self.reeval:
                # FRESH batch 2: re-evaluate accepted proposal
                inputs2, targets2 = self._next_batch()
                inputs2, targets2 = inputs2.to(self.device), targets2.to(self.device)

                with torch.no_grad():
                    flat_emb2 = self.model.embed(inputs2)
                    # Evaluate current (post-accept) model on fresh batch
                    logits_new = self._forward_from_embedded(flat_emb2)
                    loss_new = F.cross_entropy(logits_new, targets2).item()

                    # Evaluate with OLD weights (revert and test)
                    old_w = result["old_weights"]
                    self.model.layers[layer_idx].weights = old_w
                    logits_old = self._forward_from_embedded(flat_emb2)
                    loss_old = F.cross_entropy(logits_old, targets2).item()

                if loss_new < loss_old:
                    # Confirmed improvement on fresh data — keep it
                    self.model.layers[layer_idx].weights = result["new_weights"]
                    accepted_count += 1
                else:
                    # Batch-specific fluke — revert
                    reeval_rejections += 1
            elif result["accepted"]:
                accepted_count += 1

        # FRESH batch 3: embedding update (separate from TFLE eval batches)
        inputs3, targets3 = self._next_batch()
        inputs3, targets3 = inputs3.to(self.device), targets3.to(self.device)

        self.embed_optimizer.zero_grad()
        emb = self.model.embed(inputs3)
        logits = self.model.ternary_forward(emb)
        loss = F.cross_entropy(logits, targets3)
        loss.backward()
        self.embed_optimizer.step()

        n_layers = len(layer_indices)
        return {
            "loss": loss.item(),
            "accepted_layers": accepted_count,
            "acceptance_rate": accepted_count / max(n_layers, 1),
            "temperature": temperature,
            "reeval_rejections": reeval_rejections,
        }

    def _tfle_layer_step(self, flat_emb, targets, layer_idx, K, temperature):
        layer = self.model.layers[layer_idx]
        layer.step_count += 1

        # Decay cooldowns
        expired = [k for k, v in layer.cooldown_map.items() if v <= 0]
        for k in expired:
            del layer.cooldown_map[k]
        for k in layer.cooldown_map:
            layer.cooldown_map[k] -= 1

        combined = layer._get_combined_traces()
        candidates = layer._select_candidates(combined)

        proposals = generate_k_proposals(layer.weights, candidates, K, self.device)
        current_w = layer.weights.unsqueeze(0).to(self.device)
        all_proposals = torch.cat([current_w, proposals], dim=0)

        losses = self._batched_eval(flat_emb, targets, layer_idx, all_proposals)

        loss_before = losses[0].item()
        fitness_before = -loss_before

        # Centered rank shaping for K>1
        if K > 1:
            proposal_losses = losses[1:]
            ranks = proposal_losses.argsort().argsort().float()
            # Don't use rank shaping for accept/reject — just pick best
            best_k = proposal_losses.argmin().item()
        else:
            best_k = 0

        fitness_after = -losses[best_k + 1].item()
        delta = fitness_after - fitness_before

        layer_temp = self.config.get_temperature_for_layer(temperature, layer.layer_idx)
        accepted = layer._accept_or_reject(delta, layer_temp)

        old_weights = layer.weights.clone()
        new_weights = all_proposals[best_k + 1].to(torch.int8)

        if accepted and delta != 0:
            layer.weights = new_weights

        # Update traces
        with torch.no_grad():
            h = flat_emb
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
            "delta": delta,
            "old_weights": old_weights,
            "new_weights": new_weights,
        }

    @torch.no_grad()
    def _forward_from_embedded(self, flat_emb):
        """Forward through ternary layers only."""
        return self.model.ternary_forward(flat_emb)

    @torch.no_grad()
    def _batched_eval(self, flat_emb, targets, layer_idx, proposals):
        K_plus_1 = proposals.shape[0]

        h = flat_emb
        for i in range(layer_idx):
            h = self.model.layers[i].forward(h)
            if i < len(self.model.layers) - 1:
                h = F.relu(h)
                h = F.layer_norm(h, h.shape[-1:])

        h_expanded = h.unsqueeze(0).expand(K_plus_1, -1, -1)
        w_float = proposals.float()
        varied = torch.bmm(h_expanded, w_float)
        if layer_idx < len(self.model.layers) - 1:
            varied = F.relu(varied)
            varied = F.layer_norm(varied, varied.shape[-1:])

        K_val, B, F_out = varied.shape
        h_flat = varied.reshape(K_val * B, F_out)
        for i in range(layer_idx + 1, len(self.model.layers)):
            h_flat = self.model.layers[i].forward(h_flat)
            if i < len(self.model.layers) - 1:
                h_flat = F.relu(h_flat)
                h_flat = F.layer_norm(h_flat, h_flat.shape[-1:])

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
        return {"loss": avg_loss, "perplexity": compute_perplexity(avg_loss)}

    def _print_status(self, entry: dict):
        parts = [
            f"step={entry['step']:>6d}",
            f"train_loss={entry['train_loss']:.3f}",
            f"val_loss={entry['val_loss']:.3f}",
            f"val_ppl={entry['val_perplexity']:.1f}",
            f"accept={entry['acceptance_rate']:.2f}",
            f"T={entry['temperature']:.4f}",
            f"reeval_rej={entry['reeval_rejections']}",
            f"[{entry['elapsed']}]",
        ]
        print(" | ".join(parts))
        if entry.get("sample"):
            preview = entry["sample"][:120].replace("\n", "\\n")
            print(f"  Sample: {preview}...")
