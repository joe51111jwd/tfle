"""Phased co-evolution trainer for TFLE v2.

Three phases:
  Phase 1 (embedding warmup): TFLE disabled, embeddings learn through random ternary projections
  Phase 2 (gentle TFLE): conservative flips with permissive re-eval, embeddings co-adapt
  Phase 3 (full TFLE): standard params, strict re-eval, maximize ternary contribution

Supports both AttentionLM (sequence-level loss) and CharLM (single-token loss).
"""

from __future__ import annotations

import os
import sys
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tfle.annealing import TemperatureScheduler
from tfle.config import TFLEConfig, CoolingSchedule, FitnessType
from tfle.layers import generate_k_proposals

from .validation_gate import ValidationGate
from .antithetic import generate_antithetic_proposals
from .fitness_shaping import centered_rank_transform


def _compute_perplexity(loss: float) -> float:
    return math.exp(min(loss, 20.0))


class PhasedTrainer:
    """Three-phase co-evolution trainer.

    Works with any model that has:
      - .layers: list[TFLELayer]
      - .forward(x) -> logits
      - .get_float_params() -> list[nn.Parameter]
      - .generate_text(prompt, length) -> str
      - .save_checkpoint(path), .load_checkpoint(path)
    """

    def __init__(
        self,
        model,
        config: TFLEConfig,
        train_loader,
        val_loader,
        device: torch.device,
        # Phase boundaries
        phase1_steps: int = 5000,
        phase2_steps: int = 10000,
        # Phase 1: embedding warmup
        embed_lr_init: float = 1e-3,
        # Phase 2: gentle TFLE
        phase2_flip_rate: float = 0.001,
        phase2_K: int = 32,
        phase2_tolerance: float = 0.05,
        # Phase 3: full TFLE (uses config values)
        phase3_tolerance_init: float = 0.05,
        phase3_tolerance_final: float = 0.01,
        # Features
        use_antithetic: bool = False,
        use_rank_shaping: bool = False,
        active_layers: list[int] | None = None,
        seq_model: bool = True,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.phase1_end = phase1_steps
        self.phase2_end = phase1_steps + phase2_steps
        self.phase2_flip_rate = phase2_flip_rate
        self.phase2_K = phase2_K
        self.use_antithetic = use_antithetic
        self.use_rank_shaping = use_rank_shaping
        self.active_layers = active_layers
        self.seq_model = seq_model

        # Validation gate
        self.gate = ValidationGate(
            tolerance_init=phase2_tolerance,
            tolerance_final=phase3_tolerance_final,
            warmup_steps=phase1_steps,
            anneal_steps=phase2_steps + 20000,
        )

        # Temperature scheduler (for TFLE phases)
        self.temp_scheduler = TemperatureScheduler(config)

        # Float param optimizer with warmup + cosine decay
        self.float_params = model.get_float_params()
        self.optimizer = torch.optim.AdamW(
            self.float_params, lr=embed_lr_init, weight_decay=0.01
        )
        self.lr_init = embed_lr_init

        self.log: list[dict] = []
        self.step = 0
        self._data_iter = None

    def _get_phase(self) -> int:
        if self.step < self.phase1_end:
            return 1
        if self.step < self.phase2_end:
            return 2
        return 3

    def _get_embed_lr(self) -> float:
        phase = self._get_phase()
        if phase == 1:
            # Warmup for first 500 steps, then constant
            warmup = min(1.0, self.step / 500)
            return self.lr_init * warmup
        elif phase == 2:
            # Decay from init to init/10
            progress = (self.step - self.phase1_end) / max(self.phase2_end - self.phase1_end, 1)
            return self.lr_init * (1 - 0.9 * progress)
        else:
            # Phase 3: low LR
            return self.lr_init * 0.01

    def _get_K(self) -> int:
        phase = self._get_phase()
        if phase == 1:
            return 0  # No TFLE
        if phase == 2:
            return self.phase2_K
        return self.config.num_parallel_proposals

    def _get_flip_rate(self) -> float:
        phase = self._get_phase()
        if phase == 1:
            return 0.0
        if phase == 2:
            return self.phase2_flip_rate
        return self.config.flip_rate

    def _next_batch(self):
        if self._data_iter is None:
            self._data_iter = iter(self.train_loader)
        try:
            return next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(self.train_loader)
            return next(self._data_iter)

    def train(
        self,
        total_steps: int = 40000,
        eval_every: int = 500,
        checkpoint_every: int = 5000,
        checkpoint_dir: str = "checkpoints",
        results_dir: str = "results",
        prompt: str = "ROMEO:\n",
    ) -> list[dict]:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        n_ternary = self.model.get_ternary_param_count()
        n_float = self.model.get_float_param_count()
        print(f"\n{'='*60}")
        print(f"Phased Co-Evolution Trainer")
        print(f"Ternary: {n_ternary:,}, Float: {n_float:,}")
        print(f"Phase 1 (embed warmup): steps 0-{self.phase1_end}")
        print(f"Phase 2 (gentle TFLE): steps {self.phase1_end}-{self.phase2_end}")
        print(f"Phase 3 (full TFLE): steps {self.phase2_end}+")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")

        while self.step < total_steps:
            metrics = self._train_step()
            self.gate.advance()

            if self.step % eval_every == 0:
                val = self.evaluate()
                entry = {
                    "step": self.step,
                    "phase": self._get_phase(),
                    "train_loss": metrics["loss"],
                    "val_loss": val["loss"],
                    "val_ppl": _compute_perplexity(val["loss"]),
                    "embed_lr": self._get_embed_lr(),
                    "K": self._get_K(),
                    "flip_rate": self._get_flip_rate(),
                    "accepted": metrics.get("accepted", 0),
                    "reeval_passed": metrics.get("reeval_passed", 0),
                    "reeval_rejected": metrics.get("reeval_rejected", 0),
                    "temperature": metrics.get("temperature", 0),
                    "gate_tolerance": self.gate.get_tolerance(),
                }
                if self.step % (eval_every * 4) == 0:
                    entry["sample"] = self.model.generate_text(prompt, 200)
                self.log.append(entry)
                self._print(entry)

            if self.step > 0 and self.step % checkpoint_every == 0:
                self.model.save_checkpoint(os.path.join(checkpoint_dir, f"step{self.step}.pt"))
                self._save_log(results_dir)

            self.step += 1

        self.model.save_checkpoint(os.path.join(checkpoint_dir, "final.pt"))
        self._save_log(results_dir)
        return self.log

    def _train_step(self) -> dict:
        phase = self._get_phase()
        K = self._get_K()

        # Batch 1: TFLE evaluation (if phase 2+)
        x1, y1 = self._next_batch()
        x1, y1 = x1.to(self.device), y1.to(self.device)

        accepted = 0
        reeval_passed = 0
        reeval_rejected = 0
        temperature = 0.0

        if K > 0:
            temperature = self.temp_scheduler.get_temperature()
            layer_indices = self.active_layers or list(range(len(self.model.layers)))

            for layer_idx in layer_indices:
                result = self._tfle_step(x1, y1, layer_idx, K, temperature)

                if result["accepted"]:
                    # Re-evaluate on fresh batch
                    x_re, y_re = self._next_batch()
                    x_re, y_re = x_re.to(self.device), y_re.to(self.device)

                    keep = self.gate.check(
                        self.model, layer_idx,
                        result["old_weights"], result["new_weights"],
                        x_re, y_re,
                    )
                    if keep:
                        reeval_passed += 1
                        accepted += 1
                    else:
                        self.model.layers[layer_idx].weights = result["old_weights"]
                        reeval_rejected += 1

            self.temp_scheduler.step_update()

        # Batch 2: float params update (always, fresh batch)
        x2, y2 = self._next_batch()
        x2, y2 = x2.to(self.device), y2.to(self.device)

        # Update LR
        for pg in self.optimizer.param_groups:
            pg["lr"] = self._get_embed_lr()

        self.optimizer.zero_grad()
        logits = self.model.forward(x2)

        if self.seq_model:
            # Sequence model: loss over all positions
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.reshape(B * T, V), y2.reshape(B * T))
        else:
            loss = F.cross_entropy(logits, y2)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.float_params, 1.0)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "accepted": accepted,
            "reeval_passed": reeval_passed,
            "reeval_rejected": reeval_rejected,
            "temperature": temperature,
        }

    def _tfle_step(self, x, y, layer_idx, K, temperature):
        """K-proposal TFLE step for one layer."""
        layer = self.model.layers[layer_idx]
        layer.step_count += 1

        # Temporarily override flip rate for phase 2
        orig_flip_rate = self.config.flip_rate
        self.config.flip_rate = self._get_flip_rate()

        combined = layer._get_combined_traces()
        candidates = layer._select_candidates(combined)
        self.config.flip_rate = orig_flip_rate  # restore

        old_weights = layer.weights.clone()

        if self.use_antithetic and K <= 2:
            prop_a, prop_b = generate_antithetic_proposals(
                layer.weights, candidates, self.device
            )
            proposals = torch.stack([prop_a, prop_b])
        else:
            proposals = generate_k_proposals(layer.weights, candidates, K, self.device)

        current_w = layer.weights.unsqueeze(0).to(self.device)
        all_proposals = torch.cat([current_w, proposals], dim=0)

        losses = self._batched_eval(x, y, layer_idx, all_proposals)

        loss_before = losses[0].item()
        best_k = losses[1:].argmin().item()
        loss_after = losses[best_k + 1].item()
        delta = loss_before - loss_after  # positive = improvement

        layer_temp = self.config.get_temperature_for_layer(temperature, layer.layer_idx)
        accepted = layer._accept_or_reject(delta, layer_temp)

        new_weights = all_proposals[best_k + 1].to(torch.int8)
        if accepted and delta > 0:
            layer.weights = new_weights
        else:
            accepted = False
            new_weights = old_weights

        # Update traces — skip if input size doesn't match (e.g. FF layers in transformer)
        try:
            with torch.no_grad():
                if self.seq_model:
                    h = self.model.embedding(x)
                    flat = h.reshape(-1, h.shape[-1])
                else:
                    flat = self.model.embed(x)
                sample = flat[:min(256, flat.shape[0])]
                if sample.shape[-1] == layer.in_features:
                    output = layer.forward(sample)
                    layer._update_traces(sample, output, delta <= 0)
        except Exception:
            pass  # Trace update is optional; selection falls back to exploration
        layer.fitness_history.append(-loss_after if accepted else -loss_before)
        layer.acceptance_history.append(accepted)

        return {
            "accepted": accepted,
            "delta": delta,
            "old_weights": old_weights,
            "new_weights": new_weights if accepted else old_weights,
        }

    @torch.no_grad()
    def _batched_eval(self, x, y, layer_idx, proposals):
        """Evaluate proposals by swapping layer weights and running full forward."""
        K_plus_1 = proposals.shape[0]
        layer = self.model.layers[layer_idx]
        original = layer.weights

        losses = []
        for k in range(K_plus_1):
            layer.weights = proposals[k].to(torch.int8)
            logits = self.model.forward(x)
            if self.seq_model:
                B, T, V = logits.shape
                loss = F.cross_entropy(logits.reshape(B * T, V), y.reshape(B * T))
            else:
                loss = F.cross_entropy(logits, y)
            losses.append(loss.item())

        layer.weights = original
        return torch.tensor(losses, device=self.device)

    @torch.no_grad()
    def evaluate(self) -> dict:
        total_loss = 0.0
        n = 0
        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model.forward(x)
            if self.seq_model:
                B, T, V = logits.shape
                loss = F.cross_entropy(logits.reshape(B * T, V), y.reshape(B * T), reduction="sum")
                n += B * T
            else:
                loss = F.cross_entropy(logits, y, reduction="sum")
                n += y.shape[0]
            total_loss += loss.item()
        avg = total_loss / max(n, 1)
        return {"loss": avg, "perplexity": _compute_perplexity(avg)}

    def _print(self, e: dict):
        parts = [
            f"P{e['phase']}",
            f"step={e['step']:>6d}",
            f"val_loss={e['val_loss']:.3f}",
            f"ppl={e['val_ppl']:.1f}",
            f"K={e['K']}",
            f"acc={e['accepted']}",
            f"reeval={e['reeval_passed']}/{e['reeval_passed']+e['reeval_rejected']}",
            f"tol={e['gate_tolerance']:.3f}",
        ]
        print(" | ".join(parts))
        if e.get("sample"):
            preview = e["sample"][:100].replace("\n", "\\n")
            print(f"  Sample: {preview}")

    def _save_log(self, results_dir: str):
        with open(os.path.join(results_dir, "phased_log.json"), "w") as f:
            json.dump(self.log, f, indent=2, default=str)
