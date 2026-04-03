"""STE baseline trainer for character-level text prediction.

Same architecture as the TFLE CharLM, trained with backprop + STE.
"""

from __future__ import annotations

import os
import time

import torch
import torch.nn.functional as F

from .utils import compute_perplexity, save_results, Timer


class STETextTrainer:
    """Trains an STECharLM model with standard backprop."""

    def __init__(self, model, train_loader, val_loader, device, lr=1e-3):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.model.to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.log: list[dict] = []
        self.step = 0

    def train(
        self,
        total_steps: int = 20000,
        eval_every: int = 500,
        results_dir: str = "results",
        prompt: str = "ROMEO:\n",
    ) -> list[dict]:
        os.makedirs(results_dir, exist_ok=True)
        timer = Timer()
        data_iter = iter(self.train_loader)

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"\n{'='*60}")
        print(f"STE Baseline Training — {total_steps} steps")
        print(f"Total params: {n_params:,}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")

        self.model.train()
        while self.step < total_steps:
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                inputs, targets = next(data_iter)

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            logits = self.model(inputs)
            loss = F.cross_entropy(logits, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.step % eval_every == 0:
                val_metrics = self.evaluate()
                entry = {
                    "step": self.step,
                    "train_loss": loss.item(),
                    "val_loss": val_metrics["loss"],
                    "train_perplexity": compute_perplexity(loss.item()),
                    "val_perplexity": compute_perplexity(val_metrics["loss"]),
                    "elapsed": timer.elapsed_str(),
                }
                if self.step % (eval_every * 4) == 0:
                    self.model.eval()
                    sample = self.model.generate_text(prompt, length=200)
                    entry["sample"] = sample
                    self.model.train()

                self.log.append(entry)
                self._print_status(entry)

            self.step += 1

        save_results(self.log, os.path.join(results_dir, "ste_log.json"))
        return self.log

    @torch.no_grad()
    def evaluate(self) -> dict:
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            logits = self.model(inputs)
            loss = F.cross_entropy(logits, targets, reduction="sum")
            total_loss += loss.item()
            total_samples += targets.shape[0]
        self.model.train()
        avg_loss = total_loss / max(total_samples, 1)
        return {
            "loss": avg_loss,
            "perplexity": compute_perplexity(avg_loss),
        }

    def _print_status(self, entry: dict):
        parts = [
            f"step={entry['step']:>6d}",
            f"train_loss={entry['train_loss']:.3f}",
            f"val_loss={entry['val_loss']:.3f}",
            f"val_ppl={entry['val_perplexity']:.1f}",
            f"[{entry['elapsed']}]",
        ]
        print(" | ".join(parts))
        if entry.get("sample"):
            preview = entry["sample"][:120].replace("\n", "\\n")
            print(f"  Sample: {preview}...")
