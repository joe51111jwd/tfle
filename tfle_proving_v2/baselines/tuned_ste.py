"""Properly tuned STE baseline: AdamW, cosine LR, warmup, grad clip, weight decay."""

from __future__ import annotations

import json
import math
import os

import torch
import torch.nn.functional as F


class TunedSTETrainer:
    """Trains an STE model with proper hyperparameter tuning."""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device: torch.device,
        lr: float = 3e-4,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
        seq_model: bool = True,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.grad_clip = grad_clip
        self.seq_model = seq_model

        self.model.to(device)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999)
        )
        self.log: list[dict] = []
        self.step = 0
        self._data_iter = None

    def _next_batch(self):
        if self._data_iter is None:
            self._data_iter = iter(self.train_loader)
        try:
            return next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(self.train_loader)
            return next(self._data_iter)

    def _get_lr(self, total_steps: int) -> float:
        if self.step < self.warmup_steps:
            return self.lr * self.step / max(self.warmup_steps, 1)
        progress = (self.step - self.warmup_steps) / max(total_steps - self.warmup_steps, 1)
        return self.lr * 0.5 * (1 + math.cos(math.pi * progress))

    def train(
        self,
        total_steps: int = 40000,
        eval_every: int = 500,
        results_dir: str = "results",
        prompt: str = "ROMEO:\n",
    ) -> list[dict]:
        os.makedirs(results_dir, exist_ok=True)

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"\n{'='*60}")
        print(f"Tuned STE Baseline — {total_steps} steps")
        print(f"Params: {n_params:,}, LR: {self.lr}, Warmup: {self.warmup_steps}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")

        self.model.train()
        while self.step < total_steps:
            x, y = self._next_batch()
            x, y = x.to(self.device), y.to(self.device)

            # Cosine LR schedule
            lr = self._get_lr(total_steps)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            logits = self.model(x)
            if self.seq_model:
                B, T, V = logits.shape
                loss = F.cross_entropy(logits.reshape(B * T, V), y.reshape(B * T))
            else:
                loss = F.cross_entropy(logits, y)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            if self.step % eval_every == 0:
                val = self.evaluate()
                entry = {
                    "step": self.step,
                    "train_loss": loss.item(),
                    "val_loss": val["loss"],
                    "val_ppl": math.exp(min(val["loss"], 20)),
                    "lr": lr,
                }
                if self.step % (eval_every * 4) == 0:
                    self.model.eval()
                    entry["sample"] = self.model.generate_text(prompt, 200)
                    self.model.train()
                self.log.append(entry)
                print(f"step={self.step:>6d} | val_loss={val['loss']:.3f} | "
                      f"ppl={entry['val_ppl']:.1f} | lr={lr:.2e}")
                if entry.get("sample"):
                    print(f"  Sample: {entry['sample'][:100].replace(chr(10), '\\n')}")

            self.step += 1

        with open(os.path.join(results_dir, "tuned_ste_log.json"), "w") as f:
            json.dump(self.log, f, indent=2, default=str)
        return self.log

    @torch.no_grad()
    def evaluate(self) -> dict:
        self.model.eval()
        total_loss = 0.0
        n = 0
        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            if self.seq_model:
                B, T, V = logits.shape
                loss = F.cross_entropy(logits.reshape(B * T, V), y.reshape(B * T), reduction="sum")
                n += B * T
            else:
                loss = F.cross_entropy(logits, y, reduction="sum")
                n += y.shape[0]
            total_loss += loss.item()
        self.model.train()
        avg = total_loss / max(n, 1)
        return {"loss": avg, "perplexity": math.exp(min(avg, 20))}
