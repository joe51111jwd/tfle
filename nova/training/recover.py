"""STE recovery training after pruning + ternarization (CONDITIONAL).

CONDITIONAL: Only used if pruning tests pass. Part of the
prune -> ternarize -> recover pipeline.

Uses soft labels from an unpruned teacher model to recover accuracy
lost during pruning and ternarization. The loss combines KL divergence
(match teacher distribution) with cross-entropy (match ground truth).

STE (Straight-Through Estimator) allows gradients to flow through
the ternary quantization during recovery training.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class RecoveryConfig:
    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    kl_weight: float = 0.7
    ce_weight: float = 0.3
    temperature: float = 2.0
    lr_reduce_factor: float = 0.5
    loss_spike_threshold: float = 1.5
    min_lr: float = 1e-7


class RecoveryTrainer:
    """CONDITIONAL: STE recovery training with soft labels from unpruned teacher.

    Loss = kl_weight * KL(teacher || student) + ce_weight * CE(student, targets)

    Monitors loss stability and auto-reduces LR if loss spikes above
    loss_spike_threshold * running_average.
    """

    def __init__(
        self,
        config: RecoveryConfig | None = None,
        device: torch.device | None = None,
    ):
        self.config = config or RecoveryConfig()
        self.device = device or torch.device("cpu")
        self._loss_history: list[float] = []
        self._lr_reductions = 0

    def _compute_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Combined KL divergence + cross-entropy loss.

        KL: match the teacher's soft distribution (temperature-scaled).
        CE: match ground truth labels for accuracy.
        """
        T = self.config.temperature
        vocab_size = student_logits.shape[-1]

        # KL divergence with temperature scaling
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)
        kl_loss = F.kl_div(
            student_log_probs.reshape(-1, vocab_size),
            teacher_probs.reshape(-1, vocab_size),
            reduction="batchmean",
        ) * (T * T)

        # cross-entropy with ground truth
        ce_loss = F.cross_entropy(
            student_logits.reshape(-1, vocab_size),
            targets.reshape(-1),
            ignore_index=0,
        )

        return self.config.kl_weight * kl_loss + self.config.ce_weight * ce_loss

    def _check_loss_spike(
        self,
        current_loss: float,
        optimizer: torch.optim.Optimizer,
    ) -> bool:
        """Auto-reduce LR if loss spikes. Returns True if LR was reduced."""
        self._loss_history.append(current_loss)

        if len(self._loss_history) < 10:
            return False

        # running average of last 10 steps
        recent_avg = sum(self._loss_history[-10:]) / 10

        if current_loss > recent_avg * self.config.loss_spike_threshold:
            current_lr = optimizer.param_groups[0]["lr"]
            new_lr = max(current_lr * self.config.lr_reduce_factor, self.config.min_lr)

            if new_lr < current_lr:
                for pg in optimizer.param_groups:
                    pg["lr"] = new_lr
                self._lr_reductions += 1
                logger.warning(
                    f"Loss spike detected ({current_loss:.4f} vs avg {recent_avg:.4f}). "
                    f"Reducing LR: {current_lr:.2e} -> {new_lr:.2e}"
                )
                return True

        return False

    def train(
        self,
        model: nn.Module,
        teacher: nn.Module,
        data_loader,
        num_steps: int = 1000,
        lr: float | None = None,
    ) -> dict:
        """STE recovery training loop.

        Args:
            model: The pruned+ternarized student model (trainable).
            teacher: The unpruned teacher model (frozen).
            data_loader: Yields (input_ids, targets) batches.
            num_steps: Number of training steps.
            lr: Learning rate override.

        Returns:
            Training stats dict.
        """
        effective_lr = lr or self.config.lr

        # freeze teacher
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=effective_lr,
            weight_decay=self.config.weight_decay,
        )

        self._loss_history = []
        self._lr_reductions = 0

        model.train()
        data_iter = iter(data_loader)
        total_loss = 0.0
        step_losses = []

        for step in range(num_steps):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                x, y = next(data_iter)

            x = x.to(self.device)
            y = y.to(self.device)

            # teacher forward (no grad)
            with torch.no_grad():
                use_amp = self.device.type == "cuda"
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        teacher_logits = teacher(x)
                else:
                    teacher_logits = teacher(x)
                teacher_logits = teacher_logits.float()

            # student forward (STE passes gradients through ternary quantization)
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    student_logits = model(x)
            else:
                student_logits = model(x)
            student_logits = student_logits.float()

            loss = self._compute_loss(student_logits, teacher_logits, y)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
            optimizer.step()

            loss_val = loss.item()
            total_loss += loss_val
            step_losses.append(loss_val)

            # auto-reduce LR on spikes
            self._check_loss_spike(loss_val, optimizer)

            if step % 100 == 0:
                avg = total_loss / (step + 1)
                current_lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Recovery step {step}/{num_steps} | "
                    f"loss={loss_val:.4f} | avg={avg:.4f} | lr={current_lr:.2e}"
                )

        avg_loss = total_loss / max(num_steps, 1)
        final_lr = optimizer.param_groups[0]["lr"]

        result = {
            "total_steps": num_steps,
            "avg_loss": avg_loss,
            "final_loss": step_losses[-1] if step_losses else 0.0,
            "initial_lr": effective_lr,
            "final_lr": final_lr,
            "lr_reductions": self._lr_reductions,
        }
        logger.info(f"Recovery training complete: {result}")
        return result
