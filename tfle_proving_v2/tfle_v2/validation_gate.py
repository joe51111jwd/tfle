"""Calibrated re-evaluation gate for TFLE flip acceptance.

Fixes the v1 problem where re-eval rejected ALL flips. Uses:
1. Tolerance-based threshold with cosine annealing
2. Larger re-eval batch than training batch for stability
3. Fresh batch sampling for unbiased evaluation
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class ValidationGate:
    """Decides whether to keep an accepted TFLE flip after re-evaluation.

    A flip is kept if: reeval_loss <= baseline_loss + tolerance
    Tolerance anneals from permissive (early: allow neutral flips)
    to strict (late: only genuine improvements).
    """

    def __init__(
        self,
        tolerance_init: float = 0.1,
        tolerance_final: float = 0.01,
        warmup_steps: int = 5000,
        anneal_steps: int = 15000,
        reeval_batch_multiplier: int = 2,
    ):
        self.tolerance_init = tolerance_init
        self.tolerance_final = tolerance_final
        self.warmup_steps = warmup_steps
        self.anneal_steps = anneal_steps
        self.reeval_batch_multiplier = reeval_batch_multiplier

        self.step = 0
        self.total_checks = 0
        self.total_passed = 0
        self.total_rejected = 0

    def get_tolerance(self) -> float:
        if self.step < self.warmup_steps:
            return self.tolerance_init
        progress = min(
            1.0, (self.step - self.warmup_steps) / max(self.anneal_steps, 1)
        )
        # Cosine decay from init to final
        return self.tolerance_final + 0.5 * (
            self.tolerance_init - self.tolerance_final
        ) * (1 + math.cos(math.pi * progress))

    def advance(self):
        self.step += 1

    def check(
        self,
        model,
        layer_idx: int,
        old_weights: torch.Tensor,
        new_weights: torch.Tensor,
        reeval_inputs: torch.Tensor,
        reeval_targets: torch.Tensor,
    ) -> bool:
        """Re-evaluate a flip on fresh data. Returns True if flip should be kept."""
        self.total_checks += 1
        tolerance = self.get_tolerance()

        with torch.no_grad():
            # Evaluate with NEW weights (current state after acceptance)
            logits_new = model.forward(reeval_inputs)
            loss_new = F.cross_entropy(logits_new, reeval_targets).item()

            # Evaluate with OLD weights (revert temporarily)
            model.layers[layer_idx].weights = old_weights
            logits_old = model.forward(reeval_inputs)
            loss_old = F.cross_entropy(logits_old, reeval_targets).item()

            # Restore new weights for now (caller decides final state)
            model.layers[layer_idx].weights = new_weights

        keep = loss_new <= loss_old + tolerance
        if keep:
            self.total_passed += 1
        else:
            self.total_rejected += 1

        return keep

    def get_stats(self) -> dict:
        return {
            "tolerance": self.get_tolerance(),
            "total_checks": self.total_checks,
            "passed": self.total_passed,
            "rejected": self.total_rejected,
            "pass_rate": self.total_passed / max(self.total_checks, 1),
        }
