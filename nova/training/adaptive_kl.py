"""Adaptive KL distillation losses for NOVA.

Used during 18B-token distillation from a teacher whose top-K=128 logits have
been cached to disk. The sparse variant operates on top-K indices + values
(the only things kept in the cache), the full variant is used for gradcheck
and unit tests where we want to exercise the same math against a full
teacher distribution.

Adaptive KL = alpha * forward_KL(student || teacher) + (1 - alpha) * reverse_KL.
- Forward KL (alpha=1) asks the student to cover the teacher's high-probability
  modes (mass-covering, safer for diverse teachers).
- Reverse KL (alpha=0) asks the student to sharpen itself toward the teacher
  (mode-seeking, better for crisp reasoning traces).
- alpha=0.5 averages the two, which is what we use for the 18B run.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def adaptive_kl_loss_sparse(
    student_logits: torch.Tensor,
    cached_indices: torch.Tensor,
    cached_values: torch.Tensor,
    temperature: float,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Adaptive KL on sparse top-K teacher logits.

    Args:
        student_logits: [batch, seq, vocab] full student logits.
        cached_indices: [batch, seq, K] teacher's top-K token indices.
        cached_values: [batch, seq, K] teacher's top-K logit values.
        temperature: softening temperature (>= 1 softens, higher = flatter).
        alpha: weight on forward KL; (1 - alpha) goes to reverse KL.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    indices = cached_indices.long()
    student_at_topk = torch.gather(student_logits, dim=-1, index=indices)

    t_scaled = cached_values / temperature
    s_scaled = student_at_topk / temperature

    t_log = F.log_softmax(t_scaled, dim=-1)
    s_log = F.log_softmax(s_scaled, dim=-1)
    t_probs = t_log.exp()
    s_probs = s_log.exp()

    forward_kl = F.kl_div(s_log, t_probs, reduction="batchmean")
    reverse_kl = F.kl_div(t_log, s_probs, reduction="batchmean")

    loss = alpha * forward_kl + (1.0 - alpha) * reverse_kl
    return loss * (temperature ** 2)


def adaptive_kl_loss_full(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Full-vocabulary adaptive KL. Used for testing / ablations.

    Args:
        student_logits: [batch, seq, vocab]
        teacher_logits: [batch, seq, vocab]
        temperature: softening temperature.
        alpha: weight on forward KL; (1 - alpha) goes to reverse KL.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"shape mismatch: student {tuple(student_logits.shape)} vs "
            f"teacher {tuple(teacher_logits.shape)}"
        )

    t_log = F.log_softmax(teacher_logits / temperature, dim=-1)
    s_log = F.log_softmax(student_logits / temperature, dim=-1)
    t_probs = t_log.exp()
    s_probs = s_log.exp()

    forward_kl = F.kl_div(s_log, t_probs, reduction="batchmean")
    reverse_kl = F.kl_div(t_log, s_probs, reduction="batchmean")

    loss = alpha * forward_kl + (1.0 - alpha) * reverse_kl
    return loss * (temperature ** 2)


def combined_distillation_loss(
    student_logits: torch.Tensor,
    cached_indices: torch.Tensor,
    cached_values: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    akl_alpha: float = 0.5,
    ce_alpha: float = 0.5,
    distill_alpha: float = 0.5,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combined distillation loss: adaptive KL + hard-label cross-entropy.

    total = distill_alpha * adaptive_kl + ce_alpha * cross_entropy

    Args:
        student_logits: [batch, seq, vocab]
        cached_indices: [batch, seq, K] teacher top-K indices
        cached_values: [batch, seq, K] teacher top-K logit values
        labels: [batch, seq] ground-truth next-token ids
        temperature: softening temperature for the KL term
        akl_alpha: forward/reverse split inside adaptive_kl
        ce_alpha: weight on the hard-label cross-entropy term
        distill_alpha: weight on the distillation KL term
        ignore_index: label value that should be masked from the CE term

    Returns:
        (total_loss, components) where components is a dict of detached
        scalars: {'distill', 'ce', 'total'}.
    """
    distill = adaptive_kl_loss_sparse(
        student_logits,
        cached_indices,
        cached_values,
        temperature=temperature,
        alpha=akl_alpha,
    )

    vocab = student_logits.size(-1)
    ce = F.cross_entropy(
        student_logits.reshape(-1, vocab),
        labels.reshape(-1),
        ignore_index=ignore_index,
    )

    total = distill_alpha * distill + ce_alpha * ce
    components = {
        "distill": distill.detach(),
        "ce": ce.detach(),
        "total": total.detach(),
    }
    return total, components


def temperature_anneal(
    step: int,
    total_steps: int,
    t_start: float = 2.0,
    t_end: float = 1.0,
) -> float:
    """Linear temperature anneal from t_start to t_end over total_steps.

    Early in training we use a high temperature so the student can see the
    teacher's full soft distribution; we decay toward 1.0 as the student
    becomes competent and we want crisp matching.
    """
    if total_steps <= 0:
        return t_end
    if step <= 0:
        return t_start
    if step >= total_steps:
        return t_end
    progress = step / total_steps
    return t_start + progress * (t_end - t_start)
