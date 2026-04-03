"""Rank-based fitness shaping — NES-style utilities.

Makes proposal selection invariant to loss scale and eliminates
outlier domination. Lower loss = higher rank = higher utility.
"""

from __future__ import annotations

import math

import torch


def centered_rank_transform(losses: torch.Tensor) -> torch.Tensor:
    """Transform raw losses into centered ranks in [-0.5, 0.5].

    Lower loss gets higher (more positive) rank.
    """
    n = losses.shape[0]
    ranks = losses.argsort().argsort().float()  # 0 = lowest loss
    # Invert: lowest loss should get highest rank
    ranks = (n - 1) - ranks
    return ranks / (n - 1) - 0.5


def nes_utilities(losses: torch.Tensor) -> torch.Tensor:
    """NES-style utilities: positive weight only for top half.

    Lower loss = better. Top half gets positive utility,
    bottom half gets zero. Normalized to sum to 1.
    """
    n = losses.shape[0]
    mu = n // 2
    ranks = losses.argsort().argsort()  # 0 = lowest loss
    # Invert ranks: lowest loss = rank 0 = best
    inv_ranks = (n - 1) - ranks

    utilities = torch.zeros(n, device=losses.device)
    for i in range(n):
        r = inv_ranks[i].item()
        if r < mu:
            utilities[i] = max(0.0, math.log(mu + 0.5) - math.log(r + 1))

    total = utilities.sum()
    if total > 0:
        utilities /= total
    return utilities
