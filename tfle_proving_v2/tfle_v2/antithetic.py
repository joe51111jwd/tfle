"""Antithetic flip evaluation — test both directions for each candidate.

Instead of randomly picking one of two possible flip directions,
evaluate BOTH and keep whichever is better. Doubles forward passes
per candidate but halves wasted steps.
"""

from __future__ import annotations

import torch


def generate_antithetic_proposals(
    weights: torch.Tensor,
    candidates: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate two proposals per candidate set: +1 and -1 directions.

    For ternary {-1, 0, +1}:
      v=0:  proposal_A=+1, proposal_B=-1
      v=+1: proposal_A=0,  proposal_B=-1
      v=-1: proposal_A=0,  proposal_B=+1

    Returns:
        (proposal_A, proposal_B) each shape (in_features, out_features)
    """
    flat = weights.flatten().long()
    idx_vals = flat[candidates] + 1  # map {-1,0,+1} -> {0,1,2}

    prop_a = flat.clone()
    prop_b = flat.clone()

    # Direction A: +1 mod 3
    new_a = (idx_vals + 1) % 3 - 1
    prop_a[candidates] = new_a

    # Direction B: -1 mod 3 (equivalent to +2 mod 3)
    new_b = (idx_vals + 2) % 3 - 1
    prop_b[candidates] = new_b

    shape = weights.shape
    return (
        prop_a.reshape(shape).to(torch.int8),
        prop_b.reshape(shape).to(torch.int8),
    )
