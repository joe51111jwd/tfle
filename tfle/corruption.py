"""Data corruption strategies for contrastive fitness."""

from __future__ import annotations

import torch

from .config import CorruptionMethod, TFLEConfig


def corrupt_data(
    x: torch.Tensor,
    config: TFLEConfig,
    labels: torch.Tensor | None = None,
) -> torch.Tensor:
    """Generate corrupted version of input data."""
    method = config.corruption_method
    strength = config.corruption_strength

    if method == CorruptionMethod.LABEL_SHUFFLE:
        # Shuffle within batch (permute rows)
        perm = torch.randperm(x.size(0))
        return x[perm]

    elif method == CorruptionMethod.GAUSSIAN_NOISE:
        noise = torch.randn_like(x) * strength
        return x + noise

    elif method == CorruptionMethod.INPUT_MASK:
        mask = torch.rand_like(x) > strength
        return x * mask.float()

    elif method == CorruptionMethod.FEATURE_PERMUTE:
        # Shuffle features across batch dimension
        corrupted = x.clone()
        n_features = x.size(1)
        n_permute = max(1, int(n_features * strength))
        feature_indices = torch.randperm(n_features)[:n_permute]
        for fi in feature_indices:
            perm = torch.randperm(x.size(0))
            corrupted[:, fi] = x[perm, fi]
        return corrupted

    elif method == CorruptionMethod.MIXUP:
        perm = torch.randperm(x.size(0))
        lam = strength
        return lam * x + (1 - lam) * x[perm]

    raise ValueError(f"Unknown corruption method: {method}")


def generate_negative_samples(
    x: torch.Tensor,
    config: TFLEConfig,
    labels: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Generate multiple corrupted versions."""
    return [corrupt_data(x, config, labels) for _ in range(config.num_negative_samples)]
