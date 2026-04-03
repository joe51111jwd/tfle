"""Shared model components: embeddings, positional encoding, causal mask."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Float32 token + positional embedding."""

    def __init__(self, vocab_size: int, d_model: int, max_len: int = 512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T) -> (B, T, d_model)"""
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        return self.token_emb(x) + self.pos_emb(positions)


def causal_mask(T: int, device: torch.device) -> torch.Tensor:
    """Create causal attention mask: True = masked (don't attend)."""
    return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
