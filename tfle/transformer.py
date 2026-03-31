"""Ternary transformer components with TFLE training."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from .config import TFLEConfig
from .corruption import corrupt_data
from .layers import TFLELayer


class TFLEAttention:
    """Ternary multi-head self-attention with TFLE training.

    Q, K, V projection weights are ternary. Attention scores use
    standard softmax on the ternary projections.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        config: TFLEConfig,
        layer_idx: int = 0,
    ):
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.config = config
        self.layer_idx = layer_idx

        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"

        # Ternary projection layers (Q, K, V, Output)
        self.q_proj = TFLELayer(embed_dim, embed_dim, config, layer_idx)
        self.k_proj = TFLELayer(embed_dim, embed_dim, config, layer_idx)
        self.v_proj = TFLELayer(embed_dim, embed_dim, config, layer_idx)
        self.out_proj = TFLELayer(embed_dim, embed_dim, config, layer_idx)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass. x: [batch, seq_len, embed_dim]"""
        B, T, C = x.shape

        def _proj(proj):
            return proj.forward(x.reshape(-1, C)).reshape(
                B, T, self.n_heads, self.head_dim,
            ).transpose(1, 2)

        q = _proj(self.q_proj)
        k = _proj(self.k_proj)
        v = _proj(self.v_proj)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_weights = (q @ k.transpose(-2, -1)) / scale

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = attn_weights @ v

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).reshape(B, T, C)
        output = self.out_proj.forward(attn_output.reshape(-1, C)).reshape(B, T, C)
        return output

    def get_projection_layers(self) -> list[TFLELayer]:
        return [self.q_proj, self.k_proj, self.v_proj, self.out_proj]


class TFLETransformerBlock:
    """Single transformer block with ternary weights."""

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        ff_dim: int,
        config: TFLEConfig,
        layer_idx: int = 0,
    ):
        self.config = config
        self.attention = TFLEAttention(embed_dim, n_heads, config, layer_idx)
        self.ff1 = TFLELayer(embed_dim, ff_dim, config, layer_idx)
        self.ff2 = TFLELayer(ff_dim, embed_dim, config, layer_idx)
        self.embed_dim = embed_dim

        # LayerNorm parameters (these remain continuous — they're cheap)
        self.ln1_weight = torch.ones(embed_dim)
        self.ln1_bias = torch.zeros(embed_dim)
        self.ln2_weight = torch.ones(embed_dim)
        self.ln2_bias = torch.zeros(embed_dim)

    def _layer_norm(
        self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
    ) -> torch.Tensor:
        return F.layer_norm(x, (self.embed_dim,), weight, bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: [batch, seq_len, embed_dim]"""
        B, T, C = x.shape

        # Self-attention with residual
        normed = self._layer_norm(x, self.ln1_weight, self.ln1_bias)
        attn_out = self.attention.forward(normed, mask)
        x = x + attn_out

        # Feed-forward with residual
        normed = self._layer_norm(x, self.ln2_weight, self.ln2_bias)
        ff_input = normed.reshape(-1, C)
        ff_out = F.relu(self.ff1.forward(ff_input))
        ff_out = self.ff2.forward(ff_out)
        ff_out = ff_out.reshape(B, T, C)
        x = x + ff_out

        return x

    def get_all_layers(self) -> list[TFLELayer]:
        """Get all trainable TFLE layers in this block."""
        return self.attention.get_projection_layers() + [self.ff1, self.ff2]


class TFLETransformerModel:
    """Small ternary transformer (GPT-2 style) with TFLE training."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        n_heads: int,
        n_layers: int,
        ff_dim: int,
        max_seq_len: int,
        config: TFLEConfig,
    ):
        self.config = config
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Token and position embeddings (continuous — they're small)
        self.token_embedding = torch.randn(vocab_size, embed_dim) * 0.02
        self.position_embedding = torch.randn(max_seq_len, embed_dim) * 0.02

        # Transformer blocks
        self.blocks = [
            TFLETransformerBlock(embed_dim, n_heads, ff_dim, config, layer_idx=i)
            for i in range(n_layers)
        ]

        # Output projection (ternary)
        self.output_proj = TFLELayer(embed_dim, vocab_size, config, layer_idx=n_layers)

        # LN before output
        self.final_ln_weight = torch.ones(embed_dim)
        self.final_ln_bias = torch.zeros(embed_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: [batch, seq_len] -> logits: [batch, seq_len, vocab_size]"""
        B, T = token_ids.shape

        tok_emb = self.token_embedding[token_ids]  # [B, T, embed_dim]
        pos_emb = self.position_embedding[:T].unsqueeze(0)  # [1, T, embed_dim]
        x = tok_emb + pos_emb

        # Causal mask
        mask = torch.tril(torch.ones(T, T)).unsqueeze(0).unsqueeze(0)

        for block in self.blocks:
            x = block.forward(x, mask)

        x = F.layer_norm(x, (self.embed_dim,), self.final_ln_weight, self.final_ln_bias)
        logits = self.output_proj.forward(x.reshape(-1, self.embed_dim))
        logits = logits.reshape(B, T, self.vocab_size)

        return logits

    def get_all_layers(self) -> list[TFLELayer]:
        """Get all trainable TFLE layers."""
        layers = []
        for block in self.blocks:
            layers.extend(block.get_all_layers())
        layers.append(self.output_proj)
        return layers

    def get_total_params(self) -> int:
        total = 0
        for layer in self.get_all_layers():
            total += layer.in_features * layer.out_features
        # Add embedding params (continuous)
        total += self.vocab_size * self.embed_dim  # token embedding
        total += self.max_seq_len * self.embed_dim  # position embedding
        return total

    def train_step(
        self,
        token_ids: torch.Tensor,
        temperature: float,
    ) -> list[dict]:
        """Train all ternary layers via TFLE for one step."""
        all_metrics = []

        # Forward pass to get activations at each layer
        B, T = token_ids.shape
        tok_emb = self.token_embedding[token_ids]
        pos_emb = self.position_embedding[:T].unsqueeze(0)
        x = tok_emb + pos_emb

        mask = torch.tril(torch.ones(T, T)).unsqueeze(0).unsqueeze(0)

        for block in self.blocks:
            # Train each layer in the block
            for layer in block.get_all_layers():
                # Get input to this specific layer
                if x.shape[-1] == layer.in_features:
                    layer_input = x.reshape(-1, layer.in_features)
                else:
                    layer_input = torch.randn(B * T, layer.in_features)
                corrupted = corrupt_data(layer_input, self.config)
                metrics = layer.train_step(layer_input, corrupted, temperature)
                all_metrics.append(metrics)

            x = block.forward(x, mask)

        # Train output projection
        x_flat = x.reshape(-1, self.embed_dim)
        corrupted = corrupt_data(x_flat, self.config)
        metrics = self.output_proj.train_step(x_flat, corrupted, temperature)
        all_metrics.append(metrics)

        return all_metrics

    def evaluate(self, token_ids: torch.Tensor, targets: torch.Tensor) -> dict:
        """Evaluate model (next-token prediction)."""
        with torch.no_grad():
            logits = self.forward(token_ids)
            # Shift: predict next token
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.reshape(-1, self.vocab_size),
                shift_targets.reshape(-1),
            ).item()

            preds = shift_logits.argmax(dim=-1)
            accuracy = (preds == shift_targets).float().mean().item()

        return {"loss": loss, "accuracy": accuracy}
