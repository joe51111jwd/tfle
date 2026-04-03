"""Tiny causal transformer with ternary projection layers.

Architecture per the spec:
  - d_model=128, n_heads=4 (32 dims/head), n_layers=2, d_ff=512
  - Ternary (TFLE): Q, K, V, O projections + FF layers + output proj
  - Float32 (backprop): token embeddings, positional embeddings, LayerNorm
  - Pre-LN transformer blocks with causal masking
  - ~424K ternary params + ~65K float32 params
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tfle.layers import TFLELayer, ternary_matmul
from tfle.config import TFLEConfig
from tfle.baseline import STETernaryLinear

from .shared import TokenEmbedding, causal_mask


class TernaryAttentionBlock(nn.Module):
    """Single transformer block with ternary projections.

    Pre-LN: LayerNorm -> Attention -> Residual -> LayerNorm -> FF -> Residual
    All linear projections are TFLELayer (ternary, trained by TFLE).
    LayerNorm is float32 (trained by backprop).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        config: TFLEConfig,
        block_idx: int,
        device: torch.device,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # LayerNorm (float32, backprop)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # Ternary projections (TFLE)
        base_idx = block_idx * 6  # 6 projections per block
        self.W_q = TFLELayer(d_model, d_model, config, layer_idx=base_idx, device=device)
        self.W_k = TFLELayer(d_model, d_model, config, layer_idx=base_idx + 1, device=device)
        self.W_v = TFLELayer(d_model, d_model, config, layer_idx=base_idx + 2, device=device)
        self.W_o = TFLELayer(d_model, d_model, config, layer_idx=base_idx + 3, device=device)

        # Feedforward (ternary)
        self.W_ff1 = TFLELayer(d_model, d_ff, config, layer_idx=base_idx + 4, device=device)
        self.W_ff2 = TFLELayer(d_ff, d_model, config, layer_idx=base_idx + 5, device=device)

        self.ternary_layers = [self.W_q, self.W_k, self.W_v, self.W_o, self.W_ff1, self.W_ff2]
        self.device = device

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, T, d_model) -> (B, T, d_model)"""
        B, T, D = x.shape

        # Pre-LN + multi-head attention
        h = self.ln1(x)
        q = self.W_q.forward(h.reshape(-1, D)).reshape(B, T, D)
        k = self.W_k.forward(h.reshape(-1, D)).reshape(B, T, D)
        v = self.W_v.forward(h.reshape(-1, D)).reshape(B, T, D)

        # Reshape for multi-head: (B, T, D) -> (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention (no trainable params)
        scale = math.sqrt(self.d_head)
        attn = (q @ k.transpose(-2, -1)) / scale

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        out = attn @ v  # (B, n_heads, T, d_head)

        # Concat heads + output projection
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.W_o.forward(out.reshape(-1, D)).reshape(B, T, D)

        x = x + out  # Residual

        # Pre-LN + feedforward
        h = self.ln2(x)
        ff = self.W_ff1.forward(h.reshape(-1, D)).reshape(B, T, -1)
        ff = F.gelu(ff)
        ff = self.W_ff2.forward(ff.reshape(-1, self.W_ff2.in_features)).reshape(B, T, D)

        x = x + ff  # Residual
        return x


class AttentionLM:
    """Tiny causal transformer for TFLE training.

    Float32 (backprop): embeddings, positional encoding, LayerNorm
    Ternary (TFLE): all projection matrices (Q, K, V, O, FF1, FF2) + output head
    """

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 512,
        max_len: int = 512,
        config: TFLEConfig | None = None,
        device: torch.device | None = None,
    ):
        if config is None:
            config = TFLEConfig()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers_count = n_layers
        self.max_len = max_len
        self.device = device or torch.device("cpu")
        self.config = config

        # Float32 components
        self.embedding = TokenEmbedding(vocab_size, d_model, max_len).to(self.device)
        self.ln_final = nn.LayerNorm(d_model).to(self.device)

        # Transformer blocks (contain ternary layers)
        self.blocks: list[TernaryAttentionBlock] = []
        for i in range(n_layers):
            block = TernaryAttentionBlock(
                d_model, n_heads, d_ff, config, block_idx=i, device=self.device,
            )
            block.ln1 = block.ln1.to(self.device)
            block.ln2 = block.ln2.to(self.device)
            self.blocks.append(block)

        # Output projection (ternary)
        out_idx = n_layers * 6
        self.output_proj = TFLELayer(
            d_model, vocab_size, config, layer_idx=out_idx, device=self.device,
        )

        # Collect all ternary layers for TFLE training
        self.layers: list[TFLELayer] = []
        for block in self.blocks:
            self.layers.extend(block.ternary_layers)
        self.layers.append(self.output_proj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T) token indices -> (B, T, vocab_size) logits"""
        B, T = x.shape
        mask = causal_mask(T, x.device)

        h = self.embedding(x)  # (B, T, d_model)
        for block in self.blocks:
            h = block(h, mask)

        h = self.ln_final(h)
        # Output projection: (B, T, d_model) -> (B, T, vocab_size)
        logits = self.output_proj.forward(
            h.reshape(-1, self.d_model)
        ).reshape(B, T, self.vocab_size)

        return logits

    def get_float_params(self) -> list[nn.Parameter]:
        """All float32 parameters (for backprop optimizer)."""
        params = list(self.embedding.parameters()) + list(self.ln_final.parameters())
        for block in self.blocks:
            params.extend(block.ln1.parameters())
            params.extend(block.ln2.parameters())
        return params

    def get_ternary_param_count(self) -> int:
        return sum(l.in_features * l.out_features for l in self.layers)

    def get_float_param_count(self) -> int:
        return sum(p.numel() for p in self.get_float_params())

    @torch.no_grad()
    def generate(self, prompt_tokens: list[int], max_length: int, temperature: float = 0.8) -> list[int]:
        tokens = list(prompt_tokens)
        for _ in range(max_length):
            ctx = tokens[-self.max_len:]
            x = torch.tensor([ctx], dtype=torch.long, device=self.device)
            logits = self.forward(x)
            next_logits = logits[0, -1]  # Last position
            if temperature > 0:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_tok = torch.multinomial(probs, 1).item()
            else:
                next_tok = next_logits.argmax().item()
            tokens.append(next_tok)
        return tokens

    def generate_text(self, prompt: str, length: int = 200, temperature: float = 0.8) -> str:
        tokens = [ord(c) % 256 for c in prompt]
        out = self.generate(tokens, length, temperature)
        return "".join(chr(t) for t in out)

    def save_checkpoint(self, path: str):
        state = {
            "embedding": self.embedding.state_dict(),
            "ln_final": self.ln_final.state_dict(),
            "block_lns": [
                {"ln1": b.ln1.state_dict(), "ln2": b.ln2.state_dict()}
                for b in self.blocks
            ],
            "weights": [l.weights.cpu().clone() for l in self.layers],
        }
        torch.save(state, path)

    def load_checkpoint(self, path: str):
        state = torch.load(path, weights_only=False, map_location="cpu")
        self.embedding.load_state_dict(state["embedding"])
        self.embedding.to(self.device)
        self.ln_final.load_state_dict(state["ln_final"])
        self.ln_final.to(self.device)
        for i, bln in enumerate(state["block_lns"]):
            self.blocks[i].ln1.load_state_dict(bln["ln1"])
            self.blocks[i].ln1.to(self.device)
            self.blocks[i].ln2.load_state_dict(bln["ln2"])
            self.blocks[i].ln2.to(self.device)
        for i, w in enumerate(state["weights"]):
            self.layers[i].weights = w.to(self.device)

    def verify_ternary_contribution(self) -> dict:
        """Compare learned ternary weights vs re-randomized. Measures TFLE's contribution."""
        # Save current weights
        saved = [l.weights.clone() for l in self.layers]

        # Randomize all ternary weights
        from tfle.layers import random_ternary
        for l in self.layers:
            l.weights = random_ternary(l.in_features, l.out_features, self.config).to(self.device)

        return {"saved_weights": saved, "note": "Call evaluate with random, then restore"}

    def restore_weights(self, saved: list[torch.Tensor]):
        for l, w in zip(self.layers, saved):
            l.weights = w.to(self.device)


class STEAttentionLM(nn.Module):
    """Same architecture as AttentionLM but with STE for fair comparison."""

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 512,
        max_len: int = 512,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.max_len = max_len

        self.embedding = TokenEmbedding(vocab_size, d_model, max_len)
        self.ln_final = nn.LayerNorm(d_model)

        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(STEAttentionBlock(d_model, n_heads, d_ff))

        self.output_proj = STETernaryLinear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        mask = causal_mask(T, x.device)
        h = self.embedding(x)
        for block in self.blocks:
            h = block(h, mask)
        h = self.ln_final(h)
        logits = self.output_proj(h.reshape(-1, self.d_model)).reshape(B, T, self.vocab_size)
        return logits

    @torch.no_grad()
    def generate_text(self, prompt: str, length: int = 200, temperature: float = 0.8) -> str:
        tokens = [ord(c) % 256 for c in prompt]
        for _ in range(length):
            ctx = tokens[-self.max_len:]
            x = torch.tensor([ctx], dtype=torch.long, device=next(self.parameters()).device)
            logits = self.forward(x)
            next_logits = logits[0, -1]
            if temperature > 0:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_tok = torch.multinomial(probs, 1).item()
            else:
                next_tok = next_logits.argmax().item()
            tokens.append(next_tok)
        return "".join(chr(t) for t in tokens)


class STEAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.W_q = STETernaryLinear(d_model, d_model)
        self.W_k = STETernaryLinear(d_model, d_model)
        self.W_v = STETernaryLinear(d_model, d_model)
        self.W_o = STETernaryLinear(d_model, d_model)
        self.W_ff1 = STETernaryLinear(d_model, d_ff)
        self.W_ff2 = STETernaryLinear(d_ff, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, D = x.shape

        h = self.ln1(x)
        q = self.W_q(h.reshape(-1, D)).reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_k(h.reshape(-1, D)).reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_v(h.reshape(-1, D)).reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scale = math.sqrt(self.d_head)
        attn = (q @ k.transpose(-2, -1)) / scale
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        out = self.W_o(out.reshape(-1, D)).reshape(B, T, D)
        x = x + out

        h = self.ln2(x)
        ff = F.gelu(self.W_ff1(h.reshape(-1, D)).reshape(B, T, -1))
        ff = self.W_ff2(ff.reshape(-1, ff.shape[-1])).reshape(B, T, D)
        x = x + ff
        return x
